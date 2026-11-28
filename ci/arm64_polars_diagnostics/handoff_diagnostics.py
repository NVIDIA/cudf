# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary probes at the archived Polars actor's table handoff."""

import itertools
import json
import os
import threading
import time
import weakref
from functools import wraps

_CASES = ("test_datetime_truncate[", "test_datetime_offset_by[")


def _buffers(table):
    """Include masks and nested columns so sharing a slice is observable."""
    result = []

    def visit(column, path):
        for kind, span in (
            ("data", column.data()),
            ("mask", column.null_mask()),
        ):
            if span is not None and span.ptr and span.size:
                result.append((path, kind, int(span.ptr), int(span.size)))
        for index, child in enumerate(column.children()):
            visit(child, (*path, index))

    for index, column in enumerate(table.columns()):
        visit(column, (index,))
    return result


def _aliases(inputs, outputs):
    return [
        (source, result)
        for source in inputs
        for result in outputs
        if source[2] < result[2] + result[3]
        and result[2] < source[2] + source[3]
    ]


def _handle(stream):
    return int(stream.__cuda_stream__()[1])


def _install_lifetime(path):
    from cudf_polars.streaming.actor_graph import utils

    original = utils._evaluate_chunk_sync
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600
    )

    @wraps(original)
    def evaluate(chunk, ir, ir_context, br, *, ordering_metadata=None):
        test = os.environ.get("PYTEST_CURRENT_TEST", "")
        if not any(case in test for case in _CASES):
            return original(
                chunk, ir, ir_context, br, ordering_metadata=ordering_metadata
            )

        frame = utils.chunk_to_frame(
            chunk, ir.children[0], ordering_metadata=ordering_metadata
        )
        result = utils.evaluate_ir(ir, frame, context=ir_context)
        aliases = _aliases(_buffers(frame.table), _buffers(result.table))
        if aliases:
            # One write per aliased handoff. Avoid weakrefs, extra CUDA calls,
            # and retaining owners, which can each change the failure timing.
            row = (
                time.monotonic_ns(),
                threading.get_native_id(),
                _handle(frame.stream),
                _handle(result.stream),
                test,
                aliases,
            )
            os.write(
                fd, (json.dumps(row, separators=(",", ":")) + "\n").encode()
            )

        return utils.TableChunk.from_pylibcudf_table(
            result.table, result.stream, exclusive_view=True, br=br
        )

    utils._evaluate_chunk_sync = evaluate


def install(path, mode):
    if mode == "lifetime":
        _install_lifetime(path)
        return None

    from cuda.bindings import driver
    from cudf_polars.streaming.actor_graph import utils
    from cudf_polars.utils import cuda_stream

    original = utils._evaluate_chunk_sync
    original_join = cuda_stream.join_cuda_streams
    success = driver.CUresult.CUDA_SUCCESS
    not_ready = driver.CUresult.CUDA_ERROR_NOT_READY
    ids = itertools.count()
    releases = {}
    active = threading.local()
    pending = {}
    lock = threading.Lock()
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600
    )

    def log(stage, call=None, **fields):
        os.write(
            fd,
            (
                json.dumps(
                    {
                        "ns": time.monotonic_ns(),
                        "tid": threading.get_native_id(),
                        "stage": stage,
                        "call": call,
                        **fields,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
        )

    def checked(result):
        status, *values = result
        if status != success:
            raise RuntimeError(f"CUDA handoff probe failed with {status.name}")
        return values[0] if values else None

    def stream_info(stream, call, stage):
        handle = _handle(stream)
        status, context = driver.cuCtxGetCurrent()
        # Preserve the last handle even if the driver faults while looking it up.
        log(
            stage,
            call,
            stream=handle,
            context=int(context),
            status=status.name,
        )
        if handle > 2:
            status, owner = driver.cuStreamGetCtx(driver.CUstream(handle))
            log(
                stage + "-context",
                call,
                stream=handle,
                owner=int(owner),
                status=status.name,
            )
        return context

    def watch_columns(table, call, role):
        for index, column in enumerate(table.columns()):
            token = (call, role, index)

            def release(_ref, token=token):
                releases.pop(token, None)
                log(
                    "column-object-release",
                    token[0],
                    role=token[1],
                    index=token[2],
                )

            try:
                releases[token] = weakref.ref(column, release)
            except TypeError:
                log("column-weakref-unavailable", call, role=role)

    def reap(context, *, wait=False):
        key = int(context)
        with lock:
            entries = pending.pop(key, ())
        remaining = []
        for entry in entries:
            event, call, _owners, _context = entry
            (status,) = (
                driver.cuEventSynchronize(event)
                if wait
                else driver.cuEventQuery(event)
            )
            if not wait and status == not_ready:
                remaining.append(entry)
                continue
            checked((status,))
            checked(driver.cuEventDestroy(event))
            log("consumer-complete", call)
        if remaining:
            with lock:
                pending.setdefault(key, []).extend(remaining)

    def finish_test():
        # Drain only diagnostic ownership references, after pytest has finished
        # the test and the actor threads are no longer using those references.
        with lock:
            contexts = [
                entries[0][3] for entries in pending.values() if entries
            ]
        for context in contexts:
            checked(driver.cuCtxPushCurrent(context))
            try:
                reap(context, wait=True)
            finally:
                checked(driver.cuCtxPopCurrent())

    @wraps(original_join)
    def join_cuda_streams(*, downstreams, upstreams):
        call = getattr(active, "call", None)
        if call is None:
            return original_join(downstreams=downstreams, upstreams=upstreams)
        downstreams, upstreams = tuple(downstreams), tuple(upstreams)
        fields = {
            "upstream": [_handle(stream) for stream in upstreams],
            "downstream": [_handle(stream) for stream in downstreams],
        }
        log("join-begin", call, **fields)
        result = original_join(downstreams=downstreams, upstreams=upstreams)
        log("join-end", call, **fields)
        return result

    @wraps(original)
    def evaluate(chunk, ir, ir_context, br, *, ordering_metadata=None):
        test = os.environ.get("PYTEST_CURRENT_TEST", "")
        if not any(case in test for case in _CASES):
            return original(
                chunk, ir, ir_context, br, ordering_metadata=ordering_metadata
            )

        call = next(ids)
        active.call = call
        try:
            log(
                "materialize-begin",
                call,
                test=test,
                ir=type(ir).__name__,
                chunk=id(chunk),
            )
            frame = utils.chunk_to_frame(
                chunk, ir.children[0], ordering_metadata=ordering_metadata
            )
            producer = frame.stream
            context = stream_info(producer, call, "input-stream")
            if mode == "ownership":
                reap(context)
            inputs = _buffers(frame.table)
            watch_columns(frame.table, call, "input")
            log("evaluate-begin", call, table=id(frame.table), buffers=inputs)
            if mode == "fence":
                log("producer-fence-begin", call, stream=_handle(producer))
                producer.synchronize()
                log("producer-fence-end", call)
            result = utils.evaluate_ir(ir, frame, context=ir_context)
            result_table, consumer = result.table, result.stream
            stream_info(consumer, call, "output-stream")
            outputs = _buffers(result_table)
            aliases = _aliases(inputs, outputs)
            watch_columns(result_table, call, "output")
            exclusive = mode != "ownership" or not aliases
            log(
                "wrap-begin",
                call,
                table=id(result_table),
                buffers=outputs,
                aliases=aliases,
                exclusive=exclusive,
            )
            wrapped = utils.TableChunk.from_pylibcudf_table(
                result_table, consumer, exclusive_view=exclusive, br=br
            )
            log("wrap-end", call, chunk=id(wrapped))
            if not exclusive:
                event = checked(
                    driver.cuEventCreate(
                        driver.CUevent_flags.CU_EVENT_DISABLE_TIMING
                    )
                )
                checked(
                    driver.cuEventRecord(
                        event, driver.CUstream(_handle(consumer))
                    )
                )
                context = checked(driver.cuCtxGetCurrent())
                with lock:
                    pending.setdefault(int(context), []).append(
                        (
                            event,
                            call,
                            (chunk, frame, result, producer, consumer),
                            context,
                        )
                    )
                log("ownership-retained", call, context=int(context))
            return wrapped
        finally:
            active.call = None

    log("installed", mode=mode)
    cuda_stream.join_cuda_streams = join_cuda_streams
    utils._evaluate_chunk_sync = evaluate
    return finish_test
