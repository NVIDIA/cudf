# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gather per-worker evidence for the temporary Arm64 Polars CI replay."""

import ctypes
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

_output = Path(os.environ["CUDF_POLARS_DIAGNOSTICS_DIR"])
_trial = os.environ["CUDF_POLARS_DIAGNOSTICS_TRIAL"]
_trace = not _trial.startswith(("gdb-", "plain-", "signal-"))
_warmup_first = (
    "tests/expressions/test_agg.py::"
    "test_n_unique_sorted[spmd-small-data0-Int64-False-True-True-False]"
)
_warmup_join = (
    "tests/streaming/test_join.py::test_join[spmd-small-options1-False-left]"
)
_warmup_last = "tests/streaming/test_unique.py::test_unique[spmd-small-none-subset15-False]"


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(args):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    if _trial == "warmup-serial":
        args.append("--numprocesses=1")
    elif _trial == "warmup-shared":
        # Reproduce the same worker history on each worker under contention.
        args.append("--dist=each")


def _attach_debugger(worker):
    path = _output / f"{worker}.gdb.log"
    libc = ctypes.CDLL(None, use_errno=True)
    # Yama can restrict a debugger spawned by its own tracee. The CI container
    # also grants SYS_PTRACE because attachment uses a separate process.
    pr_set_ptracer = 0x59616D61
    libc.prctl(
        ctypes.c_int(pr_set_ptracer),
        ctypes.c_ulong(-1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    command_file = Path(__file__).with_name("native.gdb")
    with path.open("wb") as log:
        debugger = subprocess.Popen(
            [
                "gdb",
                "-nx",
                "-batch",
                "-p",
                str(os.getpid()),
                "-x",
                command_file,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                k: v
                for k, v in os.environ.items()
                if k not in {"GH_TOKEN", "GITHUB_TOKEN"}
            },
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if b"CUDF_GDB_ATTACHED" in path.read_bytes():
            return debugger
        if debugger.poll() is not None:
            break
        time.sleep(0.1)
    debugger.terminate()
    pytest.exit(
        f"GDB could not attach to {worker}; see {path.name}", returncode=2
    )


def _trace_partition_and_pack(worker):
    from cudf_polars.streaming.actor_graph.collectives import shuffle

    original = shuffle.py_partition_and_pack
    path = _output / f"{worker}.shuffle.log"

    def partition_and_pack(**kwargs):
        test = os.environ.get("PYTEST_CURRENT_TEST", "")
        started = time.monotonic_ns()
        with path.open("a") as trace:
            trace.write(
                f"{started} begin {test} keys={kwargs['columns_to_hash']} "
                f"partitions={kwargs['num_partitions']}\n"
            )
        result = original(**kwargs)
        with path.open("a") as trace:
            trace.write(f"{started} end\n")
        return result

    shuffle.py_partition_and_pack = partition_and_pack


def _install_fault_report(worker):
    reporter = ctypes.CDLL(
        os.environ["CUDF_POLARS_FAULT_LIBRARY"], use_errno=True
    )
    path = _output / f"{worker}.fault.log"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    if reporter.install_fault_report(ctypes.c_int(fd)) != 0:
        error = ctypes.get_errno()
        os.close(fd)
        raise OSError(error, os.strerror(error), str(path))
    return reporter


def pytest_sessionstart(session):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        with (_output / "conda-explicit.txt").open("w") as packages:
            subprocess.run(
                ["conda", "list", "--explicit"], stdout=packages, check=True
            )
        if _trial.startswith(("gdb-", "plain-", "signal-")):
            from watch_gpu_events import watch_gpu_events

            stopped = threading.Event()
            watcher = threading.Thread(
                target=watch_gpu_events,
                args=(
                    os.environ["CUDF_POLARS_GPU_UUID"],
                    _output / "gpu-xid.log",
                    stopped,
                ),
                daemon=True,
            )
            session._cudf_gpu_watcher = (stopped, watcher)
            watcher.start()
    elif _trial.startswith("signal-"):
        session._cudf_fault_reporter = _install_fault_report(worker)
    elif _trial.startswith(("gdb-", "native-", "warmup-")):
        session._cudf_debugger = _attach_debugger(worker)
        if _trace:
            _trace_partition_and_pack(worker)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    watcher = getattr(session, "_cudf_gpu_watcher", None)
    if watcher is not None:
        stopped, thread = watcher
        stopped.set()
        thread.join(timeout=3)
    debugger = getattr(session, "_cudf_debugger", None)
    if debugger is not None and debugger.poll() is None:
        # Keep the debugger attached through teardown of the session fixtures.
        debugger.terminate()
        try:
            debugger.wait(timeout=5)
        except subprocess.TimeoutExpired:
            debugger.kill()
            debugger.wait()


def pytest_runtest_logstart(nodeid):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker and _trace:
        with (_output / f"{worker}.tests.log").open("a") as log:
            log.write(f"{time.monotonic_ns()} {nodeid}\n")


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(config, items):
    result = yield
    if _trial.startswith("signal-repeat-"):
        if len({item.nodeid for item in items}) != 2 or len(items) != 2:
            raise pytest.UsageError(
                "Expected two distinct streaming stress cases"
            )
        # Alternate cases so pytest tears down function-scoped fixtures while
        # the session-scoped SPMD engine stays alive on each worker.
        items *= 768 if _trial.endswith("-aggregation") else 256
        return result
    if not _trial.startswith("warmup-"):
        return result
    # Select the recorded worker prefix after cuDF's conftest groups by engine.
    positions = {item.nodeid: index for index, item in enumerate(items)}
    try:
        first, join, last = (
            positions[node]
            for node in (_warmup_first, _warmup_join, _warmup_last)
        )
    except KeyError as error:
        raise pytest.UsageError(f"Missing warmup landmark {error}") from error
    if not first < join < last or last - first + 1 != 5820:
        raise pytest.UsageError(
            f"Unexpected warmup order or count: {first=}, {join=}, {last=}"
        )
    config.hook.pytest_deselected(items=[*items[:first], *items[last + 1 :]])
    items[:] = items[first : last + 1]
    return result
