# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bind_to_gpu() and HardwareBindingPolicy."""

from __future__ import annotations

import functools
import multiprocessing
import traceback
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch

import distributed
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection


def _wrapper(child_conn: Connection, target: Callable[[], None]) -> None:
    """Run ``target`` in the child and report the outcome via ``child_conn``."""
    try:
        target()
        child_conn.send(None)
    except BaseException as exc:
        try:
            child_conn.send(exc)
        except Exception:
            child_conn.send(
                RuntimeError(
                    f"{type(exc).__name__}: {exc}\n"
                    f"{''.join(traceback.format_tb(exc.__traceback__))}"
                )
            )
    finally:
        child_conn.close()


def _run_in_subprocess(target: Callable[[], None], timeout_seconds: int) -> None:
    """Execute ``target()`` in a spawned child process.

    Spawn (rather than fork) is used so the child starts from a clean
    interpreter state and never inherits process-wide state from the
    pytest worker - notably the active CUDA context, live pylibcudf
    objects, and any session-scoped streaming engines. Inheriting a
    CUDA context across ``fork()`` is unsafe: the child's GC of
    inherited cudf/pylibcudf objects calls into CUDA from a process
    that doesn't actually have a context, which surfaces as
    ``cudaErrorInitializationError`` and an uncaught
    ``std::terminate`` (SIGABRT) at child exit.

    ``target`` must be a module-level callable (spawn pickles the
    target by name).
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(target=_wrapper, args=(child_conn, target))
    proc.start()
    try:
        proc.join(timeout=timeout_seconds)

        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
            raise RuntimeError(f"Subprocess timed out after {timeout_seconds} seconds")

        if parent_conn.poll(timeout=timeout_seconds):
            exc = parent_conn.recv()
            if exc is not None:
                raise exc

        if proc.exitcode != 0:
            raise RuntimeError(f"Subprocess exited with code {proc.exitcode}")
    finally:
        proc.close()


def _reset_bind_state() -> None:
    """Reset the module-level bind state so each subprocess starts clean."""
    from cudf_polars.engine import hardware_binding

    hardware_binding._bind_done = False


# ---------------------------------------------------------------------------
# Subprocess test bodies. These must be module-level so they pickle across
# the spawn boundary.
# ---------------------------------------------------------------------------


def _body_bind_called_once() -> None:
    _reset_bind_state()
    with patch("cudf_polars.engine.hardware_binding.bind") as mock_bind:
        from cudf_polars.engine.hardware_binding import (
            HardwareBindingPolicy,
            bind_to_gpu,
        )

        policy = HardwareBindingPolicy()
        bind_to_gpu(policy)
        bind_to_gpu(policy)
        assert mock_bind.call_count == 1


def test_bind_called_once(timeout_seconds: int) -> None:
    """bind() is called exactly once even when bind_to_gpu() is called twice."""
    _run_in_subprocess(_body_bind_called_once, timeout_seconds)


def _body_bind_falls_back_to_gpu_0() -> None:
    _reset_bind_state()
    mock_bind = MagicMock(side_effect=[RuntimeError("no CUDA_VISIBLE_DEVICES"), None])
    with patch(
        "cudf_polars.engine.hardware_binding.bind",
        mock_bind,
    ):
        from cudf_polars.engine.hardware_binding import (
            HardwareBindingPolicy,
            bind_to_gpu,
        )

        policy = HardwareBindingPolicy()
        bind_to_gpu(policy)
        bind_kw = {
            "cpu": policy.cpu,
            "memory": policy.memory,
            "network": policy.network,
        }
        assert mock_bind.call_count == 2
        assert mock_bind.call_args_list == [
            call(**bind_kw),
            call(gpu_id=0, **bind_kw),
        ]


def test_bind_falls_back_to_gpu_0(timeout_seconds: int) -> None:
    """When bind() raises RuntimeError, falls back to gpu_id=0."""
    _run_in_subprocess(_body_bind_falls_back_to_gpu_0, timeout_seconds)


def _body_bind_raise_on_fail_propagates_exception() -> None:
    _reset_bind_state()
    mock_bind = MagicMock(side_effect=RuntimeError("binding failed"))
    with patch(
        "cudf_polars.engine.hardware_binding.bind",
        mock_bind,
    ):
        from cudf_polars.engine.hardware_binding import (
            HardwareBindingPolicy,
            bind_to_gpu,
        )

        with pytest.raises(RuntimeError, match="binding failed"):
            bind_to_gpu(HardwareBindingPolicy(raise_on_fail=True))


def test_bind_raise_on_fail_propagates_exception(timeout_seconds: int) -> None:
    """raise_on_fail=True lets RuntimeError from bind() propagate."""
    _run_in_subprocess(_body_bind_raise_on_fail_propagates_exception, timeout_seconds)


def _body_bind_raise_on_fail_false_suppresses_exception() -> None:
    _reset_bind_state()
    mock_bind = MagicMock(side_effect=RuntimeError("binding failed"))
    with patch(
        "cudf_polars.engine.hardware_binding.bind",
        mock_bind,
    ):
        from cudf_polars.engine.hardware_binding import (
            HardwareBindingPolicy,
            bind_to_gpu,
        )

        bind_to_gpu(HardwareBindingPolicy(raise_on_fail=False))


def test_bind_raise_on_fail_false_suppresses_exception(timeout_seconds: int) -> None:
    """raise_on_fail=False silently ignores RuntimeError from bind()."""
    _run_in_subprocess(
        _body_bind_raise_on_fail_false_suppresses_exception, timeout_seconds
    )


def _body_bind_thread_safe(timeout_seconds: int) -> None:
    import threading

    _reset_bind_state()
    with patch("cudf_polars.engine.hardware_binding.bind") as mock_bind:
        from cudf_polars.engine.hardware_binding import (
            HardwareBindingPolicy,
            bind_to_gpu,
        )

        policy = HardwareBindingPolicy()
        barrier = threading.Barrier(8, timeout=timeout_seconds)

        def _call_bind() -> None:
            barrier.wait()
            bind_to_gpu(policy)

        threads = [threading.Thread(target=_call_bind) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout_seconds)

        assert mock_bind.call_count == 1


def test_bind_thread_safe(timeout_seconds: int) -> None:
    """Concurrent calls from multiple threads result in exactly one bind() call."""
    _run_in_subprocess(
        functools.partial(_body_bind_thread_safe, timeout_seconds), timeout_seconds
    )


def _body_bind_done_flag_set() -> None:
    from cudf_polars.engine import hardware_binding

    _reset_bind_state()
    assert not hardware_binding._bind_done
    with patch("cudf_polars.engine.hardware_binding.bind"):
        hardware_binding.bind_to_gpu(hardware_binding.HardwareBindingPolicy())
        assert hardware_binding._bind_done


def test_bind_done_flag_set(timeout_seconds: int) -> None:
    """_bind_done is True after bind_to_gpu() succeeds."""
    _run_in_subprocess(_body_bind_done_flag_set, timeout_seconds)


def _body_bind_disabled() -> None:
    _reset_bind_state()
    with patch("cudf_polars.engine.hardware_binding.bind") as mock_bind:
        from cudf_polars.engine.hardware_binding import (
            HardwareBindingPolicy,
            bind_to_gpu,
        )

        bind_to_gpu(HardwareBindingPolicy(enabled=False))
        mock_bind.assert_not_called()


def test_bind_disabled(timeout_seconds: int) -> None:
    """enabled=False skips binding entirely."""
    _run_in_subprocess(_body_bind_disabled, timeout_seconds)


def _body_bind_enable_once_false() -> None:
    _reset_bind_state()
    with patch("cudf_polars.engine.hardware_binding.bind") as mock_bind:
        from cudf_polars.engine.hardware_binding import (
            HardwareBindingPolicy,
            bind_to_gpu,
        )

        policy = HardwareBindingPolicy(enable_once=False)
        bind_to_gpu(policy)
        bind_to_gpu(policy)
        assert mock_bind.call_count == 2


def test_bind_enable_once_false(timeout_seconds: int) -> None:
    """enable_once=False allows repeated bind() calls."""
    _run_in_subprocess(_body_bind_enable_once_false, timeout_seconds)


# ---------------------------------------------------------------------------
# dask_setup (nanny preload)
# ---------------------------------------------------------------------------


def test_get_visible_gpu_ids_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from cudf_polars.engine.dask import _get_visible_gpu_ids

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1,4")
    assert _get_visible_gpu_ids() == ["3", "1", "4"]


def test_dask_setup_assigns_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """dask_setup assigns round-robin CUDA_VISIBLE_DEVICES to each nanny."""
    import cudf_polars.engine.dask as dask_mod

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(dask_mod, "_nanny_preload_counter", 0)

    nanny0 = MagicMock(spec=distributed.Nanny)
    nanny0.env = {}
    dask_mod.dask_setup(nanny0)
    assert nanny0.env["CUDA_VISIBLE_DEVICES"] == "0"

    nanny1 = MagicMock(spec=distributed.Nanny)
    nanny1.env = {}
    dask_mod.dask_setup(nanny1)
    assert nanny1.env["CUDA_VISIBLE_DEVICES"] == "1"


def test_dask_setup_wraps_around(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter wraps around when workers exceed GPUs."""
    import cudf_polars.engine.dask as dask_mod

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(dask_mod, "_nanny_preload_counter", 0)

    nannies = []
    for _ in range(4):
        n = MagicMock(spec=distributed.Nanny)
        n.env = {}
        dask_mod.dask_setup(n)
        nannies.append(n)

    assert nannies[0].env["CUDA_VISIBLE_DEVICES"] == "0"
    assert nannies[1].env["CUDA_VISIBLE_DEVICES"] == "1"
    assert nannies[2].env["CUDA_VISIBLE_DEVICES"] == "0"  # wraps
    assert nannies[3].env["CUDA_VISIBLE_DEVICES"] == "1"  # wraps


def test_dask_setup_rejects_worker() -> None:
    """dask_setup raises TypeError when used with --preload instead of --preload-nanny."""
    import cudf_polars.engine.dask as dask_mod

    worker = MagicMock(spec=distributed.Worker)
    with pytest.raises(TypeError, match="--preload-nanny"):
        dask_mod.dask_setup(worker)
