# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Callback for the polars collect function to execute on device."""

from __future__ import annotations

import contextlib
import os
import textwrap
import time
import warnings
from functools import cache, partial
from threading import Lock
from typing import TYPE_CHECKING, Literal, assert_never, overload

import nvtx
from cuda.bindings import runtime

from polars.exceptions import ComputeError, PerformanceWarning

import pylibcudf
import rmm
from rmm._cuda import gpu

import cudf_polars.dsl.tracing
from cudf_polars.dsl.ir import (
    IRExecutionContext,
)
from cudf_polars.dsl.tracing import CUDF_POLARS_NVTX_DOMAIN
from cudf_polars.dsl.translate import Translator
from cudf_polars.utils.config import (
    MemoryResourceConfig,
    _env_get_int,
    get_total_device_memory,
)
from cudf_polars.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl
    from polars import GPUEngine

    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import NodeTraverser
    from cudf_polars.utils.config import ConfigOptions, ExecutorType

__all__: list[str] = ["execute_with_cudf"]


def _is_concurrent_managed_access_supported() -> bool:
    """
    Check the availability of concurrent managed access (UVM).

    Note that WSL2 does not support managed memory.
    """
    # Ensure CUDA is initialized before checking cudaDevAttrConcurrentManagedAccess
    runtime.cudaFree(0)

    device_id = 0
    err, supports_managed_access = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess, device_id
    )
    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"Failed to check cudaDevAttrConcurrentManagedAccess with error {err}"
        )  # pragma: no cover
    return supports_managed_access != 0


@cache
def default_memory_resource(
    device: int,
    cuda_managed_memory: bool,  # noqa: FBT001
    memory_resource_config: MemoryResourceConfig | None,
) -> rmm.mr.DeviceMemoryResource:
    """
    Return the default memory resource for cudf-polars.

    Parameters
    ----------
    device
        Disambiguating device id when selecting the device. Must be
        the active device when this function is called.
    cuda_managed_memory
        Whether to use managed memory or not.
    memory_resource_config
        Memory resource configuration to use. If ``None``, the default
        memory resource is used.

    Returns
    -------
    rmm.mr.DeviceMemoryResource
        The default memory resource that cudf-polars uses. Currently
        a managed memory resource, if `cuda_managed_memory` is `True`.
        else, an async pool resource is returned.
    """
    try:
        if memory_resource_config is not None:
            mr = memory_resource_config.create_memory_resource()
        elif cuda_managed_memory and _is_concurrent_managed_access_supported():
            # Allocating 80% of the available memory for the pool.
            # Leaving a 20% headroom to avoid OOM errors.
            free_memory, _ = rmm.mr.available_device_memory()
            free_memory = int(round(float(free_memory) * 0.80 / 256) * 256)
            pylibcudf.prefetch.enable()
            mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.ManagedMemoryResource(),
                    initial_pool_size=free_memory,
                )
            )
        else:
            mr = rmm.mr.CudaAsyncMemoryResource()
    except RuntimeError as e:  # pragma: no cover
        msg, *_ = e.args
        if (
            msg.startswith("RMM failure")
            and msg.find("not supported with this CUDA driver/runtime version") > -1
        ):
            raise ComputeError(
                "GPU engine requested, but incorrect cudf-polars package installed. "
                "cudf-polars requires CUDA 12.2+ to installed."
            ) from None
        else:
            raise
    else:
        return mr


@contextlib.contextmanager
def set_memory_resource(
    mr: rmm.mr.DeviceMemoryResource | None,
    memory_resource_config: MemoryResourceConfig | None,
    executor: ExecutorType,
) -> Generator[rmm.mr.DeviceMemoryResource, None, None]:
    """
    Set the current memory resource for an execution block.

    Parameters
    ----------
    mr
        Memory resource to use. If `None`, calls :func:`default_memory_resource`
        to obtain an mr on the currently active device.
    memory_resource_config
        Memory resource configuration to use when a concrete memory resource.
        is not provided. If ``None``, the default memory resource is used.
    executor
        Executor configuration.

    Returns
    -------
    Memory resource used.

    Notes
    -----
    At exit, the memory resource is restored to whatever was current
    at entry. If a memory resource is provided, it must be valid to
    use with the currently active device.
    """
    previous = rmm.mr.get_current_device_resource()
    if mr is None:
        # Use cuda async by default with the streaming executor.
        if (
            memory_resource_config is None
            and executor.name == "streaming"
            and (device_size := get_total_device_memory()) is not None
        ):  # pragma: no cover
            memory_resource_config = MemoryResourceConfig(
                qualname="rmm.mr.CudaAsyncMemoryResource",
                options={
                    "release_threshold": int(0.5 * device_size / 256) * 256,
                },
            )

        device: int = gpu.getDevice()
        mr = default_memory_resource(
            device=device,
            cuda_managed_memory=bool(
                _env_get_int(
                    "POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY",
                    default=1 if get_total_device_memory() is not None else 0,
                )
                != 0
            ),
            memory_resource_config=memory_resource_config,
        )

    if (
        cudf_polars.dsl.tracing.LOG_TRACES
    ):  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
        mr = rmm.mr.StatisticsResourceAdaptor(mr)

    rmm.mr.set_current_device_resource(mr)
    try:
        yield mr
    finally:
        rmm.mr.set_current_device_resource(previous)


# libcudf doesn't support executing on multiple devices from within the same process.
SEEN_DEVICE = None
SEEN_DEVICE_LOCK = Lock()


@contextlib.contextmanager
def set_device(device: int | None) -> Generator[int, None, None]:
    """
    Set the device the query is executed on.

    Parameters
    ----------
    device
        Device to use. If `None`, uses the current device.

    Returns
    -------
    Device active for the execution of the block.

    Notes
    -----
    At exit, the device is restored to whatever was current at entry.
    """
    global SEEN_DEVICE  # noqa: PLW0603
    current: int = gpu.getDevice()
    to_use = device if device is not None else current
    with SEEN_DEVICE_LOCK:
        if (
            SEEN_DEVICE is not None and to_use != SEEN_DEVICE
        ):  # pragma: no cover; requires multiple GPUs in CI
            raise RuntimeError(
                "cudf-polars does not support running queries on "
                "multiple devices in the same process. "
                f"A previous query used device-{SEEN_DEVICE}, "
                f"the current query is using device-{to_use}."
            )
        if to_use != current:
            gpu.setDevice(to_use)
        SEEN_DEVICE = to_use
    try:
        yield to_use
    finally:
        if to_use != current:
            gpu.setDevice(current)


@overload
def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    should_time: Literal[False],
    *,
    memory_resource: rmm.mr.DeviceMemoryResource | None,
    config_options: ConfigOptions,
    timer: Timer | None,
) -> pl.DataFrame: ...


@overload
def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    should_time: Literal[True],
    *,
    memory_resource: rmm.mr.DeviceMemoryResource | None,
    config_options: ConfigOptions,
    timer: Timer | None,
) -> tuple[pl.DataFrame, list[tuple[int, int, str]]]: ...


def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    should_time: bool,  # noqa: FBT001
    *,
    memory_resource: rmm.mr.DeviceMemoryResource | None,
    config_options: ConfigOptions,
    timer: Timer | None,
) -> pl.DataFrame | tuple[pl.DataFrame, list[tuple[int, int, str]]]:
    assert with_columns is None
    assert pyarrow_predicate is None
    assert n_rows is None
    if timer is not None:
        assert should_time

    with (
        nvtx.annotate(message="ExecuteIR", domain=CUDF_POLARS_NVTX_DOMAIN),
        # Device must be set before memory resource is obtained.
        set_device(config_options.device),
        set_memory_resource(
            memory_resource,
            config_options.memory_resource_config,
            config_options.executor,
        ),
    ):
        if config_options.executor.name == "in-memory":
            context = IRExecutionContext()
            df = ir.evaluate(cache={}, timer=timer, context=context).to_polars()
            if timer is None:
                return df
            else:
                return df, timer.timings
        elif config_options.executor.name == "streaming":
            from cudf_polars.streaming.parallel import evaluate_streaming

            if timer is not None:
                msg = textwrap.dedent("""\
                    LazyFrame.profile() is not supported with the streaming executor.
                    To profile execution with the streaming executor, use NVIDIA
                    NSight Systems with the 'streaming' scheduler.
                    """)
                raise NotImplementedError(msg)

            return evaluate_streaming(ir, config_options)
        assert_never(config_options.executor)


def execute_with_cudf(
    nt: NodeTraverser, duration_since_start: int | None, *, config: GPUEngine
) -> None:
    """
    A post optimization callback that attempts to execute the plan with cudf.

    Parameters
    ----------
    nt
        NodeTraverser

    duration_since_start
        Time since the user started executing the query (or None if no
        profiling should occur).

    config
        GPUEngine object. Configuration is available as ``engine.config``.

    Raises
    ------
    ValueError
        If the config contains unsupported keys.
    NotImplementedError
        If translation of the plan is unsupported.

    Notes
    -----
    The NodeTraverser is mutated if the libcudf executor can handle the plan.
    """
    if duration_since_start is None:
        timer = None
    else:
        start = time.monotonic_ns()
        timer = Timer(start - duration_since_start)

    memory_resource = config.memory_resource

    with nvtx.annotate(message="ConvertIR", domain=CUDF_POLARS_NVTX_DOMAIN):
        translator = Translator(nt, config)
        ir = translator.translate_ir()
        if timer is not None:
            timer.store(start, time.monotonic_ns(), "gpu-ir-translation")

        exception = translator.unsupported_operations_error()
        if exception is not None:
            error_message = exception.args[0]
            if bool(int(os.environ.get("POLARS_VERBOSE", "0"))):
                warnings.warn(error_message, PerformanceWarning, stacklevel=2)
            if translator.config_options.raise_on_fail:
                raise exception
        else:
            nt.set_udf(
                partial(
                    _callback,
                    ir,
                    memory_resource=memory_resource,
                    config_options=translator.config_options,
                    timer=timer,
                )
            )
