# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

from cuda.bindings import runtime

import pylibcudf
import rmm.mr

from .fast_slow_proxy import (
    as_proxy_object,
    is_proxy_instance,
    is_proxy_object,
)
from .magics import load_ipython_extension
from .profiler import Profiler

__all__ = [
    "Profiler",
    "as_proxy_object",
    "install",
    "is_proxy_instance",
    "is_proxy_object",
    "load_ipython_extension",
]


LOADED = False


def install():
    """Enable Pandas Accelerator Mode."""
    from .module_accelerator import ModuleAccelerator

    loader = ModuleAccelerator.install("pandas", "cudf", "pandas")
    global LOADED
    LOADED = loader is not None
    if (
        "RAPIDS_NO_INITIALIZE" in os.environ
        or "CUDF_NO_INITIALIZE" in os.environ
    ):
        return

    try:
        # The default mode is "managed_pool" if UVM is supported, otherwise "pool"
        # Ensure CUDA is initialized before checking cudaDevAttrConcurrentManagedAccess
        runtime.cudaFree(0)
        err, supports_managed_access = runtime.cudaDeviceGetAttribute(
            runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess, 0
        )
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"Failed to check cudaDevAttrConcurrentManagedAccess with error {err}"
            )
        managed_memory_is_supported = supports_managed_access != 0
    except RuntimeError as e:
        warnings.warn(str(e))
        return

    rmm_mode = os.getenv("CUDF_PANDAS_RMM_MODE")
    rmm_mode_explicitly_set = rmm_mode is not None
    if rmm_mode is None:
        rmm_mode = "managed_pool" if managed_memory_is_supported else "pool"

    # Check if a non-default memory resource is set
    current_mr = rmm.mr.get_current_device_resource()
    if not isinstance(current_mr, rmm.mr.CudaMemoryResource):
        # Warn only if the user explicitly set CUDF_PANDAS_RMM_MODE
        if rmm_mode_explicitly_set:
            warnings.warn(
                "cudf.pandas detected an already configured memory resource, ignoring "
                f"'CUDF_PANDAS_RMM_MODE={rmm_mode}'",
                UserWarning,
            )
        return

    free_memory, _ = rmm.mr.available_device_memory()
    free_memory = int(round(float(free_memory) * 0.80 / 256) * 256)
    new_mr = current_mr

    if rmm_mode == "pool":
        new_mr = rmm.mr.PoolMemoryResource(
            current_mr,
            initial_pool_size=free_memory,
        )
    elif rmm_mode == "async":
        new_mr = rmm.mr.CudaAsyncMemoryResource()
    elif "managed" in rmm_mode:
        if not managed_memory_is_supported:
            raise ValueError(
                "Managed memory is not supported on this system, so the "
                f"requested {rmm_mode=} is invalid."
            )
        if rmm_mode == "managed":
            new_mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.ManagedMemoryResource()
            )
        elif rmm_mode == "managed_pool":
            new_mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.ManagedMemoryResource(),
                    initial_pool_size=free_memory,
                )
            )
        else:
            raise ValueError(f"Unsupported {rmm_mode=}")
        pylibcudf.prefetch.enable()
    elif rmm_mode != "cuda":
        raise ValueError(f"Unsupported {rmm_mode=}")

    rmm.mr.set_current_device_resource(new_mr)


def pytest_load_initial_conftests(early_config, parser, args):
    # We need to install ourselves before conftest.py import (which
    # might import pandas) This hook is guaranteed to run before that
    # happens see
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
    try:
        install()
    except RuntimeError:
        raise RuntimeError(
            "An existing plugin has already loaded pandas. Interposing failed."
        )
