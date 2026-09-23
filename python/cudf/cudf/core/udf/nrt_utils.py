# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextvars
import os
from contextlib import contextmanager
from functools import cache
from importlib.util import find_spec

from numba.cuda import config as numba_config

_current_nrt_context: contextvars.ContextVar = contextvars.ContextVar(
    "current_nrt_context"
)


class CaptureNRTUsage:
    """
    Context manager for determining if NRT is needed.
    Managed types may set use_nrt to be true during
    instantiation to signal that NRT must be enabled
    during code generation.
    """

    def __init__(self):
        self.use_nrt = False

    def __enter__(self):
        self._token = _current_nrt_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_nrt_context.reset(self._token)


@cache
def _get_libcudf_rapids_include_dir():
    spec = find_spec("libcudf")
    if spec is None or spec.submodule_search_locations is None:
        return None

    for package_dir in spec.submodule_search_locations:
        include_dir = os.path.join(package_dir, "include", "rapids")
        if os.path.isfile(os.path.join(include_dir, "cuda", "atomic")):
            return include_dir

    return None


def _append_search_path(search_paths, path):
    paths = [p for p in search_paths.split(os.pathsep) if p]
    if path not in paths:
        paths.append(path)
    return os.pathsep.join(paths)


@contextmanager
def nrt_enabled():
    """
    Context manager for enabling NRT via the numba
    config. CUDA_ENABLE_NRT may be toggled dynamically
    for a single kernel launch, so we use this context
    to enable it for those that we know need it.
    """
    original_nrt_value = getattr(numba_config, "CUDA_ENABLE_NRT", False)
    original_nvrtc_search_paths = getattr(
        numba_config, "CUDA_NVRTC_EXTRA_SEARCH_PATHS", ""
    )
    nvrtc_search_paths = original_nvrtc_search_paths or ""

    try:
        numba_config.CUDA_ENABLE_NRT = True
        if (include_dir := _get_libcudf_rapids_include_dir()) is not None:
            numba_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = _append_search_path(
                nvrtc_search_paths, include_dir
            )
        yield
    finally:
        numba_config.CUDA_ENABLE_NRT = original_nrt_value
        numba_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = (
            original_nvrtc_search_paths
        )
