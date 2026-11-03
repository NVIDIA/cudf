# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np


class ProxyNDarrayBase(np.ndarray):
    def __new__(cls, arr):
        original_slow = None
        if isinstance(arr, np.ndarray) and arr.base is not None:
            # Preserve view semantics across fast/slow conversions by
            # remembering the original view. Without this, a round-trip
            # through cupy (e.g. when a proxied function first tries the
            # fast path then falls back to slow) produces a fresh numpy
            # array whose ``.base`` is None.
            original_slow = arr
        if isinstance(arr, cp.ndarray):
            arr = arr.get()
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                "Unsupported array type. Must be numpy.ndarray or cupy.ndarray"
            )
        proxy = np.asarray(arr, dtype=arr.dtype).view(cls)
        if original_slow is not None:
            proxy._fsproxy_original_slow = original_slow
        return proxy

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._fsproxy_wrapped = getattr(obj, "_fsproxy_wrapped", obj)
        original_slow = getattr(obj, "_fsproxy_original_slow", None)
        if original_slow is not None:
            self._fsproxy_original_slow = original_slow
