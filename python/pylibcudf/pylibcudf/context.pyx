# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf cimport context as cpp_context


__all__ = ["clear_jit_cache", "enable_jit_cache"]


cpdef void enable_jit_cache(bool enable):
    """Enable or disable the JIT program cache.

    When disabled, the cache will not be used for storing or retrieving
    compiled programs. When enabled, the cache will be used as normal.

    Parameters
    ----------
    enable : bool
        If ``True``, the JIT program cache is enabled; if ``False``, it is
        disabled.
    """
    cpp_context.enable_jit_cache(enable)


cpdef void clear_jit_cache():
    """Clear the JIT program cache, removing all cached programs from memory
    and disk.
    """
    cpp_context.clear_jit_cache()
