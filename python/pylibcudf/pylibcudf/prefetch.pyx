# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf cimport prefetch as cpp_prefetch


__all__ = ["disable", "disable_debugging", "enable", "enable_debugging"]

cpdef void enable():
    """Turn on prefetching of managed memory."""
    cpp_prefetch.enable()


cpdef void disable():
    """Turn off prefetching of managed memory."""
    cpp_prefetch.disable()


cpdef void enable_debugging():
    """Enable prefetch debugging."""
    cpp_prefetch.enable_debugging()


cpdef void disable_debugging():
    """Disable prefetch debugging."""
    cpp_prefetch.disable_debugging()
