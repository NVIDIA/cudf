# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.unicode_normalize cimport (
    unicode_normalizer as cpp_unicode_normalizer,
    unicode_normalization_form as cpp_unicode_normalization_form,
)
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class UnicodeNormalizer:
    cdef unique_ptr[cpp_unicode_normalizer] c_obj

cpdef Column normalize_unicode(
    Column input,
    UnicodeNormalizer normalizer,
    object stream=*,
    DeviceMemoryResource mr=*,
)
