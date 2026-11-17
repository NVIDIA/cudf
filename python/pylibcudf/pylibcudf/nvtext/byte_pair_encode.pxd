# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.byte_pair_encode cimport bpe_merge_pairs
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class BPEMergePairs:
    cdef unique_ptr[bpe_merge_pairs] c_obj

cpdef Column byte_pair_encoding(
    Column input,
    BPEMergePairs merge_pairs,
    Scalar separator=*,
    object stream = *,
    DeviceMemoryResource mr=*
)
