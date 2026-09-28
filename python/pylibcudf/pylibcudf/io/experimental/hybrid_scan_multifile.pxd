# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.libcudf.io.hybrid_scan_multifile cimport (
    hybrid_scan_multifile as cpp_hybrid_scan_multifile,
)
from pylibcudf.libcudf.types cimport size_type


cdef vector[vector[size_type]] _get_row_group_indices(object row_group_indices) except *


cdef class HybridScanMultiFile:
    cdef unique_ptr[cpp_hybrid_scan_multifile] c_obj
    cdef Stream _stream
    cdef DeviceMemoryResource mr
    cdef object _payload_page_data
