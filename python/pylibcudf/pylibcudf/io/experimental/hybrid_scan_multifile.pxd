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


cdef class RowGroupIndices:
    cdef vector[vector[size_type]] c_obj

    @staticmethod
    cdef RowGroupIndices from_libcudf(vector[vector[size_type]] indices)


cdef class HybridScanMultiFile:
    cdef unique_ptr[cpp_hybrid_scan_multifile] c_obj
    cdef Stream _stream
    cdef DeviceMemoryResource mr
    cdef object _payload_page_data
