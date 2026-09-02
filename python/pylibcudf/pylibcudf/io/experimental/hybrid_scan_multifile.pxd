# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.libcudf.io.hybrid_scan_multifile cimport (
    hybrid_scan_multifile as cpp_hybrid_scan_multifile,
)


cdef class HybridScanMultifile:
    cdef unique_ptr[cpp_hybrid_scan_multifile] c_obj
    cdef Stream _stream
    cdef DeviceMemoryResource mr
