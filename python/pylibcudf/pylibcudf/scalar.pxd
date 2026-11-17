# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.scalar.scalar cimport scalar

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .types cimport DataType


cdef class Scalar:
    cdef unique_ptr[scalar] c_obj
    cdef DataType _data_type

    # Holds a reference to the DeviceMemoryResource used for allocation.
    # Ensures the MR does not get destroyed before this DeviceBuffer. `mr` is
    # needed for deallocation
    cdef DeviceMemoryResource mr

    cdef const scalar* get(self) noexcept nogil

    cpdef DataType type(self)
    cpdef bool is_valid(self, object stream = *)

    @staticmethod
    cdef Scalar empty_like(Column column, object stream, DeviceMemoryResource mr)

    @staticmethod
    cdef Scalar from_libcudf(unique_ptr[scalar] libcudf_scalar, dtype=*)
