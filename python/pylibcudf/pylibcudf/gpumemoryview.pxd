# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint64_t, uintptr_t
from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.utilities.device_buffer cimport byte, device_buffer


cdef class _CudaDeviceBuffer:
    cdef unique_ptr[device_buffer[byte]] c_obj
    cdef object stream
    cdef object mr


cdef gpumemoryview _from_cuda_device_buffer(
    unique_ptr[device_buffer[byte]] buf, object stream, object mr
)

cdef class gpumemoryview:
    # TODO: Eventually probably want to make this opaque, but for now it's fine
    # to treat this object as something like a POD struct
    cdef readonly uintptr_t ptr
    cdef readonly object obj
    cdef readonly dict cai
    cdef readonly uint64_t nbytes
    cdef object __weakref__
