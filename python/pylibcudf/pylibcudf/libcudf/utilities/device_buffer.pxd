# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "<cstddef>" namespace "std":
    cdef cppclass byte:
        pass

cdef extern from "<cuda/buffer>" namespace "cuda" nogil:
    cdef cppclass device_buffer[T]:
        T* data()
        size_t size()
