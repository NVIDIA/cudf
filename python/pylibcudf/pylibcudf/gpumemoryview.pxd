# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint64_t, uintptr_t

cdef class gpumemoryview:
    # TODO: Eventually probably want to make this opaque, but for now it's fine
    # to treat this object as something like a POD struct
    cdef readonly uintptr_t ptr
    cdef readonly object obj
    cdef readonly dict cai
    cdef readonly uint64_t nbytes
    cdef object __weakref__
