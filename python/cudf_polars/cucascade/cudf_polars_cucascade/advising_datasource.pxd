# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource


cdef class AdvisingDatasource(Datasource):
    cdef unique_ptr[datasource] c_obj
    cdef object owner

    @staticmethod
    cdef AdvisingDatasource from_libcucascade(
        unique_ptr[datasource] source, object owner
    )

    cdef datasource* get_datasource(self) except * nogil

    cpdef size_t nbytes(self)
