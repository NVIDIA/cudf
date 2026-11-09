# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.io.datasource cimport datasource

from cudf_polars_cucascade.exception_handler cimport libcucascade_exception_handler


cdef extern from "cucascade_datasource.hpp" namespace "cudf_polars_cucascade" nogil:

    cdef cppclass cucascade_datasource(datasource):
        size_t size() except +libcucascade_exception_handler const
