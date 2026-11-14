# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cudf_polars_cucascade.libcucascade.column_admission cimport (
    column_stats as cpp_column_stats,
)


cdef class ColumnStats:
    cdef cpp_column_stats c_obj

    @staticmethod
    cdef ColumnStats from_libcucascade(cpp_column_stats stats)
