# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from pylibcudf.libcudf.types cimport null_equality

from .table cimport Table


cpdef bool tables_equal(
    Table left,
    Table right,
    null_equality nulls_equal=*,
    object stream=*,
)
