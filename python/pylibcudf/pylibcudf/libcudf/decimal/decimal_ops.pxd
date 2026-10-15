# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar

cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    cdef enum class decimal_rounding_mode:
        HALF_UP
        HALF_EVEN

cdef extern from "cudf/decimal/decimal_ops.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] divide_decimal(
        const column_view& lhs,
        const column_view& rhs,
        decimal_rounding_mode rounding_mode
    ) except +

    cdef unique_ptr[column] divide_decimal(
        const column_view& lhs,
        const scalar& rhs,
        decimal_rounding_mode rounding_mode
    ) except +

    cdef unique_ptr[column] divide_decimal(
        const scalar& lhs,
        const column_view& rhs,
        decimal_rounding_mode rounding_mode
    ) except +
