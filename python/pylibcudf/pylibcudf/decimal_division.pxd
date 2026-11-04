# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pylibcudf.libcudf.fixed_point.math_ops cimport decimal_intermediate
from pylibcudf.libcudf.round cimport rounding_method

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cpdef Column divide_decimal(
    Column lhs,
    Column rhs,
    DataType output_type,
    rounding_method round_method = *,
    decimal_intermediate intermediate = *,
    object stream = *,
    DeviceMemoryResource mr = *
)

cpdef Column divide_decimal_column_scalar(
    Column lhs,
    Scalar rhs,
    DataType output_type,
    rounding_method round_method = *,
    decimal_intermediate intermediate = *,
    object stream = *,
    DeviceMemoryResource mr = *
)

cpdef Column divide_decimal_scalar_column(
    Scalar lhs,
    Column rhs,
    DataType output_type,
    rounding_method round_method = *,
    decimal_intermediate intermediate = *,
    object stream = *,
    DeviceMemoryResource mr = *
)
