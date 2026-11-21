# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.libcudf.binaryop cimport binary_operator
from pylibcudf.libcudf.types cimport nan_semantics
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType

# Need two separate fused types to generate the cartesian product of signatures.
ctypedef fused LeftBinaryOperand:
    Column
    Scalar

ctypedef fused RightBinaryOperand:
    Column
    Scalar


cpdef Column binary_operation(
    LeftBinaryOperand lhs,
    RightBinaryOperand rhs,
    binary_operator op,
    DataType output_type,
    nan_semantics semantics = *,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef bool is_supported_operation(
    DataType out,
    DataType lhs,
    DataType rhs,
    binary_operator op
)
