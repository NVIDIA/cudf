# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.libcudf.expressions cimport (
    ast_operator,
    expression,
    table_reference,
)

from .scalar cimport Scalar


cdef class Expression:
    cdef unique_ptr[expression] c_obj

cdef class Literal(Expression):
    # Hold on to input scalar so it doesn't get gc'ed
    cdef Scalar scalar

cdef class ColumnReference(Expression):
    pass

cdef class Operation(Expression):
    # Hold on to the input expressions so
    # they don't get gc'ed
    cdef Expression third
    cdef Expression right
    cdef Expression left

cdef class ColumnNameReference(Expression):
    pass
