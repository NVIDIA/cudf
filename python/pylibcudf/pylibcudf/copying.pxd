# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool as cbool
from pylibcudf.libcudf.copying cimport (
    mask_allocation_policy,
    out_of_bounds_policy,
)
from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrTable:
    Table
    Column


ctypedef fused TableOrListOfScalars:
    Table
    # The contents of the list must be validated as Scalars at runtime.
    list


# Need two separate fused types to generate the cartesian product of signatures.
ctypedef fused LeftCopyIfElseOperand:
    Column
    Scalar

ctypedef fused RightCopyIfElseOperand:
    Column
    Scalar


cpdef Table gather(
    Table source_table,
    Column gather_map,
    out_of_bounds_policy bounds_policy,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Table scatter(
    TableOrListOfScalars source,
    Column scatter_map,
    Table target_table,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef ColumnOrTable empty_like(
    ColumnOrTable input, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column allocate_like(
    Column input_column,
    mask_allocation_policy policy,
    size=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column copy_range_in_place(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
    object stream = *,
)

cpdef Column copy_range(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column shift(
    Column input,
    size_type offset,
    Scalar fill_value,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef ColumnOrTable reverse(
    ColumnOrTable input, object stream = *, DeviceMemoryResource mr=*
)

cpdef list slice(ColumnOrTable input, list indices, object stream = *)

cpdef list split(ColumnOrTable input, list splits, object stream = *)

cpdef Column copy_if_else(
    LeftCopyIfElseOperand lhs,
    RightCopyIfElseOperand rhs,
    Column boolean_mask,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Table boolean_mask_scatter(
    TableOrListOfScalars input,
    Table target,
    Column boolean_mask,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Scalar get_element(
    Column input_column,
    size_type index,
    object stream = *,
    DeviceMemoryResource mr=*
)
