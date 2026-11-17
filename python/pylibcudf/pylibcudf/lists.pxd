# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.libcudf.types cimport (
    nan_equality, null_equality, null_order, order, size_type
)
from pylibcudf.libcudf.copying cimport out_of_bounds_policy
from pylibcudf.libcudf.lists.combine cimport concatenate_null_policy
from pylibcudf.libcudf.lists.contains cimport duplicate_find_option
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table

ctypedef fused ColumnOrScalar:
    Column
    Scalar

ctypedef fused ColumnOrSizeType:
    Column
    size_type

cpdef Table explode_outer(
    Table,
    size_type explode_column_idx,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column concatenate_rows(
    Table,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column concatenate_list_elements(
    Column,
    concatenate_null_policy null_policy,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column contains(
    Column,
    ColumnOrScalar,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column contains_nulls(
    Column,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column index_of(
    Column,
    ColumnOrScalar,
    duplicate_find_option,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column reverse(
    Column,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column segmented_gather(
    Column,
    Column,
    out_of_bounds_policy bounds_policy=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column extract_list_element(
    Column,
    ColumnOrSizeType,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column count_elements(
    Column,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column sequences(
    Column,
    Column,
    Column steps = *,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column sort_lists(
    Column,
    order,
    null_order,
    bool stable = *,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column difference_distinct(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column have_overlap(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column intersect_distinct(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column union_distinct(
    Column,
    Column,
    null_equality nulls_equal=*,
    nan_equality nans_equal=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column apply_retention_mask(
    Column,
    Column,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column apply_boolean_mask(
    Column,
    Column,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column apply_deletion_mask(
    Column,
    Column,
    object stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column distinct(
    Column,
    null_equality,
    nan_equality,
    object stream = *,
    DeviceMemoryResource mr=*,
)
