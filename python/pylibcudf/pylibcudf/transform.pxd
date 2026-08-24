# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from pylibcudf.libcudf.types cimport bitmask_type, data_type
from pylibcudf.libcudf.types cimport null_aware, output_nullability
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .gpumemoryview cimport gpumemoryview
from .table cimport Table
from .types cimport DataType


cpdef tuple[gpumemoryview, int] nans_to_nulls(
    Column input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column column_nans_to_nulls(
    Column input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column compute_column(
    Table input, Expression expr, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column compute_column_jit(
    Table input, Expression expr, object stream = *, DeviceMemoryResource mr = *
)

cpdef tuple[gpumemoryview, int] bools_to_mask(
    Column input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Column mask_to_bools(
    Py_ssize_t bitmask,
    int begin_bit,
    int end_bit,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column transform(
    inputs,
    str transform_udf,
    DataType output_type,
    bool is_ptx,
    null_aware is_null_aware,
    output_nullability null_policy,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef tuple[Table, Column] encode(
    Table input, object stream = *, DeviceMemoryResource mr = *
)

cpdef Table one_hot_encode(
    Column input_column,
    Column categories,
    object stream = *,
    DeviceMemoryResource mr = *,
)
