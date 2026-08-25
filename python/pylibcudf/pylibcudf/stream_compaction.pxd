# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.stream_compaction cimport duplicate_keep_option
from pylibcudf.libcudf.types cimport (
    nan_equality,
    null_equality,
    size_type,
)
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table


cpdef Table drop_nulls(
    Table source_table,
    list keys,
    size_type keep_threshold,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table drop_nans(
    Table source_table,
    list keys,
    size_type keep_threshold,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table apply_boolean_mask(
    Table source_table,
    Column boolean_mask,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table apply_deletion_mask(
    Table source_table,
    Column deletion_mask,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table unique(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column distinct_indices(
    Table input,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table stable_distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Table filter(
    Table predicate_table,
    Expression predicate_expr,
    Table filter_table,
    object stream = *,
    DeviceMemoryResource mr = *,
)
