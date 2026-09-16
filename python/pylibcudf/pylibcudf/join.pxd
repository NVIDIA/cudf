# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf cimport join as cpp_join
from pylibcudf.libcudf.types cimport null_equality
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table


cpdef tuple[Column, Column] inner_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef tuple[Column, Column] left_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef tuple[Column, Column] full_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column left_semi_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column left_anti_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Table cross_join(
    Table left, Table right, object stream = *, DeviceMemoryResource mr=*
)

cpdef tuple[Column, Column] conditional_inner_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef tuple[Column, Column] conditional_left_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef tuple[Column, Column] conditional_full_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column conditional_left_semi_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column conditional_left_anti_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef tuple[Column, Column] mixed_inner_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef tuple[Column, Column] mixed_left_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef tuple[Column, Column] mixed_full_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column mixed_left_semi_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column mixed_left_anti_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cdef class FilteredJoin:
    cdef unique_ptr[cpp_join.filtered_join] c_obj
