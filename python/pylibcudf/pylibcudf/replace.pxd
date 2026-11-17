# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from pylibcudf.libcudf.replace cimport replace_policy
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar

ctypedef fused ReplacementType:
    Column
    Scalar
    replace_policy
    # Allowing object is a workaround for
    # https://github.com/cython/cython/issues/5984. See the implementation of
    # replace_nulls for details.
    object


cpdef Column replace_nulls(
    Column source_column,
    ReplacementType replacement,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column find_and_replace_all(
    Column source_column,
    Column values_to_replace,
    Column replacement_values,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column clamp(
    Column source_column,
    Scalar lo,
    Scalar hi,
    Scalar lo_replace=*,
    Scalar hi_replace=*,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column normalize_nans_and_zeros(
    Column source_column,
    bool inplace=*,
    object stream = *,
    DeviceMemoryResource mr = *,
)
