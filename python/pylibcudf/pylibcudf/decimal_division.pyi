# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.round import RoundingMethod
from pylibcudf.scalar import Scalar
from pylibcudf.types import DataType
from pylibcudf.utils import CudaStreamLike

class DecimalIntermediate(IntEnum):
    STANDARD = ...
    WIDE = ...

def divide_decimal(
    lhs: Column,
    rhs: Column,
    output_type: DataType,
    round_method: RoundingMethod = RoundingMethod.HALF_UP,
    intermediate: DecimalIntermediate = DecimalIntermediate.STANDARD,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def divide_decimal_column_scalar(
    lhs: Column,
    rhs: Scalar,
    output_type: DataType,
    round_method: RoundingMethod = RoundingMethod.HALF_UP,
    intermediate: DecimalIntermediate = DecimalIntermediate.STANDARD,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def divide_decimal_scalar_column(
    lhs: Scalar,
    rhs: Column,
    output_type: DataType,
    round_method: RoundingMethod = RoundingMethod.HALF_UP,
    intermediate: DecimalIntermediate = DecimalIntermediate.STANDARD,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
