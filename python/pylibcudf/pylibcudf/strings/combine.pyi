# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

class SeparatorOnNulls(IntEnum):
    YES = ...
    NO = ...

class OutputIfEmptyList(IntEnum):
    EMPTY_STRING = ...
    NULL_ELEMENT = ...

def concatenate(
    strings_columns: Table,
    separator: Column | Scalar,
    narep: Scalar | None = None,
    col_narep: Scalar | None = None,
    separate_nulls: SeparatorOnNulls = SeparatorOnNulls.YES,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def join_strings(
    input: Column,
    separator: Scalar,
    narep: Scalar,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def join_list_elements(
    lists_strings_column: Column,
    separator: Column | Scalar,
    separator_narep: Scalar,
    string_narep: Scalar,
    separate_nulls: SeparatorOnNulls,
    empty_list_policy: OutputIfEmptyList,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
