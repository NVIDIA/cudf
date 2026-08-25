# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.expressions import Expression
from pylibcudf.table import Table
from pylibcudf.types import NanEquality, NullEquality
from pylibcudf.utils import CudaStreamLike

class DuplicateKeepOption(IntEnum):
    KEEP_ANY = ...
    KEEP_FIRST = ...
    KEEP_LAST = ...
    KEEP_NONE = ...

def drop_nulls(
    source_table: Table,
    keys: list[int],
    keep_threshold: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def drop_nans(
    source_table: Table,
    keys: list[int],
    keep_threshold: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def apply_retention_mask(
    source_table: Table,
    retention_mask: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def apply_boolean_mask(
    source_table: Table,
    boolean_mask: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def apply_deletion_mask(
    source_table: Table,
    deletion_mask: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def unique(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def distinct(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def distinct_indices(
    input: Table,
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def stable_distinct(
    input: Table,
    keys: list[int],
    keep: DuplicateKeepOption,
    nulls_equal: NullEquality,
    nans_equal: NanEquality,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
def filter(
    predicate_table: Table,
    predicate_expr: Expression,
    filter_table: Table,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
