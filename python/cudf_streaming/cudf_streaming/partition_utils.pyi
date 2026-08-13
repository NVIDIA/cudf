# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

from pylibcudf.contiguous_split import PackedColumns
from pylibcudf.table import Table

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData
from rmm.pylibrmm.stream import Stream

def partition_and_pack(
    table: Table,
    columns_to_hash: Iterable[int],
    num_partitions: int,
    stream: Stream,
    br: BufferResource,
) -> dict[int, PackedData]: ...
def split_and_pack(
    table: Table,
    splits: Iterable[int],
    stream: Stream,
    br: BufferResource,
) -> dict[int, PackedData]: ...
def unpack_and_concat(
    partitions: Iterable[PackedData],
    stream: Stream,
    br: BufferResource,
) -> Table: ...
def packed_data_from_cudf_packed_columns(
    packed_columns: PackedColumns,
    stream: Stream,
    br: BufferResource,
) -> PackedData: ...
