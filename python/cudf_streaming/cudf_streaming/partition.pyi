# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.chunks.partition import (
    PartitionMapChunk,
    PartitionVectorChunk,
)
from rapidsmpf.streaming.core.actor import CppActor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

def partition_and_pack(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[PartitionMapChunk],
    columns_to_hash: Iterable[int],
    num_partitions: int,
) -> CppActor: ...
def unpack_and_concat(
    ctx: Context,
    ch_in: Channel[PartitionMapChunk]
    | Channel[PartitionVectorChunk]
    | Channel[PartitionMapChunk | PartitionVectorChunk],
    ch_out: Channel[TableChunk],
) -> CppActor: ...
