# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuDF Streaming library."""

# If libcudf_streaming was installed as a wheel, request it to load the library
# symbols. Otherwise, assume the library is on a system path that ld can find.
try:
    import libcudf_streaming
except ModuleNotFoundError:
    pass
else:
    libcudf_streaming.load_library()
    del libcudf_streaming

from cudf_streaming.approx_distinct_count import (
    CardinalityEstimate,
    CardinalityEstimator,
)
from cudf_streaming.bloom_filter import BloomFilter
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Ordering,
    OrderKey,
    OrderScheme,
    Partitioning,
)
from cudf_streaming.partition import (
    partition_and_pack as actor_partition_and_pack,
    unpack_and_concat as actor_unpack_and_concat,
)
from cudf_streaming.partition_utils import (
    packed_data_from_cudf_packed_columns,
    partition_and_pack,
    partition_and_pack_cost,
    split_and_pack,
    split_and_pack_cost,
    unpack_and_concat,
    unpack_and_concat_cost,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)

__all__ = [
    "BloomFilter",
    "CardinalityEstimate",
    "CardinalityEstimator",
    "ChannelMetadata",
    "HashScheme",
    "OrderKey",
    "OrderScheme",
    "Ordering",
    "Partitioning",
    "TableChunk",
    "actor_partition_and_pack",
    "actor_unpack_and_concat",
    "make_table_chunks_available_or_wait",
    "packed_data_from_cudf_packed_columns",
    "partition_and_pack",
    "partition_and_pack_cost",
    "split_and_pack",
    "split_and_pack_cost",
    "unpack_and_concat",
    "unpack_and_concat_cost",
]
