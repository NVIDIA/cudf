# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type stubs for channel_metadata module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self

import pylibcudf as plc

from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.streaming.core.message import Message

class HashScheme:
    def __init__(
        self, column_indices: Sequence[int], modulus: int
    ) -> None: ...
    @property
    def column_indices(self) -> tuple[int, ...]: ...
    @property
    def modulus(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

@dataclass(frozen=True, slots=True)
class OrderKey:
    """Sort key: column index, direction, and null ordering."""

    column_index: int
    order: plc.types.Order
    null_order: plc.types.NullOrder

class Ordering:
    def __init__(
        self,
        keys: Sequence[OrderKey],
        boundaries: TableChunk,
        *,
        strict_boundaries: bool = False,
    ) -> None: ...
    @property
    def keys(self) -> tuple[OrderKey, ...]: ...
    @property
    def column_indices(self) -> tuple[int, ...]: ...
    @property
    def strict_boundaries(self) -> bool: ...
    @property
    def num_boundaries(self) -> int: ...
    def get_boundaries(self, br: BufferResource) -> TableChunk: ...
    def with_keys(self, new_keys: Sequence[OrderKey]) -> Ordering: ...
    def boundaries_aligned_with(
        self, other: Ordering, br: BufferResource
    ) -> bool: ...

class OrderScheme:
    def __init__(self, orderings: Sequence[Ordering]) -> None: ...
    @property
    def orderings(self) -> tuple[Ordering, ...]: ...
    def __repr__(self) -> str: ...

PartitioningSpecValue = HashScheme | OrderScheme | None | Literal["inherit"]

class Partitioning:
    def __init__(
        self,
        inter_rank: PartitioningSpecValue = None,
        local: PartitioningSpecValue = None,
    ) -> None: ...
    @property
    def inter_rank(self) -> PartitioningSpecValue: ...
    @property
    def local(self) -> PartitioningSpecValue: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class ChannelMetadata:
    def __init__(
        self,
        local_count: int,
        *,
        partitioning: Partitioning | None = None,
        duplicated: bool = False,
    ) -> None: ...
    @classmethod
    def from_message(
        cls: type[Self], message: Message[Self]
    ) -> ChannelMetadata: ...
    def into_message(
        self, sequence_number: int, message: Message[Self]
    ) -> None: ...
    @property
    def local_count(self) -> int: ...
    @property
    def partitioning(self) -> Partitioning: ...
    @property
    def duplicated(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
