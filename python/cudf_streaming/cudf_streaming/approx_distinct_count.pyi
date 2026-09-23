# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message

class CardinalityEstimate:
    @classmethod
    def from_message(
        cls: type[Self], message: Message[Self]
    ) -> CardinalityEstimate: ...
    @property
    def row_count(self) -> int: ...
    @property
    def distinct_count(self) -> int: ...

class CardinalityEstimator:
    def __init__(
        self,
        ctx: Context,
        comm: Communicator,
        tag: int,
        precision: int = 14,
    ) -> None: ...
    @property
    def comm(self) -> Communicator: ...
    @property
    def tag(self) -> int: ...
    @property
    def precision(self) -> int: ...
    async def estimate(
        self,
        ctx: Context,
        ch_in: Channel[TableChunk],
        ch_out: Channel[CardinalityEstimate],
        ch_sampled: Channel[TableChunk] | None = None,
        column_indices: tuple[int, ...] = (),
    ) -> None: ...
