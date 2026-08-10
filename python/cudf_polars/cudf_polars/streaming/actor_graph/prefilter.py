# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime planning helpers for optional prefilters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import pylibcudf as plc
from cudf_streaming import BloomFilter
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import TableChunk
from pylibcudf.hashing import LIBCUDF_DEFAULT_HASH_SEED
from rapidsmpf.streaming.core.message import Message

from cudf_polars.streaming.actor_graph.utils import (
    ChunkStore,
    recv_metadata,
    send_metadata,
    shutdown_channels_on_error,
)
from cudf_polars.streaming.filter_hint import (
    ExternalDomain,
    JoinInputDomain,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.containers import DataType
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.streaming.actor_graph.utils import TableSizeStats
    from cudf_polars.streaming.filter_hint import JoinSide, Prefilter


def estimate_bytes(dtypes: Sequence[DataType], row_count: int) -> int | None:
    """
    Estimate the byte count of a table containing the given datatypes.

    Parameters
    ----------
    dtypes
        Types of columns in the table.
    row_count
        Estimated total number of rows.

    Returns
    -------
    Estimated table size in bytes, or ``None`` if any dtype is not fixed width.
    """
    if not all(plc.traits.is_fixed_width(dtype.plc_type) for dtype in dtypes):
        return None

    return int(
        # Just assume everything has a validity mask
        row_count * sum(plc.types.size_of(dtype.plc_type) + 1 / 8 for dtype in dtypes)
    )


@dataclass(frozen=True, slots=True)
class PrefilterDecision:
    """Runtime decision for one optional prefilter."""

    method: Literal["skip", "bloom", "broadcast_semi_join"]
    reason: str
    target_bytes: int
    domain_rows: int | None
    estimated_cardinality: int | None = None
    bloom_bytes: int | None = None
    exact_bytes: int | None = None

    def trace(self, prefilter: Prefilter) -> dict[str, str | int | None]:
        """Return serializable actor-trace information."""
        result = asdict(self)
        result["target_side"] = prefilter.target_side
        if isinstance(prefilter.domain, JoinInputDomain):
            result["domain_side"] = prefilter.domain.side
        else:
            assert isinstance(prefilter.domain, ExternalDomain)
            result["domain"] = "external"
        return result


def project_key_chunk(
    context: Context, chunk: TableChunk, indices: Iterable[int]
) -> TableChunk:
    """Copy selected columns into an owning key chunk."""
    columns = chunk.table_view().columns()
    key_table = plc.Table([columns[index] for index in indices]).copy(
        stream=chunk.stream, mr=context.br().device_mr
    )
    return TableChunk.from_pylibcudf_table(
        key_table,
        chunk.stream,
        exclusive_view=True,
        br=context.br(),
    )


async def buffer_and_project_keys(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_keys: Channel[TableChunk],
    ch_replay: Channel[TableChunk],
    indices: Iterable[int],
) -> None:
    """
    Project owning key chunks while spill-buffering an input for replay.

    The key channel is produced in full before replay begins. Its consumer must
    therefore run concurrently with this coroutine.
    """
    chunks = ChunkStore(context)
    try:
        async with shutdown_channels_on_error(context, ch_in, ch_keys, ch_replay):
            metadata = await recv_metadata(ch_in, context)
            key_metadata = ChannelMetadata(
                local_count=metadata.local_count,
                partitioning=None,
                duplicated=metadata.duplicated,
            )
            await send_metadata(ch_replay, context, metadata)
            await send_metadata(ch_keys, context, key_metadata)
            indices = tuple(indices)
            while (msg := await ch_in.recv(context)) is not None:
                sequence_number = msg.sequence_number
                chunk = await TableChunk.from_message(
                    msg, br=context.br()
                ).make_available_or_wait(context, net_memory_delta=0)
                key_chunk = project_key_chunk(context, chunk, indices)
                chunks.insert(Message(sequence_number, chunk))
                await ch_keys.send(context, Message(sequence_number, key_chunk))

            await ch_keys.drain(context)
            for msg in chunks:
                await ch_replay.send(context, msg)
            await ch_replay.drain(context)
    finally:
        chunks.clear()


async def count_rows_passthrough(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    trace_stats: dict[str, Any],
    row_count_key: str,
) -> None:
    """Forward a table-chunk channel while recording its row count."""
    async with shutdown_channels_on_error(context, ch_in, ch_out):
        metadata = await recv_metadata(ch_in, context)
        await send_metadata(ch_out, context, metadata)
        row_count = 0
        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(msg, br=context.br())
            row_count += chunk.shape[0]
            await ch_out.send(context, Message(msg.sequence_number, chunk))
        trace_stats[row_count_key] = row_count
        await ch_out.drain(context)


class PrefilterExecution:
    """Channels and actor tasks used to apply one or more prefilters."""

    def __init__(self, context: Context) -> None:
        self.context = context
        self.tasks: list[Coroutine[Any, Any, None]] = []
        self.channels: list[Channel[Any]] = []

    def add_task(self, task: Coroutine[Any, Any, None]) -> None:
        """Add an actor task to the prefilter execution."""
        self.tasks.append(task)

    def add_channel(self, channel: Channel[Any]) -> None:
        """Register an auxiliary channel for shutdown on failure."""
        self.channels.append(channel)


class JoinPrefilterExecution(PrefilterExecution):
    """Channels and actor tasks used to apply prefilters before a join."""

    def __init__(
        self,
        context: Context,
        ch_left: Channel[TableChunk],
        ch_right: Channel[TableChunk],
    ) -> None:
        super().__init__(context)
        self.source_inputs = {"left": ch_left, "right": ch_right}
        self.join_inputs = dict(self.source_inputs)
        self.buffered_domains: set[JoinSide] = set()

    def buffer_domain(
        self,
        side: JoinSide,
        indices: Iterable[int],
    ) -> Channel[TableChunk]:
        """Buffer one original input and return its owning key channel."""
        if side in self.buffered_domains:
            raise ValueError(f"Join input {side!r} is already a prefilter domain")

        ch_keys: Channel[TableChunk] = self.context.create_channel()
        ch_replay: Channel[TableChunk] = self.context.create_channel()
        self.tasks.append(
            buffer_and_project_keys(
                self.context,
                self.source_inputs[side],
                ch_keys,
                ch_replay,
                indices,
            )
        )
        self.channels.extend((ch_keys, ch_replay))
        self.join_inputs[side] = ch_replay
        self.buffered_domains.add(side)
        return ch_keys

    def replace_join_input(
        self,
        side: JoinSide,
        channel: Channel[TableChunk],
    ) -> None:
        """Replace one join-facing input with a prefilter output channel."""
        self.join_inputs[side] = channel
        self.channels.append(channel)

    @property
    def left(self) -> Channel[TableChunk]:
        """Current left join input."""
        return self.join_inputs["left"]

    @property
    def right(self) -> Channel[TableChunk]:
        """Current right join input."""
        return self.join_inputs["right"]


def add_bloom_prefilter(
    context: Context,
    comm: Communicator,
    bloom_bytes: int,
    execution: PrefilterExecution,
    target_indices: Iterable[int],
    ch_domain_keys: Channel[TableChunk],
    ch_target: Channel[TableChunk],
    ch_filtered: Channel[TableChunk],
    collective_id: int,
    trace_stats: dict[str, Any] | None,
) -> None:
    """Add the channels and actors for an approximate Bloom prefilter."""
    bloom = BloomFilter(
        context,
        comm,
        LIBCUDF_DEFAULT_HASH_SEED,
        bloom_bytes,
    )
    ch_filter = context.create_channel()
    execution.add_channel(ch_filter)
    execution.add_task(
        bloom.build(
            context,
            ch_domain_keys,
            ch_filter,
            collective_id,
        )
    )
    ch_apply_input = ch_target
    ch_apply_output = ch_filtered
    if trace_stats is not None:
        ch_counted_input: Channel[TableChunk] = context.create_channel()
        ch_raw_output: Channel[TableChunk] = context.create_channel()
        execution.add_channel(ch_counted_input)
        execution.add_channel(ch_raw_output)
        execution.add_task(
            count_rows_passthrough(
                context,
                ch_target,
                ch_counted_input,
                trace_stats,
                "input_rows",
            )
        )
        execution.add_task(
            count_rows_passthrough(
                context,
                ch_raw_output,
                ch_filtered,
                trace_stats,
                "output_rows",
            )
        )
        ch_apply_input = ch_counted_input
        ch_apply_output = ch_raw_output
    execution.add_task(
        bloom.apply(
            context,
            ch_filter,
            ch_apply_input,
            ch_apply_output,
            target_indices,
        )
    )


def estimate_cardinality(stats: TableSizeStats) -> int | None:
    """Extrapolate sampled distinct count to the estimated full row count."""
    if stats.total_rows == 0:
        return 0
    if stats.cardinality is None or stats.cardinality.row_count == 0:
        return None
    return min(
        stats.total_rows,
        math.ceil(
            stats.cardinality.distinct_count
            * stats.total_rows
            / stats.cardinality.row_count
        ),
    )


def estimate_bloom_filter_bytes(
    cardinality: int,
    desired_false_positive_rate: float = 0.1,
) -> int:
    """Estimate Bloom-filter bytes for the block-split policy."""
    if cardinality < 0:
        raise ValueError("cardinality must be non-negative")
    if not 0 < desired_false_positive_rate < 1:
        raise ValueError("false_positive_rate must be between zero and one")
    if cardinality == 0:
        return 0
    # TODO: cuco could offer this as a static utility on the policy
    # Then we wouldn't have to hardcode these magic numbers.
    bits = (
        -8  # number of fingerprint bits
        * cardinality
        / math.log(1 - desired_false_positive_rate ** (1 / 8))
    )
    return math.ceil(bits / 8)


def choose_prefilter(
    prefilter: Prefilter,
    target: TableSizeStats,
    domain: TableSizeStats | None,
    *,
    target_requires_redistribution: bool,
    broadcast_limit: int,
    bloom_filter_max_size: int,
) -> PrefilterDecision:
    """Choose whether one join prefilter is eligible to be applied."""
    domain_rows = None if domain is None else domain.total_rows
    if not target_requires_redistribution:
        return PrefilterDecision(
            "skip",
            "target_not_redistributed",
            target.total_size,
            domain_rows,
        )
    if domain is None:
        raise ValueError("A redistributed target requires domain statistics")
    if (
        isinstance(prefilter.domain, JoinInputDomain)
        and prefilter.target_side == prefilter.domain.side
    ):
        return PrefilterDecision(
            "skip",
            "same_input",
            target.total_size,
            domain_rows,
        )

    return choose_prefilter_method(
        prefilter.domain_on,
        target,
        domain,
        broadcast_limit=broadcast_limit,
        bloom_filter_max_size=bloom_filter_max_size,
    )


def choose_prefilter_method(
    domain_on: Sequence[NamedExpr],
    target: TableSizeStats,
    domain: TableSizeStats,
    *,
    broadcast_limit: int,
    bloom_filter_max_size: int,
) -> PrefilterDecision:
    """Choose the implementation for an eligible prefilter."""
    cardinality = estimate_cardinality(domain)
    if cardinality is None:
        return PrefilterDecision(
            "skip",
            "missing_cardinality",
            target.total_size,
            domain.total_rows,
        )
    if cardinality == 0:
        return PrefilterDecision(
            "skip",
            "zero_cardinality",
            target.total_size,
            domain.total_rows,
            estimated_cardinality=0,
            bloom_bytes=0,
            exact_bytes=0,
        )

    bloom_bytes = max(
        32,
        BloomFilter.aligned_size(estimate_bloom_filter_bytes(cardinality)),
    )
    exact_bytes = estimate_bytes(
        tuple(key.value.dtype for key in domain_on),
        domain.total_rows,
    )
    if bloom_bytes <= min(bloom_filter_max_size, target.total_size):
        return PrefilterDecision(
            "bloom",
            "bloom_fits",
            target.total_size,
            domain.total_rows,
            estimated_cardinality=cardinality,
            bloom_bytes=bloom_bytes,
            exact_bytes=exact_bytes,
        )
    if exact_bytes is not None and exact_bytes <= min(
        broadcast_limit, target.total_size
    ):
        return PrefilterDecision(
            "broadcast_semi_join",
            "exact_domain_fits",
            target.total_size,
            domain.total_rows,
            estimated_cardinality=cardinality,
            bloom_bytes=bloom_bytes,
            exact_bytes=exact_bytes,
        )
    return PrefilterDecision(
        "skip",
        "no_viable_filter",
        target.total_size,
        domain.total_rows,
        estimated_cardinality=cardinality,
        bloom_bytes=bloom_bytes,
        exact_bytes=exact_bytes,
    )
