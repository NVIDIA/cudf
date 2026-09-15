# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actor-local planning state for dynamic joins and optional prefilters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cudf_polars.streaming.filter_hint import (
    ExternalDomain,
    JoinInputDomain,
    JoinWithPrefilter,
)

if TYPE_CHECKING:
    from typing import Any, Self

    from cudf_streaming.channel_metadata import ChannelMetadata
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.streaming.core.channel import Channel

    from cudf_polars.dsl.ir import IR, Join
    from cudf_polars.streaming.actor_graph.prefilter import PrefilterDecision
    from cudf_polars.streaming.actor_graph.utils import TableSizeStats
    from cudf_polars.streaming.filter_hint import Prefilter


@dataclass(slots=True)
class JoinInput:
    """Concrete runtime resources for one input to a dynamic join."""

    node: IR
    channel: Channel[TableChunk]
    metadata: ChannelMetadata
    sample: TableSizeStats | None = None


@dataclass(slots=True)
class PrefilterCandidate:
    """An optional prefilter and the runtime inputs needed to evaluate it."""

    spec: Prefilter
    target: JoinInput
    domain: JoinInput
    cardinality_tag: int
    decision: PrefilterDecision | None = None
    key_channel: Channel[TableChunk] | None = None
    trace: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class JoinPlanningState:
    """Actor-local input and prefilter state for planning a dynamic join."""

    left: JoinInput
    right: JoinInput
    candidates: tuple[PrefilterCandidate, ...] = ()

    @classmethod
    def create(
        cls,
        ir: Join,
        ch_left: Channel[TableChunk],
        ch_right: Channel[TableChunk],
        ch_prefilter_domains: tuple[Channel[TableChunk], ...],
        left_metadata: ChannelMetadata,
        right_metadata: ChannelMetadata,
        prefilter_domain_metadata: tuple[ChannelMetadata, ...],
        cardinality_tags: tuple[int, ...],
    ) -> Self:
        """Create actor-local planning state from a join and its runtime inputs."""
        left = JoinInput(ir.children[0], ch_left, left_metadata)
        right = JoinInput(ir.children[1], ch_right, right_metadata)
        if not isinstance(ir, JoinWithPrefilter):
            if ch_prefilter_domains or prefilter_domain_metadata:
                raise ValueError("A plain Join cannot have prefilter domain inputs")
            return cls(left, right)

        external_inputs = tuple(
            JoinInput(node, channel, metadata)
            for node, channel, metadata in zip(
                ir.children[2:],
                ch_prefilter_domains,
                prefilter_domain_metadata,
                strict=True,
            )
        )
        external_prefilter_count = sum(
            isinstance(prefilter.domain, ExternalDomain) for prefilter in ir.prefilters
        )
        if external_prefilter_count != len(external_inputs):
            raise ValueError("Join prefilters and external domain inputs must align")
        if len(cardinality_tags) < len(ir.prefilters):
            raise ValueError("Each join prefilter requires a cardinality collective ID")

        sides = {"left": left, "right": right}
        external_inputs_iter = iter(external_inputs)
        cardinality_tags_iter = iter(cardinality_tags)
        candidates = []
        for spec in ir.prefilters:
            target = sides[spec.target_side]
            if isinstance(spec.domain, JoinInputDomain):
                domain = sides[spec.domain.side]
            else:
                domain = next(external_inputs_iter)
            candidates.append(
                PrefilterCandidate(
                    spec,
                    target,
                    domain,
                    cardinality_tag=next(cardinality_tags_iter),
                )
            )
        return cls(left, right, tuple(candidates))
