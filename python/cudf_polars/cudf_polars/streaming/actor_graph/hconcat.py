# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HConcat helpers for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_streaming.channel_metadata import (
    HashScheme,
    OrderScheme,
    Partitioning,
)

from cudf_polars.dsl.ir import HConcat
from cudf_polars.streaming.actor_graph.utils import (
    maybe_remap_partitioning,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.channel_metadata import ChannelMetadata
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.actor_graph.utils import PartitioningScheme


def _combine_hconcat_scheme(
    schemes: Sequence[PartitioningScheme],
) -> PartitioningScheme:
    schemes = tuple(scheme for scheme in schemes if scheme is not None)
    if not schemes:
        return None
    if all(scheme == "inherit" for scheme in schemes):
        return "inherit"
    if any(scheme == "inherit" for scheme in schemes):
        return None

    hash_schemes = tuple(scheme for scheme in schemes if isinstance(scheme, HashScheme))
    order_schemes = tuple(
        scheme for scheme in schemes if isinstance(scheme, OrderScheme)
    )
    if hash_schemes and order_schemes:
        return None
    if hash_schemes:
        first = hash_schemes[0]
        return first if all(scheme == first for scheme in hash_schemes) else None
    if order_schemes:
        orderings = tuple(
            ordering for scheme in order_schemes for ordering in scheme.orderings
        )
        return OrderScheme(orderings) if orderings else None
    return None


def build_hconcat_partitioning(
    ir: IR,
    child_metadatas: Sequence[ChannelMetadata],
    context: Context,
) -> Partitioning | None:
    """Build partitioning metadata for an internal, row-aligned HConcat."""
    if not isinstance(ir, HConcat) or not ir.streaming_safe:
        return None

    indices = [
        i for i, metadata in enumerate(child_metadatas) if not metadata.duplicated
    ]
    if not indices:
        indices = list(range(len(child_metadatas)))

    remapped: list[Partitioning] = []
    for idx in indices:
        partitioning = maybe_remap_partitioning(
            ir,
            child_metadatas[idx].partitioning,
            child_ir=ir.children[idx],
            context=context,
        )
        if partitioning is not None:
            remapped.append(partitioning)

    inter_rank = _combine_hconcat_scheme(
        tuple(partitioning.inter_rank for partitioning in remapped)
    )
    local = _combine_hconcat_scheme(
        tuple(partitioning.local for partitioning in remapped)
    )
    if inter_rank is None and local == "inherit":
        local = None
    if inter_rank is None and local is None:
        return None
    return Partitioning(inter_rank, local)
