# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for replacing nodes in a DAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from cudf_polars.dsl.traversal import CachingVisitor, reuse_if_unchanged
from cudf_polars.typing import NodeT, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cudf_polars.typing import GenericTransformer

__all__ = ["replace"]


class State(Generic[NodeT], TypedDict):
    """
    State used when replacing nodes in expressions.

    Parameters
    ----------
    replacements
        Mapping from nodes to be replaced to their replacements.
        This state is generic over the type of these nodes.
    """

    replacements: Mapping[NodeT, NodeT]


def _replace(node: NodeT, fn: GenericTransformer[NodeT, NodeT, State]) -> NodeT:
    try:
        return fn.state["replacements"][node]
    except KeyError:
        return reuse_if_unchanged(node, fn)


def replace(nodes: Sequence[NodeT], replacements: Mapping[NodeT, NodeT]) -> list[NodeT]:
    """
    Replace nodes in expressions.

    Parameters
    ----------
    nodes
        Sequence of nodes to perform replacements in.
    replacements
        Mapping from nodes to be replaced to their replacements.

    Returns
    -------
    list
        Of nodes with replacements performed.
    """
    mapper: GenericTransformer[NodeT, NodeT, State] = CachingVisitor(
        _replace, state={"replacements": replacements}
    )
    return [mapper(node) for node in nodes]
