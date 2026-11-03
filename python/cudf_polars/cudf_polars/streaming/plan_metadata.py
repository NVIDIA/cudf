# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata produced while optimizing a query plan."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cudf_polars.streaming.join_filter_pushdown import JoinFilterPushdownDecision

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR

# This will be a union of all the possible operator details.
OperatorDetail = JoinFilterPushdownDecision


@dataclass
class PlanMetadata:
    """Typed details associated with nodes in an optimized IR graph."""

    _operator_details: dict[IR, list[OperatorDetail]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_operator_detail(self, node: IR, detail: OperatorDetail) -> None:
        """Associate one single-shot detail payload with an operator."""
        details = self._operator_details[node]
        for existing in details:
            if type(existing) is type(detail):
                if existing != detail:
                    raise ValueError(
                        "Conflicting operator details for structurally equal nodes: "
                        f"{existing!r} != {detail!r}"
                    )
                return
        details.append(detail)

    def operator_details(self, node: IR) -> tuple[OperatorDetail, ...]:
        """Return detail payloads associated with an operator."""
        return tuple(self._operator_details.get(node, ()))
