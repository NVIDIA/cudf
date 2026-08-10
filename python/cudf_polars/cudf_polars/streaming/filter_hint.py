# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logical filter hints for the streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias

from cudf_polars.dsl.ir import IR, Join

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.typing import Schema


JoinSide: TypeAlias = Literal["left", "right"]
HintPlacement: TypeAlias = Literal["join_input", "pushed_down"]


@dataclass(frozen=True, slots=True)
class JoinInputDomain:
    """A prefilter domain provided by an input of its owning join."""

    side: JoinSide


@dataclass(frozen=True, slots=True)
class ExternalDomain:
    """A prefilter domain provided by an additional join input."""


PrefilterDomain: TypeAlias = JoinInputDomain | ExternalDomain


@dataclass(frozen=True, slots=True)
class Prefilter:
    """Description of an optional join prefilter."""

    target_side: JoinSide
    target_on: tuple[NamedExpr, ...]
    domain: PrefilterDomain
    domain_on: tuple[NamedExpr, ...]
    nulls_equal: bool


class JoinWithPrefilter(Join):
    """Lowered join with normalized prefilter descriptors."""

    __slots__ = ("prefilters",)
    _non_child = ("schema", "left_on", "right_on", "options", "prefilters")
    _n_non_child_args = 4

    prefilters: tuple[Prefilter, ...]

    def __init__(
        self,
        schema: Schema,
        left_on: Sequence[NamedExpr],
        right_on: Sequence[NamedExpr],
        options: Any,
        prefilters: Sequence[Prefilter],
        left: IR,
        right: IR,
        *external_domains: IR,
    ):
        self.schema = schema
        self.left_on = tuple(left_on)
        self.right_on = tuple(right_on)
        self.options = options
        self.prefilters = tuple(prefilters)
        self.children = (left, right, *external_domains)
        self._non_child_args = (
            self.left_on,
            self.right_on,
            self.options,
            self.prefilters,
        )

        if not self.prefilters:
            raise ValueError("JoinWithPrefilter requires at least one prefilter")
        external_domain_count = sum(
            isinstance(prefilter.domain, ExternalDomain)
            for prefilter in self.prefilters
        )
        if external_domain_count != len(external_domains):
            raise ValueError(
                "External prefilters and additional JoinWithPrefilter children "
                "must align"
            )

    @classmethod
    def do_evaluate(
        cls,
        left_on: tuple[NamedExpr, ...],
        right_on: tuple[NamedExpr, ...],
        options: Any,
        prefilters: tuple[Prefilter, ...],
        left: DataFrame,
        right: DataFrame,
        *external_domains: DataFrame,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Evaluate the join while ignoring its optional prefilters."""
        del prefilters, external_domains
        return Join.do_evaluate(
            left_on,
            right_on,
            options,
            left,
            right,
            context=context,
        )


class PushdownFilterHint(IR):
    """
    Optional join-key filter placed in a logical plan.

    The first child is the target to filter and the second child, the
    domain, provides the keys to filter against. Applying the filter is
    optional.
    """

    __slots__ = ("domain_on", "nulls_equal", "placement", "target_on")
    _non_child: ClassVar[tuple[str, ...]] = (
        "schema",
        "target_on",
        "domain_on",
        "nulls_equal",
        "placement",
    )
    _n_non_child_args: ClassVar[int] = 4

    target_on: tuple[NamedExpr, ...]
    """Expressions selecting filter keys from the target."""
    domain_on: tuple[NamedExpr, ...]
    """Expressions selecting filter keys from the domain."""
    nulls_equal: bool
    """Whether null key values compare equal."""
    placement: HintPlacement
    """Whether the hint remains at the motivating join input."""

    def __init__(
        self,
        schema: Schema,
        target_on: Sequence[NamedExpr],
        domain_on: Sequence[NamedExpr],
        nulls_equal: bool,  # noqa: FBT001
        placement: HintPlacement,
        target: IR,
        domain: IR,
    ):
        self.schema = schema
        self.target_on = tuple(target_on)
        self.domain_on = tuple(domain_on)
        self.nulls_equal = nulls_equal
        self.placement = placement
        self._non_child_args = (
            self.target_on,
            self.domain_on,
            self.nulls_equal,
            self.placement,
        )
        self.children = (target, domain)

    @classmethod
    def do_evaluate(
        cls,
        target_on: tuple[NamedExpr, ...],
        domain_on: tuple[NamedExpr, ...],
        nulls_equal: bool,  # noqa: FBT001
        placement: HintPlacement,
        target: DataFrame,
        domain: DataFrame,
        *,
        context: IRExecutionContext,
    ) -> DataFrame:
        """Ignore the optional filter and return the target."""
        del placement
        return target
