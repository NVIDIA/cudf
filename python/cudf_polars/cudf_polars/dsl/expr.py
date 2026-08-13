# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
DSL nodes for the polars expression language.

An expression node is a function, `DataFrame -> Column`.

The evaluation context is provided by a LogicalPlan node, and can
affect the evaluation rule as well as providing the dataframe input.
In particular, the interpretation of the expression language in a
`GroupBy` node is groupwise, rather than whole frame.
"""

from __future__ import annotations

from cudf_polars.dsl.expressions.aggregation import Agg, Item
from cudf_polars.dsl.expressions.base import (
    Col,
    ColRef,
    ErrorExpr,
    Expr,
    NamedExpr,
)
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.dsl.expressions.boolean import BooleanFunction
from cudf_polars.dsl.expressions.datetime import TemporalFunction
from cudf_polars.dsl.expressions.literal import Literal, LiteralColumn
from cudf_polars.dsl.expressions.rolling import (
    FixedSizeRollingWindow,
    GroupedWindow,
    RollingWindow,
)
from cudf_polars.dsl.expressions.selection import Filter, Gather
from cudf_polars.dsl.expressions.slicing import Slice
from cudf_polars.dsl.expressions.sorting import Sort, SortBy
from cudf_polars.dsl.expressions.string import StringFunction
from cudf_polars.dsl.expressions.struct import StructFunction
from cudf_polars.dsl.expressions.ternary import Ternary
from cudf_polars.dsl.expressions.unary import Cast, Len, UnaryFunction

__all__ = [
    "Agg",
    "BinOp",
    "BooleanFunction",
    "Cast",
    "Col",
    "ColRef",
    "ErrorExpr",
    "Expr",
    "Filter",
    "FixedSizeRollingWindow",
    "Gather",
    "GroupedWindow",
    "Item",
    "Len",
    "Literal",
    "LiteralColumn",
    "NamedExpr",
    "RollingWindow",
    "Slice",
    "Sort",
    "SortBy",
    "StringFunction",
    "StructFunction",
    "TemporalFunction",
    "Ternary",
    "UnaryFunction",
]
