# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.utils.naming import names_to_indices


def test_names_to_indices_concrete_prefix() -> None:
    dtype = DataType(pl.Int64())
    schema = {"a": dtype, "b": dtype, "c": dtype}
    names = (
        expr.NamedExpr("a_alias", expr.Col(dtype, "a")),
        "b",
        expr.NamedExpr("computed", expr.Literal(dtype, 1)),
        expr.NamedExpr("c_alias", expr.Col(dtype, "c")),
    )

    assert names_to_indices(names, schema, concrete_prefix=True) == (0, 1)
