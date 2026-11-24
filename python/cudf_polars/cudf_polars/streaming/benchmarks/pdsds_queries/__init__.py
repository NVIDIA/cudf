# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB and Polars queries."""

from __future__ import annotations

import polars as pl


def sql_sum(expr: str | pl.Expr) -> pl.Expr:
    """
    Sum that returns NULL for all-null/empty groups, matching SQL SUM semantics.

    Polars sum() returns 0 for all-null or empty groups; SQL returns NULL.
    See https://github.com/rapidsai/cudf/issues/19560.

    Parameters
    ----------
    expr
        Column name or expression to sum. If a string, wraps in ``pl.col``.
        Pass a conditional expression (e.g. ``pl.when(...).then(...).otherwise(None)``)
        to implement SQL ``SUM(CASE WHEN ... END)`` without ``.filter()`` inside
        a groupby, which is not supported on GPU.
    """
    e = pl.col(expr) if isinstance(expr, str) else expr
    return pl.when(e.count() > 0).then(e.sum()).otherwise(None)
