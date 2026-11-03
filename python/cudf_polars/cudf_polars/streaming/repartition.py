# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Repartitioning Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import IR

if TYPE_CHECKING:
    from cudf_polars.typing import Schema


class Repartition(IR):
    """
    Repartition a DataFrame.

    Notes
    -----
    Repartitioning means that we are not modifying any
    data, nor are we reordering or shuffling rows. We
    are only changing the overall partition count. For
    now, we only support an N -> [1...N] repartitioning
    (inclusive). The output partition count is tracked
    separately using PartitionInfo.
    """

    __slots__ = ()
    _non_child = ("schema",)
    _n_non_child_args = 0

    def __init__(self, schema: Schema, df: IR):
        self.schema = schema
        self._non_child_args = ()
        self.children = (df,)
