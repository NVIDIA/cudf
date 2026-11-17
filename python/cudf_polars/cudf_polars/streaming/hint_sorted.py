# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming IR node for sortedness hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import IR

if TYPE_CHECKING:
    from cudf_polars.typing import Schema


type HintSortedOptions = tuple[tuple[str, ...], tuple[bool, ...], tuple[bool, ...]]


class HintSorted(IR):
    """Attach ordering metadata for data known to be sorted."""

    __slots__ = ("options",)
    _non_child = ("schema", "options")
    _n_non_child_args = 2
    options: HintSortedOptions

    def __init__(self, schema: Schema, options: HintSortedOptions, df: IR):
        self.schema = schema
        self.options = options
        self._non_child_args = (schema, options)
        self.children = (df,)
