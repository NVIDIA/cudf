# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io

import rich.console
import rich.pretty

import polars as pl

from cudf_polars.dsl.translate import Translator
from cudf_polars.streaming.base import PartitionInfo


def test_partition_info_rich_repr() -> None:
    console = rich.console.Console(file=io.StringIO())
    pi = PartitionInfo(count=1, partitioned_on=())

    rich.pretty.pprint(pi, console=console)


def test_node_rich_repr() -> None:
    file = io.StringIO()
    console = rich.console.Console(file=file)

    q = pl.LazyFrame({"a": [1, 2, 3]}).select(pl.col("a").sum())
    ir = Translator(
        q._ldf.visit(), engine=pl.GPUEngine(executor="streaming")
    ).translate_ir()

    rich.pretty.pprint(ir, console=console)
    result = file.getvalue()
    assert "Select" in result
    assert "sum" in result
    assert "DataFrameScan" in result
