# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime as dt

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(scope="module")
def frame() -> pl.LazyFrame:
    """Sorted datetime rows with interleaved groups."""
    start = dt.datetime(2026, 1, 5, 9, 30)
    return pl.LazyFrame(
        {
            "time": [start + dt.timedelta(minutes=i) for i in range(120)],
            "group": ["a", "b", None] * 40,
            "group_id": [0, 1, None] * 40,
            "price": range(120),
            "size": [10, 20, 30, 40] * 30,
            "bid": [10.0, 11.0, 12.0, 13.0] * 30,
            "ask": [10.5, 11.5, 12.5, 13.5] * 30,
            "bsize": [100, 200, 300] * 40,
            "asize": [150, 250, 350] * 40,
        }
    ).set_sorted("time")


@pytest.mark.parametrize("every", ["1m", "5m", "10m"])
@pytest.mark.parametrize("group_by", [None, "group"])
def test_supported_tumbling_windows(
    engine: pl.GPUEngine,
    frame: pl.LazyFrame,
    every: str,
    group_by: str | None,
) -> None:
    q = frame.group_by_dynamic(
        "time", every=every, closed="left", group_by=group_by
    ).agg(
        (
            (pl.col("bsize") * pl.col("bid") + pl.col("asize") * pl.col("ask"))
            / (pl.col("asize") + pl.col("bsize"))
        )
        .mean()
        .alias("avg_liq_weighted_mid"),
        (pl.col("ask") - pl.col("bid")).mean().alias("avg_spread"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").sum(),
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_slice(engine: pl.GPUEngine, frame: pl.LazyFrame) -> None:
    q = (
        frame.group_by_dynamic("time", every="10m", group_by="group_id")
        .agg(pl.col("price").sum())
        .head(4)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"every": "5m", "period": "10m"},
        {"every": "5m", "offset": "1m"},
        {"every": "5m", "closed": "right"},
        {"every": "5m", "closed": "both"},
        {"every": "5m", "label": "right"},
        {"every": "5m", "include_boundaries": True},
        {"every": "5m", "start_by": "datapoint"},
        {"every": "1mo"},
    ],
)
def test_unsupported_options(
    in_memory_engine: pl.GPUEngine,
    frame: pl.LazyFrame,
    kwargs: dict,
) -> None:
    q = frame.group_by_dynamic("time", **kwargs).agg(pl.col("price").sum())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_integer_index_unsupported(in_memory_engine: pl.GPUEngine) -> None:
    q = (
        pl.LazyFrame({"index": range(10), "value": range(10)})
        .group_by_dynamic("index", every="2i")
        .agg(pl.col("value").sum())
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
