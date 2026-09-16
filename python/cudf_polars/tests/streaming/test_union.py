# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_union_shared_fanout_no_deadlock(streaming_engine):
    # union actor can deadlock when input branches share a fanout.
    # See https://github.com/NVIDIA/cudf/issues/21750
    n = 100
    df = pl.LazyFrame({"key": list(range(50)) * (n // 50), "val": list(range(n))})
    gb = df.group_by("key").agg(pl.col("val").sum())
    project = df.select("key", "val")
    q = pl.concat([gb, project])
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_sort_slice_over_union_of_duplicated_streams(streaming_engine):
    # Sort+head over a concat of two group-by branches.
    lf1 = (
        pl.LazyFrame({"name": ["alice"], "score": [1.0]})
        .group_by("name")
        .agg(pl.col("score").sum())
    )
    lf2 = (
        pl.LazyFrame({"name": ["bob"], "score": [2.0]})
        .group_by("name")
        .agg(pl.col("score").sum())
    )
    q = pl.concat([lf1, lf2]).sort("score").head(10)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)
