# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of Index methods."""

import pandas as pd
import pytest
from config import NUM_ROWS, cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_construction(benchmark, N):
    benchmark(cudf.Index, cupy.random.rand(N))


@benchmark_with_object(cls="index", dtype="int", nulls=False)
def bench_sort_values(benchmark, index):
    benchmark(index.sort_values)


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_large_unique_categories_repr(benchmark, N):
    pi = cudf.CategoricalIndex(range(N))
    benchmark(repr, pi)


@pytest.mark.parametrize("N", NUM_ROWS)
@pytest.mark.parametrize("endpoint_type", ["numeric", "datetime-tz"])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.pandas_incompatible
def bench_interval_to_pandas(benchmark, N, endpoint_type, sliced):
    size = N + 2 if sliced else N
    if endpoint_type == "numeric":
        breaks = pd.RangeIndex(size + 1)
    else:
        breaks = pd.date_range(
            "2025-03-08", periods=size + 1, freq="s", tz="US/Eastern"
        )
    intervals = cudf.IntervalIndex.from_breaks(cudf.from_pandas(breaks))
    if sliced:
        intervals = cudf.IntervalIndex(cudf.Series(intervals).iloc[1:-1])

    benchmark(intervals.to_pandas)
