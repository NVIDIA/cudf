# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of Series methods."""

import pytest
from config import NUM_ROWS, cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_construction(benchmark, N):
    benchmark(cudf.Series, cupy.random.rand(N))


@benchmark_with_object(cls="series", dtype="int")
def bench_sort_values(benchmark, series):
    benchmark(series.sort_values)


@benchmark_with_object(cls="series", dtype="int")
@pytest.mark.parametrize("n", [10])
def bench_series_nsmallest(benchmark, series, n):
    benchmark(series.nsmallest, n)


@benchmark_with_object(cls="series", dtype="int", nulls=False)
def bench_series_cp_asarray(benchmark, series):
    benchmark(cupy.asarray, series)


@benchmark_with_object(cls="series", dtype="int", nulls=False)
@pytest.mark.pandas_incompatible
def bench_to_cupy(benchmark, series):
    benchmark(lambda: series.values)


@benchmark_with_object(cls="series", dtype="int", nulls=False)
def bench_series_values(benchmark, series):
    benchmark(lambda: series.values)


@pytest.mark.parametrize("N", NUM_ROWS)
@pytest.mark.parametrize(
    "start, stop",
    [
        pytest.param(1, 3, id="positive-bounds"),
        pytest.param(-3, None, id="negative-start"),
        pytest.param(-4, -1, id="negative-bounds"),
        pytest.param(4, 1, id="reversed-bounds"),
    ],
)
@pytest.mark.parametrize("repl", ["", "!!"])
def bench_string_slice_replace(benchmark, N, start, stop, repl):
    values = ["abcdefghij", "accénted", "a", "", None]
    series = cudf.Series((values * ((N + 4) // 5))[:N])
    benchmark(series.str.slice_replace, start, stop, repl)
