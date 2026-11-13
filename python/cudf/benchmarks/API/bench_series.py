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


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize(
    "container", [cudf.Series, cudf.Index], ids=["series", "index"]
)
@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param(r"([A-Z]+)-(\d+)", id="unnamed"),
        pytest.param(r"(?P<letters>[A-Z]+)-(?P<number>\d+)", id="named"),
        pytest.param(r"(?P<letters>[A-Z]+)-(\d+)", id="mixed"),
    ],
)
def bench_str_extract(benchmark, num_rows, container, pattern):
    values = ["AB-123", "CD-456", None, "unmatched"] * (num_rows // 4)
    strings = container(values, dtype="str")

    benchmark(strings.str.extract, pattern, expand=True)
