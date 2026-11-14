# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of Series methods."""

import pandas as pd
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
@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("operation", ["len", "count"])
def bench_nullable_string_integer_result(benchmark, N, storage, operation):
    values = (["aba", None, "abc", ""] * ((N + 3) // 4))[:N]
    series = cudf.Series(values, dtype=pd.StringDtype(storage=storage))
    method = getattr(series.str, operation)
    args = ("a",) if operation == "count" else ()
    benchmark(method, *args)
