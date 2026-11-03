# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of methods that exist for both Frame and Index."""

import operator

import numpy as np
import pytest
from utils import benchmark_with_object, make_gather_map


@benchmark_with_object(cls="frame_or_index", dtype="int")
@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
@pytest.mark.parametrize("fraction", [0.4])
def bench_take(benchmark, gather_how, fraction, frame_or_index):
    nr = len(frame_or_index)
    gather_map = make_gather_map(nr * fraction, nr, gather_how)
    benchmark(frame_or_index.take, gather_map)


@pytest.mark.pandas_incompatible  # Series/Index work, but not DataFrame
@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_argsort(benchmark, frame_or_index):
    benchmark(frame_or_index.argsort)


@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_min(benchmark, frame_or_index):
    benchmark(frame_or_index.min)


@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_where(benchmark, frame_or_index):
    cond = frame_or_index % 2 == 0
    benchmark(frame_or_index.where, cond, 0)


@benchmark_with_object(cls="frame_or_index", dtype="int", nulls=False)
def bench_values(benchmark, frame_or_index):
    benchmark(lambda: frame_or_index.values)


@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_nunique(benchmark, frame_or_index):
    benchmark(frame_or_index.nunique)


@benchmark_with_object(cls="frame_or_index", dtype="int", nulls=False)
def bench_to_numpy(benchmark, frame_or_index):
    benchmark(frame_or_index.to_numpy)


@benchmark_with_object(cls="frame_or_index", dtype="int", nulls=False)
@pytest.mark.pandas_incompatible
def bench_to_cupy(benchmark, frame_or_index):
    benchmark(frame_or_index.to_cupy)


@benchmark_with_object(cls="frame_or_index", dtype="int")
@pytest.mark.pandas_incompatible
def bench_to_arrow(benchmark, frame_or_index):
    benchmark(frame_or_index.to_arrow)


@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_astype(benchmark, frame_or_index):
    benchmark(frame_or_index.astype, float)


@pytest.mark.parametrize("ufunc", [np.add, np.logical_and])
@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_ufunc_series_binary(benchmark, frame_or_index, ufunc):
    benchmark(ufunc, frame_or_index, frame_or_index)


@pytest.mark.parametrize(
    "op",
    [operator.add, operator.mul, operator.eq],
)
@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_binops(benchmark, op, frame_or_index):
    benchmark(op, frame_or_index, frame_or_index)


@pytest.mark.parametrize(
    "op",
    [operator.add, operator.mul, operator.eq],
)
@benchmark_with_object(cls="frame_or_index", dtype="int")
def bench_scalar_binops(benchmark, op, frame_or_index):
    benchmark(op, frame_or_index, 1)
