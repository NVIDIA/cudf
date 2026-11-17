# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest


def bench_to_numpy(benchmark, rangeindex):
    benchmark(rangeindex.to_numpy)


@pytest.mark.pandas_incompatible
def bench_to_arrow(benchmark, rangeindex):
    benchmark(rangeindex.to_arrow)


def bench_argsort(benchmark, rangeindex):
    benchmark(rangeindex.argsort)


def bench_nunique(benchmark, rangeindex):
    benchmark(rangeindex.nunique)


def bench_isna(benchmark, rangeindex):
    benchmark(rangeindex.isna)


def bench_max(benchmark, rangeindex):
    benchmark(rangeindex.max)


def bench_min(benchmark, rangeindex):
    benchmark(rangeindex.min)


def bench_where(benchmark, rangeindex):
    cond = rangeindex % 2 == 0
    benchmark(rangeindex.where, cond, 0)


def bench_isin(benchmark, rangeindex):
    values = [10, 100]
    benchmark(rangeindex.isin, values)
