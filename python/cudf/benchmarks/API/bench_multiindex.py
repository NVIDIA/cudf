# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of MultiIndex methods."""

import numpy as np
import pandas as pd
import pytest
from config import NUM_ROWS, cudf


@pytest.fixture
def midx():
    num_elements = int(1e3)
    rng = np.random.default_rng(seed=0)
    a = rng.integers(0, num_elements // 10, num_elements)
    b = rng.integers(0, num_elements // 10, num_elements)
    pidx = pd.MultiIndex.from_arrays([a, b], names=("a", "b"))
    return cudf.MultiIndex(
        codes=pidx.codes, levels=pidx.levels, names=pidx.names
    )


def bench_constructor(benchmark, midx):
    benchmark(
        cudf.MultiIndex, codes=midx.codes, levels=midx.levels, names=midx.names
    )


def bench_from_frame(benchmark, midx):
    benchmark(cudf.MultiIndex.from_frame, midx.to_frame(index=False))


def bench_copy(benchmark, midx):
    benchmark(midx.copy, deep=False)


@pytest.mark.parametrize("level_size", NUM_ROWS)
@pytest.mark.parametrize("level_type", ["numeric", "datetime"])
def bench_repr_unused_levels(benchmark, level_size, level_type):
    if level_type == "datetime":
        level = pd.date_range("2025-01-01", periods=level_size, freq="h")
    else:
        level = np.arange(level_size)
    num_rows = 1000
    midx = cudf.MultiIndex(
        levels=[level, ["a", "b"]],
        codes=[np.zeros(num_rows, dtype="int32"), np.arange(num_rows) % 2],
        names=["level", "key"],
    )

    with pd.option_context("display.max_seq_items", 10):
        benchmark(repr, midx)
