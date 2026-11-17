# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for DataFrame reshaping operations (pivot, unstack)."""

import numpy as np
import pytest
from config import NUM_ROWS, cudf


@pytest.fixture(
    params=[
        (nr, card)
        for nr in NUM_ROWS
        for card in [max(4, nr // 100), max(4, nr // 10)]
    ],
    ids=lambda x: f"{x[0]}rows_card{x[1]}",
)
def data_size(request):
    return request.param


@pytest.fixture
def pivot_data(data_size):
    n_rows, cardinality = data_size
    rng = np.random.default_rng(42)

    # Generate raw data
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, cardinality, n_rows),
            "key2": rng.integers(0, cardinality // 2, n_rows),
            "value1": rng.standard_normal(n_rows),
            "value2": rng.standard_normal(n_rows),
            "value3": rng.standard_normal(n_rows),
            "value4": rng.standard_normal(n_rows),
        }
    )

    # Aggregate to ensure unique index-column pairs
    df_agg = (
        df.groupby(["key1", "key2"])
        .agg(
            {
                "value1": "sum",
                "value2": "mean",
                "value3": "min",
                "value4": "max",
            }
        )
        .reset_index()
    )

    return df_agg


@pytest.fixture
def unstack_data(data_size):
    n_rows, cardinality = data_size
    rng = np.random.default_rng(42)

    # Generate raw data
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, cardinality, n_rows),
            "key2": rng.integers(0, cardinality // 2, n_rows),
            "value1": rng.standard_normal(n_rows),
            "value2": rng.standard_normal(n_rows),
        }
    )

    # Create MultiIndex DataFrame (required for unstack)
    df_agg = df.groupby(["key1", "key2"]).agg(
        {"value1": "sum", "value2": "mean"}
    )

    return df_agg


@pytest.fixture
def unstack_data_3level(data_size):
    n_rows, cardinality = data_size
    rng = np.random.default_rng(42)

    # Generate raw data
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, cardinality, n_rows),
            "key2": rng.integers(0, cardinality // 2, n_rows),
            "key3": rng.integers(0, cardinality // 4, n_rows),
            "value": rng.standard_normal(n_rows),
        }
    )

    # Create 3-level MultiIndex DataFrame
    df_agg = df.groupby(["key1", "key2", "key3"]).agg({"value": "sum"})

    return df_agg


@pytest.mark.parametrize(
    "n_value_cols", [1, 2, 4], ids=["1col", "2cols", "4cols"]
)
def bench_pivot(benchmark, pivot_data, n_value_cols):
    value_cols = [f"value{i + 1}" for i in range(n_value_cols)]
    values = value_cols if n_value_cols > 1 else value_cols[0]

    benchmark(pivot_data.pivot, index="key1", columns="key2", values=values)


@pytest.mark.parametrize("n_value_cols", [2, 4], ids=["2cols", "4cols"])
def bench_pivot_no_values(benchmark, pivot_data, n_value_cols):
    # Select subset of columns to control how many value columns are used
    cols_to_keep = ["key1", "key2"] + [
        f"value{i + 1}" for i in range(n_value_cols)
    ]
    df_subset = pivot_data[cols_to_keep]

    benchmark(df_subset.pivot, index="key1", columns="key2")


@pytest.mark.parametrize("level", [None, 0], ids=["last_level", "first_level"])
@pytest.mark.parametrize("n_value_cols", [1, 2], ids=["1col", "2cols"])
def bench_unstack(benchmark, unstack_data, level, n_value_cols):
    if n_value_cols == 1:
        # Use only first value column
        df = unstack_data[["value1"]]
    else:
        # Use both value columns
        df = unstack_data

    if level is None:
        benchmark(df.unstack)
    else:
        benchmark(df.unstack, level=level)


def bench_unstack_multilevel(benchmark, unstack_data_3level):
    benchmark(unstack_data_3level.unstack, level=1)
