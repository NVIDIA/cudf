# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IO tests for the streaming engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_sink_result_equal
from cudf_polars.utils.config import Cluster, StreamingExecutor

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cudf_polars.engine.core import StreamingEngine

# Runs the spmd variant even under rrun with nranks > 1. The ray/dask
# variants skip themselves in that environment.
pytestmark = [
    pytest.mark.spmd,
]


@pytest.fixture(scope="module")
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "x": range(30),
            "y": [1, 2, None] * 10,
            "z": ["ẅ", "a", "z", "123", "abcd"] * 6,
        }
    )


@pytest.fixture
def engine(
    streaming_engine_factory: Callable[..., StreamingEngine],
) -> StreamingEngine:
    """Yield each supported streaming engine pinned to small partitions."""
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=10),
    )


def test_sink_parquet_directory(
    engine: StreamingEngine, df: pl.LazyFrame, tmp_path: Path
) -> None:
    """Streaming engines always sink to a directory of partition files."""
    path = tmp_path / "out.parquet"
    assert_sink_result_equal(df, path, engine=engine)

    gpu_path = path.with_name(f"{path.stem}_gpu{path.suffix}")
    assert gpu_path.is_dir()
    assert any(gpu_path.iterdir())


def test_sink_parquet_empty_rank(engine: StreamingEngine, tmp_path: Path) -> None:
    """A rank that receives no partitions."""
    lazydf = pl.LazyFrame({"x": [1]})
    path = tmp_path / "tiny.parquet"
    assert_sink_result_equal(lazydf, path, engine=engine)

    gpu_path = path.with_name(f"{path.stem}_gpu{path.suffix}")
    assert gpu_path.is_dir()


def test_sink_parquet_duplicated(engine: StreamingEngine, tmp_path: Path) -> None:
    """Duplicated input should only be written once."""
    lazydf = (
        pl.LazyFrame({"x": [1] * 4, "y": [1, 2, 3, 4]})
        .group_by("x")
        .agg(pl.col("y").sum())
    )
    path = tmp_path / "duplicated.parquet"
    assert_sink_result_equal(lazydf, path, engine=engine)


@pytest.mark.parametrize(
    "cluster",
    [Cluster.SPMD, Cluster.RAY, Cluster.DASK],
)
def test_sink_to_directory_false_raises(cluster: Cluster) -> None:
    """Explicit ``sink_to_directory=False`` is rejected for every multi-rank cluster."""
    with pytest.raises(
        ValueError,
        match=rf"The {cluster.value} cluster requires sink_to_directory=True",
    ):
        StreamingExecutor(cluster=cluster, sink_to_directory=False)
