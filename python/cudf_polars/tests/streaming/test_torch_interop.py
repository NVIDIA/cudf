# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the torch.distributed interop module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

torch = pytest.importorskip("torch")

import torch.distributed as dist  # noqa: E402

from cudf_polars.engine.torch_interop import (  # noqa: E402
    _chunk_bounds,
    _dataframe_to_torch,
    persisted_to_torch,
    polars_to_tensor,
)

# A CPU-only PyTorch build cannot hold the GPU-resident handoff.
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA build of PyTorch"
)


def test_polars_to_tensor() -> None:
    """Every column by default, or a subset in the given order with dtype overrides."""
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": [5, 6]})
    out = polars_to_tensor(df)
    assert {name: t.tolist() for name, t in out.items()} == df.to_dict(as_series=False)

    out = polars_to_tensor(df, columns=["c", "a"], dtype={"a": torch.float64})
    assert list(out) == ["c", "a"]
    assert out["a"].dtype == torch.float64
    assert out["a"].tolist() == [1.0, 2.0]


def test_from_torch_distributed_requires_initialized_pg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classmethod raises if torch.distributed has not been initialized."""
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match=r"torch\.distributed is not initialized"):
        SPMDEngine.from_torch_distributed()


@pytest.fixture
def single_rank_process_group(tmp_path: Path) -> Iterator[None]:
    """A real one-rank gloo process group, torn down afterwards."""
    if dist.is_initialized():
        pytest.skip("a torch.distributed process group is already initialized")
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.usefixtures("single_rank_process_group")
def test_from_torch_distributed_builds_engine_from_options() -> None:
    """The engine is built on the torch group and takes its options like `from_options`."""
    options = StreamingOptions(max_rows_per_partition=7)
    with SPMDEngine.from_torch_distributed(options) as engine:
        assert engine.nranks == 1
        assert engine.rank == 0
        executor = engine.config["executor_options"]
        assert executor["max_rows_per_partition"] == 7


@pytest.mark.usefixtures("single_rank_process_group")
def test_from_torch_distributed_checks_gpu_before_bootstrap() -> None:
    """A GPU selected by ordinal is rejected before any communicator is created."""
    with (
        patch("cuda.core.Device", return_value=MagicMock(device_id=1)),
        patch("cudf_polars.engine.spmd.new_communicator") as new_communicator,
        pytest.raises(RuntimeError, match="ordinal 0, but"),
    ):
        SPMDEngine.from_torch_distributed()
    new_communicator.assert_not_called()


@requires_torch_cuda
@pytest.mark.parametrize("ensure_sharded", [False, True])
def test_persisted_to_torch_roundtrip(
    spmd_engine: SPMDEngine, *, ensure_sharded: bool
) -> None:
    """The handoff returns this rank's rows, on the GPU, with the right values."""
    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6]})
    result = spmd_engine.execute(lf.select(pl.col("a") * 2, pl.col("b")))
    # The result is sharded, so `ensure_sharded` leaves it as it is.
    tensors = persisted_to_torch(
        result, engine=spmd_engine, ensure_sharded=ensure_sharded
    )

    assert set(tensors) == {"a", "b"}
    assert tensors["a"].device.type == "cuda"
    assert sorted(tensors["a"].tolist()) == [2.0, 4.0, 6.0]
    assert sorted(tensors["b"].tolist()) == [4, 5, 6]


@requires_torch_cuda
def test_persisted_to_torch_synchronizes_producing_stream(
    spmd_engine: SPMDEngine,
) -> None:
    """The query's stream is synchronized before its buffers are exposed."""
    result = spmd_engine.execute(pl.LazyFrame({"a": [1.0, 2.0]}).select("a"))
    df = result.take_local(spmd_engine.rank)
    df.stream = MagicMock(wraps=df.stream)
    _dataframe_to_torch(df, None)
    df.stream.synchronize.assert_called_once_with()


@requires_torch_cuda
def test_persisted_to_torch_column_subset(spmd_engine: SPMDEngine) -> None:
    """`columns` selects a subset, and an unknown name raises."""
    lf = pl.LazyFrame({"a": [1.0], "b": [2.0]})
    result = spmd_engine.execute(lf.select("a", "b"))
    assert set(persisted_to_torch(result, engine=spmd_engine, columns=["b"])) == {"b"}

    result = spmd_engine.execute(lf.select("a", "b"))
    with pytest.raises(KeyError, match="not in result"):
        persisted_to_torch(result, engine=spmd_engine, columns=["nope"])


@pytest.mark.parametrize(
    "values,error,match",
    [
        ([1.0, None, 3.0], ValueError, "contains nulls"),
        (["x", "y"], TypeError, "no zero-copy torch equivalent"),
    ],
)
def test_persisted_to_torch_rejects_unsupported_columns(
    spmd_engine: SPMDEngine, values: list, error: type[Exception], match: str
) -> None:
    """Nulls and non-fixed-width columns cannot become tensors, and say so."""
    result = spmd_engine.execute(pl.LazyFrame({"a": values}).select("a"))
    with pytest.raises(error, match=match):
        persisted_to_torch(result, engine=spmd_engine)


@pytest.mark.parametrize("nranks", [1, 3, 4])
@pytest.mark.parametrize("nrows", [0, 1, 9, 10])
def test_chunk_bounds_match_torch_chunk(nrows: int, nranks: int) -> None:
    """Each rank's slice is exactly the chunk `torch.chunk` gives it."""
    chunks = [c.tolist() for c in torch.arange(nrows).chunk(nranks)] if nrows else []
    chunks += [[]] * (nranks - len(chunks))
    for rank in range(nranks):
        start, length = _chunk_bounds(nrows, nranks, rank)
        assert list(range(start, start + length)) == chunks[rank]


@requires_torch_cuda
def test_ensure_sharded_slices_replicated_result(spmd_engine: SPMDEngine) -> None:
    """A replicated result is cut down to this rank's `torch.chunk` share."""
    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = spmd_engine.execute(lf.select("a"))
    # A single-rank engine never replicates, so present this rank as the
    # first of two holding a replicated copy.
    engine = MagicMock(rank=spmd_engine.rank, nranks=2)
    with patch.object(result, "local_is_duplicated", return_value=True):
        tensor = persisted_to_torch(result, engine=engine, ensure_sharded=True)["a"]
    assert tensor.tolist() == [1.0, 2.0, 3.0]


@requires_torch_cuda
def test_sliced_partition_converts_at_its_offset(spmd_engine: SPMDEngine) -> None:
    """A later rank's share is a view into the column at its offset, not a copy."""
    result = spmd_engine.execute(
        pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}).select("a")
    )
    df = result.take_local(spmd_engine.rank)
    data = df.column_map["a"].obj.data()
    assert data is not None
    start, length = _chunk_bounds(df.num_rows, 2, 1)
    tensor = _dataframe_to_torch(df.slice((start, length)), None)["a"]
    assert tensor.tolist() == [4.0, 5.0]
    assert tensor.data_ptr() == data.ptr + start * tensor.element_size()
