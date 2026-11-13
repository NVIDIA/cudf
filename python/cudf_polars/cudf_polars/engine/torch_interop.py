# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Hand ``SPMDEngine`` query results to PyTorch.

* :func:`persisted_to_torch` views this rank's GPU-resident result of
  :meth:`~cudf_polars.engine.spmd.SPMDEngine.execute` as torch tensors,
  without copying.
* :func:`polars_to_tensor` converts a host-side :class:`polars.DataFrame`,
  such as the output of :meth:`~polars.LazyFrame.collect`.

Both return plain per-rank tensors, not ``DTensor``. ``DTensor``'s ``Shard(0)``
requires each rank to hold exactly the rows ``torch.chunk`` would give it,
while the engine decides each rank's row count from the data, for example
from how a hash partitioning falls.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

import pylibcudf as plc

from cudf_polars.unstable import unstable

if TYPE_CHECKING:
    import polars as pl

    from cudf_polars.containers import Column, DataFrame
    from cudf_polars.engine.persisted_result import PersistedQueryResult
    from cudf_polars.engine.spmd import SPMDEngine


@functools.cache
def _torch_dtypes() -> dict[plc.types.TypeId, torch.dtype]:
    """Map libcudf fixed-width type ids to torch dtypes."""
    tid = plc.types.TypeId
    return {
        tid.INT8: torch.int8,
        tid.INT16: torch.int16,
        tid.INT32: torch.int32,
        tid.INT64: torch.int64,
        tid.UINT8: torch.uint8,
        tid.UINT16: torch.uint16,
        tid.UINT32: torch.uint32,
        tid.UINT64: torch.uint64,
        tid.FLOAT32: torch.float32,
        tid.FLOAT64: torch.float64,
        tid.BOOL8: torch.bool,
    }


def _column_to_tensor(name: str, column: Column) -> torch.Tensor:
    """Zero-copy view of a fixed-width, non-nullable GPU column as a tensor."""
    obj = column.obj
    type_id = obj.type().id()
    mapping = _torch_dtypes()
    if type_id not in mapping:
        raise TypeError(
            f"column {name!r} has dtype {column.dtype.polars_type}, which has no "
            "zero-copy torch equivalent. Cast it to a fixed-width numeric or "
            "boolean type before the handoff."
        )
    if obj.null_count() > 0:
        raise ValueError(
            f"column {name!r} contains nulls, which torch tensors cannot "
            "represent. Fill or drop them before the handoff, for example "
            f"with `pl.col({name!r}).fill_null(...)`."
        )
    dtype = mapping[type_id]
    buffer = obj.data()
    if buffer is None:  # empty column carries no data buffer
        return torch.empty(0, dtype=dtype, device="cuda")
    # Fixed-width columns come out of libcudf, which wraps their data this way.
    assert isinstance(buffer, plc.gpumemoryview)
    start = obj.offset() * dtype.itemsize
    data = buffer.byte_slice(slice(start, start + obj.size() * dtype.itemsize))
    return torch.as_tensor(data, device="cuda").view(dtype)


def _dataframe_to_torch(
    df: DataFrame, columns: list[str] | None
) -> dict[str, torch.Tensor]:
    """Convert selected columns of a GPU DataFrame to tensors, zero-copy."""
    names = columns if columns is not None else df.column_names
    missing = [name for name in names if name not in df.column_map]
    if missing:
        raise KeyError(
            f"column(s) {missing} not in result; available: {df.column_names}"
        )
    # The views below carry no stream in their __cuda_array_interface__.
    df.stream.synchronize()
    return {name: _column_to_tensor(name, df.column_map[name]) for name in names}


def _chunk_bounds(nrows: int, nranks: int, rank: int) -> tuple[int, int]:
    """
    Start and length of ``rank``'s slice when ``nrows`` rows are split like ``torch.chunk``.

    Every rank but the last holds ``ceil(nrows / nranks)`` rows, and trailing
    ranks may hold none, matching the split ``torch.chunk`` and so a
    ``DTensor`` ``Shard(0)`` placement would make.
    """
    step = -(-nrows // nranks)
    start = min(rank * step, nrows)
    return start, min(start + step, nrows) - start


@unstable()
def persisted_to_torch(
    result: PersistedQueryResult,
    *,
    engine: SPMDEngine,
    columns: list[str] | None = None,
    ensure_sharded: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert this rank's GPU-resident query result to ``torch.Tensor`` objects.

    Takes the result of :meth:`~cudf_polars.engine.spmd.SPMDEngine.execute` and
    views each column as a tensor that shares the column's GPU memory, so the
    data never leaves the device. Contrast with
    :meth:`~polars.LazyFrame.collect`, which copies the result to host memory
    first.

    This consumes the rank-local partition (see
    :meth:`~cudf_polars.engine.persisted_result.PersistedQueryResult.take_local`),
    so ``result`` cannot also be collected. The returned tensors keep the
    underlying GPU memory alive beyond the lifetime of the engine.

    A result is either sharded, each rank holding different rows, or
    replicated, every rank holding the same full copy. Which one a query
    produces depends on how the engine partitioned it, not only on the query.
    A replicated result is what every rank needs for values such as
    normalization constants, but for training data it means every rank trains
    on the same rows. ``ensure_sharded=True`` returns only this rank's share of
    a replicated result, and leaves a sharded result as it is, so the rows
    across ranks never overlap either way.

    Parameters
    ----------
    result
        Result of ``engine.execute(lf)``.
    engine
        The engine that produced ``result``; supplies this process's rank.
    columns
        Subset of columns to convert; defaults to all columns of the result.
    ensure_sharded
        If ``True`` and the result is replicated, return only this rank's
        slice of it, split the way ``torch.chunk`` splits rows. Has no effect
        on a result that is already sharded.

    Returns
    -------
    A dict mapping column name to a GPU :class:`torch.Tensor`.

    Raises
    ------
    KeyError
        If ``columns`` references a name that is not in the result.
    TypeError
        If a column's dtype has no zero-copy torch equivalent.
    ValueError
        If a column contains nulls.

    Examples
    --------
    >>> with SPMDEngine.from_torch_distributed() as engine:  # doctest: +SKIP
    ...     result = engine.execute(lf)
    ...     tensors = persisted_to_torch(result, engine=engine, ensure_sharded=True)
    """
    # Read the layout before taking the partition, which consumes it.
    replicated = result.local_is_duplicated(engine.rank)
    df = result.take_local(engine.rank)
    if ensure_sharded and replicated:
        df = df.slice(_chunk_bounds(df.num_rows, engine.nranks, engine.rank))
    return _dataframe_to_torch(df, columns)


@unstable()
def polars_to_tensor(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convert a Polars DataFrame to a dict of ``torch.Tensor``.

    Uses :meth:`polars.Series.to_torch` per column. If ``device`` is provided,
    each tensor is moved to that device after conversion. Per-column dtype
    overrides may be supplied via ``dtype``.

    Parameters
    ----------
    df
        Source DataFrame. Each column becomes one tensor.
    columns
        Subset of columns to convert; defaults to all columns of ``df``.
    device
        Optional target device (e.g. ``"cuda:0"`` or a :class:`torch.device`).
    dtype
        Optional per-column :class:`torch.dtype` overrides.

    Returns
    -------
    A dict mapping column name to :class:`torch.Tensor`.

    Raises
    ------
    polars.exceptions.ColumnNotFoundError
        If ``columns`` references a name that is not in ``df``.
    """
    cols = columns if columns is not None else df.columns
    dtype = dtype or {}

    out: dict[str, torch.Tensor] = {}
    for name in cols:
        series = df[name]
        tensor = series.to_torch()
        target_dtype = dtype.get(name)
        if target_dtype is not None:
            tensor = tensor.to(dtype=target_dtype)
        if device is not None:
            tensor = tensor.to(device)
        out[name] = tensor
    return out
