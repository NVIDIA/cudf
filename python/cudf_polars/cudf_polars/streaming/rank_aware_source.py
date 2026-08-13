# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-aware Python scan source for cudf-polars."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import polars as pl

    from cudf_polars.containers import DataFrame
    from cudf_polars.typing import SizedIterator


class SizedChunks:
    """
    An iterator that also reports how many chunks it will produce.

    A Polars IO source returns an iterator, so the total number of chunks is
    generally unknown until the iterator is fully consumed. When a source already
    knows its chunk count, wrapping the chunks in `SizedChunks` lets the
    cudf-polars streaming engine forward chunks lazily, keeping only a single
    device chunk resident at a time instead of materializing all chunks up front to
    learn the count.

    `SizedChunks` remains an ordinary iterator, so the default (host) Polars
    engine handles it unchanged.

    This is an interim cudf-polars API. It exists because Polars' IO-source
    contract returns a plain iterator with no length. It may be removed if Polars
    exposes a supported way to report the chunk count. Tracked in
    https://github.com/rapidsai/cudf/issues/22917.

    Parameters
    ----------
    count
        Number of chunks ``chunks`` will yield.
    chunks
        Iterable of host :class:`polars.DataFrame` or GPU-resident
        `cudf_polars.containers.DataFrame` chunks.

    Examples
    --------
    >>> import polars as pl
    >>> from polars.io.plugins import register_io_source
    >>> from cudf_polars.streaming.rank_aware_source import SizedChunks
    >>>
    >>> def source(with_columns, predicate, n_rows, batch_size):
    ...     def chunks():
    ...         yield pl.DataFrame({"a": [1, 2]})
    ...         yield pl.DataFrame({"a": [3, 4]})
    ...
    ...     return SizedChunks(2, chunks())
    >>>
    >>> lf = register_io_source(source, schema={"a": pl.Int64})
    """

    def __init__(self, count: int, chunks: Iterable[pl.DataFrame | DataFrame]) -> None:
        self._count = count
        self._chunks = iter(chunks)

    def __iter__(self) -> Iterator[pl.DataFrame | DataFrame]:
        """Return the chunk iterator."""
        return self

    def __next__(self) -> pl.DataFrame | DataFrame:
        """Return the next chunk."""
        return next(self._chunks)

    def __len__(self) -> int:
        """Return the number of chunks."""
        return self._count


class RankAwareSource(abc.ABC):
    """
    Base class for rank-aware Polars IO-plugin sources.

    Subclasses must implement :meth:`__call__`, which follows the standard
    Polars `io_source` contract with additional `rank` and `nranks` keyword
    arguments supplied by cudf-polars streaming engines.

    During multi-rank streaming execution, every rank calls the IO source once.
    The implementation should therefore use `rank` and `nranks` to produce the
    rank-local rows of the global dataframe.

    This is an interim cudf-polars API. It exists because Polars' IO-source
    contract has no way to thread rank information into a source. It may be
    removed if Polars exposes a supported mechanism. Tracked in
    https://github.com/rapidsai/cudf/issues/22917.

    The `rank` and `nranks` arguments default to `0` and `1` for the in-memory
    cudf-polars engine, the default Polars engine, and single-rank streaming.

    A single `RankAwareSource` instance may be invoked concurrently (the same
    source can appear in more than one scan of a query, and the streaming engine
    runs scans on separate threads). Implementations must therefore be safe to
    call concurrently. The streaming engine also advances a single returned
    iterator on a worker thread pool, so successive chunks may be produced on
    different threads thus do not hold thread-affine state across chunks (see the
    Threading section of the IO-plugins guide).

    Pass the `RankAwareSource` instance directly to `register_io_source`; do not
    wrap it in another callable. To inject rank information, cudf-polars must
    locate the `RankAwareSource` instance inside the registered callable. Only an
    unwrapped instance is recognized; wrapping it in anything else (a
    `functools.partial`, closure, lambda, or decorator) hides it, in which case
    the source is treated as rank-unaware and runs on rank 0 only. This
    limitation is tracked in https://github.com/rapidsai/cudf/issues/22917.

    Examples
    --------
    Register a source that stripes a shared frame across streaming ranks.

    >>> import polars as pl
    >>> from polars.io.plugins import register_io_source
    >>> from cudf_polars.streaming.rank_aware_source import RankAwareSource
    >>>
    >>> class StripedSource(RankAwareSource):
    ...     def __init__(self, df):
    ...         self.df = df
    ...
    ...     def __call__(
    ...         self,
    ...         with_columns,
    ...         predicate,
    ...         n_rows,
    ...         batch_size,
    ...         rank=0,
    ...         nranks=1,
    ...     ):
    ...         out = self.df.gather_every(nranks, offset=rank)
    ...         if with_columns is not None:
    ...             out = out.select(with_columns)
    ...         if predicate is not None:
    ...             out = out.filter(predicate)
    ...         yield out
    >>>
    >>> lf = register_io_source(
    ...     StripedSource(pl.DataFrame({"a": [1, 2, 3]})),
    ...     schema={"a": pl.Int64},
    ... )
    """

    @abc.abstractmethod
    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
        rank: int = 0,
        nranks: int = 1,
    ) -> Iterator[pl.DataFrame | DataFrame] | SizedIterator[pl.DataFrame | DataFrame]:
        """
        Produce the rank-local chunks of this IO source.

        Parameters
        ----------
        with_columns
            Projected column names. If not ``None``, the source should return
            only these columns.
        predicate
            Polars expression. The reader must filter their rows
            accordingly.
        n_rows
            Maximum number of rows requested from the scan. cudf-polars does not
            support a pushed-down row limit and rejects it during translation, so
            this is always ``None`` on a GPU engine (see
            https://github.com/rapidsai/cudf/issues/22918). It may be non-None
            on the default Polars CPU engine, where the source must honor it.
        batch_size
            Optional hint for the number of rows to yield per chunk.
        rank
            Rank running this scan function, bound by the streaming engine.
            Defaults to ``0`` for single-rank / in-memory / default
            Polars-engine execution.
        nranks
            Total number of ranks (the world size), bound by the streaming
            engine. Defaults to ``1`` for single-rank execution.

        Returns
        -------
        Chunks for this rank. Return a :class:`SizedChunks` if the total chunk count
        is known upfront to enable lazy streaming.

        Notes
        -----
        Returning one or more `cudf_polars.containers.DataFrame` objects restricts
        the source to cudf-polars engines. To support the default (host) Polars engine,
        an IO source should only return :class:`polars.DataFrame`.

        The emitted columns, after applying `with_columns`, must match the registered
        schema in name, order, and dtype. cudf-polars always validates this and raises
        :class:`polars.exceptions.SchemaError` on mismatch. Polars only performs this
        validation when `register_io_source(validate_schema=True)` is used, but that
        flag is not exposed to the GPU execution path.
        """

    def output_duplicated(self, rank: int = 0, nranks: int = 1) -> bool:
        """
        Whether this rank's output is duplicated across ranks.

        A duplicated output is an identical, complete copy held on every rank
        (for example the result of a global sort/limit). The streaming scan node
        advertises this as the output channel's ``duplicated`` flag so downstream
        collectives (joins, aggregates) treat the copies as duplicates rather
        than distinct partitions and do not double-count them.

        By default, the output is not duplicated.

        Parameters
        ----------
        rank
            Rank running this scan function.
        nranks
            Total number of ranks (the world size).

        Returns
        -------
        ``True`` if this rank's output is an identical copy held on every rank.
        """
        return False
