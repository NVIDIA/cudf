# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Process-local storage for rank-local query-result partitions.

This module contains the one unavoidable process-global used by the
persisted-result feature. It is needed because the IO source that loads a
persisted partition runs deep within the Polars IR and has no access to the
owning engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid

    from cudf_polars.containers import DataFrame


class RankLocalStore:
    """One engine's rank-local partitions on a process, keyed by ``query_id`` then rank."""

    def __init__(self) -> None:
        # Nested ``query_id -> {rank -> (partition, duplicated)}``. Each entry
        # keeps the partition and whether it is part of a duplicated output (an
        # identical, complete copy held on every rank, e.g. a global sort/limit
        # result). The flag must survive persistence so a re-scan can re-advertise
        # it and downstream collectives don't double-count.
        self._partitions: dict[uuid.UUID, dict[int, tuple[DataFrame, bool]]] = {}

    def put(
        self, query_id: uuid.UUID, rank: int, df: DataFrame, *, duplicated: bool
    ) -> None:
        """
        Store this rank's partition.

        Parameters
        ----------
        query_id
            Identifier of the query the partition belongs to.
        rank
            This rank's index within the cluster.
        df
            The partition to store.
        duplicated
            Whether this partition is part of a duplicated output (an identical
            copy held on every rank).
        """
        self._partitions.setdefault(query_id, {})[rank] = (df, duplicated)

    def pop(self, query_id: uuid.UUID, rank: int) -> DataFrame:
        """
        Remove and return this rank's partition.

        Parameters
        ----------
        query_id
            Identifier of the query the partition belongs to.
        rank
            This rank's index within the cluster.

        Returns
        -------
        This rank's partition for ``query_id``.

        Raises
        ------
        RuntimeError
            If the partition has already been read. A persisted result is
            consumed on read and cannot be scanned more than once yet.
        """
        try:
            df, _ = self._partitions[query_id].pop(rank)
        except KeyError:
            raise RuntimeError(
                "A persisted query result is consumed on read and cannot be "
                "scanned more than once (for example a self-join, or "
                "collecting the same LazyFrame twice). Call engine.execute() "
                "again for a fresh result. Re-scan support is tracked as "
                "future work (see https://github.com/rapidsai/cudf/issues/23115)."
            ) from None
        return df

    def is_duplicated(self, query_id: uuid.UUID, rank: int) -> bool:
        """
        Return whether this rank's stored partition is part of a duplicated output.

        Read-only (does not consume the partition). Returns ``False`` if the
        partition is absent, so callers that probe before reading get a safe
        default and the eventual :meth:`pop` surfaces the consumed-result error.

        Parameters
        ----------
        query_id
            Identifier of the query the partition belongs to.
        rank
            This rank's index within the cluster.

        Returns
        -------
        ``True`` if the partition is an identical copy held on every rank.
        """
        entry = self._partitions.get(query_id, {}).get(rank)
        return entry is not None and entry[1]

    def drop(self, query_id: uuid.UUID) -> None:
        """
        Drop every rank's partition for ``query_id`` (idempotent).

        Parameters
        ----------
        query_id
            Identifier of the query whose partitions are dropped.
        """
        self._partitions.pop(query_id, None)

    def clear(self) -> None:
        """Drop all partitions in this store (idempotent)."""
        self._partitions.clear()


# The process-global set of per-engine stores, keyed by uid
_stores: dict[str, RankLocalStore] = {}


def open_store(uid: str) -> RankLocalStore:
    """Return this engine's store on the current process, creating it if absent."""
    return _stores.setdefault(uid, RankLocalStore())


def require_store(uid: str) -> RankLocalStore:
    """Return this engine's store; raise :class:`KeyError` if it has been closed."""
    return _stores[uid]


def close_store(uid: str) -> None:
    """Drop this engine's store on the current process (idempotent)."""
    _stores.pop(uid, None)


def close_all() -> None:
    """
    Drop every store on the current process (idempotent).

    For a process dedicated to a single engine (a Ray actor), this is equivalent
    to :func:`close_store` for that engine but needs no uid.
    """
    _stores.clear()


def drop_query(uid: str, query_id: uuid.UUID) -> None:
    """Drop ``query_id`` from this engine's store, if the store still exists."""
    store = _stores.get(uid)
    if store is not None:
        store.drop(query_id)
