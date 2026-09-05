# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Values attached to each path of a file scan."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rmm.pylibrmm.stream import Stream

__all__ = ["PerPathValues"]


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class PerPathValues:
    """
    Values that vary per path in a file scan, one row per path.

    For example, a hive-partitioned scan of a dataset written with
    ``partition_by=["cat", "part"]`` supplies four paths and this frame::

        shape: (4, 2)
        ┌──────┬─────┐
        │ part ┆ cat │
        │ ---  ┆ --- │
        │ i64  ┆ str │
        ╞══════╪═════╡
        │ 1    ┆ u   │
        │ 2    ┆ u   │
        │ 3    ┆ v   │
        │ 4    ┆ v   │
        └──────┴─────┘

    Row ``i`` holds the values for path ``i``.
    The columns might not be in the order the keys appear in the
    paths.

    Parameters
    ----------
    df
        One row per path, one column per value to materialize.
    """

    df: pl.DataFrame

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> PerPathValues | None:
        """
        Build from the ``hive_parts`` dataframe of a polars ``Scan`` node.

        Parameters
        ----------
        df
            One row per path in the scan, one column per hive key.

        Returns
        -------
        The partition values, or ``None`` if no hive columns are needed.
        """
        if df.width == 0:
            # Polars filtered out all paths
            return None
        return cls(df)

    @functools.cached_property
    def names(self) -> tuple[str, ...]:
        """Names of the columns to materialize."""
        return tuple(self.df.columns)

    @functools.cached_property
    def dtypes(self) -> tuple[DataType, ...]:
        """Datatype of each column."""
        return tuple(DataType(dtype) for dtype in self.df.dtypes)

    @property
    def num_paths(self) -> int:
        """Number of paths these values describe."""
        return self.df.height

    @functools.cached_property
    def is_uniform(self) -> bool:
        """Whether every path shares the same values."""
        return all(series.n_unique() <= 1 for series in self.df.iter_columns())

    def slice(self, start: int, stop: int) -> PerPathValues:
        """
        Restrict to the paths in ``range(start, stop)``.

        Parameters
        ----------
        start
            Index of the first path to keep.
        stop
            Index one past the last path to keep.

        Returns
        -------
        Values for the selected paths.
        """
        return type(self)(self.df.slice(start, stop - start))

    def broadcast(self, num_rows: int, *, stream: Stream) -> list[Column]:
        """
        Materialize the values as columns of ``num_rows`` equal rows.

        Only valid when :attr:`is_uniform` holds, since every output row is
        given the values of the first path.

        Parameters
        ----------
        num_rows
            Length of the returned columns.
        stream
            CUDA stream used for device memory operations and kernel launches.

        Returns
        -------
        One column per value.
        """
        return self._to_columns(
            plc.filling.repeat(
                plc.Table.from_arrow(self.df.head(1), stream=stream),
                num_rows,
                stream=stream,
            )
        )

    def repeat(self, rows_per_path: Sequence[int], *, stream: Stream) -> list[Column]:
        """
        Materialize the values by repeating each path's row.

        Prefer gather when a source index is known.

        Parameters
        ----------
        rows_per_path
            Number of output rows contributed by each path.
        stream
            CUDA stream used for device memory operations and kernel launches.

        Returns
        -------
        One column per value, of length ``sum(rows_per_path)``.
        """
        return self._to_columns(
            plc.filling.repeat(
                plc.Table.from_arrow(self.df, stream=stream),
                plc.Column.from_arrow(
                    pl.Series(values=rows_per_path, dtype=pl.Int32()), stream=stream
                ),
                stream=stream,
            )
        )

    def gather(self, source_index: plc.Column, *, stream: Stream) -> list[Column]:
        """
        Materialize the values by indexing them with a source index.

        Parameters
        ----------
        source_index
            Column giving, for each output row, the index of the path it was
            read from.
        stream
            CUDA stream used for device memory operations and kernel launches.

        Returns
        -------
        One column per value, aligned with ``source_index``.
        """
        return self._to_columns(
            plc.copying.gather(
                plc.Table.from_arrow(self.df, stream=stream),
                source_index,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            )
        )

    def _to_columns(self, table: plc.Table) -> list[Column]:
        return [
            Column(column, name=name, dtype=dtype)
            for column, name, dtype in zip(
                table.columns(), self.names, self.dtypes, strict=True
            )
        ]

    def __repr__(self) -> str:
        """Representation showing the values."""
        return f"{type(self).__name__}(df={self.df!r})"

    def __hash__(self) -> int:
        """Hash of the schema and of every value."""
        return hash(
            (tuple(self.df.schema.items()), tuple(self.df.hash_rows().to_list()))
        )

    def __eq__(self, other: Any) -> bool:
        """Whether two sets of values agree."""
        return (
            isinstance(other, PerPathValues)
            and self.df.schema == other.df.schema
            and self.df.equals(other.df)
        )
