# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import cudf
from cudf.utils.dtypes import SIZE_TYPE_DTYPE

if TYPE_CHECKING:
    from cudf.core.column import NumericalColumn


@dataclass
class GatherMap:
    """A representation of a column as a gather map.

    This object augments the column with the information that it
    is valid as a gather map for the specified number of rows with
    the given nullification flag.

    Parameters
    ----------
    column
        The data to turn into a column and then verify
    nrows
        The number of rows to verify against
    nullify
        Will the gather map be used nullifying out of bounds
        accesses?

    Returns
    -------
    GatherMap
        New object wrapping the column bearing witness to its
        suitability as a gather map for columns with nrows.

    Raises
    ------
    TypeError
        If the column is of unsuitable dtype
    IndexError
        If the map is not in bounds.
    """

    #: The number of rows the gather map has been validated for
    nrows: int
    #: Was the validation for nullify=True?
    nullify: bool

    def __init__(self, column: Any, nrows: int, *, nullify: bool):
        #: The gather map
        self.column = cast(
            "cudf.core.column.NumericalColumn",
            cudf.core.column.as_column(column),
        )
        self.nrows = nrows
        self.nullify = nullify
        if len(self.column) == 0:
            # Any empty column is valid as a gather map
            # This is necessary because as_column([]) defaults to float64
            # TODO: we should fix this further up.
            # Alternately we can have an Optional[Column] and handle None
            # specially in _gather.
            self.column = cast(
                "NumericalColumn", self.column.astype(SIZE_TYPE_DTYPE)
            )
        else:
            if self.column.dtype.kind not in {"i", "u"}:
                raise TypeError("Gather map must have integer dtype")
            if not nullify:
                lo, hi = self.column.minmax()
                if lo < -nrows or hi >= nrows:
                    raise IndexError(
                        f"Gather map is out of bounds for [0, {nrows})"
                    )

    @classmethod
    def from_column_unchecked(
        cls, column: "NumericalColumn", nrows: int, *, nullify: bool
    ) -> Self:
        """Construct a new GatherMap from a column without checks.

        Parameters
        ----------
        column
           The column that will be used as a gather map
        nrows
           The number of rows the gather map will be used for
        nullify
           Will the gather map be used nullifying out of bounds
           accesses?

        Returns
        -------
        GatherMap

        Notes
        -----
        This method asserts, by fiat, that the column is valid.
        Behaviour is undefined if it is not.
        """
        self = cls.__new__(cls)
        self.column = column
        self.nrows = nrows
        self.nullify = nullify
        return self


@dataclass
class BooleanMask:
    """A representation of a column as a boolean mask.

    This augments the column with information that it is valid as a
    boolean mask for columns with a given number of rows

    Parameters
    ----------
    column
        The data to turn into a column to then verify
    nrows
        the number of rows to verify against

    Returns
    -------
    BooleanMask
        New object wrapping the column bearing witness to its
        suitability as a boolean mask for columns with matching
        row count.

    Raises
    ------
    TypeError
        If the column is of unsuitable dtype
    IndexError
        If the mask has the wrong number of rows
    """

    def __init__(self, column: Any, nrows: int):
        #: The boolean mask
        self.column = cast(
            "cudf.core.column.NumericalColumn",
            cudf.core.column.as_column(column),
        )
        if self.column.dtype.kind != "b":
            raise TypeError("Boolean mask must have bool dtype")
        if len(column) != nrows:
            raise IndexError(
                f"Column with {len(column)} rows not suitable "
                f"as a boolean mask for {nrows} rows"
            )

    @classmethod
    def from_column_unchecked(cls, column: "NumericalColumn") -> Self:
        """Construct a new BooleanMask from a column without checks.

        Parameters
        ----------
        column
           The column that will be used as a boolean mask

        Returns
        -------
        BooleanMask

        Notes
        -----
        This method asserts, by fiat, that the column is valid.
        Behaviour is undefined if it is not.
        """
        self = cls.__new__(cls)
        self.column = column
        return self
