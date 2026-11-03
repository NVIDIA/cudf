# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd

import pylibcudf as plc

import cudf  # noqa: TC001 (used at runtime for cudf.DataFrame etc., not type-only)
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_integer,
    is_list_like,
)
from cudf.core.column.column import as_column
from cudf.core.copy_types import BooleanMask, GatherMap
from cudf.core.dtypes import CategoricalDtype, IntervalDtype
from cudf.core.index import Index
from cudf.core.multiindex import MultiIndex
from cudf.core.series import Series
from cudf.utils.dtypes import dtype_from_pylibcudf_column

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf.core.column.column import ColumnBase
    from cudf.core.column_accessor import ColumnAccessor
    from cudf.core.dataframe import DataFrame


class EmptyIndexer:
    """An indexer that will produce an empty result."""

    pass


@dataclass
class MapIndexer:
    """An indexer for a gather map."""

    key: GatherMap


@dataclass
class MaskIndexer:
    """An indexer for a boolean mask."""

    key: BooleanMask


@dataclass
class SliceIndexer:
    """An indexer for a slice."""

    key: slice


@dataclass
class ScalarIndexer:
    """An indexer for a scalar value."""

    key: GatherMap


IndexingSpec: TypeAlias = (
    EmptyIndexer | MapIndexer | MaskIndexer | ScalarIndexer | SliceIndexer
)


def validate_scalar_key(key: Any, error_msg: str) -> None:
    """Validate that key contains only scalar values for .at/.iat indexers.

    Parameters
    ----------
    key : Any
        The key to validate
    error_msg : str
        The error message to raise if validation fails

    Raises
    ------
    ValueError
        If the key contains list-like indexers
    """
    if not isinstance(key, tuple):
        if is_list_like(key):
            raise ValueError(error_msg)
    else:
        for k in key:
            if is_list_like(k):
                raise ValueError(error_msg)


def check_dict_or_set_indexers(key: Any) -> None:
    """Disallow using a set or dict (or a tuple nesting one) as an indexer.

    Mirrors ``pandas.core.indexing.check_dict_or_set_indexers``. pandas
    rejected set/dict indexers in 2.0 (GH#42825). The recursion into tuples
    matches pandas' handling of nested ``.loc`` keys on a ``MultiIndex``
    (e.g. ``df.loc[(({1}, 2), "a")]``).
    """
    if isinstance(key, set):
        raise TypeError(
            "Passing a set as an indexer is not supported. Use a list instead."
        )
    elif isinstance(key, dict):
        raise TypeError(
            "Passing a dict as an indexer is not supported. "
            "Use a list instead."
        )
    elif isinstance(key, tuple):
        for sub_key in key:
            check_dict_or_set_indexers(sub_key)


# Helpers for code-sharing between loc and iloc paths
def expand_key(
    key: Any, frame: DataFrame | Series, method_type: Literal["iloc", "loc"]
) -> tuple[Any, ...]:
    """Slice-expand key to match dimension of the frame being indexed.

    Parameters
    ----------
    key
        Key to expand
    frame
        DataFrame or Series to expand to the dimension of.

    Returns
    -------
    tuple
        New key of length equal to the dimension of the frame.

    Raises
    ------
    IndexError
        If the provided key is a tuple and has more entries than the frame dimension.

    Notes
    -----
    If any individual entry in the key is a callable, it is called
    with the provided frame as argument and is required to be converted
    into a supported indexing type.
    """
    dim = len(frame.shape)
    if (
        isinstance(key, bool)
        or (
            isinstance(key, Series)
            and key.dtype.kind == "b"
            and method_type == "loc"
            and len(key) != len(frame)
        )
        or (
            isinstance(key, Series)
            and key.dtype.kind == "b"
            and method_type == "iloc"
        )
    ) and not (
        frame.index.dtype.kind == "b"
        or isinstance(frame.index, MultiIndex)
        and frame.index.get_level_values(0).dtype.kind == "b"
    ):
        raise KeyError(
            f"{key}: boolean label can not be used without a boolean index"
        )

    if isinstance(key, slice) and (
        isinstance(key.start, bool) or isinstance(key.stop, bool)
    ):
        raise TypeError(f"{key}: boolean values can not be used in a slice")

    if isinstance(key, tuple):
        # Key potentially indexes rows and columns, slice-expand to
        # shape of frame
        if len(key) > 1 and sum(k is Ellipsis for k in key) > 1:
            raise IndexError("indexer may only contain one '...' entry")
        indexers = key + (slice(None),) * (dim - len(key))
        if len(indexers) > dim:
            raise IndexError(
                f"Too many indexers: got {len(indexers)} expected {dim}"
            )
    else:
        # Key indexes rows, slice-expand to shape of frame
        indexers = (key, *(slice(None),) * (dim - 1))
    return tuple(k(frame) if callable(k) else k for k in indexers)


def destructure_dataframe_indexer(
    key: Any,
    frame: DataFrame,
    destructure: Callable[[Any, DataFrame], tuple[Any, Any]],
    is_scalar: Callable[[Any, ColumnAccessor], bool],
    get_ca: str,
):
    """
    Pick apart an indexing key for a DataFrame into constituent pieces.

    Parameters
    ----------
    key
        The key to unpick.
    frame
        The DataFrame being indexed.
    destructure
        Callable to split the key into a two-tuple of row keys and
        column keys.
    is_scalar
        Callable to report if the column indexer produces a single
        column.
    get_ca
        Method name to obtain the column accessor from the frame.

    Returns
    -------
    rows
        Indexing expression for the rows
    tuple
        Two-tuple indicating if the column indexer produces a scalar and
        a subsetted ColumnAccessor.

    Raises
    ------
    TypeError
        If the column indexer is invalid.
    """
    rows, cols = destructure(key, frame)

    from cudf.core.series import Series

    if cols is Ellipsis:
        cols = slice(None)
    elif isinstance(cols, (Index, Series)):
        cols = cols.to_pandas()
    try:
        ca = getattr(frame._data, get_ca)(cols)
    except TypeError as e:
        raise TypeError(
            "Column indices must be names, slices, "
            "list-like of names, or boolean mask"
        ) from e
    scalar = is_scalar(cols, ca)
    if scalar:
        assert len(ca) == 1, (
            "Scalar column indexer should not produce more than one column"
        )
    return rows, (scalar, ca)


def destructure_iloc_key(
    key: Any, frame: Series | DataFrame
) -> tuple[Any, ...]:
    """
    Destructure a potentially tuple-typed key into row and column indexers.

    Tuple arguments to iloc indexing are treated specially. They are
    picked apart into indexers for the row and column. If the number
    of entries is less than the number of modes of the frame, missing
    entries are slice-expanded.

    If the user-provided key is not a tuple, it is treated as if it
    were a singleton tuple, and then slice-expanded.

    Once this destructuring has occurred, any entries that are
    callables are then called with the indexed frame. This should
    return a valid indexing object for the rows (respectively
    columns), namely one of:

    - A boolean mask of the same length as the frame in the given
      dimension
    - A scalar integer that indexes the frame
    - An array-like of integers that index the frame
    - A slice that indexes the frame

    Integer and slice-based indexing follows usual Python conventions.

    Parameters
    ----------
    key
        The key to destructure
    frame
        DataFrame or Series to provide context

    Returns
    -------
    tuple
        Indexers with length equal to the dimension of the frame

    Raises
    ------
    IndexError
        If there are too many indexers, or any individual indexer is a tuple.
    """
    indexers = expand_key(key, frame, "iloc")
    if any(isinstance(k, tuple) for k in indexers):
        raise IndexError(
            "Too many indexers: can't have nested tuples in iloc indexing"
        )
    return indexers


def destructure_dataframe_iloc_indexer(
    key: Any, frame: DataFrame
) -> tuple[Any, tuple[bool, ColumnAccessor]]:
    """Destructure an index key for DataFrame iloc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        DataFrame to provide context context

    Returns
    -------
    tuple
        2-tuple of a key for the rows and tuple of
        (column_index_is_scalar, ColumnAccessor) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """
    return destructure_dataframe_indexer(
        key,
        frame,
        destructure_iloc_key,
        lambda col, _ca: is_integer(col),
        "select_by_index",
    )


def destructure_series_iloc_indexer(key: Any, frame: Series) -> Any:
    """Destructure an index key for Series iloc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        Series for unpacking context

    Returns
    -------
    Single key that will index the rows
    """
    (rows,) = destructure_iloc_key(key, frame)
    return rows


def parse_row_iloc_indexer(key: Any, n: int) -> IndexingSpec:
    """
    Normalize and produce structured information about a row indexer.

    Given a row indexer that has already been destructured by
    :func:`destructure_iloc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably destructured key for row indexing
    n
        Length of frame to index

    Returns
    -------
    IndexingSpec
        Structured data for indexing. A tag + parsed data.

    Raises
    ------
    IndexError
        If a valid type of indexer is provided, but it is out of
        bounds
    TypeError
        If the indexing key is otherwise invalid.
    """
    if key is Ellipsis:
        return SliceIndexer(slice(None))
    elif isinstance(key, slice):
        return SliceIndexer(key)
    elif _is_scalar_or_zero_d_array(key):
        return ScalarIndexer(GatherMap(key, n, nullify=False))
    else:
        key = as_column(key)
        if isinstance(key.dtype, CategoricalDtype):
            key = key.astype(key.codes.dtype)
        if key.dtype.kind == "b":
            return MaskIndexer(BooleanMask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        elif key.dtype.kind in "iu":
            return MapIndexer(GatherMap(key, n, nullify=False))
        else:
            raise TypeError(
                "Cannot index by location "
                f"with non-integer key of type {type(key)}"
            )


def destructure_loc_key(
    key: Any, frame: Series | DataFrame
) -> tuple[Any, ...]:
    """
    Destructure a potentially tuple-typed key into row and column indexers

    Tuple arguments to loc indexing are treated specially. They are
    picked apart into indexers for the row and column. If the number
    of entries is less than the number of modes of the frame, missing
    entries are slice-expanded.

    If the user-provided key is not a tuple, it is treated as if it
    were a singleton tuple, and then slice-expanded.

    Once this destructuring has occurred, any entries that are
    callables are then called with the indexed frame. This should
    return a valid indexing object for the rows (respectively
    columns), namely one of:

    - A boolean mask of the same length as the frame in the given
      dimension
    - A scalar label looked up in the index
    - A scalar integer that indexes the frame
    - An array-like of labels looked up in the index
    - A slice of the index
    - For multiindices, a tuple of per level indexers

    Slice-based indexing is on the closed interval [start, end], rather
    than the semi-open interval [start, end)

    Parameters
    ----------
    key
        The key to destructure
    frame
        DataFrame or Series to provide context

    Returns
    -------
    tuple of indexers with length equal to the dimension of the frame

    Raises
    ------
    IndexError
        If there are too many indexers.
    """
    return expand_key(key, frame, "loc")


def destructure_dataframe_loc_indexer(
    key: Any, frame: DataFrame
) -> tuple[Any, tuple[bool, ColumnAccessor]]:
    """Destructure an index key for DataFrame loc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        DataFrame to provide context context

    Returns
    -------
    tuple
        2-tuple of a key for the rows and tuple of
        (column_index_is_scalar, ColumnAccessor) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """

    def is_scalar(name: Any, ca: ColumnAccessor) -> bool:
        try:
            return name in ca
        except TypeError:
            return False

    return destructure_dataframe_indexer(
        key, frame, destructure_loc_key, is_scalar, "select_by_label"
    )


def destructure_series_loc_indexer(key: Any, frame: Series) -> Any:
    """Destructure an index key for Series loc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        Series for unpacking context

    Returns
    -------
    Single key that will index the rows
    """
    (rows,) = destructure_loc_key(key, frame)
    return rows


def ordered_find(needles: ColumnBase, haystack: ColumnBase) -> GatherMap:
    """Find locations of needles in a haystack preserving order

    Parameters
    ----------
    needles
        Labels to look for
    haystack
        Haystack to search in

    Returns
    -------
    NumericalColumn
        Integer gather map of locations needles were found in haystack

    Raises
    ------
    KeyError
        If not all needles were found in the haystack.
        If needles cannot be converted to the dtype of haystack.

    Notes
    -----
    This sorts the gather map so that the result comes back in the
    order the needles were specified (and are found in the haystack).
    """
    # Pre-process to match dtypes
    needle_kind = needles.dtype.kind
    haystack_kind = haystack.dtype.kind
    if haystack_kind == "O" and not isinstance(haystack.dtype, IntervalDtype):
        try:
            needles = needles.astype(haystack.dtype)
        except ValueError:
            # Pandas raise KeyError here
            raise KeyError("Dtype mismatch in label lookup")
    elif needle_kind == haystack_kind or {
        haystack_kind,
        needle_kind,
    }.issubset({"i", "u", "f"}):
        needles = needles.astype(haystack.dtype)
    elif needles.dtype != haystack.dtype:
        # Pandas raise KeyError here
        raise KeyError("Dtype mismatch in label lookup")
    # Can't always do an inner join because then we can't check if we
    # had missing keys (can't check the length because the entries in
    # the needle might appear multiple times in the haystack).

    left_rows, right_rows = plc.join.left_join(
        plc.Table([needles.plc_column]),
        plc.Table([haystack.plc_column]),
        plc.types.NullEquality.EQUAL,
    )
    right_order = plc.copying.gather(
        plc.Table(
            [
                plc.filling.sequence(
                    len(haystack), plc.Scalar.from_py(0), plc.Scalar.from_py(1)
                )
            ]
        ),
        right_rows,
        plc.copying.OutOfBoundsPolicy.NULLIFY,
    ).columns()[0]
    if right_order.null_count() > 0:
        raise KeyError("Not all keys in index")
    left_order = plc.copying.gather(
        plc.Table(
            [
                plc.filling.sequence(
                    len(needles), plc.Scalar.from_py(0), plc.Scalar.from_py(1)
                )
            ]
        ),
        left_rows,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    ).columns()[0]

    plc_right_rows = plc.sorting.stable_sort_by_key(
        plc.Table([right_rows]),
        plc.Table([left_order, right_order]),
        [plc.types.Order.ASCENDING] * 2,
        [plc.types.NullOrder.AFTER] * 2,
    ).columns()[0]
    return GatherMap.from_column_unchecked(
        cast(
            "cudf.core.column.NumericalColumn",
            type(haystack).create(
                plc_right_rows,
                dtype=dtype_from_pylibcudf_column(plc_right_rows),
            ),
        ),
        len(haystack),
        nullify=False,
    )


def find_label_range_or_mask(
    key: slice, index: Index
) -> EmptyIndexer | SliceIndexer:
    """
    Convert a slice of labels into a slice of positions

    Parameters
    ----------
    key
        Slice to convert
    index
        Index to look up in

    Returns
    -------
    IndexingSpec
        Structured data for indexing (but never a :class:`ScalarIndexer`)

    Raises
    ------
    KeyError
        If the index is unsorted and not a DatetimeIndex
    """
    parsed_key = index.find_label_range(key)
    if len(range(len(index))[parsed_key]) == 0:
        return EmptyIndexer()
    else:
        return SliceIndexer(parsed_key)


def parse_single_row_loc_key(
    key: Any,
    index: Index,
) -> IndexingSpec:
    """
    Turn a single label-based row indexer into structured information.

    This converts label-based lookups into structured positional
    lookups.

    Valid values for the key are
    - a slice (endpoints are looked up)
    - a scalar label
    - a boolean mask of the same length as the index
    - a column of labels to look up (may be empty)

    Parameters
    ----------
    key
        Key for label-based row indexing
    index
        Index to act as haystack for labels

    Returns
    -------
    IndexingSpec
        Structured information for indexing

    Raises
    ------
    KeyError
        If any label is not found
    ValueError
        If labels cannot be coerced to index dtype
    """
    n = len(index)
    if isinstance(key, slice):
        return find_label_range_or_mask(key, index)
    else:
        is_scalar = _is_scalar_or_zero_d_array(key)
        if is_scalar and isinstance(key, np.ndarray):
            key = as_column(key.item())
        elif (
            isinstance(key, Series)
            and key.dtype.kind == "b"
            and not key.index.equals(index)
        ):
            # A boolean Series indexer is aligned to the frame's index by
            # label rather than applied positionally. Only labels missing
            # from the mask make it unalignable; pre-existing NA values in
            # the mask are treated as not-selected (matching pandas), so
            # check for missing labels rather than for nulls post-reindex.
            if len(index.difference(key.index)) > 0:
                raise ValueError(
                    "Unalignable boolean Series provided as indexer "
                    "(index of the boolean Series and of the indexed "
                    "object do not match)."
                )
            key = key.reindex(index)._column
        else:
            key = as_column(key)
        if (
            isinstance(key.dtype, CategoricalDtype)
            and index.dtype != key.dtype
        ):
            # TODO: is this right?
            key = key._get_decategorized_column()
        if len(key) == 0:
            return EmptyIndexer()
        else:
            # TODO: promote to Index objects, so this can handle
            # categoricals correctly?
            if key.dtype.kind == "b":
                if is_scalar and index.dtype.kind != "b":
                    raise KeyError(
                        "boolean label cannot be used without a boolean index"
                    )
                else:
                    return MaskIndexer(BooleanMask(key, n))
            elif index.dtype.kind == "M":
                # Try to turn strings into datetimes
                try:
                    key = as_column(key, dtype=index.dtype)
                except pd.errors.OutOfBoundsDatetime:
                    if is_scalar:
                        # A label beyond the index unit's bounds cannot
                        # be present; pandas raises KeyError for scalar
                        # lookups but OutOfBoundsDatetime for list keys.
                        raise KeyError(key.element_indexing(0))
                    raise
            haystack = index._column
            gather_map = ordered_find(key, haystack)
            if is_scalar and len(gather_map.column) == 1:
                return ScalarIndexer(gather_map)
            else:
                return MapIndexer(gather_map)


def parse_row_loc_indexer(key: Any, index: Index) -> IndexingSpec:
    """
    Normalize to return structured information for a label-based row indexer.

    Given a label-based row indexer that has already been destructured by
    :func:`destructure_loc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably destructured key for row indexing
    index
        Index to provide context

    Returns
    -------
    IndexingSpec
        Structured data for indexing. A tag + parsed data.

    Raises
    ------
    KeyError
        If a valid type of indexer is provided, but not all keys are
        found
    TypeError
        If the indexing key is otherwise invalid.
    """
    if isinstance(index, MultiIndex):
        raise NotImplementedError(
            "This code path is not designed for MultiIndex"
        )
    # TODO: multiindices need to be treated separately
    if key is Ellipsis:
        # Ellipsis is handled here because multiindex level-based
        # indices don't handle ellipsis in pandas.
        return SliceIndexer(slice(None))
    else:
        return parse_single_row_loc_key(key, index)
