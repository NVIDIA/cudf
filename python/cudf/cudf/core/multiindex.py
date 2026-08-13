# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import numbers
import operator
from functools import cached_property
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
import pandas as pd

import pylibcudf as plc

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import is_integer, is_list_like, is_scalar
from cudf.core._internals import sorting
from cudf.core.algorithms import factorize
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.column.utils import access_columns
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.dtype.validators import (
    is_dtype_obj_string,
)
from cudf.core.frame import Frame
from cudf.core.index import (
    Index,
    _get_indexer_basic,
    _index_from_data,
    _lexsorted_equal_range,
    ensure_index,
)
from cudf.core.join._join_helpers import _match_join_keys
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    SIZE_TYPE_DTYPE,
    dtype_from_pylibcudf_column,
    is_column_like,
)
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.utils import (
    _external_only_api,
    _is_same_name,
    _warn_no_dask_cudf,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Hashable, Iterable, MutableMapping
    from typing import Self

    import pyarrow as pa

    from cudf._typing import DataFrameOrSeries, Dtype
    from cudf.core.dataframe import DataFrame


def _maybe_indices_to_slice(indices: cp.ndarray) -> slice | cp.ndarray:
    """Makes best effort to convert an array of indices into a python slice.
    If the conversion is not possible, return input. `indices` are expected
    to be valid.
    """
    # TODO: improve efficiency by avoiding sync.
    if len(indices) == 1:
        x = indices[0].item()
        return slice(x, x + 1)
    if len(indices) == 2:
        x1, x2 = indices[0].item(), indices[1].item()
        return slice(x1, x2 + 1, x2 - x1)
    start, step = indices[0].item(), (indices[1] - indices[0]).item()
    stop = start + step * len(indices)
    if (indices == cp.arange(start, stop, step)).all():
        return slice(start, stop, step)
    return indices


class MultiIndex(Index):
    """A multi-level or hierarchical index.

    Provides N-Dimensional indexing into Series and DataFrame objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes: sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Not yet supported
    names: optional sequence of objects
        Names for each of the index levels.
    copy : bool, default False
        Copy the levels and codes.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.
        Not yet supported

    Attributes
    ----------
    names
    nlevels
    dtypes
    levels
    codes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_loc
    drop

    Returns
    -------
    MultiIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.MultiIndex(
    ... levels=[[1, 2], ['blue', 'red']], codes=[[0, 0, 1, 1], [1, 0, 1, 0]])
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               )
    """

    _levels: list[cudf.Index] | None
    _codes: list[ColumnBase] | None

    @_performance_tracking
    def __init__(
        self,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        dtype=None,
        copy=False,
        name=None,
        verify_integrity=True,
        nan_as_null=no_default,
    ) -> None:
        if nan_as_null is no_default:
            nan_as_null = (
                False if cudf.get_option("mode.pandas_compatible") else None
            )
        if isinstance(levels, (pd.MultiIndex, MultiIndex)):
            # TODO: Figure out why cudf.Index(pd.MultiIndex(...)) goes through here twice
            # Somehow due to from_pandas calling cls?
            return
        if sortorder is not None:
            raise NotImplementedError("sortorder is not yet supported")
        if name is not None:
            raise NotImplementedError(
                "Use `names`, `name` is not yet supported"
            )
        if levels is None or codes is None:
            raise TypeError("Must pass both levels and codes")
        elif not (is_list_like(levels) and len(levels) > 0):
            raise ValueError("Must pass non-zero length sequence of levels")
        elif not (is_list_like(codes) and len(codes) > 0):
            raise ValueError("Must pass non-zero length sequence of codes")
        elif len(codes) != len(levels):
            raise ValueError(
                f"levels must have the same length ({len(levels)}) "
                f"as codes ({len(codes)})."
            )

        new_levels: list[cudf.Index] = []
        for level in levels:
            new_level = ensure_index(level, nan_as_null=nan_as_null)
            if copy and new_level is level:
                new_level = new_level.copy(deep=True)
            new_levels.append(new_level)

        new_codes: list[ColumnBase] = []
        for code in codes:
            if not (is_list_like(code) or is_column_like(code)):
                raise TypeError("Each code must be list-like")
            new_code = as_column(code, dtype=np.dtype(np.int64))
            if copy and new_code is code:
                new_code = new_code.copy(deep=True)
            new_codes.append(new_code)

        source_data: dict[Hashable, ColumnBase] = {}
        for i, (code, level) in enumerate(
            zip(new_codes, new_levels, strict=True)
        ):
            if len(code):
                lo, hi = code.minmax()
                if lo < -1 or hi > len(level) - 1:
                    raise ValueError(
                        f"Codes must be -1 <= codes <= {len(level) - 1}"
                    )
                if lo == -1:
                    # Now we can gather and insert null automatically
                    code[code == -1] = np.iinfo(SIZE_TYPE_DTYPE).min
            result_col = level._column.take(code, nullify=True)
            # TODO: Pandas-3.0: Investigate with pandas test suite and remove
            # the following checks if they are obsolete now.
            # if (
            #     cudf.get_option("mode.pandas_compatible")
            #     and nan_as_null is False
            #     and not is_dtype_obj_numeric(result_col.dtype)
            #     and not is_pandas_nullable_extension_dtype(level.dtype)
            #     and result_col.has_nulls(include_nan=False)
            # ):
            #     raise MixedTypeError(
            #         "MultiIndex levels cannot have mixed types when `mode.pandas_compatible` is True and `nan_as_null` is False."
            #     )
            # if (
            #     cudf.get_option("mode.pandas_compatible")
            #     and not is_dtype_obj_numeric(result_col.dtype)
            #     and result_col.has_nulls(include_nan=False)
            #     and nan_as_null is False
            #     and not is_pandas_nullable_extension_dtype(level.dtype)
            # ):
            #     result_col = result_col.fillna(np.nan)
            source_data[i] = ColumnBase.create(
                result_col.plc_column, level.dtype
            )

        Frame.__init__(self, ColumnAccessor(source_data))
        self._levels = new_levels
        self._codes = new_codes
        self._name = None
        self.names = names

    @property
    @_performance_tracking
    def names(self):
        return self._names

    @names.setter
    @_performance_tracking
    def names(self, value):
        if value is None:
            value = [None] * self.nlevels
        elif not is_list_like(value):
            raise ValueError("Names should be list-like for a MultiIndex")
        elif len(value) != self.nlevels:
            raise ValueError(
                "Length of names must match number of levels in MultiIndex."
            )

        if len(value) == len(set(value)):
            # IMPORTANT: if the provided names are unique,
            # we reconstruct self._data with the names as keys.
            # If they are not unique, the keys of self._data
            # and self._names will be different, which can lead
            # to unexpected behavior in some cases. This is
            # definitely buggy, but we can't disallow non-unique
            # names either...
            self._data = type(self._data)(
                dict(zip(value, self._columns, strict=True)),
                level_names=self._data.level_names,
                verify=False,
            )
        self._names = pd.core.indexes.frozen.FrozenList(value)

    @_performance_tracking
    def to_series(self, index=None, name=None):
        raise NotImplementedError(
            "MultiIndex.to_series isn't implemented yet."
        )

    @_performance_tracking
    def astype(self, dtype: Dtype, copy: bool = True) -> Self:
        if not is_dtype_obj_string(cudf.dtype(dtype)):
            raise TypeError(
                "Setting a MultiIndex dtype to anything other than string is "
                "not supported"
            )
        return self

    @_performance_tracking
    def rename(self, names, inplace: bool = False) -> Self | None:
        """
        Alter MultiIndex level names

        Parameters
        ----------
        names : list of label
            Names to set, length must be the same as number of levels
        inplace : bool, default False
            If True, modifies objects directly, otherwise returns a new
            ``MultiIndex`` instance

        Returns
        -------
        None or MultiIndex

        Examples
        --------
        Renaming each levels of a MultiIndex to specified name:

        >>> midx = cudf.MultiIndex.from_product(
        ...     [('A', 'B'), (2020, 2021)], names=['c1', 'c2'])
        >>> midx.rename(['lv1', 'lv2'])
        MultiIndex([('A', 2020),
                    ('A', 2021),
                    ('B', 2020),
                    ('B', 2021)],
                names=['lv1', 'lv2'])
        >>> midx.rename(['lv1', 'lv2'], inplace=True)
        >>> midx
        MultiIndex([('A', 2020),
                    ('A', 2021),
                    ('B', 2020),
                    ('B', 2021)],
                names=['lv1', 'lv2'])

        ``names`` argument must be a list, and must have same length as
        ``MultiIndex.levels``:

        >>> midx.rename(['lv0'])
        Traceback (most recent call last):
        ValueError: Length of names must match number of levels in MultiIndex.

        """
        return self.set_names(names, level=None, inplace=inplace)

    @_performance_tracking
    def set_names(
        self, names, level=None, inplace: bool = False
    ) -> Self | None:
        names_is_list_like = is_list_like(names)
        level_is_list_like = is_list_like(level)

        if level is not None and not level_is_list_like and names_is_list_like:
            raise TypeError(
                "Names must be a string when a single level is provided."
            )

        if not names_is_list_like and level is None and self.nlevels > 1:
            raise TypeError("Must pass list-like as `names`.")

        if not names_is_list_like:
            names = [names]
        if level is not None and not level_is_list_like:
            level = [level]

        if level is not None and len(names) != len(level):
            raise ValueError("Length of names must match length of level.")
        if level is None and len(names) != self.nlevels:
            raise ValueError(
                "Length of names must match number of levels in MultiIndex."
            )

        if level is None:
            level = range(self.nlevels)
        else:
            level = [self._level_index_from_level(lev) for lev in level]

        existing_names = list(self.names)
        for i, lev in enumerate(level):
            existing_names[lev] = names[i]
        names = existing_names

        return self._set_names(names=names, inplace=inplace)

    def _maybe_materialize_codes_and_levels(
        self: Self, *, sort: bool = True
    ) -> Self:
        """
        Set self._codes and self._levels from self._columns _when_ needed.

        Factorization of self._columns to self._codes and self._levels is delayed
        due to being expensive and sometimes unnecessary for operations.

        MultiIndex methods are responsible for calling this when needed.

        ``sort`` controls the order of the materialized ``levels``. The default
        ``sort=True`` mirrors ``MultiIndex.from_arrays`` (which also factorizes
        with ``sort=True``), matching pandas' always-sorted levels for indexes
        built from raw data. Callers that build a ``MultiIndex`` with a
        meaningful level order (e.g. the result of ``stack``/``unstack``, which
        pandas constructs with explicit, *unsorted* levels) should pass
        ``sort=False`` so that a subsequent ``to_pandas()`` preserves that order.
        """
        if self._levels is None and self._codes is None:
            levels = []
            codes = []
            for col in self._data.values():
                code, cats = factorize(col, sort=sort)
                codes.append(as_column(code.astype(np.dtype(np.int64))))
                levels.append(cats)
            self._levels = levels
            self._codes = codes
        return self

    @classmethod
    @_performance_tracking
    def _from_data(
        cls,
        data: MutableMapping,
        name: Any = None,
    ) -> Self:
        """
        Use when you have a ColumnAccessor-like mapping but no codes and levels.

        Preferable to use _simple_new if you have codes and levels.
        """
        return cls._simple_new(
            data=ColumnAccessor(data),
            levels=None,
            codes=None,
            names=pd.core.indexes.frozen.FrozenList(data.keys()),
            name=name,
        )

    @property
    def _num_columns(self) -> int:
        return len(self._data)

    @_performance_tracking
    def _from_data_like_self(self, data: MutableMapping) -> Self:
        mi = type(self)._from_data(data, name=self.name)
        if mi.nlevels == self.nlevels:
            mi.names = self.names
        return mi

    @classmethod
    def _simple_new(
        cls,
        data: ColumnAccessor,
        levels: list[cudf.Index] | None,
        codes: list[ColumnBase] | None,
        names: pd.core.indexes.frozen.FrozenList,
        name: Any = None,
    ) -> Self:
        """
        Use when you have a ColumnAccessor-like mapping, codes, and levels.
        """
        mi = object.__new__(cls)
        mi._data = data
        mi._levels = levels
        mi._codes = codes
        mi._names = names
        mi._name = name
        return mi

    @property
    @_performance_tracking
    def name(self):
        return self._name

    @name.setter
    @_performance_tracking
    def name(self, value):
        self._name = value

    @_performance_tracking
    def copy(
        self,
        names=None,
        deep=False,
        name=None,
    ) -> Self:
        """Returns copy of MultiIndex object.

        Returns a copy of `MultiIndex`. The `levels` and `codes` value can be
        set to the provided parameters. When they are provided, the returned
        MultiIndex is always newly constructed.

        Parameters
        ----------
        names : sequence of objects, optional (default None)
            Names for each of the index levels.
        deep : Bool (default False)
            If True, `._data`, `._levels`, `._codes` will be copied. Ignored if
            `levels` or `codes` are specified.
        name : object, optional (default None)
            Kept for compatibility with 1-dimensional Index. Should not
            be used.

        Returns
        -------
        Copy of MultiIndex Instance

        Examples
        --------
        >>> df = cudf.DataFrame({'Close': [3400.00, 226.58, 3401.80, 228.91]})
        >>> idx1 = cudf.MultiIndex(
        ... levels=[['2020-08-27', '2020-08-28'], ['AMZN', 'MSFT']],
        ... codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        ... names=['Date', 'Symbol'])
        >>> idx2 = idx1.copy(
        ... names=['col1', 'col2'])

        >>> df.index = idx1
        >>> df
                             Close
        Date       Symbol
        2020-08-27 AMZN    3400.00
                   MSFT     226.58
        2020-08-28 AMZN    3401.80
                   MSFT     228.91

        >>> df.index = idx2
        >>> df
                           Close
        col1       col2
        2020-08-27 AMZN  3400.00
                   MSFT   226.58
        2020-08-28 AMZN  3401.80
                   MSFT   228.91
        """
        if names is not None:
            names = pd.core.indexes.frozen.FrozenList(names)
        else:
            names = self.names
        if self._levels is not None:
            levels: list[cudf.Index] | None = [
                idx.copy(deep=deep) for idx in self._levels
            ]
        else:
            levels = self._levels
        if self._codes is not None:
            codes: list[ColumnBase] | None = [
                code.copy(deep=deep) for code in self._codes
            ]
        else:
            codes = self._codes
        return type(self)._simple_new(
            data=self._data.copy(deep=deep),
            levels=levels,
            codes=codes,
            names=names,
            name=name,
        )

    @_performance_tracking
    def __repr__(self) -> str:
        max_seq_items = pd.get_option("display.max_seq_items") or len(self)

        if len(self) > max_seq_items:
            n = int(max_seq_items / 2) + 1
            # TODO: Update the following two arange calls to
            # a single arange call once arange has support for
            # a vector start/end points.
            indices = ColumnBase.from_range(range(n))
            indices = indices.append(
                ColumnBase.from_range(range(len(self) - n, len(self), 1))
            )
            preprocess = self.take(indices)
        else:
            preprocess = self

        arrays = []
        for name, col in zip(self.names, preprocess._columns, strict=True):
            try:
                pd_idx = col.to_pandas(nullable=True)
            except NotImplementedError:
                pd_idx = col.to_pandas(nullable=False)
            pd_idx.name = name
            arrays.append(pd_idx)

        preprocess_pd = pd.MultiIndex.from_arrays(arrays)

        output = repr(preprocess_pd)
        output_prefix = self.__class__.__name__ + "("
        output = output.lstrip(output_prefix)
        lines = output.split("\n")

        if len(lines) > 1:
            if "length=" in lines[-1] and len(self) != len(preprocess_pd):
                last_line = lines[-1]
                length_index = last_line.index("length=")
                last_line = last_line[:length_index] + f"length={len(self)})"
                lines = lines[:-1]
                lines.append(last_line)

        data_output = "\n".join(lines)
        return output_prefix + data_output

    @property
    @_external_only_api("Use ._codes instead")
    @_performance_tracking
    def codes(self) -> pd.core.indexes.frozen.FrozenList:
        """
        Returns the codes of the underlying MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a':[1, 2, 3], 'b':[10, 11, 12]})
        >>> midx = cudf.MultiIndex.from_frame(df)
        >>> midx
        MultiIndex([(1, 10),
                    (2, 11),
                    (3, 12)],
                names=['a', 'b'])
        >>> midx.codes
        FrozenList([[0, 1, 2], [0, 1, 2]])
        """
        self._maybe_materialize_codes_and_levels()
        return pd.core.indexes.frozen.FrozenList(
            col.values
            for col in self._codes  # type: ignore[union-attr]
        )

    def get_slice_bound(self, label, side):
        raise NotImplementedError(
            "get_slice_bound is not currently implemented."
        )

    @property
    @_performance_tracking
    def levels(self) -> list[cudf.Index]:
        """
        Returns list of levels in the MultiIndex

        Returns
        -------
        List of Index objects

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a':[1, 2, 3], 'b':[10, 11, 12]})
        >>> cudf.MultiIndex.from_frame(df)
        MultiIndex([(1, 10),
                    (2, 11),
                    (3, 12)],
                names=['a', 'b'])
        >>> midx = cudf.MultiIndex.from_frame(df)
        >>> midx
        MultiIndex([(1, 10),
                    (2, 11),
                    (3, 12)],
                names=['a', 'b'])
        >>> midx.levels
        [Index([1, 2, 3], dtype='int64', name='a'), Index([10, 11, 12], dtype='int64', name='b')]
        """
        self._maybe_materialize_codes_and_levels()
        return [
            idx.rename(name)  # type: ignore[misc]
            for idx, name in zip(self._levels, self.names, strict=True)  # type: ignore[arg-type]
        ]

    @property
    @_performance_tracking
    def ndim(self) -> int:
        """Dimension of the data. For MultiIndex ndim is always 2."""
        return 2

    @_performance_tracking
    def _get_level_label(self, level):
        """Get name of the level.

        Parameters
        ----------
        level : int or level name
            if level is name, it will be returned as it is
            else if level is index of the level, then level
            label will be returned as per the index.
        """
        if level in self.names:
            return level
        else:
            return self.names[level]

    @_performance_tracking
    def isin(self, values, level=None) -> cp.ndarray:
        """Return a boolean array where the index values are in values.

        Compute boolean array of whether each index value is found in
        the passed set of values. The length of the returned boolean
        array matches the length of the index.

        Parameters
        ----------
        values : set, list-like, Index or Multi-Index
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index
            is a MultiIndex).

        Returns
        -------
        is_contained : cupy array
            CuPy array of boolean values.

        Notes
        -----
        When `level` is None, `values` can only be MultiIndex, or a
        set/list-like tuples.
        When `level` is provided, `values` can be Index or MultiIndex,
        or a set/list-like tuples.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> midx = cudf.from_pandas(pd.MultiIndex.from_arrays([[1,2,3],
        ...                                  ['red', 'blue', 'green']],
        ...                                  names=('number', 'color')))
        >>> midx
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(['red', 'orange', 'yellow'], level='color')
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, 'red'), (3, 'red')])
        array([ True, False, False])
        """
        if level is None:
            if isinstance(values, cudf.MultiIndex):
                values_idx = values
            elif (
                (
                    isinstance(
                        values,
                        (
                            cudf.Series,
                            cudf.Index,
                            cudf.DataFrame,
                            ColumnBase,
                        ),
                    )
                )
                or (not is_list_like(values))
                or (
                    is_list_like(values)
                    and len(values) > 0
                    and not isinstance(values[0], tuple)
                )
            ):
                raise TypeError(
                    "values need to be a Multi-Index or set/list-like tuple "
                    "squences  when `level=None`."
                )
            else:
                values_idx = cudf.MultiIndex.from_tuples(
                    values, names=self.names
                )
            self_df = self.to_frame(index=False).reset_index()
            values_df = values_idx.to_frame(index=False)
            idx = self_df.merge(values_df, how="leftsemi")._data["index"]
            res = as_column(False, length=len(self))
            res[idx] = True
            result = res.values
        else:
            level_series = self.get_level_values(level)
            result = level_series.isin(values)

        return result

    def where(self, cond, other=None, inplace=False):
        raise NotImplementedError(
            ".where is not supported for MultiIndex operations"
        )

    @_performance_tracking
    def _compute_validity_mask(self, index, row_tuple, max_length):
        """Computes the valid set of indices of values in the lookup"""
        # An all-wildcard per-level key (every component ``slice(None)``, e.g.
        # ``df.loc[(slice(None), slice(None))]``) is an identity selection of
        # all rows. Handle it directly: skipping every component below would
        # build a 0-column lookup whose merge has no common columns and raises.
        if isinstance(row_tuple, tuple) and all(
            isinstance(row, slice) and row == slice(None) for row in row_tuple
        ):
            return ColumnBase.from_range(range(max_length))
        # The pandas-matching lookup semantics below (cartesian-product of the
        # per-level keys, de-duplicated indexer labels, deterministic
        # result ordering, and strict KeyError for missing labels/combinations)
        # are intentionally gated on ``mode.pandas_compatible`` rather than
        # applied unconditionally. Ungating is correctness-safe (the cuDF unit
        # suite passes either way), but it would change and slow down the
        # classic (non-compatible) fast path:
        #   * The deterministic ``sort_values`` was already gated here before
        #     this code (see the original "deterministic order" comment and the
        #     ``# TODO: Remove this after merge/join obtain deterministic
        #     ordering`` note) -- the merge is order-unstable, so the sort is a
        #     pandas-compat-only cost that classic mode deliberately skips.
        #   * The per-level membership checks add a GPU scan per level, and the
        #     multi-level monotonic check needs a host round-trip
        #     (``to_pandas().apply(tuple, ...)``), on every ``.loc`` call.
        #   * Classic mode preserves its historical lenient behavior (positional
        #     zip, no dedup, empty-result instead of KeyError on a missing
        #     combination).
        pandas_compatible = cudf.get_option("mode.pandas_compatible")
        if pandas_compatible:
            # Build the lookup as the cartesian product, in key order, of the
            # per-level key lists (pandas semantics). A positional zip of the
            # level lists would only match the diagonal combinations.
            positions = []
            keylists = []
            for i, row in enumerate(row_tuple):
                if isinstance(row, slice) and row == slice(None):
                    continue
                positions.append(i)
                keylists.append(
                    list(row)
                    if (is_list_like(row) and not isinstance(row, tuple))
                    else [row]
                )
            product = list(itertools.product(*keylists))
            lookup_dict = {
                pos: as_column([combo[j] for combo in product])
                for j, pos in enumerate(positions)
            }
        else:
            # TODO: A non-slice(None) will probably raise in as_column
            lookup_dict = {
                i: as_column(row)
                for i, row in enumerate(row_tuple)
                if not (isinstance(row, slice) and row == slice(None))
            }
        lookup = cudf.DataFrame._from_data(lookup_dict)
        frame = cudf.DataFrame._from_data(
            ColumnAccessor(
                dict(enumerate(index._columns)),
                verify=False,
            )
        )

        data_table = cudf.concat(
            [
                frame,
                cudf.DataFrame._from_data(
                    ColumnAccessor(
                        {"idx": ColumnBase.from_range(range(len(frame)))},
                        verify=False,
                    )
                ),
            ],
            axis=1,
        )
        # Sort indices in pandas compatible mode
        # because we want the indices to be fetched
        # in a deterministic order.
        # TODO: Remove this after merge/join
        # obtain deterministic ordering.
        if pandas_compatible:
            # Drop duplicate lookup keys so that duplicate labels in the
            # indexer do not cause a cartesian explosion in the merge
            # (pandas GH#40978).
            lookup = lookup.drop_duplicates(keep="first")
            # pandas returns rows in the natural (source-index) order only when
            # the source index is lexsorted AND the lookup keys are themselves
            # monotonically increasing (so indexer order == natural order).
            # Otherwise it returns rows in the order the labels appear in the
            # indexer.
            if lookup._num_columns <= 1:
                lookup_monotonic = (
                    lookup._num_columns == 0
                    or lookup._columns[0].is_monotonic_increasing
                )
            else:
                lookup_monotonic = pd.MultiIndex.from_frame(
                    lookup.to_pandas()
                ).is_monotonic_increasing
            natural_order = lookup_monotonic and index.is_monotonic_increasing
            lookup_order = "_" + "_".join(map(str, lookup._column_names))
            lookup[lookup_order] = ColumnBase.from_range(range(len(lookup)))
            if natural_order:
                postprocess = operator.methodcaller("sort_values", by=["idx"])
            else:
                postprocess = operator.methodcaller(
                    "sort_values", by=[lookup_order, "idx"]
                )
        else:
            postprocess = lambda r: r  # noqa: E731
        result = postprocess(lookup.merge(data_table))["idx"]
        if pandas_compatible:
            # Every requested label (scalar or list element) must exist in its
            # level, and the requested combination must match at least one
            # row; otherwise pandas raises KeyError.
            for idx, row in enumerate(row_tuple):
                if isinstance(row, slice) and row == slice(None):
                    continue
                level_col = index.levels[idx]._column
                if is_list_like(row) and not isinstance(row, tuple):
                    # Bulk membership check (one pass) rather than a scan per
                    # element; only fall back to a host loop to surface the
                    # first missing label when something is actually absent.
                    present = as_column(list(row)).isin(level_col)
                    if not present.all():
                        for element, ok in zip(
                            row, present.values.get(), strict=True
                        ):
                            if not ok:
                                raise KeyError(element)
                elif row not in level_col:
                    raise KeyError(row)
            if len(result) == 0:
                raise KeyError(row_tuple)
            return result
        # Avoid computing levels unless the result of the merge is empty,
        # which suggests that a KeyError should be raised.
        if len(result) == 0:
            for idx, row in enumerate(row_tuple):
                if row == slice(None):
                    continue
                if row not in index.levels[idx]._column:
                    raise KeyError(row)
        return result

    @_performance_tracking
    def _get_valid_indices_by_tuple(self, index, row_tuple, max_length):
        # Instructions for Slicing
        # if tuple, get first and last elements of tuple
        # if open beginning tuple, get 0 to highest valid_index
        # if open ending tuple, get highest valid_index to len()
        # if not open end or beginning, get range lowest beginning index
        # to highest ending index
        if isinstance(row_tuple, slice):
            if row_tuple.step not in (None, 1):
                # Strided / reversed MultiIndex label slicing is not modeled by
                # the label-range machinery below (it ignores the step and
                # reversal, silently returning the wrong rows). Raise so that
                # pandas-compatible callers fall back to pandas' correct
                # label-slice handling instead of producing a wrong result.
                raise NotImplementedError(
                    "MultiIndex label slicing with a step is not supported"
                )
            if (
                isinstance(row_tuple.start, numbers.Number)
                or isinstance(row_tuple.stop, numbers.Number)
                or row_tuple == slice(None)
            ):
                stop = row_tuple.stop or max_length
                start, stop, step = row_tuple.indices(stop)
                return ColumnBase.from_range(range(start, stop, step))
            start_values = self._compute_validity_mask(
                index, row_tuple.start, max_length
            )
            stop_values = self._compute_validity_mask(
                index, row_tuple.stop, max_length
            )
            return ColumnBase.from_range(
                range(start_values.min(), stop_values.max() + 1)
            )
        elif isinstance(row_tuple, numbers.Number):
            # A scalar row key in .loc is a LABEL on the first level, not a
            # positional index. Look it up by label so a partial scalar key
            # (e.g. ``df.loc[10]``) selects matching rows and drops the level.
            return self._compute_validity_mask(index, (row_tuple,), max_length)
        return self._compute_validity_mask(index, row_tuple, max_length)

    @_performance_tracking
    def _index_and_downcast(self, result, index, index_key, per_level=False):
        if isinstance(index_key, (numbers.Number, slice)):
            index_key = (index_key,)
        # A bare slice or list-like of level-0 labels (not a per-level tuple)
        # keeps every level; only set the index on a DataFrame result.
        if not isinstance(index_key, tuple):
            if isinstance(result, cudf.DataFrame):
                result.index = index
            return result

        nlevels = index.nlevels
        cols = list(index._columns)
        names = list(index.names)

        if nlevels == 1:
            # On a 1-level MultiIndex, keep the result as-is (a DataFrame with
            # the MultiIndex preserved) rather than dropping the level or
            # downcasting to a Series.
            if isinstance(result, cudf.DataFrame):
                result.index = index
            return result

        if per_level:
            # Series.loc / DataFrame "Form A" (a single per-level tuple): drop
            # exactly the levels selected by a SCALAR, positionally; keep the
            # levels selected by a slice or a list-like.
            sel = list(index_key) + [slice(None)] * (nlevels - len(index_key))
            keep = [i for i, s in enumerate(sel) if not is_scalar(s)]
        else:
            # DataFrame two-axis ``loc[rowkey, colkey]``: an all-scalar partial
            # row tuple drops that many LEADING levels (pandas' all-or-nothing
            # rule); any slice/list-like keeps all levels.
            if all(is_scalar(s) for s in index_key):
                keep = list(range(len(index_key), nlevels))
            else:
                keep = list(range(nlevels))

        if len(keep) == 0:
            # Every level was scalar-selected: collapse one dimension.
            if isinstance(result, cudf.DataFrame):
                if len(result) > 1:
                    # Duplicate full-key matches: pandas keeps every matched
                    # row as a DataFrame rather than collapsing a dimension.
                    return result
                if len(result) == 1:
                    result = result.T
                    return result[result._column_names[0]]
                # Pandas returns an empty Series named by the requested key.
                return cudf.Series._from_data(
                    {},
                    name=tuple(
                        col.element_indexing(0) if len(col) else None
                        for col in cols
                    ),
                )
            return result

        if len(keep) == 1:
            result.index = cudf.Index._from_column(
                cols[keep[0]], name=names[keep[0]]
            )
        else:
            new_index = MultiIndex._from_data(
                {i: cols[k] for i, k in enumerate(keep)}
            )
            new_index.names = [names[k] for k in keep]
            result.index = new_index
        return result

    @_performance_tracking
    def _get_row_major(
        self,
        df: DataFrameOrSeries,
        row_tuple: numbers.Number
        | slice
        | tuple[Any, ...]
        | list[tuple[Any, ...]],
        per_level: bool = False,
    ) -> DataFrameOrSeries:
        def _reject_sets(x):
            # pandas rejects a set or dict anywhere in a .loc indexer. Scan
            # recursively so it survives ``__getitem__``'s re-nesting retry as
            # well.
            if isinstance(x, (set, frozenset)):
                raise TypeError(
                    "Passing a set as an indexer is not supported. "
                    "Use a list instead."
                )
            if isinstance(x, dict):
                raise TypeError(
                    "Passing a dict as an indexer is not supported. "
                    "Use a list instead."
                )
            if isinstance(x, tuple):
                for el in x:
                    _reject_sets(el)

        _reject_sets(row_tuple)
        if isinstance(row_tuple, slice):
            if row_tuple.step == 0:
                raise ValueError("slice step cannot be zero")
            if row_tuple.start is None:
                row_tuple = slice(self[0], row_tuple.stop, row_tuple.step)
            if row_tuple.stop is None:
                row_tuple = slice(row_tuple.start, self[-1], row_tuple.step)
            if isinstance(row_tuple.start, bool) or isinstance(
                row_tuple.stop, bool
            ):
                raise TypeError(
                    f"{row_tuple}: boolean values can not be used in a slice"
                )
        self._validate_indexer(row_tuple)
        valid_indices = self._get_valid_indices_by_tuple(
            df.index, row_tuple, len(df)
        )
        if isinstance(valid_indices, ColumnBase):
            indices = cudf.Series._from_column(valid_indices)
        else:
            indices = cudf.Series(valid_indices)
        result = df.take(indices)
        final = self._index_and_downcast(
            result, result.index, row_tuple, per_level=per_level
        )
        return final

    @_performance_tracking
    def _validate_indexer(
        self,
        indexer: numbers.Number
        | slice
        | tuple[Any, ...]
        | list[tuple[Any, ...]],
    ) -> None:
        if isinstance(indexer, numbers.Number):
            return
        if isinstance(indexer, tuple):
            # drop any slice(None) from the end:
            indexer = tuple(
                itertools.dropwhile(
                    lambda x: x == slice(None), reversed(indexer)
                )
            )[::-1]

            # now check for size
            if len(indexer) > self.nlevels:
                raise IndexError("Indexer size exceeds number of levels")
        elif isinstance(indexer, slice):
            self._validate_indexer(indexer.start)
            self._validate_indexer(indexer.stop)
        else:
            for i in indexer:
                self._validate_indexer(i)

    @_performance_tracking
    def __eq__(self, other):
        if isinstance(other, MultiIndex):
            return np.array(
                [
                    self_col.equals(other_col)
                    for self_col, other_col in zip(
                        self._columns, other._columns, strict=True
                    )
                ]
            )
        return NotImplemented

    @property
    @_performance_tracking
    def size(self) -> int:
        # The size of a MultiIndex is only dependent on the number of rows.
        return self._num_rows

    @_performance_tracking
    def take(self, indices) -> Self:
        if isinstance(indices, cudf.Series) and indices.has_nulls:
            raise ValueError("Column must have no nulls.")
        obj = super().take(indices)
        obj.names = self.names
        return obj

    @_performance_tracking
    def serialize(self):
        header, frames = super().serialize()
        # Overwrite the names in _data with the true names.
        header["column_names"] = self.names
        return header, frames

    @classmethod
    @_performance_tracking
    def deserialize(cls, header, frames):
        # Spoof the column names to construct the frame, then set manually.
        column_names = header["column_names"]
        header["column_names"] = range(0, len(column_names))
        obj = super().deserialize(header, frames)
        return obj._set_names(column_names)

    @_performance_tracking
    def __getitem__(self, index):
        flatten = is_integer(index)

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            idx = range(start, stop, step)
        elif is_scalar(index):
            if isinstance(index, float):
                raise IndexError("indexing with a float is disallowed.")
            idx = [index]
        else:
            idx = index

        indexer = as_column(idx)
        ca = self._data._from_columns_like_self(
            (col.take(indexer) for col in self._columns), verify=False
        )
        if self._codes is not None:
            codes = [code.take(indexer) for code in self._codes]
        else:
            codes = self._codes
        result = type(self)._simple_new(
            data=ca, codes=codes, levels=self._levels, names=self.names
        )

        # we are indexing into a single row of the MultiIndex,
        # return that row as a tuple:
        if flatten:
            return result.to_pandas()[0]
        else:
            return result

    @_performance_tracking
    def equals(self, other) -> bool:
        return Frame.equals(self, other)

    @_performance_tracking
    def to_arrow(self) -> pa.Table:
        return Frame.to_arrow(self)

    @_performance_tracking
    def to_frame(
        self,
        index: bool = True,
        name=no_default,
        allow_duplicates: bool = False,
    ) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.
        name : list / sequence of str, optional
            The passed names should substitute index level names.
        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created. Note
            that this parameter is non-functional because
            duplicates column labels aren't supported in cudf.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> mi = cudf.MultiIndex.from_tuples([('a', 'c'), ('b', 'd')])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=['x', 'y'])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        if name is no_default:
            column_names = [
                level if name is None else name
                for level, name in enumerate(self.names)
            ]
        elif not is_list_like(name):
            raise TypeError(
                "'name' must be a list / sequence of column names."
            )
        elif len(name) != self.nlevels:
            raise ValueError(
                "'name' should have the same length as "
                "number of levels on index."
            )
        else:
            column_names = name

        if len(column_names) != len(set(column_names)):
            raise ValueError("Duplicate column names are not allowed")
        ca = ColumnAccessor(
            dict(
                zip(
                    column_names,
                    (col.copy() for col in self._columns),
                    strict=True,
                )
            ),
            verify=False,
        )
        return cudf.DataFrame._from_data(
            data=ca, index=self if index else None
        )

    @_performance_tracking
    def _level_to_ca_label(self, level) -> tuple[Hashable, int]:
        """
        Convert a level to a ColumAccessor label and an integer position.

        Useful if self._column_names != self.names.

        Parameters
        ----------
        level : int or label

        Returns
        -------
        tuple[Hashable, int]
            (ColumnAccessor label corresponding to level, integer position of the level)
        """
        colnames = self._column_names
        try:
            level_idx = colnames.index(level)
        except ValueError:
            if isinstance(level, int):
                if level < 0:
                    level = level + len(colnames)
                if level < 0 or level >= len(colnames):
                    raise IndexError(f"Invalid level number: '{level}'")
                level_idx = level
                level = colnames[level_idx]
            elif level in self.names:
                level_idx = list(self.names).index(level)
                level = colnames[level_idx]
            else:
                raise KeyError(f"Level not found: '{level}'")
        return level, level_idx

    @_performance_tracking
    def get_level_values(self, level) -> Index:
        """
        Return the values at the requested level

        Parameters
        ----------
        level : int or label

        Returns
        -------
        An Index containing the values at the requested level.
        """
        level, level_idx = self._level_to_ca_label(level)
        level_values = cudf.Index._from_column(
            self._data[level], name=self.names[level_idx]
        )
        return level_values

    @classmethod
    @_performance_tracking
    def _concat(cls, objs) -> Self:
        # TODO: This will discard previously computed self._codes and self._levels.
        # Try preserving them if defined.
        source_data = [o.to_frame(index=False) for o in objs]

        # TODO: Verify if this is really necessary or if we can rely on
        # DataFrame._concat.
        if len(source_data) > 1:
            colnames = source_data[0]._data.to_pandas_index
            for obj in source_data[1:]:
                obj.columns = colnames

        source_df = cudf.DataFrame._concat(source_data)
        try:
            # Only set names if all objs have the same names
            (names,) = {o.names for o in objs} - {None}
        except ValueError:
            names = [None] * source_df._num_columns
        return cudf.MultiIndex.from_frame(source_df, names=names)

    @classmethod
    @_performance_tracking
    def from_tuples(
        cls, tuples, sortorder: int | None = None, names=None
    ) -> Self:
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> cudf.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_tuples(
            tuples, sortorder=sortorder, names=names
        )
        return cls(levels=pdi.levels, codes=pdi.codes, names=pdi.names)

    @property
    def dtypes(self) -> pd.Series:
        """
        Return the dtypes as a Series for the underlying MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.MultiIndex.from_product([(0, 1, 2), ('green', 'purple')],
        ...                                  names=['number', 'color'])
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.dtypes
        number     int64
        color       str
        dtype: object
        """
        # Not using DataFrame.dtypes to avoid expensive invocation of `._data.to_pandas_index`
        return pd.Series(dict(self.to_frame()._dtypes))

    @_performance_tracking
    def to_numpy(self) -> np.ndarray:
        """
        Return a numpy representation of the MultiIndex.

        Only the values in the MultiIndex will be returned.

        Returns
        -------
        out : numpy.ndarray
            The values of the MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(
        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...     )
        >>> midx.to_numpy()
        array([(1, 1), (1, 5), (3, 2), (4, 2), (5, 1)], dtype=object)
        >>> type(midx.to_numpy())
        <class 'numpy.ndarray'>
        """
        return self.to_pandas().values

    def to_flat_index(self):
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        This is not currently implemented
        """
        # TODO: Could implement as Index of ListDtype?
        raise NotImplementedError("to_flat_index is not currently supported.")

    @property
    @_performance_tracking
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the MultiIndex.

        Only the values in the MultiIndex will be returned.

        Returns
        -------
        out: cupy.ndarray
            The values of the MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(
        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...     )
        >>> midx.values
        array([[1, 1],
            [1, 5],
            [3, 2],
            [4, 2],
            [5, 1]])
        >>> type(midx.values)
        <class 'cupy...ndarray'>
        """
        if cudf.get_option("mode.pandas_compatible"):
            raise NotImplementedError(
                "Unable to create a cupy array with tuples."
            )
        return Frame.to_cupy(self)

    @classmethod
    @_performance_tracking
    def from_arrow(cls, data: pa.Table) -> Self:
        return cls._from_data(Frame.from_arrow(data)._data)

    @classmethod
    @_performance_tracking
    def from_frame(
        cls,
        df: pd.DataFrame | DataFrame,
        sortorder: int | None = None,
        names=None,
    ) -> Self:
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        sortorder : int, optional
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip
        >>> cudf.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> cudf.MultiIndex.from_frame(df, names=['state', 'observation'])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        if isinstance(df, pd.DataFrame):
            source_data = cudf.DataFrame(df)
        elif isinstance(df, cudf.DataFrame):
            source_data = df
        else:
            raise TypeError("Input must be a pandas or cudf DataFrame.")
        names = names if names is not None else source_data._column_names
        return cls.from_arrays(
            source_data._columns, sortorder=sortorder, names=names
        )

    @classmethod
    @_performance_tracking
    def from_product(
        cls, iterables, sortorder: int | None = None, names=None
    ) -> Self:
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.
            If not explicitly provided, names will be inferred from the
            elements of iterables if an element has a name attribute

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> cudf.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_product(
            iterables, sortorder=sortorder, names=names
        )
        return cls(levels=pdi.levels, codes=pdi.codes, names=pdi.names)

    @classmethod
    @_performance_tracking
    def from_arrays(
        cls,
        arrays,
        sortorder=None,
        names=None,
    ) -> Self:
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : optional int
            Not yet supported
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> cudf.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        error_msg = "Input must be a list / sequence of array-likes."
        if not is_list_like(arrays):
            raise TypeError(error_msg)
        codes = []
        levels = []
        names_from_arrays = []

        for array in arrays:
            if not (is_list_like(array) or is_column_like(array)):
                raise TypeError(error_msg)
            code, level = factorize(array, sort=True)
            codes.append(code)
            levels.append(level)
            names_from_arrays.append(getattr(array, "name", None))
        if names is None:
            names = names_from_arrays

        return cls(
            codes=codes, levels=levels, sortorder=sortorder, names=names
        )

    @_performance_tracking
    def swaplevel(self, i=-2, j=-1) -> Self:
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int or str, default -2
            First level of index to be swapped.
        j : int or str, default -1
            Second level of index to be swapped.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> mi = cudf.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
            ('a', 'aa'),
            ('b', 'bb'),
            ('b', 'aa')],
           )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a'),
            ('aa', 'a'),
            ('bb', 'b'),
            ('aa', 'b')],
           )
        """
        lvl_i = self._level_index_from_level(i)
        lvl_j = self._level_index_from_level(j)
        order = list(range(self.nlevels))
        order[lvl_i], order[lvl_j] = order[lvl_j], order[lvl_i]
        keys = self._column_names
        columns = self._columns
        # TODO: Preserve self._codes and self._levels if set
        midx = type(self)._from_data({keys[k]: columns[k] for k in order})
        # the accessor keys may be positional stand-ins (e.g. for
        # duplicated or unset level names), so carry the level names
        # over from ``self.names`` rather than deriving them from keys
        midx.names = [self.names[k] for k in order]
        return midx

    @_performance_tracking
    def droplevel(self, level=-1) -> Self | Index:
        """
        Removes the specified levels from the MultiIndex.

        Parameters
        ----------
        level : level name or index, list-like
            Integer, name or list of such, specifying one or more
            levels to drop from the MultiIndex

        Returns
        -------
        A MultiIndex or Index object, depending on the number of remaining
        levels.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.MultiIndex.from_frame(
        ...     cudf.DataFrame(
        ...         {
        ...             "first": ["a", "a", "a", "b", "b", "b"],
        ...             "second": [1, 1, 2, 2, 3, 3],
        ...             "third": [0, 1, 2, 0, 1, 2],
        ...         }
        ...     )
        ... )

        Dropping level by index:

        >>> idx.droplevel(0)
        MultiIndex([(1, 0),
                    (1, 1),
                    (2, 2),
                    (2, 0),
                    (3, 1),
                    (3, 2)],
                   names=['second', 'third'])

        Dropping level by name:

        >>> idx.droplevel("first")
        MultiIndex([(1, 0),
                    (1, 1),
                    (2, 2),
                    (2, 0),
                    (3, 1),
                    (3, 2)],
                   names=['second', 'third'])

        Dropping multiple levels:

        >>> idx.droplevel(["first", "second"])
        Index([0, 1, 2, 0, 1, 2], dtype='int64', name='third')
        """
        if is_scalar(level):
            level = (level,)
        elif len(level) == 0:
            return self

        new_names = list(self.names)
        new_data = self._data.copy(deep=False)
        for i in sorted(
            (self._level_index_from_level(lev) for lev in level), reverse=True
        ):
            new_names.pop(i)
            new_data.pop(self._data.names[i])

        if len(new_data) == 1:
            return Index._from_column(
                next(iter(new_data.values())), name=new_names[0]
            )
        else:
            mi = type(self)._from_data(new_data)
            mi.names = new_names
            return mi

    @_performance_tracking
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.MultiIndex:
        # cudf uses np.iinfo(SIZE_TYPE_DTYPE).min as missing code
        # pandas uses -1 as missing code
        self._maybe_materialize_codes_and_levels()
        pd_codes = (
            code.find_and_replace(
                as_column(np.iinfo(SIZE_TYPE_DTYPE).min, length=1),
                as_column(-1, length=1),
            )
            for code in self._codes  # type: ignore[union-attr]
        )
        return pd.MultiIndex(
            levels=[
                level.to_pandas(nullable=nullable, arrow_type=arrow_type)
                for level in self.levels
            ],
            codes=[col.to_numpy() for col in pd_codes],
            names=self.names,
        )

    @cached_property
    @_performance_tracking
    def is_unique(self) -> bool:
        return len(self) == self.nunique(dropna=False)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype("O")

    @_performance_tracking
    def _is_sorted(self, ascending: bool) -> bool:
        return sorting.is_sorted(
            self._columns,
            ascending=itertools.repeat(ascending, times=self._num_columns),
            na_position=itertools.repeat("first", times=self._num_columns),
        )

    @cached_property
    @_performance_tracking
    def is_monotonic_increasing(self) -> bool:
        """
        Return if the index is monotonic increasing
        (only equal or increasing) values.
        """
        return self._is_sorted(True)

    @cached_property
    @_performance_tracking
    def is_monotonic_decreasing(self) -> bool:
        """
        Return if the index is monotonic decreasing
        (only equal or decreasing) values.
        """
        return self._is_sorted(False)

    @_performance_tracking
    def fillna(self, value) -> Self:
        """
        Fill null values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill nulls. This value cannot be a
            list-likes.

        Returns
        -------
        filled : MultiIndex

        Examples
        --------
        >>> import cudf
        >>> index = cudf.MultiIndex(
        ...         levels=[["a", "b", "c", None], ["1", None, "5"]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...       )
        >>> index
        MultiIndex([('a', '1'),
                    ('a', '5'),
                    ('b', nan),
                    ('c', nan),
                    (nan, '1')],
                   names=['x', 'y'])
        >>> index.fillna('hello')
        MultiIndex([(    'a',     '1'),
                    (    'a',     '5'),
                    (    'b', 'hello'),
                    (    'c', 'hello'),
                    ('hello',     '1')],
                   names=['x', 'y'])
        """
        return super()._fillna(value=value)

    @_performance_tracking
    def unique(self, level: int | None = None) -> Self | Index:
        if level is None:
            return self.drop_duplicates(keep="first")
        else:
            return self.get_level_values(level).unique()

    def _factorize(
        self, sort: bool, use_na_sentinel: bool
    ) -> tuple[cp.ndarray, MultiIndex]:
        if any(col.has_nulls() for col in self._columns):
            raise NotImplementedError(
                "factorize on a MultiIndex with missing values is not yet "
                "supported"
            )
        if len(self) == 0:
            return cp.empty(0, dtype=np.intp), self.copy()

        level_labels = list(range(self.nlevels))
        df = cudf.DataFrame._from_data(
            dict(zip(level_labels, self._columns, strict=True))
        )
        pos_label = self.nlevels
        df[pos_label] = range(len(df))

        # the uniques are the distinct rows in order of first appearance
        # (or sorted lexicographically when requested)
        uniques = (
            df.groupby(level_labels, sort=False)
            .agg({pos_label: "min"})
            .reset_index()
        )
        uniques = uniques.sort_values(
            by=level_labels if sort else pos_label, ignore_index=True
        )
        uid_label = self.nlevels + 1
        uniques[uid_label] = range(len(uniques))

        # codes: map each row to its unique-row id, in the original order
        merged = df.merge(
            uniques[[*level_labels, uid_label]],
            on=level_labels,
            how="left",
        )
        codes = (
            merged.sort_values(by=pos_label)[uid_label]
            .astype(np.dtype(np.intp))
            .values
        )
        # pandas does not propagate the level names to the uniques
        uniques_mi = MultiIndex._from_data(
            {label: uniques._data[label] for label in level_labels}
        )
        uniques_mi.names = [None] * self.nlevels
        return codes, uniques_mi

    @_performance_tracking
    def nunique(self, dropna: bool = True) -> int:
        mi = self.dropna(how="all") if dropna else self
        return len(mi.unique())

    @_performance_tracking
    def memory_usage(self, deep: bool = False) -> int:
        usage = sum(col.memory_usage for col in self._columns)
        if self._levels is not None:
            usage += sum(
                level.memory_usage(deep=deep) for level in self._levels
            )
        if self._codes is not None:
            usage += sum(code.memory_usage for code in self._codes)
        return usage

    @_performance_tracking
    def difference(self, other, sort=None) -> Self:
        if hasattr(other, "to_pandas"):
            other = other.to_pandas()
        return cudf.from_pandas(self.to_pandas().difference(other, sort))

    @_performance_tracking
    def append(self, other) -> Self:
        """
        Append a collection of MultiIndex objects together

        Parameters
        ----------
        other : MultiIndex or list/tuple of MultiIndex objects

        Returns
        -------
        appended : Index

        Examples
        --------
        >>> import cudf
        >>> idx1 = cudf.MultiIndex(
        ...     levels=[[1, 2], ['blue', 'red']],
        ...     codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        ... )
        >>> idx2 = cudf.MultiIndex(
        ...     levels=[[3, 4], ['blue', 'red']],
        ...     codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        ... )
        >>> idx1
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   )
        >>> idx2
        MultiIndex([(3,  'red'),
                    (3, 'blue'),
                    (4,  'red'),
                    (4, 'blue')],
                   )
        >>> idx1.append(idx2)
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue'),
                    (3,  'red'),
                    (3, 'blue'),
                    (4,  'red'),
                    (4, 'blue')],
                   )
        """
        if isinstance(other, (list, tuple)):
            to_concat = [self, *other]
        else:
            to_concat = [self, other]

        for obj in to_concat:
            if not isinstance(obj, MultiIndex):
                raise TypeError(
                    f"all objects should be of type "
                    f"MultiIndex for MultiIndex.append, "
                    f"found object of type: {type(obj)}"
                )

        return type(self)._concat(to_concat)

    @_performance_tracking
    def __array_function__(self, func, types, args, kwargs):
        cudf_df_module = MultiIndex

        for submodule in func.__module__.split(".")[1:]:
            # point cudf to the correct submodule
            if hasattr(cudf_df_module, submodule):
                cudf_df_module = getattr(cudf_df_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [cudf_df_module, np.ndarray]

        for t in types:
            if t not in handled_types:
                return NotImplemented

        if hasattr(cudf_df_module, fname):
            cudf_func = getattr(cudf_df_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            else:
                return cudf_func(*args, **kwargs)
        else:
            return NotImplemented

    def _level_index_from_level(self, level) -> int:
        """
        Return level index from given level name or index
        """
        try:
            return self.names.index(level)
        except ValueError:
            if not is_integer(level):
                raise KeyError(f"Level {level} not found")
            norm = level + self.nlevels if level < 0 else level
            if not 0 <= norm < self.nlevels:
                # matches pandas MultiIndex._get_level_number
                raise IndexError(
                    f"Too many levels: Index has only {self.nlevels} "
                    f"levels, {level} is not a valid level number"
                ) from None
            return norm

    @_performance_tracking
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        if tolerance is not None:
            raise NotImplementedError(
                "Parameter tolerance is not supported yet."
            )
        if method == "nearest":
            raise NotImplementedError(
                f"{method=} is not supported yet for MultiIndex."
            )
        if method in {"ffill", "bfill", "pad", "backfill"} and not (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        ):
            raise ValueError(
                "index must be monotonic increasing or decreasing"
            )

        result = as_column(
            -1,
            length=len(target),
            dtype=SIZE_TYPE_DTYPE,
        )
        if not len(self):
            return self._return_get_indexer_result(result.values)
        try:
            target = cudf.MultiIndex.from_tuples(target)
        except TypeError as e:
            if isinstance(e, MixedTypeError):
                raise e
            return self._return_get_indexer_result(result.values)

        join_keys = [
            _match_join_keys(lcol, rcol, "inner")
            for lcol, rcol in zip(target._columns, self._columns, strict=True)
        ]
        join_keys = list(map(list, zip(*join_keys, strict=True)))
        # Flatten for access_columns
        flattened = [col for cols in join_keys for col in cols]
        with access_columns(*flattened, mode="read", scope="internal"):
            plc_tables = []
            for cols in join_keys:
                plc_tables.append(plc.Table([col.plc_column for col in cols]))
            left_plc, right_plc = plc.join.inner_join(
                plc_tables[0],
                plc_tables[1],
                plc.types.NullEquality.EQUAL,
            )
            scatter_map = ColumnBase.create(
                left_plc, dtype=dtype_from_pylibcudf_column(left_plc)
            )
            indices = ColumnBase.create(
                right_plc, dtype=dtype_from_pylibcudf_column(right_plc)
            )
        result_series = cudf.Series._from_column(
            result._scatter_by_column(scatter_map, indices)
        )

        if method in {"ffill", "bfill", "pad", "backfill"}:
            result_series = _get_indexer_basic(
                index=self,
                positions=result_series,
                method=method,
                target_col=target.to_frame(index=False)[
                    list(range(0, self.nlevels))
                ],
                tolerance=tolerance,
            )
        elif method is not None:
            raise ValueError(
                f"{method=} is unsupported, only supported values are: "
                "{['ffill'/'pad', 'bfill'/'backfill', None]}"
            )

        return self._return_get_indexer_result(result_series.to_cupy())

    def __contains__(self, key) -> bool:
        # Mirror pandas' ``MultiIndex.__contains__``: delegate to ``get_loc``,
        # which already implements correct partial- and full-tuple membership.
        # The inherited ``Index.__contains__`` only checks the first level's
        # column (treating the whole tuple as a scalar), which is incorrect.
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError, ValueError):
            return False

    @_performance_tracking
    def get_loc(self, key):
        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )
        is_unique = self.is_unique
        key = (key,) if not isinstance(key, tuple) else key

        # Handle partial key search. If length of `key` is less than `nlevels`,
        # Only search levels up to `len(key)` level.
        partial_index = self.__class__._from_data(
            data=self._data.select_by_index(slice(len(key)))
        )
        (
            lower_bound,
            upper_bound,
            sort_inds,
        ) = _lexsorted_equal_range(
            partial_index,
            [as_column(k, length=1) for k in key],
            is_sorted,
        )

        if lower_bound == upper_bound:
            raise KeyError(key)

        if is_unique and lower_bound + 1 == upper_bound:
            # Indices are unique (Pandas constraint), search result is unique,
            # return int.
            return (
                lower_bound
                if is_sorted
                else sort_inds.element_indexing(lower_bound)
            )

        if is_sorted:
            # In monotonic index, lex search result is continuous. A slice for
            # the range is returned.
            return slice(lower_bound, upper_bound)

        true_inds = sort_inds.slice(lower_bound, upper_bound).values
        true_inds = _maybe_indices_to_slice(true_inds)
        if isinstance(true_inds, slice):
            return true_inds

        # Not sorted and not unique. Return a boolean mask
        mask = cp.full(len(self), False)
        mask[true_inds] = True
        return mask

    def _get_reconciled_name_object(self, other) -> Self:
        """
        If the result of a set operation will be self,
        return self, unless the names change, in which
        case make a shallow copy of self.
        """
        names = self._maybe_match_names(other)
        if self.names != names:
            return self.rename(names)
        return self

    def _maybe_match_names(self, other) -> list[Hashable]:
        """
        Try to find common names to attach to the result of an operation
        between a and b. Return a consensus list of names if they match
        at least partly or list of None if they have completely
        different names.
        """
        if self.nlevels != other.nlevels:
            return [None] * self.nlevels
        return [
            self_name if _is_same_name(self_name, other_name) else None
            for self_name, other_name in zip(
                self.names, other.names, strict=True
            )
        ]

    @_performance_tracking
    def union(self, other, sort=None) -> Self:
        if not isinstance(other, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            try:
                other = MultiIndex.from_tuples(other, names=self.names)
            except (ValueError, TypeError) as err:
                # ValueError raised by tuples_to_object_array if we
                #  have non-object dtype
                raise TypeError(msg) from err

        if sort not in {None, False}:
            raise ValueError(
                f"The 'sort' keyword only takes the values of "
                f"None or False; {sort} was passed."
            )

        if not len(other) or self.equals(other):
            return self._get_reconciled_name_object(other)
        elif not len(self):
            return other._get_reconciled_name_object(self)

        return self._union(other, sort=sort)

    @_performance_tracking
    def _union(self, other, sort=None) -> Self:
        # TODO: When to_frame is refactored to return a
        # deep copy in future, we should push most of the common
        # logic between MultiIndex._union & Index._union into
        # Index._union.
        other_df = other.copy(deep=True).to_frame(index=False)
        self_df = self.copy(deep=True).to_frame(index=False)
        col_names = list(range(0, self.nlevels))
        self_df.columns = col_names
        other_df.columns = col_names
        self_df["order"] = self_df.index
        other_df["order"] = other_df.index

        result_df = self_df.merge(other_df, on=col_names, how="outer")
        result_df = result_df.sort_values(
            by=result_df._data.to_pandas_index[self.nlevels :],
            ignore_index=True,
        )

        midx = type(self)._from_data(result_df.iloc[:, : self.nlevels]._data)
        midx.names = self.names if self.names == other.names else None
        if sort in {None, True} and len(other):
            return midx.sort_values()
        return midx

    @_performance_tracking
    def _intersection(self, other, sort=None) -> Self:
        if self.names != other.names:
            deep = True
            col_names = list(range(0, self.nlevels))
            res_name = (None,) * self.nlevels
        else:
            deep = False
            col_names = None
            res_name = self.names

        other_df = other.copy(deep=deep).to_frame(index=False)
        self_df = self.copy(deep=deep).to_frame(index=False)
        if col_names is not None:
            other_df.columns = col_names
            self_df.columns = col_names

        result_df = cudf.merge(self_df, other_df, how="inner")
        midx = type(self)._from_data(result_df._data)
        midx.names = res_name
        if sort in {None, True} and len(other):
            return midx.sort_values()
        return midx

    @_performance_tracking
    def _from_columns_like_self(
        self,
        columns: list[ColumnBase],
        column_names: Iterable[str] | None = None,
    ):
        result = super()._from_columns_like_self(columns, column_names)
        result._names = self._names
        return result

    @_performance_tracking
    def _split_columns_by_levels(
        self, levels: tuple, *, in_levels: bool
    ) -> Generator[tuple[Any, ColumnBase], None, None]:
        # This function assumes that for levels with duplicate names, they are
        # specified by indices, not name by ``levels``. E.g. [None, None] can
        # only be specified by 0, 1, not "None".
        level_names = list(self.names)
        level_indices = {
            lv if isinstance(lv, int) else level_names.index(lv)
            for lv in levels
        }
        for i, (name, col) in enumerate(
            zip(self.names, self._columns, strict=True)
        ):
            if in_levels and i in level_indices:
                name = f"level_{i}" if name is None else name
                yield name, col
            elif not in_levels and i not in level_indices:
                yield name, col

    @_performance_tracking
    def _new_index_for_reset_index(
        self, levels: tuple | None, name
    ) -> None | Index:
        """Return the new index after .reset_index"""
        if levels is None:
            return None

        index_columns, index_names = [], []
        for name, col in self._split_columns_by_levels(
            levels, in_levels=False
        ):
            index_columns.append(col)
            index_names.append(name)

        if not index_columns:
            # None is caught later to return RangeIndex
            return None

        index = _index_from_data(
            dict(enumerate(index_columns)),
            name=name,
        )
        if isinstance(index, type(self)):
            index.names = index_names
        else:
            index.name = index_names[0]
        return index

    def _columns_for_reset_index(
        self, levels: tuple | None
    ) -> Generator[tuple[Any, ColumnBase], None, None]:
        """Return the columns and column names for .reset_index"""
        if levels is None:
            for i, (col, name) in enumerate(
                zip(self._columns, self.names, strict=True)
            ):
                yield f"level_{i}" if name is None else name, col
        else:
            yield from self._split_columns_by_levels(levels, in_levels=True)

    def repeat(self, repeats, axis=None) -> Self:
        return self._from_data(
            self._data._from_columns_like_self(
                self._repeat(self._columns, repeats, axis)
            )
        )

    @_performance_tracking
    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        return Frame.__dask_tokenize__(self)
