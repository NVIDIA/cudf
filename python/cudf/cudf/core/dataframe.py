# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import inspect
import itertools
import json
import numbers
import os
import re
import sys
import textwrap
import warnings
from collections import defaultdict
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, Self, assert_never

import cupy
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from nvtx import annotate
from pandas.io.formats import console
from pandas.io.formats.printing import pprint_thing

import pylibcudf as plc

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import (
    _is_categorical_dtype,
    _is_scalar_or_zero_d_array,
    is_decimal32_dtype,
    is_decimal64_dtype,
    is_decimal128_dtype,
    is_dict_like,
    is_dtype_equal,
    is_integer,
    is_list_like,
    is_scalar,
)
from cudf.core import indexing_utils, reshape
from cudf.core.algorithms import factorize
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    access_columns,
    as_column,
    column_empty,
    concat_columns,
)
from cudf.core.column.column import _normalize_types_column
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.copy_types import BooleanMask
from cudf.core.dtype.validators import (
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
    recursively_update_struct_names,
)
from cudf.core.groupby.groupby import DataFrameGroupBy, groupby_doc_template
from cudf.core.index import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    IntervalIndex,
    RangeIndex,
    TimedeltaIndex,
    _index_from_data,
    ensure_index,
)
from cudf.core.indexed_frame import (
    IndexedFrame,
    _check_duplicate_level_names,
    _FrameIndexer,
    _indices_from_labels,
    doc_reset_index_template,
)
from cudf.core.join import Merge, MergeSemi
from cudf.core.missing import NA
from cudf.core.mixins import GetAttrGetItemMixin
from cudf.core.multiindex import MultiIndex
from cudf.core.resample import DataFrameResampler
from cudf.core.series import Series
from cudf.core.udf.row_function import DataFrameApplyKernel
from cudf.errors import MixedTypeError
from cudf.options import get_option
from cudf.utils import docutils, ioutils, queryutils
from cudf.utils.dtypes import (
    DEFAULT_STRING_DTYPE,
    SIZE_TYPE_DTYPE,
    can_convert_to_column,
    dtype_from_pylibcudf_column,
    find_common_type,
    get_dtype_of_same_kind,
    is_column_like,
    is_mixed_with_object_dtype,
    is_pandas_nullable_extension_dtype,
    min_unsigned_type,
)
from cudf.utils.ioutils import (
    _update_pandas_metadata_types_inplace,
    buffer_write_lines,
)
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.utils import (
    _EQUALITY_OPS,
    _external_only_api,
    is_na_like,
)

if TYPE_CHECKING:
    from types import NotImplementedType

    from cudf._typing import (
        Axis,
        ColumnLike,
        Dtype,
        NoDefault,
        ScalarLike,
    )

_cupy_nan_methods_map = {
    "min": "nanmin",
    "max": "nanmax",
    "sum": "nansum",
    "prod": "nanprod",
    "product": "nanprod",
    "mean": "nanmean",
    "std": "nanstd",
    "var": "nanvar",
    "median": "nanmedian",
}


def _shape_mismatch_error(x, y):
    raise ValueError(
        f"shape mismatch: value array of shape {x} "
        f"could not be broadcast to indexing result of "
        f"shape {y}"
    )


class _RowLabelKeyError(KeyError):
    """Marks a ``KeyError`` raised while looking up a *row* label.

    ``_DataFrameLocIndexer.__getitem__`` retries a failed lookup as
    ``(arg, slice(None))`` to support pandas-like tuple indexing. That retry
    must only fire when *column* gathering failed, not when the row label
    itself is genuinely missing (otherwise a bare scalar row miss is silently
    reinterpreted positionally instead of raising ``KeyError``).
    """


class _DataFrameIndexer(_FrameIndexer):
    def __setitem__(self, key, value):
        indexing_utils.check_dict_or_set_indexers(key)
        if not isinstance(key, tuple):
            key = (key, slice(None))
        return self._setitem_tuple_arg(key, value)


class _DataFrameLocIndexer(_DataFrameIndexer):
    """
    For selection by label.
    """

    @_performance_tracking
    def __getitem__(self, arg):
        indexing_utils.check_dict_or_set_indexers(arg)
        if isinstance(self._frame.index, MultiIndex):
            # This try/except block allows the use of pandas-like
            # tuple arguments to index into MultiIndex dataframes.
            try:
                return self._getitem_tuple_arg(arg)
            except _RowLabelKeyError as e:
                # A genuine row-label miss must raise, not retry positionally.
                raise KeyError(*e.args) from None
            except (TypeError, KeyError, IndexError, ValueError):
                # Retry treating ``arg`` as a row-only key that selects across
                # the MultiIndex levels (e.g. ``df.loc[("a", "b")]`` matching
                # the first two index levels), with all columns selected via
                # ``slice(None)``. ``per_level=True`` applies the per-level
                # lookup and drops the scalar-selected index levels from the
                # result.
                return self._getitem_tuple_arg(
                    (arg, slice(None)), per_level=True
                )
        else:
            (
                row_key,
                (
                    col_is_scalar,
                    ca,
                ),
            ) = indexing_utils.destructure_dataframe_loc_indexer(
                arg, self._frame
            )
            row_spec = indexing_utils.parse_row_loc_indexer(
                row_key, self._frame.index
            )
            return self._frame._getitem_preprocessed(
                row_spec, col_is_scalar, ca
            )

    @_performance_tracking
    def _getitem_tuple_arg(self, arg, per_level=False):
        # Step 1: Gather columns
        if isinstance(arg, tuple):
            columns_df = self._frame._get_columns_by_label(arg[1])
            columns_df.index = self._frame.index
        else:
            columns_df = self._frame

        # Step 2: Gather rows
        if isinstance(columns_df.index, MultiIndex):
            if isinstance(arg, (MultiIndex, pd.MultiIndex)):
                if isinstance(arg, pd.MultiIndex):
                    arg = MultiIndex(
                        levels=arg.levels,
                        codes=arg.codes,
                        names=arg.names,
                    )

                indices = _indices_from_labels(columns_df, arg)
                return columns_df.take(indices)

            else:
                if isinstance(arg, tuple):
                    row_arg = arg[0]
                elif is_scalar(arg):
                    row_arg = (arg,)
                else:
                    row_arg = arg
                nlevels = columns_df.index.nlevels
                if (
                    is_list_like(row_arg)
                    and not isinstance(row_arg, tuple)
                    and len(row_arg) > 0
                    and all(
                        isinstance(label, tuple) and len(label) == nlevels
                        for label in row_arg
                    )
                ):
                    # A list of full-length tuples selects those exact
                    # MultiIndex rows (pandas semantics), instead of being
                    # mis-parsed as a per-level indexer.
                    key_mi = MultiIndex.from_tuples(
                        list(row_arg), names=columns_df.index.names
                    )
                    result = columns_df.take(
                        _indices_from_labels(columns_df, key_mi)
                    )
                    if (
                        isinstance(arg, tuple)
                        and len(arg) > 1
                        and is_scalar(arg[1])
                    ):
                        # A scalar column key yields a single column, so
                        # downcast the result to a Series.
                        return result[arg[1]]
                    return result
                try:
                    result = columns_df.index._get_row_major(
                        columns_df, row_arg, per_level=per_level
                    )
                except KeyError as e:
                    # Tag a row-label miss so __getitem__ raises instead of
                    # retrying it as a positional/column lookup.
                    raise _RowLabelKeyError(*e.args) from None
                # A column key that selects a single complete column (a scalar
                # label for single-level columns, or a full-length tuple for
                # MultiIndex columns) yields a Series, collapsing to a scalar
                # only when the row key is itself a full scalar label.
                col_key = (
                    arg[1]
                    if (isinstance(arg, tuple) and len(arg) > 1)
                    else None
                )
                if col_key is not None:
                    col_nlevels = self._frame._data.nlevels
                    col_is_single_label = (
                        is_scalar(col_key)
                        if col_nlevels == 1
                        # A full-length all-scalar tuple is one complete
                        # column label; a tuple containing a slice/list is a
                        # column *slicer* selecting multiple columns.
                        else (
                            isinstance(col_key, tuple)
                            and len(col_key) == col_nlevels
                            and all(is_scalar(x) for x in col_key)
                        )
                    )
                else:
                    col_is_single_label = False
                if col_is_single_label:
                    row_is_full_label = (
                        columns_df.index.nlevels == 1 and is_scalar(row_arg)
                    ) or (
                        isinstance(row_arg, tuple)
                        and len(row_arg) == columns_df.index.nlevels
                        and all(is_scalar(x) for x in row_arg)
                    )
                    if isinstance(result, cudf.DataFrame):
                        result = result[result._column_names[0]]
                    if len(result) == 1 and row_is_full_label:
                        return result._column.element_indexing(0)
                    return result
                return result
        else:
            raise RuntimeError(
                "Should have been handled by now. Please raise Github issue "
                "at https://github.com/rapidsai/cudf/issues"
            )

    @_performance_tracking
    def _setitem_tuple_arg(self, key, value):
        if (
            isinstance(self._frame.index, MultiIndex)
            or self._frame._data.multiindex
        ):
            raise NotImplementedError(
                "Setting values using df.loc[] not supported on "
                "DataFrames with a MultiIndex"
            )

        try:
            columns_df = self._frame._get_columns_by_label(key[1])
        except KeyError:
            if not self._frame.empty and isinstance(key[0], slice):
                indexer = indexing_utils.find_label_range_or_mask(
                    key[0], self._frame.index
                )
                index = self._frame.index
                if isinstance(indexer, indexing_utils.EmptyIndexer):
                    idx = index[0:0:1]
                elif isinstance(indexer, indexing_utils.SliceIndexer):
                    idx = index[indexer.key]
                else:
                    idx = index[indexer.key.column]
            elif self._frame.empty and isinstance(key[0], slice):
                idx = None
            else:
                if is_scalar(key[0]):
                    arr = [key[0]]
                else:
                    arr = key[0]
                idx = Index(arr)
            if is_scalar(value):
                length = len(idx) if idx is not None else 1
                value = as_column(value, length=length)

            if isinstance(value, ColumnBase):
                new_ser = Series._from_column(value, index=idx)
            else:
                new_ser = Series(value, index=idx)
            if len(self._frame) != 0:
                new_ser = new_ser._align_to_index(
                    self._frame.index, how="right"
                )

            if len(self._frame) == 0:
                self._frame.index = (
                    idx if idx is not None else cudf.RangeIndex(len(new_ser))
                )
            self._frame._data.insert(key[1], new_ser._column)
        else:
            if is_scalar(value):
                try:
                    if columns_df._num_columns:
                        self._frame[
                            columns_df._column_names[0]
                        ].loc._loc_to_iloc(key[0])
                    for col in columns_df._column_names:
                        self._frame[col].loc[key[0]] = value
                except KeyError:
                    if not is_scalar(key[0]):
                        raise
                    # TODO: There is a potential bug here if the inplace modifications
                    # done above fail half-way we are left with a partially modified
                    # frame. Need to handle this case better.
                    self.append_new_row(key, value, columns_df=columns_df)

            elif isinstance(value, cudf.DataFrame):
                if value.shape != self._frame.loc[key[0]].shape:
                    _shape_mismatch_error(
                        value.shape,
                        self._frame.loc[key[0]].shape,
                    )
                value_column_names = set(value._column_names)
                row_key = key[0]
                if (
                    isinstance(row_key, (Series, pd.Series))
                    and row_key.dtype.kind == "b"
                ):
                    # Align a boolean Series row mask to the frame's index by
                    # label so the scatter targets the correct rows even when
                    # the mask is reordered relative to the frame; NA mask
                    # values are treated as not-selected.
                    row_key = Series(row_key)
                    if not self._frame.index.equals(row_key.index):
                        row_key = row_key.reindex(self._frame.index)
                    row_key = row_key.fillna(False).astype("bool")
                row_key_col = (
                    as_column(row_key)
                    if (is_list_like(row_key) or is_column_like(row_key))
                    else None
                )
                if row_key_col is not None and row_key_col.dtype.kind == "b":
                    # A boolean row mask must be used directly as the scatter
                    # target; routing it through _indices_from_labels would
                    # (mis)interpret True/False as integer index labels.
                    # Align the RHS to the selected rows by label before
                    # scattering, so a reordered RHS index assigns the right
                    # value to each masked row.
                    if row_key_col.has_nulls():
                        row_key_col = row_key_col.fillna(False)
                    target_index = self._frame.loc[row_key].index
                    if not value.index.equals(target_index):
                        value = value.reindex(target_index)
                    scatter_map = row_key_col
                else:
                    scatter_map = _indices_from_labels(self._frame, row_key)
                for col in columns_df._column_names:
                    columns_df[col][scatter_map] = (
                        value._data[col] if col in value_column_names else NA
                    )

            else:
                if not is_column_like(value):
                    value = cupy.asarray(value)
                if getattr(value, "ndim", 1) == 2:
                    # If the inner dimension is 1, it's broadcastable to
                    # all columns of the dataframe.
                    indexed_shape = columns_df.loc[key[0]].shape
                    if value.shape[1] == 1:
                        if value.shape[0] != indexed_shape[0]:
                            _shape_mismatch_error(value.shape, indexed_shape)
                        for i, col in enumerate(columns_df._column_names):
                            self._frame[col].loc[key[0]] = value[:, 0]
                    else:
                        if value.shape != indexed_shape:
                            _shape_mismatch_error(value.shape, indexed_shape)
                        for i, col in enumerate(columns_df._column_names):
                            self._frame[col].loc[key[0]] = value[:, i]
                else:
                    # handle cases where value is 1d object:
                    # If the key on column axis is a scalar, we indexed
                    # a single column; The 1d value should assign along
                    # the columns.
                    if is_scalar(key[1]):
                        for col in columns_df._column_names:
                            self._frame[col].loc[key[0]] = value
                    # Otherwise, there are two situations. The key on row axis
                    # can be a scalar or 1d. In either of the situation, the
                    # ith element in value corresponds to the ith row in
                    # the indexed object.
                    # If the key is 1d, a broadcast will happen.
                    else:
                        # A scalar row key referencing an existing label lets
                        # a Series RHS be aligned to the column labels below;
                        # a brand-new label cannot (multi-column row append
                        # from a Series is unsupported).
                        align_series_to_columns = (
                            isinstance(value, (Series, pd.Series))
                            and is_scalar(key[0])
                            and key[0] in self._frame.index
                        )
                        for i, col in enumerate(columns_df._column_names):
                            if align_series_to_columns:
                                # A single existing target row: align the
                                # Series RHS to the column labels (pandas
                                # aligns by label; labels missing from the
                                # Series become NA).
                                rhs = (
                                    value.loc[col]
                                    if col in value.index
                                    else NA
                                )
                                self._frame[col].loc[key[0]] = rhs
                            elif isinstance(value, (Series, pd.Series)):
                                # A Series RHS over multiple target rows is
                                # row-indexed: assign the whole Series so .loc
                                # aligns it by label to key[0], rather than
                                # picking value[i] (which would treat it as a
                                # list of per-column values). A new scalar
                                # label also lands here so the unsupported
                                # append fails loudly instead of silently.
                                self._frame[col].loc[key[0]] = value
                            else:
                                self._frame[col].loc[key[0]] = value[i]


class _DataFrameAtIndexer(_DataFrameLocIndexer):
    @_performance_tracking
    def __getitem__(self, key):
        indexing_utils.validate_scalar_key(
            key, "Invalid call for scalar access (getting)!"
        )
        return super().__getitem__(key)

    @_performance_tracking
    def __setitem__(self, key, value):
        indexing_utils.validate_scalar_key(
            key, "Invalid call for scalar access (getting)!"
        )
        return super().__setitem__(key, value)


class _DataFrameIlocIndexer(_DataFrameIndexer):
    """
    For selection by index.
    """

    def __getitem__(self, arg):
        indexing_utils.check_dict_or_set_indexers(arg)
        (
            row_key,
            (
                col_is_scalar,
                ca,
            ),
        ) = indexing_utils.destructure_dataframe_iloc_indexer(arg, self._frame)
        row_spec = indexing_utils.parse_row_iloc_indexer(
            row_key, len(self._frame)
        )
        return self._frame._getitem_preprocessed(row_spec, col_is_scalar, ca)

    @_performance_tracking
    def _setitem_tuple_arg(self, key, value):
        columns_df = self._frame._from_data(
            self._frame._data.select_by_index(key[1]), self._frame.index
        )

        if is_scalar(value):
            for col in columns_df._column_names:
                self._frame[col].iloc[key[0]] = value

        elif isinstance(value, cudf.DataFrame):
            if value.shape != self._frame.iloc[key[0]].shape:
                _shape_mismatch_error(
                    value.shape,
                    self._frame.loc[key[0]].shape,
                )
            value_column_names = set(value._column_names)
            for col in columns_df._column_names:
                columns_df[col][key[0]] = (
                    value._data[col] if col in value_column_names else NA
                )

        else:
            # TODO: consolidate code path with identical counterpart
            # in `_DataFrameLocIndexer._setitem_tuple_arg`
            if not is_column_like(value):
                value = cupy.asarray(value)
            if getattr(value, "ndim", 1) == 2:
                indexed_shape = columns_df.iloc[key[0]].shape
                if value.shape[1] == 1:
                    if value.shape[0] != indexed_shape[0]:
                        _shape_mismatch_error(value.shape, indexed_shape)
                    for i, col in enumerate(columns_df._column_names):
                        self._frame[col].iloc[key[0]] = value[:, 0]
                else:
                    if value.shape != indexed_shape:
                        _shape_mismatch_error(value.shape, indexed_shape)
                    for i, col in enumerate(columns_df._column_names):
                        self._frame._data[col][key[0]] = value[:, i]
            else:
                if is_scalar(key[1]):
                    for col in columns_df._column_names:
                        self._frame[col].iloc[key[0]] = value
                else:
                    for i, col in enumerate(columns_df._column_names):
                        self._frame[col].iloc[key[0]] = value[i]


class _DataFrameiAtIndexer(_DataFrameIlocIndexer):
    @_performance_tracking
    def __getitem__(self, key):
        indexing_utils.validate_scalar_key(
            key, "iAt based indexing can only have integer indexers"
        )
        return super().__getitem__(key)

    @_performance_tracking
    def __setitem__(self, key, value):
        indexing_utils.validate_scalar_key(
            key, "iAt based indexing can only have integer indexers"
        )
        return super().__setitem__(key, value)


def _pd_index_level_dtypes(idx) -> tuple | None:
    """Per-level dtypes of a pandas MultiIndex, else ``None``.

    Used to preserve the dtype of each level when the column axis is an empty
    MultiIndex, whose levels would otherwise reconstruct as object (their
    dtype cannot be inferred from zero entries).
    """
    if isinstance(idx, pd.MultiIndex):
        # use the levels, not get_level_values: materializing a level that
        # has missing entries (-1 codes) upcasts e.g. int64 to float64
        return tuple(level.dtype for level in idx.levels)
    if isinstance(idx, cudf.MultiIndex):
        # the per-row columns share their dtype with the levels
        return tuple(dtype for _, dtype in idx._dtypes)
    return None


@_performance_tracking
def _listlike_to_column_accessor(
    data: Sequence,
    columns: None | pd.Index,
    index: None | Index,
    nan_as_null: bool,
) -> tuple[dict[Any, ColumnBase], Index, pd.Index]:
    """
    Convert a list-like to a dict for ColumnAccessor for DataFrame.__init__

    Returns
    -------
    tuple[dict[Any, ColumnBase], Index, pd.Index]
        - Mapping of column label: Column
        - Resulting index (Index) from the data
        - Resulting columns (pd.Index - store as host data) from the data
    """
    if len(data) == 0:
        if index is None:
            index = cudf.RangeIndex(0)
        if columns is not None:
            # An empty (zero-row) column with no data has no inferable type;
            # default it to ``object`` to match pandas (cudf otherwise uses
            # the default string dtype). Scoped to zero-row columns so the
            # padded case keeps the string dtype and does not surface the
            # inherent None-vs-NaN difference.
            empty_col_dtype = (
                np.dtype("object") if len(index) == 0 else DEFAULT_STRING_DTYPE
            )
            col_data = {
                col_label: column_empty(len(index), dtype=empty_col_dtype)
                for col_label in columns
            }
        else:
            col_data = {}
            columns = pd.RangeIndex(0)
        return (col_data, index, columns)
    # We assume that all elements in data are the same type as the first element
    first_element = data[0]
    if is_scalar(first_element):
        if columns is not None:
            if len(columns) != 1:
                raise ValueError("Passed column must be of length 1")
        else:
            columns = pd.RangeIndex(1)
        if index is not None:
            if len(index) != len(data):
                raise ValueError(
                    "Passed index must be the same length as data."
                )
        else:
            index = cudf.RangeIndex(len(data))

        return (
            {columns[0]: as_column(data, nan_as_null=nan_as_null)},
            index,
            columns,
        )
    elif isinstance(first_element, Series):
        data_length = len(data)
        if index is None:
            index = _index_from_listlike_of_series(data)
        else:
            index_length = len(index)
            if data_length != index_length:
                # If the passed `index` length doesn't match
                # length of Series objects in `data`, we must
                # check if `data` can be duplicated/expanded
                # to match the length of index. For that we
                # check if the length of index is a factor
                # of length of data.
                #
                # 1. If yes, we extend data
                # until length of data is equal to length of index.
                # 2. If no, we throw an error stating the
                # shape of resulting `data` and `index`

                # Simple example
                # >>> import pandas as pd
                # >>> s = pd.Series([1, 2, 3])
                # >>> pd.DataFrame([s], index=['a', 'b'])
                #    0  1  2
                # a  1  2  3
                # b  1  2  3
                # >>> pd.DataFrame([s], index=['a', 'b', 'c'])
                #    0  1  2
                # a  1  2  3
                # b  1  2  3
                # c  1  2  3
                if index_length % data_length == 0:
                    data = list(
                        itertools.chain.from_iterable(
                            itertools.repeat(data, index_length // data_length)
                        )
                    )
                    data_length = len(data)
                else:
                    raise ValueError(
                        f"Length of values ({data_length}) does "
                        f"not match length of index ({index_length})"
                    )
        if data_length > 1:
            common_dtype = find_common_type([ser.dtype for ser in data])
            data = [ser.astype(common_dtype) for ser in data]
        if all(len(first_element) == len(ser) for ser in data):
            if data_length == 1:
                temp_index = first_element.index
            else:
                temp_index = Index._concat(
                    [ser.index for ser in data]
                ).drop_duplicates()

            temp_data: dict[Hashable, ColumnBase] = {}
            for i, ser in enumerate(data):
                if not ser.index.is_unique:
                    raise ValueError(
                        "Reindexing only valid with uniquely valued Index "
                        "objects"
                    )
                elif not ser.index.equals(temp_index):
                    ser = ser.reindex(temp_index)
                temp_data[i] = ser._column

            combined = DataFrame._from_data(
                ColumnAccessor(
                    temp_data,
                    verify=False,
                    rangeindex=True,
                ),
                index=temp_index,
            )
        else:
            combined = cudf.concat(data, axis=1)
        # Aligning rows above can upcast integer columns that gained nulls
        # (e.g. int -> float64), leaving columns with differing dtypes;
        # transpose requires a single dtype, so promote to a common one
        # first (matching pandas, which upcasts the whole frame).
        if combined._num_columns > 1:
            common_dtype = find_common_type(
                [dtype for _, dtype in combined._dtypes]
            )
            combined = combined.astype(common_dtype)
        transpose = combined.T

        if columns is None:
            columns = pd.RangeIndex(transpose._num_columns)
            col_data = transpose._data
        else:
            col_data = {}
            for col_label in columns:
                try:
                    col_data[col_label] = transpose._data[col_label]
                except KeyError:
                    col_data[col_label] = column_empty(
                        len(index), dtype=np.dtype(np.float64)
                    )
        return (col_data, index, columns)
    elif isinstance(first_element, dict):
        from_pandas = DataFrame(
            pd.DataFrame(data),
            index=index,
            columns=columns,
            nan_as_null=nan_as_null,
        )
        return (
            from_pandas._data,
            from_pandas.index,
            from_pandas._data.to_pandas_index,
        )
    elif not can_convert_to_column(first_element):
        raise TypeError(f"Cannot convert {type(first_element)} to a column")
    else:
        if index is None:
            index = cudf.RangeIndex(len(data))
        data = list(itertools.zip_longest(*data))
        if columns is None:
            if isinstance(first_element, tuple) and hasattr(
                first_element, "_fields"
            ):
                # pandas behavior is to use the fields from the first
                # namedtuple as the column names
                columns = pd.Index(first_element._fields)
            else:
                columns = pd.RangeIndex(len(data))
        col_data = {
            col_label: as_column(col_values, nan_as_null=nan_as_null)
            for col_label, col_values in zip(columns, data, strict=True)
        }
        return (
            col_data,
            index,
            columns,
        )


@_performance_tracking
def _array_to_column_accessor(
    data: np.ndarray | cupy.ndarray,
    columns: None | pd.Index,
    nan_as_null: bool,
) -> ColumnAccessor:
    """Convert a 1D or 2D numpy or cupy array to a ColumnAccessor for DataFrame.__init__"""
    if data.ndim not in {1, 2}:
        raise ValueError(
            f"records dimension expected 1 or 2 but found: {data.ndim}"
        )

    if data.ndim == 1:
        data = data.reshape(len(data), 1)
    if columns is not None:
        if len(columns) != data.shape[1]:
            raise ValueError(
                f"columns length expected {data.shape[1]} but "
                f"found {len(columns)}"
            )
        columns_labels = columns
    else:
        columns_labels = pd.RangeIndex(data.shape[1])
    ca = ColumnAccessor(
        {
            column_label: as_column(data[:, i], nan_as_null=nan_as_null)
            for column_label, i in zip(
                columns_labels, range(data.shape[1]), strict=True
            )
        },
        verify=False,
        rangeindex=isinstance(columns_labels, pd.RangeIndex),
        multiindex=isinstance(columns_labels, pd.MultiIndex),
        label_dtype=columns_labels.dtype,
        level_names=tuple(columns_labels.names),
        level_dtypes=_pd_index_level_dtypes(columns_labels),
        pandas_index=columns_labels,
    )
    return ca


@_performance_tracking
def _mapping_to_column_accessor(
    data: Mapping,
    index: None | Index,
    dtype: None | Dtype,
    nan_as_null: bool,
) -> tuple[dict[Any, ColumnBase], Index, pd.Index]:
    """
    Convert a mapping (dict-like) to a dict for ColumnAccessor for DataFrame.__init__

    Returns
    -------
    tuple[dict[Any, ColumnBase], Index, pd.Index]
        - Mapping of column label: Column
        - Resulting index from the data
        - Resulting columns from the data
    """
    if len(data) == 0:
        return (
            {},
            cudf.RangeIndex(0) if index is None else index,
            pd.RangeIndex(0),
        )
    data = dict(data)

    # 1) Align indexes of all data.values() that are Series/dicts
    values_as_series = {
        key: Series(val, nan_as_null=nan_as_null, dtype=dtype)
        for key, val in data.items()
        if isinstance(val, (pd.Series, Series, dict))
    }
    if values_as_series:
        aligned_input_series = cudf.core.series._align_indices(
            list(values_as_series.values())
        )
        data = data.copy()
        for key, aligned_series in zip(
            values_as_series.keys(), aligned_input_series, strict=True
        ):
            if index is not None:
                aligned_series = aligned_series.reindex(index=index)
            data[key] = aligned_series
        index_from_data = aligned_series.index
    else:
        index_from_data = None

    value_lengths = set()
    result_index = None
    if index_from_data is not None:
        value_lengths.add(len(index_from_data))
        result_index = index_from_data
    elif index is not None:
        result_index = index

    # 2) Convert all array-like data.values() to columns
    scalar_keys = []
    tuple_key_count = 0
    tuple_key_lengths = set()
    col_data = {}
    for key, value in data.items():
        if is_scalar(value):
            scalar_keys.append(key)
            col_data[key] = value
        else:
            if isinstance(key, tuple):
                tuple_key_count += 1
                tuple_key_lengths.add(len(key))
            column = as_column(value, nan_as_null=nan_as_null, dtype=dtype)
            if (
                dtype is None
                and len(column) == 0
                and isinstance(value, (list, tuple, Iterator))
            ):
                # pandas' DataFrame constructor defaults untyped empty
                # sequences (list/tuple/iterator) to float64 (numpy's
                # default for np.array([])), unlike Series([]) which
                # defaults to object. An empty range stays int64 like
                # pandas (as_column already handles it via from_range).
                column = column_empty(0, dtype=np.dtype(np.float64))
            value_lengths.add(len(column))
            col_data[key] = column

    if tuple_key_count not in {0, len(data)}:
        raise ValueError("All dict keys must be tuples if a tuple key exists.")

    if len(scalar_keys) != len(data) and len(value_lengths) > 1:
        raise ValueError(
            "Found varying value lengths when all values "
            f"must have the same length: {value_lengths}"
        )
    elif len(scalar_keys) == len(data):
        # All data.values() are scalars
        if index is None:
            raise ValueError(
                "If using all scalar values, you must pass an index"
            )
        scalar_length = len(index)
    else:
        scalar_length = value_lengths.pop()

    # 3) Convert all remaining scalar data.values() to columns
    for key in scalar_keys:
        scalar = col_data[key]
        if scalar is None or scalar is cudf.NA:
            scalar = pa.scalar(None, type=pa.string())
        col_data[key] = as_column(
            scalar, nan_as_null=nan_as_null, length=scalar_length, dtype=dtype
        )

    if tuple_key_count and len(tuple_key_lengths) > 1:
        # All tuple keys must be the same length
        final_length = max(tuple_key_lengths)
        col_data = {
            old_key
            if len(old_key) == final_length
            else old_key + ("",) * (final_length - len(old_key)): column
            for old_key, column in col_data.items()
        }

    if result_index is None:
        result_index = cudf.RangeIndex(scalar_length)

    return col_data, result_index, pd.Index(col_data)


class DataFrame(IndexedFrame, GetAttrGetItemMixin):
    """
    A GPU Dataframe object.

    Parameters
    ----------
    data : array-like, Iterable, dict, or DataFrame.
        Dict can contain Series, arrays, constants, or list-like objects.
    index : Index or array-like
        Index to use for resulting frame. Will default to
        RangeIndex if no indexing information part of input data and
        no index provided.
    columns : Index or array-like
        Column labels to use for resulting frame.
        Will default to RangeIndex (0, 1, 2, …, n) if no column
        labels are provided.
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed.
        If None, infer.
    copy : bool or None, default None
        Copy data from inputs.
        Currently not implemented.
    nan_as_null : bool, Default True
        If ``None``/``True``, converts ``np.nan`` values to
        ``null`` values.
        If ``False``, leaves ``np.nan`` values as is.

    Examples
    --------
    Build dataframe with ``__setitem__``:

    >>> import cudf
    >>> df = cudf.DataFrame()
    >>> df['key'] = [0, 1, 2, 3, 4]
    >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
    >>> df
       key   val
    0    0  10.0
    1    1  11.0
    2    2  12.0
    3    3  13.0
    4    4  14.0

    Build DataFrame via dict of columns:

    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> t0 = datetime.strptime('2018-10-07 12:00:00', '%Y-%m-%d %H:%M:%S')
    >>> n = 5
    >>> df = cudf.DataFrame({
    ...     'id': np.arange(n),
    ...     'datetimes': np.array(
    ...     [(t0+ timedelta(seconds=x)) for x in range(n)])
    ... })
    >>> df
        id            datetimes
    0    0  2018-10-07 12:00:00
    1    1  2018-10-07 12:00:01
    2    2  2018-10-07 12:00:02
    3    3  2018-10-07 12:00:03
    4    4  2018-10-07 12:00:04

    Build DataFrame via list of rows as tuples:

    >>> df = cudf.DataFrame([
    ...     (5, "cats", "jump", np.nan),
    ...     (2, "dogs", "dig", 7.5),
    ...     (3, "cows", "moo", -2.1, "occasionally"),
    ... ])
    >>> df
       0     1     2     3             4
    0  5  cats  jump   NaN           NaN
    1  2  dogs   dig   7.5           NaN
    2  3  cows   moo  -2.1  occasionally

    Convert from a Pandas DataFrame:

    >>> import pandas as pd
    >>> pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
    >>> pdf
       a    b
    0  0  0.1
    1  1  0.2
    2  2  NaN
    3  3  0.3
    >>> df = cudf.from_pandas(pdf)
    >>> df
       a    b
    0  0  0.1
    1  1  0.2
    2  2  NaN
    3  3  0.3
    """

    _PROTECTED_KEYS = frozenset(
        ("_data", "_index", "_ipython_canary_method_should_not_exist_")
    )
    _accessors: set[Any] = set()
    _loc_indexer_type = _DataFrameLocIndexer
    _iloc_indexer_type = _DataFrameIlocIndexer
    _groupby = DataFrameGroupBy
    _resampler = DataFrameResampler

    @_performance_tracking
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=None,
        nan_as_null=no_default,
    ):
        if copy is not None:
            raise NotImplementedError("copy is not currently implemented.")
        if nan_as_null is no_default:
            nan_as_null = not get_option("mode.pandas_compatible")

        if columns is not None:
            if isinstance(columns, (Index, Series, cupy.ndarray)):
                columns = ensure_index(columns).to_pandas()
            elif (
                isinstance(columns, list)
                and len(columns)
                and all(cudf.api.types.is_list_like(col) for col in columns)
            ):
                columns = pd.MultiIndex.from_arrays(columns)
            elif not isinstance(columns, pd.Index):
                columns = pd.Index(columns)

            if columns.nunique(dropna=False) != len(columns):
                raise ValueError("Columns cannot contain duplicate values")

        if index is not None:
            index = ensure_index(index)

        if isinstance(data, Iterator) and not is_scalar(data):
            data = list(data)

        second_index = None
        second_columns = None
        attrs = None
        allows_duplicate_labels = True
        if isinstance(data, (DataFrame, pd.DataFrame)):
            attrs = deepcopy(data.attrs)
            allows_duplicate_labels = data.flags.allows_duplicate_labels
            if isinstance(data, pd.DataFrame):
                cols = {
                    # Pass the Series (not ``col_value.array``) so that
                    # ``as_column`` sees the column's explicit dtype. For an
                    # object-dtype string column with nulls this preserves
                    # ``object`` rather than inferring ``str`` in
                    # pandas-compatible mode, matching ``cudf.Series(col)``.
                    i: as_column(col_value, nan_as_null=nan_as_null)
                    for i, (_, col_value) in enumerate(data.items())
                }
                new_idx = from_pandas(data.index, nan_as_null=nan_as_null)
                df = type(self)._from_data(cols, new_idx)
                # Checks duplicate columns and sets column metadata
                df.columns = data.columns
                data = df
            col_accessor = data._data
            index, second_index = data.index, index
            second_columns = columns
        elif isinstance(data, (Series, pd.Series)):
            if isinstance(data, pd.Series):
                data = Series(data, nan_as_null=nan_as_null)
            index, second_index = data.index, index
            # Series.name is not None and Series.name in columns
            #   -> align
            # Series.name is not None and Series.name not in columns
            #   -> return empty DataFrame
            # Series.name is None and no columns
            #   -> return 1 column DataFrame
            # Series.name is None and columns
            #   -> return 1 column DataFrame if len(columns) in {0, 1}
            if data.name is None:
                if columns is not None:
                    if len(columns) > 1:
                        raise ValueError(
                            "Length of columns must be less than 2 if "
                            f"{type(data).__name__}.name is None."
                        )
                    name = columns[0]
                    col_is_rangeindex = False
                else:
                    name = 0
                    col_is_rangeindex = True
                col_accessor = ColumnAccessor(
                    {name: data._column},
                    verify=False,
                    rangeindex=col_is_rangeindex,
                )
            else:
                if columns is not None and not columns.isin([data.name]).any():
                    index = index[:0]
                    col_accessor = ColumnAccessor(
                        {
                            # Zero-row no-data columns default to ``object``
                            # to match pandas.
                            col: column_empty(0, dtype=np.dtype("object"))
                            for col in columns
                        },
                        verify=False,
                    )
                else:
                    col_accessor = ColumnAccessor(
                        {data.name: data._column}, verify=False
                    )
                    second_columns = columns
        elif data is None or (
            isinstance(data, dict)
            and columns is not None
            and (~columns.isin(data.keys())).all()
        ):
            if index is None:
                index = RangeIndex(0)
            if columns is not None:
                # Zero-row no-data columns default to ``object`` to match
                # pandas; padded columns keep the string dtype.
                empty_col_dtype = (
                    np.dtype("object")
                    if len(index) == 0
                    else DEFAULT_STRING_DTYPE
                )
                col_accessor = ColumnAccessor(
                    {
                        k: column_empty(len(index), dtype=empty_col_dtype)
                        for k in columns
                    },
                    verify=False,
                    level_names=tuple(columns.names),
                    multiindex=isinstance(columns, pd.MultiIndex),
                    rangeindex=isinstance(columns, pd.RangeIndex),
                    label_dtype=columns.dtype,
                    level_dtypes=_pd_index_level_dtypes(columns),
                )
            else:
                col_accessor = ColumnAccessor(
                    {}, verify=False, rangeindex=True
                )
        elif isinstance(data, ColumnAccessor):
            raise TypeError(
                "Use cudf.DataFrame._from_data for constructing a DataFrame from "
                "ColumnAccessor"
            )
        elif isinstance(data, ColumnBase):
            raise TypeError(
                "Use cudf.DataFrame._from_data({col_label: col}) for constructing with "
                "a Column."
            )
        elif hasattr(data, "__cuda_array_interface__"):
            arr_interface = data.__cuda_array_interface__
            # descr is an optional field
            if "descr" in arr_interface:
                if len(arr_interface["descr"]) == 1:
                    col_accessor = _array_to_column_accessor(
                        cupy.asarray(data, order="F"), columns, nan_as_null
                    )
                else:
                    new_df = self.from_records(
                        data, index=index, columns=columns
                    )
                    col_accessor = new_df._data
                    index = new_df.index
            else:
                col_accessor = _array_to_column_accessor(
                    cupy.asarray(data, order="F"), columns, nan_as_null
                )

            if index is None:
                index = cudf.RangeIndex(arr_interface["shape"][0])
        elif hasattr(data, "__array_interface__"):
            arr_interface = data.__array_interface__
            if len(arr_interface["descr"]) == 1:
                col_accessor = _array_to_column_accessor(
                    np.asarray(data, order="F"), columns, nan_as_null
                )
                if index is None:
                    index = cudf.RangeIndex(arr_interface["shape"][0])
            else:
                new_df = self.from_records(data, index=index, columns=columns)
                col_accessor = new_df._data
                index = new_df.index
        elif is_scalar(data):
            if index is None or columns is None:
                raise ValueError(
                    "index= and columns= must both not be None if data is a scalar."
                )
            col_accessor = ColumnAccessor(
                {
                    col_label: as_column(
                        data, nan_as_null=nan_as_null, length=len(index)
                    )
                    for col_label in columns
                },
                verify=False,
                multiindex=isinstance(columns, pd.MultiIndex),
                rangeindex=isinstance(columns, pd.RangeIndex),
                level_names=tuple(columns.names),
                label_dtype=columns.dtype,
                level_dtypes=_pd_index_level_dtypes(columns),
                pandas_index=columns,
            )
        elif isinstance(data, Mapping):
            # Note: We excluded ColumnAccessor already above
            result = _mapping_to_column_accessor(
                data,
                index,
                cudf.dtype(dtype) if dtype is not None else None,
                nan_as_null,
            )
            col_dict = result[0]
            index = result[1]
            columns, second_columns = result[2], columns
            col_accessor = ColumnAccessor(
                col_dict,
                verify=False,
                rangeindex=isinstance(columns, pd.RangeIndex),
                multiindex=isinstance(columns, pd.MultiIndex),
                level_names=tuple(columns.names),
                label_dtype=columns.dtype,
                level_dtypes=_pd_index_level_dtypes(columns),
            )
        elif is_list_like(data):
            col_dict, index, columns = _listlike_to_column_accessor(
                data, columns, index, nan_as_null
            )
            col_accessor = ColumnAccessor(
                col_dict,
                verify=False,
                rangeindex=isinstance(columns, pd.RangeIndex),
                multiindex=isinstance(columns, pd.MultiIndex),
                level_names=tuple(columns.names),
                label_dtype=columns.dtype,
                level_dtypes=_pd_index_level_dtypes(columns),
            )
        else:
            raise TypeError(
                f"data must be list or dict-like, not {type(data).__name__}"
            )

        if second_columns is not None:
            new_data = {
                second_label: col_accessor[second_label]
                if second_label in col_accessor
                else column_empty(
                    col_accessor.nrows, dtype=DEFAULT_STRING_DTYPE
                )
                for second_label in second_columns
            }
            col_accessor = ColumnAccessor(
                new_data,
                verify=False,
                rangeindex=isinstance(second_columns, pd.RangeIndex),
                multiindex=isinstance(second_columns, pd.MultiIndex),
                level_names=tuple(second_columns.names),
                label_dtype=second_columns.dtype,
                level_dtypes=_pd_index_level_dtypes(second_columns),
            )

        super().__init__(
            col_accessor,
            index=index,
            attrs=attrs,
            allows_duplicate_labels=allows_duplicate_labels,
        )
        if second_index is not None:
            reindexed = self.reindex(index=second_index, copy=False)
            self._data = reindexed._data
            self._index = second_index

        if dtype:
            self._data = self.astype(dtype)._data

        final_pd_columns = (
            second_columns if second_columns is not None else columns
        )
        if isinstance(final_pd_columns, pd.Index):
            self._data._prime_to_pandas_index(final_pd_columns)

    @classmethod
    def _from_data(  # type: ignore[override]
        cls,
        data: MutableMapping,
        index: Index | None = None,
        columns: Any = None,
        attrs: dict | None = None,
        allows_duplicate_labels: bool = True,
    ) -> Self:
        out = super()._from_data(
            data=data,
            index=index,
            attrs=attrs,
            allows_duplicate_labels=allows_duplicate_labels,
        )
        if columns is not None:
            out.columns = columns
        return out

    # The `constructor*` properties are used by `dask` (and `dask_cudf`)
    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError(
            "_constructor_expanddim not supported for DataFrames!"
        )

    @property
    @_performance_tracking
    def shape(self) -> tuple[int, int]:
        """Returns a tuple representing the dimensionality of the DataFrame."""
        return self._num_rows, self._num_columns

    @property
    @_external_only_api(
        "Use ._dtypes for an iterator over the column labels and dtypes. "
        "Use pandas.Series(dict(self._dtypes)) if you need a pandas Series "
        "of dtypes."
    )
    def dtypes(self) -> pd.Series:
        """
        Return the dtypes in this object.

        Returns
        -------
        pandas.Series
            The data type of each column.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'float': [1.0],
        ...                    'int': [1],
        ...                    'datetime': [pd.Timestamp('20180310')],
        ...                    'string': ['foo']})
        >>> df
           float  int   datetime string
        0    1.0    1 2018-03-10    foo
        >>> df.dtypes
        float              float64
        int                  int64
        datetime    datetime64[us]
        string                 str
        dtype: object
        """
        result_dict = dict(self._dtypes)
        if cudf.get_option("mode.pandas_compatible"):
            for key, value in result_dict.items():
                if isinstance(
                    value,
                    (
                        ListDtype,
                        StructDtype,
                        Decimal32Dtype,
                        Decimal64Dtype,
                        Decimal128Dtype,
                    ),
                ):
                    raise TypeError(
                        f"Column '{key}' has {type(value).__name__}, which is not supported in pandas."
                    )

        result = pd.Series(
            result_dict, index=self._data.to_pandas_index, dtype="object"
        )
        return result

    @property
    def ndim(self) -> int:
        """Dimension of the data. DataFrame ndim is always 2."""
        return 2

    def __dir__(self):
        # Add the columns of the DataFrame to the dir output.
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(
            c
            for c in self._column_names
            if isinstance(c, str) and c.isidentifier()
        )
        return list(o)

    def __setattr__(self, key, col):
        try:
            # Preexisting attributes may be set. We cannot rely on checking the
            # `_PROTECTED_KEYS` because we must also allow for settable
            # properties, and we must call object.__getattribute__ to bypass
            # the `__getitem__` behavior inherited from `GetAttrGetItemMixin`.
            object.__getattribute__(self, key)
        except AttributeError:
            if key not in self._PROTECTED_KEYS:
                try:
                    # Check key existence.
                    self[key]
                    # If a column already exists, set it.
                    self[key] = col
                    return
                except KeyError:
                    pass

            # Set a new attribute that is not already a column.
            super().__setattr__(key, col)

        except RuntimeError as e:
            # TODO: This allows setting properties that are marked as forbidden
            # for internal usage. It is necessary because the __getattribute__
            # call in the try block will trigger the error. We should see if
            # setting these variables can also always be disabled
            if "External-only API" not in str(e):
                raise
            super().__setattr__(key, col)
        else:
            super().__setattr__(key, col)

    def _getitem_preprocessed(
        self,
        spec: indexing_utils.IndexingSpec,
        col_is_scalar: bool,
        ca: ColumnAccessor,
    ) -> Self | Series:
        """Get a subset of rows and columns given structured data

        Parameters
        ----------
        spec
            Indexing specification for the rows
        col_is_scalar
            Was the indexer of the columns a scalar (return a Series)
        ca
            ColumnAccessor representing the subsetted column data

        Returns
        -------
        Subsetted DataFrame or Series (if a scalar row is requested
        and the concatenation of the column types is possible)

        Notes
        -----
        This function performs no bounds-checking or massaging of the
        inputs.
        """
        if col_is_scalar:
            series = Series._from_data(ca, index=self.index)
            self._propagate_metadata(series)
            return series._getitem_preprocessed(spec)
        if ca.names != self._column_names:
            frame = self._from_data(ca, index=self.index)
            self._propagate_metadata(frame)
        else:
            frame = self
        if isinstance(spec, indexing_utils.MapIndexer):
            return frame._gather(spec.key, keep_index=True)
        elif isinstance(spec, indexing_utils.MaskIndexer):
            return frame._apply_boolean_mask(spec.key, keep_index=True)
        elif isinstance(spec, indexing_utils.SliceIndexer):
            return frame._slice(spec.key)
        elif isinstance(spec, indexing_utils.ScalarIndexer):
            result = frame._gather(spec.key, keep_index=True)
            # Attempt to turn into series.
            try:
                # Behaviour difference from pandas, which will merrily
                # turn any heterogeneous set of columns into a series if
                # you only ask for one row.
                new_name = result.index[0]
                pd_new_index = result.keys()
                if isinstance(pd_new_index, pd.MultiIndex):
                    result_index = MultiIndex(
                        pd_new_index.levels,
                        pd_new_index.codes,
                        names=pd_new_index.names,
                    )
                else:
                    result_index = Index(pd_new_index)
                result = Series._concat(
                    [result[name] for name in frame._column_names],
                    index=False,
                )
                result.index = result_index
                result.name = new_name
                frame._propagate_metadata(result)
                return result
            except TypeError:
                if get_option("mode.pandas_compatible"):
                    raise
                # Couldn't find a common type, just return a 1xN dataframe.
                return result
        elif isinstance(spec, indexing_utils.EmptyIndexer):
            return frame._empty_like(keep_index=True)
        assert_never(spec)

    @_performance_tracking
    def __getitem__(self, arg):
        """
        If *arg* is a ``str`` or ``int`` type, return the column Series.
        If *arg* is a ``slice``, return a new DataFrame with all columns
        sliced to the specified range.
        If *arg* is an ``array`` containing column names, return a new
        DataFrame with the corresponding columns.
        If *arg* is a ``dtype.bool array``, return the rows marked True

        Examples
        --------
        >>> df = cudf.DataFrame({
        ...     'a': list(range(10)),
        ...     'b': list(range(10)),
        ...     'c': list(range(10)),
        ... })

        Get first 4 rows of all columns.

        >>> df[:4]
           a  b  c
        0  0  0  0
        1  1  1  1
        2  2  2  2
        3  3  3  3

        Get last 5 rows of all columns.

        >>> df[-5:]
           a  b  c
        5  5  5  5
        6  6  6  6
        7  7  7  7
        8  8  8  8
        9  9  9  9

        Get columns a and c.

        >>> df[['a', 'c']]
           a  c
        0  0  0
        1  1  1
        2  2  2
        3  3  3
        4  4  4
        5  5  5
        6  6  6
        7  7  7
        8  8  8
        9  9  9

        Return the rows specified in the boolean mask.

        >>> df[[True, False, True, False, True,
        ...     False, True, False, True, False]]
           a  b  c
        0  0  0  0
        2  2  2  2
        4  4  4  4
        6  6  6  6
        8  8  8  8
        """
        if _is_scalar_or_zero_d_array(arg) or isinstance(arg, tuple):
            if isinstance(arg, tuple) and any(
                isinstance(k, slice) for k in arg
            ):
                # A tuple key in __getitem__ is a single (possibly
                # MultiIndex) column label; a slice inside it is invalid
                # (use .loc for slicing), matching pandas' InvalidIndexError.
                raise pd.errors.InvalidIndexError(
                    next(k for k in arg if isinstance(k, slice))
                )
            out = self._get_columns_by_label(arg)
            if is_scalar(arg):
                nlevels = 1
            elif isinstance(arg, tuple):
                nlevels = len(arg)

            if (
                self._data.multiindex is False
                or nlevels == self._data.nlevels
                or (
                    out._data.multiindex is False
                    and self._data.multiindex is True
                    and out._num_columns
                    and all(n == "" for n in out._column_names)
                )
                or (
                    out._data.multiindex is True
                    and self._data.multiindex is True
                    and out._num_columns
                    and all(n == "" for n in out._column_names[0])
                )
            ):
                out = self._constructor_sliced._from_data(out._data)
                out._data.multiindex = False
                out.index = self.index
                out.name = arg
                self._propagate_metadata(out)
            return out

        elif isinstance(arg, slice):
            return self._slice(arg)

        elif can_convert_to_column(arg):
            mask = arg
            if is_list_like(mask):
                dtype = None
                mask = pd.Series(mask, dtype=dtype)
            if getattr(mask.dtype, "kind", None) == "b":
                return self._apply_boolean_mask(BooleanMask(mask, len(self)))
            else:
                return self._get_columns_by_label(mask)
        elif isinstance(arg, DataFrame):
            return self.where(arg)
        else:
            raise TypeError(
                f"__getitem__ on type {type(arg)} is not supported"
            )

    @_performance_tracking
    def __setitem__(self, arg, value):
        """Add/set column by *arg or DataFrame*"""
        if isinstance(arg, DataFrame):
            # not handling set_item where arg = df & value = df
            if isinstance(value, DataFrame):
                raise TypeError(
                    f"__setitem__ with arg = {type(value)} and "
                    f"value = {type(arg)} is not supported"
                )
            else:
                if not self.index.equals(arg.index):
                    # Align a boolean DataFrame mask to this frame by label
                    # before scattering, so a reordered or partial mask
                    # selects the correct rows. Labels missing from the mask
                    # are treated as not-selected.
                    arg = arg.reindex(self.index)
                for col_name in self._data:
                    scatter_map = arg._data[col_name]
                    if scatter_map.has_nulls():
                        scatter_map = scatter_map.fillna(False)
                    if is_scalar(value):
                        self._data[col_name][scatter_map] = value
                    else:
                        self._data[col_name][scatter_map] = as_column(value)[
                            scatter_map
                        ]
        elif is_scalar(arg) or isinstance(arg, tuple):
            if isinstance(value, DataFrame):
                _setitem_with_dataframe(
                    input_df=self,
                    replace_df=value,
                    input_cols=[arg],
                    mask=None,
                )
            else:
                if arg in self._data:
                    if not is_scalar(value) and len(self) == 0:
                        value = as_column(value)
                        length = len(value)
                        new_columns = (
                            value
                            if key == arg
                            else column_empty(
                                row_count=length, dtype=col.dtype
                            )
                            for key, col in self._column_labels_and_values
                        )
                        self._data = self._data._from_columns_like_self(
                            new_columns, verify=False
                        )
                        if isinstance(value, (pd.Series, Series)):
                            self._index = Index(value.index)
                        elif len(value) > 0:
                            self._index = RangeIndex(length)
                        return
                    elif isinstance(value, (pd.Series, Series)):
                        value = Series(value)._align_to_index(
                            self.index,
                            how="right",
                            sort=False,
                            allow_non_unique=True,
                        )
                    if is_scalar(value):
                        self._data[arg] = as_column(value, length=len(self))
                    else:
                        value = as_column(value)
                        self._data[arg] = value
                else:
                    # disc. with pandas here
                    # pandas raises key error here
                    self._insert(
                        loc=self._num_columns,
                        name=arg,
                        value=value,
                        ignore_index=False,
                    )

        elif can_convert_to_column(arg):
            mask = arg
            if is_list_like(mask):
                mask = np.array(mask)

            if getattr(mask.dtype, "kind", None) == "b":
                if isinstance(arg, (cudf.Series, pd.Series)):
                    arg = cudf.Series(arg)
                    if not self.index.equals(arg.index):
                        # Align a boolean Series mask to this frame by
                        # label; unaligned labels are not selected.
                        arg = arg.reindex(self.index)
                    # Normalize NA mask values to False (pandas treats NA in
                    # a boolean mask as not-selected) and collapse a nullable
                    # ``boolean`` dtype to plain ``bool`` for scattering,
                    # regardless of whether reindexing was needed.
                    arg = arg.fillna(False).astype("bool")
                mask = as_column(arg)

                if isinstance(value, DataFrame):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=value,
                        input_cols=None,
                        mask=mask,
                    )
                else:
                    if not is_scalar(value):
                        value = as_column(value)[mask]
                    for col_name in self._data:
                        self._data[col_name][mask] = value
            else:
                if isinstance(value, (cupy.ndarray, np.ndarray)):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=cudf.DataFrame(value),
                        input_cols=arg,
                        mask=None,
                        ignore_index=True,
                    )
                elif isinstance(value, DataFrame):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=value,
                        input_cols=arg,
                        mask=None,
                    )
                else:
                    for col in arg:
                        if is_scalar(value):
                            self._data[col] = as_column(
                                value, length=len(self)
                            )
                        else:
                            self._data[col] = as_column(value)

        else:
            raise TypeError(
                f"__setitem__ on type {type(arg)} is not supported"
            )

    def __delitem__(self, name):
        self._drop_column(name)

    @_performance_tracking
    def memory_usage(self, index: bool = True, deep: bool = False) -> Series:
        """
        Return the memory usage of the DataFrame.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the index.
        deep : bool, default False
            The deep parameter is ignored and is only included for pandas
            compatibility.

        Returns
        -------
        Series
            A Series whose index is the original column names
            and whose values is the memory usage of each column in bytes.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> dtypes = [int, float, str, bool]
        >>> data = {typ.__name__: [typ(1)] * 5000 for typ in dtypes}
        >>> df = cudf.DataFrame(data)
        >>> df.head()
        int  float str  bool
        0    1    1.0   1  True
        1    1    1.0   1  True
        2    1    1.0   1  True
        3    1    1.0   1  True
        4    1    1.0   1  True
        >>> df.memory_usage(index=False)
        int      40000
        float    40000
        str      25004
        bool      5000
        dtype: int64

        Use a Categorical for efficient storage of an object-dtype column with
        many repeated values.

        >>> df['str'].astype('category').memory_usage(deep=True)
        5009
        """
        mem_usage: Iterable[int] = (col.memory_usage for col in self._columns)
        result_index = self._data.to_pandas_index
        if index:
            mem_usage = itertools.chain([self.index.memory_usage()], mem_usage)
            result_index = pd.Index(["Index"]).append(result_index.astype(str))
        return Series._from_column(
            as_column(list(mem_usage)),
            index=Index(result_index),
        )

    @_performance_tracking
    def __array_function__(self, func, types, args, kwargs):
        if "out" in kwargs or not all(
            issubclass(t, (Series, DataFrame)) for t in types
        ):
            return NotImplemented

        try:
            if func.__name__ in {"any", "all"}:
                # NumPy default for `axis` is
                # different from `cudf`/`pandas`
                # hence need this special handling.
                kwargs.setdefault("axis", None)
            if cudf_func := getattr(self.__class__, func.__name__, None):
                out = cudf_func(*args, **kwargs)
                # The dot product of two DataFrames returns an array in pandas.
                if (
                    func is np.dot
                    and isinstance(args[0], (DataFrame, pd.DataFrame))
                    and isinstance(args[1], (DataFrame, pd.DataFrame))
                ):
                    return out.values
                return out
        except Exception:
            # The rare instance where a "silent" failure is preferable. Except
            # in the (highly unlikely) case that some other library
            # interoperates with cudf objects, the result will be that numpy
            # raises a TypeError indicating that the operation is not
            # implemented, which is much friendlier than an arbitrary internal
            # cudf error.
            pass
        return NotImplemented

    def __arrow_c_stream__(self, requested_schema=None):
        """
        Export the cudf DataFrame as an Arrow C stream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the dataframe should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema. Currently not implemented.

        Returns
        -------
        PyCapsule
        """
        if requested_schema is not None:
            raise NotImplementedError("requested_schema is not supported")
        return self.to_arrow().__arrow_c_stream__()

    # The _get_numeric_data method is necessary for dask compatibility.
    @_performance_tracking
    def _get_numeric_data(self):
        """Return a dataframe with only numeric data types"""
        columns = [c for c, dt in self._dtypes if is_dtype_obj_numeric(dt)]
        return self[columns]

    @_performance_tracking
    def assign(self, **kwargs: Callable[[Self], Any] | Any):
        """
        Assign columns to DataFrame from keyword arguments.

        Parameters
        ----------
        **kwargs: dict mapping string column names to values
            The value for each key can either be a literal column (or
            something that can be converted to a column), or
            a callable of one argument that will be given the
            dataframe as an argument and should return the new column
            (without modifying the input argument).
            Columns are added in-order, so callables can refer to
            column names constructed in the assignment.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df = df.assign(a=[0, 1, 2], b=[3, 4, 5])
        >>> df
           a  b
        0  0  3
        1  1  4
        2  2  5
        """
        new_df = self.copy(deep=False)
        for k, v in kwargs.items():
            new_df[k] = v(new_df) if callable(v) else v
        return new_df

    @classmethod
    @_performance_tracking
    def _concat(
        cls, objs, axis=0, join="outer", ignore_index=False, sort=False
    ):
        # flag to indicate at least one empty input frame also has an index
        empty_has_index = False
        # length of output frame's RangeIndex if all input frames are empty,
        # and at least one has an index
        result_index_length = 0
        # the number of empty input frames
        num_empty_input_frames = 0

        # flag to indicate if all DataFrame's have
        # RangeIndex as their index
        are_all_range_index = False

        for i, obj in enumerate(objs):
            # shallow-copy the input DFs in case the same DF instance
            # is concatenated with itself
            objs[i] = obj.copy(deep=False)

            # If ignore_index is true, determine if
            # all or some objs are empty(and have index).
            # 1. If all objects are empty(and have index), we
            # should set the index separately using RangeIndex.
            # 2. If some objects are empty(and have index), we
            # create empty columns later while populating `columns`
            # variable. Detailed explanation of second case before
            # allocation of `columns` variable below.
            if ignore_index and obj.empty:
                num_empty_input_frames += 1
                result_index_length += len(obj)
                empty_has_index = empty_has_index or len(obj) > 0

            are_all_range_index = (
                True if i == 0 else are_all_range_index
            ) and isinstance(obj.index, cudf.RangeIndex)

        if join == "inner":
            sets_of_column_names = [set(obj._column_names) for obj in objs]

            intersecting_columns = functools.reduce(
                set.intersection, sets_of_column_names
            )
            union_of_columns = functools.reduce(
                set.union, sets_of_column_names
            )
            non_intersecting_columns = union_of_columns.symmetric_difference(
                intersecting_columns
            )

            # Get an ordered list of the intersecting columns to preserve input
            # order, which is promised by pandas for inner joins.
            ordered_intersecting_columns = [
                name
                for obj in objs
                for name in obj._column_names
                if name in intersecting_columns
            ]

            names = dict.fromkeys(ordered_intersecting_columns).keys()

            if axis == 0:
                if ignore_index and (
                    num_empty_input_frames > 0
                    or len(intersecting_columns) == 0
                ):
                    # When ignore_index is True and if there is
                    # at least 1 empty dataframe and no
                    # intersecting columns are present, an empty dataframe
                    # needs to be returned just with an Index.
                    empty_has_index = True
                    num_empty_input_frames = len(objs)
                    result_index_length = sum(len(obj) for obj in objs)

                # remove columns not present in all objs
                for obj in objs:
                    obj.drop(
                        columns=non_intersecting_columns,
                        inplace=True,
                        errors="ignore",
                    )
        elif join == "outer":
            # Get a list of the unique table column names
            names = [name for f in objs for name in f._column_names]
            names = dict.fromkeys(names).keys()

        else:
            raise ValueError(
                "Only can inner (intersect) or outer (union) when joining"
                "the other axis"
            )

        if sort:
            try:
                # Sorted always returns a list, but will fail to sort if names
                # include different types that are not comparable.
                names = sorted(names)
            except TypeError:
                # For pandas compatibility, we also try to handle the case
                # where some column names are strings and others are ints. Just
                # assume that everything that isn't a str is numerical, we
                # can't sort anything else.
                try:
                    str_names = sorted(n for n in names if isinstance(n, str))
                    non_str_names = sorted(
                        n for n in names if not isinstance(n, str)
                    )
                    names = non_str_names + str_names
                except TypeError:
                    names = list(names)
        else:
            names = list(names)

        # Combine the index and table columns for each Frame into a list of
        # [...index_cols, ...table_cols].
        #
        # If any of the input frames have a non-empty index, include these
        # columns in the list of columns to concatenate, even if the input
        # frames are empty and `ignore_index=True`.
        columns = [
            (
                []
                if are_all_range_index
                or (ignore_index and not empty_has_index)
                else list(f.index._columns)
            )
            + [f._data[name] if name in f._data else None for name in names]
            for f in objs
        ]

        # Get a list of the combined index and table column indices
        indices = list(range(functools.reduce(max, map(len, columns))))
        # The position of the first table column in each
        # combined index + table columns list
        first_data_column_position = len(indices) - len(names)

        # Get the non-null columns and their dtypes
        non_null_cols, dtypes = _get_non_null_cols_and_dtypes(indices, columns)

        # Infer common dtypes between numeric columns
        # and combine CategoricalColumn categories
        categories = _find_common_dtypes_and_categories(non_null_cols, dtypes)

        # Cast all columns to a common dtype, assign combined categories,
        # and back-fill missing columns with all-null columns
        _cast_cols_to_common_dtypes(indices, columns, dtypes, categories)

        # Construct input tables with the index and data columns in the same
        # order. This strips the given index/column names and replaces the
        # names with their integer positions in the `cols` list
        tables = []
        for cols in columns:
            table_index = None
            if 1 == first_data_column_position:
                table_index = Index._from_column(cols[0])
            elif first_data_column_position > 1:
                table_index = cudf.MultiIndex._from_data(
                    data=dict(
                        zip(
                            indices[:first_data_column_position],
                            cols[:first_data_column_position],
                            strict=True,
                        )
                    )
                )
            tables.append(
                DataFrame._from_data(
                    data=dict(
                        zip(
                            indices[first_data_column_position:],
                            cols[first_data_column_position:],
                            strict=True,
                        )
                    ),
                    index=table_index,
                )
            )

        # Concatenate the Tables
        ignore = ignore_index or are_all_range_index
        index_names = None if ignore else tables[0]._index_names
        column_names = tables[0]._column_names
        with access_columns(
            *(
                col
                for table in tables
                for col in (
                    table._columns
                    if ignore
                    else itertools.chain(table.index._columns, table._columns)
                )
            ),
            mode="read",
            scope="internal",
        ) as accessed_cols:
            # Build mapping from original columns to accessed columns
            col_map = {}
            accessed_idx = 0
            for table in tables:
                for col in (
                    table._columns
                    if ignore
                    else itertools.chain(table.index._columns, table._columns)
                ):
                    col_map[id(col)] = accessed_cols[accessed_idx]
                    accessed_idx += 1

            plc_tables = [
                plc.Table(
                    [
                        col_map[id(c)].plc_column
                        for c in (
                            table._columns
                            if ignore
                            else itertools.chain(
                                table.index._columns, table._columns
                            )
                        )
                    ]
                )
                for table in tables
            ]
            # Preserve dtypes from the first table
            if ignore:
                reference_dtypes = [col.dtype for col in tables[0]._columns]
            else:
                reference_dtypes = [
                    col.dtype
                    for col in itertools.chain(
                        tables[0].index._columns, tables[0]._columns
                    )
                ]
            plc_result = plc.concatenate.concatenate(plc_tables)
            if ignore:
                index = None
                data = {
                    col_name: ColumnBase.create(col, dtype)
                    for col_name, col, dtype in zip(
                        column_names,
                        plc_result.columns(),
                        reference_dtypes,
                        strict=True,
                    )
                }
            else:
                result_columns = [
                    ColumnBase.create(col, dtype)
                    for col, dtype in zip(
                        plc_result.columns(), reference_dtypes, strict=True
                    )
                ]
                index = _index_from_data(
                    dict(
                        zip(
                            index_names,
                            result_columns[: len(index_names)],
                            strict=True,
                        )
                    )
                )
                data = dict(
                    zip(
                        column_names,
                        result_columns[len(index_names) :],
                        strict=True,
                    )
                )
        out = cls._from_data(data=data, index=index)

        # If ignore_index is True, all input frames are empty, and at
        # least one input frame has an index, assign a new RangeIndex
        # to the result frame.
        if empty_has_index and num_empty_input_frames == len(objs):
            out.index = cudf.RangeIndex(result_index_length)
        elif are_all_range_index and not ignore_index:
            out.index = Index._concat([o.index for o in objs])

        # Reassign the categories for any categorical table cols
        _reassign_categories(
            categories, out._data, indices[first_data_column_position:]
        )

        # Reassign the categories for any categorical index cols
        if not isinstance(out.index, cudf.RangeIndex):
            # If the index column was constructed and not generated via concatenation,
            # then reassigning categories is neither needed nor a valid operation.
            if first_data_column_position > 0:
                _reassign_categories(
                    categories,
                    out.index._data,
                    indices[:first_data_column_position],
                )
        if not isinstance(out.index, MultiIndex) and isinstance(
            out.index.dtype, CategoricalDtype
        ):
            out = out.set_index(out.index)

        # Reassign index and column names
        if objs[0]._data.multiindex:
            out._set_columns_like(objs[0]._data)
        elif (
            all(obj._data.rangeindex for obj in objs)
            and all(
                obj._num_columns == 0
                or (
                    obj._column_names[0] == 0
                    and obj._column_names[-1] == obj._num_columns - 1
                )
                for obj in objs
            )
            and tuple(names) == tuple(range(len(names)))
        ):
            out.columns = cudf.RangeIndex(len(names))
        else:
            out.columns = names
        if not ignore_index:
            out.index.name = objs[0].index.name
            out.index.names = objs[0].index.names

        if isinstance(out.index, DatetimeIndex):
            try:
                out.index._freq = out.index.inferred_freq
            except NotImplementedError:
                out.index._freq = None
        return out

    def astype(
        self,
        dtype: Dtype | dict[Hashable, Dtype],
        copy: bool | None = None,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> Self:
        if copy is None:
            copy = True
        if is_dict_like(dtype):
            if len(set(dtype.keys()) - set(self._column_names)) > 0:  # type: ignore[union-attr]
                raise KeyError(
                    "Only a column name can be used for the "
                    "key in a dtype mappings argument."
                )
            dtype = {
                col_name: cudf.dtype(dtype)
                for col_name, dtype in dtype.items()  # type: ignore[union-attr]
            }
        else:
            dtype = {cc: cudf.dtype(dtype) for cc in self._column_names}
        return super().astype(dtype, copy, errors)

    def _clean_renderable_dataframe(self, output: Self) -> str:
        """
        This method takes in partial/preprocessed dataframe
        and returns correct representation of it with correct
        dimensions (rows x columns)
        """

        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        max_cols = pd.get_option("display.max_columns")
        max_colwidth = pd.get_option("display.max_colwidth")
        show_dimensions = pd.get_option("display.show_dimensions")
        if pd.get_option("display.expand_frame_repr"):
            width, _ = console.get_console_size()
        else:
            width = None

        output = output.to_pandas().to_string(
            max_rows=max_rows,
            min_rows=min_rows,
            max_cols=max_cols,
            line_width=width,
            max_colwidth=max_colwidth,
            show_dimensions=show_dimensions,
        )

        lines = output.split("\n")

        if lines[-1].startswith("["):
            lines = lines[:-1]
            lines.append(
                "[%d rows x %d columns]" % (len(self), self._num_columns)
            )
        return "\n".join(lines)

    def _get_renderable_dataframe(self) -> Self:
        """
        Takes rows and columns from pandas settings or estimation from size.
        pulls quadrants based off of some known parameters then style for
        multiindex as well producing an efficient representative string
        for printing with the dataframe.
        """
        max_rows = pd.options.display.max_rows
        if max_rows in {0, None}:
            max_rows = len(self)
        nrows = max(max_rows, 1)
        ncols = (
            pd.options.display.max_columns
            if pd.options.display.max_columns
            else pd.options.display.width / 2
        )

        if len(self) <= nrows and self._num_columns <= ncols:
            output = self
        elif self.empty and len(self) > 0:
            max_seq_items = pd.options.display.max_seq_items
            # In case of Empty DataFrame with index, Pandas prints
            # first `pd.options.display.max_seq_items` index values
            # followed by ... To obtain ... at the end of index list,
            # adding 1 extra value.
            # If `pd.options.display.max_seq_items` is None,
            # entire sequence/Index is to be printed.
            # Note : Pandas truncates the dimensions at the end of
            # the resulting dataframe when `display.show_dimensions`
            # is set to truncate. Hence to display the dimensions we
            # need to extract maximum of `max_seq_items` and `nrows`
            # and have 1 extra value for ... to show up in the output
            # string.
            if max_seq_items is not None:
                output = self.head(max(max_seq_items, nrows) + 1)
            else:
                output = self.copy(deep=False)
        else:
            left_cols = self._num_columns
            right_cols = 0
            upper_rows = len(self)
            lower_rows = 0
            if len(self) > nrows and nrows > 0:
                upper_rows = int(nrows / 2.0) + 1
                lower_rows = upper_rows + (nrows % 2)
            if left_cols > ncols:
                right_cols = left_cols - int(ncols / 2.0)
                # adjust right columns for output if multiindex.
                right_cols = (
                    right_cols - 1
                    if isinstance(self.index, MultiIndex)
                    else right_cols
                )
                left_cols = int(ncols / 2.0) + 1
            if right_cols > 0:
                # Pick ncols - left_cols number of columns
                # from the right side/from the end.
                right_cols = -(int(ncols) - left_cols + 1)
            else:
                # If right_cols is 0 or negative, it means
                # self has lesser number of columns than ncols.
                # Hence assign self._num_columns which
                # will result in empty `*_right` quadrants.
                # This is because `*_left` quadrants will
                # contain all columns.
                right_cols = self._num_columns

            upper_left = self.head(upper_rows).iloc[:, :left_cols]
            upper_right = self.head(upper_rows).iloc[:, right_cols:]
            lower_left = self.tail(lower_rows).iloc[:, :left_cols]
            lower_right = self.tail(lower_rows).iloc[:, right_cols:]

            upper = cudf.concat([upper_left, upper_right], axis=1)
            lower = cudf.concat([lower_left, lower_right], axis=1)
            output = cudf.concat([upper, lower])

        return output._pandas_repr_compatible()

    @_performance_tracking
    def __repr__(self):
        output = self._get_renderable_dataframe()
        return self._clean_renderable_dataframe(output)

    @_performance_tracking
    def _repr_html_(self):
        lines = (
            self._get_renderable_dataframe()
            .to_pandas()
            ._repr_html_()
            .split("\n")
        )
        if lines[-2].startswith("<p>"):
            lines = lines[:-2]
            lines.append(
                "<p>%d rows × %d columns</p>" % (len(self), self._num_columns)
            )
            lines.append("</div>")
        return "\n".join(lines)

    @_performance_tracking
    def _repr_latex_(self):
        return self._get_renderable_dataframe().to_pandas()._repr_latex_()

    def _make_operands_and_index_for_binop(
        self,
        other: Any,
        fn: str,
        fill_value: Any = None,
        reflect: bool = False,
        can_reindex: bool = False,
    ) -> tuple[
        dict[str | None, tuple[ColumnBase, Any, bool, Any]]
        | NotImplementedType,
        Index | None,
        dict[str, Any],
    ]:
        lhs, rhs = self._data, other
        index = self.index
        fill_requires_key = False
        left_default: Any = False
        equal_columns = False
        ca_attributes: dict[str, Any] = {}

        def _fill_same_ca_attributes(
            attrs: dict[str, Any], ca: ColumnAccessor
        ) -> dict[str, Any]:
            attrs["rangeindex"] = ca.rangeindex
            attrs["multiindex"] = ca.multiindex
            attrs["label_dtype"] = ca.label_dtype
            attrs["level_names"] = ca.level_names
            return attrs

        if _is_scalar_or_zero_d_array(other):
            rhs = {name: other for name in self._data}
            equal_columns = True
            ca_attributes = _fill_same_ca_attributes(ca_attributes, self._data)
        elif isinstance(other, Series):
            if (
                not (self_pd_columns := self._data.to_pandas_index).equals(
                    other_pd_index := other.index.to_pandas()
                )
                and not can_reindex
                and fn in _EQUALITY_OPS
            ):
                raise ValueError(
                    "Can only compare DataFrame & Series objects "
                    "whose columns & index are same respectively, "
                    "please reindex."
                )
            rhs = dict(zip(other_pd_index, other.to_numpy(), strict=True))
            # For keys in right but not left, perform binops between NaN (not
            # NULL!) and the right value (result is NaN).
            left_default = as_column(np.nan, length=len(self))
            equal_columns = other_pd_index.equals(self_pd_columns)
            if equal_columns:
                ca_attributes = _fill_same_ca_attributes(
                    ca_attributes, self._data
                )
            elif other_pd_index.names == self_pd_columns.names:
                ca_attributes["level_names"] = self._data.level_names
        elif isinstance(other, DataFrame):
            if (
                not can_reindex
                and fn in _EQUALITY_OPS
                and (
                    not self.index.equals(other.index)
                    or not self._data.to_pandas_index.equals(
                        other._data.to_pandas_index
                    )
                )
            ):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            new_lhs, new_rhs = _align_indices(self, other)
            index = new_lhs.index
            lhs, rhs = new_lhs._data, new_rhs._data
            fill_requires_key = True
            # For DataFrame-DataFrame ops, always default to operating against
            # the fill value.
            left_default = fill_value
            equal_columns = self._column_names == other._column_names
            if self._data.to_pandas_index.equals(other._data.to_pandas_index):
                ca_attributes = _fill_same_ca_attributes(
                    ca_attributes, self._data
                )
            elif self._data._level_names == other._data._level_names:
                ca_attributes["level_names"] = self._data.level_names
                if self._data.multiindex == other._data.multiindex:
                    # equal labels can still fail the ``equals`` check above
                    # on level-dtype differences (e.g. Int8 vs int64); the
                    # result keeps hierarchical columns like pandas
                    ca_attributes["multiindex"] = self._data.multiindex
        elif isinstance(other, (dict, Mapping)):
            # Need to fail early on host mapping types because we ultimately
            # convert everything to a dict.
            return NotImplemented, None, ca_attributes

        if not isinstance(rhs, (dict, Mapping)):
            return NotImplemented, None, ca_attributes

        operands = {
            k: (
                v,
                rhs.get(k, fill_value),
                reflect,
                fill_value if (not fill_requires_key or k in rhs) else None,
            )
            for k, v in lhs.items()
        }

        if left_default is not False:
            for k, v in rhs.items():
                if k not in lhs:
                    operands[k] = (left_default, v, reflect, None)

        if not equal_columns:
            if isinstance(other, DataFrame):
                column_names_list = self._data.to_pandas_index.join(
                    other._data.to_pandas_index, how="outer"
                )
            elif isinstance(other, Series):
                column_names_list = self._data.to_pandas_index.join(
                    other.index.to_pandas(), how="outer"
                )
            else:
                raise ValueError("other must be a DataFrame or Series.")

            if isinstance(column_names_list, pd.MultiIndex):
                ca_attributes["multiindex"] = True
                ca_attributes["level_names"] = tuple(column_names_list.names)
            sorted_dict = {key: operands[key] for key in column_names_list}
            return sorted_dict, index, ca_attributes
        return operands, index, ca_attributes

    @classmethod
    @_performance_tracking
    def from_dict(
        cls,
        data: dict,
        orient: str = "columns",
        dtype: Dtype | None = None,
        columns: list | None = None,
    ) -> DataFrame:
        """
        Construct DataFrame from dict of array-like or dicts.
        Creates DataFrame object from dictionary by columns or by index
        allowing dtype specification.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        orient : {'columns', 'index', 'tight'}, default 'columns'
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass 'columns'
            (default). Otherwise if the keys should be rows, pass 'index'.
            If 'tight', assume a dict with keys ['index', 'columns', 'data',
            'index_names', 'column_names'].
        dtype : dtype, default None
            Data type to force, otherwise infer.
        columns : list, default None
            Column labels to use when ``orient='index'``. Raises a ``ValueError``
            if used with ``orient='columns'`` or ``orient='tight'``.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_records : DataFrame from structured ndarray, sequence
            of tuples or dicts, or DataFrame.
        DataFrame : DataFrame object creation using constructor.
        DataFrame.to_dict : Convert the DataFrame to a dictionary.

        Examples
        --------
        By default the keys of the dict become the DataFrame columns:

        >>> import cudf
        >>> data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
        >>> cudf.DataFrame.from_dict(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Specify ``orient='index'`` to create the DataFrame using dictionary
        keys as rows:

        >>> data = {'row_1': [3, 2, 1, 0], 'row_2': [10, 11, 12, 13]}
        >>> cudf.DataFrame.from_dict(data, orient='index')
                0   1   2   3
        row_1   3   2   1   0
        row_2  10  11  12  13

        When using the 'index' orientation, the column names can be
        specified manually:

        >>> cudf.DataFrame.from_dict(data, orient='index',
        ...                          columns=['A', 'B', 'C', 'D'])
                A   B   C   D
        row_1   3   2   1   0
        row_2  10  11  12  13

        Specify ``orient='tight'`` to create the DataFrame using a 'tight'
        format:

        >>> data = {'index': [('a', 'b'), ('a', 'c')],
        ...         'columns': [('x', 1), ('y', 2)],
        ...         'data': [[1, 3], [2, 4]],
        ...         'index_names': ['n1', 'n2'],
        ...         'column_names': ['z1', 'z2']}
        >>> cudf.DataFrame.from_dict(data, orient='tight')
        z1     x  y
        z2     1  2
        n1 n2
        a  b   1  3
           c   2  4
        """

        orient = orient.lower()
        if orient == "index":
            if isinstance(
                next(iter(data.values()), None), (Series, cupy.ndarray)
            ):
                result = cls(data).T
                result.columns = (
                    columns
                    if columns is not None
                    else range(result._num_columns)
                )
                if dtype is not None:
                    result = result.astype(dtype)
                return result
            else:
                return cls(
                    pd.DataFrame.from_dict(
                        data=data,
                        orient=orient,
                        dtype=dtype,
                        columns=columns,
                    )
                )
        elif orient == "columns":
            if columns is not None:
                raise ValueError(
                    "Cannot use columns parameter with orient='columns'"
                )
            return cls(data, columns=None, dtype=dtype)
        elif orient == "tight":
            if columns is not None:
                raise ValueError(
                    "Cannot use columns parameter with orient='right'"
                )

            index = _from_dict_create_index(
                data["index"], data["index_names"], cudf
            )
            columns = _from_dict_create_index(
                data["columns"], data["column_names"], pd
            )
            return cls(data["data"], index=index, columns=columns, dtype=dtype)
        else:
            raise ValueError(
                "Expected 'index', 'columns' or 'tight' for orient "
                f"parameter. Got '{orient}' instead"
            )

    @_performance_tracking
    def to_dict(
        self,
        orient: str = "dict",
        into: type[dict] = dict,
        index: bool = True,
    ) -> dict | list[dict]:
        """
        Convert the DataFrame to a dictionary.

        The type of the key-value pairs can be customized with the parameters
        (see below).

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
            Determines the type of the values of the dictionary.

            - 'dict' (default) : dict like {column -> {index -> value}}
            - 'list' : dict like {column -> [values]}
            - 'series' : dict like {column -> Series(values)}
            - 'split' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            - 'tight' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
              'index_names' -> [index.names], 'column_names' -> [column.names]}
            - 'records' : list like
              [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}

            Abbreviations are allowed. `s` indicates `series` and `sp`
            indicates `split`.

        into : class, default dict
            The collections.abc.Mapping subclass used for all Mappings
            in the return value.  Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        index : bool, default True
            Whether to include the index item (and index_names item if `orient`
            is 'tight') in the returned dictionary. Can only be ``False``
            when `orient` is 'split' or 'tight'. Note that when `orient` is
            'records', this parameter does not take effect (index item always
            not included).

        Returns
        -------
        dict, list or collections.abc.Mapping
            Return a collections.abc.Mapping object representing the DataFrame.
            The resulting transformation depends on the `orient` parameter.

        See Also
        --------
        DataFrame.from_dict: Create a DataFrame from a dictionary.
        DataFrame.to_json: Convert a DataFrame to JSON format.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'col1': [1, 2],
        ...                      'col2': [0.5, 0.75]},
        ...                     index=['row1', 'row2'])
        >>> df
              col1  col2
        row1     1  0.50
        row2     2  0.75
        >>> df.to_dict()
        {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 0.5, 'row2': 0.75}}

        You can specify the return orientation.

        >>> df.to_dict('series')
        {'col1': row1    1
                 row2    2
        Name: col1, dtype: int64,
        'col2': row1    0.50
                row2    0.75
        Name: col2, dtype: float64}

        >>> df.to_dict('split')
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]]}

        >>> df.to_dict('records')
        [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]

        >>> df.to_dict('index')
        {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}

        >>> df.to_dict('tight')
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]], 'index_names': [None], 'column_names': [None]}

        You can also specify the mapping type.

        >>> from collections import OrderedDict, defaultdict
        >>> df.to_dict(into=OrderedDict)  # doctest: +SKIP
        OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])),
                     ('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])

        If you want a `defaultdict`, you need to initialize it:

        >>> dd = defaultdict(list)
        >>> df.to_dict('records', into=dd)
        [defaultdict(<class 'list'>, {'col1': 1, 'col2': 0.5}),
         defaultdict(<class 'list'>, {'col1': 2, 'col2': 0.75})]
        """
        orient = orient.lower()

        if orient == "series":
            # Special case needed to avoid converting
            # Series objects into pd.Series
            if not inspect.isclass(into):
                cons = type(into)
                if isinstance(into, defaultdict):
                    cons = functools.partial(cons, into.default_factory)
            elif issubclass(into, Mapping):
                cons = into
                if issubclass(into, defaultdict):
                    raise TypeError(
                        "to_dict() only accepts initialized defaultdicts"
                    )
            else:
                raise TypeError(f"unsupported type: {into}")
            return cons(self.items())

        return self.to_pandas().to_dict(orient=orient, into=into, index=index)

    @_performance_tracking
    def to_string(
        self,
        buf=None,
        *,
        columns=None,
        col_space=None,
        header=True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters=None,
        float_format=None,
        sparsify: bool | None = None,
        index_names: bool = True,
        justify: str | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: bool = False,
        decimal: str = ".",
        line_width: int | None = None,
        min_rows: int | None = None,
        max_colwidth: int | None = None,
        encoding: str | None = None,
    ) -> str | None:
        r"""
        Render a DataFrame to a console-friendly tabular output.

        cuDF uses pandas internals for string formatting, so this mirrors
        :meth:`pandas.DataFrame.to_string` and accepts the same arguments.
        Unlike ``repr``, the output is not truncated by the
        ``display.max_rows``/``display.max_columns`` options unless
        ``max_rows``/``max_cols`` are passed explicitly.

        cuDF supports `null/None` as a value in any column type, which
        is transparently supported during this output process.

        Parameters
        ----------
        buf : writable buffer, optional
            File path or buffer to write to. If ``None`` (default), the
            output is returned as a string.
        columns : sequence, optional
            The subset of columns to write. Writes all columns by default.
        col_space : int, list or dict of int, optional
            The minimum width of each column.
        header : bool or list of str, default True
            Whether to write the column names. If a list of strings is
            given, it is assumed to be aliases for the column names.
        index : bool, default True
            Whether to print index (row) labels.
        na_rep : str, default "NaN"
            String representation of ``NaN``/null values.
        formatters : list, tuple or dict of callables, optional
            One-parameter formatter functions to apply to columns' elements
            by position or name.
        float_format : callable, optional
            One-parameter formatter function applied to floating point
            values.
        sparsify : bool, optional
            Set to ``False`` for a DataFrame with a hierarchical index to
            print every multiindex key at each row. Defaults to ``True``.
        index_names : bool, default True
            Whether to print the names of the indexes.
        justify : str, optional
            How to justify the column labels (e.g. ``"left"``, ``"right"``,
            ``"center"``). Defaults to the pandas display option.
        max_rows : int, optional
            Maximum number of rows to display before truncating.
        max_cols : int, optional
            Maximum number of columns to display before truncating.
        show_dimensions : bool, default False
            Whether to display the DataFrame dimensions (number of rows
            and columns).
        decimal : str, default "."
            Character recognized as the decimal separator, e.g. ``","`` in
            Europe.
        line_width : int, optional
            Width, in characters, at which to wrap a line.
        min_rows : int, optional
            Number of rows to show in a truncated output (when the number
            of rows exceeds ``max_rows``).
        max_colwidth : int, optional
            Maximum width, in characters, to truncate each column. By
            default there is no limit.
        encoding : str, optional
            Character encoding to use when writing to ``buf``.

        Returns
        -------
        str or None
            The string representation of the DataFrame when ``buf`` is
            ``None``; otherwise the result is written to ``buf`` and
            ``None`` is returned.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2]
        >>> df['val'] = [float(i + 10) for i in range(3)]
        >>> df.to_string()
        '   key   val\n0    0  10.0\n1    1  11.0\n2    2  12.0'
        """
        return self.to_pandas().to_string(
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            line_width=line_width,
            min_rows=min_rows,
            max_colwidth=max_colwidth,
            encoding=encoding,
        )

    @_performance_tracking
    def scatter_by_map(
        self,
        map_index,
        map_size: int | None = None,
        keep_index: bool = True,
        debug: bool = False,
    ) -> list[Self]:
        """Scatter to a list of dataframes.

        Uses map_index to determine the destination
        of each row of the original DataFrame.

        Parameters
        ----------
        map_index : Series, str or list-like
            Scatter assignment for each row
        map_size : int
            Length of output list. Must be >= uniques in map_index
        keep_index : bool
            Conserve original index values for each row

        Returns
        -------
        A list of cudf.DataFrame objects.

        Raises
        ------
        ValueError
            If the map_index has invalid entries (not all in [0,
            num_partitions)).
        """
        # map_index might be a column name or array,
        # make it a Column
        if isinstance(map_index, str):
            map_index = self._data[map_index]
        elif isinstance(map_index, Series):
            map_index = map_index._column
        else:
            map_index = as_column(map_index)

        # Convert float to integer
        if map_index.dtype.kind == "f":
            map_index = map_index.astype(SIZE_TYPE_DTYPE)

        # Convert string or categorical to integer
        if is_dtype_obj_string(map_index.dtype):
            map_index = map_index._label_encoding(map_index.unique())
            warnings.warn(
                "Using StringColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )
        elif isinstance(map_index.dtype, CategoricalDtype):
            map_index = map_index.codes
            warnings.warn(
                "Using CategoricalColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )

        if debug and map_size is not None:
            count = map_index.distinct_count()
            if map_size < count:
                raise ValueError(
                    f"ERROR: map_size must be >= {count} (got {map_size})."
                )

        source_columns = (
            itertools.chain(self.index._columns, self._columns)
            if keep_index
            else self._columns
        )

        if map_size is None:
            map_size = map_index.distinct_count(dropna=True)

        if map_index.size > 0:
            lo, hi = map_index.minmax()
            if lo < 0 or hi >= map_size:
                raise ValueError("Partition map has invalid values")

        # Materialize iterator to avoid consuming it during access context setup
        source_columns_list = list(source_columns)
        with access_columns(
            *source_columns_list, map_index, mode="read", scope="internal"
        ) as (*source_columns_list, map_index):
            plc_table, offsets = plc.partitioning.partition(
                plc.Table([col.plc_column for col in source_columns_list]),
                map_index.plc_column,
                map_size,
            )
        return self._wrap_from_partitions(
            plc_table,
            offsets,
            keep_index=keep_index,
            size=map_size,
            by_hash=False,
        )

    @_performance_tracking
    def update(
        self,
        other,
        join="left",
        overwrite=True,
        filter_func=None,
        errors="ignore",
    ):
        """
        Modify a DataFrame in place using non-NA values from another DataFrame.

        Aligns on indices. There is no return value.

        Parameters
        ----------
        other : DataFrame, or object coercible into a DataFrame
            Should have at least one matching index/column label with the
            original DataFrame. If a Series is passed, its name attribute must
            be set, and that will be used as the column name to align with the
            original DataFrame.

        join : {'left'}, default 'left'
            Only left join is implemented, keeping the index and
            columns of the original object.

        overwrite : {True, False}, default True
            How to handle non-NA values for overlapping keys:
            True: overwrite original DataFrame's values with values from other.
            False: only update values that are NA in the original DataFrame.

        filter_func : None
            filter_func is not supported yet
            Return True for values that should be updated.S

        errors : {'raise', 'ignore'}, default 'ignore'
            If 'raise', will raise a ValueError if the DataFrame and other
            both contain non-NA data in the same place.


        Returns
        -------
        None : method directly changes calling object

        Raises
        ------
        ValueError
            - When ``errors`` = 'raise' and there's overlapping non-NA data.
            - When ``errors`` is not either 'ignore' or 'raise'

        NotImplementedError
            - If ``join`` != 'left'
        """
        # TODO: Support other joins
        if join != "left":
            raise NotImplementedError("Only left join is supported")
        if errors not in {"ignore", "raise"}:
            raise ValueError(
                "The parameter errors must be either 'ignore' or 'raise'"
            )
        if filter_func is not None:
            raise NotImplementedError("filter_func is not supported yet")

        if not isinstance(other, DataFrame):
            other = DataFrame(other)

        self_cols = self._data.to_pandas_index
        if not self_cols.equals(other._data.to_pandas_index):
            other = other.reindex(self_cols, axis=1)
        if not self.index.equals(other.index):
            other = other.reindex(self.index, axis=0)

        source_df = self.copy(deep=False)
        for col in source_df._column_names:
            this = source_df[col]
            that = other[col]

            if errors == "raise":
                mask_this = that.notna()
                mask_that = this.notna()
                if (mask_this & mask_that).any():
                    raise ValueError("Data overlaps.")

            if overwrite:
                mask = that.isna()
            else:
                mask = this.notna()

            # don't overwrite columns unnecessarily
            if mask.all():
                continue
            source_df[col] = source_df[col].where(mask, that)

        self._mimic_inplace(source_df, inplace=True)

    @_performance_tracking
    def __iter__(self):
        return iter(self._column_names)

    @_performance_tracking
    def __contains__(self, item):
        # This must check against containment in the pandas Index and not
        # self._column_names to handle NA, None, nan, etc. correctly.
        return item in self._data.to_pandas_index

    @_performance_tracking
    def items(self):
        """Iterate over column names and series pairs"""
        for k in self:
            yield (k, self[k])

    @_performance_tracking
    def equals(self, other) -> bool:
        # Check that column labels match too
        return super().equals(other) and all(
            self_name == other_name
            for self_name, other_name in zip(
                self._column_names, other._column_names, strict=True
            )
        )

    @property
    def iat(self):
        """
        Alias for ``DataFrame.iloc``; provided for compatibility with Pandas.
        """
        return _DataFrameiAtIndexer(self)

    @property
    def at(self):
        """
        Alias for ``DataFrame.loc``; provided for compatibility with Pandas.
        """
        return _DataFrameAtIndexer(self)

    @property
    @_external_only_api(
        "Use _column_names instead, or _data.to_pandas_index if a pandas "
        "index is absolutely necessary. For checking if the columns are a "
        "MultiIndex, use _data.multiindex."
    )
    @_performance_tracking
    def columns(self):
        """Returns a tuple of columns"""
        return self._data.to_pandas_index

    @columns.setter
    @_performance_tracking
    def columns(self, columns):
        multiindex = False
        rangeindex = False
        label_dtype = None
        level_names = None
        level_dtypes = None
        if isinstance(columns, (pd.MultiIndex, cudf.MultiIndex)):
            multiindex = True
            if isinstance(columns, cudf.MultiIndex):
                pd_columns = columns.to_pandas()
            else:
                pd_columns = columns
            if pd_columns.nunique(dropna=False) != len(pd_columns):
                raise ValueError("Duplicate column names are not allowed")
            level_names = list(pd_columns.names)
            level_dtypes = _pd_index_level_dtypes(pd_columns)
        elif isinstance(columns, (Index, ColumnBase, Series)):
            level_names = (getattr(columns, "name", None),)
            rangeindex = isinstance(columns, cudf.RangeIndex)
            if rangeindex:
                unique_count = len(columns)
            else:
                columns = as_column(columns)
                unique_count = columns.distinct_count(dropna=False)
            if unique_count != len(columns):
                raise ValueError("Duplicate column names are not allowed")
            pd_columns = pd.Index(columns.to_pandas())
            label_dtype = pd_columns.dtype
        else:
            has_dtype = isinstance(columns, (np.ndarray, pd.Series, pd.Index))
            pd_columns = pd.Index(columns)
            if pd_columns.nunique(dropna=False) != len(pd_columns):
                raise ValueError("Duplicate column names are not allowed")
            rangeindex = isinstance(pd_columns, pd.RangeIndex)
            level_names = (pd_columns.name,)
            if not has_dtype and not rangeindex and len(pd_columns) == 0:
                label_dtype = self._data.label_dtype
            else:
                label_dtype = pd_columns.dtype

        if len(pd_columns) != self._num_columns:
            raise ValueError(
                f"Length mismatch: expected {self._num_columns} elements, "
                f"got {len(pd_columns)} elements"
            )

        self._data = ColumnAccessor(
            data=dict(zip(pd_columns, self._columns, strict=True)),
            multiindex=multiindex,
            level_names=level_names,
            label_dtype=label_dtype,
            rangeindex=rangeindex,
            level_dtypes=level_dtypes,
            verify=False,
            pandas_index=pd_columns,
        )

    def _set_columns_like(self, other: ColumnAccessor) -> None:
        """
        Modify self with the column properties of other.

        * Whether .columns is a MultiIndex/RangeIndex
        * The possible .columns.dtype
        * The .columns.names/name (depending on if it's a MultiIndex)
        """
        if self._num_columns != len(other):
            raise ValueError(
                f"Length mismatch: expected {len(other)} elements, "
                f"got {len(self)} elements"
            )
        self._data = ColumnAccessor(
            data=dict(zip(other.names, self._columns, strict=True)),
            multiindex=other.multiindex,
            rangeindex=other.rangeindex,
            level_names=other.level_names,
            label_dtype=other.label_dtype,
            level_dtypes=other._level_dtypes,
            verify=False,
        )

    @_performance_tracking
    def reindex(
        self,
        labels=None,
        *,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=True,
        level=None,
        fill_value=NA,
        limit=None,
        tolerance=None,
    ):
        """
        Conform DataFrame to new index. Places NA/NaN in locations
        having no value in the previous index. A new object is produced
        unless the new index is equivalent to the current one and copy=False.

        Parameters
        ----------
        labels : Index, Series-convertible, optional, default None
            New labels / index to conform the axis specified by ``axis`` to.
        index : Index, Series-convertible, optional, default None
            The index labels specifying the index to conform to.
        columns : array-like, optional, default None
            The column labels specifying the columns to conform to.
        axis : Axis to target.
            Can be either the axis name
            (``index``, ``columns``) or number (0, 1).
        method : Not supported
        copy : boolean, default True
            Return a new object, even if the passed indexes are the same.
        level : Not supported
        fill_value : Value to use for missing values.
            Defaults to ``NA``, but can be any "compatible" value.
        limit : Not supported
        tolerance : Not supported

        Returns
        -------
        DataFrame with changed index.

        Examples
        --------
        ``DataFrame.reindex`` supports two calling conventions
        * ``(index=index_labels, columns=column_labels, ...)``
        * ``(labels, axis={'index', 'columns'}, ...)``
        We _highly_ recommend using keyword arguments to clarify your intent.

        Create a dataframe with some fictional data.

        >>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
        >>> df = cudf.DataFrame({'http_status': [200, 200, 404, 404, 301],
        ...                    'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
        ...                      index=index)
        >>> df
                http_status  response_time
        Firefox            200           0.04
        Chrome             200           0.02
        Safari             404           0.07
        IE10               404           0.08
        Konqueror          301           1.00
        >>> new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
        ...              'Chrome']
        >>> df.reindex(new_index)
                      http_status response_time
        Safari              404.0          0.07
        Iceweasel             NaN           NaN
        Comodo Dragon         NaN           NaN
        IE10                404.0          0.08
        Chrome              200.0          0.02

        We can fill in the missing values by
        passing a value to the keyword ``fill_value``.

        >>> df.reindex(new_index, fill_value=0)
                    http_status  response_time
        Safari                 404           0.07
        Iceweasel                0           0.00
        Comodo Dragon            0           0.00
        IE10                   404           0.08
        Chrome                 200           0.02

        We can also reindex the columns.

        >>> df.reindex(columns=['http_status', 'user_agent'])
                http_status user_agent
        Firefox            200        NaN
        Chrome             200        NaN
        Safari             404        NaN
        IE10               404        NaN
        Konqueror          301        NaN

        Or we can use "axis-style" keyword arguments

        >>> df.reindex(columns=['http_status', 'user_agent'])
                http_status user_agent
        Firefox            200        NaN
        Chrome             200        NaN
        Safari             404        NaN
        IE10               404        NaN
        Konqueror          301        NaN
        """

        if labels is None and index is None and columns is None:
            return self.copy(deep=copy)

        if labels is not None and index is not None and columns is not None:
            raise TypeError(
                "Cannot specify all of 'labels', 'index', 'columns'."
            )

        # pandas simply ignores the labels keyword if it is provided in
        # addition to index and columns, but it prohibits the axis arg.
        if (index is not None or columns is not None) and axis is not None:
            raise TypeError(
                "Cannot specify both 'axis' and any of 'index' or 'columns'."
            )

        axis = 0 if axis is None else self._get_axis_from_axis_arg(axis)
        if axis == 0:
            if index is None:
                index = labels
        else:
            if columns is None:
                columns = labels
        if columns is None:
            df = self
        else:
            if (
                not isinstance(columns, (pd.Index, Index))
                and len(columns) == 0
            ):
                # pandas' Index.reindex treats an empty non-Index target as
                # ``columns[:0]``, preserving the columns' metadata.
                columns = self._data.to_pandas_index[:0]
            columns = Index(columns)
            intersection = self._data.to_pandas_index.intersection(
                columns.to_pandas()
            )
            df = self.loc[:, intersection]

        return df._reindex(
            column_names=columns,
            dtypes=dict(self._dtypes),
            deep=copy,
            index=index,
            inplace=False,
            fill_value=fill_value,
            level=level,
            method=method,
            limit=limit,
            tolerance=tolerance,
        )

    @_performance_tracking
    def set_index(
        self,
        keys,
        drop: bool = True,
        append: bool = False,
        inplace: bool = False,
        verify_integrity: bool = False,
    ) -> Self | None:
        """Return a new DataFrame with a new index

        Parameters
        ----------
        keys : Index, Series-convertible, label-like, or list
            Index : the new index.
            Series-convertible : values for the new index.
            Label-like : Label of column to be used as index.
            List : List of items from above.
        drop : boolean, default True
            Whether to drop corresponding column for str index argument
        append : boolean, default True
            Whether to append columns to the existing index,
            resulting in a MultiIndex.
        inplace : boolean, default False
            Modify the DataFrame in place (do not create a new object).
        verify_integrity : boolean, default False
            Check for duplicates in the new index.

        Examples
        --------
        >>> df = cudf.DataFrame({
        ...     "a": [1, 2, 3, 4, 5],
        ...     "b": ["a", "b", "c", "d","e"],
        ...     "c": [1.0, 2.0, 3.0, 4.0, 5.0]
        ... })
        >>> df
           a  b    c
        0  1  a  1.0
        1  2  b  2.0
        2  3  c  3.0
        3  4  d  4.0
        4  5  e  5.0

        Set the index to become the 'b' column:

        >>> df.set_index('b')
           a    c
        b
        a  1  1.0
        b  2  2.0
        c  3  3.0
        d  4  4.0
        e  5  5.0

        Create a MultiIndex using columns 'a' and 'b':

        >>> df.set_index(["a", "b"])
               c
        a b
        1 a  1.0
        2 b  2.0
        3 c  3.0
        4 d  4.0
        5 e  5.0

        Set new Index instance as index:

        >>> df.set_index(cudf.RangeIndex(10, 15))
            a  b    c
        10  1  a  1.0
        11  2  b  2.0
        12  3  c  3.0
        13  4  d  4.0
        14  5  e  5.0

        Setting `append=True` will combine current index with column `a`:

        >>> df.set_index("a", append=True)
             b    c
          a
        0 1  a  1.0
        1 2  b  2.0
        2 3  c  3.0
        3 4  d  4.0
        4 5  e  5.0

        `set_index` supports `inplace` parameter too:

        >>> df.set_index("a", inplace=True)
        >>> df
           b    c
        a
        1  a  1.0
        2  b  2.0
        3  c  3.0
        4  d  4.0
        5  e  5.0
        """

        if not isinstance(keys, list):
            keys = [keys]
        if len(keys) == 0:
            raise ValueError("No valid columns to be added to index.")
        if not isinstance(drop, bool):
            raise TypeError("drop must be a boolean")
        if not isinstance(append, bool):
            raise TypeError("append must be a boolean")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be a boolean")
        if inplace and not self._flags.allows_duplicate_labels:
            raise ValueError(
                "Cannot specify 'inplace=True' when "
                "'self.flags.allows_duplicate_labels' is False."
            )

        if append:
            keys = [self.index, *keys]

        # Preliminary type check
        labels_not_found = []
        data_to_add = []
        names = []
        to_drop = []
        for col in keys:
            # label-like
            if is_scalar(col) or isinstance(col, tuple):
                if col in self._column_names:
                    if drop and inplace:
                        data_to_add.append(self[col]._column)
                    else:
                        data_to_add.append(self[col]._column.copy(deep=True))
                    names.append(col)
                    if drop:
                        to_drop.append(col)
                else:
                    labels_not_found.append(col)
            # index-like
            elif isinstance(col, (MultiIndex, pd.MultiIndex)):
                if isinstance(col, pd.MultiIndex):
                    col = MultiIndex(
                        levels=col.levels, codes=col.codes, names=col.names
                    )
                data_to_add.extend(col._columns)
                names.extend(col.names)
            elif isinstance(col, (Series, Index, pd.Series, pd.Index)):
                data_to_add.append(as_column(col))
                names.append(col.name)
            else:
                try:
                    col = as_column(col)
                except TypeError as err:
                    msg = f"{col} cannot be converted to column-like."
                    raise TypeError(msg) from err
                data_to_add.append(col)
                names.append(None)

        if labels_not_found:
            raise KeyError(f"None of {labels_not_found} are in the columns")

        if (
            len(data_to_add) == 1
            and len(keys) == 1
            and not isinstance(keys[0], (cudf.MultiIndex, pd.MultiIndex))
        ):
            # Don't turn single level MultiIndex into an Index
            freq = (
                getattr(keys[0], "freq", None)
                if isinstance(keys[0], (cudf.Index, pd.Index))
                else None
            )
            if freq is not None and data_to_add[0].dtype.kind == "M":
                # Preserve the freq of a DatetimeIndex passed as the key.
                idx = cudf.DatetimeIndex._from_column(
                    data_to_add[0], name=names[0], freq=freq
                )
            else:
                idx = Index._from_column(data_to_add[0], name=names[0])
        else:
            idx = MultiIndex._from_data(dict(enumerate(data_to_add)))
            idx.names = names

        df = self if inplace else self.copy(deep=False)

        if verify_integrity and not idx.is_unique:
            raise ValueError(f"Values in Index are not unique: {idx}")

        if to_drop:
            df.drop(columns=to_drop, inplace=True)

        df.index = idx
        return df if not inplace else None

    @_performance_tracking
    def fillna(
        self,
        value,
        *,
        axis: Axis | None = None,
        inplace: bool = False,
        limit: int | None = None,
    ) -> Self | None:
        """Fill null values with ``value``.

        Parameters
        ----------
        value : scalar, Series-like or dict
            Value to use to fill nulls. If Series-like, null values
            are filled with values in corresponding indices.
            A dict can be used to provide different values to fill nulls
            in different columns.

        Returns
        -------
        result : DataFrame, Series, or Index
            Copy with nulls filled.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, None], 'b': [3, None, 5]})
        >>> df
              a     b
        0     1     3
        1     2  <NA>
        2  <NA>     5
        >>> df.fillna(4)
           a  b
        0  1  3
        1  2  4
        2  4  5
        >>> df.fillna({'a': 3, 'b': 4})
           a  b
        0  1  3
        1  2  4
        2  3  5

        ``fillna`` can also supports inplace operation:

        >>> df.fillna({'a': 3, 'b': 4}, inplace=True)
        >>> df
           a  b
        0  1  3
        1  2  4
        2  3  5
        """
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = from_pandas(value)
        if isinstance(value, Series):
            # Align value.index to self.columns
            value = value.reindex(self._column_names)
        elif isinstance(value, cudf.DataFrame):
            if not self.index.equals(value.index):
                # Align value.index to self.index
                value = value.reindex(self.index)
            value = dict(value.items())
        elif isinstance(value, Mapping):
            # Align value.indexes to self.index
            value = {
                key: value.reindex(self.index)
                if isinstance(value, Series)
                else value
                for key, value in value.items()
            }
        return super()._fillna(
            value=value, axis=axis, inplace=inplace, limit=limit
        )

    @_performance_tracking
    def where(
        self, cond, other=None, inplace: bool = False, axis=None, level=None
    ) -> Self | None:
        if axis is not None:
            raise NotImplementedError("axis is not supported.")
        elif level is not None:
            raise NotImplementedError("level is not supported.")

        # First process the condition.
        if isinstance(cond, Series):
            cond = self._from_data(
                self._data._from_columns_like_self(
                    itertools.repeat(cond._column, self._num_columns),
                    verify=False,
                ),
                index=cond.index,
            )
        elif hasattr(cond, "__cuda_array_interface__"):
            cond = DataFrame(
                cond, columns=self._column_names, index=self.index
            )
        elif (
            hasattr(cond, "__array_interface__")
            and cond.__array_interface__["shape"] != self.shape
        ):
            raise ValueError("conditional must be same shape as self")
        elif not isinstance(cond, DataFrame):
            cond = DataFrame(cond)

        if set(self._column_names).intersection(set(cond._column_names)):
            if not self.index.equals(cond.index):
                cond = cond.reindex(self.index)
        else:
            if cond.shape != self.shape:
                raise ValueError(
                    "Array conditional must be same shape as self"
                )
            # Setting `self` column names to `cond` as it has no column names.
            cond._set_columns_like(self._data)

        # If other was provided, process that next.
        if isinstance(other, DataFrame):
            other_cols = [other._data[col] for col in self._column_names]
        elif is_scalar(other):
            other_cols = [other] * self._num_columns
        elif isinstance(other, Series):
            other_cols = other.to_pandas()
        else:
            other_cols = other

        if self._num_columns != len(other_cols):
            raise ValueError(
                "other must contain the same number of columns or elements "
                f"as self ({self._num_columns})"
            )

        out = []
        for (name, col), other_col in zip(
            self._column_labels_and_values, other_cols, strict=True
        ):
            if cond_col := cond._data.get(name):
                out.append(col.where(cond_col, other_col, inplace))
            else:
                out.append(column_empty(len(col), dtype=col.dtype))

        return self._mimic_inplace(
            self._from_data_like_self(self._data._from_columns_like_self(out)),
            inplace=inplace,
        )

    @docutils.doc_apply(
        doc_reset_index_template.format(
            klass="DataFrame",
            argument="",
            return_type="DataFrame or None",
            return_doc="",
            example="""
        >>> df = cudf.DataFrame([('bird', 389.0),
        ...                    ('bird', 24.0),
        ...                    ('mammal', 80.5),
        ...                    ('mammal', np.nan)],
        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
        ...                   columns=('class', 'max_speed'))
        >>> df
                 class max_speed
        falcon    bird     389.0
        parrot    bird      24.0
        lion    mammal      80.5
        monkey  mammal       NaN
        >>> df.reset_index()
            index   class max_speed
        0  falcon    bird     389.0
        1  parrot    bird      24.0
        2    lion  mammal      80.5
        3  monkey  mammal       NaN
        >>> df.reset_index(drop=True)
            class max_speed
        0    bird     389.0
        1    bird      24.0
        2  mammal      80.5
        3  mammal       NaN

        You can also use ``reset_index`` with MultiIndex.

        >>> index = cudf.MultiIndex.from_tuples([('bird', 'falcon'),
        ...                                     ('bird', 'parrot'),
        ...                                     ('mammal', 'lion'),
        ...                                     ('mammal', 'monkey')],
        ...                                     names=['class', 'name'])
        >>> df = cudf.DataFrame([(389.0, 'fly'),
        ...                      ( 24.0, 'fly'),
        ...                      ( 80.5, 'run'),
        ...                      (np.nan, 'jump')],
        ...                      index=index,
        ...                      columns=('speed', 'type'))
        >>> df
                       speed  type
        class  name
        bird   falcon  389.0   fly
               parrot   24.0   fly
        mammal lion     80.5   run
               monkey    NaN  jump
        >>> df.reset_index(level='class')
                 class  speed  type
        name
        falcon    bird  389.0   fly
        parrot    bird   24.0   fly
        lion    mammal   80.5   run
        monkey  mammal    NaN  jump
        """,
        )
    )
    def reset_index(
        self,
        level=None,
        drop=False,
        inplace=False,
        col_level=0,
        col_fill="",
        allow_duplicates: bool = False,
        names: Hashable | Sequence[Hashable] | None = None,
    ):
        if inplace and not self._flags.allows_duplicate_labels:
            raise ValueError(
                "Cannot specify 'inplace=True' when "
                "'self.flags.allows_duplicate_labels' is False."
            )
        data, index = self._reset_index(
            level=level,
            drop=drop,
            col_level=col_level,
            col_fill=col_fill,
            allow_duplicates=allow_duplicates,
            names=names,
        )
        out = DataFrame._from_data(data=data, index=index)
        self._propagate_metadata(out)
        return self._mimic_inplace(out, inplace=inplace)

    @_external_only_api(
        "Use ._insert with ignore_index=True to avoid expensive index "
        "equality checking and reindexing when the data is already aligned."
    )
    @_performance_tracking
    def insert(
        self,
        loc,
        column,
        value,
        allow_duplicates: bool = False,
        nan_as_null=no_default,
    ):
        """Add a column to DataFrame at the index specified by loc.

        Parameters
        ----------
        loc : int
            location to insert by index, cannot be greater then num columns + 1
        column : number or string
            column or label of column to be inserted
        value : Series or array-like
        nan_as_null : bool, Default None
            If ``None``/``True``, converts ``np.nan`` values to
            ``null`` values.
            If ``False``, leaves ``np.nan`` values as is.
        """
        if allow_duplicates is not False:
            raise NotImplementedError(
                "allow_duplicates is currently not implemented."
            )
        if nan_as_null is no_default:
            nan_as_null = not get_option("mode.pandas_compatible")
        return self._insert(
            loc=loc,
            name=column,
            value=value,
            nan_as_null=nan_as_null,
            ignore_index=False,
        )

    @_performance_tracking
    def _insert(self, loc, name, value, nan_as_null=None, ignore_index=True):
        """
        Same as `insert`, with additional `ignore_index` param.

        ignore_index : bool, default True
            If True, there will be no index equality check & reindexing
            happening.
            If False, a reindexing operation is performed if
            `value.index` is not equal to `self.index`.
        """
        num_cols = self._num_columns
        if loc < 0:
            loc += num_cols + 1

        if not (0 <= loc <= num_cols):
            raise ValueError(
                f"insert location must be within range "
                f"{-(num_cols + 1) * (num_cols > 0)}, "
                f"{num_cols * (num_cols > 0)}"
            )

        # TODO: This check is currently necessary because
        # _is_scalar_or_zero_d_array below will treat a length 1 pd.Categorical
        # as a scalar and attempt to use column.full, which can't handle it.
        # Maybe _is_scalar_or_zero_d_array should be changed, or maybe we just
        # shouldn't support pd.Categorical at all, but those changes will at
        # least require a deprecation cycle because we currently support
        # inserting a pd.Categorical.
        if isinstance(value, pd.Categorical):
            value = as_column(value)

        if _is_scalar_or_zero_d_array(value):
            # TODO: as_column should be able to handle these outputs outright
            dtype = None
            if isinstance(value, (np.ndarray, cupy.ndarray)):
                dtype = value.dtype
                if dtype.kind == "U":
                    dtype = DEFAULT_STRING_DTYPE
                value = value.item()
                if dtype.kind == "O" and isinstance(value, str):
                    dtype = DEFAULT_STRING_DTYPE
            if is_na_like(value):
                if isinstance(value, (np.timedelta64, np.datetime64)):
                    # Fall back to "ns" for unsupported units
                    # (e.g. ``np.timedelta64("NaT")``), matching pandas.
                    unit, _ = np.datetime_data(value.dtype)
                    if unit not in {"s", "ms", "us", "ns"}:
                        kind = (
                            "timedelta64"
                            if isinstance(value, np.timedelta64)
                            else "datetime64"
                        )
                        dtype = np.dtype(f"{kind}[ns]")
                    else:
                        dtype = value.dtype
                else:
                    dtype = DEFAULT_STRING_DTYPE
            value = as_column(
                value,
                length=len(self),
                dtype=dtype,
            )

        if len(self) == 0:
            if isinstance(value, (pd.Series, Series)):
                if not ignore_index:
                    self.index = Index(value.index)
            elif (length := len(value)) > 0:
                if num_cols != 0:
                    ca = self._data._from_columns_like_self(
                        (
                            column_empty(row_count=length, dtype=dtype)
                            for _, dtype in self._dtypes
                        ),
                        verify=False,
                    )
                else:
                    ca = ColumnAccessor({})
                self._data = ca
                self._index = RangeIndex(length)

        elif isinstance(value, (pd.Series, Series)):
            value = Series(
                value, nan_as_null=nan_as_null, copy=isinstance(value, Series)
            )
            if not ignore_index:
                value = value._align_to_index(
                    self.index, how="right", sort=False
                )

        if isinstance(value, (list, tuple)) and len(value) == 0:
            # An empty list-like assigned as a DataFrame column has no
            # inferable dtype and becomes float64, matching pandas (note
            # pd.Series([]) is object, but DataFrame column assignment is
            # float64).
            value = as_column(
                value, nan_as_null=nan_as_null, dtype=np.dtype(np.float64)
            )
        else:
            value = as_column(value, nan_as_null=nan_as_null)
        self._data.insert(name, value, loc=loc)

    @property
    @_performance_tracking
    def axes(self):
        """
        Return a list representing the axes of the DataFrame.

        DataFrame.axes returns a list of two elements:
        element zero is the row index and element one is the columns.

        Examples
        --------
        >>> import cudf
        >>> cdf1 = cudf.DataFrame()
        >>> cdf1["key"] = [0,0,1,1]
        >>> cdf1["k2"] = [1,2,2,3]
        >>> cdf1["val"] = [1,2,3,4]
        >>> cdf1["temp"] = [-1,2,2,3]
        >>> cdf1.axes
        [RangeIndex(start=0, stop=4, step=1), Index(['key', 'k2', 'val', 'temp'], dtype='str')]

        """
        return [self.index, self._data.to_pandas_index]

    def diff(
        self, periods: int = 1, axis: int | Literal["index", "columns"] = 0
    ) -> Self:
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another
        element in the DataFrame (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference,
            accepts negative values.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Take difference over rows (0) or columns (1).
            Only row-wise (0) shift is supported.

        Returns
        -------
        DataFrame
            First differences of the DataFrame.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                       'b': [1, 1, 2, 3, 5, 8],
        ...                       'c': [1, 4, 9, 16, 25, 36]})
        >>> gdf
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36
        >>> gdf.diff(periods=2)
              a     b     c
        0  <NA>  <NA>  <NA>
        1  <NA>  <NA>  <NA>
        2     2     1     8
        3     2     2    12
        4     2     3    16
        5     2     5    20

        .. pandas-compat::
            :meth:`pandas.DataFrame.diff`

            Diff currently only supports numeric dtype columns.
        """
        if not isinstance(periods, int):
            if not (isinstance(periods, float) and periods.is_integer()):
                raise ValueError("periods must be an integer")
            periods = int(periods)

        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise NotImplementedError("Only axis=0 is supported.")

        if abs(periods) > len(self):
            return self._from_data_like_self(
                self._data._from_columns_like_self(
                    (
                        column_empty(len(self), dtype=dtype)
                        for _, dtype in self._dtypes
                    ),
                    verify=False,
                )
            )
        return self - self.shift(periods=periods)

    @_performance_tracking
    def drop_duplicates(
        self,
        subset=None,
        keep: Literal["first", "last", False] = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Self | None:
        """
        Return DataFrame with duplicate rows removed.

        Considering certain columns is optional. Indexes, including time
        indexes are ignored.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', ``False``}, default 'first'
            Determines which duplicates (if any) to keep.
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        inplace : bool, default ``False``
            Whether to drop duplicates in place or to return a copy.
        ignore_index : bool, default ``False``
            If True, the resulting axis will be labeled 0, 1, ..., n - 1.

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed or None if ``inplace=True``.

        See Also
        --------
        DataFrame.value_counts: Count unique combinations of columns.

        Examples
        --------
        Consider a dataset containing ramen ratings.

        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
        ...     'rating': [4, 4, 3.5, 15, 5]
        ... })
        >>> df
             brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        By default, it removes duplicate rows based on all columns.

        >>> df.drop_duplicates()
             brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        To remove duplicates on specific column(s), use ``subset``.

        >>> df.drop_duplicates(subset=['brand'])
             brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5

        To remove duplicates and keep last occurrences, use ``keep``.

        >>> df.drop_duplicates(subset=['brand', 'style'], keep='last')
             brand style  rating
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        4  Indomie  pack     5.0
        """
        outdf = super().drop_duplicates(
            subset=subset,
            keep=keep,
            ignore_index=ignore_index,
        )

        return self._mimic_inplace(outdf, inplace=inplace)

    @_performance_tracking
    def pop(self, item):
        """Return a column and drop it from the DataFrame."""
        popped = self[item]
        del self[item]
        return popped

    @_performance_tracking
    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=0,
        copy=True,
        inplace=False,
        level=None,
        errors="ignore",
    ):
        """Alter column and index labels.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don't throw an
        error.

        ``DataFrame.rename`` supports two calling conventions:
            - ``(index=index_mapper, columns=columns_mapper, ...)``
            - ``(mapper, axis={0/'index' or 1/'column'}, ...)``

        We highly recommend using keyword arguments to clarify your intent.

        Parameters
        ----------
        mapper : dict-like or function, default None
            optional dict-like or functions transformations to apply to
            the index/column values depending on selected ``axis``.
        index : dict-like, default None
            Optional dict-like transformations to apply to the index axis'
            values. Does not support functions for axis 0 yet.
        columns : dict-like or function, default None
            optional dict-like or functions transformations to apply to
            the columns axis' values.
        axis : int, default 0
            Axis to rename with mapper.
            0 or 'index' for index
            1  or 'columns' for columns
        copy : boolean, default True
            Also copy underlying data
        inplace : boolean, default False
            Return new DataFrame.  If True, assign columns without copy
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified level.
        errors : {'raise', 'ignore', 'warn'}, default 'ignore'
            *Only 'ignore' supported*
            Control raising of exceptions on invalid data for provided dtype.

            -   ``raise`` : allow exceptions to be raised
            -   ``ignore`` : suppress exceptions. On error return original
                object.
            -   ``warn`` : prints last exceptions as warnings and
                return original object.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> df
           A  B
        0  1  4
        1  2  5
        2  3  6

        Rename columns using a mapping:

        >>> df.rename(columns={"A": "a", "B": "c"})
           a  c
        0  1  4
        1  2  5
        2  3  6

        Rename index using a mapping:

        >>> df.rename(index={0: 10, 1: 20, 2: 30})
            A  B
        10  1  4
        20  2  5
        30  3  6

        .. pandas-compat::
            :meth:`pandas.DataFrame.rename`

            * Not Supporting: level

            Rename will not overwrite column names. If a list with
            duplicates is passed, column names will be postfixed
            with a number.
        """
        if errors != "ignore":
            raise NotImplementedError(
                "Only errors='ignore' is currently supported"
            )

        if mapper is None and index is None and columns is None:
            return self.copy(deep=copy)

        index = mapper if index is None and axis in (0, "index") else index
        columns = (
            mapper if columns is None and axis in (1, "columns") else columns
        )

        result = self if inplace else self.copy(deep=copy)

        out_index = None
        if index:
            if (
                any(isinstance(item, str) for item in index.values())
                and self.index.dtype != "object"
            ):
                raise NotImplementedError(
                    "Implicit conversion of index to "
                    "mixed type is not yet supported."
                )

            if level is not None and isinstance(self.index, MultiIndex):
                # Resolve to the ColumnAccessor label (e.g. the positional key
                # ``0`` for an unnamed level), NOT the level *name* (which is
                # ``None`` for unnamed levels and would insert a spurious extra
                # column rather than overwrite the level being renamed).
                ca_label, _ = self.index._level_to_ca_label(level)
                level_values = self.index.get_level_values(level)
                ca = self.index._data.copy(deep=copy)
                vals = list(index.values())
                is_all_na = all(val is None for val in vals)
                ca[ca_label] = level_values._column.find_and_replace(
                    to_replace=list(index.keys()),
                    replacement=vals,
                    all_nan=is_all_na,
                )
                out_index = type(self.index)._from_data(
                    ca, name=self.index.name
                )
                # ``_from_data`` derives level names from the ColumnAccessor
                # keys (positional ints for an unnamed MultiIndex); restore the
                # original level names so renaming a level does not rename it.
                out_index.names = self.index.names
            else:
                to_replace = list(index.keys())
                vals = list(index.values())
                is_all_na = all(val is None for val in vals)

                try:
                    out_index = _index_from_data(
                        {
                            name: col.find_and_replace(
                                to_replace, vals, is_all_na
                            )
                            for name, col in self.index._column_labels_and_values
                        }
                    )
                except OverflowError:
                    pass

        if out_index is not None:
            result.index = out_index

        if columns:
            new_ca = result._data.rename_levels(mapper=columns, level=level)
            # pandas' rename rebuilds the columns Index from the transformed
            # labels (``Index(items, tupleize_cols=False)`` in
            # ``_transform_index``), re-inferring dtypes rather than
            # preserving the originals: renaming object-dtype columns to
            # all-string labels yields ``str``, and MultiIndex level dtypes
            # are likewise re-inferred.
            if new_ca.multiindex:
                new_ca._level_dtypes = None
            else:
                new_ca.label_dtype = pd.Index(
                    new_ca.names, tupleize_cols=False
                ).dtype
            result._data = new_ca

        return result

    @_performance_tracking
    def add_prefix(self, prefix, axis=None):
        if axis is not None:
            raise NotImplementedError("axis is currently not implemented.")
        out = self.copy(deep=False)
        out.columns = [prefix + col_name for col_name in self._column_names]
        return out

    @_performance_tracking
    def add_suffix(self, suffix, axis=None):
        if axis is not None:
            raise NotImplementedError("axis is currently not implemented.")
        out = self.copy(deep=False)
        out.columns = [col_name + suffix for col_name in self._column_names]
        return out

    @_performance_tracking
    def agg(self, aggs, axis=None):
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        aggs : Iterable (set, list, string, tuple or dict)
            Function to use for aggregating data. Accepted types are:
             * string name, e.g. ``"sum"``
             * list of functions, e.g. ``["sum", "min", "max"]``
             * dict of axis labels specified operations per column,
               e.g. ``{"a": "sum"}``

        axis : not yet supported

        Returns
        -------
        Aggregation Result : ``Series`` or ``DataFrame``
            When ``DataFrame.agg`` is called with single agg,
            ``Series`` is returned.
            When ``DataFrame.agg`` is called with several aggs,
            ``DataFrame`` is returned.

        .. pandas-compat::
            :meth:`pandas.DataFrame.agg`

            * Not supporting: ``axis``, ``*args``, ``**kwargs``

        """
        dtypes = [dtype for _, dtype in self._dtypes]
        common_dtype = find_common_type(dtypes)
        if common_dtype.kind != "b" and any(
            dtype.kind == "b" for dtype in dtypes
        ):
            raise MixedTypeError("Cannot create a column with mixed types")

        if any(is_dtype_obj_string(dt) for dt in dtypes):
            raise NotImplementedError(
                "DataFrame.agg() is not supported for "
                "frames containing string columns"
            )

        if axis == 0 or axis is not None:
            raise NotImplementedError("axis not implemented yet")

        if isinstance(aggs, Iterable) and not isinstance(aggs, (str, dict)):
            result = DataFrame()
            # TODO : Allow simultaneous pass for multi-aggregation as
            # a future optimization
            for agg in aggs:
                result[agg] = getattr(self, agg)()
            return result.T.sort_index(axis=1, ascending=True)

        elif isinstance(aggs, str):
            if not hasattr(self, aggs):
                raise AttributeError(
                    f"{aggs} is not a valid function for 'DataFrame' object"
                )
            result = DataFrame()
            result[aggs] = getattr(self, aggs)()
            result = result.iloc[:, 0]
            result.name = None
            return result

        elif isinstance(aggs, dict):
            cols = aggs.keys()
            if any(callable(val) for val in aggs.values()):
                raise NotImplementedError(
                    "callable parameter is not implemented yet"
                )
            elif all(isinstance(val, str) for val in aggs.values()):
                res = {}
                for key, value in aggs.items():
                    col = self[key]
                    if not hasattr(col, value):
                        raise AttributeError(
                            f"{value} is not a valid function for "
                            f"'Series' object"
                        )
                    res[key] = getattr(col, value)()
                result = Series(list(res.values()), index=res.keys())
            elif all(isinstance(val, Iterable) for val in aggs.values()):
                idxs = set()
                for val in aggs.values():
                    if isinstance(val, str):
                        idxs.add(val)
                    elif isinstance(val, Iterable):
                        idxs.update(val)
                idxs = sorted(list(idxs))
                for agg in idxs:
                    if agg is callable:
                        raise NotImplementedError(
                            "callable parameter is not implemented yet"
                        )
                result = DataFrame(index=idxs, columns=cols)
                for key in aggs.keys():
                    col = self[key]
                    col_empty = column_empty(len(idxs), dtype=col.dtype)
                    ans = Series._from_column(col_empty, index=Index(idxs))
                    if isinstance(aggs.get(key), Iterable):
                        # TODO : Allow simultaneous pass for multi-aggregation
                        # as a future optimization
                        for agg in aggs.get(key):
                            if not hasattr(col, agg):
                                raise AttributeError(
                                    f"{agg} is not a valid function for "
                                    f"'Series' object"
                                )
                            ans[agg] = getattr(col, agg)()
                    elif isinstance(aggs.get(key), str):
                        if not hasattr(col, aggs.get(key)):
                            raise AttributeError(
                                f"{aggs.get(key)} is not a valid function for "
                                f"'Series' object"
                            )
                        ans[aggs.get(key)] = getattr(col, agg)()
                    result[key] = ans
            else:
                raise ValueError("values of dict must be a string or list")

            return result

        elif callable(aggs):
            raise NotImplementedError(
                "callable parameter is not implemented yet"
            )

        else:
            raise ValueError("argument must be a string, list or dict")

    @_performance_tracking
    def nlargest(self, n, columns, keep="first"):
        """Return the first *n* rows ordered by *columns* in descending order.

        Return the first *n* rows with the largest values in *columns*, in
        descending order. The columns that are not specified are returned as
        well, but not used for ordering.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : label or list of labels
            Column label(s) to order by.
        keep : {'first', 'last'}, default 'first'
            Where there are duplicate values:

            - `first` : prioritize the first occurrence(s)
            - `last` : prioritize the last occurrence(s)

        Returns
        -------
        DataFrame
            The first `n` rows ordered by the given columns in descending
            order.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'population': [59000000, 65000000, 434000,
        ...                                   434000, 434000, 337000, 11300,
        ...                                   11300, 11300],
        ...                    'GDP': [1937894, 2583560 , 12011, 4520, 12128,
        ...                            17036, 182, 38, 311],
        ...                    'alpha-2': ["IT", "FR", "MT", "MV", "BN",
        ...                                "IS", "NR", "TV", "AI"]},
        ...                   index=["Italy", "France", "Malta",
        ...                          "Maldives", "Brunei", "Iceland",
        ...                          "Nauru", "Tuvalu", "Anguilla"])
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru          11300      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI
        >>> df.nlargest(3, 'population')
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Malta       434000    12011      MT
        >>> df.nlargest(3, 'population', keep='last')
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Brunei      434000    12128      BN

        .. pandas-compat::
            :meth:`pandas.DataFrame.nlargest`

            - Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest(True, n, columns, keep)

    def nsmallest(self, n, columns, keep="first"):
        """Return the first *n* rows ordered by *columns* in ascending order.

        Return the first *n* rows with the smallest values in *columns*, in
        ascending order. The columns that are not specified are returned as
        well, but not used for ordering.

        Parameters
        ----------
        n : int
            Number of items to retrieve.
        columns : list or str
            Column name or names to order by.
        keep : {'first', 'last'}, default 'first'
            Where there are duplicate values:

            - ``first`` : take the first occurrence.
            - ``last`` : take the last occurrence.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'population': [59000000, 65000000, 434000,
        ...                                   434000, 434000, 337000, 337000,
        ...                                   11300, 11300],
        ...                    'GDP': [1937894, 2583560 , 12011, 4520, 12128,
        ...                            17036, 182, 38, 311],
        ...                    'alpha-2': ["IT", "FR", "MT", "MV", "BN",
        ...                                "IS", "NR", "TV", "AI"]},
        ...                   index=["Italy", "France", "Malta",
        ...                          "Maldives", "Brunei", "Iceland",
        ...                          "Nauru", "Tuvalu", "Anguilla"])
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru         337000      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI

        In the following example, we will use ``nsmallest`` to select the
        three rows having the smallest values in column "population".

        >>> df.nsmallest(3, 'population')
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS

        When using ``keep='last'``, ties are resolved in reverse order:

        >>> df.nsmallest(3, 'population', keep='last')
                  population  GDP alpha-2
        Anguilla       11300  311      AI
        Tuvalu         11300   38      TV
        Nauru         337000  182      NR

        .. pandas-compat::
            :meth:`pandas.DataFrame.nsmallest`

            - Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest(False, n, columns, keep)

    @_performance_tracking
    def swaplevel(self, i=-2, j=-1, axis=0):
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int or str, default -2
            First level of index to be swapped.
        j : int or str, default -1
            Second level of index to be swapped.
        axis : The axis to swap levels on.
            0 or 'index' for row-wise, 1 or 'columns' for column-wise.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(levels=[['llama', 'cow', 'falcon'],
        ...   ['speed', 'weight', 'length'],['first','second']],
        ...   codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2],
        ...             [0, 0, 0, 0, 0, 0, 1, 1, 1]])
        >>> cdf = cudf.DataFrame(index=midx, columns=['big', 'small'],
        ...  data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
        ...         [250, 150], [1.5, 0.8], [320, 250], [1, 0.8], [0.3, 0.2]])

        >>> cdf
                                     big  small
             llama  speed  first    45.0   30.0
                    weight first   200.0  100.0
                    length first     1.5    1.0
             cow    speed  first    30.0   20.0
                    weight first   250.0  150.0
                    length first     1.5    0.8
             falcon speed  second  320.0  250.0
                    weight second    1.0    0.8
                    length second    0.3    0.2

        >>> cdf.swaplevel()
                                     big  small
             llama  first  speed    45.0   30.0
                           weight  200.0  100.0
                           length    1.5    1.0
             cow    first  speed    30.0   20.0
                           weight  250.0  150.0
                           length    1.5    0.8
             falcon second speed   320.0  250.0
                           weight    1.0    0.8
                           length    0.3    0.2
        """
        result = self.copy(deep=False)

        # To get axis number
        axis = self._get_axis_from_axis_arg(axis)

        if axis == 0:
            if not isinstance(result.index, MultiIndex):
                raise TypeError("Can only swap levels on a hierarchical axis.")
            result.index = result.index.swaplevel(i, j)
        else:
            if not result._data.multiindex:
                raise TypeError("Can only swap levels on a hierarchical axis.")
            result._data = result._data.swaplevel(i, j)

        return result

    @_performance_tracking
    def transpose(self) -> Self:
        """Transpose index and columns.

        Returns
        -------
        a new (ncol x nrow) dataframe. self is (nrow x ncol)

        .. pandas-compat::
            :meth:`pandas.DataFrame.transpose`, :attr:`pandas.DataFrame.T`

            Not supporting *copy* because default and only behavior is
            copy=True
        """
        index = self._data.to_pandas_index
        if not isinstance(index, pd.MultiIndex) and index.dtype == np.dtype(
            np.object_
        ):
            # Potentially convert mixed objects to strings
            index = index.astype(str)
        if self._num_columns == 0 or self._num_rows == 0:
            return type(self)(index=index, columns=self.index)

        # No column from index is transposed with libcudf.
        source_columns = self._columns
        source_dtype = source_columns[0].dtype
        if isinstance(source_dtype, CategoricalDtype):
            if any(
                not isinstance(c.dtype, CategoricalDtype)
                for c in source_columns
            ):
                raise ValueError("Columns must all have the same dtype")
            cats = concat_columns(
                [c.categories for c in source_columns]  # type: ignore[attr-defined]
            ).unique()
            source_columns = [  # type: ignore[assignment]
                col._set_categories(cats, is_unique=True).codes  # type: ignore[attr-defined]
                for col in source_columns
            ]
            # TODO: Do we need to pass ordered=source_dtype.ordered as well?
            source_dtype = CategoricalDtype(categories=cats)
        elif any(col.dtype != source_dtype for col in source_columns):
            raise ValueError("Columns must all have the same dtype")

        with access_columns(
            *source_columns, mode="read", scope="internal"
        ) as source_columns:
            result_table = plc.transpose.transpose(
                plc.table.Table([col.plc_column for col in source_columns])
            )
            result_columns = (
                ColumnBase.create(col, source_dtype)
                for col in result_table.columns()
            )

        # Set the old column names as the new index
        result = type(self)._from_data(
            ColumnAccessor(dict(enumerate(result_columns)), verify=False),
            index=Index(index),
            attrs=self.attrs,
        )
        # Set the old index as the new column names
        result.columns = self.index
        return result

    T = property(transpose, doc=transpose.__doc__)

    @_performance_tracking
    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
        ignore_index: bool = True,
    ):
        """Unpivots a DataFrame from wide format to long format,
        optionally leaving identifier variables set.

        Parameters
        ----------
        frame : DataFrame
        id_vars : tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
            default: None
        value_vars : tuple, list, or ndarray, optional
            Column(s) to unpivot.
            default: all columns that are not set as `id_vars`.
        var_name : scalar
            Name to use for the `variable` column.
            default: frame.columns.name or 'variable'
        value_name : str
            Name to use for the `value` column.
            default: 'value'

        Returns
        -------
        out : DataFrame
            Melted result
        """
        return reshape.melt(
            self,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
            col_level=col_level,
            ignore_index=ignore_index,
        )

    @_performance_tracking
    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        indicator=False,
        validate=None,
    ):
        """Merge GPU DataFrame objects by performing a database-style join
        operation by columns or indexes.

        Parameters
        ----------
        right : DataFrame
        on : label or list; defaults to None
            Column or index level names to join on. These must be found in
            both DataFrames.

            If on is None and not merging on indexes then
            this defaults to the intersection of the columns
            in both DataFrames.
        how : {'left', 'right', 'outer', 'inner', 'cross', 'leftsemi', 'leftanti'}, \
            default 'inner'
            Type of merge to be performed.

            - left : use only keys from left frame, similar to a SQL left
              outer join.
            - right : not supported.
            - outer : use union of keys from both frames, similar to a SQL
              full outer join.
            - inner : use intersection of keys from both frames, similar to
              a SQL inner join.
            - cross: creates the cartesian product from both frames, preserves the order
              of the left keys.
            - leftsemi : similar to ``inner`` join, but only returns columns
               from the left dataframe and ignores all columns from the
               right dataframe.
            - leftanti : returns only rows columns from the left dataframe
              for non-matched records. This is exact opposite to ``leftsemi``
              join.
        left_on : label or list, or array-like
            Column or index level names to join on in the left DataFrame.
            Can also be an array or list of arrays of the length of the
            left DataFrame. These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame.
            Can also be an array or list of arrays of the length of the
            right DataFrame. These arrays are treated as if they are columns.
        left_index : bool, default False
            Use the index from the left DataFrame as the join key(s).
        right_index : bool, default False
            Use the index from the right DataFrame as the join key.
        sort : bool, default False
            Sort the resulting dataframe by the columns that were merged on,
            starting from the left.
        suffixes: Tuple[str, str], defaults to ('_x', '_y')
            Suffixes applied to overlapping column names on the left and right
            sides

        Returns
        -------
            merged : DataFrame

        Examples
        --------
        >>> import cudf
        >>> df_a = cudf.DataFrame()
        >>> df_a['key'] = [0, 1, 2, 3, 4]
        >>> df_a['vals_a'] = [float(i + 10) for i in range(5)]
        >>> df_b = cudf.DataFrame()
        >>> df_b['key'] = [1, 2, 4]
        >>> df_b['vals_b'] = [float(i+10) for i in range(3)]
        >>> df_merged = df_a.merge(df_b, on=['key'], how='left')
        >>> df_merged.sort_values('key')  # doctest: +SKIP
           key  vals_a  vals_b
        3    0    10.0
        0    1    11.0    10.0
        1    2    12.0    11.0
        4    3    13.0
        2    4    14.0    12.0

        **Merging on categorical variables is only allowed in certain cases**

        Categorical variable typecasting logic depends on both `how`
        and the specifics of the categorical variables to be merged.
        Merging categorical variables when only one side is ordered
        is ambiguous and not allowed. Merging when both categoricals
        are ordered is allowed, but only when the categories are
        exactly equal and have equal ordering, and will result in the
        common dtype.
        When both sides are unordered, the result categorical depends
        on the kind of join:
        - For inner joins, the result will be the intersection of the
        categories
        - For left or right joins, the result will be the left or
        right dtype respectively. This extends to semi and anti joins.
        - For outer joins, the result will be the union of categories
        from both sides.

        .. pandas-compat::
            :meth:`pandas.DataFrame.merge`

            DataFrames merges in cuDF result in non-deterministic row
            ordering.
        """
        if indicator:
            raise NotImplementedError(
                "Only indicator=False is currently supported"
            )
        if validate is not None:
            raise NotImplementedError("validate is currently not supported.")

        lhs, rhs = self, right
        merge_cls = Merge
        is_right_join = how == "right"
        if is_right_join:
            # Merge doesn't support right, so just swap.
            # Capture original key params before swap so we can restore
            # the expected column order (self columns first) after the merge.
            orig_on = on
            orig_left_on = left_on
            orig_right_on = right_on
            orig_left_index = left_index
            orig_right_index = right_index
            how = "left"
            lhs, rhs = right, self
            left_on, right_on = right_on, left_on
            left_index, right_index = right_index, left_index
            suffixes = (suffixes[1], suffixes[0])
            if (
                orig_on is None
                and orig_left_on is None
                and orig_right_on is None
                and not orig_left_index
                and not orig_right_index
            ):
                # Merge infers the common key columns from its (post-swap)
                # left frame, but pandas keys an inferred merge by the
                # *original* left frame's column order regardless of ``how``.
                # Pass the keys explicitly to preserve that order (it decides
                # both the key column order and the sort priority).
                right_names = set(right._column_names)
                inferred_on = [
                    name for name in self._column_names if name in right_names
                ]
                if inferred_on:
                    on = inferred_on
        elif how in {"leftsemi", "leftanti"}:
            merge_cls = MergeSemi

        result = merge_cls(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            indicator=indicator,
            suffixes=suffixes,
        ).perform_merge()

        if is_right_join:
            # After the swap, result columns are ordered as:
            #   [key_cols (k), right_non_key_cols, self_non_key_cols]
            # Pandas expects:
            #   [key_cols (k), self_non_key_cols, right_non_key_cols]
            # Reorder using integer positions to avoid issues with None/NaN
            # column names: new positions = [0..k-1, N_r..n-1, k..N_r-1]
            N_r = right._num_columns
            n_result = result._num_columns
            if orig_on is not None:
                # `on` names data columns shared by both sides; they appear once.
                k = 1 if is_scalar(orig_on) else len(orig_on)
            elif (
                orig_left_on is not None
                and orig_right_on is not None
                and not orig_left_index
                and not orig_right_index
            ):
                # Count same-name (lk, rk) key-column pairs.
                orig_left_on_list = (
                    [orig_left_on]
                    if is_scalar(orig_left_on)
                    else list(orig_left_on)
                )
                orig_right_on_list = (
                    [orig_right_on]
                    if is_scalar(orig_right_on)
                    else list(orig_right_on)
                )
                k = sum(
                    lk == rk
                    for lk, rk in zip(
                        orig_left_on_list, orig_right_on_list, strict=True
                    )
                )
            elif (
                not orig_left_on
                and not orig_right_on
                and not orig_left_index
                and not orig_right_index
            ):
                # Auto-detected keys sit wherever they appear within each
                # frame, so a segment swap cannot reproduce pandas' layout
                # (the original left frame's columns in their own order,
                # then the right frame's non-key columns). No suffixing can
                # occur here -- any label shared by both frames is a key --
                # so the result labels are unchanged and a label-based
                # reorder is exact.
                common = set(self._column_names) & set(right._column_names)
                expected = list(self._column_names) + [
                    name for name in right._column_names if name not in common
                ]
                positions = {
                    label: i for i, label in enumerate(result._column_names)
                }
                if len(expected) == n_result and all(
                    label in positions for label in expected
                ):
                    result = result.iloc[
                        :, [positions[label] for label in expected]
                    ]
                # skip the positional segment swap below
                k = n_result
            else:
                k = 0
            # Only reorder when there are both right non-key cols and self
            # non-key cols to swap; use integer positions to avoid NaN issues.
            if k < N_r < n_result:
                new_indices = (
                    list(range(k))
                    + list(range(N_r, n_result))
                    + list(range(k, N_r))
                )
                result = result.iloc[:, new_indices]

        if how != "cross":
            orig_how = "right" if is_right_join else how
            if is_right_join:
                key_on, key_lon, key_ron = orig_on, orig_left_on, orig_right_on
                key_li, key_ri = orig_left_index, orig_right_index
            else:
                key_on, key_lon, key_ron = on, left_on, right_on
                key_li, key_ri = left_index, right_index

            def _restore_key_dtype(name, target, right_dtype):
                # Restore a result key column to ``target`` to match pandas.
                if name not in result._data:
                    return
                # Categorical keys follow the (de)categorization rules applied
                # during the join itself.
                if isinstance(target, CategoricalDtype) or isinstance(
                    right_dtype, CategoricalDtype
                ):
                    return
                if result._data[name].dtype == target:
                    return
                # Do not undo the numpy int -> float64 upcast that unmatched
                # rows require.
                if (
                    isinstance(target, np.dtype)
                    and result._data[name].null_count
                ):
                    return
                result[name] = result[name].astype(target)

            def _keep_left_dtype(target, right_dtype):
                # pandas presents a shared-name key with the LEFT dtype when an
                # extension dtype is involved (all joins) or for inner/left
                # joins; right/outer numpy keys take the common type.
                one_extension = (not isinstance(target, np.dtype)) or (
                    right_dtype is not None
                    and not isinstance(right_dtype, np.dtype)
                )
                return one_extension or orig_how in {
                    "inner",
                    "left",
                    "leftsemi",
                    "leftanti",
                }

            if not key_li and not key_ri:
                if key_lon is not None and key_ron is not None:
                    lon = [key_lon] if is_scalar(key_lon) else list(key_lon)
                    ron = [key_ron] if is_scalar(key_ron) else list(key_ron)
                    for lk, rk in zip(lon, ron, strict=True):
                        if lk == rk:
                            if lk in self._data:
                                target = self._data[lk].dtype
                                rd = (
                                    right._data[lk].dtype
                                    if lk in right._data
                                    else None
                                )
                                if _keep_left_dtype(target, rd):
                                    _restore_key_dtype(lk, target, rd)
                        else:
                            # Differently-named keys both survive, each with
                            # its own operand's dtype.
                            if lk in self._data:
                                _restore_key_dtype(
                                    lk, self._data[lk].dtype, None
                                )
                            if rk in right._data:
                                _restore_key_dtype(
                                    rk, right._data[rk].dtype, None
                                )
                else:
                    if key_on is not None:
                        key_names = (
                            [key_on] if is_scalar(key_on) else list(key_on)
                        )
                    else:
                        key_names = list(
                            set(self._column_names) & set(right._column_names)
                        )
                    for name in key_names:
                        if name not in self._data:
                            continue
                        target = self._data[name].dtype
                        rd = (
                            right._data[name].dtype
                            if name in right._data
                            else None
                        )
                        if _keep_left_dtype(target, rd):
                            _restore_key_dtype(name, target, rd)

        return result

    @_performance_tracking
    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
        validate: str | None = None,
    ):
        """Join columns with other DataFrame on index or on a key column.

        Parameters
        ----------
        other : DataFrame
        how : {'left', 'right', 'outer', 'inner', 'cross'}, default 'left'
            How to handle the operation of the two objects.

            * left: use calling frame's index (or column if on is specified)
            * right: use `other`'s index.
            * outer: form union of calling frame's index (or column if on is
              specified) with `other`'s index, and sort it lexicographically.
            * inner: form intersection of calling frame's index (or column if
              on is specified) with `other`'s index, preserving the order
              of the calling's one.
            * cross: creates the cartesian product from both frames, preserves the order
              of the left keys.
        lsuffix, rsuffix : str
            The suffices to add to the left (*lsuffix*) and right (*rsuffix*)
            column names when avoiding conflicts.
        sort : bool
            Set to True to ensure sorted ordering.
        validate : str, optional
            If specified, checks if join is of specified type.

            * "one_to_one" or "1:1": check if join keys are unique in both left
              and right datasets.
            * "one_to_many" or "1:m": check if join keys are unique in left dataset.
            * "many_to_one" or "m:1": check if join keys are unique in right dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.

            Currently not supported.

        Returns
        -------
        joined : DataFrame

        .. pandas-compat::
            :meth:`pandas.DataFrame.join`

            - *other* must be a single DataFrame for now.
            - *on* is not supported yet due to lack of multi-index support.
        """
        if on is not None:
            raise NotImplementedError("The on parameter is not yet supported")
        elif validate is not None:
            raise NotImplementedError(
                "The validate parameter is not yet supported"
            )

        df = self.merge(
            other,
            left_index=True,
            right_index=True,
            how=how,
            suffixes=(lsuffix, rsuffix),
            sort=sort,
        )
        df.index.name = (
            None if self.index.name != other.index.name else self.index.name
        )

        # Preserve a CategoricalIndex columns axis when both inputs share the
        # same categorical dtype on their column labels (matches pandas).
        self_pd_cols = self._data.to_pandas_index
        other_pd_cols = other._data.to_pandas_index
        if (
            isinstance(self_pd_cols, pd.CategoricalIndex)
            and isinstance(other_pd_cols, pd.CategoricalIndex)
            and self_pd_cols.dtype == other_pd_cols.dtype
        ):
            df.columns = self_pd_cols.append(other_pd_cols)
        return df

    @_performance_tracking
    @docutils.doc_apply(
        groupby_doc_template.format(  # type: ignore[has-type]
            ret=textwrap.dedent(
                """
                Returns
                -------
                DataFrameGroupBy
                    Returns a DataFrameGroupBy object that contains
                    information about the groups.
                """
            )
        )
    )
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=no_default,
        group_keys=no_default,
        observed=True,
        dropna=True,
    ):
        return super().groupby(
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            observed,
            dropna,
        )

    def query(
        self,
        expr: str,
        local_dict: None | dict[str, Any] = None,
        global_dict: None | dict[str, Any] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Query with a boolean expression using Numba to compile a GPU kernel.

        See :meth:`pandas.DataFrame.query`.

        Parameters
        ----------
        expr : str
            A boolean expression. Names in expression refer to columns.
            `index` can be used instead of index name, but this is not
            supported for MultiIndex.
            Names starting with `@` refer to Python variables.
            An output value will be `null` if any of the input values are
            `null` regardless of expression.
        local_dict : dict
            Containing the local variable to be used in query.
        global_dict : dict, optional
            A dictionary of global variables. If not provided,
            the globals from the calling environment are used.
        **kwargs
            Not supported.

        Returns
        -------
        filtered : DataFrame

        Examples
        --------
        >>> df = cudf.DataFrame({
        ...     "a": [1, 2, 2],
        ...     "b": [3, 4, 5],
        ... })
        >>> expr = "(a == 2 and b == 4) or (b == 3)"
        >>> df.query(expr)
           a  b
        0  1  3
        1  2  4

        DateTime conditionals:

        >>> import numpy as np
        >>> import datetime
        >>> df = cudf.DataFrame()
        >>> data = np.array(['2018-10-07', '2018-10-08'], dtype='datetime64')
        >>> df['datetimes'] = data
        >>> search_date = datetime.datetime.strptime('2018-10-08', '%Y-%m-%d')
        >>> df.query('datetimes==@search_date')
           datetimes
        1 2018-10-08

        Using local_dict:

        >>> import numpy as np
        >>> import datetime
        >>> df = cudf.DataFrame()
        >>> data = np.array(['2018-10-07', '2018-10-08'], dtype='datetime64')
        >>> df['datetimes'] = data
        >>> search_date2 = datetime.datetime.strptime('2018-10-08', '%Y-%m-%d')
        >>> df.query('datetimes==@search_date',
        ...          local_dict={'search_date': search_date2})
           datetimes
        1 2018-10-08

        .. pandas-compat::
            :meth:`pandas.DataFrame.query`

            One difference from pandas is that ``query`` currently only
            supports numeric, datetime, timedelta, or bool dtypes.
        """
        if kwargs:
            raise ValueError(
                "Keyword arguments other than `local_dict`"
                "and `global_dict` are not supported."
            )
        # can't use `annotate` decorator here as we inspect the calling
        # environment.
        with annotate("DATAFRAME_QUERY", color="purple", domain="cudf_python"):
            if local_dict is None:
                local_dict = {}

            if global_dict is None:
                global_dict = {}

            if self.empty:
                return self.copy()

            if not isinstance(local_dict, dict):
                raise TypeError(
                    f"local_dict type: expected dict but found "
                    f"{type(local_dict)}"
                )

            if not isinstance(global_dict, dict):
                raise TypeError(
                    f"global_dict type: expected dict but found "
                    f"{type(global_dict)}"
                )

            # Get calling environment
            if (frame := inspect.currentframe()) is not None and (
                callframe := frame.f_back
            ) is not None:
                pass
            else:
                raise RuntimeError("Failed to get the calling frame.")
            callenv = {
                "locals": callframe.f_locals,
                "globals": callframe.f_globals,
                "local_dict": local_dict,
                "global_dict": global_dict,
            }
            # Run query
            boolmask = queryutils.query_execute(self, expr, callenv)
            return self._apply_boolean_mask(
                BooleanMask.from_column_unchecked(boolmask)
            )

    @_performance_tracking
    def apply(
        self,
        func,
        axis=1,
        raw=False,
        result_type=None,
        args=(),
        by_row: Literal[False, "compat"] = "compat",
        engine: Literal["python", "numba"] = "python",
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs,
    ):
        """
        Apply a function along an axis of the DataFrame.
        ``apply`` relies on Numba to JIT compile ``func``.
        Thus the allowed operations within ``func`` are limited to `those
        supported by the CUDA Python Numba target
        <https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html>`__.
        For more information, see the `cuDF guide to user defined functions
        <https://docs.rapids.ai/api/cudf/stable/cudf/guide-to-udfs/>`__.

        Some string functions and methods are supported. Refer to the guide
        to UDFs for details.

        Parameters
        ----------
        func : function
            Function to apply to each row.
        axis : {0 or 'index', 1 or 'columns'}, default 1
            Axis along which the function is applied.
            - 0 or 'index': apply function to each column (not yet supported).
            - 1 or 'columns': apply function to each row.
        raw: bool, default False
            Not yet supported
        result_type: {'expand', 'reduce', 'broadcast', None}, default None
            Not yet supported
        args: tuple
            Positional arguments to pass to func in addition to the dataframe.
        by_row : False or "compat", default "compat"
            Only has an effect when ``func`` is a listlike or dictlike of funcs
            and the func isn't a string.
            If "compat", will if possible first translate the func into pandas
            methods (e.g. ``Series().apply(np.sum)`` will be translated to
            ``Series().sum()``). If that doesn't work, will try call to apply again with
            ``by_row=True`` and if that fails, will call apply again with
            ``by_row=False`` (backward compatible).
            If False, the funcs will be passed the whole Series at once.

            Currently not supported.
        engine : {'python', 'numba'}, default 'python'
            Unused. Added for compatibility with pandas.
        engine_kwargs : dict
            Unused. Added for compatibility with pandas.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Examples
        --------
        Simple function of a single variable which could be NA:

        >>> def f(row):
        ...     if row['a'] is cudf.NA:
        ...             return 0
        ...     else:
        ...             return row['a'] + 1
        ...
        >>> df = cudf.DataFrame({'a': [1, cudf.NA, 3]})
        >>> df.apply(f, axis=1)
        0    2
        1    0
        2    4
        dtype: int64

        Function of multiple variables will operate in
        a null aware manner:

        >>> def f(row):
        ...     return row['a'] - row['b']
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, cudf.NA, 3, cudf.NA],
        ...     'b': [5, 6, cudf.NA, cudf.NA]
        ... })
        >>> df.apply(f)
        0      -4
        1    <NA>
        2    <NA>
        3    <NA>
        dtype: int64

        Functions may conditionally return NA as in pandas:

        >>> def f(row):
        ...     if row['a'] + row['b'] > 3:
        ...             return cudf.NA
        ...     else:
        ...             return row['a'] + row['b']
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [2, 1, 1]
        ... })
        >>> df.apply(f, axis=1)
        0       3
        1       3
        2    <NA>
        dtype: int64

        Mixed types are allowed, but will return the common
        type, rather than object as in pandas:

        >>> def f(row):
        ...     return row['a'] + row['b']
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [0.5, cudf.NA, 3.14]
        ... })
        >>> df.apply(f, axis=1)
        0     1.5
        1     NaN
        2    6.14
        dtype: float64

        Functions may also return scalar values, however the
        result will be promoted to a safe type regardless of
        the data:

        >>> def f(row):
        ...     if row['a'] > 3:
        ...             return row['a']
        ...     else:
        ...             return 1.5
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 3, 5]
        ... })
        >>> df.apply(f, axis=1)
        0    1.5
        1    1.5
        2    5.0
        dtype: float64

        Ops against N columns are supported generally:

        >>> def f(row):
        ...     v, w, x, y, z = (
        ...         row['a'], row['b'], row['c'], row['d'], row['e']
        ...     )
        ...     return x + (y - (z / w)) % v
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [cudf.NA, 4, 4],
        ...     'd': [8, 7, 8],
        ...     'e': [7, 1, 6]
        ... })
        >>> df.apply(f, axis=1)
        0    NaN
        1    4.8
        2    5.0
        dtype: float64

        UDFs manipulating string data are allowed, as long as
        they neither modify strings in place nor create new strings.
        For example, the following UDF is allowed:

        >>> def f(row):
        ...     st = row['str_col']
        ...     scale = row['scale']
        ...     if len(st) == 0:
        ...             return -1
        ...     elif st.startswith('a'):
        ...             return 1 - scale
        ...     elif 'example' in st:
        ...             return 1 + scale
        ...     else:
        ...             return 42
        ...
        >>> df = cudf.DataFrame({
        ...     'str_col': ['', 'abc', 'some_example'],
        ...     'scale': [1, 2, 3]
        ... })
        >>> df.apply(f, axis=1)  # doctest: +SKIP
        0   -1
        1   -1
        2    4
        dtype: int64

        However, the following UDF is not allowed since it includes an
        operation that requires the creation of a new string: a call to the
        ``upper`` method. Methods that are not supported in this manner
        will raise an ``AttributeError``.

        >>> def f(row):
        ...     st = row['str_col'].upper()
        ...     return 'ABC' in st
        >>> df.apply(f, axis=1)  # doctest: +SKIP

        For a complete list of supported functions and methods that may be
        used to manipulate string data, see the UDF guide,
        <https://docs.rapids.ai/api/cudf/stable/cudf/guide-to-udfs/>
        """
        if axis != 1:
            raise NotImplementedError(
                "DataFrame.apply currently only supports row wise ops"
            )
        if raw:
            raise NotImplementedError("The `raw` kwarg is not yet supported.")
        if result_type is not None:
            raise NotImplementedError(
                "The `result_type` kwarg is not yet supported."
            )
        if by_row != "compat":
            raise NotImplementedError("by_row is currently not supported.")

        return self._apply(func, DataFrameApplyKernel, *args, **kwargs)

    def map(
        self,
        func: Callable[[Any], Any],
        na_action: str | None = None,
        **kwargs,
    ) -> DataFrame:
        """
        Apply a function to a Dataframe elementwise.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.

        Returns
        -------
        DataFrame
            Transformed DataFrame.
        """

        if kwargs:
            raise NotImplementedError(
                "DataFrame.map does not yet support **kwargs."
            )

        if na_action not in {"ignore", None}:
            raise ValueError(
                f"na_action must be 'ignore' or None. Got {na_action!r}"
            )

        if na_action == "ignore":
            devfunc = numba.cuda.jit(device=True)(func)

            # promote to a null-ignoring function
            # this code is never run in python, it only
            # exists to provide numba with the correct
            # bytecode to generate the equivalent PTX
            # as a null-ignoring version of the function
            def _func(x):  # pragma: no cover
                if x is NA:
                    return NA
                else:
                    return devfunc(x)

        else:
            _func = func

        # TODO: naive implementation
        # this could be written as a single kernel
        result = {}
        for name, col in self._column_labels_and_values:
            apply_sr = Series._from_column(col)
            result[name] = apply_sr.apply(_func)._column

        return DataFrame._from_data(result, index=self.index, attrs=self.attrs)

    @_performance_tracking
    def partition_by_hash(
        self, columns: Sequence[Hashable], nparts: int, keep_index: bool = True
    ) -> list[Self]:
        """Partition the dataframe by the hashed value of data in *columns*.

        Parameters
        ----------
        columns : sequence of str
            The names of the columns to be hashed.
            Must have at least one name.
        nparts : int
            Number of output partitions
        keep_index : boolean
            Whether to keep the index or drop it

        Returns
        -------
        partitioned: list of DataFrame
        """
        key_indices = [self._column_names.index(k) for k in columns]
        if keep_index:
            cols: Iterable[ColumnBase] = itertools.chain(
                self.index._columns, self._columns
            )
            key_indices = [i + self.index._num_columns for i in key_indices]
        else:
            cols = self._columns

        # Materialize iterator to avoid consuming it during access context setup
        cols_list = list(cols)
        with access_columns(*cols_list, mode="read", scope="internal"):
            plc_table, offsets = plc.partitioning.hash_partition(
                plc.Table([col.plc_column for col in cols_list]),
                key_indices,
                nparts,
            )
        return self._wrap_from_partitions(
            plc_table,
            offsets,
            keep_index=keep_index,
            size=nparts,
            by_hash=True,
        )

    def _wrap_from_partitions(
        self,
        table: plc.Table,
        offsets: list[int],
        *,
        keep_index: bool,
        size: int,
        by_hash: bool,
    ) -> list[Self]:
        # Remove first element (always 0) and last element (total row count) from offsets
        offsets = offsets[1:-1]
        # Get the original dtypes to preserve (index columns + data columns if keep_index)
        original_dtypes = (
            [col.dtype for col in self.index._columns]
            + [col.dtype for col in self._columns]
            if keep_index
            else [col.dtype for col in self._columns]
        )
        output_columns = [
            ColumnBase.create(col, dtype)
            for col, dtype in zip(
                table.columns(), original_dtypes, strict=True
            )
        ]
        partitioned = self._from_columns_like_self(
            output_columns,
            column_names=self._column_names,
            index_names=self._index_names if keep_index else None,  # type: ignore[arg-type]
        )
        result = partitioned._split(offsets, keep_index=keep_index)
        if size:
            result.extend(
                self._empty_like(keep_index) for _ in range(size - len(result))
            )
        return result

    def info(
        self,
        verbose=None,
        buf=None,
        max_cols=None,
        memory_usage=None,
        null_counts=None,
    ):
        """
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and column dtypes, non-null values and memory usage.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary. By default, the setting in
            ``pandas.options.display.max_info_columns`` is followed.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        max_cols : int, optional
            When to switch from the verbose to the truncated output. If the
            DataFrame has more than `max_cols` columns, the truncated output
            is used. By default, the setting in
            ``pandas.options.display.max_info_columns`` is used.
        memory_usage : bool, str, optional
            Specifies whether total memory usage of the DataFrame
            elements (including the index) should be displayed. By default,
            this follows the ``pandas.options.display.memory_usage`` setting.
            True always show memory usage. False never shows memory usage.
            A value of 'deep' is equivalent to "True with deep introspection".
            Memory usage is shown in human-readable units (base-2
            representation). Without deep introspection a memory estimation is
            made based in column dtype and number of rows assuming values
            consume the same memory amount for corresponding dtypes. With deep
            memory introspection, a real memory usage calculation is performed
            at the cost of computational resources.
        null_counts : bool, optional
            Whether to show the non-null counts. By default, this is shown
            only if the frame is smaller than
            ``pandas.options.display.max_info_rows`` and
            ``pandas.options.display.max_info_columns``. A value of True always
            shows the counts, and False never shows the counts.

        Returns
        -------
        None
            This method prints a summary of a DataFrame and returns None.

        See Also
        --------
        DataFrame.describe: Generate descriptive statistics of DataFrame
            columns.
        DataFrame.memory_usage: Memory usage of DataFrame columns.

        Examples
        --------
        >>> import cudf
        >>> int_values = [1, 2, 3, 4, 5]
        >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> df = cudf.DataFrame({"int_col": int_values,
        ...                     "text_col": text_values,
        ...                     "float_col": float_values})
        >>> df
           int_col text_col  float_col
        0        1    alpha       0.00
        1        2     beta       0.25
        2        3    gamma       0.50
        3        4    delta       0.75
        4        5  epsilon       1.00

        Prints information of all columns:

        >>> df.info(verbose=True)
        <class 'cudf.core.dataframe.DataFrame'>
        RangeIndex: 5 entries, 0 to 4
        Data columns (total 3 columns):
         #   Column     Non-Null Count  Dtype
        ---  ------     --------------  -----
         0   int_col    5 non-null      int64
         1   text_col   5 non-null      str
         2   float_col  5 non-null      float64
        dtypes: float64(1), int64(1), str(1)
        memory usage: 130.0 bytes

        Prints a summary of columns count and its dtypes but not per column
        information:

        >>> df.info(verbose=False)
        <class 'cudf.core.dataframe.DataFrame'>
        RangeIndex: 5 entries, 0 to 4
        Columns: 3 entries, int_col to float_col
        dtypes: float64(1), int64(1), str(1)
        memory usage: 130.0 bytes

        Pipe output of DataFrame.info to a buffer instead of sys.stdout and
        print buffer contents:

        >>> import io
        >>> buffer = io.StringIO()
        >>> df.info(buf=buffer)
        >>> print(buffer.getvalue())
        <class 'cudf.core.dataframe.DataFrame'>
        RangeIndex: 5 entries, 0 to 4
        Data columns (total 3 columns):
         #   Column     Non-Null Count  Dtype
        ---  ------     --------------  -----
         0   int_col    5 non-null      int64
         1   text_col   5 non-null      str
         2   float_col  5 non-null      float64
        dtypes: float64(1), int64(1), str(1)
        memory usage: 130.0 bytes

        The `memory_usage` parameter allows deep introspection mode, specially
        useful for big DataFrames and fine-tune memory optimization:

        >>> import numpy as np
        >>> rng = np.random.default_rng(seed=0)
        >>> random_strings_array = rng.choice(['a', 'b', 'c'], 10 ** 6)
        >>> df = cudf.DataFrame({
        ...     'column_1': rng.choice(['a', 'b', 'c'], 10 ** 6),
        ...     'column_2': rng.choice(['a', 'b', 'c'], 10 ** 6),
        ...     'column_3': rng.choice(['a', 'b', 'c'], 10 ** 6)
        ... })
        >>> df.info(memory_usage='deep')
        <class 'cudf.core.dataframe.DataFrame'>
        RangeIndex: 1000000 entries, 0 to 999999
        Data columns (total 3 columns):
         #   Column    Non-Null Count    Dtype
        ---  ------    --------------    -----
         0   column_1  1000000 non-null  str
         1   column_2  1000000 non-null  str
         2   column_3  1000000 non-null  str
        dtypes: str(3)
        memory usage: 14.3 MB
        """
        if buf is None:
            buf = sys.stdout

        lines = [str(type(self))]

        index_name = type(self.index).__name__
        if len(self) > 0:
            entries_summary = f", {self.index[0]} to {self.index[-1]}"
        else:
            entries_summary = ""
        index_summary = f"{index_name}: {len(self)} entries{entries_summary}"
        lines.append(index_summary)

        if self._num_columns == 0:
            lines.append(f"Empty {type(self).__name__}")
            buffer_write_lines(buf, lines)
            return

        cols = self._column_names
        col_count = len(cols)

        if max_cols is None:
            max_cols = pd.options.display.max_info_columns

        max_rows = pd.options.display.max_info_rows

        if null_counts is None:
            show_counts = (col_count <= max_cols) and (len(self) < max_rows)
        else:
            show_counts = null_counts

        exceeds_info_cols = col_count > max_cols

        def _put_str(s, space):
            return str(s)[:space].ljust(space)

        def _verbose_repr():
            lines.append(f"Data columns (total {col_count} columns):")

            id_head = " # "
            column_head = "Column"
            col_space = 2

            max_col = max(len(pprint_thing(k)) for k in cols)
            len_column = len(pprint_thing(column_head))
            space = max(max_col, len_column) + col_space

            max_id = len(pprint_thing(col_count))
            len_id = len(pprint_thing(id_head))
            space_num = max(max_id, len_id) + col_space
            counts = None

            header = _put_str(id_head, space_num) + _put_str(
                column_head, space
            )
            if show_counts:
                counts = self.count().to_pandas().tolist()
                if col_count != len(counts):
                    raise AssertionError(
                        f"Columns must equal "
                        f"counts ({col_count} != {len(counts)})"
                    )
                count_header = "Non-Null Count"
                len_count = len(count_header)
                non_null = " non-null"
                max_count = max(len(pprint_thing(k)) for k in counts) + len(
                    non_null
                )
                space_count = max(len_count, max_count) + col_space
                count_temp = "{count}" + non_null
            else:
                count_header = ""
                space_count = len(count_header)
                len_count = space_count
                count_temp = "{count}"

            dtype_header = "Dtype"
            len_dtype = len(dtype_header)
            max_dtypes = max(
                len(pprint_thing(dtype)) for _, dtype in self._dtypes
            )
            space_dtype = max(len_dtype, max_dtypes)
            header += (
                _put_str(count_header, space_count)
                + _put_str(dtype_header, space_dtype).rstrip()
            )

            lines.append(header)
            lines.append(
                _put_str("-" * len_id, space_num)
                + _put_str("-" * len_column, space)
                + _put_str("-" * len_count, space_count)
                + _put_str("-" * len_dtype, space_dtype).rstrip()
            )

            for i, (col_name, col_dtype) in enumerate(self._dtypes):
                col = pprint_thing(col_name)

                line_no = _put_str(f" {i}", space_num)
                count = ""
                if show_counts:
                    count = counts[i]

                lines.append(
                    line_no
                    + _put_str(col, space)
                    + _put_str(count_temp.format(count=count), space_count)
                    + _put_str(col_dtype, space_dtype).rstrip()
                )

        def _non_verbose_repr():
            if col_count > 0:
                entries_summary = f", {cols[0]} to {cols[-1]}"
            else:
                entries_summary = ""
            columns_summary = f"Columns: {col_count} entries{entries_summary}"
            lines.append(columns_summary)

        def _sizeof_fmt(num, size_qualifier):
            # returns size in human readable format
            for x in ["bytes", "KB", "MB", "GB", "TB"]:
                if num < 1024.0:
                    return f"{num:3.1f}{size_qualifier} {x}"
                num /= 1024.0
            return f"{num:3.1f}{size_qualifier} PB"

        if verbose:
            _verbose_repr()
        elif verbose is False:  # specifically set to False, not nesc None
            _non_verbose_repr()
        else:
            if exceeds_info_cols:
                _non_verbose_repr()
            else:
                _verbose_repr()

        dtype_counts = defaultdict(int)
        for col in self._data:
            dtype_counts[self._data[col].dtype.name] += 1

        dtypes = [f"{k[0]}({k[1]:d})" for k in sorted(dtype_counts.items())]
        lines.append(f"dtypes: {', '.join(dtypes)}")

        if memory_usage is None:
            memory_usage = pd.options.display.memory_usage

        if memory_usage:
            # append memory usage of df to display
            size_qualifier = ""
            if memory_usage == "deep":
                deep = True
            else:
                deep = False
                if "object" in dtype_counts or self.index.dtype == "object":
                    size_qualifier = "+"
            mem_usage = self.memory_usage(index=True, deep=deep).sum()
            lines.append(
                f"memory usage: {_sizeof_fmt(mem_usage, size_qualifier)}\n"
            )

        buffer_write_lines(buf, lines)

    @_performance_tracking
    @docutils.doc_describe()
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ):
        """{docstring}"""

        if not include and not exclude:
            default_include = [np.number, "datetime"]
            data_to_describe = self.select_dtypes(include=default_include)
            if data_to_describe._num_columns == 0:
                data_to_describe = self

        elif include == "all":
            if exclude is not None:
                raise ValueError("exclude must be None when include is 'all'")

            data_to_describe = self
        else:
            data_to_describe = self.select_dtypes(
                include=include, exclude=exclude
            )

            if data_to_describe.empty:
                raise ValueError("No data of included types.")

        describe_series_list = [
            data_to_describe[col].describe(
                percentiles=percentiles,
            )
            for col in data_to_describe._column_names
        ]
        if len(describe_series_list) == 1:
            result = describe_series_list[0].to_frame()
        else:
            ldesc_indexes = sorted(
                (x.index for x in describe_series_list), key=len
            )
            names = dict.fromkeys(
                [
                    name
                    for idxnames in ldesc_indexes
                    for name in idxnames.to_pandas()
                ],
                None,
            )

            result = cudf.concat(
                [
                    series.reindex(names, copy=False)
                    for series in describe_series_list
                ],
                axis=1,
                sort=False,
            )
        # ``describe`` produces one column per described column in the same
        # order, so preserve the source column index (incl. its dtype/name,
        # e.g. a ``CategoricalIndex``), which ``concat``/``to_frame`` drop.
        result.columns = data_to_describe._data.to_pandas_index
        return result

    @_performance_tracking
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.DataFrame:
        """
        Convert to a Pandas DataFrame.

        Parameters
        ----------
        nullable : Boolean, Default False
            If ``nullable`` is ``True``, the resulting columns
            in the dataframe will be having a corresponding
            nullable Pandas dtype. If there is no corresponding
            nullable Pandas dtype present, the resulting dtype
            will be a regular pandas dtype.
            If ``nullable`` is ``False``,
            the resulting columns will either convert null
            values to ``np.nan`` or ``None`` depending on the dtype.
        arrow_type : bool, Default False
            Return the columns with a ``pandas.ArrowDtype``

        Returns
        -------
        out : Pandas DataFrame

        Notes
        -----
        nullable and arrow_type cannot both be set to ``True``

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [0, 1, 2], 'b': [-3, 2, 0]})
        >>> pdf = df.to_pandas()
        >>> pdf
           a  b
        0  0 -3
        1  1  2
        2  2  0
        >>> type(pdf)
        <class 'pandas.DataFrame'>

        ``nullable=True`` converts the result to pandas nullable types:

        >>> df = cudf.DataFrame({'a': [0, None, 2], 'b': [True, False, None]})
        >>> df
              a      b
        0     0   True
        1  <NA>  False
        2     2   <NA>
        >>> pdf = df.to_pandas(nullable=True)
        >>> pdf
              a      b
        0     0   True
        1  <NA>  False
        2     2   <NA>
        >>> pdf.dtypes
        a      Int64
        b    boolean
        dtype: object
        >>> pdf = df.to_pandas(nullable=False)
        >>> pdf
             a      b
        0  0.0   True
        1  NaN  False
        2  2.0   None
        >>> pdf.dtypes
        a    float64
        b     object
        dtype: object

        ``arrow_type=True`` converts the result to ``pandas.ArrowDtype``:

        >>> df.to_pandas(arrow_type=True).dtypes
        a    int64[pyarrow]
        b     bool[pyarrow]
        dtype: object
        """
        out_index = self.index.to_pandas()
        out_data = {
            i: col.to_pandas(nullable=nullable, arrow_type=arrow_type)
            for i, col in enumerate(self._columns)
        }

        out_df = pd.DataFrame(out_data, index=out_index)
        out_df.columns = self._data.to_pandas_index
        out_df.attrs = deepcopy(self.attrs)
        out_df.flags.allows_duplicate_labels = (
            self._flags.allows_duplicate_labels
        )

        return out_df

    @classmethod
    @_performance_tracking
    def from_arrow(cls, table: pa.Table) -> Self:
        """
        Convert from PyArrow Table to DataFrame.

        Parameters
        ----------
        table : pyarrow.Table
            PyArrow Table to convert to a cudf DataFrame.

        Raises
        ------
        TypeError for invalid input type.

        Returns
        -------
        cudf DataFrame

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> data = pa.table({"a":[1, 2, 3], "b":[4, 5, 6]})
        >>> cudf.DataFrame.from_arrow(data)
           a  b
        0  1  4
        1  2  5
        2  3  6

        .. pandas-compat::
            `pandas.DataFrame.from_arrow`

            This method does not exist in pandas but it is similar to
            how :external+pyarrow:meth:`pyarrow.Table.to_pandas` works for PyArrow Tables i.e.
            it does not support automatically setting index column(s).
        """
        out = super().from_arrow(table)
        if table.num_columns == 0:
            # Match pyarrow's to_pandas: empty tables produce object dtype
            # columns, not the cudf default string dtype.
            out._data.label_dtype = np.dtype("O")
        if isinstance(table.schema.pandas_metadata, dict):
            # Maybe set .index or .columns attributes
            pd_meta = table.schema.pandas_metadata
            if "column_indexes" in pd_meta:
                out._data._level_names = [
                    col_meta["name"] for col_meta in pd_meta["column_indexes"]
                ]
            if index_col := pd_meta["index_columns"]:
                if isinstance(index_col[0], dict):
                    range_meta = index_col[0]
                    idx = cudf.RangeIndex(
                        start=range_meta["start"],
                        stop=range_meta["stop"],
                        step=range_meta["step"],
                        name=range_meta["name"],
                    )
                    if len(idx) == len(out):
                        # `idx` is generated from arrow `pandas_metadata`
                        # which can get out of date with many of the
                        # arrow operations. Hence verifying if the
                        # lengths match, or else don't need to set
                        # an index at all i.e., Default RangeIndex
                        # will be set.
                        # See more about the discussion here:
                        # https://github.com/apache/arrow/issues/15178
                        out = out.set_index(idx)
                else:
                    out = out.set_index(index_col)
            if (
                len(out.index.names) == 1
                and out.index.names[0] == "__index_level_0__"
            ):
                for col_meta in pd_meta["columns"]:
                    if col_meta["field_name"] == "__index_level_0__":
                        out.index.name = col_meta["name"]
                        break
                else:
                    out.index.name = None
        return out

    @_performance_tracking
    def to_arrow(self, preserve_index: bool | None = None) -> pa.Table:
        """
        Convert to a PyArrow Table.

        Parameters
        ----------
        preserve_index : bool, optional
            whether index column and its meta data needs to be saved
            or not. The default of None will store the index as a
            column, except for a RangeIndex which is stored as
            metadata only. Setting preserve_index to True will force
            a RangeIndex to be materialized.

        Returns
        -------
        PyArrow Table

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame(
        ...     {"a":[1, 2, 3], "b":[4, 5, 6]}, index=[1, 2, 3])
        >>> df.to_arrow()
        pyarrow.Table
        a: int64
        b: int64
        index: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
        index: [[1,2,3]]
        >>> df.to_arrow(preserve_index=False)
        pyarrow.Table
        a: int64
        b: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
        """

        data = self
        index_descr: Sequence[dict[str, Any]] | Sequence[str] = []
        write_index = preserve_index is not False
        keep_range_index = write_index and preserve_index is None
        index = self.index
        index_levels = [self.index]
        if write_index:
            if isinstance(index, cudf.RangeIndex) and keep_range_index:
                index_descr = [
                    {
                        "kind": "range",
                        "name": index.name,
                        "start": index.start,
                        "stop": index.stop,
                        "step": index.step,
                    }
                ]
            else:
                if isinstance(index, cudf.RangeIndex):
                    index = index._as_int_index()
                    index.name = "__index_level_0__"
                if isinstance(index, MultiIndex):
                    index_descr = index._column_names
                    index_levels = index.levels
                else:
                    index_descr = (
                        index.names if index.name is not None else ("index",)
                    )
                data = data.copy(deep=False)
                for gen_name, col_name in zip(
                    index_descr, index._column_names, strict=True
                ):
                    data._insert(
                        data.shape[1],
                        gen_name,
                        index._data[col_name],
                    )

        out = super(DataFrame, data).to_arrow()
        # PyArrow stubs don't recognize pandas_compat attribute
        metadata = pa.pandas_compat.construct_metadata(  # type: ignore[attr-defined]
            columns_to_convert=[self[col] for col in self._column_names],
            df=self,
            column_names=out.schema.names,
            index_levels=index_levels,
            index_descriptors=index_descr,
            preserve_index=preserve_index,
            types=out.schema.types,
        )
        md_dict = json.loads(metadata[b"pandas"])

        _update_pandas_metadata_types_inplace(self, md_dict)

        return out.replace_schema_metadata({b"pandas": json.dumps(md_dict)})

    @_performance_tracking
    def to_records(self, index=True, column_dtypes=None, index_dtypes=None):
        """Convert to a numpy recarray

        Parameters
        ----------
        index : bool
            Whether to include the index in the output.
        column_dtypes : str, type, dict, default None
            If a string or type, the data type to store all columns. If
            a dictionary, a mapping of column names and indices (zero-indexed)
            to specific data types. Currently not supported.
        index_dtypes : str, type, dict, default None
            If a string or type, the data type to store all index levels. If
            a dictionary, a mapping of index level names and indices
            (zero-indexed) to specific data types.
            This mapping is applied only if `index=True`.
            Currently not supported.

        Returns
        -------
        numpy recarray
        """
        if column_dtypes is not None:
            raise NotImplementedError(
                "column_dtypes is currently not supported."
            )
        elif index_dtypes is not None:
            raise NotImplementedError(
                "column_dtypes is currently not supported."
            )
        members = [("index", self.index.dtype)] if index else []
        members += list(self._dtypes)
        dtype = np.dtype(members)
        ret = np.recarray(len(self), dtype=dtype)
        if index:
            ret["index"] = self.index.to_numpy()
        for col in self._column_names:
            ret[col] = self[col].to_numpy()
        return ret

    @classmethod
    @_performance_tracking
    def from_records(
        cls,
        data,
        index=None,
        exclude=None,
        columns=None,
        coerce_float: bool = False,
        nrows: int | None = None,
        nan_as_null=False,
    ):
        """
        Convert structured or record ndarray to DataFrame.

        Parameters
        ----------
        data : numpy structured dtype or recarray of ndim=2
        index : str, array-like
            The name of the index column in *data*.
            If None, the default index is used.
        exclude : sequence, default None
            Columns or fields to exclude.
            Currently not implemented.
        columns : list of str
            List of column names to include.
        coerce_float : bool, default False
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
            Currently not implemented.
        nrows : int, default None
            Number of rows to read if data is an iterator.
            Currently not implemented.

        Returns
        -------
        DataFrame
        """
        if exclude is not None:
            raise NotImplementedError("exclude is currently not supported.")
        if coerce_float is not False:
            raise NotImplementedError(
                "coerce_float is currently not supported."
            )
        if nrows is not None:
            raise NotImplementedError("nrows is currently not supported.")
        if not isinstance(data, (np.ndarray, cupy.ndarray)):
            raise TypeError("data must be a numpy ndarray or cupy ndarray")
        if data.ndim != 1 and data.ndim != 2:
            raise ValueError(
                f"records dimension expected 1 or 2 but found {data.ndim}"
            )

        num_cols = len(data[0])
        if columns is not None:
            if (names := data.dtype.names) is not None:
                if diff := set(columns) - set(names):
                    raise ValueError(
                        f"Found columns that don't exist in data: {diff}"
                    )
            if len(columns) > num_cols:
                raise ValueError(
                    f"columns length expected {num_cols} "
                    f"but found {len(columns)}"
                )
            names = columns
        elif data.dtype.names is None:
            names = range(num_cols)
        else:
            names = data.dtype.names

        if data.ndim == 2:
            ca_data = {
                k: as_column(data[:, i], nan_as_null=nan_as_null)
                for i, k in enumerate(names)
            }
        elif data.ndim == 1:
            ca_data = {
                name: as_column(data[name], nan_as_null=nan_as_null)
                for name in names
            }

        if not is_scalar(index):
            new_index = ensure_index(index)
        else:
            new_index = None

        if isinstance(columns, (pd.Index, Index)):
            level_names = tuple(columns.names)
        else:
            level_names = None

        df = cls._from_data(
            ColumnAccessor(
                data=ca_data,
                multiindex=isinstance(
                    columns, (pd.MultiIndex, cudf.MultiIndex)
                ),
                rangeindex=isinstance(
                    columns, (range, pd.RangeIndex, cudf.RangeIndex)
                ),
                level_names=level_names,
                label_dtype=getattr(columns, "dtype", None),
                verify=False,
            ),
            index=new_index,
        )
        if is_scalar(index) and index is not None:
            df = df.set_index(index)
        return df

    @_performance_tracking
    def quantile(
        self,
        q=0.5,
        axis=0,
        numeric_only=False,
        interpolation=None,
        method="single",
        columns=None,
        exact=True,
    ):
        """
        Return values at the given quantile.

        Parameters
        ----------
        q : float or array-like
            0 <= q <= 1, the quantile(s) to compute
        axis : int
            axis is a NON-FUNCTIONAL parameter
        numeric_only : bool, default False
            If True, compute the quantile only over numeric columns.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j.
            Default is ``'linear'`` for ``method="single"``, and ``'nearest'``
            for ``method="table"``.

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.
        method : {'single', 'table'}, default `'single'`
            Whether to compute quantiles per-column ('single') or over all
            columns ('table'). When 'table', the only allowed interpolation
            methods are 'nearest', 'lower', and 'higher'.
        columns : list of str
            List of column names to include.
        exact : boolean
            Whether to use approximate or exact quantile algorithm.

        Returns
        -------
        Series or DataFrame
            If q is an array, a DataFrame will be returned where the index
            is q, the columns are the columns of self, and the values are
            the quantiles.

            If q is a float, a Series will be returned where the index is
            the columns of self and the values are the quantiles.

        Examples
        --------
        >>> import cupy as cp
        >>> import cudf
        >>> df = cudf.DataFrame(cp.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
        ...                   columns=['a', 'b'])
        >>> df
           a    b
        0  1    1
        1  2   10
        2  3  100
        3  4  100
        >>> df.quantile(0.1)
        a    1.3
        b    3.7
        Name: 0.1, dtype: float64
        >>> df.quantile([.1, .5])
               a     b
        0.1  1.3   3.7
        0.5  2.5  55.0

        .. pandas-compat::
            :meth:`pandas.DataFrame.quantile`

            When ``q`` is a scalar and the columns do not share a common
            dtype (for example a mix of datetime and numeric columns),
            pandas returns an object-dtype Series. cuDF does not support
            mixed types under a Series and raises a ``MixedTypeError``
            instead.
        """
        if axis not in (0, None):
            raise NotImplementedError("axis is not implemented yet")

        data_df = self
        if numeric_only:
            data_df = data_df.select_dtypes(
                include=[np.number], exclude=["datetime64", "timedelta64"]
            )

        if columns is not None:
            requested = set(columns)
            data_df = data_df[
                [k for k in data_df._column_names if k in requested]
            ]

        if isinstance(q, numbers.Number):
            q_is_number = True
            qs = [float(q)]
        elif not is_scalar(q):
            q_is_number = False
            qs = q
        else:
            msg = "`q` must be either a single element or list"
            raise TypeError(msg)

        if method not in {"single", "table"}:
            raise ValueError(f"Invalid method: {method}")

        if method == "table" and len(data_df) > 0 and data_df._num_columns > 0:
            with access_columns(
                *data_df._columns, mode="read", scope="internal"
            ) as input_columns:
                plc_table = plc.quantiles.quantiles(
                    plc.Table([c.plc_column for c in input_columns]),
                    qs,
                    plc.types.Interpolation[
                        (interpolation or "nearest").upper()
                    ],
                    plc.types.Sorted.NO,
                    [],
                    [],
                )
                result = {
                    # The table method selects whole rows, so each result
                    # column has the same dtype as its source column (which
                    # ``dtype_from_pylibcudf_column`` would erase for types
                    # like tz-aware datetimes).
                    name: ColumnBase.create(plc_col, dtype=source_col.dtype)
                    for name, plc_col, source_col in zip(
                        data_df._column_names,
                        plc_table.columns(),
                        data_df._columns,
                        strict=True,
                    )
                }
        else:
            # ``method="table"`` also lands here when there are no rows or
            # no columns to compute over, where the all-null result matches
            # the per-column result.
            # Ensure that qs is non-scalar so that we always get a column back.
            interpolation = interpolation or "linear"
            result = {}
            for k in data_df._column_names:
                ser = data_df[k]
                res = ser.quantile(
                    qs,
                    interpolation=interpolation,
                    exact=exact,
                    quant_index=False,
                )._column
                if len(res) == 0:
                    res = column_empty(row_count=len(qs), dtype=ser.dtype)
                result[k] = res
        result_ca = ColumnAccessor(
            result,
            multiindex=data_df._data.multiindex,
            level_names=data_df._data.level_names,
            rangeindex=data_df._data.rangeindex,
            label_dtype=data_df._data.label_dtype,
            level_dtypes=data_df._data.level_dtypes,
            verify=False,
        )
        result = DataFrame._from_data(result_ca, attrs=self.attrs)

        if q_is_number:
            # For a scalar q pandas returns a Series indexed by the column
            # labels, which requires the columns to share a common dtype.
            if result._num_columns == 0:
                new_ser = Series._from_column(
                    column_empty(0, dtype=np.dtype(np.float64)),
                    name=q,
                    index=ensure_index(result.keys()),
                    attrs=self.attrs,
                )
            else:
                dtypes = [col.dtype for col in result._columns]
                if all(
                    isinstance(dtype, np.dtype) and dtype.kind in "iufb"
                    for dtype in dtypes
                ):
                    result = result.fillna(np.nan)
                    dtypes = [col.dtype for col in result._columns]
                if all(dtype == dtypes[0] for dtype in dtypes):
                    common_dtype = dtypes[0]
                elif all(
                    is_dtype_obj_numeric(dtype, include_decimal=False)
                    for dtype in dtypes
                ):
                    common_dtype = find_common_type(dtypes)
                else:
                    raise MixedTypeError(
                        f"q={q} requires returning a Series, which is only "
                        "possible when the resulting columns share a common "
                        "dtype; pass a list of quantiles to get a DataFrame "
                        "instead"
                    )
                new_ser = Series._from_column(
                    concat_columns(
                        [col.astype(common_dtype) for col in result._columns]
                    ),
                    name=q,
                    index=ensure_index(result.keys()),
                    attrs=self.attrs,
                )
            return new_ser

        result.index = Index(list(map(float, qs)), dtype="float64")
        return result

    @_performance_tracking
    def isin(self, values):
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all
            the labels match. If values is a Series, that's the index.
            If values is a dict, the keys must be the column names,
            which must match. If values is a DataFrame, then both the
            index and column labels must match.

        Returns
        -------
        DataFrame:
            DataFrame of booleans showing whether each element in
            the DataFrame is contained in values.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
        ...                     index=['falcon', 'dog'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        When ``values`` is a list check whether every value in the DataFrame
        is present in the list (which animals have 0 or 2 legs or wings)

        >>> df.isin([0, 2])
                num_legs  num_wings
        falcon      True       True
        dog        False       True

        When ``values`` is a dict, we can pass values to check for each
        column separately:

        >>> df.isin({'num_wings': [0, 3]})
                num_legs  num_wings
        falcon     False      False
        dog        False       True

        When ``values`` is a Series or DataFrame the index and column must
        match. Note that 'falcon' does not match based on the number of legs
        in other.

        >>> other = cudf.DataFrame({'num_legs': [8, 2], 'num_wings': [0, 2]},
        ...                         index=['spider', 'falcon'])
        >>> df.isin(other)
                num_legs  num_wings
        falcon      True       True
        dog        False      False
        """
        # TODO: propagate nulls through isin
        # https://github.com/rapidsai/cudf/issues/7556

        def make_false_column_like_self():
            return as_column(False, length=len(self), dtype=np.dtype("bool"))

        # Preprocess different input types into a mapping from column names to
        # a list of values to check.
        result = {}
        if isinstance(values, IndexedFrame):
            # Note: In the case where values is a Series, computing some
            # information about the values column outside the loop may result
            # in performance gains.  However, since categorical conversion
            # depends on the current column in the loop, using the correct
            # precomputed variables inside the loop requires nontrivial logic.
            # This optimization could be attempted if `isin` ever becomes a
            # bottleneck.
            if (
                isinstance(values, (Series, DataFrame))
                and not values.index.is_unique
            ):
                # if DataFrame ever supports duplicate columns
                # would need to check that here
                raise ValueError("cannot compute isin with a duplicate axis.")
            values = values.reindex(self.index)
            other_cols = (
                values._data
                if isinstance(values, DataFrame)
                else {name: values._column for name in self._data}
            )
            for col, self_col in self._column_labels_and_values:
                if col in other_cols:
                    other_col = other_cols[col]
                    self_is_cat = isinstance(self_col, CategoricalColumn)
                    other_is_cat = isinstance(other_col, CategoricalColumn)

                    if self_is_cat != other_is_cat:
                        # It is valid to compare the levels of a categorical
                        # column to a non-categorical column.
                        if self_is_cat:
                            self_col = self_col._get_decategorized_column()
                        else:
                            other_col = other_col._get_decategorized_column()

                    # We use the type checks from _before_ the conversion
                    # because if only one was categorical then it's already
                    # been converted and we have to check if they're strings.
                    if self_is_cat and other_is_cat:
                        self_is_str = other_is_str = False
                    else:
                        # These checks must happen after the conversions above
                        # since numpy can't handle categorical dtypes.
                        self_is_str = is_dtype_obj_string(self_col.dtype)
                        other_is_str = is_dtype_obj_string(other_col.dtype)

                    if self_is_str != other_is_str:
                        # Strings can't compare to anything else.
                        result[col] = make_false_column_like_self()
                    else:
                        result[col] = (self_col == other_col).fillna(False)
                else:
                    result[col] = make_false_column_like_self()
        elif is_dict_like(values):
            for name, col in self._column_labels_and_values:
                if name in values:
                    result[name] = col.isin(values[name])
                else:
                    result[name] = make_false_column_like_self()
        elif is_list_like(values):
            for name, col in self._column_labels_and_values:
                result[name] = col.isin(values)
        else:
            raise TypeError(
                "only list-like or dict-like objects are "
                "allowed to be passed to DataFrame.isin(), "
                "you passed a "
                f"'{type(values).__name__}'"
            )

        # TODO: Update this logic to properly preserve MultiIndex columns.
        return DataFrame._from_data(result, self.index, attrs=self.attrs)

    #
    # Stats
    #
    @_performance_tracking
    def _prepare_for_rowwise_op(self, method, skipna, numeric_only):
        """Prepare a DataFrame for CuPy-based row-wise operations."""

        if method not in _cupy_nan_methods_map and any(
            col.nullable for col in self._columns
        ):
            msg = (
                f"Row-wise operations to calculate '{method}' do not "
                f"currently support columns with null values. "
                f"Consider removing them with .dropna() "
                f"or using .fillna()."
            )
            raise ValueError(msg)

        if numeric_only:
            filtered = self.select_dtypes(include=[np.number, np.bool_])
        else:
            filtered = self.copy(deep=False)

        dtypes = [dtype for _, dtype in filtered._dtypes]
        is_pure_dt = all(dt.kind == "M" for dt in dtypes)

        common_dtype = find_common_type(dtypes) if dtypes else None
        if (
            not numeric_only
            and common_dtype is not None
            and is_dtype_obj_string(common_dtype)
            and any(not is_dtype_obj_string(dtype) for dtype in dtypes)
        ):
            raise TypeError(
                f"Cannot perform row-wise {method} across mixed-dtype columns,"
                " try type-casting all the columns to same dtype."
            )

        if not skipna and any(col.nullable for col in filtered._columns):
            length = filtered._data.nrows
            ca = ColumnAccessor(
                {
                    name: col._get_mask_as_column()
                    if col.nullable
                    else as_column(True, length=length)
                    for name, col in filtered._data.items()
                },
                verify=False,
            )
            mask = DataFrame._from_data(ca)
            mask = mask.all(axis=1)
        else:
            mask = None

        coerced = filtered.astype(common_dtype, copy=False)
        if is_pure_dt:
            # Further convert into cupy friendly types
            coerced = coerced.astype(np.dtype(np.int64), copy=False)
        return coerced, mask, common_dtype

    @_performance_tracking
    def count(self, axis: Axis = 0, numeric_only: bool = False) -> Series:
        """
        Count ``non-NA`` cells for each column or row.

        The values ``None``, ``NaN``, ``NaT`` are considered ``NA``.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> df = cudf.DataFrame({"Person":
        ...        ["John", "Myla", "Lewis", "John", "Myla"],
        ...        "Age": [24., np.nan, 21., 33, 26],
        ...        "Single": [False, True, True, True, False]})
        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64

        Count non-NA entries per row with ``axis=1``:

        >>> df.count(axis=1)
        0    3
        1    2
        2    3
        3    3
        4    3
        dtype: int64

        .. pandas-compat::
            :meth:`pandas.DataFrame.count`
        """
        axis = self._get_axis_from_axis_arg(axis)
        source = self
        if numeric_only:
            numeric_cols = (
                name
                for name, dtype in self._dtypes
                if is_dtype_obj_numeric(dtype)
            )
            source = self._get_columns_by_label(numeric_cols)
        if axis == 0:
            return Series._from_column(
                as_column([col.count for col in source._columns]),
                index=Index(source._data.to_pandas_index),
                attrs=self.attrs,
            )
        elif axis == 1:
            # count non-NA cells per row, i.e. the row-wise sum of each
            # column's validity. Implemented on-device to avoid a cudf.pandas
            # fallback that would copy the whole frame to host.
            if len(source._columns) == 0:
                result_col = as_column(
                    0, length=len(self), dtype=np.dtype(np.int64)
                )
            else:
                result_col = functools.reduce(
                    lambda left, right: left + right,
                    (
                        col.notnull().astype(np.dtype(np.int64))
                        for col in source._columns
                    ),
                )
            return Series._from_column(
                result_col, index=self.index, attrs=self.attrs
            )
        else:
            raise NotImplementedError(
                "Only axis=0 and axis=1 are currently supported."
            )

    _SUPPORT_AXIS_LOOKUP = {
        0: 0,
        1: 1,
        "index": 0,
        "columns": 1,
    }

    @_performance_tracking
    def _reduce(
        self,
        op: str,
        axis: Axis | None = 0,
        numeric_only: bool = False,
        **kwargs,
    ) -> ScalarLike:
        source = self

        if axis is None:
            # Both axis
            reduction_axis = 2
        else:
            reduction_axis = source._get_axis_from_axis_arg(axis)

        if numeric_only:
            numeric_cols = (
                name
                for name, dtype in self._dtypes
                if is_dtype_obj_numeric(dtype)
            )
            source = self._get_columns_by_label(numeric_cols)
            if source.empty:
                res = Series(
                    index=self._data.to_pandas_index[:0]
                    if reduction_axis == 0
                    else source.index,
                    dtype="float64",
                )
                res._attrs = self._attrs
                return res

        def _apply_reduction(col, op, kwargs):
            return getattr(col, op)(**kwargs)

        if (
            reduction_axis == 2
            and op in {"kurtosis", "skew"}
            and self._num_rows < 4
            and self._num_columns > 1
        ):
            return getattr(concat_columns(source._columns), op)(**kwargs)
        elif reduction_axis == 1:
            return source._apply_cupy_method_axis_1(op, **kwargs)
        else:
            axis_0_results = []
            for col_label, col in source._column_labels_and_values:
                try:
                    axis_0_results.append(_apply_reduction(col, op, kwargs))
                except (AttributeError, ValueError) as err:
                    if numeric_only:
                        raise NotImplementedError(
                            f"Column {col_label} with type {col.dtype} does not support {op}"
                        ) from err
                    elif not is_dtype_obj_numeric(col.dtype):
                        raise TypeError(
                            "Non numeric columns passed with "
                            "`numeric_only=False`, pass `numeric_only=True` "
                            f"to perform DataFrame.{op}"
                        ) from err
                    else:
                        raise
            if reduction_axis == 2:
                return _apply_reduction(
                    as_column(axis_0_results, nan_as_null=False), op, kwargs
                )
            else:
                source_dtypes = [dtype for _, dtype in source._dtypes]
                common_dtype = (
                    find_common_type(source_dtypes) if source_dtypes else None
                )
                if (
                    common_dtype is not None
                    and is_dtype_obj_string(common_dtype)
                    and any(
                        not is_dtype_obj_string(dtype)
                        for dtype in source_dtypes
                    )
                    or common_dtype is not None
                    and common_dtype.kind != "b"
                    and any(dtype.kind == "b" for dtype in source_dtypes)
                ):
                    raise TypeError(
                        "Columns must all have the same dtype to "
                        f"perform {op=} with {axis=}"
                    )
                pd_index = source._data.to_pandas_index
                idx = from_pandas(pd_index)
                if (
                    op == "std"
                    and common_dtype is not None
                    and common_dtype.kind == "M"
                ):
                    # TODO: Columns should probably signal the result type of their scalar
                    # Especially for this case where NaT could be datetime or timedelta
                    unit = np.datetime_data(common_dtype)[0]
                    axis_0_results = pd.Index(
                        axis_0_results, dtype=f"m8[{unit}]"
                    )
                # For max/min operations, preserve the original dtype since
                # Python scalars (int, float) would otherwise widen to int64/float64
                result_dtype = common_dtype if op in {"max", "min"} else None
                # For duration/timestamp reductions, preserve the kind so
                # NaT scalars don't default to datetime64[s]
                if (
                    result_dtype is None
                    and common_dtype is not None
                    and (
                        (
                            common_dtype.kind == "m"
                            and op in {"sum", "mean", "median", "std"}
                        )
                        or (
                            common_dtype.kind == "M"
                            and op in {"mean", "median"}
                        )
                    )
                ):
                    result_dtype = common_dtype
                res = as_column(
                    axis_0_results,
                    nan_as_null=False,
                    dtype=result_dtype,
                )

                res_dtype = res.dtype
                if res.isnull().all():
                    if cudf.api.types.is_numeric_dtype(common_dtype):
                        if op in {"sum", "product"}:
                            if (
                                common_dtype is not None
                                and common_dtype.kind == "f"
                            ):
                                res_dtype = (
                                    np.dtype("float64")
                                    if isinstance(common_dtype, pd.ArrowDtype)
                                    else common_dtype
                                )
                            elif (
                                common_dtype is not None
                                and common_dtype.kind == "u"
                            ):
                                res_dtype = np.dtype("uint64")
                            elif (
                                common_dtype is not None
                                and common_dtype.kind == "b"
                                and isinstance(common_dtype, pd.ArrowDtype)
                            ):
                                res_dtype = np.dtype("uint64")
                            else:
                                res_dtype = np.dtype("int64")
                        elif op == "sum_of_squares":
                            res_dtype = find_common_type(
                                (common_dtype, np.dtype(np.uint64))
                            )
                        elif op in {
                            "var",
                            "std",
                            "mean",
                            "skew",
                            "median",
                        }:
                            if (
                                common_dtype is not None
                                and common_dtype.kind == "f"
                            ):
                                res_dtype = (
                                    np.dtype("float64")
                                    if isinstance(common_dtype, pd.ArrowDtype)
                                    else common_dtype
                                )
                            else:
                                res_dtype = np.dtype("float64")
                        elif op in {"max", "min"}:
                            res_dtype = common_dtype
                    elif (
                        common_dtype is not None
                        and common_dtype.kind == "m"
                        and op
                        in {"sum", "mean", "median", "std", "max", "min"}
                    ):
                        res_dtype = common_dtype
                    elif (
                        common_dtype is not None
                        and common_dtype.kind == "M"
                        and op in {"mean", "median", "max", "min"}
                    ):
                        res_dtype = common_dtype
                    if op in {"any", "all"}:
                        res_dtype = np.dtype(np.bool_)
                res = res.nans_to_nulls()
                new_dtype = get_dtype_of_same_kind(common_dtype, res_dtype)
                res = res.astype(new_dtype)

            return Series._from_column(res, index=idx, attrs=self.attrs)

    @_performance_tracking
    def _scan(
        self,
        op: str,
        axis: Axis | None = None,
        skipna: bool = True,
        *args,
        **kwargs,
    ) -> Self:
        if axis is None:
            axis = 0
        axis = self._get_axis_from_axis_arg(axis)

        if axis == 0:
            return super()._scan(op, axis=axis, skipna=skipna, *args, **kwargs)
        elif axis == 1:
            return self._apply_cupy_method_axis_1(op, skipna=skipna, **kwargs)
        else:
            raise ValueError(f"{axis=} should be None, 0 or 1")

    @_performance_tracking
    def mode(self, axis=0, numeric_only=False, dropna=True):
        """
        Get the mode(s) of each element along the selected axis.

        The mode of a set of values is the value that appears most often.
        It can be multiple values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to iterate over while searching for the mode:

            - 0 or 'index' : get mode of each column
            - 1 or 'columns' : get mode of each row.
        numeric_only : bool, default False
            If True, only apply to numeric columns.
        dropna : bool, default True
            Don't consider counts of NA/NaN/NaT.

        Returns
        -------
        DataFrame
            The modes of each column or row.

        See Also
        --------
        cudf.Series.mode : Return the highest frequency value
            in a Series.
        cudf.Series.value_counts : Return the counts of values
            in a Series.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     "species": ["bird", "mammal", "arthropod", "bird"],
        ...     "legs": [2, 4, 8, 2],
        ...     "wings": [2.0, None, 0.0, None]
        ... })
        >>> df
             species  legs wings
        0       bird     2   2.0
        1     mammal     4   NaN
        2  arthropod     8   0.0
        3       bird     2   NaN

        By default, missing values are not considered, and the mode of wings
        are both 0 and 2. The second row of species and legs contains ``NaN``,
        because they have only one mode, but the DataFrame has two rows.

        >>> df.mode()
          species legs  wings
        0    bird  2.0    0.0
        1     NaN  NaN    2.0

        Setting ``dropna=False``, ``NA`` values are considered and they can be
        the mode (like for wings).

        >>> df.mode(dropna=False)
          species  legs wings
        0    bird     2   NaN

        Setting ``numeric_only=True``, only the mode of numeric columns is
        computed, and columns of other types are ignored.

        >>> df.mode(numeric_only=True)
          legs  wings
        0  2.0    0.0
        1  NaN    2.0

        .. pandas-compat::
            :meth:`pandas.DataFrame.transpose`

            ``axis`` parameter is currently not supported.
        """
        if axis not in (0, "index"):
            raise NotImplementedError("Only axis=0 is currently supported")

        if numeric_only:
            data_df = self.select_dtypes(
                include=[np.number], exclude=["datetime64", "timedelta64"]
            )
        else:
            data_df = self

        mode_results = [
            data_df[col].mode(dropna=dropna) for col in data_df._data
        ]

        if len(mode_results) == 0:
            return DataFrame()

        df = cudf.concat(mode_results, axis=1)

        if isinstance(df, Series):
            df = df.to_frame()

        df._set_columns_like(data_df._data)

        return df

    @_performance_tracking
    def all(
        self,
        axis: Axis | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ):
        obj = (
            self.select_dtypes(include=np.dtype(np.bool_))
            if bool_only
            else self
        )
        return super(DataFrame, obj).all(axis, skipna, **kwargs)  # type: ignore[misc]

    @_performance_tracking
    def any(
        self,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs,
    ):
        obj = (
            self.select_dtypes(include=np.dtype(np.bool_))
            if bool_only
            else self
        )
        return super(DataFrame, obj).any(axis, skipna, **kwargs)  # type: ignore[misc]

    @_performance_tracking
    def _apply_cupy_method_axis_1(self, method: str, *args, **kwargs):
        # This method uses cupy to perform scans and reductions along rows of a
        # DataFrame. Since cuDF is designed around columnar storage and
        # operations, we convert DataFrames to 2D cupy arrays for these ops.

        # for dask metadata compatibility
        skipna = kwargs.pop("skipna", None)
        skipna = True if skipna is None else skipna
        if method not in _cupy_nan_methods_map and skipna not in (
            None,
            True,
            1,
        ):
            raise NotImplementedError(
                f"Row-wise operations to calculate '{method}'"
                f" currently do not support `skipna=False`."
            )

        level = kwargs.pop("level", None)
        if level not in (None,):
            raise NotImplementedError(
                "Row-wise operations currently do not support `level`."
            )

        numeric_only = kwargs.pop("numeric_only", False)

        min_count = kwargs.pop("min_count", None)
        if min_count not in (None, 0):
            raise NotImplementedError(
                "Row-wise operations currently do not support `min_count`."
            )

        bool_only = kwargs.pop("bool_only", None)
        if bool_only not in (None, True):
            raise NotImplementedError(
                "Row-wise operations currently do not support `bool_only`."
            )

        # This parameter is only necessary for axis 0 reductions that cuDF
        # performs internally. cupy already upcasts smaller integer/bool types
        # to int64 when accumulating.
        kwargs.pop("cast_to_int", None)

        prepared, mask, common_dtype = self._prepare_for_rowwise_op(
            method, skipna, numeric_only
        )

        for col in prepared._column_names:
            if prepared._data[col].nullable:
                prepared._data[col] = (
                    prepared._data[col]
                    .astype(
                        prepared._data[col]._min_column_type(
                            np.dtype(np.float32)
                        )
                        if common_dtype.kind != "M"
                        else np.dtype(np.float64)
                    )
                    .fillna(np.nan)
                )
        arr = prepared.to_cupy()

        if skipna is not False and method in _cupy_nan_methods_map:
            method = _cupy_nan_methods_map[method]

        if len(arr) == 0 and method == "nanmedian":
            # Workaround for a cupy limitation, cupy
            # errors for zero dim array in nanmedian
            # https://github.com/cupy/cupy/issues/9332
            method = "median"
        result = getattr(cupy, method)(arr, axis=1, **kwargs)

        if result.ndim == 1:
            type_coerced_methods = {
                "count",
                "min",
                "nanmin",
                "max",
                "nanmax",
                "sum",
                "nansum",
                "prod",
                "nanprod",
                "product",
                "cummin",
                "cummax",
                "cumsum",
                "cumprod",
            }
            result_dtype = (
                common_dtype
                if method in type_coerced_methods
                or (common_dtype is not None and common_dtype.kind == "M")
                else None
            )

            if result_dtype is None and is_pandas_nullable_extension_dtype(
                common_dtype
            ):
                if (
                    method
                    in {
                        "kurt",
                        "kurtosis",
                        "mean",
                        "nanmean",
                        "median",
                        "nanmedian",
                        "sem",
                        "skew",
                        "std",
                        "nanstd",
                        "var",
                        "nanvar",
                    }
                    and common_dtype.kind != "f"
                ):
                    result_dtype = get_dtype_of_same_kind(
                        common_dtype, np.dtype(np.float64)
                    )
                else:
                    result_dtype = get_dtype_of_same_kind(
                        common_dtype, result.dtype
                    )
            if (
                result_dtype is not None
                and result_dtype.kind == "b"
                and result.dtype.kind != "b"
            ):
                result_dtype = get_dtype_of_same_kind(
                    common_dtype, result.dtype
                )
            result = as_column(result, dtype=result_dtype)
            if mask is not None:
                mask_buff, null_count = mask._column.as_mask()
                result = result.set_mask(mask_buff, null_count)
            return Series._from_column(
                result, index=self.index, attrs=self.attrs
            )
        else:
            result_df = DataFrame(result, index=self.index)
            result_df._set_columns_like(prepared._data)
            result_df._attrs = self.attrs
            return result_df

    @_performance_tracking
    def select_dtypes(self, include=None, exclude=None):
        """Return a subset of the DataFrame's columns based on the column dtypes.

        Parameters
        ----------
        include : str or list
            which columns to include based on dtypes
        exclude : str or list
            which columns to exclude based on dtypes

        Returns
        -------
        DataFrame
            The subset of the frame including the dtypes
            in ``include`` and excluding the dtypes in ``exclude``.

        Raises
        ------
        ValueError
            - If both of ``include`` and ``exclude`` are empty
            - If ``include`` and ``exclude`` have overlapping elements

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2] * 3,
        ...                    'b': [True, False] * 3,
        ...                    'c': [1.0, 2.0] * 3})
        >>> df
           a      b    c
        0  1   True  1.0
        1  2  False  2.0
        2  1   True  1.0
        3  2  False  2.0
        4  1   True  1.0
        5  2  False  2.0
        >>> df.select_dtypes(include='bool')
               b
        0   True
        1  False
        2   True
        3  False
        4   True
        5  False
        >>> df.select_dtypes(include=['float64'])
             c
        0  1.0
        1  2.0
        2  1.0
        3  2.0
        4  1.0
        5  2.0
        >>> df.select_dtypes(exclude=['int'])
               b    c
        0   True  1.0
        1  False  2.0
        2   True  1.0
        3  False  2.0
        4   True  1.0
        5  False  2.0
        """

        # Mirrors pandas.DataFrame.select_dtypes:
        # https://github.com/pandas-dev/pandas/blob/main/pandas/core/frame.py

        if not isinstance(include, (list, tuple)):
            include = (include,) if include is not None else ()
        if not isinstance(exclude, (list, tuple)):
            exclude = (exclude,) if exclude is not None else ()

        selection = tuple(map(frozenset, (include, exclude)))

        if not any(selection):
            raise ValueError(
                "at least one of include or exclude must be nonempty"
            )

        def convert_dtype_entry(entry):
            """Convert an include/exclude entry to the generic type(s) used
            for matching, like pandas' ``check_int_infer_dtype``.
            """
            if _is_categorical_dtype(entry):
                return (pd.core.dtypes.dtypes.CategoricalDtypeType,)
            elif is_decimal32_dtype(entry):
                return (Decimal32Dtype,)
            elif is_decimal64_dtype(entry):
                return (Decimal64Dtype,)
            elif is_decimal128_dtype(entry):
                return (Decimal128Dtype,)
            elif (isinstance(entry, str) and entry == "int") or entry is int:
                # Numpy maps int to different types (int32, int64) on Windows
                # and Linux
                return (np.int32, np.int64)
            elif (
                isinstance(entry, str) and entry == "float"
            ) or entry is float:
                # np.dtype("float") coerces to np.float64 from Numpy 1.20
                return (np.float64, np.float32)
            return (pd.core.dtypes.common.infer_dtype_from_object(entry),)

        include, exclude = (
            frozenset(
                itertools.chain.from_iterable(map(convert_dtype_entry, x))
            )
            for x in selection
        )

        for dtypes_set in (include, exclude):
            if dtypes_set & {np.dtype("S").type, np.dtype("<U").type}:
                raise TypeError(
                    "numpy string dtypes are not allowed, use 'str' or "
                    "'object' instead"
                )

        # can't both include AND exclude!
        if not include.isdisjoint(exclude):
            raise ValueError(
                f"include and exclude overlap on {(include & exclude)}"
            )

        def dtype_predicate(dtype, dtypes_set):
            if _is_categorical_dtype(dtype):
                # cudf's CategoricalDtype.type is the categories' scalar
                # type, so it must not participate in the generic matching
                return pd.core.dtypes.dtypes.CategoricalDtypeType in dtypes_set
            if isinstance(
                dtype, (Decimal32Dtype, Decimal64Dtype, Decimal128Dtype)
            ):
                return type(dtype) in dtypes_set
            if isinstance(dtype, pd.ArrowDtype):
                dtype = dtype.numpy_dtype
            classes = tuple(t for t in dtypes_set if inspect.isclass(t))
            return (
                (
                    inspect.isclass(dtype.type)
                    and issubclass(dtype.type, classes)
                )
                # BooleanDtype._is_numeric == True but should be excluded
                or (
                    np.number in dtypes_set
                    and getattr(dtype, "_is_numeric", False)
                    and dtype.kind != "b"
                )
                # backwards compat for the default `str` dtype being
                # selected by object
                or (
                    isinstance(dtype, pd.StringDtype)
                    and dtype.na_value is np.nan
                    and np.object_ in dtypes_set
                )
            )

        if (
            np.object_ in include
            and str not in include
            and str not in exclude
            and any(
                isinstance(dtype, pd.StringDtype) and dtype.na_value is np.nan
                for _, dtype in self._dtypes
            )
        ):
            warnings.warn(
                "For backward compatibility, 'str' dtypes are included by "
                "select_dtypes when 'object' dtype is specified. "
                "This behavior is deprecated and will be removed in a future "
                "version. Explicitly pass 'str' to `include` to select them, "
                "or to `exclude` to remove them and silence this warning.",
                FutureWarning,
                stacklevel=2,
            )

        to_select = [
            label
            for label, dtype in self._dtypes
            if (not include or dtype_predicate(dtype, include))
            and (not exclude or not dtype_predicate(dtype, exclude))
        ]
        result = self.loc[:, to_select]
        if not to_select and self._data.rangeindex:
            # Preserve RangeIndex columns through an empty selection so that
            # downstream operations match pandas' column metadata.
            result._data.rangeindex = True
        return result

    @ioutils.doc_to_parquet()
    def to_parquet(
        self,
        path,
        engine="cudf",
        compression="snappy",
        index=None,
        partition_cols=None,
        partition_file_name=None,
        partition_offsets=None,
        statistics="ROWGROUP",
        metadata_file_path=None,
        int96_timestamps=False,
        row_group_size_bytes=None,
        row_group_size_rows=None,
        max_page_size_bytes=None,
        max_page_size_rows=None,
        storage_options=None,
        return_metadata=False,
        use_dictionary=True,
        header_version="1.0",
        skip_compression=None,
        column_encoding=None,
        column_type_length=None,
        output_as_binary=None,
        *args,
        **kwargs,
    ):
        """{docstring}"""
        from cudf.io import parquet

        return parquet.to_parquet(
            self,
            path=path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            partition_file_name=partition_file_name,
            partition_offsets=partition_offsets,
            statistics=statistics,
            metadata_file_path=metadata_file_path,
            int96_timestamps=int96_timestamps,
            row_group_size_bytes=row_group_size_bytes,
            row_group_size_rows=row_group_size_rows,
            max_page_size_bytes=max_page_size_bytes,
            max_page_size_rows=max_page_size_rows,
            storage_options=storage_options,
            return_metadata=return_metadata,
            use_dictionary=use_dictionary,
            header_version=header_version,
            skip_compression=skip_compression,
            column_encoding=column_encoding,
            column_type_length=column_type_length,
            output_as_binary=output_as_binary,
            *args,
            **kwargs,
        )

    @ioutils.doc_to_feather()
    def to_feather(self, path, *args, **kwargs):
        """{docstring}"""
        from cudf.io import feather

        feather.to_feather(self, path, *args, **kwargs)

    @ioutils.doc_dataframe_to_csv()
    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        columns=None,
        header=True,
        index=True,
        encoding=None,
        compression=None,
        quoting=None,
        lineterminator=None,
        chunksize=None,
        storage_options=None,
    ):
        """{docstring}"""
        from cudf.io import csv

        if lineterminator is None:
            lineterminator = os.linesep
        return csv.to_csv(
            self,
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            columns=columns,
            header=header,
            index=index,
            lineterminator=lineterminator,
            chunksize=chunksize,
            encoding=encoding,
            compression=compression,
            storage_options=storage_options,
            quoting=quoting,
        )

    @ioutils.doc_to_orc()
    def to_orc(
        self,
        fname,
        compression="snappy",
        statistics="ROWGROUP",
        stripe_size_bytes=None,
        stripe_size_rows=None,
        row_index_stride=None,
        cols_as_map_type=None,
        storage_options=None,
        index=None,
    ):
        """{docstring}"""
        from cudf.io import orc

        return orc.to_orc(
            df=self,
            fname=fname,
            compression=compression,
            statistics=statistics,
            stripe_size_bytes=stripe_size_bytes,
            stripe_size_rows=stripe_size_rows,
            row_index_stride=row_index_stride,
            cols_as_map_type=cols_as_map_type,
            storage_options=storage_options,
            index=index,
        )

    @_performance_tracking
    def stack(
        self,
        level=-1,
        dropna: bool | NoDefault = no_default,
        future_stack: bool = True,
    ) -> DataFrame | Series:
        """Stack the prescribed level(s) from columns to index

        Return a reshaped DataFrame or Series having a multi-level
        index with one or more new inner-most levels compared to
        the current DataFrame. The new inner-most levels are created
        by pivoting the columns of the current dataframe:

          - if the columns have a single level, the output is a Series;
          - if the columns have multiple levels, the new index
            level(s) is (are) taken from the prescribed level(s) and
            the output is a DataFrame.

        Parameters
        ----------
        level : int, str, list default -1
            Level(s) to stack from the column axis onto the index axis,
            defined as one index or label, or a list of indices or labels.
        dropna : bool, default True
            Whether to drop rows in the resulting Frame/Series with missing
            values. When multiple levels are specified, `dropna==False` is
            unsupported.

        Returns
        -------
        DataFrame or Series
            Stacked dataframe or series.

        See Also
        --------
        DataFrame.unstack : Unstack prescribed level(s) from index axis
             onto column axis.
        DataFrame.pivot : Reshape dataframe from long format to wide
             format.
        DataFrame.pivot_table : Create a spreadsheet-style pivot table
             as a DataFrame.

        Notes
        -----
        The function is named by analogy with a collection of books
        being reorganized from being side by side on a horizontal
        position (the columns of the dataframe) to being stacked
        vertically on top of each other (in the index of the
        dataframe).

        Examples
        --------
        **Single level columns**

        >>> df_single_level_cols = cudf.DataFrame([[0, 1], [2, 3]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=['weight', 'height'])

        Stacking a dataframe with a single level column axis returns a Series:

        >>> df_single_level_cols
             weight height
        cat       0      1
        dog       2      3
        >>> df_single_level_cols.stack()
        cat  weight    0
             height    1
        dog  weight    2
             height    3
        dtype: int64

        **Multi level columns: simple case**

        >>> import pandas as pd
        >>> multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('weight', 'pounds')])
        >>> df_multi_level_cols1 = cudf.DataFrame([[1, 2], [2, 4]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol1)

        Stacking a dataframe with a multi-level column axis:

        >>> df_multi_level_cols1
             weight
                 kg    pounds
        cat       1        2
        dog       2        4
        >>> df_multi_level_cols1.stack()
                    weight
        cat kg           1
            pounds       2
        dog kg           2
            pounds       4

        **Missing values**

        >>> multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('height', 'm')])
        >>> df_multi_level_cols2 = cudf.DataFrame([[1.0, 2.0], [3.0, 4.0]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol2)

        It is common to have missing values when stacking a dataframe
        with multi-level columns, as the stacked dataframe typically
        has more values than the original dataframe. Missing values
        are filled with NULLs:

        >>> df_multi_level_cols2
            weight height
                kg      m
        cat    1.0    2.0
        dog    3.0    4.0
        >>> df_multi_level_cols2.stack()
               weight height
        cat kg    1.0    NaN
            m     NaN    2.0
        dog kg    3.0    NaN
            m     NaN    4.0

        **Prescribing the level(s) to be stacked**

        The first parameter controls which level or levels are stacked:

        >>> df_multi_level_cols2.stack(0)
                    kg     m
        cat weight   1.0   NaN
            height   NaN   2.0
        dog weight   3.0   NaN
            height   NaN   4.0

        >>> df_multi_level_cols2.stack([0, 1])
        cat  weight  kg    1.0
             height  m     2.0
        dog  weight  kg    3.0
             height  m     4.0
        dtype: float64
        """

        if future_stack:
            if dropna is not no_default:
                raise ValueError(
                    "dropna must be unspecified with future_stack=True as "
                    "the new implementation does not introduce rows of NA "
                    "values. This argument will be removed in a future "
                    "version of cudf."
                )
        else:
            warnings.warn(
                "The previous implementation of stack is deprecated and "
                "will be removed in a future version of cudf. Do not specify the "
                "future_stack argument to adopt the new implementation and "
                "silence this warning.",
                FutureWarning,
            )
            if dropna is no_default:
                dropna = True

        if isinstance(level, (int, str)):
            level = [level]
        elif isinstance(level, list):
            if not all(isinstance(lv, (int, str)) for lv in level):
                raise ValueError(
                    "level must be either an int/str, or a list of int/str."
                )
        else:
            raise ValueError(
                "level must be either an int/str, or a list of int/str."
            )

        level = [level] if not isinstance(level, list) else level

        if not future_stack and len(level) > 1 and not dropna:
            raise NotImplementedError(
                "When stacking multiple levels, setting `dropna` to False "
                "will generate new column combination that does not exist "
                "in original dataframe. This behavior is unsupported in "
                "cuDF. See pandas deprecation note: "
                "https://github.com/pandas-dev/pandas/issues/53515"
            )

        _check_duplicate_level_names(
            [lv for lv in level if not is_integer(lv)],
            self._data.level_names,
        )

        # Compute the columns to stack based on specified levels

        level_indices: list[int] = []

        # If all passed in level names match up to the dataframe column's level
        # names, cast them to indices
        if all(lv in self._data.level_names for lv in level):
            level_indices = [self._data.level_names.index(lv) for lv in level]
        elif not all(isinstance(lv, int) for lv in level):
            raise ValueError(
                "`level` must either be a list of names or positions, not a "
                "mixture of both."
            )
        else:
            # Must be a list of positions; normalize negative positions
            # and validate bounds to match pandas MultiIndex._get_level_number
            nlevels = self._data.nlevels
            for lv in level:
                if lv < 0:
                    if lv + nlevels < 0:
                        raise IndexError(
                            f"Too many levels: Index has only {nlevels} "
                            f"levels, {lv} is not a valid level number"
                        )
                    level_indices.append(lv + nlevels)
                else:
                    if lv >= nlevels:
                        raise IndexError(
                            f"Too many levels: Index has only {nlevels} "
                            f"levels, not {lv + 1}"
                        )
                    level_indices.append(lv)

        unnamed_levels_indices = [
            i for i in range(self._data.nlevels) if i not in level_indices
        ]
        has_unnamed_levels = len(unnamed_levels_indices) > 0

        column_name_idx = self._data.to_pandas_index
        # pandas' Index.get_level_values resolves an integer argument by
        # name first: if a level is *named* that integer, that level is
        # returned regardless of position. All lookups below use positional
        # indices, so strip the names to force positional resolution and
        # re-attach the real names afterwards.
        nameless_column_name_idx = column_name_idx.set_names(
            [None] * column_name_idx.nlevels
        )
        # Construct new index from the levels specified by `level`
        if isinstance(column_name_idx, pd.MultiIndex):
            # build from codes/levels to keep the level dtypes: materializing
            # via get_level_values/from_arrays turns missing entries into NaN
            # and upcasts e.g. int64 levels to float64
            named_levels = pd.MultiIndex(
                levels=[column_name_idx.levels[i] for i in level_indices],
                codes=[column_name_idx.codes[i] for i in level_indices],
                names=[column_name_idx.names[i] for i in level_indices],
                verify_integrity=False,
            )
        else:
            named_levels = pd.MultiIndex.from_arrays(
                [
                    nameless_column_name_idx.get_level_values(lv).rename(
                        column_name_idx.names[lv]
                    )
                    for lv in level_indices
                ]
            )

        # Since `level` may only specify a subset of all levels, `unique()` is
        # required to remove duplicates. In pandas legacy stack, the keys of
        # the specified levels are sorted by their level *codes* when the
        # columns have multiple levels (flat column labels keep their
        # original order): level order is preserved even for unsorted levels
        # and missing labels (code -1) come first.
        unique_named_levels = named_levels.unique()
        if not future_stack and self._data.nlevels > 1:
            unique_named_levels = unique_named_levels.take(
                np.lexsort(tuple(reversed(unique_named_levels.codes)))
            )

        # Each index from the original dataframe should repeat by the number
        # of unique values in the named_levels
        repeated_index = self.index.repeat(len(unique_named_levels))

        # Each column name should tile itself by len(df) times
        nameless_unique_named_levels = unique_named_levels.set_names(
            [None] * unique_named_levels.nlevels
        )
        cols = []
        for i in range(unique_named_levels.nlevels):
            if future_stack:
                # pandas future stack materializes the level values (a
                # level with missing entries becomes e.g. float64 with NaN)
                cols.append(
                    as_column(nameless_unique_named_levels.get_level_values(i))
                )
            else:
                # pandas legacy stack keeps the original level dtype and
                # represents missing entries as nulls (-1 codes)
                level_col = as_column(unique_named_levels.levels[i])
                level_codes = np.asarray(unique_named_levels.codes[i]).astype(
                    "int64"
                )
                level_codes[level_codes == -1] = np.iinfo(SIZE_TYPE_DTYPE).min
                cols.append(
                    level_col.take(as_column(level_codes), nullify=True)
                )
        with access_columns(*cols, mode="read", scope="internal"):
            plc_table = plc.reshape.tile(
                plc.Table([col.plc_column for col in cols]),
                self.shape[0],
            )
            tiled_index = [
                ColumnBase.create(plc_col, dtype=src_col.dtype)
                for src_col, plc_col in zip(
                    cols, plc_table.columns(), strict=True
                )
            ]

        # Assemble the final index — build levels/codes first so the
        # MultiIndex can be constructed in one step via _simple_new.
        # Codes/levels are attached eagerly, matching how pandas' stack builds
        # the result MultiIndex, so a later unstack can restore the original
        # row/column order (lazy materialization would sort the levels):
        # the original index contributes its own levels/codes (repeated);
        # a flat original index and the tiled stacked level(s) get
        # appearance-order factorization.
        index_names = [*self.index.names, *unique_named_levels.names]
        new_levels: list[cudf.Index] = []
        new_codes: list[ColumnBase] = []
        n_tile = len(unique_named_levels)
        if isinstance(self.index, MultiIndex):
            src = self.index._maybe_materialize_codes_and_levels()
            for src_level, src_code in zip(
                src._levels,
                src._codes,
                strict=True,
            ):
                new_levels.append(src_level)
                new_codes.append(
                    Index._from_column(src_code.astype(np.dtype(np.int64)))
                    .repeat(n_tile)
                    ._column
                )
        else:
            code, cats = factorize(self.index)
            new_levels.append(cats)
            new_codes.append(
                Index._from_column(as_column(code).astype(np.dtype(np.int64)))
                .repeat(n_tile)
                ._column
            )
        for tiled_col in tiled_index:
            code, cats = factorize(Index._from_column(tiled_col))
            new_codes.append(as_column(code).astype(np.dtype(np.int64)))
            new_levels.append(cats)
        new_index_columns = [*repeated_index._columns, *tiled_index]
        new_index = MultiIndex._simple_new(
            ColumnAccessor(dict(enumerate(new_index_columns))),
            new_levels,
            new_codes,
            pd.core.indexes.frozen.FrozenList(index_names),
        )

        # Compute the column indices that serves as the input for
        # `interleave_columns`
        column_idx_df = pd.DataFrame(
            data=range(self._num_columns), index=named_levels
        )

        if has_unnamed_levels:
            # the columns axis has multiple levels here, so column_name_idx
            # is always a pd.MultiIndex; build from codes/levels to keep the
            # level dtypes and to resolve the levels positionally
            unnamed_level_values = pd.MultiIndex(
                levels=[
                    column_name_idx.levels[i] for i in unnamed_levels_indices
                ],
                codes=[
                    column_name_idx.codes[i] for i in unnamed_levels_indices
                ],
                names=[
                    column_name_idx.names[i] for i in unnamed_levels_indices
                ],
                verify_integrity=False,
            )

        def unnamed_group_generator():
            if has_unnamed_levels:
                # sort=False iterates groups in first-appearance order, i.e.
                # exactly ``unnamed_level_values.unique()`` order (also for
                # NaN-containing tuple keys, which sorted groupby would
                # reorder via codes), so the stacked columns can be zipped
                # 1:1 with those keys when assembling the result.
                for _, grpdf in column_idx_df.groupby(
                    by=unnamed_level_values, sort=False, dropna=False
                ):
                    # When stacking part of the levels, some combinations
                    # of keys may not be present in this group but can be
                    # present in others. Reindexing with the globally computed
                    # `unique_named_levels` assigns -1 to these key
                    # combinations, representing an all-null column that
                    # is used in the subsequent libcudf call.
                    # ``reindex`` returns rows in target order, so the
                    # legacy path needs no further sorting (the target was
                    # already sorted above).
                    yield grpdf.reindex(
                        unique_named_levels, axis=0, fill_value=-1
                    ).values
            else:
                if future_stack or self._data.nlevels == 1:
                    yield column_idx_df.values
                else:
                    yield column_idx_df.reindex(unique_named_levels).values

        # For each of the group constructed from the unnamed levels,
        # invoke `interleave_columns` to stack the values.
        stacked = []

        for column_idx in unnamed_group_generator():
            # Collect columns based on indices, append None for -1 indices.
            columns = [
                None if i == -1 else self._data.select_by_index(i).columns[0]
                for i in column_idx
            ]

            # Collect datatypes and cast columns as that type
            common_type = find_common_type(
                [col.dtype for col in columns if col is not None]
            )

            all_nulls = functools.cache(
                functools.partial(column_empty, self.shape[0], common_type)
            )

            # homogenize the dtypes of the columns
            homogenized = [
                col.astype(common_type) if col is not None else all_nulls()
                for col in columns
            ]
            if (
                cudf.get_option("mode.pandas_compatible")
                and common_type == "object"
            ):
                for col, hcol in zip(columns, homogenized, strict=True):
                    if is_mixed_with_object_dtype(col, hcol):
                        raise TypeError(
                            "Stacking a DataFrame with mixed object and "
                            "non-object dtypes is not supported. "
                        )

            with access_columns(  # type: ignore[assignment]
                *homogenized, mode="read", scope="internal"
            ) as homogenized:
                interleaved_col = plc.reshape.interleave_columns(
                    plc.Table([col.plc_column for col in homogenized])
                )
            stacked.append(ColumnBase.create(interleaved_col, common_type))

        # Construct the resulting dataframe / series
        if not has_unnamed_levels:
            result = Series._from_column(
                stacked[0], index=new_index, attrs=self.attrs
            )
        else:
            if unnamed_level_values.nlevels == 1:
                unnamed_level_values = unnamed_level_values.get_level_values(0)
            unnamed_level_values = unnamed_level_values.unique()

            if isinstance(unnamed_level_values, pd.MultiIndex):
                # build the labels from levels/codes to preserve scalar
                # types: iterating a MultiIndex materializes e.g. an int64
                # level containing a missing entry as float
                keys: list[tuple[Any, ...]] = [
                    tuple(
                        unnamed_level_values.levels[j][c]
                        if c != -1
                        else np.nan
                        for j, c in enumerate(row)
                    )
                    for row in zip(*unnamed_level_values.codes, strict=True)
                ]
            else:
                keys = unnamed_level_values

            # ``stacked`` is in group first-appearance order (groupby with
            # sort=False above), which is exactly the order of
            # ``unnamed_level_values.unique()``: zip 1:1.
            data = ColumnAccessor(
                dict(zip(keys, stacked, strict=True)),
                isinstance(unnamed_level_values, pd.MultiIndex),
                unnamed_level_values.names,
                label_dtype=(
                    None
                    if isinstance(unnamed_level_values, pd.MultiIndex)
                    else unnamed_level_values.dtype
                ),
                level_dtypes=_pd_index_level_dtypes(unnamed_level_values),
            )

            result = DataFrame._from_data(
                data, index=new_index, attrs=self.attrs
            )

        if not future_stack and dropna:
            # Compute the row mask explicitly so the eagerly-attached
            # codes can be subset alongside the data; pandas keeps the full
            # pre-drop level set through dropna.
            # _apply_boolean_mask propagates pre-set levels/codes on the
            # index automatically.
            if isinstance(result, Series):
                keep = result.notna()
            else:
                keep = ~result.isna().all(axis=1)
            return result._apply_boolean_mask(
                BooleanMask(keep._column, len(result))
            )
        else:
            return result

    @_performance_tracking
    def cov(self, min_periods=None, ddof: int = 1, numeric_only: bool = False):
        """Compute the covariance matrix of a DataFrame.

        Parameters
        ----------
        min_periods : int, optional
            Minimum number of observations required per pair of columns to
            have a valid result.
            Currently not supported.

        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
            Currently not supported.

        Returns
        -------
        cov : DataFrame
        """
        if min_periods is not None:
            raise NotImplementedError(
                "min_periods is currently not supported."
            )

        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        if any(col.has_nulls(include_nan=True) for col in self._columns):
            raise NotImplementedError("cupy-based cov does not support nulls")

        cov = cupy.cov(self.values, ddof=ddof, rowvar=False)
        cols = self._data.to_pandas_index
        df = DataFrame(cupy.asfortranarray(cov), index=cols)
        df._set_columns_like(self._data)
        df._attrs = self.attrs
        return df

    def corr(
        self, method="pearson", min_periods=None, numeric_only: bool = False
    ):
        """Compute the correlation matrix of a DataFrame.

        Parameters
        ----------
        method : {'pearson', 'spearman'}, default 'pearson'
            Method used to compute correlation:

            - pearson : Standard correlation coefficient
            - spearman : Spearman rank correlation

        min_periods : int, optional
            Minimum number of observations required per pair of columns to
            have a valid result.

        Returns
        -------
        DataFrame
            The requested correlation matrix.
        """
        if any(col.has_nulls(include_nan=True) for col in self._columns):
            raise NotImplementedError("cupy-based corr does not support nulls")

        if method == "pearson":
            values = self.values
        elif method == "spearman":
            values = self.rank().values
        else:
            raise ValueError("method must be either 'pearson', 'spearman'")

        if min_periods is not None:
            raise NotImplementedError("Unsupported argument 'min_periods'")

        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        corr = cupy.corrcoef(values, rowvar=False)
        cols = self._data.to_pandas_index
        df = DataFrame(cupy.asfortranarray(corr), index=cols)
        df._set_columns_like(self._data)
        df._attrs = self.attrs
        return df

    @_performance_tracking
    def to_struct(self, name=None):
        """
        Return a struct Series composed of the columns of the DataFrame.

        Parameters
        ----------
        name: optional
            Name of the resulting Series

        Notes
        -----
        Note: a copy of the columns is made.
        """
        if not all(isinstance(name, str) for name in self._column_names):
            warnings.warn(
                "DataFrame contains non-string column name(s). Struct column "
                "requires field name to be string. Non-string column names "
                "will be casted to string as the field name."
            )
        dtype = StructDtype(
            fields={str(name): dtype for name, dtype in self._dtypes}
        )
        if self._num_columns == 0:
            col = column_empty(len(self), dtype=dtype)
        else:
            first_null_count = self._columns[0].null_count
            children = (
                col.copy(deep=True).plc_column for col in self._columns
            )
            if all(
                col.null_count == first_null_count for col in self._columns
            ):
                plc_column = plc.Column.struct_from_children(children)
            else:
                plc_column = plc.Column(
                    plc.DataType(plc.TypeId.STRUCT),
                    len(self),
                    None,
                    None,
                    0,
                    0,
                    list(children),
                )
            col = ColumnBase.create(plc_column, dtype)
        return Series._from_column(
            col,
            index=self.index,
            name=name,
        )

    @_performance_tracking
    def keys(self):
        """
        Get the columns.
        This is index for Series, columns for DataFrame.

        Returns
        -------
        Index
            Columns of DataFrame.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'one' : [1, 2, 3], 'five' : ['a', 'b', 'c']})
        >>> df
           one five
        0    1    a
        1    2    b
        2    3    c
        >>> df.keys()
        Index(['one', 'five'], dtype='str')
        >>> df = cudf.DataFrame(columns=[0, 1, 2, 3])
        >>> df
        Empty DataFrame
        Columns: [0, 1, 2, 3]
        Index: []
        >>> df.keys()
        Index([0, 1, 2, 3], dtype='int64')
        """
        return self._data.to_pandas_index

    def itertuples(self, index=True, name="Pandas"):
        """
        Iteration is unsupported.

        See :ref:`iteration <pandas-comparison/iteration>` for more
        information.
        """
        raise TypeError(
            "cuDF does not support iteration of DataFrame "
            "via itertuples. Consider using "
            "`.to_pandas().itertuples()` "
            "if you wish to iterate over namedtuples."
        )

    def iterrows(self):
        """
        Iteration is unsupported.

        See :ref:`iteration <pandas-comparison/iteration>` for more
        information.
        """
        raise TypeError(
            "cuDF does not support iteration of DataFrame "
            "via iterrows. Consider using "
            "`.to_pandas().iterrows()` "
            "if you wish to iterate over each row."
        )

    @_performance_tracking
    @docutils.copy_docstring(reshape.pivot)
    def pivot(self, *, columns, index=no_default, values=no_default):
        return reshape.pivot(self, index=index, columns=columns, values=values)

    @_performance_tracking
    @docutils.copy_docstring(reshape.pivot_table)
    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=None,
        margins_name="All",
        observed=False,
        sort=True,
    ):
        return reshape.pivot_table(
            self,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
            sort=sort,
        )

    @_performance_tracking
    @docutils.copy_docstring(reshape.unstack)
    def unstack(self, level=-1, fill_value=None, sort: bool = True):
        return reshape.unstack(
            self, level=level, fill_value=fill_value, sort=sort
        )

    @_performance_tracking
    def explode(self, column, ignore_index=False):
        """
        Transform each element of a list-like to a row, replicating index
        values.

        Parameters
        ----------
        column : str
            Column to explode.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     "a": [[1, 2, 3], [], None, [4, 5]],
        ...     "b": [11, 22, 33, 44],
        ... })
        >>> df
                   a   b
        0  [1, 2, 3]  11
        1         []  22
        2       None  33
        3     [4, 5]  44
        >>> df.explode('a')
              a   b
        0     1  11
        0     2  11
        0     3  11
        1  <NA>  22
        2  <NA>  33
        3     4  44
        3     5  44
        """
        return super()._explode(column, ignore_index)

    def nunique(self, axis=0, dropna: bool = True) -> Series:
        """
        Count number of distinct elements in specified axis.
        Return Series with number of distinct elements. Can ignore NaN values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
            column-wise.
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A': [4, 5, 6], 'B': [4, 1, 1]})
        >>> df.nunique()
        A    3
        B    2
        dtype: int64
        """
        if axis != 0:
            raise NotImplementedError("axis parameter is not supported yet.")
        counts = [col.distinct_count(dropna=dropna) for col in self._columns]
        res = self._constructor_sliced(
            counts,
            index=self._data.to_pandas_index,
            dtype="float64" if len(counts) == 0 else None,
        )
        res._attrs = self.attrs
        return res

    def _sample_axis_1(
        self,
        n: int,
        weights: ColumnLike | None,
        replace: bool,
        random_state: np.random.RandomState,
        ignore_index: bool,
    ):
        if replace:
            # Since cuDF does not support multiple columns with same name,
            # sample with replace=True at axis 1 is unsupported.
            raise NotImplementedError(
                "Sample is not supported for axis 1/`columns` when"
                "`replace=True`."
            )

        sampled_column_labels = random_state.choice(
            self._column_names, size=n, replace=False, p=weights
        )

        result = self._get_columns_by_label(sampled_column_labels)
        if ignore_index:
            result.reset_index(drop=True)

        return result

    def _from_columns_like_self(
        self,
        columns: list[ColumnBase],
        column_names: Iterable[str] | None = None,
        index_names: list[str] | None = None,
    ) -> Self:
        result = super()._from_columns_like_self(
            columns,
            column_names,
            index_names,
        )
        result._set_columns_like(self._data)
        return result

    @_performance_tracking
    def interleave_columns(self):
        """
        Interleave Series columns of a table into a single column.

        Converts the column major table `cols` into a row major column.

        Parameters
        ----------
        cols : input Table containing columns to interleave.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({0: ['A1', 'A2', 'A3'], 1: ['B1', 'B2', 'B3']})
        >>> df
            0   1
        0  A1  B1
        1  A2  B2
        2  A3  B3
        >>> df.interleave_columns()
        0    A1
        1    B1
        2    A2
        3    B2
        4    A3
        5    B3
        dtype: str

        Returns
        -------
        The interleaved columns as a single column

        .. pandas-compat::
            `pandas.DataFrame.interleave_columns`

            This method does not exist in pandas but it can be run
            as ``pd.Series(np.vstack(df.to_numpy()).reshape((-1,)))``.
        """
        if any(
            isinstance(dtype, CategoricalDtype) for _, dtype in self._dtypes
        ):
            raise ValueError(
                "interleave_columns does not support 'category' dtype."
            )
        with access_columns(*self._columns, mode="read", scope="internal"):
            _, result_dtype = next(iter(self._dtypes))
            result_col = ColumnBase.create(
                plc.reshape.interleave_columns(
                    plc.Table([col.plc_column for col in self._columns])
                ),
                result_dtype,
            )
        return self._constructor_sliced._from_column(result_col)

    def _compute_column(self, expr: str) -> ColumnBase:
        """Helper function for eval"""
        with access_columns(*self._columns, mode="read", scope="internal"):
            plc_column = plc.transform.compute_column(
                plc.Table([col.plc_column for col in self._columns]),
                plc.expressions.to_expression(expr, self._column_names),
            )
            return ColumnBase.create(
                plc_column, dtype=dtype_from_pylibcudf_column(plc_column)
            )

    @_performance_tracking
    def eval(self, expr: str, inplace: bool = False, **kwargs):
        """Evaluate a string describing operations on DataFrame columns.

        Operates on columns only, not specific rows or elements.

        Parameters
        ----------
        expr : str
            The expression string to evaluate.
        inplace : bool, default False
            If the expression contains an assignment, whether to perform the
            operation inplace and mutate the existing DataFrame. Otherwise,
            a new DataFrame is returned.
        **kwargs
            Not supported.

        Returns
        -------
        DataFrame, Series, or None
            Series if a single column is returned (the typical use case),
            DataFrame if any assignment statements are included in
            ``expr``, or None if ``inplace=True``.


        Examples
        --------
        >>> df = cudf.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2)})
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2
        >>> df.eval('A + B')
        0    11
        1    10
        2     9
        3     8
        4     7
        dtype: int64

        Assignment is allowed though by default the original DataFrame is not
        modified.

        >>> df.eval('C = A + B')
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2

        Use ``inplace=True`` to modify the original DataFrame.

        >>> df.eval('C = A + B', inplace=True)
        >>> df
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7

        Multiple columns can be assigned to using multi-line expressions:

        >>> df.eval(
        ...     '''
        ... C = A + B
        ... D = A - B
        ... '''
        ... )
           A   B   C  D
        0  1  10  11 -9
        1  2   8  10 -6
        2  3   6   9 -3
        3  4   4   8  0
        4  5   2   7  3

        .. pandas-compat::
            :meth:`pandas.DataFrame.eval`

            * Additional kwargs are not supported.
            * Bitwise and logical operators are not dtype-dependent.
              Specifically, `&` must be used for bitwise operators on integers,
              not `and`, which is specifically for the logical and between
              booleans.
            * Only numerical types are currently supported.
            * Operators generally will not cast automatically. Users are
              responsible for casting columns to suitable types before
              evaluating a function.
            * Multiple assignments to the same name (i.e. a sequence of
              assignment statements where later statements are conditioned upon
              the output of earlier statements) is not supported.
        """
        if kwargs:
            raise ValueError(
                "Keyword arguments other than `inplace` are not supported"
            )

        # Have to use a regex match to avoid capturing ==, >=, or <=
        equals_sign_regex = "[^=><]=[^=]"
        includes_assignment = re.search(equals_sign_regex, expr) is not None

        # Check if there were multiple statements. Filter out empty lines.
        statements = tuple(filter(None, expr.strip().split("\n")))
        if len(statements) > 1 and any(
            re.search(equals_sign_regex, st) is None for st in statements
        ):
            raise ValueError(
                "Multi-line expressions are only valid if all expressions "
                "contain an assignment."
            )

        if not includes_assignment:
            if inplace:
                raise ValueError(
                    "Cannot operate inplace if there is no assignment"
                )
            return Series._from_column(self._compute_column(statements[0]))

        targets = []
        exprs = []
        for st in statements:
            try:
                t, e = re.split("[^=]=[^=]", st)
            except ValueError as err:
                if "too many values" in str(err):
                    raise ValueError(
                        f"Statement {st} contains too many assignments ('=')"
                    )
                raise
            targets.append(t.strip())
            exprs.append(e.strip())

        ret = self if inplace else self.copy(deep=False)
        for name, expr in zip(targets, exprs, strict=True):
            ret._data[name] = self._compute_column(expr)
        if not inplace:
            return ret

    def value_counts(
        self,
        subset=None,
        normalize=False,
        sort=True,
        ascending=False,
        dropna=True,
    ):
        """
        Return a Series containing counts of unique rows in the DataFrame.

        Parameters
        ----------
        subset: list-like, optional
            Columns to use when counting unique combinations.
        normalize: bool, default False
            Return proportions rather than frequencies.
        sort: bool, default True
            Sort by frequencies.
        ascending: bool, default False
            Sort in ascending order.
        dropna: bool, default True
            Don't include counts of rows that contain NA values.

        Returns
        -------
        Series

        Notes
        -----
        The returned Series will have a MultiIndex with one level per input
        column. By default, rows that contain any NA values are omitted from
        the result. By default, the resulting Series will be in descending
        order so that the first element is the most frequently-occurring row.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'num_legs': [2, 4, 4, 6],
        ...                    'num_wings': [2, 0, 0, 0]},
        ...                    index=['falcon', 'dog', 'cat', 'ant'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0
        cat            4          0
        ant            6          0
        >>> df.value_counts().sort_index()
        num_legs  num_wings
        2         2            1
        4         0            2
        6         0            1
        Name: count, dtype: int64
        """
        if subset:
            diff = set(subset) - set(self._data)
            if len(diff) != 0:
                raise KeyError(f"columns {diff} do not exist")
        columns = list(self._column_names) if subset is None else subset
        name = "proportion" if normalize else "count"

        df = self[columns].copy()
        taken = set(df._column_names)

        def _free_name(base: str) -> str:
            while base in taken:
                base = f"_{base}"
            taken.add(base)
            return base

        cnt_col = _free_name("__count")
        pos_col = _free_name("__pos")
        # cudf's groupby does not preserve first-appearance order, so track the
        # first row index of each unique row and order by it, matching pandas'
        # stable handling of tied counts.
        df[cnt_col] = 1
        df[pos_col] = as_column(range(len(df)))
        result = (
            df.groupby(by=columns, dropna=dropna, sort=False)
            .agg({cnt_col: "count", pos_col: "min"})
            .reset_index()
        )
        result[cnt_col] = result[cnt_col].astype(np.dtype(np.int64))
        result = result.sort_values(pos_col)
        if sort:
            seq_col = _free_name("__seq")
            result[seq_col] = as_column(range(len(result)))
            result = result.sort_values(
                [cnt_col, seq_col], ascending=[ascending, True]
            )
        result = result.set_index(columns)[cnt_col]
        if normalize:
            result = result / result._column.sum()
        # Pandas always returns MultiIndex even if only one column.
        if not isinstance(result.index, MultiIndex):
            result.index = MultiIndex._from_data(result.index._data)
        result.name = name
        return result

    @_performance_tracking
    def to_pylibcudf(self) -> tuple[plc.Table, dict]:
        """
        Convert this DataFrame to a pylibcudf.Table.

        Returns
        -------
        pylibcudf.Table
            A pylibcudf.Table referencing the same data.
        dict
            Dict of metadata (includes column names and dataframe indices)

        Notes
        -----
        This is always a zero-copy operation. The result is a view of the
        existing data. Changes to the pylibcudf data will be reflected back
        to the cudf object and vice versa.
        """
        metadata = {"index": self.index, "columns": self._data.to_pandas_index}
        return plc.Table(
            [col.to_pylibcudf() for col in self._columns]
        ), metadata

    @classmethod
    @_performance_tracking
    def from_pylibcudf(
        cls,
        table: plc.Table | plc.io.TableWithMetadata,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """
        Create a DataFrame from a pylibcudf.Table.

        Parameters
        ----------
        table : pylibcudf.Table, pylibcudf.io.TableWithMetadata
            The input Table.
        metadata : dict, default None
            Metadata necessary to reconstruct the dataframe
            if table is a pylibcudf.Table.

        Returns
        -------
        table : cudf.DataFrame
            A cudf.DataFrame referencing the columns in the pylibcudf.Table.

        Notes
        -----
        This function will generate a DataFrame which contains a tuple of columns
        pointing to the same columns the input table points to.  It will directly access
        the data and mask buffers of the pylibcudf columns, so the newly created
        object is not tied to the lifetime of the original pylibcudf.Table.
        """
        if isinstance(table, plc.io.TableWithMetadata):
            tbl = table.tbl
            if metadata is not None:
                raise ValueError(
                    "metadata must be None when table is a pylibcudf.io.TableWithMetadata"
                )
            column_names = table.column_names(include_children=False)
            child_names = table.child_names
            index = None
        elif isinstance(table, plc.Table):
            tbl = table
            if not (
                isinstance(metadata, dict)
                and 1 <= len(metadata) <= 3
                and "columns" in metadata
                and (not set(metadata) - {"columns", "child_names", "index"})
            ):
                raise ValueError(
                    "Must pass a metadata dict with keys 'columns' and "
                    "optionally 'child_names' and 'index' only "
                    "when table is a pylibcudf.Table "
                )
            column_names = metadata["columns"]
            child_names = metadata.get("child_names")  # type: ignore[assignment]
            index = metadata.get("index")
        else:
            raise ValueError(
                "table must be a pylibcudf.Table or pylibcudf.io.TableWithMetadata"
            )

        columns = tbl.columns()
        normalized_columns = [_normalize_types_column(col) for col in columns]
        if not all(
            normalized_col is col
            for normalized_col, col in zip(
                normalized_columns, columns, strict=True
            )
        ):
            tbl = plc.Table(normalized_columns)

        plc_columns = tbl.columns()
        cudf_cols = (
            ColumnBase.create(
                plc_col, dtype=dtype_from_pylibcudf_column(plc_col)
            )
            for plc_col in plc_columns
        )
        # We only have child names if the source is a pylibcudf.io.TableWithMetadata.
        if child_names is not None:
            cudf_cols = (
                ColumnBase.create(
                    col.plc_column,
                    recursively_update_struct_names(col.dtype, cn),
                )
                for col, cn in zip(
                    cudf_cols, child_names.values(), strict=True
                )
            )
        col_accessor = ColumnAccessor(
            {
                name: cudf_col
                for name, cudf_col in zip(column_names, cudf_cols, strict=True)
            },
            verify=False,
            rangeindex=len(plc_columns) == 0,
        )
        return cls._from_data(col_accessor, index=index)


def make_binop_func(op, postprocess=None):
    # This function is used to wrap binary operations in Frame with an
    # appropriate API for DataFrame as required for pandas compatibility. The
    # main effect is reordering and error-checking parameters in
    # DataFrame-specific ways. The postprocess argument is a callable that may
    # optionally be provided to modify the result of the binop if additional
    # processing is needed for pandas compatibility. The callable must have the
    # signature
    # def postprocess(left, right, output)
    # where left and right are the inputs to the binop and output is the result
    # of calling the wrapped Frame binop.
    wrapped_func = getattr(IndexedFrame, op)

    @functools.wraps(wrapped_func)
    def wrapper(self, other, axis="columns", level=None, fill_value=None):
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")
        output = wrapped_func(self, other, axis, level, fill_value)
        if postprocess is None:
            return output
        return postprocess(self, other, output)

    # functools.wraps copies module level attributes to `wrapper` and sets
    # __wrapped__ attributes to `wrapped_func`. Cpython looks up the signature
    # string of a function by recursively delving into __wrapped__ until
    # it hits the first function that has __signature__ attribute set. To make
    # the signature string of `wrapper` matches with its actual parameter list,
    # we directly set the __signature__ attribute of `wrapper` below.

    new_sig = inspect.signature(
        lambda self, other, axis="columns", level=None, fill_value=None: None
    )

    wrapper.__signature__ = new_sig
    return wrapper


# Wrap arithmetic Frame binop functions with the expected API for Series.
for binop in [
    "add",
    "radd",
    "subtract",
    "sub",
    "rsub",
    "multiply",
    "mul",
    "rmul",
    "mod",
    "rmod",
    "pow",
    "rpow",
    "floordiv",
    "rfloordiv",
    "truediv",
    "div",
    "divide",
    "rtruediv",
    "rdiv",
]:
    setattr(DataFrame, binop, make_binop_func(binop))


def _make_replacement_func(value):
    # This function generates a postprocessing function suitable for use with
    # make_binop_func that fills null columns with the desired fill value.

    def func(left, right, output):
        # This function may be passed as the postprocess argument to
        # make_binop_func. Columns that are only present in one of the inputs
        # will be null in the output. This function postprocesses the output to
        # replace those nulls with some desired output.
        if isinstance(right, Series):
            uncommon_columns = set(left._column_names) ^ set(right.index)
        elif isinstance(right, DataFrame):
            uncommon_columns = set(left._column_names) ^ set(
                right._column_names
            )
        elif _is_scalar_or_zero_d_array(right):
            for name, col in output._column_labels_and_values:
                output._data[name] = col.fillna(value)
            return output
        else:
            return output

        for name in uncommon_columns:
            output._data[name] = as_column(
                value, length=len(output), dtype="bool"
            )
        return output

    return func


# The ne comparator needs special postprocessing because elements that missing
# in one operand should be treated as null and result in True in the output
# rather than simply propagating nulls.
DataFrame.ne = make_binop_func("ne", _make_replacement_func(True))


# All other comparison operators needs return False when one of the operands is
# missing in the input.
for binop in [
    "eq",
    "lt",
    "le",
    "gt",
    "ge",
]:
    setattr(
        DataFrame, binop, make_binop_func(binop, _make_replacement_func(False))
    )


@_performance_tracking
def from_pandas(obj, nan_as_null=no_default):
    """
    Convert certain Pandas objects into the cudf equivalent.

    Supports DataFrame, Series, Index, or MultiIndex.

    Returns
    -------
    DataFrame/Series/Index/MultiIndex
        Return type depends on the passed input.

    Raises
    ------
    TypeError for invalid input type.

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> data = [[0, 1], [1, 2], [3, 4]]
    >>> pdf = pd.DataFrame(data, columns=['a', 'b'], dtype=int)
    >>> pdf
       a  b
    0  0  1
    1  1  2
    2  3  4
    >>> gdf = cudf.from_pandas(pdf)
    >>> gdf
       a  b
    0  0  1
    1  1  2
    2  3  4
    >>> type(gdf)
    <class 'cudf.core.dataframe.DataFrame'>
    >>> type(pdf)
    <class 'pandas.DataFrame'>

    Converting a Pandas Series to cuDF Series:

    >>> psr = pd.Series(['a', 'b', 'c', 'd'], name='apple', dtype='str')
    >>> psr
    0    a
    1    b
    2    c
    3    d
    Name: apple, dtype: str
    >>> gsr = cudf.from_pandas(psr)
    >>> gsr
    0    a
    1    b
    2    c
    3    d
    Name: apple, dtype: str
    >>> type(gsr)
    <class 'cudf.core.series.Series'>
    >>> type(psr)
    <class 'pandas.Series'>

    Converting a Pandas Index to cuDF Index:

    >>> pidx = pd.Index([1, 2, 10, 20])
    >>> pidx
    Index([1, 2, 10, 20], dtype='int64')
    >>> gidx = cudf.from_pandas(pidx)
    >>> gidx
    Index([1, 2, 10, 20], dtype='int64')
    >>> type(gidx)
    <class 'cudf.core.index.Index'>
    >>> type(pidx)
    <class 'pandas.Index'>

    Converting a Pandas MultiIndex to cuDF MultiIndex:

    >>> pmidx = pd.MultiIndex(
    ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
    ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
    ...         names=["x", "y"],
    ...     )
    >>> pmidx
    MultiIndex([(1, 1),
                (1, 5),
                (3, 2),
                (4, 2),
                (5, 1)],
               names=['x', 'y'])
    >>> gmidx = cudf.from_pandas(pmidx)
    >>> gmidx
    MultiIndex([(1, 1),
                (1, 5),
                (3, 2),
                (4, 2),
                (5, 1)],
               names=['x', 'y'])
    >>> type(gmidx)
    <class 'cudf.core.multiindex.MultiIndex'>
    >>> type(pmidx)
    <class 'pandas.MultiIndex'>
    """
    if nan_as_null is no_default:
        nan_as_null = False if get_option("mode.pandas_compatible") else None

    if isinstance(obj, pd.DataFrame):
        return DataFrame(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.Series):
        return Series(obj, nan_as_null=nan_as_null)
    # This carveout for cudf.pandas is undesirable, but fixes crucial issues
    # for core RAPIDS projects like cuML and cuGraph that rely on
    # `cudf.from_pandas`, so we allow it for now.
    elif (ret := getattr(obj, "_fsproxy_wrapped", None)) is not None:
        return ret
    elif isinstance(obj, pd.MultiIndex):
        return MultiIndex(
            levels=obj.levels,
            codes=obj.codes,
            names=obj.names,
            nan_as_null=nan_as_null,
        )
    elif isinstance(obj, pd.RangeIndex):
        return RangeIndex(
            start=obj.start, stop=obj.stop, step=obj.step, name=obj.name
        )
    elif isinstance(obj, pd.CategoricalIndex):
        return CategoricalIndex(obj, name=obj.name, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.DatetimeIndex):
        return DatetimeIndex(obj, name=obj.name, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.TimedeltaIndex):
        return TimedeltaIndex(obj, name=obj.name, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.IntervalIndex):
        return IntervalIndex(obj, name=obj.name, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.Index):
        return Index(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.CategoricalDtype):
        return CategoricalDtype(obj.categories, obj.ordered)
    elif isinstance(obj, pd.IntervalDtype):
        return IntervalDtype(obj.subtype, obj.closed)
    else:
        raise TypeError(
            f"from_pandas unsupported for object of type {type(obj).__name__}"
        )


@_performance_tracking
def merge(left, right, *args, **kwargs):
    if isinstance(left, Series):
        left = left.to_frame()
    return left.merge(right, *args, **kwargs)


# a bit of fanciness to inject docstring with left parameter
merge_doc = DataFrame.merge.__doc__
if merge_doc is not None:
    idx = merge_doc.find("right")
    merge.__doc__ = "".join(
        [
            merge_doc[:idx],
            "\n\tleft : Series or DataFrame\n\t",
            merge_doc[idx:],
        ]
    )


def _align_indices(lhs, rhs):
    """
    Internal util to align the indices of two DataFrames. Returns a tuple of
    the aligned dataframes, or the original arguments if the indices are the
    same, or if rhs isn't a DataFrame.
    """
    lhs_out, rhs_out = lhs, rhs
    if isinstance(rhs, DataFrame) and not lhs.index.equals(rhs.index):
        df = lhs.merge(
            rhs,
            sort=True,
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("_x", "_y"),
        )
        df = df.sort_index()
        lhs_out = DataFrame(index=df.index)
        rhs_out = DataFrame(index=df.index)
        common = set(lhs._column_names) & set(rhs._column_names)
        common_x = {f"{x}_x": x for x in common}
        common_y = {f"{x}_y": x for x in common}
        for col in df._column_names:
            if col in common_x:
                lhs_out[common_x[col]] = df[col]
            elif col in common_y:
                rhs_out[common_y[col]] = df[col]
            elif col in lhs:
                lhs_out[col] = df[col]
            elif col in rhs:
                rhs_out[col] = df[col]

    return lhs_out, rhs_out


def _setitem_with_dataframe(
    input_df: DataFrame,
    replace_df: DataFrame,
    input_cols: Any = None,
    mask: ColumnBase | None = None,
    ignore_index: bool = False,
):
    """
    This function sets item dataframes relevant columns with replacement df
    :param input_df: Dataframe to be modified inplace
    :param replace_df: Replacement DataFrame to replace values with
    :param input_cols: columns to replace in the input dataframe
    :param mask: boolean mask in case of masked replacing
    :param ignore_index: Whether to conduct index equality and reindex
    """

    if input_cols is None:
        input_cols = input_df._column_names

    if len(input_cols) != replace_df._num_columns:
        raise ValueError(
            "Number of Input Columns must be same replacement Dataframe"
        )

    if (
        not ignore_index
        and len(input_df) != 0
        and not input_df.index.equals(replace_df.index)
    ):
        replace_df = replace_df.reindex(input_df.index)

    for col_1, col_2 in zip(input_cols, replace_df._column_names, strict=True):
        if col_1 in input_df._column_names:
            if mask is not None:
                input_df._data[col_1][mask] = as_column(replace_df[col_2])
            else:
                input_df._data[col_1] = as_column(replace_df[col_2])
        else:
            if mask is not None:
                raise ValueError("Can not insert new column with a bool mask")
            else:
                # handle append case
                input_df._insert(
                    loc=input_df._num_columns,
                    name=col_1,
                    value=replace_df[col_2],
                )


def _index_from_listlike_of_series(
    series_list: Sequence[Series],
) -> Index:
    names_list: Sequence[Hashable] = []
    unnamed_count = 0
    for series in series_list:
        if series.name is None:
            names_list.append(f"Unnamed {unnamed_count}")  # type: ignore[attr-defined]
            unnamed_count += 1
        else:
            names_list.append(series.name)  # type: ignore[attr-defined]
    if unnamed_count == len(series_list):
        names_list = range(len(series_list))

    return Index(names_list)


# Create a dictionary of the common, non-null columns
def _get_non_null_cols_and_dtypes(col_idxs, list_of_columns):
    # A mapping of {idx: np.dtype}
    dtypes = {}
    # A mapping of {idx: [...columns]}, where `[...columns]`
    # is a list of columns with at least one valid value for each
    # column name across all input frames
    non_null_columns = {}
    for idx in col_idxs:
        for cols in list_of_columns:
            # Skip columns not in this frame
            if idx >= len(cols) or cols[idx] is None:
                continue
            # Store the first dtype we find for a column, even if it's
            # all-null. This ensures we always have at least one dtype
            # for each name. This dtype will be overwritten later if a
            # non-null Column with the same name is found.
            if idx not in dtypes:
                dtypes[idx] = cols[idx].dtype
            if cols[idx].null_count != len(cols[idx]):
                if idx not in non_null_columns:
                    non_null_columns[idx] = [cols[idx]]
                else:
                    non_null_columns[idx].append(cols[idx])
    return non_null_columns, dtypes


def _find_common_dtypes_and_categories(
    non_null_columns, dtypes
) -> dict[Any, ColumnBase]:
    # A mapping of {idx: categories}, where `categories` is a
    # column of all the unique categorical values from each
    # categorical column across all input frames. This function
    # also modifies the input dtypes dictionary in place to capture
    # the common dtype across columns being concatenated.
    categories = {}
    for idx, cols in non_null_columns.items():
        # default to the first non-null dtype
        dtypes[idx] = cols[0].dtype
        # If all the non-null dtypes are int/float, find a common dtype
        if all(is_dtype_obj_numeric(col.dtype) for col in cols):
            dtypes[idx] = find_common_type([col.dtype for col in cols])
        # If all categorical dtypes, combine the categories
        elif all(isinstance(col.dtype, CategoricalDtype) for col in cols):
            # Combine and de-dupe the categories
            categories[idx] = concat_columns(
                [col.categories for col in cols]
            ).unique()
            # Set the column dtype to the codes' dtype. The categories
            # will be re-assigned at the end
            dtypes[idx] = min_unsigned_type(len(categories[idx]))
        # Otherwise raise an error if columns have different dtypes
        elif not all(is_dtype_equal(c.dtype, dtypes[idx]) for c in cols):
            raise ValueError("All columns must be the same type")
    return categories


def _cast_cols_to_common_dtypes(col_idxs, list_of_columns, dtypes, categories):
    # Cast all columns to a common dtype, assign combined categories,
    # and back-fill missing columns with all-null columns
    for idx in col_idxs:
        dtype = dtypes[idx]
        for cols in list_of_columns:
            # If column not in this df, fill with an all-null column
            if idx >= len(cols) or cols[idx] is None:
                n = len(next(x for x in cols if x is not None))
                cols[idx] = column_empty(row_count=n, dtype=dtype)
            else:
                # If column is categorical, rebase the codes with the
                # combined categories, and cast the new codes to the
                # min-scalar-sized dtype
                if idx in categories:
                    cols[idx] = (
                        cols[idx]
                        ._set_categories(
                            categories[idx],
                            is_unique=True,
                        )
                        .codes
                    )
                cols[idx] = cols[idx].astype(dtype)


def _reassign_categories(categories, cols, col_idxs):
    for name, idx in zip(cols, col_idxs, strict=True):
        if idx in categories:
            cols[name] = ColumnBase.create(
                cols[name].plc_column,
                CategoricalDtype(categories=categories[idx], ordered=False),
            )


def _from_dict_create_index(indexlist, namelist, library):
    if len(namelist) > 1:
        index = library.MultiIndex.from_tuples(indexlist, names=namelist)
    else:
        index = library.Index(indexlist, name=namelist[0])
    return index
