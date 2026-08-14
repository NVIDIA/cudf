# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for Frame types that have an index."""

from __future__ import annotations

import copy
import itertools
import textwrap
import warnings
from collections import Counter
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypeVar,
    cast,
    overload,
)
from uuid import uuid4

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf._lib import strings_udf
from cudf.api.extensions import no_default
from cudf.api.types import (
    is_dict_like,
    is_list_like,
    is_scalar,
    is_string_dtype,
)
from cudf.core._internals import copying
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    access_columns,
    as_column,
    column_empty,
)
from cudf.core.column.column import concat_columns
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.common import pipe
from cudf.core.copy_types import BooleanMask, GatherMap
from cudf.core.dtype.validators import (
    is_dtype_obj_list,
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.frame import Frame
from cudf.core.groupby.groupby import GroupBy
from cudf.core.index import Index, RangeIndex, _index_from_data, ensure_index
from cudf.core.missing import NA
from cudf.core.multiindex import MultiIndex
from cudf.core.resample import _Resampler
from cudf.core.udf.utils import (
    _get_input_args_from_frame,
    _make_free_string_kernel,
    _return_arr_from_dtype,
)
from cudf.core.window import ExponentialMovingWindow, Rolling
from cudf.errors import MixedTypeError
from cudf.utils import docutils, ioutils
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    SIZE_TYPE_DTYPE,
    can_convert_to_column,
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    find_common_type,
    get_dtype_of_same_kind,
    is_column_like,
    is_mixed_with_object_dtype,
    is_pandas_nullable_extension_dtype,
    is_pandas_nullable_numpy_dtype,
)
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.utils import _warn_no_dask_cudf

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        MutableMapping,
        Sequence,
    )
    from types import NotImplementedType

    from cudf._typing import (
        Axis,
        ColumnLike,
        DataFrameOrSeries,
        Dtype,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column.lists import ListColumn
    from cudf.core.series import Series


doc_reset_index_template = """
        Reset the index of the {klass}, or a level of it.

        Parameters
        ----------
        level : int, str, tuple, or list, default None
            Only remove the given levels from the index. Removes all levels by
            default.
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
{argument}
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).
        allow_duplicates : bool, default False
            Allow duplicate column labels to be created.
            Currently not supported.

        Returns
        -------
        {return_type}
            {klass} with the new index or None if ``inplace=True``.{return_doc}

        Examples
        --------
        {example}
"""


doc_binop_template = textwrap.dedent(
    """
    Get {operation} of DataFrame or Series and other, element-wise (binary
    operator `{op_name}`).

    Equivalent to ``frame + other``, but with support to substitute a
    ``fill_value`` for missing data in one of the inputs.

    Parameters
    ----------
    other : scalar, sequence, Series, or DataFrame
        Any single or multiple element data structure, or list-like object.
    axis : int or string
        Only ``0`` is supported for series, ``1`` or ``columns`` supported
        for dataframe
    level : int or name
        Broadcast across a level, matching Index values on the
        passed MultiIndex level. Not yet supported.
    fill_value  : float or None, default None
        Fill existing missing (NaN) values, and any new element needed
        for successful DataFrame alignment, with this value before
        computation. If data in both corresponding DataFrame locations
        is missing the result will be missing.

    Returns
    -------
    DataFrame or Series
        Result of the arithmetic operation.

    Examples
    --------

    **DataFrame**

    >>> df = cudf.DataFrame(
    ...     {{'angles': [0, 3, 4], 'degrees': [360, 180, 360]}},
    ...     index=['circle', 'triangle', 'rectangle']
    ... )
    {df_op_example}

    **Series**

    >>> a = cudf.Series([1, 1, 1, None], index=['a', 'b', 'c', 'd'])
    >>> b = cudf.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])
    {ser_op_example}
    """
)


def _get_unique_drop_labels(array):
    """Return labels to be dropped for IndexFrame.drop."""
    if isinstance(array, (cudf.Series, cudf.Index, ColumnBase)):
        yield from np.unique(as_column(array).to_numpy())
    elif is_scalar(array):
        yield array
    else:
        yield from set(array)


def _indices_from_labels(obj, labels):
    if not isinstance(labels, cudf.MultiIndex):
        labels = as_column(labels)
        labels = labels.astype(obj.index.dtype)
        idx_labels = cudf.Index._from_column(labels)
    else:
        idx_labels = labels

    # join is not guaranteed to maintain the index ordering
    # so we will sort it with its initial ordering which is stored
    # in column "__"
    lhs = cudf.DataFrame(
        {"__": ColumnBase.from_range(range(len(idx_labels)))}, index=idx_labels
    )
    rhs = cudf.DataFrame(
        {"_": ColumnBase.from_range(range(len(obj)))}, index=obj.index
    )
    result = lhs.join(rhs).sort_values(by=["__", "_"])["_"]
    if result.dtype != rhs._data["_"].dtype:
        # The merge upcasts the positional column to float64 when some
        # labels are missing; row positions are exactly representable in
        # float64, so restore the integer dtype (nulls are preserved).
        result = result.astype(rhs._data["_"].dtype)
    return result


def _gather_map_from_positions(positions, nrows: int) -> GatherMap:
    """Build a nullifying GatherMap from a joined positional column.

    A merge that leaves some rows unmatched upcasts a positional column to
    float64 with nulls; row positions are exactly representable in float64,
    so cast back and replace nulls with an out-of-bounds sentinel that a
    nullifying gather turns back into nulls.
    """
    positions = positions.astype(SIZE_TYPE_DTYPE)
    if positions.null_count:
        positions = positions.fillna(np.int32(np.iinfo(np.int32).min))
    return GatherMap.from_column_unchecked(positions, nrows, nullify=True)


def _unify_categorical_indexes(lhs_index, rhs_index):
    """Cast two unordered categorical indexes with differing categories to
    a common merged-categories dtype ahead of a join.

    The merged categories keep first-appearance order (the left categories
    followed by the right's unseen ones), so a sorted join on the unified
    codes orders rows by category -- like pandas' union of categoricals --
    rather than lexically, and the joined index remains categorical.
    """
    ldtype = getattr(lhs_index, "dtype", None)
    rdtype = getattr(rhs_index, "dtype", None)
    if (
        isinstance(ldtype, cudf.CategoricalDtype)
        and isinstance(rdtype, cudf.CategoricalDtype)
        and not ldtype.ordered
        and not rdtype.ordered
        and not ldtype._internal_eq(rdtype)
    ):
        merged_categories = cudf.concat(
            [ldtype.categories, rdtype.categories]
        ).unique()
        common = cudf.CategoricalDtype(
            categories=merged_categories, ordered=False
        )
        return lhs_index.astype(common), rhs_index.astype(common)
    return lhs_index, rhs_index


class _FrameIndexer:
    """Parent class for indexers."""

    def __init__(self, frame):
        self._frame = frame

    def append_new_row(self, key, value, columns_df=None, column=None):
        idx = self._frame.index

        # Normalize key for tuple/list or int
        key_val = key[0] if isinstance(key, (tuple, list)) else key

        if isinstance(idx, RangeIndex):
            if isinstance(key_val, int) and (key_val == idx[-1] + idx.step):
                idx_copy = cudf.RangeIndex(
                    start=idx.start,
                    stop=idx.stop + idx.step,
                    step=idx.step,
                    name=idx.name,
                )
            else:
                int_idx = idx._as_int_index()
                new_index_col = _append_new_row(int_idx._column, key_val)
                idx_copy = cudf.Index._from_column(
                    new_index_col, name=int_idx.name
                )
        else:
            new_index_col = _append_new_row(idx._column, key_val)
            idx_copy = cudf.Index._from_column(new_index_col, name=idx.name)

        # Append value(s) to column(s) without mutating the existing
        # column object. Column objects can be shared between Series
        # and DataFrame views, so in-place mutation would corrupt views.
        # Bypass ColumnAccessor's length-validation by writing to the
        # underlying mapping directly; both `nrows` and the cached
        # `columns` tuple are invalidated so the frame's view of its
        # columns reflects the new lengths.
        old_ncols = len(self._frame._data)
        if columns_df is not None:
            for col in columns_df._column_names:
                self._frame._data._data[col] = _append_new_row(
                    self._frame._data[col], value
                )
        elif column is not None:
            new_col = _append_new_row(self._frame._column, value)
            self._frame._data._data[self._frame.name] = new_col

        try:
            del self._frame._data.nrows
        except AttributeError:
            pass
        self._frame._data._clear_cache(old_ncols, len(self._frame._data))
        self._frame._index = idx_copy


_LocIndexerClass = TypeVar("_LocIndexerClass", bound="_FrameIndexer")
_IlocIndexerClass = TypeVar("_IlocIndexerClass", bound="_FrameIndexer")


class IndexedFrame(Frame):
    """A frame containing an index.

    This class encodes the common behaviors for core user-facing classes like
    DataFrame and Series that consist of a sequence of columns along with a
    special set of index columns.

    Parameters
    ----------
    data : dict
        An dict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    # mypy can't handle bound type variables as class members
    _loc_indexer_type: type[_LocIndexerClass]  # type: ignore[valid-type]
    _iloc_indexer_type: type[_IlocIndexerClass]  # type: ignore[valid-type]
    _groupby = GroupBy
    _resampler = _Resampler

    _VALID_SCANS = {
        "cumsum",
        "cumprod",
        "cummin",
        "cummax",
    }

    # Necessary because the function names don't directly map to the docs.
    _SCAN_DOCSTRINGS = {
        "cumsum": {"op_name": "cumulative sum"},
        "cumprod": {"op_name": "cumulative product"},
        "cummin": {"op_name": "cumulative min"},
        "cummax": {"op_name": "cumulative max"},
    }

    def __init__(
        self,
        data: ColumnAccessor | MutableMapping[Any, ColumnBase],
        index: Index,
        attrs: dict[Hashable, Any] | None = None,
        allows_duplicate_labels: bool = True,
    ):
        super().__init__(data=data)
        if not isinstance(index, Index):
            raise ValueError(
                f"index must be a cudf index not {type(index).__name__}"
            )
        elif self._data.nrows > 0 and self._data.nrows != len(index):
            raise ValueError(
                f"Length of values ({self._data.nrows}) does not "
                f"match length of index ({len(index)})"
            )
        self._index = index
        if attrs is None:
            self._attrs = {}
        else:
            self._attrs = attrs
        self._flags = pd.Flags(
            self, allows_duplicate_labels=allows_duplicate_labels
        )

    @property
    def _num_rows(self) -> int:
        # Important to use the index because the data may be empty.
        return len(self.index)

    @property
    def _index_names(self) -> tuple[Any, ...]:  # TODO: Tuple[str]?
        return self.index._column_names

    @property
    def attrs(self) -> dict[Hashable, Any]:
        """
        Dictionary of global attributes of this dataset.

        Notes
        -----
        Many operations that create new datasets will copy ``attrs``. Copies
        are always deep so that changing ``attrs`` will only affect the
        present dataset. ``cudf.concat`` copies ``attrs`` only if all input
        datasets have the same ``attrs``.

        Examples
        --------
        For Series:

        >>> import cudf
        >>> ser = cudf.Series([1, 2, 3])
        >>> ser.attrs = {"A": [10, 20, 30]}
        >>> ser.attrs
        {'A': [10, 20, 30]}

        For DataFrame:

        >>> df = cudf.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df.attrs = {"A": [10, 20, 30]}
        >>> df.attrs
        {'A': [10, 20, 30]}
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)

    @property
    def flags(self) -> pd.Flags:
        """
        Get the properties associated with this cudf object.

        The available flags are

        * :attr:`pandas.Flags.allows_duplicate_labels`

        See Also
        --------
        pandas.Flags : Flags that apply to cudf objects.
        DataFrame.attrs : Global metadata applying to this dataset.

        Notes
        -----
        "Flags" differ from "metadata". Flags reflect properties of the
        :class:`Series` or :class:`DataFrame` (e.g. whether it allows
        duplicate labels). Metadata refer to properties of the dataset,
        and should be stored in ``DataFrame.attrs``.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 2]})
        >>> df.flags
        <Flags(allows_duplicate_labels=True)>

        Flags can be set via ``set_flags`` or by directly mutating the
        ``pandas.Flags`` object:

        >>> df.flags.allows_duplicate_labels = False
        >>> df.flags
        <Flags(allows_duplicate_labels=False)>
        """
        return self._flags

    def set_flags(
        self,
        *,
        copy: bool = False,
        allows_duplicate_labels: bool | None = None,
    ) -> Self:
        """
        Return a new object with updated flags.

        Parameters
        ----------
        copy : bool, default False
            Specify if a copy of the object should be made. The default
            value is preserved for compatibility with pandas.
        allows_duplicate_labels : bool, optional
            Whether the returned object allows duplicate labels.

        Returns
        -------
        Series or DataFrame
            The same type as the caller.

        See Also
        --------
        DataFrame.attrs : Global metadata applying to this dataset.
        DataFrame.flags : Global flags applying to this object.

        Notes
        -----
        This method returns a new object that's a view on the same data.
        The returned object will have ``flags.allows_duplicate_labels``
        set to the value passed. Unlike the pandas version, the default
        of ``copy=False`` does not share memory between the two objects
        because cudf backs all operations with copy-on-write buffers.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 2]})
        >>> df.flags.allows_duplicate_labels
        True
        >>> df2 = df.set_flags(allows_duplicate_labels=False)
        >>> df2.flags.allows_duplicate_labels
        False
        """
        df = self.copy(deep=copy)
        if allows_duplicate_labels is not None:
            df.flags["allows_duplicate_labels"] = allows_duplicate_labels
        return df

    @classmethod
    def _from_data(
        cls,
        data: MutableMapping,
        index: Index | None = None,
        attrs: dict | None = None,
        allows_duplicate_labels: bool = True,
    ):
        out = super()._from_data(data)
        if index is None:
            # out._num_rows requires .index to be defined
            index = RangeIndex(out._data.nrows)
        elif not isinstance(index, Index):
            raise ValueError(
                f"index must be a cudf.Index not {type(index).__name__}"
            )
        out._index = index
        out._attrs = {} if attrs is None else attrs
        out._flags = pd.Flags(
            out, allows_duplicate_labels=allows_duplicate_labels
        )
        return out

    def _propagate_metadata(self, result: "IndexedFrame") -> "IndexedFrame":
        """Copy ``attrs`` and ``flags`` from ``self`` onto ``result``.

        Mirrors pandas' ``__finalize__``: ``attrs`` are deep-copied and
        ``allows_duplicate_labels`` is propagated unchanged. The
        :class:`~pandas.errors.DuplicateLabelError` check is performed by
        :class:`pandas.Flags` when setting ``allows_duplicate_labels`` to
        ``False``.
        """
        result._attrs = copy.deepcopy(self._attrs)
        result._flags = pd.Flags(result, allows_duplicate_labels=True)
        result.flags.allows_duplicate_labels = (
            self._flags.allows_duplicate_labels
        )
        return result

    @_performance_tracking
    def _get_columns_by_label(self, labels) -> Self:
        """
        Returns columns of the Frame specified by `labels`.

        Akin to cudf.DataFrame(...).loc[:, labels]
        """
        out = self._from_data(
            self._data.select_by_label(labels),
            index=self.index,
        )
        self._propagate_metadata(out)
        return out

    @_performance_tracking
    def _from_data_like_self(self, data: MutableMapping):
        out = super()._from_data_like_self(data)
        out.index = self.index
        out._attrs = copy.deepcopy(self._attrs)
        out._flags = pd.Flags(
            out, allows_duplicate_labels=self._flags.allows_duplicate_labels
        )
        return out

    @_performance_tracking
    def _from_columns_like_self(
        self,
        columns: list[ColumnBase],
        column_names: Iterable[str] | None = None,
        index_names: list[str] | None = None,
    ) -> Self:
        """Construct a `Frame` from a list of columns with metadata from self.

        If `index_names` is set, the first `len(index_names)` columns are
        used to construct the index of the frame.
        """
        if column_names is None:
            column_names = self._column_names

        data_columns = columns
        index = None

        if index_names is not None:
            n_index_columns = len(index_names)
            data_columns = columns[n_index_columns:]
            index = _index_from_data(
                dict(enumerate(columns[:n_index_columns]))
            )
            # Handle special index types that need to be reconstructed
            if (
                isinstance(self.index, cudf.CategoricalIndex)
                and not isinstance(index, cudf.CategoricalIndex)
            ) or (
                isinstance(self.index, cudf.MultiIndex)
                and not isinstance(index, cudf.MultiIndex)
            ):
                index = type(self.index)._from_data(index._data)
            if isinstance(index, cudf.MultiIndex):
                index.names = index_names
            else:
                index.name = index_names[0]
            # Preserve DatetimeIndex frequency if the original index had one
            if isinstance(self.index, cudf.DatetimeIndex) and isinstance(
                index, cudf.DatetimeIndex
            ):
                index._freq = self.index._freq

        data = dict(zip(column_names, data_columns, strict=True))
        out = type(self)._from_data(data, index)
        self._propagate_metadata(out)
        return out

    def __round__(self, digits=0):
        # Shouldn't be added to BinaryOperand
        # because pandas Index doesn't implement
        # this method.
        return self.round(decimals=digits)

    def _mimic_inplace(
        self, result: Self, inplace: bool = False
    ) -> Self | None:
        if inplace:
            self._index = result.index
            self._attrs = result._attrs
            self._flags = pd.Flags(
                self,
                allows_duplicate_labels=result._flags.allows_duplicate_labels,
            )
        return super()._mimic_inplace(result, inplace)

    @_performance_tracking
    def _scan(
        self,
        op: str,
        axis: Axis | None = None,
        skipna: bool = True,
        *args,
        **kwargs,
    ) -> Self:
        """
        Return {op_name} of the {cls}.

        Parameters
        ----------
        axis: {{index (0), columns(1)}}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        {cls}

        Examples
        --------
        **Series**

        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cumsum()
        0     1
        1     6
        2     8
        3    12
        4    15
        dtype: int64

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({{'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]}})
        >>> df.cumsum()
            a   b
        0   1   7
        1   3  15
        2   6  24
        3  10  34
        """
        if "numeric_only" in kwargs:
            raise TypeError(
                "got an unexpected keyword argument 'numeric_only'"
            )

        for col in self._columns:
            if isinstance(col.dtype, pd.ArrowDtype):
                if pa.types.is_decimal(col.dtype.pyarrow_dtype):
                    raise NotImplementedError(
                        "Decimal ArrowDtype does not support cumulative operations"
                    )

        cast_to_int = op in ("cumsum", "cumprod")
        skipna = True if skipna is None else skipna

        results = []
        for col in self._columns:
            if skipna:
                result_col = col.nans_to_nulls()
            else:
                if col.has_nulls(include_nan=True):
                    first_index = col.isnull().find_first_value(True)
                    result_col = col.copy()
                    result_col[first_index:] = None
                else:
                    result_col = col

            if cast_to_int and result_col.dtype.kind in "uib":
                # pandas returns an int64 dtype for all int or bool dtypes.
                dtype = get_dtype_of_same_kind(
                    result_col.dtype, np.dtype(np.int64)
                )
                result_col = result_col.astype(dtype)
            results.append(getattr(result_col, op)(inclusive=True))
        return self._from_data_like_self(
            self._data._from_columns_like_self(results)
        )

    @_performance_tracking
    @ioutils.doc_to_json()
    def to_json(self, path_or_buf=None, *args, **kwargs):
        """{docstring}"""

        return cudf.io.json.to_json(
            self, path_or_buf=path_or_buf, *args, **kwargs
        )

    @_performance_tracking
    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""

        cudf.io.hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    def copy(self, deep: bool = True) -> Self:
        """Make a copy of this object's indices and data.

        When ``deep=True`` (default), a new object will be created with a
        copy of the calling object's data and indices. Modifications to
        the data or indices of the copy will not be reflected in the
        original object (see notes below).
        When ``deep=False``, a new object will be created without copying
        the calling object's data or index (only references to the data
        and index are copied). Data is shared via copy-on-write, so
        modifications to either object will trigger a physical copy and
        will not be reflected in the other.

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With ``deep=False`` neither the indices nor the data are copied.

        Returns
        -------
        copy : Series or DataFrame
            Object type matches caller.

        Examples
        --------
        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> s
        a    1
        b    2
        dtype: int64
        >>> s_copy = s.copy()
        >>> s_copy
        a    1
        b    2
        dtype: int64

        **Shallow copy versus default (deep) copy:**

        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> deep = s.copy()
        >>> shallow = s.copy(deep=False)

        Shallow copy shares data via copy-on-write. Modifications to
        either the original or the shallow copy trigger a physical copy,
        so changes are not reflected in the other.

        >>> s['a'] = 3
        >>> shallow['b'] = 4
        >>> s
        a    3
        b    2
        dtype: int64
        >>> shallow
        a    1
        b    4
        dtype: int64
        >>> deep
        a    1
        b    2
        dtype: int64
        """
        return self._from_data(
            self._data.copy(deep=deep),
            self.index.copy(deep=deep),
            attrs=copy.deepcopy(self.attrs) if deep else self._attrs,
            allows_duplicate_labels=self._flags.allows_duplicate_labels,
        )

    @_performance_tracking
    def equals(self, other) -> bool:
        return super().equals(other) and self.index.equals(other.index)

    @property
    def index(self):
        """Get the labels for the rows."""
        return self._index

    @index.setter
    def index(self, value):
        old_length = len(self)
        new_length = len(value)

        # A DataFrame with 0 columns can have an index of arbitrary length.
        if self._num_columns > 0 and new_length != old_length:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_length} elements, "
                f"new values have {len(value)} elements"
            )
        # avoid unnecessary cast to Index
        value = ensure_index(value)
        self._index = value

    @_performance_tracking
    def replace(
        self,
        to_replace=None,
        value=no_default,
        *,
        inplace: bool = False,
        regex: bool = False,
    ) -> Self | None:
        """Replace values given in ``to_replace`` with ``value``.

        Parameters
        ----------
        to_replace : numeric, str or list-like
            Value(s) to replace.

            * numeric or str:
                - values equal to ``to_replace`` will be replaced
                  with ``value``
            * list of numeric or str:
                - If ``value`` is also list-like, ``to_replace`` and
                  ``value`` must be of same length.
            * dict:
                - Dicts can be used to specify different replacement values
                  for different existing values. For example, {'a': 'b',
                  'y': 'z'} replaces the value 'a' with 'b' and
                  'y' with 'z'.
                  To use a dict in this way the ``value`` parameter should
                  be ``None``.
        value : scalar, dict, list-like, str, default None
            Value to replace any values matching ``to_replace`` with.
        inplace : bool, default False
            If True, in place.

        See Also
        --------
        Series.fillna

        Raises
        ------
        TypeError
            - If ``to_replace`` is not a scalar, array-like, dict, or None
            - If ``to_replace`` is a dict and value is not a list, dict,
              or Series
        ValueError
            - If a list is passed to ``to_replace`` and ``value`` but they
              are not the same length.

        Returns
        -------
        result : Series
            Series after replacement. The mask and index are preserved.

        Examples
        --------
        **Series**

        Scalar ``to_replace`` and ``value``

        >>> import cudf
        >>> s = cudf.Series([0, 1, 2, 3, 4])
        >>> s
        0    0
        1    1
        2    2
        3    3
        4    4
        dtype: int64
        >>> s.replace(0, 5)
        0    5
        1    1
        2    2
        3    3
        4    4
        dtype: int64

        List-like ``to_replace``

        >>> s.replace([1, 2], 10)
        0     0
        1    10
        2    10
        3     3
        4     4
        dtype: int64

        dict-like ``to_replace``

        >>> s.replace({1:5, 3:50})
        0     0
        1     5
        2     2
        3    50
        4     4
        dtype: int64
        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])
        >>> s
        0     b
        1     a
        2     a
        3     b
        4     a
        dtype: str
        >>> s.replace({'a': None})
        0       b
        1     NaN
        2     NaN
        3       b
        4     NaN
        dtype: str

        If there is a mismatch in types of the values in
        ``to_replace`` & ``value`` with the actual series, then
        cudf raises a ``TypeError``:

        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])
        >>> s
        0    b
        1    a
        2    a
        3    b
        4    a
        dtype: str
        >>> s.replace('a', 1)
        Traceback (most recent call last):
            ...
        TypeError: to_replace and value should be of same types,got to_replace dtype: str and value dtype: int64
        >>> s.replace(['a', 'c'], [1, 2])
        Traceback (most recent call last):
            ...
        TypeError: to_replace and value should be of same types,got to_replace dtype: str and value dtype: int64

        **DataFrame**

        Scalar ``to_replace`` and ``value``

        >>> import cudf
        >>> df = cudf.DataFrame({'A': [0, 1, 2, 3, 4],
        ...                    'B': [5, 6, 7, 8, 9],
        ...                    'C': ['a', 'b', 'c', 'd', 'e']})
        >>> df
           A  B  C
        0  0  5  a
        1  1  6  b
        2  2  7  c
        3  3  8  d
        4  4  9  e
        >>> df.replace(0, 5)
           A  B  C
        0  5  5  a
        1  1  6  b
        2  2  7  c
        3  3  8  d
        4  4  9  e

        List-like ``to_replace``

        >>> df.replace([0, 1, 2, 3], 4)
           A  B  C
        0  4  5  a
        1  4  6  b
        2  4  7  c
        3  4  8  d
        4  4  9  e
        >>> df.replace([0, 1, 2, 3], [4, 3, 2, 1])
           A  B  C
        0  4  5  a
        1  3  6  b
        2  2  7  c
        3  1  8  d
        4  4  9  e

        dict-like ``to_replace``

        >>> df.replace({0: 10, 1: 100})
             A  B  C
        0   10  5  a
        1  100  6  b
        2    2  7  c
        3    3  8  d
        4    4  9  e
        >>> df.replace({'A': 0, 'B': 5}, 100)
             A    B  C
        0  100  100  a
        1    1    6  b
        2    2    7  c
        3    3    8  d
        4    4    9  e

        .. pandas-compat::
            :meth:`pandas.DataFrame.replace`, :meth:`pandas.Series.replace`

            `regex` is not currently supported.
        """
        if regex:
            raise NotImplementedError("regex parameter is not implemented yet")

        if value is no_default and not (
            is_dict_like(to_replace) or is_dict_like(regex)
        ):
            raise ValueError(
                f"{type(self).__name__}.replace must specify either 'value', "
                "a dict-like 'to_replace', or dict-like 'regex'."
            )

        if not (to_replace is None and value is no_default):
            (
                all_na_per_column,
                to_replace_per_column,
                replacements_per_column,
            ) = _get_replacement_values_for_columns(
                to_replace=to_replace,
                value=value,
                columns_dtype_map=dict(self._dtypes),
            )
            copy_data = []
            for name, col in self._column_labels_and_values:
                try:
                    replaced = col.find_and_replace(
                        to_replace_per_column[name],
                        replacements_per_column[name],
                        all_na_per_column[name],
                    )
                except (KeyError, OverflowError):
                    # We need to create a deep copy if:
                    # i. `find_and_replace` was not successful or any of
                    #    `to_replace_per_column`, `replacements_per_column`,
                    #    `all_na_per_column` don't contain the `name`
                    #    that exists in `copy_data`.
                    # ii. There is an OverflowError while trying to cast
                    #     `to_replace_per_column` to `replacements_per_column`.
                    replaced = col.copy(deep=True)
                copy_data.append(replaced)
            result = self._from_data_like_self(
                self._data._from_columns_like_self(copy_data)
            )
        else:
            result = self.copy()

        return self._mimic_inplace(result, inplace=inplace)

    @_performance_tracking
    def clip(self, lower=None, upper=None, axis=1, inplace=False):
        """
        Trim values at input threshold(s).

        Assigns values outside boundary to boundary values.
        Thresholds can be singular values or array like,
        and in the latter case the clipping is performed
        element-wise in the specified axis. Currently only
        `axis=1` is supported.

        Parameters
        ----------
        lower : scalar or array_like, default None
            Minimum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on lower.
            In case of Series/Index, lower is expected to be
            a scalar or an array of size 1.
        upper : scalar or array_like, default None
            Maximum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on upper.
            In case of Series, upper is expected to be
            a scalar or an array of size 1.
        inplace : bool, default False

        Returns
        -------
        Clipped DataFrame/Series/Index/MultiIndex

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"a":[1, 2, 3, 4], "b":['a', 'b', 'c', 'd']})
        >>> df.clip(lower=[2, 'b'], upper=[3, 'c'])
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=None, upper=[3, 'c'])
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=[2, 'b'], upper=None)
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  4  d

        >>> df = cudf.DataFrame({"a":[1, 2, 3, 4], "b":[5, 6, 7, 8]})
        >>> df.clip(lower=2, upper=7, inplace=True)
        >>> df
           a  b
        0  2  5
        1  2  6
        2  3  7
        3  4  7

        >>> import cudf
        >>> sr = cudf.Series([1, 2, 3, 4])
        >>> sr.clip(lower=2, upper=3)
        0    2
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=None, upper=3)
        0    1
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=2, upper=None, inplace=True)
        >>> sr
        0    2
        1    2
        2    3
        3    4
        dtype: int64
        """
        if axis != 1:
            raise NotImplementedError("`axis is not yet supported in clip`")

        if lower is None and upper is None:
            return None if inplace is True else self.copy(deep=True)

        if is_scalar(lower):
            lower = np.full(self._num_columns, lower)
        if is_scalar(upper):
            upper = np.full(self._num_columns, upper)

        if len(lower) != len(upper):
            raise ValueError("Length of lower and upper should be equal")

        if len(lower) != self._num_columns:
            raise ValueError(
                "Length of lower/upper should be equal to number of columns"
            )

        if self.ndim == 1:
            # In case of series and Index,
            # swap lower and upper if lower > upper
            if (
                lower[0] is not None
                and upper[0] is not None
                and (lower[0] > upper[0])
            ):
                lower[0], upper[0] = upper[0], lower[0]

        data = (
            col.clip(low, high)
            for col, low, high in zip(self._columns, lower, upper, strict=True)
        )
        output = self._from_data_like_self(
            self._data._from_columns_like_self(data)
        )
        return self._mimic_inplace(output, inplace=inplace)

    @_performance_tracking
    def abs(self):
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        This function only applies to elements that are all numeric.

        Returns
        -------
        DataFrame/Series
            Absolute value of each element.

        Examples
        --------
        Absolute numeric values in a Series

        >>> s = cudf.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        dtype: float64
        """
        return self._unaryop("abs")

    @_performance_tracking
    def dot(self, other, reflect=False):
        """
        Get dot product of frame and other, (binary operator `dot`).

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`,
        `dot`) to arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`,
        `@`.

        Parameters
        ----------
        other : Sequence, Series, or DataFrame
            Any multiple element data structure, or list-like object.
        reflect : bool, default False
            If ``True``, swap the order of the operands. See
            https://docs.python.org/3/reference/datamodel.html#object.__ror__
            for more information on when this is necessary.

        Returns
        -------
        scalar, Series, or DataFrame
            The result of the operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame([[1, 2, 3, 4],
        ...                      [5, 6, 7, 8]])
        >>> df @ df.T
            0    1
        0  30   70
        1  70  174
        >>> s = cudf.Series([1, 1, 1, 1])
        >>> df @ s
        0    10
        1    26
        dtype: int64
        >>> [1, 2, 3, 4] @ s
        np.int64(10)
        """
        # TODO: This function does not currently support nulls.
        lhs = self.values
        result_index = None
        result_cols = None
        if isinstance(self, cudf.Series) and isinstance(
            other, (cudf.Series, cudf.DataFrame)
        ):
            common = self.index.union(other.index)
            if len(common) > max(len(self), len(other)):
                raise ValueError("matrices are not aligned")

            lhs = self.reindex(index=common, copy=False).values
            rhs = other.reindex(index=common, copy=False).values
            if isinstance(other, cudf.DataFrame):
                result_index = other._data.to_pandas_index
        elif isinstance(self, cudf.DataFrame) and isinstance(
            other, (cudf.Series, cudf.DataFrame)
        ):
            common = self._data.to_pandas_index.union(other.index.to_pandas())
            if len(common) > max(self._num_columns, len(other)):
                raise ValueError("matrices are not aligned")

            lhs = self.reindex(columns=common, copy=False)
            result_index = lhs.index

            rhs = other.reindex(index=common, copy=False).values
            lhs = lhs.values
            if isinstance(other, cudf.DataFrame):
                result_cols = other._data.to_pandas_index

        elif isinstance(
            other, (cp.ndarray, np.ndarray)
        ) or can_convert_to_column(other):
            rhs = cp.asarray(other)
        else:
            # TODO: This should raise an exception, not return NotImplemented,
            # but __matmul__ relies on the current behavior. We should either
            # move this implementation to __matmul__ and call it from here
            # (checking for NotImplemented and raising NotImplementedError if
            # that's what's returned), or __matmul__ should catch a
            # NotImplementedError from here and return NotImplemented. The
            # latter feels cleaner (putting the implementation in this method
            # rather than in the operator) but will be slower in the (highly
            # unlikely) case that we're multiplying a cudf object with another
            # type of object that somehow supports this behavior.
            return NotImplemented
        if reflect:
            lhs, rhs = rhs, lhs

        result = lhs.dot(rhs)
        if result.ndim == 1:
            return cudf.Series(
                result,
                index=self.index if result_index is None else result_index,
            )
        if result.ndim == 2:
            return cudf.DataFrame(
                result,
                index=self.index if result_index is None else result_index,
                columns=result_cols,
            )
        # 0-d result: return a numpy scalar (e.g. np.int64) to match
        # pandas' ``np.matmul(Series, Series)`` semantics rather than the
        # Python ``int`` produced by ``cupy.ndarray.item``.
        return result.get()[()]

    @_performance_tracking
    def __matmul__(self, other):
        return self.dot(other)

    @_performance_tracking
    def __rmatmul__(self, other):
        return self.dot(other, reflect=True)

    @_performance_tracking
    def head(self, n=5):
        """
        Return the first `n` rows.
        This function returns the first `n` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.
        For negative values of `n`, this function returns all rows except
        the last `n` rows, equivalent to ``df[:-n]``.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        DataFrame or Series
            The first `n` rows of the caller object.

        Examples
        --------
        **Series**

        >>> ser = cudf.Series(['alligator', 'bee', 'falcon',
        ... 'lion', 'monkey', 'parrot', 'shark', 'whale', 'zebra'])
        >>> ser
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        5       parrot
        6        shark
        7        whale
        8        zebra
        dtype: str

        Viewing the first 5 lines

        >>> ser.head()
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        dtype: str

        Viewing the first `n` lines (three in this case)

        >>> ser.head(3)
        0    alligator
        1          bee
        2       falcon
        dtype: str

        For negative values of `n`

        >>> ser.head(-3)
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        5       parrot
        dtype: str

        **DataFrame**

        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> df.head(2)
           key   val
        0    0  10.0
        1    1  11.0
        """
        return self.iloc[:n]

    @_performance_tracking
    def tail(self, n=5):
        """
        Returns the last n rows as a new DataFrame or Series

        Examples
        --------
        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> df.tail(2)
           key   val
        3    3  13.0
        4    4  14.0

        **Series**

        >>> import cudf
        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.tail(2)
        3    1
        4    0
        dtype: int64
        """
        if n == 0:
            return self.iloc[0:0]

        return self.iloc[-n:]

    @_performance_tracking
    def pipe(self, func, *args, **kwargs):
        """
        Apply ``func(self, *args, **kwargs)``.

        Parameters
        ----------
        func : function
            Function to apply to the Series/DataFrame.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the Series/DataFrame.
        args : iterable, optional
            Positional arguments passed into ``func``.
        kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Examples
        --------
        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects.

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3]})
        >>> def add_one(x):
        ...     return x + 1
        >>> df.pipe(add_one)
           a
        0  2
        1  3
        2  4
        """
        return pipe(self, func, *args, **kwargs)

    @_performance_tracking
    def sum(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ):
        """
        Return sum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.sum()
        a    10
        b    34
        dtype: int64
        """
        return self._reduce(
            "sum",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    @_performance_tracking
    def product(
        self,
        axis: Axis | None = 0,
        skipna=True,
        numeric_only=False,
        min_count=0,
        **kwargs,
    ):
        """
        Return product of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.product()
        a      24
        b    5040
        dtype: int64
        """

        return self._reduce(
            # cuDF columns use "product" as the op name, but cupy uses "prod"
            # and we need cupy if axis == 1.
            "prod" if axis in {1, "columns"} else "product",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    # Alias for pandas compatibility.
    prod = product

    @_performance_tracking
    def mean(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return the mean of the values for the requested axis.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        mean : Series or DataFrame (if level specified)

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.mean()
        a    2.5
        b    8.5
        dtype: float64
        """
        return self._reduce(
            "mean",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_performance_tracking
    def median(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return the median of the values for the requested axis.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on. For Series this
            parameter is unused and defaults to 0.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        scalar

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([10, 25, 3, 25, 24, 6])
        >>> ser
        0    10
        1    25
        2     3
        3    25
        4    24
        5     6
        dtype: int64
        >>> ser.median()
        np.float64(17.0)
        """
        if "overwrite_input" in kwargs:
            raise ValueError(
                "the 'overwrite_input' parameter is not supported in the "
                "pandas implementation of median()"
            )
        return self._reduce(
            "median",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_performance_tracking
    def std(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return sample standard deviation of the DataFrame.

        Normalized by N-1 by default. This can be changed using
        the `ddof` argument

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations
            is N - ddof, where N represents the number of elements.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.std()
        a    1.290994
        b    1.290994
        dtype: float64
        """

        return self._reduce(
            "std",
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_performance_tracking
    def var(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return unbiased variance of the DataFrame.

        Normalized by N-1 by default. This can be changed using the
        ddof argument.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is
            N - ddof, where N represents the number of elements.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        scalar

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.var()
        a    1.666667
        b    1.666667
        dtype: float64
        """
        return self._reduce(
            "var",
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_performance_tracking
    def kurtosis(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return Fisher's unbiased kurtosis of a sample.

        Kurtosis obtained using Fisher's definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series or scalar

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4])
        >>> series.kurtosis()
        np.float64(-1.200000000000001)

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.kurt()
        a   -1.2
        b   -1.2
        dtype: float64
        """
        if axis not in (0, "index", None, no_default):
            raise NotImplementedError("Only axis=0 is currently supported.")

        return self._reduce(
            "kurtosis",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    # Alias for kurtosis.
    kurt = kurtosis

    @_performance_tracking
    def skew(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        Return unbiased Fisher-Pearson skew of a sample.

        Parameters
        ----------
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4, 5, 6, 6])
        >>> series
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        6    6
        dtype: int64

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 8, 10, 10]})
        >>> df.skew()
        a    0.00000
        b   -0.37037
        dtype: float64

        .. pandas-compat::
            :meth:`pandas.DataFrame.skew`, :meth:`pandas.Series.skew`

            The `axis` parameter is not currently supported.
        """
        if axis not in (0, "index", None, no_default):
            raise NotImplementedError("Only axis=0 is currently supported.")

        return self._reduce(
            "skew",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_performance_tracking
    def mask(
        self,
        cond,
        other=None,
        inplace: bool = False,
        axis=None,
        level=None,
    ) -> Self | None:
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is False, keep the original value.
            Where True, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is True are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
        >>> df.mask(df % 2 == 0, [-1, -1])
           A  B
        0  1  3
        1 -1  5
        2  5 -1

        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.mask(ser > 2, 10)
        0    10
        1    10
        2     2
        3     1
        4     0
        dtype: int64
        >>> ser.mask(ser > 2)
        0    <NA>
        1    <NA>
        2       2
        3       1
        4       0
        dtype: int64
        """
        if axis is not None:
            raise NotImplementedError("axis is not supported.")
        elif level is not None:
            raise NotImplementedError("level is not supported.")

        if not hasattr(cond, "__invert__"):
            # We Invert `cond` below and call `where`, so
            # making sure the object supports
            # `~`(inversion) operator or `__invert__` method
            cond = cp.asarray(cond)

        return self.where(cond=~cond, other=other, inplace=inplace)

    @_performance_tracking
    @copy_docstring(Rolling)
    def rolling(
        self,
        window,
        min_periods=None,
        center: bool = False,
        win_type: str | None = None,
        on=None,
        axis=0,
        closed: str | None = None,
        step: int | None = None,
        method: str = "single",
    ):
        return Rolling(
            self,  # type: ignore[arg-type]
            window,
            min_periods=min_periods,
            center=center,
            axis=axis,
            on=on,
            win_type=win_type,
            closed=closed,
            step=step,
            method=method,
        )

    @copy_docstring(ExponentialMovingWindow)
    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: int = 0,
        times: str | np.ndarray | None = None,
        method: Literal["single", "table"] = "single",
    ):
        return ExponentialMovingWindow(
            self,
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            axis=axis,
            times=times,
            method=method,
        )

    @_performance_tracking
    def nans_to_nulls(self) -> Self:
        """
        Convert nans (if any) to nulls

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        **Series**

        >>> import cudf, numpy as np
        >>> series = cudf.Series([1, 2, np.nan, None, 10], nan_as_null=False)
        >>> series
        0     1.0
        1     2.0
        2     NaN
        3     NaN
        4    10.0
        dtype: float64
        >>> series.nans_to_nulls()
        0     1.0
        1     2.0
        2     NaN
        3     NaN
        4    10.0
        dtype: float64

        **DataFrame**

        >>> df = cudf.DataFrame()
        >>> df['a'] = cudf.Series([1, None, np.nan], nan_as_null=False)
        >>> df['b'] = cudf.Series([None, 3.14, np.nan], nan_as_null=False)
        >>> df
              a     b
        0  1.0   NaN
        1  NaN  3.14
        2  NaN   NaN
        >>> df.nans_to_nulls()
              a     b
        0  1.0   NaN
        1  NaN  3.14
        2  NaN   NaN
        """
        return super().nans_to_nulls()

    @_performance_tracking
    def interpolate(
        self,
        *,
        method: str = "linear",
        axis: Axis = 0,
        limit: int | None = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        **kwargs,
    ):
        """
        Interpolate data values between some points.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. Currently,
            only 'linear` is supported.
            * 'linear': Ignore the index and treat the values as
            equally spaced. This is the only method supported on MultiIndexes.
            * 'index', 'values': linearly interpolate using the index as
            an x-axis. Unsorted indices can lead to erroneous results.
        axis : int, default 0
            Axis to interpolate along. Currently,
            only 'axis=0' is supported.
        inplace : bool, default False
            Update the data in place if possible.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller, interpolated at
            some or all ``NaN`` values

        """
        if method not in {"linear", "values", "index"}:
            raise ValueError(f"Interpolation method `{method}` not found")

        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean")
        elif inplace is True:
            raise NotImplementedError("inplace is not supported")

        if limit is not None:
            raise NotImplementedError("limit is not supported")
        if limit_direction is not None:
            raise NotImplementedError("limit_direction is not supported")
        if limit_area is not None:
            raise NotImplementedError("limit_area is not supported")

        data = self
        if not isinstance(data.index, cudf.RangeIndex):
            perm_sort = data.index.argsort()
            data = data._gather(
                GatherMap.from_column_unchecked(
                    as_column(perm_sort),  # type: ignore[arg-type]
                    len(data),
                    nullify=False,
                )
            )

        if method == "linear":
            interp_index = RangeIndex(self._num_rows)
        else:
            interp_index = data.index

        result = self._from_data_like_self(
            self._data._from_columns_like_self(
                [col.interpolate(index=interp_index) for col in data._columns]
            )
        )
        result.index = data.index

        return (
            result
            if isinstance(data.index, cudf.RangeIndex)
            # TODO: This should be a scatter, avoiding an argsort.
            else result._gather(
                GatherMap.from_column_unchecked(
                    as_column(perm_sort.argsort()),  # type: ignore[arg-type]
                    len(result),
                    nullify=False,
                )
            )
        )

    _SUPPORT_AXIS_LOOKUP = {0: 0, "index": 0}

    @classmethod
    @_performance_tracking
    def _get_axis_from_axis_arg(cls, axis: int | str) -> int:
        try:
            return cls._SUPPORT_AXIS_LOOKUP[axis]
        except KeyError:
            raise ValueError(f"No axis named {axis} for object type {cls}")

    @_performance_tracking
    def shift(
        self,
        periods=1,
        freq=None,
        axis=0,
        fill_value=None,
        suffix: str | None = None,
    ):
        """Shift values by `periods` positions."""
        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise NotImplementedError("Only axis=0 is supported.")
        if freq is not None:
            raise NotImplementedError(
                "The freq argument is not yet supported."
            )
        if suffix is not None:
            raise NotImplementedError(
                "The suffix argument is not yet supported."
            )

        data_columns = (
            col.shift(periods, fill_value) for col in self._columns
        )
        return self._from_data_like_self(
            self._data._from_columns_like_self(data_columns)
        )

    @_performance_tracking
    def truncate(self, before=None, after=None, axis=0, copy=True):
        """
        Truncate a Series or DataFrame before and after some index value.

        This is a useful shorthand for boolean indexing based on index
        values above or below certain thresholds.

        Parameters
        ----------
        before : date, str, int
            Truncate all rows before this index value.
        after : date, str, int
            Truncate all rows after this index value.
        axis : {0 or 'index', 1 or 'columns'}, optional
            Axis to truncate. Truncates the index (rows) by default.
        copy : bool, default is True,
            Return a copy of the truncated section.

        Returns
        -------
            The truncated Series or DataFrame.

        Notes
        -----
        If the index being truncated contains only datetime values,
        `before` and `after` may be specified as strings instead of
        Timestamps.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> cs1 = cudf.Series([1, 2, 3, 4])
        >>> cs1
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> cs1.truncate(before=1, after=2)
        1    2
        2    3
        dtype: int64

        >>> import cudf
        >>> dates = cudf.date_range(
        ...     '2021-01-01 23:45:00', '2021-01-01 23:46:00', freq='s'
        ... )
        >>> cs2 = cudf.Series(range(len(dates)), index=dates)
        >>> cs2
        2021-01-01 23:45:00     0
        2021-01-01 23:45:01     1
        2021-01-01 23:45:02     2
        2021-01-01 23:45:03     3
        2021-01-01 23:45:04     4
                                ..
        2021-01-01 23:45:56    56
        2021-01-01 23:45:57    57
        2021-01-01 23:45:58    58
        2021-01-01 23:45:59    59
        2021-01-01 23:46:00    60
        Length: 61, dtype: int64


        >>> cs2.truncate(
        ...     before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ... )
        2021-01-01 23:45:18    18
        2021-01-01 23:45:19    19
        2021-01-01 23:45:20    20
        2021-01-01 23:45:21    21
        2021-01-01 23:45:22    22
        2021-01-01 23:45:23    23
        2021-01-01 23:45:24    24
        2021-01-01 23:45:25    25
        2021-01-01 23:45:26    26
        2021-01-01 23:45:27    27
        dtype: int64

        >>> cs3 = cudf.Series({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        >>> cs3
        A    1
        B    2
        C    3
        D    4
        dtype: int64

        >>> cs3.truncate(before='B', after='C')
        B    2
        C    3
        dtype: int64

        **DataFrame**

        >>> df = cudf.DataFrame({
        ...     'A': ['a', 'b', 'c', 'd', 'e'],
        ...     'B': ['f', 'g', 'h', 'i', 'j'],
        ...     'C': ['k', 'l', 'm', 'n', 'o']
        ... }, index=[1, 2, 3, 4, 5])
        >>> df
           A  B  C
        1  a  f  k
        2  b  g  l
        3  c  h  m
        4  d  i  n
        5  e  j  o

        >>> df.truncate(before=2, after=4)
           A  B  C
        2  b  g  l
        3  c  h  m
        4  d  i  n

        >>> df.truncate(before="A", after="B", axis="columns")
           A  B
        1  a  f
        2  b  g
        3  c  h
        4  d  i
        5  e  j

        >>> import cudf
        >>> dates = cudf.date_range(
        ...     '2021-01-01 23:45:00', '2021-01-01 23:46:00', freq='s'
        ... )
        >>> df2 = cudf.DataFrame(data={'A': 1, 'B': 2}, index=dates)
        >>> df2.head()
                             A  B
        2021-01-01 23:45:00  1  2
        2021-01-01 23:45:01  1  2
        2021-01-01 23:45:02  1  2
        2021-01-01 23:45:03  1  2
        2021-01-01 23:45:04  1  2

        >>> df2.truncate(
        ...     before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ... )
                             A  B
        2021-01-01 23:45:18  1  2
        2021-01-01 23:45:19  1  2
        2021-01-01 23:45:20  1  2
        2021-01-01 23:45:21  1  2
        2021-01-01 23:45:22  1  2
        2021-01-01 23:45:23  1  2
        2021-01-01 23:45:24  1  2
        2021-01-01 23:45:25  1  2
        2021-01-01 23:45:26  1  2
        2021-01-01 23:45:27  1  2

        .. pandas-compat::
            :meth:`pandas.DataFrame.truncate`, :meth:`pandas.Series.truncate`

            The ``copy`` parameter is only present for API compatibility, but
            ``copy=False`` is not supported. This method always generates a
            copy.
        """
        if not copy:
            raise ValueError("Truncating with copy=False is not supported.")
        axis = self._get_axis_from_axis_arg(axis)
        ax = self.index if axis == 0 else self._data.to_pandas_index

        if not ax.is_monotonic_increasing and not ax.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")

        if isinstance(ax, cudf.DatetimeIndex):
            before = pd.to_datetime(before)
            after = pd.to_datetime(after)

        if before is not None and after is not None and before > after:
            raise ValueError(f"Truncate: {after} must be after {before}")

        if len(ax) > 1 and ax.is_monotonic_decreasing and ax.nunique() > 1:
            before, after = after, before

        slicer = [slice(None, None)] * self.ndim
        slicer[axis] = slice(before, after)
        return self.loc[tuple(slicer)].copy()

    @property
    def loc(self):
        """Select rows and columns by label or boolean mask.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([10, 11, 12], index=['a', 'b', 'c'])
        >>> series
        a    10
        b    11
        c    12
        dtype: int64
        >>> series.loc['b']
        11

        **DataFrame**

        DataFrame with string index.

        >>> df
           a  b
        a  0  5
        b  1  6
        c  2  7
        d  3  8
        e  4  9

        Select a single row by label.

        >>> df.loc['a']
        a    0
        b    5
        Name: a, dtype: int64

        Select multiple rows and a single column.

        >>> df.loc[['a', 'c', 'e'], 'b']
        a    5
        c    7
        e    9
        Name: b, dtype: int64

        Selection by boolean mask.

        >>> df.loc[df.a > 2]
           a  b
        d  3  8
        e  4  9

        Setting values using loc.

        >>> df.loc[['a', 'c', 'e'], 'a'] = 0
        >>> df
           a  b
        a  0  5
        b  1  6
        c  0  7
        d  3  8
        e  0  9

        """
        return self._loc_indexer_type(self)

    @property
    def iloc(self):
        """Select values by position.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> s = cudf.Series([10, 20, 30])
        >>> s
        0    10
        1    20
        2    30
        dtype: int64
        >>> s.iloc[2]
        30

        **DataFrame**

        Selecting rows and column by position.

        >>> df = cudf.DataFrame({'a': range(20),
        ...                      'b': range(20),
        ...                      'c': range(20)})

        Select a single row using an integer index.

        >>> df.iloc[1]
        a    1
        b    1
        c    1
        Name: 1, dtype: int64

        Select multiple rows using a list of integers.

        >>> df.iloc[[0, 2, 9, 18]]
              a    b    c
         0    0    0    0
         2    2    2    2
         9    9    9    9
        18   18   18   18

        Select rows using a slice.

        >>> df.iloc[3:10:2]
             a    b    c
        3    3    3    3
        5    5    5    5
        7    7    7    7
        9    9    9    9

        Select both rows and columns.

        >>> df.iloc[[1, 3, 5, 7], 2]
        1    1
        3    3
        5    5
        7    7
        Name: c, dtype: int64

        Setting values in a column using iloc.

        >>> df.iloc[:4] = 0
        >>> df
           a  b  c
        0  0  0  0
        1  0  0  0
        2  0  0  0
        3  0  0  0
        4  4  4  4
        5  5  5  5
        6  6  6  6
        7  7  7  7
        8  8  8  8
        9  9  9  9
        [10 more rows]

        """
        return self._iloc_indexer_type(self)

    @property
    @_performance_tracking
    def axes(self):
        """
        Return a list representing the axes of the Series.

        Series.axes returns a list containing the row index.

        Examples
        --------
        >>> import cudf
        >>> csf1 = cudf.Series([1, 2, 3, 4])
        >>> csf1.axes
        [RangeIndex(start=0, stop=4, step=1)]

        """
        return [self.index]

    def squeeze(self, axis: Literal["index", "columns", 0, 1, None] = None):
        """
        Squeeze 1 dimensional axis objects into scalars.

        Series or DataFrames with a single element are squeezed to a scalar.
        DataFrames with a single column or a single row are squeezed to a
        Series. Otherwise the object is unchanged.

        This method is most useful when you don't know if your
        object is a Series or DataFrame, but you do know it has just a single
        column. In that case you can safely call `squeeze` to ensure you have a
        Series.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default None
            A specific axis to squeeze. By default, all length-1 axes are
            squeezed. For `Series` this parameter is unused and defaults
            to `None`.

        Returns
        -------
        DataFrame, Series, or scalar
            The projection after squeezing `axis` or all the axes.

        See Also
        --------
        Series.iloc : Integer-location based indexing for selecting scalars.
        DataFrame.iloc : Integer-location based indexing for selecting Series.
        Series.to_frame : Inverse of DataFrame.squeeze for a
            single-column DataFrame.

        Examples
        --------
        >>> primes = cudf.Series([2, 3, 5, 7])

        Slicing might produce a Series with a single value:

        >>> even_primes = primes[primes % 2 == 0]
        >>> even_primes
        0    2
        dtype: int64

        >>> even_primes.squeeze()
        np.int64(2)

        Squeezing objects with more than one value in every axis does nothing:

        >>> odd_primes = primes[primes % 2 == 1]
        >>> odd_primes
        1    3
        2    5
        3    7
        dtype: int64

        >>> odd_primes.squeeze()
        1    3
        2    5
        3    7
        dtype: int64

        Squeezing is even more effective when used with DataFrames.

        >>> df = cudf.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        >>> df
           a  b
        0  1  2
        1  3  4

        Slicing a single column will produce a DataFrame with the columns
        having only one value:

        >>> df_a = df[["a"]]
        >>> df_a
           a
        0  1
        1  3

        So the columns can be squeezed down, resulting in a Series:

        >>> df_a.squeeze("columns")
        0    1
        1    3
        Name: a, dtype: int64

        Slicing a single row from a single column will produce a single
        scalar DataFrame:

        >>> df_0a = df.loc[df.index < 1, ["a"]]
        >>> df_0a
           a
        0  1

        Squeezing the rows produces a single scalar Series:

        >>> df_0a.squeeze("index")
        a    1
        Name: 0, dtype: int64

        Squeezing all axes will project directly into a scalar:

        >>> df_0a.squeeze()
        np.int64(1)
        """
        axes = (
            range(self.ndim)
            if axis is None
            else (self._get_axis_from_axis_arg(axis),)
        )
        indexer = tuple(
            0 if i in axes and len(a) == 1 else slice(None)
            for i, a in enumerate(self.axes)
        )
        return self.iloc[indexer]

    @_performance_tracking
    def scale(self):
        """
        Scale values to [0, 1] in float64

        Returns
        -------
        DataFrame or Series
            Values scaled to [0, 1].

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 11, 12, 0.5, 1])
        >>> series
        0    10.0
        1    11.0
        2    12.0
        3     0.5
        4     1.0
        dtype: float64
        >>> series.scale()
        0    0.826087
        1    0.913043
        2    1.000000
        3    0.000000
        4    0.043478
        dtype: float64
        """
        vmin = self.min()
        vmax = self.max()
        scaled = (self - vmin) / (vmax - vmin)
        scaled.index = self.index.copy(deep=False)
        return scaled

    @overload
    def sort_index(
        self,
        axis: Axis = ...,
        level=...,
        ascending: bool | Iterable[bool] = ...,
        inplace: Literal[False] = ...,
        kind: str = ...,
        na_position: Literal["first", "last"] = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key=...,
    ) -> Self: ...

    @overload
    def sort_index(
        self,
        axis: Axis = ...,
        level=...,
        ascending: bool | Iterable[bool] = ...,
        inplace: Literal[True] = ...,
        kind: str = ...,
        na_position: Literal["first", "last"] = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key=...,
    ) -> None: ...

    @_performance_tracking
    def sort_index(
        self,
        axis: Axis = 0,
        level=None,
        ascending: bool | Iterable[bool] = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: Literal["first", "last"] = "last",
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key=None,
    ) -> Self | None:
        """Sort object by labels (along an axis).

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis along which to sort. The value 0 identifies the rows,
            and 1 identifies the columns.
        level : int or level name or list of ints or list of level names
            If not None, sort on values in specified index level(s).
            This is only useful in the case of MultiIndex.
        ascending : bool, default True
            Sort ascending vs. descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : sorting method such as `quick sort` and others.
            Not yet supported.
        na_position : {'first', 'last'}, default 'last'
            Puts NaNs at the beginning if first; last puts NaNs at the end.
        sort_remaining : bool, default True
            When sorting a multiindex on a subset of its levels,
            should entries be lexsorted by the remaining
            (non-specified) levels as well?
        ignore_index : bool, default False
            if True, index will be replaced with RangeIndex.
        key : callable, optional
            If not None, apply the key function to the index values before
            sorting. This is similar to the key argument in the builtin
            sorted() function, with the notable difference that this key
            function should be vectorized. It should expect an Index and return
            an Index of the same shape. For MultiIndex inputs, the key is
            applied per level.

        Returns
        -------
        Frame or None

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
        >>> series
        3    a
        2    b
        1    c
        4    d
        dtype: str
        >>> series.sort_index()
        1    c
        2    b
        3    a
        4    d
        dtype: str

        Sort Descending

        >>> series.sort_index(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: str

        **DataFrame**

        >>> df = cudf.DataFrame(
        ... {"b":[3, 2, 1], "a":[2, 1, 3]}, index=[1, 3, 2])
        >>> df.sort_index(axis=0)
           b  a
        1  3  2
        2  1  3
        3  2  1
        >>> df.sort_index(axis=1)
           a  b
        1  2  3
        3  1  2
        2  3  1

        .. pandas-compat::
            :meth:`pandas.DataFrame.sort_index`, :meth:`pandas.Series.sort_index`

            * Not supporting: kind, sort_remaining=False
        """
        if kind != "quicksort":
            raise NotImplementedError("kind is not yet supported")

        if key is not None:
            raise NotImplementedError("key is not yet supported.")

        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        if axis in (0, "index"):
            idx = self.index
            if isinstance(idx, MultiIndex):
                if level is not None:
                    if not is_list_like(level):
                        level = [level]
                    by = list(map(idx._get_level_label, level))
                    if sort_remaining:
                        handled = set(by)
                        by.extend(
                            filter(
                                lambda n: n not in handled,
                                self.index._column_names,
                            )
                        )
                else:
                    by = list(idx._column_names)

                inds = idx._get_sorted_inds(
                    by=by, ascending=ascending, na_position=na_position
                )
                out = self._gather(
                    GatherMap.from_column_unchecked(
                        inds, len(self), nullify=False
                    )
                )
                # TODO: frame factory function should handle multilevel column
                # names
                if isinstance(self, cudf.DataFrame) and self._data.multiindex:
                    out._set_columns_like(self._data)
            elif (ascending and idx.is_monotonic_increasing) or (
                not ascending and idx.is_monotonic_decreasing
            ):
                out = self.copy()
            else:
                inds = idx.argsort(
                    ascending=ascending, na_position=na_position
                )
                out = self._gather(
                    GatherMap.from_column_unchecked(
                        as_column(inds),  # type: ignore[arg-type]
                        len(self),
                        nullify=False,
                    )
                )
                if isinstance(self, cudf.DataFrame) and self._data.multiindex:
                    out._set_columns_like(self._data)
            if ignore_index:
                out = out.reset_index(drop=True)
        else:
            # The column labels are host-side pandas metadata, so have
            # pandas compute the sorted column order: a Series holding the
            # original positions, indexed by the column labels, sorted by
            # its index yields the positional indexer while inheriting
            # pandas' exact level resolution and validation, per-level
            # ascending, sort_remaining, and na_position semantics.
            pd_columns = self._data.to_pandas_index
            indexer = pd.Series(
                range(len(pd_columns)), index=pd_columns
            ).sort_index(
                level=level,
                ascending=ascending,
                sort_remaining=sort_remaining,
                na_position=na_position,
                kind="stable",
            )
            labels = [self._column_names[i] for i in indexer]
            result_columns = (self._data[label] for label in labels)
            if ignore_index:
                ca = ColumnAccessor(
                    dict(enumerate(result_columns)),
                    rangeindex=True,
                    verify=False,
                )
            else:
                ca = ColumnAccessor(
                    dict(zip(labels, result_columns, strict=True)),
                    rangeindex=self._data.rangeindex,
                    multiindex=self._data.multiindex,
                    level_names=self._data.level_names,
                    label_dtype=self._data.label_dtype,
                    level_dtypes=self._data._level_dtypes,
                    verify=False,
                )
            out = self._from_data_like_self(ca)

        return self._mimic_inplace(out, inplace=inplace)

    def hash_values(
        self,
        method: Literal[
            "murmur3",
            "xxhash64",
            "md5",
            "sha1",
            "sha224",
            "sha256",
            "sha384",
            "sha512",
        ] = "murmur3",
        seed: int | None = None,
    ) -> Series:
        """Compute the hash of values in this column.

        Parameters
        ----------
        method : {'murmur3', 'xxhash32', 'xxhash64', 'md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512'}, default 'murmur3'
            Hash function to use:

            * murmur3: MurmurHash3 hash function
            * xxhash32: xxHash32 hash function
            * xxhash64: xxHash64 hash function
            * md5: MD5 hash function
            * sha1: SHA-1 hash function
            * sha224: SHA-224 hash function
            * sha256: SHA-256 hash function
            * sha384: SHA-384 hash function
            * sha512: SHA-512 hash function

        seed : int, optional
            Seed value to use for the hash function. This parameter is only
            supported for 'murmur3', 'xxhash32', and 'xxhash64'.


        Returns
        -------
        Series
            A Series with hash values.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([10, 120, 30])
        >>> series
        0     10
        1    120
        2     30
        dtype: int64
        >>> series.hash_values(method="murmur3")
        0    4004982076
        1    2063150778
        2     699010651
        dtype: uint32
        >>> series.hash_values(method="md5")
        0    7be4bbacbfdb05fb3044e36c22b41e8b
        1    947ca8d2c5f0f27437f156cfbfab0969
        2    d0580ef52d27c043c8e341fd5039b166
        dtype: str
        >>> series.hash_values(method="murmur3", seed=42)
        0    4279022973
        1    1654398277
        2    1497690768
        dtype: uint32

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({"a": [10, 120, 30], "b": [0.0, 0.25, 0.50]})
        >>> df
             a     b
        0   10  0.00
        1  120  0.25
        2   30  0.50
        >>> df.hash_values(method="murmur3")
        0      92443192
        1     116637743
        2    2602001908
        dtype: uint32
        >>> df.hash_values(method="md5")
        0    57ce879751b5169c525907d5c563fae1
        1    948d6221a7c4963d4be411bcead7e32b
        2    fe061786ea286a515b772d91b0dfcd70
        dtype: str
        """
        seed_hash_methods = {"murmur3", "xxhash32", "xxhash64"}
        if seed is None:
            seed = 0
        elif method not in seed_hash_methods:
            warnings.warn(
                "Provided seed value has no effect for the hash method "
                f"`{method}`. Only {seed_hash_methods} support seeds."
            )
        with access_columns(*self._columns, mode="read", scope="internal"):
            plc_table = plc.Table([c.plc_column for c in self._columns])
            if method == "murmur3":
                plc_column = plc.hashing.murmurhash3_x86_32(plc_table, seed)
            elif method == "xxhash32":
                plc_column = plc.hashing.xxhash_32(plc_table, seed)
            elif method == "xxhash64":
                plc_column = plc.hashing.xxhash_64(plc_table, seed)
            elif method == "md5":
                plc_column = plc.hashing.md5(plc_table)
            elif method == "sha1":
                plc_column = plc.hashing.sha1(plc_table)
            elif method == "sha224":
                plc_column = plc.hashing.sha224(plc_table)
            elif method == "sha256":
                plc_column = plc.hashing.sha256(plc_table)
            elif method == "sha384":
                plc_column = plc.hashing.sha384(plc_table)
            elif method == "sha512":
                plc_column = plc.hashing.sha512(plc_table)
            else:
                raise ValueError(f"Unsupported hashing algorithm {method}.")
            result = ColumnBase.create(
                plc_column, dtype=dtype_from_pylibcudf_column(plc_column)
            )
        return cudf.Series._from_column(
            result,
            index=self.index,
        )

    def _gather(
        self,
        gather_map: GatherMap,
        keep_index=True,
    ):
        """Gather rows of frame specified by indices in `gather_map`.

        Maintain the index if keep_index is True.

        This function does no expensive bounds checking, but does
        check that the number of rows of self matches the validated
        number of rows.
        """

        if not gather_map.nullify and len(self) != gather_map.nrows:
            raise IndexError("Gather map is out of bounds")
        index_names = self.index.names if keep_index else None

        try:
            # No need to gather if the gather map is already in the correct order
            can_gather_with_copy = (
                len(gather_map.column) == len(self)
                and len(self) > 0
                and gather_map.column.equals(
                    ColumnBase.from_range(range(len(self)))
                )
            )
        except (AttributeError, TypeError):
            can_gather_with_copy = False
        if can_gather_with_copy:
            if keep_index:
                return self.copy(deep=True)
            return self._from_columns_like_self(
                [col.copy(deep=True) for col in self._columns],
                self._column_names,
                index_names,
            )
        columns_to_gather = (
            list(itertools.chain(self.index._columns, self._columns))
            if keep_index
            else self._columns
        )
        return self._from_columns_like_self(
            [
                ColumnBase.create(col, orig_col.dtype)
                for col, orig_col in zip(
                    copying.gather(
                        columns_to_gather,
                        gather_map.column,
                        nullify=gather_map.nullify,
                    ),
                    columns_to_gather,
                    strict=True,
                )
            ],
            self._column_names,
            index_names,
        )

    def _slice(self, arg: slice, keep_index: bool = True) -> Self:
        """Slice a frame.

        Parameters
        ----------
        arg
            The slice
        keep_index
            Preserve the index when slicing?

        Returns
        -------
        Sliced frame

        Notes
        -----
        This slicing has normal python semantics.
        """
        if arg.step == 0:
            raise ValueError("slice step cannot be zero")
        num_rows = len(self)
        if num_rows == 0:
            return self
        start, stop, stride = arg.indices(num_rows)
        index = self.index
        has_range_index = isinstance(index, RangeIndex)
        if len(range(start, stop, stride)) == 0:
            # Avoid materialising the range index column
            result = self._empty_like(
                keep_index=keep_index and not has_range_index
            )
            if keep_index and has_range_index:
                lo = index.start + start * index.step
                hi = index.start + stop * index.step
                step = index.step * stride
                result.index = RangeIndex(
                    start=lo, stop=hi, step=step, name=index.name
                )
            return result
        if start < 0:
            start = start + num_rows

        # At this point, we have converted slice arguments into
        # indices that no longer wrap around.
        # For example slice(4, None, -1) will produce the
        # start, stop, stride tuple (4, -1, -1)
        # This check makes sure -1 is not wrapped (again) to
        # produce -1 + num_rows.
        if stop < 0 and not (stride < 0 and stop == -1):
            stop = stop + num_rows
        stride = 1 if stride is None else stride

        if (stop - start) * stride <= 0:
            return self._empty_like(keep_index=True)

        start = min(start, num_rows)
        stop = min(stop, num_rows)

        if stride != 1:
            return self._gather(
                GatherMap.from_column_unchecked(
                    cast(
                        "cudf.core.column.numerical.NumericalColumn",
                        ColumnBase.from_range(
                            range(start, stop, stride)
                        ).astype(SIZE_TYPE_DTYPE),
                    ),
                    len(self),
                    nullify=False,
                ),
                keep_index=keep_index,
            )

        cols_list = list(
            itertools.chain(self.index._columns, self._columns)
            if keep_index and not has_range_index
            else self._columns
        )
        with access_columns(
            *cols_list, mode="read", scope="internal"
        ) as flattened_cols_list:
            plc_tables = plc.copying.slice(
                plc.Table([col.plc_column for col in flattened_cols_list]),
                [start, stop],
            )
            sliced = [
                ColumnBase.create(plc_result, dtype=reference_col.dtype)
                for plc_result, reference_col in zip(
                    plc_tables[0].columns(), flattened_cols_list, strict=True
                )
            ]
        result = self._from_columns_like_self(
            sliced,
            self._column_names,
            None if has_range_index or not keep_index else self.index.names,
        )

        if keep_index and has_range_index:
            result.index = self.index[start:stop]
        return result

    def _positions_from_column_names(
        self,
        column_names: set[Hashable],
        offset_by_index_columns: bool = True,
    ) -> list[int]:
        """Map each column name into their positions in the frame.

        Return positions of the provided column names, offset by the number of
        index columns if `offset_by_index_columns` is True. The order of
        indices returned corresponds to the column order in this Frame.
        """
        start = self.index.nlevels if offset_by_index_columns else 0
        return [
            i
            for i, name in enumerate(self._column_names, start=start)
            if name in column_names
        ]

    def drop_duplicates(
        self,
        subset=None,
        keep: Literal["first", "last", False] = "first",
        nulls_are_equal: bool = True,
        ignore_index: bool = False,
    ):
        """
        Drop duplicate rows in frame.

        subset : list, optional
            List of columns to consider when dropping rows.
        keep : ["first", "last", False]
            "first" will keep the first duplicate entry, "last" will keep the
            last duplicate entry, and False will drop all duplicates.
        nulls_are_equal: bool, default True
            Null elements are considered equal to other null elements.
        ignore_index: bool, default False
            If True, the resulting axis will be labeled 0, 1, ..., n - 1.
        """
        if not isinstance(ignore_index, (np.bool_, bool)):
            raise ValueError(
                f"{ignore_index=} must be bool, "
                f"not {type(ignore_index).__name__}"
            )
        subset = self._preprocess_subset(subset)
        subset_cols = [name for name in self._column_names if name in subset]
        if len(subset_cols) == 0:
            return self.copy(deep=True)

        columns = (
            list(self._columns)
            if ignore_index
            else list(self.index._columns + self._columns)
        )
        original_dtypes = [col.dtype for col in columns]
        keys = self._positions_from_column_names(
            subset, offset_by_index_columns=not ignore_index
        )
        result_columns = self._drop_duplicates_columns(
            columns,
            keys=keys,
            keep=keep,
            nulls_are_equal=nulls_are_equal,
        )
        return self._from_columns_like_self(
            [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    result_columns, original_dtypes, strict=True
                )
            ],
            self._column_names,
            self.index.names if not ignore_index else None,
        )

    @_performance_tracking
    def duplicated(
        self, subset=None, keep: Literal["first", "last", False] = "first"
    ) -> Series:
        """
        Return boolean Series denoting duplicate rows.

        Considering certain columns is optional.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to mark.

            - ``'first'`` : Mark duplicates as ``True`` except for the first
                occurrence.
            - ``'last'`` : Mark duplicates as ``True`` except for the last
                occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series
            Boolean series indicating duplicated rows.

        See Also
        --------
        Index.duplicated : Equivalent method on index.
        Series.duplicated : Equivalent method on Series.
        Series.drop_duplicates : Remove duplicate values from Series.
        DataFrame.drop_duplicates : Remove duplicate values from DataFrame.

        Examples
        --------
        Consider a dataset containing ramen product ratings.

        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     'brand': ['Yum Yum', 'Yum Yum', 'Maggie', 'Maggie', 'Maggie'],
        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
        ...     'rating': [4, 4, 3.5, 15, 5]
        ... })
        >>> df
             brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2   Maggie   cup     3.5
        3   Maggie  pack    15.0
        4   Maggie  pack     5.0

        By default, for each set of duplicated values, the first occurrence
        is set to False and all others to True.

        >>> df.duplicated()
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set to False and all others to True.

        >>> df.duplicated(keep='last')
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        By setting ``keep`` to False, all duplicates are True.

        >>> df.duplicated(keep=False)
        0     True
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        To find duplicates on specific column(s), use ``subset``.

        >>> df.duplicated(subset=['brand'])
        0    False
        1     True
        2    False
        3     True
        4     True
        dtype: bool
        """
        subset = self._preprocess_subset(subset)

        name = None
        if isinstance(self, cudf.Series):
            columns = [self._column]
            name = self.name
        else:
            columns = [self._data[n] for n in subset]

        _keep_options = {
            "first": plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            "last": plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
            False: plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
        }

        if (keep_option := _keep_options.get(keep)) is None:
            raise ValueError('keep must be either "first", "last" or False')

        with access_columns(*columns, mode="read", scope="internal"):
            distinct_indices = plc.stream_compaction.distinct_indices(
                plc.Table([col.plc_column for col in columns]),
                keep_option,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
            )
            source = [
                plc.Scalar.from_py(False, dtype=plc.DataType(plc.TypeId.BOOL8))
            ]
            target = plc.Table(
                [
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(
                            True, dtype=plc.DataType(plc.TypeId.BOOL8)
                        ),
                        len(self),
                    )
                ]
            )
            plc_result = plc.copying.scatter(
                source,
                distinct_indices,
                target,
            ).columns()[0]
            result = ColumnBase.create(
                plc_result, dtype=dtype_from_pylibcudf_column(plc_result)
            )
        return cudf.Series._from_column(
            result, index=self.index, name=name, attrs=self.attrs
        )

    @_performance_tracking
    def _empty_like(self, keep_index: bool = True) -> Self:
        columns_to_access = list(
            itertools.chain(self.index._columns, self._columns)
            if keep_index
            else self._columns
        )
        original_dtypes = [col.dtype for col in columns_to_access]
        with access_columns(*columns_to_access, mode="read", scope="internal"):
            plc_table = plc.copying.empty_like(
                plc.Table(
                    [
                        col.plc_column
                        for col in (
                            itertools.chain(self.index._columns, self._columns)
                            if keep_index
                            else self._columns
                        )
                    ]
                )
            )
            columns = [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    plc_table.columns(), original_dtypes, strict=True
                )
            ]
        result = self._from_columns_like_self(
            columns,
            self._column_names,
            self.index.names if keep_index else None,
        )
        result._data.label_dtype = self._data.label_dtype
        result._data.rangeindex = self._data.rangeindex
        return result

    def _split(self, splits: list[int], keep_index: bool = True) -> list[Self]:
        if self._num_rows == 0:
            return []

        # Materialize the iterator and enter contexts
        source_columns_list = list(
            itertools.chain(self.index._columns, self._columns)
            if keep_index
            else self._columns
        )
        original_dtypes = [col.dtype for col in source_columns_list]
        with access_columns(
            *source_columns_list, mode="read", scope="internal"
        ):
            columns_split = copying.columns_split(
                source_columns_list,
                splits,
            )

            def split_with_dtypes(
                split: tuple[plc.Column, ...],
            ) -> list[ColumnBase]:
                return [
                    ColumnBase.create(col, dtype)
                    for col, dtype in zip(split, original_dtypes, strict=True)
                ]

            return [
                self._from_columns_like_self(
                    split_with_dtypes(split),
                    self._column_names,
                    self.index.names if keep_index else None,
                )
                for split in columns_split
            ]

    @_performance_tracking
    def bfill(
        self,
        *,
        axis: Axis | None = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside", None] = None,
    ) -> Self | None:
        """
        Fill NA/NaN values by using the next valid observation to fill the gap.

        Returns
        -------
            Object with missing values filled or None if ``inplace=True``.
        """
        return self._fillna(
            method=plc.replace.ReplacePolicy.FOLLOWING,
            axis=axis,
            inplace=inplace,
            limit=limit,
            limit_area=limit_area,
        )

    @_performance_tracking
    def ffill(
        self,
        *,
        axis: Axis | None = None,
        inplace: bool = False,
        limit: None | int = None,
        limit_area: Literal["inside", "outside", None] = None,
    ) -> Self | None:
        """
        Fill NA/NaN values by propagating the last valid observation to next valid.

        Returns
        -------
            Object with missing values filled or None if ``inplace=True``.
        """
        return self._fillna(
            method=plc.replace.ReplacePolicy.PRECEDING,
            axis=axis,
            inplace=inplace,
            limit=limit,
            limit_area=limit_area,
        )

    def add_prefix(self, prefix, axis=None):
        """
        Prefix labels with string `prefix`.

        For Series, the row labels are prefixed.
        For DataFrame, the column labels are prefixed.

        Parameters
        ----------
        prefix : str
            The string to add before each label.

        Returns
        -------
        Series or DataFrame
            New Series with updated labels or DataFrame with updated labels.

        See Also
        --------
        Series.add_suffix: Suffix row labels with string 'suffix'.
        DataFrame.add_suffix: Suffix column labels with string 'suffix'.

        Examples
        --------
        **Series**

        >>> s = cudf.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        >>> s.add_prefix('item_')
        item_0    1
        item_1    2
        item_2    3
        item_3    4
        dtype: int64

        **DataFrame**

        >>> df = cudf.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6
        >>> df.add_prefix('col_')
             col_A  col_B
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        raise NotImplementedError(
            "`IndexedFrame.add_prefix` not currently implemented. \
                Use `Series.add_prefix` or `DataFrame.add_prefix`"
        )

    def add_suffix(self, suffix, axis=None):
        """
        Suffix labels with string `suffix`.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        prefix : str
            The string to add after each label.

        Returns
        -------
        Series or DataFrame
            New Series with updated labels or DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: prefix row labels with string 'prefix'.
        DataFrame.add_prefix: Prefix column labels with string 'prefix'.

        Examples
        --------
        **Series**

        >>> s = cudf.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        >>> s.add_suffix('_item')
        0_item    1
        1_item    2
        2_item    3
        3_item    4
        dtype: int64

        **DataFrame**

        >>> df = cudf.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6
        >>> df.add_suffix('_col')
             A_col  B_col
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        raise NotImplementedError

    @_performance_tracking
    def _apply(self, func, kernel_class, *args, **kwargs):
        """Apply `func` across the rows of the frame."""
        if kwargs:
            raise ValueError("UDFs using **kwargs are not yet supported.")
        try:
            kr = kernel_class(self, func, args)
            kernel, retty = kr.get_kernel()
        except Exception as e:
            raise ValueError(
                "user defined function compilation failed."
            ) from e

        # Mask and data column preallocated
        ans_col = _return_arr_from_dtype(retty, len(self))
        ans_mask = as_column(True, length=len(self), dtype=np.dtype("bool"))
        output_args = [(ans_col, ans_mask), len(self)]
        input_args = _get_input_args_from_frame(self)
        launch_args = output_args + input_args + list(args)
        try:
            with _CUDFNumbaConfig():
                kernel.forall(len(self))(*launch_args)
        except Exception as e:
            raise RuntimeError("UDF kernel execution failed.") from e

        if is_dtype_obj_string(retty):
            plc_col = strings_udf.column_from_managed_udf_string_array(ans_col)
            col = ColumnBase.create(
                plc_col, dtype=dtype_from_pylibcudf_column(plc_col)
            )
            free_kernel = _make_free_string_kernel()
            with _CUDFNumbaConfig():
                free_kernel.forall(len(col))(ans_col, len(col))
        else:
            col = as_column(ans_col, retty)

        mask_buff, null_count = ans_mask.as_mask()
        col = col.set_mask(mask_buff, null_count)
        result = cudf.Series._from_column(
            col, index=self.index, attrs=self.attrs
        )

        return result

    def sort_values(
        self,
        by,
        axis: Axis = 0,
        ascending: bool | Iterable[bool] = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: Literal["first", "last"] = "last",
        ignore_index: bool = False,
        key=None,
    ):
        """Sort by the values along either axis.

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders. If this is a list of bools, must match the length of the
            by.
        na_position : {'first', 'last'}, default 'last'
            'first' puts nulls at the beginning, 'last' puts nulls at the end
        ignore_index : bool, default False
            If True, index will not be sorted.
        key : callable, optional
            Apply the key function to the values
            before sorting. This is similar to the ``key`` argument in the
            builtin ``sorted`` function, with the notable difference that
            this ``key`` function should be *vectorized*. It should expect a
            ``Series`` and return a Series with the same shape as the input.
            It will be applied to each column in `by` independently.
            Currently not supported.

        Returns
        -------
        Frame : Frame with sorted values.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['a'] = [0, 1, 2]
        >>> df['b'] = [-3, 2, 0]
        >>> df.sort_values('b')
           a  b
        0  0 -3
        2  2  0
        1  1  2

        .. pandas-compat::
            :meth:`pandas.DataFrame.sort_values`, :meth:`pandas.Series.sort_values`

            * Support axis='index' only.
            * Not supporting: inplace, kind
        """
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")
        if inplace:
            raise NotImplementedError("`inplace` not currently implemented.")
        if kind != "quicksort":
            if kind not in {"mergesort", "heapsort", "stable"}:
                raise AttributeError(
                    f"{kind} is not a valid sorting algorithm for "
                    f"'DataFrame' object"
                )
            warnings.warn(
                f"GPU-accelerated {kind} is currently not supported, "
                f"defaulting to quicksort."
            )
        if axis != 0:
            raise NotImplementedError("`axis` not currently implemented.")
        if key is not None:
            raise NotImplementedError("key is not currently supported.")

        if len(self) == 0:
            return self

        try:
            by_in_columns = self._get_columns_by_label(by)
        except KeyError:
            by_in_columns = None
        if self.ndim == 1:
            # For Series case, we're never selecting an index level.
            by_in_index = None
        else:
            try:
                by_in_index = self.index._get_columns_by_label(by)
            except KeyError:
                by_in_index = None

        if by_in_columns is not None and by_in_index is not None:
            raise ValueError(
                f"{by=} appears in the {type(self).__name__} columns "
                "and as an index level which is ambiguous."
            )
        elif by_in_columns is not None:
            by_columns = by_in_columns
        elif by_in_index is not None:
            by_columns = by_in_index
        else:
            raise KeyError(by)

        if cudf.get_option("mode.pandas_compatible"):
            by_columns = by_columns.nans_to_nulls()
        # argsort the `by` column
        out = self._gather(
            GatherMap.from_column_unchecked(
                by_columns._get_sorted_inds(
                    ascending=ascending, na_position=na_position
                ),
                len(self),
                nullify=False,
            ),
            keep_index=not ignore_index,
        )
        return out

    def _n_largest_or_smallest(
        self, largest: bool, n: int, columns, keep: Literal["first", "last"]
    ) -> Self:
        # Get column to operate on
        if isinstance(columns, str):
            columns = [columns]

        method = "nlargest" if largest else "nsmallest"
        for col in columns:
            col_dtype = self._data[col].dtype
            # pandas rejects nlargest/nsmallest on object, complex, string,
            # and categorical dtypes.
            if (
                is_dtype_obj_string(col_dtype)
                or getattr(col_dtype, "kind", None) in {"O", "c"}
                or isinstance(col_dtype, pd.CategoricalDtype)
            ):
                if isinstance(self, cudf.DataFrame):
                    error_msg = (
                        f"Column '{col}' has dtype {col_dtype}, "
                        f"cannot use method '{method}' with this dtype"
                    )
                else:
                    error_msg = (
                        f"Cannot use method '{method}' with dtype {col_dtype}"
                    )
                raise TypeError(error_msg)
        if len(self) == 0:
            return self

        if keep == "first":
            if n < 0:
                n = 0

            # argsort the `by` column
            return self._gather(
                GatherMap.from_column_unchecked(
                    self._get_columns_by_label(columns)
                    ._get_sorted_inds(ascending=not largest)
                    .slice(*slice(None, n).indices(len(self))),
                    len(self),
                    nullify=False,
                ),
                keep_index=True,
            )
        elif keep == "last":
            indices = self._get_columns_by_label(columns)._get_sorted_inds(
                ascending=largest
            )

            if n <= 0:
                # Empty slice.
                indices = indices.slice(0, 0)
            else:
                indices = indices.slice(
                    *slice(None, -n - 1, -1).indices(len(self))
                )
            return self._gather(
                GatherMap.from_column_unchecked(
                    indices, len(self), nullify=False
                ),
                keep_index=True,
            )
        else:
            raise ValueError('keep must be either "first", "last"')

    def _align_to_index(
        self,
        index: ColumnLike,
        how: str = "outer",
        sort: bool = True,
        allow_non_unique: bool = False,
    ) -> Self:
        index = ensure_index(index)

        if self.index.equals(index):
            return self
        if not allow_non_unique:
            if not self.index.is_unique or not index.is_unique:
                raise ValueError("Cannot align indices with non-unique values")

        # Join only the indexes, with a positional column standing in for
        # this frame's rows, and gather the data columns afterwards. Routing
        # the data columns through the merge would subject them to pandas'
        # merge dtype semantics (e.g. the numpy int -> float64 upcast on
        # unmatched rows), which apply to user-facing merges but not to
        # alignment.
        lhs_index, rhs_index = _unify_categorical_indexes(self.index, index)
        pos_col_id = str(uuid4())
        lhs = cudf.DataFrame._from_data(
            {pos_col_id: ColumnBase.from_range(range(len(self)))},
            index=lhs_index,
        )
        rhs = cudf.DataFrame._from_data({}, index=rhs_index)

        # create a temporary column that we will later sort by
        # to recover ordering after index alignment.
        sort_col_id = str(uuid4())
        if how == "right":
            rhs[sort_col_id] = ColumnBase.from_range(range(len(rhs)))

        result = lhs.join(rhs, how=how, sort=sort)
        if how == "left":
            result = result.sort_values(pos_col_id)
        elif how == "right":
            result = result.sort_values(sort_col_id)

        out = self._gather(
            _gather_map_from_positions(result._data[pos_col_id], len(self)),
            keep_index=False,
        )
        out.index = result.index
        out.index.names = self.index.names
        return out

    @_performance_tracking
    def _reindex(
        self,
        column_names,
        dtypes=None,
        deep=False,
        index=None,
        inplace=False,
        fill_value=NA,
        level=None,
        method=None,
        limit=None,
        tolerance=None,
    ):
        """
        Helper for `.reindex`

        Parameters
        ----------
        columns_names : array-like
            array-like of columns to select from the Frame,
            if ``columns`` is a superset of ``Frame.columns`` new
            columns are created.
        dtypes : dict
            Mapping of dtypes for the empty columns being created.
        deep : boolean, optional, default False
            Whether to make deep copy or shallow copy of the columns.
        index : Index or array-like, default None
            The ``index`` to be used to reindex the Frame with.
        inplace : bool, default False
            Whether to perform the operation in place on the data.
        fill_value : value with which to replace nulls in the result

        Returns
        -------
        Series or DataFrame
        """
        if method is not None:
            raise NotImplementedError("method is not currently supported.")
        if level is not None:
            raise NotImplementedError("level is not currently supported.")
        if limit is not None:
            raise NotImplementedError("limit is not currently supported.")
        if tolerance is not None:
            raise NotImplementedError("tolerance is not currently supported.")

        if dtypes is None:
            dtypes = {}

        df = self
        # Original column dtypes, captured before any index handling
        # mutates ``df``; used to infer the dtype of brand-new columns.
        orig_col_dtypes = [dtype for _, dtype in self._dtypes]
        frame_common_dtype = (
            orig_col_dtypes[0]
            if (
                orig_col_dtypes
                and all(dt == orig_col_dtypes[0] for dt in orig_col_dtypes)
                and isinstance(orig_col_dtypes[0], np.dtype)
            )
            else None
        )
        row_reindex = index is not None
        rows_added = False
        if index is not None:
            if not df.index.is_unique:
                raise ValueError(
                    "cannot reindex on an axis with duplicate labels"
                )
            index = cudf.Index(
                index, name=getattr(index, "name", self.index.name)
            )

            idx_dtype_match = (df.index.nlevels == index.nlevels) and all(
                _is_same_dtype(left_dtype, right_dtype)
                for left_dtype, right_dtype in zip(
                    (dtype for _, dtype in df.index._dtypes),
                    (dtype for _, dtype in index._dtypes),
                    strict=True,
                )
            )

            if not idx_dtype_match:
                column_names = (
                    column_names
                    if column_names is not None
                    else list(df._column_names)
                )
                df = cudf.DataFrame()
            else:
                rhs = cudf.DataFrame._from_data(
                    {
                        # bookkeeping workaround for unnamed series
                        (name or 0)
                        if isinstance(self, cudf.Series)
                        else name: col
                        for name, col in df._column_labels_and_values
                    },
                    index=df.index,
                )
                diff = index.difference(df.index)
                rows_added = len(diff) > 0
                # Join only the indexes and gather the data columns natively
                # (see ``_align_to_index``): the merge's pandas dtype
                # semantics must not leak into reindexing, whose own dtype
                # rules are applied below. Row order comes from a positional
                # column on the target rather than from sorting the joined
                # key: a value sort's collation can disagree with the
                # target's own ordering (e.g. a categorical target joined
                # against a string source is decategorized and would sort
                # lexically rather than by category).
                pos_col_id = str(uuid4())
                order_col_id = str(uuid4())
                lhs = cudf.DataFrame._from_data(
                    {order_col_id: ColumnBase.from_range(range(len(index)))},
                    index=index,
                )
                pos_rhs = cudf.DataFrame._from_data(
                    {pos_col_id: ColumnBase.from_range(range(len(rhs)))},
                    index=rhs.index,
                )
                joined = lhs.join(pos_rhs, how="left", sort=False)
                joined = joined.sort_values(order_col_id)
                df = rhs._gather(
                    _gather_map_from_positions(
                        joined._data[pos_col_id], len(rhs)
                    ),
                    keep_index=False,
                )
                df.index = joined.index
                if fill_value is not NA and rows_added:
                    df.loc[diff] = fill_value

        index = index if index is not None else df.index

        label_dtype = None
        if column_names is None:
            names = list(df._column_names)
            level_names = self._data.level_names
            multiindex = self._data.multiindex
            rangeindex = self._data.rangeindex
            label_dtype = self._data.label_dtype
        elif isinstance(column_names, (pd.Index, cudf.Index)):
            if isinstance(column_names, (pd.MultiIndex, cudf.MultiIndex)):
                multiindex = True
                if isinstance(column_names, cudf.MultiIndex):
                    names = list(iter(column_names.to_pandas()))
                else:
                    names = list(iter(column_names))
                rangeindex = False
            else:
                multiindex = False
                names = column_names
                if isinstance(names, cudf.Index):
                    names = names.to_pandas()
                rangeindex = isinstance(
                    column_names, (pd.RangeIndex, cudf.RangeIndex)
                )
                if not rangeindex:
                    label_dtype = (
                        column_names.dtype
                        if isinstance(column_names, pd.Index)
                        else column_names.to_pandas().dtype
                    )
            level_names = tuple(column_names.names)
        else:
            names = column_names
            level_names = None
            multiindex = False
            rangeindex = False
            if len(names) == 0:
                # pandas' Index.reindex treats an empty non-Index target as
                # ``columns[:0]``, preserving the original columns' metadata
                # (label dtype, names, RangeIndex/MultiIndex-ness).
                level_names = self._data.level_names
                multiindex = self._data.multiindex
                rangeindex = self._data.rangeindex
                label_dtype = self._data.label_dtype

        def _new_nulls_column(name):
            # Build a brand-new column produced by reindex (entirely missing
            # or fill values), choosing a dtype that matches pandas.
            if fill_value is NA:
                # All-null new column: keep a homogeneous float dtype,
                # else float64 (numpy integer/bool cannot hold NaN).
                if name in dtypes:
                    target = dtypes[name]
                elif (
                    frame_common_dtype is not None
                    and frame_common_dtype.kind == "f"
                ):
                    target = frame_common_dtype
                else:
                    target = np.dtype(np.float64)
                # A numpy integer dtype cannot hold NA, so pandas upcasts an
                # all-null reindexed column to float64. Match that here,
                # mirroring the upcast applied to existing integer columns
                # below so both paths agree. An empty result (row_count == 0)
                # holds no NA, so the integer dtype is preserved -- matching
                # pandas and cudf's prior behavior for e.g. reindex to an
                # empty index.
                if (
                    isinstance(target, np.dtype)
                    and target.kind in "iu"
                    and len(index) > 0
                ):
                    target = np.dtype(np.float64)
                return column_empty(dtype=target, row_count=len(index))
            # Non-null fill. A numeric scalar fill on a brand-new column of a
            # homogeneous numpy frame promotes the frame dtype against the fill
            # value (uint8 + 10 -> uint8, uint8 + 300 -> int64); otherwise the
            # dtype is the fill value's own. Any other fill keeps the source /
            # float64 default. ``fillna`` raises on incompatible fills so
            # cudf.pandas falls back to pandas.
            if (
                name not in dtypes
                and is_scalar(fill_value)
                and (scalar_col := as_column(fill_value, length=1)).dtype.kind
                in "iuf"
            ):
                if row_reindex and frame_common_dtype is not None:
                    target = (
                        frame_common_dtype
                        if scalar_col.can_cast_safely(frame_common_dtype)
                        else find_common_type(
                            [frame_common_dtype, scalar_col.dtype]
                        )
                    )
                else:
                    target = scalar_col.dtype
            else:
                target = dtypes.get(name, np.dtype(np.float64))
            return column_empty(dtype=target, row_count=len(index)).fillna(
                fill_value
            )

        # cudf cannot represent duplicate column names; pandas can. Raise
        # so cudf.pandas falls back to pandas rather than silently
        # collapsing duplicates.
        names_list = list(names)
        if len(names_list) != len(set(names_list)):
            raise ValueError("Duplicate column names are not allowed")

        # pandas upcasts integer columns to float64 when default-NaN
        # filling newly added rows on reindex (a numpy integer column
        # cannot hold NA), so match that unconditionally.
        upcast_int_nulls = rows_added and fill_value is NA

        cols = {}
        for name in names_list:
            if name in df._data:
                col = df._data[name].copy(deep=deep)
                if (
                    upcast_int_nulls
                    # Only plain numpy integer columns cannot hold NA;
                    # nullable extension integers (masked ``IntX``,
                    # ``ArrowDtype``) natively represent NA, so pandas
                    # keeps their dtype rather than upcasting to float64.
                    and isinstance(col.dtype, np.dtype)
                    and col.dtype.kind in "iu"
                    and col.null_count
                ):
                    col = col.astype(np.dtype(np.float64))
            else:
                col = _new_nulls_column(name)
            cols[name] = col

        result = self.__class__._from_data(
            data=ColumnAccessor(
                cols,
                multiindex=multiindex,
                level_names=level_names,
                rangeindex=rangeindex,
                label_dtype=label_dtype,
            ),
            index=index,
            attrs=self.attrs,
        )

        return self._mimic_inplace(result, inplace=inplace)

    def round(self, decimals=0, how="half_even"):
        """
        Round to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. This parameter
            must be an int for a Series. For a DataFrame, a dict or a Series
            are also valid inputs. If an int is given, round each column to the
            same number of places. Otherwise dict and Series round to variable
            numbers of places. Column names should be in the keys if
            `decimals` is a dict-like, or in the index if `decimals` is a
            Series. Any columns not included in `decimals` will be left as is.
            Elements of `decimals` which are not columns of the input will be
            ignored.
        how : str, optional
            Type of rounding. Can be either "half_even" (default)
            or "half_up" rounding.

        Returns
        -------
        Series or DataFrame
            A Series or DataFrame with the affected columns rounded to the
            specified number of decimal places.

        Examples
        --------
        **Series**

        >>> s = cudf.Series([0.1, 1.4, 2.9])
        >>> s.round()
        0    0.0
        1    1.0
        2    3.0
        dtype: float64

        **DataFrame**

        >>> df = cudf.DataFrame(
        ...     [(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
        ...     columns=['dogs', 'cats'],
        ... )
        >>> df
           dogs  cats
        0  0.21  0.32
        1  0.01  0.67
        2  0.66  0.03
        3  0.21  0.18

        By providing an integer each column is rounded to the same number
        of decimal places.

        >>> df.round(1)
           dogs  cats
        0   0.2   0.3
        1   0.0   0.7
        2   0.7   0.0
        3   0.2   0.2

        With a dict, the number of places for specific columns can be
        specified with the column names as keys and the number of decimal
        places as values.

        >>> df.round({'dogs': 1, 'cats': 0})
           dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0

        Using a Series, the number of places for specific columns can be
        specified with the column names as the index and the number of
        decimal places as the values.

        >>> decimals = cudf.Series([0, 1], index=['cats', 'dogs'])
        >>> df.round(decimals)
           dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0
        """
        if isinstance(decimals, cudf.Series):
            decimals = decimals.to_pandas()

        if isinstance(decimals, pd.Series):
            if not decimals.index.is_unique:
                raise ValueError("Index of decimals must be unique")
            decimals = decimals.to_dict()
        elif isinstance(decimals, int):
            decimals = {name: decimals for name in self._column_names}
        elif not isinstance(decimals, Mapping):
            raise TypeError(
                "decimals must be an integer, a dict-like or a Series"
            )

        cols = (
            col.round(decimals[name], how=how)
            if name in decimals
            and (
                col.dtype.kind in "fiu"
                or isinstance(
                    col.dtype, (cudf.Decimal32Dtype, cudf.Decimal64Dtype)
                )
            )
            else col.copy(deep=True)
            for name, col in self._column_labels_and_values
        )
        return self._from_data_like_self(
            self._data._from_columns_like_self(cols)
        )

    def resample(
        self,
        rule,
        closed: Literal["right", "left"] | None = None,
        label: Literal["right", "left"] | None = None,
        convention: Literal["start", "end", "s", "e"] = "start",
        on=None,
        level=None,
        origin="start_day",
        offset=None,
        group_keys: bool = False,
    ):
        """
        Convert the frequency of ("resample") the given time series data.

        Parameters
        ----------
        rule: str
            The offset string representing the frequency to use.
            Note that DateOffset objects are not yet supported.
        closed: {"right", "left"}, default None
            Which side of bin interval is closed. The default is
            "left" for all frequency offsets except for "M" and "W",
            which have a default of "right".
        label: {"right", "left"}, default None
            Which bin edge label to label bucket with. The default is
            "left" for all frequency offsets except for "M" and "W",
            which have a default of "right".
        on: str, optional
            For a DataFrame, column to use instead of the index for
            resampling.  Column must be a datetime-like.
        level: str or int, optional
            For a MultiIndex, level to use instead of the index for
            resampling.  The level must be a datetime-like.

        Returns
        -------
        A Resampler object

        Examples
        --------
        First, we create a time series with 1 minute intervals:

        >>> index = cudf.date_range(start="2001-01-01", periods=10, freq="1min")
        >>> sr = cudf.Series(range(10), index=index)
        >>> sr
        2001-01-01 00:00:00    0
        2001-01-01 00:01:00    1
        2001-01-01 00:02:00    2
        2001-01-01 00:03:00    3
        2001-01-01 00:04:00    4
        2001-01-01 00:05:00    5
        2001-01-01 00:06:00    6
        2001-01-01 00:07:00    7
        2001-01-01 00:08:00    8
        2001-01-01 00:09:00    9
        dtype: int64

        Downsampling to 3 minute intervals, followed by a "sum" aggregation:

        >>> sr.resample("3min").sum()
        2001-01-01 00:00:00     3
        2001-01-01 00:03:00    12
        2001-01-01 00:06:00    21
        2001-01-01 00:09:00     9
        dtype: int64

        Use the right side of each interval to label the bins:

        >>> sr.resample("3min", label="right").sum()
        2001-01-01 00:03:00     3
        2001-01-01 00:06:00    12
        2001-01-01 00:09:00    21
        2001-01-01 00:12:00     9
        dtype: int64

        Close the right side of the interval instead of the left:

        >>> sr.resample("3min", closed="right").sum()
        2000-12-31 23:57:00     0
        2001-01-01 00:00:00     6
        2001-01-01 00:03:00    15
        2001-01-01 00:06:00    24
        dtype: int64

        Upsampling to 30 second intervals:

        >>> sr.resample("30s").asfreq()[:5]  # show the first 5 rows
        2001-01-01 00:00:00       0
        2001-01-01 00:00:30    <NA>
        2001-01-01 00:01:00       1
        2001-01-01 00:01:30    <NA>
        2001-01-01 00:02:00       2
        dtype: int64

        Upsample and fill nulls using the "bfill" method:

        >>> sr.resample("30s").bfill()[:5]
        2001-01-01 00:00:00    0
        2001-01-01 00:00:30    1
        2001-01-01 00:01:00    1
        2001-01-01 00:01:30    2
        2001-01-01 00:02:00    2
        dtype: int64

        Resampling by a specified column of a Dataframe:

        >>> df = cudf.DataFrame({
        ...     "price": [10, 11, 9, 13, 14, 18, 17, 19],
        ...     "volume": [50, 60, 40, 100, 50, 100, 40, 50],
        ...     "week_starting": cudf.date_range(
        ...         "2018-01-01", periods=8, freq="7D"
        ...     )
        ... })
        >>> df
        price  volume week_starting
        0     10      50    2018-01-01
        1     11      60    2018-01-08
        2      9      40    2018-01-15
        3     13     100    2018-01-22
        4     14      50    2018-01-29
        5     18     100    2018-02-05
        6     17      40    2018-02-12
        7     19      50    2018-02-19
        >>> df.resample("2W", on="week_starting").mean()
                       price  volume
        week_starting
        2017-12-31      10.5    55.0
        2018-01-14      11.0    70.0
        2018-01-28      16.0    75.0
        2018-02-11      18.0    45.0


        .. pandas-compat::
            :meth:`pandas.DataFrame.resample`, :meth:`pandas.Series.resample`

            Note that the dtype of the index (or the 'on' column if using
            'on=') in the result will be of a frequency closest to the
            resampled frequency.  For example, if resampling from
            nanoseconds to milliseconds, the index will be of dtype
            'datetime64[ms]'.
        """
        from cudf.core.resample import DataFrameResampler, SeriesResampler

        if convention != "start":
            raise NotImplementedError("convention is currently not supported.")
        if origin != "start_day":
            raise NotImplementedError("origin is currently not supported.")
        if offset is not None:
            raise NotImplementedError("offset is currently not supported.")
        if group_keys is not False:
            raise NotImplementedError("group_keys is currently not supported.")
        by = cudf.Grouper(
            key=on, freq=rule, closed=closed, label=label, level=level
        )
        return (
            SeriesResampler(self, by=by)
            if isinstance(self, cudf.Series)
            else DataFrameResampler(self, by=by)
        )

    def dropna(
        self,
        axis=0,
        how="any",
        thresh=None,
        subset=None,
        inplace=False,
        ignore_index: bool = False,
    ):
        """
        Drop rows (or columns) containing nulls from a Column.

        Parameters
        ----------
        axis : {0, 1}, optional
            Whether to drop rows (axis=0, default) or columns (axis=1)
            containing nulls.
        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row (or column).
            any (default) drops rows (or columns) containing at least
            one null value. all drops only rows (or columns) containing
            *all* null values.
        thresh: int, optional
            If specified, then drops every row (or column) containing
            less than `thresh` non-null values
        subset : list, optional
            List of columns to consider when dropping rows (all columns
            are considered by default). Alternatively, when dropping
            columns, subset is a list of rows to consider.
        inplace : bool, default False
            If True, do operation inplace and return None.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Copy of the DataFrame with rows/columns containing nulls dropped.

        See Also
        --------
        cudf.DataFrame.isna
            Indicate null values.
        cudf.DataFrame.notna
            Indicate non-null values.
        cudf.DataFrame.fillna
            Replace null values.
        cudf.Series.dropna
            Drop null values.
        cudf.Index.dropna
            Drop null indices.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
        ...                    "toy": ['Batmobile', None, 'Bullwhip'],
        ...                    "born": [np.datetime64("1940-04-25"),
        ...                             np.datetime64("NaT"),
        ...                             np.datetime64("NaT")]})
        >>> df
               name        toy                           born
        0    Alfred  Batmobile  1940-04-25 00:00:00.000000000
        1    Batman        NaN                            NaT
        2  Catwoman   Bullwhip                            NaT

        Drop the rows where at least one element is null.

        >>> df.dropna()
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Drop the columns where at least one element is null.

        >>> df.dropna(axis='columns')
               name
        0    Alfred
        1    Batman
        2  Catwoman

        Drop the rows where all elements are null.

        >>> df.dropna(how='all')
               name        toy                           born
        0    Alfred  Batmobile  1940-04-25 00:00:00.000000000
        1    Batman        NaN                            NaT
        2  Catwoman   Bullwhip                            NaT

        Keep only the rows with at least 2 non-null values.

        >>> df.dropna(thresh=2)
               name        toy                           born
        0    Alfred  Batmobile  1940-04-25 00:00:00.000000000
        2  Catwoman   Bullwhip                            NaT

        Define in which columns to look for null values.

        >>> df.dropna(subset=['name', 'born'])
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Keep the DataFrame with valid entries in the same variable.

        >>> df.dropna(inplace=True)
        >>> df
             name        toy       born
        0  Alfred  Batmobile 1940-04-25
        """
        if axis in [0, "index"]:
            result = self._drop_na_rows(how=how, subset=subset, thresh=thresh)
            if ignore_index:
                result.index = RangeIndex(len(result))
        else:
            result = self._drop_na_columns(
                how=how, subset=subset, thresh=thresh
            )

        return self._mimic_inplace(result, inplace=inplace)

    @_performance_tracking
    def _drop_na_columns(self, how="any", subset=None, thresh=None):
        """
        Drop columns containing nulls
        """
        out_cols = []

        if subset is None:
            df = self
        else:
            df = self.take(subset)

        if thresh is None:
            if how == "all":
                thresh = 1
            else:
                thresh = len(df)

        for name, col in df._column_labels_and_values:
            check_col = col.nans_to_nulls()
            no_threshold_valid_count = check_col.valid_count < thresh
            if no_threshold_valid_count:
                continue
            out_cols.append(name)

        return self[out_cols]

    def _drop_na_rows(self, how="any", subset=None, thresh=None):
        """
        Drop null rows from `self`.

        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row.
            any (default) drops rows containing at least
            one null value. all drops only rows containing
            *all* null values.
        subset : list, optional
            List of columns to consider when dropping rows.
        thresh : int, optional
            If specified, then drops every row containing
            less than `thresh` non-null values.
        """
        subset = self._preprocess_subset(subset)

        if len(subset) == 0:
            return self.copy(deep=True)

        data_columns = [col.nans_to_nulls() for col in self._columns]
        columns = [*self.index._columns, *data_columns]
        original_dtypes = [col.dtype for col in columns]
        keys = self._positions_from_column_names(subset)

        result_columns = self._drop_nulls_columns(
            columns,
            keys=keys,
            how=how,
            thresh=thresh,
        )
        return self._from_columns_like_self(
            [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    result_columns, original_dtypes, strict=True
                )
            ],
            self._column_names,
            self.index.names,
        )

    def _apply_boolean_mask(self, boolean_mask: BooleanMask, keep_index=True):
        """Apply boolean mask to each row of `self`.

        Rows corresponding to `False` is dropped.

        If keep_index is False, the index is not preserved.
        """
        if len(boolean_mask.column) != len(self):
            raise IndexError(
                "Boolean mask has wrong length: "
                f"{len(boolean_mask.column)} not {len(self)}"
            )
        columns = (
            list(self.index._columns + self._columns)
            if keep_index
            else list(self._columns)
        )
        original_dtypes = [col.dtype for col in columns]
        mask = boolean_mask.column
        with access_columns(*columns, mask, mode="read", scope="internal") as (
            *cols,
            mask_col,
        ):
            plc_table = plc.stream_compaction.apply_boolean_mask(
                plc.Table([col.plc_column for col in cols]),
                mask_col.plc_column,
            )
            result = self._from_columns_like_self(
                [
                    ColumnBase.create(col, dtype)
                    for col, dtype in zip(
                        plc_table.columns(), original_dtypes, strict=True
                    )
                ],
                column_names=self._column_names,
                index_names=self.index.names if keep_index else None,
            )
        if (
            keep_index
            and isinstance(self.index, MultiIndex)
            and self.index._levels is not None
        ):
            result.index._levels = self.index._levels
            result.index._codes = [
                code.apply_boolean_mask(boolean_mask.column)
                for code in self.index._codes
            ]
        return result

    def _pandas_repr_compatible(self, nan_rep=None) -> Self:
        """Return Self but with columns prepared for a pandas-like repr."""
        if self.size == 0 and nan_rep is None:
            nan_rep = "nan"
        result = super()._pandas_repr_compatible(nan_rep=nan_rep)
        result.index = self.index._pandas_repr_compatible(nan_rep=nan_rep)
        return result

    def take(self, indices, axis=0):
        """Return a new frame containing the rows specified by *indices*.

        Parameters
        ----------
        indices : array-like
            Array of ints indicating which positions to take.
        axis : Unsupported

        Returns
        -------
        out : Series or DataFrame
            New object with desired subset of rows.

        Examples
        --------
        **Series**
        >>> s = cudf.Series(['a', 'b', 'c', 'd', 'e'])
        >>> s.take([2, 0, 4, 3])
        2    c
        0    a
        4    e
        3    d
        dtype: str

        **DataFrame**

        >>> a = cudf.DataFrame({'a': [1.0, 2.0, 3.0],
        ...                    'b': cudf.Series(['a', 'b', 'c'])})
        >>> a.take([0, 2, 2])
             a  b
        0  1.0  a
        2  3.0  c
        2  3.0  c
        """
        if self._get_axis_from_axis_arg(axis) != 0:
            raise NotImplementedError("Only axis=0 is supported.")

        return self._gather(GatherMap(indices, len(self), nullify=False))

    def _reset_index(
        self,
        level,
        drop,
        col_level=0,
        col_fill="",
        allow_duplicates: bool = False,
        names: Hashable | Sequence[Hashable] | None = None,
    ):
        """Shared path for DataFrame.reset_index and Series.reset_index."""
        if allow_duplicates is not False:
            raise NotImplementedError(
                "allow_duplicates is not currently supported."
            )
        elif names is not None:
            raise NotImplementedError("names is not currently supported.")
        if level is not None:
            if (
                isinstance(level, int)
                and level > 0
                and not isinstance(self.index, MultiIndex)
            ):
                raise IndexError(
                    f"Too many levels: Index has only 1 level, not {level + 1}"
                )
            if not isinstance(level, (tuple, list)):
                level = (level,)
        _check_duplicate_level_names(level, self.index.names)

        index = self.index._new_index_for_reset_index(level, self.index.name)
        if index is None:
            index = RangeIndex(len(self))
        if drop:
            return self._data, index

        new_column_data = {}
        for name, col in self.index._columns_for_reset_index(level):
            if name == "index" and "index" in self._data:
                name = "level_0"
            name = (
                tuple(
                    name if i == col_level else col_fill
                    for i in range(self._data.nlevels)
                )
                if self._data.multiindex
                else name
            )
            new_column_data[name] = col
        # This is to match pandas where the new data columns are always
        # inserted to the left of existing data columns.
        label_dtype = None
        if not self._data.multiindex:
            # pandas computes the result columns by Index.insert into the
            # existing columns Index, which preserves its dtype (e.g.
            # Index([None], dtype=object).insert(0, "a") stays object);
            # rebuilding from the merged labels would re-infer (pandas 3.0
            # infers "str" for all-string labels). Emulate the insert
            # provenance.
            pd_columns = self._data.to_pandas_index
            for new_name in reversed(list(new_column_data)):
                pd_columns = pd_columns.insert(0, new_name)
            label_dtype = pd_columns.dtype
        return (
            ColumnAccessor(
                {**new_column_data, **self._data},
                self._data.multiindex,
                self._data._level_names,
                label_dtype=label_dtype,
            ),
            index,
        )

    @_performance_tracking
    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
        ignore_index=False,
    ):
        """Return a random sample of items from an axis of object.

        If reproducible results are required, a random number generator may be
        provided via the `random_state` parameter. This function will always
        produce the same sample given an identical `random_state`.

        Parameters
        ----------
        n : int, optional
            Number of items from axis to return. Cannot be used with `frac`.
            Default = 1 if frac = None.
        frac : float, optional
            Fraction of axis items to return. Cannot be used with n.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
            `replace == True` is not supported for axis = 1/"columns".
            `replace == False` is not supported for axis = 0/"index" given
            `random_state` is `None` or a cupy random state, and `weights` is
            specified.
        weights : ndarray-like, optional
            Default `None` for uniform probability distribution over rows to
            sample from. If `ndarray` is passed, the length of `weights` should
            equal to the number of rows to sample from, and will be normalized
            to have a sum of 1. Unlike pandas, index alignment is not currently
            not performed.
        random_state : int, numpy/cupy RandomState, or None, default None
            If None, default cupy random state is chosen.
            If int, the seed for the default cupy random state.
            If RandomState, rows-to-sample are generated from the RandomState.
        axis : {0 or `index`, 1 or `columns`, None}, default None
            Axis to sample. Accepts axis number or name.
            Default is stat axis for given data type
            (0 for Series and DataFrames). Series doesn't support axis=1.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing n items
            randomly sampled from the caller object.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"a":{1, 2, 3, 4, 5}})
        >>> df.sample(3, random_state=0)  # doctest: +SKIP
           a
        0  1
        1  2
        4  5

        >>> sr = cudf.Series([1, 2, 3, 4, 5])
        >>> sr.sample(10, replace=True, random_state=0)
        1    2
        4    5
        1    2
        2    3
        4    5
        1    2
        4    5
        2    3
        4    5
        1    2
        dtype: int64

        >>> df = cudf.DataFrame(
        ...     {"a": [1, 2], "b": [2, 3], "c": [3, 4], "d": [4, 5]}
        ... )
        >>> df.sample(2, axis=1, random_state=0)
           c  d
        0  3  4
        1  4  5

        .. pandas-compat::
            :meth:`pandas.DataFrame.sample`, :meth:`pandas.Series.sample`

            When sampling from ``axis=0/'index'``, ``random_state`` can be
            either a numpy random state (``numpy.random.RandomState``)
            or a cupy random state (``cupy.random.RandomState``). When a numpy
            random state is used, the output is guaranteed to match the output
            of the corresponding pandas method call, but generating the sample
            maybe slow. If exact pandas equivalence is not required, using a
            cupy random state will achieve better performance,
            especially when sampling large number of
            items. It's advised to use the matching `ndarray` type to
            the random state for the `weights` array.
        """
        axis = 0 if axis is None else self._get_axis_from_axis_arg(axis)
        size = self.shape[axis]

        # Compute `n` from parameter `frac`.
        if frac is None:
            n = 1 if n is None else n
        else:
            if frac > 1 and not replace:
                raise ValueError(
                    "Replace has to be set to `True` when upsampling the "
                    "population `frac` > 1."
                )
            if n is not None:
                raise ValueError(
                    "Please enter a value for `frac` OR `n`, not both."
                )
            n = round(size * frac)

        if n > 0 and size == 0:
            raise ValueError(
                "Cannot take a sample larger than 0 when axis is empty."
            )

        if isinstance(random_state, cp.random.RandomState):
            lib = cp
        elif isinstance(random_state, np.random.RandomState):
            lib = np
        else:
            # Construct random state if `random_state` parameter is None or a
            # seed. By default, cupy random state is used to sample rows
            # and numpy is used to sample columns. This is because row data
            # is stored on device, and the column objects are stored on host.
            lib = cp if axis == 0 else np
            random_state = lib.random.RandomState(seed=random_state)

        # Normalize `weights` array.
        if weights is not None:
            if isinstance(weights, str):
                raise NotImplementedError(
                    "Weights specified by string is unsupported yet."
                )

            if size != len(weights):
                raise ValueError(
                    "Weights and axis to be sampled must be of same length."
                )

            weights = lib.asarray(weights)
            weights = weights / weights.sum()

        if axis == 0:
            return self._sample_axis_0(
                n, weights, replace, random_state, ignore_index
            )
        else:
            if isinstance(random_state, cp.random.RandomState):
                raise ValueError(
                    "Sampling from `axis=1`/`columns` with cupy random state"
                    "isn't supported."
                )
            return self._sample_axis_1(
                n, weights, replace, random_state, ignore_index
            )

    def _sample_axis_0(
        self,
        n: int,
        weights: ColumnLike | None,
        replace: bool,
        random_state: np.random.RandomState | cp.random.RandomState,
        ignore_index: bool,
    ):
        try:
            gather_map = GatherMap.from_column_unchecked(
                cast(
                    "cudf.core.column.numerical.NumericalColumn",
                    as_column(
                        random_state.choice(
                            len(self), size=n, replace=replace, p=weights
                        )
                    ),
                ),
                len(self),
                nullify=False,
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                "Random sampling with cupy does not support these inputs."
            ) from e

        return self._gather(gather_map, keep_index=not ignore_index)

    def _sample_axis_1(
        self,
        n: int,
        weights: ColumnLike | None,
        replace: bool,
        random_state: np.random.RandomState,
        ignore_index: bool,
    ):
        raise NotImplementedError(
            f"Sampling from axis 1 is not implemented for {self.__class__}."
        )

    def _binaryop(
        self,
        other: Any,
        op: str,
        fill_value: Any = None,
        can_reindex: bool = False,
        *args,
        **kwargs,
    ):
        reflect, op = self._check_reflected_op(op)
        (
            operands,
            out_index,
            ca_attributes,
        ) = self._make_operands_and_index_for_binop(
            other, op, fill_value, reflect, can_reindex
        )
        if operands is NotImplemented:
            return NotImplemented
        # Mirror pandas' `__finalize__(self).__finalize__(other)` for binary
        # ops: `other`'s attrs override `self`'s when non-empty, and
        # `allows_duplicate_labels` is the AND of both inputs.
        attrs = self.attrs
        allows_duplicate_labels = self._flags.allows_duplicate_labels
        if isinstance(other, IndexedFrame):
            if other.attrs:
                attrs = other.attrs
            allows_duplicate_labels = (
                allows_duplicate_labels
                and other._flags.allows_duplicate_labels
            )
        return self._from_data(
            ColumnAccessor(
                type(self)._colwise_binop(operands, op),
                **ca_attributes,
            ),
            index=out_index,
            attrs=copy.deepcopy(attrs),
            allows_duplicate_labels=allows_duplicate_labels,
        )

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
        cudf.Index | None,
        dict[str, Any],
    ]:
        raise NotImplementedError(
            f"Binary operations are not supported for {self.__class__}"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        fname = ufunc.__name__

        if ret is not None:
            return ret

        # Attempt to dispatch all other functions to cupy.
        cupy_func = getattr(cp, fname)
        if cupy_func:
            if ufunc.nin == 2:
                other = inputs[self is inputs[0]]
                inputs, index, _ = self._make_operands_and_index_for_binop(
                    other, fname
                )
            else:
                # This works for Index too
                inputs = {
                    name: (col, None, False, None)
                    for name, col in self._column_labels_and_values
                }
                index = self.index

            data = self._apply_cupy_ufunc_to_operands(
                ufunc, cupy_func, inputs, **kwargs
            )

            out = tuple(self._from_data(out, index=index) for out in data)
            return out[0] if ufunc.nout == 1 else out

        return NotImplemented

    @_performance_tracking
    def repeat(self, repeats, axis=None):
        """Repeats elements consecutively.

        Returns a new object of caller type(DataFrame/Series) where each
        element of the current object is repeated consecutively a given
        number of times.

        Parameters
        ----------
        repeats : int, or array of ints
            The number of repetitions for each element. This should
            be a non-negative integer. Repeating 0 times will return
            an empty object.

        Returns
        -------
        Series/DataFrame
            A newly created object of same type as caller
            with repeated elements.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        >>> df
           a   b
        0  1  10
        1  2  20
        2  3  30
        >>> df.repeat(3)
           a   b
        0  1  10
        0  1  10
        0  1  10
        1  2  20
        1  2  20
        1  2  20
        2  3  30
        2  3  30
        2  3  30

        Repeat on Series

        >>> s = cudf.Series([0, 2])
        >>> s
        0    0
        1    2
        dtype: int64
        >>> s.repeat([3, 4])
        0    0
        0    0
        0    0
        1    2
        1    2
        1    2
        1    2
        dtype: int64
        >>> s.repeat(2)
        0    0
        0    0
        1    2
        1    2
        dtype: int64
        """
        res = self._from_columns_like_self(
            self._repeat(
                [*self.index._columns, *self._columns], repeats, axis
            ),
            self._column_names,
            self._index_names,
        )
        if isinstance(res.index, cudf.DatetimeIndex):
            res.index._freq = None
        return res

    def astype(
        self,
        dtype: Dtype | dict[Hashable, Dtype],
        copy: bool | None = None,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> Self:
        """Cast the object to the given dtype.

        Parameters
        ----------
        dtype : data type, or dict of column name -> data type
            Use a :class:`numpy.dtype` or Python type to cast entire DataFrame
            object to the same type. Alternatively, use ``{col: dtype, ...}``,
            where col is a column label and dtype is a :class:`numpy.dtype`
            or Python type to cast one or more of the DataFrame's columns to
            column-specific types.
        copy : bool, default False
            Return a deep-copy when ``copy=True``. Note by default
            ``copy=False`` setting is used and hence changes to
            values then may propagate to other cudf objects.
        errors : {'raise', 'ignore', 'warn'}, default 'raise'
            Control raising of exceptions on invalid data for provided dtype.

            -   ``raise`` : allow exceptions to be raised
            -   ``ignore`` : suppress exceptions. On error return original
                object.

        Returns
        -------
        DataFrame/Series

        Examples
        --------
        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3]})
        >>> df
            a  b
        0  10  1
        1  20  2
        2  30  3
        >>> df.dtypes
        a    int64
        b    int64
        dtype: object

        Cast all columns to `int32`:

        >>> df.astype('int32').dtypes
        a    int32
        b    int32
        dtype: object

        Cast `a` to `float32` using a dictionary:

        >>> df.astype({'a': 'float32'}).dtypes
        a    float32
        b      int64
        dtype: object
        >>> df.astype({'a': 'float32'})
              a  b
        0  10.0  1
        1  20.0  2
        2  30.0  3

        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2], dtype='int32')
        >>> series
        0    1
        1    2
        dtype: int32
        >>> series.astype('int64')
        0    1
        1    2
        dtype: int64

        Convert to categorical type:

        >>> series.astype('category')
        0    1
        1    2
        dtype: category
        Categories (2, int64): [1, 2]

        Convert to ordered categorical type with custom ordering:

        >>> cat_dtype = cudf.CategoricalDtype(categories=[2, 1], ordered=True)
        >>> series.astype(cat_dtype)
        0    1
        1    2
        dtype: category
        Categories (2, int64): [2 < 1]

        Note that using ``copy=False`` (enabled by default)
        and changing data on a new Series will
        propagate changes:

        >>> s1 = cudf.Series([1, 2])
        >>> s1
        0    1
        1    2
        dtype: int64
        >>> s2 = s1.astype('int64', copy=False)
        >>> s2[0] = 10
        >>> s1
        0    10
        1     2
        dtype: int64
        """
        if errors not in ("ignore", "raise"):
            raise ValueError("invalid error value specified")

        try:
            return super().astype(dtype, copy)
        except Exception as e:
            if errors == "raise":
                raise e
            return self

    @_performance_tracking
    def _drop_column(
        self, name: Hashable, errors: Literal["ignore", "raise"] = "raise"
    ) -> None:
        """Drop a column by *name* inplace."""
        try:
            del self._data[name]
        except KeyError as err:
            if errors != "ignore":
                raise KeyError(f"column '{name}' does not exist") from err

    @_performance_tracking
    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace: bool = False,
        errors: Literal["ignore", "raise"] = "raise",
    ) -> Self | None:
        """Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding
        axis, or by specifying directly index or column names. When using a
        multi-index, labels on different levels can be removed by specifying
        the level.

        Parameters
        ----------
        labels : single label or list-like
            Index or column labels to drop.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') or
            columns (1 or 'columns').
        index : single label or list-like
            Alternative to specifying axis (``labels, axis=0``
            is equivalent to ``index=labels``).
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).
        level : int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
            If False, return a copy. Otherwise, do operation
            inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are
            dropped.

        Returns
        -------
        DataFrame or Series
            DataFrame or Series without the removed index or column labels.

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis.

        See Also
        --------
        DataFrame.loc : Label-location based indexer for selection by label.
        DataFrame.dropna : Return DataFrame with labels on given axis omitted
            where (all or any) data are missing.
        DataFrame.drop_duplicates : Return DataFrame with duplicate rows
            removed, optionally only considering certain columns.
        Series.reindex
            Return only specified index labels of Series
        Series.dropna
            Return series without null values
        Series.drop_duplicates
            Return series with duplicate values removed

        Examples
        --------
        **Series**

        >>> s = cudf.Series([1,2,3], index=['x', 'y', 'z'])
        >>> s
        x    1
        y    2
        z    3
        dtype: int64

        Drop labels x and z

        >>> s.drop(labels=['x', 'z'])
        y    2
        dtype: int64

        Drop a label from the second level in MultiIndex Series.

        >>> midx = cudf.MultiIndex.from_product([[0, 1, 2], ['x', 'y']])
        >>> s = cudf.Series(range(6), index=midx)
        >>> s
        0  x    0
           y    1
        1  x    2
           y    3
        2  x    4
           y    5
        dtype: int64
        >>> s.drop(labels='y', level=1)
        0  x    0
        1  x    2
        2  x    4
        dtype: int64

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 2, 3, 4],
        ...                      "B": [5, 6, 7, 8],
        ...                      "C": [10, 11, 12, 13],
        ...                      "D": [20, 30, 40, 50]})
        >>> df
           A  B   C   D
        0  1  5  10  20
        1  2  6  11  30
        2  3  7  12  40
        3  4  8  13  50

        Drop columns

        >>> df.drop(['B', 'C'], axis=1)
           A   D
        0  1  20
        1  2  30
        2  3  40
        3  4  50
        >>> df.drop(columns=['B', 'C'])
           A   D
        0  1  20
        1  2  30
        2  3  40
        3  4  50

        Drop a row by index

        >>> df.drop([0, 1])
           A  B   C   D
        2  3  7  12  40
        3  4  8  13  50

        Drop columns and/or rows of MultiIndex DataFrame

        >>> midx = cudf.MultiIndex(levels=[['lama', 'cow', 'falcon'],
        ...                              ['speed', 'weight', 'length']],
        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> df = cudf.DataFrame(index=midx, columns=['big', 'small'],
        ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
        ...                         [250, 150], [1.5, 0.8], [320, 250],
        ...                         [1, 0.8], [0.3, 0.2]])
        >>> df
                         big  small
        lama   speed    45.0   30.0
               weight  200.0  100.0
               length    1.5    1.0
        cow    speed    30.0   20.0
               weight  250.0  150.0
               length    1.5    0.8
        falcon speed   320.0  250.0
               weight    1.0    0.8
               length    0.3    0.2
        >>> df.drop(index='cow', columns='small')
                         big
        lama   speed    45.0
               weight  200.0
               length    1.5
        falcon speed   320.0
               weight    1.0
               length    0.3
        >>> df.drop(index='length', level=1)
                         big  small
        lama   speed    45.0   30.0
               weight  200.0  100.0
        cow    speed    30.0   20.0
               weight  250.0  150.0
        falcon speed   320.0  250.0
               weight    1.0    0.8
        """
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError(
                    "Cannot specify both 'labels' and 'index'/'columns'"
                )
            target = labels
        elif index is not None:
            target = index
            axis = 0
        elif columns is not None:
            target = columns
            axis = 1
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', "
                "'index' or 'columns'"
            )

        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean")
        elif inplace:
            out = self
        else:
            out = self.copy()

        if axis in (1, "columns"):
            for label in _get_unique_drop_labels(target):
                out._drop_column(label, errors=errors)
        elif axis in (0, "index"):
            dropped = _drop_rows_by_labels(out, target, level, errors)

            if columns is not None:
                for label in _get_unique_drop_labels(columns):
                    dropped._drop_column(label, errors=errors)

            out._mimic_inplace(dropped, inplace=True)

        if not inplace:
            return out
        return None

    @_performance_tracking
    def _explode(self, explode_label: Hashable, ignore_index: bool):
        # Helper function for `explode` in `Series` and `Dataframe`, explodes a
        # specified nested column. Other columns' corresponding rows are
        # duplicated. If ignore_index is set, the original index is not
        # exploded and will be replaced with a `RangeIndex`.
        explode_column = self._data[explode_label]
        if not is_dtype_obj_list(explode_column.dtype):
            result = self.copy()
            if ignore_index:
                result.index = RangeIndex(len(result))
            return result

        column_index = self._column_names.index(explode_label)
        idx_cols = self.index._columns if not ignore_index else ()

        # Build result dtypes: for the exploded column, use its element type
        # to preserve struct dtype key names; for all others, preserve original dtype
        element_type = cast("ListColumn", explode_column).element_type
        explode_column_idx = column_index + len(idx_cols)
        result_dtypes = [
            element_type if i == explode_column_idx else col.dtype
            for i, col in enumerate(itertools.chain(idx_cols, self._columns))
        ]

        with access_columns(
            *itertools.chain(idx_cols, self._columns),
            mode="read",
            scope="internal",
        ):
            plc_table = plc.lists.explode_outer(
                plc.Table(
                    [
                        col.plc_column
                        for col in itertools.chain(idx_cols, self._columns)
                    ]
                ),
                explode_column_idx,
            )
            exploded = [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    plc_table.columns(), result_dtypes, strict=True
                )
            ]

        data = type(self._data)(
            dict(
                zip(self._column_names, exploded[len(idx_cols) :], strict=True)
            ),
            multiindex=self._data.multiindex,
            level_names=self._data.level_names,
            rangeindex=self._data.rangeindex,
            label_dtype=self._data.label_dtype,
            verify=False,
        )
        if len(idx_cols):
            index = _index_from_data(
                dict(enumerate(exploded[: len(idx_cols)]))
            )
            if (
                isinstance(self.index, cudf.CategoricalIndex)
                and not isinstance(index, cudf.CategoricalIndex)
            ) or (
                isinstance(self.index, cudf.MultiIndex)
                and not isinstance(index, cudf.MultiIndex)
            ):
                index = type(self.index)._from_data(index._data)
            if isinstance(self.index, cudf.MultiIndex):
                index.names = self.index.names
            else:
                index.name = self.index.name
                if isinstance(self.index, cudf.DatetimeIndex):
                    index._freq = self.index._freq
        else:
            index = None

        result = type(self)._from_data(data, index)
        return result

    @_performance_tracking
    def tile(self, count: int):
        """Repeats the rows `count` times to form a new Frame.

        Parameters
        ----------
        self : input Table containing columns to interleave.
        count : Number of times to tile "rows". Must be non-negative.

        Examples
        --------
        >>> import cudf
        >>> df  = cudf.DataFrame([[8, 4, 7], [5, 2, 3]])
        >>> count = 2
        >>> df.tile(count)
           0  1  2
        0  8  4  7
        1  5  2  3
        0  8  4  7
        1  5  2  3

        Returns
        -------
        The indexed frame containing the tiled "rows".
        """
        source_columns = list(
            itertools.chain(self.index._columns, self._columns)
        )
        original_dtypes = [col.dtype for col in source_columns]
        with access_columns(
            *source_columns,
            mode="read",
            scope="internal",
        ):
            plc_table = plc.reshape.tile(
                plc.Table([col.plc_column for col in source_columns]),
                count,
            )
            tiled = [
                ColumnBase.create(plc, dtype)
                for plc, dtype in zip(
                    plc_table.columns(), original_dtypes, strict=True
                )
            ]
        return self._from_columns_like_self(
            tiled,
            column_names=self._column_names,
            index_names=self._index_names,
        )

    @_performance_tracking
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
        if sort is no_default:
            sort = cudf.get_option("mode.pandas_compatible")
        if group_keys is no_default:
            group_keys = cudf.get_option("mode.pandas_compatible")

        if axis not in (0, "index"):
            raise NotImplementedError("axis parameter is not yet implemented")

        if not observed:
            raise NotImplementedError(
                "observed parameter is not yet implemented"
            )

        if by is None and level is None:
            raise TypeError(
                "groupby() requires either by or level to be specified."
            )
        if group_keys is None:
            group_keys = False

        return (
            self.__class__._resampler(self, by=by)
            if isinstance(by, cudf.Grouper) and by.freq
            else self.__class__._groupby(
                self,
                by=by,
                level=level,
                as_index=as_index,
                dropna=dropna,
                sort=sort,
                group_keys=group_keys,
            )
        )

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Addition",
            op_name="add",
            equivalent_op="frame + other",
            df_op_example=textwrap.dedent(
                """
                >>> df.add(1)
                        angles  degrees
                circle          1      361
                triangle        4      181
                rectangle       5      361
                """,
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.add(b)
                a       2
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.add(b, fill_value=0)
                a       2
                b       1
                c       1
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def add(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__add__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Addition",
            op_name="radd",
            equivalent_op="other + frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.radd(1)
                        angles  degrees
                circle          1      361
                triangle        4      181
                rectangle       5      361
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.radd(b)
                a       2
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.radd(b, fill_value=0)
                a       2
                b       1
                c       1
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def radd(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__radd__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Subtraction",
            op_name="sub",
            equivalent_op="frame - other",
            df_op_example=textwrap.dedent(
                """
                >>> df.sub(1)
                        angles  degrees
                circle         -1      359
                triangle        2      179
                rectangle       3      359
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.sub(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.sub(b, fill_value=0)
                a       0
                b       1
                c       1
                d      -1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def subtract(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__sub__", fill_value)

    sub = subtract

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Subtraction",
            op_name="rsub",
            equivalent_op="other - frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rsub(1)
                        angles  degrees
                circle          1     -359
                triangle       -2     -179
                rectangle      -3     -359
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rsub(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rsub(b, fill_value=0)
                a       0
                b      -1
                c      -1
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def rsub(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rsub__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Multiplication",
            op_name="mul",
            equivalent_op="frame * other",
            df_op_example=textwrap.dedent(
                """
                >>> df.multiply(1)
                        angles  degrees
                circle          0      360
                triangle        3      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.multiply(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.multiply(b, fill_value=0)
                a       1
                b       0
                c       0
                d       0
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def multiply(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__mul__", fill_value)

    mul = multiply

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Multiplication",
            op_name="rmul",
            equivalent_op="other * frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rmul(1)
                        angles  degrees
                circle          0      360
                triangle        3      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rmul(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rmul(b, fill_value=0)
                a       1
                b       0
                c       0
                d       0
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def rmul(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rmul__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Modulo",
            op_name="mod",
            equivalent_op="frame % other",
            df_op_example=textwrap.dedent(
                """
                >>> df.mod(1)
                        angles  degrees
                circle          0        0
                triangle        0        0
                rectangle       0        0
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.mod(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.mod(b, fill_value=0)
                a     0.0
                b     NaN
                c     NaN
                d     0.0
                e     NaN
                dtype: float64
                """
            ),
        )
    )
    def mod(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__mod__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Modulo",
            op_name="rmod",
            equivalent_op="other % frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rmod(1)
                           angles  degrees
                circle        NaN        1
                triangle      1.0        1
                rectangle     1.0        1
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rmod(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rmod(b, fill_value=0)
                a     0.0
                b     0.0
                c     0.0
                d     NaN
                e     NaN
                dtype: float64
                """
            ),
        )
    )
    def rmod(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rmod__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Exponential",
            op_name="pow",
            equivalent_op="frame ** other",
            df_op_example=textwrap.dedent(
                """
                >>> df.pow(1)
                           angles  degrees
                circle          0      360
                triangle        3      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.pow(b)
                a       1
                b       1
                c       1
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.pow(b, fill_value=0)
                a       1
                b       1
                c       1
                d       0
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def pow(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__pow__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Exponential",
            op_name="rpow",
            equivalent_op="other ** frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rpow(1)
                        angles  degrees
                circle          1        1
                triangle        1        1
                rectangle       1        1
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rpow(b)
                a       1
                b    <NA>
                c    <NA>
                d       1
                e    <NA>
                dtype: int64
                >>> a.rpow(b, fill_value=0)
                a       1
                b       0
                c       0
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def rpow(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rpow__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Integer division",
            op_name="floordiv",
            equivalent_op="frame // other",
            df_op_example=textwrap.dedent(
                """
                >>> df.floordiv(1)
                        angles  degrees
                circle          0      360
                triangle        3      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.floordiv(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.floordiv(b, fill_value=0)
                a     1.0
                b     Inf
                c     Inf
                d     0.0
                e     NaN
                dtype: float64
                """
            ),
        )
    )
    def floordiv(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__floordiv__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Integer division",
            op_name="rfloordiv",
            equivalent_op="other // frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rfloordiv(1)
                           angles  degrees
                circle        inf        0
                triangle      0.0        0
                rectangle     0.0        0
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rfloordiv(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rfloordiv(b, fill_value=0)
                a     1.0
                b     0.0
                c     0.0
                d     Inf
                e     NaN
                dtype: float64
                """
            ),
        )
    )
    def rfloordiv(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rfloordiv__", fill_value)

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Floating division",
            op_name="truediv",
            equivalent_op="frame / other",
            df_op_example=textwrap.dedent(
                """
                >>> df.truediv(1)
                        angles  degrees
                circle        0.0    360.0
                triangle      3.0    180.0
                rectangle     4.0    360.0
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.truediv(b)
                a    1.0
                b    NaN
                c    NaN
                d    NaN
                e    NaN
                dtype: float64
                >>> a.truediv(b, fill_value=0)
                a    1.0
                b    Inf
                c    Inf
                d    0.0
                e    NaN
                dtype: float64
                """
            ),
        )
    )
    def truediv(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__truediv__", fill_value)

    # Alias for truediv
    div = truediv
    divide = truediv

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Floating division",
            op_name="rtruediv",
            equivalent_op="other / frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rtruediv(1)
                            angles   degrees
                circle          inf  0.002778
                triangle   0.333333  0.005556
                rectangle  0.250000  0.002778
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rtruediv(b)
                a    1.0
                b    NaN
                c    NaN
                d    NaN
                e    NaN
                dtype: float64
                >>> a.rtruediv(b, fill_value=0)
                a    1.0
                b    0.0
                c    0.0
                d    Inf
                e    NaN
                dtype: float64
                """
            ),
        )
    )
    def rtruediv(self, other, axis, level=None, fill_value=None):
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rtruediv__", fill_value)

    # Alias for rtruediv
    rdiv = rtruediv

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Equal to",
            op_name="eq",
            equivalent_op="frame == other",
            df_op_example=textwrap.dedent(
                """
                >>> df.eq(1)
                        angles  degrees
                circle      False    False
                triangle    False    False
                rectangle   False    False
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.eq(b)
                a    True
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.eq(b, fill_value=0)
                a    True
                b   False
                c   False
                d   False
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def eq(self, other, axis="columns", level=None, fill_value=None):
        return self._binaryop(
            other=other, op="__eq__", fill_value=fill_value, can_reindex=True
        )

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Not equal to",
            op_name="ne",
            equivalent_op="frame != other",
            df_op_example=textwrap.dedent(
                """
                >>> df.ne(1)
                        angles  degrees
                circle       True     True
                triangle     True     True
                rectangle    True     True
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.ne(b)
                a    False
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.ne(b, fill_value=0)
                a   False
                b    True
                c    True
                d    True
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def ne(self, other, axis="columns", level=None, fill_value=None):
        return self._binaryop(
            other=other, op="__ne__", fill_value=fill_value, can_reindex=True
        )

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Less than",
            op_name="lt",
            equivalent_op="frame < other",
            df_op_example=textwrap.dedent(
                """
                >>> df.lt(1)
                        angles  degrees
                circle       True    False
                triangle    False    False
                rectangle   False    False
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.lt(b)
                a   False
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.lt(b, fill_value=0)
                a   False
                b   False
                c   False
                d    True
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def lt(self, other, axis="columns", level=None, fill_value=None):
        return self._binaryop(
            other=other, op="__lt__", fill_value=fill_value, can_reindex=True
        )

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Less than or equal to",
            op_name="le",
            equivalent_op="frame <= other",
            df_op_example=textwrap.dedent(
                """
                >>> df.le(1)
                        angles  degrees
                circle       True    False
                triangle    False    False
                rectangle   False    False
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.le(b)
                a    True
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.le(b, fill_value=0)
                a    True
                b   False
                c   False
                d    True
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def le(self, other, axis="columns", level=None, fill_value=None):
        return self._binaryop(
            other=other, op="__le__", fill_value=fill_value, can_reindex=True
        )

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Greater than",
            op_name="gt",
            equivalent_op="frame > other",
            df_op_example=textwrap.dedent(
                """
                >>> df.gt(1)
                        angles  degrees
                circle      False     True
                triangle     True     True
                rectangle    True     True
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.gt(b)
                a   False
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.gt(b, fill_value=0)
                a   False
                b    True
                c    True
                d   False
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def gt(self, other, axis="columns", level=None, fill_value=None):
        return self._binaryop(
            other=other, op="__gt__", fill_value=fill_value, can_reindex=True
        )

    @_performance_tracking
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Greater than or equal to",
            op_name="ge",
            equivalent_op="frame >= other",
            df_op_example=textwrap.dedent(
                """
                >>> df.ge(1)
                        angles  degrees
                circle      False     True
                triangle     True     True
                rectangle    True     True
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.ge(b)
                a    True
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.ge(b, fill_value=0)
                a   True
                b    True
                c    True
                d   False
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def ge(self, other, axis="columns", level=None, fill_value=None):
        return self._binaryop(
            other=other, op="__ge__", fill_value=fill_value, can_reindex=True
        )

    def _preprocess_subset(self, subset) -> set[Hashable]:
        if subset is None:
            subset = self._column_names
        elif (
            is_scalar(subset)
            or isinstance(subset, tuple)
            and subset in self._column_names
        ):
            subset = (subset,)
        diff = set(subset) - set(self._column_names)
        if len(diff) != 0:
            raise KeyError(f"columns {diff} do not exist")
        return subset

    @_performance_tracking
    def rank(
        self,
        axis: Literal[0, "index"] = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> Self:
        """
        Compute numerical data ranks (1 through n) along axis.

        By default, equal values are assigned a rank that is the average of the
        ranks of those values.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Index to direct ranking.
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records that have the same value
            (i.e. ties):
            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups.
        numeric_only : bool, default False
            For DataFrame objects, rank only numeric columns if set to True.
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values:
            * keep: assign NaN rank to NaN values
            * top: assign smallest rank to NaN values if ascending
            * bottom: assign highest rank to NaN values if ascending.
        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.

        Returns
        -------
        same type as caller
            Return a Series or DataFrame with data ranks as values.
        """
        if method not in {"average", "min", "max", "first", "dense"}:
            raise KeyError(method)

        method_enum = plc.aggregation.RankMethod[method.upper()]
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError(
                "na_option must be one of 'keep', 'top', or 'bottom'"
            )

        if axis not in (0, "index"):
            raise NotImplementedError(
                f"axis must be `0`/`index`, "
                f"axis={axis} is not yet supported in rank"
            )

        num_cols = self._num_columns
        dropped_cols = False
        source = self
        if numeric_only:
            if isinstance(source, cudf.Series) and not is_dtype_obj_numeric(
                source.dtype, include_decimal=False
            ):
                raise TypeError(
                    "Series.rank does not allow numeric_only=True with "
                    "non-numeric dtype."
                )
            numeric_cols = (
                name
                for name, dtype in self._dtypes
                if is_dtype_obj_numeric(dtype, include_decimal=False)
            )
            source = self._get_columns_by_label(numeric_cols)
            if source.empty:
                return source.astype(np.dtype(np.float64))
            elif source._num_columns != num_cols:
                dropped_cols = True

        column_order = (
            plc.types.Order.ASCENDING
            if ascending
            else plc.types.Order.DESCENDING
        )
        # ascending
        #    #top    = na_is_smallest
        #    #bottom = na_is_largest
        #    #keep   = na_is_largest
        # descending
        #    #top    = na_is_largest
        #    #bottom = na_is_smallest
        #    #keep   = na_is_smallest
        if ascending:
            if na_option == "top":
                null_precedence = plc.types.NullOrder.BEFORE
            else:
                null_precedence = plc.types.NullOrder.AFTER
        else:
            if na_option == "top":
                null_precedence = plc.types.NullOrder.AFTER
            else:
                null_precedence = plc.types.NullOrder.BEFORE
        c_null_handling = (
            plc.types.NullPolicy.EXCLUDE
            if na_option == "keep"
            else plc.types.NullPolicy.INCLUDE
        )

        if cudf.get_option("mode.pandas_compatible"):
            source = source.nans_to_nulls()
        result_columns = [
            col.rank(
                method=method_enum,
                column_order=column_order,
                null_handling=c_null_handling,
                null_precedence=null_precedence,
                pct=pct,
            )
            for col in source._columns
        ]

        if dropped_cols:
            result = type(source)._from_data(
                ColumnAccessor(
                    dict(
                        zip(source._column_names, result_columns, strict=True)
                    ),
                    multiindex=source._data.multiindex,
                    level_names=source._data.level_names,
                    label_dtype=source._data.label_dtype,
                    verify=False,
                ),
            )
        else:
            result = source._from_data_like_self(
                self._data._from_columns_like_self(result_columns)
            )
        result.index = source.index
        return result

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend=None,
    ) -> Self:
        """
        Convert columns from numpy dtypes to the best dtypes that support ``pd.NA``.

        Parameters
        ----------
        infer_objects : bool, default True
            Whether object dtypes should be converted to the best possible types.
        convert_string : bool, default True
            Whether object dtypes should be converted to ``StringDtype()``.
        convert_integer : bool, default True
            Whether, if possible, conversion can be done to integer extension types.
        convert_boolean : bool, defaults True
            Whether object dtypes should be converted to ``BooleanDtypes()``.
        convert_floating : bool, defaults True
            Whether, if possible, conversion can be done to floating extension types.
            If `convert_integer` is also True, preference will be give to integer
            dtypes if the floats can be faithfully casted to integers.
        dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
            Back-end data type applied to the resultant :class:`DataFrame` or
            :class:`Series` (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed
              :class:`DataFrame` or :class:`Series`.
            * ``"pyarrow"``: returns pyarrow-backed nullable ``ArrowDtype``
              :class:`DataFrame` or :class:`Series`.

        Returns
        -------
        Series or DataFrame
            Copy of input object with new dtype.

        See Also
        --------
        to_datetime : Convert argument to datetime.
        to_numeric : Convert argument to a numeric type.

        Notes
        -----
        By default, ``convert_dtypes`` will attempt to convert a Series (or each
        Series in a DataFrame) to dtypes that support ``pd.NA``. By using the options
        ``convert_string``, ``convert_integer``, ``convert_boolean`` and
        ``convert_floating``, it is possible to turn off individual conversions
        to ``StringDtype``, the integer extension types, ``BooleanDtype``
        or floating extension types, respectively.

        For object-dtyped columns, if ``infer_objects`` is ``True``, use the inference
        rules as during normal Series/DataFrame construction.  Then, if possible,
        convert to ``StringDtype``, ``BooleanDtype`` or an appropriate integer
        or floating extension type, otherwise leave as ``object``.

        If the dtype is integer, convert to an appropriate integer extension type.

        If the dtype is numeric, and consists of all integers, convert to an
        appropriate integer extension type. Otherwise, convert to an
        appropriate floating extension type.

        In the future, as new dtypes are added that support ``pd.NA``, the results
        of this method will change to support those new dtypes.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> df = cudf.DataFrame(
        ...     {
        ...         "a": cudf.Series([1, 2, 3], dtype=np.dtype("int32")),
        ...         "b": cudf.Series(["x", "y", "z"], dtype=np.dtype("O")),
        ...         "e": cudf.Series([10, np.nan, 20], dtype=np.dtype("float")),
        ...         "f": cudf.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
        ...     }
        ... )

        Start with a DataFrame with default dtypes.

        >>> df
           a  b     e      f
        0  1  x  10.0    NaN
        1  2  y   NaN  100.5
        2  3  z  20.0  200.0

        >>> df.dtypes
        a      int32
        b     object
        e    float64
        f    float64
        dtype: object

        Convert the DataFrame to use best possible dtypes.

        >>> dfn = df.convert_dtypes()
        >>> dfn
           a  b     e      f
        0  1  x    10   <NA>
        1  2  y  <NA>  100.5
        2  3  z    20  200.0

        >>> dfn.dtypes
        a      Int32
        b     string
        e      Int64
        f    Float64
        dtype: object

        Start with a Series of strings and missing data represented by ``np.nan``.

        >>> s = cudf.Series(["a", "b", np.nan])
        >>> s
        0      a
        1      b
        2    NaN
        dtype: str

        Obtain a Series with dtype ``StringDtype``.

        >>> s.convert_dtypes()
        0       a
        1       b
        2    <NA>
        dtype: string
        """
        if dtype_backend not in (None, "numpy_nullable", "pyarrow"):
            raise ValueError(
                f"dtype_backend {dtype_backend} is invalid, only "
                "'numpy_nullable' and 'pyarrow' are allowed."
            )
        numpy_to_nullable = {
            "int8": pd.Int8Dtype(),
            "int16": pd.Int16Dtype(),
            "int32": pd.Int32Dtype(),
            "int64": pd.Int64Dtype(),
            "uint8": pd.UInt8Dtype(),
            "uint16": pd.UInt16Dtype(),
            "uint32": pd.UInt32Dtype(),
            "uint64": pd.UInt64Dtype(),
            "float32": pd.Float32Dtype(),
            "float64": pd.Float64Dtype(),
        }
        cols = []
        # Tracks whether each column was "selected" by the convert-* flags
        # (used by the ``dtype_backend="pyarrow"`` path below to decide whether
        # the column should be converted to ``pd.ArrowDtype``).
        converted = []
        for col in self._columns:
            dtype = col.dtype
            new_col = None
            # Map ``pd.ArrowDtype`` inputs to their numpy-nullable equivalents
            # only when explicitly asked via ``dtype_backend="numpy_nullable"``.
            # Otherwise keep ``pd.ArrowDtype`` columns unchanged (matches pandas).
            if (
                isinstance(dtype, pd.ArrowDtype)
                and dtype_backend == "numpy_nullable"
            ):
                pa_dtype = dtype.pyarrow_dtype
                pa_name = str(pa_dtype)
                if pa_name in numpy_to_nullable:
                    eff_dtype = numpy_to_nullable[pa_name]
                elif pa.types.is_boolean(pa_dtype):
                    eff_dtype = pd.BooleanDtype()
                elif pa.types.is_string(pa_dtype) or pa.types.is_large_string(
                    pa_dtype
                ):
                    eff_dtype = pd.StringDtype(na_value=pd.NA)
                else:
                    eff_dtype = dtype
            else:
                eff_dtype = dtype

            # ``selected`` indicates that the column was actually transformed
            # by one of the convert-* flags (used by the pyarrow-backend path
            # to decide whether to map to ``pd.ArrowDtype``).
            selected = False
            if eff_dtype == np.dtype("bool") or isinstance(
                eff_dtype, pd.BooleanDtype
            ):
                if convert_boolean:
                    selected = True
                    new_col = col.astype(pd.BooleanDtype(), copy=False)
                elif convert_integer:
                    selected = True
                    new_col = col.astype(pd.Int64Dtype(), copy=False)
                elif convert_floating:
                    selected = True
                    new_col = col.astype(pd.Float64Dtype(), copy=False)
            elif (
                isinstance(eff_dtype, np.dtype)
                and eff_dtype.kind in ("i", "u")
                and eff_dtype.name in numpy_to_nullable
            ):
                if convert_integer:
                    selected = True
                    new_col = col.astype(
                        numpy_to_nullable[eff_dtype.name], copy=False
                    )
            elif is_pandas_nullable_numpy_dtype(
                eff_dtype
            ) and eff_dtype.kind in (
                "i",
                "u",
            ):
                if convert_integer:
                    selected = True
                    new_col = col.astype(eff_dtype, copy=False)
            elif (
                isinstance(eff_dtype, np.dtype)
                and eff_dtype.kind == "f"
                and eff_dtype.name in numpy_to_nullable
            ):
                col_filled = col.fillna(0)
                as_int = col_filled.astype(np.dtype(np.int64))
                nan_blocks_int = (
                    pd.options.future.distinguish_nan_and_na
                    and col.nan_count > 0
                )
                if (
                    convert_integer
                    and not nan_blocks_int
                    and bool(cp.allclose(col_filled, as_int))
                ):
                    selected = True
                    new_col = col.nans_to_nulls().astype(
                        pd.Int64Dtype(), copy=False
                    )
                elif convert_floating:
                    selected = True
                    new_col = col.astype(
                        numpy_to_nullable[eff_dtype.name], copy=False
                    )
            elif (
                is_pandas_nullable_numpy_dtype(eff_dtype)
                and eff_dtype.kind == "f"
            ):
                if convert_floating:
                    selected = True
                    new_col = col.astype(eff_dtype, copy=False)
            elif isinstance(eff_dtype, pd.StringDtype):
                # Empty string columns have no values to infer from; pandas
                # leaves the corresponding object column unchanged in that
                # case. Match that behavior by skipping conversion.
                if convert_string and len(col) > 0:
                    selected = True
                    if eff_dtype.na_value is not pd.NA:
                        new_col = col.astype(
                            pd.StringDtype(na_value=pd.NA), copy=False
                        )
            elif eff_dtype == np.dtype("O"):
                # Empty object columns have no values to infer from, so
                # pandas leaves them as object. Skip conversion to match.
                if len(col) == 0:
                    pass
                elif convert_string:
                    selected = True
                    new_col = col.astype(
                        pd.StringDtype(na_value=pd.NA), copy=False
                    )
                elif infer_objects:
                    selected = True
                    new_col = col.astype(
                        pd.StringDtype(na_value=np.nan), copy=False
                    )
            was_converted = selected
            if new_col is None:
                # No conversion applied: still return a copy so that
                # modifications to the result don't propagate to ``self``
                # (matches pandas' ``convert_dtypes`` copy semantics).
                new_col = col.copy(deep=True)
            elif new_col is col:
                # ``astype(..., copy=False)`` can return the same column
                # when the requested dtype already matches; ensure a copy.
                new_col = col.copy(deep=True)
            cols.append(new_col)
            converted.append(was_converted)

        if dtype_backend == "pyarrow":
            # Convert columns that the convert-* flags selected into the
            # equivalent ``pd.ArrowDtype``; leave others untouched (matches
            # pandas' ``test_pyarrow_backend_no_conversion`` semantics).
            # Datetime / timedelta columns aren't matched by any convert-*
            # flag, but pandas converts them when *any* flag is enabled.
            any_convert = (
                convert_floating
                or convert_integer
                or convert_boolean
                or convert_string
            )
            arrow_cols = []
            for new_col, was_converted in zip(cols, converted, strict=True):
                if not was_converted:
                    dt = new_col.dtype
                    is_datetimelike = (
                        isinstance(dt, np.dtype) and dt.kind in ("M", "m")
                    ) or isinstance(dt, pd.DatetimeTZDtype)
                    if not (any_convert and is_datetimelike):
                        arrow_cols.append(new_col)
                        continue
                # Empty object columns stay as object (matches pandas'
                # ``test_pyarrow_dtype_empty_object``).
                if len(new_col) == 0 and is_dtype_obj_string(new_col.dtype):
                    arrow_cols.append(new_col)
                    continue
                if len(new_col) != 0 and new_col.null_count == len(new_col):
                    arrow_cols.append(
                        as_column(new_col, dtype=pd.ArrowDtype(pa.null()))
                    )
                elif isinstance(new_col.dtype, pd.ArrowDtype):
                    arrow_cols.append(new_col)
                else:
                    arrow_dtype = pd.ArrowDtype(
                        cudf_dtype_to_pa_type(new_col.dtype)
                    )
                    arrow_cols.append(
                        ColumnBase.create(new_col.plc_column, arrow_dtype)
                    )
            cols = arrow_cols

        return self._from_data_like_self(
            self._data._from_columns_like_self(cols, verify=False)
        )

    @_performance_tracking
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq=None,
        **kwargs,
    ):
        """
        Calculates the percent change between sequential elements.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : None
            Must be None.
        freq : str, optional
            Increment to use from time series API.
            Not yet implemented.
        **kwargs
            Additional keyword arguments are passed into shift.

        Returns
        -------
        Same type as caller.
        """
        if freq is not None:
            raise NotImplementedError("freq parameter not supported yet.")
        if fill_method is not None:
            raise ValueError(f"fill_method must be None; got {fill_method=}.")

        return self.diff(periods=periods) / self.shift(  # type: ignore[attr-defined]
            periods=periods, freq=freq, **kwargs
        )

    @_performance_tracking
    def serialize(self):
        header, frames = super().serialize()

        header["index"], index_frames = self.index.device_serialize()
        header["index_frame_count"] = len(index_frames)
        # For backwards compatibility with older versions of cuDF, index
        # columns are placed before data columns.
        frames = index_frames + frames

        return header, frames

    @classmethod
    @_performance_tracking
    def deserialize(cls, header, frames):
        index_nframes = header["index_frame_count"]
        obj = super().deserialize(
            header, frames[header["index_frame_count"] :]
        )

        index = cls.device_deserialize(header["index"], frames[:index_nframes])
        obj.index = index

        return obj

    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return [
            type(self),
            str(dict(self._dtypes)),
            *[
                normalize_token(col.dtype.categories)
                for col in self._columns
                if col.dtype == "category"
            ],
            normalize_token(self.index),
            normalize_token(self.hash_values().to_numpy()),
        ]


def _check_duplicate_level_names(specified, level_names):
    """Raise if any of `specified` has duplicates in `level_names`."""
    if specified is None:
        return
    if len(set(level_names)) == len(level_names):
        return
    duplicates = {key for key, val in Counter(level_names).items() if val > 1}

    duplicates_specified = [spec for spec in specified if spec in duplicates]
    if not len(duplicates_specified) == 0:
        # Note: pandas raises first encountered duplicates, cuDF raises all.
        raise ValueError(
            f"The names {duplicates_specified} occurs multiple times, use a"
            " level number"
        )


@_performance_tracking
def _get_replacement_values_for_columns(
    to_replace: Any, value: Any, columns_dtype_map: dict[Any, DtypeObj]
) -> tuple[
    dict[Any, bool], dict[Any, ColumnBase | list], dict[Any, ColumnBase | list]
]:
    """
    Returns a per column mapping for the values to be replaced, new
    values to be replaced with and if all the values are empty.

    Parameters
    ----------
    to_replace : numeric, str, list-like or dict
        Contains the values to be replaced.
    value : numeric, str, list-like, or dict
        Contains the values to replace `to_replace` with.
    columns_dtype_map : dict
        A column to dtype mapping representing dtype of columns.

    Returns
    -------
    all_na_columns : dict
        A dict mapping of all columns if they contain all na values
    to_replace_columns : dict
        A dict mapping of all columns and the existing values that
        have to be replaced.
    values_columns : dict
        A dict mapping of all columns and the corresponding values
        to be replaced with.
    """
    to_replace_columns: dict[Any, ColumnBase | list] = {}
    values_columns: dict[Any, ColumnBase | list] = {}
    all_na_columns: dict[Any, bool] = {}

    if is_scalar(to_replace) and is_scalar(value):
        to_replace_columns = {col: [to_replace] for col in columns_dtype_map}
        values_columns = {col: [value] for col in columns_dtype_map}
    elif is_list_like(to_replace) or isinstance(
        to_replace, (ColumnBase, Index)
    ):
        if is_scalar(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {
                col: [value]
                if is_dtype_obj_numeric(dtype, include_decimal=False)
                else as_column(
                    value,
                    length=len(to_replace),
                    dtype=cudf.dtype(type(value)),
                )
                for col, dtype in columns_dtype_map.items()
            }
        elif is_list_like(value) or is_column_like(value):
            if len(to_replace) != len(value):
                raise ValueError(
                    f"Replacement lists must be "
                    f"of same length."
                    f" Expected {len(to_replace)}, got {len(value)}."
                )
            else:
                to_replace_columns = {
                    col: to_replace for col in columns_dtype_map
                }
                values_columns = {col: value for col in columns_dtype_map}
        else:
            raise TypeError(
                "value argument must be scalar, list-like or Series"
            )
    elif _is_series(to_replace):
        if value is None or value is no_default:
            to_replace_columns = {
                col: as_column(to_replace.index) for col in columns_dtype_map
            }
            values_columns = {col: to_replace for col in columns_dtype_map}
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in to_replace_columns if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value[col]
                for col in to_replace_columns
                if col in value
            }
        else:
            raise ValueError(
                "Series.replace cannot use dict-like to_replace and non-None "
                "value"
            )
    elif is_dict_like(to_replace):
        if value is None or value is no_default:
            to_replace_columns = {
                col: list(to_replace.keys()) for col in columns_dtype_map
            }
            values_columns = {
                col: list(to_replace.values()) for col in columns_dtype_map
            }
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in columns_dtype_map if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value
                for col in columns_dtype_map
                if col in to_replace
            }
        else:
            raise TypeError("value argument must be scalar, dict, or Series")
    else:
        raise TypeError(
            "Expecting 'to_replace' to be either a scalar, array-like, "
            "dict or None, got invalid type "
            f"'{type(to_replace).__name__}'"
        )

    to_replace_columns = {
        key: [value]
        if is_scalar(value)
        else (value if isinstance(value, list) else as_column(value))
        for key, value in to_replace_columns.items()
    }
    values_columns = {
        key: [value]
        if is_scalar(value)
        else (value if isinstance(value, list) else as_column(value))
        for key, value in values_columns.items()
    }

    for i in to_replace_columns:
        if i in values_columns:
            if isinstance(values_columns[i], list):
                val_col = cast("list", values_columns[i])
                all_na = any(val is None for val in val_col)
            else:
                all_na = False
            all_na_columns[i] = all_na

    return all_na_columns, to_replace_columns, values_columns


def _is_series(obj: Any) -> bool:
    """
    Checks if the `obj` is of type `cudf.Series`
    instead of checking for isinstance(obj, cudf.Series)
    to avoid circular imports.
    """
    return isinstance(obj, IndexedFrame) and obj.ndim == 1


@_performance_tracking
def _drop_rows_by_labels(
    obj: DataFrameOrSeries,
    labels: ColumnLike | Iterable | str,
    level: int | str,
    errors: str,
) -> DataFrameOrSeries:
    """Remove rows specified by `labels`.

    If `errors="raise"`, an error is raised if some items in `labels` do not
    exist in `obj.index`.

    Will raise if level(int) is greater or equal to index nlevels.
    """
    if isinstance(level, int) and level >= obj.index.nlevels:
        raise ValueError("Param level out of bounds.")

    if not isinstance(labels, (cudf.Series, cudf.Index)):
        labels = as_column(labels)

    if isinstance(obj.index, cudf.MultiIndex):
        if level is None:
            level = 0

        levels_index = obj.index.get_level_values(level)
        if errors == "raise" and not labels.isin(levels_index).all():
            raise KeyError("One or more values not found in axis")

        if isinstance(level, int):
            ilevel = level
        else:
            ilevel = obj.index.names.index(level)

        # 1. Merge Index df and data df along column axis:
        # | id | .index df | data column(s) |
        idx_nlv = obj.index.nlevels
        working_df = obj.index.to_frame(index=False)
        working_df.columns = list(range(idx_nlv))
        for i, col in enumerate(obj._data):
            working_df[idx_nlv + i] = obj._data[col]
        # 2. Set `level` as common index:
        # | level | .index df w/o level | data column(s) |
        working_df = working_df.set_index(level)

        # 3. Use "leftanti" join to drop
        # TODO: use internal API with "leftanti" and specify left and right
        # join keys to bypass logic check
        if isinstance(labels, ColumnBase):
            join_index = cudf.Index._from_column(labels, name=level)
        else:
            join_index = cudf.Index(labels, name=level)
        to_join = cudf.DataFrame(index=join_index)
        join_res = working_df.join(to_join, how="leftanti")

        # 4. Reconstruct original layout, and rename
        join_res._insert(
            ilevel, name=join_res.index.name, value=join_res.index
        )

        midx = cudf.MultiIndex.from_frame(
            join_res.iloc[:, 0:idx_nlv], names=obj.index.names
        )

        if isinstance(obj, cudf.Series):
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data,
                index=midx,
                name=obj.name,
                attrs=obj.attrs,
            )
        else:
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data,
                index=midx,
                columns=obj._data.to_pandas_index,
                attrs=obj.attrs,
            )

    else:
        orig_index_type = obj.index.dtype
        if errors == "raise" and not labels.isin(obj.index).all():
            raise KeyError("One or more values not found in axis")

        if isinstance(labels, ColumnBase):
            idx = cudf.Index._from_column(labels, name=obj.index.name)
        else:
            idx = cudf.Index(labels, name=labels.name)
        key_df = cudf.DataFrame._from_data(data={}, index=idx)
        if isinstance(obj, cudf.DataFrame):
            res = obj.join(key_df, how="leftanti")
        else:
            res = obj.to_frame(name="tmp").join(key_df, how="leftanti")["tmp"]
            res.name = obj.name
        # Join changes the index to common type,
        # but we need to preserve the type of
        # index being returned, Hence this type-cast.
        res.index = res.index.astype(orig_index_type)
        res._attrs = obj.attrs
        return res


def _is_same_dtype(lhs_dtype, rhs_dtype):
    # Utility specific to `_reindex` to check
    # for matching column dtype.
    if lhs_dtype == rhs_dtype:
        return True
    elif (
        isinstance(lhs_dtype, cudf.CategoricalDtype)
        and isinstance(rhs_dtype, cudf.CategoricalDtype)
        and lhs_dtype.categories.dtype == rhs_dtype.categories.dtype
    ):
        # OK if categories are not all the same
        return True
    elif (
        isinstance(lhs_dtype, cudf.CategoricalDtype)
        and not isinstance(rhs_dtype, cudf.CategoricalDtype)
        and lhs_dtype.categories.dtype == rhs_dtype
    ):
        return True
    elif (
        isinstance(rhs_dtype, cudf.CategoricalDtype)
        and not isinstance(lhs_dtype, cudf.CategoricalDtype)
        and rhs_dtype.categories.dtype == lhs_dtype
    ):
        return True
    elif is_dtype_obj_string(lhs_dtype) and is_dtype_obj_string(rhs_dtype):
        return True
    elif getattr(lhs_dtype, "kind", None) in {"i", "u", "f"} and getattr(
        rhs_dtype, "kind", None
    ) in {"i", "u", "f"}:
        # Numeric index labels are compatible across int/float for
        # reindexing; the join matches labels by equality (e.g. 4 == 4.0)
        # instead of bailing to an all-null result.
        return True
    else:
        return False


def _append_new_row(col: ColumnBase, value: ScalarLike) -> ColumnBase:
    """Return a new column with a scalar `value` appended to the end of `col`.
    Cast to common type if possible
    """
    val_col = as_column(
        value,
        dtype=col.dtype
        if (cudf.utils.utils.is_na_like(value) or value in {None, np.nan})
        else None,
    )
    if val_col.dtype.kind != "f" and val_col.can_cast_safely(col.dtype):
        # If the value can be cast to the column dtype, do so
        val_col = val_col.astype(col.dtype)
        to_type = col.dtype
    else:
        to_type = None
        if is_pandas_nullable_extension_dtype(col.dtype):
            if val_col.dtype.kind == "f":
                # If the column is a pandas nullable extension type, we need to
                # convert the nans to a nullable type as well.
                val_col = val_col.nans_to_nulls()
                if len(val_col) == val_col.null_count:
                    # If the column is all nulls, we can use the column dtype
                    # to avoid unnecessary casting.
                    val_col = val_col.astype(col.dtype)
            if val_col.can_cast_safely(col.dtype):
                to_type = col.dtype
        if to_type is None:
            to_type = find_common_type([val_col.dtype, col.dtype])

        if (
            cudf.get_option("mode.pandas_compatible")
            and is_string_dtype(to_type)
            and is_mixed_with_object_dtype(val_col, col)
        ):
            raise MixedTypeError("Cannot append mixed types")
        if is_pandas_nullable_extension_dtype(
            col.dtype
        ) and val_col.can_cast_safely(col.dtype):
            to_type = col.dtype
    val_col = val_col.astype(to_type)
    old_col = col.astype(to_type)
    res_col = ColumnBase.create(
        concat_columns([old_col, val_col]).plc_column, to_type
    )
    if (
        cudf.get_option("mode.pandas_compatible")
        and res_col.dtype != col.dtype
        and isinstance(col, CategoricalColumn)
    ):
        raise MixedTypeError(
            "Cannot append mixed types: "
            f"Column dtype {col.dtype} is not compatible with {res_col.dtype}"
        )
    return res_col
