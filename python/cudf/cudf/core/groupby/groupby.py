# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import functools
import itertools
import textwrap
import warnings
from collections.abc import Mapping
from functools import cached_property, singledispatch
from typing import TYPE_CHECKING, Any, Literal, cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

from cudf.api.types import is_list_like, is_scalar
from cudf.core._internals import aggregation, sorting
from cudf.core.abc import Serializable
from cudf.core.column import access_columns
from cudf.core.column.column import (
    ColumnBase,
    as_column,
    column_empty,
    deserialize_columns,
    serialize_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.common import pipe
from cudf.core.copy_types import GatherMap
from cudf.core.dtype.validators import (
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.dtypes import (
    CategoricalDtype,
    DecimalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.index import Index, RangeIndex, _index_from_data
from cudf.core.join._join_helpers import _match_join_keys
from cudf.core.mixins import GetAttrGetItemMixin, Reducible, Scannable
from cudf.core.multiindex import MultiIndex
from cudf.core.reshape import concat
from cudf.core.udf.groupby_utils import _can_be_jitted, jit_groupby_apply
from cudf.options import get_option
from cudf.utils.dtypes import (
    SIZE_TYPE_DTYPE,
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    get_dtype_of_same_kind,
    is_pandas_nullable_extension_dtype,
    is_pandas_nullable_numpy_dtype,
)
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.scalar import pa_scalar_to_plc_scalar

if TYPE_CHECKING:
    from collections.abc import Generator, Hashable, Iterable, Sequence

    from cudf._typing import (
        AggType,
        DataFrameOrSeries,
        DtypeObj,
        MultiColumnAggType,
        ScalarLike,
    )
    from cudf.core.dataframe import DataFrame
    from cudf.core.series import Series

# The sets below define the possible aggregations that can be performed on
# different dtypes. These strings must be elements of the AggregationKind enum.
# The libcudf infrastructure exists for "COLLECT" support on
# categoricals, but the dtype support in python does not.
# Reductions whose result for a group becomes null when that group contains
# any null value and ``skipna=False`` (libcudf otherwise always drops nulls).
_NULL_PROPAGATING_REDUCTIONS = {
    "sum",
    "prod",
    "product",
    "mean",
    "median",
    "var",
    "std",
    "min",
    "max",
}

_CATEGORICAL_AGGS = {"COUNT", "NUNIQUE", "SIZE", "UNIQUE"}
_STRING_AGGS = {
    "COLLECT",
    "COUNT",
    "MAX",
    "MIN",
    "NTH",
    "NUNIQUE",
    "SIZE",
    "UNIQUE",
}
_LIST_AGGS = {"COLLECT"}
_STRUCT_AGGS = {"COLLECT", "CORRELATION", "COVARIANCE"}
_INTERVAL_AGGS = {"COLLECT"}
_DECIMAL_AGGS = {
    "ARGMIN",
    "ARGMAX",
    "COLLECT",
    "COUNT",
    "MAX",
    "MIN",
    "NTH",
    "NUNIQUE",
    "SUM",
}


@singledispatch
def get_valid_aggregation(dtype):
    return "ALL"


@get_valid_aggregation.register
def _(dtype: pd.StringDtype):
    return _STRING_AGGS


@get_valid_aggregation.register
def _(dtype: ListDtype):
    return _LIST_AGGS


@get_valid_aggregation.register
def _(dtype: CategoricalDtype):
    return _CATEGORICAL_AGGS


@get_valid_aggregation.register
def _(dtype: ListDtype):
    return _LIST_AGGS


@get_valid_aggregation.register
def _(dtype: StructDtype):
    return _STRUCT_AGGS


@get_valid_aggregation.register
def _(dtype: IntervalDtype):
    return _INTERVAL_AGGS


@get_valid_aggregation.register
def _(dtype: DecimalDtype):
    return _DECIMAL_AGGS


@singledispatch
def _is_unsupported_agg_for_type(dtype, str_agg: str) -> bool:
    return False


@_is_unsupported_agg_for_type.register
def _(dtype: pd.StringDtype, str_agg: str) -> bool:
    cumulative_agg = str_agg in {"cumsum", "cummin", "cummax"}
    basic_agg = any(
        a in str_agg
        for a in (
            "count",
            "max",
            "min",
            "first",
            "last",
            "nunique",
            "unique",
            "nth",
        )
    )
    return str_agg not in _STRING_AGGS and (
        cumulative_agg or not (basic_agg or str_agg == "<class 'list'>")
    )


@_is_unsupported_agg_for_type.register
def _(dtype: CategoricalDtype, str_agg: str) -> bool:
    cumulative_agg = str_agg in {"cumsum", "cummin", "cummax"}
    not_basic_agg = not any(
        a in str_agg for a in ("count", "max", "min", "unique")
    )
    return str_agg not in _CATEGORICAL_AGGS and (
        cumulative_agg or not_basic_agg
    )


def _is_all_scan_aggregate(all_aggs: list[list[str]]) -> bool:
    """
    Returns True if all are scan aggregations.

    Raises
    ------
    NotImplementedError
        If both reduction aggregations and scan aggregations are present.
    """
    groupby_scans = {
        "cumcount",
        "cumsum",
        "cummin",
        "cummax",
        "cumprod",
        "rank",
    }

    def get_name(agg):
        if not callable(agg):
            return agg
        if agg is not list:
            # A ``lambda x: x.cumsum()``-style aggregation carries its
            # scan-ness only in the aggregation name it resolves to
            # (``Aggregation.cumsum`` is an alias of ``sum``; libcudf
            # separates scan from reduction by the *call*, not the
            # aggregation object). Probe the callable with a
            # name-recording stand-in mirroring ``make_aggregation``'s
            # ``op(Aggregation)`` protocol; true UDFs raise inside the
            # probe and fall back to ``__name__``.
            class _NameProbe:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: name

            try:
                name = agg(_NameProbe())
            except Exception:
                return agg.__name__
            if isinstance(name, str):
                return name
        return agg.__name__

    all_scan = all(
        get_name(agg_name) in groupby_scans
        for aggs in all_aggs
        for agg_name in aggs
    )
    any_scan = any(
        get_name(agg_name) in groupby_scans
        for aggs in all_aggs
        for agg_name in aggs
    )

    if not all_scan and any_scan:
        raise NotImplementedError(
            "Cannot perform both aggregation and scan in one operation"
        )
    return all_scan and any_scan


# The three functions below return the quantiles [25%, 50%, 75%]
# respectively, which are called in the describe() method to output
# the summary stats of a GroupBy object
def _quantile_25(x):
    return x.quantile(0.25)


def _quantile_50(x):
    return x.quantile(0.50)


def _quantile_75(x):
    return x.quantile(0.75)


def _is_row_of(chunk, obj):
    from cudf.core.dataframe import DataFrame
    from cudf.core.series import Series

    return (
        isinstance(chunk, Series)
        and isinstance(obj, DataFrame)
        and len(chunk) == obj._num_columns
        and (chunk.index.to_pandas() == pd.Index(obj._column_names)).all()
    )


NamedAgg = pd.NamedAgg


NamedAgg.__doc__ = """
Helper for column specific aggregation with control over output column names.

Subclass of typing.NamedTuple.

Parameters
----------
column : Hashable
    Column label in the DataFrame to apply aggfunc.
aggfunc : function or str
    Function to apply to the provided column.

Examples
--------
>>> df = cudf.DataFrame({"key": [1, 1, 2], "a": [-1, 0, 1], 1: [10, 11, 12]})
>>> agg_a = cudf.NamedAgg(column="a", aggfunc="min")
>>> agg_1 = cudf.NamedAgg(column=1, aggfunc=lambda x: x.mean())
>>> df.groupby("key", sort=True).agg(result_a=agg_a, result_1=agg_1)
        result_a  result_1
key
1          -1      10.5
2           1      12.0
"""


groupby_doc_template = textwrap.dedent(
    """Group using a mapper or by a Series of columns.

A groupby operation involves some combination of splitting the object,
applying a function, and combining the results. This can be used to
group large amounts of data and compute operations on these groups.

Parameters
----------
by : mapping, function, label, or list of labels
    Used to determine the groups for the groupby. If by is a
    function, it's called on each value of the object's index.
    If a dict or Series is passed, the Series or dict VALUES will
    be used to determine the groups (the Series' values are first
    aligned; see .align() method). If an cupy array is passed, the
    values are used as-is determine the groups. A label or list
    of labels may be passed to group by the columns in self.
    Notice that a tuple is interpreted as a (single) key.
level : int, level name, or sequence of such, default None
    If the axis is a MultiIndex (hierarchical), group by a particular
    level or levels.
as_index : bool, default True
    For aggregated output, return object with group labels as
    the index. Only relevant for DataFrame input.
    as_index=False is effectively "SQL-style" grouped output.
sort : bool, default False
    Sort result by group key. Differ from Pandas, cudf defaults to
    ``False`` for better performance. Note this does not influence
    the order of observations within each group. Groupby preserves
    the order of rows within each group.
group_keys : bool, optional
    When calling apply and the ``by`` argument produces a like-indexed
    result, add group keys to index to identify pieces. By default group
    keys are not included when the result's index (and column) labels match
    the inputs, and are included otherwise. This argument has no effect if
    the result produced is not like-indexed with respect to the input.
{ret}
Examples
--------
**Series**

>>> ser = cudf.Series([390., 350., 30., 20.],
...                 index=['Falcon', 'Falcon', 'Parrot', 'Parrot'],
...                 name="Max Speed")
>>> ser
Falcon    390.0
Falcon    350.0
Parrot     30.0
Parrot     20.0
Name: Max Speed, dtype: float64
>>> ser.groupby(level=0, sort=True).mean()
Falcon    370.0
Parrot     25.0
Name: Max Speed, dtype: float64
>>> ser.groupby(ser > 100, sort=True).mean()
Max Speed
False     25.0
True     370.0
Name: Max Speed, dtype: float64

**DataFrame**

>>> import cudf
>>> import pandas as pd
>>> df = cudf.DataFrame({{
...     'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
...     'Max Speed': [380., 370., 24., 26.],
... }})
>>> df
   Animal  Max Speed
0  Falcon      380.0
1  Falcon      370.0
2  Parrot       24.0
3  Parrot       26.0
>>> df.groupby(['Animal'], sort=True).mean()
        Max Speed
Animal
Falcon      375.0
Parrot       25.0

>>> arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
...           ['Captive', 'Wild', 'Captive', 'Wild']]
>>> index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
>>> df = cudf.DataFrame({{'Max Speed': [390., 350., 30., 20.]}},
...     index=index)
>>> df
                Max Speed
Animal Type
Falcon Captive      390.0
        Wild         350.0
Parrot Captive       30.0
        Wild          20.0
>>> df.groupby(level=0, sort=True).mean()
        Max Speed
Animal
Falcon      370.0
Parrot       25.0
>>> df.groupby(level="Type", sort=True).mean()
        Max Speed
Type
Captive      210.0
Wild         185.0

>>> df = cudf.DataFrame({{'A': 'a a b'.split(),
...                      'B': [1,2,3],
...                      'C': [4,6,5]}})
>>> g1 = df.groupby('A', group_keys=False, sort=True)
>>> g2 = df.groupby('A', group_keys=True, sort=True)

Notice that ``g1`` have ``g2`` have two groups, ``a`` and ``b``, and only
differ in their ``group_keys`` argument. Calling `apply` in various ways,
we can get different grouping results:

>>> g1[['B', 'C']].apply(lambda x: x / x.sum())
          B    C
0  0.333333  0.4
1  0.666667  0.6
2  1.000000  1.0

In the above, the groups are not part of the index. We can have them included
by using ``g2`` where ``group_keys=True``:

>>> g2[['B', 'C']].apply(lambda x: x / x.sum())
            B    C
A
a 0  0.333333  0.4
  1  0.666667  0.6
b 2  1.000000  1.0
"""
)


class _GroupByContextManager:
    """Context manager for safe access to pylibcudf GroupBy object.

    This context manager creates and holds the pylibcudf GroupBy object,
    entering access contexts for key columns on each entry and exiting
    them on exit. The same instance can be safely entered multiple times,
    including nested entries.
    """

    __slots__ = (
        "_grouping",
        "_plc_groupby",
        "_stack_list",
    )

    def __init__(self, grouping, dropna):
        self._grouping = grouping
        self._stack_list = []

        # Create pylibcudf GroupBy eagerly
        with access_columns(
            *grouping._key_columns, mode="read", scope="internal"
        ) as key_columns:
            self._plc_groupby = plc.groupby.GroupBy(
                plc.Table([col.plc_column for col in key_columns]),
                plc.types.NullPolicy.EXCLUDE
                if dropna
                else plc.types.NullPolicy.INCLUDE,
            )

    def __enter__(self):
        stack = access_columns(
            *self._grouping._key_columns, mode="read", scope="internal"
        )
        stack.__enter__()
        self._stack_list.append(stack)

        # Return the private pylibcudf GroupBy object
        return self._plc_groupby

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stack_list:
            stack = self._stack_list.pop()
            return stack.__exit__(exc_type, exc_val, exc_tb)
        return False


def _collect_series_key_column_names(obj, by) -> dict[int, Hashable]:
    """For each Series grouping key in ``by``, map ``id`` of the Series'
    underlying column to the name of the matching column in ``obj`` (when
    one exists by object identity). Mirrors pandas' behavior of excluding
    such columns from aggregation values.

    Only applies when ``obj`` is a DataFrame: for Series inputs, the single
    column *is* the value column, so identity-based exclusion would empty
    the aggregation result. Keying by ``id(series._column)`` makes the
    match robust to ordering, the presence of non-Series keys, and to
    repeated Series keys.
    """
    import cudf

    result: dict[int, Hashable] = {}
    if not isinstance(obj, cudf.DataFrame):
        return result
    by_list = by if isinstance(by, list) else [by]
    for key in by_list:
        if isinstance(key, cudf.Series):
            for col_name, col in obj._column_labels_and_values:
                if col is key._column:
                    result[id(key._column)] = col_name
                    break
    return result


class GroupByNthSelector:
    """Mirror of :class:`pandas.core.groupby.indexing.GroupByNthSelector`.

    ``GroupBy.nth`` supports both the call form ``gb.nth(n, dropna=...)``
    and the index form ``gb.nth[n]``.
    """

    def __init__(self, groupby_object: GroupBy) -> None:
        self.groupby_object = groupby_object

    def __call__(
        self, n, dropna: Literal["any", "all", None] = None
    ) -> Series | DataFrame:
        return self.groupby_object._nth(n, dropna)

    def __getitem__(self, n) -> Series | DataFrame:
        return self.groupby_object._nth(n)


class GroupBy(Serializable, Reducible, Scannable):
    obj: Series | DataFrame

    _VALID_REDUCTIONS = {
        "sum",
        "prod",
        "idxmin",
        "idxmax",
        "min",
        "max",
        "mean",
        "median",
        "nunique",
        "first",
        "last",
        "var",
        "std",
    }

    _VALID_SCANS = {
        "cumsum",
        "cumprod",
        "cummin",
        "cummax",
    }

    # Necessary because the function names don't directly map to the docs.
    _SCAN_DOCSTRINGS = {
        "cumsum": {"op_name": "Cumulative sum"},
        "cumprod": {"op_name": "Cumulative product"},
        "cummin": {"op_name": "Cumulative min"},
        "cummax": {"op_name": "Cumulative max"},
    }

    _MAX_GROUPS_BEFORE_WARN = 100

    def __init__(
        self,
        obj,
        by=None,
        level=None,
        sort=False,
        as_index=True,
        dropna=True,
        group_keys=True,
    ):
        """
        Group a DataFrame or Series by a set of columns.

        Parameters
        ----------
        by : optional
            Specifies the grouping columns. Can be any of the following:
            - A Python function called on each value of the object's index
            - A dict or Series that maps index labels to group names
            - A Index object
            - A str indicating a column name
            - An array of the same length as the object
            - A Grouper object
            - A list of the above
        level : int, level_name or list, optional
            For objects with a MultiIndex, `level` can be used to specify
            grouping by one or more levels of the MultiIndex.
        sort : bool, default False
            Sort the result by group keys. Differ from Pandas, cudf defaults
            to False for better performance.
        as_index : bool, optional
            If as_index=True (default), the group names appear
            as the keys of the resulting DataFrame.
            If as_index=False, the groups are returned as ordinary
            columns of the resulting DataFrame, *if they are named columns*.
        dropna : bool, optional
            If True (default), do not include the "null" group.
        """
        # Determine which column names in `obj` correspond to the grouping
        # key Series by column identity (mirrors pandas' behavior).
        # Must be done before ``nans_to_nulls`` which breaks identity.
        by_series_col_names = _collect_series_key_column_names(obj, by)

        # Row-filter operations (``nth``) must return the original values,
        # preserving the NaN-vs-null distinction that ``nans_to_nulls``
        # erases below.
        self._obj_original = obj
        if get_option("mode.pandas_compatible"):
            obj = obj.nans_to_nulls()
        self.obj = obj
        self._as_index = as_index
        self._by = by.copy(deep=True) if isinstance(by, _Grouping) else by
        self._level = level
        self._sort = sort
        self._dropna = dropna
        self._group_keys = group_keys
        self._selection: tuple[Any, ...] | None = None

        if isinstance(self._by, _Grouping):
            self._by._obj = self.obj
            self.grouping = self._by
        else:
            self.grouping = _Grouping(
                obj, self._by, level, by_series_col_names, dropna=self._dropna
            )

        self._groupby_manager = _GroupByContextManager(
            self.grouping, self._dropna
        )

    @cached_property
    def _range_column_from_obj(self) -> ColumnBase:
        return ColumnBase.from_range(range(len(self.obj)))

    def __iter__(self):
        group_names, offsets, _, grouped_values = self._grouped()
        if isinstance(group_names, Index):
            group_names = group_names.to_pandas()
        if self._sort or len(offsets) <= 2:
            order: Iterable[int] = range(len(offsets) - 1)
        else:
            # libcudf returns groups sorted by key, but with ``sort=False``
            # pandas iterates groups in order of first appearance. Reorder by
            # the earliest original row position in each group (group order
            # matches between the two ``_groups`` calls since the grouping is
            # identical).
            pos_offsets, _, (positions,) = self._groups(
                [self._range_column_from_obj]
            )
            # Gather the earliest original row position of each group and sort
            # the groups by it entirely on the device; only the small ``order``
            # array (one entry per group) is copied back to the host.
            first_pos = positions.take(as_column(pos_offsets[:-1]))
            order = first_pos.argsort().to_numpy()
        for i in order:
            name = group_names[i]
            yield (
                (name,)
                if isinstance(self._by, list) and len(self._by) == 1
                else name,
                grouped_values[offsets[i] : offsets[i + 1]],
            )

    def __len__(self) -> int:
        return self.ngroups

    @property
    def ngroups(self) -> int:
        _, offsets, _, _ = self._grouped()
        return len(offsets) - 1

    @property
    def ndim(self) -> int:
        return self.obj.ndim

    @cached_property
    def groups(self):
        """
        Returns a dictionary mapping group keys to row labels.
        """
        group_names, offsets, _, grouped_values = self._grouped()
        grouped_index = grouped_values.index

        if len(group_names) > self._MAX_GROUPS_BEFORE_WARN:
            warnings.warn(
                f"GroupBy.groups() performance scales poorly with "
                f"number of groups. Got {len(group_names)} groups."
            )
        if isinstance(self._by, list) and len(self._by) == 1:
            warnings.warn(
                "In a future version, the keys of `groups` will be a "
                f"tuple with a single element, e.g. ({self._by[0]},) , "
                f"instead of a scalar, e.g. {self._by[0]}, when grouping "
                "by a list with a single element. Use ``df.groupby(by='a').groups`` "
                "instead of ``df.groupby(by=['a']).groups`` to avoid this warning",
                FutureWarning,
            )

        return dict(
            zip(
                group_names.to_pandas(),
                grouped_index._split(offsets[1:-1]),
                strict=True,
            )
        )

    @cached_property
    def indices(self) -> dict[ScalarLike, cp.ndarray]:
        """
        Dict {group name -> group indices}.

        Examples
        --------
        >>> import cudf
        >>> data = [[10, 20, 30], [10, 30, 40], [40, 50, 30]]
        >>> df = cudf.DataFrame(data, columns=["a", "b", "c"])
        >>> df
            a   b   c
        0  10  20  30
        1  10  30  40
        2  40  50  30
        >>> df.groupby(by=["a"]).indices  # doctest: +SKIP
        {10: array([0, 1]), 40: array([2])}
        """
        offsets, group_keys, (indices,) = self._groups(
            [self._range_column_from_obj]
        )

        key_dtypes = [col.dtype for col in group_keys]
        with access_columns(
            *group_keys, mode="read", scope="internal"
        ) as cols:
            plc_table = plc.stream_compaction.stable_distinct(
                plc.Table([col.plc_column for col in cols]),
                list(range(len(cols))),
                plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
            )
            group_keys = [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    plc_table.columns(), key_dtypes, strict=True
                )
            ]
        if len(group_keys) > 1:
            index = MultiIndex.from_arrays(group_keys)
        else:
            index = Index._from_column(group_keys[0])
        split = cp.split(indices.values, offsets[1:-1])
        return dict(
            zip(
                index.to_pandas(),
                split,
                strict=True,
            )
        )

    @_performance_tracking
    def get_group(self, name):
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.

        Returns
        -------
        group : same type as obj

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
        >>> df
           X  Y
        0  A  1
        1  B  4
        2  A  3
        3  B  2
        >>> df.groupby("X").get_group("A")
           X  Y
        0  A  1
        2  A  3
        """
        if is_list_like(self._by) and len(self._by) == 1:
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            else:
                raise KeyError(name)
        return self.obj.iloc[self.indices[name]]

    @_performance_tracking
    def size(self) -> Series:
        """
        Return the size of each group.
        """
        from cudf.core.series import Series

        col = column_empty(len(self.obj), np.dtype(np.int8))
        result = (
            Series._from_column(col, name=getattr(self.obj, "name", None))
            .groupby(self.grouping, sort=self._sort, dropna=self._dropna)
            .agg("size")
        )
        obj_dtype = getattr(self.obj, "dtype", None)
        if isinstance(obj_dtype, pd.ArrowDtype):
            # TODO: Remove once groupby.agg preserves pandas extension dtypes.
            arrow_dtype = pd.ArrowDtype(pa.int64())
            if isinstance(result, Series):
                result._column = ColumnBase.create(
                    result._column.plc_column, arrow_dtype
                )
            elif "size" in result._column_names:
                result._data["size"] = ColumnBase.create(
                    result._data["size"].plc_column, arrow_dtype
                )
        elif (
            isinstance(obj_dtype, pd.StringDtype)
            and obj_dtype.storage == "pyarrow"
            and obj_dtype.na_value is pd.NA
        ) or (
            self.obj.ndim == 1
            and not isinstance(obj_dtype, pd.StringDtype)
            and is_pandas_nullable_extension_dtype(obj_dtype)
        ):
            # Series.groupby.size() returns Int64 for ``string[pyarrow]``
            # and for masked (Int*/UInt*/Float*/boolean) dtypes
            # (pandas GH#54132).
            int64_dtype = pd.Int64Dtype()
            if isinstance(result, Series):
                result = Series._from_column(
                    ColumnBase.create(result._column.plc_column, int64_dtype),
                    name=result.name,
                    index=result.index,
                )
            elif "size" in result._column_names:
                result._data["size"] = ColumnBase.create(
                    result._data["size"].plc_column, int64_dtype
                )
        if not self._as_index:
            result = result.rename("size").reset_index()
        return result

    @_performance_tracking
    def cumcount(self, ascending: bool = True):
        """
        Return the cumulative count of keys in each group.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.
            Currently not supported
        """
        if ascending is not True:
            raise NotImplementedError(
                "ascending is currently not implemented."
            )
        from cudf.core.series import Series

        return (
            Series._from_column(
                column_empty(len(self.obj), np.dtype(np.int8)),
                index=self.obj.index,
            )
            .groupby(self.grouping, sort=self._sort, dropna=self._dropna)
            .agg("cumcount")
        )

    @_performance_tracking
    def rank(
        self,
        method="average",
        ascending=True,
        na_option="keep",
        pct=False,
        axis=0,
    ):
        """
        Return the rank of values within each group.
        """
        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")

        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError(
                f"na_option must be one of 'keep', 'top', or 'bottom', "
                f"but got {na_option}"
            )

        # TODO: in pandas compatibility mode, we should convert any
        # NaNs to nulls in any float value columns, as Pandas
        # treats NaNs the way we treat nulls.
        if get_option("mode.pandas_compatible"):
            if any(
                col.dtype.kind == "f" for col in self.grouping.values._columns
            ):
                raise NotImplementedError(
                    "NaNs are not supported in groupby.rank."
                )

        def rank(x):
            return getattr(x, "rank")(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )

        result = self.agg(rank)

        # pandas always returns floats, staying within the value column's
        # dtype family: numpy -> float64, masked (Int64/Float32/...) ->
        # Float64, arrow -> double[pyarrow]
        target = np.dtype(np.float64)
        if result.ndim == 1:
            source_dtype = (
                self.obj.dtype if self.obj.ndim == 1 else result.dtype
            )
            return result.astype(get_dtype_of_same_kind(source_dtype, target))
        return result.astype(
            {
                label: get_dtype_of_same_kind(
                    self.obj._data[label].dtype
                    if self.obj.ndim == 2 and label in self.obj._data
                    else result_dtype,
                    target,
                )
                for label, result_dtype in result._dtypes
            }
        )

    @property
    def _groupby(self):
        """Returns the cached context manager for safe access to the pylibcudf GroupBy."""
        return self._groupby_manager

    def _groups(
        self, values: Iterable[ColumnBase]
    ) -> tuple[list[int], list[ColumnBase], list[ColumnBase]]:
        # Materialize iterator to avoid consuming it during access context setup
        values_list = list(values)
        key_dtypes = [col.dtype for col in self.grouping._key_columns]
        value_dtypes = [col.dtype for col in values_list]
        with access_columns(*values_list, mode="read", scope="internal"):
            plc_columns = [col.plc_column for col in values_list]
            if not plc_columns:
                plc_table = None
            else:
                plc_table = plc.Table(plc_columns)

            with self._groupby as plc_groupby:
                offsets, grouped_keys, grouped_values = plc_groupby.get_groups(
                    plc_table
                )

        return (
            offsets,
            [
                ColumnBase.create(col, dtype)
                for col, dtype in zip(
                    grouped_keys.columns(), key_dtypes, strict=True
                )
            ],
            (
                [
                    ColumnBase.create(col, dtype)
                    for col, dtype in zip(
                        grouped_values.columns(), value_dtypes, strict=True
                    )
                ]
                if grouped_values is not None
                else []
            ),
        )

    def _aggregate(
        self, values: tuple[ColumnBase, ...], aggregations
    ) -> tuple[
        list[list[plc.Column]],
        list[ColumnBase],
        list[list[tuple[str, str]]],
    ]:
        included_aggregations = []
        column_included = []
        requests = []
        result_columns: list[list[plc.Column]] = []

        for i, (col, aggs) in enumerate(
            zip(values, aggregations, strict=True)
        ):
            valid_aggregations = get_valid_aggregation(col.dtype)
            included_aggregations_i = []
            col_aggregations = []
            for agg in aggs:
                str_agg = str(agg)
                if _is_unsupported_agg_for_type(col.dtype, str_agg):
                    raise TypeError(
                        f"{col.dtype} type does not support {agg} operations"
                    )
                agg_obj = aggregation.make_aggregation(agg)
                if (
                    valid_aggregations == "ALL"
                    or agg_obj.kind in valid_aggregations
                ):
                    included_aggregations_i.append((agg, agg_obj.kind))
                    col_aggregations.append(agg_obj.plc_obj)
            included_aggregations.append(included_aggregations_i)
            result_columns.append([])
            if col_aggregations:
                requests.append(
                    plc.groupby.GroupByRequest(
                        col.plc_column, col_aggregations
                    )
                )
                column_included.append(i)

        if not requests and any(len(v) > 0 for v in aggregations):
            raise pd.errors.DataError(
                "All requested aggregations are unsupported."
            )

        key_dtypes = [col.dtype for col in self.grouping._key_columns]
        with access_columns(*values, mode="read", scope="internal"):
            with self._groupby as plc_groupby:
                keys, results = (
                    plc_groupby.scan(requests)
                    if _is_all_scan_aggregate(aggregations)
                    else plc_groupby.aggregate(requests)
                )

        for i, result in zip(column_included, results, strict=True):
            result_columns[i] = result.columns()

        return (
            result_columns,
            [
                ColumnBase.create(key, dtype)
                for key, dtype in zip(keys.columns(), key_dtypes, strict=True)
            ],
            included_aggregations,
        )

    def _shift(
        self, values: tuple[ColumnBase, ...], periods: int, fill_values: list
    ) -> Generator[ColumnBase]:
        with access_columns(*values, mode="read", scope="internal"):
            with self._groupby as plc_groupby:
                _, shifts = plc_groupby.shift(
                    plc.table.Table([col.plc_column for col in values]),
                    [periods] * len(values),
                    [
                        pa_scalar_to_plc_scalar(
                            pa.scalar(
                                val, type=cudf_dtype_to_pa_type(col.dtype)
                            )
                        )
                        for val, col in zip(fill_values, values, strict=True)
                    ],
                )
                return (
                    ColumnBase.create(col, orig.dtype)
                    for col, orig in zip(shifts.columns(), values, strict=True)
                )

    def _replace_nulls(
        self, values: tuple[ColumnBase, ...], method: plc.replace.ReplacePolicy
    ) -> Generator[ColumnBase]:
        with access_columns(*values, mode="read", scope="internal"):
            with self._groupby as plc_groupby:
                _, replaced = plc_groupby.replace_nulls(
                    plc.Table([col.plc_column for col in values]),
                    [method] * len(values),
                )

                return (
                    ColumnBase.create(col, orig.dtype)
                    for col, orig in zip(
                        replaced.columns(), values, strict=True
                    )
                )

    @_performance_tracking
    def agg(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        """
        Apply aggregation(s) to the groups.

        Parameters
        ----------
        func : str, callable, list or dict
            Argument specifying the aggregation(s) to perform on the
            groups. `func` can be any of the following:

              - string: the name of a supported aggregation
              - callable: a function that accepts a Series/DataFrame and
                performs a supported operation on it.
              - list: a list of strings/callables specifying the
                aggregations to perform on every column.
              - dict: a mapping of column names to string/callable
                specifying the aggregations to perform on those
                columns.

        See :ref:`the user guide <basics.groupby>` for supported
        aggregations.

        Returns
        -------
        A Series or DataFrame containing the combined results of the
        aggregation(s).

        Examples
        --------
        >>> import cudf
        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> a.groupby('a', sort=True).agg('sum')
           b  c
        a
        1  3  4
        2  3  1

        Specifying a list of aggregations to perform on each column.

        >>> import cudf
        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> a.groupby('a', sort=True).agg(['sum', 'min'])
            b       c
          sum min sum min
        a
        1   3   1   4   2
        2   3   3   1   1

        Using a dict to specify aggregations to perform per column.

        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> a.groupby('a', sort=True).agg({'a': 'max', 'b': ['min', 'mean']})
            a   b
          max min mean
        a
        1   1   1  1.5
        2   2   3  3.0

        Using lambdas/callables to specify aggregations taking parameters.

        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> f1 = lambda x: x.quantile(0.5); f1.__name__ = "q0.5"
        >>> f2 = lambda x: x.quantile(0.75); f2.__name__ = "q0.75"
        >>> a.groupby('a').agg([f1, f2])
             b          c
          q0.5 q0.75 q0.5 q0.75
        a
        1  1.5  1.75  2.0   2.0
        2  3.0  3.00  1.0   1.0
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if args:
            raise NotImplementedError(
                "Passing args to func is currently not supported."
            )
        from cudf.core.dataframe import DataFrame

        column_names, columns, normalized_aggs = self._normalize_aggs(
            func, **kwargs
        )
        orig_dtypes = tuple(c.dtype for c in columns)

        # Note: When there are no key columns, the below produces
        # an Index with float64 dtype, while Pandas returns
        # an Index with int64 dtype.
        # (GH: 6945)
        (
            result_columns,
            grouped_key_cols,
            included_aggregations,
        ) = self._aggregate(columns, normalized_aggs)

        result_index = self.grouping.keys._from_columns_like_self(
            grouped_key_cols,
        )

        multilevel = _is_multi_agg(func)
        data = {}
        for col_name, aggs, cols, orig_dtype in zip(
            column_names,
            included_aggregations,
            result_columns,
            orig_dtypes,
            strict=True,
        ):
            for agg_tuple, plc_result in zip(aggs, cols, strict=True):
                agg, agg_kind = agg_tuple
                agg_name = agg.__name__ if callable(agg) else agg
                if multilevel:
                    key = (col_name, agg_name)
                else:
                    key = col_name

                create_dtype = dtype_from_pylibcudf_column(plc_result)
                cast_dtype = None
                if agg in {list, "collect"}:
                    # Collect wraps the original dtype in ListDtype (e.g., int -> list<int>)
                    create_dtype = get_dtype_of_same_kind(
                        orig_dtype, ListDtype(orig_dtype)
                    )
                if agg_kind in {"ARGMIN", "ARGMAX"} and not isinstance(
                    self.obj.index, MultiIndex
                ):
                    # libcudf returns the integer row-position of the
                    # min/max element within each group (null if the
                    # group's values are all NA); pandas returns the
                    # *label* of that row from the source index and raises
                    # for all-NA groups. Gather from the raw position
                    # column before any dtype morphing (a masked gather
                    # map cannot feed ``take``). MultiIndex sources fall
                    # through and stay positional: pandas maps them to
                    # tuple labels in an object column, which is not
                    # currently supported.
                    pos_col = ColumnBase.create(plc_result, create_dtype)
                    if pos_col.has_nulls():
                        how = "idxmin" if agg_kind == "ARGMIN" else "idxmax"
                        raise ValueError(
                            f"{how} with skipna=True encountered all NA "
                            "values in a group."
                        )
                    data[key] = self.obj.index._column.take(pos_col)
                    continue
                # Override for specific aggregation types that need dtype adjustments
                if agg_kind in {"COUNT", "SIZE", "ARGMIN", "ARGMAX"}:
                    if isinstance(orig_dtype, pd.StringDtype):
                        cast_dtype = np.dtype(np.int64)
                    else:
                        cast_dtype = get_dtype_of_same_kind(
                            orig_dtype, np.dtype(np.int64)
                        )
                elif agg_kind == "NUNIQUE":
                    cast_dtype = np.dtype(np.int64)
                elif (
                    agg_name in {"cumsum", "cumprod"}
                    and is_pandas_nullable_extension_dtype(orig_dtype)
                    and orig_dtype.kind in {"i", "u"}
                ):
                    # libcudf's SUM/PRODUCT scans promote narrow integers
                    # to 64-bit. pandas does the same for numpy dtypes
                    # (int8 -> int64, GH#37493) but preserves masked
                    # extension dtypes (Int16 stays Int16, GH#58811),
                    # wrapping on overflow.
                    cast_dtype = orig_dtype
                elif (
                    (
                        isinstance(agg_name, str)
                        and agg_name in Reducible._SUPPORTED_REDUCTIONS
                    )
                    and plc_result.size() == 0
                    and not isinstance(
                        create_dtype,
                        (ListDtype, StructDtype, DecimalDtype),
                    )
                ):
                    cast_dtype = orig_dtype
                elif agg not in {list, "collect"}:
                    if (
                        isinstance(orig_dtype, np.dtype)
                        and orig_dtype.kind == "O"
                        and is_dtype_obj_string(create_dtype)
                    ):
                        # a string-producing aggregation (first/last/min/
                        # max/nth) on an object-dtype column stays object,
                        # matching pandas. Scoped here rather than in
                        # get_dtype_of_same_kind: other callers (e.g. merge
                        # key coalescing) re-infer str for object inputs.
                        create_dtype = orig_dtype
                    elif (
                        isinstance(orig_dtype, pd.DatetimeTZDtype)
                        and isinstance(create_dtype, np.dtype)
                        and create_dtype.kind == "M"
                    ):
                        # libcudf has no timezone notion: a DatetimeTZColumn
                        # feeds its stored UTC instants to libcudf and the
                        # result comes back as a tz-naive timestamp column.
                        # Reattach the original tz (the values are unchanged
                        # UTC instants, so this is lossless).
                        create_dtype = pd.DatetimeTZDtype(
                            np.datetime_data(create_dtype)[0], orig_dtype.tz
                        )
                    else:
                        create_dtype = get_dtype_of_same_kind(
                            orig_dtype, create_dtype
                        )

                result_col = ColumnBase.create(plc_result, create_dtype)
                if agg == "cumcount":
                    # pandas 0-indexes cumulative count, see
                    # https://github.com/rapidsai/cudf/issues/10237
                    result_col = result_col - 1
                if cast_dtype is not None:
                    result_col = result_col.astype(cast_dtype)
                data[key] = result_col
        # Preserve the column axis label-dtype/level_names from the source
        # DataFrame so that aggregations such as ``nunique`` keep the column
        # axis name (matching pandas behavior).
        if len(data) == 0 and not multilevel and self.obj.ndim == 2:
            # No columns were aggregated (e.g. a frame with no value
            # columns): mirror the source column axis so its dtype and
            # RangeIndex-ness are preserved. Otherwise an empty
            # ColumnAccessor reconstructs its columns as a string/object
            # Index, whereas pandas keeps the original (e.g. empty
            # RangeIndex) columns.
            data = ColumnAccessor(
                data,
                multiindex=self.obj._data.multiindex,
                level_names=self.obj._data.level_names,
                rangeindex=self.obj._data.rangeindex,
                label_dtype=self.obj._data.label_dtype,
                level_dtypes=self.obj._data.level_dtypes,
            )
        elif (
            not multilevel
            and self.obj.ndim == 2
            and self.obj._data.level_names != (None,)
        ):
            mi_kwargs: dict[str, Any] = {}
            if self.obj._data.multiindex and all(
                isinstance(label, tuple)
                and len(label) == self.obj._data.nlevels
                for label in data
            ):
                # the aggregation kept the source's tuple labels: preserve
                # the MultiIndex columns and their per-level metadata.
                # Relabeling aggregations (``agg(new=(col, func))``) emit
                # new flat labels the source's multi-level metadata does
                # not describe, so they keep the flat default.
                mi_kwargs = {
                    "multiindex": True,
                    "level_dtypes": self.obj._data.level_dtypes,
                }
            data = ColumnAccessor(
                data,
                level_names=self.obj._data.level_names,
                label_dtype=self.obj._data.label_dtype,
                **mi_kwargs,
            )
        else:
            data = ColumnAccessor(data, multiindex=multilevel)
        if not multilevel and len(data) > 0:
            # Skip when there are no columns: there is nothing to rename, and
            # rebuilding the ColumnAccessor would discard column-axis metadata
            # (e.g. the preserved RangeIndex/dtype set above).
            data = data.rename_levels({np.nan: None}, level=0)

        result = DataFrame._from_data(data, index=result_index)

        if self._sort:
            result = result.sort_index()
        else:
            if get_option(
                "mode.pandas_compatible"
            ) and not _is_all_scan_aggregate(normalized_aggs):
                # Even with `sort=False`, pandas guarantees that
                # groupby preserves the order of rows within each group.
                left_cols = self.grouping.keys.drop_duplicates()._columns
                right_cols = result_index._columns
                join_keys = [
                    _match_join_keys(lcol, rcol, "inner")
                    for lcol, rcol in zip(left_cols, right_cols, strict=True)
                ]
                # TODO: In future, see if we can centralize
                # logic else where that has similar patterns.
                join_keys = map(list, zip(*join_keys, strict=True))
                # By construction, left and right keys are related by
                # a permutation, so we can use an inner join.
                join_keys_list = list(join_keys)
                # Flatten nested list of columns for access_columns
                all_cols = [col for cols in join_keys_list for col in cols]
                with access_columns(
                    *all_cols, mode="read", scope="internal"
                ) as all_cols:
                    # Reconstruct join_keys_list structure from flattened all_cols
                    idx = 0
                    plc_tables = []
                    for cols in join_keys_list:
                        cols_len = len(cols)
                        plc_tables.append(
                            plc.Table(
                                [
                                    col.plc_column
                                    for col in all_cols[idx : idx + cols_len]
                                ]
                            )
                        )
                        idx += cols_len
                    left_plc, right_plc = plc.join.inner_join(
                        plc_tables[0],
                        plc_tables[1],
                        plc.types.NullEquality.EQUAL,
                    )
                    left_order = ColumnBase.create(
                        left_plc, dtype=dtype_from_pylibcudf_column(left_plc)
                    )
                    right_order = ColumnBase.create(
                        right_plc, dtype=dtype_from_pylibcudf_column(right_plc)
                    )
                # TODO: Perform inner_join and sort_by_key all in pylibcudf
                # left order is some permutation of the ordering we
                # want, and right order is a matching gather map for
                # the result table. Get the correct order by sorting
                # the right gather map.
                plc_right_order = sorting.sort_by_key(
                    [right_order],
                    [left_order],
                    [True],
                    ["first"],
                    stable=False,
                )[0]

                result = result._gather(
                    GatherMap.from_column_unchecked(
                        ColumnBase.create(
                            plc_right_order,
                            dtype=dtype_from_pylibcudf_column(plc_right_order),
                        ),
                        len(result),
                        nullify=False,
                    )
                )

        is_scan = _is_all_scan_aggregate(normalized_aggs)
        if not self._as_index and not is_scan:
            result = result.reset_index()
        if is_scan:
            # Scan aggregations are transforms: rows are returned in the
            # original index order and the grouping keys are never part of
            # the output, regardless of ``as_index``.
            return self._mimic_pandas_order(result)

        return result

    def _reduce_numeric_only(self, op: str):
        raise NotImplementedError(
            f"numeric_only is not implemented for {type(self)}"
        )

    def _reduce(
        self,
        op: str,
        numeric_only: bool = False,
        min_count: int = 0,
        *args,
        **kwargs,
    ):
        """Compute {op} of group values.

        Parameters
        ----------
        numeric_only : bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data.
        min_count : int, default 0
            The required number of valid values to perform the operation. If
            fewer than ``min_count`` non-NA values are present the result will
            be NA.

        Returns
        -------
        Series or DataFrame
            Computed {op} of values within each group.

        .. pandas-compat::
            :meth:`pandas.core.groupby.DataFrameGroupBy.{op}`,
             :meth:`pandas.core.groupby.SeriesGroupBy.{op}`

            The numeric_only, min_count
        """
        if numeric_only:
            return self._reduce_numeric_only(op)

        if op == "sum" and self._has_string_value_column():
            return self._string_sum(
                skipna=kwargs.get("skipna", True), min_count=min_count
            )

        skipna = kwargs.get("skipna", True)
        agg_op: str | _FirstLastAggSpec = op
        if op in {"first", "last"} and not skipna:
            # ``first``/``last`` default to dropping nulls (skipna=True). With
            # ``skipna=False`` the actual first/last element of each group is
            # returned even when it is null, matching pandas.
            agg_op = _FirstLastAggSpec(op, skipna=False)

        result = self.agg(agg_op)
        if op in _NULL_PROPAGATING_REDUCTIONS and not skipna:
            # libcudf reductions always drop nulls. With ``skipna=False`` a
            # group containing any null in a column yields a null result for
            # that (group, column), matching pandas. A (group, column) is
            # all-non-null when its non-null count equals the group size
            # (``size()`` is used instead of the ``size`` aggregation because
            # the latter is unsupported for string columns).
            from cudf.core.dataframe import DataFrame

            non_null_counts = self.agg("count")
            group_sizes = self.size()
            if isinstance(group_sizes, DataFrame):
                # With ``as_index=False`` the per-group counts are returned as
                # the "size" column of a DataFrame; reduce it to a Series so it
                # aligns with each value column below.
                group_sizes = group_sizes["size"]
            if isinstance(result, DataFrame):
                # ``as_index=False`` keeps the grouping keys as columns of
                # ``result``; they must never be nulled out, so mask only the
                # value columns.
                key_names = set(self.grouping.names)
                for name in result._column_names:
                    if name in key_names:
                        continue
                    result[name] = result[name].where(
                        non_null_counts[name] == group_sizes, None
                    )
            else:
                result = result.where(non_null_counts == group_sizes, None)
        if min_count and min_count > 0:
            counts = self.agg("count")
            result = result.where(counts >= min_count, None)
        return result

    def _scan(self, op: str, *args, **kwargs):
        """{op_name} for each group."""
        return self.agg(op)

    def _has_string_value_column(self) -> bool:
        from cudf.core.series import Series

        if isinstance(self.obj, Series):
            return isinstance(self.obj.dtype, pd.StringDtype)
        for col_name in self.grouping._values_column_names:
            if isinstance(self.obj._data[col_name].dtype, pd.StringDtype):
                return True
        return False

    def _string_sum(self, *, skipna: bool, min_count: int):
        """Implement groupby sum for StringDtype columns as per-group
        string concatenation.
        """
        from cudf.core.column import ColumnBase
        from cudf.core.dataframe import DataFrame
        from cudf.core.series import Series

        is_series = isinstance(self.obj, Series)
        if is_series:
            value_cols: list[tuple[Any, ColumnBase]] = [
                (self.obj.name, self.obj._column)
            ]
        else:
            value_cols = []
            for col_name in self.grouping._values_column_names:
                col = self.obj._data[col_name]
                if not isinstance(col.dtype, pd.StringDtype):
                    # TODO: handle mixed dtype frames
                    raise NotImplementedError(
                        "sum on mixed string and non-string columns is "
                        "not yet supported"
                    )
                value_cols.append((col_name, col))

        # Build a single batched groupby aggregation: one request per value
        # column, computing collect_list and (when min_count > 0) count.
        aggs = [plc.aggregation.collect_list()]
        if min_count > 0:
            aggs.append(plc.aggregation.count())
        requests = [
            plc.groupby.GroupByRequest(col.plc_column, aggs)
            for _, col in value_cols
        ]
        columns_for_access = [col for _, col in value_cols]
        with access_columns(
            *columns_for_access, mode="read", scope="internal"
        ):
            with self._groupby_manager as plc_groupby:
                keys, results = plc_groupby.aggregate(requests)

        sep = plc.Scalar.from_py("")
        sep_narep = plc.Scalar.from_py("")
        if skipna:
            string_narep = plc.Scalar.from_py("")
            empty_policy = plc.strings.combine.OutputIfEmptyList.EMPTY_STRING
        else:
            string_narep = plc.Scalar.from_py(
                None, plc.DataType(plc.TypeId.STRING)
            )
            empty_policy = plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT
        null_str = plc.Scalar.from_py(None, plc.DataType(plc.TypeId.STRING))

        out_data: dict[Any, ColumnBase] = {}
        for (col_name, col), table in zip(value_cols, results, strict=True):
            agg_columns = table.columns()
            joined = plc.strings.combine.join_list_elements(
                agg_columns[0],
                sep,
                sep_narep,
                string_narep,
                plc.strings.combine.SeparatorOnNulls.YES,
                empty_policy,
            )
            if min_count > 0:
                keep_mask_plc = plc.binaryop.binary_operation(
                    agg_columns[1],
                    plc.Scalar.from_py(min_count),
                    plc.binaryop.BinaryOperator.GREATER_EQUAL,
                    plc.DataType(plc.TypeId.BOOL8),
                )
                joined = plc.copying.copy_if_else(
                    joined, null_str, keep_mask_plc
                )
            out_data[col_name] = ColumnBase.create(joined, col.dtype)

        key_dtypes = [col.dtype for col in self.grouping._key_columns]
        index = self.grouping.keys._from_columns_like_self(
            [
                ColumnBase.create(key, dtype)
                for key, dtype in zip(keys.columns(), key_dtypes, strict=True)
            ]
        )

        if is_series:
            return Series._from_column(
                out_data[self.obj.name], name=self.obj.name, index=index
            )
        return DataFrame._from_data(out_data, index=index)

    aggregate = agg

    def _head_tail(self, n, *, take_head: bool, preserve_order: bool):
        """Return the head or tail of each group

        Parameters
        ----------
        n
           Number of entries to include (if negative, number of
           entries to exclude)
        take_head
           Do we want the head or the tail of the group
        preserve_order
            If True, return the n rows from each group in original
            dataframe order (this mimics pandas behavior though is
            more expensive).

        Returns
        -------
        New DataFrame or Series

        Notes
        -----
        Unlike pandas, this returns an object in group order, not
        original order, unless ``preserve_order`` is ``True``.
        """
        # A more memory-efficient implementation would merge the take
        # into the grouping, but that probably requires a new
        # aggregation scheme in libcudf. This is probably "fast
        # enough" for most reasonable input sizes.
        _, offsets, _, group_values = self._grouped()
        group_offsets = np.asarray(offsets, dtype=SIZE_TYPE_DTYPE)
        size_per_group = np.diff(group_offsets)
        # "Out of bounds" n for the group size either means no entries
        # (negative) or all the entries (positive)
        if n < 0:
            size_per_group = np.maximum(
                size_per_group + n, 0, out=size_per_group
            )
        else:
            size_per_group = np.minimum(size_per_group, n, out=size_per_group)
        if take_head:
            group_offsets = group_offsets[:-1]
        else:
            group_offsets = group_offsets[1:] - size_per_group
        to_take_indices = np.arange(
            size_per_group.sum(), dtype=SIZE_TYPE_DTYPE
        )
        fixup = np.empty_like(size_per_group)
        fixup[0] = 0
        np.cumsum(size_per_group[:-1], out=fixup[1:])
        to_take_indices += np.repeat(group_offsets - fixup, size_per_group)
        to_take = as_column(to_take_indices)
        result = group_values.iloc[to_take]
        if preserve_order:
            # Can't use _mimic_pandas_order because we need to
            # subsample the gather map from the full input ordering,
            # rather than permuting the gather map of the output.
            _, _, (ordering,) = self._groups([self._range_column_from_obj])
            # Invert permutation from original order to groups on the
            # subset of entries we want.
            gather_map = ordering.take(to_take).argsort()
            return result.take(gather_map)
        else:
            return result

    @_performance_tracking
    def head(self, n: int = 5, *, preserve_order: bool = True):
        """Return first n rows of each group

        Parameters
        ----------
        n
            If positive: number of entries to include from start of group
            If negative: number of entries to exclude from end of group

        preserve_order
            If True (default), return the n rows from each group in
            original dataframe order (this mimics pandas behavior
            though is more expensive). If you don't need rows in
            original dataframe order you will see a performance
            improvement by setting ``preserve_order=False``. In both
            cases, the original index is preserved, so ``.loc``-based
            indexing will work identically.

        Returns
        -------
        Series or DataFrame
            Subset of the original grouped object as determined by n

        See Also
        --------
        .tail

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame(
        ...     {
        ...         "a": [1, 0, 1, 2, 2, 1, 3, 2, 3, 3, 3],
        ...         "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     }
        ... )
        >>> df.groupby("a").head(1)
           a  b
        0  1  0
        1  0  1
        3  2  3
        6  3  6
        >>> df.groupby("a").head(-2)
           a  b
        0  1  0
        3  2  3
        6  3  6
        8  3  8
        """
        return self._head_tail(
            n, take_head=True, preserve_order=preserve_order
        )

    @_performance_tracking
    def tail(self, n: int = 5, *, preserve_order: bool = True):
        """Return last n rows of each group

        Parameters
        ----------
        n
            If positive: number of entries to include from end of group
            If negative: number of entries to exclude from start of group

        preserve_order
            If True (default), return the n rows from each group in
            original dataframe order (this mimics pandas behavior
            though is more expensive). If you don't need rows in
            original dataframe order you will see a performance
            improvement by setting ``preserve_order=False``. In both
            cases, the original index is preserved, so ``.loc``-based
            indexing will work identically.

        Returns
        -------
        Series or DataFrame
            Subset of the original grouped object as determined by n


        See Also
        --------
        .head

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame(
        ...     {
        ...         "a": [1, 0, 1, 2, 2, 1, 3, 2, 3, 3, 3],
        ...         "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     }
        ... )
        >>> df.groupby("a").tail(1)
            a   b
        1   0   1
        5   1   5
        7   2   7
        10  3  10
        >>> df.groupby("a").tail(-2)
            a   b
        5   1   5
        7   2   7
        9   3   9
        10  3  10
        """
        return self._head_tail(
            n, take_head=False, preserve_order=preserve_order
        )

    @property
    def nth(self):
        """
        Take the nth row from each group if n is an int, otherwise a
        subset of rows.

        Like pandas, supports both the call form ``gb.nth(n, dropna=...)``
        and the index form ``gb.nth[n]``.

        Parameters
        ----------
        n : int, slice or list of ints and slices
            A single nth value for the row, a slice with non-negative
            step or a list of nth values and slices. Negative values
            count from the end of each group.
        dropna : {'any', 'all', None}, default None
            Apply the specified dropna operation before counting which
            row is the nth row. Only supported in the call form and not
            currently implemented in cuDF (raises ``NotImplementedError``;
            falls back to pandas under ``cudf.pandas``).

        Returns
        -------
        Series or DataFrame
            The nth row(s) of each group, keeping the original index and
            row order (like a filter operation, the group keys are not
            added as an index level).

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 1, 2, 1, 2],
        ...                      "B": [None, 2, 3, 4, 5]})
        >>> gb = df.groupby("A")
        >>> gb.nth(0)
           A     B
        0  1  <NA>
        2  2     3
        >>> gb.nth(-1)
           A  B
        3  1  4
        4  2  5
        >>> gb.nth[:2]
           A     B
        0  1  <NA>
        1  1     2
        2  2     3
        4  2     5
        """
        return GroupByNthSelector(self)

    @_performance_tracking
    def _nth(self, n, dropna: Literal["any", "all", None] = None):
        """Positional row filter mirroring pandas' GroupBy.nth."""
        if dropna is not None:
            raise NotImplementedError("dropna is not currently supported.")

        # Normalize and validate ``n`` like pandas'
        # GroupByIndexingMixin._make_mask_from_positional_indexer.
        if isinstance(n, (int, np.integer)):
            args: list = [int(n)]
        elif isinstance(n, slice):
            args = [n]
        elif isinstance(n, (list, tuple, np.ndarray)):
            args = list(n)
        else:
            raise TypeError(
                f"Invalid index {type(n)}. "
                "Must be integer, list-like, slice or a tuple of "
                "integers and slices"
            )
        for arg in args:
            if isinstance(arg, slice):
                if (arg.step or 1) < 0:
                    raise ValueError(
                        f"Invalid step {arg.step}. Must be non-negative"
                    )
            elif not isinstance(arg, (int, np.integer)):
                raise TypeError(
                    f"Invalid index {type(n)}. "
                    "Must be integer, list-like, slice or a tuple of "
                    "integers and slices"
                )

        # Per-row position within its group and group size, in group-major
        # order (same construction as ``_head_tail``).
        _, offsets, _, _ = self._grouped()
        group_offsets = np.asarray(offsets, dtype=SIZE_TYPE_DTYPE)
        size_per_group = np.diff(group_offsets)
        sizes = np.repeat(size_per_group, size_per_group)
        pos = np.arange(len(sizes), dtype=SIZE_TYPE_DTYPE) - np.repeat(
            group_offsets[:-1], size_per_group
        )

        mask = np.zeros(len(sizes), dtype=bool)
        for arg in args:
            if isinstance(arg, slice):
                step = arg.step or 1
                if arg.start is None:
                    start = np.zeros_like(sizes)
                elif arg.start >= 0:
                    start = np.full_like(sizes, arg.start)
                else:
                    # ``slice.indices`` clamps a negative start at 0 and
                    # the step alignment begins at the clamped value
                    start = np.maximum(sizes + arg.start, 0)
                submask = pos >= start
                if step > 1:
                    submask &= (pos - start) % step == 0
                if arg.stop is not None:
                    if arg.stop >= 0:
                        submask &= pos < arg.stop
                    else:
                        submask &= pos < np.maximum(sizes + arg.stop, 0)
                mask |= submask
            elif arg >= 0:
                mask |= pos == arg
            else:
                mask |= pos == sizes + arg

        # Map the selected group-major rows back to positions in the
        # original object and gather from the *pre-nans_to_nulls* object:
        # pandas' nth is a row filter, so values, dtypes, index and row
        # order are those of the original rows.
        to_take = as_column(np.nonzero(mask)[0].astype(SIZE_TYPE_DTYPE))
        _, _, (ordering,) = self._groups([self._range_column_from_obj])
        original_positions = ordering.take(to_take)
        original_positions = original_positions.take(
            original_positions.argsort()
        )
        return self._obj_original.take(original_positions)

    @_performance_tracking
    def ngroup(self, ascending=True):
        """
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount. Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A": list("aaabba")})
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').ngroup()
        0    0
        1    0
        2    0
        3    1
        4    1
        5    0
        dtype: int64
        >>> df.groupby('A').ngroup(ascending=False)
        0    1
        1    1
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby(["A", [1,1,2,3,2,1]]).ngroup()
        0    0
        1    0
        2    1
        3    2
        4    3
        5    0
        dtype: int64
        """
        from cudf.core.series import Series

        index = self.grouping.keys.unique()
        # Groups are numbered in the order they would be iterated over: sorted
        # order when ``sort=True``, otherwise order of first appearance.
        if self._sort:
            index = index.sort_values()
        num_groups = len(index)

        if not self._dropna:
            # ``dropna=False``: a group whose key contains a null is a regular
            # group numbered like any other, so the labels are simply the
            # group positions in iteration order.
            seq = (
                range(num_groups)
                if ascending
                else range(num_groups - 1, -1, -1)
            )
            group_ids = Series._from_column(
                as_column(seq, dtype=np.dtype(np.int64))
            )
        else:
            # ``dropna=True``: pandas labels rows whose key contains a null
            # with NA and excludes those groups from the numbering. A group is
            # a "null group" when any of its key columns is null there.
            null_group_col = functools.reduce(
                lambda a, b: a | b,
                (col.isnull() for col in index._columns),
            )
            non_null_mask = ~null_group_col
            non_null = non_null_mask.astype(SIZE_TYPE_DTYPE)
            # 0-based position of each labeled (non-null) group; the value at
            # null positions is irrelevant as it is replaced with NA below.
            rank = non_null.cumsum() - non_null
            if not ascending:
                rank = (int(non_null.sum()) - 1) - rank
            group_ids = Series._from_column(
                rank.astype(np.dtype(np.int64)).copy_if_else(
                    pa_scalar_to_plc_scalar(pa.scalar(None, type=pa.int64())),
                    non_null_mask,
                )
            )
        group_ids.index = index
        return self._broadcast(group_ids)

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: Sequence | Series | None = None,
        random_state: np.random.RandomState | int | None = None,
    ):
        """Return a random sample of items in each group.

        Parameters
        ----------
        n
            Number of items to return for each group, if sampling
            without replacement must be at most the size of the
            smallest group. Cannot be used with frac. Default is
            ``n=1`` if frac is None.
        frac
            Fraction of items to return. Cannot be used with n.
        replace
            Should sampling occur with or without replacement?
        weights
            Sampling probability for each element. Must be the same
            length as the grouped frame. Not currently supported.
        random_state
            Seed for random number generation.

        Returns
        -------
        New dataframe or series with samples of appropriate size drawn
        from each group.

        """
        if weights is not None:
            # To implement this case again needs different algorithms
            # in both cases.
            #
            # Without replacement, use the weighted reservoir sampling
            # approach of Efraimidas and Spirakis (2006)
            # https://doi.org/10.1016/j.ipl.2005.11.003, essentially,
            # do a segmented argsort sorting on weight-scaled
            # logarithmic deviates. See
            # https://timvieira.github.io/blog/algorithms-for-sampling-without-replacement/
            #
            # With replacement is trickier, one might be able to use
            # the alias method, otherwise we're back to bucketed
            # rejection sampling.
            raise NotImplementedError("Sampling with weights is not supported")
        if frac is not None and n is not None:
            raise ValueError("Cannot supply both of frac and n")
        elif n is None and frac is None:
            n = 1
        elif frac is not None and not (0 <= frac <= 1):
            raise ValueError(
                "Sampling with fraction must provide fraction in "
                f"[0, 1], got {frac=}"
            )
        # TODO: handle random states properly.
        if random_state is not None and not isinstance(random_state, int):
            raise NotImplementedError(
                "Only integer seeds are supported for random_state "
                "in this case"
            )
        # Get the groups
        # TODO: convince Cython to convert the std::vector offsets
        # into a numpy array directly, rather than a list.
        # TODO: this uses the sort-based groupby, could one use hash-based?
        _, offsets, _, group_values = self._grouped()
        group_offsets = np.asarray(offsets, dtype=SIZE_TYPE_DTYPE)
        size_per_group = np.diff(group_offsets)
        if n is not None:
            samples_per_group = np.broadcast_to(
                SIZE_TYPE_DTYPE.type(n), size_per_group.shape
            )
            if not replace and (minsize := size_per_group.min()) < n:
                raise ValueError(
                    f"Cannot sample {n=} without replacement, "
                    f"smallest group is {minsize}"
                )
        else:
            # Pandas uses round-to-nearest, ties to even to
            # pick sample sizes for the fractional case (unlike IEEE
            # which is round-to-nearest, ties to sgn(x) * inf).
            samples_per_group = np.round(
                size_per_group * frac, decimals=0
            ).astype(SIZE_TYPE_DTYPE)
        if replace:
            # We would prefer to use cupy here, but their rng.integers
            # interface doesn't take array-based low and high
            # arguments.
            low = 0
            high: np.ndarray = np.repeat(size_per_group, samples_per_group)
            rng = np.random.default_rng(seed=random_state)
            indices = rng.integers(low, high, dtype=SIZE_TYPE_DTYPE)
            indices += np.repeat(group_offsets[:-1], samples_per_group)
        else:
            # Approach: do a segmented argsort of the index array and take
            # the first samples_per_group entries from sorted array.
            # We will shuffle the group indices and then pick them out
            # from the grouped dataframe index.
            nrows = len(group_values)
            indices = cp.arange(nrows, dtype=SIZE_TYPE_DTYPE)
            if len(size_per_group) < 500:
                # Empirically shuffling with cupy is faster at this scale
                rs = cp.random.get_random_state()
                rs.seed(seed=random_state)
                for off, size in zip(
                    group_offsets[:-1], size_per_group, strict=True
                ):
                    rs.shuffle(indices[off : off + size])
            else:
                keys = cp.random.default_rng(seed=random_state).random(
                    size=nrows
                )
                indices_col = as_column(indices)
                keys_col = as_column(keys)
                group_offsets_col = as_column(group_offsets)
                with access_columns(
                    indices_col,
                    keys_col,
                    group_offsets_col,
                    mode="read",
                    scope="internal",
                ) as (indices_col, keys_col, group_offsets_col):
                    plc_column = plc.sorting.stable_segmented_sort_by_key(
                        plc.Table([indices_col.plc_column]),
                        plc.Table([keys_col.plc_column]),
                        group_offsets_col.plc_column,
                        [plc.types.Order.ASCENDING],
                        [plc.types.NullOrder.AFTER],
                    ).columns()[0]
                    indices = cp.array(plc_column.data())
            # Which indices are we going to want?
            want = np.arange(samples_per_group.sum(), dtype=SIZE_TYPE_DTYPE)
            scan = np.empty_like(samples_per_group)
            scan[0] = 0
            np.cumsum(samples_per_group[:-1], out=scan[1:])
            want += np.repeat(group_offsets[:-1] - scan, samples_per_group)
            indices = indices[want]
        return group_values.iloc[indices]

    def serialize(self):
        header = {}
        frames = []

        header["kwargs"] = {
            "sort": self._sort,
            "dropna": self._dropna,
            "as_index": self._as_index,
        }

        obj_header, obj_frames = self.obj.serialize()
        header["obj"] = obj_header
        header["obj_type_name"] = type(self.obj).__name__
        header["num_obj_frames"] = len(obj_frames)
        frames.extend(obj_frames)

        grouping_header, grouping_frames = self.grouping.serialize()
        header["grouping"] = grouping_header
        header["num_grouping_frames"] = len(grouping_frames)
        frames.extend(grouping_frames)

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        kwargs = header["kwargs"]

        obj_type = Serializable._name_type_map[header["obj_type_name"]]
        obj = obj_type.deserialize(
            header["obj"], frames[: header["num_obj_frames"]]
        )
        grouping = _Grouping.deserialize(
            header["grouping"], frames[header["num_obj_frames"] :]
        )
        return cls(obj, grouping, **kwargs)

    def _grouped(self, *, include_groups: bool = True):
        from cudf.core.dataframe import DataFrame

        offsets, grouped_key_cols, grouped_value_cols = self._groups(
            itertools.chain(self.obj.index._columns, self.obj._columns)
        )
        grouped_keys = _index_from_data(dict(enumerate(grouped_key_cols)))
        if isinstance(self.grouping.keys, MultiIndex):
            grouped_keys.names = self.grouping.keys.names
            to_drop = self.grouping.keys.names
        else:
            grouped_keys.name = self.grouping.keys.name
            to_drop = (self.grouping.keys.name,)
        grouped_values = self.obj._from_columns_like_self(
            grouped_value_cols,
            column_names=self.obj._column_names,
            index_names=self.obj.index.names,
        )
        if not include_groups and isinstance(grouped_values, DataFrame):
            selection = getattr(self, "_selection", None)
            for col_name in to_drop:
                if col_name in grouped_values._column_names and (
                    selection is None or col_name not in selection
                ):
                    del grouped_values[col_name]
        group_names = grouped_keys.unique().sort_values()
        return (group_names, offsets, grouped_keys, grouped_values)

    def _normalize_aggs(
        self, aggs: MultiColumnAggType, **kwargs
    ) -> tuple[Iterable[Any], tuple[ColumnBase, ...], list[list[AggType]]]:
        """
        Normalize aggs to a list of list of aggregations, where `out[i]`
        is a list of aggregations for column `self.obj[i]`. We support four
        different form of `aggs` input here:
        - A single agg, such as "sum". This agg is applied to all value
        columns.
        - A list of aggs, such as ["sum", "mean"]. All aggs are applied to all
        value columns.
        - A mapping of column name to aggs, such as
        {"a": ["sum"], "b": ["mean"]}, the aggs are applied to specified
        column.
        - Pairs of column name and agg tuples passed as kwargs
        eg. col1=("a", "sum"), col2=("b", "prod"). The output column names are
        the keys. The aggs are applied to the corresponding column in the tuple.
        Each agg can be string or lambda functions.
        """
        aggs_per_column: Iterable[AggType | Iterable[AggType]]
        # TODO: Remove isinstance condition when the legacy dask_cudf API is removed.
        # See https://github.com/rapidsai/cudf/pull/16528#discussion_r1715482302 for information.
        if aggs or isinstance(aggs, dict):
            if isinstance(aggs, dict):
                if any(
                    is_list_like(values) and len(set(values)) != len(values)  # type: ignore[arg-type]
                    for values in aggs.values()
                ):
                    if get_option("mode.pandas_compatible"):
                        raise NotImplementedError(
                            "Duplicate aggregations per column are currently not supported."
                        )
                    else:
                        warnings.warn(
                            "Duplicate aggregations per column found. "
                            "The resulting duplicate columns will be dropped.",
                            UserWarning,
                        )
                column_names, aggs_per_column = aggs.keys(), aggs.values()
                columns = tuple(self.obj._data[col] for col in column_names)
            else:
                if isinstance(aggs, list) and len(aggs) != len(set(aggs)):
                    raise pd.errors.SpecificationError(
                        "Function names must be unique if there is no new column names assigned"
                    )
                values = self.grouping.values
                column_names = values._column_names
                columns = values._columns
                aggs_per_column = (aggs,) * len(columns)
        elif not aggs and kwargs:
            column_names = kwargs.keys()

            def _raise_invalid_type(x):
                raise TypeError(
                    f"Invalid keyword argument {x} of type {type(x)} was passed to agg"
                )

            columns, aggs_per_column = zip(
                *(
                    (self.obj._data[x[0]], x[1])
                    if isinstance(x, (tuple, NamedAgg))
                    else _raise_invalid_type(x)
                    for x in kwargs.values()
                ),
                strict=True,
            )
        else:
            raise TypeError("Must provide at least one aggregation function.")

        # is_list_like performs type narrowing but type-checkers don't
        # know it. One could add a TypeGuard annotation to
        # is_list_like (see PEP647), but that is less useful than it
        # seems because unlike the builtin narrowings it only performs
        # narrowing in the positive case.
        normalized_aggs = [
            list(agg) if is_list_like(agg) else [agg]  # type: ignore[arg-type]
            for agg in aggs_per_column
        ]
        return column_names, columns, normalized_aggs  # type: ignore[return-value]  # (list-like narrowing is not represented)

    @_performance_tracking
    def pipe(self, func, *args, **kwargs):
        """
        Apply a function `func` with arguments to this GroupBy
        object and return the function's result.

        Parameters
        ----------
        func : function
            Function to apply to this GroupBy object or,
            alternatively, a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the GroupBy object.
        args : iterable, optional
            Positional arguments passed into ``func``.
        kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        See Also
        --------
        cudf.Series.pipe
            Apply a function with arguments to a series.

        cudf.DataFrame.pipe
            Apply a function with arguments to a dataframe.

        apply
            Apply function to each group instead of to the full GroupBy object.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value
        in one pass, you can do

        >>> df.groupby('A', sort=True).pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2
        """
        return pipe(self, func, *args, **kwargs)

    @_performance_tracking
    def _jit_groupby_apply(
        self, function, group_names, offsets, group_keys, grouped_values, *args
    ):
        chunk_results = jit_groupby_apply(
            offsets, grouped_values, function, *args
        )
        return self._post_process_chunk_results(
            chunk_results, group_names, group_keys, grouped_values
        )

    @_performance_tracking
    def _iterative_groupby_apply(
        self, function, group_names, offsets, group_keys, grouped_values, *args
    ):
        ngroups = len(offsets) - 1
        if ngroups > self._MAX_GROUPS_BEFORE_WARN:
            warnings.warn(
                f"GroupBy.apply() performance scales poorly with "
                f"number of groups. Got {ngroups} groups. Some functions "
                "may perform better by passing engine='jit'",
                RuntimeWarning,
            )

        chunks = [grouped_values[s:e] for s, e in itertools.pairwise(offsets)]
        chunk_results = [function(chk, *args) for chk in chunks]
        return self._post_process_chunk_results(
            chunk_results, group_names, group_keys, grouped_values
        )

    def _post_process_chunk_results(
        self, chunk_results, group_names, group_keys, grouped_values
    ):
        from cudf.core.dataframe import DataFrame
        from cudf.core.series import Series

        if not len(chunk_results):
            return self.obj.head(0)
        if (
            isinstance(self.obj, DataFrame)
            and not isinstance(chunk_results, ColumnBase)
            and all(res is None for res in chunk_results)
        ):
            # pandas GH9684/GH57775: an all-None DataFrameGroupBy.apply
            # returns an empty frame keeping the (non-grouping) columns and
            # dtypes. (An all-None SeriesGroupBy.apply stays in the scalar
            # branch below: pandas returns an object Series of Nones.)
            return grouped_values.head(0).reset_index(drop=True)
        if isinstance(chunk_results, ColumnBase) or is_scalar(
            chunk_results[0]
        ):
            data = ColumnAccessor(
                {None: as_column(chunk_results)}, verify=False
            )
            ty = Series if self._as_index else DataFrame
            result = ty._from_data(data, index=group_names)
            result.index.names = self.grouping.names
            return result

        elif isinstance(chunk_results[0], Series) and isinstance(
            self.obj, DataFrame
        ):
            # When the UDF is like df.sum(), the result for each
            # group is a row-like "Series" where the index labels
            # are the same as the original calling DataFrame
            if _is_row_of(chunk_results[0], self.obj):
                result = concat(chunk_results, axis=1).T
                result.index = group_names
                result.index.names = self.grouping.names
                # pandas names the columns axis after the row-like Series
                # returned by the UDF (e.g. ``iloc[0]`` carries the original
                # row label as its name); ``concat(..., axis=1).T`` otherwise
                # drops it, leaving an unnamed columns axis.
                result.columns = result.columns.set_names(
                    [chunk_results[0].name]
                )
            # pandas stacks Series results that share an identical index
            # into a DataFrame with one row per group and columns given by
            # the common index (DataFrameGroupBy._wrap_applied_output_series)
            elif all(
                chunk_results[0].index.equals(chk.index)
                for chk in chunk_results[1:]
            ):
                # a consistent Series name becomes the columns-axis name
                # (pandas GH6124). Chunks are renamed positionally before
                # the axis=1 concat because cuDF rejects duplicate column
                # names.
                names = {chk.name for chk in chunk_results}
                result = concat(
                    [chk.rename(i) for i, chk in enumerate(chunk_results)],
                    axis=1,
                ).T
                result.index = group_names
                result.index.names = self.grouping.names
                if len(names) == 1:
                    result._data._level_names = (names.pop(),)
            else:
                # pandas GH8467: Series results with differing indexes are
                # concatenated along axis 0 into a Series with the group
                # keys prepended as the outer index level(s), each key
                # repeated by its chunk's actual length and the UDF-returned
                # index kept as the inner level
                # (GroupBy._concat_objects with ``not_indexed_same=True``).
                # This also covers transform-like UDFs: chunks indexed like
                # their input concatenate back to the grouped input's index.
                lengths = [len(chk) for chk in chunk_results]
                result = concat(chunk_results)
                gather = as_column(
                    np.repeat(np.arange(len(group_names)), lengths)
                )
                index_data = {
                    i: col.take(gather)
                    for i, col in enumerate(group_names._columns)
                }
                inner_name = result.index.name
                index_data[None] = result.index._column
                mi = MultiIndex._from_data(index_data)
                mi.names = [*self.grouping.names, inner_name]
                result.index = mi
        else:
            result = concat(chunk_results)
            if self._group_keys:
                index_data = group_keys._data.copy(deep=True)
                # The inner index level is the index returned by the UDF for
                # each group (preserved through ``concat``), not the original
                # row positions of the grouped values. This matches pandas,
                # e.g. a UDF returning ``DataFrame({"values": range(len(grp))})``
                # contributes a fresh 0..len(grp)-1 range per group.
                inner_name = result.index.name
                index_data[None] = result.index._column
                mi = MultiIndex._from_data(index_data)
                # ColumnAccessor keys must be unique, so the inner level's
                # name (which may duplicate a key name) is restored after
                # construction.
                mi.names = [*mi.names[:-1], inner_name]
                result.index = mi
            elif len(result) == len(grouped_values) and result.index.equals(
                grouped_values.index
            ):
                # Every chunk result is indexed like its input chunk, i.e.
                # the UDF acted as a transform. pandas restores the original
                # row order in this case (GroupBy._concat_objects) regardless
                # of ``sort``. The concatenated chunks are in key-sorted
                # group order, so gather back through the inverse of the
                # grouping permutation.
                _, _, (positions,) = self._groups(
                    [self._range_column_from_obj]
                )
                result = result.take(positions.argsort().values)
        return result

    @_performance_tracking
    def apply(
        self,
        func,
        *args,
        engine="auto",
        include_groups: bool = False,
        **kwargs,
    ):
        """Apply a python transformation function over the grouped chunk.

        Parameters
        ----------
        func : callable
          The python transformation function that will be applied
          on the grouped chunk.
        args : tuple
            Optional positional arguments to pass to the function.
        engine: 'auto', 'cudf', or 'jit', default 'auto'
          Selects the GroupBy.apply implementation. Use `jit` to
          select the numba JIT pipeline. Only certain operations are allowed
          within the function when using this option: min, max, sum, mean, var,
          std, idxmax, and idxmin and any arithmetic formula involving them are
          allowed. Binary operations are not yet supported, so syntax like
          `df['x'] * 2` is not yet allowed.
          For more information, see the `cuDF guide to user defined functions
          <https://docs.rapids.ai/api/cudf/stable/cudf/guide-to-udfs/>`__.
          Use `cudf` to select the iterative groupby apply algorithm which aims
          to provide maximum flexibility at the expense of performance.
          The default value `auto` will attempt to use the numba JIT pipeline
          where possible and will fall back to the iterative algorithm if
          necessary.
        include_groups : bool, default False
            Only ``False`` is accepted (matching pandas 3.0, where
            ``include_groups=True`` raises a ``ValueError``).
        kwargs : dict
            Optional keyword arguments to pass to the function.
            Currently not supported

        Examples
        --------
        .. code-block:: python

          from cudf import DataFrame
          df = DataFrame()
          df['key'] = [0, 0, 1, 1, 2, 2, 2]
          df['val'] = [0, 1, 2, 3, 4, 5, 6]
          groups = df.groupby(['key'])

          # Define a function to apply to each row in a group
          def mult(df):
            df['out'] = df['key'] * df['val']
            return df

          result = groups.apply(mult)
          print(result)

        Output:

        .. code-block:: python

             key  val  out
          0    0    0    0
          1    0    1    0
          2    1    2    2
          3    1    3    3
          4    2    4    8
          5    2    5   10
          6    2    6   12

        .. pandas-compat::
            :meth:`pandas.core.groupby.DataFrameGroupBy.apply`,
             :meth:`pandas.core.groupby.SeriesGroupBy.apply`

            cuDF's ``groupby.apply`` is limited compared to pandas.
            In some situations, Pandas returns the grouped keys as part of
            the index while cudf does not due to redundancy. For example:

            .. code-block::

                >>> import pandas as pd
                >>> df = pd.DataFrame({
                ...     'a': [1, 1, 2, 2],
                ...     'b': [1, 2, 1, 2],
                ...     'c': [1, 2, 3, 4],
                ... })
                >>> gdf = cudf.from_pandas(df)
                >>> df.groupby('a')[["b", "c"]].apply(lambda x: x.iloc[[0]])
                     b  c
                a
                1 0  1  1
                2 2  1  3
                >>> gdf.groupby('a')[["b", "c"]].apply(lambda x: x.iloc[[0]])
                   b  c
                0  1  1
                2  1  3

        ``engine='jit'`` may be used to accelerate certain functions,
        initially those that contain reductions and arithmetic operations
        between results of those reductions:

        >>> import cudf
        >>> df = cudf.DataFrame({'a':[1,1,2,2,3,3], 'b':[1,2,3,4,5,6]})
        >>> df.groupby('a').apply(
        ...   lambda group: group['b'].max() - group['b'].min(),
        ...   engine='jit'
        ... )
        a
        1    1
        2    1
        3    1
        dtype: int64

        """
        if include_groups:
            # matches pandas 3.0
            raise ValueError("include_groups=True is no longer allowed.")
        if kwargs:
            raise NotImplementedError(
                "Passing kwargs to func is currently not supported."
            )
        if self.obj.empty:
            from cudf.core.series import Series

            if func in {"count", "size", "idxmin", "idxmax"}:
                res = Series([], dtype=np.dtype(np.int64))
            else:
                res = self.obj.copy(deep=True)
            res.index = self.grouping.keys
            if func in {"sum", "product"}:
                # For `sum` & `product`, boolean types
                # will need to result in `int64` type.
                for name, col in res._column_labels_and_values:
                    if col.dtype.kind == "b":
                        res._data[name] = col.astype(np.dtype(np.int64))
            return res

        if not callable(func):
            raise TypeError(f"type {type(func)} is not callable")
        group_names, offsets, group_keys, grouped_values = self._grouped(
            include_groups=include_groups
        )

        if not self._sort and len(offsets) > 2:
            # libcudf returns groups sorted by key, but with ``sort=False``
            # pandas processes groups in order of first appearance. Permute
            # the grouped layout accordingly so both engines and the result
            # assembly see pandas' iteration order.
            pos_offsets, _, (positions,) = self._groups(
                [self._range_column_from_obj]
            )
            first_pos = positions.take(as_column(pos_offsets[:-1]))
            group_order = first_pos.argsort().to_numpy()
            sizes = np.diff(np.asarray(offsets, dtype=SIZE_TYPE_DTYPE))
            row_order = as_column(
                np.concatenate(
                    [
                        np.arange(
                            offsets[i], offsets[i + 1], dtype=SIZE_TYPE_DTYPE
                        )
                        for i in group_order
                    ]
                )
            )
            group_names = group_names.take(group_order)
            group_keys = group_keys.take(row_order)
            grouped_values = grouped_values.take(row_order)
            new_offsets = np.zeros(len(sizes) + 1, dtype=SIZE_TYPE_DTYPE)
            np.cumsum(sizes[group_order], out=new_offsets[1:])
            offsets = new_offsets.tolist()

        if engine == "auto":
            if _can_be_jitted(grouped_values, func, args):
                engine = "jit"
            else:
                engine = "cudf"
        if engine == "jit":
            result = self._jit_groupby_apply(
                func,
                group_names,
                offsets,
                group_keys,
                grouped_values,
                *args,
            )
        elif engine == "cudf":
            result = self._iterative_groupby_apply(
                func,
                group_names,
                offsets,
                group_keys,
                grouped_values,
                *args,
            )
        else:
            raise ValueError(f"Unsupported engine '{engine}'")

        # No final sort: group-keyed results are already produced in
        # sorted group-key order, and pandas preserves the UDF's
        # within-group row order (and a transform's original row order)
        # regardless of ``sort`` (pandas GH52444).
        if self._as_index is False:
            result = result.reset_index()
        return result

    @_performance_tracking
    def _broadcast(self, values: Series) -> Series:
        """
        Broadcast the results of an aggregation to the group

        Parameters
        ----------
        values: Series
            A Series representing the results of an aggregation.  The
            index of the Series must be the (unique) values
            representing the group keys.

        Returns
        -------
        A Series of the same size and with the same index as
        ``self.obj``.
        """
        if not values.index.equals(self.grouping.keys):
            values = values._align_to_index(
                self.grouping.keys, how="right", allow_non_unique=True
            )
        # Even when no alignment is needed (every group is a single row,
        # so the aggregated index already equals the group keys), the
        # result must be indexed like the input rows, not the group
        # labels (pandas GH#9941: transform returns an obj-indexed
        # result).
        values.index = self.obj.index
        return values

    @_performance_tracking
    def transform(
        self, func, *args, engine=None, engine_kwargs=None, **kwargs
    ):
        """Apply an aggregation, then broadcast the result to the group size.

        Parameters
        ----------
        func: str or callable
            Aggregation to apply to each group. Note that the set of
            operations currently supported by `transform` is identical
            to that supported by the `agg` method.

        Returns
        -------
        A Series or DataFrame of the same size as the input, with the
        result of the aggregation per group broadcasted to the group
        size.

        Examples
        --------
        .. code-block:: python

          import cudf
          df = DataFrame({'a': [2, 1, 1, 2, 2], 'b': [1, 2, 3, 4, 5]})
          df.groupby('a').transform('max')
             b
          0  5
          1  3
          2  3
          3  5
          4  5

        See Also
        --------
        agg
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if args:
            raise NotImplementedError(
                "Passing args to func is currently not supported."
            )
        if kwargs:
            raise NotImplementedError(
                "Passing kwargs to func is currently not supported."
            )

        if not (isinstance(func, str) or callable(func)):
            raise TypeError(
                "Aggregation must be a named aggregation or a callable"
            )
        gb = self
        if not self._as_index:
            # as_index has no effect on transform in pandas (GH#49834):
            # the key-column reset that agg/size apply for as_index=False
            # must not leak into the broadcast result.
            gb = copy.copy(self)
            gb._as_index = True
        if func == "size":
            # size counts group rows rather than aggregating each value
            # column, so pandas broadcasts GroupBy.size() as a single
            # Series (unnamed for DataFrameGroupBy, keeping the source
            # name for SeriesGroupBy) instead of going per-column.
            return gb._broadcast(gb.size())
        if func == "cumcount":
            # cumcount numbers the rows of each group: always an unnamed
            # Series over the original index, never a per-column result.
            return gb.cumcount()
        if func == "ngroup":
            return gb.ngroup()
        try:
            result = gb.agg(func)
        except TypeError as e:
            raise NotImplementedError(
                "Currently, `transform()` supports only aggregations."
            ) from e
        # If the aggregation is a scan, don't broadcast
        if _is_all_scan_aggregate([[func]]):
            if len(result) != len(self.obj):
                raise AssertionError(
                    "Unexpected result length for scan transform"
                )
            return result
        return gb._broadcast(result)

    def rolling(self, *args, **kwargs):
        """
        Returns a `RollingGroupby` object that enables rolling window
        calculations on the groups.

        See Also
        --------
        cudf.core.window.rolling.RollingGroupby
        """
        from cudf.core.window.rolling import RollingGroupby

        return RollingGroupby(self, *args, **kwargs)

    @_performance_tracking
    def count(self, dropna=True):
        """Compute the number of values in each column.

        Parameters
        ----------
        dropna : bool
            If ``True``, don't include null values in the count.
        """

        def func(x):
            return getattr(x, "count")(dropna=dropna)

        return self.agg(func)

    @_performance_tracking
    def describe(self, percentiles=None, include=None, exclude=None):
        """
        Generate descriptive statistics that summarizes the central tendency,
        dispersion and shape of a dataset's distribution, excluding NaN values.

        Analyzes numeric DataFrames only

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output.
            Currently not supported.

        include: 'all', list-like of dtypes or None (default), optional
            list of data types to include in the result.
            Ignored for Series.

        exclude: list-like of dtypes or None (default), optional,
            list of data types to omit from the result.
            Ignored for Series.

        Returns
        -------
        Series or DataFrame
            Summary statistics of the Dataframe provided.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({
        ...     "Speed": [380.0, 370.0, 24.0, 26.0],
        ...      "Score": [50, 30, 90, 80],
        ... })
        >>> gdf
           Speed  Score
        0  380.0     50
        1  370.0     30
        2   24.0     90
        3   26.0     80
        >>> gdf.groupby('Score').describe()
              Speed
              count   mean  std    min    25%    50%    75%    max
        Score
        30        1  370.0  NaN  370.0  370.0  370.0  370.0  370.0
        50        1  380.0  NaN  380.0  380.0  380.0  380.0  380.0
        80        1   26.0  NaN   26.0   26.0   26.0   26.0   26.0
        90        1   24.0  NaN   24.0   24.0   24.0   24.0   24.0

        """
        if percentiles is not None:
            raise NotImplementedError("percentiles is currently not supported")
        if exclude is not None:
            raise NotImplementedError("exclude is currently not supported")
        if include is not None:
            raise NotImplementedError("include is currently not supported")

        res = self.agg(
            [
                "count",
                "mean",
                "std",
                "min",
                _quantile_25,
                _quantile_50,
                _quantile_75,
                "max",
            ]
        )
        res.rename(
            columns={
                "_quantile_25": "25%",
                "_quantile_50": "50%",
                "_quantile_75": "75%",
            },
            level=1,
            inplace=True,
        )
        return res

    @_performance_tracking
    def cov(self, min_periods=0, ddof=1, numeric_only: bool = False):
        """
        Compute the pairwise covariance among the columns of a DataFrame,
        excluding NA/null values.

        The returned DataFrame is the covariance matrix of the columns of
        the DataFrame.

        Both NA and null values are automatically excluded from the
        calculation. See the note below about bias from missing values.

        A threshold can be set for the minimum number of observations
        for each value created. Comparisons with observations below this
        threshold will be returned as `NA`.

        This method is generally used for the analysis of time series data to
        understand the relationship between different measures across time.

        Parameters
        ----------
        min_periods: int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.

        ddof: int, optional
            Delta degrees of freedom, default is 1.

        Returns
        -------
        DataFrame
            Covariance matrix.

        Notes
        -----
        Returns the covariance matrix of the DataFrame's time series.
        The covariance is normalized by N-ddof.

        For DataFrames that have Series that are missing data
        (assuming that data is missing at random) the returned covariance
        matrix will be an unbiased estimate of the variance and covariance
        between the member Series.

        However, for many applications this estimate may not be acceptable
        because the estimate covariance matrix is not guaranteed to be
        positive semi-definite. This could lead to estimate correlations
        having absolute values which are greater than one, and/or a
        non-invertible covariance matrix. See
        `Estimation of covariance matrices
        <https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices>`
        for more details.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({
        ...     "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        ...     "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
        ...     "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        ...     "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        ... })
        >>> gdf
          id  val1  val2  val3
        0  a     5     4     4
        1  a     4     5     5
        2  a     6     6     6
        3  b     4     1     1
        4  b     8     2     2
        5  b     7     9     9
        6  c     4     8     8
        7  c     5     5     5
        8  c     2     1     1
        >>> gdf.groupby("id").cov()
                    val1       val2       val3
        id
        a  val1  1.000000   0.500000   0.500000
           val2  0.500000   1.000000   1.000000
           val3  0.500000   1.000000   1.000000
        b  val1  4.333333   3.500000   3.500000
           val2  3.500000  19.000000  19.000000
           val3  3.500000  19.000000  19.000000
        c  val1  2.333333   3.833333   3.833333
           val2  3.833333  12.333333  12.333333
           val3  3.833333  12.333333  12.333333
        """
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        return self._cov_or_corr(
            lambda x: x.cov(min_periods, ddof), "Covariance"
        )

    def _cov_or_corr(self, func, method_name):
        """
        Internal function that is called by either corr() or cov()
        for sort groupby correlation and covariance computations,
        respectively.
        """
        # create expanded dataframe consisting all combinations of the
        # struct columns-pairs to be used in the correlation or covariance
        # i.e. (('col1', 'col1'), ('col1', 'col2'), ('col2', 'col2'))
        column_names = self.grouping._values_column_names
        num_cols = len(column_names)

        column_pair_structs = {}
        for x, y in itertools.combinations_with_replacement(column_names, 2):
            # The number of output columns is the number of input columns
            # squared. We directly call the struct column factory here to
            # reduce overhead and avoid copying data. Since libcudf groupby
            # maintains a cache of aggregation requests, reusing the same
            # column also makes use of previously cached column means and
            # reduces kernel costs.

            # checks if input column names are string, raise a warning if
            # not so and cast them to strings
            if not (isinstance(x, str) and isinstance(y, str)):
                warnings.warn(
                    "DataFrame contains non-string column name(s). "
                    "Struct columns require field names to be strings. "
                    "Non-string column names will be cast to strings "
                    "in the result's field names."
                )
                x, y = str(x), str(y)

            plc_column = plc.Column.struct_from_children(
                [
                    self.obj._data[x].plc_column,
                    self.obj._data[y].plc_column,
                ]
            )
            struct_column = ColumnBase.create(
                plc_column, dtype=dtype_from_pylibcudf_column(plc_column)
            ).set_mask(None, 0)
            column_pair_structs[(x, y)] = struct_column

        from cudf.core.dataframe import DataFrame

        column_pair_groupby = DataFrame._from_data(
            column_pair_structs
        ).groupby(by=self.grouping)

        try:
            gb_cov_corr = column_pair_groupby.agg(func)
        except RuntimeError as e:
            if "Unsupported groupby reduction type-agg combination" in str(e):
                raise TypeError(
                    f"{method_name} accepts only numerical column-pairs"
                )
            raise

        # ensure that column-pair labels are arranged in ascending order
        cols_list = [
            (y, x) if i > j else (x, y)
            for j, y in enumerate(column_names)
            for i, x in enumerate(column_names)
        ]
        cols_split = [
            cols_list[i : i + num_cols]
            for i in range(0, len(cols_list), num_cols)
        ]

        # interleave: combines the correlation or covariance results for each
        # column-pair into a single column

        def interleave_columns(source_columns: list[ColumnBase]) -> ColumnBase:
            # Note: assume non-empty
            result_type = source_columns[0].dtype
            with access_columns(
                *source_columns, mode="read", scope="internal"
            ) as accessed_source_columns:
                return ColumnBase.create(
                    plc.reshape.interleave_columns(
                        plc.Table(
                            [c.plc_column for c in accessed_source_columns]
                        )
                    ),
                    result_type,
                )

        res = DataFrame._from_data(
            {
                x: interleave_columns([gb_cov_corr._data[y] for y in ys])
                for ys, x in zip(cols_split, column_names, strict=True)
            }
        )

        # create a multiindex for the groupby covariance or correlation
        # dataframe, to match pandas behavior
        unsorted_idx = gb_cov_corr.index.repeat(num_cols)
        idx_sort_order = unsorted_idx._get_sorted_inds()
        sorted_idx = unsorted_idx._gather(idx_sort_order)
        if len(gb_cov_corr):
            # TO-DO: Should the operation below be done on the CPU instead?
            sorted_idx._data[None] = as_column(
                np.tile(column_names, len(gb_cov_corr.index))
            )
        res.index = MultiIndex._from_data(sorted_idx._data)

        return res

    @_performance_tracking
    def var(
        self,
        ddof=1,
        engine=None,
        engine_kwargs=None,
        numeric_only: bool = False,
    ):
        """Compute the column-wise variance of the values in each group.

        Parameters
        ----------
        ddof : int
            The delta degrees of freedom. N - ddof is the divisor used to
            normalize the variance.
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        def func(x):
            return getattr(x, "var")(ddof=ddof)

        return self.agg(func)

    @_performance_tracking
    def nunique(self, dropna: bool = True):
        """
        Return number of unique elements in the group.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.
        """

        def func(x):
            return getattr(x, "nunique")(dropna=dropna)

        return self.agg(func)

    @_performance_tracking
    def std(
        self,
        ddof=1,
        engine=None,
        engine_kwargs=None,
        numeric_only: bool = False,
    ):
        """Compute the column-wise std of the values in each group.

        Parameters
        ----------
        ddof : int
            The delta degrees of freedom. N - ddof is the divisor used to
            normalize the standard deviation.
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        def func(x):
            return getattr(x, "std")(ddof=ddof)

        return self.agg(func)

    @_performance_tracking
    def quantile(
        self, q=0.5, interpolation="linear", numeric_only: bool = False
    ):
        """Compute the column-wise quantiles of the values in each group.

        Parameters
        ----------
        q : float or array-like
            The quantiles to compute.
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}
            The interpolation method to use when the desired quantile lies
            between two data points. Defaults to "linear".
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
            Currently not supported
        """
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is not currently supported."
            )

        if is_list_like(q):
            return self._quantile_array(list(q), interpolation=interpolation)

        def func(x):
            return getattr(x, "quantile")(q=q, interpolation=interpolation)

        result = self.agg(func)
        return self._align_quantile_dtypes(result, interpolation)

    @staticmethod
    def _quantile_result_dtype(
        orig_dtype: DtypeObj | None, exact: bool
    ) -> DtypeObj | None:
        """Return the dtype pandas produces for a quantile of ``orig_dtype``.

        libcudf's quantile aggregation always yields ``float64``. pandas,
        however, chooses the result dtype from the input dtype and the
        interpolation method (``exact`` is ``True`` for ``lower``/``higher``/
        ``nearest``, where the quantile is an actual element of the input):

        * numpy / pyarrow integer columns -> ``int64`` when ``exact`` else
          ``float64``; numpy / pyarrow floating columns -> always ``float64``.
        * pandas nullable (masked) integer columns -> keep their dtype when
          ``exact`` else ``Float64``; nullable floating columns -> always keep
          their dtype (e.g. ``Float32`` stays ``Float32``).

        Returns ``None`` to leave the (float64) result untouched.
        """
        if orig_dtype is None:
            return None
        if is_pandas_nullable_numpy_dtype(orig_dtype) and not isinstance(
            orig_dtype, pd.ArrowDtype
        ):
            # pandas nullable (masked) extension dtype.
            if orig_dtype.kind == "f":
                return orig_dtype
            if orig_dtype.kind in "iu":
                return orig_dtype if exact else pd.Float64Dtype()
            return None
        # numpy and pyarrow-backed columns do not preserve their dtype; the
        # result is always numpy-backed.
        if orig_dtype.kind == "f":
            return np.dtype(np.float64)
        if orig_dtype.kind in "iu":
            return np.dtype(np.int64) if exact else np.dtype(np.float64)
        return None

    def _align_quantile_dtypes(
        self, result: DataFrameOrSeries, interpolation: str
    ) -> DataFrameOrSeries:
        """Cast quantile result columns to the dtype pandas would produce."""
        exact = interpolation in {"lower", "higher", "nearest"}
        orig_dtypes = dict(self.grouping.values._dtypes)
        for name, col in list(result._data.items()):
            target = self._quantile_result_dtype(orig_dtypes.get(name), exact)
            if target is not None and target != col.dtype:
                result._data[name] = col.astype(target)
        return result

    def _quantile_array(self, qs, interpolation="linear"):
        """Compute multiple quantiles and return result with proper
        MultiIndex including quantile values as the innermost level.
        """
        # Compute each quantile separately and collect results
        results = [self.quantile(qi, interpolation=interpolation) for qi in qs]
        nqs = len(qs)
        first = results[0]
        idx = first.index
        ngroups = len(idx)

        # Concatenate results (order: all groups for q0, then q1, ...)
        combined = concat(results, ignore_index=True)

        # Reorder to interleave: group0-q0, group0-q1, group1-q0, group1-q1
        order = (
            np.arange(ngroups * nqs)
            .reshape(ngroups, nqs, order="F")
            .reshape(-1)
        )

        combined = combined.iloc[order]

        # Build new MultiIndex with quantile as innermost level
        q_level = Index(qs, dtype=np.float64)

        if isinstance(idx, MultiIndex):
            levels = [*list(idx.levels), q_level]
            new_codes = [cp.repeat(code.values, nqs) for code in idx._codes]
            new_codes.append(cp.tile(cp.arange(nqs), ngroups))

            new_index = MultiIndex(
                levels=levels,
                codes=new_codes,
                names=[*list(idx.names), None],
            )
        else:
            new_index = MultiIndex(
                levels=[idx, q_level],
                codes=[
                    cp.repeat(cp.arange(ngroups, dtype=np.int64), nqs),
                    cp.tile(cp.arange(nqs, dtype=np.int64), ngroups),
                ],
                names=[idx.name, None],
            )

        combined.index = new_index

        # If operating on a SeriesGroupBy, ``combined`` is already a Series;
        # a DataFrameGroupBy yields a single-/multi-column DataFrame.
        return combined

    @_performance_tracking
    def unique(self):
        """Get a list of the unique values for each column in each group."""
        return self.agg("unique")

    @_performance_tracking
    def diff(self, periods=1, axis=0):
        """Get the difference between the values in each group.

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
        Series or DataFrame
            First differences of the Series or DataFrame.
        """

        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")
        values = self.grouping.values
        values.index = self.obj.index
        return values - self.shift(periods=periods)

    def _scan_fill(
        self, method: plc.replace.ReplacePolicy, limit: int | None
    ) -> DataFrameOrSeries:
        """Internal implementation for `ffill` and `bfill`"""
        values = self.grouping.values
        from cudf.core.dataframe import DataFrame

        value_columns = values._columns
        if limit is not None and limit < 0:
            # pandas treats a negative limit as unlimited
            limit = None
        if limit is None:
            replaced = tuple(self._replace_nulls(value_columns, method))
        else:
            # pandas accepts integer-valued floats
            limit = int(limit)
            # Group-relative row position, masked null exactly where the
            # value is null: group-filling the positions with the same
            # policy yields, per row, the position of the value that
            # sourced its fill, making (own position - source position)
            # the fill distance. The unmasked positions column is passed
            # through the same replace_nulls call because the output rows
            # come back in grouped order, not the original row order.
            cum = self.cumcount()._column
            pos_columns = tuple(
                cum.set_mask(col.mask, col.null_count) if col.nullable else cum
                for col in value_columns
            )
            n = len(value_columns)
            filled = tuple(
                self._replace_nulls(
                    (*value_columns, *pos_columns, cum), method
                )
            )
            grouped_cum = filled[-1]
            limited = []
            for fcol, fpos in zip(filled[:n], filled[n : 2 * n], strict=True):
                if method == plc.replace.ReplacePolicy.PRECEDING:
                    dist = grouped_cum - fpos
                else:
                    dist = fpos - grouped_cum
                # Rows within limit of their fill source stay valid
                # (originally-valid rows are their own source, distance
                # 0); nulls no fill reached have a null distance and stay
                # null, so ``keep`` alone is a valid final null mask.
                keep = (dist <= limit).fillna(False)
                limited.append(fcol.set_mask(*keep.as_mask()))
            replaced = tuple(limited)

        result = self.obj._from_data(
            dict(zip(values._column_names, replaced, strict=True))
        )
        # Pandas' groupby.ffill/bfill builds the result columns via a ``take``
        # on the input columns, which converts integer-valued column labels
        # to object dtype. Reproduce that here so column metadata matches.
        if (
            isinstance(result, DataFrame)
            and isinstance(self.obj, DataFrame)
            and result._num_columns < self.obj._num_columns
        ):
            source_pd_cols = self.obj._data.to_pandas_index
            if (
                source_pd_cols.dtype.kind in {"i", "u"}
                or source_pd_cols.dtype == object
            ):
                indexer = source_pd_cols.get_indexer(result._column_names)
                if not (indexer == -1).any():
                    taken = source_pd_cols.take(indexer)
                    if (
                        not isinstance(taken, pd.MultiIndex)
                        and taken.dtype != object
                    ):
                        taken = taken.astype(object)
                    result.columns = taken
        return self._mimic_pandas_order(result)

    def ffill(self, limit: int | None = None):
        """Forward fill NA values.

        Parameters
        ----------
        limit : int, default None
            The maximum number of consecutive NA values within a group
            filled forward from the most recent valid value. ``None``
            fills without limit.
        """
        return self._scan_fill(plc.replace.ReplacePolicy.PRECEDING, limit)

    def bfill(self, limit: int | None = None):
        """Backward fill NA values.

        Parameters
        ----------
        limit : int, default None
            The maximum number of consecutive NA values within a group
            filled backward from the next valid value. ``None`` fills
            without limit.
        """
        return self._scan_fill(plc.replace.ReplacePolicy.FOLLOWING, limit)

    @_performance_tracking
    def shift(
        self,
        periods: int = 1,
        freq=None,
        axis=0,
        fill_value=None,
        suffix: str | None = None,
    ):
        """
        Shift each group by ``periods`` positions.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift.
        freq : str, unsupported
        axis : 0, axis to shift
            Shift direction. Only row-wise shift is supported
        fill_value : scalar or list of scalars, optional
            The scalar value to use for newly introduced missing values. Can be
            specified with `None`, a single value or multiple values:

            - `None` (default): sets all indeterminable values to null.
            - Single value: fill all shifted columns with this value. Should
              match the data type of all columns.
            - List of values: fill shifted columns with corresponding value in
              the list. The length of the list should match the number of
              columns shifted. Each value should match the data type of the
              column to fill.
        suffix : str, optional
            A string to add to each shifted column if there are multiple periods.
            Ignored otherwise.
            Currently not supported.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        .. pandas-compat::
            :meth:`pandas.core.groupby.DataFrameGroupBy.shift`,
             :meth:`pandas.core.groupby.SeriesGroupBy.shift`

            Parameter ``freq`` is unsupported.
        """

        if freq is not None:
            raise NotImplementedError("Parameter freq is unsupported.")

        if axis != 0:
            raise NotImplementedError("Only axis=0 is supported.")

        if suffix is not None:
            raise NotImplementedError("shift is not currently supported.")

        values = self.grouping.values
        if is_list_like(fill_value):
            if len(fill_value) != values._num_columns:
                raise ValueError(
                    "Mismatched number of columns and values to fill."
                )
        else:
            fill_value = [fill_value] * values._num_columns

        result = self.obj.__class__._from_data(
            dict(
                zip(
                    values._column_names,
                    self._shift(values._columns, periods, fill_value),
                    strict=True,
                )
            )
        )
        return self._mimic_pandas_order(result)

    @_performance_tracking
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq=None,
    ):
        """
        Calculates the percent change between sequential elements
        in the group.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : None
            Must be None.
        freq : str, optional
            Increment to use from time series API.
            Not yet implemented.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group
        """
        if freq is not None:
            raise NotImplementedError("freq parameter not supported yet.")
        if fill_method is not None:
            raise ValueError(f"fill_method must be None; got {fill_method=}.")

        # pandas 3.0 removed fill_method: no filling is performed, so NaN
        # appears wherever the value or the group-shifted value is NA.
        values = self.grouping.values
        values.index = self.obj.index
        value_grp = values.groupby(
            self.grouping, sort=self._sort, dropna=self._dropna
        )
        shifted = value_grp.shift(periods=periods, freq=freq)
        return (values / shifted) - 1

    def _mimic_pandas_order(
        self, result: DataFrameOrSeries
    ) -> DataFrameOrSeries:
        """Given a groupby result from libcudf, reconstruct the row orders
        matching that of pandas. This also adds appropriate indices.
        """
        # TODO: copy metadata after this method is a common pattern, should
        # merge in this method.

        # This function is used to reorder the results of scan-based
        # groupbys which have the same output size as input size.
        # However, if the grouping key has NAs and dropna=True, the
        # result coming back from libcudf has null_count few rows than
        # the input, so we must produce an ordering from the full
        # input range.
        _, _, (ordering,) = self._groups([self._range_column_from_obj])
        if self._dropna and any(
            c.has_nulls(include_nan=True) > 0
            for c in self.grouping._key_columns
        ):
            # Scan aggregations with null/nan keys put nulls in the
            # corresponding output rows in pandas, to do that here
            # expand the result by reindexing.
            ri = RangeIndex(0, len(self.obj))
            result.index = Index._from_column(ordering)
            # This reorders and expands
            result = result.reindex(ri)
        else:
            # Just reorder according to the groupings
            result = result.take(ordering.argsort())
        # Now produce the actual index we first thought of
        result.index = self.obj.index
        return result

    def ohlc(self):
        """
        Compute open, high, low and close values of a group, excluding missing values.

        Currently not implemented.
        """
        raise NotImplementedError("ohlc is currently not implemented")

    @property
    def plot(self):
        """
        Make plots of a grouped Series or DataFrame.

        Currently not implemented.
        """
        raise NotImplementedError("plot is currently not implemented")

    def resample(self, rule, *args, include_groups: bool = True, **kwargs):
        """
        Provide resampling when using a TimeGrouper.

        Currently not implemented.
        """
        raise NotImplementedError("resample is currently not implemented")

    def take(self, indices):
        """
        Return the elements in the given *positional* indices in each group.

        Currently not implemented.
        """
        raise NotImplementedError("take is currently not implemented")

    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Filter elements from groups that don't satisfy a criterion.

        Currently not implemented.
        """
        raise NotImplementedError("filter is currently not implemented")

    def expanding(self, *args, **kwargs):
        """
        Return an expanding grouper, providing expanding
        functionality per group.

        Currently not implemented.
        """
        raise NotImplementedError("expanding is currently not implemented")

    def ewm(self, *args, **kwargs):
        """
        Return an ewm grouper, providing ewm functionality per group.

        Currently not implemented.
        """
        raise NotImplementedError("expanding is currently not implemented")

    def any(self, skipna: bool = True, min_count: int = 0, **kwargs: Any):
        """
        Return True if any value in the group is truthful, else False.
        """
        return self._bool_reduce("any", skipna=skipna, min_count=min_count)

    def all(self, skipna: bool = True, min_count: int = 0, **kwargs: Any):
        """
        Return True if all values in the group are truthful, else False.
        """
        return self._bool_reduce("all", skipna=skipna, min_count=min_count)

    def _bool_reduce(self, op: str, *, skipna: bool, min_count: int):
        """Implement all/any as min/max on bool-coerced value columns."""
        from cudf.core.dataframe import DataFrame
        from cudf.core.series import Series

        agg_name = {"all": "min", "any": "max"}[op]
        # Empty-group fill value: vacuously True for all, vacuously False for any
        fill_value = op == "all"

        is_series = isinstance(self.obj, Series)

        # Coerce each value column to a (nullable) bool column so that
        # nulls are preserved through the aggregation (min/max skip
        # nulls). For ``skipna=False``, nulls are replaced with True so
        # they don't flip ``all`` to False and always make ``any`` True.
        bool_dtype = np.dtype(np.bool_)

        def _to_bool_col(col):
            if is_dtype_obj_string(col.dtype):
                bool_col = col.count_characters() > np.int8(0)
            else:
                # For numeric/bool inputs, cast to bool preserving nulls.
                bool_col = col != 0
            if col.has_nulls() and bool_col.null_count != col.null_count:
                # ``na_value=np.nan`` dtypes don't propagate missingness
                # through the comparison above (NaN compares as False), so
                # restore the source column's null positions.
                bool_col = bool_col.set_mask(col.mask, col.null_count)
            if not skipna:
                # NA values must not flip ``all`` to False nor stop ``any``
                # from being True, so treat them as True.
                bool_col = bool_col.fillna(True)
            # Normalize away pandas-extension bool dtypes so the downstream
            # aggregation sees ``np.bool_``, but only when no nulls remain:
            # casting a null-containing extension dtype to numpy bool is
            # (intentionally) rejected in pandas-compatible mode. A nullable
            # bool column aggregates correctly as-is, and the result is
            # normalized to ``np.bool_`` after empty/skipna groups are
            # filled below.
            if not bool_col.has_nulls():
                bool_col = bool_col.astype(bool_dtype, copy=False)
            return bool_col

        if is_series:
            new_obj = Series._from_column(
                _to_bool_col(self.obj._column), name=self.obj.name
            )
        else:
            new_data = {
                col_name: _to_bool_col(self.obj._data[col_name])
                for col_name in self.grouping._values_column_names
            }
            new_obj = DataFrame._from_data(new_data, index=self.obj.index)

        # Reuse the same grouping so key columns match ``new_obj`` exactly,
        # avoiding label-based lookup when the key column was excluded.
        bool_gb = type(self)(
            new_obj,
            by=self.grouping,
            level=None,
            sort=self._sort,
            as_index=self._as_index,
            dropna=self._dropna,
        )
        result = bool_gb.agg(agg_name)

        def _bool_result_dtype(input_dtype):
            # Mirror pandas' any/all output dtype to the input's "flavor":
            # masked nullable -> ``boolean``, pyarrow -> ``bool[pyarrow]``,
            # numpy/string -> numpy ``bool``.
            if isinstance(input_dtype, pd.ArrowDtype):
                return pd.ArrowDtype(pa.bool_())
            if is_pandas_nullable_extension_dtype(
                input_dtype
            ) and not is_dtype_obj_string(input_dtype):
                return pd.BooleanDtype()
            return np.dtype(np.bool_)

        # Empty groups (skipna=True with all-NA values) yield NA from
        # min/max — pandas treats these as ``True`` for ``all`` and
        # ``False`` for ``any``.
        if isinstance(result, Series):
            result = result.fillna(fill_value).astype(
                _bool_result_dtype(self.obj.dtype)
            )
        else:
            # With ``as_index=False`` the group-key columns are present in the
            # result; only the aggregated value columns must be coerced to
            # bool (casting a key column would corrupt it, e.g. a categorical
            # key turning into ``[False, True]``).
            key_names = set(self.grouping.names)
            for col_name in result._column_names:
                if col_name in key_names:
                    continue
                target = _bool_result_dtype(self.obj._data[col_name].dtype)
                result[col_name] = (
                    result[col_name].fillna(fill_value).astype(target)
                )

        if min_count and min_count > 0:
            counts = self.agg("count")
            result = result.where(counts >= min_count, None)
        return result


class DataFrameGroupBy(GroupBy, GetAttrGetItemMixin):
    obj: DataFrame

    _PROTECTED_KEYS = frozenset(("obj",))

    def _reduce_numeric_only(self, op: str):
        columns = list(
            name
            for name, dtype in self.obj._dtypes
            if (
                is_dtype_obj_numeric(dtype) and name not in self.grouping.names
            )
        )
        return self[columns].agg(op)

    def __getitem__(self, key):
        new = self.obj[key].groupby(
            by=self.grouping.keys,
            dropna=self._dropna,
            sort=self._sort,
            group_keys=self._group_keys,
            as_index=self._as_index,
        )
        # Track explicit column selection so include_groups=False does not
        # strip columns the user explicitly asked for (matches pandas
        # behavior of returning group-key columns when reselected).
        new._selection = (
            tuple(key) if isinstance(key, (list, tuple)) else (key,)
        )
        return new

    def idxmin(
        self,
        skipna: bool = True,
        min_count: int = 0,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        if not skipna:
            # pandas does not support positional idxmin with skipna=False
            # (it cannot represent "the label of a NA").
            raise ValueError("idxmin with skipna=False")
        return self._reduce("idxmin", numeric_only=numeric_only)

    def idxmax(
        self,
        skipna: bool = True,
        min_count: int = 0,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        if not skipna:
            raise ValueError("idxmax with skipna=False")
        return self._reduce("idxmax", numeric_only=numeric_only)

    def value_counts(
        self,
        subset=None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame | Series:
        """
        Return a Series or DataFrame containing counts of unique rows.

        Parameters
        ----------
        subset : list-like, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Don't include counts of rows that contain NA values.

        Returns
        -------
        Series or DataFrame
            Series if the groupby as_index is True, otherwise DataFrame.

        See Also
        --------
        SeriesGroupBy.value_counts: Equivalent method on SeriesGroupBy.

        Notes
        -----
        - If the groupby as_index is True then the returned Series will have a
          MultiIndex with one level per input column.
        - If the groupby as_index is False then the returned DataFrame will
          have an additional column with the value_counts. The column is
          labelled 'count' or 'proportion', depending on the ``normalize``
          parameter.

        By default, rows that contain any NA values are omitted from
        the result.

        By default, the result will be in descending order so that the
        first element of each group is the most frequently-occurring row.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({
        ...    'gender': ['male', 'male', 'female', 'male', 'female', 'male'],
        ...    'education': ['low', 'medium', 'high', 'low', 'high', 'low'],
        ...    'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']
        ... })

        >>> df
                gender  education   country
        0       male    low         US
        1       male    medium      FR
        2       female  high        US
        3       male    low         FR
        4       female  high        FR
        5       male    low         FR

        >>> df.groupby('gender').value_counts()
        gender  education  country
        male    low        FR         2
                           US         1
                medium     FR         1
        female  high       US         1
                           FR         1
        Name: count, dtype: int64

        >>> df.groupby('gender').value_counts(ascending=True)
        gender  education  country
        male    low        US         1
                medium     FR         1
        female  high       US         1
                           FR         1
        male    low        FR         2
        Name: count, dtype: int64

        >>> df.groupby('gender').value_counts(normalize=True)
        gender  education  country
        male    low        FR         0.50
                           US         0.25
                medium     FR         0.25
        female  high       US         0.50
                           FR         0.50
        Name: proportion, dtype: float64

        >>> df.groupby('gender', as_index=False).value_counts()
           gender education country  count
        0    male       low      FR      2
        1    male       low      US      1
        2    male    medium      FR      1
        3  female      high      US      1
        4  female      high      FR      1

        >>> df.groupby('gender', as_index=False).value_counts(normalize=True)
           gender education country  proportion
        0    male       low      FR        0.50
        1    male       low      US        0.25
        2    male    medium      FR        0.25
        3  female      high      US        0.50
        4  female      high      FR        0.50
        """

        df = self.obj.copy()
        groupings = self.grouping.names
        name = "proportion" if normalize else "count"

        if subset is None:
            subset = [i for i in df._column_names if i not in groupings]
        # Check subset exists in dataframe
        elif set(subset) - set(df._column_names):
            raise ValueError(
                f"Keys {set(subset) - set(df._column_names)} in subset "
                f"do not exist in the DataFrame."
            )
        # Catch case where groupby and subset share an element
        elif set(subset) & set(groupings):
            raise ValueError(
                f"Keys {set(subset) & set(groupings)} in subset "
                "cannot be in the groupby column keys."
            )

        subset = list(subset)
        keys = list(groupings) + subset

        # Use the grouping's actual key values rather than the raw object
        # columns: a ``Grouper(freq=...)`` bins its key (e.g. floors a datetime
        # to the resampling frequency), so the raw column would give the wrong
        # group labels.
        for kname, kcol in zip(
            self.grouping.names, self.grouping._key_columns, strict=True
        ):
            if kname in df._column_names:
                df[kname] = kcol

        # Bookkeeping columns for counting and for recovering pandas' ordering
        # (see below). Choose names that cannot clash with the user's columns.
        taken = set(df._column_names)

        def _free_name(base: str) -> str:
            while base in taken:
                base = f"_{base}"
            taken.add(base)
            return base

        cnt_col = _free_name("__count")
        pos_col = _free_name("__pos")
        seq_col = _free_name("__seq")

        # cudf's groupby does not preserve first-appearance order, so track the
        # first row index of each unique (key + subset) combination alongside
        # the count and reorder by it below. Group with ``dropna=False`` and
        # filter afterwards so the groupby ``dropna`` (group keys) and the
        # ``value_counts`` ``dropna`` (subset) can be applied independently,
        # matching pandas.
        df[cnt_col] = 1
        df[pos_col] = as_column(range(len(df)))
        result = (
            df.groupby(keys, dropna=False, sort=False)
            .agg({cnt_col: "count", pos_col: "min"})
            .reset_index()
        )
        result[cnt_col] = result[cnt_col].astype(np.dtype(np.int64))

        drop_cols = (list(groupings) if self._dropna else []) + (
            subset if dropna else []
        )
        keep = None
        for col in drop_cols:
            mask = result[col].notna()
            keep = mask if keep is None else (keep & mask)
        if keep is not None:
            result = result[keep]

        # pandas includes unobserved categorical combinations (count 0). cudf's
        # groupby only emits observed combinations, so when a subset column is
        # categorical, expand to the full product of the observed group-key
        # combinations and the subset categories. The base order is the
        # category product when the groupby is sorted, else first appearance
        # with the unobserved combinations placed last.
        cat_subset = [
            s
            for s in subset
            if isinstance(self.obj._data[s].dtype, CategoricalDtype)
        ]
        if cat_subset and len(result):
            # TODO: This conversion to host objects can be avoided once
            # MultiIndex.from_product supports GPU inputs.
            subset_levels = [
                cast("CategoricalDtype", self.obj._data[s].dtype)
                .categories.to_pandas()
                .tolist()
                if s in cat_subset
                else result[s].dropna().unique().to_pandas().tolist()
                for s in subset
            ]
            subset_prod = MultiIndex.from_product(
                subset_levels, names=subset
            ).to_frame(index=False)
            for s in subset:
                subset_prod[s] = subset_prod[s].astype(result[s].dtype)
            group_combos = result[list(groupings)].drop_duplicates()
            cross_col = _free_name("__cross")
            group_combos[cross_col] = 1
            subset_prod[cross_col] = 1
            full = group_combos.merge(subset_prod, on=cross_col).drop(
                columns=[cross_col]
            )
            result = full.merge(result, on=keys, how="left")
            result[cnt_col] = (
                result[cnt_col].fillna(0).astype(np.dtype(np.int64))
            )
            if self._sort:
                result[pos_col] = as_column(range(len(result)))
            else:
                result[pos_col] = (
                    result[pos_col].fillna(len(df) + 1).astype(SIZE_TYPE_DTYPE)
                )

        # Mirror pandas: order by first appearance, then -- as two independent
        # steps -- optionally by value (``sort``) and by group key (the groupby
        # ``sort``). A running sequence keeps each sort stable for ties (cudf's
        # value sort is not guaranteed stable).
        result = result.sort_values(pos_col)
        result[seq_col] = as_column(range(len(result)))
        if sort:
            result = result.sort_values(
                [cnt_col, seq_col], ascending=[ascending, True]
            )
            result[seq_col] = as_column(range(len(result)))
        if self._sort:
            result = result.sort_values([*list(groupings), seq_col])

        if normalize:
            # Divide each count by its group total. ``fillna`` handles
            # non-observed categorical groups (0 / 0).
            group_size = result.groupby(list(groupings), sort=False)[
                cnt_col
            ].transform("sum")
            result[cnt_col] = (result[cnt_col] / group_size).fillna(0.0)

        result = result.set_index(keys)[cnt_col]
        result.name = name

        if not self._as_index:
            if name in keys:
                raise ValueError(
                    f"Column label '{name}' is duplicate of result column"
                )
            result = result.reset_index()
        return result

    @_performance_tracking
    def corr(
        self, method="pearson", min_periods=1, numeric_only: bool = False
    ):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method: {"pearson", "kendall", "spearman"} or callable,
            default "pearson". Currently only the pearson correlation
            coefficient is supported.

        min_periods: int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.

        Returns
        -------
        DataFrame
            Correlation matrix.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({
        ...             "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        ...             "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
        ...             "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        ...             "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1]})
        >>> gdf
           id  val1  val2  val3
        0  a     5     4     4
        1  a     4     5     5
        2  a     6     6     6
        3  b     4     1     1
        4  b     8     2     2
        5  b     7     9     9
        6  c     4     8     8
        7  c     5     5     5
        8  c     2     1     1
        >>> gdf.groupby("id").corr(method="pearson")
                    val1      val2      val3
        id
        a   val1  1.000000  0.500000  0.500000
            val2  0.500000  1.000000  1.000000
            val3  0.500000  1.000000  1.000000
        b   val1  1.000000  0.385727  0.385727
            val2  0.385727  1.000000  1.000000
            val3  0.385727  1.000000  1.000000
        c   val1  1.000000  0.714575  0.714575
            val2  0.714575  1.000000  1.000000
            val3  0.714575  1.000000  1.000000
        """

        if method != "pearson":
            raise NotImplementedError(
                "Only pearson correlation is currently supported"
            )
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        return self._cov_or_corr(
            lambda x: x.corr(method, min_periods), "Correlation"
        )

    def hist(
        self,
        column=None,
        by=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        ax=None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("hist is not currently implemented")

    def boxplot(
        self,
        subplots: bool = True,
        column=None,
        fontsize: int | None = None,
        rot: int = 0,
        grid: bool = True,
        ax=None,
        figsize: tuple[float, float] | None = None,
        layout=None,
        sharex: bool = False,
        sharey: bool = True,
        backend=None,
        **kwargs,
    ):
        raise NotImplementedError("boxplot is not currently implemented")


DataFrameGroupBy.__doc__ = groupby_doc_template.format(ret="")


class SeriesGroupBy(GroupBy):
    obj: Series

    def agg(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        result = super().agg(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

        # downcast the result to a Series:
        if result._num_columns:
            if result.shape[1] == 1 and not is_list_like(func):
                return result.iloc[:, 0]

        # Collapse the column MultiIndex produced by a list aggregation down to
        # the aggregation names. With ``as_index=False`` the group-key columns
        # have already been inserted (as ``(key, "")`` tuples by
        # ``reset_index``); blindly dropping level 0 would replace each key
        # name with the empty padding level, so keep the name for those.
        if result._data.nlevels > 1:
            key_names = set(self.grouping.names)
            result.columns = [
                top if (second == "" and top in key_names) else second
                for top, second in result._data.to_pandas_index
            ]

        return result

    aggregate = agg

    def apply(self, func, *args, **kwargs):
        result = super().apply(func, *args, **kwargs)

        # apply Series name to result
        result.name = self.obj.name

        return result

    def idxmin(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> Series:
        if not skipna:
            raise ValueError("idxmin with skipna=False")
        return self._reduce("idxmin")

    def idxmax(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> Series:
        if not skipna:
            raise ValueError("idxmax with skipna=False")
        return self._reduce("idxmax")

    @property
    def dtype(self) -> pd.Series:
        raise NotImplementedError("dtype is currently not implemented.")

    def hist(
        self,
        by=None,
        ax=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("hist is currently not implemented.")

    @property
    def is_monotonic_increasing(self) -> Series:
        """
        Return whether each group's values are monotonically increasing.

        Currently not implemented
        """
        raise NotImplementedError(
            "is_monotonic_increasing is currently not implemented."
        )

    @property
    def is_monotonic_decreasing(self) -> Series:
        """
        Return whether each group's values are monotonically decreasing.

        Currently not implemented
        """
        raise NotImplementedError(
            "is_monotonic_decreasing is currently not implemented."
        )

    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        """
        Return the largest n elements.

        Currently not implemented
        """
        raise NotImplementedError("nlargest is currently not implemented.")

    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        """
        Return the smallest n elements.

        Currently not implemented
        """
        raise NotImplementedError("nsmallest is currently not implemented.")

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series | DataFrame:
        raise NotImplementedError("value_counts is currently not implemented.")

    def corr(
        self,
        other: Series,
        method: str = "pearson",
        min_periods: int | None = None,
    ) -> Series:
        raise NotImplementedError("corr is currently not implemented.")


SeriesGroupBy.__doc__ = groupby_doc_template.format(ret="")


# TODO: should we define this as a dataclass instead?
class Grouper:
    def __init__(
        self, key=None, level=None, freq=None, closed=None, label=None
    ):
        if key is not None and level is not None:
            raise ValueError("Grouper cannot specify both key and level")
        if (key, level) == (None, None) and not freq:
            raise ValueError("Grouper must specify either key or level")
        self.key = key
        self.level = level
        self.freq = freq
        self.closed = closed
        self.label = label


class _Grouping(Serializable):
    def __init__(
        self,
        obj,
        by=None,
        level=None,
        series_key_column_names=None,
        dropna=True,
    ):
        self._obj = obj
        self._key_columns = []
        self.names = []

        # Need to keep track of named key columns
        # to support `as_index=False` correctly
        self._named_columns = []
        # ``id(series._column)`` -> name of the matching ``obj`` column,
        # for each Series-typed grouping key that is identical (by object
        # identity) to one of ``obj``'s columns. Used by ``_handle_series``
        # to mirror pandas' exclusion of such columns from value columns.
        self._series_key_column_names = dict(series_key_column_names or {})
        self._handle_by_or_level(by, level)

        # pandas treats NaN and null group keys identically, and labels an
        # all-null object key with a float64 NaN. Externally supplied key
        # columns (e.g. a Series or array passed as ``by``) also bypass the
        # ``nans_to_nulls`` conversion applied to ``obj``. Normalize the key
        # columns here so that, e.g., a float key of ``[None, NaN]`` collapses
        # to a single null group and an all-null object key produces a float64
        # NaN group label, matching pandas.
        normalized = []
        for col in self._key_columns:
            if (
                # Only when ``dropna=False`` is the all-null group actually
                # kept and labelled. With ``dropna=True`` the group is dropped,
                # and pandas leaves the (empty) result index as the original
                # object dtype rather than promoting it to float64.
                not dropna
                and isinstance(col.dtype, np.dtype)
                and col.dtype.kind == "O"
                and len(col)
                and col.null_count == len(col)
            ):
                col = column_empty(len(col), np.dtype("float64"))
            normalized.append(col.nans_to_nulls())
        self._key_columns = normalized

        if len(obj) and not len(self._key_columns):
            raise ValueError("No group keys passed")

    def _handle_by_or_level(self, by=None, level=None):
        if level is not None:
            if by is not None:
                raise ValueError("Cannot specify both by and level")
            level_list = level if isinstance(level, list) else [level]
            for level in level_list:
                self._handle_level(level)
        else:
            by_list = by if isinstance(by, list) else [by]
            if not len(self._obj) and not len(by_list):
                # We pretend to groupby an empty column
                by_list = [Index._from_column(column_empty(0))]

            from cudf.core.dataframe import Series

            for by in by_list:
                if callable(by):
                    self._handle_callable(by)
                elif isinstance(by, Series):
                    self._handle_series(by)
                elif isinstance(by, Index):
                    self._handle_index(by)
                elif isinstance(by, Mapping):
                    self._handle_mapping(by)
                elif isinstance(by, Grouper):
                    self._handle_grouper(by)
                elif isinstance(by, pd.Series):
                    self._handle_series(Series(by))
                elif isinstance(by, pd.Index):
                    self._handle_index(Index(by))
                else:
                    try:
                        self._handle_label(by)
                    except (KeyError, TypeError):
                        self._handle_misc(by)

    @functools.cached_property
    def keys(self):
        """Return grouping key columns as index"""
        if len(self._key_columns) > 1:
            return MultiIndex._from_data(
                dict(enumerate(self._key_columns))
            )._set_names(self.names)
        else:
            return Index._from_column(self._key_columns[0], name=self.names[0])

    @property
    def _values_column_names(self) -> list[Hashable]:
        # If the key columns are in `obj`, filter them out
        return [
            x for x in self._obj._column_names if x not in self._named_columns
        ]

    @property
    def values(self) -> DataFrame | Series:
        """Return value columns as a frame.

        Note that in aggregation, value columns can be arbitrarily
        specified. While this method returns all non-key columns from `obj` as
        a frame.

        This is mainly used in transform-like operations.
        """
        value_columns = self._obj._data.select_by_label(
            self._values_column_names
        )
        return self._obj.__class__._from_data(value_columns)

    def _handle_callable(self, by):
        by = by(self._obj.index)
        self.__init__(self._obj, by)

    def _handle_series(self, by):
        # Mirror pandas: if the grouping Series' underlying column was one
        # of the obj's columns (identity captured pre-transformation),
        # exclude that column name from value columns during aggregation.
        # Look up by ``id`` of the original column *before* alignment may
        # produce a fresh column object.
        matched = self._series_key_column_names.get(id(by._column))
        by = by._align_to_index(self._obj.index, how="right")
        self._key_columns.append(by._column)
        self.names.append(by.name)
        if matched is not None:
            self._named_columns.append(matched)

    def _handle_index(self, by):
        self._key_columns.extend(by._columns)
        self.names.extend(by._column_names)

    def _handle_mapping(self, by):
        from cudf.core.series import Series

        by = Series(by.values(), index=by.keys())
        self._handle_series(by)

    def _handle_label(self, by):
        try:
            self._key_columns.append(self._obj._data[by])
        except KeyError as e:
            # `by` can be index name(label) too.
            if by in self._obj.index.names:
                self._key_columns.append(self._obj.index._data[by])
            else:
                raise e
        self.names.append(by)
        self._named_columns.append(by)

    def _handle_grouper(self, by):
        if by.freq:
            self._handle_frequency_grouper(by)
        elif by.key:
            self._handle_label(by.key)
        else:
            self._handle_level(by.level)

    def _handle_frequency_grouper(self, by):
        raise NotImplementedError()

    def _handle_level(self, by):
        level_values = self._obj.index.get_level_values(by)
        self._key_columns.append(level_values._column)
        self.names.append(level_values.name)

    def _handle_misc(self, by):
        by = as_column(by)
        if len(by) != len(self._obj):
            raise ValueError("Grouper and object must have same length")
        self._key_columns.append(by)
        self.names.append(None)

    def serialize(self):
        header = {}
        frames = []
        header["names"] = self.names
        header["_named_columns"] = self._named_columns
        column_header, column_frames = serialize_columns(self._key_columns)
        header["columns"] = column_header
        frames.extend(column_frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        names = header["names"]
        _named_columns = header["_named_columns"]
        key_columns = deserialize_columns(header["columns"], frames)
        out = _Grouping.__new__(_Grouping)
        out.names = names
        out._named_columns = _named_columns
        out._key_columns = key_columns
        return out

    def copy(self, deep=True):
        out = _Grouping.__new__(_Grouping)
        out.names = copy.deepcopy(self.names)
        out._named_columns = copy.deepcopy(self._named_columns)
        out._key_columns = [col.copy(deep=deep) for col in self._key_columns]
        return out


class _FirstLastAggSpec:
    """Callable aggregation spec for groupby ``first``/``last``.

    Lets :meth:`GroupBy._reduce` thread ``skipna`` to
    :meth:`Aggregation.first`/:meth:`Aggregation.last` through
    ``make_aggregation``'s callable path. ``__str__``/``__name__`` report the
    op name so aggregation-validity checks and result-column naming behave
    exactly as they do for the plain ``"first"``/``"last"`` string specs.
    """

    def __init__(self, op: str, skipna: bool) -> None:
        self._op = op
        self._skipna = skipna
        self.__name__ = op

    def __call__(self, agg):
        return getattr(agg, self._op)(skipna=self._skipna)

    def __str__(self) -> str:
        return self._op


def _is_multi_agg(aggs):
    """
    Returns True if more than one aggregation is performed
    on any of the columns as specified in `aggs`.
    """
    if isinstance(aggs, Mapping):
        return any(is_list_like(agg) for agg in aggs.values())
    if is_list_like(aggs):
        return True
    return False
