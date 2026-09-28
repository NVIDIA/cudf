# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools

import numpy as np
import pandas as pd

from dask.dataframe.core import _concat
from dask.dataframe.groupby import Aggregation

from cudf.utils.performance_tracking import _dask_cudf_performance_tracking

from dask_cudf._expr import (
    DecomposableGroupbyAggregation,
    DXGroupBy,
    DXSeriesGroupBy,
    GroupbyAggregation,
    SingleAggregation,
    is_scalar,
    new_collection,
)

##
## Fused groupby aggregations
##

OPTIMIZED_AGGS = (
    "count",
    "mean",
    "std",
    "var",
    "sum",
    "min",
    "max",
    list,
    "first",
    "last",
)


def _make_name(col_name, sep="_"):
    """Combine elements of `col_name` into a single string, or no-op if
    `col_name` is already a string
    """
    if isinstance(col_name, str):
        return col_name
    return sep.join(name for name in col_name if name != "")


@_dask_cudf_performance_tracking
def _groupby_partition_agg(df, gb_cols, aggs, columns, dropna, sort, sep):
    """Initial partition-level aggregation task.

    This is the first operation to be executed on each input
    partition in `groupby_agg`.  Depending on `aggs`, four possible
    groupby aggregations ("count", "sum", "min", and "max") are
    performed.  The result is then partitioned (by hashing `gb_cols`)
    into a number of distinct dictionary elements.  The number of
    elements in the output dictionary (`split_out`) corresponds to
    the number of partitions in the final output of `groupby_agg`.
    """

    # Modify dict for initial (partition-wise) aggregations
    _agg_dict = {}
    for col, agg_list in aggs.items():
        _agg_dict[col] = set()
        for agg in agg_list:
            if agg in ("mean", "std", "var"):
                _agg_dict[col].add("count")
                _agg_dict[col].add("sum")
            else:
                _agg_dict[col].add(agg)
        _agg_dict[col] = list(_agg_dict[col])
        if set(agg_list).intersection({"std", "var"}):
            pow2_name = _make_name((col, "pow2"), sep=sep)
            df[pow2_name] = df[col].astype("float64").pow(2)
            _agg_dict[pow2_name] = ["sum"]

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False, sort=sort).agg(
        _agg_dict
    )
    output_columns = [_make_name(name, sep=sep) for name in gb.columns]
    gb.columns = output_columns
    # Return with deterministic column ordering
    return gb[sorted(output_columns)]


@_dask_cudf_performance_tracking
def _tree_node_agg(df, gb_cols, dropna, sort, sep):
    """Node in groupby-aggregation reduction tree.

    The input DataFrame (`df`) corresponds to the
    concatenated output of one or more `_groupby_partition_agg`
    tasks. In this function, "sum", "min" and/or "max" groupby
    aggregations will be used to combine the statistics for
    duplicate keys.
    """

    agg_dict = {}
    for col in df.columns:
        if col in gb_cols:
            continue
        agg = col.split(sep)[-1]
        if agg in ("count", "sum"):
            agg_dict[col] = ["sum"]
        elif agg == "list":
            agg_dict[col] = [list]
        elif agg in OPTIMIZED_AGGS:
            agg_dict[col] = [agg]
        else:
            raise ValueError(f"Unexpected aggregation: {agg}")

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False, sort=sort).agg(
        agg_dict
    )

    # Don't include the last aggregation in the column names
    output_columns = [
        _make_name(name[:-1] if isinstance(name, tuple) else name, sep=sep)
        for name in gb.columns
    ]
    gb.columns = output_columns
    # Return with deterministic column ordering
    return gb[sorted(output_columns)]


@_dask_cudf_performance_tracking
def _var_agg(df, col, count_name, sum_name, pow2_sum_name, ddof=1):
    """Calculate variance (given count, sum, and sum-squared columns)."""

    # Select count, sum, and sum-squared
    n = df[count_name]
    x = df[sum_name]
    x2 = df[pow2_sum_name]

    # Use sum-squared approach to get variance
    var = x2 - x**2 / n
    div = n - ddof
    div[div < 1] = 1  # Avoid division by 0
    var /= div

    # Set appropriate NaN elements
    # (since we avoided 0-division)
    var[(n - ddof) == 0] = np.nan

    return var


@_dask_cudf_performance_tracking
def _finalize_gb_agg(
    gb_in,
    gb_cols,
    aggs,
    columns,
    final_columns,
    as_index,
    dropna,
    sort,
    sep,
    str_cols_out,
    aggs_renames,
):
    """Final aggregation task.

    This is the final operation on each output partitions
    of the `groupby_agg` algorithm.  This function must
    take care of higher-order aggregations, like "mean",
    "std" and "var".  We also need to deal with the column
    index, the row index, and final sorting behavior.
    """

    gb = _tree_node_agg(gb_in, gb_cols, dropna, sort, sep)

    # Deal with higher-order aggregations
    for col in columns:
        agg_list = aggs.get(col, [])
        agg_set = set(agg_list)
        if agg_set.intersection({"mean", "std", "var"}):
            count_name = _make_name((col, "count"), sep=sep)
            sum_name = _make_name((col, "sum"), sep=sep)
            if agg_set.intersection({"std", "var"}):
                pow2_sum_name = _make_name((col, "pow2", "sum"), sep=sep)
                var = _var_agg(gb, col, count_name, sum_name, pow2_sum_name)
                if "var" in agg_list:
                    name_var = _make_name((col, "var"), sep=sep)
                    gb[name_var] = var
                if "std" in agg_list:
                    name_std = _make_name((col, "std"), sep=sep)
                    gb[name_std] = np.sqrt(var)
                gb.drop(columns=[pow2_sum_name], inplace=True)
            if "mean" in agg_list:
                mean_name = _make_name((col, "mean"), sep=sep)
                gb[mean_name] = gb[sum_name] / gb[count_name]
            if "sum" not in agg_list:
                gb.drop(columns=[sum_name], inplace=True)
            if "count" not in agg_list:
                gb.drop(columns=[count_name], inplace=True)
        if list in agg_list:
            collect_name = _make_name((col, "list"), sep=sep)
            gb[collect_name] = gb[collect_name].list.concat()

    # Ensure sorted keys if `sort=True`
    if sort:
        gb = gb.sort_values(gb_cols)

    # Set index if necessary
    if as_index:
        gb.set_index(gb_cols, inplace=True)

    # Unflatten column names
    col_array = []
    agg_array = []
    for col in gb.columns:
        if col in gb_cols:
            col_array.append(col)
            agg_array.append("")
        else:
            name, agg = col.split(sep)
            col_array.append(name)
            agg_array.append(aggs_renames.get((name, agg), agg))
    if str_cols_out:
        gb.columns = col_array
    else:
        gb.columns = pd.MultiIndex.from_arrays([col_array, agg_array])

    return gb[final_columns]


@_dask_cudf_performance_tracking
def _redirect_aggs(arg):
    """Redirect aggregations to their corresponding name in cuDF"""
    redirects = {
        sum: "sum",
        max: "max",
        min: "min",
        "collect": list,
        "list": list,
    }
    if isinstance(arg, dict):
        new_arg = dict()
        for col in arg:
            if isinstance(arg[col], list):
                new_arg[col] = [redirects.get(agg, agg) for agg in arg[col]]
            elif isinstance(arg[col], dict):
                new_arg[col] = {
                    k: redirects.get(v, v) for k, v in arg[col].items()
                }
            else:
                new_arg[col] = redirects.get(arg[col], arg[col])
        return new_arg
    if isinstance(arg, list):
        return [redirects.get(agg, agg) for agg in arg]
    return redirects.get(arg, arg)


@_dask_cudf_performance_tracking
def _aggs_optimized(arg, supported: set):
    """Check that aggregations in `arg` are a subset of `supported`"""
    if isinstance(arg, (list, dict)):
        if isinstance(arg, dict):
            _global_set: set[str] = set()
            for col in arg:
                if isinstance(arg[col], list):
                    _global_set = _global_set.union(set(arg[col]))
                elif isinstance(arg[col], dict):
                    _global_set = _global_set.union(set(arg[col].values()))
                else:
                    _global_set.add(arg[col])
        else:
            _global_set = set(arg)

        return bool(_global_set.issubset(supported))
    elif isinstance(arg, (str, type)):
        return arg in supported
    return False


def _get_spec_info(gb):
    if isinstance(gb.arg, (dict, list)):
        aggs = gb.arg.copy()
    else:
        aggs = gb.arg

    if gb._slice and not isinstance(aggs, dict):
        aggs = {gb._slice: aggs}

    gb_cols = gb._by_columns
    if isinstance(gb_cols, str):
        gb_cols = [gb_cols]
    columns = [c for c in gb.frame.columns if c not in gb_cols]
    if not isinstance(aggs, dict):
        aggs = {col: aggs for col in columns}

    # Assert if our output will have a MultiIndex; this will be the case if
    # any value in the `aggs` dict is not a string (i.e. multiple/named
    # aggregations per column)
    str_cols_out = True
    aggs_renames = {}
    for col in aggs:
        if isinstance(aggs[col], str) or callable(aggs[col]):
            aggs[col] = [aggs[col]]
        elif isinstance(aggs[col], dict):
            str_cols_out = False
            col_aggs = []
            for k, v in aggs[col].items():
                aggs_renames[col, v] = k
                col_aggs.append(v)
            aggs[col] = col_aggs
        else:
            str_cols_out = False
        if col in gb_cols:
            columns.append(col)

    return {
        "aggs": aggs,
        "columns": columns,
        "str_cols_out": str_cols_out,
        "aggs_renames": aggs_renames,
    }


def _get_meta(gb):
    spec_info = gb.spec_info
    gb_cols = gb._by_columns
    aggs = spec_info["aggs"].copy()
    aggs_renames = spec_info["aggs_renames"]
    if spec_info["str_cols_out"]:
        # Metadata should use `str` for dict values if that is
        # what the user originally specified (column names will
        # be str, rather than tuples).
        for col in aggs:
            aggs[col] = aggs[col][0]
    _meta = gb.frame._meta.groupby(gb_cols).agg(aggs)
    if aggs_renames:
        col_array = []
        agg_array = []
        for col, agg in _meta.columns:
            col_array.append(col)
            agg_array.append(aggs_renames.get((col, agg), agg))
        _meta.columns = pd.MultiIndex.from_arrays([col_array, agg_array])
    return _meta


class DecomposableCudfGroupbyAgg(DecomposableGroupbyAggregation):
    sep = "___"

    @functools.cached_property
    def spec_info(self):
        return _get_spec_info(self)

    @functools.cached_property
    def _meta(self):
        return _get_meta(self)

    @property
    def shuffle_by_index(self):
        return False  # We always group by column(s)

    @classmethod
    def chunk(cls, df, *by, **kwargs):
        return _groupby_partition_agg(df, **kwargs)

    @classmethod
    def combine(cls, inputs, **kwargs):
        return _tree_node_agg(_concat(inputs), **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        return _finalize_gb_agg(_concat(inputs), **kwargs)

    @property
    def chunk_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        return {
            "gb_cols": self._by_columns,
            "aggs": self.spec_info["aggs"],
            "columns": self.spec_info["columns"],
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
        }

    @property
    def combine_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        return {
            "gb_cols": self._by_columns,
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        final_columns = self._slice or self._meta.columns
        return {
            "gb_cols": self._by_columns,
            "aggs": self.spec_info["aggs"],
            "columns": self.spec_info["columns"],
            "final_columns": final_columns,
            "as_index": True,
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
            "str_cols_out": self.spec_info["str_cols_out"],
            "aggs_renames": self.spec_info["aggs_renames"],
        }


class CudfGroupbyAgg(GroupbyAggregation):
    @functools.cached_property
    def spec_info(self):
        return _get_spec_info(self)

    @functools.cached_property
    def _meta(self):
        return _get_meta(self)

    def _lower(self):
        return DecomposableCudfGroupbyAgg(
            self.frame,
            self.arg,
            self.observed,
            self.dropna,
            self.split_every,
            self.split_out,
            self.sort,
            self.shuffle_method,
            self._slice,
            *self.by,
        )


def _maybe_get_custom_expr(
    gb,
    aggs,
    split_every=None,
    split_out=None,
    shuffle_method=None,
    **kwargs,
):
    if kwargs:
        # Unsupported key-word arguments
        return None

    if not hasattr(gb.obj._meta, "to_pandas"):
        # Not cuDF-backed data
        return None

    _aggs = _redirect_aggs(aggs)
    if not _aggs_optimized(_aggs, OPTIMIZED_AGGS):
        # One or more aggregations are unsupported
        return None

    return CudfGroupbyAgg(
        gb.obj.expr,
        _aggs,
        gb.observed,
        gb.dropna,
        split_every,
        split_out,
        gb.sort,
        shuffle_method,
        gb._slice,
        *gb.by,
    )


##
## Custom groupby classes
##


class ListAgg(SingleAggregation):
    @staticmethod
    def groupby_chunk(arg):
        return arg.agg(list)

    @staticmethod
    def groupby_aggregate(arg):
        gb = arg.agg(list)
        if gb.ndim > 1:
            for col in gb.columns:
                gb[col] = gb[col].list.concat()
            return gb
        else:
            return gb.list.concat()


list_aggregation = Aggregation(
    name="list",
    chunk=ListAgg.groupby_chunk,
    agg=ListAgg.groupby_aggregate,
)


def _translate_arg(arg):
    # Helper function to translate args so that
    # they can be processed correctly by upstream
    # dask & dask-expr. Right now, the only necessary
    # translation is list aggregations.
    if isinstance(arg, dict):
        return {k: _translate_arg(v) for k, v in arg.items()}
    elif isinstance(arg, list):
        return [_translate_arg(x) for x in arg]
    elif arg in ("collect", "list", list):
        return list_aggregation
    else:
        return arg


# We define our own GroupBy classes in Dask cuDF for
# the following reasons:
#  (1) We want to use a custom `aggregate` algorithm
#      that performs multiple aggregations on the
#      same dataframe partition at once. The upstream
#      algorithm breaks distinct aggregations into
#      separate tasks.
#  (2) We need to work around missing `observed=False`
#      support:
#      https://github.com/NVIDIA/cudf/issues/15173


class GroupBy(DXGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)

    def __getitem__(self, key):
        if is_scalar(key):
            return SeriesGroupBy(
                self.obj,
                by=self.by,
                slice=key,
                sort=self.sort,
                dropna=self.dropna,
                observed=self.observed,
            )
        g = GroupBy(
            self.obj,
            by=self.by,
            slice=key,
            sort=self.sort,
            dropna=self.dropna,
            observed=self.observed,
            group_keys=self.group_keys,
        )
        return g

    def aggregate(self, arg, fused=True, **kwargs):
        if (
            fused
            and (expr := _maybe_get_custom_expr(self, arg, **kwargs))
            is not None
        ):
            return new_collection(expr)
        else:
            return super().aggregate(_translate_arg(arg), **kwargs)


class SeriesGroupBy(DXSeriesGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)

    def aggregate(self, arg, **kwargs):
        return super().aggregate(_translate_arg(arg), **kwargs)
