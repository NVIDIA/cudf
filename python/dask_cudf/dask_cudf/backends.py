# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Iterator
from functools import partial
from typing import cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

from dask import config
from dask.array.dispatch import percentile_lookup
from dask.dataframe.backends import (
    DataFrameBackendEntrypoint,
    PandasBackendEntrypoint,
)
from dask.dataframe.dispatch import (
    categorical_dtype_dispatch,
    concat_dispatch,
    from_pyarrow_table_dispatch,
    group_split_dispatch,
    grouper_dispatch,
    hash_object_dispatch,
    is_categorical_dtype_dispatch,
    make_meta_dispatch,
    meta_nonempty,
    partd_encode_dispatch,
    pyarrow_schema_dispatch,
    to_pyarrow_table_dispatch,
    tolist_dispatch,
    union_categoricals_dispatch,
)
from dask.dataframe.utils import (
    UNKNOWN_CATEGORIES,
    _nonempty_scalar,
    _scalar_from_dtype,
    make_meta_obj,
)
from dask.sizeof import sizeof as sizeof_dispatch
from dask.utils import Dispatch, is_arraylike

import cudf
from cudf.api.types import is_scalar, is_string_dtype
from cudf.utils.performance_tracking import _dask_cudf_performance_tracking


@meta_nonempty.register(cudf.Index)
@_dask_cudf_performance_tracking
def _nonempty_index(idx):
    """Return a non-empty cudf.Index as metadata."""
    # TODO: IntervalIndex, TimedeltaIndex?
    if isinstance(idx, cudf.RangeIndex):
        return cudf.RangeIndex(2, name=idx.name)
    elif isinstance(idx, cudf.DatetimeIndex):
        data = np.array(["1970-01-01", "1970-01-02"], dtype=idx.dtype)
        return cudf.DatetimeIndex(data, name=idx.name)
    elif isinstance(idx, cudf.CategoricalIndex):
        return cudf.CategoricalIndex.from_codes(
            np.array([0, 0], dtype=idx.codes.dtype),
            categories=idx.dtype.categories,
            ordered=idx.dtype.ordered,
            name=idx.name,
        )
    elif isinstance(idx, cudf.MultiIndex):
        levels = [meta_nonempty(lev) for lev in idx.levels]
        codes = [[0, 0]] * idx.nlevels
        return cudf.MultiIndex(levels=levels, codes=codes, names=idx.names)
    elif is_string_dtype(idx.dtype):
        return cudf.Index(["cat", "dog"], name=idx.name)
    elif isinstance(idx, cudf.Index):
        return cudf.Index(np.arange(2, dtype=idx.dtype), name=idx.name)

    raise TypeError(
        f"Don't know how to handle index of type {type(idx).__name__}"
    )


def _nest_list_data(data, leaf_type):
    """
    Helper for _get_non_empty_data which creates
    nested list data
    """
    data = [data]
    while isinstance(leaf_type, cudf.ListDtype):
        leaf_type = leaf_type.element_type
        data = [data]
    return data


@_dask_cudf_performance_tracking
def _get_non_empty_data(
    s: cudf.core.column.ColumnBase,
) -> cudf.core.column.ColumnBase:
    """Return a non-empty column as metadata from a column."""
    if isinstance(s.dtype, cudf.CategoricalDtype):
        s = cast("cudf.core.column.CategoricalColumn", s)
        categories = (
            s.categories if len(s.categories) else [UNKNOWN_CATEGORIES]
        )
        codes = cudf.core.column.as_column(
            0,
            dtype=s.codes.dtype,
            length=2,
        )
        return cudf.core.column.ColumnBase.create(
            codes.plc_column,
            cudf.CategoricalDtype(
                categories=categories, ordered=s.dtype.ordered
            ),
        )
    elif isinstance(s.dtype, cudf.ListDtype):
        leaf_type = s.dtype.leaf_type
        if is_string_dtype(leaf_type):
            data = ["cat", "dog"]
        else:
            data = np.array([0, 1], dtype=leaf_type).tolist()
        data = _nest_list_data(data, s.dtype) * 2
        return cudf.core.column.as_column(data, dtype=s.dtype)
    elif isinstance(s.dtype, cudf.StructDtype):
        # Handles IntervalColumn
        struct_dtype = s.dtype
        struct_data = [{key: None for key in struct_dtype.fields.keys()}] * 2
        return cudf.core.column.as_column(struct_data, dtype=s.dtype)
    elif is_string_dtype(s.dtype):
        return cudf.core.column.as_column(pa.array(["cat", "dog"]))
    elif isinstance(s.dtype, pd.DatetimeTZDtype):
        date_data = cudf.date_range("2001-01-01", periods=2, freq=s.time_unit)  # type: ignore[attr-defined]
        return date_data.tz_localize(str(s.dtype.tz))._column
    elif s.dtype.kind in "fiubmM":
        return cudf.core.column.as_column(np.arange(0, 2), dtype=s.dtype)
    elif isinstance(s.dtype, cudf.core.dtypes.DecimalDtype):
        return cudf.core.column.as_column(range(2), dtype=s.dtype)
    else:
        raise TypeError(
            f"Don't know how to handle column of type {type(s).__name__}"
        )


@meta_nonempty.register(cudf.Series)
@_dask_cudf_performance_tracking
def _nonempty_series(s, idx=None):
    if idx is None:
        idx = _nonempty_index(s.index)
    data = _get_non_empty_data(s._column)

    return cudf.Series._from_column(data, name=s.name, index=idx)


@meta_nonempty.register(cudf.DataFrame)
@_dask_cudf_performance_tracking
def meta_nonempty_cudf(x):
    idx = meta_nonempty(x.index)
    columns_with_dtype = dict()
    res = {}
    for col_label, col in x._data.items():
        dtype = col.dtype
        if isinstance(
            dtype,
            (cudf.ListDtype, cudf.StructDtype, cudf.CategoricalDtype),
        ):
            # 1. Not possible to hash and store list & struct types
            #    as they can contain different levels of nesting or
            #    fields.
            # 2. Not possible to hash `category` types as
            #    they often contain an underlying types to them.
            res[col_label] = _get_non_empty_data(col)
        else:
            if dtype not in columns_with_dtype:
                columns_with_dtype[dtype] = _get_non_empty_data(col)
            res[col_label] = columns_with_dtype[dtype]

    return cudf.DataFrame._from_data(res, index=idx)


@make_meta_dispatch.register((cudf.Series, cudf.DataFrame))
@_dask_cudf_performance_tracking
def make_meta_cudf(x, index=None):
    return x.head(0)


@make_meta_dispatch.register(cudf.Index)
@_dask_cudf_performance_tracking
def make_meta_cudf_index(x, index=None):
    return x[:0]


@_dask_cudf_performance_tracking
def _empty_series(name, dtype, index=None):
    if isinstance(dtype, str) and dtype == "category":
        dtype = cudf.CategoricalDtype(categories=[UNKNOWN_CATEGORIES])
    return cudf.Series([], dtype=dtype, name=name, index=index)


@make_meta_obj.register(object)
@_dask_cudf_performance_tracking
def make_meta_object_cudf(x, index=None):
    """Create an empty cudf object containing the desired metadata.

    Parameters
    ----------
    x : dict, tuple, list, cudf.Series, cudf.DataFrame, cudf.Index,
        dtype, scalar
        To create a DataFrame, provide a `dict` mapping of `{name: dtype}`, or
        an iterable of `(name, dtype)` tuples. To create a `Series`, provide a
        tuple of `(name, dtype)`. If a cudf object, names, dtypes, and index
        should match the desired output. If a dtype or scalar, a scalar of the
        same dtype is returned.
    index :  cudf.Index, optional
        Any cudf index to use in the metadata. If none provided, a
        `RangeIndex` will be used.

    Examples
    --------
    >>> make_meta([('a', 'i8'), ('b', 'O')])
    Empty DataFrame
    Columns: [a, b]
    Index: []
    >>> make_meta(('a', 'f8'))
    Series([], Name: a, dtype: float64)
    >>> make_meta('i8')
    1
    """
    if hasattr(x, "_meta"):
        return x._meta
    elif is_arraylike(x) and x.shape:
        return x[:0]

    if index is not None:
        index = make_meta_dispatch(index)

    if isinstance(x, dict):
        return cudf.DataFrame(
            {c: _empty_series(c, d, index=index) for (c, d) in x.items()},
            index=index,
        )
    if isinstance(x, tuple) and len(x) == 2:
        return _empty_series(x[0], x[1], index=index)
    elif isinstance(x, (list, tuple)):
        if not all(isinstance(i, tuple) and len(i) == 2 for i in x):
            raise ValueError(
                f"Expected iterable of tuples of (name, dtype), got {x}"
            )
        return cudf.DataFrame(
            {c: _empty_series(c, d, index=index) for (c, d) in x},
            columns=[c for c, d in x],
            index=index,
        )
    elif not hasattr(x, "dtype") and x is not None:
        # could be a string, a dtype object, or a python type. Skip `None`,
        # because it is implicitly converted to `dtype('f8')`, which we don't
        # want here.
        try:
            dtype = np.dtype(x)
            return _scalar_from_dtype(dtype)
        except Exception:
            # Continue on to next check
            pass

    if is_scalar(x):
        return _nonempty_scalar(x)

    raise TypeError(f"Don't know how to create metadata from {x}")


@concat_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
@_dask_cudf_performance_tracking
def concat_cudf(
    dfs,
    axis=0,
    join="outer",
    uniform=False,
    filter_warning=True,
    sort=None,
    ignore_index=False,
    **kwargs,
):
    assert join == "outer"

    ignore_order = kwargs.get("ignore_order", False)
    if ignore_order:
        raise NotImplementedError(
            "ignore_order parameter is not yet supported in dask-cudf"
        )

    return cudf.concat(dfs, axis=axis, ignore_index=ignore_index)


@categorical_dtype_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
@_dask_cudf_performance_tracking
def categorical_dtype_cudf(categories=None, ordered=False):
    return cudf.CategoricalDtype(categories=categories, ordered=ordered)


@tolist_dispatch.register((cudf.Series, cudf.Index))
@_dask_cudf_performance_tracking
def tolist_cudf(obj):
    return obj.to_pandas().tolist()


@is_categorical_dtype_dispatch.register(
    (cudf.Series, cudf.Index, cudf.CategoricalDtype)  # , Series)
)
@_dask_cudf_performance_tracking
def is_categorical_dtype_cudf(obj):
    return cudf.api.types._is_categorical_dtype(obj)


@grouper_dispatch.register((cudf.Series, cudf.DataFrame))
def get_grouper_cudf(obj):
    return cudf.core.groupby.Grouper


@percentile_lookup.register((cudf.Series, cp.ndarray, cudf.Index))
@_dask_cudf_performance_tracking
def percentile_cudf(a, q, interpolation="linear"):
    # Cudf dispatch to the equivalent of `np.percentile`:
    # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    a = cudf.Series(a)
    # a is series.
    n = len(a)
    if not len(a):
        return None, n
    if isinstance(q, Iterator):
        q = list(q)

    if isinstance(a.dtype, cudf.CategoricalDtype):
        result = cp.percentile(a.cat.codes, q, interpolation=interpolation)

        return (
            pd.Categorical.from_codes(
                result, a.dtype.categories, a.dtype.ordered
            ),
            n,
        )
    if a.dtype.kind == "M":
        result = a.quantile(
            [i / 100.0 for i in q], interpolation=interpolation
        )

        if q[0] == 0:
            # https://github.com/dask/dask/issues/6864
            result[0] = min(result[0], a.min())
        return result.to_pandas(), n
    if a.dtype.kind not in "iufm":
        # TODO: Do we want to include timedelta?
        interpolation = "nearest"
    return (
        a.quantile(
            [i / 100.0 for i in q], interpolation=interpolation
        ).to_pandas(),
        n,
    )


@pyarrow_schema_dispatch.register((cudf.DataFrame,))
def _get_pyarrow_schema_cudf(obj, preserve_index=None, **kwargs):
    if kwargs:
        warnings.warn(
            "Ignoring the following arguments to "
            f"`pyarrow_schema_dispatch`: {list(kwargs)}"
        )

    return _cudf_to_table(
        meta_nonempty(obj), preserve_index=preserve_index
    ).schema


@to_pyarrow_table_dispatch.register(cudf.DataFrame)
def _cudf_to_table(obj, preserve_index=None, **kwargs):
    if kwargs:
        warnings.warn(
            "Ignoring the following arguments to "
            f"`to_pyarrow_table_dispatch`: {list(kwargs)}"
        )
    return obj.to_arrow(preserve_index=preserve_index)


@from_pyarrow_table_dispatch.register(cudf.DataFrame)
def _table_to_cudf(obj, table, self_destruct=None, **kwargs):
    # cudf ignores self_destruct.
    kwargs.pop("self_destruct", None)
    if kwargs:
        warnings.warn(
            f"Ignoring the following arguments to "
            f"`from_pyarrow_table_dispatch`: {list(kwargs)}"
        )
    return obj.from_arrow(table)


@union_categoricals_dispatch.register((cudf.Series, cudf.Index))
@_dask_cudf_performance_tracking
def union_categoricals_cudf(
    to_union, sort_categories=False, ignore_order=False
):
    return cudf.api.types._union_categoricals(
        to_union, sort_categories=False, ignore_order=False
    )


@hash_object_dispatch.register((cudf.DataFrame, cudf.Series))
@_dask_cudf_performance_tracking
def hash_object_cudf(frame, index=True):
    if index:
        frame = frame.reset_index()
    return frame.hash_values()


@hash_object_dispatch.register(cudf.Index)
@_dask_cudf_performance_tracking
def hash_object_cudf_index(ind, index=None):
    if isinstance(ind, cudf.MultiIndex):
        return ind.to_frame(index=False).hash_values()

    col = cudf.core.column.as_column(ind)
    return cudf.Series._from_column(col).hash_values()


@group_split_dispatch.register((cudf.Series, cudf.DataFrame))
@_dask_cudf_performance_tracking
def group_split_cudf(df, c, k, ignore_index=False):
    return dict(
        zip(
            range(k),
            df.scatter_by_map(
                c.astype(np.int32, copy=False),
                map_size=k,
                keep_index=not ignore_index,
            ),
            strict=True,
        )
    )


@sizeof_dispatch.register(cudf.DataFrame)
@_dask_cudf_performance_tracking
def sizeof_cudf_dataframe(df):
    return int(
        sum(col.memory_usage for col in df._data.columns)
        + df._index.memory_usage()
    )


@sizeof_dispatch.register((cudf.Series, cudf.Index))
@_dask_cudf_performance_tracking
def sizeof_cudf_series_index(obj):
    return obj.memory_usage()


@partd_encode_dispatch.register(cudf.DataFrame)
def _simple_cudf_encode(_):
    # Basic pickle-based encoding for a partd k-v store
    import pickle

    import partd

    def join(dfs):
        if not dfs:
            return cudf.DataFrame()
        else:
            return cudf.concat(dfs)

    dumps = partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)
    return partial(partd.Encode, dumps, pickle.loads, join)


def _default_backend(func, *args, **kwargs):
    # Utility to call a dask.dataframe function with
    # the default ("pandas") backend

    # NOTE: Some `CudfBackendEntrypoint` methods need to
    # invoke the "pandas"-version of the same method, but
    # with custom kwargs (e.g. `engine`). In these cases,
    # an explicit "pandas" config context is needed to
    # avoid a recursive loop
    with config.set({"dataframe.backend": "pandas"}):
        return func(*args, **kwargs)


def _unsupported_kwargs(old, new, kwargs):
    # Utility to raise a meaningful error when
    # unsupported kwargs are encountered within
    # ``to_backend_dispatch``
    if kwargs:
        raise ValueError(
            f"Unsupported key-word arguments used in `to_backend` "
            f"for {old}-to-{new} conversion: {kwargs}"
        )


def _raise_unsupported_parquet_kwargs(
    open_file_options=None, filesystem=None, **kwargs
):
    import fsspec

    if open_file_options is not None:
        raise ValueError(
            "The open_file_options argument is no longer supported "
            "by the 'cudf' backend."
        )

    if filesystem not in ("fsspec", None) and not isinstance(
        filesystem, fsspec.AbstractFileSystem
    ):
        raise ValueError(
            f"filesystem={filesystem} is not supported by the 'cudf' backend."
        )


# Register cudf->pandas
to_pandas_dispatch = PandasBackendEntrypoint.to_backend_dispatch()


@to_pandas_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def to_pandas_dispatch_from_cudf(data, nullable=False, **kwargs):
    _unsupported_kwargs("cudf", "pandas", kwargs)
    return data.to_pandas(nullable=nullable)


# Register pandas->cudf
to_cudf_dispatch = Dispatch("to_cudf_dispatch")


@to_cudf_dispatch.register((pd.DataFrame, pd.Series, pd.Index))
def to_cudf_dispatch_from_pandas(data, nan_as_null=None, **kwargs):
    _unsupported_kwargs("pandas", "cudf", kwargs)
    return cudf.from_pandas(data, nan_as_null=nan_as_null)


@to_cudf_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def to_cudf_dispatch_from_cudf(data, **kwargs):
    _unsupported_kwargs("cudf", "cudf", kwargs)
    return data


# Define the "cudf" backend for expr-based Dask DataFrame
class CudfBackendEntrypoint(DataFrameBackendEntrypoint):
    """Backend-entrypoint class for Dask-Expressions

    This class is registered under the name "cudf" for the
    ``dask_expr.dataframe.backends`` entrypoint in ``pyproject.toml``.
    Dask-DataFrame will use the methods defined in this class
    in place of ``dask_expr.<creation-method>`` when the
    "dataframe.backend" configuration is set to "cudf":

    Examples
    --------
    >>> import dask
    >>> import dask.dataframe as dd
    >>> with dask.config.set({"dataframe.backend": "cudf"}):
    ...     ddf = dd.from_dict({"a": range(10)})
    >>> type(ddf._meta)
    <class 'cudf.core.dataframe.DataFrame'>
    """

    @staticmethod
    def to_backend(data, **kwargs):
        from dask_cudf._expr import new_collection
        from dask_cudf._expr.expr import ToCudfBackend

        return new_collection(ToCudfBackend(data, kwargs))

    @staticmethod
    def from_dict(
        data,
        npartitions,
        orient="columns",
        dtype=None,
        columns=None,
        constructor=cudf.DataFrame,
    ):
        from dask_cudf._expr import from_dict

        return _default_backend(
            from_dict,
            data,
            npartitions=npartitions,
            orient=orient,
            dtype=dtype,
            columns=columns,
            constructor=constructor,
        )

    @staticmethod
    def read_parquet(*args, **kwargs):
        from dask_cudf.io.parquet import read_parquet as read_parquet_expr

        return read_parquet_expr(*args, **kwargs)

    @staticmethod
    def read_csv(
        path,
        *args,
        header="infer",
        dtype_backend=None,
        storage_options=None,
        **kwargs,
    ):
        from dask_cudf.io.csv import read_csv

        return read_csv(
            path,
            *args,
            header=header,
            storage_options=storage_options,
            **kwargs,
        )

    @staticmethod
    def read_json(*args, **kwargs):
        from dask_cudf.io.json import read_json as read_json_impl

        return read_json_impl(*args, **kwargs)

    @staticmethod
    def read_orc(*args, **kwargs):
        from dask_cudf.io.orc import read_orc as legacy_read_orc

        return legacy_read_orc(*args, **kwargs)
