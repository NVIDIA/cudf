# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import is_integer, is_list_like, is_scalar
from cudf.core.column import (
    ColumnBase,
    as_column,
    column_empty,
    concat_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.dtype.validators import is_dtype_obj_string
from cudf.core.dtypes import CategoricalDtype, dtype as cudf_dtype
from cudf.utils.dtypes import (
    DEFAULT_STRING_DTYPE,
    SIZE_TYPE_DTYPE,
    find_common_type,
    is_pandas_nullable_extension_dtype,
    min_unsigned_type,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cudf._typing import DtypeObj
    from cudf.core.dataframe import DataFrame
    from cudf.core.index import Index
    from cudf.core.multiindex import MultiIndex
    from cudf.core.series import Series

_AXIS_MAP = {0: 0, 1: 1, "index": 0, "columns": 1}


def _align_objs(objs, how="outer", sort=None):
    """
    Align a set of Series or Dataframe objects.

    Parameters
    ----------
    objs : list of DataFrame, Series, or Index
    how : How to handle indexes on other axis (or axes),
    similar to join in concat
    sort : Whether to sort the resulting Index

    Returns
    -------
    A list of reindexed and aligned objects
    ready for concatenation
    """
    # Check if multiindex then check if indexes match. Index
    # returns ndarray tuple of bools requiring additional filter.
    # Then check for duplicate index value.
    i_objs = iter(objs)
    first = next(i_objs)

    not_matching_index = any(
        not first.index.equals(rest.index) for rest in i_objs
    )

    if not_matching_index:
        if not all(o.index.is_unique for o in objs):
            raise ValueError("cannot reindex on an axis with duplicate labels")

        index = objs[0].index
        name = index.name

        final_index = _get_combined_index(
            [obj.index for obj in objs], intersect=how == "inner", sort=sort
        )

        final_index.name = name
        return [
            obj.reindex(final_index)
            if not final_index.equals(obj.index)
            else obj
            for obj in objs
        ]
    else:
        if sort:
            if not first.index.is_monotonic_increasing:
                final_index = first.index.sort_values()
                return [obj.reindex(final_index) for obj in objs]
        return objs


def _get_combined_index(indexes, intersect: bool = False, sort=None):
    if len(indexes) == 0:
        index = cudf.Index([])
    elif len(indexes) == 1:
        index = indexes[0]
    elif intersect:
        sort = True
        index = indexes[0]
        for other in indexes[1:]:
            # Don't sort for every intersection,
            # let the sorting happen in the end.
            index = index.intersection(other, sort=False)
    else:
        index = indexes[0]
        if sort is None:
            # pandas keeps string and categorical unions in order of
            # appearance (a categorical union decategorizes, so sorting it
            # here would order lexically rather than by category).
            sort = not (
                is_dtype_obj_string(index.dtype)
                or isinstance(index.dtype, CategoricalDtype)
            )
        for other in indexes[1:]:
            index = index.union(other, sort=False)

    if sort:
        if not index.is_monotonic_increasing:
            index = index.sort_values()

    return index


def _normalize_series_and_dataframe(
    objs: list[Series | DataFrame], axis: Literal[0, 1]
) -> None:
    """Convert any cudf.Series objects in objs to DataFrames in place."""
    # Default to naming series by a numerical id if they are not named.
    sr_name = 0
    for idx, obj in enumerate(objs):
        if isinstance(obj, cudf.Series):
            name = obj.name
            if name is None:
                if axis == 0:
                    # Preserve "unnamed" semantics so the resulting frame has
                    # a RangeIndex columns object (matching pandas).
                    objs[idx] = obj.to_frame()
                    continue
                else:
                    name = sr_name
                    sr_name += 1

            objs[idx] = obj.to_frame(name=name)


def _finalize_concat_metadata(result, inputs):
    """Propagate ``attrs`` and ``flags`` onto a concat result.

    Mirrors pandas' ``__finalize__`` with ``input_objs``: ``attrs`` are
    propagated only when all inputs carry identical non-empty ``attrs``,
    and ``allows_duplicate_labels`` is the AND across inputs. The
    :class:`~pandas.errors.DuplicateLabelError` check is performed by
    :class:`pandas.Flags` when setting ``allows_duplicate_labels`` to
    ``False``.
    """
    inputs = [
        obj for obj in inputs if isinstance(obj, (cudf.Series, cudf.DataFrame))
    ]
    if not inputs or not isinstance(result, (cudf.Series, cudf.DataFrame)):
        return result
    if all(bool(obj.attrs) for obj in inputs):
        first_attrs = inputs[0].attrs
        if all(obj.attrs == first_attrs for obj in inputs[1:]):
            result._attrs = copy.deepcopy(first_attrs)
    allows = all(obj.flags.allows_duplicate_labels for obj in inputs)
    result.flags.allows_duplicate_labels = allows
    return result


def concat(
    objs,
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    sort=None,
):
    """Concatenate DataFrames, Series, or Indices row-wise.

    Parameters
    ----------
    objs : list or dictionary of DataFrame, Series, or Index
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.
        `axis=1` must be passed if a dictionary is passed.
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
    ignore_index : bool, default False
        Set True to ignore the index of the *objs* and provide a
        default range index instead.
    keys : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level.
        Currently not supported.
    levels : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys.
        Currently not supported.
    names : list, default None
        Names for the levels in the resulting hierarchical index.
        Currently not supported.
    verify_integrity : bool, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation.
        Currently not supported.
    sort : bool, default False
        Sort non-concatenation axis if it is not already aligned.

    Returns
    -------
    A new object of like type with rows from each object in ``objs``.

    Examples
    --------
    Combine two ``Series``.

    >>> import cudf
    >>> s1 = cudf.Series(['a', 'b'])
    >>> s2 = cudf.Series(['c', 'd'])
    >>> s1
    0    a
    1    b
    dtype: str
    >>> s2
    0    c
    1    d
    dtype: str
    >>> cudf.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    dtype: str

    Clear the existing index and reset it in the
    result by setting the ``ignore_index`` option to ``True``.

    >>> cudf.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    dtype: str

    Combine two DataFrame objects with identical columns.

    >>> df1 = cudf.DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1
      letter  number
    0      a       1
    1      b       2
    >>> df2 = cudf.DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2
      letter  number
    0      c       3
    1      d       4
    >>> cudf.concat([df1, df2])
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine DataFrame objects with overlapping columns and return
    everything. Columns outside the intersection will
    be filled with ``null`` values.

    >>> df3 = cudf.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3
      letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> cudf.concat([df1, df3], sort=False)
      letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> cudf.concat([df1, df3], join="inner")
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects horizontally along the
    x axis by passing in ``axis=1``.

    >>> df4 = cudf.DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])
    >>> df4
       animal    name
    0    bird   polly
    1  monkey  george
    >>> cudf.concat([df1, df4], axis=1)
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    Combine a dictionary of DataFrame objects horizontally:

    >>> d = {'first': df1, 'second': df2}
    >>> cudf.concat(d, axis=1)
      first           second
      letter  number  letter  number
    0      a       1       c       3
    1      b       2       d       4
    """
    # Materialize the input sequence so both `concat`'s implementation and
    # `_finalize_concat_metadata` see identical objects (concat consumes
    # dicts/iterators).
    if isinstance(objs, dict):
        inputs = list(objs.values())
    else:
        inputs = list(objs)
    result = _concat_impl(
        objs,
        axis=axis,
        join=join,
        ignore_index=ignore_index,
        keys=keys,
        levels=levels,
        names=names,
        verify_integrity=verify_integrity,
        sort=sort,
    )
    return _finalize_concat_metadata(result, inputs)


def _concat_impl(
    objs,
    axis=0,
    join="outer",
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    sort=None,
):
    if keys is not None:
        raise NotImplementedError("keys is currently not supported")
    if levels is not None:
        raise NotImplementedError("levels is currently not supported")
    if names is not None:
        raise NotImplementedError("names is currently not supported")
    # TODO: Do we really need to have different error messages for an empty
    # list and a list of None?
    if not objs:
        raise ValueError("No objects to concatenate")

    axis = _AXIS_MAP.get(axis, None)
    if axis is None:
        raise ValueError(
            f'`axis` must be 0 / "index" or 1 / "columns", got: {axis}'
        )

    if isinstance(objs, dict):
        if axis != 1:
            raise NotImplementedError(
                f"Can only concatenate dictionary input along axis=1, not {axis}"
            )
        objs = {k: obj for k, obj in objs.items() if obj is not None}
        keys_objs = list(objs)
        objs = list(objs.values())
        if any(isinstance(o, cudf.Index) for o in objs):
            raise TypeError(
                "cannot concatenate a dictionary containing indices"
            )
    else:
        objs = [obj for obj in objs if obj is not None]
        keys_objs = None

    if not objs:
        raise ValueError("All objects passed were None")

    # Retrieve the base types of `objs`. In order to support sub-types
    # and object wrappers, we use `isinstance()` instead of comparing
    # types directly
    allowed_typs = {
        cudf.Series,
        cudf.DataFrame,
        cudf.Index,
    }
    if not all(isinstance(o, tuple(allowed_typs)) for o in objs):
        raise TypeError(
            f"can only concatenate objects which are instances of "
            f"{allowed_typs}, instead received {[type(o) for o in objs]}"
        )

    if any(isinstance(o, cudf.Index) for o in objs):
        if not all(isinstance(o, cudf.Index) for o in objs):
            raise TypeError(
                "when concatenating indices you must provide ONLY indices"
            )

    only_series = all(isinstance(o, cudf.Series) for o in objs)

    # Return for single object
    if len(objs) == 1:
        obj = objs[0]
        if ignore_index:
            if axis == 1:
                if isinstance(obj, cudf.Series):
                    result = obj.to_frame()
                else:
                    result = obj.copy(deep=True)
                result.columns = cudf.RangeIndex(result._num_columns)
            else:
                result = type(obj)._from_data(
                    data=obj._data.copy(deep=True),
                    index=cudf.RangeIndex(len(obj)),
                )
        elif axis == 0:
            result = obj.copy(deep=True)
        else:
            if isinstance(obj, cudf.Series):
                result = obj.to_frame()
            else:
                result = obj.copy(deep=True)
            if keys_objs is not None and isinstance(result, cudf.DataFrame):
                k = keys_objs[0]
                result.columns = pd.MultiIndex.from_tuples(
                    [
                        (k, *c) if isinstance(c, tuple) else (k, c)
                        for c in result._column_names
                    ]
                )

        if isinstance(result, cudf.Series) and axis == 0:
            # sort has no effect for series concatted along axis 0
            return result
        else:
            return result.sort_index(axis=(1 - axis)) if sort else result

    # when axis is 1 (column) we can concat with Series and Dataframes
    if axis == 1:
        if not all(isinstance(o, (cudf.Series, cudf.DataFrame)) for o in objs):
            raise TypeError(
                "Can only concatenate Series and DataFrame objects when axis=1"
            )
        _normalize_series_and_dataframe(objs, axis=axis)

        # Inner joins involving empty data frames always return empty dfs, but
        # We must delay returning until we have set the column names.
        empty_inner = join == "inner" and any(obj.empty for obj in objs)

        if len(objs) == 0:
            # TODO: https://github.com/rapidsai/cudf/issues/16550
            return cudf.DataFrame()

        # Don't need to align indices of all `objs` since we
        # would anyway return an empty dataframe below
        if not empty_inner:
            objs = _align_objs(objs, how=join, sort=sort)
            result_index = objs[0].index
        else:
            result_index = None

        result_data = {}
        result_columns = None
        if keys_objs is None:
            for o in objs:
                for name, col in o._column_labels_and_values:
                    if name in result_data:
                        raise NotImplementedError(
                            f"A Column with duplicate name found: {name}, cuDF "
                            f"doesn't support having multiple columns with "
                            f"same names yet."
                        )
                    if empty_inner:
                        # if join is inner and it contains an empty df
                        # we return an empty df, hence creating an empty
                        # column with dtype metadata retained.
                        result_data[name] = column_empty(
                            row_count=0, dtype=col.dtype
                        )
                    else:
                        result_data[name] = col

            result_columns = (
                objs[0]
                ._data.to_pandas_index.append(
                    [obj._data.to_pandas_index for obj in objs[1:]]
                )
                .unique()
            )

        # need to create a MultiIndex column
        else:
            # All levels in the multiindex label must have the same type
            has_multiple_level_types = (
                len({type(name) for o in objs for name in o._column_names}) > 1
            )
            if has_multiple_level_types:
                raise NotImplementedError(
                    "Cannot construct a MultiIndex column with multiple "
                    "label types in cuDF at this time. You must convert "
                    "the labels to the same type."
                )
            for k, o in zip(keys_objs, objs, strict=True):
                for name, col in o._column_labels_and_values:
                    # if only series, then only keep keys_objs as column labels
                    # if the existing column is multiindex, prepend it
                    # to handle cases where dfs and srs are concatenated
                    if only_series:
                        col_label = k
                    elif isinstance(name, tuple):
                        col_label = (k, *name)
                    else:
                        col_label = (k, name)
                    if empty_inner:
                        result_data[col_label] = column_empty(
                            row_count=0, dtype=col.dtype
                        )
                    else:
                        result_data[col_label] = col

        df = cudf.DataFrame._from_data(
            ColumnAccessor(result_data, verify=False), index=result_index
        )
        if ignore_index:
            df.columns = cudf.RangeIndex(df._num_columns)
        elif result_columns is not None:
            df.columns = result_columns
        elif not only_series:
            df.columns = pd.MultiIndex.from_tuples(df._column_names)

        if empty_inner:
            # if join is inner and it contains an empty df
            # we return an empty df
            return df.head(0)

        return df

    # If we get here, we are always concatenating along axis 0 (the rows).
    typ = type(objs[0])
    if len({type(o) for o in objs}) > 1:
        _normalize_series_and_dataframe(objs, axis=axis)
        typ = cudf.DataFrame

    if typ is cudf.DataFrame:
        old_objs = objs
        objs = [obj for obj in objs if obj.shape != (0, 0)]
        if len(objs) == 0:
            # If objs is empty, that indicates all of
            # objs are empty dataframes.
            # TODO: https://github.com/rapidsai/cudf/issues/16550
            return cudf.DataFrame()
        elif len(objs) == 1:
            obj = objs[0]
            result = cudf.DataFrame._from_data(
                data={} if join == "inner" else obj._data.copy(deep=True),
                index=cudf.RangeIndex(len(obj))
                if ignore_index
                else obj.index.copy(deep=True),
            )
            return result
        else:
            if join == "inner" and len(old_objs) != len(objs):
                # don't filter out empty df's
                objs = old_objs
            result = cudf.DataFrame._concat(
                objs,
                axis=axis,
                join=join,
                ignore_index=ignore_index,
                # Explicitly cast rather than relying on None being falsy.
                sort=bool(sort),
            )
        return result

    elif typ is cudf.Series:
        new_objs = [obj for obj in objs if len(obj)]
        if len(new_objs) == 1 and not ignore_index:
            result = new_objs[0]
            # Promote dtype for empty series with explicit dtypes,
            # e.g. int64 + empty float64 should yield float64.
            all_promote_dtypes = {obj.dtype for obj in objs if len(obj) == 0}
            all_promote_dtypes.add(result.dtype)
            if len(all_promote_dtypes) > 1:
                common_dtype = find_common_type(list(all_promote_dtypes))
                result = result.astype(common_dtype)
            return result
        else:
            return cudf.Series._concat(objs, axis=axis, index=not ignore_index)
    elif typ is cudf.MultiIndex:
        return cudf.MultiIndex._concat(objs)
    elif issubclass(typ, cudf.Index):
        return cudf.Index._concat(objs)
    else:
        raise TypeError(f"cannot concatenate object of type {typ}")


def melt(
    frame: DataFrame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name: Hashable = "value",
    col_level=None,
    ignore_index: bool = True,
) -> DataFrame:
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

    Difference from pandas:
     * Does not support 'col_level' because cuDF does not have multi-index

    Examples
    --------
    >>> import cudf
    >>> df = cudf.DataFrame({'A': ['a', 'b', 'c'],
    ...                      'B': [1, 3, 5],
    ...                      'C': [2, 4, 6]})
    >>> df
       A  B  C
    0  a  1  2
    1  b  3  4
    2  c  5  6
    >>> cudf.melt(df, id_vars=['A'], value_vars=['B'])
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5
    >>> cudf.melt(df, id_vars=['A'], value_vars=['B', 'C'])
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5
    3  a        C      2
    4  b        C      4
    5  c        C      6

    The names of 'variable' and 'value' columns can be customized:

    >>> cudf.melt(df, id_vars=['A'], value_vars=['B'],
    ...         var_name='myVarname', value_name='myValname')
       A myVarname  myValname
    0  a         B          1
    1  b         B          3
    2  c         B          5
    """
    if col_level is not None:
        raise NotImplementedError("col_level != None is not supported yet.")

    # Arg cleaning

    # id_vars
    if id_vars is not None:
        if is_scalar(id_vars):
            id_vars = [id_vars]
        id_vars = list(id_vars)
        missing = set(id_vars) - set(frame._column_names)
        if not len(missing) == 0:
            raise KeyError(
                f"The following 'id_vars' are not present"
                f" in the DataFrame: {list(missing)}"
            )
    else:
        id_vars = []

    # value_vars
    if value_vars is not None:
        if is_scalar(value_vars):
            value_vars = [value_vars]
        value_vars = list(value_vars)
        missing = set(value_vars) - set(frame._column_names)
        if not len(missing) == 0:
            raise KeyError(
                f"The following 'value_vars' are not present"
                f" in the DataFrame: {list(missing)}"
            )
    else:
        # then all remaining columns in frame
        unique_id = set(id_vars)
        value_vars = [c for c in frame._column_names if c not in unique_id]

    # Error for unimplemented support for datatype
    if any(
        isinstance(frame[col].dtype, CategoricalDtype)
        for col in itertools.chain(id_vars, value_vars)
    ):
        raise NotImplementedError(
            "Categorical columns are not yet supported for function"
        )

    # Check dtype homogeneity in value_var
    # Because heterogeneous concat is unimplemented
    if len(value_vars) > 1:
        dtype = frame[value_vars[0]].dtype
        if any(frame[col].dtype != dtype for col in value_vars):
            raise ValueError("all cols in value_vars must have the same dtype")

    # overlap
    overlap = set(id_vars).intersection(set(value_vars))
    if not len(overlap) == 0:
        raise KeyError(
            f"'value_vars' and 'id_vars' cannot have overlap."
            f" The following 'value_vars' are ALSO present"
            f" in 'id_vars': {list(overlap)}"
        )

    N = len(frame)
    K = len(value_vars)

    def _tile(base_col: ColumnBase, reps: int) -> ColumnBase:
        if reps > 0:
            return concat_columns([base_col] * reps)
        else:
            return column_empty(0, dtype=base_col.dtype)

    # Step 1: tile id_vars
    mdata = {col: _tile(frame[col]._column, K) for col in id_vars}

    # Step 2: add variable
    nval = len(value_vars)
    dtype = min_unsigned_type(nval)

    if var_name is None:
        var_name = "variable"

    if not value_vars:
        # TODO: Use frame._data.label_dtype when it's more consistently set
        var_data = column_empty(
            0, dtype=cudf_dtype(frame._data.to_pandas_index.dtype)
        )
    else:
        var_data = as_column(value_vars).take(
            as_column(np.repeat(np.arange(nval, dtype=dtype), N)),
            check_bounds=False,
        )
    mdata[var_name] = var_data

    # Step 3: add values
    mdata[value_name] = concat_columns(
        [frame[val]._column for val in value_vars]
    )

    result = cudf.DataFrame._from_data(mdata, attrs=frame.attrs)
    if not ignore_index:
        taker = np.tile(np.arange(len(frame)), frame.shape[1] - len(id_vars))
        result.index = frame.index.take(taker)

    return result


def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype="bool",
):
    """Returns a dataframe whose columns are the one hot encodings of all
    columns in `df`

    Parameters
    ----------
    data : array-like, Series, or DataFrame
        Data of which to get dummy indicators.
    prefix : str, dict, or sequence, optional
        Prefix to append. Either a str (to apply a constant prefix), dict
        mapping column names to prefixes, or sequence of prefixes to apply with
        the same length as the number of columns. If not supplied, defaults
        to the empty string
    prefix_sep : str, dict, or sequence, optional, default '_'
        Separator to use when appending prefixes
    dummy_na : boolean, optional
        Add a column to indicate Nones, if False Nones are ignored.
    sparse : boolean, optional
        Right now this is NON-FUNCTIONAL argument in rapids.
    drop_first : boolean, optional
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    columns : sequence of str, optional
        Names of columns to encode. If not provided, will attempt to encode all
        columns. Note this is different from pandas default behavior, which
        encodes all columns with dtype object or categorical
    dtype : str, optional
        Output dtype, default 'bool'

    Examples
    --------
    >>> import cudf
    >>> df = cudf.DataFrame({"a": ["value1", "value2", None], "b": [0, 0, 0]})
    >>> cudf.get_dummies(df)
       b  a_value1  a_value2
    0  0      True     False
    1  0     False      True
    2  0     False     False

    >>> cudf.get_dummies(df, dummy_na=True)
       b  a_value1  a_value2  a_<NA>
    0  0      True     False   False
    1  0     False      True   False
    2  0     False     False    True

    >>> import numpy as np
    >>> df = cudf.DataFrame({"a":cudf.Series([1, 2, np.nan, None],
    ...                     nan_as_null=False)})
    >>> df
          a
    0   1.0
    1   2.0
    2   NaN
    3   NaN

    >>> cudf.get_dummies(df, dummy_na=True, columns=["a"])
       a_1.0  a_2.0  a_nan  a_<NA>
    0   True  False  False   False
    1  False   True  False   False
    2  False  False   True   False
    3  False  False  False    True

    >>> series = cudf.Series([1, 2, None, 2, 4])
    >>> series
    0       1
    1       2
    2    <NA>
    3       2
    4       4
    dtype: int64
    >>> cudf.get_dummies(series, dummy_na=True)
           1      2      4   <NA>
    0   True  False  False  False
    1  False   True  False  False
    2  False  False  False   True
    3  False   True  False  False
    4  False  False   True  False
    """
    if sparse:
        raise NotImplementedError("sparse is not supported yet")

    dtype = cudf_dtype(dtype)

    if isinstance(data, cudf.DataFrame):
        encode_fallback_dtypes = [DEFAULT_STRING_DTYPE, "category"]

        if columns is None or len(columns) == 0:
            columns = data.select_dtypes(
                include=encode_fallback_dtypes
            )._column_names

        _length_check_params(prefix, columns, "prefix")
        _length_check_params(prefix_sep, columns, "prefix_sep")

        if prefix is None:
            prefix = columns

        if isinstance(prefix, str):
            prefix_map = {}
        elif isinstance(prefix, dict):
            prefix_map = prefix
        else:
            prefix_map = dict(zip(columns, prefix, strict=True))

        if isinstance(prefix_sep, str):
            prefix_sep_map = {}
        elif isinstance(prefix_sep, dict):
            prefix_sep_map = prefix_sep
        else:
            prefix_sep_map = dict(zip(columns, prefix_sep, strict=True))

        # If we have no columns to encode, we need to drop
        # fallback columns(if any)
        if len(columns) == 0:
            return data.select_dtypes(exclude=encode_fallback_dtypes)
        else:
            result_data = {
                col_name: col
                for col_name, col in data._column_labels_and_values
                if col_name not in columns
            }

            for name in columns:
                col_enc_data = _one_hot_encode_column(
                    column=data._data[name],
                    prefix=prefix_map.get(name, prefix),
                    prefix_sep=prefix_sep_map.get(name, prefix_sep),
                    dtype=dtype,
                    drop_first=drop_first,
                    dummy_na=dummy_na,
                )
                result_data.update(col_enc_data)
            return cudf.DataFrame._from_data(result_data, index=data.index)
    else:
        ser = cudf.Series(data)
        data = _one_hot_encode_column(
            column=ser._column,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dtype=dtype,
            drop_first=drop_first,
            dummy_na=dummy_na,
        )
        return cudf.DataFrame._from_data(data, index=ser.index)


def _pivot(
    col_accessor: ColumnAccessor,
    index_labels: Index | MultiIndex,
    index_idx: ColumnBase,
    columns_labels: Index | MultiIndex,
    columns_idx: ColumnBase,
    promote_ints_on_missing: bool = False,
) -> DataFrame:
    """
    Reorganize the values of the DataFrame according to the given
    index and columns.

    Parameters
    ----------
    col_accessor : ColumnAccessor
        Values to pivot into the result's columns.
    index_labels : Index
        Distinct index keys; row labels of the result.
    index_idx : ColumnBase
        Position of each source row's key within ``index_labels``.
    columns_labels : Index
        Distinct column keys; labels of the result's new column level(s).
    columns_idx : ColumnBase
        Position of each source row's key within ``columns_labels``.
    promote_ints_on_missing : bool
        Promote integer source columns to float64 when the reshape
        introduces missing cells, as pandas' unstack does. The unstack
        and pivot paths want this; pivot_table/crosstab fill missing
        cells afterwards and keep the integer dtype.
    """
    column_labels = columns_labels.to_pandas().to_flat_index()

    result = {}
    if len(index_labels) != 0 and len(columns_labels) != 0:

        def as_tuple(x):
            return x if isinstance(x, tuple) else (x,)

        nrows = len(index_labels)
        promote_ints = promote_ints_on_missing and cudf.get_option(
            "mode.pandas_compatible"
        )
        for col_label, col in col_accessor.items():
            names = [
                as_tuple(col_label) + as_tuple(name) for name in column_labels
            ]
            new_size = nrows * len(names)
            scatter_map = (columns_idx * np.int32(nrows)) + index_idx
            if (
                promote_ints
                and new_size > len(col)
                and isinstance(col.dtype, np.dtype)
                and col.dtype.kind in "iu"
            ):
                # pandas builds one 2-D values block per source column and
                # promotes the whole block to float64 when the reshape
                # introduces missing entries, so even gap-free result
                # columns become float64
                col = col.astype(np.dtype(np.float64))
            target_col = column_empty(row_count=new_size, dtype=col.dtype)
            target_col[scatter_map] = col
            result.update(
                dict(
                    zip(
                        names,
                        target_col.split_by_offsets(
                            list(range(nrows, new_size, nrows))
                        ),
                        strict=True,
                    )
                )
            )

    # the result of pivot always has a MultiIndex; the leading level(s)
    # come from the source frame's column labels, so preserve their names
    ca = ColumnAccessor(
        result,
        multiindex=True,
        level_names=(
            *col_accessor.level_names,
            *columns_labels.names,
        ),
        verify=False,
    )
    return cudf.DataFrame._from_data(ca, index=index_labels)


def _unstack_encode_by_codes(
    mi: MultiIndex, level
) -> tuple[Index | MultiIndex, ColumnBase, Index | MultiIndex, ColumnBase]:
    """Encode unstack keys ordered by the MultiIndex level codes.

    libcudf's ``encode`` orders distinct keys by sorted value with nulls
    last, but pandas' unstack orders keys by the index's level codes: the
    level order is preserved and missing entries (code -1) come first.
    Encoding the integer code columns instead of the level values yields
    exactly that order.
    """
    lvl_idx = mi._level_index_from_level(level)
    mi._maybe_materialize_codes_and_levels()
    names = mi.names

    def encode_side(sel: list[int]) -> tuple[Index | MultiIndex, ColumnBase]:
        code_cols = []
        for i in sel:
            code = mi._codes[i].astype(np.dtype(np.int64)).copy()  # type: ignore[index]
            # Normalize the NA sentinel (``MultiIndex.__init__`` stores
            # ``iinfo(SIZE_TYPE_DTYPE).min``, lazy factorization stores -1)
            # so the missing-key group encodes as one key that sorts first.
            code[code < 0] = -1
            code_cols.append(code)
        code_frame = cudf.DataFrame._from_data(
            ColumnAccessor(dict(enumerate(code_cols)), verify=False)
        )
        key_codes, idx = code_frame._encode()
        labels_data = {}
        out_levels = []
        out_codes = []
        for j, i in enumerate(sel):
            kc = key_codes._columns[j].astype(np.dtype(np.int64))
            out_levels.append(mi._levels[i])  # type: ignore[index]
            out_codes.append(kc)
            gather_codes = kc.copy()
            gather_codes[gather_codes == -1] = np.iinfo(SIZE_TYPE_DTYPE).min
            # key by position: level names may be duplicated or None
            labels_data[j] = mi._levels[i]._column.take(  # type: ignore[index]
                gather_codes, nullify=True
            )
        if len(labels_data) == 1:
            labels: Index | MultiIndex = cudf.Index._from_column(
                next(iter(labels_data.values())), name=names[sel[0]]
            )
        else:
            mi_labels = cudf.MultiIndex._from_data(labels_data)
            mi_labels.names = [names[i] for i in sel]
            # carry the original level objects and the keys' codes so that
            # a subsequent unstack/stack keeps ordering by the original
            # levels, exactly like pandas (which reuses the level objects)
            mi_labels._levels = out_levels
            mi_labels._codes = out_codes
            labels = mi_labels
        return labels, idx

    remaining = [i for i in range(mi.nlevels) if i != lvl_idx]
    index_labels, index_idx = encode_side(remaining)
    columns_labels, columns_idx = encode_side([lvl_idx])
    return index_labels, index_idx, columns_labels, columns_idx


def pivot(
    data: DataFrame, columns=None, index=no_default, values=no_default
) -> DataFrame:
    """
    Return reshaped DataFrame organized by the given index and column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame.

    Parameters
    ----------
    columns : scalar or list of scalars, optional
        Column label(s) used to construct the columns of the result.
    index : scalar or list of scalars, optional
        Column label(s) used to construct the index of the result.
    values : column name or list of column names, optional
        Column(s) whose values are rearranged to produce the result.
        If not specified, all remaining columns of the DataFrame
        are used.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> a = cudf.DataFrame()
    >>> a['a'] = [1, 1, 2, 2]
    >>> a['b'] = ['a', 'b', 'a', 'b']
    >>> a['c'] = [1, 2, 3, 4]
    >>> a.pivot(index='a', columns='b')
       c
    b  a  b
    a
    1  1  2
    2  3  4

    Pivot with missing values in result:

    >>> a = cudf.DataFrame()
    >>> a['a'] = [1, 1, 2]
    >>> a['b'] = [1, 2, 3]
    >>> a['c'] = ['one', 'two', 'three']
    >>> a.pivot(index='a', columns='b')
              c
        b     1     2      3
        a
        1   one   two    NaN
        2   NaN   NaN  three

    """
    values_is_list = True
    if values is no_default:
        already_selected = set(
            itertools.chain(
                [index] if is_scalar(index) else index,
                [columns] if is_scalar(columns) else columns,
            )
        )
        cols_to_select = [
            col for col in data._column_names if col not in already_selected
        ]
    elif not isinstance(values, (list, tuple)):
        cols_to_select = [values]
        values_is_list = False
    else:
        cols_to_select = values  # type: ignore[assignment]
    if index is no_default:
        index_data = data.index
    else:
        index_data = data.loc[:, index]
        if index_data.ndim == 2:
            index_data = cudf.MultiIndex.from_frame(index_data)
            if not is_scalar(index) and len(index) == 1:
                # pandas converts single level MultiIndex to Index
                index_data = index_data.get_level_values(0)
        else:
            index_data = cudf.Index(index_data)
        # An entirely empty input pivots to an empty result. Pandas uses the
        # default ``object`` dtype for the resulting index axis in that case;
        # mirror this so index metadata (dtype/inferred_type) matches.
        if (
            len(data) == 0
            and not isinstance(index_data, cudf.MultiIndex)
            and is_pandas_nullable_extension_dtype(index_data.dtype)
            and is_dtype_obj_string(index_data.dtype)
        ):
            index_data = cudf.Index(
                pd.Index([], name=index_data.name, dtype=object)
            )

    column_data = data.loc[:, columns]
    # When `columns` is a scalar but the source DataFrame has a MultiIndex on
    # the row axis, ``loc`` may return a 2-D selection in cuDF. Treat the
    # selection as 1-D so we end up with a flat Index of column labels.
    if is_scalar(columns) and column_data.ndim == 2:
        column_data = column_data.iloc[:, 0]
    if column_data.ndim == 2:
        column_data = cudf.MultiIndex.from_frame(column_data)
    else:
        column_data = cudf.Index(column_data)
    # An entirely empty input pivots to an empty result. Pandas reports the
    # default ``object`` dtype for the resulting columns axis in that case;
    # mirror this so column metadata (dtype/inferred_type) matches.
    if (
        len(data) == 0
        and not isinstance(column_data, cudf.MultiIndex)
        and is_pandas_nullable_extension_dtype(column_data.dtype)
        and is_dtype_obj_string(column_data.dtype)
    ):
        column_data = cudf.Index(
            pd.Index([], name=column_data.name, dtype=object)
        )

    # Create a DataFrame composed of columns from both
    # columns and index
    ca = ColumnAccessor(
        dict(
            enumerate(
                itertools.chain(index_data._columns, column_data._columns)
            )
        ),
        verify=False,
    )
    columns_index = cudf.DataFrame._from_data(ca)

    # Check that each row is unique:
    if len(columns_index) != len(columns_index.drop_duplicates()):
        raise ValueError("Duplicate index-column pairs found. Cannot reshape.")

    selection = data._data.select_by_label(cols_to_select)
    if values is not no_default:
        # pandas rebuilds the columns axis from ``values`` and drops the
        # original columns-axis name(s)
        selection._level_names = (None,) * selection.nlevels
    columns_labels, columns_idx = column_data._encode()
    index_labels, index_idx = index_data._encode()
    result = _pivot(
        selection,
        index_labels,
        index_idx,
        columns_labels,
        columns_idx,
        promote_ints_on_missing=True,
    )
    result._attrs = data.attrs

    # MultiIndex to Index
    if not values_is_list:
        result._data.droplevel(0)
        result._data.label_dtype = column_data.dtype

    return result


def unstack(df, level, fill_value=None, sort: bool = True):
    """
    Pivot one or more levels of the (necessarily hierarchical) index labels.

    Pivots the specified levels of the index labels of df to the innermost
    levels of the columns labels of the result.

    * If the index of ``df`` has multiple levels, returns a ``Dataframe`` with
      specified level of the index pivoted to the column levels.
    * If the index of ``df`` has single level, returns a ``Series`` with all
      column levels pivoted to the index levels.

    Parameters
    ----------
    df : DataFrame
    level : level name or index, list-like
        Integer, name or list of such, specifying one or more
        levels of the index to pivot
    fill_value
        Non-functional argument provided for compatibility with Pandas.
    sort : bool, default True
        Sort the level(s) in the resulting MultiIndex columns.


    Returns
    -------
    Series or DataFrame

    Examples
    --------
    >>> df = cudf.DataFrame()
    >>> df['a'] = [1, 1, 1, 2, 2]
    >>> df['b'] = [1, 2, 3, 1, 2]
    >>> df['c'] = [5, 6, 7, 8, 9]
    >>> df['d'] = ['a', 'b', 'a', 'd', 'e']
    >>> df = df.set_index(['a', 'b', 'd'])
    >>> df
           c
    a b d
    1 1 a  5
      2 b  6
      3 a  7
    2 1 d  8
      2 e  9

    Unstacking level 'a':

    >>> df.unstack('a')
            c
    a       1     2
    b d
    1 a     5  <NA>
      d  <NA>     8
    2 b     6  <NA>
      e  <NA>     9
    3 a     7  <NA>

    Unstacking level 'd' :

    >>> df.unstack('d')
            c
    d       a     b     d     e
    a b
    1 1     5  <NA>  <NA>  <NA>
      2  <NA>     6  <NA>  <NA>
      3     7  <NA>  <NA>  <NA>
    2 1  <NA>  <NA>     8  <NA>
      2  <NA>  <NA>  <NA>     9

    Unstacking multiple levels:

    >>> df.unstack(['b', 'd'])
          c
    b     1           2           3
    d     a     d     b     e     a
    a
    1     5  <NA>     6  <NA>     7
    2  <NA>     8  <NA>     9  <NA>

    Unstacking single level index dataframe:

    >>> df = cudf.DataFrame({('c', 1): [1, 2, 3], ('c', 2):[9, 8, 7]})
    >>> df.unstack()
    c  1  0    1
          1    2
          2    3
       2  0    9
          1    8
          2    7
    dtype: int64
    """
    return _unstack(df, level, fill_value=fill_value, sort=sort)


def _unstack(
    df,
    level,
    fill_value=None,
    sort: bool = True,
    promote_ints_on_missing: bool = True,
):
    """``unstack`` implementation.

    ``promote_ints_on_missing`` promotes integer source columns to float64
    when the reshape introduces missing cells, like pandas' unstack.
    ``pivot_table`` (and thereby ``crosstab``) disables it because those fill
    the missing cells afterwards and keep the integer dtype.
    """
    if not isinstance(df, cudf.DataFrame):
        raise ValueError("`df` should be a cudf Dataframe object.")

    if df.empty:
        raise ValueError("Cannot unstack an empty dataframe.")

    if (
        not is_scalar(level)
        and len(level) > 1
        and df.index.nlevels > 1
        and cudf.get_option("mode.pandas_compatible")
    ):
        # We currently produce columns in the wrong order vs pandas
        # See https://github.com/rapidsai/cudf/issues/20446.
        # We should plan to remove once we rewrite pivot and unstack
        # for better performance. See https://github.com/rapidsai/cudf/issues/20469
        raise NotImplementedError(
            "Unstacking multiple index levels is not yet pandas compatible"
        )

    if fill_value is not None:
        raise NotImplementedError("fill_value is not supported.")
    elif sort is False:
        raise NotImplementedError(f"{sort=} is not supported.")
    if not is_scalar(level):
        if not level:
            return df
        if len(level) == 1:
            # pandas normalizes a length-1 list-like level to a scalar
            level = level[0]
    if not isinstance(df.index, cudf.MultiIndex):
        if not is_integer(level):
            # pandas validates non-integer levels against the flat index
            # name and raises KeyError on a mismatch
            df.index._validate_index_level(level)
        dtype = df._columns[0].dtype
        if any(col_dtype != dtype for _, col_dtype in df._dtypes):
            raise ValueError(
                "Calling unstack() on single index dataframe"
                " with different column datatype is not supported."
            )
        res = df.T.stack(future_stack=True)
        # Result's index is a multiindex
        res.index.names = (
            tuple(df._data.to_pandas_index.names) + df.index.names
        )
        res._attrs = df.attrs
        return res
    else:
        from cudf.core.indexed_frame import _check_duplicate_level_names

        specified = [level] if is_scalar(level) else list(level)
        _check_duplicate_level_names(
            [lv for lv in specified if not is_integer(lv)],
            df.index.names,
        )
        if is_scalar(level):
            # order rows/columns by the removed level's codes (pandas
            # semantics: level order preserved, missing entries first),
            # not by sorted level values
            index_labels, index_idx, columns_labels, columns_idx = (
                _unstack_encode_by_codes(df.index, level)
            )
        else:
            index = df.index.droplevel(level)
            new_names = []
            ca_data = {}
            for lev in level:
                ca_level, level_idx = df.index._level_to_ca_label(lev)
                new_names.append(df.index.names[level_idx])
                ca_data[ca_level] = df.index._data[ca_level]
            columns = type(df.index)._from_data(
                ColumnAccessor(ca_data, verify=False)
            )
            columns.names = new_names
            columns_labels, columns_idx = columns._encode()
            index_labels, index_idx = index._encode()
        result = _pivot(
            df._data,
            index_labels,
            index_idx,
            columns_labels,
            columns_idx,
            promote_ints_on_missing=promote_ints_on_missing,
        )
        result._attrs = df.attrs
        if is_scalar(level) and result._data.multiindex:
            # pandas keeps unused categories of the removed level
            # ("removed_level_full", pandas GH 17845) in
            # result.columns.levels even though no columns are created
            # for them.
            _, level_idx = df.index._level_to_ca_label(level)
            full_level = df.index.levels[level_idx].to_pandas()
            pdi = result._data.to_pandas_index
            level_values = pdi.get_level_values(-1)
            new_codes = full_level.get_indexer(level_values)
            # -1 codes for NA labels are pandas' canonical missing
            # representation; only bail out when a non-NA label failed
            # to map into the full level
            if ((new_codes >= 0) | pd.isna(level_values)).all():
                result._data.to_pandas_index = pd.MultiIndex(
                    levels=[*pdi.levels[:-1], full_level],
                    codes=[*pdi.codes[:-1], new_codes],
                    names=pdi.names,
                    verify_integrity=False,
                )
        if result.index.nlevels == 1:
            result.index = result.index.get_level_values(result.index.names[0])
        return result


def _one_hot_encode_column(
    column: ColumnBase,
    prefix: str | None,
    prefix_sep: str | None,
    dtype: DtypeObj,
    drop_first: bool,
    dummy_na: bool,
) -> dict[str, ColumnBase]:
    """Encode a single column with one hot encoding. The return dictionary
    contains pairs of (category, encodings). The keys may be prefixed with
    `prefix`, separated with category name with `prefix_sep`. The encoding
    columns maybe coerced into `dtype`.
    """
    if isinstance(column.dtype, CategoricalDtype):
        categories = column.categories  # type: ignore[attr-defined]
    else:
        categories = column.unique().sort_values()
    if not dummy_na:
        categories = categories.nans_to_nulls().dropna()

    if isinstance(column.dtype, CategoricalDtype):
        if column.size == column.null_count:
            column = column_empty(
                row_count=column.size, dtype=categories.dtype
            )
        else:
            column = column._get_decategorized_column()  # type: ignore[attr-defined]

    if column.size * categories.size >= np.iinfo(SIZE_TYPE_DTYPE).max:
        raise ValueError(
            "Size limitation exceeded: column.size * category.size < "
            f"np.iinfo({SIZE_TYPE_DTYPE}).max. Consider reducing "
            "size of category"
        )
    result_labels = (
        x if x is not None else "<NA>"
        for x in categories.to_arrow().to_pylist()
    )
    data = dict(
        zip(result_labels, column.one_hot_encode(categories), strict=True)
    )

    if drop_first and len(data):
        data.pop(next(iter(data)))
    if prefix is not None and prefix_sep is not None:
        data = {f"{prefix}{prefix_sep}{col}": enc for col, enc in data.items()}
    data = {k: v.astype(dtype) for k, v in data.items()}
    return data


def _length_check_params(obj, columns, name):
    if is_list_like(obj):
        if len(obj) != len(columns):
            raise ValueError(
                f"Length of '{name}' ({len(obj)}) did not match the "
                f"length of the columns being "
                f"encoded ({len(columns)})."
            )


def _get_pivot_names(arrs, names, prefix):
    """
    Generates unique names for rows/columns
    """
    if names is None:
        names = []
        for i, arr in enumerate(arrs):
            if isinstance(arr, cudf.Series) and arr.name is not None:
                names.append(arr.name)
            else:
                names.append(f"{prefix}_{i}")
    else:
        if len(names) != len(arrs):
            raise ValueError("arrays and names must have the same length")
        if not isinstance(names, list):
            names = list(names)

    return names


def crosstab(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins=False,
    margins_name="All",
    dropna=None,
    normalize=False,
):
    """
    Compute a simple cross tabulation of two (or more) factors. By default
    computes a frequency table of the factors unless an array of values and an
    aggregation function are passed.

    Parameters
    ----------
    index : array-like, Series, or list of arrays/Series
        Values to group by in the rows.
    columns : array-like, Series, or list of arrays/Series
        Values to group by in the columns.
    values : array-like, optional
        Array of values to aggregate according to the factors.
        Requires `aggfunc` be specified.
    rownames : list of str, default None
        If passed, must match number of row arrays passed.
    colnames : list of str, default None
        If passed, must match number of column arrays passed.
    aggfunc : function, optional
        If specified, requires `values` be specified as well.
    margins : Not supported
    margins_name : Not supported
    dropna : Not supported
    normalize : Not supported

    Returns
    -------
    DataFrame
        Cross tabulation of the data.

    Examples
    --------
    >>> a = cudf.Series(["foo", "foo", "foo", "foo", "bar", "bar",
    ...               "bar", "bar", "foo", "foo", "foo"], dtype=object)
    >>> b = cudf.Series(["one", "one", "one", "two", "one", "one",
    ...               "one", "two", "two", "two", "one"], dtype=object)
    >>> c = cudf.Series(["dull", "dull", "shiny", "dull", "dull", "shiny",
    ...               "shiny", "dull", "shiny", "shiny", "shiny"],
    ...              dtype=object)
    >>> cudf.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
    b   one        two
    c   dull shiny dull shiny
    a
    bar    1     2    1     0
    foo    2     2    1     2
    """
    if normalize is not False:
        raise NotImplementedError("normalize is not supported yet")

    if values is None and aggfunc is not None:
        raise ValueError("aggfunc cannot be used without values.")

    if values is not None and aggfunc is None:
        raise ValueError("values cannot be used without an aggfunc.")

    if not isinstance(index, (list, tuple)):
        index = [index]
    if not isinstance(columns, (list, tuple)):
        columns = [columns]

    if not rownames:
        rownames = _get_pivot_names(index, rownames, prefix="row")
    if not colnames:
        colnames = _get_pivot_names(columns, colnames, prefix="col")

    if len(index) != len(rownames):
        raise ValueError("index and rownames must have same length")
    if len(columns) != len(colnames):
        raise ValueError("columns and colnames must have same length")

    if len(set(rownames)) != len(rownames):
        raise ValueError("rownames must be unique")
    if len(set(colnames)) != len(colnames):
        raise ValueError("colnames must be unique")

    data = {
        **dict(zip(rownames, map(as_column, index), strict=True)),
        **dict(zip(colnames, map(as_column, columns), strict=True)),
    }

    df = cudf.DataFrame._from_data(data)

    if values is None:
        df["__dummy__"] = 0
        kwargs = {"aggfunc": "count", "fill_value": 0}
    else:
        df["__dummy__"] = values
        kwargs = {"aggfunc": aggfunc}

    table = pivot_table(
        data=df,
        index=rownames,
        columns=colnames,
        values="__dummy__",
        margins=margins,
        margins_name=margins_name,
        dropna=dropna,
        **kwargs,
    )

    return table


def pivot_table(
    data: DataFrame,
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
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    Parameters
    ----------
    data : DataFrame
    values : column name or list of column names to aggregate, optional
    index : scalar or list of column names
            Values to group by in the rows.
    columns : scalar or list of column names
            Values to group by in the columns.
    aggfunc : str or dict, default "mean"
            If dict is passed, the key is column to aggregate
            and value is function name.
    fill_value : scalar, default None
        Value to replace missing values with
        (in the resulting pivot table, after aggregation).
    margins : Not supported
    dropna : Not supported
    margins_name : Not supported
    observed : Not supported
    sort : Not supported

    Returns
    -------
    DataFrame
        An Excel style pivot table.
    """
    if margins is not False:
        raise NotImplementedError("margins is not supported yet")

    if margins_name != "All":
        raise NotImplementedError("margins_name is not supported yet")

    if dropna is not None:
        raise NotImplementedError("dropna is not supported yet")

    if observed is not False:
        raise NotImplementedError("observed is not supported yet")

    if sort is not True:
        raise NotImplementedError("sort is not supported yet")

    if is_scalar(index):
        index = [index]
    if is_scalar(columns):
        columns = [columns]

    keys = index + columns

    values_passed = values is not None
    if values_passed:
        if not is_scalar(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]

        for i in values:
            if i not in data:
                raise KeyError(i)

        to_filter = []
        for x in keys + values:
            if isinstance(x, cudf.Grouper):
                x = x.key
            try:
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        if len(to_filter) < data._num_columns:
            data = data[to_filter]

    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)

    grouped = data.groupby(keys)
    agged = grouped.agg(aggfunc)

    table = agged

    if table.index.nlevels > 1 and index:
        # If index_names are integers, determine whether the integers refer
        # to the level position or name.
        index_names = agged.index.names[: len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = _unstack(
            agged,
            to_unstack,
            # pandas keeps the integer dtype when the missing cells are
            # filled afterwards, and promotes to float64 when they are not
            promote_ints_on_missing=fill_value is None,
        )

    if fill_value is not None:
        table = table.fillna(fill_value)

    # discard the top level
    if values_passed and not values_multi and table._data.nlevels > 1:
        table.columns = table._data.to_pandas_index.droplevel(0)
    if len(index) == 0 and len(columns) > 0:
        table = table.T
    table._attrs = data.attrs
    return table
