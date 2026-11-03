# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

import cudf
from cudf.core.column import as_column
from cudf.core.dtypes import CategoricalDtype
from cudf.options import get_option
from cudf.utils.dtypes import can_convert_to_column

if TYPE_CHECKING:
    from cudf.core.index import Index


def factorize(
    values,
    sort: bool = False,
    use_na_sentinel: bool = True,
    size_hint: int | None = None,
) -> tuple[cp.ndarray, cp.ndarray | Index]:
    """Encode the input values as integer labels

    Parameters
    ----------
    values: Series, Index, or CuPy array
        The data to be factorized.
    sort : bool, default True
        Sort uniques and shuffle codes to maintain the relationship.
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NA values.
        If False, NA values will be encoded as non-negative
        integers and will not drop the NA from the uniques
        of the values.

    Returns
    -------
    (labels, cats) : (cupy.ndarray, cupy.ndarray or Index)
        - *labels* contains the encoded values
        - *cats* contains the categories in order that the N-th
            item corresponds to the (N-1) code.

    See Also
    --------
    cudf.Series.factorize : Encode the input values of Series.

    Examples
    --------
    >>> import cudf
    >>> import numpy as np
    >>> data = cudf.Series(['a', 'c', 'c'])
    >>> codes, uniques = cudf.factorize(data)
    >>> codes
    array([0, 1, 1])
    >>> uniques
    Index(['a', 'c'], dtype='str')

    When ``use_na_sentinel=True`` (the default), missing values are indicated
    in the `codes` with the sentinel value ``-1`` and missing values are not
    included in `uniques`.

    >>> codes, uniques = cudf.factorize(['b', None, 'a', 'c', 'b'])
    >>> codes
    array([ 0, -1,  1,  2,  0])
    >>> uniques
    Index(['b', 'a', 'c'], dtype='str')

    If NA is in the values, and we want to include NA in the uniques of the
    values, it can be achieved by setting ``use_na_sentinel=False``.

    >>> values = np.array([1, 2, 1, np.nan])
    >>> codes, uniques = cudf.factorize(values)
    >>> codes
    array([ 0,  1,  0, -1])
    >>> uniques
    array([1., 2.])
    >>> codes, uniques = cudf.factorize(values, use_na_sentinel=False)
    >>> codes
    array([0, 1, 0, 2])
    >>> uniques
    array([ 1.,  2., nan])
    """
    if size_hint:
        warnings.warn("size_hint is not applicable for cudf.factorize")

    # cupy/numpy arrays: run column logic directly and return an array
    # for ``cats`` so the call is closed under array inputs (mirrors
    # ``pd.factorize(np.ndarray)``).
    if isinstance(values, cp.ndarray):
        labels, cats = as_column(values).factorize(sort, use_na_sentinel)
        return labels, cats.values
    if isinstance(values, np.ndarray):
        labels, cats = as_column(values).factorize(sort, use_na_sentinel)
        return labels.get(), cats.to_pandas().to_numpy()

    # cudf objects: delegate to the type's ``_factorize`` so subclasses
    # (e.g. ``RangeIndex``) can specialize while sharing the common
    # column-level ``Column.factorize`` path by default.
    if isinstance(values, (cudf.Series, cudf.Index, cudf.MultiIndex)):
        return values._factorize(sort=sort, use_na_sentinel=use_na_sentinel)

    if not can_convert_to_column(values):
        raise TypeError(
            "'values' can only be a Series, Index, or CuPy array, "
            f"got {type(values)}"
        )

    labels, cats = as_column(values).factorize(sort, use_na_sentinel)
    # TODO: Avoid accessing Index from the top level namespace
    return labels, cudf.Index._from_column(cats)


def unique(values):
    """
    Return unique values from array-like

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    cudf.Series,

        The return can be:

        * Index : when the input is an Index
        * cudf.Series : when the input is a Series
        * cupy.ndarray : when the input is a cupy.ndarray

        Return cudf.Series, cudf.Index, or cupy.ndarray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> cudf.unique(cudf.Series([2, 1, 3, 3]))
    0    2
    1    1
    2    3
    dtype: int64

    >>> cudf.unique(cudf.Series([2] + [1] * 5))
    0    2
    1    1
    dtype: int64

    >>> import pandas as pd
    >>> cudf.unique(cudf.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    0   2016-01-01
    dtype: datetime64[us]

    >>> cudf.unique(
    ...     cudf.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160103", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    0   2016-01-01 00:00:00-05:00
    1   2016-01-03 00:00:00-05:00
    dtype: datetime64[us, US/Eastern]

    >>> cudf.unique(
    ...     cudf.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160103", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00', '2016-01-03 00:00:00-05:00'], dtype='datetime64[us, US/Eastern]')

    An unordered Categorical will return categories in the
    order of appearance.

    >>> cudf.unique(cudf.Series(pd.Categorical(list("baabc"))))
    0    b
    1    a
    2    c
    dtype: category
    Categories (3, str): ['a', 'b', 'c']

    >>> cudf.unique(cudf.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    0    b
    1    a
    2    c
    dtype: category
    Categories (3, str): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> cudf.unique(
    ...     cudf.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    0    b
    1    a
    2    c
    dtype: category
    Categories (3, str): ['a' < 'b' < 'c']
    """
    # TODO: Avoid accessing Index and Series from the top level namespace
    if not isinstance(values, (cudf.Series, cudf.Index, cp.ndarray)):
        raise ValueError(
            "Must pass cudf.Series, cudf.Index, or cupy.ndarray object"
        )
    if isinstance(values, cp.ndarray):
        # pandas.unique will not sort the values in the result
        # while cupy.unique documents it will, so we pass cupy.ndarray
        # through cudf.Index to maintain the original order.
        return cp.asarray(cudf.Index(values).unique())
    if isinstance(values, cudf.Series):
        if get_option("mode.pandas_compatible"):
            if isinstance(values.dtype, CategoricalDtype):
                raise NotImplementedError(
                    "cudf.Categorical is not implemented"
                )
            else:
                return cp.asarray(values.unique())
    return values.unique()
