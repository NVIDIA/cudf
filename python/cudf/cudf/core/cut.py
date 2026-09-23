# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import cupy
import numpy as np
import pandas as pd

import cudf
from cudf.api.types import is_list_like
from cudf.core.column import as_column
from cudf.core.index import IntervalIndex, interval_range


def cut(
    x,
    bins,
    right: bool = True,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):
    """Bin values into discrete intervals.

    Use cut when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.

        * int : Defines the number of equal-width bins in the range of `x`. The
          range of `x` is extended by .1% on each side to include the minimum
          and maximum values of `x`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    right : bool, default True
        Indicates whether bins includes the rightmost edge or not.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same
        length as the resulting bins. If False, returns only integer
        indicators of the bins. If True,raises an error. When ordered=False,
        labels must be provided.
    retbins : bool, default False
        Whether to return the bins or not.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.
    ordered : bool, default True
        Whether the labels are ordered or not. Applies to returned types
        Categorical and Series (with Categorical dtype). If True,
        the resulting categorical will be ordered. If False, the resulting
        categorical will be unordered (labels must be provided).

    Returns
    -------
    out : CategoricalIndex
        An array-like object representing the respective bin for each value
        of x. The type depends on the value of labels.
    bins : numpy.ndarray or IntervalIndex.
        The computed or specified bins. Only returned when retbins=True.
        For scalar or sequence bins, this is an ndarray with the computed
        bins. If set duplicates=drop, bins will drop non-unique bin. For
        an IntervalIndex bins, this is equal to bins.

    Examples
    --------
    Discretize into three equal-sized bins.

    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]), 3)  # doctest: +SKIP
    CategoricalIndex([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0],
                (5.0, 7.0], (0.994, 3.0]], categories=[(0.994, 3.0],
                (3.0, 5.0], (5.0, 7.0]], ordered=True, dtype='category')

    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)  # doctest: +SKIP
    (CategoricalIndex([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0],
                (5.0, 7.0], (0.994, 3.0]], categories=[(0.994, 3.0],
                (3.0, 5.0], (5.0, 7.0]], ordered=True, dtype='category'),
     array([0.994, 3.   , 5.   , 7.   ]))

    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]),
    ...          3, labels=["bad", "medium", "good"])
    CategoricalIndex(['bad', 'medium', 'good', 'good', 'medium', 'bad'], categories=['bad', 'medium', 'good'], ordered=True, dtype='category')

    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]), 3,
    ...          labels=["B", "A", "B"], ordered=False)
    CategoricalIndex(['B', 'B', 'A', 'A', 'B', 'B'], categories=['A', 'B'],
               ordered=False, dtype='category')

    >>> cudf.cut([0, 1, 1, 2], bins=4, labels=False)
    array([0, 1, 1, 3], dtype=int32)

    Passing a Series as an input returns a Series with categorical dtype:

    >>> s = cudf.Series(np.array([2, 4, 6, 8, 10]),
    ...        index=['a', 'b', 'c', 'd', 'e'])
    >>> cudf.cut(s, 3)  # doctest: +SKIP
    a    (1.992, 4.667]
    b    (1.992, 4.667]
    c    (4.667, 7.333]
    d     (7.333, 10.0]
    e     (7.333, 10.0]
    dtype: category
    Categories (3, interval[float64, right]): [(1.992, 4.667] < (4.667, 7.333] < (7.333, 10.0]]
    """
    left_inclusive = False
    right_inclusive = True
    # saving the original input x for use in case its a series
    orig_x = x
    old_bins = bins

    if not ordered and labels is None:
        raise ValueError("'labels' must be provided if 'ordered = False'")

    if duplicates not in ["raise", "drop"]:
        raise ValueError(
            "invalid value for 'duplicates' parameter, valid options are: "
            "raise, drop"
        )

    if labels is not False:
        if not (labels is None or is_list_like(labels)):
            raise ValueError(
                "Bin labels must either be False, None or passed in as a "
                "list-like argument"
            )
        if ordered and labels is not None:
            if len(set(labels)) != len(labels):
                raise ValueError(
                    "labels must be unique if ordered=True;"
                    "pass ordered=False for duplicate labels"
                )

    # bins can either be an int, sequence of scalars or an intervalIndex
    if isinstance(bins, Sequence):
        if len(set(bins)) != len(bins):
            if duplicates == "raise":
                raise ValueError(
                    f"Bin edges must be unique: {bins!r}.\n"
                    f"You can drop duplicate edges by setting the 'duplicates'"
                    "kwarg"
                )
            elif duplicates == "drop":
                # get unique values but maintain list dtype
                bins = list(dict.fromkeys(bins))

    # if bins is an intervalIndex we ignore the value of right
    elif isinstance(bins, (pd.IntervalIndex, cudf.IntervalIndex)):
        right = bins.closed == "right"

    # create bins if given an int or single scalar
    if not isinstance(bins, pd.IntervalIndex):
        if not isinstance(bins, Sequence):
            if isinstance(
                x, (pd.Series, cudf.Series, np.ndarray, cupy.ndarray)
            ):
                mn = x.min()
                mx = x.max()
            else:
                mn = min(x)
                mx = max(x)
            if mn == mx:
                raise NotImplementedError(
                    "cut on homogeneous data is not supported."
                )
            bins = np.linspace(mn, mx, bins + 1, endpoint=True)
            adj = (mx - mn) * 0.001
            if right:
                bins[0] -= adj
            else:
                bins[-1] += adj

        # if right and include lowest we adjust the first
        # bin edge to make sure it is included
        if right and include_lowest:
            bins[0] = bins[0] - 10 ** (-precision)

        # if right is false the last bin edge is not included
        if not right:
            right_edge = bins[-1]
            x = cupy.asarray(x)
            x[x == right_edge] = right_edge + 1

        # adjust bin edges decimal precision
        int_label_bins = np.around(bins, precision)

    # checking for the correct inclusivity values
    if right:
        closed = "right"
    else:
        closed = "left"
        left_inclusive = True

    if isinstance(bins, pd.IntervalIndex):
        interval_labels = bins
    elif labels is None:
        if duplicates == "drop" and len(bins) == 1 and len(old_bins) != 1:
            if right and include_lowest:
                old_bins[0] = old_bins[0] - 10 ** (-precision)
                interval_labels = interval_range(
                    old_bins[0], old_bins[1], periods=1, closed=closed
                )
            else:
                interval_labels = IntervalIndex.from_breaks(
                    old_bins, closed=closed
                )
        else:
            # get labels for categories
            interval_labels = IntervalIndex.from_breaks(
                int_label_bins, closed=closed
            )
    elif labels is not False:
        if not (is_list_like(labels)):
            raise ValueError(
                "Bin labels must either be False, None or passed in as a "
                "list-like argument"
            )
        if ordered and len(set(labels)) != len(labels):
            raise ValueError(
                "labels must be unique if ordered=True; "
                "pass ordered=False for"
                "duplicate labels"
            )

        if len(labels) != len(bins) - 1:
            raise ValueError(
                "Bin labels must be one fewer than the number of bin edges"
            )
        if not ordered and len(set(labels)) != len(labels):
            interval_labels = cudf.CategoricalIndex(
                labels, categories=None, ordered=False
            )
        else:
            interval_labels = (
                labels if len(set(labels)) == len(labels) else None
            )

    # the inputs is a column of the values in the array x
    input_arr = as_column(x)

    if isinstance(bins, pd.IntervalIndex):
        # get the left and right edges of the bins as columns
        # we cannot typecast an IntervalIndex, so we need to
        # make the edges the same type as the input array
        left_edges = as_column(bins.left).astype(input_arr.dtype)
        right_edges = as_column(bins.right).astype(input_arr.dtype)
    else:
        # get the left and right edges of the bins as columns
        left_edges = as_column(bins[:-1:], dtype=np.dtype("float64"))
        right_edges = as_column(bins[+1::], dtype=np.dtype("float64"))
        # the input arr must be changed to the same type as the edges
        input_arr = input_arr.astype(left_edges.dtype)
    # get the indexes for the appropriate number
    index_labels = input_arr.label_bins(
        left_edge=left_edges,
        left_inclusive=left_inclusive,
        right_edge=right_edges,
        right_inclusive=right_inclusive,
    )

    if labels is False:
        # if labels is false we return the index labels, we return them
        # as a series if we have a series input
        if isinstance(orig_x, (pd.Series, cudf.Series)):
            # need to run more tests but looks like in this case pandas
            # always returns a float64 dtype
            indx_arr_series = cudf.Series(index_labels, dtype="float64")
            # if retbins we return the bins as well
            if retbins:
                return indx_arr_series, bins
            else:
                return indx_arr_series
        elif retbins:
            return index_labels.values, bins
        else:
            return index_labels.values

    if labels is not None:
        if labels is not ordered and len(set(labels)) != len(labels):
            # when we have duplicate labels and ordered is False, we
            # should allow duplicate categories.
            return interval_labels[index_labels]

    categorical_index = cudf.CategoricalIndex.from_codes(
        categories=interval_labels,
        codes=index_labels,
        ordered=ordered,
    )

    if isinstance(orig_x, (pd.Series, cudf.Series)):
        # if we have a series input we return a series output
        res_series = cudf.Series(categorical_index, index=orig_x.index)
        if retbins:
            return res_series, bins
        else:
            return res_series
    elif retbins:
        # if retbins is true we return the bins as well
        return categorical_index, bins
    else:
        return categorical_index
