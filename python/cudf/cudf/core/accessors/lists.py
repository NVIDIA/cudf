# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow as pa

from cudf.api.types import is_scalar
from cudf.core.accessors.base_accessor import BaseAccessor
from cudf.core.column.column import as_column
from cudf.core.dtype.validators import is_dtype_obj_list
from cudf.core.dtypes import dtype as cudf_dtype
from cudf.utils.scalar import pa_scalar_to_plc_scalar

if TYPE_CHECKING:
    from cudf._typing import ColumnLike, Dtype, ScalarLike
    from cudf.core.column.lists import ListColumn
    from cudf.core.index import Index
    from cudf.core.series import Series


class ListMethods(BaseAccessor):
    """
    List methods for Series
    """

    _column: ListColumn

    def __init__(self, parent: Series | Index):
        if not is_dtype_obj_list(parent.dtype):
            raise AttributeError(
                "Can only use the '.list' accessor with 'list[pyarrow]' "
                f"dtype, not {parent.dtype}."
            )
        super().__init__(parent=parent)

    def __getitem__(self, key) -> Series | Index:
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)  # type: ignore[attr-defined]
        else:
            return self.get(key)

    def get(
        self,
        index: int | ColumnLike,
        default: ScalarLike | ColumnLike | None = None,
    ) -> Series | Index:
        """
        Extract element at the given index from each list in a Series of lists.

        ``index`` can be an integer or a sequence of integers.  If
        ``index`` is an integer, the element at position ``index`` is
        extracted from each list.  If ``index`` is a sequence, it must
        be of the same length as the Series, and ``index[i]``
        specifies the position of the element to extract from the
        ``i``-th list in the Series.

        If the index is out of bounds for any list, return <NA> or, if
        provided, ``default``.  Thus, this method never raises an
        ``IndexError``.

        Parameters
        ----------
        index : int or sequence of ints
        default : scalar, optional

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> s.list.get(-1)
        0    3
        1    5
        2    6
        dtype: int64

        >>> s = cudf.Series([[1, 2], [3, 4, 5], [4, 5, 6]])
        >>> s.list.get(2)
        0    <NA>
        1       5
        2       6
        dtype: int64

        >>> s.list.get(2, default=0)
        0   0
        1   5
        2   6
        dtype: int64

        >>> s.list.get([0, 1, 2])
        0   1
        1   4
        2   6
        dtype: int64
        """
        if isinstance(index, int):
            out = self._column.extract_element_scalar(index)
        else:
            index = as_column(index)
            out = self._column.extract_element_column(index)

        if not (default is None or default is pd.NA):
            # determine rows for which `index` is out-of-bounds
            lengths = self._column.count_elements()
            out_of_bounds_mask = ((-1 * index) > lengths) | (index >= lengths)

            # replace the value in those rows (should be NA) with `default`
            if out_of_bounds_mask.any():
                out = out._scatter_by_column(
                    out_of_bounds_mask,
                    pa_scalar_to_plc_scalar(pa.scalar(default)),
                )
        return self._return_or_inplace(out)

    def contains(self, search_key: ScalarLike) -> Series | Index:
        """
        Returns boolean values indicating whether the specified scalar
        is an element of each row.

        Parameters
        ----------
        search_key : scalar
            element being searched for in each row of the list column

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> s.list.contains(4)
        0    False
        1     True
        2     True
        dtype: bool
        """
        return self._return_or_inplace(
            self._column.contains_scalar(pa.scalar(search_key))
        )

    def index(self, search_key: ScalarLike | ColumnLike) -> Series | Index:
        """
        Returns integers representing the index of the search key for each row.

        If ``search_key`` is a sequence, it must be the same length as the
        Series and ``search_key[i]`` represents the search key for the
        ``i``-th row of the Series.

        If the search key is not contained in a row, -1 is returned. If either
        the row or the search key are null, <NA> is returned. If the search key
        is contained multiple times, the smallest matching index is returned.

        Parameters
        ----------
        search_key : scalar or sequence of scalars
            Element or elements being searched for in each row of the list
            column

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> s.list.index(4)
        0   -1
        1    1
        2    0
        dtype: int32

        >>> s = cudf.Series([["a", "b", "c"], ["x", "y", "z"]])
        >>> s.list.index(["b", "z"])
        0    1
        1    2
        dtype: int32

        >>> s = cudf.Series([[4, 5, 6], None, [-3, -2, -1]])
        >>> s.list.index([None, 3, -2])
        0    <NA>
        1    <NA>
        2       1
        dtype: int32
        """
        if is_scalar(search_key):
            result = self._column.index_of_scalar(pa.scalar(search_key))
        else:
            result = self._column.index_of_column(as_column(search_key))
        return self._return_or_inplace(result)

    @property
    def leaves(self) -> Series | Index:
        """
        From a Series of (possibly nested) lists, obtain the elements from
        the innermost lists as a flat Series (one value per row).

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> a = cudf.Series([[[1, None], [3, 4]], None, [[5, 6]]])
        >>> a.list.leaves
        0       1
        1    <NA>
        2       3
        3       4
        4       5
        5       6
        dtype: int64
        """
        return self._return_or_inplace(
            self._column.leaves(), retain_index=False
        )

    def len(self) -> Series | Index:
        """
        Computes the length of each element in the Series/Index.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> s
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> s.list.len()
        0       3
        1    <NA>
        2       2
        dtype: int32
        """
        return self._return_or_inplace(self._column.count_elements())

    def take(self, lists_indices: ColumnLike) -> Series | Index:
        """
        Collect list elements based on given indices.

        Parameters
        ----------
        lists_indices: Series-like of lists
            Specifies what to collect from each row

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> s
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> s.list.take([[0, 1], [], []])
        0    [1, 2]
        1      None
        2        []
        dtype: list
        """
        return self._return_or_inplace(
            self._column.segmented_gather(as_column(lists_indices))
        )

    def unique(self) -> Series | Index:
        """
        Returns the unique elements in each list.
        The ordering of elements is not guaranteed.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 1, 2, None, None], None, [4, 4], []])
        >>> s
        0    [1, 1, 2, None, None]
        1                     None
        2                   [4, 4]
        3                       []
        dtype: list
        >>> s.list.unique() # Order of list element is not guaranteed
        0    [1, 2, None]
        1            None
        2             [4]
        3              []
        dtype: list
        """
        return self._return_or_inplace(
            self._column.distinct(nulls_equal=True, nans_all_equal=True)
        )

    def sort_values(
        self,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: Literal["first", "last"] = "last",
        ignore_index: bool = False,
    ) -> Series | Index:
        """
        Sort each list by the values.

        Sort the lists in ascending or descending order by some criterion.

        Parameters
        ----------
        ascending : bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {'first', 'last'}, default 'last'
            'first' puts nulls at the beginning, 'last' puts nulls at the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, ..., n - 1.

        Returns
        -------
        Series or Index with each list sorted

        Examples
        --------
        >>> s = cudf.Series([[4, 2, None, 9], [8, 8, 2], [2, 1]])
        >>> s.list.sort_values(ascending=True, na_position="last")
        0    [2, 4, 9, None]
        1          [2, 8, 8]
        2             [1, 2]
        dtype: list

        .. pandas-compat::
            `pandas.Series.list.sort_values`

            This method does not exist in pandas but it can be run
            as:

            >>> import pandas as pd
            >>> s = pd.Series([[3, 2, 1], [2, 4, 3]])
            >>> print(s.apply(sorted))
            0    [1, 2, 3]
            1    [2, 3, 4]
            dtype: object
        """
        if inplace:
            raise NotImplementedError("`inplace` not currently implemented.")
        if kind != "quicksort":
            raise NotImplementedError("`kind` not currently implemented.")
        return self._return_or_inplace(
            self._column.sort_lists(ascending, na_position),
            retain_index=not ignore_index,
        )

    def concat(self, dropna: bool = True) -> Series | Index:
        """
        For a column with at least one level of nesting, concatenate the
        lists in each row.

        Parameters
        ----------
        dropna: bool, optional
            If True (default), ignores top-level null elements in each row.
            If False, and top-level null elements are present, the resulting
            row in the output is null.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s1 = cudf.Series([[[1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, None], [7.0], [8.0, 9.0]]])
        >>> s1
        0      [[1.0, 2.0], [3.0, 4.0, 5.0]]
        1    [[6.0, None], [7.0], [8.0, 9.0]]
        dtype: list
        >>> s1.list.concat()
        0    [1.0, 2.0, 3.0, 4.0, 5.0]
        1    [6.0, None, 7.0, 8.0, 9.0]
        dtype: list

        Null values at the top-level in each row are dropped by default:

        >>> s2 = cudf.Series([[[1.0, 2.0], None, [3.0, 4.0, 5.0]], [[6.0, None], [7.0], [8.0, 9.0]]])
        >>> s2
        0    [[1.0, 2.0], None, [3.0, 4.0, 5.0]]
        1        [[6.0, None], [7.0], [8.0, 9.0]]
        dtype: list
        >>> s2.list.concat()
        0    [1.0, 2.0, 3.0, 4.0, 5.0]
        1    [6.0, None, 7.0, 8.0, 9.0]
        dtype: list

        Use ``dropna=False`` to produce a null instead:

        >>> s2.list.concat(dropna=False)
        0                          None
        1    [6.0, None, 7.0, 8.0, 9.0]
        dtype: list
        """
        return self._return_or_inplace(
            self._column.concatenate_list_elements(dropna)
        )

    def astype(self, dtype: Dtype) -> Series | Index:
        """
        Return a new list Series with the leaf values casted
        to the specified data type.

        Parameters
        ----------
        dtype: data type to cast leaves values to

        Returns
        -------
        A new Series of lists

        Examples
        --------
        >>> s = cudf.Series([[1, 2], [3, 4]])
        >>> s.dtype
        ListDtype(int64)
        >>> s2 = s.list.astype("float64")
        >>> s2.dtype
        ListDtype(float64)
        """
        return self._return_or_inplace(
            self._column._transform_leaves(
                lambda col, dtype: col.astype(cudf_dtype(dtype)), dtype
            )
        )
