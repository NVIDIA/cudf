# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class StructMethods:
    def __init__(self, d_series):
        self.d_series = d_series

    def field(self, key):
        """
        Extract children of the specified struct column
        in the Series

        Parameters
        ----------
        key: int or str
            index/position or field name of the respective
            struct column

        Returns
        -------
        Series

        Examples
        --------
        >>> s = cudf.Series([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.struct.field(0).compute()
        0    1
        1    3
        dtype: int64
        >>> ds.struct.field('a').compute()
        0    1
        1    3
        dtype: int64
        """
        meta_result = self.d_series._meta.struct.field(key)
        typ = meta_result.dtype
        name = meta_result.name

        return self.d_series.map_partitions(
            lambda s: s.struct.field(key),
            meta=self.d_series._meta._constructor([], dtype=typ, name=name),
        )

    def explode(self):
        """
        Creates a dataframe view of the struct column, one column per field.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf, dask_cudf
        >>> ds = dask_cudf.from_cudf(cudf.Series(
        ...     [{'a': 42, 'b': 'str1', 'c': [-1]},
        ...      {'a': 0,  'b': 'str2', 'c': [400, 500]},
        ...      {'a': 7,  'b': '',     'c': []}]), npartitions=2)
        >>> ds.struct.explode().compute()
            a     b           c
        0  42  str1        [-1]
        1   0  str2  [400, 500]
        2   7                []
        """
        return self.d_series.map_partitions(
            lambda s: s.struct.explode(),
            meta=self.d_series._meta.struct.explode(),
        )


class ListMethods:
    def __init__(self, d_series):
        self.d_series = d_series

    def len(self):
        """
        Computes the length of each element in the Series/Index.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> ds.list.len().compute()
        0       3
        1    <NA>
        2       2
        dtype: int32
        """
        return self.d_series.map_partitions(
            lambda s: s.list.len(), meta=self.d_series._meta.list.len()
        )

    def contains(self, search_key):
        """
        Creates a column of bool values indicating whether the specified scalar
        is an element of each row of a list column.

        Parameters
        ----------
        search_key : scalar
            element being searched for in each row of the list column

        Returns
        -------
        Column

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.contains(4).compute()
        Series([False, True, True])
        dtype: bool
        """
        return self.d_series.map_partitions(
            lambda s: s.list.contains(search_key),
            meta=self.d_series._meta.list.contains(search_key),
        )

    def get(self, index):
        """
        Extract element at the given index from each component
        Extract element from lists, tuples, or strings in
        each element in the Series/Index.

        Parameters
        ----------
        index : int

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.get(-1).compute()
        0    3
        1    5
        2    6
        dtype: int64
        """
        return self.d_series.map_partitions(
            lambda s: s.list.get(index),
            # The result dtype does not depend on the index, which may also
            # be a list of per-row positions that the empty meta cannot take.
            meta=self.d_series._meta.list.get(0),
        )

    @property
    def leaves(self):
        """
        From a Series of (possibly nested) lists, obtain the elements from
        the innermost lists as a flat Series (one value per row).

        Returns
        -------
        Series

        Examples
        --------
        >>> s = cudf.Series([[[1, None], [3, 4]], None, [[5, 6]]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.leaves.compute()
        0       1
        1    <NA>
        2       3
        3       4
        4       5
        5       6
        dtype: int64
        """
        return self.d_series.map_partitions(
            lambda s: s.list.leaves, meta=self.d_series._meta.list.leaves
        )

    def take(self, lists_indices):
        """
        Collect list elements based on given indices.

        Parameters
        ----------
        lists_indices: List type arrays
            Specifies what to collect from each row

        Returns
        -------
        ListColumn

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> ds.list.take([[0, 1], [], []]).compute()
        0    [1, 2]
        1      None
        2        []
        dtype: list
        """
        return self.d_series.map_partitions(
            lambda s: s.list.take(lists_indices), meta=self.d_series._meta
        )

    def unique(self):
        """
        Returns unique element for each list in the column, order for each
        unique element is not guaranteed.

        Returns
        -------
        ListColumn

        Examples
        --------
        >>> s = cudf.Series([[1, 1, 2, None, None], None, [4, 4], []])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds
        0    [1.0, 1.0, 2.0, nan, nan]
        1                         None
        2                   [4.0, 4.0]
        3                           []
        dtype: list
        >>> ds.list.unique().compute() # Order of elements not guaranteed
        0              [1.0, 2.0, nan]
        1                         None
        2                        [4.0]
        3                           []
        dtype: list
        """
        return self.d_series.map_partitions(
            lambda s: s.list.unique(), meta=self.d_series._meta
        )

    def sort_values(
        self,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
    ):
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
        ListColumn with each list sorted

        Notes
        -----
        Difference from pandas:
          * Not supporting: `inplace`, `kind`

        Examples
        --------
        >>> s = cudf.Series([[4, 2, None, 9], [8, 8, 2], [2, 1]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.sort_values(ascending=True, na_position="last").compute()
        0    [2.0, 4.0, 9.0, nan]
        1         [2.0, 8.0, 8.0]
        2              [1.0, 2.0]
        dtype: list
        """
        return self.d_series.map_partitions(
            lambda s: s.list.sort_values(
                ascending, inplace, kind, na_position, ignore_index
            ),
            meta=self.d_series._meta,
        )
