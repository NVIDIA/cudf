# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf.core.accessors.base_accessor import BaseAccessor
from cudf.core.dtypes import CategoricalDtype

if TYPE_CHECKING:
    from cudf.core.column.categorical import CategoricalColumn
    from cudf.core.index import Index
    from cudf.core.series import Series


class CategoricalAccessor(BaseAccessor):
    """
    Accessor object for categorical properties of the Series values.
    Be aware that assigning to `categories` is a inplace operation,
    while all methods return new categorical data per default.

    Parameters
    ----------
    column : Column
    parent : Series or CategoricalIndex

    Examples
    --------
    >>> s = cudf.Series([1,2,3], dtype='category')
    >>> s
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [1, 2, 3]
    >>> s.cat.categories
    Index([1, 2, 3], dtype='int64')
    >>> s.cat.reorder_categories([3,2,1])
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [3, 2, 1]
    >>> s.cat.remove_categories([1])
    0    <NA>
    1       2
    2       3
    dtype: category
    Categories (2, int64): [2, 3]
    >>> s.cat.set_categories(list('abcde'))
    0    <NA>
    1    <NA>
    2    <NA>
    dtype: category
    Categories (5, str): ['a', 'b', 'c', 'd', 'e']
    >>> s.cat.as_ordered()
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [1 < 2 < 3]
    >>> s.cat.as_unordered()
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [1, 2, 3]
    """

    _column: CategoricalColumn

    def __init__(self, parent: Series | Index):
        if not isinstance(parent.dtype, CategoricalDtype):
            raise AttributeError(
                "Can only use .cat accessor with a 'category' dtype"
            )
        super().__init__(parent=parent)

    @property
    def categories(self) -> Index:
        """
        The categories of this categorical.
        """
        return self._column.dtype.categories

    @property
    def codes(self) -> Series:
        """
        Return Series of codes as well as the index.
        """
        from cudf.core.series import Series

        index = (
            self._parent.index if isinstance(self._parent, Series) else None
        )
        codes = self._column.codes.astype(
            self._column.dtype._pandas_codes_dtype
        )
        return Series._from_column(codes, index=index)

    @property
    def ordered(self) -> bool | None:
        """
        Whether the categories have an ordered relationship.
        """
        return self._column.ordered

    def as_ordered(self) -> Series | Index | None:
        """
        Set the Categorical to be ordered.

        Returns
        -------
        Categorical
            Ordered Categorical.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.as_ordered()
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1 < 2 < 10]
        """
        return self._return_or_inplace(self._column.as_ordered(ordered=True))

    def as_unordered(self) -> Series | Index | None:
        """
        Set the Categorical to be unordered.

        Returns
        -------
        Categorical
            Unordered Categorical or None if inplace.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s = s.cat.as_ordered()
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1 < 2 < 10]
        >>> s.cat.as_unordered()
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        """
        return self._return_or_inplace(self._column.as_ordered(ordered=False))

    def add_categories(self, new_categories: Any) -> Series | Index | None:
        """
        Add new categories.

        `new_categories` will be included at the last/highest
        place in the categories and will be unused directly
        after this call.

        Parameters
        ----------
        new_categories : category or list-like of category
            The new categories to be included.

        Returns
        -------
        cat
            Categorical with new categories added.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 2], dtype="category")
        >>> s
        0    1
        1    2
        dtype: category
        Categories (2, int64): [1, 2]
        >>> s.cat.add_categories([0, 3, 4])
        0    1
        1    2
        dtype: category
        Categories (5, int64): [1, 2, 0, 3, 4]
        >>> s
        0    1
        1    2
        dtype: category
        Categories (2, int64): [1, 2]
        """
        return self._return_or_inplace(
            self._column.add_categories(new_categories=new_categories)
        )

    def remove_categories(
        self,
        removals: Any,
    ) -> Series | Index | None:
        """
        Remove the specified categories.

        `removals` must be included in the
        old categories. Values which were in the
        removed categories will be set to null.

        Parameters
        ----------
        removals : category or list-like of category
            The categories which should be removed.

        Returns
        -------
        cat
            Categorical with removed categories

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.remove_categories([1])
        0      10
        1    <NA>
        2    <NA>
        3       2
        4      10
        5       2
        6      10
        dtype: category
        Categories (2, int64): [2, 10]
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        """
        return self._return_or_inplace(
            self._column.remove_categories(removals=removals)
        )

    def set_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
        rename: bool = False,
    ) -> Series | Index | None:
        """
        Set the categories to the specified new_categories.


        `new_categories` can include new categories (which
        will result in unused categories) or remove old categories
        (which results in values set to null). If `rename==True`,
        the categories will simple be renamed (less or more items
        than in old categories will result in values set to null or
        in unused categories respectively).

        This method can be used to perform more than one action
        of adding, removing, and reordering simultaneously and
        is therefore faster than performing the individual steps
        via the more specialised methods.

        On the other hand this methods does not do checks
        (e.g., whether the old categories are included in the
        new categories on a reorder), which can result in
        surprising changes.

        Parameters
        ----------
        new_categories : list-like
            The categories in new order.
        ordered : bool, default None
            Whether or not the categorical is treated as
            a ordered categorical. If not given, do
            not change the ordered information.
        rename : bool, default False
            Whether or not the `new_categories` should be
            considered as a rename of the old categories
            or as reordered categories.

        Returns
        -------
        cat
            Categorical with reordered categories

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 1, 2, 10, 2, 10], dtype='category')
        >>> s
        0     1
        1     1
        2     2
        3    10
        4     2
        5    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.set_categories([1, 10])
        0       1
        1       1
        2    <NA>
        3      10
        4    <NA>
        5      10
        dtype: category
        Categories (2, int64): [1, 10]
        """
        return self._return_or_inplace(
            self._column.set_categories(
                new_categories=new_categories, ordered=ordered, rename=rename
            )
        )

    def reorder_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
    ) -> Series | Index | None:
        """
        Reorder categories as specified in new_categories.

        `new_categories` need to include all old categories
        and no new category items.

        Parameters
        ----------
        new_categories : Index-like
            The categories in new order.
        ordered : bool, optional
            Whether or not the categorical is treated
            as a ordered categorical. If not given, do
            not change the ordered information.

        Returns
        -------
        cat
            Categorical with reordered categories

        Raises
        ------
        ValueError
            If the new categories do not contain all old
            category items or any new ones.


        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.reorder_categories([10, 1, 2])
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [10, 1, 2]
        >>> s.cat.reorder_categories([10, 1])
        Traceback (most recent call last):
        ...
        ValueError: items in new_categories are not the same as in old categories
        """
        return self._return_or_inplace(
            self._column.reorder_categories(new_categories, ordered=ordered),
        )
