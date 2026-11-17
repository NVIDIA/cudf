# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import cudf
from cudf.core.mixins import NotIterable

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cudf.core.index import Index
    from cudf.core.series import Series


class BaseAccessor(NotIterable):
    _parent: Series | Index

    def __init__(self, parent: Series | Index):
        self._parent = parent
        self._column = self._parent._column

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: Literal[True],
        expand: bool = False,
        retain_index: bool = True,
        replace_name: Hashable | None = None,
    ) -> None: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: Literal[False],
        expand: bool = False,
        retain_index: bool = True,
        replace_name: Hashable | None = None,
    ) -> Series | Index: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        expand: bool = False,
        retain_index: bool = True,
        replace_name: Hashable | None = None,
    ) -> Series | Index: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: bool = False,
        expand: bool = False,
        retain_index: bool = True,
        replace_name: Hashable | None = None,
    ) -> Series | Index | None: ...

    def _return_or_inplace(  # type: ignore[misc]
        self,
        new_col,
        inplace: bool = False,
        expand: bool = False,
        retain_index: bool = True,
        replace_name: Hashable | None = None,
    ):
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """
        result_name = (
            self._parent.name if replace_name is None else replace_name
        )
        if inplace:
            self._parent._mimic_inplace(
                type(self._parent)._from_column(new_col, name=result_name),
                inplace=True,
            )
            return None
        else:
            if expand:
                # This branch indicates the passed as new_col
                # is a Table
                table = new_col

                if isinstance(self._parent, cudf.Index):
                    idx = self._parent._constructor_expanddim._from_data(table)
                    idx.names = None
                    return idx
                else:
                    df = self._parent._constructor_expanddim._from_data(
                        data=table,
                        index=self._parent.index,  # type: ignore[union-attr]
                        attrs=self._parent.attrs,  # type: ignore[union-attr]
                    )
                    keys = (
                        tuple(table.keys()) if isinstance(table, dict) else ()
                    )
                    if len(table) == 0 or (
                        keys and keys == tuple(range(len(keys)))
                    ):
                        df._data.rangeindex = True
                    return df
            elif isinstance(self._parent, cudf.Series):
                return cudf.Series._from_column(
                    new_col,
                    name=result_name,
                    index=self._parent.index if retain_index else None,
                    attrs=self._parent.attrs,
                )
            elif isinstance(self._parent, cudf.Index):
                return cudf.Index._from_column(new_col, name=result_name)
            else:
                return self._parent._mimic_inplace(new_col, inplace=False)

    def __setattr__(self, key, value):
        if key in {"_parent", "_column"}:
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"You cannot add any new attribute '{key}'")
