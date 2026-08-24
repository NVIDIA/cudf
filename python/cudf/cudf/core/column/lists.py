# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core.column.column import (
    ColumnBase,
    ColumnList,
    PylibcudfFunction,
    as_column,
    column_empty,
    int32_same_kind_policy,
    same_dtype_policy,
)
from cudf.core.dtype.conversions import element_type_from_list_dtype
from cudf.core.dtype.validators import is_dtype_obj_list
from cudf.core.dtypes import ListDtype
from cudf.core.missing import NA
from cudf.utils.dtypes import (
    dtype_from_pylibcudf_column,
    get_dtype_of_same_kind,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cudf._typing import ColumnBinaryOperand, ColumnLike, DtypeObj
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn


class ListColumn(ColumnBase):
    _VALID_BINARY_OPERATIONS = {"__add__", "__radd__", "__eq__", "__ne__"}

    def _get_sliced_child(self) -> ColumnBase:
        """Get a child column properly sliced to match the parent's view."""
        sliced_plc_col = self.plc_column.list_view().get_sliced_child()
        return ColumnBase.create(sliced_plc_col, self.element_type)

    def _prep_pandas_compat_repr(
        self, nan_rep: str | None = None
    ) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        # TODO: handle if self.has_nulls(): case
        return self

    def _cast_setitem_value(self, value: Any) -> plc.Scalar:
        pa_type = (
            self.dtype.to_arrow()
            if isinstance(self.dtype, ListDtype)
            else cast("pd.ArrowDtype", self.dtype).pyarrow_dtype
        )
        if isinstance(value, list) or value is None:
            return pa_scalar_to_plc_scalar(pa.scalar(value, type=pa_type))
        elif value is NA or value is None:
            return pa_scalar_to_plc_scalar(pa.scalar(None, type=pa_type))
        else:
            raise ValueError(f"Can not set {value} into ListColumn")

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        # Lists only support __add__, which concatenates lists.
        _reflect, op = self._check_reflected_op(op)
        if not isinstance(other, type(self)):
            return NotImplemented
        if is_dtype_obj_list(other.dtype):
            if op == "__add__":
                return self.concatenate_rows([other])
            else:
                raise NotImplementedError(
                    "Lists concatenation for this operation is not yet"
                    "supported"
                )
        else:
            raise TypeError("can only concatenate list to list")

    @property
    def elements(self) -> ColumnBase:
        """Column containing the elements of each list (may itself be a ListColumn)"""
        return self._get_sliced_child()

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Lists are not yet supported via `__cuda_array_interface__`"
        )

    def leaves(self) -> ColumnBase:
        if isinstance(self.elements, ListColumn):
            return self.elements.leaves()
        else:
            return self.elements

    @classmethod
    def from_sequences(cls, arbitrary: Sequence[ColumnLike]) -> Self:
        """
        Create a list column for list of column-like sequences
        """
        data_col = column_empty(0)
        mask_bools = []
        offset_vals = [0]
        offset = 0

        # Build Data, Mask & Offsets
        for data in arbitrary:
            if is_na_like(data):
                mask_bools.append(False)
                offset_vals.append(offset)
            else:
                mask_bools.append(True)
                data_col = data_col.append(as_column(data))
                offset += len(data)
                offset_vals.append(offset)

        offset_col = plc.Column.from_iterable_of_py(
            offset_vals, dtype=plc.types.SIZE_TYPE
        )
        data_plc_col = data_col.plc_column
        mask, null_count = plc.transform.bools_to_mask(
            plc.Column.from_iterable_of_py(mask_bools)
        )
        plc_column = plc.Column(
            plc.DataType(plc.TypeId.LIST),
            len(offset_vals) - 1,
            None,
            mask,
            null_count,
            0,
            [offset_col, data_plc_col],
        )
        return cast(
            "Self",
            ColumnBase.create(plc_column, ListDtype(data_col.dtype)),
        )

    @cached_property
    def _string_separators(self) -> plc.Column:
        # Separator strings to match the Python format
        return plc.Column.from_iterable_of_py(
            [", ", "[", "]"], dtype=plc.DataType(plc.TypeId.STRING)
        )

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        """
        Create a strings column from a list column
        """
        if cudf.get_option("mode.pandas_compatible"):
            if isinstance(dtype, np.dtype) and dtype.kind == "O":
                raise TypeError(
                    f"Cannot cast a list from {self.dtype} to {dtype}"
                )
        if isinstance(dtype, np.dtype) and dtype.kind == "U":
            dtype = np.dtype("object")
        lc = self._transform_leaves(
            lambda col, dtype: col.as_string_column(dtype), dtype
        )

        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_lists.format_list_column(
                lc.plc_column,
                plc.Scalar.from_py(
                    "None", dtype=plc.DataType(plc.TypeId.STRING)
                ),
                self._string_separators,
            )
            # format_list_column converts top-level nulls to the na_rep
            # string ("None"). Re-apply the original null mask so that
            # top-level nulls remain as actual nulls, matching pandas.
            if self.null_count > 0:
                plc_column = plc_column.with_mask(
                    self.plc_column.null_mask(),
                    self.null_count,
                )
            return cast(
                "cudf.core.column.string.StringColumn",
                ColumnBase.create(plc_column, dtype),
            )

    def _transform_leaves(
        self, func: Callable[[ColumnBase, DtypeObj], ColumnBase], *args: Any
    ) -> Self:
        """
        Return a new column like Self but with func applied to the last leaf column.
        """
        # Store metadata for reconstruction: (plc_column, list_view)
        # We need to keep the full plc_column for accessing size, mask, null_count, offset
        leaf_queue: list[plc.Column] = []
        curr_plc_col = self.plc_column
        leaf_dtype = self.dtype

        while is_dtype_obj_list(leaf_dtype):
            leaf_queue.append(curr_plc_col)
            curr_plc_col = curr_plc_col.list_view().child()
            leaf_dtype = (
                leaf_dtype.element_type
                if isinstance(leaf_dtype, ListDtype)
                else pd.ArrowDtype(
                    cast("pd.ArrowDtype", leaf_dtype).pyarrow_dtype.value_type
                )
            )

        # Apply the transformation to the leaf column
        # TODO: For now we convert plc.Column to ColumnBase for the func, then back to
        # plc.Column, but we should be able to eventually avoid this double conversion.
        leaf_col_base = ColumnBase.create(curr_plc_col, leaf_dtype)
        transformed_leaf = func(leaf_col_base, *args)
        plc_leaf_col = transformed_leaf.plc_column

        # Rebuild the list column hierarchy from leaf back to root
        result_dtype = transformed_leaf.dtype
        while leaf_queue:
            parent_plc_col = leaf_queue.pop()
            offsets = parent_plc_col.list_view().offsets()
            # parent_plc_col.null_mask() is a Span which is Span-compliant
            plc_leaf_col = plc.Column(
                plc.DataType(plc.TypeId.LIST),
                parent_plc_col.size(),
                None,
                parent_plc_col.null_mask(),
                parent_plc_col.null_count(),
                parent_plc_col.offset(),
                [offsets, plc_leaf_col],
            )
            result_dtype = (
                ListDtype(result_dtype)
                if not isinstance(result_dtype, pd.ArrowDtype)
                else pd.ArrowDtype(pa.list_(result_dtype.pyarrow_dtype))
            )
        return cast(
            "Self",
            ColumnBase.create(plc_leaf_col, result_dtype),
        )

    @cached_property
    def element_type(self) -> DtypeObj:
        """
        Returns the element type of the list column.
        """
        return element_type_from_list_dtype(self.dtype)

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type or isinstance(self.dtype, pd.ArrowDtype):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        else:
            return pd.Index(self.to_arrow().tolist(), dtype="object")

    def count_elements(self) -> ColumnBase:
        return PylibcudfFunction(
            plc.lists.count_elements,
            int32_same_kind_policy,
        ).execute_with_args(self)

    def distinct(self, nulls_equal: bool, nans_all_equal: bool) -> ColumnBase:
        if is_dtype_obj_list(self.element_type):
            raise NotImplementedError(
                "distinct is not supported for nested lists."
            )
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.distinct(
                    self.plc_column,
                    (
                        plc.types.NullEquality.EQUAL
                        if nulls_equal
                        else plc.types.NullEquality.UNEQUAL
                    ),
                    (
                        plc.types.NanEquality.ALL_EQUAL
                        if nans_all_equal
                        else plc.types.NanEquality.UNEQUAL
                    ),
                ),
                self.dtype,
            )

    def sort_lists(
        self, ascending: bool, na_position: Literal["first", "last"]
    ) -> ColumnBase:
        if is_dtype_obj_list(self.element_type):
            raise NotImplementedError(
                "sort_lists is not supported for nested lists."
            )
        if na_position not in {"first", "last"}:
            raise ValueError(
                f"`na_position` must be 'first' or 'last', not {na_position}."
            )
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.sort_lists(
                    self.plc_column,
                    plc.types.Order.ASCENDING
                    if ascending
                    else plc.types.Order.DESCENDING,
                    (
                        plc.types.NullOrder.BEFORE
                        if na_position == "first"
                        else plc.types.NullOrder.AFTER
                    ),
                    False,
                ),
                self.dtype,
            )

    def extract_element_scalar(self, index: int) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.extract_list_element(
                    self.plc_column,
                    index,
                ),
                self.element_type,
            )

    def extract_element_column(self, index: ColumnBase) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.extract_list_element(
                    self.plc_column,
                    index.plc_column,
                ),
                self.element_type,
            )

    def contains_scalar(self, search_key: pa.Scalar) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.lists.contains(
                self.plc_column,
                pa_scalar_to_plc_scalar(search_key),
            )
            return ColumnBase.create(
                plc_result,
                get_dtype_of_same_kind(
                    self.dtype, dtype_from_pylibcudf_column(plc_result)
                ),
            )

    def index_of_scalar(self, search_key: pa.Scalar) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.lists.index_of(
                self.plc_column,
                pa_scalar_to_plc_scalar(search_key),
                plc.lists.DuplicateFindOption.FIND_FIRST,
            )
            return ColumnBase.create(
                plc.lists.index_of(
                    self.plc_column,
                    pa_scalar_to_plc_scalar(search_key),
                    plc.lists.DuplicateFindOption.FIND_FIRST,
                ),
                get_dtype_of_same_kind(
                    self.dtype, dtype_from_pylibcudf_column(plc_result)
                ),
            )

    def index_of_column(self, search_keys: ColumnBase) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.lists.index_of(
                self.plc_column,
                search_keys.plc_column,
                plc.lists.DuplicateFindOption.FIND_FIRST,
            )
            return ColumnBase.create(
                plc.lists.index_of(
                    self.plc_column,
                    search_keys.plc_column,
                    plc.lists.DuplicateFindOption.FIND_FIRST,
                ),
                get_dtype_of_same_kind(
                    self.dtype, dtype_from_pylibcudf_column(plc_result)
                ),
            )

    def concatenate_rows(self, other_columns: list[ColumnBase]) -> ColumnBase:
        return PylibcudfFunction(
            plc.lists.concatenate_rows,
            same_dtype_policy,
        ).execute_with_args(
            ColumnList(*itertools.chain([self], other_columns))
        )

    def concatenate_list_elements(self, dropna: bool) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.lists.concatenate_list_elements(
                    self.plc_column,
                    plc.lists.ConcatenateNullPolicy.IGNORE
                    if dropna
                    else plc.lists.ConcatenateNullPolicy.NULLIFY_OUTPUT_ROW,
                ),
                self.element_type,
            )

    def segmented_gather(self, gather_map: ColumnBase) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            try:
                plc_result = plc.lists.segmented_gather(
                    self.plc_column,
                    gather_map.plc_column,
                )
            except (TypeError, RuntimeError, ValueError) as e:
                raise ValueError(
                    f"input must be list type with length {len(self)} "
                    "and only integer elements"
                ) from e
            return ColumnBase.create(
                plc_result,
                self.dtype,
            )

    def join_list_elements(
        self,
        separator: str | StringColumn,
        sep_na_rep: str,
        string_na_rep: str,
        result_dtype: DtypeObj,
    ) -> StringColumn:
        with self.access(mode="read", scope="internal"):
            if isinstance(separator, str):
                sep: plc.Scalar | plc.Column = plc.Scalar.from_py(
                    separator, dtype=plc.DataType(plc.TypeId.STRING)
                )
            else:
                sep = separator.plc_column
            plc_column = plc.strings.combine.join_list_elements(
                self.plc_column,
                sep,
                plc.Scalar.from_py(
                    sep_na_rep, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                plc.Scalar.from_py(
                    string_na_rep, dtype=plc.DataType(plc.TypeId.STRING)
                ),
                plc.strings.combine.SeparatorOnNulls.YES,
                plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT,
            )
            return cast(
                "cudf.core.column.string.StringColumn",
                ColumnBase.create(plc_column, result_dtype),
            )

    def minhash_ngrams(
        self,
        width: int,
        seed: int | np.uint32,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint32 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint32).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                    )
                seed = np.uint32(seed)
            return cast(
                "Self",
                ColumnBase.create(
                    plc.nvtext.minhash.minhash_ngrams(
                        self.plc_column,
                        width,
                        seed,
                        a.plc_column,
                        b.plc_column,
                    ),
                    ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint32))
                    ),
                ),
            )

    def minhash64_ngrams(
        self,
        width: int,
        seed: int | np.uint64,
        a: NumericalColumn,
        b: NumericalColumn,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint64 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint64).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint64).max}]"
                    )
                seed = np.uint64(seed)
            return cast(
                "Self",
                ColumnBase.create(
                    plc.nvtext.minhash.minhash64_ngrams(
                        self.plc_column,
                        width,
                        seed,
                        a.plc_column,
                        b.plc_column,
                    ),
                    ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint64))
                    ),
                ),
            )
