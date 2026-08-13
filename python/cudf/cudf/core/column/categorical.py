# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core.column.column import (
    ColumnBase,
    ColumnList,
    PylibcudfFunction,
    as_column,
    column_empty,
    concat_columns,
    same_dtype_policy,
)
from cudf.core.column.utils import access_columns
from cudf.core.dtypes import CategoricalDtype, IntervalDtype
from cudf.utils.dtypes import (
    SIZE_TYPE_DTYPE,
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
    find_common_type,
    is_mixed_with_object_dtype,
    min_signed_type,
    min_unsigned_type,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableSequence, Sequence

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column import (
        DatetimeColumn,
        NumericalColumn,
        StringColumn,
        TimeDeltaColumn,
    )


# Using np.int8(-1) to allow silent wrap-around when casting to uint
# it may make sense to make this dtype specific or a function.
_DEFAULT_CATEGORICAL_VALUE = np.int8(-1)


def _sort_column(col: ColumnBase) -> ColumnBase:
    """Sort a column in ascending order with nulls after."""
    return PylibcudfFunction(
        plc.sorting.sort,
        same_dtype_policy,
        result_index=0,
    ).execute_with_args(
        ColumnList(col),
        column_order=[plc.types.Order.ASCENDING],
        null_precedence=[plc.types.NullOrder.AFTER],
    )


class CategoricalColumn(ColumnBase):
    """Implements operations for Columns of Categorical type"""

    dtype: CategoricalDtype
    _VALID_REDUCTIONS = {
        "max",
        "min",
    }
    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
    }

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            encoded = self._encode(item)
        except ValueError:
            return False
        return encoded in self.codes

    def _process_values_for_isin(
        self, values: Sequence | ColumnBase
    ) -> tuple[ColumnBase, ColumnBase]:
        # Convert values to categorical dtype like self
        return self, as_column(values, dtype=self.dtype)

    @property
    def categories(self) -> ColumnBase:
        return self.dtype.categories._column

    @cached_property
    def codes(self) -> NumericalColumn:
        """The integer codes representing each category.

        This is a NumericalColumn wrapping self.plc_column, which is necessary because
        many operations on categoricals need to delegate to the codes column.
        """
        return cast(
            "NumericalColumn",
            cudf.core.column.NumericalColumn.create(
                self.plc_column, self.dtype._codes_dtype, validate=False
            ),
        )

    @property
    def ordered(self) -> bool | None:
        return self.dtype.ordered

    def __setitem__(self, key: Any, value: Any) -> None:
        # Ensure that we are not setting a non-existent category.
        val_is_scalar = is_scalar(value)
        if not val_is_scalar or not is_na_like(value):
            length = 1 if val_is_scalar else None
            arr = as_column(value, length=length, nan_as_null=False)
            if isinstance(arr, CategoricalColumn):
                arr = arr._get_decategorized_column()
            if not arr.is_all_null:
                if arr.dtype != self.categories.dtype:
                    arr = arr.astype(self.categories.dtype)
                with arr.access(mode="read", scope="internal"):
                    arr_plc = arr.plc_column
                    if arr_plc.null_count() > 0:
                        (arr_plc,) = plc.stream_compaction.drop_nulls(
                            plc.Table([arr_plc]), [0], 1
                        ).columns()
                    if arr_plc.size() > 0:
                        with self.categories.access(
                            mode="read", scope="internal"
                        ):
                            contains_col = plc.search.contains(
                                self.categories.plc_column,
                                arr_plc,
                            )
                        contains_all = plc.reduce.reduce(
                            contains_col,
                            plc.aggregation.all(),
                            dtype_to_pylibcudf_type(np.dtype("bool")),
                        )
                        if not contains_all.to_py():
                            raise TypeError(
                                "Cannot setitem on a Categorical with a new "
                                "category, set the categories first"
                            )

        if val_is_scalar:
            value = self._encode(value) if value is not None else value
        else:
            value = cast(
                "CategoricalColumn", as_column(value).astype(self.dtype)
            )
            value = value.codes
        codes = self.codes
        codes[key] = value
        self.plc_column = codes.plc_column
        self._clear_cache()

    def _fill(
        self,
        fill_value: plc.Scalar,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Self | None:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        fill_code = self._encode(fill_value.to_arrow())
        fill_code_scalar = pa_scalar_to_plc_scalar(pa.scalar(fill_code))

        return super()._fill(fill_code_scalar, begin, end, inplace)

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        """Custom reduction for categorical columns - delegates to codes."""
        # Only valid reductions are min and max
        if not self.ordered:
            raise TypeError(
                f"Categorical is not ordered for operation {op}. "
                "You can use .as_ordered() to change the Categorical "
                "to an ordered one."
            )

        # Delegate to underlying codes column via public method
        result = getattr(self.codes, op)(
            skipna=skipna, min_count=min_count, **kwargs
        )

        # Decode the result back to a category
        return self._decode(result)

    def all(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> bool:
        """Categorical columns don't support all() reduction."""
        raise TypeError(
            f"'{type(self).__name__}' with dtype {self.dtype} "
            f"does not support reduction 'all'"
        )

    def any(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> bool:
        """Categorical columns don't support any() reduction."""
        raise TypeError(
            f"'{type(self).__name__}' with dtype {self.dtype} "
            f"does not support reduction 'any'"
        )

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        if isinstance(other, ColumnBase):
            if isinstance(other, CategoricalColumn):
                if self.dtype == other.dtype:
                    # Dtypes are the same, but ordering may differ
                    # in which case we need to align them for their codes
                    # to match.
                    if not self.dtype._internal_eq(other.dtype):
                        other = other.as_categorical_column(self.dtype)
                else:
                    raise TypeError(
                        "Categoricals can only compare with the same type"
                    )
            # We'll compare self's decategorized values later for non-CategoricalColumn
        else:
            encoded = self._encode(other)
            if isinstance(encoded, np.generic):
                encoded = encoded.item()
            plc_scalar = plc.Scalar.from_py(
                encoded,
                dtype_to_pylibcudf_type(self.dtype._codes_dtype),
            )
            plc_col = plc.Column.from_scalar(plc_scalar, len(self))
            other = cast(
                "CategoricalColumn",
                ColumnBase.create(plc_col, self.dtype),
            )
        equality_ops = {"__eq__", "__ne__", "NULL_EQUALS", "NULL_NOT_EQUALS"}
        if not self.ordered and op not in equality_ops:
            raise TypeError(
                "The only binary operations supported by unordered "
                "categorical columns are equality and inequality."
            )
        if not isinstance(other, CategoricalColumn):
            if op not in equality_ops:
                raise TypeError(
                    f"Cannot compare a Categorical for op {op} with a "
                    "non-categorical type. If you want to compare values, "
                    "decategorize the Categorical first."
                )
            elif op not in equality_ops.union(
                {"__gt__", "__lt__", "__ge__", "__le__"}
            ):
                # TODO: Other non-comparison ops may raise or be supported
                return NotImplemented
            return self._get_decategorized_column()._binaryop(other, op)
        return self.codes._binaryop(other.codes, op)

    def element_indexing(self, index: int) -> ScalarLike:
        val = super().element_indexing(index)
        if isinstance(val, int):
            return self._decode(val)
        return val

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise TypeError(
            "Categorical does not support `__cuda_array_interface__`."
            " Please consider using `.codes` or `.categories`"
            " if you need this functionality."
        )

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if nullable:
            raise NotImplementedError(f"{nullable=} is not supported.")
        if arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not supported.")

        codes = (
            self.codes.astype(self.dtype._pandas_codes_dtype)
            .fillna(_DEFAULT_CATEGORICAL_VALUE)
            .to_numpy()
        )

        if self.categories.dtype.kind == "f":
            # NaN categories need to be mapped to "missing" (-1) in pandas.
            codes[~self.notnull().fillna(False).to_numpy()] = (
                _DEFAULT_CATEGORICAL_VALUE
            )

        col = self

        cats = col.categories.nans_to_nulls()
        if not isinstance(cats.dtype, IntervalDtype):
            # leaving out dropna because it temporarily changes an interval
            # index into a struct and throws off results.
            # TODO: work on interval index dropna
            cats = cats.dropna()
        data = pd.Categorical.from_codes(
            codes, categories=cats.to_pandas(), ordered=col.ordered
        )
        return pd.Index(data)

    def to_arrow(self) -> pa.Array:
        # match pandas which uses signed codes
        signed_type = (
            min_signed_type(self.codes.max())
            if self.size > 0
            else np.dtype(np.int8)
        )
        assert self.ordered is not None, (
            "Categorical 'ordered' attribute must be set to convert to Arrow"
        )
        return pa.DictionaryArray.from_arrays(
            self.codes.astype(signed_type).to_arrow(),
            self.categories.to_arrow(),
            ordered=self.ordered,
        )

    def clip(self, lo: ScalarLike, hi: ScalarLike) -> Self:
        return (
            self.astype(self.categories.dtype).clip(lo, hi).astype(self.dtype)  # type: ignore[return-value]
        )

    def where(
        self, cond: ColumnBase, other: ScalarLike | ColumnBase, inplace: bool
    ) -> ColumnBase:
        if is_scalar(other):
            try:
                other = self._encode(other)
            except ValueError:
                # When other is not present in categories,
                # fill with Null.
                other = None
            other = pa_scalar_to_plc_scalar(
                pa.scalar(
                    other,
                    type=cudf_dtype_to_pa_type(self.dtype._codes_dtype),
                )
            )
        elif isinstance(other.dtype, CategoricalDtype):
            other = other.codes  # type: ignore[union-attr]

        with access_columns(
            self.codes,
            other,
            cond,
            mode="read",
            scope="internal",
        ) as (codes, other, cond):
            other_plc = (
                other.plc_column if isinstance(other, ColumnBase) else other
            )
            result_plc = plc.copying.copy_if_else(
                codes.plc_column,
                other_plc,
                cond.plc_column,
            )
        return ColumnBase.create(result_plc, self.dtype)

    def _encode(self, value: ScalarLike) -> ScalarLike:
        return self.categories.find_first_value(value)

    def _decode(self, value: int) -> ScalarLike:
        if value == _DEFAULT_CATEGORICAL_VALUE:
            return None
        return self.categories.element_indexing(value)

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        """Return col with *to_replace* replaced with *replacement*."""
        to_replace_col, replacement_col = (
            ColumnBase._prepare_find_and_replace_columns(
                to_replace,
                replacement,
                null_cast_dtype=self.categories.dtype,
            )
        )
        old_plc, new_plc = ColumnBase._dedupe_find_and_replace_mapping(
            to_replace_col, replacement_col
        )

        replaced = self
        if old_plc.null_count() == 1:
            # Get the replacement value for the null in old_col
            old_isnull_plc = plc.unary.is_null(old_plc)
            filtered_table = plc.stream_compaction.apply_boolean_mask(
                plc.Table([new_plc]), old_isnull_plc
            )
            # We know there's exactly 1 null, so filtered result has 1 row
            fill_value = filtered_table.columns()[0].to_scalar().to_py()

            # The `in` operator will only work on certain column types
            # (NumericalColumn, StringColumn).
            if fill_value not in self.categories:  # type: ignore[operator]
                replaced = replaced.add_categories(fill_value)
            replaced = replaced.fillna(fill_value)

            # Once we've replaced self's values with the fill_value, we can drop the
            # mapping from old_plc and new_plc to simplify the rest of the logic
            old_plc, new_plc = plc.stream_compaction.drop_nulls(
                plc.Table([old_plc, new_plc]), [0], 1
            ).columns()

        if old_plc.size() == 0:
            if replaced.dtype != self.dtype:
                raise TypeError(
                    f"Cannot setitem on a Categorical with a new"
                    f" category ({replaced.dtype}), set the"
                    " categories first"
                )
            return replaced.copy()

        remaining_to_replace = ColumnBase.create(old_plc, to_replace_col.dtype)
        if not replaced.categories.isin(remaining_to_replace).any():
            if replaced.dtype != self.dtype:
                raise TypeError(
                    f"Cannot setitem on a Categorical with a new"
                    f" category ({replaced.dtype}), set the"
                    " categories first"
                )
            return replaced.copy()

        if new_plc.null_count() > 0:
            # Any value mapped to null is dropped in the result
            new_isnull_plc = plc.unary.is_null(new_plc)
            filtered_table = plc.stream_compaction.apply_boolean_mask(
                plc.Table([old_plc]), new_isnull_plc
            )
            drop_values = ColumnBase.create(
                filtered_table.columns()[0], to_replace_col.dtype
            )
            cur_categories = replaced.categories
            new_categories = cur_categories.apply_boolean_mask(
                cur_categories.isin(drop_values).unary_operator("not")
            )
            replaced = replaced._set_categories(new_categories)

            old_plc, new_plc = plc.stream_compaction.drop_nulls(
                plc.Table([old_plc, new_plc]), [1], 1
            ).columns()

        # Construct the new categorical labels. If either the set of values to replace
        # or the replacement values columns match the current column's dtype, we must
        # attempt to reconcile the new category values. If a category is being replaced
        # by an existing one, we want to map it to None. If it's totally new, we want to
        # map it to the new label it is to be replaced by.
        cats_col = replaced.dtype.categories._column
        with cats_col.access(mode="read", scope="internal"):
            if (
                to_replace_col.dtype == cats_col.dtype
                or replacement_col.dtype == cats_col.dtype
            ):
                null_scalar = plc.Scalar.from_py(None, new_plc.type())
                is_in_cats = plc.search.contains(cats_col.plc_column, new_plc)
                cats_replace_plc = plc.replace.find_and_replace_all(
                    cats_col.plc_column,
                    old_plc,
                    new_plc,
                )
                dtype_replace_plc = plc.copying.copy_if_else(
                    null_scalar,
                    new_plc,
                    is_in_cats,
                )
                new_cats_plc = plc.replace.find_and_replace_all(
                    cats_col.plc_column,
                    old_plc,
                    dtype_replace_plc,
                )
            else:
                new_cats_plc = cats_col.plc_column
                cats_replace_plc = cats_col.plc_column

            # Filter out categories that were mapped to None
            (new_cats_plc,) = plc.stream_compaction.drop_nulls(
                plc.Table([new_cats_plc]), [0], 1
            ).columns()

            # Build a mapping from old codes to new codes
            left_gather_map, right_gather_map = plc.join.inner_join(
                plc.Table([cats_replace_plc]),
                plc.Table([new_cats_plc]),
                plc.types.NullEquality.UNEQUAL,
            )

        # Gather from a range column using right_gather_map to get new codes
        new_index_plc = plc.Column.from_iterable_of_py(
            range(new_cats_plc.size())
        )
        (gathered_right,) = plc.copying.gather(
            plc.Table([new_index_plc]),
            right_gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()
        codes_replacement = ColumnBase.create(
            gathered_right, dtype_from_pylibcudf_column(new_index_plc)
        ).astype(replaced.codes.dtype)

        # The left_gather_map contains the old category indices that matched
        # These are the "to_replace" codes
        # left_gather_map is INT32 from pylibcudf, need to cast to codes dtype
        codes_to_replace = ColumnBase.create(
            left_gather_map, dtype_from_pylibcudf_column(left_gather_map)
        ).astype(replaced.codes.dtype)

        with access_columns(
            replaced.codes,
            codes_to_replace,
            codes_replacement,
            mode="read",
            scope="internal",
        ) as (codes, codes_to_replace, codes_replacement):
            new_codes = plc.replace.find_and_replace_all(
                codes.plc_column,
                codes_to_replace.plc_column,
                codes_replacement.plc_column,
            )
        new_cats_col = ColumnBase.create(new_cats_plc, cats_col.dtype)
        result_dtype = CategoricalDtype(new_cats_col, self.dtype.ordered)
        if result_dtype != self.dtype:
            raise TypeError(
                f"Cannot setitem on a Categorical with a new category ({result_dtype}), set the categories first"
            )
        return cast("Self", ColumnBase.create(new_codes, result_dtype))

    def isnull(self) -> ColumnBase:
        """
        Identify missing values in a CategoricalColumn.
        """
        result = super().isnull()

        if self.categories.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of an underlying float column
            categories = self.categories.isnan()
            if categories.any():
                code = self._encode(np.nan)
                result = result | (self.codes == code)

        return result

    def notnull(self) -> ColumnBase:
        """
        Identify non-missing values in a CategoricalColumn.
        """
        result = super().is_valid()

        if self.categories.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of an underlying float column
            categories = self.categories.isnan()
            if categories.any():
                code = self._encode(np.nan)
                result = result & (self.codes != code)

        return result

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value):
            if fill_value != _DEFAULT_CATEGORICAL_VALUE:
                try:
                    fill_value = self._encode(fill_value)
                except ValueError as err:
                    raise ValueError(
                        f"{fill_value=} must be in categories"
                    ) from err
            return pa_scalar_to_plc_scalar(
                pa.scalar(
                    fill_value,
                    type=cudf_dtype_to_pa_type(self.dtype._codes_dtype),
                )
            )
        else:
            fill_value = as_column(fill_value, nan_as_null=False)
            if isinstance(fill_value.dtype, CategoricalDtype):
                if self.dtype != fill_value.dtype:
                    raise TypeError(
                        "Cannot set a categorical with another without identical categories"
                    )
            else:
                raise TypeError(
                    "Cannot set a categorical with non-categorical data"
                )
            fill_value = cast("CategoricalColumn", fill_value)._set_categories(
                self.categories,
            )
            return fill_value.codes.astype(self.dtype._codes_dtype)

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
        return self.codes.indices_of(self._encode(value))

    @cached_property
    def is_monotonic_increasing(self) -> bool:
        return bool(self.ordered) and super().is_monotonic_increasing

    @cached_property
    def is_monotonic_decreasing(self) -> bool:
        return bool(self.ordered) and super().is_monotonic_decreasing

    def as_categorical_column(self, dtype: CategoricalDtype) -> Self:
        if not isinstance(self.categories, type(dtype.categories._column)):
            if isinstance(
                self.categories.dtype, cudf.StructDtype
            ) and isinstance(dtype.categories.dtype, cudf.IntervalDtype):
                return cast(
                    "Self",
                    ColumnBase.create(self.plc_column, dtype),
                )
            else:
                # Otherwise if both categories are of different Column types,
                # return a column full of nulls.
                codes_plc = plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        None,
                        dtype_to_pylibcudf_type(dtype._codes_dtype),
                    ),
                    self.size,
                )
                return cast(
                    "Self",
                    ColumnBase.create(codes_plc, dtype),
                )

        return self.set_categories(
            new_categories=self.dtype.categories
            if dtype._categories is None
            else dtype.categories,
            ordered=bool(dtype.ordered),
        )

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        if (
            isinstance(dtype, np.dtype)
            and dtype.kind in "iu"
            and self.null_count > 0
        ):
            # pandas promotes a null-containing Categorical's categories to
            # float, so converting to a non-nullable integer dtype raises.
            raise ValueError("Cannot convert float NaN to integer")
        return self._get_decategorized_column().as_numerical_column(dtype)

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        if isinstance(dtype, np.dtype) and dtype.kind == "U":
            dtype = np.dtype("object")
        return self._get_decategorized_column().as_string_column(dtype)

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        return self._get_decategorized_column().as_datetime_column(dtype)

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        return self._get_decategorized_column().as_timedelta_column(dtype)

    def _get_decategorized_column(self) -> ColumnBase:
        if self.is_all_null:
            # self.categories is empty; just return codes
            return self.codes
        gather_map = self.codes.astype(SIZE_TYPE_DTYPE).fillna(0)
        out = self.categories.take(gather_map)
        mask = self.mask
        new_null_count = self.null_count
        if self.offset > 0 and mask is not None:
            with mask.access(mode="read", scope="internal"):
                mask = cudf.core.buffer.as_buffer(
                    plc.null_mask.copy_bitmask_from_bitmask(
                        mask,
                        self.offset,
                        mask.size - self.offset,
                    )
                )
                new_null_count = plc.null_mask.null_count(
                    mask,
                    0,
                    mask.size,
                )
        out = out.set_mask(mask, new_null_count)
        return out

    def copy(self, deep: bool = True) -> Self:
        if not deep:
            return cast("Self", super().copy(deep=deep))
        dtype_copy = CategoricalDtype(
            categories=self.categories.copy(),
            ordered=self.ordered,
        )
        return cast(
            "Self",
            ColumnBase.create(self.plc_column.copy(), dtype_copy),
        )

    @cached_property
    def memory_usage(self) -> int:
        return self.categories.memory_usage + self.codes.memory_usage

    @staticmethod
    def _concat(
        objs: MutableSequence[CategoricalColumn],
    ) -> CategoricalColumn:
        # TODO: This function currently assumes it is being called from
        # column.concat_columns, at least to the extent that all the
        # preprocessing in that function has already been done. That should be
        # improved as the concatenation API is solidified.

        # Combine and de-dupe the categories
        cats = concat_columns([o.categories for o in objs]).unique()
        objs = [o._set_categories(cats, is_unique=True) for o in objs]
        result_dtype = CategoricalDtype(categories=cats)
        codes_dtype = result_dtype._codes_dtype
        codes = [o.codes for o in objs]
        expected_plc_dtype = dtype_to_pylibcudf_type(codes_dtype)

        newsize = sum(map(len, codes))
        if newsize > np.iinfo(SIZE_TYPE_DTYPE).max:
            raise MemoryError(
                f"Result of concat cannot have size > {SIZE_TYPE_DTYPE}_MAX"
            )
        elif newsize == 0:
            codes_col = column_empty(0, codes_dtype)
        else:
            normalized_codes = []
            for code in codes:
                if code.plc_column.type() != expected_plc_dtype:
                    normalized_codes.append(
                        cast(
                            "NumericalColumn",
                            ColumnBase.create(
                                plc.unary.cast(
                                    code.plc_column, expected_plc_dtype
                                ),
                                codes_dtype,
                            ),
                        )
                    )
                else:
                    normalized_codes.append(code)
            codes = normalized_codes
            codes_col = concat_columns(codes)

        return cast(
            "Self",
            ColumnBase.create(
                codes_col.plc_column,
                result_dtype,
            ),
        )

    def set_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
        rename: bool = False,
    ) -> Self:
        # See CategoricalAccessor.set_categories.

        ordered = ordered if ordered is not None else self.ordered
        new_categories = as_column(new_categories)

        if isinstance(new_categories, CategoricalColumn):
            new_categories = new_categories.categories

        # when called with rename=True, the pandas behavior is
        # to replace the current category values with the new
        # categories.
        if rename:
            # enforce same length
            if len(new_categories) != len(self.categories):
                raise ValueError(
                    "new_categories must have the same "
                    "number of items as old categories"
                )
            return cast(
                "Self",
                ColumnBase.create(
                    self.plc_column,
                    CategoricalDtype(
                        categories=new_categories, ordered=ordered
                    ),
                ),
            )
        else:
            out_col = self
            if type(out_col.categories) is not type(new_categories):
                # If both categories are of different Column types,
                # return a column full of Nulls.
                new_dtype = CategoricalDtype(
                    categories=new_categories, ordered=ordered
                )
                new_codes_plc = plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        None,
                        dtype_to_pylibcudf_type(new_dtype._codes_dtype),
                    ),
                    self.size,
                )
                out_col = cast(
                    "Self",
                    ColumnBase.create(
                        new_codes_plc,
                        new_dtype,
                    ),
                )
            elif (
                not out_col._categories_equal(new_categories, ordered=True)
                or not self.ordered == ordered
            ):
                out_col = out_col._set_categories(
                    new_categories,
                    ordered=ordered,
                )
        return out_col

    def _categories_equal(
        self, new_categories: ColumnBase, ordered: bool = False
    ) -> bool:
        cur_categories = self.categories
        if len(new_categories) != len(cur_categories):
            return False
        if new_categories.dtype != cur_categories.dtype:
            return False
        # if order doesn't matter, sort before the equals call below
        if not ordered:
            cur_categories = _sort_column(cur_categories)
            new_categories = _sort_column(new_categories)
        return cur_categories.equals(new_categories)

    def _set_categories(
        self,
        new_categories: Any,
        is_unique: bool = False,
        ordered: bool = False,
    ) -> Self:
        """Returns a new CategoricalColumn with the categories set to the
        specified *new_categories*.

        Notes
        -----
        Assumes ``new_categories`` is the same dtype as the current categories
        """
        new_cats = as_column(new_categories)
        if not (is_unique or new_cats.is_unique):
            new_cats = new_cats.unique()

        cur_cats = self.categories
        if cur_cats.equals(new_cats, check_dtypes=True):
            # TODO: Internal usages don't always need a copy; add a copy keyword
            # as_ordered shallow copies
            return self.copy().as_ordered(ordered=ordered)

        # Construct the output dtype now since we may cast the categories to join later
        ordered = ordered if ordered is not None else self.ordered
        result_dtype = CategoricalDtype(categories=new_cats, ordered=ordered)

        cur_codes = self.codes
        out_plc_dtype = dtype_to_pylibcudf_type(
            min_unsigned_type(max(len(cur_cats), len(new_cats)))
        )
        null_scalar = plc.Scalar.from_py(None, out_plc_dtype)

        if cur_codes.null_count == len(cur_codes):
            result_codes = plc.Column.from_scalar(null_scalar, len(cur_codes))
        else:
            # Cast to common dtype for the join
            if cur_cats.dtype != new_cats.dtype:
                common_dtype = find_common_type(
                    [cur_cats.dtype, new_cats.dtype]
                )
                cur_cats = cur_cats.astype(common_dtype)
                new_cats = new_cats.astype(common_dtype)

            # Left join to map old categories to new codes via category values
            with cur_cats.access(mode="read", scope="internal"):
                with new_cats.access(mode="read", scope="internal"):
                    left_map, right_map = plc.join.left_join(
                        plc.Table([cur_cats.plc_column]),
                        plc.Table([new_cats.plc_column]),
                        plc.types.NullEquality.UNEQUAL,
                    )
            new_codes_plc = plc.Column.from_iterable_of_py(
                range(len(new_cats)), dtype=out_plc_dtype
            )
            joined_new_codes_table = plc.copying.gather(
                plc.Table([new_codes_plc]),
                right_map,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            )

            # We must scatter the joins according to the left_map in order to reverse
            # any shuffling of order caused by the join.
            null_target = plc.Column.from_scalar(null_scalar, len(cur_cats))
            matched_new_codes_table = plc.copying.scatter(
                joined_new_codes_table,
                left_map,
                plc.Table([null_target]),
            )
            with cur_codes.access(mode="read", scope="internal"):
                # Applying the mapping to the actual codes is just a gather from
                # matched_new_codes_table using cur_codes as indices, but gather doesn't
                # accept nulls in cur_codes, so if there are nulls we need to replace
                # nulls with a valid index, gather, then restore nulls in the result.
                if cur_codes.null_count == 0:
                    (result_codes,) = plc.copying.gather(
                        matched_new_codes_table,
                        cur_codes.plc_column,
                        plc.copying.OutOfBoundsPolicy.NULLIFY,
                    ).columns()
                else:
                    zero_scalar = plc.Scalar.from_py(
                        0, cur_codes.plc_column.type()
                    )
                    cur_codes_valid = plc.unary.is_valid(cur_codes.plc_column)
                    filled_codes = plc.copying.copy_if_else(
                        cur_codes.plc_column,
                        zero_scalar,
                        cur_codes_valid,
                    )
                    gathered_table = plc.copying.gather(
                        matched_new_codes_table,
                        filled_codes,
                        plc.copying.OutOfBoundsPolicy.NULLIFY,
                    )
                    result_codes = plc.copying.copy_if_else(
                        gathered_table.columns()[0],
                        null_scalar,
                        cur_codes_valid,
                    )

        return cast("Self", ColumnBase.create(result_codes, result_dtype))

    def add_categories(self, new_categories: Any) -> Self:
        old_categories = self.categories
        new_categories = as_column(
            new_categories,
            dtype=old_categories.dtype if len(new_categories) == 0 else None,
        )
        if is_mixed_with_object_dtype(old_categories, new_categories):
            raise TypeError(
                f"cudf does not support adding categories with existing "
                f"categories of dtype `{old_categories.dtype}` and new "
                f"categories of dtype `{new_categories.dtype}`, please "
                f"type-cast new_categories to the same type as "
                f"existing categories."
            )
        common_dtype = find_common_type(
            [old_categories.dtype, new_categories.dtype]
        )

        new_categories = new_categories.astype(common_dtype)
        old_categories = old_categories.astype(common_dtype)

        if old_categories.isin(new_categories).any():
            raise ValueError("new categories must not include old categories")

        new_categories = old_categories.append(new_categories)
        if not self._categories_equal(new_categories):
            return self._set_categories(new_categories)
        return self

    def remove_categories(
        self,
        removals: Any,
    ) -> Self:
        removals = as_column(removals).astype(self.categories.dtype)
        removals_mask = removals.isin(self.categories)

        # ensure all the removals are in the current categories
        # list. If not, raise an error to match Pandas behavior
        if not removals_mask.all():
            raise ValueError("removals must all be in old categories")

        new_categories = self.categories.apply_boolean_mask(
            self.categories.isin(removals).unary_operator("not")
        )
        if not self._categories_equal(new_categories):
            return self._set_categories(new_categories)
        return self

    def reorder_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
    ) -> CategoricalColumn:
        new_categories = as_column(new_categories)
        # Compare new_categories against current categories.
        # Ignore order for comparison because we're only interested
        # in whether new_categories has all the same values as the
        # current set of categories.
        if not self._categories_equal(new_categories, ordered=False):
            raise ValueError(
                "items in new_categories are not the same as in old categories"
            )
        return self._set_categories(new_categories, ordered=ordered)

    def rename_categories(self, new_categories: Any) -> CategoricalColumn:
        raise NotImplementedError(
            "rename_categories is currently not supported."
        )

    def remove_unused_categories(self) -> Self:
        raise NotImplementedError(
            "remove_unused_categories is currently not supported."
        )

    def as_ordered(self, ordered: bool) -> Self:
        if self.dtype.ordered == ordered:
            return self
        return cast(
            "Self",
            ColumnBase.create(
                self.plc_column,
                CategoricalDtype(categories=self.categories, ordered=ordered),
            ),
        )
