# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from cudf.api.types import is_dtype_equal
from cudf.core.dtype.validators import (
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
)
from cudf.utils.dtypes import (
    find_common_type,
    get_dtype_of_same_kind,
)

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase
    from cudf.core.column.categorical import CategoricalColumn
    from cudf.core.dataframe import DataFrame


class _Indexer:
    # Indexer into a column (either a data column or index level).
    #
    # >>> df
    #    a
    # b
    # 4  1
    # 5  2
    # 6  3
    # >>> _Indexer("a", column=True).get(df)  # returns column "a" of df
    # >>> _Indexer("b", index=True).get(df)  # returns index level "b" of df

    def __init__(self, name: Any):
        self.name = name


class _ColumnIndexer(_Indexer):
    def get(self, obj: DataFrame) -> ColumnBase:
        return obj._data[self.name]

    def set(self, obj: DataFrame, value: ColumnBase):
        obj._data.set_by_label(self.name, value)


class _IndexIndexer(_Indexer):
    def get(self, obj: DataFrame) -> ColumnBase:
        return obj.index._data[self.name]

    def set(self, obj: DataFrame, value: ColumnBase):
        if isinstance(obj.index.dtype, CategoricalDtype):
            # We need to update the categories of the index as well as the
            # codes when setting a new column on a categorical index. The
            # categories are stored in the dtype of the index, so we need to
            # update the dtype of the index to reflect the new categories.
            obj.index = obj.index._from_data_like_self({self.name: value})
        else:
            obj.index._data.set_by_label(self.name, value)


def _match_join_keys(
    lcol: ColumnBase, rcol: ColumnBase, how: str
) -> tuple[ColumnBase, ColumnBase]:
    # Casts lcol and rcol to a common dtype for use as join keys. If no casting
    # is necessary, they are returned as is.
    common_type = None

    # cast the keys lcol and rcol to a common dtype
    ltype = lcol.dtype
    rtype = rcol.dtype

    # https://github.com/rapidsai/cudf/issues/9981
    # For outer joins, every output row's key value comes from whichever side
    # is non-empty (the empty side contributes only nulls), so if exactly one
    # side has no rows there is no data whose dtype could actually be lost.
    # Adopt the non-empty side's dtype instead of falling through to the
    # generic promotion logic below, which would otherwise promote e.g. an
    # int64 column paired with an empty object column to float64. Decimal
    # dtypes are excluded so mismatched precision/scale still raises below,
    # matching the behavior when both sides are non-empty.
    if (
        how == "outer"
        and ltype != rtype
        and not isinstance(
            ltype, (Decimal32Dtype, Decimal64Dtype, Decimal128Dtype)
        )
        and not isinstance(
            rtype, (Decimal32Dtype, Decimal64Dtype, Decimal128Dtype)
        )
    ):
        if len(lcol) and not len(rcol):
            return lcol, rcol.astype(ltype)
        elif len(rcol) and not len(lcol):
            return lcol.astype(rtype), rcol

    # if either side is categorical, different logic
    left_is_categorical = isinstance(ltype, CategoricalDtype)
    right_is_categorical = isinstance(rtype, CategoricalDtype)
    if left_is_categorical and right_is_categorical:
        return _match_categorical_dtypes_both(lcol, rcol, how)  # type: ignore[arg-type]
    elif left_is_categorical or right_is_categorical:
        if left_is_categorical:
            if how in {"leftsemi", "leftanti"}:
                # Decategorize both sides and find a common type
                # instead of casting right to left's categorical
                # type, which would introduce spurious nulls for
                # right values not in the left categories.
                common_type = find_common_type(
                    (ltype.categories.dtype, rtype)  # type: ignore[union-attr]
                )
                return lcol.astype(common_type), rcol.astype(common_type)
            # pandas decategorizes the key when only one side is categorical,
            # regardless of the join type.
            common_type = get_dtype_of_same_kind(rtype, ltype.categories.dtype)  # type: ignore[union-attr]
        else:
            common_type = get_dtype_of_same_kind(ltype, rtype.categories.dtype)  # type: ignore[union-attr]
        return lcol.astype(common_type), rcol.astype(common_type)

    if is_dtype_equal(ltype, rtype):
        return lcol, rcol

    if isinstance(
        ltype, (Decimal32Dtype, Decimal64Dtype, Decimal128Dtype)
    ) or isinstance(rtype, (Decimal32Dtype, Decimal64Dtype, Decimal128Dtype)):
        raise TypeError(
            "Decimal columns can only be merged with decimal columns "
            "of the same precision and scale"
        )

    if (
        is_dtype_obj_numeric(ltype)
        and is_dtype_obj_numeric(rtype)
        and not (ltype.kind == "m" or rtype.kind == "m")
    ):
        if ltype.kind == rtype.kind:
            try:
                common_type = max(ltype, rtype)
            except TypeError:
                # e.g. numpy bool and pandas BooleanDtype both have kind "b"
                # but are not orderable via ``>``; fall back to the
                # type-promotion helper that understands extension dtypes.
                common_type = find_common_type((ltype, rtype))
        else:
            common_type = find_common_type((ltype, rtype))
    elif (ltype.kind == "M" and rtype.kind == "M") or (
        ltype.kind == "m" and rtype.kind == "m"
    ):
        common_type = max(ltype, rtype)
    elif ltype.kind in "mM" and not rcol.fillna(0).can_cast_safely(ltype):
        raise TypeError(
            f"Cannot join between {ltype} and {rtype}, please type-cast both "
            "columns to the same type."
        )
    elif rtype.kind in "mM" and not lcol.fillna(0).can_cast_safely(rtype):
        raise TypeError(
            f"Cannot join between {rtype} and {ltype}, please type-cast both "
            "columns to the same type."
        )

    if how == "left" and rcol.fillna(0).can_cast_safely(ltype):
        return lcol, rcol.astype(ltype)
    if common_type is None:
        if is_dtype_obj_string(ltype) and is_dtype_obj_string(rtype):
            common_type = find_common_type((ltype, rtype))
        else:
            common_type = np.dtype(np.float64)
    return lcol.astype(common_type), rcol.astype(common_type)


def _match_categorical_dtypes_both(
    lcol: CategoricalColumn, rcol: CategoricalColumn, how: str
) -> tuple[ColumnBase, ColumnBase]:
    ltype, rtype = lcol.dtype, rcol.dtype

    # when both are ordered and both have the same categories,
    # no casting required:
    if ltype._internal_eq(rtype):
        return lcol, rcol

    # Merging categorical variables when only one side is ordered is
    # ambiguous and not allowed.
    if ltype.ordered != rtype.ordered:
        raise TypeError(
            "Merging on categorical variables with mismatched"
            " ordering is ambiguous"
        )

    if ltype.ordered and rtype.ordered:
        # if we get to here, categories must be what causes the
        # dtype equality check to fail. And we can never merge
        # two ordered categoricals with different categories
        raise TypeError(
            f"{how} merge between categoricals with "
            "different categories is only valid when "
            "neither side is ordered"
        )

    if how in {"leftsemi", "leftanti"}:
        # These join types have no pandas equivalent; decategorize both sides
        # to a common type to compare values.
        return _match_join_keys(
            lcol._get_decategorized_column(),
            rcol._get_decategorized_column(),
            how,
        )

    if ltype._internal_eq(rtype, strict=False):
        # Categories match up to permutation (same set, both unordered).
        # pandas keeps the result categorical; re-encode both to a unified
        # CategoricalDtype so the join compares by value rather than by code.
        # Decategorize first: a categorical->categorical astype is a no-op
        # here because ``CategoricalDtype.__eq__`` treats permuted categories
        # as equal, which would leave the codes (and thus the join) misaligned.
        common = find_common_type((ltype, rtype))
        return (
            lcol._get_decategorized_column().astype(common),
            rcol._get_decategorized_column().astype(common),
        )

    # Different category sets: pandas decategorizes both sides to their common
    # categories dtype (the result is not categorical), for every join type.
    common_type = find_common_type(
        (ltype.categories.dtype, rtype.categories.dtype)
    )
    return (
        lcol._get_decategorized_column().astype(common_type),
        rcol._get_decategorized_column().astype(common_type),
    )


def _coerce_to_tuple(obj):
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return tuple(obj)
    else:
        return (obj,)
