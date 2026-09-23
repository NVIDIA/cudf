# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pylibcudf as plc

from cudf.core._internals import binaryop
from cudf.core.column.column import (
    ColumnBase,
    dtype_from_pylibcudf_column,
)
from cudf.core.dtype.conversions import subtype_from_interval_dtype
from cudf.core.dtype.validators import is_dtype_obj_interval
from cudf.core.dtypes import IntervalDtype
from cudf.utils.dtypes import get_dtype_of_same_kind

if TYPE_CHECKING:
    from cudf._typing import ColumnBinaryOperand, DtypeObj


class IntervalColumn(ColumnBase):
    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
    }

    @functools.cached_property
    def subtype(self) -> DtypeObj:
        return subtype_from_interval_dtype(self.dtype)

    @functools.cached_property
    def closed(self) -> Literal["left", "right", "neither", "both"]:
        if isinstance(self.dtype, IntervalDtype):
            closed = self.dtype.closed
            return "right" if closed is None else closed
        else:
            return cast("pd.ArrowDtype", self.dtype).pyarrow_dtype.closed

    def to_arrow(self) -> pa.Array:
        pa_array = super().to_arrow()
        pa_type = (
            self.dtype.to_arrow()
            if isinstance(self.dtype, IntervalDtype)
            else cast("pd.ArrowDtype", self.dtype).pyarrow_dtype
        )
        return pa.ExtensionArray.from_storage(pa_type, pa_array)

    @functools.cached_property
    def is_empty(self) -> ColumnBase:
        left_equals_right = (self.right == self.left).fillna(False)
        not_closed_both = ColumnBase.create(
            plc.Column.from_scalar(
                plc.Scalar.from_py(self.closed != "both"),
                len(self),
            ),
            left_equals_right.dtype,
        )
        return left_equals_right & not_closed_both

    @functools.cached_property
    def is_non_overlapping_monotonic(self) -> bool:
        raise NotImplementedError(
            "is_overlapping is currently not implemented."
        )

    @functools.cached_property
    def is_overlapping(self) -> bool:
        raise NotImplementedError(
            "is_overlapping is currently not implemented."
        )

    @functools.cached_property
    def length(self) -> ColumnBase:
        return self.right - self.left

    @functools.cached_property
    def left(self) -> ColumnBase:
        return ColumnBase.create(
            self.plc_column.children()[0],
            self.subtype,
        )

    @functools.cached_property
    def mid(self) -> ColumnBase:
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * self.length

    @functools.cached_property
    def right(self) -> ColumnBase:
        return ColumnBase.create(
            self.plc_column.children()[1],
            self.subtype,
        )

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        raise NotImplementedError(
            "Intervals are not yet supported via `__cuda_array_interface__`"
        )

    def overlaps(other) -> ColumnBase:
        raise NotImplementedError("overlaps is not currently implemented.")

    def set_closed(
        self, closed: Literal["left", "right", "both", "neither"]
    ) -> Self:
        new_dtype = (
            IntervalDtype(self.subtype, closed)
            if isinstance(self.dtype, IntervalDtype)
            else pd.ArrowDtype(
                ArrowIntervalType(
                    cast("pd.ArrowDtype", self.dtype).pyarrow_dtype.subtype,
                    closed,
                )
            )
        )
        return cast(
            "Self",
            ColumnBase.create(
                self.plc_column,
                new_dtype,
            ),
        )

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        _reflect, op = self._check_reflected_op(op)
        if not isinstance(other, type(self)):
            return NotImplemented
        if op in {"__eq__", "__ne__", "NULL_EQUALS", "NULL_NOT_EQUALS"}:
            lefts_equal = self.left._binaryop(other.left, "NULL_EQUALS")
            rights_equal = self.right._binaryop(other.right, "NULL_EQUALS")
            result = binaryop.binaryop(
                lefts_equal,
                rights_equal,
                "__and__",
                get_dtype_of_same_kind(self.dtype, lefts_equal.dtype),
            )
            if op in {"__ne__", "NULL_NOT_EQUALS"}:
                result = ~result
            return result
        else:
            raise TypeError(f"{op} not supported with {type(other).__name__}")

    def as_interval_column(self, dtype: IntervalDtype) -> Self:
        if not is_dtype_obj_interval(dtype):
            raise ValueError(
                f"dtype must be IntervalDtype or interval-type pd.ArrowDtype, got {dtype}"
            )

        # If subtype is changing, cast children to match new subtype
        if dtype.subtype != self.dtype.subtype:  # type: ignore[union-attr]
            new_children = tuple(
                ColumnBase.create(
                    child, dtype_from_pylibcudf_column(child)
                ).astype(dtype.subtype)
                for child in self.plc_column.children()
            )
            # Reconstruct plc_column with cast children
            plc_column = plc.Column(
                plc.DataType(plc.TypeId.STRUCT),
                self.plc_column.size(),
                self.plc_column.data(),
                self.plc_column.null_mask(),
                self.plc_column.null_count(),
                self.plc_column.offset(),
                [child.plc_column for child in new_children],
            )
        else:
            plc_column = self.plc_column

        return ColumnBase.create(plc_column, dtype)  # type: ignore[return-value]

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type or isinstance(self.dtype, pd.ArrowDtype):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif nullable and not arrow_type:
            raise NotImplementedError(
                f"pandas does not have a native nullable type for {self.dtype}."
            )
        pd_type = cast("IntervalDtype", self.dtype).to_pandas()
        return pd.Index(pd_type.__from_arrow__(self.to_arrow()), dtype=pd_type)

    def element_indexing(self, index: int) -> pd.Interval | None:
        result = super().element_indexing(index)
        if isinstance(result, dict):
            return pd.Interval(**result, closed=self.closed)
        return result
