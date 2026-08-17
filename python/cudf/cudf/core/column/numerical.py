# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Self, cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core._internals import binaryop
from cudf.core.column.column import (
    ColumnBase,
    ColumnList,
    PylibcudfFunction,
    as_column,
    column_empty,
    fixed_dtype_policy,
    pylibcudf_result_dtype_policy,
    same_dtype_policy,
)
from cudf.core.column.numerical_base import NumericalBaseColumn
from cudf.core.dtype.validators import (
    is_dtype_obj_string,
)
from cudf.core.mixins import BinaryOperand
from cudf.utils.dtypes import (
    DEFAULT_STRING_DTYPE,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
    find_common_type,
    get_dtype_of_same_kind,
    is_pandas_nullable_extension_dtype,
    min_signed_type,
    min_unsigned_type,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like


def _needs_flip(
    rhs_has_negative_zero: bool, result_sign: cp.ndarray, lhs_sign: cp.ndarray
) -> bool:
    if rhs_has_negative_zero:
        return bool(cp.all(result_sign == lhs_sign))
    else:
        return bool(cp.all(result_sign != lhs_sign))


def _is_null_or_nan_plc(plc_column: plc.Column) -> plc.Column:
    is_null_plc = plc.unary.is_null(plc_column)
    is_nan_plc = plc.unary.is_nan(plc_column)
    return plc.binaryop.binary_operation(
        is_null_plc,
        is_nan_plc,
        plc.binaryop.BinaryOperator.BITWISE_OR,
        plc.types.DataType(plc.types.TypeId.BOOL8),
    )


def _is_valid_and_not_nan_plc(plc_column: plc.Column) -> plc.Column:
    is_valid_plc = plc.unary.is_valid(plc_column)
    is_not_nan_plc = plc.unary.is_not_nan(plc_column)
    return plc.binaryop.binary_operation(
        is_valid_plc,
        is_not_nan_plc,
        plc.binaryop.BinaryOperator.BITWISE_AND,
        plc.types.DataType(plc.types.TypeId.BOOL8),
    )


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column import DecimalColumn
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.string import StringColumn
    from cudf.core.column.timedelta import TimeDeltaColumn
    from cudf.core.dtypes import DecimalDtype


class NumericalColumn(NumericalBaseColumn):
    """A Column object for Numeric types."""

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    @property
    def _PANDAS_NA_VALUE(self) -> ScalarLike:
        """Float columns return np.nan as NA value in pandas compatibility mode."""
        if self.dtype.kind == "f" and isinstance(self.dtype, np.dtype):
            return np.nan
        return super()._PANDAS_NA_VALUE

    def __contains__(self, item: ScalarLike) -> bool:
        """
        Returns True if column contains item, else False.
        """
        # Handles improper item types
        # Fails if item is of type None, so the handler.
        try:
            search_item = self.dtype.type(item)
            if search_item != item and self.dtype.kind != "f":
                return False
        except (TypeError, ValueError):
            return False
        # TODO: Use `scalar`-based `contains` wrapper
        return self.contains(
            as_column(
                [search_item],
                dtype=self.dtype,
                nan_as_null=not cudf.get_option("mode.pandas_compatible"),
            ),
        ).any()

    def dropna(self) -> Self:
        if self.has_nulls(include_nan=True):
            with self.access(mode="read", scope="internal") as accessed:
                plc_table = plc.Table([accessed.plc_column])
                plc_result = plc.stream_compaction.drop_nulls(
                    plc_table, [0], 1
                )
                if self.dtype.kind == "f":
                    plc_result = plc.stream_compaction.drop_nans(
                        plc_result, [0], 1
                    )
            return cast(
                "Self",
                ColumnBase.create(plc_result.columns()[0], dtype=self.dtype),
            )
        else:
            return self.copy()

    @property
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the NumericalColumn.
        """
        if not isinstance(self.dtype, np.dtype):
            dtype = self.dtype.numpy_dtype
        else:
            dtype = self.dtype

        if len(self) == 0:
            return cp.empty(0, dtype=dtype)

        col = self
        if col.has_nulls():
            if dtype.kind == "b":
                raise ValueError(
                    f"Column must have no nulls for dtype={col.dtype}"
                )
            elif dtype.kind != "f":
                dtype = np.dtype(np.float64)
                col = col.astype(dtype)  # type: ignore[assignment]
            col = col.fillna(np.nan)

        return cp.asarray(col).view(dtype)

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
        if isinstance(value, (bool, np.bool_)) and self.dtype.kind != "b":
            raise ValueError(
                f"Cannot use a {type(value).__name__} to find an index of "
                f"a {self.dtype} Index."
            )
        elif (
            self.dtype.kind in {"c", "f"}
            and isinstance(value, (float, np.floating))
            and np.isnan(value)
        ):
            return self.isnan().indices_of(True)
        else:
            return super().indices_of(value)

    def has_nulls(self, include_nan: bool = False) -> bool:
        return bool(self.null_count != 0) or (
            include_nan and bool(self.nan_count != 0)
        )

    def isnan(self) -> ColumnBase:
        """Identify NaN values in a Column.

        Only meaningful for float dtypes. For integer and boolean columns,
        returns a column of False values.
        """
        if self.dtype.kind != "f":
            return as_column(False, length=len(self))
        return PylibcudfFunction(
            plc.unary.is_nan, pylibcudf_result_dtype_policy
        ).execute_with_args(self)

    def notnan(self) -> ColumnBase:
        """Identify non-NaN values in a Column.

        Only meaningful for float dtypes. For integer and boolean columns,
        returns a column of True values.
        """
        if self.dtype.kind != "f":
            return as_column(True, length=len(self))
        return PylibcudfFunction(
            plc.unary.is_not_nan,
            pylibcudf_result_dtype_policy,
        ).execute_with_args(self)

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column.

        For float columns, NaN values are also considered null.
        """
        if not self.has_nulls(include_nan=self.dtype.kind == "f"):
            return as_column(False, length=len(self))
        if self.dtype.kind == "f":
            return PylibcudfFunction(
                _is_null_or_nan_plc,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self)
        return PylibcudfFunction(
            plc.unary.is_null,
            pylibcudf_result_dtype_policy,
        ).execute_with_args(self)

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column.

        For float columns, NaN values are considered null and excluded.
        """
        if not self.has_nulls(include_nan=self.dtype.kind == "f"):
            result = as_column(True, length=len(self))
        else:
            if self.dtype.kind == "f":
                result = PylibcudfFunction(
                    _is_valid_and_not_nan_plc,
                    pylibcudf_result_dtype_policy,
                ).execute_with_args(self)
            else:
                result = PylibcudfFunction(
                    plc.unary.is_valid,
                    pylibcudf_result_dtype_policy,
                ).execute_with_args(self)

        return result

    def element_indexing(self, index: int) -> ScalarLike | None:
        result = super().element_indexing(index)
        if isinstance(result, (int, float, bool)):
            return self.dtype.type(result)
        return result

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if is_scalar(value):
            nan_value = isinstance(value, float) and np.isnan(value)
            pandas_compatible = cudf.get_option("mode.pandas_compatible")
            # Only treat ``np.nan`` as null in pandas-compatible mode. In
            # legacy cudf mode, ``np.nan`` is stored as the float value
            # ``nan`` in the data buffer (no null mask). Changing this
            # silently turns any ``loc[...] = np.nan`` assignment into a
            # masked-null, which breaks consumers that read the raw data
            # buffer (e.g. the ``engine="jit"`` groupby kernel).
            if (
                value is cudf.NA
                or value is None
                or (nan_value and pandas_compatible)
            ):
                if self.dtype.kind == "b" and pandas_compatible:
                    # pandas raises TypeError when setting a null into bool
                    # because bool dtype cannot hold NA without dtype change.
                    raise TypeError(
                        f"Invalid value '{value}' for dtype '{self.dtype}'"
                    )
                scalar = pa.scalar(
                    None, type=cudf_dtype_to_pa_type(self.dtype)
                )
            else:
                try:
                    scalar = pa.scalar(value)
                except ValueError as err:
                    raise TypeError(
                        f"Cannot set value of type {type(value)} to column of type {self.dtype}"
                    ) from err
            is_scalar_bool = pa.types.is_boolean(scalar.type)
            if (is_scalar_bool and self.dtype.kind != "b") or (
                not is_scalar_bool and self.dtype.kind == "b"
            ):
                raise TypeError(
                    f"Invalid value {value} for dtype {self.dtype}"
                )
            if pa.types.is_temporal(scalar.type):
                # Reject datetime64/timedelta64 scalars (including NaT)
                # assigned into a numeric column. pyarrow would otherwise
                # silently cast timestamp[ns]/duration[ns] to int64 via the
                # underlying integer representation.
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'"
                )
            return pa_scalar_to_plc_scalar(
                scalar.cast(cudf_dtype_to_pa_type(self.dtype))
            )
        else:
            col = as_column(value)
            if col.dtype.kind == "b" and self.dtype.kind != "b":
                raise TypeError(
                    f"Invalid value {value} for dtype {self.dtype}"
                )
            if col.dtype.kind in "mM":
                # See the temporal-scalar rejection above: assigning a
                # datetime64/timedelta64 array into a numeric column is
                # invalid.
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'"
                )
            if (
                cudf.get_option("mode.pandas_compatible")
                and self.dtype.kind in {"i", "u"}
                and col.dtype.kind == "f"
                and not col.can_cast_safely(self.dtype)
            ):
                # pandas rejects float→int assignment when values can't be
                # losslessly cast.
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'"
                )
            return col.astype(self.dtype)

    def __invert__(self) -> ColumnBase:
        if (dtype_kind := self.dtype.kind) in "ui":
            return self.unary_operator("invert")
        elif dtype_kind == "b":
            return self.unary_operator("not")
        else:
            return super().__invert__()

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        int_float_dtype_mapping = {
            np.int8: np.float32,
            np.int16: np.float32,
            np.int32: np.float32,
            np.int64: np.float64,
            np.uint8: np.float32,
            np.uint16: np.float32,
            np.uint32: np.float64,
            np.uint64: np.float64,
            np.bool_: np.float32,
        }
        if cudf.get_option("mode.pandas_compatible"):
            int_float_dtype_mapping = {
                np.int8: np.float64,
                np.int16: np.float64,
                np.int32: np.float64,
                np.int64: np.float64,
                np.uint8: np.float64,
                np.uint16: np.float64,
                np.uint32: np.float64,
                np.uint64: np.float64,
                np.bool_: np.float64,
            }

        cmp_ops = {
            "__lt__",
            "__gt__",
            "__le__",
            "__ge__",
            "__eq__",
            "__ne__",
        }
        out_dtype = None
        if op in {"__truediv__", "__rtruediv__"}:
            # Division with integer types results in a suitable float.
            if truediv_type := int_float_dtype_mapping.get(
                self.dtype.numpy_dtype.type
                if is_pandas_nullable_extension_dtype(self.dtype)
                else self.dtype.type
            ):
                return self.astype(
                    get_dtype_of_same_kind(self.dtype, np.dtype(truediv_type))
                )._binaryop(other, op)
        elif op in cmp_ops:
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))

            # If `other` is a Python integer and it is out-of-bounds
            # promotion could fail but we can trivially define the result
            # in terms of `notnull` or `NULL_NOT_EQUALS`.
            if type(other) is int and self.dtype.kind in "iu":
                truthiness = None
                if not isinstance(self.dtype, np.dtype):
                    info_type = self.dtype.numpy_dtype
                else:
                    info_type = self.dtype
                iinfo = np.iinfo(info_type)
                if iinfo.min > other:
                    truthiness = op in {"__ne__", "__gt__", "__ge__"}
                elif iinfo.max < other:
                    truthiness = op in {"__ne__", "__lt__", "__le__"}

                # Compare with minimum value so that the result is true/false
                if truthiness is True:
                    other = iinfo.min
                    op = "__ge__"
                elif truthiness is False:
                    other = iinfo.min
                    op = "__lt__"

        elif op in {"NULL_EQUALS", "NULL_NOT_EQUALS"}:
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))

        reflect, op = self._check_reflected_op(op)
        if (other := self._normalize_binop_operand(other)) is NotImplemented:
            return NotImplemented
        other_cudf_dtype = (
            cudf_dtype_from_pa_type(other.type)
            if isinstance(other, pa.Scalar)
            else other.dtype
        )
        self_is_arrow = isinstance(self.dtype, pd.ArrowDtype)
        other_is_arrow = isinstance(other_cudf_dtype, pd.ArrowDtype)
        if self_is_arrow or other_is_arrow:
            if op == "__mod__":
                raise NotImplementedError("ArrowDtype does not support modulo")
            # pyarrow.compute raises ArrowInvalid for integer divide by zero in
            # floordiv. Mirror that behavior for ArrowDtype operands instead of
            # silently returning inf via float promotion.
            if op == "__floordiv__":
                divisor = self if reflect else other
                divisor_dtype = self.dtype if reflect else other_cudf_dtype
                if divisor_dtype.kind in "iu" and (
                    (isinstance(divisor, NumericalColumn) and 0 in divisor)
                    or (
                        isinstance(divisor, pa.Scalar) and divisor.as_py() == 0
                    )
                ):
                    raise pa.ArrowInvalid("divide by zero")
            # pyarrow.compute.subtract_checked raises ArrowInvalid on overflow
            # for unsigned integer subtraction. Mirror that for ArrowDtype, but
            # only when both operands are unsigned: a Python int operand is
            # promoted to signed int64, so the subtraction is no longer
            # unsigned and cannot underflow.
            if (
                op == "__sub__"
                and self_is_arrow
                and self.dtype.kind == "u"
                and other_is_arrow
                and other_cudf_dtype.kind == "u"
            ):
                actual_lhs = other if reflect else self
                actual_rhs = self if reflect else other
                overflow = False
                if isinstance(actual_lhs, NumericalColumn) and isinstance(
                    actual_rhs, NumericalColumn
                ):
                    overflow = bool(
                        actual_lhs._binaryop(actual_rhs, "__lt__").any()
                    )
                elif isinstance(actual_lhs, NumericalColumn) and isinstance(
                    actual_rhs, pa.Scalar
                ):
                    if actual_rhs.is_valid:
                        overflow = bool(
                            actual_lhs._binaryop(
                                actual_rhs.as_py(), "__lt__"
                            ).any()
                        )
                elif isinstance(actual_lhs, pa.Scalar) and isinstance(
                    actual_rhs, NumericalColumn
                ):
                    if actual_lhs.is_valid:
                        overflow = bool(
                            actual_rhs._binaryop(
                                actual_lhs.as_py(), "__gt__"
                            ).any()
                        )
                if overflow:
                    raise pa.ArrowInvalid("overflow")
            bool_unsupported_ops = {
                "__add__",
                "__radd__",
                "__sub__",
                "__rsub__",
                "__mul__",
                "__rmul__",
                "__truediv__",
                "__rtruediv__",
                "__floordiv__",
                "__rfloordiv__",
                "__pow__",
                "__rpow__",
            }
            if op in bool_unsupported_ops and (
                (self_is_arrow and self.dtype.kind == "b")
                or (other_is_arrow and other_cudf_dtype.kind == "b")
            ):
                # pandas raises TypeError for arithmetic on bool[pyarrow]:
                # operation 'X' not supported for dtype 'bool[pyarrow]' ...
                other_py = (
                    other.as_py() if isinstance(other, pa.Scalar) else other
                )
                raise TypeError(
                    f"operation '{op[2:-2]}' not supported for dtype "
                    f"'bool[pyarrow]' with object of type "
                    f"{type(other_py).__name__!r}"
                )

        if out_dtype is None:
            out_dtype = find_common_type((self.dtype, other_cudf_dtype))
            if op in {"__mod__", "__floordiv__"}:
                tmp = self if reflect else other
                tmp_dtype = self.dtype if reflect else other_cudf_dtype
                # Guard against division by zero for integers.
                if tmp_dtype.kind in "iu" and (
                    (isinstance(tmp, NumericalColumn) and 0 in tmp)
                    or (isinstance(tmp, pa.Scalar) and tmp.as_py() == 0)
                ):
                    out_dtype = get_dtype_of_same_kind(
                        out_dtype, np.dtype(np.float64)
                    )
            # pandas pyarrow preserves the LHS dtype for integer floordiv
            # rather than upcasting based on the operand widths. The LHS is
            # `other` when the operation is reflected, otherwise `self`.
            if (
                op == "__floordiv__"
                and self.dtype.kind in "iu"
                and other_cudf_dtype.kind in "iu"
            ):
                lhs_is_arrow = other_is_arrow if reflect else self_is_arrow
                if lhs_is_arrow:
                    out_dtype = other_cudf_dtype if reflect else self.dtype
            # For ArrowDtype integer + Python int (now a pa.Scalar with a
            # pyarrow int type), pandas pyarrow promotes to int64[pyarrow]
            # regardless of width. numpy promotion would instead pick float64
            # for (uint64, int64); override to match pandas pyarrow.
            elif (
                self_is_arrow
                and self.dtype.kind in "iu"
                and isinstance(other, pa.Scalar)
                and other_cudf_dtype.kind in "iu"
                and op in {"__add__", "__sub__", "__mul__", "__pow__"}
            ):
                out_dtype = pd.ArrowDtype(pa.int64())

        if op in {"__and__", "__or__", "__xor__"}:
            if self.dtype.kind == "f" or other_cudf_dtype.kind == "f":
                raise TypeError(
                    f"Operation 'bitwise {op[2:-2]}' not supported between "
                    f"{self.dtype.type.__name__} and "
                    f"{other_cudf_dtype.type.__name__}"
                )
            if self.dtype.kind == "b" and other_cudf_dtype.kind == "b":
                out_dtype = get_dtype_of_same_kind(
                    self.dtype, np.dtype(np.bool_)
                )
            elif self.dtype.kind == "b" or other_cudf_dtype.kind == "b":
                out_dtype = get_dtype_of_same_kind(
                    out_dtype, np.dtype(np.bool_)
                )

        elif (
            op == "__pow__"
            and self.dtype.kind in "iu"
            and (other_cudf_dtype.kind in "iu")
        ):
            op = "INT_POW"

        lhs_dtype, rhs_dtype = (
            (other_cudf_dtype, self.dtype)
            if reflect
            else (self.dtype, other_cudf_dtype)
        )
        lhs, rhs = (other, self) if reflect else (self, other)
        if out_dtype.kind == "f" and is_pandas_nullable_extension_dtype(
            out_dtype
        ):
            if (
                not is_pandas_nullable_extension_dtype(lhs_dtype)
                and lhs_dtype.kind == "f"
                and isinstance(lhs, NumericalColumn)
            ):
                lhs = lhs.nans_to_nulls()
            if (
                not is_pandas_nullable_extension_dtype(rhs_dtype)
                and rhs_dtype.kind == "f"
                and isinstance(rhs, NumericalColumn)
            ):
                rhs = rhs.nans_to_nulls()
        if op in cmp_ops and is_pandas_nullable_extension_dtype(out_dtype):
            if lhs_dtype.kind == "f" and isinstance(lhs, NumericalColumn):
                lhs = lhs.nans_to_nulls()
            if rhs_dtype.kind == "f" and isinstance(rhs, NumericalColumn):
                rhs = rhs.nans_to_nulls()
        lhs_binaryop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(lhs) if isinstance(lhs, pa.Scalar) else lhs
        )
        rhs_binaryop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(rhs) if isinstance(rhs, pa.Scalar) else rhs
        )

        res = binaryop.binaryop(lhs_binaryop, rhs_binaryop, op, out_dtype)
        if (
            is_pandas_nullable_extension_dtype(out_dtype)
            and out_dtype.kind == "f"
        ):
            # If the output dtype is a pandas nullable extension type,
            # we need to ensure that the result is a NumericalColumn.
            res = res.nans_to_nulls()
        if op in {"__truediv__", "__floordiv__"}:
            rhs_all_zero = False
            rhs_has_negative_zero = False
            if isinstance(rhs, ColumnBase):
                if rhs.null_count == 0:
                    rhs_values = rhs.values
                    rhs_all_zero = cp.all(rhs_values == 0)
                    if rhs_all_zero and rhs.dtype.kind == "f":
                        rhs_has_negative_zero = cp.any(cp.signbit(rhs_values))
            else:
                assert isinstance(rhs, pa.Scalar)
                if rhs.is_valid:
                    rhs_value = rhs.as_py()
                    rhs_all_zero = rhs_value == 0
                    if rhs_all_zero:
                        rhs_has_negative_zero = np.signbit(rhs_value)
            if (
                rhs_all_zero
                and not res.dtype.kind == "b"
                or res.null_count == 0
            ):
                result_values = res.values
                if result_values.size != 0:
                    result_is_inf = cp.isinf(result_values)
                    if isinstance(lhs, ColumnBase):
                        if not lhs.dtype.kind == "b" or lhs.null_count == 0:
                            lhs_values = lhs.values
                            mask = result_is_inf & (lhs_values != 0)
                            if cp.any(mask) and _needs_flip(
                                rhs_has_negative_zero,
                                cp.signbit(result_values[mask]),
                                cp.signbit(lhs_values[mask]),
                            ):
                                res *= -1
                    elif isinstance(lhs, pa.Scalar):
                        if lhs.is_valid:
                            num_value = lhs.as_py()
                            if not (
                                num_value == 0 or not cp.any(result_is_inf)
                            ):
                                mask = result_is_inf
                                if _needs_flip(
                                    rhs_has_negative_zero,
                                    cp.signbit(result_values[mask]),
                                    np.signbit(num_value),
                                ):
                                    res *= -1
        if op in {"__mod__", "__floordiv__"} and tmp_dtype.kind == "b":
            res = res.astype(
                get_dtype_of_same_kind(out_dtype, np.dtype(np.int8))
            )
        elif (
            op in {"__pow__", "INT_POW"}
            and res.null_count
            and not (self_is_arrow or other_is_arrow)
        ):
            # pandas: ``1 ** x == 1`` and ``x ** 0 == 1`` hold even when
            # ``x`` is missing, for both NaN (numpy semantics) and NA
            # (masked semantics). libcudf propagates nulls unconditionally,
            # so patch the positions covered by these identities. pandas'
            # ArrowDtype follows pyarrow and propagates nulls, hence the
            # arrow exclusion.
            fix_mask = None
            if isinstance(lhs, pa.Scalar):
                if lhs.is_valid and lhs.as_py() == 1:
                    res = res.fillna(1)
            else:
                fix_mask = lhs._binaryop(1, "__eq__").fillna(False)
            if isinstance(rhs, pa.Scalar):
                if rhs.is_valid and rhs.as_py() == 0:
                    res = res.fillna(1)
            else:
                exp_is_zero = rhs._binaryop(0, "__eq__").fillna(False)
                fix_mask = (
                    exp_is_zero
                    if fix_mask is None
                    else fix_mask._binaryop(exp_is_zero, "__or__")
                )
            if fix_mask is not None and res.null_count:
                res = res.fillna(1).copy_if_else(res, fix_mask)
        elif (
            cudf.get_option("mode.pandas_compatible")
            and op in cmp_ops
            and not is_pandas_nullable_extension_dtype(self.dtype)
            and self.dtype.kind != "b"
        ):
            # For numeric dtypes, pandas converts nulls to NaN and the
            # comparison follows NumPy semantics (NaN == anything is False,
            # NaN != anything is True). Fill so we match that. Bool dtype
            # is excluded: a nullable cudf bool column maps to pandas'
            # ``boolean`` extension dtype, which preserves nulls in
            # comparisons.
            res = res.fillna(op == "__ne__")
        return res

    def nans_to_nulls(self: Self) -> Self:
        if self.dtype.kind != "f" or self.nan_count == 0:
            return self
        result = PylibcudfFunction(
            plc.transform.column_nans_to_nulls,
            same_dtype_policy,
        ).execute_with_args(self)
        return cast("Self", result)

    def _normalize_binop_operand(self, other: Any) -> pa.Scalar | ColumnBase:
        if isinstance(other, ColumnBase):
            if not isinstance(other, type(self)):
                return NotImplemented
            return other
        # TODO: cupy scalars are just aliases for numpy scalars, so extracting a scalar
        # from a cupy array would always require a D2H copy. As a result, cupy does not
        # produce scalars without explicit casting requests
        # https://docs.cupy.dev/en/stable/user_guide/difference.html#zero-dimensional-array
        # The below logic for type inference relies on numpy, however, so we need to go
        # that route for now. If possible we should find a way to avoid this.
        if isinstance(other, cp.ndarray) and other.ndim == 0:
            other = cp.asnumpy(other)[()]
        elif isinstance(other, np.ndarray) and other.ndim == 0:
            other = other[()]

        if is_scalar(other):
            if is_na_like(other):
                if other is pd.NaT or (
                    isinstance(other, (np.datetime64, np.timedelta64))
                    and np.isnat(other)
                ):
                    return NotImplemented
                return pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
            if not isinstance(other, (int, float, complex)):
                # Go via NumPy to get the value
                other = np.array(other)
                if other.dtype.kind in "uifc":
                    other = other.item()

            # Try and match pandas and hence numpy. Deduce the common
            # dtype via the _value_ of other, and the dtype of self on NumPy 1.x
            # with NumPy 2, we force weak promotion even for our/NumPy scalars
            # to match pandas 2.2.
            # Weak promotion is not at all simple:
            # np.result_type(0, np.uint8)
            #   => np.uint8
            # np.result_type(np.asarray([0], dtype=np.int64), np.uint8)
            #   => np.int64
            # np.promote_types(np.int64(0), np.uint8)
            #   => np.int64
            # np.promote_types(np.asarray([0], dtype=np.int64).dtype, np.uint8)
            #   => np.int64
            if is_pandas_nullable_extension_dtype(self.dtype):
                if isinstance(self.dtype, pd.ArrowDtype):
                    if self.dtype.kind in "iu" and type(other) is int:
                        # pandas pyarrow treats a Python int as int64 when
                        # combined with an ArrowDtype integer column, so the
                        # promoted scalar type is int64 regardless of width.
                        common_dtype = pd.ArrowDtype(pa.int64())
                    else:
                        common_dtype = cudf.utils.dtypes.find_common_type(
                            [self.dtype, other]
                        )
                else:
                    common_dtype = get_dtype_of_same_kind(
                        self.dtype,
                        np.result_type(self.dtype.numpy_dtype, other),  # noqa: TID251
                    )
            else:
                common_dtype = np.result_type(self.dtype, other)  # noqa: TID251
            if common_dtype.kind in {"b", "i", "u", "f"}:
                if self.dtype.kind == "b" and not isinstance(other, bool):
                    common_dtype = min_signed_type(other)
                return pa.scalar(
                    other, type=cudf_dtype_to_pa_type(common_dtype)
                )
            else:
                return NotImplemented
        else:
            return NotImplemented

    def int2ip(self) -> StringColumn:
        if self.dtype != np.dtype(np.uint32):
            raise TypeError("Only uint32 type can be converted to ip")
        return cast(
            "cudf.core.column.string.StringColumn",
            PylibcudfFunction(
                plc.strings.convert.convert_ipv4.integers_to_ipv4,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self),
        )

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        col = self
        if (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(dtype, np.dtype)
            and dtype.kind == "O"
        ):
            raise ValueError(
                "Cannot convert numerical column to string column "
                "when dtype is an object dtype in pandas compatibility mode."
            )
        if isinstance(dtype, np.dtype) and dtype.kind == "U":
            dtype = np.dtype("object")
        if len(self) == 0:
            return cast(
                "cudf.core.column.StringColumn",
                column_empty(0, dtype=DEFAULT_STRING_DTYPE),
            )

        conv_func: Callable[[plc.Column], plc.Column]
        if self.dtype.kind == "b":
            conv_func = functools.partial(
                plc.strings.convert.convert_booleans.from_booleans,
                true_string=plc.Scalar.from_py(
                    "True", dtype=plc.DataType(plc.TypeId.STRING)
                ),
                false_string=plc.Scalar.from_py(
                    "False", dtype=plc.DataType(plc.TypeId.STRING)
                ),
            )
        elif self.dtype.kind in {"i", "u"}:
            conv_func = plc.strings.convert.convert_integers.from_integers
        elif self.dtype.kind == "f":
            if cudf.get_option(
                "mode.pandas_compatible"
            ) and is_pandas_nullable_extension_dtype(dtype):
                # In pandas compatibility mode, we convert nans to nulls
                col = self.nans_to_nulls()
            conv_func = plc.strings.convert.convert_floats.from_floats
        else:
            raise ValueError(f"No string conversion from type {self.dtype}")

        with col.access(mode="read", scope="internal"):
            return cast(
                "cudf.core.column.string.StringColumn",
                ColumnBase.create(conv_func(col.plc_column), dtype),
            )

    def _as_temporal_column(self, dtype: np.dtype) -> plc.Column:
        """Convert Self to a temporal pylibcudf Column for as_datetime_column and as_timedelta_column"""
        return plc.Column(
            data_type=dtype_to_pylibcudf_type(dtype),
            size=self.size,
            data=self.astype(np.dtype(np.int64)).data,
            mask=self.mask,
            null_count=self.null_count,
            offset=self.offset,
            children=[],
        )

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        return cast(
            "cudf.core.column.datetime.DatetimeColumn",
            ColumnBase.create(self._as_temporal_column(dtype), dtype),
        )

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        return cast(
            "cudf.core.column.timedelta.TimeDeltaColumn",
            ColumnBase.create(self._as_temporal_column(dtype), dtype),
        )

    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalColumn:
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    def as_numerical_column(self, dtype: DtypeObj) -> NumericalColumn:
        if dtype == self.dtype:
            return self

        if cudf.get_option("mode.pandas_compatible"):
            if (
                is_pandas_nullable_extension_dtype(self.dtype)
                and isinstance(dtype, np.dtype)
                and self.null_count > 0
            ):
                if dtype.kind in "iu":
                    raise ValueError("cannot convert NA to integer")
                elif dtype.kind == "b":
                    raise ValueError("cannot convert float NaN to bool")

            if (
                not is_pandas_nullable_extension_dtype(self.dtype)
                and is_pandas_nullable_extension_dtype(dtype)
                and dtype.kind == "f"
            ):
                res = self.nans_to_nulls().cast(dtype=dtype)
                res._dtype = dtype
                return res  # type: ignore[return-value]

            if (
                self.dtype.kind == "f"
                and dtype.kind == "b"
                and not is_pandas_nullable_extension_dtype(dtype)
                and self.has_nulls()
            ):
                return self.fillna(np.nan).cast(dtype=dtype)  # type: ignore[return-value]

            if dtype_to_pylibcudf_type(dtype) == dtype_to_pylibcudf_type(
                self.dtype
            ):
                # Short-circuit the cast if the dtypes are equivalent
                # but not the same type object. Do NOT mutate self._dtype:
                # the column object may be shared with the caller's frame.
                if (
                    is_pandas_nullable_extension_dtype(dtype)
                    and isinstance(self.dtype, np.dtype)
                    and self.dtype.kind == "f"
                ):
                    # NaNs must become nulls before viewing as a masked dtype.
                    res = self.nans_to_nulls()
                    return cast(
                        "NumericalColumn",
                        ColumnBase.create(res.plc_column, dtype),
                    )
                return cast(
                    "NumericalColumn",
                    ColumnBase.create(self.plc_column, dtype),
                )
            if self.dtype.kind == "f" and dtype.kind in "iu":
                if not is_pandas_nullable_extension_dtype(dtype) and (
                    self.nan_count > 0
                    or np.isinf(self.min())
                    or np.isinf(self.max())
                ):
                    raise TypeError(
                        "Cannot convert non-finite values (NA or inf) to integer"
                    )
                # If casting from float to int, we need to convert nans to nulls
                res = self.nans_to_nulls().cast(dtype=dtype)
                res._dtype = dtype
                return res  # type: ignore[return-value]

        return self.cast(dtype=dtype)  # type: ignore[return-value]

    @functools.cached_property
    def nan_count(self) -> int:
        if self.dtype.kind != "f":
            return super().nan_count
        return self.isnan().sum()

    def isin(self, values: Sequence | ColumnBase) -> ColumnBase:
        if isinstance(self.dtype, (np.dtype, pd.ArrowDtype)):
            return super().isin(values)
        # Mirror pandas' BaseMaskedArray.isin for masked (nullable
        # integer/float/boolean) dtypes:
        #  * matching is done on the underlying numpy values, so e.g. a
        #    boolean element equals the integer 1,
        #  * an NA element is considered present only when pd.NA itself
        #    is one of the passed values (a plain NaN/None/NaT does not
        #    match), and
        #  * the result is a nullable BooleanDtype with no missing values.
        # Validity-only nulls (ColumnBase.isnull): a genuine NaN value in a
        # masked float column is data, not NA - it must keep matching a NaN
        # needle instead of being folded into the mask
        # (NumericalColumn.isnull would count it as null).
        na_mask = ColumnBase.isnull(self)
        # View the column through its numpy dtype (identical physical
        # layout). astype is not used because casting a masked column with
        # nulls to numpy raises in pandas-compatible mode, and its
        # short-circuit mutates the live column's dtype in place; fillna is
        # not used because it folds genuine NaN values into the fill. Null
        # rows come back False from ``isin`` (the needles below never
        # contain nulls) and are then corrected from ``na_mask``.
        data_col = ColumnBase.create(self.plc_column, self.dtype.numpy_dtype)
        cleaned_values, values_have_na = self._process_values_for_masked_isin(
            values
        )
        if len(cleaned_values) == 0:
            # Nothing can match; also avoids the object-dtype column an
            # empty needle list produces, which an all-null data column
            # cannot be compared against in pandas-compatible mode.
            result = as_column(
                False, length=len(self), dtype=np.dtype(np.bool_)
            )
        else:
            result = data_col.isin(cleaned_values)
        if na_mask.any():
            result = result | na_mask if values_have_na else result & ~na_mask
        return result.astype(pd.BooleanDtype())

    @staticmethod
    def _process_values_for_masked_isin(
        values: Sequence | ColumnBase,
    ) -> tuple[Sequence, bool]:
        """Normalize ``values`` for a masked ``isin``.

        Drops NA-like needles (they are handled through the mask by the
        caller) and reports whether ``pd.NA`` itself was among them.
        """
        # The normalization below rebinds across container types, so use a
        # loosely typed alias.
        host_values: Any = values
        # Bring device-backed ``values`` to host in a single transfer; the
        # per-element inspection below would otherwise read them
        # element-wise or fail on non-iterable cudf objects.
        if isinstance(host_values, (ColumnBase, cudf.Series, cudf.Index)):
            host_values = host_values.to_pandas()
        elif isinstance(host_values, cp.ndarray):
            host_values = cp.asnumpy(host_values)
        elif isinstance(host_values, (pa.Array, pa.ChunkedArray)):
            host_values = host_values.to_pylist()
        if isinstance(
            host_values,
            (pd.Series, pd.Index, pd.api.extensions.ExtensionArray),
        ):
            # np.asarray mirrors pandas' BaseMaskedArray.isin: a masked (or
            # other extension) container decays to numpy, where pd.NA
            # becomes NaN and no longer matches NA rows - only object
            # containers can carry pd.NA identity.
            host_values = np.asarray(host_values)
        elif not isinstance(host_values, np.ndarray):
            # Materialize one-shot iterators so the single pass below is
            # the only traversal.
            host_values = list(host_values)
        # NA-like sentinels never match a real value and would otherwise
        # raise when mixed with numeric values, so they are dropped from
        # the needles; NA rows are handled by the caller through the mask
        # instead.
        if isinstance(host_values, np.ndarray) and host_values.dtype != object:
            # A non-object array cannot hold pd.NA, and NaT is the only
            # NA-like value it can hold (NaN is a matchable value), so no
            # Python-level scan is needed.
            if host_values.dtype.kind in "mM":
                host_values = host_values[~np.isnat(host_values)]
            return cast("Sequence", host_values), False
        # ``host_values`` is a list or an object array here; a single
        # Python pass mirrors pandas' pd.NA identity scan.
        cleaned: list[Any] = []
        values_have_na = False
        for value in host_values:
            if value is pd.NA:
                values_have_na = True
            elif not is_na_like(value):
                cleaned.append(value)
        return cleaned, values_have_na

    def _process_values_for_isin(
        self, values: Sequence | ColumnBase
    ) -> tuple[ColumnBase, ColumnBase]:
        try:
            lhs, rhs = super()._process_values_for_isin(values)
        except TypeError:
            # Can remove once dask 25.04 is the minimum version
            # https://github.com/dask/dask/pull/11869
            if isinstance(values, np.ndarray) and values.dtype.kind == "O":
                return super()._process_values_for_isin(values.tolist())
            else:
                raise
        if lhs.dtype != rhs.dtype and not is_dtype_obj_string(rhs.dtype):
            if rhs.can_cast_safely(lhs.dtype):
                rhs = rhs.astype(lhs.dtype)
            elif lhs.can_cast_safely(rhs.dtype):
                lhs = lhs.astype(rhs.dtype)
            elif (
                isinstance(lhs.dtype, np.dtype)
                and isinstance(rhs.dtype, np.dtype)
                and lhs.dtype.kind in "biuf"
                and rhs.dtype.kind in "biuf"
                and "b" in (lhs.dtype.kind, rhs.dtype.kind)
            ):
                # A boolean column compares by value against numeric needles
                # (``True == 1``) like numpy/pandas. ``can_cast_safely`` reports
                # bool<->numeric as unsafe, so promote the boolean side to the
                # numeric dtype (a bool always fits) rather than bailing out to
                # an all-False result.
                if lhs.dtype.kind == "b":
                    lhs = lhs.astype(rhs.dtype)
                else:
                    rhs = rhs.astype(lhs.dtype)
        return lhs, rhs

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls(include_nan=True)

    def _min_column_type(self, expected_type: np.dtype) -> np.dtype:
        """
        Return the smallest dtype which can represent all elements of self.
        """
        if self.is_all_null:
            return self.dtype

        min_value, max_value = self.minmax()
        either_is_inf = np.isinf(min_value) or np.isinf(max_value)
        if not either_is_inf and expected_type.kind == "i":
            max_bound_dtype = min_signed_type(max_value)
            min_bound_dtype = min_signed_type(min_value)
            return np.promote_types(max_bound_dtype, min_bound_dtype)
        elif not either_is_inf and expected_type.kind == "u":
            max_bound_dtype = min_unsigned_type(max_value)
            min_bound_dtype = min_unsigned_type(min_value)
            return np.promote_types(max_bound_dtype, min_bound_dtype)
        elif self.dtype.kind == "f" or expected_type.kind == "f":
            return np.promote_types(
                expected_type,
                np.promote_types(
                    np.min_scalar_type(float(max_value)),
                    np.min_scalar_type(float(min_value)),
                ),
            )
        else:
            return self.dtype

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        """Return col with *to_replace* replaced with *value*."""
        # TODO: The all_nan argument is only used for this
        # this subclass, try to factor these cases out of this method
        to_replace_col, replacement_col = (
            ColumnBase._prepare_find_and_replace_columns(
                to_replace,
                replacement,
                null_cast_dtype=self.dtype,
            )
        )

        if not isinstance(to_replace_col, NumericalColumn) and not isinstance(
            replacement_col, NumericalColumn
        ):
            return self.copy()

        try:
            to_replace_col = _normalize_find_and_replace_input(
                self.dtype, to_replace
            )
        except TypeError:
            # if `to_replace` cannot be normalized to the current dtype,
            # that means no value of `to_replace` is present in self,
            # Hence there is no point of proceeding further.
            return self.copy()

        if all_nan:
            replacement_col = as_column(replacement, dtype=self.dtype)
        else:
            try:
                replacement_col = _normalize_find_and_replace_input(
                    self.dtype, replacement
                )
            except TypeError:
                # Some floating values can never be converted into signed or unsigned integers
                # for those cases, we just need a column of `replacement` constructed
                # with its own type for the final type determination below at `find_common_type`
                # call.
                replacement_col = as_column(
                    replacement,
                    dtype=self.dtype if len(replacement) <= 0 else None,
                )
        if len(replacement_col) == 1 and len(to_replace_col) > 1:
            replacement_col = replacement_col.repeat(len(to_replace_col))
        elif len(replacement_col) == 1 and len(to_replace_col) == 0:
            return self.copy()

        common_type = find_common_type(
            (to_replace_col.dtype, replacement_col.dtype, self.dtype)
        )
        assert common_type is not None
        replaced = cast("Self", self.astype(common_type))
        old_col = to_replace_col.astype(common_type)
        new_col = replacement_col.astype(common_type)

        result = replaced._find_and_replace_with_dedup(
            old_col,
            new_col,
            common_type,
        )
        return cast("Self", result)

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value):
            plc_col = plc.Column.from_scalar(
                pa_scalar_to_plc_scalar(pa.scalar(fill_value)), 1
            )
            cudf_obj = ColumnBase.create(
                plc_col,
                dtype=dtype_from_pylibcudf_column(plc_col),
            )
            if not cudf_obj.can_cast_safely(self.dtype):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{type(fill_value).__name__} to {self.dtype.name}"
                )
            return super()._validate_fillna_value(fill_value)
        else:
            cudf_obj = as_column(fill_value, nan_as_null=False)
            if not cudf_obj.can_cast_safely(self.dtype):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{cudf_obj.dtype.type.__name__} to "
                    f"{self.dtype.type.__name__}"
                )
            return cudf_obj.astype(self.dtype)

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        """
        Returns true if all the values in self can be
        safely cast to dtype
        """
        # Convert potential pandas extension dtypes to numpy dtypes
        # For example, convert Int32Dtype to np.dtype('int32')
        self_dtype_numpy = (
            np.dtype(self.dtype.numpy_dtype)
            if hasattr(self.dtype, "numpy_dtype")
            else self.dtype
        )
        to_dtype_numpy = (
            np.dtype(to_dtype.numpy_dtype)
            if hasattr(to_dtype, "numpy_dtype")
            else to_dtype
        )

        if len(self) == 0 and to_dtype_numpy.kind in "iufb":
            # An empty column can be cast to any numeric dtype losslessly.
            return True

        if self_dtype_numpy.kind == to_dtype_numpy.kind:
            # Check if self dtype can be safely cast to to_dtype
            # For same kinds, we can compare the sizes
            if self_dtype_numpy <= to_dtype_numpy:
                return True
            else:
                if self_dtype_numpy.kind == "f":
                    # Exclude 'np.inf', '-np.inf'
                    not_inf = (self != np.inf) & (self != -np.inf)
                    col = self.apply_boolean_mask(not_inf)
                else:
                    col = self

                min_ = col.min()
                # TODO: depending on implementation of cudf scalar and future
                # refactor of min/max, change the test method
                if np.isnan(min_):
                    # Column contains only infs
                    return True

                # Kinds are the same but to_dtype is smaller
                if "float" in to_dtype_numpy.name:
                    finfo = np.finfo(to_dtype_numpy)
                    lower_: int | float
                    upper_: int | float
                    # assignment ignore not needed for numpy>=2.4.0
                    lower_, upper_ = finfo.min, finfo.max  # type: ignore[assignment,unused-ignore]

                    # Check specifically for np.pi values when casting to lower precision
                    if self_dtype_numpy.itemsize > to_dtype_numpy.itemsize:
                        # Check if column contains pi value
                        if len(col) > 0:
                            # Create a simple column with pi to test if the precision matters
                            pi_col = self == np.pi
                            # Test if pi can be correctly represented after casting
                            if pi_col.any():
                                # If pi is present, we cannot safely cast to lower precision
                                return False
                elif "int" in to_dtype_numpy.name:
                    iinfo = np.iinfo(to_dtype_numpy)
                    lower_, upper_ = iinfo.min, iinfo.max

                return (min_ >= lower_) and (col.max() <= upper_)

        # want to cast int to uint
        elif self_dtype_numpy.kind == "i" and to_dtype_numpy.kind == "u":
            i_max_ = np.iinfo(self_dtype_numpy).max
            u_max_ = np.iinfo(to_dtype_numpy).max

            return (self.min() >= 0) and (
                (i_max_ <= u_max_) or (self.max() <= u_max_)
            )

        # want to cast uint to int
        elif self_dtype_numpy.kind == "u" and to_dtype_numpy.kind == "i":
            u_max_ = np.iinfo(self_dtype_numpy).max
            i_max_ = np.iinfo(to_dtype_numpy).max

            return (u_max_ <= i_max_) or (self.max() <= i_max_)

        # want to cast int to float
        elif (
            self_dtype_numpy.kind in {"i", "u"} and to_dtype_numpy.kind == "f"
        ):
            info = np.finfo(to_dtype_numpy)
            biggest_exact_int = 2 ** (info.nmant + 1)
            if (self.min() >= -biggest_exact_int) and (
                self.max() <= biggest_exact_int
            ):
                return True
            else:
                filled = self.fillna(0)
                return (
                    filled.astype(to_dtype).astype(filled.dtype) == filled
                ).all()

        # want to cast float to int:
        elif self_dtype_numpy.kind == "f" and to_dtype_numpy.kind in {
            "i",
            "u",
        }:
            if self.nan_count > 0:
                return False
            iinfo = np.iinfo(to_dtype_numpy)
            min_, max_ = iinfo.min, iinfo.max

            # best we can do is hope to catch it here and avoid compare
            # Use Python floats, which have precise comparison for float64.
            # NOTE(seberg): it would make sense to limit to the mantissa range.
            min_val, max_val = self.minmax()
            if (float(min_val) >= min_) and (float(max_val) <= max_):
                filled = self.fillna(0)
                return (filled % 1 == 0).all()
            else:
                return False

        return False

    def _reduction_result_dtype(self, reduction_op: str) -> DtypeObj:
        if reduction_op in {"sum", "product"}:
            if self.dtype.kind == "f":
                return self.dtype
            elif self.dtype.kind == "u":
                return np.dtype("uint64")
            elif self.dtype.kind == "b" and isinstance(
                self.dtype, pd.ArrowDtype
            ):
                return np.dtype("uint64")
            return np.dtype("int64")
        elif reduction_op == "sum_of_squares":
            return find_common_type((self.dtype, np.dtype(np.uint64)))
        elif reduction_op in {"var", "std", "mean"}:
            if self.dtype.kind == "f":
                return self.dtype
            else:
                return np.dtype("float64")

        return super()._reduction_result_dtype(reduction_op)

    def digitize(self, bins: np.ndarray, right: bool = False) -> Self:
        """Return the indices of the bins to which each value in column belongs.

        Parameters
        ----------
        bins : np.ndarray
            1-D column-like object of bins with same type as `column`, should be
            monotonically increasing.
        right : bool
            Indicates whether interval contains the right or left bin edge.

        Returns
        -------
        A column containing the indices
        """
        if self.dtype != bins.dtype:
            raise ValueError(
                "digitize() expects bins and input column have the same dtype."
            )

        bin_col = as_column(bins, dtype=bins.dtype)
        if bin_col.nullable:
            raise ValueError("`bins` cannot contain null entries.")

        return cast(
            "Self",
            PylibcudfFunction(
                getattr(
                    plc.search,
                    "lower_bound" if right else "upper_bound",
                ),
                fixed_dtype_policy(
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                ),
            ).execute_with_args(
                ColumnList(bin_col),
                ColumnList(self),
                [plc.types.Order.ASCENDING],
                [plc.types.NullOrder.BEFORE],
            ),
        )


def _normalize_find_and_replace_input(
    input_column_dtype: DtypeObj, col_to_normalize: ColumnBase | list
) -> ColumnBase:
    normalized_column = as_column(
        col_to_normalize,
        dtype=input_column_dtype if len(col_to_normalize) <= 0 else None,
    )
    col_to_normalize_dtype = normalized_column.dtype
    if isinstance(col_to_normalize, list):
        if normalized_column.is_all_null:
            normalized_column = normalized_column.astype(input_column_dtype)
        if normalized_column.can_cast_safely(input_column_dtype):
            return normalized_column.astype(input_column_dtype)
        col_to_normalize_dtype = normalized_column._min_column_type(  # type: ignore[attr-defined]
            input_column_dtype
        )
        # Scalar case
        if len(col_to_normalize) == 1:
            if is_na_like(col_to_normalize[0]):
                return normalized_column.astype(input_column_dtype)
            if np.isinf(col_to_normalize[0]):
                return normalized_column
            col_to_normalize_casted = np.array(col_to_normalize[0]).astype(
                col_to_normalize_dtype
            )

            if not np.isnan(col_to_normalize_casted) and (
                col_to_normalize_casted != col_to_normalize[0]
            ):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{col_to_normalize[0]} "
                    f"to {input_column_dtype.name}"
                )
        if normalized_column.can_cast_safely(col_to_normalize_dtype):
            return normalized_column.astype(col_to_normalize_dtype)
    elif hasattr(col_to_normalize, "dtype"):
        col_to_normalize_dtype = col_to_normalize.dtype
    else:
        raise TypeError(f"Type {type(col_to_normalize)} not supported")

    if (
        col_to_normalize_dtype.kind == "f"
        and input_column_dtype.kind in {"i", "u"}
    ) or (col_to_normalize_dtype.num > input_column_dtype.num):
        raise TypeError(
            f"Potentially unsafe cast for non-equivalent "
            f"{col_to_normalize_dtype.name} "
            f"to {input_column_dtype.name}"
        )
    if not normalized_column.can_cast_safely(input_column_dtype):
        return normalized_column
    return normalized_column.astype(input_column_dtype)
