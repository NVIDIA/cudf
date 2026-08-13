# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from decimal import Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyarrow as pa
from packaging import version

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core._internals import binaryop
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.column.numerical_base import NumericalBaseColumn
from cudf.core.dtypes import (
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    DecimalDtype,
)
from cudf.core.mixins import BinaryOperand
from cudf.utils.dtypes import (
    cudf_dtype_to_pa_type,
    get_dtype_of_same_kind,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn


def _to_plc_scalar(scalar: int | Decimal, dtype: DecimalDtype) -> plc.Scalar:
    pa_scalar = pa.scalar(scalar, type=dtype.to_arrow())
    return pa_scalar_to_plc_scalar(pa_scalar)


class DecimalColumn(NumericalBaseColumn):
    """Base column for decimal32, decimal64 or decimal128 columns"""

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    @cached_property
    def scale(self) -> int:
        if isinstance(self.dtype, DecimalDtype):
            return self.dtype.scale
        else:
            return cast("pd.ArrowDtype", self.dtype).pyarrow_dtype.scale

    @cached_property
    def precision(self) -> int:
        if isinstance(self.dtype, DecimalDtype):
            return self.dtype.precision
        else:
            return cast("pd.ArrowDtype", self.dtype).pyarrow_dtype.precision

    def _adjust_reduce_result_dtype(
        self,
        reduction_op: str,
        col_dtype: DtypeObj,
        plc_scalar: plc.Scalar,
    ) -> DtypeObj:
        """Adjust decimal precision based on reduction operation."""
        scale = -plc_scalar.type().scale()
        # Narrow type for mypy - we know col_dtype is a decimal type
        assert isinstance(col_dtype, DecimalDtype)
        p = col_dtype.precision
        # https://learn.microsoft.com/en-us/sql/t-sql/data-types/precision-scale-and-length-transact-sql
        nrows = len(self)
        if reduction_op in {"min", "max"}:
            new_p = p
        elif reduction_op == "sum":
            new_p = p + nrows - 1
        elif reduction_op == "product":
            new_p = p * nrows + nrows - 1
        elif reduction_op == "sum_of_squares":
            new_p = 2 * p + nrows
        else:
            raise NotImplementedError(
                f"{reduction_op} not implemented for decimal types."
            )
        precision = max(min(new_p, col_dtype.MAX_PRECISION), 0)
        return type(col_dtype)(precision, scale)

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Decimals are not yet supported via `__cuda_array_interface__`"
        )

    def as_decimal_column(
        self,
        dtype: DecimalDtype,
    ) -> DecimalColumn:
        if isinstance(dtype, DecimalDtype) and dtype.scale < self.scale:
            warnings.warn(
                "cuDF truncates when downcasting decimals to a lower scale. "
                "To round, use Series.round() or DataFrame.round()."
            )

        if dtype == self.dtype:
            return self
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        if isinstance(dtype, np.dtype) and dtype.kind == "U":
            dtype = np.dtype("object")
        if len(self) > 0:
            with self.access(mode="read", scope="internal"):
                plc_column = (
                    plc.strings.convert.convert_fixed_point.from_fixed_point(
                        self.plc_column,
                    )
                )
                return cast(
                    "cudf.core.column.string.StringColumn",
                    ColumnBase.create(plc_column, dtype),
                )
        else:
            return cast(
                "cudf.core.column.StringColumn",
                cudf.core.column.column_empty(0, dtype=dtype),
            )

    def __pow__(self, other: ColumnBinaryOperand) -> ColumnBase:
        if isinstance(other, int):
            if other == 0:
                res = cudf.core.column.as_column(
                    1, dtype=self.dtype, length=len(self)
                )
                if self.nullable:
                    res = res.set_mask(self.mask, self.null_count)
                return res
            elif other < 0:
                raise TypeError("Power of negative integers not supported.")
            res = self
            for _ in range(other - 1):
                res = self * res
            return res
        else:
            raise NotImplementedError(
                f"__pow__ of types {self.dtype} and {type(other)} is "
                "not yet implemented."
            )

    # Decimals in libcudf don't support truediv, see
    # https://github.com/rapidsai/cudf/pull/7435 for explanation.
    def __truediv__(self, other: ColumnBinaryOperand) -> ColumnBase:
        return self._binaryop(other, "__div__")

    def __rtruediv__(self, other: ColumnBinaryOperand) -> ColumnBase:
        return self._binaryop(other, "__rdiv__")

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)

        # Inline _normalize_binop_operand functionality
        if isinstance(other, ColumnBase):
            if not isinstance(other, NumericalBaseColumn):
                return NotImplemented
            elif other.dtype.kind == "f":
                return self.astype(other.dtype)._binaryop(other, op)
            elif other.dtype.kind == "b":
                raise TypeError(
                    "Decimal columns only support binary operations with "
                    "integer numerical columns."
                )
            elif other.dtype.kind in {"i", "u"}:
                other = other.astype(
                    type(self.dtype)(self.dtype.MAX_PRECISION, 0)  # type: ignore[call-overload, union-attr]
                )
            elif not isinstance(self.dtype, other.dtype.__class__):
                # This branch occurs if we have a DecimalColumn of a
                # different size (e.g. 64 instead of 32).
                if (
                    self.precision == other.dtype.precision  # type: ignore[union-attr]
                    and self.scale == other.dtype.scale  # type: ignore[union-attr]
                ):
                    other = other.astype(self.dtype)
            other_cudf_dtype = other.dtype
        elif isinstance(other, (int, Decimal)):
            if cudf.get_option("mode.pandas_compatible") and not isinstance(
                self.dtype, DecimalDtype
            ):
                raise NotImplementedError(
                    "binary operations with arbitrary decimal types are not supported in pandas compatibility mode"
                )
            other_cudf_dtype = self.dtype._from_decimal(Decimal(other))  # type: ignore[union-attr]
        elif isinstance(other, float):
            return self._binaryop(as_column(other, length=len(self)), op)
        elif is_na_like(other):
            other = pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
            other_cudf_dtype = self.dtype
        else:
            return NotImplemented
        if reflect:
            lhs_dtype = other_cudf_dtype
            rhs_dtype = self.dtype
            lhs = other
            rhs = self
        else:
            lhs_dtype = self.dtype
            rhs_dtype = other_cudf_dtype
            lhs = self
            rhs = other  # type: ignore[assignment]

        # Binary Arithmetics between decimal columns. `Scale` and `precision`
        # are computed outside of libcudf
        if op in {"__add__", "__sub__", "__mul__", "__div__"}:
            output_type = _get_decimal_type(lhs_dtype, rhs_dtype, op)  # type: ignore[arg-type]
            new_lhs_dtype = type(output_type)(
                lhs_dtype.precision,  # type: ignore[union-attr]
                lhs_dtype.scale,  # type: ignore[union-attr]
            )
            new_rhs_dtype = type(output_type)(
                rhs_dtype.precision,  # type: ignore[union-attr]
                rhs_dtype.scale,  # type: ignore[union-attr]
            )
            lhs_binop: plc.Scalar | ColumnBase
            rhs_binop: plc.Scalar | ColumnBase
            if isinstance(lhs, (int, Decimal)):
                lhs_binop = _to_plc_scalar(lhs, new_lhs_dtype)
            else:
                lhs_binop = lhs.astype(new_lhs_dtype)  # type: ignore[union-attr]  # (lhs is decimal column here)
            if isinstance(rhs, (int, Decimal)):
                rhs_binop = _to_plc_scalar(rhs, new_rhs_dtype)
            else:
                rhs_binop = rhs.astype(new_rhs_dtype)
            result = binaryop.binaryop(lhs_binop, rhs_binop, op, output_type)
            return result
        elif op in {
            "__eq__",
            "__ne__",
            "__lt__",
            "__gt__",
            "__le__",
            "__ge__",
        }:
            lhs_comp: plc.Scalar | ColumnBase = lhs  # type: ignore[assignment]
            rhs_comp: plc.Scalar | ColumnBase = (
                _to_plc_scalar(rhs, self.dtype)
                if isinstance(rhs, (int, Decimal))
                else rhs
            )
            return binaryop.binaryop(
                lhs_comp,
                rhs_comp,
                op,
                get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
            )
        else:
            raise TypeError(
                f"{op} not supported for the following dtypes: "
                f"{self.dtype}, {other_cudf_dtype}"
            )

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if isinstance(value, np.integer):
            value = value.item()
        if is_scalar(value):
            return self._scalar_to_plc_scalar(value)
        return super()._cast_setitem_value(value)

    def _scalar_to_plc_scalar(self, scalar: ScalarLike) -> plc.Scalar:
        """Return a pylibcudf.Scalar that matches the type of self.dtype"""
        if not isinstance(scalar, pa.Scalar):
            # e.g casting int to decimal type isn't allow, but OK in the constructor?
            pa_scalar = pa.scalar(
                scalar, type=cudf_dtype_to_pa_type(self.dtype)
            )
        else:
            pa_scalar = scalar.cast(cudf_dtype_to_pa_type(self.dtype))
        return pa_scalar_to_plc_scalar(pa_scalar)

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if isinstance(fill_value, (int, Decimal)):
            return super()._validate_fillna_value(fill_value)
        elif isinstance(fill_value, ColumnBase) and (
            isinstance(self.dtype, DecimalDtype) or self.dtype.kind in "iu"
        ):
            return super()._validate_fillna_value(fill_value)
        raise TypeError(
            "Decimal columns only support using fillna with decimal and "
            "integer values"
        )

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    # This overload can be removed once we require pyarrow 20, see
    # https://github.com/apache/arrow/issues/45570
    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        col = self
        if version.parse(pa.__version__) < version.parse("20") and isinstance(
            col.dtype, (Decimal32Dtype, Decimal64Dtype)
        ):
            col = cast(
                "DecimalColumn",
                self.astype(
                    Decimal128Dtype(
                        self.precision,
                        self.scale,
                    )
                ),
            )
        return super(DecimalColumn, col).to_pandas(
            nullable=nullable, arrow_type=arrow_type
        )


def _get_decimal_type(
    lhs_dtype: DecimalDtype,
    rhs_dtype: DecimalDtype,
    op: str,
) -> DecimalDtype:
    """
    Returns the resulting decimal type after calculating
    precision & scale when performing the binary operation
    `op` for the given dtypes.

    For precision & scale calculations see : https://learn.microsoft.com/en-us/sql/t-sql/data-types/precision-scale-and-length-transact-sql
    """

    # This should at some point be hooked up to libcudf's
    # binary_operation_fixed_point_scale
    # Note: libcudf decimal types don't have a concept of precision

    p1, p2 = lhs_dtype.precision, rhs_dtype.precision
    s1, s2 = lhs_dtype.scale, rhs_dtype.scale

    if op in {"__add__", "__sub__"}:
        scale = max(s1, s2)
        precision = scale + max(p1 - s1, p2 - s2) + 1
        if precision > Decimal128Dtype.MAX_PRECISION:
            precision = Decimal128Dtype.MAX_PRECISION
            scale = Decimal128Dtype.MAX_PRECISION - max(p1 - s1, p2 - s2)
    elif op in {"__mul__", "__div__"}:
        if op == "__mul__":
            scale = s1 + s2
            precision = p1 + p2 + 1
        else:
            scale = max(6, s1 + p2 + 1)
            precision = p1 - s1 + s2 + scale
        if precision > Decimal128Dtype.MAX_PRECISION:
            integral = precision - scale
            if integral < 32:
                scale = min(scale, Decimal128Dtype.MAX_PRECISION - integral)
            elif scale > 6 and integral > 32:
                scale = 6
            precision = Decimal128Dtype.MAX_PRECISION
    else:
        raise NotImplementedError()

    try:
        if isinstance(lhs_dtype, type(rhs_dtype)):
            # SCENARIO 1: If `lhs_dtype` & `rhs_dtype` are same, then try to
            # see if `precision` & `scale` can be fit into this type.
            return lhs_dtype.__class__(precision=precision, scale=scale)
        else:
            # SCENARIO 2: If `lhs_dtype` & `rhs_dtype` are of different dtypes,
            # then try to see if `precision` & `scale` can be fit into the type
            # with greater MAX_PRECISION (i.e., the bigger dtype).
            if lhs_dtype.MAX_PRECISION >= rhs_dtype.MAX_PRECISION:
                return lhs_dtype.__class__(precision=precision, scale=scale)
            else:
                return rhs_dtype.__class__(precision=precision, scale=scale)
    except ValueError:
        # Call to _validate fails, which means we need
        # to goto SCENARIO 3.
        pass

    # SCENARIO 3: If either of the above two scenarios fail, then get the
    # MAX_PRECISION of `lhs_dtype` & `rhs_dtype` so that we can only check
    # and return a dtype that is greater than or equal to input dtype that
    # can fit `precision` & `scale`.
    max_precision = max(lhs_dtype.MAX_PRECISION, rhs_dtype.MAX_PRECISION)
    for decimal_type in (
        Decimal32Dtype,
        Decimal64Dtype,
        Decimal128Dtype,
    ):
        if decimal_type.MAX_PRECISION >= max_precision:
            try:
                return decimal_type(precision=precision, scale=scale)
            except ValueError:
                # Call to _validate fails, which means we need
                # to try the next dtype
                continue

    # if we've reached this point, we cannot create a decimal type without
    # overflow; raise an informative error
    raise ValueError(
        f"Performing {op} between columns of type {lhs_dtype!r} and "
        f"{rhs_dtype!r} would result in overflow"
    )
