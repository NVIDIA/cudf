# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from pandas.core.computation.common import result_type_many

import pylibcudf as plc

import cudf
from cudf.core._internals.timezones import get_compatible_timezone

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cudf._typing import DtypeObj, ScalarLike
    from cudf.core.dtypes import DecimalDtype

DEFAULT_STRING_DTYPE = pd.StringDtype(na_value=np.nan)
np_dtypes_to_pandas_dtypes: dict[
    np.dtype[Any], pd.core.dtypes.base.ExtensionDtype
] = {
    np.dtype("uint8"): pd.UInt8Dtype(),
    np.dtype("uint16"): pd.UInt16Dtype(),
    np.dtype("uint32"): pd.UInt32Dtype(),
    np.dtype("uint64"): pd.UInt64Dtype(),
    np.dtype("int8"): pd.Int8Dtype(),
    np.dtype("int16"): pd.Int16Dtype(),
    np.dtype("int32"): pd.Int32Dtype(),
    np.dtype("int64"): pd.Int64Dtype(),
    np.dtype("bool_"): pd.BooleanDtype(),
    np.dtype("object"): DEFAULT_STRING_DTYPE,
    np.dtype("str"): DEFAULT_STRING_DTYPE,
    np.dtype("float32"): pd.Float32Dtype(),
    np.dtype("float64"): pd.Float64Dtype(),
}
pandas_dtypes_to_np_dtypes = {
    pd_dtype: np_dtype
    for np_dtype, pd_dtype in np_dtypes_to_pandas_dtypes.items()
}
pandas_dtypes_to_np_dtypes[pd.StringDtype("pyarrow")] = np.dtype("object")

pyarrow_dtypes_to_pandas_dtypes = {
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.string(): pd.StringDtype(),
}


SIGNED_INTEGER_TYPES = {"int8", "int16", "int32", "int64"}
UNSIGNED_TYPES = {"uint8", "uint16", "uint32", "uint64"}
INTEGER_TYPES = SIGNED_INTEGER_TYPES | UNSIGNED_TYPES
FLOAT_TYPES = {"float32", "float64"}
SIGNED_TYPES = SIGNED_INTEGER_TYPES | FLOAT_TYPES
NUMERIC_TYPES = SIGNED_TYPES | UNSIGNED_TYPES
DATETIME_TYPES = {
    "datetime64[s]",
    "datetime64[ms]",
    "datetime64[us]",
    "datetime64[ns]",
}
TIMEDELTA_TYPES = {
    "timedelta64[s]",
    "timedelta64[ms]",
    "timedelta64[us]",
    "timedelta64[ns]",
}
OTHER_TYPES = {"bool", "category", "str"}
STRING_TYPES = {"object", "str", "string"}
BOOL_TYPES = {"bool"}
ALL_TYPES = NUMERIC_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | OTHER_TYPES


def _find_common_type_decimal(dtypes: Iterable[DecimalDtype]) -> DecimalDtype:
    # Find the largest scale and the largest difference between
    # precision and scale of the columns to be concatenated
    s = max(dtype.scale for dtype in dtypes)
    lhs = max(dtype.precision - dtype.scale for dtype in dtypes)
    # Combine to get the necessary precision and clip at the maximum
    # precision
    p = s + lhs

    if p > cudf.Decimal64Dtype.MAX_PRECISION:
        return cudf.Decimal128Dtype(
            min(cudf.Decimal128Dtype.MAX_PRECISION, p), s
        )
    elif p > cudf.Decimal32Dtype.MAX_PRECISION:
        return cudf.Decimal64Dtype(
            min(cudf.Decimal64Dtype.MAX_PRECISION, p), s
        )
    else:
        return cudf.Decimal32Dtype(
            min(cudf.Decimal32Dtype.MAX_PRECISION, p), s
        )


def cudf_dtype_to_pa_type(dtype: DtypeObj) -> pa.DataType:
    """Given a cudf pandas dtype, converts it into the equivalent cuDF
    Python dtype.
    """
    from cudf.core.dtype.validators import is_dtype_obj_string

    dtype = getattr(dtype, "numpy_dtype", dtype)
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "No conversion from Categorical to pyarrow type"
        )
    elif isinstance(
        dtype,
        (cudf.StructDtype, cudf.ListDtype, cudf.core.dtypes.DecimalDtype),
    ):
        return dtype.to_arrow()
    elif isinstance(dtype, pd.DatetimeTZDtype):
        return pa.timestamp(dtype.unit, str(dtype.tz))
    elif is_dtype_obj_string(dtype):
        return pa.string()
    else:
        return pa.from_numpy_dtype(dtype)


def cudf_dtype_from_pa_type(typ: pa.DataType) -> DtypeObj:
    """Given a pyarrow dtype, converts it into the equivalent cudf dtype."""
    if pa.types.is_list(typ):
        return cudf.ListDtype.from_arrow(typ)
    elif pa.types.is_struct(typ):
        return cudf.StructDtype.from_arrow(typ)
    elif pa.types.is_decimal(typ):
        if isinstance(typ, pa.Decimal256Type):
            raise NotImplementedError("cudf does not support Decimal256Type")
        if isinstance(typ, pa.Decimal32Type):
            return cudf.Decimal32Dtype.from_arrow(typ)
        if isinstance(typ, pa.Decimal64Type):
            return cudf.Decimal64Dtype.from_arrow(typ)
        return cudf.Decimal128Dtype.from_arrow(typ)
    elif pa.types.is_timestamp(typ) and typ.tz is not None:
        return get_compatible_timezone(pd.DatetimeTZDtype(typ.unit, typ.tz))
    elif (
        pa.types.is_large_string(typ)
        or pa.types.is_string(typ)
        or pa.types.is_string_view(typ)
    ):
        return DEFAULT_STRING_DTYPE
    elif pa.types.is_date(typ):
        # typ.to_pandas_dtype() produces np.dtype("datetime64[ms]").
        # Conversely pylibcudf will produce TIMESTAMP_DAYS for date types - the most
        # correct answer - and to match pandas cudf will cast to TIMESTAMP_SECONDS
        # (see ColumnBase._wrap_buffers). Therefore we should return a seconds
        # resolution datetime type here. The pyarrow may be changed
        # https://github.com/apache/arrow/issues/49168.
        return np.dtype("datetime64[s]")
    elif pa.types.is_null(typ):
        # Similar to PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.EMPTY]
        return np.dtype("object")
    elif isinstance(typ, ArrowIntervalType):
        return cudf.IntervalDtype.from_arrow(typ)
    else:
        try:
            return cudf.api.types.pandas_dtype(typ.to_pandas_dtype())
        except NotImplementedError as err:
            raise TypeError(f"Unsupported type: {typ}") from err


def is_column_like(obj):
    """
    This function checks if the given `obj`
    is a column-like (Series, Index...)
    type or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is column-like or not.
    """
    return (
        isinstance(
            obj,
            (
                cudf.core.column.ColumnBase,
                cudf.Series,
                cudf.Index,
                pd.Series,
                pd.Index,
            ),
        )
        or (
            hasattr(obj, "__cuda_array_interface__")
            and len(obj.__cuda_array_interface__["shape"]) == 1
        )
        or (
            hasattr(obj, "__array_interface__")
            and len(obj.__array_interface__["shape"]) == 1
        )
    )


def can_convert_to_column(obj):
    """
    This function checks if the given `obj`
    can be used to create a column or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is column-compatible or not.
    """
    return is_column_like(obj) or cudf.api.types.is_list_like(obj)


def min_signed_type(x: int, min_size: int = 8) -> np.dtype:
    """
    Return the smallest *signed* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in (np.int8, np.int16, np.int32, np.int64):
        dtype: np.dtype[Any] = np.dtype(int_dtype)
        if (dtype.itemsize * 8) >= min_size:
            if np.iinfo(int_dtype).min <= x <= np.iinfo(int_dtype).max:
                return dtype
    # resort to using `int64` and let numpy raise appropriate exception:
    return np.int64(x).dtype


def min_unsigned_type(x: int, min_size: int = 8) -> np.dtype:
    """
    Return the smallest *unsigned* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        dtype: np.dtype[Any] = np.dtype(int_dtype)
        if (dtype.itemsize * 8) >= min_size:
            if 0 <= x <= np.iinfo(int_dtype).max:
                return dtype
    # resort to using `uint64` and let numpy raise appropriate exception:
    return np.uint64(x).dtype


def is_mixed_with_object_dtype(lhs, rhs):
    if isinstance(lhs.dtype, cudf.CategoricalDtype):
        return is_mixed_with_object_dtype(lhs.dtype.categories, rhs)
    elif isinstance(rhs.dtype, cudf.CategoricalDtype):
        return is_mixed_with_object_dtype(lhs, rhs.dtype.categories)

    return (
        cudf.api.types.is_string_dtype(lhs.dtype)
        and not cudf.api.types.is_string_dtype(rhs.dtype)
    ) or (
        cudf.api.types.is_string_dtype(rhs.dtype)
        and not cudf.api.types.is_string_dtype(lhs.dtype)
    )


def _get_nan_for_dtype(dtype: DtypeObj) -> ScalarLike:
    """Return the appropriate NaN/NaT value for the given dtype.

    Returns the null value for the dtype (e.g., np.float64('nan'),
    pd.NaT, or the dtype's ``na_value`` for pandas nullable extension
    dtypes).
    """
    if dtype.kind in "mM":
        # pandas datetime/timedelta reductions return the pd.NaT
        # singleton for null results, and callers rely on identity
        # (``result is pd.NaT``), not a unit-qualified
        # np.datetime64/np.timedelta64 "NaT".
        return pd.NaT
    elif dtype.kind == "f":
        if is_pandas_nullable_extension_dtype(dtype):
            return dtype.na_value
        return dtype.type("nan")
    else:
        if isinstance(dtype, pd.StringDtype) or (
            is_pandas_nullable_extension_dtype(dtype)
            and getattr(dtype, "kind", "c") in "biuU"
        ):
            # dtype.na_value is pd.NA for masked, "string", and arrow
            # string (kind "U") dtypes, and the np.nan float singleton
            # for "str" dtypes (pandas>=3). Reductions like
            # min/max(skipna=False) must return exactly that object so
            # that ``result is dtype.na_value`` holds. Kind "O" extension
            # dtypes other than StringDtype (e.g. categorical, arrow
            # decimal/binary) are excluded: pandas coerces their
            # skew/cov/corr results to a float NaN, which the fallback
            # below matches.
            return dtype.na_value
        return np.float64("nan")


def find_common_type(dtypes: Iterable[DtypeObj]) -> DtypeObj:
    """
    Wrapper over np.result_type to handle cudf specific types.

    Parameters
    ----------
    dtypes : iterable
        sequence of dtypes to find common types

    Returns
    -------
    dtype : DtypeObj

    Raises
    ------
    ValueError
        If ``dtypes`` is empty (matches ``pandas.core.dtypes.cast.find_common_type``).
    """
    if len(dtypes) == 0:  # type: ignore[arg-type]
        # Match pandas.core.dtypes.cast.find_common_type, which raises.
        raise ValueError("no types given")

    pandas_compatible = cudf.get_option("mode.pandas_compatible")

    # Early exit for categoricals since they're not hashable and therefore
    # can't be put in a set.
    if any(isinstance(dtype, cudf.CategoricalDtype) for dtype in dtypes):
        if all(
            (
                isinstance(dtype, cudf.CategoricalDtype)
                and (not dtype.ordered if hasattr(dtype, "ordered") else True)
            )
            for dtype in dtypes
        ):
            if len({dtype._categories.dtype for dtype in dtypes}) == 1:  # type: ignore[union-attr]
                return cudf.CategoricalDtype(
                    cudf.core.column.concat_columns(
                        [dtype._categories for dtype in dtypes]  # type: ignore[union-attr]
                    ).unique()
                )
            else:
                raise NotImplementedError(
                    "Only unordered categories of the same underlying type "
                    "may be currently coerced to a common type."
                )
        else:
            # extract the categories' dtype
            non_cat_dtypes = [
                x.categories.dtype
                if isinstance(x, cudf.CategoricalDtype)
                else x
                for x in dtypes
            ]
            return find_common_type(non_cat_dtypes)

    # Aggregate same types
    dtypes = set(dtypes)
    if len(dtypes) == 1:
        return dtypes.pop()

    if any(
        isinstance(dtype, cudf.core.dtypes.DecimalDtype) for dtype in dtypes
    ):
        from cudf.core.dtype.validators import is_dtype_obj_numeric

        if all(
            is_dtype_obj_numeric(dtype, include_decimal=True)
            for dtype in dtypes
        ):
            return _find_common_type_decimal(
                [
                    dtype
                    for dtype in dtypes
                    if isinstance(dtype, cudf.core.dtypes.DecimalDtype)
                ]
            )
        else:
            return DEFAULT_STRING_DTYPE
    elif any(
        isinstance(
            dtype, (cudf.ListDtype, cudf.StructDtype, cudf.IntervalDtype)
        )
        for dtype in dtypes
    ):
        # TODO: As list dtypes allow casting
        # to identical types, improve this logic of returning a
        # common dtype, for example:
        # ListDtype(int64) & ListDtype(int32) common
        # dtype could be ListDtype(int64).
        raise NotImplementedError(
            "Finding a common type for `ListDtype` or `StructDtype` is currently "
            "not supported"
        )

    if pandas_compatible:
        # cudf follows NumPy promotion: bool+int->int, bool+float->float,
        # datetime64+timedelta64->datetime64. Pandas returns `object` for
        # these mixes. Raise so that, when used as the cudf.pandas fast path
        # for `pandas.core.dtypes.cast.find_common_type`, we fall back to
        # pandas' implementation rather than silently producing a different
        # answer.
        kinds = {dtype.kind for dtype in dtypes if isinstance(dtype, np.dtype)}
        if "b" in kinds and kinds & set("iuf"):
            raise NotImplementedError(
                "Common type of bool with numeric dtypes is not supported "
                "in pandas-compatible mode."
            )
        if "M" in kinds and "m" in kinds:
            raise NotImplementedError(
                "Common type of datetime64 with timedelta64 is not supported "
                "in pandas-compatible mode."
            )

    try:
        common_dtype = np.result_type(*dtypes)  # noqa: TID251
    except TypeError:
        common_dtype = result_type_many(*dtypes)

    if common_dtype == np.dtype(np.float16):
        return np.dtype(np.float32)
    return common_dtype


def _maybe_convert_to_default_type(dtype: DtypeObj) -> DtypeObj:
    """Convert `dtype` to default if specified by user.

    If not specified, return as is.
    """
    if ib := cudf.get_option("default_integer_bitwidth"):
        if dtype.kind == "i":
            return np.dtype(f"i{ib // 8}")
        elif dtype.kind == "u":
            return np.dtype(f"u{ib // 8}")
    if (fb := cudf.get_option("default_float_bitwidth")) and dtype.kind == "f":
        return np.dtype(f"f{fb // 8}")
    return dtype


def _get_base_dtype(dtype: pd.DatetimeTZDtype | pd.ArrowDtype) -> np.dtype:
    if isinstance(dtype, pd.DatetimeTZDtype):
        # TODO: replace below with dtype.base when Pandas 2.1.0 is the minimum version we support:
        # https://github.com/pandas-dev/pandas/pull/52706
        return np.dtype(f"<M8[{dtype.unit}]")
    else:
        return dtype.numpy_dtype


def pyarrow_dtype_to_cudf_dtype(dtype: pd.ArrowDtype) -> DtypeObj:
    """Convert a pandas.ArrowDtype to a default cuDF dtype."""
    if pa.types.is_date(dtype.pyarrow_dtype):
        raise TypeError(f"Unsupported type: {dtype.pyarrow_dtype}")
    else:
        return cudf_dtype_from_pa_type(dtype.pyarrow_dtype)


def is_pandas_nullable_numpy_dtype(dtype_to_check) -> bool:
    return isinstance(
        dtype_to_check,
        (
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.Float32Dtype,
            pd.Float64Dtype,
            pd.BooleanDtype,
            pd.StringDtype,
        ),
    )


def is_pandas_nullable_extension_dtype(
    dtype_to_check: Any,
) -> TypeGuard[pd.core.dtypes.base.ExtensionDtype]:
    if is_pandas_nullable_numpy_dtype(dtype_to_check) or isinstance(
        dtype_to_check, pd.ArrowDtype
    ):
        return True
    elif isinstance(dtype_to_check, pd.CategoricalDtype):
        if dtype_to_check.categories is None:
            return False
        return is_pandas_nullable_extension_dtype(
            dtype_to_check.categories.dtype
        )
    elif isinstance(dtype_to_check, pd.IntervalDtype):
        return is_pandas_nullable_extension_dtype(dtype_to_check.subtype)
    return False


def dtype_to_pylibcudf_type(dtype: DtypeObj) -> plc.DataType:
    if isinstance(dtype, pd.ArrowDtype):
        dtype = pyarrow_dtype_to_cudf_dtype(dtype)
    if isinstance(dtype, cudf.ListDtype):
        return plc.DataType(plc.TypeId.LIST)
    elif isinstance(dtype, cudf.IntervalDtype):
        return plc.DataType(plc.TypeId.STRUCT)
    elif isinstance(dtype, cudf.StructDtype):
        return plc.DataType(plc.TypeId.STRUCT)
    elif isinstance(dtype, cudf.Decimal128Dtype):
        tid = plc.TypeId.DECIMAL128
        return plc.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal64Dtype):
        tid = plc.TypeId.DECIMAL64
        return plc.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal32Dtype):
        tid = plc.TypeId.DECIMAL32
        return plc.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.CategoricalDtype):
        dtype = dtype._codes_dtype
    # libcudf types don't support timezones so convert to the base type
    elif isinstance(dtype, pd.DatetimeTZDtype):
        dtype = _get_base_dtype(dtype)
    elif isinstance(dtype, pd.StringDtype):
        dtype = np.dtype("object")
    elif is_pandas_nullable_numpy_dtype(dtype):
        dtype = dtype.numpy_dtype  # type: ignore[union-attr]
    return plc.DataType(SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES[dtype])


def dtype_to_pandas_arrowdtype(dtype: DtypeObj) -> pd.ArrowDtype:
    if isinstance(dtype, pd.ArrowDtype):
        return dtype
    elif isinstance(
        dtype,
        (cudf.ListDtype, cudf.StructDtype, cudf.core.dtypes.DecimalDtype),
    ):
        return pd.ArrowDtype(dtype.to_arrow())
    elif isinstance(dtype, pd.StringDtype):
        return pd.ArrowDtype(pa.large_string())
    # libcudf types don't support timezones so convert to the base type
    elif isinstance(dtype, pd.DatetimeTZDtype):
        return pd.ArrowDtype(pa.timestamp(dtype.unit, str(dtype.tz)))
    elif isinstance(dtype, pd.StringDtype):
        return pd.ArrowDtype(pa.string())
    elif is_pandas_nullable_numpy_dtype(dtype):
        dtype = dtype.numpy_dtype  # type: ignore[union-attr]
    elif dtype == np.dtype("object"):
        # pa.from_numpy_dtype doesn't map object to string
        dtype = np.dtype("str")
    return pd.ArrowDtype(pa.from_numpy_dtype(dtype))


def dtype_to_pandas_nullable_extension_type(dtype) -> DtypeObj:
    if isinstance(dtype, pd.ArrowDtype):
        return dtype_to_pandas_nullable_extension_type(
            pyarrow_dtype_to_cudf_dtype(dtype)
        )
    else:
        return np_dtypes_to_pandas_dtypes.get(dtype, dtype)


def get_dtype_of_same_kind(source_dtype: DtypeObj, target_dtype: DtypeObj):
    """
    Given a dtype, return a dtype of the same kind.
    If no such dtype exists, return the default dtype.
    """
    if isinstance(source_dtype, pd.ArrowDtype):
        # Preserve the source Arrow string variant (string vs large_string)
        # when the target is a string-like dtype.
        if pa.types.is_string(
            source_dtype.pyarrow_dtype
        ) or pa.types.is_large_string(source_dtype.pyarrow_dtype):
            if isinstance(target_dtype, pd.StringDtype) or (
                isinstance(target_dtype, pd.ArrowDtype)
                and (
                    pa.types.is_string(target_dtype.pyarrow_dtype)
                    or pa.types.is_large_string(target_dtype.pyarrow_dtype)
                )
            ):
                return source_dtype
        return dtype_to_pandas_arrowdtype(target_dtype)
    elif is_pandas_nullable_extension_dtype(source_dtype):
        # Preserve StringDtype storage/na_value when target is also a StringDtype.
        if isinstance(source_dtype, pd.StringDtype) and isinstance(
            target_dtype, pd.StringDtype
        ):
            return source_dtype
        if (
            isinstance(source_dtype, pd.StringDtype)
            and source_dtype.na_value is np.nan
        ):
            return target_dtype
        elif (
            isinstance(source_dtype, pd.StringDtype)
            and source_dtype.storage == "pyarrow"
        ):
            return dtype_to_pandas_arrowdtype(target_dtype)
        return dtype_to_pandas_nullable_extension_type(target_dtype)
    else:
        return target_dtype


def get_dtype_of_same_type(lhs_dtype: DtypeObj, rhs_dtype: DtypeObj):
    """
    Given two dtypes, checks if `lhs_dtype` translates to same libcudf
    type as `rhs_dtype`, if yes, returns `lhs_dtype`.
    Else, returns `rhs_dtype` in `lhs_dtype`'s kind.
    """
    if dtype_to_pylibcudf_type(lhs_dtype) == dtype_to_pylibcudf_type(
        rhs_dtype
    ):
        return lhs_dtype
    else:
        return get_dtype_of_same_kind(lhs_dtype, rhs_dtype)


def dtype_from_pylibcudf_column(col: plc.Column) -> DtypeObj:
    type_ = col.type()
    tid = type_.id()

    if tid == plc.TypeId.LIST:
        child = col.list_view().child()
        return cudf.ListDtype(dtype_from_pylibcudf_column(child))
    elif tid == plc.TypeId.STRUCT:
        fields = {
            str(i): dtype_from_pylibcudf_column(col.child(i))
            for i in range(col.num_children())
        }
        return cudf.StructDtype(fields)
    elif tid == plc.TypeId.DECIMAL64:
        return cudf.Decimal64Dtype(
            precision=cudf.Decimal64Dtype.MAX_PRECISION, scale=-type_.scale()
        )
    elif tid == plc.TypeId.DECIMAL32:
        return cudf.Decimal32Dtype(
            precision=cudf.Decimal32Dtype.MAX_PRECISION, scale=-type_.scale()
        )
    elif tid == plc.TypeId.DECIMAL128:
        return cudf.Decimal128Dtype(
            precision=cudf.Decimal128Dtype.MAX_PRECISION, scale=-type_.scale()
        )
    elif tid == plc.TypeId.STRING:
        return DEFAULT_STRING_DTYPE
    else:
        return PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[tid]


def is_arrow_null_dtype(dtype: DtypeObj) -> bool:
    """Check if dtype is a pandas ArrowDtype wrapping pa.null()."""
    return isinstance(dtype, pd.ArrowDtype) and pa.types.is_null(
        dtype.pyarrow_dtype
    )


SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES: dict[np.dtype[Any], plc.types.TypeId] = {
    np.dtype("int8"): plc.types.TypeId.INT8,
    np.dtype("int16"): plc.types.TypeId.INT16,
    np.dtype("int32"): plc.types.TypeId.INT32,
    np.dtype("int64"): plc.types.TypeId.INT64,
    np.dtype("uint8"): plc.types.TypeId.UINT8,
    np.dtype("uint16"): plc.types.TypeId.UINT16,
    np.dtype("uint32"): plc.types.TypeId.UINT32,
    np.dtype("uint64"): plc.types.TypeId.UINT64,
    np.dtype("float32"): plc.types.TypeId.FLOAT32,
    np.dtype("float64"): plc.types.TypeId.FLOAT64,
    np.dtype("datetime64[s]"): plc.types.TypeId.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): plc.types.TypeId.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): plc.types.TypeId.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): plc.types.TypeId.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): plc.types.TypeId.STRING,
    np.dtype("str"): plc.types.TypeId.STRING,
    np.dtype("bool"): plc.types.TypeId.BOOL8,
    np.dtype("timedelta64[s]"): plc.types.TypeId.DURATION_SECONDS,
    np.dtype("timedelta64[ms]"): plc.types.TypeId.DURATION_MILLISECONDS,
    np.dtype("timedelta64[us]"): plc.types.TypeId.DURATION_MICROSECONDS,
    np.dtype("timedelta64[ns]"): plc.types.TypeId.DURATION_NANOSECONDS,
}
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES = {
    plc_type: np_type
    for np_type, plc_type in SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES.items()
}
# There's no equivalent to EMPTY in cudf.  We translate EMPTY
# columns from libcudf to ``object`` columns of all nulls in Python.
# ``object`` is chosen to match pandas behavior.
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.EMPTY] = np.dtype("object")
# TIMESTAMP_DAYS is converted to TIMESTAMP_SECONDS to match the default resolution
# choice used by pandas for datetime.date objects.
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.TIMESTAMP_DAYS] = np.dtype(
    "datetime64[s]"
)
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.STRUCT] = np.dtype(
    "object"
)
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.LIST] = np.dtype("object")
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.STRING] = np.dtype(
    "object"
)

SIZE_TYPE_DTYPE = PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.SIZE_TYPE_ID]
