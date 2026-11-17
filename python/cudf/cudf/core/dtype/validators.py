# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf

if TYPE_CHECKING:
    from cudf._typing import DtypeObj


def is_dtype_obj_string(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF string type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF string type.
    """
    return (
        obj == np.dtype("object")
        or isinstance(obj, pd.StringDtype)
        or (
            isinstance(obj, pd.ArrowDtype)
            and (
                pa.types.is_string(obj.pyarrow_dtype)
                or pa.types.is_large_string(obj.pyarrow_dtype)
                or pa.types.is_null(obj.pyarrow_dtype)
            )
        )
    )


def is_dtype_obj_list(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF list type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF list type.
    """
    return isinstance(obj, cudf.ListDtype) or (
        isinstance(obj, pd.ArrowDtype) and pa.types.is_list(obj.pyarrow_dtype)
    )


def is_dtype_obj_struct(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF struct type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF struct type.
    """
    return isinstance(obj, cudf.StructDtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_struct(obj.pyarrow_dtype)
    )


def is_dtype_obj_interval(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF interval type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF interval type.
    """
    return isinstance(obj, cudf.IntervalDtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_interval(obj.pyarrow_dtype)
    )


def is_dtype_obj_decimal(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal type.
    """
    return (
        is_dtype_obj_decimal32(obj)
        or is_dtype_obj_decimal64(obj)
        or is_dtype_obj_decimal128(obj)
    )


def is_dtype_obj_decimal32(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal32 type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal32 type.
    """
    return isinstance(obj, cudf.Decimal32Dtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_decimal32(obj.pyarrow_dtype)
    )


def is_dtype_obj_decimal64(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal64 type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal64 type.
    """
    return isinstance(obj, cudf.Decimal64Dtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_decimal64(obj.pyarrow_dtype)
    )


def is_dtype_obj_decimal128(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal128 type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal128 type.
    """
    return isinstance(obj, cudf.Decimal128Dtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_decimal128(obj.pyarrow_dtype)
    )


def is_dtype_obj_numeric(
    dtype: DtypeObj, include_decimal: bool = True
) -> bool:
    """
    Check whether the provided dtype object is a numeric type.

    Parameters
    ----------
    dtype: DtypeObj
        The dtype object to check.
    include_decimal: bool, default True
        Whether to include decimal types in the check.

    Returns
    -------
    bool
        Whether or not the dtype object is a numeric type.
    """
    valid_kinds = set("iufb")
    is_non_decimal = (
        (isinstance(dtype, np.dtype) and dtype.kind in valid_kinds)
        or (isinstance(dtype, pd.ArrowDtype) and dtype.kind in valid_kinds)
        or (
            isinstance(
                dtype,
                (
                    pd.Int8Dtype,
                    pd.Int16Dtype,
                    pd.Int32Dtype,
                    pd.Int64Dtype,
                    pd.UInt8Dtype,
                    pd.UInt16Dtype,
                    pd.UInt32Dtype,
                    pd.UInt64Dtype,
                    pd.Float32Dtype,
                    pd.Float64Dtype,
                    pd.BooleanDtype,
                ),
            )
        )
    )
    if include_decimal:
        return is_non_decimal or is_dtype_obj_decimal(dtype)
    else:
        return is_non_decimal


def is_dtype_obj_datetime_tz(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a datetime with timezone type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
    """
    return isinstance(obj, pd.DatetimeTZDtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_timestamp(obj.pyarrow_dtype)
        and obj.pyarrow_dtype.tz is not None
    )
