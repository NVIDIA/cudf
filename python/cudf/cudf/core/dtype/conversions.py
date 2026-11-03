# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from cudf.core.dtype.validators import (
    is_dtype_obj_interval,
    is_dtype_obj_list,
    is_dtype_obj_struct,
)

if TYPE_CHECKING:
    import cudf
    from cudf._typing import DtypeObj


def element_type_from_list_dtype(dtype: DtypeObj) -> DtypeObj:
    """
    Return the element type of a list dtype.
    """
    if not is_dtype_obj_list(dtype):
        raise ValueError(f"Expected a list dtype, got {dtype}")
    if isinstance(dtype, pd.ArrowDtype):
        return pd.ArrowDtype(dtype.pyarrow_dtype.value_type)
    else:
        return cast("cudf.ListDtype", dtype).element_type


def fields_from_struct_dtype(dtype: DtypeObj) -> dict[str, DtypeObj]:
    """
    Return the fields of a struct dtype.
    """
    if not is_dtype_obj_struct(dtype):
        raise ValueError(f"Expected a struct dtype, got {dtype}")
    if isinstance(dtype, pd.ArrowDtype):
        return {
            field.name: pd.ArrowDtype(field.type)
            for field in cast("pd.ArrowDtype", dtype).pyarrow_dtype
        }
    else:
        return cast("cudf.StructDtype", dtype).fields


def subtype_from_interval_dtype(dtype: DtypeObj) -> DtypeObj:
    """
    Return the subtype of an interval dtype.
    """
    if not is_dtype_obj_interval(dtype):
        raise ValueError(f"Expected an interval dtype, got {dtype}")
    if isinstance(dtype, pd.ArrowDtype):
        return pd.ArrowDtype(dtype.pyarrow_dtype.subtype)
    else:
        return cast("cudf.IntervalDtype", dtype).subtype
