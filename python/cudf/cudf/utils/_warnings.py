# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cudf.api.types import infer_dtype
from cudf.core._compat import PANDAS_GE_31
from cudf.options import get_option

if TYPE_CHECKING:
    from types import FrameType


def _warn_pandas_31_deprecation(message: str) -> None:
    if PANDAS_GE_31:
        frame: FrameType | None = sys._getframe(1)
        stacklevel = 2
        while (
            frame is not None
            and frame.f_globals.get("__name__", "").startswith("cudf.")
            and not frame.f_globals.get("__name__", "").startswith(
                "cudf.tests."
            )
        ):
            stacklevel += 1
            frame = frame.f_back
        warnings.warn(message, pd.errors.Pandas4Warning, stacklevel=stacklevel)


def _warn_pandas_31_date_indexer(indexer, *, infer: bool) -> None:
    if (
        PANDAS_GE_31
        and isinstance(indexer, (list, tuple, np.ndarray))
        and infer_dtype(indexer, skipna=True) == "date"
    ):
        if get_option("mode.pandas_compatible"):
            raise NotImplementedError(
                "Deprecated Python-date indexing requires pandas coercion semantics"
            )
        _warn_pandas_31_deprecation(
            "Inferring datetime64 from data containing datetime.date objects "
            "is deprecated. In a future version, these will be left as object "
            "dtype. Use pd.to_datetime to explicitly convert to datetime64."
            if infer
            else "Indexing a DatetimeIndex with a sequence of datetime.date "
            "objects is deprecated. Use Timestamp objects instead."
        )


def _warn_pandas_31_numeric_only(numeric_only, op: str) -> None:
    if (
        PANDAS_GE_31
        and not isinstance(numeric_only, (bool, np.bool_))
        and op
        in {
            "mean",
            "min",
            "max",
            "median",
            "skew",
            "kurt",
            "kurtosis",
            "std",
            "var",
            "sem",
            "sum",
            "prod",
            "product",
        }
    ):
        _warn_pandas_31_deprecation(
            "Passing non-boolean values for 'numeric_only' is deprecated and "
            "will raise in a future version of pandas."
        )
