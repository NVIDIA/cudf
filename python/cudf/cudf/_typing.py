# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.api.typing import NoDefault  # noqa: F401

import pylibcudf as plc

if TYPE_CHECKING:
    from pandas.api.extensions import ExtensionDtype

    import cudf

# Many of these are from
# https://github.com/pandas-dev/pandas/blob/master/pandas/_typing.py

# Dtype should ideally only used for public facing APIs
Dtype = Union["ExtensionDtype", str, np.dtype]
# DtypeObj should be used otherwise
DtypeObj = Union["ExtensionDtype", np.dtype]


class DtypePolicy(Protocol):
    def __call__(
        self, result: plc.Column, dtypes: list[DtypeObj]
    ) -> DtypeObj: ...


# scalars
DatetimeLikeScalar = TypeVar(
    "DatetimeLikeScalar", pd.Period, pd.Timestamp, pd.Timedelta
)
ScalarLike = Any

# columns
ColumnLike = Any

# binary operation
ColumnBinaryOperand = Union[ScalarLike, "cudf.core.column.ColumnBase"]

DataFrameOrSeries = Union["cudf.Series", "cudf.DataFrame"]
SeriesOrIndex = Union["cudf.Series", "cudf.Index"]
SeriesOrSingleColumnIndex = Union["cudf.Series", "cudf.Index"]

# Groupby aggregation
AggType = Union[str, Callable]  # noqa: UP007
MultiColumnAggType = Union[  # noqa: UP007
    AggType, Iterable[AggType], dict[Any, Iterable[AggType]]
]

Axis = Literal[0, 1, "index", "columns"]
