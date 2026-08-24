# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numba.np import numpy_support

import pylibcudf as plc

from cudf.api.types import is_scalar
from cudf.core.udf.utils import compile_udf
from cudf.utils.dtypes import SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

_agg_name_map = {
    "COUNT_VALID": "COUNT",
    "COUNT_ALL": "SIZE",
    "VARIANCE": "VAR",
    "NTH_ELEMENT": "NTH",
    "COLLECT_LIST": "COLLECT",
    "COLLECT_SET": "UNIQUE",
}


class Aggregation:
    def __init__(self, agg: plc.aggregation.Aggregation) -> None:
        self.plc_obj = agg

    @property
    def kind(self) -> str:
        name = self.plc_obj.kind().name
        return _agg_name_map.get(name, name)

    @classmethod
    def sum(cls) -> Self:
        return cls(plc.aggregation.sum())

    @classmethod
    def min(cls) -> Self:
        return cls(plc.aggregation.min())

    @classmethod
    def max(cls) -> Self:
        return cls(plc.aggregation.max())

    @classmethod
    def idxmin(cls) -> Self:
        return cls(plc.aggregation.argmin())

    @classmethod
    def idxmax(cls) -> Self:
        return cls(plc.aggregation.argmax())

    @classmethod
    def mean(cls) -> Self:
        return cls(plc.aggregation.mean())

    @classmethod
    def count(cls, dropna: bool = True) -> Self:
        return cls(
            plc.aggregation.count(
                plc.types.NullPolicy.EXCLUDE
                if dropna
                else plc.types.NullPolicy.INCLUDE
            )
        )

    @classmethod
    def ewma(cls, com: float = 1.0, adjust: bool = True) -> Self:
        return cls(
            plc.aggregation.ewma(
                com,
                plc.aggregation.EWMHistory.INFINITE
                if adjust
                else plc.aggregation.EWMHistory.FINITE,
            )
        )

    @classmethod
    def size(cls) -> Self:
        return cls(plc.aggregation.count(plc.types.NullPolicy.INCLUDE))

    @classmethod
    def collect(cls) -> Self:
        return cls(plc.aggregation.collect_list(plc.types.NullPolicy.INCLUDE))

    @classmethod
    def nunique(cls, dropna: bool = True) -> Self:
        return cls(
            plc.aggregation.nunique(
                plc.types.NullPolicy.EXCLUDE
                if dropna
                else plc.types.NullPolicy.INCLUDE
            )
        )

    @classmethod
    def nth(cls, size: int) -> Self:
        return cls(plc.aggregation.nth_element(size))

    @classmethod
    def product(cls) -> Self:
        return cls(plc.aggregation.product())

    prod = product

    @classmethod
    def sum_of_squares(cls) -> Self:
        return cls(plc.aggregation.sum_of_squares())

    @classmethod
    def var(cls, ddof: int = 1) -> Self:
        return cls(plc.aggregation.variance(ddof))

    @classmethod
    def std(cls, ddof: int = 1) -> Self:
        return cls(plc.aggregation.std(ddof))

    @classmethod
    def median(cls) -> Self:
        return cls(plc.aggregation.median())

    @classmethod
    def quantile(
        cls,
        q: float | list[float] = 0.5,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "linear",
    ) -> Self:
        q_list: list[float] = [q] if is_scalar(q) else q  # type: ignore[assignment,list-item]
        return cls(
            plc.aggregation.quantile(
                q_list,
                plc.types.Interpolation[interpolation.upper()],
            )
        )

    @classmethod
    def unique(cls) -> Self:
        return cls(
            plc.aggregation.collect_set(
                plc.types.NullPolicy.INCLUDE,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
            )
        )

    @classmethod
    def first(cls, skipna: bool = True) -> Self:
        return cls(
            plc.aggregation.nth_element(
                0,
                plc.types.NullPolicy.EXCLUDE
                if skipna
                else plc.types.NullPolicy.INCLUDE,
            )
        )

    @classmethod
    def last(cls, skipna: bool = True) -> Self:
        return cls(
            plc.aggregation.nth_element(
                -1,
                plc.types.NullPolicy.EXCLUDE
                if skipna
                else plc.types.NullPolicy.INCLUDE,
            )
        )

    @classmethod
    def corr(cls, method, min_periods) -> Self:
        return cls(
            plc.aggregation.correlation(
                plc.aggregation.CorrelationType[method.upper()], min_periods
            )
        )

    @classmethod
    def cov(cls, min_periods: int, ddof: int = 1) -> Self:
        return cls(plc.aggregation.covariance(min_periods, ddof))

    # scan aggregations
    @classmethod
    def cumcount(cls) -> Self:
        return cls.count(False)

    cumsum = sum
    cummin = min
    cummax = max
    cumprod = product

    @classmethod
    def rank(
        cls,
        method: Literal["first", "average", "min", "max", "dense"],
        ascending: bool,
        na_option: Literal["keep", "top", "bottom"],
        pct: bool,
    ) -> Self:
        return cls(
            plc.aggregation.rank(
                plc.aggregation.RankMethod[method.upper()],
                (
                    plc.types.Order.ASCENDING
                    if ascending
                    else plc.types.Order.DESCENDING
                ),
                (
                    plc.types.NullPolicy.EXCLUDE
                    if na_option == "keep"
                    else plc.types.NullPolicy.INCLUDE
                ),
                (
                    plc.types.NullOrder.BEFORE
                    if (na_option == "top") == ascending
                    else plc.types.NullOrder.AFTER
                ),
                (
                    plc.aggregation.RankPercentage.ZERO_NORMALIZED
                    if pct
                    else plc.aggregation.RankPercentage.NONE
                ),
            )
        )

    # Reduce aggregations
    @classmethod
    def any(cls) -> Self:
        return cls(plc.aggregation.any())

    @classmethod
    def all(cls) -> Self:
        return cls(plc.aggregation.all())

    # Rolling aggregations
    @classmethod
    def from_udf(cls, op, *args, **kwargs) -> Self:
        # Handling UDF type
        nb_type = numpy_support.from_dtype(kwargs["dtype"])
        type_signature = (nb_type[:],)
        ptx_code, output_dtype = compile_udf(op, type_signature)
        output_np_dtype = np.dtype(output_dtype)
        if output_np_dtype not in SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES:
            raise TypeError(
                f"Result of window function has unsupported dtype {op[1]}"
            )

        return cls(
            plc.aggregation.udf(
                ptx_code,
                plc.DataType(
                    SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES[output_np_dtype]
                ),
            )
        )


def make_aggregation(
    op: str | Callable, kwargs: dict | None = None
) -> Aggregation:
    r"""
    Parameters
    ----------
    op : str or callable
        If callable, must meet one of the following requirements:

        * Is of the form lambda x: x.agg(*args, **kwargs), where
          `agg` is the name of a supported aggregation. Used to
          to specify aggregations that take arguments, e.g.,
          `lambda x: x.quantile(0.5)`.
        * Is a user defined aggregation function that operates on
          group values. In this case, the output dtype must be
          specified in the `kwargs` dictionary.
    \*\*kwargs : dict, optional
        Any keyword arguments to be passed to the op.

    Returns
    -------
    Aggregation
    """
    if kwargs is None:
        kwargs = {}

    if isinstance(op, str):
        return getattr(Aggregation, op)(**kwargs)
    elif callable(op):
        if op is list:
            return Aggregation.collect()
        elif "dtype" in kwargs:
            return Aggregation.from_udf(op, **kwargs)
        else:
            return op(Aggregation)
    raise TypeError(f"Unknown aggregation {op}")
