# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 datetime bounds checks"
)
@pytest.mark.parametrize(
    "values,kwargs",
    [
        (np.array([float(2**63)]), {"unit": "ns"}),
        (np.array([np.iinfo(np.uint64).max], dtype="uint64"), {"unit": "ns"}),
        (["1677-09-21 00:12:43.145224192"], {}),
        (
            ["21/09/1677 00:12:43.145224192"],
            {"format": "%d/%m/%Y %H:%M:%S.%f"},
        ),
    ],
)
@pytest.mark.parametrize("container", [list, cudf.Series, cudf.Index])
def test_to_datetime_rejects_reserved_or_overflowed_storage(
    values, kwargs, container
):
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        pd.to_datetime(values, **kwargs)
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        cudf.to_datetime(container(values), **kwargs)


@pytest.mark.parametrize(
    "values,kwargs",
    [
        (np.array([np.nextafter(float(2**63), -np.inf)]), {"unit": "ns"}),
        (np.array([np.iinfo(np.int64).max], dtype="uint64"), {"unit": "ns"}),
        (["1677-09-21 00:12:43.145224193"], {}),
        (
            ["21/09/1677 00:12:43.145224193"],
            {"format": "%d/%m/%Y %H:%M:%S.%f"},
        ),
    ],
)
def test_to_datetime_keeps_boundary_neighbors(values, kwargs):
    expected = pd.to_datetime(values, **kwargs)
    actual = cudf.to_datetime(values, **kwargs)
    assert_eq(expected, actual)


def test_to_datetime_keeps_literal_nat_and_nulls():
    values = ["1970-01-01 00:00:00.000000000", "NaT", None]
    kwargs = {"format": "%Y-%m-%d %H:%M:%S.%f"}
    assert_eq(
        pd.to_datetime(values, **kwargs), cudf.to_datetime(values, **kwargs)
    )
