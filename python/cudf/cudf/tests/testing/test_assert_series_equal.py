# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_series_equal
from cudf.testing._utils import assert_asserters_equal


def test_series_different_type_cases(
    numeric_types_as_str, check_exact, check_dtype
):
    data = [0, 1, 2, 3]

    psr1 = pd.Series(data, dtype="uint8")
    psr2 = pd.Series(data, dtype=numeric_types_as_str)

    sr1 = cudf.from_pandas(psr1)
    sr2 = cudf.from_pandas(psr2)

    assert_asserters_equal(
        pd.testing.assert_series_equal,
        assert_series_equal,
        psr1,
        psr2,
        sr1,
        sr2,
        check_exact=check_exact,
        check_dtype=check_dtype,
    )


@pytest.mark.parametrize("rdata", [3, 4], ids=["same", "different"])
def test_datetime_like_compaibility(rdata, check_datetimelike_compat):
    psr1 = pd.Series([0, 1, 2, 3], dtype="datetime64[ns]")
    psr2 = pd.Series([0, 1, 2, rdata], dtype="datetime64[ns]").astype("str")

    sr1 = cudf.from_pandas(psr1)
    sr2 = cudf.from_pandas(psr2)
    with pytest.warns(pd.errors.Pandas4Warning):
        assert_asserters_equal(
            pd.testing.assert_series_equal,
            assert_series_equal,
            psr1,
            psr2,
            sr1,
            sr2,
            check_datetimelike_compat=check_datetimelike_compat,
        )


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("rname", ["a", "b"])
@pytest.mark.parametrize("check_category_order", [True, False])
@pytest.mark.parametrize("check_categorical", [True, False])
def test_basic_assert_series_equal(
    rdata,
    rname,
    check_names,
    check_category_order,
    check_categorical,
    all_supported_types_as_str,
):
    p_left = pd.Series([1, 2, 3], name="a", dtype=all_supported_types_as_str)
    p_right = pd.Series(rdata, name=rname, dtype=all_supported_types_as_str)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    assert_asserters_equal(
        pd.testing.assert_series_equal,
        assert_series_equal,
        p_left,
        p_right,
        left,
        right,
        check_names=check_names,
        check_categorical=check_categorical,
        check_category_order=check_category_order,
    )
