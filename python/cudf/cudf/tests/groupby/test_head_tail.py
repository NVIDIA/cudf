# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import operator

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(params=[-3, -1, 0, 1, 2])
def n(request):
    return request.param


@pytest.fixture(
    params=[False, True], ids=["no-preserve-order", "preserve-order"]
)
def preserve_order(request):
    return request.param


@pytest.fixture(scope="module")
def df():
    # The test matrix only reads this source dataframe.
    return cudf.DataFrame(
        {
            "a": [1, 0, 1, 2, 2, 1, 3, 2, 3, 3, 3],
            "b": [0, 1, 2, 4, 3, 5, 6, 7, 9, 8, 10],
        }
    )


@pytest.fixture(params=[True, False], ids=["head", "tail"])
def take_head(request):
    return request.param


@pytest.fixture
def expected(df, n, take_head, preserve_order):
    if n == 0:
        # We'll get an empty dataframe in this case
        return df._empty_like(keep_index=True)
    else:
        if preserve_order:
            # Should match pandas here
            g = df.to_pandas().groupby("a")
            if take_head:
                return g.head(n=n)
            else:
                return g.tail(n=n)
        else:
            # We groupby "a" which is the first column. This
            # possibly relies on an implementation detail that for
            # integer group keys, cudf produces groups in sorted
            # (ascending) order.
            keyfunc = operator.itemgetter(0)
            if take_head or n == 0:
                # Head does group[:n] as does tail for n == 0
                slicefunc = operator.itemgetter(slice(None, n))
            else:
                # Tail does group[-n:] except when n == 0
                slicefunc = operator.itemgetter(
                    slice(-n, None) if n else slice(0)
                )
            values_to_sort = np.hstack(
                [df.to_numpy(), np.arange(len(df)).reshape(-1, 1)]
            )
            expect_a, expect_b, index = zip(
                *itertools.chain.from_iterable(
                    slicefunc(list(group))
                    for _, group in itertools.groupby(
                        sorted(values_to_sort.tolist(), key=keyfunc),
                        key=keyfunc,
                    )
                ),
                strict=True,
            )
            return cudf.DataFrame({"a": expect_a, "b": expect_b}, index=index)


def test_head_tail(df, n, take_head, expected, preserve_order):
    if take_head:
        actual = df.groupby("a").head(n=n, preserve_order=preserve_order)
    else:
        actual = df.groupby("a").tail(n=n, preserve_order=preserve_order)
    assert_eq(actual, expected)


def test_head_tail_empty():
    # GH #13397

    values = [1, 2, 3]
    pdf = pd.DataFrame({}, index=values)
    df = cudf.DataFrame({}, index=values)

    expected = pdf.groupby(pd.Series(values)).head()
    got = df.groupby(cudf.Series(values)).head()
    assert_eq(expected, got, check_column_type=False)

    expected = pdf.groupby(pd.Series(values)).tail()
    got = df.groupby(cudf.Series(values)).tail()

    assert_eq(expected, got, check_column_type=False)
