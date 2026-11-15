# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in Series\\.drop_duplicates is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize("keep", ["first", "last", False])
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4, 5, 6, 6],
        [],
        ["a", "b", "s", "sd", "a", "b"],
        pd.Series(["aaa"] * 10),
    ],
)
def test_drop_duplicates_series(data, keep, ignore_index):
    pds = pd.Series(data)
    gds = cudf.from_pandas(pds)
    if isinstance(data, list) and len(data) == 0:
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        pds = pds.astype(gds.dtype)

    assert_eq(
        pds.drop_duplicates(keep=keep, ignore_index=ignore_index),
        gds.drop_duplicates(keep=keep, ignore_index=ignore_index),
    )

    expected = pds.drop_duplicates(keep=keep, ignore_index=ignore_index)
    gds.drop_duplicates(keep=keep, inplace=True, ignore_index=ignore_index)
    assert_eq(expected, gds)
