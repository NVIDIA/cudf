# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_index_astype(all_supported_types_as_str, copy):
    pdi = pd.Index([1, 2, 3])
    gdi = cudf.from_pandas(pdi)

    actual = gdi.astype(dtype=all_supported_types_as_str, copy=copy)
    expected = pdi.astype(dtype=all_supported_types_as_str, copy=copy)

    assert_eq(expected, actual)
    assert_eq(pdi, gdi)


@pytest.mark.parametrize("copy", [True, False])
def test_index_astype_no_copy(copy):
    gidx = cudf.Index([1, 2, 3], dtype="int64")
    result = gidx.astype("int64", copy=copy)
    assert_eq(result, gidx)
    # Under CoW, copy=False returns a shallow copy (distinct object).
    assert result is not gidx
