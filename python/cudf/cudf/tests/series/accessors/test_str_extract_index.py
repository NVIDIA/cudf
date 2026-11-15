# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("klass", ["Series", "Index"])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize(
    "pattern", [r"([AB])", r"(?P<letter>[AB])", r"([AB])([123])"]
)
@pytest.mark.parametrize("dtype", ["str", "string", object])
def test_extract_index_requires_single_group_without_expansion(
    klass, expand, pattern, dtype
):
    values = ["A1", "B2", "A3"]
    expected = getattr(pd, klass)(values, dtype=dtype)
    actual = getattr(cudf, klass)(values, dtype=dtype)
    if klass == "Index" and not expand and pattern == r"([AB])([123])":
        message = "only one regex group is supported with Index"
        with pytest.raises(ValueError, match=message):
            expected.str.extract(pattern, expand=expand)
        with pytest.raises(ValueError, match=message):
            actual.str.extract(pattern, expand=expand)
    else:
        assert_eq(
            expected.str.extract(pattern, expand=expand),
            actual.str.extract(pattern, expand=expand),
        )
