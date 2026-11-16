# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as xpd
import pytest

from cudf.pandas import LOADED
from cudf.pandas.module_accelerator import disable_module_accelerator
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("klass", ["Series", "Index"])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize(
    "pattern", [r"([AB])", r"(?P<letter>[AB])", r"([AB])([123])"]
)
def test_extract_index_capture_group_validation(klass, expand, pattern):
    values = ["A1", "B2", "A3"]
    with disable_module_accelerator():
        expected = getattr(pd, klass)(values)
    actual = getattr(xpd, klass)(values)
    if klass == "Index" and not expand and pattern == r"([AB])([123])":
        message = "only one regex group is supported with Index"
        with pytest.raises(ValueError, match=message):
            actual.str.extract(pattern, expand=expand)
    else:
        with disable_module_accelerator():
            expected_result = expected.str.extract(pattern, expand=expand)
        assert_eq(
            expected_result,
            actual.str.extract(pattern, expand=expand)._fsproxy_slow,
        )
