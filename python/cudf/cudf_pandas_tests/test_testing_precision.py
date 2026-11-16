# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="New pandas 3.1 tolerance precision"
)
@pytest.mark.parametrize("kind", ["Series", "Index", "DataFrame"])
@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64"])
@pytest.mark.parametrize("atol", [0.5, 1.0])
def test_explicit_tolerance_preserves_integer_precision(kind, dtype, atol):
    left = getattr(xpd, kind)([2**60 + 1], dtype=dtype)
    right = getattr(xpd, kind)([float(2**60)], dtype="float64")
    name = "frame" if kind == "DataFrame" else kind.lower()
    compare = getattr(xpd.testing, f"assert_{name}_equal")
    kwargs = {"exact": False} if kind == "Index" else {"check_dtype": False}
    if atol < 1:
        with pytest.raises(AssertionError):
            compare(
                left, right, check_exact=False, rtol=0, atol=atol, **kwargs
            )
    else:
        compare(left, right, check_exact=False, rtol=0, atol=atol, **kwargs)


@pytest.mark.parametrize("kind", ["Series", "Index", "DataFrame"])
def test_float_tolerance_keeps_gpu_execution(monkeypatch, kind):
    left = getattr(xpd, kind)([1.0, 2.0])
    right = getattr(xpd, kind)([1.0, 2.0 + 1e-8])
    name = "frame" if kind == "DataFrame" else kind.lower()
    compare = getattr(xpd.testing, f"assert_{name}_equal")
    with monkeypatch.context() as context:
        context.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "1")
        compare(left, right, check_exact=False, rtol=1e-6, atol=0)
