# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import cupy as cp
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import gen_rand_ufunc_input


def test_ufunc_index(request, numpy_ufunc):
    # Note: This test assumes that all ufuncs are unary or binary.
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc in {np.ceil, np.floor, np.trunc}
            and parse(np.__version__) >= parse("2.0")
            and parse(np.__version__) < parse("2.1"),
            reason="https://github.com/cupy/cupy/issues/9018",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc == np.matmul,
            reason="cuDF doesn't support matmul for Indexes yet",
        )
    )

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    rng = np.random.default_rng(0)
    pandas_args = args = [
        cudf.Index(
            gen_rand_ufunc_input(numpy_ufunc, rng, N),
        )
        for _ in range(numpy_ufunc.nin)
    ]

    got = numpy_ufunc(*args)

    expect = numpy_ufunc(*(arg.to_pandas() for arg in pandas_args))

    if numpy_ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
            assert_eq(g, e, check_exact=False)
    else:
        assert_eq(got, expect, check_exact=False)


@pytest.mark.parametrize(
    "ufunc", [np.add, np.logaddexp, np.greater, np.logical_and]
)
@pytest.mark.parametrize("reflect", [True, False])
def test_binary_ufunc_index_series_returns_series(ufunc, reflect):
    # ``np.ufunc(Index, Series)`` should defer to ``Series`` (which has
    # higher priority than ``Index``) and return a ``Series``, matching
    # pandas. Previously cudf's ``Index.__array_ufunc__`` handled the call
    # itself and returned an ``Index``.
    rng = np.random.default_rng(0)
    a = rng.integers(low=1, high=10, size=20)
    b = rng.integers(low=1, high=10, size=20)

    gidx = cudf.Index(a, name="name")
    gser = cudf.Series(b, name="name")
    pidx = pd.Index(a, name="name")
    pser = pd.Series(b, name="name")

    if reflect:
        got = ufunc(gser, gidx)
        expect = ufunc(pser, pidx)
    else:
        got = ufunc(gidx, gser)
        expect = ufunc(pidx, pser)

    assert type(got).__name__ == type(expect).__name__
    assert_eq(got, expect, check_exact=False)


@pytest.mark.parametrize(
    "ufunc", [np.add, np.greater, np.greater_equal, np.logical_and]
)
@pytest.mark.parametrize("reflect", [True, False])
def test_binary_ufunc_index_array(ufunc, reflect):
    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    rng = np.random.default_rng(0)
    args = [cudf.Index(rng.random(N)) for _ in range(ufunc.nin)]

    arg1 = args[1].to_cupy()

    if reflect:
        got = ufunc(arg1, args[0])
        expect = ufunc(args[1].to_numpy(), args[0].to_pandas())
    else:
        got = ufunc(args[0], arg1)
        expect = ufunc(args[0].to_pandas(), args[1].to_numpy())

    if ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
            if reflect:
                assert (cp.asnumpy(g) == e).all()
            else:
                assert_eq(g, e, check_exact=False)
    else:
        if reflect:
            assert (cp.asnumpy(got) == expect).all()
        else:
            assert_eq(got, expect, check_exact=False)
