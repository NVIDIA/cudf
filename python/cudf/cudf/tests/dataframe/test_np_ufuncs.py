# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
from functools import reduce

import numpy as np
import pytest
from packaging.version import parse

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    gen_rand_ufunc_input,
    set_random_null_mask_inplace,
)


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
def test_ufunc_dataframe(request, numpy_ufunc, has_nulls, indexed):
    # Note: This test assumes that all ufuncs are unary or binary.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                indexed
                and numpy_ufunc
                in {
                    np.greater,
                    np.greater_equal,
                    np.less,
                    np.less_equal,
                    np.not_equal,
                    np.equal,
                }
            ),
            reason="Comparison operators do not support misaligned indexes.",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc in {np.ceil, np.floor, np.trunc}
            and not has_nulls
            and parse(np.__version__) >= parse("2.0")
            and parse(np.__version__) < parse("2.1"),
            reason="https://github.com/cupy/cupy/issues/9018",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc == np.matmul,
            reason=f"{numpy_ufunc} is not supported in cuDF",
        )
    )

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    # TODO: Add tests of mismatched columns etc.
    rng = np.random.default_rng(0)
    pandas_args = args = [
        cudf.DataFrame(
            {"foo": gen_rand_ufunc_input(numpy_ufunc, rng, N)},
            index=rng.choice(range(N), N, False) if indexed else None,
        )
        for _ in range(numpy_ufunc.nin)
    ]

    if has_nulls:
        # Converting nullable integer cudf.Series to pandas will produce a
        # float pd.Series, so instead we replace nulls with an arbitrary
        # integer value, precompute the mask, and then reapply it afterwards.
        for arg in args:
            set_random_null_mask_inplace(arg["foo"])
        pandas_args = [arg.copy() for arg in args]
        for arg in pandas_args:
            arg["foo"] = arg["foo"].fillna(0)

        # Note: Different indexes must be aligned before the mask is computed.
        # This requires using an internal function (_align_indices), and that
        # is unlikely to change for the foreseeable future.
        aligned = (
            cudf.core.dataframe._align_indices(*args)
            if indexed and numpy_ufunc.nin == 2
            else args
        )
        mask = reduce(
            operator.or_, (a["foo"].isna() for a in aligned)
        ).to_pandas()
        if numpy_ufunc in (np.power, np.float_power):
            # pandas honors 1 ** x == 1 and x ** 0 == 1 even when x is
            # missing, and cudf matches, so those positions are valid in the
            # result. The 0-filled pandas args already compute 1 there
            # (1 ** 0 and 0 ** 0), so just unmask them.
            base, exponent = (a["foo"] for a in aligned)
            identity = ((base == 1).fillna(False) & exponent.isna()) | (
                (exponent == 0).fillna(False) & base.isna()
            )
            mask &= ~identity.to_pandas()

    got = numpy_ufunc(*args)

    expect = numpy_ufunc(*(arg.to_pandas() for arg in pandas_args))

    if numpy_ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
            if has_nulls:
                e[mask] = np.nan
            assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            if numpy_ufunc in (
                np.isfinite,
                np.isinf,
                np.isnan,
                np.logical_and,
                np.logical_not,
                np.logical_or,
                np.logical_xor,
                np.signbit,
                np.equal,
                np.greater,
                np.greater_equal,
                np.less,
                np.less_equal,
                np.not_equal,
            ):
                # cuDF .to_pandas for bools with nulls represents missing as None,
                # should this be np.nan?
                expect = expect.astype(object).mask(mask, None)
            else:
                expect[mask] = np.nan
        assert_eq(got, expect, check_exact=False)
