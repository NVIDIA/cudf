# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def shape_checker(expected, got):
    assert expected.shape == got.shape


def exact_checker(expected, got):
    assert_eq(expected, got)


@pytest.fixture(
    params=itertools.product([0, 2, None], [0.3, None]),
    ids=lambda arg: f"n={arg[0]}-frac={arg[1]}",
)
def sample_n_frac(request):
    """
    Specific to `test_sample*` tests.
    """
    n, frac = request.param
    if n is not None and frac is not None:
        pytest.skip("Cannot specify both n and frac.")
    return n, frac


@pytest.fixture(params=[None, "builtin_list", "ndarray"])
def make_weights_axis_0(request):
    """Specific to `test_sample*_axis_0` tests.
    Only testing weights array that matches type with random state.
    """

    if request.param is None:
        return lambda *_: (None, None)
    elif request.param == "builtin-list":
        return lambda size, _: ([1] * size, [1] * size)
    else:

        def wrapped(size, numpy_weights_for_cudf):
            # Uniform distribution, non-normalized
            if numpy_weights_for_cudf:
                return np.ones(size), np.ones(size)
            else:
                return np.ones(size), cp.ones(size)

        return wrapped


@pytest.mark.parametrize(
    "make_weights_axis_1",
    [lambda _: None, lambda s: [1] * s, lambda s: np.ones(s)],
)
@pytest.mark.parametrize(
    "pd_random_state, gd_random_state, checker",
    [
        (None, None, shape_checker),
        (42, 42, shape_checker),
        (np.random.RandomState(42), np.random.RandomState(42), exact_checker),
    ],
    ids=["None", "IntSeed", "NumpyRandomState"],
)
def test_sample_axis_1(
    sample_n_frac,
    pd_random_state,
    gd_random_state,
    checker,
    make_weights_axis_1,
):
    n, frac = sample_n_frac

    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
    )
    df = cudf.DataFrame(pdf)

    weights = make_weights_axis_1(len(pdf.columns))

    expected = pdf.sample(
        n=n,
        frac=frac,
        replace=False,
        random_state=pd_random_state,
        weights=weights,
        axis=1,
    )
    got = df.sample(
        n=n,
        frac=frac,
        replace=False,
        random_state=gd_random_state,
        weights=weights,
        axis=1,
    )
    checker(expected, got)


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "float": [0.05, 0.2, 0.3, 0.2, 0.25],
                "int": [1, 3, 5, 4, 2],
            },
        ),
        pd.Series([1, 2, 3, 4, 5]),
    ],
)
@pytest.mark.parametrize(
    "pd_random_state, gd_random_state, checker",
    [
        (None, None, shape_checker),
        (42, 42, shape_checker),
        (np.random.RandomState(42), np.random.RandomState(42), exact_checker),
        (np.random.RandomState(42), cp.random.RandomState(42), shape_checker),
    ],
    ids=["None", "IntSeed", "NumpyRandomState", "CupyRandomState"],
)
@pytest.mark.parametrize("replace", [True, False])
def test_sample_axis_0(
    pdf,
    sample_n_frac,
    replace,
    pd_random_state,
    gd_random_state,
    checker,
    make_weights_axis_0,
):
    n, frac = sample_n_frac

    df = cudf.from_pandas(pdf)

    pd_weights, gd_weights = make_weights_axis_0(
        len(pdf), isinstance(gd_random_state, np.random.RandomState)
    )
    if (
        not replace
        and not isinstance(gd_random_state, np.random.RandomState)
        and gd_weights is not None
    ):
        pytest.skip(
            "`cupy.random.RandomState` doesn't support weighted sampling "
            "without replacement."
        )

    expected = pdf.sample(
        n=n,
        frac=frac,
        replace=replace,
        random_state=pd_random_state,
        weights=pd_weights,
        axis=0,
    )

    got = df.sample(
        n=n,
        frac=frac,
        replace=replace,
        random_state=gd_random_state,
        weights=gd_weights,
        axis=0,
    )
    checker(expected, got)


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "random_state_lib", [cp.random.RandomState, np.random.RandomState]
)
def test_sample_reproducibility(replace, random_state_lib):
    df = cudf.DataFrame({"a": cp.arange(0, 25)})

    n = 25
    expected = df.sample(n, replace=replace, random_state=random_state_lib(10))
    out = df.sample(n, replace=replace, random_state=random_state_lib(10))

    assert_eq(expected, out)


def test_sample_invalid_n_frac_combo(axis):
    n, frac = 2, 0.5
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
    )
    df = cudf.DataFrame(pdf)

    assert_exceptions_equal(
        lfunc=pdf.sample,
        rfunc=df.sample,
        lfunc_args_and_kwargs=([], {"n": n, "frac": frac, "axis": axis}),
        rfunc_args_and_kwargs=([], {"n": n, "frac": frac, "axis": axis}),
    )


@pytest.mark.parametrize("n, frac", [(100, None), (None, 3)])
def test_oversample_without_replace(n, frac, axis):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    df = cudf.DataFrame(pdf)

    assert_exceptions_equal(
        lfunc=pdf.sample,
        rfunc=df.sample,
        lfunc_args_and_kwargs=(
            [],
            {"n": n, "frac": frac, "axis": axis, "replace": False},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"n": n, "frac": frac, "axis": axis, "replace": False},
        ),
    )


@pytest.mark.parametrize("random_state", [None, cp.random.RandomState(42)])
def test_sample_unsupported_arguments(random_state):
    df = cudf.DataFrame({"float": [0.05, 0.2, 0.3, 0.2, 0.25]})
    with pytest.raises(
        NotImplementedError,
        match=r"Random sampling with cupy does not support these inputs.",
    ):
        df.sample(
            n=2, replace=False, random_state=random_state, weights=[1] * 5
        )
