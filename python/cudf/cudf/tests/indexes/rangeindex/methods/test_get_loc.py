# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("key", list(range(1, 110, 13)))
def test_get_loc_rangeindex(key):
    pi = pd.RangeIndex(3, 100, 4)
    gi = cudf.from_pandas(pi)
    if (
        (key not in pi)
        # Get key before the first element is KeyError
        or (key < pi.start)
        # Get key after the last element is KeyError
        or (key >= pi.stop)
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "labels, key",
    [
        (range(3, -4, -2), np.uint64(1)),
        (range(0, 2**54), 2**53 + 1),
        (range(-(2**63), 2**63), np.uint64(0)),
    ],
)
def test_get_loc_exact_integer_position(labels, key):
    assert cudf.RangeIndex(labels).get_loc(key) == pd.RangeIndex(
        labels
    ).get_loc(key)


@pytest.mark.parametrize("labels", [range(0), range(4), range(4, 0, -1)])
def test_get_loc_excludes_stop(labels):
    with pytest.raises(KeyError):
        cudf.RangeIndex(labels).get_loc(labels.stop)
