# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf


@pytest.fixture
def mi_pair():
    arrays = [[1, 1, 2, 2], ["a", "b", "a", "b"], [10, 20, 30, 40]]
    pmi = pd.MultiIndex.from_arrays(arrays, names=["x", "y", "z"])
    return cudf.from_pandas(pmi), pmi


@pytest.mark.parametrize(
    "key",
    [
        1,  # scalar leading-level label
        5,  # scalar absent label
        (1,),  # partial 1-tuple present
        (5,),  # partial 1-tuple absent
        (1, "a"),  # partial 2-tuple present (leading levels match)
        (1, "z"),  # partial 2-tuple absent
        (1, "a", 10),  # full tuple present
        (1, "a", 99),  # full tuple absent
        (2, "b", 40),  # another full tuple present
        (1, "a", 10, "extra"),  # too-long tuple -> False (not error)
    ],
)
def test_multiindex_contains_matches_pandas(mi_pair, key):
    gmi, pmi = mi_pair
    assert (key in gmi) == (key in pmi)


def test_multiindex_contains_unhashable_raises(mi_pair):
    gmi, _ = mi_pair
    with pytest.raises(TypeError):
        [1, "a"] in gmi
