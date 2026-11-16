# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from collections import deque

import pandas as xpd
import pandas._testing as tm
import pytest

pd = xpd._fsproxy_slow


def test_str_cat_iterator_flags():
    series = xpd.Series(["a", "b", "c", "d"])
    other = xpd.Series(["d", "a", "e", "b"], index=[3, 0, 4, 1])
    result = series.str.cat(
        iter([other, series.to_numpy()]), join="outer", na_rep="-"
    )
    expected = pd.Series(["aaa", "bbb", "c-c", "ddd", "-e-"])
    tm.assert_series_equal(result, expected)


def test_concat_opaque_iterable_flags():
    frames = deque([xpd.DataFrame({"a": [1]}), xpd.DataFrame({"a": [2]})])
    result = xpd.concat(frames, ignore_index=True)
    assert result.to_dict("list") == {"a": [1, 2]}
    assert result.flags.allows_duplicate_labels


@pytest.mark.parametrize("offset", ["Day", "BusinessDay", "MonthEnd"])
@pytest.mark.parametrize("protocol", [0, 2, pickle.HIGHEST_PROTOCOL])
def test_pickle_offset_restores_pandas_scalar(offset, protocol):
    original = getattr(xpd.offsets, offset)(2)
    result = pickle.loads(pickle.dumps(original, protocol=protocol))

    assert isinstance(result, pd.offsets.BaseOffset)
    assert type(result) is getattr(pd.offsets, offset)
    assert result == getattr(pd.offsets, offset)(2)
