# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pandas as pd
import pytest

from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize(
    "name", ["MonthEnd", "YearEnd", "BusinessDay", "Day", "DateOffset"]
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
@pytest.mark.parametrize("reflect", [False, True])
def test_numpy_nat_offset(name, op, reflect):
    nat = np.datetime64("NaT", "ns")

    def apply(offset):
        return op(nat, offset) if reflect else op(offset, nat)

    try:
        with disable_module_accelerator():
            expected = apply(getattr(pd.offsets, name)())
    except (TypeError, ValueError, AttributeError) as error:
        with pytest.raises(type(error)):
            apply(getattr(pd.offsets, name)())
    else:
        assert expected is pd.NaT
        assert apply(getattr(pd.offsets, name)()) is pd.NaT
