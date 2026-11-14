# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cudf.pandas.fast_slow_proxy import _Unusable
from cudf.pandas.module_accelerator import ModuleAccelerator


def test_module_constant_without_hashing(monkeypatch):
    dtype = pd.SparseDtype("float64")

    def unexpected_hash(self):
        raise AssertionError("Module constants must not be hashed")

    monkeypatch.setattr(type(dtype), "__hash__", unexpected_hash)
    loader = ModuleAccelerator._instance
    assert loader._wrap_attribute(dtype, _Unusable(), "dtype") is dtype
    assert (
        loader.getattr_real_or_wrapped(
            "dtype", real={"dtype": dtype}, wrapped_objs={}, loader=loader
        )
        is dtype
    )


@pytest.mark.parametrize("value", [np.add, np.where, object()])
def test_registered_module_callable_or_constant(value, monkeypatch):
    loader = ModuleAccelerator._instance
    proxy = object()
    wrapped_objs = {value: proxy}
    monkeypatch.setattr(loader, "_wrapped_objs", wrapped_objs)
    assert loader._wrap_attribute(value, _Unusable(), "value") is proxy
    assert (
        loader.getattr_real_or_wrapped(
            "value",
            real={"value": value},
            wrapped_objs=wrapped_objs,
            loader=loader,
        )
        is proxy
    )
