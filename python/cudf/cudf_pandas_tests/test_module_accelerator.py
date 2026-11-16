# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

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
    wrapped_objs = {id(value): (value, proxy)}
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


def test_registered_constant_uses_identity_without_hashing(monkeypatch):
    class Constant:
        def __hash__(self):
            raise AssertionError("Registered constants must not be hashed")

        def __eq__(self, other):
            raise AssertionError("Registered constants must not be compared")

    value = Constant()
    proxy = object()
    loader = ModuleAccelerator._instance
    wrapped_objs = {id(value): (value, proxy)}
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
    assert (
        loader._wrap_attribute(Constant(), _Unusable(), "other") is not proxy
    )

    reference = weakref.ref(value)
    del value
    gc.collect()
    assert reference() is not None


def test_pandas_na_keeps_singleton_identity():
    assert pd.NA is pd._fsproxy_slow.NA


@pytest.mark.parametrize("kind", ["Series", "DataFrame", "Index"])
def test_registered_accessor_does_not_retain_parent(kind):
    class Accessor:
        def __init__(self, obj):
            self.obj = obj

    cls = getattr(pd, kind)
    name = "_test_no_reference_cycle"
    registrar = getattr(pd.api.extensions, f"register_{kind.lower()}_accessor")
    registrar(name)(Accessor)
    try:
        obj = cls([0])
        reference = weakref.ref(obj)
        assert getattr(obj, name).obj is obj
        del obj
        assert reference() is None
    finally:
        delattr(cls, name)
        cls._accessors.discard(name)


def test_public_module_aliases_and_numpy_ufunc_dispatch():
    assert pd.Series is pd.core.series.Series
    assert pd.Index is pd.core.indexes.base.Index
    result = np.add(pd.Series([1, 2, 3]), 1)
    pd._fsproxy_slow.testing.assert_series_equal(
        result._fsproxy_slow, pd._fsproxy_slow.Series([2, 3, 4])
    )
