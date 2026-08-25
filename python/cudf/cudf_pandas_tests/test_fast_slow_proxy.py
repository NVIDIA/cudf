# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from functools import cache, partial
from io import StringIO

import numpy as np
import pytest

import cudf.pandas.fast_slow_proxy
from cudf.pandas.fast_slow_proxy import (
    _fast_arg,
    _FastSlowAttribute,
    _FunctionProxy,
    _setattr_fsproxy_no_mirror,
    _slow_arg,
    _transform_arg,
    _Unusable,
    make_final_proxy_type,
    make_intermediate_proxy_type,
)


@pytest.fixture(autouse=True)
def ensure_temp_proxy_maps(monkeypatch):
    # Ensure these tests do not pollute the global, cached final/intermediate type maps
    @cache
    def mock_get_final_type_map():
        return {}

    @cache
    def mock_get_intermediate_type_map():
        return {}

    with monkeypatch.context() as m:
        m.setattr(
            cudf.pandas.fast_slow_proxy,
            "get_final_type_map",
            mock_get_final_type_map,
        )
        m.setattr(
            cudf.pandas.fast_slow_proxy,
            "get_intermediate_type_map",
            mock_get_intermediate_type_map,
        )
        yield
    mock_get_final_type_map.cache_clear()
    mock_get_intermediate_type_map.cache_clear()


@pytest.fixture
def final_proxy():
    class Fast:
        def __init__(self, x):
            self.x = x

        def to_slow(self):
            return Slow(self.x)

        @classmethod
        def from_slow(cls, slow):
            return cls(slow)

        def __eq__(self, other):
            return self.x == other.x

        def method(self):
            return "fast method"

    class Slow:
        def __init__(self, x):
            self.x = x

        def __eq__(self, other):
            return self.x == other.x

        def method(self):
            return "slow method"

    Pxy = make_final_proxy_type(
        "Pxy",
        Fast,
        Slow,
        fast_to_slow=lambda fast: fast.to_slow(),
        slow_to_fast=lambda slow: Fast.from_slow(slow),
    )
    return Fast(1), Slow(1), Pxy(1)


@pytest.fixture
def function_proxy():
    def fast_func():
        """
        Fast doc
        """
        return "fast func"

    def slow_func():
        """
        Slow doc
        """
        return "slow_func"

    return fast_func, slow_func, _FunctionProxy(fast_func, slow_func)


def test_repr_no_fast_object():
    # test that __repr__ falls back to slow object
    # when we don't have a corresponding fast object:
    class Slow:
        def __repr__(self):
            return "slow object"

    Pxy = make_final_proxy_type(
        "Pxy",
        _Unusable,
        Slow,
        fast_to_slow=lambda fast: Slow(),
        slow_to_fast=lambda slow: _Unusable(),
    )
    assert repr(Pxy()) == repr(Slow())


def test_fast_slow_arg_function_basic():
    def func1():
        return 1

    assert _fast_arg(func1)() == _slow_arg(func1)() == 1

    def func2(x, y):
        return x + y

    assert _fast_arg(func2)(1, 2) == _slow_arg(func2)(1, 2) == 3


def test_fast_slow_arg_function_closure(function_proxy, final_proxy):
    fast_x, slow_x, x = function_proxy
    fast_y, slow_y, y = final_proxy

    def func():
        return x, y.method()

    assert _slow_arg(func)() == (slow_x, slow_y.method())
    assert _fast_arg(func)() == (fast_x, fast_y.method())


def test_fast_slow_arg_function_global(
    monkeypatch, function_proxy, final_proxy
):
    fast_x, slow_x, x = function_proxy
    fast_y, slow_y, y = final_proxy

    # temporarily set x, y as globals
    monkeypatch.setitem(globals(), "__x", x)
    monkeypatch.setitem(globals(), "__y", y)

    def func():
        global __x, __y
        return __x, __y.method()

    assert _slow_arg(func)() == (slow_x, slow_y.method())
    assert _fast_arg(func)() == (fast_x, fast_y.method())


def test_fast_slow_arg_function_np():
    # test that _fast_arg() and _slow_arg() return "externally"
    # defined functions like numpy functions as-is:
    assert _slow_arg(np.mean) is np.mean
    assert _slow_arg(np.unique) is np.unique
    assert _fast_arg(np.mean) is np.mean
    assert _fast_arg(np.unique) is np.unique


def test_fast_slow_arg_builtins(function_proxy):
    # test that builtins are accessible in the result of
    # _fast_arg() and _slow_arg()

    _, _, x = function_proxy

    def func():
        x  # nonlocal x ensures _fast_arg() makes a copy
        return len([1])

    assert _slow_arg(func)() == 1
    assert _fast_arg(func)() == 1


def test_function_proxy_decorating_super_method():
    # test that we can use a function proxy as a decorator to a method
    # that invokes super() (GH: #254)
    deco = _FunctionProxy(_Unusable(), lambda func: func)

    class Foo:
        @deco
        def method(self):
            super()


@pytest.mark.xfail(
    reason="Mutually recursive functions are known to be handled incorrectly."
)
def test_fast_slow_arg_recursion(final_proxy):
    fast_x, slow_x, x = final_proxy

    def foo(n):
        if n <= 0:
            return x
        else:
            return bar(n - 1)

    def bar(n):
        return foo(n - 1)

    assert _slow_arg(foo)(0) == slow_x
    assert _slow_arg(bar)(1) == slow_x
    assert _slow_arg(foo)(1) == slow_x
    assert _slow_arg(bar)(2) == slow_x
    assert _fast_arg(foo)(0) == fast_x
    assert _fast_arg(bar)(1) == fast_x
    assert _fast_arg(foo)(1) == fast_x
    assert _fast_arg(bar)(2) == fast_x


def test_fallback_with_stringio():
    def slow(s):
        return s.read()

    def fast(s):
        s.read()
        raise ValueError()

    pxy = _FunctionProxy(fast=fast, slow=slow)
    assert pxy(StringIO("hello")) == "hello"


# "Pure" (resource-free) iterators that ``_transform_arg`` handles specially:
# they force a fallback on the fast path and are transformed lazily into a new
# generator on the slow path.
_PURE_ITERATOR_FACTORIES = [
    lambda items: (item for item in items),
    lambda items: map(lambda item: item, items),
    lambda items: filter(lambda item: True, items),
    lambda items: zip(items, strict=True),
    lambda items: enumerate(items),
]
_PURE_ITERATOR_IDS = ["generator", "map", "filter", "zip", "enumerate"]


@pytest.mark.parametrize(
    "make_iterator", _PURE_ITERATOR_FACTORIES, ids=_PURE_ITERATOR_IDS
)
def test_pure_iterator_forces_fallback_and_stays_iterator(
    make_iterator, final_proxy
):
    # A pure iterator argument forces a fallback to the slow path (the fast
    # path raises before consuming it), and the slow callable must receive a
    # *bare* iterator rather than a materialized list (pandas treats iterators
    # and lists differently, e.g. ``is_nested_list_like``).
    _, _, x = final_proxy
    received = {}

    def fast(it):
        raise AssertionError("fast path must not run for pure iterators")

    def slow(it):
        received["bare_iterator"] = iter(it) is it and not isinstance(
            it, (list, tuple)
        )
        return list(it)

    pxy = _FunctionProxy(fast=fast, slow=slow)
    result = pxy(make_iterator([x, x]))
    assert received["bare_iterator"]
    assert len(result) == 2


@pytest.mark.parametrize(
    "make_iterator",
    _PURE_ITERATOR_FACTORIES[:3],
    ids=_PURE_ITERATOR_IDS[:3],
)
def test_pure_iterator_slow_path_unwraps_elements(make_iterator, final_proxy):
    # The lazily produced generator still unwraps proxy elements to their slow
    # counterparts (the motivation for transforming rather than passing the
    # iterator through untouched, which dropped type info such as the
    # ``category`` dtype in ``MultiIndex.from_product(map(...))``).
    _, slow_x, x = final_proxy
    transformed = _slow_arg(make_iterator([x, x]))
    assert iter(transformed) is transformed
    items = list(transformed)
    assert items == [slow_x, slow_x]
    # Each element is the unwrapped *slow* object, not the proxy.
    assert all(type(item) is type(slow_x) for item in items)


def test_access_class():
    def func():
        pass

    pxy = _FunctionProxy(fast=_Unusable(), slow=func)
    pxy.__class__


def test_class_attribute_error(final_proxy, function_proxy):
    _, _, x = final_proxy
    _, _, y = function_proxy
    # Test that an attribute error is raised when attempting to
    # access undefined class attributes:
    with pytest.raises(AttributeError):
        x.foo

    with pytest.raises(AttributeError):
        y.foo

    with pytest.raises(AttributeError):
        y.__abs__


def test_function_proxy_doc(function_proxy):
    _, slow, pxy = function_proxy
    assert pxy.__doc__ == slow.__doc__


def test_special_methods():
    class Fast:
        def __abs__(self):
            pass

        def __gt__(self):
            pass

    class Slow:
        def __gt__(self):
            pass

    Pxy = make_final_proxy_type(
        "Pxy",
        Fast,
        Slow,
        fast_to_slow=lambda _: Slow(),
        slow_to_fast=lambda _: Fast(),
    )

    # test that special methods defined _only_ on the
    # fast type are not accessible on the proxy:
    assert not hasattr(Pxy, "__abs__")
    assert not hasattr(Pxy(), "__abs__")

    # test that special methods defined on the
    # slow type are accessible on the proxy:
    assert hasattr(Pxy, "__gt__")
    assert hasattr(Pxy(), "__gt__")


@pytest.fixture(scope="module")
def fast_and_intermediate_with_doc():
    class FastIntermediate:
        """The fast intermediate docstring."""

        def method(self):
            """The fast intermediate method docstring."""

    class Fast:
        """The fast docstring."""

        @property
        def prop(self):
            """The fast property docstring."""

        def method(self):
            """The fast method docstring."""

        def intermediate(self):
            """The fast intermediate docstring."""
            return FastIntermediate()

    return Fast, FastIntermediate


@pytest.fixture(scope="module")
def slow_and_intermediate_with_doc():
    class SlowIntermediate:
        """The slow intermediate docstring."""

        def method(self):
            """The slow intermediate method docstring."""

    class Slow:
        """The slow docstring."""

        @property
        def prop(self):
            """The slow property docstring."""

        def method(self):
            """The slow method docstring."""

        def intermediate(self):
            """The slow intermediate docstring."""
            return SlowIntermediate()

    return Slow, SlowIntermediate


def test_doc(fast_and_intermediate_with_doc, slow_and_intermediate_with_doc):
    Fast, FastIntermediate = fast_and_intermediate_with_doc
    Slow, SlowIntermediate = slow_and_intermediate_with_doc

    Pxy = make_final_proxy_type(
        "Pxy",
        Fast,
        Slow,
        fast_to_slow=lambda _: Slow(),
        slow_to_fast=lambda _: Fast(),
    )

    IntermediatePxy = make_intermediate_proxy_type(  # noqa: F841
        "IntermediatePxy",
        FastIntermediate,
        SlowIntermediate,
    )

    assert inspect.getdoc(Pxy) == inspect.getdoc(Slow)
    assert inspect.getdoc(Pxy()) == inspect.getdoc(Slow())
    assert inspect.getdoc(Pxy.prop) == inspect.getdoc(Slow.prop)
    assert inspect.getdoc(Pxy().prop) == inspect.getdoc(Slow().prop)
    assert inspect.getdoc(Pxy.method) == inspect.getdoc(Slow.method)
    assert inspect.getdoc(Pxy().method) == inspect.getdoc(Slow().method)
    assert inspect.getdoc(Pxy().intermediate()) == inspect.getdoc(
        Slow().intermediate()
    )
    assert inspect.getdoc(Pxy().intermediate().method) == inspect.getdoc(
        Slow().intermediate().method
    )


def test_dir(fast_and_intermediate_with_doc, slow_and_intermediate_with_doc):
    Fast, FastIntermediate = fast_and_intermediate_with_doc
    Slow, SlowIntermediate = slow_and_intermediate_with_doc

    Pxy = make_final_proxy_type(
        "Pxy",
        Fast,
        Slow,
        fast_to_slow=lambda _: Slow(),
        slow_to_fast=lambda _: Fast(),
    )

    IntermediatePxy = make_intermediate_proxy_type(  # noqa: F841
        "IntermediatePxy",
        FastIntermediate,
        SlowIntermediate,
    )

    assert dir(Pxy) == dir(Slow)
    assert dir(Pxy()) == dir(Slow())
    assert dir(Pxy.prop) == dir(Slow.prop)
    assert dir(Pxy().prop) == dir(Slow().prop)
    assert dir(Pxy.method) == dir(Slow.method)
    assert dir(Pxy().intermediate()) == dir(Slow().intermediate())


@pytest.mark.xfail
@pytest.mark.parametrize(
    "check",
    [
        lambda Pxy, Slow: dir(Pxy().method) == dir(Slow().method),
        lambda Pxy, Slow: (
            dir(Pxy().intermediate().method)
            == dir(Slow().intermediate().method)
        ),
    ],
)
def test_dir_bound_method(
    fast_and_intermediate_with_doc, slow_and_intermediate_with_doc, check
):
    """This test will fail because dir for bound methods is currently
    incorrect, but we have no way to fix it without materializing the slow
    type, which is unnecessarily expensive.
    """
    Fast, FastIntermediate = fast_and_intermediate_with_doc
    Slow, SlowIntermediate = slow_and_intermediate_with_doc

    Pxy = make_final_proxy_type(
        "Pxy",
        Fast,
        Slow,
        fast_to_slow=lambda _: Slow(),
        slow_to_fast=lambda _: Fast(),
    )

    IntermediatePxy = make_intermediate_proxy_type(  # noqa: F841
        "IntermediatePxy",
        FastIntermediate,
        SlowIntermediate,
    )
    assert check(Pxy, Slow)


def test_proxy_binop():
    class Foo:
        pass

    class Bar:
        def __add__(self, other):
            if isinstance(other, Foo):
                return "sum"
            return NotImplemented

        def __radd__(self, other):
            return self.__add__(other)

    FooProxy = make_final_proxy_type(
        "FooProxy",
        _Unusable,
        Foo,
        fast_to_slow=Foo(),
        slow_to_fast=_Unusable(),
    )

    BarProxy = make_final_proxy_type(
        "BarProxy",
        _Unusable,
        Bar,
        fast_to_slow=Bar(),
        slow_to_fast=_Unusable(),
    )

    assert Foo() + Bar() == "sum"
    assert Bar() + Foo() == "sum"
    assert FooProxy() + BarProxy() == "sum"
    assert BarProxy() + FooProxy() == "sum"


def test_slow_attr_still_proxy():
    class A:
        pass

    class B:
        @property
        def _private(self):
            return A()

    pxy_a = make_final_proxy_type(
        "A",
        _Unusable,
        A,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
    )

    pxy_b = make_final_proxy_type(
        "B",
        _Unusable,
        B,
        fast_to_slow=_Unusable(),
        slow_to_fast=_Unusable(),
    )

    result = pxy_b()._private
    assert isinstance(result, pxy_a)


def tuple_with_attrs(name, fields: list[str], extra_fields: set[str]):
    # Build a tuple-like class with some extra attributes and a custom
    # pickling scheme with __getnewargs_ex__
    args = ", ".join(fields)
    kwargs = ", ".join(sorted(extra_fields))
    code = f"""
def __new__(cls, {args}, *, {kwargs}):
    return tuple.__new__(cls, ({args}, ))

def __init__(self, {args}, *, {kwargs}):
    for key, val in zip({sorted(extra_fields)}, [{kwargs}]):
        self.__dict__[key] = val

def __eq__(self, other):
    return (
        type(other) is type(self)
        and tuple.__eq__(self, other)
        and all(getattr(self, k) == getattr(other, k) for k in self._fields)
    )

def __ne__(self, other):
    return not (self == other)

def __getnewargs_ex__(self):
    return tuple(self), self.__dict__
"""
    namespace = {
        "__builtins__": {
            "AttributeError": AttributeError,
            "tuple": tuple,
            "zip": zip,
            "super": super,
            "frozenset": frozenset,
            "type": type,
            "all": all,
            "getattr": getattr,
        }
    }
    exec(code, namespace)
    return type(
        name,
        (tuple,),
        {
            "_fields": frozenset(extra_fields),
            "__eq__": namespace["__eq__"],
            "__getnewargs_ex__": namespace["__getnewargs_ex__"],
            "__init__": namespace["__init__"],
            "__ne__": namespace["__ne__"],
            "__new__": namespace["__new__"],
        },
    )


@pytest.mark.parametrize("attribute_name", ["_fsproxy_fast", "_fsproxy_slow"])
def test_transform_arg_preserves_object_ndarray_identity(
    attribute_name, final_proxy
):
    # An object-dtype ndarray whose elements need no transformation must
    # be returned as-is rather than rebuilt: an equivalent copy breaks
    # aliasing checks such as ``np.may_share_memory(np.asarray(x), x)``
    # that numpy uses (e.g. in ``Generator.permutation``) to decide
    # whether to defensively copy before an in-place shuffle.
    transform = partial(
        _transform_arg, attribute_name=attribute_name, seen=set()
    )
    arr = np.array(["a", "bb", "ccc"], dtype=object)
    assert transform(arr) is arr

    # An object-dtype ndarray containing a proxy is still rebuilt with
    # the unwrapped elements.
    fast_x, slow_x, x = final_proxy
    expected = fast_x if attribute_name == "_fsproxy_fast" else slow_x
    arr_with_proxy = np.array(["a", x], dtype=object)
    result = transform(arr_with_proxy)
    assert result is not arr_with_proxy
    assert result[0] == "a"
    assert type(result[1]) is type(expected)


@pytest.mark.parametrize("attribute_name", ["_fsproxy_fast", "_fsproxy_slow"])
def test_transform_arg_preserves_list_and_dict_identity(attribute_name):
    # A list or dict whose entries need no transformation must be
    # returned as-is rather than rebuilt: a user function may close over
    # a mutable container and mutate it for its side effects (e.g.
    # ``names.append(group.name)`` inside groupby.apply), and rebuilding
    # the closure around an equivalent copy would silently discard those
    # mutations.
    transform = partial(
        _transform_arg, attribute_name=attribute_name, seen=set()
    )
    for unchanged in (
        [],
        [1, "a"],
        {},
        {"k": 1, 2: "v"},
        [[1], {"k": (2,)}],
        {"k": [1, {"nested": 2}]},
    ):
        assert transform(unchanged) is unchanged


@pytest.mark.parametrize("attribute_name", ["_fsproxy_fast", "_fsproxy_slow"])
def test_transform_arg_rebuilds_containers_holding_proxies(
    attribute_name, final_proxy
):
    transform = partial(
        _transform_arg, attribute_name=attribute_name, seen=set()
    )
    fast_x, slow_x, x = final_proxy
    expected_type = type(
        fast_x if attribute_name == "_fsproxy_fast" else slow_x
    )

    lst = [1, x]
    result = transform(lst)
    assert result is not lst
    assert result[0] == 1
    assert type(result[1]) is expected_type

    dct = {"k": x}
    result = transform(dct)
    assert result is not dct
    assert type(result["k"]) is expected_type

    # A proxy nested deeper down rebuilds every enclosing container.
    nested = [{"k": x}]
    result = transform(nested)
    assert result is not nested
    assert type(result[0]["k"]) is expected_type

    # Rebuilt lists preserve list subclasses.
    class MyList(list):
        pass

    my_list = MyList([1, x])
    result = transform(my_list)
    assert result is not my_list
    assert type(result) is MyList
    assert type(result[1]) is expected_type


@pytest.mark.parametrize("attribute_name", ["_fsproxy_fast", "_fsproxy_slow"])
def test_transform_arg_dict_subclass_identity_avoids_overrides(
    attribute_name,
):
    # The unchanged-identity detection must compare entries via ``items()``
    # (which the transformation itself already consumed) rather than
    # ``__iter__``/``__getitem__``, which a mapping subclass may override
    # with semantics that break the comparison.
    class NoDunderDict(dict):
        def __iter__(self):
            raise AssertionError("__iter__ must not be used")

        def __getitem__(self, key):
            raise AssertionError("__getitem__ must not be used")

    transform = partial(
        _transform_arg, attribute_name=attribute_name, seen=set()
    )
    unchanged = NoDunderDict({"k": 1})
    assert transform(unchanged) is unchanged


def test_tuple_with_attrs_transform():
    Bunch = tuple_with_attrs("Bunch", ["a", "b"], {"c", "d"})
    Bunch2 = tuple_with_attrs("Bunch", ["a", "b"], {"c", "d"})
    a = Bunch(1, 2, c=3, d=4)
    b = (1, 2)
    c = Bunch(1, 2, c=4, d=3)
    d = Bunch2(1, 2, c=3, d=4)
    assert a != c
    assert a != b
    assert b != c
    assert a != d
    transform = partial(
        _transform_arg, attribute_name="_fsproxy_fast", seen=set()
    )
    aprime = transform(a)
    bprime = transform(b)
    cprime = transform(c)
    dprime = transform(d)
    assert a == aprime and a is not aprime
    # A plain tuple with no transformable elements keeps its identity.
    assert b is bprime
    assert c == cprime and c is not cprime
    assert d == dprime and d is not dprime


def _make_mirror_proxy():
    class Fast:
        pass

    class SlowBase:
        def inherited(self):
            return "base"

    class Slow(SlowBase):
        const = 42

        def existing(self):
            return "slow original"

        @property
        def prop(self):
            return "slow prop"

        @staticmethod
        def smethod(x):
            return x + 1

        @classmethod
        def cmethod(cls):
            return cls.__name__

    Pxy = make_final_proxy_type(
        "Pxy",
        Fast,
        Slow,
        fast_to_slow=lambda fast: Slow(),
        slow_to_fast=lambda slow: Fast(),
    )
    return Fast, Slow, Pxy


def test_class_attr_mirroring_enabled_after_construction():
    # ``make_*_proxy_type`` enables per-type mirroring once the proxy type is
    # fully built (it starts disabled so the methods installed during
    # construction are not forwarded to the real type).
    _, _, Pxy = _make_mirror_proxy()
    assert Pxy.__dict__["_fsproxy_mirror_slow_overrides"] is True


def test_class_attr_setattr_mirrored_to_slow():
    # A class-level attribute write on the proxy is mirrored onto the
    # underlying "slow" (real) type so it is visible to fallback code.
    _, Slow, Pxy = _make_mirror_proxy()

    def patched(self):
        return "patched"

    Pxy.new_method = patched
    assert Slow.__dict__.get("new_method") is patched


def test_class_attr_delattr_mirrored_to_slow():
    # Deleting a class-level attribute on the proxy mirrors the deletion
    # onto the slow type (the ``__delattr__`` path).
    _, Slow, Pxy = _make_mirror_proxy()

    def patched(self):
        return "patched"

    Pxy.new_method = patched
    assert "new_method" in Slow.__dict__

    del Pxy.new_method
    assert "new_method" not in Pxy.__dict__
    assert "new_method" not in Slow.__dict__


def test_class_attr_delattr_existing_removes_from_slow():
    # Deleting is deleting, not restoring: patching an existing attribute
    # and then deleting it removes it from both the proxy and the slow type,
    # exactly as the same sequence would on a plain Python class.
    _, Slow, Pxy = _make_mirror_proxy()

    def patched(self):
        return "patched"

    Pxy.existing = patched
    assert Slow.__dict__["existing"] is patched

    del Pxy.existing
    assert "existing" not in Pxy.__dict__
    assert "existing" not in Slow.__dict__


def test_class_attr_restore_existing_slow_attr():
    # Re-assigning the proxy's pristine class-dict entry (as ``monkeypatch``
    # and ``mock.patch`` teardown do) restores the slow type's pristine
    # attribute. Note: no prior class-level getattr — the saved descriptor
    # is unresolved, as in the ``mock.patch.object`` flow.
    _, Slow, Pxy = _make_mirror_proxy()
    saved = Pxy.__dict__["existing"]
    assert isinstance(saved, _FastSlowAttribute)
    original_slow = Slow.__dict__["existing"]

    def patched(self):
        return "patched"

    Pxy.existing = patched
    assert Slow.__dict__["existing"] is patched

    Pxy.existing = saved
    assert Slow.__dict__["existing"] is original_slow


def test_class_attr_restore_via_saved_method_proxy():
    # Re-assigning a saved ``Pxy.method`` (a ``_MethodProxy`` over the
    # resolved slow attribute, not the class-dict descriptor) also restores
    # the slow type's pristine attribute.
    _, Slow, Pxy = _make_mirror_proxy()
    saved = getattr(Pxy, "existing")
    original_slow = Slow.__dict__["existing"]

    def patched(self):
        return "patched"

    Pxy.existing = patched
    assert Slow.__dict__["existing"] is patched

    Pxy.existing = saved
    assert Slow.__dict__["existing"] is original_slow


def test_class_attr_monkeypatch_roundtrip(monkeypatch):
    # End-to-end: ``monkeypatch`` of a brand-new attribute mirrors onto the
    # slow type, and teardown (which deletes it) mirrors the deletion.
    _, Slow, Pxy = _make_mirror_proxy()

    def fake(self):
        return "fake"

    monkeypatch.setattr(Pxy, "brand_new", fake, raising=False)
    assert Slow.__dict__.get("brand_new") is fake

    monkeypatch.undo()
    assert "brand_new" not in Pxy.__dict__
    assert "brand_new" not in Slow.__dict__


def test_class_attr_monkeypatch_existing_roundtrip(monkeypatch):
    # End-to-end: ``monkeypatch`` of a pre-existing method mirrors the patch
    # onto the slow type, and teardown (which re-assigns the saved proxy
    # descriptor) restores the slow type's original implementation.
    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["existing"]

    def fake(self):
        return "fake"

    monkeypatch.setattr(Pxy, "existing", fake)
    assert Slow.__dict__["existing"] is fake

    monkeypatch.undo()
    assert isinstance(Pxy.__dict__["existing"], _FastSlowAttribute)
    assert Slow.__dict__["existing"] is original_slow


def test_class_attr_nested_monkeypatch_existing_roundtrip():
    # Nested patches of the same pre-existing method unwind in order,
    # each level restoring the slow type to the previous state.
    from _pytest.monkeypatch import MonkeyPatch

    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["existing"]

    def fake1(self):
        return "fake1"

    def fake2(self):
        return "fake2"

    mp1, mp2 = MonkeyPatch(), MonkeyPatch()
    mp1.setattr(Pxy, "existing", fake1)
    assert Slow.__dict__["existing"] is fake1
    mp2.setattr(Pxy, "existing", fake2)
    assert Slow.__dict__["existing"] is fake2

    mp2.undo()
    assert Slow.__dict__["existing"] is fake1
    mp1.undo()
    assert Slow.__dict__["existing"] is original_slow


def test_class_attr_mock_patch_object_roundtrip():
    # ``unittest.mock.patch.object`` saves the raw class-dict entry without
    # a prior getattr (so the saved descriptor is never resolved); undo must
    # still restore the slow type's pristine attribute.
    from unittest import mock

    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["existing"]

    def fake(self):
        return "fake"

    with mock.patch.object(Pxy, "existing", fake):
        assert Slow.__dict__["existing"] is fake
    assert Slow.__dict__["existing"] is original_slow


def test_class_attr_property_monkeypatch_roundtrip(monkeypatch):
    # Patching a property mirrors it onto the slow type; undo restores the
    # slow type's pristine property object.
    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["prop"]

    fake = property(lambda self: "fake")
    monkeypatch.setattr(Pxy, "prop", fake)
    assert Slow.__dict__["prop"] is fake
    assert Slow().prop == "fake"

    monkeypatch.undo()
    assert Slow.__dict__["prop"] is original_slow
    assert Slow().prop == "slow prop"


def test_class_attr_data_attr_monkeypatch_roundtrip(monkeypatch):
    # Patching a plain class data attribute round-trips on the slow type.
    _, Slow, Pxy = _make_mirror_proxy()

    monkeypatch.setattr(Pxy, "const", 99)
    assert Slow.const == 99

    monkeypatch.undo()
    assert Slow.const == 42


def test_class_attr_staticmethod_monkeypatch_roundtrip(monkeypatch):
    # Undo restores the slow type's pristine ``staticmethod`` descriptor,
    # not the plain function it resolves to (which would break instance
    # calls by receiving ``self``).
    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["smethod"]
    assert isinstance(original_slow, staticmethod)

    monkeypatch.setattr(Pxy, "smethod", staticmethod(lambda x: x - 1))
    assert Slow.smethod(1) == 0

    monkeypatch.undo()
    assert Slow.__dict__["smethod"] is original_slow
    assert Slow().smethod(1) == 2


def test_class_attr_classmethod_monkeypatch_roundtrip(monkeypatch):
    # Undo restores the slow type's pristine ``classmethod`` descriptor,
    # not the class-bound method it resolves to (which would pin ``cls``
    # for subclasses).
    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["cmethod"]
    assert isinstance(original_slow, classmethod)

    monkeypatch.setattr(Pxy, "cmethod", classmethod(lambda cls: "fake"))
    assert Slow.cmethod() == "fake"

    monkeypatch.undo()
    assert Slow.__dict__["cmethod"] is original_slow
    assert Slow.cmethod() == "Slow"


def test_class_attr_inherited_method_monkeypatch_roundtrip(monkeypatch):
    # Patching a method the slow type only inherits mirrors it into the slow
    # type's own dict; undo removes that entry again (rather than copying
    # the base-class implementation into the subclass), leaving the
    # inherited implementation visible.
    _, Slow, Pxy = _make_mirror_proxy()
    assert "inherited" not in Slow.__dict__

    def fake(self):
        return "fake"

    monkeypatch.setattr(Pxy, "inherited", fake)
    assert Slow.__dict__["inherited"] is fake

    monkeypatch.undo()
    assert "inherited" not in Slow.__dict__
    assert Slow().inherited() == "base"


def test_class_attr_monkeypatch_delattr_roundtrip(monkeypatch):
    # ``monkeypatch.delattr`` mirrors the deletion onto the slow type, and
    # undo (which re-assigns the saved pristine descriptor) restores the
    # slow type's pristine attribute.
    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["existing"]

    monkeypatch.delattr(Pxy, "existing")
    assert "existing" not in Pxy.__dict__
    assert "existing" not in Slow.__dict__

    monkeypatch.undo()
    assert Slow.__dict__["existing"] is original_slow


def test_setattr_fsproxy_no_mirror_skips_slow():
    # ``_setattr_fsproxy_no_mirror`` sets a class attribute on the proxy
    # without forwarding it to the slow type (used for cudf.pandas' own custom
    # methods such as ``DataFrame.query``/``eval``).
    _, Slow, Pxy = _make_mirror_proxy()

    def custom(self):
        return "custom"

    _setattr_fsproxy_no_mirror(Pxy, "custom_method", custom)
    assert Pxy.__dict__["custom_method"] is custom
    assert "custom_method" not in Slow.__dict__


def test_setattr_fsproxy_no_mirror_monkeypatch_roundtrip(monkeypatch):
    # A cudf-installed custom attribute (e.g. ``DataFrame.eval``/``query``)
    # participates in the pristine state: monkeypatching it and undoing
    # restores the proxy's custom object AND the slow type's own genuine
    # implementation — the cudf-internal object is never forwarded to the
    # slow type.
    _, Slow, Pxy = _make_mirror_proxy()
    original_slow = Slow.__dict__["existing"]

    def custom(self):
        return "cudf custom"

    _setattr_fsproxy_no_mirror(Pxy, "existing", custom)
    assert Slow.__dict__["existing"] is original_slow

    def fake(self):
        return "fake"

    monkeypatch.setattr(Pxy, "existing", fake)
    assert Slow.__dict__["existing"] is fake

    monkeypatch.undo()
    assert Pxy.__dict__["existing"] is custom
    assert Slow.__dict__["existing"] is original_slow
