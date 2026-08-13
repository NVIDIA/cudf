# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import functools
import inspect
import operator
import pickle
import types
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import IntEnum
from typing import Any, Literal

import numpy as np

from rmm import RMMError

import cudf

from ..options import _env_get_bool
from ..testing import assert_eq
from .annotation import nvtx
from .proxy_base import ProxyNDarrayBase


def call_operator(fn, args, kwargs):
    return fn(*args, **kwargs)


_CUDF_PANDAS_NVTX_COLORS = {
    "COPY_SLOW_TO_FAST": 0xCA0020,
    "COPY_FAST_TO_SLOW": 0xF4A582,
    "EXECUTE_FAST": 0x92C5DE,
    "EXECUTE_SLOW": 0x0571B0,
}

# This is a dict of functions that are known to have arguments that
# need to be transformed from fast to slow only. i.e., Some cudf functions
# error on passing a device object but don't error on passing a host object.
# For example: DataFrame.__setitem__(arg, value) errors on passing a
# cudf.Index object but doesn't error on passing a pd.Index object.
# Hence we need to transform the arg from fast to slow only. So, we use
# a dictionary like:
# {"DataFrame.__setitem__": {0}}
# where the keys are the function names and the values are the indices
# (0-based) of the arguments that need to be transformed.

_SPECIAL_FUNCTIONS_ARGS_MAP = {
    "DataFrame.__setitem__": {0},
}

_WRAPPER_ASSIGNMENTS = tuple(
    attr
    for attr in functools.WRAPPER_ASSIGNMENTS
    # Skip __doc__ because we assign it on class creation using exec_body
    # callable that updates the namespace of the class.
    # Skip __annotations__ / __annotate__ (PEP 749, added on 3.14) because
    # there are differences between Python versions on how they're
    # initialized for a class that doesn't explicitly define them and we
    # don't want to force eager evaluation of anything that would normally
    # be lazy. On 3.14 specifically, simply reading ``__annotate__`` causes
    # ``__annotate_func__`` to materialize on the source class, which then
    # shows up in ``dir()`` and diverges from the cached ``_fsproxy_slow_dir``
    # captured a few lines earlier.
    if attr not in ("__annotations__", "__annotate__", "__doc__")
)


def callers_module_name():
    # Call f_back twice since this function adds an extra frame
    return inspect.currentframe().f_back.f_back.f_globals["__name__"]


class _State(IntEnum):
    """Simple enum to track the type of wrapped object of a final proxy"""

    SLOW = 0
    FAST = 1


class _BlockState(IntEnum):
    """Simple enum to track blocking direction of wrapped object of a final proxy"""

    TO_SLOW = 0
    TO_FAST = 1


class _Unusable:
    """
    A totally unusable type. When a "fast" object is not available,
    it's useful to set it to _Unusable() so that any operations
    on it fail, and ensure fallback to the corresponding
    "slow" object.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError(
            "Fast implementation not available. "
            "Falling back to the slow implementation"
        )

    def __getattribute__(self, name: str) -> Any:
        if name in {"__class__"}:  # needed for type introspection
            return super().__getattribute__(name)
        raise TypeError("Unusable type. Falling back to the slow object")

    def __repr__(self) -> str:
        raise AttributeError("Unusable type. Falling back to the slow object")


class _PickleConstructor:
    """A pickleable object to support construction in __reduce__.

    This object is used to avoid having unpickling call __init__ on the
    objects, instead only invoking __new__. __init__ may have required
    arguments or otherwise perform invalid initialization that we could skip
    altogether since we're going to overwrite the wrapped object.
    """

    def __init__(self, type_):
        self._type = type_

    def __call__(self):
        return object.__new__(get_final_type_map().get(self._type, self._type))


_DELETE = object()


def _make_proxy_getattr():
    """
    Create an optimized ``__getattr__`` for proxy types.

    Uses ``_fsproxy_wrapped`` directly instead of the
    ``_fsproxy_fast``/``_fsproxy_slow`` properties to avoid
    triggering expensive fast-slow type conversions on every
    attribute miss.

    The fallback uses ``_fsproxy_fast_to_slow()`` /
    ``_fsproxy_slow_to_fast()`` directly (not the ``_fsproxy_slow``
    / ``_fsproxy_fast`` properties) so that the stored
    ``_fsproxy_wrapped`` is **not** mutated.  This avoids the
    fast-slow-fast ping-pong that would otherwise occur on
    consecutive attribute misses.

    A per-instance negative cache (``_fsproxy_missing_attrs``)
    records attribute names that were not found on either the fast
    or slow representation.  The first miss for a given name pays
    the conversion cost, but all subsequent misses are O(1).
    """

    def __getattr__(self, name):
        # Fast path: check per-instance negative cache.
        inst_dict = object.__getattribute__(self, "__dict__")
        missing = inst_dict.get("_fsproxy_missing_attrs")
        if missing is not None and name in missing:
            raise AttributeError(name)

        wrapped = self._fsproxy_wrapped
        try:
            result = getattr(wrapped, name)
        except AttributeError:
            # The current wrapped object doesn't have this attr.
            # Try the other representation without storing the
            # conversion (avoids ping-pong).
            try:
                if isinstance(wrapped, self._fsproxy_fast_type):
                    other = self._fsproxy_fast_to_slow()
                else:
                    other = self._fsproxy_slow_to_fast()
            except Exception:
                # Conversion failed — treat as missing.
                raise AttributeError(name) from None
            try:
                result = getattr(other, name)
            except AttributeError:
                # Neither side has it — cache and re-raise.
                if missing is None:
                    missing = set()
                    inst_dict["_fsproxy_missing_attrs"] = missing
                missing.add(name)
                raise
        except Exception:
            # Non-AttributeError (e.g., cudf internal error):
            # fall back to the other representation.
            if isinstance(wrapped, self._fsproxy_fast_type):
                result = getattr(self._fsproxy_fast_to_slow(), name)
            else:
                result = getattr(self._fsproxy_slow_to_fast(), name)
        return _maybe_wrap_result(result, getattr, self, name)

    return __getattr__


def make_final_proxy_type(
    name: str,
    fast_type: type,
    slow_type: type,
    *,
    fast_to_slow: Callable,
    slow_to_fast: Callable,
    module: str | None = None,
    additional_attributes: Mapping[str, Any] | None = None,
    postprocess: Callable[[_FinalProxy, Any, Any], Any] | None = None,
    bases: tuple = (),
    metaclasses: tuple = (),
) -> type[_FinalProxy]:
    """
    Defines a fast-slow proxy type for a pair of "final" fast and slow
    types. Final types are types for which known operations exist for
    converting an object of "fast" type to "slow" and vice-versa.

    Parameters
    ----------
    name: str
        The name of the class returned
    fast_type: type
    slow_type: type
    fast_to_slow: callable
        Function that accepts a single argument of type `fast_type`
        and returns an object of type `slow_type`
    slow_to_fast: callable
        Function that accepts a single argument of type `slow_type`
        and returns an object of type `fast_type`
    module: str | None
        Module name to assign to the generated proxy type.
    additional_attributes
        Mapping of additional attributes to add to the class
       (optional), these will override any defaulted attributes (e.g.
       ``__init__``). If you want to remove a defaulted attribute
       completely, pass the special sentinel ``_DELETE`` as a value.
    postprocess
        Optional function called to allow the proxy to postprocess
        itself when being wrapped up, called with the proxy object,
        the unwrapped result object, and the function that was used to
        construct said unwrapped object. See also `_maybe_wrap_result`.
    bases
        Optional tuple of base classes to insert into the mro.
    metaclasses
        Optional tuple of metaclasses to unify with the base proxy metaclass.

    Notes
    -----
    As a side-effect, this function adds `fast_type` and `slow_type`
    to a global mapping of final types to their corresponding proxy
    types, accessible via `get_final_type_map()`.
    """

    def __init__(self, *args, **kwargs):
        _fast_slow_function_call(
            lambda cls, args, kwargs: setattr(
                self, "_fsproxy_wrapped", cls(*args, **kwargs)
            ),
            None,
            type(self),
            args,
            kwargs,
        )

    @nvtx.annotate(
        "COPY_SLOW_TO_FAST",
        color=_CUDF_PANDAS_NVTX_COLORS["COPY_SLOW_TO_FAST"],
        domain="cudf_pandas",
    )
    def _fsproxy_slow_to_fast(self):
        # if we are wrapping a slow object,
        # convert it to a fast one
        if self._fsproxy_state is _State.SLOW:
            return slow_to_fast(self._fsproxy_wrapped)
        return self._fsproxy_wrapped

    @nvtx.annotate(
        "COPY_FAST_TO_SLOW",
        color=_CUDF_PANDAS_NVTX_COLORS["COPY_FAST_TO_SLOW"],
        domain="cudf_pandas",
    )
    def _fsproxy_fast_to_slow(self):
        # if we are wrapping a fast object,
        # convert it to a slow one
        if self._fsproxy_state is _State.FAST:
            return fast_to_slow(self._fsproxy_wrapped)
        return self._fsproxy_wrapped

    def as_gpu_object(self):
        return self._fsproxy_slow_to_fast()

    def as_cpu_object(self):
        return self._fsproxy_fast_to_slow()

    @property  # type: ignore[misc]
    def _fsproxy_state(self) -> _State:
        return (
            _State.FAST
            if isinstance(self._fsproxy_wrapped, self._fsproxy_fast_type)
            else _State.SLOW
        )

    slow_dir = dir(slow_type)
    cls_dict = {
        "__init__": __init__,
        "__doc__": inspect.getdoc(slow_type),
        "_fsproxy_slow_dir": slow_dir,
        "_fsproxy_fast_type": fast_type,
        "_fsproxy_slow_type": slow_type,
        "_fsproxy_slow_to_fast": _fsproxy_slow_to_fast,
        "_fsproxy_fast_to_slow": _fsproxy_fast_to_slow,
        "as_gpu_object": as_gpu_object,
        "as_cpu_object": as_cpu_object,
        "_fsproxy_state": _fsproxy_state,
    }

    if additional_attributes is None:
        additional_attributes = {}
    slow_type_dir = dir(slow_type)
    for method in _SPECIAL_METHODS:
        if method in slow_type_dir and getattr(slow_type, method, False):
            cls_dict[method] = _FastSlowAttribute(method)

    if hasattr(slow_type, "__getattr__"):
        cls_dict["__getattr__"] = _make_proxy_getattr()

    for k, v in additional_attributes.items():
        if v is _DELETE and k in cls_dict:
            del cls_dict[k]
        elif v is not _DELETE:
            cls_dict[k] = v

    for slow_name in slow_type_dir:
        if slow_name in cls_dict or slow_name.startswith("__"):
            continue
        else:
            cls_dict[slow_name] = _FastSlowAttribute(
                slow_name, private=slow_name.startswith("_")
            )

    metaclass = _FastSlowProxyMeta
    if metaclasses:
        metaclass = types.new_class(  # type: ignore[assignment]
            f"{name}_Meta",
            (*metaclasses, _FastSlowProxyMeta),
            {},
        )
    cls = types.new_class(
        name,
        (*bases, _FinalProxy),
        {"metaclass": metaclass},
        lambda ns: ns.update(cls_dict),
    )
    functools.update_wrapper(
        cls,
        slow_type,
        assigned=_WRAPPER_ASSIGNMENTS,
        updated=(),
    )
    cls.__module__ = module if module is not None else callers_module_name()

    final_type_map = get_final_type_map()
    if fast_type is not _Unusable:
        final_type_map[fast_type] = cls
    final_type_map[slow_type] = cls

    # Proxy type fully constructed: snapshot its pristine state and, from
    # here on, mirror class-level attribute writes (genuine runtime
    # monkeypatches) onto the underlying "slow" (real) type.
    _enable_fsproxy_mirroring(cls)

    return cls


def make_intermediate_proxy_type(
    name: str,
    fast_type: type,
    slow_type: type,
    *,
    module: str | None = None,
    additional_attributes: Mapping[str, Any] | None = None,
) -> type[_IntermediateProxy]:
    """
    Defines a proxy type for a pair of "intermediate" fast and slow
    types. Intermediate types are the types of the results of
    operations invoked on final types.

    As a side-effect, this function adds `fast_type` and `slow_type`
    to a global mapping of intermediate types to their corresponding
    proxy types, accessible via `get_intermediate_type_map()`.

    Parameters
    ----------
    name: str
        The name of the class returned
    fast_type: type
    slow_type: type
    module: str | None
        Module name to assign to the generated proxy type.
    additional_attributes
        Mapping of additional attributes to add to the class
       (optional), these will override any defaulted attributes (e.g.
       ``__init__``). If you want to remove a defaulted attribute
       completely, pass the special sentinel ``_DELETE`` as a value.
    """

    def __init__(self, *args, **kwargs):
        # disallow __init__. An intermediate proxy type can only be
        # instantiated from (possibly chained) operations on a final
        # proxy type.
        raise TypeError(
            f"Cannot directly instantiate object of type {type(self)}"
        )

    @property  # type: ignore[misc]
    def _fsproxy_state(self):
        return (
            _State.FAST
            if isinstance(self._fsproxy_wrapped, self._fsproxy_fast_type)
            else _State.SLOW
        )

    @nvtx.annotate(
        "COPY_SLOW_TO_FAST",
        color=_CUDF_PANDAS_NVTX_COLORS["COPY_SLOW_TO_FAST"],
        domain="cudf_pandas",
    )
    def _fsproxy_slow_to_fast(self):
        if self._fsproxy_state is _State.SLOW:
            return super(type(self), self)._fsproxy_slow_to_fast()
        # Already fast: re-derive from the originating call if a parent proxy
        # was mutated/replaced since creation, so the intermediate reflects the
        # live parent (e.g. a column added to a frame after its groupby).
        if self._fsproxy_parents_changed():
            result = super(type(self), self)._fsproxy_slow_to_fast()
            # Refresh the cache (not just the parent-id snapshot) so later
            # conversions return the re-derived object instead of falling back
            # to the stale one once the snapshot matches the live parents again.
            self._fsproxy_wrapped = result
            self._fsproxy_record_parent_ids()
            return result
        return self._fsproxy_wrapped

    @nvtx.annotate(
        "COPY_FAST_TO_SLOW",
        color=_CUDF_PANDAS_NVTX_COLORS["COPY_FAST_TO_SLOW"],
        domain="cudf_pandas",
    )
    def _fsproxy_fast_to_slow(self):
        if self._fsproxy_state is _State.FAST:
            return super(type(self), self)._fsproxy_fast_to_slow()
        # Already slow: re-derive from the originating call if a parent proxy
        # was mutated/replaced since creation (see _fsproxy_slow_to_fast).
        if self._fsproxy_parents_changed():
            result = super(type(self), self)._fsproxy_fast_to_slow()
            # See _fsproxy_slow_to_fast: refresh the cache so subsequent
            # conversions don't revert to the stale wrapped object.
            self._fsproxy_wrapped = result
            self._fsproxy_record_parent_ids()
            return result
        return self._fsproxy_wrapped

    slow_dir = dir(slow_type)
    cls_dict = {
        "__init__": __init__,
        "__doc__": inspect.getdoc(slow_type),
        "_fsproxy_slow_dir": slow_dir,
        "_fsproxy_fast_type": fast_type,
        "_fsproxy_slow_type": slow_type,
        "_fsproxy_slow_to_fast": _fsproxy_slow_to_fast,
        "_fsproxy_fast_to_slow": _fsproxy_fast_to_slow,
        "_fsproxy_state": _fsproxy_state,
    }
    for method in _SPECIAL_METHODS:
        if method in slow_dir and getattr(slow_type, method, False):
            cls_dict[method] = _FastSlowAttribute(method)

    if hasattr(slow_type, "__getattr__"):
        cls_dict["__getattr__"] = _make_proxy_getattr()

    if additional_attributes is None:
        additional_attributes = {}
    for k, v in additional_attributes.items():
        cls_dict[k] = v

    for slow_name in dir(slow_type):
        if slow_name in cls_dict or slow_name.startswith("__"):
            continue
        else:
            cls_dict[slow_name] = _FastSlowAttribute(
                slow_name, private=slow_name.startswith("_")
            )

    for slow_name in getattr(slow_type, "_attributes", []):
        if slow_name in cls_dict:
            continue
        else:
            cls_dict[slow_name] = _FastSlowAttribute(
                slow_name, private=slow_name.startswith("_")
            )

    cls = types.new_class(
        name,
        (_IntermediateProxy,),
        {"metaclass": _FastSlowProxyMeta},
        lambda ns: ns.update(cls_dict),
    )
    functools.update_wrapper(
        cls,
        slow_type,
        assigned=_WRAPPER_ASSIGNMENTS,
        updated=(),
    )
    cls.__module__ = module if module is not None else callers_module_name()

    intermediate_type_map = get_intermediate_type_map()
    if fast_type is not _Unusable:
        intermediate_type_map[fast_type] = cls
    intermediate_type_map[slow_type] = cls

    # Proxy type fully constructed: snapshot its pristine state and, from
    # here on, mirror class-level attribute writes (genuine runtime
    # monkeypatches) onto the underlying "slow" (real) type.
    _enable_fsproxy_mirroring(cls)

    return cls


def register_proxy_func(slow_func: Callable):
    """
    Decorator to register custom function as a proxy for slow_func.

    Parameters
    ----------
    slow_func: Callable
        The function to register a wrapper for.

    Returns
    -------
    Callable
    """

    def wrapper(func):
        registered_functions = get_registered_functions()
        registered_functions[slow_func] = func
        functools.update_wrapper(func, slow_func)
        return func

    return wrapper


@functools.lru_cache(maxsize=None)
def get_final_type_map():
    """
    Return the mapping of all known fast and slow final types to their
    corresponding proxy types.
    """
    return dict()


@functools.lru_cache(maxsize=None)
def get_intermediate_type_map():
    """
    Return a mapping of all known fast and slow intermediate types to their
    corresponding proxy types.
    """
    return dict()


@functools.lru_cache(maxsize=None)
def get_registered_functions():
    return dict()


_SLOW_ABSENT = object()


def _enable_fsproxy_mirroring(cls: type) -> None:
    """Finalize a proxy type for class-level patch mirroring.

    Snapshots the proxy type's pristine public class attributes together
    with the slow type's pristine class-dict entries for the same names,
    then enables mirroring of class-level attribute writes/deletions onto
    the slow type (see ``_FastSlowProxyMeta.__setattr__``/``__delattr__``).

    The snapshot is a fixed translation table, not runtime patch tracking:
    re-assigning the proxy's pristine attribute for ``name`` (which is what
    ``monkeypatch``/``mock.patch`` save and re-assign on undo) translates
    to restoring the slow type's pristine attribute for ``name``.
    """
    slow = cls._fsproxy_slow_type  # type: ignore[attr-defined]
    pristine = {
        name: (value, slow.__dict__.get(name, _SLOW_ABSENT))
        for name, value in cls.__dict__.items()
        if not name.startswith("_")
    }
    type.__setattr__(cls, "_fsproxy_pristine_attrs", pristine)
    type.__setattr__(cls, "_fsproxy_mirror_slow_overrides", True)


def _setattr_fsproxy_no_mirror(cls: type, name: str, value: Any) -> None:
    """Install a cudf.pandas-internal attribute on a proxy type.

    ``_FastSlowProxyMeta.__setattr__`` mirrors class-level attribute writes
    onto the underlying "slow" (real) type so that runtime monkeypatches stay
    visible to the pandas fallback path. cudf.pandas itself installs a handful
    of custom methods (e.g. ``DataFrame.query``/``DataFrame.eval``) onto the
    proxy classes that must *not* clobber pandas' genuine implementations; use
    this helper for those (rare) assignments, after the proxy type has been
    fully constructed by ``make_*_proxy_type``. The attribute is registered as
    part of the proxy's pristine state so that a later save/patch/re-assign
    cycle restores the slow type's own attribute rather than forwarding the
    cudf-internal object to it.
    """
    type.__setattr__(cls, name, value)
    if not name.startswith("_"):
        slow = cls._fsproxy_slow_type  # type: ignore[attr-defined]
        cls._fsproxy_pristine_attrs[name] = (  # type: ignore[attr-defined]
            value,
            slow.__dict__.get(name, _SLOW_ABSENT),
        )


class _FastSlowProxyMeta(type):
    """
    Metaclass used to dynamically find class attributes and
    classmethods of fast-slow proxy types.
    """

    _fsproxy_slow_dir: list
    _fsproxy_slow_type: type
    _fsproxy_fast_type: type

    @property
    def _fsproxy_slow(self) -> type:
        return self._fsproxy_slow_type

    @property
    def _fsproxy_fast(self) -> type:
        return self._fsproxy_fast_type

    def __new__(mcls, *args, **kwargs):
        cls = super().__new__(mcls, *args, **kwargs)
        # Per-proxy-type switch controlling whether class-level attribute
        # writes are mirrored onto the underlying "slow" (real) type (see
        # ``__setattr__``/``__delattr__``). It starts disabled so that the
        # attributes installed while the proxy type is being built are not
        # forwarded to the real type; ``make_*_proxy_type`` enables it once
        # construction is complete. Initialized in ``__new__`` rather than
        # ``__init__`` because cooperating metaclasses may perform
        # class-level attribute writes from their own ``__new__`` — e.g.
        # ``ABCMeta.__new__`` assigns ``__abstractmethods__``, dispatching
        # to ``__setattr__`` below before ``__init__`` ever runs.
        type.__setattr__(cls, "_fsproxy_mirror_slow_overrides", False)
        return cls

    def __setattr__(cls, name, value):
        # Class-level attribute assignments on a proxy type (e.g.
        # ``monkeypatch.setattr(pd.ExcelFile, "parse", fn)``) must also be
        # mirrored onto the underlying "slow" (real) type. Code that runs
        # under ``disable_module_accelerator()`` (e.g. the pandas fallback
        # path of ``pd.read_excel``) resolves attributes from the real
        # class, not the proxy, so a patch applied only to the proxy would
        # otherwise be invisible to that code. The assigned value is first
        # translated into its slow-space equivalent: re-assigning the
        # proxy's pristine attribute translates to the slow type's pristine
        # attribute, and proxy machinery is unwrapped to the slow object it
        # delegates to, so save/patch/re-assign cycles round-trip on the
        # real type as well.
        type.__setattr__(cls, name, value)
        if not cls._fsproxy_mirror_slow_overrides:
            # The proxy type is still being constructed (or this is a
            # non-pandas proxy): only mirror user/runtime monkeypatches,
            # never the custom methods cudf.pandas installs on the proxy
            # classes itself.
            return
        if name.startswith("_"):
            return
        slow = cls._fsproxy_slow_type
        try:
            # Mirroring is best-effort: translating a wrapped proxy instance
            # can require a fast-to-slow conversion, which may itself fail;
            # never let that escape an otherwise-successful assignment.
            pristine = cls._fsproxy_pristine_attrs
            entry = pristine.get(name)
            if entry is not None and value is entry[0]:
                # The proxy's pristine attribute for ``name`` is being
                # re-assigned (e.g. ``monkeypatch``/``mock.patch`` undo
                # re-setting the saved class-dict entry). Its slow-space
                # equivalent is the slow type's pristine attribute.
                if entry[1] is _SLOW_ABSENT:
                    # The slow type never defined ``name`` itself: the proxy
                    # mirrors the slow type's *dir*, so it has pristine
                    # attributes for methods the slow type only inherits
                    # (e.g. ``DataFrame.head`` lives on ``NDFrame``).
                    # Mirroring a patch for such a name added a shadowing
                    # entry to the slow type's dict; undoing the patch must
                    # remove that entry again so the inherited
                    # implementation becomes visible. It may legitimately be
                    # missing (the mirror is best-effort), hence the guard.
                    if name in slow.__dict__:
                        delattr(slow, name)
                else:
                    setattr(slow, name, entry[1])
                return
            # Otherwise translate the assigned value into "slow" space
            # before mirroring it: a value read off a proxy type (e.g. the
            # original that a caller saves before patching and re-assigns
            # to undo) is proxy machinery wrapping a slow-side object, and
            # mirroring it verbatim would install that machinery on the
            # real type. Unwrap it to the slow object it delegates to, so
            # save/patch/re-assign cycles round-trip on the real type;
            # values with no determinable slow-side equivalent are not
            # mirrored at all.
            if isinstance(value, _FastSlowAttribute):
                # The proxy's own delegating descriptor (a *pristine* one is
                # already handled by identity above; this covers a
                # descriptor obtained some other way): its slow equivalent
                # is the method it wraps, if it ever resolved one.
                attr = value._attr
                if not isinstance(attr, _MethodProxy):
                    return
                value = attr
            if isinstance(value, _FunctionProxy):
                unwrapped = value._fsproxy_slow
                if entry is not None and entry[1] is not _SLOW_ABSENT:
                    # Re-assigning a saved ``cls.method`` (a ``_MethodProxy``
                    # over the *resolved* slow attribute): if it resolves
                    # back to the slow type's pristine attribute, restore
                    # the pristine class-dict entry itself so
                    # ``classmethod``/``staticmethod`` descriptors are not
                    # degraded to their bound/plain-function forms.
                    descriptor = entry[1]
                    try:
                        resolved = (
                            descriptor.__get__(None, slow)
                            if hasattr(type(descriptor), "__get__")
                            else descriptor
                        )
                        if unwrapped is resolved or unwrapped == resolved:
                            setattr(slow, name, descriptor)
                            return
                    except Exception:
                        pass
                setattr(slow, name, unwrapped)
            elif isinstance(value, _FastSlowProxy):
                setattr(slow, name, value._fsproxy_slow)
            elif isinstance(value, _FastSlowProxyMeta):
                slow_type = getattr(value, "_fsproxy_slow_type", None)
                if slow_type is not None:
                    setattr(slow, name, slow_type)
            else:
                setattr(slow, name, value)
        except Exception:
            pass

    def __delattr__(cls, name):
        # Mirror class-level attribute *deletions* onto the underlying "slow"
        # (real) type as well, for the same reason as ``__setattr__``: after
        # ``del cls.name`` the attribute is gone from the proxy, so it must
        # also be gone from the real type seen by fallback code.
        type.__delattr__(cls, name)
        if not cls._fsproxy_mirror_slow_overrides:
            return
        if name.startswith("_"):
            return
        try:
            delattr(cls._fsproxy_slow_type, name)
        except (AttributeError, TypeError):
            pass

    def __dir__(self):
        # Try to return the cached dir of the slow object, but if it
        # doesn't exist, fall back to the default implementation.
        try:
            return self._fsproxy_slow_dir
        except AttributeError:
            return type.__dir__(self)

    @property
    def _is_proxy_base_class(self) -> bool:
        # True if this class is the base proxy class registered for its
        # slow type, as opposed to e.g. a user-defined subclass of a
        # proxy type (which shares ``_fsproxy_slow_type`` with its
        # parent but must not match arbitrary instances of the slow
        # type).
        slow = getattr(self, "_fsproxy_slow_type", None)
        return slow is not None and (
            get_final_type_map().get(slow) is self
            or get_intermediate_type_map().get(slow) is self
        )

    def __subclasscheck__(self, __subclass: type) -> bool:
        if super().__subclasscheck__(__subclass):
            return True
        if hasattr(__subclass, "_fsproxy_slow"):
            return issubclass(__subclass._fsproxy_slow, self._fsproxy_slow)
        if self._is_proxy_base_class:
            # An unproxied class (e.g. a user-defined subclass of the
            # slow type such as ``pandas._testing.SubclassedDataFrame``)
            # is a subclass of the proxy standing in for its parent.
            return issubclass(__subclass, self._fsproxy_slow)
        return False

    def __instancecheck__(self, __instance: Any) -> bool:
        if super().__instancecheck__(__instance):
            return True
        if hasattr(type(__instance), "_fsproxy_slow"):
            return issubclass(type(__instance), self)
        if self._is_proxy_base_class:
            # A raw (unproxied) slow object, e.g. produced by operations
            # on an unproxied pandas subclass, is an instance of the
            # proxy type standing in for its class.
            return isinstance(__instance, self._fsproxy_slow)
        return False


class _FastSlowProxy:
    """
    Base class for all fast=slow proxy types.

    A fast-slow proxy is proxy for a pair of types that provide "fast"
    and "slow" implementations of the same API.  At any time, a
    fast-slow proxy wraps an object of either "fast" type, or "slow"
    type. Operations invoked on the fast-slow proxy are first
    delegated to the "fast" type, and if that fails, to the "slow"
    type.
    """

    _fsproxy_wrapped: Any
    # Instance-level transfer blocking flag
    _fsproxy_transfer_block: _BlockState | None = None

    def _fsproxy_fast_to_slow(self) -> Any:
        """
        If the wrapped object is of "fast" type, returns the
        corresponding "slow" object. Otherwise, returns the wrapped
        object as-is.
        """
        raise NotImplementedError("Abstract base class")

    def _fsproxy_slow_to_fast(self) -> Any:
        """
        If the wrapped object is of "slow" type, returns the
        corresponding "fast" object. Otherwise, returns the wrapped
        object as-is.
        """
        raise NotImplementedError("Abstract base class")

    @property
    def _fsproxy_fast(self) -> Any:
        """
        Returns the wrapped object. If the wrapped object is of "slow"
        type, replaces it with the corresponding "fast" object before
        returning it.
        """
        # Check for transfer blocking before attempting conversion
        if (
            self._fsproxy_transfer_block is _BlockState.TO_FAST
            and self._fsproxy_state is _State.SLOW  # type: ignore[attr-defined]
        ):
            raise RuntimeError("Fast-to-slow transfer is blocked")

        self._fsproxy_wrapped = self._fsproxy_slow_to_fast()
        return self._fsproxy_wrapped

    @property
    def _fsproxy_slow(self) -> Any:
        """
        Returns the wrapped object. If the wrapped object is of "fast"
        type, replaces it with the corresponding "slow" object before
        returning it.
        """
        self._fsproxy_wrapped = self._fsproxy_fast_to_slow()
        return self._fsproxy_wrapped

    def set_transfer_block_state(
        self, transfer_block: _BlockState | None = None
    ):
        """Set transfer blocking flag for this specific instance.

        Parameters
        ----------
        transfer_block : _BlockState | None
            None: No blocking (default)
            _BlockState.TO_FAST: Block slow-to-fast transfers.
            _BlockState.TO_SLOW: Block fast-to-slow transfers.
        """
        self._fsproxy_transfer_block = transfer_block

    def get_transfer_blocking(self):
        """Get current instance-level transfer blocking flag."""
        return getattr(self, "_fsproxy_transfer_block", None)

    def force_state(self, state: _State):
        """Force the proxy to a specific state (FAST or SLOW) and block opposite transfers."""
        if state == _State.FAST:
            # Force to fast state and block fast-to-slow transfers
            self._fsproxy_wrapped = self._fsproxy_slow_to_fast()
            self.set_transfer_block_state(_BlockState.TO_SLOW)
        elif state == _State.SLOW:
            # Force to slow state and block slow-to-fast transfers
            self._fsproxy_wrapped = self._fsproxy_fast_to_slow()
            self.set_transfer_block_state(_BlockState.TO_FAST)
        else:
            raise ValueError(
                f"Invalid state: {state}. Must be _State.FAST or _State.SLOW"
            )

    def unblock_transfers(self):
        """Remove all transfer blocking for this instance."""
        self.set_transfer_block_state(None)

    def __dir__(self):
        # Try to return the cached dir of the slow object, but if it
        # doesn't exist, fall back to the default implementation.
        try:
            return self._fsproxy_slow_dir
        except AttributeError:
            return object.__dir__(self)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            # If the class declares this private name as a
            # ``_FastSlowAttribute``, forward the write to the wrapped
            # object so that behaviors driven by that attribute
            # (e.g. ``_readonly`` propagation in
            # ``ArrowExtensionArray.__getitem__``) are preserved.
            descriptor = inspect.getattr_static(type(self), name, None)
            if isinstance(descriptor, _FastSlowAttribute):
                try:
                    wrapped = object.__getattribute__(self, "_fsproxy_wrapped")
                except AttributeError:
                    wrapped = None
                if wrapped is not None:
                    try:
                        setattr(wrapped, name, value)
                        return
                    except (AttributeError, TypeError):
                        pass
            object.__setattr__(self, name, value)
            return
        return _FastSlowAttribute("__setattr__").__get__(self, type(self))(
            name, value
        )


class _FinalProxy(_FastSlowProxy):
    """
    Proxy type for a pair of fast and slow "final" types for which
    there is a known conversion from fast to slow, and vice-versa.
    The conversion between fast and slow types is done using
    user-provided conversion functions.

    Do not attempt to use this class directly. Instead, use
    `make_final_proxy_type` to create subtypes.
    """

    @classmethod
    def _fsproxy_wrap(cls, value, func):
        """Default mechanism to wrap a value in a proxy type

        Parameters
        ----------
        cls
            The proxy type
        value
            The value to wrap up
        func
            The function called that constructed value

        Returns
        -------
        A new proxied object

        Notes
        -----
        _FinalProxy subclasses can override this classmethod if they
        need particular behaviour when wrapped up.
        """
        # TODO: Replace the if-elif-else using singledispatch helper function
        base_class = _get_proxy_base_class(cls)
        if base_class is object:
            proxy = base_class.__new__(cls)
        elif base_class is ProxyNDarrayBase:
            proxy = base_class.__new__(cls, value)
        elif base_class is datetime.datetime:
            proxy = base_class.__new__(
                cls,
                year=value.year,
                month=value.month,
                day=value.day,
                hour=value.hour,
                minute=value.minute,
                second=value.second,
                microsecond=value.microsecond,
                tzinfo=value.tzinfo,
                fold=value.fold,
            )
        elif base_class is datetime.timedelta:
            proxy = base_class.__new__(
                cls,
                days=value.days,
                seconds=value.seconds,
                microseconds=value.microseconds,
            )
        else:
            raise TypeError(
                f"Cannot create an proxy instance of {cls.__name__} using base class {base_class.__name__}. "
                f"Expected either 'object' or another type in 'PROXY_BASE_CLASSES'"
            )
        proxy._fsproxy_wrapped = value
        return proxy

    def __reduce__(self):
        """
        In conjunction with `__proxy_setstate__`, this effectively enables
        proxy types to be pickled and unpickled by pickling and unpickling
        the underlying wrapped types.
        """
        # Need a local import to avoid circular import issues
        from .module_accelerator import disable_module_accelerator

        with disable_module_accelerator():
            pickled_wrapped_obj = pickle.dumps(self._fsproxy_wrapped)
        return (_PickleConstructor(type(self)), (), pickled_wrapped_obj)

    def __setstate__(self, state):
        # Need a local import to avoid circular import issues
        from .module_accelerator import disable_module_accelerator

        if isinstance(state, bytes):
            # State produced by ``__reduce__`` above.
            with disable_module_accelerator():
                unpickled_wrapped_obj = pickle.loads(state)
            self._fsproxy_wrapped = unpickled_wrapped_obj
        else:
            # Native state from the slow class. Reached when raw
            # ``pickle.load`` resolves a reference to one of our proxy
            # classes via ordinary attribute lookup (no
            # ``disable_module_accelerator`` shielding) and then drives
            # ``__setstate__`` with whatever the slow class's own
            # ``__reduce__`` produced (e.g. dict for CategoricalDtype,
            # tuple for PeriodArray). The state may itself contain proxy
            # objects, so unwrap them to the underlying slow values
            # before handing the state to the slow class.
            with disable_module_accelerator():
                slow_state = _slow_arg(state)
                slow_type = type(self)._fsproxy_slow_type
                wrapped = slow_type.__new__(slow_type)
                if hasattr(wrapped, "__setstate__"):
                    wrapped.__setstate__(slow_state)
                elif isinstance(slow_state, dict):
                    wrapped.__dict__.update(slow_state)
                else:
                    raise TypeError(
                        f"Cannot restore proxy {type(self).__name__} from "
                        f"state of type {type(slow_state).__name__}"
                    )
            self._fsproxy_wrapped = wrapped


class _IntermediateProxy(_FastSlowProxy):
    """
    Proxy type for a pair of "intermediate" types that appear as
    intermediate values when invoking operations on "final" types.
    The conversion between fast and slow types is done by keeping
    track of the sequence of operations that created the wrapped
    object, and "playing back" that sequence starting from the "slow"
    version of the originating _FinalProxy.

    Do not attempt to use this class directly. Instead, use
    `make_intermediate_proxy_type` to create subtypes.
    """

    _method_chain: tuple[Callable, tuple, dict]
    # The proxy objects this intermediate was derived from and a snapshot of
    # their wrapped objects' identities. Both are always (re)assigned together
    # in ``_fsproxy_wrap`` / ``__setstate__`` / ``_fsproxy_record_parent_ids``;
    # the empty class-level defaults just guarantee the attributes always exist
    # (e.g. on a bare ``object.__new__`` instance before it is populated).
    _fsproxy_parents: Sequence[_FastSlowProxy] = ()
    _fsproxy_parent_ids: Sequence[int] = ()

    @classmethod
    def _fsproxy_wrap(
        cls,
        obj: Any,
        method_chain: tuple[Callable, tuple, dict],
    ):
        """
        Parameters
        ----------
        obj: The object to wrap
        method_chain: A tuple of the form (func, args, kwargs) where
            `func` is the function that was called to create `obj`,
            and `args` and `kwargs` are the arguments that were passed
            to `func`.
        """
        proxy = object.__new__(cls)
        proxy._fsproxy_wrapped = obj
        proxy._method_chain = method_chain
        proxy._fsproxy_parents = _collect_parent_proxies(method_chain)
        proxy._fsproxy_record_parent_ids()
        return proxy

    def _fsproxy_record_parent_ids(self) -> None:
        """Snapshot the identity of each parent proxy's wrapped object.

        ``_fsproxy_parent_ids[i]`` is the recorded identity of
        ``_fsproxy_parents[i]``: the two lists are built from the same source
        in the same order, so they stay positionally aligned.
        """
        self._fsproxy_parent_ids = [
            id(parent._fsproxy_wrapped) for parent in self._fsproxy_parents
        ]

    def _fsproxy_parents_changed(self) -> bool:
        """
        Return True if any parent proxy's wrapped object has been replaced
        (e.g. mutated in place or transferred between fast and slow) since this
        intermediate was last derived from it. When that happens the cached
        wrapped object is stale and must be re-derived from the live parents.
        """
        parents = self._fsproxy_parents
        if not parents:
            return False
        # ``_fsproxy_parent_ids`` was recorded from ``_fsproxy_parents`` in the
        # same order (see ``_fsproxy_record_parent_ids``), so the ``zip`` pairs
        # each parent with its own prior identity. The comparison is
        # deliberately positional rather than set-based: we must detect whether
        # a *given* parent's object was replaced, which a set of identities
        # would lose (and it would not save any re-derivation, since parents are
        # already de-duplicated when collected).
        return any(
            id(parent._fsproxy_wrapped) != old_id
            for parent, old_id in zip(
                parents, self._fsproxy_parent_ids, strict=True
            )
        )

    @nvtx.annotate(
        "COPY_SLOW_TO_FAST",
        color=_CUDF_PANDAS_NVTX_COLORS["COPY_SLOW_TO_FAST"],
        domain="cudf_pandas",
    )
    def _fsproxy_slow_to_fast(self) -> Any:
        func, args, kwargs = self._method_chain
        args, kwargs = _fast_arg(args), _fast_arg(kwargs)
        return func(*args, **kwargs)

    @nvtx.annotate(
        "COPY_FAST_TO_SLOW",
        color=_CUDF_PANDAS_NVTX_COLORS["COPY_FAST_TO_SLOW"],
        domain="cudf_pandas",
    )
    def _fsproxy_fast_to_slow(self) -> Any:
        func, args, kwargs = self._method_chain
        args, kwargs = _slow_arg(args), _slow_arg(kwargs)
        return func(*args, **kwargs)

    def __reduce__(self):
        """
        In conjunction with `__proxy_setstate__`, this effectively enables
        proxy types to be pickled and unpickled by pickling and unpickling
        the underlying wrapped types.
        """
        # Need a local import to avoid circular import issues
        from .module_accelerator import disable_module_accelerator

        with disable_module_accelerator():
            pickled_wrapped_obj = pickle.dumps(self._fsproxy_wrapped)
        pickled_method_chain = pickle.dumps(self._method_chain)
        return (
            _PickleConstructor(type(self)),
            (),
            (pickled_wrapped_obj, pickled_method_chain),
        )

    def __setstate__(self, state):
        # Need a local import to avoid circular import issues
        from .module_accelerator import disable_module_accelerator

        with disable_module_accelerator():
            unpickled_wrapped_obj = pickle.loads(state[0])
        unpickled_method_chain = pickle.loads(state[1])
        self._fsproxy_wrapped = unpickled_wrapped_obj
        self._method_chain = unpickled_method_chain
        self._fsproxy_parents = _collect_parent_proxies(unpickled_method_chain)
        self._fsproxy_record_parent_ids()


class _CallableProxyMixin:
    """
    Mixin class that implements __call__ for fast-slow proxies.
    """

    # For wrapped callables isinstance(self, FunctionType) should return True
    __class__ = types.FunctionType  # type: ignore[assignment]

    def __call__(self, *args, **kwargs) -> Any:
        result, _ = _fast_slow_function_call(
            # We cannot directly call self here because we need it to be
            # converted into either the fast or slow object (by
            # _fast_slow_function_call) to avoid infinite recursion.
            # TODO: When Python 3.11 is the minimum supported Python version
            # this can use operator.call
            call_operator,
            self.get_transfer_blocking(),  # type: ignore[attr-defined]
            self,
            args,
            kwargs,
        )
        return result

    @property
    def __code__(self):
        return self._fsproxy_slow.__code__

    @property
    def __defaults__(self):
        return self._fsproxy_slow.__defaults__

    @property
    def __kwdefaults__(self):
        return self._fsproxy_slow.__kwdefaults__


class _FunctionProxy(_CallableProxyMixin):
    """
    Proxy for a pair of fast and slow functions.
    """

    __name__: str

    def __init__(
        self,
        fast: Callable | _Unusable,
        slow: Callable,
        *,
        assigned=None,
        updated=None,
        _fsproxy_transfer_block: _BlockState | None = None,
    ):
        self._fsproxy_fast = fast
        self._fsproxy_slow = slow
        self._fsproxy_transfer_block = _fsproxy_transfer_block
        if assigned is None:
            assigned = functools.WRAPPER_ASSIGNMENTS
        if updated is None:
            updated = functools.WRAPPER_UPDATES
        functools.update_wrapper(
            self,
            slow,
            assigned=assigned,
            updated=updated,
        )

    def __reduce__(self):
        """
        In conjunction with `__proxy_setstate__`, this effectively enables
        proxy types to be pickled and unpickled by pickling and unpickling
        the underlying wrapped types.
        """
        # Need a local import to avoid circular import issues
        from .module_accelerator import disable_module_accelerator

        with disable_module_accelerator():
            pickled_fast = pickle.dumps(self._fsproxy_fast)
            pickled_slow = pickle.dumps(self._fsproxy_slow)
        return (
            _PickleConstructor(type(self)),
            (),
            (pickled_fast, pickled_slow),
        )

    def __setstate__(self, state):
        # Need a local import to avoid circular import issues
        from .module_accelerator import disable_module_accelerator

        with disable_module_accelerator():
            unpickled_fast = pickle.loads(state[0])
            unpickled_slow = pickle.loads(state[1])
        self._fsproxy_fast = unpickled_fast
        self._fsproxy_slow = unpickled_slow

    def get_transfer_blocking(self):
        """Get current instance-level transfer blocking flag."""
        return self._fsproxy_transfer_block


def is_bound_method(obj):
    return inspect.ismethod(obj) and not inspect.isfunction(obj)


def is_function(obj):
    return inspect.isfunction(obj) or isinstance(obj, types.FunctionType)


class _FastSlowAttribute:
    """
    A descriptor type used to define attributes of fast-slow proxies.
    """

    _attr: Any

    def __init__(self, name: str, *, private: bool = False):
        self._name = name
        self._private = private
        self._attr = None
        self._doc = None
        self._dir = None

    def __get__(self, instance, owner) -> Any:
        from .module_accelerator import disable_module_accelerator

        if self._attr is None:
            if self._private:
                fast_attr = _Unusable()
            else:
                fast_attr = getattr(
                    owner._fsproxy_fast, self._name, _Unusable()
                )

            try:
                slow_attr = getattr(owner._fsproxy_slow, self._name)
            except AttributeError as e:
                if instance is not None:
                    return _maybe_wrap_result(
                        getattr(instance._fsproxy_slow, self._name),
                        None,  # type: ignore[arg-type]
                    )
                else:
                    raise e

            if _is_function_or_method(slow_attr):
                self._attr = _MethodProxy(
                    fast_attr,
                    slow_attr,
                    _fsproxy_transfer_block=instance.get_transfer_blocking()
                    if instance is not None
                    else None,
                )
            else:
                # for anything else, use a fast-slow attribute:
                self._attr, _ = _fast_slow_function_call(
                    getattr,
                    None,
                    owner,
                    self._name,
                )

                if isinstance(
                    self._attr, (property, functools.cached_property)
                ):
                    with disable_module_accelerator():
                        self._attr.__doc__ = inspect.getdoc(slow_attr)

        if instance is not None:
            if isinstance(self._attr, _MethodProxy):
                if is_bound_method(self._attr._fsproxy_slow):
                    return self._attr
                else:
                    return types.MethodType(self._attr, instance)
            else:
                if self._private:
                    return _maybe_wrap_result(
                        getattr(instance._fsproxy_slow, self._name),
                        None,  # type: ignore[arg-type]
                    )
                return _fast_slow_function_call(
                    getattr,
                    None,
                    instance,
                    self._name,
                )[0]

        return self._attr

    def __set__(self, instance, value) -> None:
        # Implementing ``__set__`` makes this a data descriptor, which takes
        # precedence over an entry of the same name in ``instance.__dict__``.
        # Without it, ``object.__setattr__`` (called from the proxy's
        # ``__setattr__`` for underscore-prefixed names) would write to the
        # instance dict and shadow the descriptor on reads, so private attrs
        # declared in ``additional_attributes`` would never reach the slow
        # object.
        if self._private:
            # Forward to the slow object so pandas-internal state
            # (e.g. the ``_readonly`` flag consulted by ``__setitem__``) stays
            # in sync with what the user assigned on the proxy.
            object.__setattr__(instance._fsproxy_slow, self._name, value)
        else:
            # Preserve the pre-data-descriptor behavior for non-private
            # entries by writing to the instance dict directly (assigning via
            # ``object.__setattr__`` would recurse through this descriptor).
            instance.__dict__[self._name] = value


class _MethodProxy(_FunctionProxy):
    def __init__(self, fast, slow, _fsproxy_transfer_block=None):
        super().__init__(
            fast,
            slow,
            updated=functools.WRAPPER_UPDATES,
            assigned=(
                tuple(filter(lambda x: x != "__name__", _WRAPPER_ASSIGNMENTS))
            ),
            _fsproxy_transfer_block=_fsproxy_transfer_block,
        )

    def __dir__(self):
        return self._fsproxy_slow.__dir__()

    @property
    def __doc__(self):
        return self._fsproxy_slow.__doc__

    @property
    def __func__(self):
        return self._fsproxy_slow.__func__

    @property
    def __name__(self):
        return self._fsproxy_slow.__name__

    @__name__.setter
    def __name__(self, value):
        try:
            setattr(self._fsproxy_fast, "__name__", value)
        except AttributeError:
            pass
        setattr(self._fsproxy_slow, "__name__", value)

    @property
    def _customqualname(self):
        return self._fsproxy_slow.__qualname__


def _assert_fast_slow_eq(left, right):
    if _is_final_type(type(left)) or type(left) in NUMPY_TYPES:
        assert_eq(left, right)


class FallbackError(Exception):
    """Raises when fallback occurs"""

    pass


class OOMFallbackError(FallbackError):
    """Raises when cuDF produces a MemoryError or an rmm.RMMError"""

    pass


class NotImplementedFallbackError(FallbackError):
    """Raises cuDF produces a NotImplementedError"""

    pass


class AttributeFallbackError(FallbackError):
    """Raises when cuDF produces an AttributeError"""

    pass


class TypeFallbackError(FallbackError):
    """Raises when cuDF produces a TypeError"""

    pass


def _raise_fallback_error(err, name):
    """Raises a fallback error."""
    err_message = f"Falling back to the slow path. The exception was {err}. \
        The function called was {name}."
    exception_map = {
        (RMMError, MemoryError): OOMFallbackError,
        NotImplementedError: NotImplementedFallbackError,
        AttributeError: AttributeFallbackError,
        TypeError: TypeFallbackError,
    }
    for err_type, fallback_err_type in exception_map.items():
        if isinstance(err, err_type):
            raise fallback_err_type(err_message) from err
    raise FallbackError(err_message) from err


def _fast_function_call():
    """
    Placeholder fast function for pytest profiling purposes.
    """
    return None


def _slow_function_call():
    """
    Placeholder slow function for pytest profiling purposes.
    """
    return None


def _fast_slow_function_call(
    func: Callable,
    transfer_block: _BlockState | None = None,
    /,
    *args,
    **kwargs,
) -> Any:
    """
    Call `func` with all `args` and `kwargs` converted to their
    respective fast type. If that fails, call `func` with all
    `args` and `kwargs` converted to their slow type.

    Wrap the result in a fast-slow proxy if it is a type we know how
    to wrap.
    """
    from .module_accelerator import disable_module_accelerator

    fast = False
    block_transfer_to_fast = False
    try:
        if transfer_block is _BlockState.TO_FAST:
            raise Exception("Forcing slow path due to transfer blocking")
        with nvtx.annotate(
            "EXECUTE_FAST",
            color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_FAST"],
            domain="cudf_pandas",
        ):
            fast_args, fast_kwargs = _fast_arg(args), _fast_arg(kwargs)
            result = func(*fast_args, **fast_kwargs)
            if result is NotImplemented:
                # try slow path
                raise Exception()
            fast = True
            _fast_function_call()
            if _env_get_bool("CUDF_PANDAS_DEBUGGING", False):
                try:
                    with nvtx.annotate(
                        "EXECUTE_SLOW_DEBUG",
                        color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
                        domain="cudf_pandas",
                    ):
                        slow_args, slow_kwargs = (
                            _slow_arg(args),
                            _slow_arg(kwargs),
                        )
                        with disable_module_accelerator():
                            slow_result = func(*slow_args, **slow_kwargs)
                except Exception as e:
                    warnings.warn(
                        "The result from pandas could not be computed. "
                        f"The exception was {e}."
                    )
                else:
                    try:
                        _assert_fast_slow_eq(result, slow_result)
                    except AssertionError as e:
                        warnings.warn(
                            "The results from cudf and pandas were different. "
                            f"The exception was {e}."
                        )
                    except Exception as e:
                        warnings.warn(
                            "Pandas debugging mode failed. "
                            f"The exception was {e}."
                        )
    except Exception as err:
        if type(err) is cudf.errors.MixedTypeError:
            block_transfer_to_fast = True
        with nvtx.annotate(
            "EXECUTE_SLOW",
            color=_CUDF_PANDAS_NVTX_COLORS["EXECUTE_SLOW"],
            domain="cudf_pandas",
        ):
            slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
            if _env_get_bool("CUDF_PANDAS_FAIL_ON_FALLBACK", False):
                _raise_fallback_error(err, slow_args[0].__name__)
            if _env_get_bool("LOG_FAST_FALLBACK", False):
                from ._logger import log_fallback

                log_fallback(slow_args, slow_kwargs, err)
            _slow_function_call()
            with disable_module_accelerator():
                result = func(*slow_args, **slow_kwargs)
    result = _maybe_wrap_result(result, func, *args, **kwargs)
    if block_transfer_to_fast and isinstance(result, _FastSlowProxy):
        result.force_state(_State.SLOW)
    return result, fast


def _transform_arg(
    arg: Any,
    attribute_name: Literal["_fsproxy_slow", "_fsproxy_fast"],
    seen: set[int],
) -> Any:
    """
    Transform "arg" into its corresponding slow (or fast) type.
    """
    import numpy as np

    if isinstance(arg, (_FastSlowProxy, _FastSlowProxyMeta, _FunctionProxy)):
        typ = getattr(arg, attribute_name)
        if typ is _Unusable:
            raise Exception("Cannot transform _Unusable")
        return typ
    elif isinstance(arg, types.ModuleType) and attribute_name in arg.__dict__:
        return arg.__dict__[attribute_name]
    elif isinstance(arg, list):
        transformed_list = [
            _transform_arg(a, attribute_name, seen) for a in arg
        ]
        if all(
            new is old for new, old in zip(transformed_list, arg, strict=True)
        ):
            # No element needed transforming: return the original list (as
            # the object-ndarray branch below already does) to preserve
            # identity. A user function may close over a mutable container
            # and mutate it for its side effects (e.g.
            # ``names.append(group.name)`` inside groupby.apply); copying
            # here would silently discard those side effects on both the
            # fast attempt and the pandas fallback, and also defeats
            # _replace_closurevars' unchanged-check so the original
            # function object is never passed through.
            return arg
        # Pass an iterator rather than the materialized list so list
        # subclasses see the same non-list iterable constructor argument
        # they always received here.
        return type(arg)(iter(transformed_list))
    elif isinstance(arg, tuple):
        # This attempts to handle arbitrary subclasses of tuple by
        # assuming that if you've subclassed tuple with some special
        # behaviour you'll also make the object pickleable by
        # implementing the custom pickle protocol interface (either
        # __getnewargs_ex__ or __getnewargs__). Perhaps this should
        # use __reduce_ex__ instead...
        if type(arg) is tuple:
            # Must come first to avoid infinite recursion
            if (
                len(arg) > 0
                and isinstance(arg[0], _MethodProxy)
                and arg[0]._customqualname in _SPECIAL_FUNCTIONS_ARGS_MAP
            ):
                indices_map = _SPECIAL_FUNCTIONS_ARGS_MAP[
                    arg[0]._customqualname
                ]
                method_proxy, original_args, original_kwargs = arg

                original_args = tuple(
                    _transform_arg(a, "_fsproxy_slow", seen)
                    if i - 1 in indices_map
                    else _transform_arg(a, attribute_name, seen)
                    for i, a in enumerate(original_args)
                )
                original_kwargs = _transform_arg(
                    original_kwargs, attribute_name, seen
                )
                return tuple(
                    (
                        _transform_arg(method_proxy, attribute_name, seen),
                        original_args,
                        original_kwargs,
                    )
                )
            else:
                transformed_tuple = [
                    _transform_arg(a, attribute_name, seen) for a in arg
                ]
                if all(
                    new is old
                    for new, old in zip(transformed_tuple, arg, strict=True)
                ):
                    # No element needed transforming: return the original
                    # tuple (immutable, so this is safe) so containers
                    # enclosing it also keep their identity (see the list
                    # branch above).
                    return arg
                return tuple(transformed_tuple)
        elif hasattr(arg, "__getnewargs_ex__"):
            # Partial implementation of to reconstruct with
            # transformed pieces
            # This handles scipy._lib._bunch._make_tuple_bunch
            args, kwargs = (
                _transform_arg(a, attribute_name, seen)
                for a in arg.__getnewargs_ex__()
            )
            obj = type(arg).__new__(type(arg), *args, **kwargs)
            if hasattr(obj, "__setstate__"):
                raise NotImplementedError(
                    "Transforming tuple-like with __getnewargs_ex__ and "
                    "__setstate__ not implemented"
                )
            if not hasattr(obj, "__dict__") and kwargs:
                raise NotImplementedError(
                    "Transforming tuple-like with kwargs from "
                    "__getnewargs_ex__ and no __dict__ not implemented"
                )
            obj.__dict__.update(kwargs)
            return obj
        elif hasattr(arg, "__getnewargs__"):
            # This handles namedtuple, and would catch tuple if we
            # didn't handle it above.
            args = _transform_arg(arg.__getnewargs__(), attribute_name, seen)
            return type(arg).__new__(type(arg), *args)
        else:
            # Hope we can just call the constructor with transformed entries.
            return type(arg)(
                _transform_arg(a, attribute_name, seen) for a in args
            )
    elif isinstance(arg, dict):
        transformed_dict = {
            _transform_arg(k, attribute_name, seen): _transform_arg(
                a, attribute_name, seen
            )
            for k, a in arg.items()
        }
        if len(transformed_dict) == len(arg) and all(
            new_k is old_k and new_v is old_v
            for (new_k, new_v), (old_k, old_v) in zip(
                transformed_dict.items(), arg.items(), strict=True
            )
        ):
            # see the list branch above: preserve identity when unchanged
            return arg
        return transformed_dict
    elif isinstance(arg, np.ndarray) and arg.dtype == "O":
        transformed: list[Any] = [
            _transform_arg(a, attribute_name, seen) for a in arg.flat
        ]
        if all(
            new is old for new, old in zip(transformed, arg.flat, strict=True)
        ):
            # No element needed transforming: return the original array
            # (as we already do for non-object ndarrays below) to
            # preserve buffer identity. Rebuilding an equivalent copy
            # breaks aliasing checks such as
            # ``np.may_share_memory(np.asarray(x), x)`` that numpy uses
            # (e.g. in ``Generator.permutation``) to decide whether to
            # defensively copy before an in-place shuffle; the views
            # pandas returns under copy-on-write are read-only, so
            # skipping that copy raises "ValueError: array is read-only".
            return arg
        # Keep the same memory layout as arg (the default is C_CONTIGUOUS)
        if arg.flags["F_CONTIGUOUS"] and not arg.flags["C_CONTIGUOUS"]:
            order = "F"
        else:
            order = "C"
        result = np.empty(  # type: ignore[call-overload]
            int(np.prod(arg.shape)), dtype=np.object_, order=order
        )
        result[...] = transformed
        return result.reshape(arg.shape)
    elif isinstance(arg, (types.GeneratorType, map, filter, zip, enumerate)):
        # "Pure" (resource-free) iterators. On the fast path we force a
        # fallback because they are consumable (see below). On the slow path we
        # transform the elements lazily into a new generator -- a bare iterator
        # would otherwise hand wrapped proxies straight to the slow library,
        # which can lose type information (e.g. ``MultiIndex.from_product`` of a
        # ``map`` over categoricals dropping the ``category`` dtype). We yield a
        # generator (rather than materializing a ``list``) so the slow library
        # still sees a bare iterator: pandas treats iterators differently from
        # lists in places (e.g. ``is_nested_list_like`` returns ``False`` for an
        # iterator, and ``Index.get_loc`` reports the iterator in its error
        # message), and materializing would silently change that behavior.
        if attribute_name == "_fsproxy_fast":
            raise Exception()
        return (_transform_arg(a, attribute_name, seen) for a in arg)
    elif isinstance(arg, Iterator) and attribute_name == "_fsproxy_fast":
        # Other consumable objects such as generators or IOBase objects, which
        # we don't want unavailable to the slow path in case of fallback. So,
        # we raise here and ensure the slow path is taken (and the object is
        # left intact for it):
        raise Exception()
    elif isinstance(arg, types.FunctionType):
        if id(arg) in seen:
            # `arg` is mutually recursive with another function.  We
            # can't handle these cases yet:
            return arg
        seen.add(id(arg))
        return _replace_closurevars(arg, attribute_name, seen)
    else:
        return arg


def _collect_parent_proxies(
    method_chain: tuple[Callable, tuple, dict],
) -> list[_FastSlowProxy]:
    """
    Collect the ``_FastSlowProxy`` objects referenced by an intermediate
    proxy's method chain.

    An ``_IntermediateProxy`` (e.g. a groupby, rolling, or accessor object)
    is created by calling ``func(*args, **kwargs)`` on one or more "parent"
    proxies (the originating frame/series and any proxy arguments such as a
    grouping key). We record those parents so that, if one of them is later
    mutated or replaced in place (e.g. ``df["new"] = ...`` after
    ``gp = df.groupby(...)``), the intermediate can re-derive itself from the
    live parent instead of returning a stale snapshot. Method/function
    proxies are intentionally skipped: they carry the operation, not the data,
    and consumable intermediates (e.g. file readers) have no proxy parents at
    all, so they are never re-derived.

    The traversal is intentionally iterative (an explicit stack) rather than a
    nested recursive helper. A recursive closure would hold a cell referencing
    itself, forming a reference cycle that keeps the closure -- and everything
    it captured, including the collected parent proxies -- alive until the
    cyclic garbage collector runs. That delayed collection breaks code that
    relies on prompt reference-counted teardown of a proxy (e.g. pandas'
    ``Series.str``-accessor circular-reference test).
    """
    import numpy as np

    _, args, kwargs = method_chain
    parents: list[_FastSlowProxy] = []
    seen: set[int] = set()
    stack: list[Any] = [args, kwargs]

    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, _FastSlowProxy):
            parents.append(obj)
        elif isinstance(obj, (tuple, list)):
            stack.extend(obj)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                stack.append(key)
                stack.append(value)
        elif isinstance(obj, np.ndarray) and obj.dtype == "O":
            stack.extend(obj.flat)

    return parents


def _fast_arg(arg: Any) -> Any:
    """
    Transform "arg" into its corresponding fast type.
    """
    seen: set[int] = set()
    return _transform_arg(arg, "_fsproxy_fast", seen)


def _slow_arg(arg: Any) -> Any:
    """
    Transform "arg" into its corresponding slow type.
    """
    seen: set[int] = set()
    return _transform_arg(arg, "_fsproxy_slow", seen)


def _maybe_wrap_result(result: Any, func: Callable, /, *args, **kwargs) -> Any:
    """
    Wraps "result" in a fast-slow proxy if is a "proxiable" object.
    """
    if isinstance(result, (int, str, float, bool, type(None))):
        return result
    elif _is_final_type(result):
        typ = get_final_type_map()[type(result)]
        return typ._fsproxy_wrap(result, func)
    elif _is_intermediate_type(result):
        typ = get_intermediate_type_map()[type(result)]
        return typ._fsproxy_wrap(result, method_chain=(func, args, kwargs))
    elif _is_final_class(result):
        return get_final_type_map()[result]
    elif isinstance(result, list):
        return type(result)(
            [
                _maybe_wrap_result(r, operator.getitem, result, i)
                for i, r in enumerate(result)
            ]
        )
    elif isinstance(result, tuple):
        wrapped = (
            _maybe_wrap_result(r, operator.getitem, result, i)
            for i, r in enumerate(result)
        )
        if hasattr(result, "_make"):
            # namedtuple
            return type(result)._make(wrapped)
        else:
            return type(result)(wrapped)
    elif isinstance(result, Iterator):
        return (_maybe_wrap_result(r, lambda x: x, r) for r in result)
    else:
        return result


def _is_final_type(result: Any) -> bool:
    return type(result) in get_final_type_map()


def _is_final_class(result: Any) -> bool:
    if not isinstance(result, type):
        return False
    return result in get_final_type_map()


def _is_intermediate_type(result: Any) -> bool:
    return type(result) in get_intermediate_type_map()


def _is_function_or_method(obj: Any) -> bool:
    res = isinstance(
        obj,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.WrapperDescriptorType,
            types.MethodWrapperType,
            types.MethodDescriptorType,
            types.BuiltinMethodType,
        ),
    )
    if not res:
        try:
            return "cython_function_or_method" in str(type(obj))
        except Exception:
            return False
    return res


def _replace_closurevars(
    f: types.FunctionType,
    attribute_name: Literal["_fsproxy_slow", "_fsproxy_fast"],
    seen: set[int],
) -> Callable[..., Any]:
    """
    Return a copy of `f` with its closure variables replaced with
    their corresponding slow (or fast) types.
    """
    if f.__closure__:
        # GH #254: If empty cells are present - which can happen in
        # situations like when `f` is a method that invokes the
        # "empty" `super()` - the call to `getclosurevars` below will
        # fail.  For now, we just return `f` in this case.  If needed,
        # we can consider populating empty cells with a placeholder
        # value to allow the call to `getclosurevars` to succeed.
        if any(c == types.CellType() for c in f.__closure__):
            return f

    f_nonlocals, f_globals, _, _ = inspect.getclosurevars(f)

    g_globals = _transform_arg(f_globals, attribute_name, seen)
    g_nonlocals = _transform_arg(f_nonlocals, attribute_name, seen)

    # if none of the globals/nonlocals were transformed, we
    # can just return f:
    if all(f_globals[k] is g_globals[k] for k in f_globals) and all(
        g_nonlocals[k] is f_nonlocals[k] for k in f_nonlocals
    ):
        return f

    g_closure = tuple(types.CellType(val) for val in g_nonlocals.values())

    # https://github.com/rapidsai/cudf/issues/15548
    new_g_globals = f.__globals__.copy()
    new_g_globals.update(g_globals)

    g = types.FunctionType(
        f.__code__,
        new_g_globals,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=g_closure,
    )
    return functools.update_wrapper(
        g,
        f,
        assigned=(*functools.WRAPPER_ASSIGNMENTS, "__kwdefaults__"),
    )


def is_proxy_object(obj: Any) -> bool:
    """Determine if an object is proxy object

    Parameters
    ----------
    obj : object
        Any python object.

    """
    if _FastSlowProxyMeta in type(type(obj)).__mro__:
        return True
    return False


def _get_proxy_base_class(cls):
    """Returns the proxy base class if one exists"""
    for proxy_class in PROXY_BASE_CLASSES:
        if proxy_class in cls.__mro__:
            return proxy_class
    return object


def as_proxy_object(obj: Any) -> Any:
    """
    Wraps a cudf or pandas object in a proxy object if applicable.

    There will be no memory transfer, i.e., GPU objects stay on GPU and
    CPU objects stay on CPU. The object will be wrapped in a
    proxy object. This is useful for ensuring that the object is
    compatible with the fast-slow proxy system.

    Parameters
    ----------
    obj : Any
        The object to wrap.

    Returns
    -------
    Any
        The wrapped proxy object if applicable, otherwise the original object.
    """
    if _is_final_type(obj):
        typ = get_final_type_map()[type(obj)]
        return typ._fsproxy_wrap(obj, None)
    return obj


def is_proxy_instance(obj, type):
    return is_proxy_object(obj) and obj.__class__.__name__ == type.__name__


PROXY_BASE_CLASSES: set[type] = {
    ProxyNDarrayBase,
    datetime.datetime,
    datetime.timedelta,
}


NUMPY_TYPES: set[type[np.generic]] = set(np.sctypeDict.values())


_SPECIAL_METHODS: set[str] = {
    "__abs__",
    "__add__",
    "__and__",
    "__bool__",
    "__call__",
    "__complex__",
    "__contains__",
    "__copy__",
    "__dataframe__",
    "__deepcopy__",
    "__delitem__",
    "__delslice__",
    "__divmod__",
    "__enter__",
    "__eq__",
    "__exit__",
    "__float__",
    "__floordiv__",
    "__format__",
    "__ge__",
    "__getitem__",
    "__getslice__",
    "__gt__",
    # Added on a per-proxy basis
    # https://github.com/rapidsai/xdf/pull/306#pullrequestreview-1636155428
    # "__hash__",
    "__iadd__",
    "__iand__",
    "__iconcat__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__int__",
    "__invert__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__iter__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__len__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__next__",
    "__or__",
    "__pos__",
    "__pow__",
    "__radd__",
    "__rand__",
    "__rdivmod__",
    "__repr__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmatmul__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__round__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__setitem__",
    "__setslice__",
    "__str__",
    "__sub__",
    "__truediv__",
    "__xor__",
}
