# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cupy
import cupy._core.flags
import numpy
from numpy._core.multiarray import flagsobj as _numpy_flagsobj

from cudf.options import _env_get_bool

from ..fast_slow_proxy import (
    _fast_arg,
    _fast_slow_function_call,
    _FastSlowAttribute,
    _maybe_wrap_result,
    _raise_fallback_error,
    _slow_arg,
    is_proxy_object,
    make_final_proxy_type,
    make_intermediate_proxy_type,
)
from ..proxy_base import ProxyNDarrayBase
from .common import (
    array_interface,
    array_method,
    cuda_array_interface,
    custom_iter,
)

# https://docs.cupy.dev/en/stable/reference/creation.html
_CONSTRUCTORS = frozenset(
    [
        cupy.empty,
        cupy.empty_like,
        cupy.eye,
        cupy.identity,
        cupy.ones,
        cupy.ones_like,
        cupy.zeros,
        cupy.zeros_like,
        cupy.full,
        cupy.full_like,
        cupy.array,
        cupy.asarray,
        cupy.asanyarray,
        cupy.ascontiguousarray,
        cupy.copy,
        cupy.frombuffer,
        cupy.fromfile,
        cupy.fromfunction,
        cupy.fromiter,
        cupy.fromstring,
        cupy.loadtxt,
        cupy.arange,
        cupy.linspace,
        cupy.logspace,
        cupy.meshgrid,
        cupy.diag,
        cupy.diagflat,
        cupy.tri,
        cupy.tril,
        cupy.triu,
        cupy.vander,
    ]
)


def wrap_ndarray(cls, arr: cupy.ndarray | numpy.ndarray, constructor):
    """Wrap an ndarray in a proxy type

    Parameters
    ----------
    cls
        Proxy type for ndarray
    arr
        Concrete result ndarray (cupy or numpy)
    constructor
        Function that was called to construct the concrete array, used
        to check against a denylist to avoid unwrapping.

    Returns
    -------
    The scalar .item() wrapped in its numpy dtype if arr is a
    zero-dimensional cupy array (and wasn't just constructed as such),
    a new proxy type otherwise.

    Notes
    -----
    Axis-reducing operations in numpy return scalar objects but
    zero-dimensional arrays in cupy. This confuses downstream
    libraries when they get a fast (device-based) zero-dim array when
    they were expecting a scalar. To avoid this, if the provided array
    is a cupy array, and its shape is zero, unwrap it.
    """
    if (
        isinstance(arr, cupy.ndarray)
        and arr.shape == ()
        and constructor not in _CONSTRUCTORS
    ):
        value = arr.item()
        if arr.dtype.kind in "mM":
            unit, _ = numpy.datetime_data(arr.dtype)
            return arr.dtype.type(value, unit)
        return arr.dtype.type(value)
    else:
        # Note, this super call means that the constructed ndarray
        # class cannot be subclassed (because then super(cls,
        # cls)._fsproxy_wrap produces an infinite loop). Really this
        # should be super(ndarray, cls), but we don't have access to
        # the ndarray type until after we need to pass this function
        # in. So it works for now since without subclassing,
        # super(ndarray, cls) == super(ndarray, ndarray) == super(cls,
        # cls)
        return super(cls, cls)._fsproxy_wrap(arr, constructor)


def _other_has_higher_priority(self, other) -> bool:
    self_priority = float(getattr(self, "__array_priority__", 0.0))
    try:
        other_priority = float(getattr(other, "__array_priority__", 0.0))
    except (TypeError, ValueError):
        return False
    return other_priority > self_priority


def ndarray__array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if method == "__call__" and len(inputs) > 1:
        for inp in inputs:
            if _other_has_higher_priority(self, inp):
                return NotImplemented
    result, _ = _fast_slow_function_call(
        getattr(ufunc, method),
        None,
        *inputs,
        **kwargs,
    )
    if isinstance(result, tuple):
        if is_proxy_object(result[0]) and isinstance(
            result[0]._fsproxy_wrapped, numpy.ndarray
        ):
            return tuple(numpy.asarray(x) for x in result)
    elif is_proxy_object(result) and isinstance(
        result._fsproxy_wrapped, numpy.ndarray
    ):
        return numpy.asarray(result)
    return result


def make_binary_op_method(op_func, reflected=False):
    if reflected:

        def method(self, other):
            return op_func(other, self)
    else:
        # Delegate to the other operand if it has higher priority
        def method(self, other):
            if _other_has_higher_priority(self, other):
                return NotImplemented
            return op_func(self, other)

    return method


def ndarray__reduce__(self):
    # As it stands the custom pickling logic used for all other
    # proxy types is incompatible with our proxy ndarray. The pickle
    # constructor we use to deserialize the other proxy types calls
    # object.__new__(type) which you cannot call on subclasses of
    # numpy arrays because the new array won't be created with numpy's
    # specific memory management logic. Therefore, we have to handle
    # serialization separately for proxy arrays.
    return (
        ndarray.__new__,
        (
            ndarray,
            self._fsproxy_wrapped,
        ),
    )


def _is_cupy_backed_and_non_datetime_array(x) -> bool:
    if is_proxy_object(x):
        x = x._fsproxy_wrapped
    return (
        isinstance(x, cupy.ndarray)
        and x.dtype is not None
        and x.dtype.kind not in ("M", "m")
    )


def ndarray__array_function__(self, func, types, args, kwargs):
    name = func.__name__
    try:
        cupy_func = getattr(cupy, name)
    except AttributeError:
        if getattr(func, "__module__", "").startswith("numpy.linalg"):
            cupy_func = getattr(cupy.linalg, name, None)
        else:
            cupy_func = None

    if cupy_func is not None and all(
        _is_cupy_backed_and_non_datetime_array(a) for a in args
    ):
        fast_args, fast_kwargs = _fast_arg(args), _fast_arg(kwargs)
        if name == "fft":
            cupy_func = cupy_func.fft
        try:
            res = cupy_func(*fast_args, **fast_kwargs)
        except Exception as err:
            slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
            if _env_get_bool("CUDF_PANDAS_FAIL_ON_FALLBACK", False):
                _raise_fallback_error(err, slow_args[0].__name__)
            res = func(*slow_args, **slow_kwargs)
        return _maybe_wrap_result(res, func, *args, **kwargs)

    slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
    return _maybe_wrap_result(
        func(*slow_args, **slow_kwargs), func, *args, **kwargs
    )


def _ndarray_fsproxy_fast_to_slow(self):
    from ..fast_slow_proxy import _State

    # Preserve view semantics when the proxy was originally wrapped
    # around a numpy view (``.base is not None``).  Without this, a
    # round-trip through cupy would convert the stored numpy view into
    # a cupy array and then back into a freshly-allocated numpy array
    # whose ``.base`` is None, breaking callers that rely on the view
    # relationship.
    if self._fsproxy_state is _State.FAST:
        original = getattr(self, "_fsproxy_original_slow", None)
        if original is not None:
            return original
        return cupy.ndarray.get(self._fsproxy_wrapped)
    return self._fsproxy_wrapped


ndarray = make_final_proxy_type(
    "ndarray",
    cupy.ndarray,
    numpy.ndarray,
    fast_to_slow=cupy.ndarray.get,
    slow_to_fast=cupy.asarray,
    bases=(ProxyNDarrayBase,),
    additional_attributes={
        "__array__": array_method,
        "__array_function__": ndarray__array_function__,
        # Note: __arrow_array__ is intentionally NOT defined here. The proxy
        # already exposes __array_interface__, which pyarrow uses to ingest
        # numpy data. Exposing __arrow_array__ on a plain numpy.ndarray proxy
        # makes pa.array(proxy, mask=...) reject the mask= kwarg, since pyarrow
        # does not allow combining mask= with the __arrow_array__ protocol.
        "__cuda_array_interface__": cuda_array_interface,
        "__array_interface__": array_interface,
        "__array_ufunc__": ndarray__array_ufunc__,
        # Emulate numpy's __array_priority__ behavior
        "__add__": make_binary_op_method(numpy.add),
        "__radd__": make_binary_op_method(numpy.add, reflected=True),
        "__sub__": make_binary_op_method(numpy.subtract),
        "__rsub__": make_binary_op_method(numpy.subtract, reflected=True),
        "__mul__": make_binary_op_method(numpy.multiply),
        "__rmul__": make_binary_op_method(numpy.multiply, reflected=True),
        "__truediv__": make_binary_op_method(numpy.true_divide),
        "__rtruediv__": make_binary_op_method(
            numpy.true_divide, reflected=True
        ),
        "__floordiv__": make_binary_op_method(numpy.floor_divide),
        "__rfloordiv__": make_binary_op_method(
            numpy.floor_divide, reflected=True
        ),
        "__mod__": make_binary_op_method(numpy.mod),
        "__rmod__": make_binary_op_method(numpy.mod, reflected=True),
        "__pow__": make_binary_op_method(numpy.power),
        "__rpow__": make_binary_op_method(numpy.power, reflected=True),
        "__divmod__": make_binary_op_method(numpy.divmod),
        "__rdivmod__": make_binary_op_method(numpy.divmod, reflected=True),
        "__matmul__": make_binary_op_method(numpy.matmul),
        "__rmatmul__": make_binary_op_method(numpy.matmul, reflected=True),
        "__and__": make_binary_op_method(numpy.bitwise_and),
        "__rand__": make_binary_op_method(numpy.bitwise_and, reflected=True),
        "__or__": make_binary_op_method(numpy.bitwise_or),
        "__ror__": make_binary_op_method(numpy.bitwise_or, reflected=True),
        "__xor__": make_binary_op_method(numpy.bitwise_xor),
        "__rxor__": make_binary_op_method(numpy.bitwise_xor, reflected=True),
        "__lshift__": make_binary_op_method(numpy.left_shift),
        "__rlshift__": make_binary_op_method(numpy.left_shift, reflected=True),
        "__rshift__": make_binary_op_method(numpy.right_shift),
        "__rrshift__": make_binary_op_method(
            numpy.right_shift, reflected=True
        ),
        "__lt__": make_binary_op_method(numpy.less),
        "__le__": make_binary_op_method(numpy.less_equal),
        "__gt__": make_binary_op_method(numpy.greater),
        "__ge__": make_binary_op_method(numpy.greater_equal),
        "__eq__": make_binary_op_method(numpy.equal),
        "__ne__": make_binary_op_method(numpy.not_equal),
        "__reduce__": ndarray__reduce__,
        # ndarrays are unhashable
        "__hash__": None,
        # iter(cupy-array) produces an iterable of zero-dim device
        # arrays, which is not usable in many settings (whereas
        # iter(numpy-array) produces an iterable of scalars)
        "__iter__": custom_iter,
        # Special wrapping to handle scalar values
        "_fsproxy_wrap": classmethod(wrap_ndarray),
        "_fsproxy_fast_to_slow": _ndarray_fsproxy_fast_to_slow,
        "base": _FastSlowAttribute("base", private=True),
        "data": _FastSlowAttribute("data", private=True),
    },
)


flatiter = make_final_proxy_type(
    "flatiter",
    cupy.flatiter,
    numpy.flatiter,
    fast_to_slow=lambda fast: cupy.asnumpy(fast.base).flat,
    slow_to_fast=lambda slow: cupy.asarray(slow).flat,
    additional_attributes={
        "__array__": array_method,
    },
)

# Mapping flags between slow and fast types
_ndarray_flags = make_intermediate_proxy_type(
    "_ndarray_flags",
    cupy._core.flags.Flags,
    _numpy_flagsobj,
)
