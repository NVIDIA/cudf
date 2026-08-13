# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython cimport bool as py_bool
from cython cimport no_gc_clear
from libc.math cimport isinf
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libcpp.limits cimport numeric_limits
from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.scalar.scalar cimport (
    scalar,
    duration_scalar,
    fixed_point_scalar,
    numeric_scalar,
    string_scalar,
    timestamp_scalar,
)
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_default_constructed_scalar,
    make_duration_scalar,
    make_empty_scalar_like,
    make_fixed_point_scalar,
    make_string_scalar,
    make_numeric_scalar,
    make_timestamp_scalar,
)
from pylibcudf.libcudf.types cimport type_id
from pylibcudf.libcudf.types cimport int128 as int128_t
from pylibcudf.libcudf.wrappers.durations cimport (
    duration_ms,
    duration_ns,
    duration_us,
    duration_s,
    duration_D,
)
from pylibcudf.libcudf.fixed_point.fixed_point cimport scale_type, decimal128
from pylibcudf.libcudf.wrappers.timestamps cimport (
    timestamp_s,
    timestamp_ms,
    timestamp_us,
    timestamp_ns,
    timestamp_D,
)

from rmm.pylibrmm.memory_resource cimport (
    DeviceMemoryResource,
    get_current_device_resource,
)
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

from .column cimport Column
from .traits cimport is_floating_point
from .types cimport DataType
from .utils cimport _get_memory_resource, _get_stream
from functools import singledispatch
from ._interop_helpers import ArrowLike, ColumnMetadata

try:
    import pyarrow as pa
    pa_err = None
except ImportError as e:
    pa = None
    pa_err = e

import datetime
import decimal

try:
    import numpy as np
    np_error = None
except ImportError as err:
    np = None
    np_error = err

__all__ = ["Scalar"]


# The no_gc_clear decorator on this class is necessary for the following reason:
#
# The object underlying a Scalar is a libcudf scalar. The underlying storage
# type within the scalar depends on the scalar's data type, but regardless of
# the proximate storage class all of the types ultimately store their data in
# an rmm::device_buffer that has an associated rmm memory resource used for
# allocation and deallocation. That memory resource must therefore still be
# alive when the Scalar is destroyed. With the current architecture of cudf we
# do not know exactly what mr was used to construct the scalar, so until then
# the best we can do is to grab the current memory resource at the time of
# construction and keep it alive until the Scalar is destroyed (for potential
# problems with this approach, see https://github.com/rapidsai/rmm/issues/1515;
# the solution will be to address https://github.com/rapidsai/cudf/issues/15170
# and also pass mrs all the way down to every rmm Python API to avoid its
# default mrs). This is done in the `__cinit__` method below.
#
# However, even in the most common case where this approach gives us the
# correct mr, we still have a problem. If a Scalar participates in a reference
# cycle, then when the garbage collector goes to clear that cycle its default
# behavior will be to clear all attributes of the object, including the mr
# attribute (see
# https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#dealloc-intro).
# This is fine in the immediate Cython code because Scalar does not define a
# `__dealloc__` method, so there is no need for the mr in the Cython code.
# However, if after the Scalar was created some other code called
# `set_current_device_resource`, then there may be no other references left to
# the mr used to create the scalar. In that case, the reference count of the
# Python DeviceMemoryResource will drop to zero and it will immediately be
# destroyed, resulting in the destruction of the underlying C++ memory resource
# as well (rmm::device_buffer only has a non-owning reference to it because all
# mrs in rmm are managed with unique_ptr semantics). That will result in a
# segmentation fault when the device_buffer goes to deallocate its memory using
# a freed memory resources. To prevent this, we use the `no_gc_clear` decorator
# to prevent the garbage collector from clearing the `mr` attribute when it
# clears the Scalar object as described in
# https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#disabling-cycle-breaking-tp-clear.
@no_gc_clear
cdef class Scalar:
    """A scalar value in device memory.

    This is the Cython representation of :cpp:class:`cudf::scalar`.
    """
    # Unlike for columns, libcudf does not support scalar views. All APIs that
    # accept scalar values accept references to the owning object rather than a
    # special view type. As a result, pylibcudf.Scalar has a simpler structure
    # than pylibcudf.Column because it can be a true wrapper around a libcudf
    # column

    def __cinit__(self, *args, **kwargs):
        self.mr = get_current_device_resource()

    def __init__(self, *args, **kwargs):
        # TODO: This case is not something we really want to
        # support, but it here for now to ease the transition of
        # DeviceScalar.
        raise ValueError("Scalar should be constructed with a factory")

    __hash__ = None

    cdef const scalar* get(self) noexcept nogil:
        return self.c_obj.get()

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef bool is_valid(self, object stream = None):
        """True if the scalar is valid, false if not"""
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        return self.get().is_valid(_cs)

    def to_arrow(
        self,
        metadata: ColumnMetadata | str | None = None,
        stream: Stream | None = None,
    ) -> ArrowLike:
        """Create a PyArrow array from a pylibcudf scalar.

        Parameters
        ----------
        metadata : ColumnMetadata | str | None
            The metadata to attach to the scalar.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        pyarrow.Scalar
        """
        # Note that metadata for scalars is primarily important for preserving
        # information on nested types since names are otherwise irrelevant.
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        return Column.from_scalar(self, 1, _stream).to_arrow(metadata=metadata)[0]

    @staticmethod
    def from_arrow(
        pa_val,
        dtype: DataType | None = None,
        stream: Stream | None = None
    ) -> Scalar:
        """
        Convert a pyarrow scalar to a pylibcudf.Scalar.

        Parameters
        ----------
        pa_val: pyarrow scalar
            Value to convert to a pylibcudf.Scalar
        dtype: DataType | None
            The datatype to cast the value to. If None,
            the type is inferred from the pyarrow scalar.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Scalar
            New pylibcudf.Scalar
        """
        return _from_arrow(pa_val, dtype, stream)

    @staticmethod
    cdef Scalar empty_like(Column column, object stream, DeviceMemoryResource mr):
        """Construct a null scalar with the same type as column.

        Parameters
        ----------
        column
            Column to take type from
        stream : Stream
            CUDA stream on which to perform the operation.
        mr : DeviceMemoryResource
            Memory resource for allocations

        Returns
        -------
        New empty (null) scalar of the given type.
        """
        cdef Stream _stream = <Stream>stream
        cdef cudaStream_t _cs = _stream.view().value()
        return Scalar.from_libcudf(
            move(make_empty_scalar_like(column.view(), _cs, mr.get_mr()))
        )

    @staticmethod
    cdef Scalar from_libcudf(unique_ptr[scalar] libcudf_scalar, dtype=None):
        """Construct a Scalar object from a libcudf scalar.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_obj.swap(libcudf_scalar)
        s._data_type = DataType.from_libcudf(s.get().type())
        return s

    @classmethod
    def from_py(
        cls,
        py_val,
        dtype: DataType | None = None,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None
    ):
        """
        Convert a Python standard library object to a Scalar.

        Parameters
        ----------
        py_val: None, bool, int, float, str, datetime, timedelta, list, dict
            Value to convert to a pylibcudf.Scalar
        dtype: DataType | None
            The datatype to cast the value to. If None,
            the type is inferred from `py_val`.
        stream : Stream | None
            CUDA stream on which to perform the operation.
        mr : DeviceMemoryResource | None
            Memory resource for allocations

        Returns
        -------
        Scalar
            New pylibcudf.Scalar
        """
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)
        return _from_py(py_val, dtype, _stream, mr)

    @classmethod
    def from_numpy(
        cls,
        np_val,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None
    ):
        """
        Convert a NumPy scalar to a Scalar.

        Parameters
        ----------
        np_val: numpy.generic
            Value to convert to a pylibcudf.Scalar
        stream : Stream | None
            CUDA stream on which to perform the operation.
        mr : DeviceMemoryResource | None
            Memory resource for allocations

        Returns
        -------
        Scalar
            New pylibcudf.Scalar
        """
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)
        return _from_numpy(np_val, _stream, mr)

    def to_py(self, stream: Stream | None = None):
        """
        Convert a Scalar to a Python scalar.

        Parameters
        ----------
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Python scalar
            A Python scalar associated with the type of the Scalar.
        """
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        if not self.is_valid(stream):
            return None

        cdef type_id tid = self.type().id()
        cdef const scalar* slr = self.c_obj.get()
        if tid == type_id.BOOL8:
            return (<numeric_scalar[cbool]*>slr).value(_cs)
        elif tid == type_id.STRING:
            return (<string_scalar*>slr).to_string(_cs).decode()
        elif tid == type_id.FLOAT32:
            return (<numeric_scalar[float]*>slr).value(_cs)
        elif tid == type_id.FLOAT64:
            return (<numeric_scalar[double]*>slr).value(_cs)
        elif tid == type_id.INT8:
            return (<numeric_scalar[int8_t]*>slr).value(_cs)
        elif tid == type_id.INT16:
            return (<numeric_scalar[int16_t]*>slr).value(_cs)
        elif tid == type_id.INT32:
            return (<numeric_scalar[int32_t]*>slr).value(_cs)
        elif tid == type_id.INT64:
            return (<numeric_scalar[int64_t]*>slr).value(_cs)
        elif tid == type_id.UINT8:
            return (<numeric_scalar[uint8_t]*>slr).value(_cs)
        elif tid == type_id.UINT16:
            return (<numeric_scalar[uint16_t]*>slr).value(_cs)
        elif tid == type_id.UINT32:
            return (<numeric_scalar[uint32_t]*>slr).value(_cs)
        elif tid == type_id.UINT64:
            return (<numeric_scalar[uint64_t]*>slr).value(_cs)
        elif tid == type_id.DECIMAL128:
            return decimal.Decimal(
                (<fixed_point_scalar[decimal128]*>slr).value(_cs).value()
            ).scaleb(
                (<fixed_point_scalar[decimal128]*>slr).type().scale()
            )
        else:
            raise NotImplementedError(
                f"Converting to Python scalar for type {self.type().id()!r} "
                "is not supported."
            )


cdef Scalar _new_scalar(unique_ptr[scalar] c_obj, DataType dtype):
    cdef Scalar s = Scalar.__new__(Scalar)
    s.c_obj.swap(c_obj)
    s._data_type = dtype
    return s


@singledispatch
def _from_py(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    raise TypeError(
        f"{type(py_val).__name__} cannot be converted to pylibcudf.Scalar"
    )


@_from_py.register(type(None))
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    cdef DataType c_dtype
    if dtype is None:
        raise ValueError("Must specify a dtype for a None value.")
    else:
        c_dtype = <DataType>dtype
    cdef unique_ptr[scalar] c_obj = make_default_constructed_scalar(
        c_dtype.c_obj,
        _cs,
        mr.get_mr()
    )
    return _new_scalar(move(c_obj), dtype)


@_from_py.register(dict)
@_from_py.register(list)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    raise NotImplementedError(
        f"Conversion from {type(py_val).__name__} is currently not supported."
    )


@_from_py.register(float)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    cdef unique_ptr[scalar] c_obj
    cdef DataType c_dtype
    if dtype is None:
        c_dtype = dtype = DataType(type_id.FLOAT64)
    else:
        c_dtype = <DataType>dtype

    cdef type_id tid = c_dtype.id()

    if tid == type_id.FLOAT32:
        if not isinf(py_val) and abs(py_val) > numeric_limits[float].max():
            raise OverflowError(f"{py_val} out of range for FLOAT32 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[float]*>c_obj.get()).set_value(py_val, _cs)
    elif tid == type_id.FLOAT64:
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[double]*>c_obj.get()).set_value(py_val, _cs)
    else:
        typ = c_dtype.id()
        raise TypeError(f"Cannot convert float to Scalar with dtype {typ.name}")

    return _new_scalar(move(c_obj), dtype)


@_from_py.register(int)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    cdef unique_ptr[scalar] c_obj
    cdef DataType c_dtype
    cdef duration_ns c_duration_ns
    cdef duration_us c_duration_us
    cdef duration_ms c_duration_ms
    cdef duration_s c_duration_s
    cdef duration_D c_duration_D
    if dtype is None:
        c_dtype = dtype = DataType(type_id.INT64)
    elif is_floating_point(dtype):
        return _from_py(float(py_val), dtype, _stream, mr)
    else:
        c_dtype = <DataType>dtype
    cdef type_id tid = c_dtype.id()

    if tid == type_id.INT8:
        if not (
            numeric_limits[int8_t].min() <= py_val <= numeric_limits[int8_t].max()
        ):
            raise OverflowError(f"{py_val} out of range for INT8 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[int8_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.INT16:
        if not (
            numeric_limits[int16_t].min() <= py_val <= numeric_limits[int16_t].max()
        ):
            raise OverflowError(f"{py_val} out of range for INT16 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[int16_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.INT32:
        if not (
            numeric_limits[int32_t].min() <= py_val <= numeric_limits[int32_t].max()
        ):
            raise OverflowError(f"{py_val} out of range for INT32 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[int32_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.INT64:
        if not (
            numeric_limits[int64_t].min() <= py_val <= numeric_limits[int64_t].max()
        ):
            raise OverflowError(f"{py_val} out of range for INT64 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[int64_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.UINT8:
        if py_val < 0:
            raise ValueError("Cannot assign negative value to UINT8 scalar")
        if py_val > numeric_limits[uint8_t].max():
            raise OverflowError(f"{py_val} out of range for UINT8 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[uint8_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.UINT16:
        if py_val < 0:
            raise ValueError("Cannot assign negative value to UINT16 scalar")
        if py_val > numeric_limits[uint16_t].max():
            raise OverflowError(f"{py_val} out of range for UINT16 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[uint16_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.UINT32:
        if py_val < 0:
            raise ValueError("Cannot assign negative value to UINT32 scalar")
        if py_val > numeric_limits[uint32_t].max():
            raise OverflowError(f"{py_val} out of range for UINT32 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[uint32_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.UINT64:
        if py_val < 0:
            raise ValueError("Cannot assign negative value to UINT64 scalar")
        if py_val > numeric_limits[uint64_t].max():
            raise OverflowError(f"{py_val} out of range for UINT64 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[uint64_t]*>c_obj.get()).set_value(py_val, _cs)

    elif tid == type_id.BOOL8:
        if py_val not in (0, 1):
            raise ValueError(f"Cannot convert {py_val} to BOOL8 scalar")
        c_obj = make_numeric_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        (<numeric_scalar[cbool]*>c_obj.get()).set_value(py_val != 0, _cs)

    elif tid == type_id.DURATION_NANOSECONDS:
        if py_val > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{py_val} nanoseconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_ns = duration_ns(<int64_t>py_val)
        (<duration_scalar[duration_ns]*>c_obj.get()).set_value(
            c_duration_ns, _cs
        )

    elif tid == type_id.DURATION_MICROSECONDS:
        if py_val > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{py_val} microseconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_us = duration_us(<int64_t>py_val)
        (<duration_scalar[duration_us]*>c_obj.get()).set_value(
            c_duration_us, _cs
        )

    elif tid == type_id.DURATION_MILLISECONDS:
        if py_val > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{py_val} milliseconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_ms = duration_ms(<int64_t>py_val)
        (<duration_scalar[duration_ms]*>c_obj.get()).set_value(
            c_duration_ms, _cs
        )

    elif tid == type_id.DURATION_SECONDS:
        if py_val > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{py_val} seconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_s = duration_s(<int64_t>py_val)
        (<duration_scalar[duration_s]*>c_obj.get()).set_value(
            c_duration_s, _cs
        )

    elif tid == type_id.DURATION_DAYS:
        if py_val > numeric_limits[int32_t].max():
            raise OverflowError(
                f"{py_val} days out of range for INT32 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_D = duration_D(<int32_t>py_val)
        (<duration_scalar[duration_D]*>c_obj.get()).set_value(
            c_duration_D, _cs
        )

    else:
        typ = c_dtype.id()
        raise TypeError(f"Cannot convert int to Scalar with dtype {typ.name}")

    return _new_scalar(move(c_obj), dtype)


@_from_py.register(py_bool)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    if dtype is None:
        dtype = DataType(type_id.BOOL8)
    elif dtype.id() != type_id.BOOL8:
        tid = (<DataType>dtype).id()
        raise TypeError(
            f"Cannot convert bool to Scalar with dtype {tid.name}"
        )

    cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
        (<DataType>dtype).c_obj,
        _cs,
        mr.get_mr()
    )
    (<numeric_scalar[cbool]*>c_obj.get()).set_value(py_val, _cs)
    return _new_scalar(move(c_obj), dtype)


@_from_py.register(str)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    if dtype is None:
        dtype = DataType(type_id.STRING)
    elif dtype.id() != type_id.STRING:
        tid = (<DataType>dtype).id()
        raise TypeError(
            f"Cannot convert str to Scalar with dtype {tid.name}"
        )
    cdef unique_ptr[scalar] c_obj = make_string_scalar(
        py_val.encode(), _cs, mr.get_mr()
    )
    return _new_scalar(move(c_obj), dtype)


@_from_py.register(datetime.timedelta)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    cdef unique_ptr[scalar] c_obj
    cdef duration_us c_duration_us
    cdef duration_ns c_duration_ns
    cdef duration_ms c_duration_ms
    cdef duration_s c_duration_s
    cdef duration_D c_duration_D
    if dtype is None:
        dtype = DataType(type_id.DURATION_MICROSECONDS)

    cdef DataType c_dtype = dtype
    cdef type_id tid = c_dtype.id()
    total_seconds = py_val.total_seconds()
    if tid == type_id.DURATION_NANOSECONDS:
        total_nanoseconds = int(total_seconds * 1_000_000_000)
        if total_nanoseconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{total_nanoseconds} nanoseconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_ns = duration_ns(<int64_t>total_nanoseconds)
        (<duration_scalar[duration_ns]*>c_obj.get()).set_value(
            c_duration_ns, _cs
        )
    elif tid == type_id.DURATION_MICROSECONDS:
        total_microseconds = int(total_seconds * 1_000_000)
        if total_microseconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{total_microseconds} microseconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_us = duration_us(<int64_t>total_microseconds)
        (<duration_scalar[duration_us]*>c_obj.get()).set_value(
            c_duration_us, _cs
        )
    elif tid == type_id.DURATION_MILLISECONDS:
        total_milliseconds = int(total_seconds * 1_000)
        if total_milliseconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{total_milliseconds} milliseconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_ms = duration_ms(<int64_t>total_milliseconds)
        (<duration_scalar[duration_ms]*>c_obj.get()).set_value(
            c_duration_ms, _cs
        )
    elif tid == type_id.DURATION_SECONDS:
        total_seconds = int(total_seconds)
        if total_seconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{total_seconds} seconds out of range for INT64 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_s = duration_s(<int64_t>total_seconds)
        (<duration_scalar[duration_s]*>c_obj.get()).set_value(
            c_duration_s, _cs
        )
    elif tid == type_id.DURATION_DAYS:
        total_days = int(total_seconds // 86400)
        if total_days > numeric_limits[int32_t].max():
            raise OverflowError(
                f"{total_days} days out of range for INT32 limit."
            )
        c_obj = make_duration_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_D = duration_D(<int32_t>total_days)
        (<duration_scalar[duration_D]*>c_obj.get()).set_value(
            c_duration_D, _cs
        )
    else:
        typ = c_dtype.id()
        raise TypeError(f"Cannot convert timedelta to Scalar with dtype {typ.name}")
    return _new_scalar(move(c_obj), dtype)


@_from_py.register(datetime.date)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    cdef unique_ptr[scalar] c_obj
    cdef duration_us c_duration_us
    cdef duration_ns c_duration_ns
    cdef duration_ms c_duration_ms
    cdef duration_s c_duration_s
    cdef duration_D c_duration_D
    cdef timestamp_s c_timestamp_s
    cdef timestamp_ms c_timestamp_ms
    cdef timestamp_us c_timestamp_us
    cdef timestamp_ns c_timestamp_ns
    cdef timestamp_D c_timestamp_D
    if dtype is None:
        dtype = DataType(type_id.TIMESTAMP_MICROSECONDS)

    cdef DataType c_dtype = dtype
    cdef type_id tid = c_dtype.id()
    if isinstance(py_val, datetime.datetime):
        if py_val.tzinfo is None:
            # Treat tz-naive datetime as UTC so .timestamp()
            # does not account for the system's timezone.
            epoch_seconds = py_val.replace(tzinfo=datetime.timezone.utc).timestamp()
        else:
            epoch_seconds = py_val.timestamp()
    else:
        epoch_seconds = (py_val - datetime.date(1970, 1, 1)).total_seconds()
    if tid == type_id.TIMESTAMP_NANOSECONDS:
        epoch_nanoseconds = int(epoch_seconds * 1_000_000_000)
        if epoch_nanoseconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{epoch_nanoseconds} nanoseconds out of range for INT64 limit."
            )
        c_obj = make_timestamp_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_ns = duration_ns(<int64_t>epoch_nanoseconds)
        c_timestamp_ns = timestamp_ns(c_duration_ns)
        (<timestamp_scalar[timestamp_ns]*>c_obj.get()).set_value(
            c_timestamp_ns, _cs
        )
    elif tid == type_id.TIMESTAMP_MICROSECONDS:
        epoch_microseconds = int(epoch_seconds * 1_000_000)
        if epoch_microseconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{epoch_microseconds} microseconds out of range for INT64 limit."
            )
        c_obj = make_timestamp_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_us = duration_us(<int64_t>epoch_microseconds)
        c_timestamp_us = timestamp_us(c_duration_us)
        (<timestamp_scalar[timestamp_us]*>c_obj.get()).set_value(
            c_timestamp_us, _cs
        )
    elif tid == type_id.TIMESTAMP_MILLISECONDS:
        epoch_milliseconds = int(epoch_seconds * 1_000)
        if epoch_milliseconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{epoch_milliseconds} milliseconds out of range for INT64 limit."
            )
        c_obj = make_timestamp_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_ms = duration_ms(<int64_t>epoch_milliseconds)
        c_timestamp_ms = timestamp_ms(c_duration_ms)
        (<timestamp_scalar[timestamp_ms]*>c_obj.get()).set_value(
            c_timestamp_ms, _cs
        )
    elif tid == type_id.TIMESTAMP_SECONDS:
        epoch_seconds = int(epoch_seconds)
        if epoch_seconds > numeric_limits[int64_t].max():
            raise OverflowError(
                f"{epoch_seconds} seconds out of range for INT64 limit."
            )
        c_obj = make_timestamp_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_s = duration_s(<int64_t>epoch_seconds)
        c_timestamp_s = timestamp_s(c_duration_s)
        (<timestamp_scalar[timestamp_s]*>c_obj.get()).set_value(
            c_timestamp_s, _cs
        )
    elif tid == type_id.TIMESTAMP_DAYS:
        epoch_days = int(epoch_seconds // 86400)
        if epoch_days > numeric_limits[int32_t].max():
            raise OverflowError(
                f"{epoch_days} days out of range for INT32 limit."
            )
        c_obj = make_timestamp_scalar(c_dtype.c_obj, _cs, mr.get_mr())
        c_duration_D = duration_D(<int32_t>epoch_days)
        c_timestamp_D = timestamp_D(c_duration_D)
        (<timestamp_scalar[timestamp_D]*>c_obj.get()).set_value(
            c_timestamp_D, _cs
        )
    else:
        typ = c_dtype.id()
        raise TypeError(f"Cannot convert datetime to Scalar with dtype {typ.name}")
    return _new_scalar(move(c_obj), dtype)


@_from_py.register(decimal.Decimal)
def _(
    py_val, dtype: DataType | None, stream: Stream, mr: DeviceMemoryResource
):
    cdef Stream _stream = stream
    cdef cudaStream_t _cs = _stream.view().value()
    scale = py_val.as_tuple().exponent
    as_int = int(py_val.scaleb(-scale))

    cdef int128_t val = <int128_t>as_int

    dtype = DataType(type_id.DECIMAL128, scale)

    if dtype.id() != type_id.DECIMAL128:
        raise TypeError("Expected dtype to be DECIMAL128")

    cdef unique_ptr[scalar] c_obj = make_fixed_point_scalar[decimal128](
        val,
        scale_type(<int32_t>scale),
        _cs,
        mr.get_mr()
    )
    return _new_scalar(move(c_obj), dtype)


@singledispatch
def _from_numpy(np_val, stream: Stream, mr: DeviceMemoryResource):
    if np_error is not None:
        raise np_error
    raise TypeError(
        f"{type(np_val).__name__} cannot be converted to pylibcudf.Scalar"
    )


if np is not None:
    @_from_numpy.register(np.datetime64)
    @_from_numpy.register(np.timedelta64)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        raise NotImplementedError(
            f"{type(np_val).__name__} is currently not supported."
        )

    @_from_numpy.register(np.bool_)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        cdef DataType dtype = DataType(type_id.BOOL8)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        cdef cbool c_val = np_val
        (<numeric_scalar[cbool]*>c_obj.get()).set_value(c_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.str_)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        cdef DataType dtype = DataType(type_id.STRING)
        cdef unique_ptr[scalar] c_obj = make_string_scalar(
            np_val.item().encode(),
            _cs,
            mr.get_mr()
        )
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.int8)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.INT8)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[int8_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.int16)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.INT16)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[int16_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.int32)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.INT32)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[int32_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.int64)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.INT64)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[int64_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.uint8)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.UINT8)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[uint8_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.uint16)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.UINT16)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[uint16_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.uint32)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.UINT32)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[uint32_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.uint64)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.UINT64)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[uint64_t]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.float32)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.FLOAT32)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[float]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr

    @_from_numpy.register(np.float64)
    def _(np_val, stream: Stream, mr: DeviceMemoryResource):
        cdef Stream _stream = stream
        cdef cudaStream_t _cs = _stream.view().value()
        dtype = DataType(type_id.FLOAT64)
        cdef unique_ptr[scalar] c_obj = make_numeric_scalar(
            dtype.c_obj, _cs, mr.get_mr()
        )
        (<numeric_scalar[double]*>c_obj.get()).set_value(np_val, _cs)
        cdef Scalar slr = _new_scalar(move(c_obj), dtype)
        return slr


def _from_arrow(
    obj: pa.Scalar,
    dtype: DataType | None = None,
    stream: Stream | None = None
) -> Scalar:
    if pa_err is not None:
        raise RuntimeError(
            "pyarrow was not found on your system. Please "
            "pip install pylibcudf with the [pyarrow] extra for a "
            "compatible pyarrow version."
        ) from pa_err
    if isinstance(obj.type, pa.ListType) and obj.as_py() is None:
        # pyarrow doesn't correctly handle None values for list types, so
        # we have to create this one manually.
        # https://github.com/apache/arrow/issues/40319
        pa_array = pa.array([None], type=obj.type)
    else:
        pa_array = pa.array([obj])
    return Column.from_arrow(pa_array, dtype=dtype, stream=stream).to_scalar()
