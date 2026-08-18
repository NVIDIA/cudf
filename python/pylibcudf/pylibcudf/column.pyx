# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from cython.operator cimport dereference

from cpython.exc cimport PyErr_Occurred
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from cpython.ref cimport PyObject

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t, uintptr_t
from libcpp.limits cimport numeric_limits
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from pylibcudf.libcudf.column.column cimport column, column_contents
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.column.column_factories cimport make_column_from_scalar
from pylibcudf.libcudf.copying cimport get_element
from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    ArrowDeviceArray,
    arrow_column,
    column_metadata,
    release_arrow_array_raw,
    to_arrow_device_raw,
    to_arrow_host_raw,
    to_arrow_schema_raw,
)
from pylibcudf.libcudf.null_mask cimport bitmask_allocation_size_bytes
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.strings.strings_column_view cimport strings_column_view
from pylibcudf.libcudf.structs.structs_column_view cimport structs_column_view
from pylibcudf.libcudf.types cimport bitmask_type, size_of as cpp_size_of, size_type
from pylibcudf.libcudf.utilities.traits cimport is_fixed_width

from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from ._interop_helpers cimport (
    _metadata_to_libcudf,
    _release_array,
    _release_device_array,
    _release_schema,
)
from .filling cimport sequence
from .gpumemoryview cimport gpumemoryview
from .scalar cimport Scalar
from .span import is_span as py_is_span
from .traits cimport (
    is_fixed_width as plc_is_fixed_width,
    is_nested,
)
from .types cimport DataType, size_of, type_id
from pylibcudf.types import TypeId
from .utils cimport _get_stream, _get_memory_resource
from pylibcudf.typing import SupportsArrayInterface, SupportsCudaArrayInterface
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike

from .gpumemoryview import _datatype_from_dtype_desc
from ._interop_helpers import ArrowLike, ColumnMetadata, _ObjectWithArrowMetadata

from itertools import accumulate
import array
import functools
import operator
from cuda.bindings.cyruntime cimport cudaStream_t

try:
    import pyarrow as pa
    pa_err = None
except ImportError as e:
    pa = None
    pa_err = e


__all__ = [
    "Column",
    "ListsColumnView",
    "StructsColumnView",
    "is_c_contiguous",
]


cdef is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    return True


cdef class _ArrowColumnHolder:
    """A holder for an Arrow column for gpumemoryview lifetime management."""
    cdef unique_ptr[arrow_column] col
    cdef DeviceMemoryResource mr


cdef class OwnerWithCAI:
    """An interface for column view's data with gpumemoryview via CAI."""
    @staticmethod
    cdef create(column_view cv, object owner, object stream):
        obj = OwnerWithCAI()
        obj.owner = owner
        # The default size of 0 will be applied for any type that stores data in the
        # children (such that the parent size is 0).
        size = 0
        if cv.type().id() == type_id.EMPTY:
            size = cv.size()
        elif is_fixed_width(cv.type()):
            # Cast to Python integers before multiplying to avoid overflow.
            size = int(cv.size()) * int(cpp_size_of(cv.type()))
        elif cv.type().id() == type_id.STRING:
            size = strings_column_view(cv).chars_size((<Stream>stream).view().value())

        obj.cai = {
            "shape": (size,),
            "strides": None,
            # For the purposes in this function, just treat all of the types as byte
            # streams of the appropriate size. This matches what we currently get from
            # rmm.DeviceBuffer
            "typestr": "|u1",
            "data": (<uintptr_t> cv.head[char](), False),
            "version": 3,
        }
        return obj

    @property
    def __cuda_array_interface__(self):
        return self.cai


cdef class OwnerMaskWithCAI:
    """An interface for column view's null mask with gpumemoryview via CAI."""
    @staticmethod
    cdef create(column_view cv, object owner):
        obj = OwnerMaskWithCAI()
        obj.owner = owner

        obj.cai = {
            "shape": (bitmask_allocation_size_bytes(cv.size(), 64),),
            "strides": None,
            # For the purposes in this function, just treat all of the types as byte
            # streams of the appropriate size. This matches what we currently get from
            # rmm.DeviceBuffer
            "typestr": "|u1",
            "data": (<uintptr_t> cv.null_mask(), False),
            "version": 3,
        }
        return obj

    @property
    def __cuda_array_interface__(self):
        return self.cai


class ArrayInterfaceWrapper:
    def __init__(self, iface):
        self.__array_interface__ = iface


cdef gpumemoryview _copy_array_to_device(object buf, object stream: CudaStreamLike | None = None):
    """
    Copy a host-side array.array buffer to device memory.

    Parameters
    ----------
    buf : array.array
        Array of bytes.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    gpumemoryview
        A device memory view backed by an rmm.DeviceBuffer.
    """
    cdef memoryview mv = memoryview(buf)
    cdef uintptr_t ptr = <uintptr_t>mv.obj.buffer_info()[0]
    cdef size_t nbytes = len(mv) * mv.itemsize
    cdef Stream _stream = _get_stream(stream)

    return gpumemoryview(DeviceBuffer.to_device(
        <const unsigned char[:nbytes:1]><const unsigned char*>ptr,
        _stream
    ))


def _infer_list_depth_and_dtype(obj: list) -> tuple[int, type]:
    """Infer the nesting depth and final scalar type."""
    depth = 0
    current = obj

    while isinstance(current, list) and current:
        current = current[0]
        depth += 1

    if not current and depth == 0:
        raise ValueError("Cannot infer dtype from empty input")

    if not isinstance(current, (int, float, bool, str)):
        raise TypeError(f"Unsupported scalar type: {type(current).__name__}")

    return depth, type(current)


def _flatten_nested_list(obj: list, depth: int) -> tuple[list, tuple[int, ...]]:
    """Flatten a nested list and compute the shape"""
    shape = _infer_shape(obj, depth)

    flat = [None] * functools.reduce(operator.mul, shape)
    _flatten(obj, flat, 0)
    return flat, shape


def _infer_shape(obj: list, depth: int) -> tuple[int, ...]:
    shape = []
    current = obj

    for i in range(depth):
        if not current:
            raise ValueError("Cannot infer shape from empty list")

        shape.append(len(current))

        if i < depth - 1:
            first = current[0]
            if not all(
                isinstance(sub, list) and len(sub) == len(first) for sub in current
            ):
                raise ValueError("Inconsistent inner list shapes")
            current = first

    return tuple(shape)


def _flatten(obj: list, out: list, offset: int) -> int:
    if not isinstance(obj[0], list):
        out[offset:offset + len(obj)] = obj
        return offset + len(obj)
    for sub in obj:
        offset = _flatten(sub, out, offset)
    return offset


def _prepare_array_metadata(
    iface: dict,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...] | None, DataType]:
    """
    Parse and validate a CUDA or NumPy array interface dictionary.

    Parameters
    ----------
    iface : dict
        A dictionary conforming to the __cuda_array_interface__
        or __array_interface__ spec.

    Returns
    -------
    tuple
        - data pointer (int)
        - total number of bytes (int)
        - shape (tuple[int, ...])
        - strides (tuple[int, ...] | None)
        - data type (pylibcudf.DataType)

    Raises
    ------
    ValueError
        If the interface is invalid, big-endian, non-contiguous,
        or exceed the size_type limit.
    """
    if iface.get("mask") is not None:
        raise NotImplementedError("Masked array-interface inputs are not supported")

    if iface["typestr"][0] == ">":
        raise ValueError("Big-endian data is not supported")

    if not (
        isinstance(iface.get("data"), tuple)
        and isinstance(iface["data"][0], int)
    ):
        raise ValueError(
            "Expected a data field with an integer pointer in the array interface. "
            "Objects with data set to None or a buffer object are not supported."
        )

    if not isinstance(iface["shape"], tuple) or len(iface["shape"]) == 0:
        raise ValueError("shape must be a non-empty tuple")

    dtype = _datatype_from_dtype_desc(iface["typestr"][1:])
    itemsize = size_of(dtype)

    shape = iface["shape"]
    strides = iface.get("strides")

    if not is_c_contiguous(shape, strides, itemsize):
        raise ValueError("Data must be C-contiguous")

    size_type_row_limit = numeric_limits[size_type].max()
    if (
        shape[0] > size_type_row_limit if len(shape) == 1
        # >= because we do list column construction _wrap_nested_list_column
        else shape[0] >= size_type_row_limit
    ):
        raise ValueError(
            "Number of rows exceeds size_type limit for offsets column construction."
        )

    flat_size = functools.reduce(operator.mul, shape)
    if flat_size > numeric_limits[size_type].max():
        raise ValueError("Flat size exceeds size_type limit")

    return iface["data"][0], flat_size * itemsize, shape, strides, dtype


cdef class Column:
    """A container of nullable device data as a column of elements.

    This class is an implementation of `Arrow columnar data specification
    <https://arrow.apache.org/docs/format/Columnar.html>`__ for data stored on
    GPUs. It relies on Python memoryview-like semantics to maintain shared
    ownership of the data it is constructed with, so any input data may also be
    co-owned by other data structures. The Column is designed to be operated on
    using algorithms backed by libcudf.

    Parameters
    ----------
    data_type : DataType
        The type of data in the column.
    size : size_type
        The number of rows in the column.
    data : gpumemoryview
        The data the column will refer to.
    mask : gpumemoryview
        The null mask for the column.
    null_count : int
        The number of null rows in the column.
    offset : int
        The offset into the data buffer where the column's data begins.
    children : Iterable[Column]
        The children of this column if it is a compound column type.
    validate : bool, default True
        Whether to validate that data and mask satisfy Span protocol.
    """
    __hash__ = None

    def __init__(
        self, DataType data_type not None, size_type size, object data,
        object mask, size_type null_count, size_type offset,
        children: Iterable[Column], bint validate=True
    ):
        children = list(children)
        if not all(isinstance(c, Column) for c in children):
            raise ValueError("All children must be pylibcudf Column objects")

        if validate:
            if data is not None and not py_is_span(data):
                raise TypeError(
                    f"data must satisfy Span protocol (have .ptr and .size), "
                    f"got {type(data).__name__}"
                )
            if mask is not None and not py_is_span(mask):
                raise TypeError(
                    f"mask must satisfy Span protocol (have .ptr and .size), "
                    f"got {type(mask).__name__}"
                )

        self._data_type = data_type
        self._size = size
        self._data = data
        self._mask = mask
        self._null_count = null_count
        self._offset = offset
        self._children = children
        self._num_children = len(children)

    def to_arrow(
        self,
        metadata: ColumnMetadata | str | None = None,
        stream: Stream | None = None,
    ) -> ArrowLike:
        """Create a pyarrow array from a pylibcudf column.

        Parameters
        ----------
        metadata : ColumnMetadata | str | None
            The metadata to attach to the column.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        pyarrow.Array
        """
        if pa_err is not None:
            raise RuntimeError(
                "pyarrow was not found on your system. Please "
                "pip install pylibcudf with the [pyarrow] extra for a "
                "compatible pyarrow version."
            ) from pa_err
        # TODO: Once the arrow C device interface registers more
        # types that it supports, we can call pa.array(self) if
        # no metadata is passed.
        return pa.array(_ObjectWithArrowMetadata(self, metadata, stream))

    @staticmethod
    def from_arrow(
        obj: ArrowLike,
        dtype: DataType | None = None,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None
    ) -> ArrowLike:
        """
        Create a Column from an Arrow-like object using the Arrow C Data Interface.

        This method supports host and device Arrow arrays or streams. It detects
        the type of Arrow object provided and constructs a `pylibcudf.Column`
        accordingly using the appropriate Arrow C pointer-based interface.

        Parameters
        ----------
        obj : Arrow-like
            An object implementing one of the following:
            - `__arrow_c_array__` (host Arrow array)
            - `__arrow_c_device_array__` (device Arrow array)
            - `__arrow_c_stream__` (host Arrow stream)
            - `__arrow_c_device_stream__` (device Arrow stream)
        dtype : DataType | None
            The pylibcudf data type.
        stream : Stream | None
            CUDA stream on which to perform the operation.
        mr : DeviceMemoryResource | None
            Device memory resource for allocations.

        Returns
        -------
        Column
            A `pylibcudf.Column` representing the Arrow data.

        Raises
        ------
        NotImplementedError
            If the Arrow-like object is a device stream (`__arrow_c_device_stream__`).
            If the dtype argument is not None.
        ValueError
            If the object does not implement a known Arrow C interface.

        Notes
        -----
        - This method supports zero-copy construction for device arrays.
        """
        if dtype is not None:
            raise NotImplementedError(
                "Creating a Column with the dtype argument specified."
            )
        cdef ArrowSchema* c_schema
        cdef ArrowArray* c_array
        cdef ArrowDeviceArray* c_device_array
        cdef _ArrowColumnHolder result
        cdef unique_ptr[arrow_column] c_result

        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)

        if hasattr(obj, "__arrow_c_device_array__"):
            schema, d_array = obj.__arrow_c_device_array__()
            c_schema = <ArrowSchema*>PyCapsule_GetPointer(schema, "arrow_schema")
            c_device_array = (
                <ArrowDeviceArray*>PyCapsule_GetPointer(d_array, "arrow_device_array")
            )

            result = _ArrowColumnHolder()
            result.mr = mr
            with nogil:
                c_result = make_unique[arrow_column](
                    move(dereference(c_schema)),
                    move(dereference(c_device_array)),
                    _cs,
                    result.mr.get_mr(),
                )
            result.col.swap(c_result)

            return Column.from_column_view_of_arbitrary(
                result.col.get().view(),
                result,
                _stream,
            )
        elif hasattr(obj, "__arrow_c_array__"):
            schema, h_array = obj.__arrow_c_array__()
            c_schema = <ArrowSchema*>PyCapsule_GetPointer(schema, "arrow_schema")
            c_array = <ArrowArray*>PyCapsule_GetPointer(h_array, "arrow_array")

            result = _ArrowColumnHolder()
            result.mr = mr
            with nogil:
                c_result = make_unique[arrow_column](
                    move(dereference(c_schema)),
                    move(dereference(c_array)),
                    _cs,
                    result.mr.get_mr(),
                )
            result.col.swap(c_result)

            return Column.from_column_view_of_arbitrary(
                result.col.get().view(),
                result,
                _stream,
            )
        elif hasattr(obj, "__arrow_c_stream__"):
            arrow_stream = obj.__arrow_c_stream__()
            c_arrow_stream = (
                <ArrowArrayStream*>PyCapsule_GetPointer(
                    arrow_stream,
                    "arrow_array_stream",
                )
            )

            result = _ArrowColumnHolder()
            result.mr = mr
            with nogil:
                c_result = make_unique[arrow_column](
                    move(dereference(c_arrow_stream)),
                    _cs,
                    result.mr.get_mr(),
                )
            result.col.swap(c_result)

            return Column.from_column_view_of_arbitrary(
                result.col.get().view(),
                result,
                _stream,
            )
        elif hasattr(obj, "__arrow_c_device_stream__"):
            # TODO: When we add support for this case, it should be moved above
            # the __arrow_c_array__ case since we should prioritize device
            # data if possible.
            raise NotImplementedError("Device streams not yet supported")
        else:
            raise ValueError("Invalid Arrow-like object")

    cdef column_view view(self):
        """Generate a libcudf column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef const void * data = NULL
        cdef const bitmask_type * null_mask = NULL
        cdef size_t data_ptr
        cdef size_t mask_ptr

        if self._data is not None:
            data_ptr = <size_t>self._data.ptr
            data = <void*>data_ptr
        if self._mask is not None:
            mask_ptr = <size_t>self._mask.ptr
            null_mask = <bitmask_type*>mask_ptr

        # TODO: Check if children can ever change. If not, this could be
        # computed once in the constructor and always be reused.
        cdef vector[column_view] c_children
        if self._children is not None:
            for child in self._children:
                c_children.push_back((<Column> child).view())

        return column_view(
            self._data_type.c_obj, self._size, data, null_mask,
            self._null_count, self._offset, c_children
        )

    cdef mutable_column_view mutable_view(self):
        """Generate a libcudf mutable_column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef void * data = NULL
        cdef bitmask_type * null_mask = NULL
        cdef size_t data_ptr
        cdef size_t mask_ptr

        if self._data is not None:
            data_ptr = <size_t>self._data.ptr
            data = <void*>data_ptr
        if self._mask is not None:
            mask_ptr = <size_t>self._mask.ptr
            null_mask = <bitmask_type*>mask_ptr

        cdef vector[mutable_column_view] c_children
        if self._children is not None:
            for child in self._children:
                # See the view method for why this needs to be cast.
                c_children.push_back((<Column> child).mutable_view())

        return mutable_column_view(
            self._data_type.c_obj, self._size, data, null_mask,
            self._null_count, self._offset, c_children
        )

    @staticmethod
    def from_rmm_buffer(
        DeviceBuffer buff,
        DataType dtype,
        size_type size,
        children: Iterable[Column],
    ) -> Column:
        """
        Create a Column from an RMM DeviceBuffer.

        Parameters
        ----------
        buff : DeviceBuffer
            The data rmm.DeviceBuffer.
        size : size_type
            The number of rows in the column.
        dtype : DataType
            The type of the data in the buffer.
        children : Iterable[Column]
            The child columns.

        Notes
        -----
        To provide a mask and null count, use `Column.with_mask` after
        this method.
        """
        children = list(children)
        if plc_is_fixed_width(dtype) and len(children) != 0:
            raise ValueError("Fixed-width types must have zero children.")
        elif dtype.id() == type_id.STRING and len(children) != 1:
            raise ValueError("String columns have have 1 child column of offsets.")
        elif is_nested(dtype) and len(children) == 0:
            raise ValueError(
                "List and struct columns must have at least one child column."
            )

        cdef gpumemoryview data = gpumemoryview(buff)
        return Column(
            dtype,
            size,
            data,
            None,
            0,
            # Initial offset when capturing a C++ column is always 0.
            0,
            children,
        )

    @staticmethod
    cdef Column from_libcudf(
        unique_ptr[column] libcudf_col,
        object stream,
        DeviceMemoryResource mr
    ):
        """Create a Column from a libcudf column.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        assert stream is not None, "stream cannot be None"
        assert mr is not None, "mr cannot be None"
        cdef Stream _stream = <Stream>stream
        cdef DataType dtype = DataType.from_libcudf(libcudf_col.get().type())
        cdef size_type size = libcudf_col.get().size()

        cdef size_type null_count = libcudf_col.get().null_count()

        cdef column_contents contents = libcudf_col.get().release()

        # Note that when converting to cudf Column objects we'll need to pull
        # out the base object.
        cdef gpumemoryview data = gpumemoryview(
            DeviceBuffer.c_from_unique_ptr(move(contents.data), _stream, mr)
        )

        cdef gpumemoryview mask = None
        if null_count > 0:
            mask = gpumemoryview(
                DeviceBuffer.c_from_unique_ptr(move(contents.null_mask), _stream, mr)
            )

        children = []
        if contents.children.size() != 0:
            for i in range(contents.children.size()):
                children.append(
                    Column.from_libcudf(move(contents.children[i]), stream, mr)
                )

        return Column(
            dtype,
            size,
            data,
            mask,
            null_count,
            # Initial offset when capturing a C++ column is always 0.
            0,
            children,
        )

    cpdef Column with_mask(self, object mask, size_type null_count, bint validate=True):
        """Augment this column with a new null mask.

        Parameters
        ----------
        mask : Span-like or None
            New mask (or None to unset the mask). Must satisfy Span protocol.
        null_count : int
            New null count. If this is incorrect, bad things happen.
        validate : bool, default True
            Whether to validate that mask satisfies Span protocol.

        Returns
        -------
        New Column object sharing data with self (except for the mask which is new).
        """
        if validate and mask is not None and not py_is_span(mask):
            raise TypeError(
                f"mask must satisfy Span protocol or None, "
                f"got {type(mask).__name__}"
            )
        if mask is None and null_count > 0:
            raise ValueError("Empty mask must have null count of zero")
        return Column(
            self._data_type,
            self._size,
            self._data,
            mask,
            null_count,
            self._offset,
            self._children,
            validate=False,  # Already validated above if needed
        )

    @staticmethod
    cdef Column from_column_view(const column_view& cv, Column owner):
        """Create a Column from a libcudf column_view into a Column owner.

        This method accepts shared ownership of the underlying data from the
        owner and relies on the offset from the view.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        children = []
        cdef size_type i
        if cv.num_children() != 0:
            for i in range(cv.num_children()):
                children.append(Column.from_column_view(cv.child(i), owner.child(i)))

        return Column(
            DataType.from_libcudf(cv.type()),
            cv.size(),
            owner._data,
            owner._mask,
            cv.null_count(),
            cv.offset(),
            children,
        )

    # Ideally this function would simply be handled via a fused type in
    # from_column_view, but this does not work due to
    # https://github.com/cython/cython/issues/6740
    @staticmethod
    cdef Column from_column_view_of_arbitrary(
        const column_view& cv,
        object owner,
        object stream,
    ):
        """Create a Column from a libcudf column_view into an arbitrary owner.

        This method accepts shared ownership of the underlying data from the owner.
        Since the owner may be any arbitrary object, every child Column also shares
        ownership of the same buffer since they do not have the information available to
        choose to only own subsets of it.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # For efficiency, prohibit calling this overload with a Column owner.
        assert not isinstance(owner, Column)
        stream = _get_stream(stream)

        children = []
        cdef size_type i
        if cv.num_children() != 0:
            for i in range(cv.num_children()):
                children.append(
                    Column.from_column_view_of_arbitrary(cv.child(i), owner, stream)
                )

        cdef gpumemoryview owning_data = gpumemoryview(
            OwnerWithCAI.create(cv, owner, stream)
        )
        cdef gpumemoryview owning_mask = None
        if cv.null_count() > 0:
            owning_mask = gpumemoryview(OwnerMaskWithCAI.create(cv, owner))

        return Column(
            DataType.from_libcudf(cv.type()),
            cv.size(),
            owning_data,
            owning_mask,
            cv.null_count(),
            cv.offset(),
            children,
        )

    @staticmethod
    def from_scalar(
        Scalar slr,
        size_type size,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None,
    ) -> Column:
        """Create a Column from a Scalar.

        Parameters
        ----------
        slr : Scalar
            The scalar to create a column from.
        size : size_type
            The number of elements in the column.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Column
            A Column containing the scalar repeated `size` times.
        """
        cdef const scalar* c_scalar = slr.get()
        cdef unique_ptr[column] c_result
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)
        with nogil:
            c_result = make_column_from_scalar(
                dereference(c_scalar),
                size,
                _cs,
                mr.get_mr()
            )
        return Column.from_libcudf(move(c_result), _stream, mr)

    cpdef Scalar to_scalar(self, object stream: CudaStreamLike | None = None, DeviceMemoryResource mr=None):
        """
        Return the first value of 1-element column as a Scalar.

        Raises
        ------
        ValueError
            If the column has more than one row.
        stream : Stream | None
            CUDA stream on which to perform the operation.
        mr : DeviceMemoryResource | None
            Device memory resource used to allocate the returned scalar's device memory.

        Returns
        -------
        Scalar
            A Scalar representing the only value in the column, including nulls.
        """
        if self._size != 1:
            raise ValueError("to_scalar only works for columns of size 1")

        cdef column_view cv = self.view()
        cdef unique_ptr[scalar] result
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)

        with nogil:
            result = get_element(cv, 0, _cs, mr.get_mr())

        return Scalar.from_libcudf(move(result))

    @staticmethod
    def all_null_like(
        Column like,
        size_type size,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None,
    ) -> Column:
        """Create an all null column from a template.

        Parameters
        ----------
        like : Column
            Column whose type we should mimic
        size : int
            Number of rows in the resulting column.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Column
            An all-null column of `size` rows and type matching `like`.
        """
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)
        cdef Scalar slr = Scalar.empty_like(like, _stream, mr)
        cdef unique_ptr[column] c_result
        with nogil:
            c_result = make_column_from_scalar(
                dereference(slr.get()),
                size,
                _cs,
                mr.get_mr()
            )
        return Column.from_libcudf(move(c_result), _stream, mr)

    @staticmethod
    cdef Column _wrap_nested_list_column(
        gpumemoryview data,
        tuple shape,
        DataType dtype,
        Column base=None,
        object stream: CudaStreamLike | None = None,
    ):
        """
        Construct a list Column from a gpumemoryview and array
        metadata, or wrap an existing Column in a nested list
        column matching the given shape.

        This non-public method does not perform validation. It assumes
        all arguments have been checked for correctness (e.g., shape,
        strides, size_type overflow) by a prior call to
        `_prepare_array_metadata`.
        """
        ndim = len(shape)
        flat_size = functools.reduce(operator.mul, shape)
        cdef Stream _stream = _get_stream(stream)

        if base is None:
            base = Column(
                data_type=dtype,
                size=flat_size,
                data=data,
                mask=None,
                null_count=0,
                offset=0,
                children=[],
            )

        int32_dtype = DataType(type_id.INT32)
        nested = base

        for i in range(ndim - 1, 0, -1):
            outer_len = functools.reduce(operator.mul, shape[:i])

            offsets_col = sequence(
                outer_len + 1,
                Scalar.from_py(0, int32_dtype, stream=_stream),
                Scalar.from_py(shape[i], int32_dtype, stream=_stream),
                _stream,
            )

            nested = Column(
                data_type=DataType(type_id.LIST),
                size=outer_len,
                data=None,
                mask=None,
                null_count=0,
                offset=0,
                children=[offsets_col, nested],
            )

        return nested

    @classmethod
    def from_array_interface(
        cls,
        obj: SupportsArrayInterface,
        object stream: CudaStreamLike | None = None,
    ) -> Column:
        """
        Create a Column from an object implementing the NumPy Array Interface.

        If the object provides a raw memory pointer via the "data" field,
        we use that pointer directly and avoid copying. Otherwise, a ValueError
        is raised.

        Parameters
        ----------
        obj : Any
            Must implement the ``__array_interface__`` protocol.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Column
            A Column containing the data from the array interface.

        Raises
        ------
        TypeError
            If the object does not implement ``__array_interface__``.
        ValueError
            If the array is not 1D or 2D, or is not C-contiguous.
            If the number of rows exceeds size_type limit.
            If the 'data' field is invalid.
        NotImplementedError
            If the object has a mask.
        """
        try:
            iface = obj.__array_interface__
        except AttributeError:
            raise TypeError("Object does not implement __array_interface__")

        data_ptr, nbytes, shape, _, dtype = _prepare_array_metadata(iface)

        cdef const unsigned char* ptr
        cdef const unsigned char[:] view
        cdef Stream _stream = _get_stream(stream)

        if nbytes > 0:
            ptr = <const unsigned char*><uintptr_t>data_ptr
            view = (<const unsigned char[:nbytes]> ptr)[:nbytes]
            dbuf = DeviceBuffer.to_device(view, _stream)
        else:
            dbuf = DeviceBuffer(size=0, stream=_stream)

        return Column._wrap_nested_list_column(
            gpumemoryview(dbuf), shape, dtype, None, _stream
        )

    @classmethod
    def from_cuda_array_interface(
        cls,
        obj: SupportsCudaArrayInterface,
        object stream: CudaStreamLike | None = None,
    ) -> Column:
        """
        Create a Column from an object implementing the CUDA Array Interface.

        Parameters
        ----------
        obj : Any
            Must implement the ``__cuda_array_interface__`` protocol.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Column
            A Column containing the data from the CUDA array interface.

        Raises
        ------
        TypeError
            If the object does not support ``__cuda_array_interface__``.
        ValueError
            If the object is not 1D or 2D, or is not C-contiguous.
            If the number of rows exceeds size_type limit.
        NotImplementedError
            If the object has a mask.
        """
        try:
            iface = obj.__cuda_array_interface__
        except AttributeError:
            raise TypeError("Object does not implement __cuda_array_interface__")

        _, _, shape, _, dtype = _prepare_array_metadata(iface)
        stream = _get_stream(stream)

        return Column._wrap_nested_list_column(
            gpumemoryview(obj), shape, dtype, None, stream
        )

    @classmethod
    def from_array(
        cls,
        obj: SupportsCudaArrayInterface | SupportsArrayInterface,
        object stream: CudaStreamLike | None = None,
    ) -> Column:
        """
        Create a Column from any object which supports the NumPy
        or CUDA array interface.

        Parameters
        ----------
        obj : object
            The input array to be converted into a `pylibcudf.Column`.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Column

        Raises
        ------
        TypeError
            If the input does not implement a supported array interface.

        Notes
        -----
        - Only C-contiguous host and device ndarrays are supported.
          For device arrays, the data is not copied.

        Examples
        --------
        >>> import pylibcudf as plc
        >>> import cupy as cp
        >>> cp_arr = cp.array([[1,2],[3,4]])
        >>> col = plc.Column.from_array(cp_arr)
        """
        if hasattr(obj, "__cuda_array_interface__"):
            return cls.from_cuda_array_interface(obj, stream=stream)
        if hasattr(obj, "__array_interface__"):
            return cls.from_array_interface(obj, stream=stream)

        raise TypeError(
            f"Cannot convert object of type {type(obj)} to a pylibcudf Column"
        )

    @staticmethod
    def from_iterable_of_py(
        obj: Iterable,
        dtype: DataType | None = None,
        object stream: CudaStreamLike | None = None
    ) -> Column:
        """
        Create a Column from a Python iterable of scalar values or nested iterables.

        Parameters
        ----------
        obj : Iterable
            An iterable of Python scalar values (int, float, bool, str) or nested lists.
        dtype : DataType | None
            The type of the leaf elements. If not specified, the type is inferred.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        Column
            A Column containing the data from the input iterable.

        Raises
        ------
        TypeError
            If the input contains unsupported scalar types.
        ValueError
            If the iterable is empty and dtype is not provided.

        Notes
        -----
        - Only scalar types int, float, bool, and str are supported.
        - Nested iterables must be materialized as lists.
        - Jagged nested lists are not supported. Inner lists must have the same shape.
        - Nulls (None) are not currently supported in input values.
        - dtype must match the inferred or actual type of the scalar values
        - Large strings are supported, meaning the combined length of all strings
          (in bytes) can exceed the maximum 32-bit integer value. In that case,
          the offsets column is automatically promoted to use 64-bit integers.
        """
        if not is_iterable(obj):
            raise ValueError(f"{obj=} is not iterable")

        if (
            hasattr(obj, "__cuda_array_interface__")
            or hasattr(obj, "__array_interface__")
        ):
            raise TypeError(
                "Object has __cuda_array_interface__ or __array_interface__. "
                "Please call Column.from_array(obj)."
            )

        if (
            hasattr(obj, "__arrow_c_array__")
            or hasattr(obj, "__arrow_c_device_array__")
            or hasattr(obj, "__arrow_c_stream__")
            or hasattr(obj, "__arrow_c_device_stream__")
        ):
            raise TypeError(
                "Object implements the Arrow C data interface protocol. "
                "Please call Column.from_arrow(obj)."
            )

        if not isinstance(obj, (list, tuple)):
            obj = list(obj)

        if not obj:
            if dtype is None:
                raise ValueError("Cannot infer dtype from empty iterable object")
            return Column(dtype, 0, None, None, 0, 0, [])

        if dtype is None:
            depth, py_dtype = _infer_list_depth_and_dtype(obj)
            dtype = DataType.from_py(py_dtype)
        else:
            depth, _ = _infer_list_depth_and_dtype(obj)

        flat, shape = _flatten_nested_list(obj, depth)

        if dtype.id() == type_id.STRING:
            encoded = [s.encode() for s in flat]
            offsets = [0] + list(accumulate(len(s) for s in encoded))

            offset_dtype = (
                DataType(type_id.INT64)
                if offsets[-1] > numeric_limits[size_type].max()
                else DataType(type_id.INT32)
            )

            offsets_data = _copy_array_to_device(
                array.array(offset_dtype._python_typecode, offsets),
                stream,
            )
            chars_data = _copy_array_to_device(
                array.array("B", b"".join(encoded)),
                stream,
            )

            offsets_col = Column(
                offset_dtype,
                len(offsets),
                offsets_data,
                None,
                0,
                0,
                [],
            )

            base = Column(
                DataType(type_id.STRING),
                len(flat),
                chars_data,
                None,
                0,
                0,
                [offsets_col],
            )

            return (
                base if depth == 1
                else Column._wrap_nested_list_column(
                    None, shape, dtype, base=base, stream=stream
                )
            )

        buf = array.array(dtype._python_typecode, flat)
        mv = memoryview(buf).cast("B")

        iface = {
            "data": (mv.obj.buffer_info()[0], False),
            "shape": shape,
            "typestr": dtype.typestr,
            "strides": None,
            "version": 3,
        }

        return Column.from_array_interface(ArrayInterfaceWrapper(iface), stream)

    cpdef list to_pylist(self):
        """
        Convert the Column to a Python list.

        Only supports fixed-width and string columns.

        Returns
        -------
        list
            A list of Python objects representing the column data.

        Raises
        ------
        NotImplementedError
            If the column type is not fixed-width or string.
        """
        cdef column_view c_self

        cdef type_id dtype = self.type().id()
        cdef bint large_offsets = (
            dtype == type_id.STRING
            and self.num_children() > 0
            and self.child(0).type().id() == type_id.INT64
        )
        cdef Stream _stream = _get_stream(None)
        cdef cudaStream_t _cs = _stream.view().value()
        cdef ArrowArray* raw_host_array_ptr = NULL
        c_self = self.view()
        with nogil:
            raw_host_array_ptr = to_arrow_host_raw(c_self, _cs)
        try:
            return _arrow_to_pylist_impl(dtype, raw_host_array_ptr, large_offsets)
        finally:
            if raw_host_array_ptr != NULL:
                with nogil:
                    release_arrow_array_raw(raw_host_array_ptr)

    @classmethod
    def struct_from_children(cls, children: Iterable[Column]) -> Column:
        """
        Create a struct Column from a list of child columns.

        Parameters
        ----------
        children : Iterable[Column]
            A list of child columns.

        Returns
        -------
        Column
            A struct Column with the provided the child columns.

        Notes
        -----
        The null count and null mask is taken from the first child column.
        Use `Column.with_mask` on the result of struct_from_children to reset
        the null count and mask.
        """
        if not isinstance(children, list):
            children = list(children)
        if len(children) == 0:
            raise ValueError("Must provide at least one child column")
        reference_child = children[0]
        if not all(
            isinstance(child, Column)
            and reference_child.size() == child.size()
            and reference_child.null_count() == child.null_count()
            # We assume the null masks are equivalent but may be expensive to
            # check: https://github.com/NVIDIA/cudf/pull/19357#issuecomment-3071033448
            for child in children
        ):
            raise ValueError(
                "All child columns must be of type Column and have the same size "
                "and null count."
            )

        return cls(
            DataType(type_id.STRUCT),
            reference_child.size(),
            None,
            reference_child.null_mask(),
            reference_child.null_count(),
            0,
            children,
        )

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef Column child(self, size_type index):
        """Get a child column of this column.

        Parameters
        ----------
        index : size_type
            The index of the child column to get.

        Returns
        -------
        Column
            The child column.
        """
        return self._children[index]

    cpdef size_type num_children(self):
        """The number of children of this column."""
        return self._num_children

    cpdef ListsColumnView list_view(self):
        """Accessor for methods of a Column that are specific to lists."""
        return ListsColumnView(self)

    cpdef StructsColumnView struct_view(self):
        """Accessor for methods of a Column that are specific to structs."""
        return StructsColumnView(self)

    cpdef object data(self):
        """The data buffer of the column."""
        return self._data

    cpdef object null_mask(self):
        """The null mask of the column."""
        return self._mask

    cpdef size_type size(self):
        """The number of elements in the column."""
        return self._size

    cpdef size_type offset(self):
        """The offset of the column."""
        return self._offset

    cpdef size_type null_count(self):
        """The number of null elements in the column."""
        return self._null_count

    cpdef void set_null_count(self, size_type null_count):
        """Set the null count of the column.

        Parameters
        ----------
        null_count : size_type
            The new null count of the column.
        """
        self._null_count = null_count

    cpdef list children(self):
        """The children of the column."""
        return self._children

    cpdef Column copy(self, object stream: CudaStreamLike | None = None, DeviceMemoryResource mr=None):
        """Create a copy of the column."""
        cdef unique_ptr[column] c_result
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()

        mr = _get_memory_resource(mr)
        cdef column_view c_self = self.view()
        with nogil:
            c_result = make_unique[column](c_self, _cs, mr.get_mr())
        return Column.from_libcudf(move(c_result), _stream, mr)

    cpdef uint64_t device_buffer_size(self):
        """
        The total size of the device buffers used by the Column.

        Notes
        -----
        Since Columns rely on Python memoryview-like semantics to maintain
        shared ownership of the data, the device buffers underlying this column
        might be shared between other data structures including other columns.

        Returns
        -------
        Number of bytes.
        """
        cdef uint64_t ret = 0
        if self.data() is not None:
            ret += self.data().nbytes
        if self.null_mask() is not None:
            ret += self.null_mask().nbytes
        if self.children() is not None:
            for child in self.children():
                ret += (<Column?>child).device_buffer_size()
        return ret

    def _create_nested_column_metadata(self):
        return ColumnMetadata(
            children_meta=[
                child._create_nested_column_metadata() for child in self.children()
            ]
        )

    def _to_schema(self, metadata=None):
        """Create an Arrow schema from this Column."""

        cdef column_view c_self

        if metadata is None:
            metadata = self._create_nested_column_metadata()
        elif isinstance(metadata, str):
            metadata = ColumnMetadata(metadata)

        cdef column_metadata c_metadata = _metadata_to_libcudf(metadata)

        cdef ArrowSchema* raw_schema_ptr
        c_self = self.view()
        with nogil:
            raw_schema_ptr = to_arrow_schema_raw(c_self, c_metadata)

        return PyCapsule_New(<void*>raw_schema_ptr, 'arrow_schema', _release_schema)

    def _to_host_array(self, object stream: CudaStreamLike):
        cdef ArrowArray* raw_host_array_ptr
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()

        cdef column_view c_self = self.view()
        with nogil:
            raw_host_array_ptr = to_arrow_host_raw(c_self, _cs)

        return PyCapsule_New(<void*>raw_host_array_ptr, "arrow_array", _release_array)

    def _to_device_array(self):
        cdef ArrowDeviceArray* raw_device_array_ptr

        cdef column_view c_self = self.view()
        with nogil:
            raw_device_array_ptr = to_arrow_device_raw(c_self, self)

        return PyCapsule_New(
            <void*>raw_device_array_ptr,
            "arrow_device_array",
            _release_device_array
        )

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Column does not support alternative schema")

        return self._to_schema(), self._to_host_array(_get_stream(None))

    def __arrow_c_device_array__(self, requested_schema=None, **kwargs):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Column does not support alternative schema")

        non_default_kwargs = [
            name for name, value in kwargs.items() if value is not None
        ]
        if non_default_kwargs:
            raise NotImplementedError(
                f"Received unsupported keyword argument(s): {non_default_kwargs}"
            )

        return self._to_schema(), self._to_device_array()


cdef class ListsColumnView:
    """Accessor for methods of a Column that are specific to lists."""
    def __init__(self, Column col):
        if col.type().id() != type_id.LIST:
            raise TypeError("Column is not a list type")
        self._column = col

    __hash__ = None

    cpdef child(self):
        """The data column of the underlying list column."""
        return self._column.child(1)

    cpdef offsets(self):
        """The offsets column of the underlying list column."""
        return self._column.child(0)

    cdef lists_column_view view(self):
        """Generate a libcudf lists_column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        return lists_column_view(self._column.view())

    cpdef Column get_sliced_child(self, object stream: CudaStreamLike | None = None):
        """
        Get the list elements child properly sliced to match parent's view.

        Parameters
        ----------
        stream : Stream, optional
            CUDA stream to use

        Returns
        -------
        Column
            The sliced elements column
        """
        cdef Stream _stream = _get_stream(stream)

        cdef column_view c_child = self.view().get_sliced_child(_stream.view().value())
        return Column.from_column_view(c_child, self._column.child(1))


cdef class StructsColumnView:
    """Accessor for methods of a Column that are specific to structs."""
    def __init__(self, Column col):
        if col.type().id() != type_id.STRUCT:
            raise TypeError("Column is not a struct type")
        self._column = col

    __hash__ = None

    cdef structs_column_view view(self):
        """Generate a libcudf structs_column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        return structs_column_view(self._column.view())

    cpdef Column get_sliced_child(self, int index, object stream: CudaStreamLike | None = None):
        """
        Get the struct elements child properly sliced to match parent's view.

        Parameters
        ----------
        index : int
            The index of the child to get.
        stream : Stream, optional
            CUDA stream to use

        Returns
        -------
        Column
            The sliced elements column
        """
        cdef Stream _stream = _get_stream(stream)

        cdef cudaStream_t _cs = _stream.view().value()
        cdef column_view c_child = self.view().get_sliced_child(index, _cs)
        return Column.from_column_view(c_child, self._column.child(index))


def is_c_contiguous(
    shape: Sequence[int], strides: None | Sequence[int], itemsize: int
) -> bool:
    """Determine if shape and strides are C-contiguous

    Parameters
    ----------
    shape : Sequence[int]
        Number of elements in each dimension.
    strides : None | Sequence[int]
        The stride of each dimension in bytes.
        If None, the memory layout is C-contiguous.
    itemsize : int
        Size of an element in bytes.

    Returns
    -------
    bool
        The boolean answer.
    """

    if strides is None:
        return True
    if any(dim == 0 for dim in shape):
        return True
    cumulative_stride = itemsize
    for dim, stride in zip(reversed(shape), reversed(strides)):
        if dim > 1 and stride != cumulative_stride:
            return False
        cumulative_stride *= dim
    return True


cdef extern from *:
    """
    #include <Python.h>
    #include <cstdint>
    #include <type_traits>
    #include <cudf/types.hpp>
    #include <cudf/utilities/type_dispatcher.hpp>

    // See https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps
    // for more details on how the validity bitmap works.
    static inline bool _is_valid(const uint8_t* bm, size_t idx) {
      if (!bm) return true;
      return ((bm[idx >> 3] >> (idx & 7)) & 1) != 0;
    }

    // Helper: PyList_SetItem steals a reference, so we incref
    // borrowed singletons before calling it.
    // https://docs.python.org/3/c-api/list.html#c.PyList_SetItem
    static inline void _set_item(PyObject* list, Py_ssize_t i, PyObject* obj) {
      Py_INCREF(obj);
      PyList_SetItem(list, i, obj);
    }

    template <bool check_valid, typename T>
    static inline PyObject* _make_pylist(int64_t n,
                                         int64_t offset,
                                         const uint8_t* null_bm,
                                         const T& conv)
    {
      auto out = PyList_New(n);
      if (!out) return nullptr;
      for (int64_t i = 0; i < n; ++i) {
        auto bit_index = static_cast<size_t>(offset + i);
        if constexpr (check_valid) {
          if (!_is_valid(null_bm, bit_index)) {
            // TODO: Py_INCREF(Py_None) unnecessary once Python<3.12 is
            // dropped (Py_None is immortal in 3.12+).
            _set_item(out, i, Py_None);
            continue;
          }
        }
        auto obj = conv(i, bit_index);
        if (!obj) {
          Py_DECREF(out);
          return nullptr;
        }
        PyList_SetItem(out, i, obj);
      }
      return out;
    }

    // Maps supported type_ids to their C++ types, unsupported ones to void.
    // This limits type_dispatcher instantiations to only the types we handle.
    template <cudf::type_id Id>
    struct dispatch_pylist_type {
      using mapped = typename cudf::id_to_type_impl<Id>::type;
      using type   = std::conditional_t<std::is_arithmetic_v<mapped> ||
                                           std::is_same_v<mapped, cudf::string_view>,
                                         mapped,
                                         void>;
    };

    template <bool check_valid>
    struct pylist_functor {
      ArrowArray* arr;
      bool large_offsets;

      template <typename T, std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
      PyObject* operator()() const { return nullptr; }

      template <typename T, std::enable_if_t<std::is_same_v<T, bool>>* = nullptr>
      PyObject* operator()() const
      {
        auto null_bm = check_valid
                          ? static_cast<const uint8_t*>(arr->buffers[0]) : nullptr;
        return _make_pylist<check_valid>(
          arr->length, arr->offset, null_bm,
          [bools = static_cast<const uint8_t*>(arr->buffers[1])]
          (int64_t, size_t bit_index) -> PyObject* {
            bool val = ((bools[bit_index >> 3] >> (bit_index & 7)) & 1) != 0;
            // TODO: Py_INCREF unnecessary for Py_True/Py_False once
            // Python<3.12 is dropped (immortal in 3.12+).
            // See: https://docs.python.org/3/c-api/bool.html
            auto obj = val ? Py_True : Py_False;
            Py_INCREF(obj);
            return obj;
          });
      }

      // For strings, Arrow uses offsets (buffers[1]) + char data (buffers[2]).
      // Offsets may be int32_t (standard) or int64_t (large strings > 2GB).
      template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
      PyObject* operator()() const
      {
        auto null_bm = check_valid
                          ? static_cast<const uint8_t*>(arr->buffers[0]) : nullptr;
        auto str_base = static_cast<const char*>(arr->buffers[2]);
        if (large_offsets) {
          auto str_offsets = static_cast<const int64_t*>(arr->buffers[1]) + arr->offset;
          return _make_pylist<check_valid>(
            arr->length, arr->offset, null_bm,
            [str_offsets, str_base](int64_t i, size_t) -> PyObject* {
              return PyUnicode_DecodeUTF8(
                str_base + str_offsets[i], str_offsets[i + 1] - str_offsets[i], nullptr);
            });
        } else {
          auto str_offsets = static_cast<const int32_t*>(arr->buffers[1]) + arr->offset;
          return _make_pylist<check_valid>(
            arr->length, arr->offset, null_bm,
            [str_offsets, str_base](int64_t i, size_t) -> PyObject* {
              return PyUnicode_DecodeUTF8(
                str_base + str_offsets[i], str_offsets[i + 1] - str_offsets[i], nullptr);
            });
        }
      }

      template <typename T,
                std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>>* = nullptr>
      PyObject* operator()() const
      {
        auto null_bm = check_valid
                          ? static_cast<const uint8_t*>(arr->buffers[0]) : nullptr;
        auto base = static_cast<const T*>(arr->buffers[1]) + arr->offset;
        return _make_pylist<check_valid>(
          arr->length, arr->offset, null_bm,
          [base](int64_t i, size_t) -> PyObject* {
            auto v = base[i];
            if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
              return PyLong_FromLongLong(static_cast<long long>(v));
            } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
              return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(v));
            } else {
              return PyFloat_FromDouble(static_cast<double>(v));
            }
          });
      }
    };

    static inline PyObject* cpp_arrow_to_pylist_impl(
      cudf::type_id dtype, ArrowArray* arr, bool large_offsets)
    {
      if (arr->length == 0) return PyList_New(0);

      if (arr->null_count != 0) {
        return cudf::type_dispatcher<dispatch_pylist_type>(
          cudf::data_type{dtype}, pylist_functor<true>{arr, large_offsets});
      } else {
        return cudf::type_dispatcher<dispatch_pylist_type>(
          cudf::data_type{dtype}, pylist_functor<false>{arr, large_offsets});
      }
    }
    """
    PyObject* cpp_arrow_to_pylist_impl(
        type_id dtype, ArrowArray* arr, bint large_offsets)

cdef list _arrow_to_pylist_impl(type_id dtype, ArrowArray* arr, bint large_offsets):
    cdef PyObject* list_ = cpp_arrow_to_pylist_impl(dtype, arr, large_offsets)
    if list_ == NULL:
        if PyErr_Occurred():
            raise
        raise NotImplementedError(
            f"Column with dtype={TypeId(dtype)} not supported"
        )
    return <list>list_
