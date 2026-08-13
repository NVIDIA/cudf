# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_New,
)

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowArrayStream,
    ArrowDeviceArray,
    ArrowSchema,
    arrow_table,
    column_metadata,
    to_arrow_device_raw,
    to_arrow_host_raw,
    to_arrow_schema_raw,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource
from pylibcudf._interop_helpers cimport (
    _release_schema,
    _release_array,
    _release_device_array,
    _metadata_to_libcudf,
)
from ._interop_helpers import ArrowLike, ColumnMetadata, _ObjectWithArrowMetadata
from cuda.bindings.cyruntime cimport cudaStream_t

try:
    import pyarrow as pa
    pa_err = None
except ImportError as e:
    pa = None
    pa_err = e


__all__ = ["Table"]


cdef class _ArrowTableHolder:
    """A holder for an Arrow table for gpumemoryview lifetime management."""
    cdef unique_ptr[arrow_table] tbl
    cdef DeviceMemoryResource mr


cdef class Table:
    """A list of columns of the same size.

    If the list of columns is empty, the table's row count may still be non-zero.

    Parameters
    ----------
    columns : Sequence[Column]
        The columns in this table.
    num_rows : int | None
        Optional explicit row count. Only used to preserve the row count of a
        table with zero columns. When `columns` is non-empty, `num_rows` must
        equal the size of every column.
    """
    __hash__ = None

    def __init__(self, columns, num_rows=None):
        columns = tuple(columns)
        if not all(isinstance(c, Column) for c in columns):
            raise ValueError("All columns must be pylibcudf Column objects")
        self._columns = columns
        if num_rows is None:
            self._num_rows = columns[0].size() if len(columns) else 0
        else:
            if not isinstance(num_rows, int):
                raise TypeError("num_rows must be an int or None")
            if num_rows < 0:
                raise ValueError("num_rows cannot be negative")
            if any(c.size() != num_rows for c in columns):
                raise ValueError(
                    "num_rows does not match the size of the provided columns"
                )
            self._num_rows = num_rows

    def to_arrow(
        self,
        metadata: list[ColumnMetadata | str] | None = None,
        stream: Stream | None = None,
    ) -> ArrowLike:
        """Create a pyarrow table from a pylibcudf table.

        Parameters
        ----------
        metadata : list[ColumnMetadata | str] | None
            The metadata to attach to the columns of the table.
        stream : Stream | None
            CUDA stream on which to perform the operation.

        Returns
        -------
        pyarrow.Table
        """
        if pa_err is not None:
            raise RuntimeError(
                "pyarrow was not found on your system. Please "
                "pip install pylibcudf with the [pyarrow] extra for a "
                "compatible pyarrow version."
            ) from pa_err
        # TODO: Once the arrow C device interface registers more
        # types that it supports, we can call pa.table(self) if
        # no metadata is passed.
        return pa.table(_ObjectWithArrowMetadata(self, metadata, stream))

    @staticmethod
    def from_arrow(
        obj: ArrowLike,
        dtype: DataType | None = None,
        object stream=None,
        DeviceMemoryResource mr=None
    ) -> Table:
        """
        Create a Table from an Arrow-like object using the Arrow C data interface.

        This method supports constructing a `pylibcudf.Table` from an object that
        implements one of the Arrow C data interface protocols, such as:

        - `__arrow_c_device_array__`: Returns a tuple of (ArrowSchema, ArrowDeviceArray)
          representing columnar device memory.
        - `__arrow_c_stream__`: Returns an ArrowArrayStream pointer representing
          a stream of host columnar batches.
        - `__arrow_c_device_stream__`: Not yet implemented.
        - `__arrow_c_array__`: Not yet implemented.

        Parameters
        ----------
        obj : Arrow-like type
            An object implementing one of the Arrow C data interface methods.
        dtype: DataType
            The pylibcudf data type.
        stream : Stream | None
            CUDA stream on which to perform the operation.
        mr : DeviceMemoryResource | None
            Device memory resource for allocations.

        Returns
        -------
        Table
            A Table constructed from the Arrow-like input.

        Raises
        ------
        NotImplementedError
            If the input is a device or host stream not yet supported.
            If the dtype argument is not None.
        ValueError
            If the input does not implement a supported Arrow C interface.
        """
        if dtype is not None:
            raise NotImplementedError(
                "Creating a Table with the dtype argument specified."
            )
        cdef ArrowSchema* c_schema
        cdef ArrowDeviceArray* c_array
        cdef _ArrowTableHolder result
        cdef unique_ptr[arrow_table] c_result

        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)

        if hasattr(obj, "__arrow_c_device_array__"):
            schema, array = obj.__arrow_c_device_array__()
            c_schema = <ArrowSchema*>PyCapsule_GetPointer(schema, "arrow_schema")
            c_array = (
                <ArrowDeviceArray*>PyCapsule_GetPointer(array, "arrow_device_array")
            )

            result = _ArrowTableHolder()
            result.mr = mr
            with nogil:
                c_result = make_unique[arrow_table](
                    move(dereference(c_schema)),
                    move(dereference(c_array)),
                    _cs,
                    result.mr.get_mr(),
                )
            result.tbl.swap(c_result)

            return Table.from_table_view_of_arbitrary(
                result.tbl.get().view(),
                result,
                stream,
            )
        elif hasattr(obj, "__arrow_c_stream__"):
            arrow_stream = obj.__arrow_c_stream__()
            c_stream = (
                <ArrowArrayStream*>PyCapsule_GetPointer(
                    arrow_stream, "arrow_array_stream"
                )
            )

            result = _ArrowTableHolder()
            result.mr = mr
            with nogil:
                c_result = make_unique[arrow_table](
                    move(dereference(c_stream)),
                    _cs,
                    result.mr.get_mr(),
                )
            result.tbl.swap(c_result)

            return Table.from_table_view_of_arbitrary(
                result.tbl.get().view(),
                result,
                stream,
            )
        elif hasattr(obj, "__arrow_c_device_stream__"):
            # TODO: When we add support for this case, it should be moved above
            # the __arrow_c_stream__ case since we should prioritize device
            # data if possible.
            raise NotImplementedError("Device streams not yet supported")
        elif hasattr(obj, "__arrow_c_array__"):
            raise NotImplementedError("Arrow host arrays not yet supported")
        else:
            raise ValueError("Invalid Arrow-like object")

    cdef table_view view(self):
        """Generate a libcudf table_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # TODO: Make c_columns a class attribute that is updated along with
        # self._columns whenever new columns are added or columns are removed.
        cdef vector[column_view] c_columns

        for col in self._columns:
            c_columns.push_back((<Column> col).view())

        return table_view(c_columns, self.num_rows())

    @staticmethod
    cdef Table from_libcudf(
        unique_ptr[table] libcudf_tbl,
        object stream,
        DeviceMemoryResource mr
    ):
        """Create a Table from a libcudf table.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        assert stream is not None, "stream cannot be None"
        assert mr is not None, "mr cannot be None"
        # Capture the row count before release() (which zeroes it) so a
        # zero-column table preserves its rows.
        cdef size_type nrows = dereference(libcudf_tbl).num_rows()
        cdef vector[unique_ptr[column]] c_columns = dereference(libcudf_tbl).release()

        cdef vector[unique_ptr[column]].size_type i
        cols = [
            Column.from_libcudf(move(c_columns[i]), stream, mr)
            for i in range(c_columns.size())
        ]
        return Table(cols, num_rows=nrows)

    @staticmethod
    cdef Table from_table_view(const table_view& tv, Table owner):
        """Create a Table from a libcudf table_view into a Table owner.

        This method accepts shared ownership of the underlying data from the owner.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef int i
        return Table(
            [
                Column.from_column_view(tv.column(i), owner.columns()[i])
                for i in range(tv.num_columns())
            ],
            num_rows=tv.num_rows(),
        )

    # Ideally this function would simply be handled via a fused type in
    # from_table_view, but this does not work due to
    # https://github.com/cython/cython/issues/6740
    @staticmethod
    cdef Table from_table_view_of_arbitrary(
        const table_view& tv,
        object owner,
        object stream,
    ):
        """Create a Table from a libcudf table_view into an arbitrary owner.

        This method accepts shared ownership of the underlying data from the owner.
        Since the owner may be any arbitrary object, every buffer view that is part of
        the table (each column and all of their children) share ownership of the same
        buffer since they do not have the information available to choose to only own
        subsets of it.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # For efficiency, prohibit calling this overload with a Table owner.
        assert not isinstance(owner, Table)
        cdef int i
        cdef Stream _stream = <Stream>stream
        return Table(
            [
                Column.from_column_view_of_arbitrary(tv.column(i), owner, _stream)
                for i in range(tv.num_columns())
            ],
            num_rows=tv.num_rows(),
        )

    cpdef int num_columns(self):
        """The number of columns in this table."""
        return len(self._columns)

    cpdef int num_rows(self):
        """The number of rows in this table."""
        return self._num_rows

    cpdef tuple columns(self):
        """The columns in this table."""
        return self._columns

    cpdef list release(self):
        """Release ownership of this table's columns and leave it empty.

        Returns
        -------
        list
            The columns that were in this table.
        """
        cdef list columns = list(self._columns)
        self._columns = ()
        self._num_rows = 0
        return columns

    cpdef tuple shape(self):
        """The shape of this table"""
        return (self.num_rows(), self.num_columns())

    cpdef Table copy(self, object stream=None, DeviceMemoryResource mr=None):
        """Create a deep copy of the table.

        Parameters
        ----------
        stream : Stream | None
            CUDA stream on which to perform the operation.
        mr : DeviceMemoryResource | None
            Device memory resource for allocations.

        Returns
        -------
        Table
            A new Table with deep copies of all columns.
        """
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        return Table(
            [col.copy(_stream, mr) for col in self._columns],
            num_rows=self.num_rows(),
        )

    def _to_schema(self, metadata=None):
        """Create an Arrow schema from this table."""
        if metadata is None:
            metadata = [
                col._create_nested_column_metadata() for col in self.columns()
            ]
        else:
            metadata = [
                ColumnMetadata(m) if isinstance(m, str) else m for m in metadata
            ]

        cdef vector[column_metadata] c_metadata
        c_metadata.reserve(len(metadata))
        for meta in metadata:
            c_metadata.push_back(_metadata_to_libcudf(meta))

        cdef ArrowSchema* raw_schema_ptr
        cdef table_view c_self = self.view()
        with nogil:
            raw_schema_ptr = to_arrow_schema_raw(c_self, c_metadata)

        return PyCapsule_New(<void*>raw_schema_ptr, "arrow_schema", _release_schema)

    def _to_host_array(self, object stream):
        cdef ArrowArray* raw_host_array_ptr
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        cdef table_view c_self = self.view()

        with nogil:
            raw_host_array_ptr = to_arrow_host_raw(c_self, _cs)

        return PyCapsule_New(<void*>raw_host_array_ptr, "arrow_array", _release_array)

    def _to_device_array(self):
        cdef ArrowDeviceArray* raw_device_array_ptr
        cdef table_view c_self = self.view()
        with nogil:
            raw_device_array_ptr = to_arrow_device_raw(c_self, self)

        return PyCapsule_New(
            <void*>raw_device_array_ptr,
            "arrow_device_array",
            _release_device_array
        )

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Table does not support alternative schema")

        return self._to_schema(), self._to_host_array(_get_stream(None))

    def __arrow_c_device_array__(self, requested_schema=None, **kwargs):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Table does not support alternative schema")

        non_default_kwargs = [
            name for name, value in kwargs.items() if value is not None
        ]
        if non_default_kwargs:
            raise NotImplementedError(
                f"Received unsupported keyword argument(s): {non_default_kwargs}"
            )

        return self._to_schema(), self._to_device_array()
