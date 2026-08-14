# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int32_t
from pylibcudf.libcudf.types cimport (
    data_type,
    size_of as cpp_size_of,
    size_type,
    type_id,
    bitmask_type
)
from pylibcudf.libcudf.utilities.type_dispatcher cimport type_to_id

from pylibcudf.libcudf.types import type_id as TypeId  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import nan_policy as NanPolicy  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import null_policy as NullPolicy  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import interpolation as Interpolation  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import mask_state as MaskState  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import nan_equality as NanEquality  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import null_equality as NullEquality  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import null_aware as NullAware  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import output_nullability as OutputNullability  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import null_order as NullOrder  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import order as Order  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import sorted as Sorted  # no-cython-lint, isort:skip

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

try:
    import pyarrow as pa

    pa_err = None

    ARROW_TO_PYLIBCUDF_TYPES = {
        pa.int8(): type_id.INT8,
        pa.int16(): type_id.INT16,
        pa.int32(): type_id.INT32,
        pa.int64(): type_id.INT64,
        pa.uint8(): type_id.UINT8,
        pa.uint16(): type_id.UINT16,
        pa.uint32(): type_id.UINT32,
        pa.uint64(): type_id.UINT64,
        pa.float32(): type_id.FLOAT32,
        pa.float64(): type_id.FLOAT64,
        pa.bool_(): type_id.BOOL8,
        pa.string(): type_id.STRING,
        pa.large_string(): type_id.STRING,
        pa.string_view(): type_id.STRING,
        pa.duration('s'): type_id.DURATION_SECONDS,
        pa.duration('ms'): type_id.DURATION_MILLISECONDS,
        pa.duration('us'): type_id.DURATION_MICROSECONDS,
        pa.duration('ns'): type_id.DURATION_NANOSECONDS,
        pa.timestamp('s'): type_id.TIMESTAMP_SECONDS,
        pa.timestamp('ms'): type_id.TIMESTAMP_MILLISECONDS,
        pa.timestamp('us'): type_id.TIMESTAMP_MICROSECONDS,
        pa.timestamp('ns'): type_id.TIMESTAMP_NANOSECONDS,
        pa.date32(): type_id.TIMESTAMP_DAYS,
        pa.null(): type_id.EMPTY,
    }

    LIBCUDF_TO_ARROW_TYPES = {
        v: k for k, v in ARROW_TO_PYLIBCUDF_TYPES.items()
    }
    # Because we map 3 pyarrow string types to type_id.STRING,
    # just map type_id.STRING to pa.string
    LIBCUDF_TO_ARROW_TYPES[type_id.STRING] = pa.string()
except ImportError as e:
    pa = None
    pa_err = e
    ARROW_TO_PYLIBCUDF_TYPES = {}
    LIBCUDF_TO_ARROW_TYPES = {}


__all__ = [
    "DataType",
    "Interpolation",
    "MaskState",
    "NanEquality",
    "NanPolicy",
    "NullEquality",
    "NullOrder",
    "NullAware",
    "OutputNullability",
    "NullPolicy",
    "Order",
    "SIZE_TYPE",
    "SIZE_TYPE_ID",
    "Sorted",
    "TypeId",
    "size_of"
]

cdef class DataType:
    """Indicator for the logical data type of an element in a column.

    This is the Cython representation of :cpp:class:`cudf::data_type`.

    Parameters
    ----------
    id : type_id
        The type's identifier
    scale : int
        The scale associated with the data. Only used for decimal data types.
    """
    def __cinit__(self, type_id id, int32_t scale=0):
        if (
            id == type_id.DECIMAL32
            or id == type_id.DECIMAL64
            or id == type_id.DECIMAL128
        ):
            self.c_obj = data_type(id, scale)
        else:
            self.c_obj = data_type(id)

    # TODO: Consider making both id and scale cached properties.
    cpdef type_id id(self):
        """Get the id associated with this data type."""
        return self.c_obj.id()

    cpdef int32_t scale(self):
        """Get the scale associated with this data type."""
        return self.c_obj.scale()

    @property
    def _python_typecode(self) -> str:
        """The Python struct module typecode string."""
        try:
            return {
                type_id.INT8: 'b',
                type_id.INT16: 'h',
                type_id.INT32: 'i',
                type_id.INT64: 'q',
                type_id.UINT8: 'B',
                type_id.UINT16: 'H',
                type_id.UINT32: 'I',
                type_id.UINT64: 'Q',
                type_id.FLOAT32: 'f',
                type_id.FLOAT64: 'd',
                type_id.BOOL8: 'b',
            }[self.id()]
        except KeyError:
            raise NotImplementedError(
                f"No Python typecode for DataType {self.id()}"
            )

    @property
    def typestr(self) -> str:
        """The array interface type string."""
        try:
            return {
                type_id.INT8: "|i1",
                type_id.INT16: "<i2",
                type_id.INT32: "<i4",
                type_id.INT64: "<i8",
                type_id.UINT8: "|u1",
                type_id.UINT16: "<u2",
                type_id.UINT32: "<u4",
                type_id.UINT64: "<u8",
                type_id.FLOAT32: "<f4",
                type_id.FLOAT64: "<f8",
                type_id.BOOL8: "|b1",
            }[self.id()]
        except KeyError:
            raise NotImplementedError(
                f"No array interface typestr for DataType {self.id()}"
            )

    def __eq__(self, other):
        return type(self) is type(other) and (
            self.c_obj == (<DataType>other).c_obj
        )

    def __hash__(self):
        return hash((self.c_obj.id(), self.c_obj.scale()))

    def __reduce__(self):
        return (type(self), (self.c_obj.id(), self.c_obj.scale()))

    @staticmethod
    cdef DataType from_libcudf(data_type dt):
        """Create a DataType from a libcudf data_type.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # Spoof an empty data type then swap in the real one.
        cdef DataType ret = DataType.__new__(DataType, type_id.EMPTY)
        ret.c_obj = dt
        return ret

    def to_arrow(self, **kwargs) -> pa.DataType:
        """
        Convert a datatype to arrow.

        Returns
        -------
        pyarrow.DataType

        Notes
        -----
        Translation of some types requires extra information as a keyword
        argument. Specifically:

        - When translating a decimal type, provide ``precision``
        - When translating a struct type, provide ``fields``
        - When translating a list type, provide the wrapped ``value_type``
        """
        if pa_err is not None:
            raise RuntimeError(
                "pyarrow was not found on your system. Please "
                "pip install pylibcudf with the [pyarrow] extra for a "
                "compatible pyarrow version."
            ) from pa_err
        if self.id() in {
            type_id.DECIMAL32,
            type_id.DECIMAL64,
            type_id.DECIMAL128
        }:
            if not (precision := kwargs.get("precision")):
                raise ValueError(
                    "Precision must be provided for decimal types"
                )
                # no pa.decimal32 or pa.decimal64
            return pa.decimal128(precision, -self.scale())
        elif self.id() == type_id.STRUCT:
            if not (fields := kwargs.get("fields")):
                raise ValueError(
                    "Fields must be provided for struct types"
                )
            return pa.struct(fields)
        elif self.id() == type_id.LIST:
            if not (value_type := kwargs.get("value_type")):
                raise ValueError(
                    "Value type must be provided for list types"
                )
            return pa.list_(value_type)
        else:
            try:
                return LIBCUDF_TO_ARROW_TYPES[self.id()]
            except KeyError:
                raise TypeError(
                    f"Unable to convert {self.id()} to arrow datatype"
                )

    @staticmethod
    def from_arrow(pa_typ: pa.DataType) -> DataType:
        """
        Construct a DataType from a Python type.

        Parameters
        ----------
        pa_typ : pyarrow type
            A Pyarrow type (eg. pa)

        Returns
        -------
        DataType
            The corresponding pylibcudf DataType.

        Raises
        ------
        ImportError
            If pyarrow is not installed.
        TypeError
            If the Python type is not supported.
        """
        return _from_arrow(pa_typ)

    @staticmethod
    def from_py(typ: type) -> DataType:
        """
        Construct a DataType from a Python type.

        Parameters
        ----------
        typ : type
            A Python type (eg. int, str, list)

        Returns
        -------
        DataType
            The corresponding pylibcudf DataType.

        Raises
        ------
        TypeError
            If the Python type is not supported.
        """
        if typ is bool:
            return DataType(type_id.BOOL8)
        elif typ is int:
            return DataType(type_id.INT64)
        elif typ is float:
            return DataType(type_id.FLOAT64)
        elif typ is str:
            return DataType(type_id.STRING)
        elif typ is list:
            return DataType(type_id.LIST)
        elif typ is dict:
            return DataType(type_id.STRUCT)
        else:
            raise TypeError(f"Cannot infer DataType from Python type {typ}")

cpdef size_t size_of(DataType t):
    """Returns the size in bytes of elements of the specified data_type.

    Only fixed-width types are supported.

    For details, see :cpp:func:`size_of`.

    Parameters
    ----------
    t : DataType
        The DataType to get the size of.

    Returns
    -------
    int
        Size in bytes of an element of the specified type.
    """
    with nogil:
        return cpp_size_of(t.c_obj)


@cache
def _from_arrow(obj: pa.DataType) -> DataType:
    if pa_err is not None:
        raise RuntimeError(
            "pyarrow was not found on your system. Please "
            "pip install pylibcudf with the [pyarrow] extra for a "
            "compatible pyarrow version."
        ) from pa_err
    if (
        getattr(pa, "Decimal32Type", None) is not None
        and isinstance(obj, pa.Decimal32Type)
    ):
        return DataType(type_id.DECIMAL32, scale=-obj.scale)
    if (
        getattr(pa, "Decimal64Type", None) is not None
        and isinstance(obj, pa.Decimal64Type)
    ):
        return DataType(type_id.DECIMAL64, scale=-obj.scale)
    if isinstance(obj, pa.Decimal128Type):
        return DataType(type_id.DECIMAL128, scale=-obj.scale)
    elif isinstance(obj, pa.StructType):
        # Recurse to catch unsupported field types
        for field in obj:
            _from_arrow(field.type)
        return DataType(type_id.STRUCT)
    elif isinstance(obj, pa.ListType):
        # Recurse to catch unsupported inner types
        _from_arrow(obj.value_type)
        return DataType(type_id.LIST)
    else:
        try:
            return DataType(ARROW_TO_PYLIBCUDF_TYPES[obj])
        except KeyError:
            raise TypeError(f"Unable to convert {obj} to cudf datatype")


SIZE_TYPE = DataType(type_to_id[size_type]())
SIZE_TYPE_ID = SIZE_TYPE.id()
BITMASK_TYPE = DataType(type_to_id[bitmask_type]())


TypeId.__str__ = TypeId.__repr__
NanPolicy.__str__ = NanPolicy.__repr__
NullPolicy.__str__ = NullPolicy.__repr__
Interpolation.__str__ = Interpolation.__repr__
MaskState.__str__ = MaskState.__repr__
NanEquality.__str__ = NanEquality.__repr__
NullEquality.__str__ = NullEquality.__repr__
NullAware.__str__ = NullAware.__repr__
OutputNullability.__str__ = OutputNullability.__repr__
NullOrder.__str__ = NullOrder.__repr__
Order.__str__ = Order.__repr__
Sorted.__str__ = Sorted.__repr__
