# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import decimal
import inspect
import operator
import re
import textwrap
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api import types as pd_types  # noqa: TID251
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pylibcudf as plc

import cudf
from cudf.core.abc import Serializable
from cudf.core.dtype.validators import (
    is_dtype_obj_datetime_tz,
    is_dtype_obj_interval,
    is_dtype_obj_list,
    is_dtype_obj_numeric,
    is_dtype_obj_string,
    is_dtype_obj_struct,
)
from cudf.utils.docutils import doc_apply
from cudf.utils.dtypes import (
    DEFAULT_STRING_DTYPE,
    SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    min_unsigned_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Self

    from cudf._typing import Dtype, DtypeObj
    from cudf.core.buffer import Buffer
    from cudf.core.column.column import ColumnBase
    from cudf.core.index import Index


PANDAS_NUMPY_DTYPE = pd.core.dtypes.dtypes.NumpyEADtype


def dtype(arbitrary: Any) -> DtypeObj:
    """
    Return the cuDF-supported dtype corresponding to `arbitrary`.

    This function should only be used when converting a dtype
    provided from a public API user (i.e. not internally).

    Parameters
    ----------
    arbitrary: dtype or scalar-like

    Returns
    -------
    dtype: the cuDF-supported dtype that best matches `arbitrary`
    """
    #  first, check if `arbitrary` is one of our extension types:
    if isinstance(arbitrary, _BaseDtype):
        return arbitrary
    if inspect.isclass(arbitrary) and issubclass(
        arbitrary, pd.api.extensions.ExtensionDtype
    ):
        msg = (
            f"Expected an instance of {arbitrary.__name__}, "
            "but got the class instead. Try instantiating 'dtype'."
        )
        raise TypeError(msg)

    if arbitrary is str or (isinstance(arbitrary, str) and arbitrary == "str"):
        # "str" -> pd.StringDtype
        # str -> pd.StringDtype
        return pd.api.types.pandas_dtype(arbitrary)  # noqa: TID251

    # next, try interpreting arbitrary as a NumPy dtype that we support:
    try:
        np_dtype = np.dtype(arbitrary)
    except TypeError:
        pass
    else:
        if np_dtype.kind == "U":
            # Like pandas, allow users to pass this object to signal "string"
            # but the dtype metadata should result in np.dtype(object)
            return np_dtype
        elif np_dtype not in SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES:
            raise TypeError(f"Unsupported type {np_dtype}")
        return np_dtype

    # lastly, try interpreting arbitrary as a pandas dtype that we support:
    pd_dtype = pd.api.types.pandas_dtype(arbitrary)  # noqa: TID251
    if isinstance(pd_dtype, PANDAS_NUMPY_DTYPE):
        return dtype(pd_dtype.numpy_dtype)
    elif isinstance(pd_dtype, pd.CategoricalDtype):
        return CategoricalDtype(pd_dtype.categories, pd_dtype.ordered)
    elif isinstance(pd_dtype, pd.IntervalDtype):
        return IntervalDtype(pd_dtype.subtype, pd_dtype.closed)
    elif (
        pd_dtype.kind in "mM"
        or is_dtype_obj_string(pd_dtype)
        or is_dtype_obj_list(pd_dtype)
        or is_dtype_obj_struct(pd_dtype)
        or is_dtype_obj_interval(pd_dtype)
        or is_dtype_obj_numeric(pd_dtype, include_decimal=True)
        or is_dtype_obj_datetime_tz(pd_dtype)
    ):
        return pd_dtype
    else:
        raise TypeError(f"Cannot interpret {arbitrary} as a valid cuDF dtype")


def _check_type(
    cls: type,
    header: dict,
    frames: list,
    is_valid_class: Callable[[type, type], bool] = operator.is_,
) -> None:
    """Perform metadata-encoded type and check validity

    Parameters
    ----------
    cls : type
        class performing deserialization
    header : dict
        metadata for deserialization
    frames : list
        buffers containing data for deserialization
    is_valid_class : Callable
        function to call to check if the encoded class type is valid for
        serialization by `cls` (default is to check type equality), called
        as `is_valid_class(decoded_class, cls)`.

    Raises
    ------
    AssertionError
        if the number of frames doesn't match the count encoded in the
        headers, or `is_valid_class` is not true.
    """
    assert header["frame_count"] == len(frames), (
        f"Deserialization expected {header['frame_count']} frames, "
        f"but received {len(frames)}."
    )
    klass = Serializable._name_type_map[header["type-serialized-name"]]
    assert is_valid_class(
        klass,
        cls,
    ), f"Header-encoded {klass=} does not match decoding {cls=}."


class _BaseDtype(ExtensionDtype, Serializable):
    # Base type for all cudf-specific dtypes
    pass


class CategoricalDtype(_BaseDtype):
    """
    Type for categorical data with the categories and orderedness.

    Parameters
    ----------
    categories : sequence, optional
        Must be unique, and must not contain any nulls.
        The categories are stored in an Index,
        and if an index is provided the dtype of that index will be used.
    ordered : bool or None, default False
        Whether or not this categorical is treated as a ordered categorical.
        None can be used to maintain the ordered value of existing categoricals
        when used in operations that combine categoricals, e.g. astype, and
        will resolve to False if there is no existing ordered to maintain.

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    from_pandas
    to_pandas

    Examples
    --------
    >>> import cudf
    >>> dtype = cudf.CategoricalDtype(categories=['b', 'a'], ordered=True)
    >>> cudf.Series(['a', 'b', 'a', 'c'], dtype=dtype)
    0       a
    1       b
    2       a
    3    <NA>
    dtype: category
    Categories (2, str): ['b' < 'a']
    """

    def __init__(self, categories=None, ordered: bool | None = False) -> None:
        if not (ordered is None or isinstance(ordered, bool)):
            raise ValueError("ordered must be a boolean or None")
        self._categories = self._init_categories(categories)
        self._ordered = ordered

    @property
    def categories(self) -> Index:
        """
        An ``Index`` containing the unique categories allowed.

        Examples
        --------
        >>> import cudf
        >>> dtype = cudf.CategoricalDtype(categories=['b', 'a'], ordered=True)
        >>> dtype.categories
        Index(['b', 'a'], dtype='str')
        """
        if self._categories is None:
            col = cudf.core.column.column_empty(0, dtype=DEFAULT_STRING_DTYPE)
        else:
            col = self._categories
        return cudf.Index._from_column(col)

    @property
    def type(self):
        return self._categories.dtype.type

    @property
    def name(self):
        return "category"

    @property
    def str(self):
        return "|O08"

    @property
    def ordered(self) -> bool | None:
        """
        Whether the categories have an ordered relationship.
        """
        return self._ordered

    @cached_property
    def _codes_dtype(self) -> np.dtype:
        """Return the dtype used for categorical codes."""
        return min_unsigned_type(len(self.categories))

    @cached_property
    def _pandas_codes_dtype(self) -> np.dtype:
        """The signed dtype pandas uses for these categorical codes.

        pandas coerces codes when constructing categorical data (see
        ``pandas.core.arrays.categorical.coerce_indexer_dtype``): a signed
        integer that widens int8 -> int16 -> int32 -> int64 once the
        category count reaches the narrower type's max (int8 is used only
        for *fewer* than ``np.iinfo(np.int8).max`` (127) categories). cudf
        stores codes as the unsigned ``_codes_dtype`` (a zero-copy view of
        the underlying dictionary indices), so pandas-facing code paths
        cast to this dtype instead.
        """
        n_categories = len(self.categories)
        for candidate in (np.int8, np.int16, np.int32):
            if n_categories < np.iinfo(candidate).max:
                return np.dtype(candidate)
        return np.dtype(np.int64)

    def to_pandas(self) -> pd.CategoricalDtype:
        """
        Convert a ``cudf.CategoricalDtype`` to ``pandas.CategoricalDtype``

        Examples
        --------
        >>> import cudf
        >>> dtype = cudf.CategoricalDtype(categories=['b', 'a'], ordered=True)
        >>> dtype
        CategoricalDtype(categories=['b', 'a'], ordered=True, categories_dtype=str)
        >>> dtype.to_pandas()
        CategoricalDtype(categories=['b', 'a'], ordered=True, categories_dtype=str)
        """
        if self._categories is None:
            categories = None
        elif self._categories.dtype.kind == "f":
            categories = self._categories.dropna().to_pandas()
        else:
            categories = self._categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(self, categories: Any) -> ColumnBase | None:
        if categories is None:
            return categories
        from cudf.api.types import is_scalar

        if is_scalar(categories):
            raise ValueError("categories must be a list-like object")
        if len(categories) == 0 and not isinstance(
            getattr(categories, "dtype", None),
            (IntervalDtype, pd.IntervalDtype),
        ):
            dtype = getattr(categories, "dtype", DEFAULT_STRING_DTYPE)
        else:
            dtype = None

        column = cudf.core.column.as_column(categories, dtype=dtype)

        if isinstance(column.dtype, CategoricalDtype):
            return column.categories  # type: ignore[attr-defined]
        else:
            return column

    def _internal_eq(self, other: Dtype, strict=True) -> bool:
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        elif other.ordered is None and other._categories is None:
            # other is equivalent to the string "category"
            return True
        elif self._categories is None or other._categories is None:
            return self._categories is other._categories
        elif self.ordered or other.ordered:
            return (self.ordered == other.ordered) and self._categories.equals(
                other._categories
            )
        else:
            left_cats = self._categories
            right_cats = other._categories
            if left_cats.dtype != right_cats.dtype:
                return False
            if len(left_cats) != len(right_cats):
                return False
            if self.ordered in {None, False} and other.ordered in {
                None,
                False,
            }:
                if strict:
                    return left_cats.equals(right_cats)
                else:
                    return left_cats.sort_values().equals(
                        right_cats.sort_values()
                    )
            return self.ordered == other.ordered and left_cats.equals(
                right_cats
            )

    def __eq__(self, other: Dtype) -> bool:
        return self._internal_eq(other, strict=False)

    def construct_from_string(self):
        raise NotImplementedError()

    def serialize(self):
        header = {}
        header["ordered"] = self.ordered

        frames = []

        if self.categories is not None:
            categories_header, categories_frames = (
                self.categories.device_serialize()
            )
        header["categories"] = categories_header
        frames.extend(categories_frames)
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        _check_type(cls, header, frames)
        ordered = header["ordered"]
        categories_header = header["categories"]
        categories_frames = frames
        categories = Serializable.device_deserialize(
            categories_header, categories_frames
        )
        return cls(categories=categories, ordered=ordered)

    def __repr__(self):
        return self.to_pandas().__repr__()


class ListDtype(_BaseDtype):
    """
    Type to represent list data.

    Parameters
    ----------
    element_type : object
        A dtype with which represents the element types in the list.

    Attributes
    ----------
    element_type
    leaf_type

    Methods
    -------
    from_arrow
    to_arrow

    Examples
    --------
    >>> import cudf
    >>> list_dtype = cudf.ListDtype("int32")
    >>> list_dtype
    ListDtype(int32)

    A nested list dtype can be created by:

    >>> nested_list_dtype = cudf.ListDtype(list_dtype)
    >>> nested_list_dtype
    ListDtype(ListDtype(int32))
    """

    name: str = "list"

    def __init__(self, element_type: Dtype) -> None:
        self._element_type = cudf.dtype(element_type)

    @cached_property
    def element_type(self) -> DtypeObj:
        """
        Returns the element type of the ``ListDtype``.

        Returns
        -------
        Dtype

        Examples
        --------
        >>> import cudf
        >>> deep_nested_type = cudf.ListDtype(cudf.ListDtype(cudf.ListDtype("float32")))
        >>> deep_nested_type
        ListDtype(ListDtype(ListDtype(float32)))
        >>> deep_nested_type.element_type
        ListDtype(ListDtype(float32))
        >>> deep_nested_type.element_type.element_type
        ListDtype(float32)
        >>> deep_nested_type.element_type.element_type.element_type  # doctest: +SKIP
        'float32'
        """
        return self._element_type

    @cached_property
    def leaf_type(self) -> DtypeObj:
        """
        Returns the type of the leaf values.

        Examples
        --------
        >>> import cudf
        >>> deep_nested_type = cudf.ListDtype(cudf.ListDtype(cudf.ListDtype("float32")))
        >>> deep_nested_type
        ListDtype(ListDtype(ListDtype(float32)))
        >>> deep_nested_type.leaf_type # doctest: +SKIP
        'float32'
        """
        if isinstance(self.element_type, ListDtype):
            return self.element_type.leaf_type
        else:
            return self.element_type

    @property
    def type(self):
        # TODO: we should change this to return something like a
        # ListDtypeType, once we figure out what that should look like
        return pa.array

    @classmethod
    def from_arrow(cls, typ: pa.ListType) -> Self:
        """
        Creates a ``ListDtype`` from ``pyarrow.ListType``.

        Parameters
        ----------
        typ : pyarrow.ListType
            A ``pyarrow.ListType`` that has to be converted to
            ``ListDtype``.

        Returns
        -------
        obj : ``ListDtype``

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> arrow_type = pa.infer_type([[1]])
        >>> arrow_type
        ListType(list<item: int64>)
        >>> list_dtype = cudf.ListDtype.from_arrow(arrow_type)
        >>> list_dtype
        ListDtype(int64)
        """
        # PyArrow infers empty lists as list<null>, but libcudf uses int8 as
        # the default for empty lists. cudf uses `np.dtype(object)`
        # to match pandas.
        if pa.types.is_null(typ.value_type):
            return cls(np.dtype("object"))
        return cls(cudf_dtype_from_pa_type(typ.value_type))

    def to_arrow(self) -> pa.ListType:
        """
        Convert to a ``pyarrow.ListType``

        Examples
        --------
        >>> import cudf
        >>> list_dtype = cudf.ListDtype(cudf.ListDtype("float32"))
        >>> list_dtype
        ListDtype(ListDtype(float32))
        >>> list_dtype.to_arrow()
        ListType(list<item: list<item: float>>)
        """
        return pa.list_(cudf_dtype_to_pa_type(self.element_type))

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, ListDtype):
            return False
        return self.element_type == other.element_type

    def __repr__(self) -> str:
        if isinstance(self.element_type, (ListDtype, StructDtype)):
            return f"{type(self).__name__}({self.element_type!r})"
        else:
            return f"{type(self).__name__}({self.element_type})"

    def __hash__(self) -> int:
        return hash(self.to_arrow())

    def serialize(self) -> tuple[dict, list]:
        header: dict[str, Dtype] = {}

        frames = []

        if isinstance(self.element_type, _BaseDtype):
            header["element-type"], frames = (
                self.element_type.device_serialize()
            )
        else:
            header["element-type"] = getattr(
                self.element_type, "name", self.element_type
            )
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Self:
        _check_type(cls, header, frames)
        if isinstance(header["element-type"], dict):
            element_type = Serializable.device_deserialize(
                header["element-type"], frames
            )
        else:
            element_type = header["element-type"]
        return cls(element_type=element_type)

    @cached_property
    def itemsize(self) -> int:
        return self.element_type.itemsize

    @classmethod
    def from_list_dtype(cls, obj) -> Self:
        if isinstance(obj, ListDtype):
            return obj
        elif isinstance(obj, pa.ListType):
            return cls.from_arrow(obj)
        elif isinstance(obj, pd.ArrowDtype):
            return cls.from_arrow(obj.pyarrow_dtype)
        else:
            raise TypeError(f"Cannot convert {type(obj)} to StructDtype")


class StructDtype(_BaseDtype):
    """
    Type to represent a struct data.

    Parameters
    ----------
    fields : dict
        A mapping of field names to dtypes, the dtypes can themselves
        be of ``StructDtype`` too.

    Attributes
    ----------
    fields
    itemsize

    Methods
    -------
    from_arrow
    to_arrow

    Examples
    --------
    >>> import cudf
    >>> struct_dtype = cudf.StructDtype({"a": "int64", "b": "float64"})
    >>> struct_dtype
    StructDtype({'a': dtype('int64'), 'b': dtype('float64')})

    A nested ``StructDtype`` can also be constructed in the following way:

    >>> nested_struct_dtype = cudf.StructDtype({"dict_data": struct_dtype, "c": "uint8"})
    >>> nested_struct_dtype
    StructDtype({'dict_data': StructDtype({'a': dtype('int64'), 'b': dtype('float64')}), 'c': dtype('uint8')})
    """

    name = "struct"

    def __init__(self, fields: dict[str, Dtype]) -> None:
        self._fields = {k: dtype(v) for k, v in fields.items()}

    @property
    def fields(self) -> dict[str, DtypeObj]:
        """
        Returns an ordered dict of column name and dtype key-value.

        Examples
        --------
        >>> import cudf
        >>> struct_dtype = cudf.StructDtype({"a": "int64", "b": "float64"})
        >>> struct_dtype
        StructDtype({'a': dtype('int64'), 'b': dtype('float64')})
        >>> struct_dtype.fields
        {'a': dtype('int64'), 'b': dtype('float64')}
        """
        return self._fields

    @property
    def type(self):
        # TODO: we should change this to return something like a
        # StructDtypeType, once we figure out what that should look like
        return dict

    @classmethod
    def from_arrow(cls, typ: pa.StructType) -> Self:
        """
        Convert a ``pyarrow.StructType`` to ``StructDtype``.

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> pa_struct_type = pa.struct({'x': pa.int32(), 'y': pa.string()})
        >>> pa_struct_type
        StructType(struct<x: int32, y: string>)
        >>> cudf.StructDtype.from_arrow(pa_struct_type)
        StructDtype({'x': dtype('int32'), 'y': <StringDtype(na_value=nan)>})
        """
        return cls(
            {field.name: cudf_dtype_from_pa_type(field.type) for field in typ}
        )

    def to_arrow(self) -> pa.StructType:
        """
        Convert a ``StructDtype`` to a ``pyarrow.StructType``.

        Examples
        --------
        >>> import cudf
        >>> struct_type = cudf.StructDtype({"x": "int32", "y": "float64"})
        >>> struct_type
        StructDtype({'x': dtype('int32'), 'y': dtype('float64')})
        >>> struct_type.to_arrow()
        StructType(struct<x: int32, y: double>)
        """
        return pa.struct(
            # dict[str, DataType] should be compatible but pyarrow stubs are too strict
            {  # type: ignore[arg-type]
                k: cudf_dtype_to_pa_type(dtype)
                for k, dtype in self.fields.items()
            }
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, StructDtype):
            return False
        return self.to_arrow().equals(other.to_arrow())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.fields})"

    def __hash__(self) -> int:
        return hash(self.to_arrow())

    def serialize(self) -> tuple[dict, list]:
        header: dict[str, Any] = {}

        frames: list[Buffer] = []

        fields: dict[str, str | tuple[Any, tuple[int, int]]] = {}

        for k, dtype in self.fields.items():
            if isinstance(dtype, _BaseDtype):
                dtype_header, dtype_frames = dtype.device_serialize()
                fields[k] = (
                    dtype_header,
                    (len(frames), len(frames) + len(dtype_frames)),
                )
                frames.extend(dtype_frames)
            else:
                fields[k] = dtype.str
        header["fields"] = fields
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Self:
        _check_type(cls, header, frames)
        fields = {}
        for k, dtype in header["fields"].items():
            if isinstance(dtype, tuple):
                dtype_header, (start, stop) = dtype
                fields[k] = Serializable.device_deserialize(
                    dtype_header,
                    frames[start:stop],
                )
            else:
                fields[k] = np.dtype(dtype)
        return cls(fields)

    @cached_property
    def itemsize(self) -> int:
        return sum(field.itemsize for field in self.fields.values())

    @classmethod
    def from_struct_dtype(cls, obj) -> Self:
        if isinstance(obj, StructDtype):
            return obj
        elif isinstance(obj, pa.StructType):
            return cls.from_arrow(obj)
        elif isinstance(obj, pd.ArrowDtype):
            return cls.from_arrow(obj.pyarrow_dtype)
        else:
            raise TypeError(f"Cannot convert {type(obj)} to StructDtype")


decimal_dtype_template = textwrap.dedent(
    """
        Type to represent a ``decimal{size}`` data.

        Parameters
        ----------
        precision : int
            The total number of digits in each value of this dtype
        scale : int, optional
            The scale of the dtype. See Notes below.

        Attributes
        ----------
        precision
        scale
        itemsize

        Methods
        -------
        to_arrow
        from_arrow

        Notes
        -----
        When the scale is positive:
            - numbers with fractional parts (e.g., 0.0042) can be represented
            - the scale is the total number of digits to the right of the
              decimal point

        When the scale is negative:
            - only multiples of powers of 10 (including 10**0) can be
              represented (e.g., 1729, 4200, 1000000)
            - the scale represents the number of trailing zeros in the value.

        For example, 42 is representable with precision=2 and scale=0.
        13.0051 is representable with precision=6 and scale=4,
        and *not* representable with precision<6 or scale<4.

        Examples
        --------
        >>> import cudf
        >>> decimal{size}_dtype = cudf.Decimal{size}Dtype(precision=9, scale=2)
        >>> decimal{size}_dtype
        Decimal{size}Dtype(precision=9, scale=2)
    """
)


class DecimalDtype(_BaseDtype):
    _metadata = ("precision", "scale")

    def __init__(self, precision: int, scale: int = 0) -> None:
        self._validate(precision, scale)
        self._precision = precision
        self._scale = scale

    @property
    def str(self) -> str:
        return f"{self.name!s}({self.precision}, {self.scale})"

    @property
    def precision(self) -> int:
        """
        The decimal precision, in number of decimal digits (an integer).
        """
        return self._precision

    @precision.setter
    def precision(self, value: int) -> None:
        self._validate(value, self.scale)
        self._precision = value

    @property
    def scale(self) -> int:
        """
        The decimal scale (an integer).
        """
        return self._scale

    @property
    def itemsize(self) -> int:
        """
        Length of one column element in bytes.
        """
        return self.ITEMSIZE

    @property
    def type(self):
        # might need to account for precision and scale here
        return decimal.Decimal

    def to_arrow(
        self,
    ) -> pa.Decimal128Type | pa.Decimal32Type | pa.Decimal64Type:
        """
        Return the equivalent ``pyarrow`` dtype.
        """
        return type(self).PA_TYPE(self.precision, self.scale)

    @classmethod
    def from_arrow(
        cls, typ: pa.Decimal32Type | pa.Decimal64Type | pa.Decimal128Type
    ) -> Self:
        # TODO: Eventually narrow this to only accept the appropriate decimal type
        # for each specific DecimalNDtype subclass
        """
        Construct a cudf decimal dtype from a ``pyarrow`` dtype

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> pa_type = pa.decimal128(precision=9, scale=2)

        Constructing a ``Decimal32Dtype``:

        >>> cudf.Decimal32Dtype.from_arrow(pa_type)
        Decimal64Dtype(precision=9, scale=2)

        Constructing a ``Decimal64Dtype``:

        >>> cudf.Decimal64Dtype.from_arrow(pa_type)
        Decimal64Dtype(precision=9, scale=2)

        Constructing a ``Decimal128Dtype``:

        >>> cudf.Decimal128Dtype.from_arrow(pa_type)
        Decimal128Dtype(precision=9, scale=2)
        """
        return cls(typ.precision, typ.scale)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(precision={self.precision}, scale={self.scale})"
        )

    @classmethod
    def _validate(cls, precision: int, scale: int) -> None:
        if precision > cls.MAX_PRECISION:
            raise ValueError(
                f"Cannot construct a {cls.__name__}"
                f" with precision > {cls.MAX_PRECISION}"
            )
        if abs(scale) > precision:
            raise ValueError(f"{scale=} cannot exceed {precision=}")

    @classmethod
    def _from_decimal(cls, decimal: decimal.Decimal) -> Self:
        """
        Create a cudf.DecimalDtype from a decimal.Decimal object
        """
        metadata = decimal.as_tuple()
        precision = max(len(metadata.digits), -metadata.exponent)  # type: ignore[operator]
        return cls(precision, -metadata.exponent)  # type: ignore[operator]

    def serialize(self) -> tuple[dict, list]:
        return (
            {
                "precision": self.precision,
                "scale": self.scale,
                "frame_count": 0,
            },
            [],
        )

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Self:
        _check_type(cls, header, frames, is_valid_class=issubclass)
        return cls(header["precision"], header["scale"])

    def __eq__(self, other: Dtype) -> bool:
        if other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        return self.precision == other.precision and self.scale == other.scale

    def __hash__(self) -> int:
        return hash(self.to_arrow())


@doc_apply(
    decimal_dtype_template.format(
        size="32",
    )
)
class Decimal32Dtype(DecimalDtype):
    name = "decimal32"
    MAX_PRECISION = np.floor(np.log10(np.iinfo("int32").max))
    ITEMSIZE = 4
    PA_TYPE = pa.decimal32


@doc_apply(
    decimal_dtype_template.format(
        size="64",
    )
)
class Decimal64Dtype(DecimalDtype):
    name = "decimal64"
    MAX_PRECISION = np.floor(np.log10(np.iinfo("int64").max))
    ITEMSIZE = 8
    PA_TYPE = pa.decimal64


@doc_apply(
    decimal_dtype_template.format(
        size="128",
    )
)
class Decimal128Dtype(DecimalDtype):
    name = "decimal128"
    MAX_PRECISION = 38
    ITEMSIZE = 16
    PA_TYPE = pa.decimal128


class IntervalDtype(_BaseDtype):
    """
    A data type for Interval data.

    Parameters
    ----------
    subtype: str, np.dtype
        The dtype of the Interval bounds.
    closed: {'right', 'left', 'both', 'neither'}, default 'right'
        Whether the interval is closed on the left-side, right-side,
        both or neither. See the Notes for more detailed explanation.

    Attributes
    ----------
    fields
    type

    Methods
    -------
    from_arrow
    to_arrow
    """

    name = "interval"
    closed: Literal["left", "right", "neither", "both"] | None

    def __init__(
        self,
        subtype: None | Dtype = None,
        closed: Literal["left", "right", "neither", "both", None] = None,
    ) -> None:
        if closed not in {"left", "right", "neither", "both", None}:
            raise ValueError(f"{closed=} is not valid")

        # If subtype is already an IntervalDtype (cudf or pandas), unwrap it,
        # validating any closed conflict.
        if isinstance(subtype, (IntervalDtype, pd.IntervalDtype)):
            inner_closed = cast(
                "Literal['left', 'right', 'neither', 'both'] | None",
                subtype.closed,
            )
            if (
                closed is not None
                and inner_closed is not None
                and closed != inner_closed
            ):
                raise ValueError(
                    "dtype.closed and 'closed' do not match. "
                    "Try IntervalDtype(dtype.subtype, closed) instead."
                )
            closed = closed if closed is not None else inner_closed
            subtype = subtype.subtype

        # If subtype is an "interval[...]" spec string, let pandas parse it
        # ("interval", "interval[int64]", "interval[int64, right]"),
        # reconciling any explicitly-passed ``closed`` (pandas raises on a
        # closed-keyword conflict or an unsupported subtype). This avoids
        # re-implementing pandas' parsing logic. A bare subtype string (e.g.
        # "int64") falls through to ``dtype(subtype)`` below.
        if isinstance(subtype, str) and subtype.lower().startswith("interval"):
            parsed = pd.IntervalDtype(subtype, closed)
            closed = cast(
                "Literal['left', 'right', 'neither', 'both'] | None",
                parsed.closed,
            )
            subtype = parsed.subtype  # may be None for bare "interval"

        if subtype is None:
            self._subtype = None
            self._fields = {}
        else:
            try:
                resolved = dtype(subtype)
            except TypeError as err:
                raise TypeError(
                    f"could not construct IntervalDtype from {subtype!r}"
                ) from err
            # TODO: Remove resolved.kind == "U" once cudf.dtype no longer accepts
            # numpy string types
            if (
                isinstance(resolved, CategoricalDtype)
                or is_dtype_obj_string(resolved)
                or getattr(resolved, "kind", None) == "U"
            ):
                raise TypeError(
                    "category, object, and string subtypes are not supported "
                    "for IntervalDtype"
                )
            self._subtype = resolved
            self._fields = {"left": resolved, "right": resolved}

        self.closed = closed

    @classmethod
    def construct_from_string(cls, string: str) -> Self:
        # Defer string validation/parsing to pandas, then wrap the result.
        return cls(pd.IntervalDtype.construct_from_string(string))

    @property
    def subtype(self) -> DtypeObj | None:
        return self._subtype

    @property
    def fields(self) -> dict[str, DtypeObj]:
        """
        Returns an ordered dict of column name and dtype key-value.

        For IntervalDtype, this always returns {"left": subtype, "right": subtype}.
        """
        return self._fields

    @property
    def type(self):
        # TODO: we should change this to return something like an
        # IntervalDtypeType, once we figure out what that should look like
        return pd.Interval

    @cached_property
    def itemsize(self) -> int:
        if self._subtype is None:
            return 0
        return sum(field.itemsize for field in self.fields.values())

    def __repr__(self) -> str:
        if self.subtype is None:
            return "interval"
        if self.closed is None:
            return f"interval[{self.subtype}]"
        return f"interval[{self.subtype}, {self.closed}]"

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def from_arrow(cls, typ: ArrowIntervalType) -> Self:
        return cls(typ.subtype.to_pandas_dtype(), typ.closed)

    def to_arrow(self) -> ArrowIntervalType:
        return ArrowIntervalType(
            cudf_dtype_to_pa_type(self.subtype),
            self.closed if self.closed is not None else "right",
        )

    def to_pandas(self) -> pd.IntervalDtype:
        return pd.IntervalDtype(
            subtype=self.subtype,
            closed=self.closed,
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            # This means equality isn't transitive but mimics pandas
            return other in (self.name, str(self))
        elif type(self) is not type(other):
            return False
        elif other.subtype is None:
            # Equivalent to the string "interval"
            return True
        return self.subtype == other.subtype and self.closed == other.closed

    def __hash__(self) -> int:
        return hash((self.subtype, self.closed))

    def serialize(self) -> tuple[dict, list]:
        header = {
            "fields": (
                self.subtype.str if self.subtype is not None else self.subtype,
                self.closed,
            ),
            "frame_count": 0,
        }
        return header, []

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Self:
        _check_type(cls, header, frames)
        subtype, closed = header["fields"]
        return cls(subtype, closed=closed)

    @classmethod
    def from_interval_dtype(cls, obj) -> Self:
        if isinstance(obj, IntervalDtype):
            return obj
        elif isinstance(obj, ArrowIntervalType):
            return cls.from_arrow(obj)
        elif isinstance(obj, pd.ArrowDtype):
            return cls.from_arrow(obj.pyarrow_dtype)
        else:
            raise TypeError(f"Cannot convert {type(obj)} to IntervalDtype")


def _is_categorical_dtype(obj):
    if obj is None:
        return False

    if isinstance(
        obj,
        (
            pd.CategoricalDtype,
            cudf.CategoricalDtype,
            cudf.core.index.CategoricalIndex,
            cudf.core.column.CategoricalColumn,
            pd.Categorical,
            pd.CategoricalIndex,
        ),
    ):
        return True
    # Note that we cannot directly use `obj in (...)`  because that triggers
    # equality as well as identity checks and pandas extension dtypes won't
    # allow converting that equality check to a boolean; `__nonzero__` is
    # disabled because they treat dtypes as "array-like".
    if any(
        obj is t
        for t in (
            cudf.CategoricalDtype,
            pd.CategoricalDtype,
            pd.CategoricalDtype.type,
        )
    ):
        return True
    if isinstance(obj, (np.ndarray, np.dtype)):
        return False
    if isinstance(obj, str) and obj == "category":
        return True
    if isinstance(
        obj,
        (cudf.Index, cudf.core.column.ColumnBase, cudf.Series),
    ):
        return isinstance(obj.dtype, cudf.CategoricalDtype)
    if isinstance(obj, (pd.Series, pd.Index)):
        return isinstance(obj.dtype, pd.CategoricalDtype)
    if hasattr(obj, "type"):
        if obj.type is pd.CategoricalDtype.type:
            return True
    # TODO: A lot of the above checks are probably redundant and should be
    # farmed out to this function here instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd_types.is_categorical_dtype(obj)


def is_categorical_dtype(obj):
    """Check whether an array-like or dtype is of the Categorical dtype.

    .. deprecated:: 24.04
       Use isinstance(dtype, cudf.CategoricalDtype) instead

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of a categorical dtype.
    """
    warnings.warn(
        "is_categorical_dtype is deprecated and will be removed in a future "
        "version. Use isinstance(dtype, cudf.CategoricalDtype) instead",
        DeprecationWarning,
    )
    return _is_categorical_dtype(obj)


def is_list_dtype(obj):
    """Check whether an array-like or dtype is of the list dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the list dtype.
    """
    return (
        type(obj) is ListDtype
        or obj is ListDtype
        or type(obj) is cudf.core.column.ListColumn
        or obj is cudf.core.column.ListColumn
        or (isinstance(obj, str) and obj == ListDtype.name)
        or (hasattr(obj, "dtype") and isinstance(obj.dtype, ListDtype))
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_list(obj.pyarrow_dtype)
        )
    )


def is_struct_dtype(obj):
    """Check whether an array-like or dtype is of the struct dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the struct dtype.
    """
    # TODO: This behavior is currently inconsistent for interval types. the
    # actual class IntervalDtype will return False, but instances (e.g.
    # IntervalDtype(int)) will return True. For now this is not being changed
    # since the interval dtype is being modified as part of the array refactor,
    # but this behavior should be made consistent afterwards.
    return (
        isinstance(obj, StructDtype)
        or obj is StructDtype
        or (isinstance(obj, str) and obj == StructDtype.name)
        or (hasattr(obj, "dtype") and isinstance(obj.dtype, StructDtype))
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_struct(obj.pyarrow_dtype)
        )
    )


def is_decimal_dtype(obj):
    """Check whether an array-like or dtype is of the decimal dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the decimal dtype.
    """
    return (
        is_decimal32_dtype(obj)
        or is_decimal64_dtype(obj)
        or is_decimal128_dtype(obj)
    )


# Regex used only to detect whether a string names an IntervalDtype; actual
# parsing of such strings is delegated to pandas in ``IntervalDtype``. Matches
# "interval", "Interval", "interval[<sub>]", "interval[<sub>, <closed>]"
# (subtype itself may contain "[...]" e.g. "datetime64[ns]").
_INTERVAL_DTYPE_STRING_MATCH = re.compile(
    r"^(I|i)nterval"
    r"(\[(?P<subtype>[^,]+?)(,\s*(?P<closed>left|right|both|neither))?\])?$"
)


def _is_interval_dtype(obj):
    return (
        isinstance(
            obj,
            (
                IntervalDtype,
                pd.IntervalDtype,
            ),
        )
        or obj is IntervalDtype
        or (
            isinstance(obj, cudf.Index)
            and isinstance(obj.dtype, IntervalDtype)
        )
        or (
            isinstance(obj, str)
            and _INTERVAL_DTYPE_STRING_MATCH.match(obj) is not None
        )
        or (
            isinstance(
                getattr(obj, "dtype", None),
                (pd.IntervalDtype, IntervalDtype),
            )
        )
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_interval(obj.pyarrow_dtype)
        )
    )


def is_interval_dtype(obj):
    """Check whether an array-like or dtype is of the interval dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the interval dtype.
    """
    warnings.warn(
        "is_interval_dtype is deprecated and will be removed in a "
        "future version. Use `isinstance(dtype, cudf.IntervalDtype)` instead",
        DeprecationWarning,
    )
    return _is_interval_dtype(obj)


def is_decimal32_dtype(obj):
    return (
        type(obj) is Decimal32Dtype
        or (hasattr(obj, "dtype") and is_decimal32_dtype(obj.dtype))
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_decimal(obj.pyarrow_dtype)
            and isinstance(obj.pyarrow_dtype, pa.lib.Decimal32Type)
        )
    )


def is_decimal64_dtype(obj):
    return (
        type(obj) is Decimal64Dtype
        or (hasattr(obj, "dtype") and is_decimal64_dtype(obj.dtype))
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_decimal(obj.pyarrow_dtype)
            and isinstance(obj.pyarrow_dtype, pa.lib.Decimal64Type)
        )
    )


def is_decimal128_dtype(obj):
    return (
        type(obj) is Decimal128Dtype
        or (hasattr(obj, "dtype") and is_decimal128_dtype(obj.dtype))
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_decimal128(obj.pyarrow_dtype)
        )
    )


def recursively_update_struct_names(
    dtype: DtypeObj, child_names: Mapping[Any, Any]
) -> DtypeObj:
    """
    Update dtype's field names (namely StructDtype and IntervalDtype) recursively with child_names.

    Needed for nested types that come from libcudf which do not carry struct field names.
    """
    if isinstance(dtype, IntervalDtype):
        # For IntervalDtype, child_names should have "left" and "right" keys
        # But we need to recursively update the subtype if it's nested
        if dtype.subtype is None:
            return dtype
        # child_names should be {"left": {...}, "right": {...}}
        left_names = child_names.get("left", {})
        # Since left and right have the same dtype, we only need one of them
        if isinstance(dtype.subtype, (StructDtype, ListDtype)):
            new_subtype = recursively_update_struct_names(
                dtype.subtype, left_names
            )
            return IntervalDtype(subtype=new_subtype, closed=dtype.closed)
        return dtype
    elif isinstance(dtype, StructDtype):
        return StructDtype(
            {
                new_name: recursively_update_struct_names(
                    child_type, new_child_names
                )
                for (new_name, new_child_names), child_type in zip(
                    child_names.items(), dtype.fields.values(), strict=True
                )
            }
        )
    elif isinstance(dtype, ListDtype):
        # child_names here should be {"offsets": {}, "<values_key>": {...}}
        values_key = next(reversed(child_names))
        return ListDtype(
            element_type=recursively_update_struct_names(
                dtype.element_type, child_names[values_key]
            )
        )
    else:
        return dtype


def _dtype_to_metadata(dtype: DtypeObj) -> plc.interop.ColumnMetadata:
    # Convert a cudf or pandas dtype to pylibcudf ColumnMetadata for arrow conversion
    cm = plc.interop.ColumnMetadata()
    if isinstance(dtype, IntervalDtype):
        # IntervalDtype is stored as a struct with "left" and "right" fields
        for name, field_dtype in dtype.fields.items():
            cm.children_meta.append(_dtype_to_metadata(field_dtype))
            cm.children_meta[-1].name = name
    elif isinstance(dtype, StructDtype):
        for name, dtype in dtype.fields.items():
            cm.children_meta.append(_dtype_to_metadata(dtype))
            cm.children_meta[-1].name = name
    elif isinstance(dtype, ListDtype):
        # Offsets column must be added manually
        cm.children_meta.append(plc.interop.ColumnMetadata())
        cm.children_meta.append(_dtype_to_metadata(dtype.element_type))
    elif isinstance(dtype, DecimalDtype):
        cm.precision = dtype.precision
    elif isinstance(dtype, pd.DatetimeTZDtype):
        cm.timezone = str(dtype.tz)
    elif isinstance(dtype, pd.ArrowDtype):
        if pa.types.is_struct(dtype.pyarrow_dtype):
            for field in dtype.pyarrow_dtype:
                cm.children_meta.append(
                    _dtype_to_metadata(pd.ArrowDtype(field.type))
                )
                cm.children_meta[-1].name = field.name
        elif isinstance(dtype.pyarrow_dtype, ArrowIntervalType):
            left_meta = _dtype_to_metadata(
                pd.ArrowDtype(dtype.pyarrow_dtype.subtype)
            )
            left_meta.name = "left"
            right_meta = left_meta.replace(name="right")  # type: ignore[attr-defined]
            cm.children_meta.append(left_meta)
            cm.children_meta.append(right_meta)
        elif pa.types.is_decimal(dtype.pyarrow_dtype):
            cm.precision = dtype.pyarrow_dtype.precision
        elif (
            pa.types.is_timestamp(dtype.pyarrow_dtype)
            and (tz := dtype.pyarrow_dtype.tz) is not None
        ):
            cm.timezone = str(tz)
        elif pa.types.is_list(dtype.pyarrow_dtype) or pa.types.is_large_list(
            dtype.pyarrow_dtype
        ):
            # Offsets column must be added manually
            cm.children_meta.append(plc.interop.ColumnMetadata())
            cm.children_meta.append(
                _dtype_to_metadata(
                    pd.ArrowDtype(dtype.pyarrow_dtype.value_type)
                )
            )
    return cm
