# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import pickle
import warnings
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    MutableSequence,
    Sequence,
)
from contextlib import ExitStack
from decimal import Decimal
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

import cudf
from cudf.api.types import (
    _is_categorical_dtype,
    infer_dtype,
    is_dtype_equal,
    is_scalar,
)
from cudf.core._internals import (
    aggregation,
    copying,
    sorting,
)
from cudf.core._internals.timezones import get_compatible_timezone
from cudf.core.abc import Serializable
from cudf.core.buffer import (
    Buffer,
    as_buffer,
)
from cudf.core.column.utils import access_columns
from cudf.core.copy_types import GatherMap
from cudf.core.dtype.conversions import (
    element_type_from_list_dtype,
    fields_from_struct_dtype,
    subtype_from_interval_dtype,
)
from cudf.core.dtype.validators import (
    is_dtype_obj_datetime_tz,
    is_dtype_obj_decimal,
    is_dtype_obj_interval,
    is_dtype_obj_list,
    is_dtype_obj_numeric,
    is_dtype_obj_string,
    is_dtype_obj_struct,
)
from cudf.core.dtypes import (
    CategoricalDtype,
    DecimalDtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
    _dtype_to_metadata,
)
from cudf.core.mixins import BinaryOperand, Reducible
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    DEFAULT_STRING_DTYPE,
    SIZE_TYPE_DTYPE,
    _get_nan_for_dtype,
    _maybe_convert_to_default_type,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
    find_common_type,
    get_dtype_of_same_kind,
    is_arrow_null_dtype,
    is_column_like,
    is_mixed_with_object_dtype,
    is_pandas_nullable_extension_dtype,
    is_pandas_nullable_numpy_dtype,
    min_signed_type,
    np_dtypes_to_pandas_dtypes,
    pyarrow_dtype_to_cudf_dtype,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import (
    _array_ufunc,
    is_na_like,
)

if TYPE_CHECKING:
    import builtins
    from collections.abc import Generator, Mapping
    from types import TracebackType

    from cudf._typing import ColumnLike, DtypeObj, DtypePolicy, ScalarLike
    from cudf.core.column.categorical import CategoricalColumn
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.decimal import DecimalColumn
    from cudf.core.column.interval import IntervalColumn
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn
    from cudf.core.column.timedelta import TimeDeltaColumn
    from cudf.core.index import Index


class ColumnList:
    """Tag type to mark a group of columns for plc.Table construction.

    Used with PylibcudfFunction to signal that these columns should be
    packed into a plc.Table before being passed to the pylibcudf function.
    """

    __slots__ = ("columns",)

    def __init__(self, *columns: "ColumnBase"):
        self.columns = columns


class PylibcudfFunction:
    def __init__(
        self,
        pylibcudf_function: Callable[..., Any],
        dtype_policy: "DtypePolicy",
        mode: Literal["read", "write"] = "read",
        result_index: int | None = None,
    ) -> None:
        self._pylibcudf_function = pylibcudf_function
        self._dtype_policy = dtype_policy
        self._mode: Literal["read", "write"] = mode
        self._result_index = result_index

    def _access_column(
        self,
        col: "ColumnBase",
        stack: ExitStack,
        dtypes: list["DtypeObj"],
    ) -> plc.Column:
        accessed = stack.enter_context(
            col.access(mode=self._mode, scope="internal")
        )
        dtypes.append(accessed.dtype)
        return accessed.plc_column

    def _process_arg(
        self,
        arg: Any,
        stack: ExitStack,
        dtypes: list["DtypeObj"],
    ) -> Any:
        if isinstance(arg, ColumnBase):
            return self._access_column(arg, stack, dtypes)
        if isinstance(arg, ColumnList):
            return plc.Table(
                [
                    self._access_column(col, stack, dtypes)
                    for col in arg.columns
                ]
            )
        return arg

    def execute_with_args(self, *args: Any, **kwargs: Any) -> Any:
        dtypes: list["DtypeObj"] = []
        with ExitStack() as stack:
            plc_args = [self._process_arg(arg, stack, dtypes) for arg in args]
            plc_kwargs = {
                k: self._process_arg(v, stack, dtypes)
                for k, v in kwargs.items()
            }
            plc_result = self._pylibcudf_function(*plc_args, **plc_kwargs)
        if self._result_index is not None:
            plc_result = plc_result.columns()[self._result_index]
        output_dtype = self._dtype_policy(plc_result, dtypes)
        return ColumnBase.create(plc_result, dtype=output_dtype)


def same_dtype_policy(
    result: plc.Column, dtypes: "list[DtypeObj]"
) -> "DtypeObj":
    return dtypes[0]


def fixed_dtype_policy(dtype: "DtypeObj") -> "DtypePolicy":
    def policy(result: plc.Column, dtypes: "list[DtypeObj]") -> "DtypeObj":
        return dtype

    return policy


def same_kind_dtype_policy(target_dtype: "DtypeObj") -> "DtypePolicy":
    def policy(result: plc.Column, dtypes: "list[DtypeObj]") -> "DtypeObj":
        return get_dtype_of_same_kind(dtypes[0], target_dtype)

    return policy


bool_same_kind_policy = same_kind_dtype_policy(np.dtype(np.bool_))
int16_same_kind_policy = same_kind_dtype_policy(np.dtype(np.int16))
int32_same_kind_policy = same_kind_dtype_policy(np.dtype(np.int32))
int64_same_kind_policy = same_kind_dtype_policy(np.dtype(np.int64))


def list_dtype_policy(
    result: plc.Column, dtypes: "list[DtypeObj]"
) -> "DtypeObj":
    return cudf.ListDtype(dtypes[0])


def pylibcudf_result_dtype_policy(
    result: plc.Column, dtypes: "list[DtypeObj]"
) -> "DtypeObj":
    return dtype_from_pylibcudf_column(result)


def pylibcudf_result_same_kind_dtype_policy(
    result: plc.Column, dtypes: "list[DtypeObj]"
) -> "DtypeObj":
    """Like ``pylibcudf_result_dtype_policy`` but preserves the first input
    dtype's "kind" (pandas-nullable / ArrowDtype) on the result.
    """
    return get_dtype_of_same_kind(
        dtypes[0], dtype_from_pylibcudf_column(result)
    )


def pandas_compatible_string_dtype_policy(
    target_dtype: np.dtype,
) -> "DtypePolicy":
    def policy(result: plc.Column, dtypes: "list[DtypeObj]") -> "DtypeObj":
        source_dtype = dtypes[0]
        if (
            isinstance(source_dtype, pd.StringDtype)
            and source_dtype.na_value is np.nan
        ):
            return get_dtype_of_same_kind(source_dtype, target_dtype)

        return get_dtype_of_same_kind(
            pd.StringDtype()
            if isinstance(source_dtype, pd.StringDtype)
            else source_dtype,
            target_dtype,
        )

    return policy


pandas_string_int32_policy = pandas_compatible_string_dtype_policy(
    np.dtype(np.int32)
)


def _replace_nested_none_with_NA(py_scalar: Any) -> Any:
    """
    Replace the None values from pyarrow.Scalar.as_py() with pd.NA.

    Used in ColumnBase.element_indexing().

    Parameters
    ----------
    py_scalar: Any

    Returns
    -------
    Any
        Python scalar
    """
    if isinstance(py_scalar, dict):
        return {
            key: _replace_nested_none_with_NA(value)
            for key, value in py_scalar.items()
        }
    elif isinstance(py_scalar, list):
        return [_replace_nested_none_with_NA(value) for value in py_scalar]
    elif py_scalar is None:
        return pd.NA
    else:
        return py_scalar


def _can_values_be_equal(left: DtypeObj, right: DtypeObj) -> bool:
    """
    Given 2 possibly not equal dtypes, can they both hold equivalent values.

    Helper function for .equals when check_dtypes is False.
    """
    if left == right:
        return True
    if isinstance(left, CategoricalDtype):
        return _can_values_be_equal(left.categories.dtype, right)
    elif isinstance(right, CategoricalDtype):
        return _can_values_be_equal(left, right.categories.dtype)
    elif isinstance(left, pd.StringDtype) and is_dtype_obj_string(right):
        return True
    elif isinstance(right, pd.StringDtype) and is_dtype_obj_string(left):
        return True
    elif is_dtype_obj_numeric(left) and is_dtype_obj_numeric(right):
        return True
    elif left.kind == right.kind and left.kind in "mM":
        return True
    return False


def _wrap_buffer_or_span(
    buffer_or_span: Buffer | Any | None,
) -> Buffer | None:
    """Wrap a buffer or span in a cudf Buffer.

    If the input is already a Buffer, it will be shallow-copied to track
    that we are now sharing the same BufferOwner. Any other non-None object will be
    wrapped in a buffer via as_buffer.
    """
    if buffer_or_span is None:
        return None
    if isinstance(buffer_or_span, Buffer):
        return buffer_or_span.copy(deep=False)
    return as_buffer(buffer_or_span)


def _rebuild_column(
    col: plc.Column,
    children: list[plc.Column],
    *,
    wrap_buffers: bool,
) -> plc.Column:
    data = col.data()
    mask = col.null_mask()
    if wrap_buffers:
        data = _wrap_buffer_or_span(data)
        mask = _wrap_buffer_or_span(mask)
    return plc.Column(
        data_type=col.type(),
        size=col.size(),
        data=data,
        mask=mask,
        null_count=col.null_count(),
        offset=col.offset(),
        children=children,
        validate=False,
    )


def _normalize_types_column(col: plc.Column) -> plc.Column:
    """Normalize unsupported types from external sources."""
    type_id = col.type().id()
    if type_id == plc.TypeId.TIMESTAMP_DAYS:
        return plc.unary.cast(col, plc.DataType(plc.TypeId.TIMESTAMP_SECONDS))

    if type_id == plc.TypeId.EMPTY:
        return plc.Column.from_scalar(
            plc.Scalar.from_py(None, plc.DataType(plc.TypeId.STRING)),
            col.size(),
        )

    normalized_children = [
        _normalize_types_column(child) for child in col.children()
    ]
    if all(
        normalized_child is child
        for normalized_child, child in zip(
            normalized_children, col.children(), strict=True
        )
    ):
        return col

    return _rebuild_column(col, normalized_children, wrap_buffers=False)


def _wrap_and_validate(col: plc.Column, dtype: DtypeObj) -> plc.Column:
    if dtype_to_pylibcudf_type(dtype) != col.type():
        raise ValueError(
            f"dtype {dtype} does not match the type of the plc_column "
            f"{col.type().id()}. If normalization is required, please run the "
            "column through _normalize_types_column first."
        )
    if isinstance(dtype, np.dtype) and dtype.kind == "U":
        raise ValueError(
            f"dtype {dtype} is a numpy Unicode dtype. "
            "Normalize to np.dtype('O') before calling "
            "ColumnBase.create."
        )
    if is_arrow_null_dtype(dtype) and col.null_count() != col.size():
        raise ValueError(
            f"dtype {dtype} can only be used with all-null columns."
        )

    dtype_kind = dtype.kind
    children: list[plc.Column] = []
    if is_dtype_obj_list(dtype):
        valid_types = {plc.TypeId.LIST}
        child_dtype = element_type_from_list_dtype(dtype)
        values = _wrap_and_validate(col.list_view().child(), child_dtype)
        offsets_plc = col.list_view().offsets()
        offsets = _wrap_and_validate(
            offsets_plc, dtype_from_pylibcudf_column(offsets_plc)
        )
        children = [offsets, values]
    elif is_dtype_obj_interval(dtype):
        valid_types = {plc.TypeId.STRUCT}
        interval_subtype = subtype_from_interval_dtype(dtype)
        try:
            paired = list(zip(("Left", "Right"), col.children(), strict=True))
        except ValueError as e:
            raise ValueError(
                "plc_column must have two children (left edges, right edges)."
            ) from e
        children = [
            _wrap_and_validate(child, interval_subtype)
            for side, child in paired
        ]
    elif is_dtype_obj_struct(dtype):
        valid_types = {plc.TypeId.STRUCT}
        struct_fields = fields_from_struct_dtype(dtype)
        for child, (field_name, field_dtype) in zip(
            col.children(), struct_fields.items(), strict=True
        ):
            try:
                wrapped_child = _wrap_and_validate(child, field_dtype)
            except ValueError as e:
                raise ValueError(
                    f"Field '{field_name}' validation failed"
                ) from e
            children.append(wrapped_child)
    elif is_dtype_obj_string(dtype):
        valid_types = {plc.TypeId.STRING}
        # An empty string column may have no children.
        if col.num_children() == 1:
            child = col.children()[0]
            if child.type().id() not in (plc.TypeId.INT32, plc.TypeId.INT64):
                raise ValueError(
                    f"String column offsets must be INT32 or INT64, "
                    f"got {child.type().id()}."
                )
            children = [_rebuild_column(child, [], wrap_buffers=True)]
    elif is_dtype_obj_decimal(dtype):
        valid_types = {
            plc.TypeId.DECIMAL128,
            plc.TypeId.DECIMAL64,
            plc.TypeId.DECIMAL32,
        }
    elif isinstance(dtype, CategoricalDtype):
        valid_types = {
            plc.TypeId.UINT8,
            plc.TypeId.UINT16,
            plc.TypeId.UINT32,
            plc.TypeId.UINT64,
        }
    elif dtype_kind == "M":
        valid_types = {
            plc.TypeId.TIMESTAMP_SECONDS,
            plc.TypeId.TIMESTAMP_MILLISECONDS,
            plc.TypeId.TIMESTAMP_MICROSECONDS,
            plc.TypeId.TIMESTAMP_NANOSECONDS,
        }
        if isinstance(dtype, pd.DatetimeTZDtype):
            if dtype != get_compatible_timezone(dtype):
                raise ValueError(
                    f"DatetimeTZDtype {dtype} has a non-zoneinfo timezone. "
                    "Normalize with get_compatible_timezone before calling "
                    "ColumnBase.create."
                )
    elif dtype_kind == "m":
        valid_types = {
            plc.TypeId.DURATION_SECONDS,
            plc.TypeId.DURATION_MILLISECONDS,
            plc.TypeId.DURATION_MICROSECONDS,
            plc.TypeId.DURATION_NANOSECONDS,
        }
    elif dtype_kind in "iufb":
        valid_types = {
            plc.TypeId.INT8,
            plc.TypeId.INT16,
            plc.TypeId.INT32,
            plc.TypeId.INT64,
            plc.TypeId.UINT8,
            plc.TypeId.UINT16,
            plc.TypeId.UINT32,
            plc.TypeId.UINT64,
            plc.TypeId.FLOAT32,
            plc.TypeId.FLOAT64,
            plc.TypeId.BOOL8,
        }
    else:
        raise TypeError(f"Unsupported dtype {dtype}.")

    if col.type().id() not in valid_types:
        raise ValueError(
            "plc_column must be a pylibcudf.Column with a TypeId in "
            f"{valid_types}. If normalization is required, please run the "
            "column through _normalize_types_column first."
        )
    return _rebuild_column(col, children, wrap_buffers=True)


class MaskCAIWrapper:
    # A wrapper that exposes the __cuda_array_interface__ of a mask that accounts for
    # the mask being a bitmask in the mask size calculation.

    def __init__(self, mask: Any) -> None:
        self._mask = mask

    @property
    def __cuda_array_interface__(self) -> Mapping:
        cai = self._mask.__cuda_array_interface__.copy()
        cai["shape"] = (
            plc.null_mask.bitmask_allocation_size_bytes(cai["shape"][0]),
        )
        return cai

    @property
    def ptr(self) -> int:
        """Device pointer (Span protocol)."""
        return self._mask.__cuda_array_interface__["data"][0]

    @property
    def size(self) -> int:
        """Size in bytes (Span protocol)."""
        return plc.null_mask.bitmask_allocation_size_bytes(
            self._mask.__cuda_array_interface__["shape"][0]
        )


def _handle_nulls(arrow_array: pa.Array, nested: bool = False) -> pa.Array:
    # Recursively replace all-null nested arrays with arrow NullArrays and make all
    # types nullable in the schema even if the columns contain no nulls
    array_type = arrow_array.type

    # All-null nested (except interval types) should become pyarrow null
    # arrays to match how pyarrow ingests such pandas objects.
    # All-null string arrays nested inside list/struct children should also
    # become null arrays, but top-level string columns should preserve their
    # string type to match pandas str dtype behavior.
    is_interval_type = (
        pa.types.is_struct(array_type)
        and len(array_type) == 2
        and array_type[0].name == "left"
        and array_type[1].name == "right"
    )
    is_all_null = arrow_array.null_count == len(arrow_array)
    is_nested_type = pa.types.is_nested(array_type) and not is_interval_type
    is_string_type = pa.types.is_string(
        array_type
    ) or pa.types.is_large_string(array_type)
    if is_all_null and (is_nested_type or (nested and is_string_type)):
        return pa.NullArray.from_buffers(
            pa.null(), len(arrow_array), [pa.py_buffer(b"")]
        )

    if pa.types.is_struct(array_type):
        arrow_array = cast("pa.StructArray", arrow_array)

        new_fields = []
        requires_reconstruction = False
        for i, subfield in enumerate(array_type):
            new_field_array = field_array = arrow_array.field(i)
            field_type = subfield.type
            if (
                pa.types.is_nested(field_type)
                or pa.types.is_string(field_type)
                or pa.types.is_large_string(field_type)
            ):
                new_field_array = _handle_nulls(field_array, nested=True)
            new_fields.append(new_field_array)
            # Reconstruct if we replaced nulls in children or need nullability change
            requires_reconstruction = (
                requires_reconstruction
                or (new_field_array is not field_array)
                or not subfield.nullable
                and not pa.types.is_null(field_type)
            )

        if requires_reconstruction:
            new_struct_type = pa.struct(
                [
                    pa.field(at.name, nf.type, nullable=True)
                    for at, nf in zip(array_type, new_fields, strict=True)
                ]
            )
            # Only need validity buffer for structs
            buffers = cast("list[pa.Buffer | None]", arrow_array.buffers()[:1])
            return pa.StructArray.from_buffers(
                new_struct_type,
                len(arrow_array),
                buffers,
                children=new_fields,
                null_count=arrow_array.null_count,
            )
    elif pa.types.is_list(array_type):
        arrow_array = cast("pa.ListArray", arrow_array)

        values = arrow_array.values
        new_values = _handle_nulls(values, nested=True)

        value_field = array_type.value_field
        has_non_nullable_field = (
            not value_field.nullable and not pa.types.is_null(value_field.type)
        )

        if new_values is not values or has_non_nullable_field:
            buffers = cast("list[pa.Buffer | None]", arrow_array.buffers()[:2])
            list_type = pa.list_(
                pa.field(value_field.name, new_values.type, nullable=True)
            )
            return pa.ListArray.from_buffers(
                list_type,
                len(arrow_array),
                buffers,
                children=[new_values],
                null_count=arrow_array.null_count,
            )
    # For other primitives (int/float/etc), preserve type even if all null
    return arrow_array


class _ColumnAccessContext:
    """Context manager for access mode control on underlying buffers."""

    __slots__ = ("_column", "_kwargs", "_stack")

    _stack: ExitStack

    def __init__(self, column: ColumnBase, **kwargs: Any):
        """Initialize column access context.

        Parameters
        ----------
        column : ColumnBase
            The column to manage access for.
        **kwargs
            Parameters to propagate to buffer access (e.g., mode, scope).
        """
        self._column = column
        self._kwargs = kwargs
        self._stack = ExitStack()

    def _enter_plc(self, plc_column: plc.Column) -> None:
        """Enter context for all child columns."""
        if (data := plc_column.data()) is not None:
            self._stack.enter_context(
                cast("Buffer", data).access(**self._kwargs)
            )
        if (mask := plc_column.null_mask()) is not None:
            self._stack.enter_context(
                cast("Buffer", mask).access(**self._kwargs)
            )
        for child in plc_column.children():
            self._enter_plc(child)

    def __enter__(self) -> ColumnBase:
        """Enter the context, setting up access for all column buffers."""
        # Propagate all kwargs transparently to all buffers
        self._enter_plc(self._column.plc_column)
        return self._column

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self._stack.__exit__(exc_type, exc_val, exc_tb)
        return False


class ColumnBase(Serializable, BinaryOperand, Reducible):
    """
    A ColumnBase stores columnar data in device memory.

    A ColumnBase is composed of:

    * A pylibcudf.Column
    * A valid pandas data type object reflecting the type of the values.
    """

    _VALID_REDUCTIONS = {
        "any",
        "all",
        "min",
        "max",
    }
    plc_column: plc.Column
    _dtype: DtypeObj
    _distinct_count: dict[bool, int]
    _has_nulls: dict[bool, bool]
    _exposed_buffers: set[Buffer]
    _CACHED_PROPERTY_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Pre-compute the set of all cached properties for efficient cache clearing
        cached_props = set()
        for base_cls in cls.__mro__:
            for attr_name, attr_value in base_cls.__dict__.items():
                if isinstance(attr_value, cached_property):
                    cached_props.add(attr_name)
        cls._CACHED_PROPERTY_NAMES = frozenset(cached_props)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(
            "ColumnBase and its subclasses must be instantiated via from_pylibcudf."
        )

    @property
    def _PANDAS_NA_VALUE(self) -> ScalarLike:
        """Return appropriate NA value based on dtype."""
        if cudf.get_option(
            "mode.pandas_compatible"
        ) and is_pandas_nullable_extension_dtype(self.dtype):
            return self.dtype.na_value
        return pd.NA

    @property
    def dtype(self) -> DtypeObj:
        return self._dtype

    @cached_property
    def size(self) -> int:
        return self.plc_column.size()

    @cached_property
    def data(self) -> None | Buffer:
        """Get data buffer from pylibcudf column."""
        return cast("Buffer | None", self.plc_column.data())

    @cached_property
    def nullable(self) -> bool:
        return self.mask is not None

    def has_nulls(self, include_nan: bool = False) -> bool:
        """Check if column has null values.

        NaN inclusion is supported for specific dtypes only.
        """
        try:
            return self._has_nulls[include_nan]
        except KeyError:
            result = int(self.null_count) != 0
            self._has_nulls[include_nan] = result
            return result

    @cached_property
    def is_all_null(self) -> bool:
        """Check if all values in the column are null.

        Returns True if the column is empty or all values are null.
        This is more readable than checking null_count == len(self).
        """
        return self.null_count == len(self)

    @cached_property
    def valid_count(self) -> int:
        """Return the number of non-null values in the column.

        This is equivalent to len(self) - self.null_count but cached
        as a property for convenience.
        """
        return len(self) - self.null_count

    @cached_property
    def nan_count(self) -> int:
        return 0

    @cached_property
    def count(self) -> int:  # type: ignore[override]
        """Return the number non-NA and NaN values in the column for the public count API."""
        return self.valid_count - (
            self.nan_count
            if not is_pandas_nullable_extension_dtype(self.dtype)
            else 0
        )

    @cached_property
    def mask(self) -> None | Buffer:
        """Get mask buffer from pylibcudf column."""
        return cast("Buffer | None", self.plc_column.null_mask())

    def access(self, **kwargs: Any) -> _ColumnAccessContext:
        """Context manager for controlled buffer access.

        Mediates access to all the underlying buffers of the column. Within this
        context, all their ptr accesses will respect the specified access parameters.

        Applies all parameters transparently to all underlying buffers in the
        column hierarchy (data, mask, and children).

        Parameters
        ----------
        **kwargs
            Parameters for buffer access (e.g., mode, scope).
            - mode : {"read", "write"} - Access mode for copy-on-write
            - scope : {"internal", "external"} - Spill scope (SpillableBuffer only)

        Returns
        -------
        _ColumnAccessContext
            A context manager that manages access to all column buffers.
        """
        return _ColumnAccessContext(self, **kwargs)

    def _clear_cache(self) -> None:
        self._distinct_count.clear()
        self._has_nulls.clear()
        for attr_name in self._CACHED_PROPERTY_NAMES:
            try:
                delattr(self, attr_name)
            except AttributeError:
                pass

    def set_mask(self, mask: Buffer | None, null_count: int) -> Self:
        """
        Replaces the mask buffer of the column and returns a new column.

        Parameters
        ----------
        mask : Buffer or None
            The new null mask buffer, or None to clear the mask.
        null_count : int
            The number of null values.
        """
        if isinstance(mask, Buffer):
            new_mask = mask
            new_null_count = null_count
        elif mask is None:
            new_mask = None
            new_null_count = 0
        else:
            raise ValueError(
                f"Expected a Buffer object or None for mask, got {type(mask).__name__}"
            )
        new_plc_column = self.plc_column.with_mask(new_mask, new_null_count)
        return cast(
            "Self",
            ColumnBase.create(new_plc_column, self.dtype),
        )

    @cached_property
    def null_count(self) -> int:
        return self.plc_column.null_count()

    @cached_property
    def offset(self) -> int:
        return self.plc_column.offset()

    def _mimic_inplace(
        self, other_col: Self, inplace: bool = False
    ) -> None | Self:
        """
        Given another column, update the attributes of this column to mimic an
        inplace operation. This does not modify the memory of Buffers, but
        instead replaces the Buffers and other attributes underneath the column
        object with the Buffers and attributes from the other column.
        """
        if inplace:
            self._dtype = other_col._dtype
            self.plc_column = other_col.plc_column
            self._clear_cache()
            return None
        else:
            return other_col

    @staticmethod
    def _unwrap_buffer_to_span(buffer: Buffer | Any | None) -> Any:
        """Unwrap a cudf Buffer to a span.

        If the input is a cudf Buffer, extract the underlying span. Otherwise,
        return the input as-is (already a span or None).

        Parameters
        ----------
        buffer : Buffer or None
            The buffer to unwrap.

        Returns
        -------
        plc.span.Span or None
            The underlying span, or None if input was None.
        """
        if buffer is None:
            return None
        return cast("plc.span.Span", cast("Buffer", buffer).owner.owner)

    @staticmethod
    def _unwrap_buffers(plc_column: plc.Column) -> plc.Column:
        """Recursively unwrap all buffers in a pylibcudf.Column.

        This function will traverse the provided pylibcudf Column and unwrap
        all data and mask buffers from cudf Buffers, removing cudf memory
        semantics.

        Parameters
        ----------
        col : pylibcudf.Column
            The column to unwrap.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column with unwrapped buffers.
        """
        data = ColumnBase._unwrap_buffer_to_span(plc_column.data())
        mask = ColumnBase._unwrap_buffer_to_span(plc_column.null_mask())

        unwrapped_children = [
            ColumnBase._unwrap_buffers(child)
            for child in plc_column.children()
        ]

        return plc.Column(
            data_type=plc_column.type(),
            size=plc_column.size(),
            data=data,
            mask=mask,
            null_count=plc_column.null_count(),
            offset=plc_column.offset(),
            children=unwrapped_children,
            validate=False,
        )

    def to_pylibcudf(self) -> plc.Column:
        """Convert this Column to a pylibcudf.Column.

        This function will generate a pylibcudf Column with unwrapped buffers,
        removing cudf memory semantics. The result is a view of the existing data.
        It is always a zero-copy operation, so changes to the pylibcudf data will
        be reflected back to the cudf object and vice versa. Users are responsible
        for making a copy if they wish to avoid this behavior. Note that due to
        copy-on-write semantics in cudf modifying the pylibcudf object may affect
        multiple preexisting cudf objects viewing the same buffer.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column with unwrapped buffers.
        """
        return ColumnBase._unwrap_buffers(self.plc_column)

    @staticmethod
    def create(
        col: plc.Column, dtype: DtypeObj, validate: bool = True
    ) -> ColumnBase:
        """
        Create a Column from a pylibcudf.Column with an explicit cudf dtype.

        This is the primary factory for ColumnBase construction. It always requires
        an explicit dtype to ensure type safety. If you need to infer the dtype from
        the pylibcudf Column, use dtype_from_pylibcudf_column() first:

            dtype = dtype_from_pylibcudf_column(plc_col)
            col = ColumnBase.create(plc_col, dtype)

        Note that when validation is enabled, the input col is never directly placed
        into the resulting ColumnBase. Rather, a new plc.Column is created with the
        exact same properties but with suitable shallow copies of the buffers wrapped
        in cudf Buffers to ensure consistent behavior with respect to memory semantics
        like copy-on-write. When validation is disabled, the caller is responsible for
        ensuring that col and its children are already normalized and wrapped.
        """
        # Dispatch to the appropriate subclass based on dtype
        target_cls = ColumnBase._dispatch_subclass_from_dtype(dtype)
        self = target_cls.__new__(target_cls)
        self.plc_column = _wrap_and_validate(col, dtype) if validate else col
        self._dtype = dtype
        self._distinct_count = {}
        self._has_nulls = {}
        # The set of exposed buffers associated with this column. These buffers must be
        # kept alive for the lifetime of this column since anything that accessed the
        # CAI of this column will still be pointing to those buffers. As such objects
        # are destroyed, all references to this column will be removed as well,
        # triggering the destruction of the exposed buffers.
        self._exposed_buffers = set()
        return self

    @staticmethod
    def _dispatch_subclass_from_dtype(dtype: DtypeObj) -> type[ColumnBase]:
        """
        Dispatch to the appropriate ColumnBase subclass based on dtype.

        This function determines which ColumnBase subclass should be used
        to construct a column with the given dtype.
        """
        dtype_kind = dtype.kind

        if is_dtype_obj_datetime_tz(dtype):
            return cudf.core.column.datetime.DatetimeTZColumn
        if isinstance(dtype, CategoricalDtype):
            return cudf.core.column.CategoricalColumn

        # Temporal types (by kind)
        if dtype_kind == "M":
            return cudf.core.column.DatetimeColumn
        if dtype_kind == "m":
            return cudf.core.column.TimeDeltaColumn

        # String types
        if is_dtype_obj_string(dtype):
            return cudf.core.column.StringColumn

        # cuDF custom types
        if is_dtype_obj_list(dtype):
            return cudf.core.column.ListColumn
        if is_dtype_obj_interval(dtype):
            return cudf.core.column.IntervalColumn
        if is_dtype_obj_struct(dtype):
            return cudf.core.column.StructColumn
        if is_dtype_obj_decimal(dtype):
            return cudf.core.column.DecimalColumn

        # Numerical types
        if is_dtype_obj_numeric(dtype, include_decimal=False):
            return cudf.core.column.NumericalColumn

        raise TypeError(f"Unrecognized dtype: {dtype}")

    @staticmethod
    def from_pylibcudf(col: plc.Column) -> ColumnBase:
        """Create a Column from a pylibcudf.Column.

        This function will generate a Column pointing to the provided pylibcudf
        Column. It will directly access the data and mask buffers of the
        pylibcudf Column, so the newly created object is not tied to the
        lifetime of the original pylibcudf.Column.

        Parameters
        ----------
        col : pylibcudf.Column
            The object to copy.

        Returns
        -------
        pylibcudf.Column
            A new pylibcudf.Column referencing the same data.
        """
        dtype = dtype_from_pylibcudf_column(col)
        return ColumnBase.create(col, dtype)

    @classmethod
    def from_cuda_array_interface(cls, arbitrary: Any) -> ColumnBase:
        """
        Create a Column from an object implementing the CUDA array interface.

        Parameters
        ----------
        arbitrary : Any
            The object to convert.

        Returns
        -------
        Column
        """
        if (
            cai := getattr(arbitrary, "__cuda_array_interface__", None)
        ) is None:
            raise ValueError(
                "Object does not implement __cuda_array_interface__"
            )

        cai_dtype = np.dtype(cai["typestr"])
        check_invalid_array(cai["shape"], cai_dtype)
        # Reshape ndarrays compatible with cuDF columns
        if len(cai["shape"]) == 0:
            arbitrary = cp.asarray(arbitrary)[np.newaxis]
        if not plc.column.is_c_contiguous(
            cai["shape"], cai["strides"], cai_dtype.itemsize
        ):
            arbitrary = cp.ascontiguousarray(arbitrary)

        # TODO: Can remove once from_cuda_array_interface can handle masks
        # https://github.com/rapidsai/cudf/issues/19122
        if (mask := cai.get("mask", None)) is not None:
            cai_copy = cai.copy()
            cai_copy.pop("mask")
            arbitrary = SimpleNamespace(__cuda_array_interface__=cai_copy)
        else:
            mask = None

        plc_column = plc.Column.from_cuda_array_interface(arbitrary)
        column = ColumnBase.create(
            plc_column, dtype_from_pylibcudf_column(plc_column)
        )
        if mask is not None:
            cai_mask = mask.__cuda_array_interface__
            if cai_mask["typestr"][1] == "t":
                mask_buff = as_buffer(MaskCAIWrapper(mask))
            elif cai_mask["typestr"][1] == "b":
                mask_buff, null_count = ColumnBase.from_cuda_array_interface(
                    mask,
                ).as_mask()
            else:
                mask_buff = as_buffer(mask)
            required_num_bytes = -(-column.size // 8)  # ceiling divide
            if mask_buff.size < required_num_bytes:
                raise ValueError(
                    f"The value for mask is smaller than expected, got {mask.size} bytes, "
                    f"expected {required_num_bytes} bytes."
                )
            # Compute null_count for external masks that don't provide it
            if cai_mask["typestr"][1] != "b":
                null_count = plc.null_mask.null_count(
                    mask_buff, 0, column.size
                )
            column = column.set_mask(mask_buff, null_count)
        return column

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"{object.__repr__(self)}\n"
            f"{self.to_arrow().to_string()}\n"
            f"dtype: {self.dtype}"
        )

    def _prep_pandas_compat_repr(
        self, nan_rep: str | None = None
    ) -> StringColumn | Self:
        """
        Preprocess Column to be compatible with pandas repr, namely handling nulls.

        * null (datetime/timedelta) = str(pd.NaT)
        * null (other types)= str(pd.NA)
        """
        if nan_rep is None:
            nan_rep = "NaN"
        if self.has_nulls():
            return cast(
                "cudf.core.column.StringColumn",
                self.astype(np.dtype("str")).fillna(
                    nan_rep
                    if self._PANDAS_NA_VALUE is np.nan
                    else str(self._PANDAS_NA_VALUE)
                ),
            )
        return self

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        """Convert object to pandas type.

        The default implementation falls back to PyArrow for the conversion.
        """
        if arrow_type and nullable:
            raise ValueError(
                f"{arrow_type=} and {nullable=} cannot both be set."
            )
        pa_array = self.to_arrow()

        if arrow_type or (
            not nullable and isinstance(self.dtype, pd.ArrowDtype)
        ):
            return pd.Index(
                pd.arrays.ArrowExtensionArray(pa_array), copy=False
            )
        elif nullable or (
            not arrow_type and is_pandas_nullable_extension_dtype(self.dtype)
        ):
            pandas_nullable_dtype = np_dtypes_to_pandas_dtypes.get(
                self.dtype, self.dtype
            )
            pandas_array = pandas_nullable_dtype.__from_arrow__(pa_array)
            return pd.Index(pandas_array, copy=False)
        else:
            # xref https://github.com/rapidsai/cudf/issues/21120
            # TODO: Revisit using pa_array.to_pandas() once pandas 3.0 is supported
            np_array = pa_array.to_numpy(zero_copy_only=False, writable=True)
            return pd.Index(
                np_array,
                dtype=np_array.dtype,
                copy=False,
            )

    def to_numpy(self) -> np.ndarray:
        """
        Convert the Column to a NumPy array.

        Returns
        -------
        numpy.ndarray
        """
        if is_dtype_obj_numeric(self.dtype):
            return self.values.get()
        return self.to_pandas().to_numpy()

    @property
    def values(self) -> cp.ndarray:
        """Return a CuPy representation of the Column."""
        raise NotImplementedError(f"CuPy does not support {self.dtype}")

    @staticmethod
    def _prepare_find_and_replace_columns(
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        *,
        null_cast_dtype: DtypeObj | None = None,
    ) -> tuple[ColumnBase, ColumnBase]:
        # If all of `to_replace`/`replacement` are `None`, the dtype of
        # `to_replace_col`/`replacement_col` is inferred as `string`so we need to
        # type-cast to self.dtype.
        to_replace_col = as_column(to_replace)
        if null_cast_dtype is not None and to_replace_col.is_all_null:
            to_replace_col = to_replace_col.astype(null_cast_dtype)

        replacement_col = as_column(replacement)
        if null_cast_dtype is not None and replacement_col.is_all_null:
            replacement_col = replacement_col.astype(null_cast_dtype)

        if type(to_replace_col) is not type(replacement_col):
            raise TypeError(
                "to_replace and value should be of same types,"
                f"got to_replace dtype: {to_replace_col.dtype} and "
                f"value dtype: {replacement_col.dtype}"
            )

        return to_replace_col, replacement_col

    @staticmethod
    def _dedupe_find_and_replace_mapping(
        old_col: ColumnBase,
        new_col: ColumnBase,
    ) -> tuple[plc.Column, plc.Column]:
        """Deduplicate old/new mapping with keep-last semantics.

        This replicates pandas' behavior when to_replace has duplicates:
        pandas processes replacements sequentially, so the last occurrence wins.
        """
        with access_columns(old_col, new_col, mode="read", scope="internal"):
            columns = plc.stream_compaction.stable_distinct(
                plc.Table([old_col.plc_column, new_col.plc_column]),
                keys=[0],
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                nulls_equal=plc.types.NullEquality.EQUAL,
                nans_equal=plc.types.NanEquality.ALL_EQUAL,
            ).columns()
        return columns[0], columns[1]

    def _find_and_replace_with_dedup(
        self,
        old_col: ColumnBase,
        new_col: ColumnBase,
        result_dtype: DtypeObj,
    ) -> ColumnBase:
        """Apply find/replace using keep-last mapping and null handling."""
        old_plc, new_plc = ColumnBase._dedupe_find_and_replace_mapping(
            old_col, new_col
        )

        replaced = self
        if old_plc.null_count() == 1:
            old_isnull_plc = plc.unary.is_null(old_plc)
            (filtered_column,) = plc.stream_compaction.apply_boolean_mask(
                plc.Table([new_plc]), old_isnull_plc
            ).columns()
            replacement_for_null = filtered_column.to_scalar().to_py()
            replaced = replaced.fillna(replacement_for_null)

            old_plc, new_plc = plc.stream_compaction.drop_nulls(
                plc.Table([old_plc, new_plc]),
                keys=[0],
                keep_threshold=1,
            ).columns()

        with replaced.access(mode="read", scope="internal"):
            result_plc = plc.replace.find_and_replace_all(
                replaced.plc_column,
                old_plc,
                new_plc,
            )
        return ColumnBase.create(result_plc, result_dtype)

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        raise NotImplementedError

    def clip(self, lo: ScalarLike, hi: ScalarLike) -> Self:
        lo_scalar = pa_scalar_to_plc_scalar(
            pa.scalar(lo, type=cudf_dtype_to_pa_type(self.dtype))
        )
        hi_scalar = pa_scalar_to_plc_scalar(
            pa.scalar(hi, type=cudf_dtype_to_pa_type(self.dtype))
        )

        return cast(
            "Self",
            PylibcudfFunction(
                plc.replace.clamp,
                same_dtype_policy,
            ).execute_with_args(self, lo_scalar, hi_scalar),
        )

    def equals(self, other: ColumnBase, check_dtypes: bool = False) -> bool:
        if not isinstance(other, ColumnBase) or len(self) != len(other):
            return False
        elif self is other:
            return True
        elif check_dtypes and self.dtype != other.dtype:
            return False
        elif not check_dtypes and not _can_values_be_equal(
            self.dtype, other.dtype
        ):
            return False
        elif self.null_count != other.null_count:
            return False
        ret = self._binaryop(other, "NULL_EQUALS")
        if ret is NotImplemented:
            return False
        return bool(ret.all())

    def all(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        if self.size == 0:
            return True
        if self.null_count == self.size:
            if not skipna:
                return _get_nan_for_dtype(self.dtype)
            else:
                return True

        # For all(), we want NaN values to be treated as truthy.
        # Call _reduce() with skipna=True to get the boolean result.
        result = self._reduce(
            "all", skipna=True, min_count=min_count, **kwargs
        )
        if result is pd.NA or np.isnan(result):
            # Empty after dropping NaN/nulls - return np.bool_
            result = np.bool_(True)

        # For pandas nullable extension dtypes with skipna=False and nulls, return NaN
        if (
            result
            and not skipna
            and self.null_count > 0
            and is_pandas_nullable_extension_dtype(self.dtype)
        ):
            return _get_nan_for_dtype(self.dtype)
        return result

    def any(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        if self.size == 0:
            return False
        is_masked_dtype = is_pandas_nullable_extension_dtype(self.dtype)
        if not skipna and (
            self.nan_count > 0 or (not is_masked_dtype and self.has_nulls())
        ):
            # NaN values (and the NaN null sentinel of numpy dtypes) are
            # truthy. For pandas nullable extension dtypes <NA> is not
            # truthy; Kleene logic below decides between True and <NA>.
            return True
        if self.null_count == self.size:
            if not skipna:
                # All-null nullable column with skipna=False: Kleene
                # any([NA, ...]) with no True values is <NA>.
                return _get_nan_for_dtype(self.dtype)
            return False

        # For any(), we want NaN values to be treated as truthy.
        # Call _reduce() with skipna=True to get the boolean result.
        result = self._reduce(
            "any", skipna=True, min_count=min_count, **kwargs
        )
        if result is pd.NA or np.isnan(result):
            # Empty after dropping NaN/nulls
            # If skipna=False, NaN values should be treated as truthy
            result = np.bool_(not skipna)

        # For pandas nullable extension dtypes with skipna=False, a False
        # result in the presence of nulls is <NA> under Kleene logic.
        if (
            not result
            and not skipna
            and self.null_count > 0
            and is_masked_dtype
        ):
            return _get_nan_for_dtype(self.dtype)
        return result

    def dropna(self) -> Self:
        if self.has_nulls():
            return cast(
                "Self",
                PylibcudfFunction(
                    plc.stream_compaction.drop_nulls,
                    same_dtype_policy,
                    result_index=0,
                ).execute_with_args(ColumnList(self), [0], 1),
            )
        else:
            return self.copy()

    def to_arrow(self) -> pa.Array:
        if is_arrow_null_dtype(self.dtype):
            return pa.nulls(len(self))
        with self.access(mode="read", scope="internal"):
            return _handle_nulls(
                self.plc_column.to_arrow(
                    metadata=_dtype_to_metadata(self.dtype)
                )
            )

    @classmethod
    def from_range(cls, rng: range) -> ColumnBase:
        """
        Create a Column from a range object.

        Parameters
        ----------
        rng : range
            The range to create the Column from.

        Returns
        -------
        Column
        """
        plc_result = plc.filling.sequence(
            len(rng),
            plc.Scalar.from_py(rng.start),
            plc.Scalar.from_py(rng.step),
        )
        return cls.create(plc_result, dtype_from_pylibcudf_column(plc_result))

    @classmethod
    def from_arrow(cls, array: pa.Array | pa.ChunkedArray) -> ColumnBase:
        """
        Convert PyArrow Array/ChunkedArray to column

        Parameters
        ----------
        array : PyArrow Array/ChunkedArray

        Returns
        -------
        column

        Examples
        --------
        >>> import pyarrow as pa
        >>> import cudf
        >>> cudf.core.column.ColumnBase.from_arrow(pa.array([1, 2, 3, 4]))
        <cudf.core.column.numerical.NumericalColumn object at 0x7f8865497ef0>
        [
          1,
          2,
          3,
          4
        ]
        dtype: int8
        """
        if not isinstance(array, (pa.Array, pa.ChunkedArray)):
            raise TypeError("array should be PyArrow array or chunked array")
        elif pa.types.is_float16(array.type):
            raise NotImplementedError(
                "Type casting from `float16` to `float32` is not "
                "yet supported in pyarrow, see: "
                "https://github.com/apache/arrow/issues/20213"
            )
        elif pa.types.is_null(array.type):
            # default "empty" type
            array = array.cast(pa.string())

        if pa.types.is_dictionary(array.type):
            if isinstance(array, pa.Array):
                dict_array = cast("pa.DictionaryArray", array)
                codes: pa.Array | pa.ChunkedArray = dict_array.indices
                dictionary: pa.Array | pa.ChunkedArray = dict_array.dictionary
            else:
                codes = pa.chunked_array(
                    [
                        cast("pa.DictionaryArray", chunk).indices
                        for chunk in array.chunks
                    ],
                    type=array.type.index_type,
                )
                dictionary = pc.unique(
                    pa.chunked_array(
                        [
                            cast("pa.DictionaryArray", chunk).dictionary
                            for chunk in array.chunks
                        ],
                        type=array.type.value_type,
                    )
                )
            categories_dtype = cudf_dtype_from_pa_type(dictionary.type)
            categories_plc = _normalize_types_column(
                plc.Column.from_arrow(dictionary)
            )
            categories = cls.create(categories_plc, categories_dtype)

            categorical_dtype = CategoricalDtype(
                categories=categories, ordered=array.type.ordered
            )
            expected_codes_dtype = dtype_to_pylibcudf_type(
                categorical_dtype._codes_dtype
            )
            codes_plc = _normalize_types_column(plc.Column.from_arrow(codes))
            if codes_plc.type() != expected_codes_dtype:
                codes_plc = plc.unary.cast(codes_plc, expected_codes_dtype)
            return ColumnBase.create(codes_plc, categorical_dtype)
        else:
            plc_column = plc.Column.from_arrow(array)
            dtype = cudf_dtype_from_pa_type(array.type)
            normalized = _normalize_types_column(plc_column)
            return cls.create(normalized, dtype)

    def _get_mask_as_column(self) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.transform.mask_to_bools(
                self.mask.ptr,  # type: ignore[union-attr]
                self.offset,
                self.offset + len(self),
            )
            return ColumnBase.create(plc_column, np.dtype(np.bool_))

    @staticmethod
    def _plc_memory_usage(col: plc.Column) -> int:
        n = 0
        if (data := col.data()) is not None:
            try:
                # Only count the actual data in use, not the entire base buffer
                # For sliced columns, col.size() * itemsize gives actual usage
                typestr: str = col.type().typestr  # type: ignore[assignment]
                itemsize = np.dtype(typestr).itemsize
                n += col.size() * itemsize
            except NotImplementedError:
                # No typestr available (e.g. STRING)
                n += data.size
        if col.null_mask() is not None:
            n += plc.null_mask.bitmask_allocation_size_bytes(col.size())
        for child in col.children():
            n += ColumnBase._plc_memory_usage(child)
        return n

    @cached_property
    def memory_usage(self) -> int:
        return self._plc_memory_usage(self.plc_column)

    def _fill(
        self,
        fill_value: plc.Scalar,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Self | None:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        if not inplace or is_dtype_obj_string(self.dtype):
            with self.access(mode="read", scope="internal"):
                result = cast(
                    "Self",
                    ColumnBase.create(
                        plc.filling.fill(
                            self.plc_column,
                            begin,
                            end,
                            fill_value,
                        ),
                        self.dtype,
                    ),
                )
            if is_dtype_obj_string(self.dtype):
                return self._mimic_inplace(result, inplace=True)
            return result

        if not fill_value.is_valid(DEFAULT_STREAM) and not self.nullable:
            # Create mask sized for base buffer to preserve view semantics
            mask = as_buffer(
                plc.null_mask.create_null_mask(
                    self.size, plc.types.MaskState.ALL_VALID
                )
            )
            self.plc_column = self.plc_column.with_mask(
                mask, 0, validate=False
            )
            self._clear_cache()

        with self.access(mode="read", scope="internal"):
            with self.access(mode="write"):
                plc.filling.fill_in_place(
                    self.plc_column,
                    begin,
                    end,
                    fill_value,
                )
            self._clear_cache()
        return self

    def shift(self, offset: int, fill_value: ScalarLike) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_fill_value = self._scalar_to_plc_scalar(fill_value)
            plc_col = plc.copying.shift(
                self.plc_column,
                offset,
                plc_fill_value,
            )
            return cast(
                "Self",
                ColumnBase.create(plc_col, self.dtype),
            )

    def copy(self, deep: bool = True) -> Self:
        """
        Makes a copy of the Column.

        Parameters
        ----------
        deep : bool, default True
            If True, a true physical copy of the column
            is made.
            If False, the same memory is shared between the
            buffers of the Column via copy-on-write. A physical
            copy is made when a write operation is performed.
        """
        plc_col = self.plc_column
        if deep:
            plc_col = plc_col.copy()
        return cast("Self", ColumnBase.create(plc_col, self.dtype))

    def element_indexing(self, index: int) -> ScalarLike:
        """
        Access an element from the column at the given index.

        Parameters
        ----------
        index : int
            The index of the element to access.

        Returns
        -------
        ScalarLike
            The element at the given index.

        Raises
        ------
        ``IndexError`` if out-of-bound
        """
        if index < 0:
            index = len(self) + index
        if index > len(self) - 1 or index < 0:
            raise IndexError("single positional indexer is out-of-bounds")
        with self.access(mode="read", scope="internal"):
            plc_scalar = plc.copying.get_element(
                self.plc_column,
                index,
            )
        pa_scalar = plc_scalar.to_arrow(
            metadata=_dtype_to_metadata(self.dtype),
        )
        if not pa_scalar.is_valid:
            return self._PANDAS_NA_VALUE
        else:
            return _replace_nested_none_with_NA(pa_scalar.as_py())

    def slice(self, start: int, stop: int, stride: int | None = None) -> Self:
        stride = 1 if stride is None else stride
        if stop < 0 and not (stride < 0 and stop == -1 and start >= 0):
            stop = stop + len(self)
        if start < 0:
            start = start + len(self)
        if (stride > 0 and start >= stop) or (stride < 0 and start <= stop):
            return cast("Self", column_empty(0, self.dtype))
        # compute mask slice
        if stride == 1:
            with self.access(mode="read", scope="internal"):
                (result,) = plc.copying.slice(
                    self.plc_column,
                    [start, stop],
                )
            return cast("Self", ColumnBase.create(result, self.dtype))
        else:
            # Need to create a gather map for given slice with stride
            gather_map = ColumnBase.from_range(
                range(start, stop, stride)
            ).astype(SIZE_TYPE_DTYPE)
            return self.take(gather_map)

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if is_scalar(value):
            if value is cudf.NA:
                value = None
            try:
                pa_scalar = pa.scalar(
                    value, type=cudf_dtype_to_pa_type(self.dtype)
                )
            except ValueError as err:
                raise TypeError(
                    f"Cannot set value of type {type(value)} to column of type {self.dtype}"
                ) from err
            return pa_scalar_to_plc_scalar(pa_scalar)
        else:
            return as_column(value, dtype=self.dtype)

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Set the value of ``self[key]`` to ``value``.

        If ``value`` and ``self`` are of different types, ``value`` is coerced
        to ``self.dtype``. Assumes ``self`` and ``value`` are index-aligned.
        """
        value_normalized = self._cast_setitem_value(value)
        if isinstance(key, slice):
            out: ColumnBase | None = self._scatter_by_slice(
                key, value_normalized
            )
        else:
            key = as_column(key)
            if len(key) == 0:
                key = key.astype(SIZE_TYPE_DTYPE)
            if not is_dtype_obj_numeric(key.dtype):
                raise ValueError(f"Invalid scatter map type {key.dtype}.")
            out = self._scatter_by_column(key, value_normalized)

        if out:
            self._mimic_inplace(out, inplace=True)

    def _all_bools_with_nulls(
        self, other: ColumnBase | ScalarLike, bool_fill_value: bool
    ) -> ColumnBase:
        # Might be able to remove if we share more of
        # DatetimeColumn._binaryop & TimedeltaColumn._binaryop
        result_mask = None
        if self.has_nulls():
            if isinstance(other, ColumnBase) and other.has_nulls():
                result_mask = (
                    self._get_mask_as_column() & other._get_mask_as_column()
                )
            elif (
                not isinstance(other, ColumnBase)
                and other not in {np.nan, None, pd.NaT, float("nan")}
                and not (isinstance(other, Decimal) and other.is_nan())
                and not (isinstance(other, float) and np.isnan(other))
            ):
                result_mask = self._get_mask_as_column()
        elif isinstance(other, ColumnBase) and other.has_nulls():
            result_mask = other._get_mask_as_column()

        result_col = as_column(
            bool_fill_value,
            dtype=get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
            length=len(self),
        )
        if result_mask is not None:
            mask_buff, null_count = result_mask.as_mask()
            result_col = result_col.set_mask(mask_buff, null_count)
        return result_col

    def _scatter_by_slice(
        self,
        key: builtins.slice,
        value: plc.Scalar | ColumnBase,
    ) -> Self | None:
        """If this function returns None, it's either a no-op (slice is empty),
        or the inplace replacement is already performed (fill-in-place).
        """
        start, stop, step = key.indices(len(self))
        if start >= stop:
            return None
        rng = range(start, stop, step)
        num_keys = len(rng)

        self._check_scatter_key_length(num_keys, value)

        if step == 1 and not isinstance(self.dtype, (StructDtype, ListDtype)):
            # NOTE: List & Struct dtypes aren't supported by both
            # inplace & out-of-place fill. Hence we need to use scatter for
            # these two types.
            if isinstance(value, plc.Scalar):
                return self._fill(value, start, stop, inplace=True)
            else:
                with self.access(mode="read", scope="internal"):
                    with self.access(mode="write"):
                        return cast(
                            "Self",
                            ColumnBase.create(
                                plc.copying.copy_range(
                                    value.plc_column,
                                    self.plc_column,
                                    0,
                                    num_keys,
                                    start,
                                ),
                                self.dtype,
                            ),
                        )

        # step != 1, create a scatter map with arange
        scatter_map = cast(
            "cudf.core.column.NumericalColumn",
            ColumnBase.from_range(rng).astype(SIZE_TYPE_DTYPE),
        )

        return self._scatter_by_column(scatter_map, value)

    def _scatter_by_column(
        self,
        key: NumericalColumn,
        value: plc.Scalar | ColumnBase,
        bounds_check: bool = True,
    ) -> Self:
        if key.dtype.kind == "b":
            # `key` is boolean mask
            if len(key) != len(self):
                raise ValueError(
                    "Boolean mask must be of same length as column"
                )
            if isinstance(value, ColumnBase) and len(self) == len(value):
                # Both value and key are aligned to self. Thus, the values
                # corresponding to the false values in key should be
                # ignored.
                value = value.apply_boolean_mask(key)
                # After applying boolean mask, the length of value equals
                # the number of elements to scatter, we can skip computing
                # the sum of ``key`` below.
                num_keys = len(value)
            else:
                # Compute the number of element to scatter by summing all
                # `True`s in the boolean mask.
                num_keys = key.sum()
        else:
            # `key` is integer scatter map
            num_keys = len(key)

        self._check_scatter_key_length(num_keys, value)

        if key.dtype.kind == "b":
            with self.access(mode="read", scope="internal"):
                with self.access(mode="write"):
                    plc_table = plc.copying.boolean_mask_scatter(
                        plc.Table([value.plc_column])
                        if isinstance(value, ColumnBase)
                        else [value],
                        plc.Table([self.plc_column]),
                        key.plc_column,
                    )
                return cast(
                    "Self",
                    ColumnBase.create(plc_table.columns()[0], self.dtype),
                )
        else:
            return cast(
                "Self",
                ColumnBase.create(
                    copying.scatter(
                        cast("list[plc.Scalar]", [value])
                        if isinstance(value, plc.Scalar)
                        else cast("list[ColumnBase]", [value]),
                        key,
                        [self],
                        bounds_check=bounds_check,
                    )[0],
                    self.dtype,
                ),
            )

    def _check_scatter_key_length(
        self, num_keys: int, value: plc.Scalar | ColumnBase
    ) -> None:
        """`num_keys` is the number of keys to scatter. Should equal to the
        number of rows in ``value`` if ``value`` is a column.
        """
        if isinstance(value, ColumnBase) and len(value) != num_keys:
            raise ValueError(
                f"Size mismatch: cannot set value "
                f"of size {len(value)} to indexing result of size "
                f"{num_keys}"
            )

    def _scalar_to_plc_scalar(self, scalar: ScalarLike) -> plc.Scalar:
        """Return a pylibcudf.Scalar that matches the type of self.dtype"""
        if not isinstance(scalar, pa.Scalar):
            scalar = pa.scalar(scalar)
        return pa_scalar_to_plc_scalar(
            scalar.cast(cudf_dtype_to_pa_type(self.dtype))
        )

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value):
            return self._scalar_to_plc_scalar(fill_value)
        return as_column(fill_value).astype(self.dtype)

    def replace(
        self, values_to_replace: Self, replacement_values: Self
    ) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.replace.find_and_replace_all,
                same_dtype_policy,
            ).execute_with_args(self, values_to_replace, replacement_values),
        )

    def repeat(self, repeats: int) -> Self:
        return cast(
            "Self",
            PylibcudfFunction(
                plc.filling.repeat,
                same_dtype_policy,
                result_index=0,
            ).execute_with_args(ColumnList(self), repeats),
        )

    def fillna(
        self,
        fill_value: ScalarLike | ColumnLike,
        method: plc.replace.ReplacePolicy | None = None,
    ) -> Self:
        """
        Fill null values with ``value``. Supports .fillna, .bfill, .ffill.

        Returns a copy with null filled.
        """
        if not self.has_nulls(include_nan=True):
            return self.copy()
        elif method is None:
            if is_scalar(fill_value) and (
                is_na_like(fill_value)
                or (
                    self.dtype.kind == "f"
                    and is_pandas_nullable_extension_dtype(self.dtype)
                    and isinstance(fill_value, float)
                    and np.isnan(fill_value)
                )
            ):
                return self.copy()
            else:
                fill_value = self._validate_fillna_value(fill_value)

        if fill_value is None and method is None:
            raise ValueError("Must specify a fill 'value' or 'method'.")

        if fill_value and method:
            raise ValueError("Cannot specify both 'value' and 'method'.")

        input_col = self.nans_to_nulls()

        with self.access(mode="read", scope="internal"):
            plc_replace: plc.replace.ReplacePolicy | plc.Scalar | plc.Column
            if method is None:
                if isinstance(fill_value, plc.Scalar):
                    plc_replace = fill_value
                else:
                    plc_replace = fill_value.plc_column
            else:
                plc_replace = method
            plc_column = plc.replace.replace_nulls(
                input_col.plc_column,
                plc_replace,
            )
        return cast(
            "Self",
            ColumnBase.create(plc_column, self.dtype),
        )

    def is_valid(self) -> ColumnBase:
        """Identify non-null values"""
        return PylibcudfFunction(
            plc.unary.is_valid, pylibcudf_result_dtype_policy
        ).execute_with_args(self)

    def isnan(self) -> ColumnBase:
        """Identify NaN values in a Column."""
        return as_column(False, length=len(self))

    def notnan(self) -> ColumnBase:
        """Identify non-NaN values in a Column."""
        return as_column(True, length=len(self))

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column."""
        if not self.has_nulls(include_nan=False):
            return as_column(False, length=len(self))
        return PylibcudfFunction(
            plc.unary.is_null, pylibcudf_result_dtype_policy
        ).execute_with_args(self)

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column."""
        if not self.has_nulls(include_nan=False):
            return as_column(True, length=len(self))
        return PylibcudfFunction(
            plc.unary.is_valid, pylibcudf_result_dtype_policy
        ).execute_with_args(self)

    def interpolate(self, index: Index) -> ColumnBase:
        if is_dtype_obj_string(self.dtype):
            # TODO: Should raise for other types?
            raise TypeError(
                f"Cannot interpolate with column of dtype {self.dtype}"
            )
        if self.nullable:
            col = self.astype(np.dtype(np.float64)).fillna(np.nan)
        else:
            col = self

        # figure out where the nans are
        mask = self.isnull()

        # trivial cases, all nan or no nans
        if not mask.any() or mask.all():
            return col.copy()

        from cudf.core.index import RangeIndex

        valid_locs = ~mask
        if isinstance(index, RangeIndex):
            # Each point is evenly spaced, index values don't matter
            known_x = cp.flatnonzero(valid_locs.values)
        else:
            known_x = index._column.apply_boolean_mask(valid_locs).values
        known_y = col.apply_boolean_mask(valid_locs).values

        result = cp.interp(index.to_cupy(), known_x, known_y)

        first_nan_idx = valid_locs.values.argmax().item()
        result[:first_nan_idx] = np.nan
        return as_column(result)

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
        """
        Find locations of value in the column

        Parameters
        ----------
        value
            Scalar to look for (cast to dtype of column), or a length-1 column

        Returns
        -------
        Column of indices that match value
        """
        if not is_scalar(value):
            raise ValueError("value must be a scalar")
        else:
            value = as_column(value, dtype=self.dtype, length=1)
        mask = value.contains(self)
        return (
            ColumnBase.from_range(range(len(self)))  # type: ignore[return-value]
            .astype(SIZE_TYPE_DTYPE)
            .apply_boolean_mask(mask)
        )

    def _find_first_and_last(self, value: ScalarLike) -> tuple[int, int]:
        indices = self.indices_of(value)
        if n := len(indices):
            return (  # type: ignore[return-value]
                indices.element_indexing(0),
                indices.element_indexing(n - 1),
            )
        else:
            raise ValueError(f"Value {value} not found in column")

    def find_first_value(self, value: ScalarLike) -> int:
        """
        Return index of first value that matches

        Parameters
        ----------
        value
            Value to search for (cast to dtype of column)

        Returns
        -------
        Index of value

        Raises
        ------
        ValueError if value is not found
        """
        first, _ = self._find_first_and_last(value)
        return first

    def find_last_value(self, value: ScalarLike) -> int:
        """
        Return index of last value that matches

        Parameters
        ----------
        value
            Value to search for (cast to dtype of column)

        Returns
        -------
        Index of value

        Raises
        ------
        ValueError if value is not found
        """
        _, last = self._find_first_and_last(value)
        return last

    def append(self, other: ColumnBase) -> ColumnBase:
        return concat_columns([self, other])

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase:
        raise TypeError(f"cannot perform quantile with type {self.dtype}")

    def take(
        self,
        indices: ColumnBase,
        *,
        nullify: bool = False,
        check_bounds: bool = True,
    ) -> Self:
        """Return Column by taking values from the corresponding *indices*.

        Skip bounds checking if check_bounds is False.
        Set rows to null for all out of bound indices if nullify is `True`.
        """
        # Handle zero size
        if indices.size == 0:
            return cast("Self", column_empty(row_count=0, dtype=self.dtype))

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if indices.dtype.kind not in {"u", "i"}:
            indices = indices.astype(SIZE_TYPE_DTYPE)
        GatherMap(indices, len(self), nullify=not check_bounds or nullify)
        gathered = copying.gather([self], indices, nullify=nullify)[0]  # type: ignore[arg-type]
        return cast(
            "Self",
            ColumnBase.create(gathered, self.dtype),
        )

    def isin(self, values: Sequence | ColumnBase) -> ColumnBase:
        """Check whether values are contained in the Column.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test. Passing in a single string will
            raise a TypeError. Instead, turn a single string into a list
            of one element.

        Returns
        -------
        result: Column
            Column of booleans indicating if each element is in values.
        """
        lhs, rhs = self._process_values_for_isin(values)
        if lhs.dtype != rhs.dtype:
            if lhs.null_count and rhs.null_count:
                return lhs.isnull()
            else:
                return as_column(
                    False, length=len(self), dtype=np.dtype(np.bool_)
                )
        elif lhs.null_count == 0 and rhs.is_all_null:
            return as_column(False, length=len(self), dtype=np.dtype(np.bool_))

        result = rhs.contains(lhs)
        if lhs.null_count > 0:
            # If one of the needles is null, then the result contains
            # nulls, these nulls should be replaced by whether or not the
            # haystack contains a null.
            # TODO: this is unnecessary if we resolve
            # https://github.com/rapidsai/cudf/issues/14515 by
            # providing a mode in which cudf::contains does not mask
            # the result.
            result = result.fillna(rhs.null_count > 0)
        return result

    def _process_values_for_isin(
        self, values: Sequence | ColumnBase
    ) -> tuple[ColumnBase, ColumnBase]:
        """
        Helper function for `isin` which pre-process `values` based on `self`.
        """
        lhs = self
        rhs = as_column(values, nan_as_null=False)
        if lhs.is_all_null:
            lhs = lhs.astype(rhs.dtype)
        elif rhs.is_all_null:
            rhs = rhs.astype(lhs.dtype)
        return lhs, rhs

    def as_mask(self) -> tuple[Buffer, int]:
        """Convert booleans to bitmask

        Returns
        -------
        tuple[Buffer, int]
            The mask buffer and the null count (number of False values).
        """
        if self.has_nulls():
            raise ValueError("Column must have no nulls.")

        with self.access(mode="read", scope="internal"):
            mask, null_count = plc.transform.bools_to_mask(self.plc_column)
            return as_buffer(mask), null_count

    @property
    def is_unique(self) -> bool:
        return self.distinct_count(dropna=False) == len(self)

    @cached_property
    def is_monotonic_increasing(self) -> bool:
        return not self.has_nulls(include_nan=True) and sorting.is_sorted(
            [self], [True], ["first"]
        )

    @cached_property
    def is_monotonic_decreasing(self) -> bool:
        return not self.has_nulls(include_nan=True) and sorting.is_sorted(
            [self], [False], ["first"]
        )

    def contains(self, other: ColumnBase) -> ColumnBase:
        """
        Check whether column contains multiple values.

        Parameters
        ----------
        other : Column
            A column of values to search for
        """
        return PylibcudfFunction(
            plc.search.contains,
            pylibcudf_result_dtype_policy,
        ).execute_with_args(self, other)

    def sort_values(
        self: Self,
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> Self:
        if (not ascending and self.is_monotonic_decreasing) or (
            ascending and self.is_monotonic_increasing
        ):
            return self.copy()
        order = sorting.ordering([ascending], [na_position])
        return cast(
            "Self",
            PylibcudfFunction(
                plc.sorting.sort,
                same_dtype_policy,
                result_index=0,
            ).execute_with_args(ColumnList(self), order[0], order[1]),
        )

    def distinct_count(self, dropna: bool = True) -> int:
        """Get the (null-aware) number of distinct values in this column."""
        try:
            return self._distinct_count[dropna]
        except KeyError:
            with self.access(mode="read", scope="internal"):
                result = plc.reduce.distinct_count(
                    self.plc_column,
                    plc.types.NullPolicy.EXCLUDE
                    if dropna
                    else plc.types.NullPolicy.INCLUDE,
                    plc.types.NanPolicy.NAN_IS_NULL
                    if dropna
                    else plc.types.NanPolicy.NAN_IS_VALID,
                )
            self._distinct_count[dropna] = result
            return result

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        raise NotImplementedError()

    def cast(self, dtype: DtypeObj) -> ColumnBase:
        cast_dtype = dtype_to_pylibcudf_type(dtype)

        return PylibcudfFunction(
            plc.unary.cast,
            fixed_dtype_policy(dtype),
        ).execute_with_args(self, cast_dtype)

    def astype(self, dtype: DtypeObj, copy: bool | None = False) -> ColumnBase:
        if self.dtype == dtype:
            result = self
        elif isinstance(dtype, CategoricalDtype):
            result = self.as_categorical_column(dtype)
        elif len(self) == 0:
            if isinstance(dtype, np.dtype) and dtype.kind == "U":
                dtype = np.dtype("object")
            result = (
                self if self.dtype == dtype else column_empty(0, dtype=dtype)
            )
        else:
            if is_dtype_obj_interval(dtype):
                result = self.as_interval_column(dtype)  # type: ignore[arg-type]
            elif is_dtype_obj_list(dtype) or is_dtype_obj_struct(dtype):
                if isinstance(dtype, pd.ArrowDtype):
                    equiv_dtype = pyarrow_dtype_to_cudf_dtype(dtype)
                else:
                    equiv_dtype = dtype
                if self.dtype != equiv_dtype:
                    raise NotImplementedError(
                        f"Casting {self.dtype} columns not currently supported"
                    )
                elif self.dtype != dtype:
                    result = ColumnBase.create(self.plc_column, dtype)
                else:
                    result = self
            elif is_dtype_obj_decimal(dtype):
                result = self.as_decimal_column(dtype)  # type: ignore[arg-type]
            elif dtype.kind == "M":
                result = self.as_datetime_column(dtype)
            elif dtype.kind == "m":
                result = self.as_timedelta_column(dtype)
            elif dtype.kind in {"O", "U"}:
                if (
                    cudf.get_option("mode.pandas_compatible")
                    and isinstance(dtype, pd.ArrowDtype)
                    and not cudf.api.types.is_string_dtype(dtype)
                ):
                    raise TypeError(f"Unsupported dtype for astype: {dtype}")
                result = self.as_string_column(dtype)
            else:
                result = self.as_numerical_column(dtype)

        if copy and result is self:
            return result.copy(deep=copy)
        return result

    def as_categorical_column(
        self, dtype: CategoricalDtype
    ) -> CategoricalColumn:
        na_sentinel = -1
        if dtype._categories is not None:
            # Re-label self w.r.t. the provided categories
            codes = self._label_encoding(cats=dtype._categories)
        else:
            # Compute categories from self
            cats = self.unique().sort_values()
            # Only dropna if cats actually has nulls (self having nulls doesn't mean cats does)
            if cats.has_nulls():
                cats = cats.dropna()
            codes = self._label_encoding(cats=cats)
            dtype = CategoricalDtype(categories=cats, ordered=dtype.ordered)
        codes_with_mask = codes.set_mask(self.mask, self.null_count)
        codes_with_mask = codes_with_mask.copy_if_else(
            plc.Scalar.from_py(None, dtype_to_pylibcudf_type(codes.dtype)),
            codes_with_mask != na_sentinel,
        )
        codes_typed = cast(
            "cudf.core.column.numerical.NumericalColumn", codes_with_mask
        ).astype(dtype._codes_dtype)
        return ColumnBase.create(codes_typed.plc_column, dtype)  # type: ignore[return-value]

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        raise NotImplementedError()

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        raise NotImplementedError()

    def as_interval_column(self, dtype: IntervalDtype) -> IntervalColumn:
        raise NotImplementedError()

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        raise NotImplementedError()

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        raise NotImplementedError()

    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalColumn:
        raise NotImplementedError()

    def apply_boolean_mask(self, mask: ColumnBase) -> ColumnBase:
        if mask.dtype.kind != "b":
            raise ValueError("boolean_mask is not boolean type.")

        return PylibcudfFunction(
            plc.stream_compaction.apply_boolean_mask,
            same_dtype_policy,
            result_index=0,
        ).execute_with_args(ColumnList(self), mask)

    def argsort(
        self,
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> NumericalColumn:
        if (ascending and self.is_monotonic_increasing) or (
            not ascending and self.is_monotonic_decreasing
        ):
            return cast(
                "cudf.core.column.NumericalColumn", as_column(range(len(self)))
            )
        elif (ascending and self.is_monotonic_decreasing) or (
            not ascending and self.is_monotonic_increasing
        ):
            return cast(
                "cudf.core.column.NumericalColumn",
                ColumnBase.from_range(range(len(self) - 1, -1, -1)),
            )
        else:
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    sorting.order_by(
                        [self], [ascending], [na_position], stable=True
                    ),
                    SIZE_TYPE_DTYPE,
                ),
            )

    def __arrow_array__(self, type: pa.DataType | None = None) -> None:
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        # pandas produces non-writeable numpy arrays when CoW is enabled, which cupy
        # does not support. Our options are either to copy the data preemptively to
        # avoid handing out a pointer that allows modification of the data, or to allow
        # such modification even though it doesn't match pandas default behavior. Since
        # numpy arrays can be trivially made writeable by just changing their flag,
        # allowing modification here seems like the better option since otherwise we
        # would forbid modification altogether. Moreover, preemptive copying would
        # increase memory pressure unnecessarily.
        data_buf = self.data
        if data_buf is None:
            raise ValueError(
                "__cuda_array_interface__ not supported for columns with no data buffer"
            )
        self._exposed_buffers.add(data_buf)
        with data_buf.access(mode="read", scope="external"):
            output = {
                "shape": (len(self),),
                "strides": (self.dtype.itemsize,),
                "typestr": self.dtype.numpy_dtype.str
                if not isinstance(self.dtype, np.dtype)
                else self.dtype.str,
                "data": (
                    data_buf.ptr + self.offset * self.dtype.itemsize,
                    False,
                ),
                "version": 3,
            }
        if self.nullable and self.has_nulls():
            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            mask = self.mask
            assert mask is not None
            with mask.access(mode="read", scope="external"):
                output["mask"] = mask
                self._exposed_buffers.add(mask)
        return output

    def __array_ufunc__(
        self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any
    ) -> ColumnBase:
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    def __invert__(self) -> ColumnBase:
        raise TypeError(
            f"Operation `~` not supported on {self.dtype.type.__name__}"
        )

    def searchsorted(
        self,
        value: ColumnBase,
        side: Literal["left", "right"] = "left",
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ) -> Self:
        if not isinstance(value, ColumnBase) or value.dtype != self.dtype:
            raise ValueError(
                "Column searchsorted expects values to be column of same dtype"
            )
        if is_pandas_nullable_extension_dtype(self.dtype) and self.has_nulls(
            include_nan=True
        ):
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        return cast(
            "Self",
            ColumnBase.create(
                sorting.search_sorted(
                    [self],
                    [value],
                    side=side,
                    ascending=[ascending],
                    na_position=[na_position],
                ),
                SIZE_TYPE_DTYPE,
            ),
        )

    def unique(self) -> Self:
        """
        Get unique values in the data
        """
        if self.is_unique:
            return self.copy()
        else:
            return cast(
                "Self",
                PylibcudfFunction(
                    plc.stream_compaction.stable_distinct,
                    same_dtype_policy,
                    result_index=0,
                ).execute_with_args(
                    ColumnList(self),
                    [0],
                    plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                    plc.types.NullEquality.EQUAL,
                    plc.types.NanEquality.ALL_EQUAL,
                ),
            )

    def factorize(
        self, sort: bool, use_na_sentinel: bool
    ) -> tuple["cp.ndarray", Self]:
        """
        Column-level factorize. Returns ``(labels, cats_col)``.

        ``cats_col`` is the column of unique values; ``labels`` are the
        codes mapping each element of ``self`` to its position in
        ``cats_col`` (with ``-1`` for nulls when ``use_na_sentinel=True``).
        """
        cats = self.dropna() if use_na_sentinel else self
        cats = cast("Self", cats.unique().astype(self.dtype))
        if sort:
            cats = cast("Self", cats.sort_values())
        labels = self._label_encoding(
            cats=cats,
            dtype=np.dtype("int64"),
        ).values
        return labels, cats

    def _is_sliced(self) -> bool:
        """
        Check if this column is a sliced view of a larger buffer.

        A column is considered sliced if:
        - It has a non-zero offset, OR
        - Its logical size is less than the base buffer size

        Returns
        -------
        bool
            True if the column is sliced, False otherwise
        """
        if self.offset != 0:
            return True

        # Check if size is less than base buffer size
        # Note: Some dtypes (like CategoricalDtype) don't have itemsize
        if self.data is not None and hasattr(self.dtype, "itemsize"):
            base_size_elements = self.data.size // self.dtype.itemsize
            if self.size < base_size_elements:
                return True

        return False

    def _compact_for_transfer(self) -> ColumnBase:
        """
        Create a compact copy of this column for network transfer.

        If the column is sliced (offset != 0 or size < base buffer size),
        this creates a copy that contains only the data that's actually
        needed, not the entire base buffer. This significantly reduces
        network transfer overhead when serializing sliced columns.

        Returns
        -------
        ColumnBase
            A compacted column (or self if not sliced)
        """
        if self._is_sliced():
            # Check if data is spilled - if so, don't compact it
            # Spilled data is already on CPU and calling copy() would
            # unnecessarily bring it back to GPU and lose the spill state
            if self.data is not None and hasattr(self.data, "owner"):
                if (
                    hasattr(self.data.owner, "is_spilled")
                    and self.data.owner.is_spilled
                ):
                    return self

            # Create a copy to compact the data - only the actual slice
            # will be copied, not the entire base buffer
            return self.copy()
        return self

    @staticmethod
    def _serialize_plc_column_recursive(
        plc_col: plc.Column,
    ) -> tuple[dict, list]:
        """Recursively serialize a plc.Column and all its children."""
        header: dict[Any, Any] = {}
        frames = []

        if (plc_data := plc_col.data()) is not None:
            header["data"], data_frames = cast(
                "Buffer", plc_data
            ).device_serialize()
            frames.extend(data_frames)

        if (plc_mask := plc_col.null_mask()) is not None:
            header["mask"], mask_frames = cast(
                "Buffer", plc_mask
            ).device_serialize()
            frames.extend(mask_frames)

        header["plc_type"] = pickle.dumps(plc_col.type())
        header["size"] = plc_col.size()
        header["offset"] = plc_col.offset()
        header["null_count"] = plc_col.null_count()

        # Recursively serialize children
        if plc_children := plc_col.children():
            child_subheaders = []
            for plc_child in plc_children:
                child_header, child_frames = (
                    ColumnBase._serialize_plc_column_recursive(plc_child)
                )
                child_subheaders.append(child_header)
                frames.extend(child_frames)
            header["subheaders"] = child_subheaders

        header["frame_count"] = len(frames)
        return header, frames

    @staticmethod
    def _unpack_buffer(header: dict, frames: list) -> tuple[Any, list]:
        """Unpack a serialized buffer from frames, returning (object, remaining_frames)."""
        count = header["frame_count"]
        return Serializable.device_deserialize(header, frames[:count]), frames[
            count:
        ]

    @classmethod
    def _deserialize_plc_column_recursive(
        cls, header: dict, frames: list
    ) -> tuple[plc.Column, list]:
        """Recursively deserialize a plc.Column and all its children."""
        # Deserialize data and mask buffers
        data = None
        if "data" in header:
            data, frames = cls._unpack_buffer(header["data"], frames)

        mask = None
        if "mask" in header:
            mask, frames = cls._unpack_buffer(header["mask"], frames)

        children = []
        for child_header in header.get("subheaders", []):
            child_plc_col, frames = cls._deserialize_plc_column_recursive(
                child_header, frames
            )
            children.append(child_plc_col)

        try:
            plc_type = pickle.loads(header["plc_type"])
        except (pickle.UnpicklingError, KeyError, TypeError, EOFError) as e:
            raise ValueError(f"Failed to deserialize plc_type: {e}") from e

        return plc.Column(
            plc_type,
            header["size"],
            data,
            mask,
            header["null_count"],
            header.get("offset", 0),
            children,
            validate=False,
        ), frames

    def serialize(self) -> tuple[dict, list]:
        """Serialize column to header dict and frames list.

        Produces nested metadata header and flattened list of buffers (frames).
        Each header has frame_count indicating frames consumed during deserialization.
        """
        # Compact sliced columns before serialization to avoid transferring
        # entire base buffers over the network. This is critical for
        # performance when working with distributed systems like Dask.
        # Only affects sliced columns (offset != 0 or size < base buffer size).
        col_to_serialize = self._compact_for_transfer()

        header: dict[Any, Any] = {}
        frames = []

        # Serialize dtype
        try:
            header["dtype"], dtype_frames = (
                col_to_serialize.dtype.device_serialize()  # type: ignore[union-attr]
            )
            frames.extend(dtype_frames)
            header["dtype-is-cudf-serialized"] = True
        except AttributeError:
            header["dtype"] = (
                col_to_serialize.dtype.str
                if isinstance(col_to_serialize.dtype, np.dtype)
                else pickle.dumps(col_to_serialize.dtype)
            )
            header["dtype-is-cudf-serialized"] = False

        # Serialize entire plc_column (data, mask, children)
        plc_header, plc_frames = self._serialize_plc_column_recursive(
            col_to_serialize.plc_column
        )
        frames.extend(plc_frames)
        header["plc_column"] = plc_header

        header["size"] = col_to_serialize.size
        header["frame_count"] = len(frames)
        header["offset"] = col_to_serialize.offset
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> ColumnBase:
        assert header["frame_count"] == len(frames), (
            f"Expected {header['frame_count']} frames, got {len(frames)}"
        )

        # Deserialize dtype
        if header["dtype-is-cudf-serialized"]:
            dtype, frames = cls._unpack_buffer(header["dtype"], frames)
        else:
            try:
                dtype = np.dtype(header["dtype"])
            except TypeError:
                dtype = pickle.loads(header["dtype"])

        # Deserialize plc_column recursively
        plc_column, frames = cls._deserialize_plc_column_recursive(
            header["plc_column"], frames
        )
        assert len(frames) == 0, (
            f"{len(frames)} frame(s) remaining after deserialization"
        )
        return ColumnBase.create(plc_column, dtype)

    def unary_operator(self, unaryop: str) -> ColumnBase:
        raise TypeError(
            f"Operation {unaryop} not supported for dtype {self.dtype}."
        )

    def nans_to_nulls(self: Self) -> Self:
        """Convert NaN to NA."""
        return self

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls(include_nan=False)

    def _reduction_result_dtype(self, op: str) -> DtypeObj:
        """
        Determine the correct dtype to pass to libcudf based on
        the input dtype, data dtype, and specific reduction op
        """
        if op in {"any", "all"}:
            return np.dtype(np.bool_)
        return self.dtype

    def _label_encoding(
        self,
        cats: ColumnBase,
        dtype: DtypeObj | None = None,
    ) -> NumericalColumn:
        """
        Convert each value in `self` into an integer code, with `cats`
        providing the mapping between codes and values.

        Examples
        --------
        >>> from cudf.core.column import as_column
        >>> col = as_column(['foo', 'bar', 'foo', 'baz'])
        >>> cats = as_column(['foo', 'bar', 'baz'])
        >>> col._label_encoding(cats)
        <cudf.core.column.numerical.NumericalColumn object at 0x7f99bf3155c0>
        [
          0,
          1,
          0,
          2
        ]
        dtype: int8
        >>> cats = as_column(['foo', 'bar'])
        >>> col._label_encoding(cats)
        <cudf.core.column.numerical.NumericalColumn object at 0x7f99bfde0e40>
        [
          0,
          1,
          0,
          -1
        ]
        dtype: int8
        """
        na_sentinel = -1

        def _return_sentinel_column(dtype: DtypeObj) -> NumericalColumn:
            # TODO: Validate that dtype is numeric type
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(
                            na_sentinel, dtype=dtype_to_pylibcudf_type(dtype)
                        ),
                        len(self),
                    ),
                    dtype,
                ),
            )

        if dtype is None:
            dtype = min_signed_type(max(len(cats), na_sentinel), 8)

        if is_mixed_with_object_dtype(self, cats):
            return _return_sentinel_column(dtype)

        try:
            # Where there is a type-cast failure, we have
            # to catch the exception and return encoded labels
            # with na_sentinel values as there would be no corresponding
            # encoded values of cats in self.
            cats = cats.astype(self.dtype)
        except ValueError:
            return _return_sentinel_column(dtype)

        left_rows, right_rows = plc.join.left_join(
            plc.Table([self.plc_column]),
            plc.Table([cats.plc_column]),
            plc.types.NullEquality.EQUAL,
        )
        left_gather_map = ColumnBase.create(left_rows, SIZE_TYPE_DTYPE)
        right_gather_map = ColumnBase.create(right_rows, SIZE_TYPE_DTYPE)

        codes = (
            ColumnBase.from_range(range(len(cats)))
            .astype(dtype)
            .take(right_gather_map, nullify=True)
        )
        del right_gather_map
        del right_rows
        # reorder `codes` so that its values correspond to the
        # values of `self`:
        plc_codes = sorting.sort_by_key(
            [codes], [left_gather_map], [True], ["last"], stable=True
        )[0]
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            ColumnBase.create(
                plc_codes, dtype_from_pylibcudf_column(plc_codes)
            ).fillna(na_sentinel),
        )

    def copy_if_else(
        self, other: Self | plc.Scalar, boolean_mask: NumericalColumn
    ) -> Self:
        with access_columns(
            self, other, boolean_mask, mode="read", scope="internal"
        ) as (self, other, boolean_mask):
            return cast(
                "Self",
                ColumnBase.create(
                    plc.copying.copy_if_else(
                        self.plc_column,
                        other
                        if isinstance(other, plc.Scalar)
                        else other.plc_column,
                        boolean_mask.plc_column,
                    ),
                    self.dtype,
                ),
            )

    def split_by_offsets(
        self, offsets: list[int]
    ) -> Generator[Self, None, None]:
        for cols in copying.columns_split([self], offsets):
            for col in cols:
                yield cast(
                    "Self",
                    ColumnBase.create(col, self.dtype),
                )

    def one_hot_encode(self, categories: ColumnBase) -> Generator[ColumnBase]:
        with access_columns(
            self, categories, mode="read", scope="internal"
        ) as (self, categories):
            plc_table = plc.transform.one_hot_encode(
                self.plc_column,
                categories.plc_column,
            )
            return (
                ColumnBase.create(col, np.dtype(np.bool_))
                for col in plc_table.columns()
            )

    def _scan(self, op: str, inclusive: bool = True, **kwargs: Any) -> Self:
        """Private method for scan operations. Called by mixin-generated methods."""
        # `inclusive` controls scan type, not passed to aggregation
        scan_type = (
            plc.reduce.ScanType.INCLUSIVE
            if inclusive
            else plc.reduce.ScanType.EXCLUSIVE
        )
        agg = aggregation.make_aggregation(op, kwargs).plc_obj

        return cast(
            "Self",
            PylibcudfFunction(
                plc.reduce.scan, same_dtype_policy
            ).execute_with_args(self, agg, scan_type),
        )

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        """Private method for reduction operations. Called by mixin-generated methods."""
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )
        # Early return if we can return NaN
        if self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)

        # Handle skipna by converting nans to nulls and potentially dropping
        col = self.nans_to_nulls() if skipna else self
        if col.has_nulls():
            if skipna:
                col = col.dropna()
            else:
                return _get_nan_for_dtype(self.dtype)

        # Handle min_count
        if min_count > 0:
            if col.valid_count < min_count:
                return _get_nan_for_dtype(self.dtype)

        # Compute reduction result dtype
        col_dtype = col._reduction_result_dtype(op)

        # Handle empty case - if column is all null
        if col.is_all_null:
            if op == "sum" or op == "sum_of_squares":
                return col_dtype.type(0)
            if op == "product":
                return col_dtype.type(1)
            if is_pandas_nullable_extension_dtype(self.dtype):
                # pandas returns <NA> for empty/all-null reductions of
                # nullable dtypes even when the reduction result dtype is
                # a plain numpy dtype (e.g. Int64.mean() -> float64).
                return _get_nan_for_dtype(self.dtype)
            return _get_nan_for_dtype(col_dtype)

        # Perform the actual reduction
        with col.access(mode="read", scope="internal"):
            plc_scalar = plc.reduce.reduce(
                col.plc_column,
                aggregation.make_aggregation(op, kwargs).plc_obj,
                dtype_to_pylibcudf_type(col_dtype),
            )
            # Hook for subclasses (e.g., DecimalColumn adjusts precision)
            col_dtype = col._adjust_reduce_result_dtype(
                op, col_dtype, plc_scalar
            )
            result_col = ColumnBase.create(
                plc.Column.from_scalar(plc_scalar, 1), col_dtype
            )
        return result_col.element_indexing(0)

    def _adjust_reduce_result_dtype(
        self,
        op: str,
        col_dtype: DtypeObj,
        plc_scalar: plc.Scalar,
    ) -> DtypeObj:
        """Hook for subclasses to adjust reduction result."""
        return col_dtype

    def minmax(self) -> tuple[ScalarLike, ScalarLike]:
        with self.access(mode="read", scope="internal"):
            min_val, max_val = plc.reduce.minmax(self.plc_column)
            return (
                type(self)
                .create(plc.Column.from_scalar(min_val, 1), self.dtype)
                .element_indexing(0),
                type(self)
                .create(plc.Column.from_scalar(max_val, 1), self.dtype)
                .element_indexing(0),
            )

    def rank(
        self,
        *,
        method: plc.aggregation.RankMethod,
        column_order: plc.types.Order,
        null_handling: plc.types.NullPolicy,
        null_precedence: plc.types.NullOrder,
        pct: bool,
    ) -> Self:
        result = cast(
            "Self",
            PylibcudfFunction(
                plc.sorting.rank,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(
                self, method, column_order, null_handling, null_precedence, pct
            ),
        )
        return result.astype(_rank_out_dtype(self.dtype, method, pct))  # type: ignore[return-value]

    def label_bins(
        self,
        *,
        left_edge: Self,
        left_inclusive: bool,
        right_edge: Self,
        right_inclusive: bool,
    ) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc.labeling.label_bins(
                        self.plc_column,
                        left_edge.plc_column,
                        plc.labeling.Inclusive.YES
                        if left_inclusive
                        else plc.labeling.Inclusive.NO,
                        right_edge.plc_column,
                        plc.labeling.Inclusive.YES
                        if right_inclusive
                        else plc.labeling.Inclusive.NO,
                    ),
                    SIZE_TYPE_DTYPE,
                ),
            )

    def _cast_self_and_other_for_where(
        self, other: ScalarLike | ColumnBase, inplace: bool
    ) -> tuple[ColumnBase, plc.Scalar | ColumnBase]:
        other_is_scalar = is_scalar(other)

        if other_is_scalar:
            if isinstance(other, (float, np.floating)) and not np.isnan(other):
                try:
                    is_safe = self.dtype.type(other) == other
                except OverflowError:
                    is_safe = False

                if not is_safe:
                    raise TypeError(
                        f"Cannot safely cast non-equivalent "
                        f"{type(other).__name__} to {self.dtype}"
                    )

            if is_na_like(other):
                is_nat = other is pd.NaT or (
                    isinstance(other, (np.datetime64, np.timedelta64))
                    and np.isnat(other)
                )
                if (
                    is_nat
                    and is_pandas_nullable_extension_dtype(self.dtype)
                    and self.dtype.kind not in {"m", "M"}
                ):
                    # pandas only accepts NaT as a missing value for
                    # datetime/timedelta dtypes, not for masked dtypes
                    raise TypeError(
                        f"Invalid value '{other}' for dtype '{self.dtype}'"
                    )
                if (
                    cudf.get_option("mode.pandas_compatible")
                    and not is_pandas_nullable_extension_dtype(self.dtype)
                    and self.dtype.kind not in {"i", "f", "u"}
                ):
                    raise MixedTypeError(
                        "Cannot use None or np.nan with non-Pandas nullable dtypes."
                    )
                return self, pa_scalar_to_plc_scalar(
                    pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
                )

        mixed_err = (
            "cudf does not support mixed types, please type-cast the column of "
            "dataframe/series and other to same dtypes."
        )

        if inplace:
            other_col = as_column(other)
            if is_mixed_with_object_dtype(other_col, self):
                raise TypeError(mixed_err)

            if other_col.dtype != self.dtype:
                if cudf.get_option("mode.pandas_compatible") and (
                    (
                        self.dtype.kind in {"i", "u"}
                        and other_col.dtype.kind == "f"
                        and not other_col.can_cast_safely(self.dtype)
                    )
                    or (
                        self.dtype.kind in {"M", "m"}
                        and other_col.dtype.kind in {"i", "u", "f", "b"}
                    )
                ):
                    raise TypeError(
                        f"Invalid value '{other}' for dtype '{self.dtype}'"
                    )
                try:
                    warn = is_dtype_obj_string(
                        find_common_type((other_col.dtype, self.dtype)),
                    )
                except NotImplementedError:
                    warn = True
                if warn:
                    warnings.warn(
                        f"Type-casting from {other_col.dtype} "
                        f"to {self.dtype}, there could be potential data loss"
                    )
            other_out: plc.Scalar | ColumnBase
            if other_is_scalar:
                other_out = pa_scalar_to_plc_scalar(
                    pa.scalar(other, type=cudf_dtype_to_pa_type(self.dtype))
                )
            else:
                other_out = other_col.astype(self.dtype)
            return self, other_out

        if is_dtype_obj_numeric(
            self.dtype, include_decimal=False
        ) and as_column(other).can_cast_safely(self.dtype):
            common_dtype = self.dtype
        else:
            common_dtype = find_common_type(
                [
                    self.dtype,
                    # can_cast_safely should imply this is safe
                    np.min_scalar_type(other)  # type: ignore[arg-type]
                    if other_is_scalar
                    else other.dtype,
                ]
            )
        other_col = as_column(other)
        if (
            is_mixed_with_object_dtype(other_col, self)
            or (self.dtype.kind == "b" and common_dtype.kind != "b")
            or (other_col.dtype.kind == "b" and self.dtype.kind != "b")
        ):
            raise TypeError(mixed_err)

        if other_is_scalar:
            other_out = pa_scalar_to_plc_scalar(
                pa.scalar(other, type=cudf_dtype_to_pa_type(common_dtype))
            )
        else:
            other_out = other.astype(common_dtype)

        return self.astype(common_dtype), other_out

    def where(
        self, cond: ColumnBase, other: ScalarLike | ColumnBase, inplace: bool
    ) -> ColumnBase:
        casted_col, casted_other = self._cast_self_and_other_for_where(
            other, inplace
        )
        result = casted_col.copy_if_else(casted_other, cond)  # type: ignore[arg-type]
        return ColumnBase.create(result.plc_column, casted_col.dtype)


def column_empty(
    row_count: int,
    dtype: DtypeObj = DEFAULT_STRING_DTYPE,
) -> ColumnBase:
    """
    Allocate a new column with the given row_count and dtype.

    * Passing row_count == 0 creates a size 0 column without a mask buffer.
    * Passing row_count > 0 creates an all null column with a mask buffer.

    Parameters
    ----------
    row_count : int
        Number of elements in the column.

    dtype : Dtype
        Type of the column.
    """
    if (
        is_struct := isinstance(dtype, (StructDtype, IntervalDtype))
    ) or isinstance(dtype, ListDtype):
        if is_struct:
            children = tuple(
                column_empty(row_count, field_dtype)
                for field_dtype in dtype.fields.values()
            )
        else:
            children = (
                as_column(0, length=row_count + 1, dtype=SIZE_TYPE_DTYPE),
                column_empty(row_count, dtype=dtype.element_type),
            )
        mask = (
            None
            if row_count == 0
            else plc.null_mask.create_null_mask(
                row_count, plc.types.MaskState.ALL_NULL
            )
        )
        plc_column = plc.Column(
            dtype_to_pylibcudf_type(dtype),
            row_count,
            None,
            mask,
            row_count,
            0,
            [child.plc_column for child in children],
        )
        return ColumnBase.create(plc_column, dtype)
    else:
        if isinstance(dtype, CategoricalDtype):
            # CategoricalDtype needs special handling - create codes column
            # with the correct dtype, then construct the categorical column.
            codes_dtype = dtype._codes_dtype
            plc_dtype = dtype_to_pylibcudf_type(codes_dtype)
            plc_column = plc.Column.from_scalar(
                plc.Scalar.from_py(None, plc_dtype),
                row_count,
            )
            return ColumnBase.create(plc_column, dtype)
        else:
            plc_dtype = dtype_to_pylibcudf_type(dtype)
            return ColumnBase.create(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(None, plc_dtype),
                    row_count,
                ),
                dtype,
            )


def check_invalid_array(shape: tuple, dtype: np.dtype) -> None:
    """Invalid ndarrays properties that are not supported"""
    if len(shape) > 1:
        raise ValueError("Data must be 1-dimensional")
    elif dtype == "float16":
        raise TypeError("Unsupported type float16")


def as_column(
    arbitrary: Any,
    nan_as_null: bool | None = None,
    dtype: DtypeObj | None = None,
    length: int | None = None,
) -> ColumnBase:
    """
    Create a Column from an arbitrary object.

    Consider using ColumnBase classmethod constructors if
    arbitrary is a known type.

    Parameters
    ----------
    arbitrary : object
        Object to construct the Column from. See *Notes*.
    nan_as_null : bool, optional, default None
        If None (default), treats NaN values in arbitrary as null if there is
        no mask passed along with it. If True, combines the mask and NaNs to
        form a new validity mask. If False, leaves NaN values as is.
        Only applies when arbitrary is not a cudf object
        (Index, Series, Column).
    dtype : DtypeObj, optional
        Optionally typecast the constructed Column to the given
        dtype.
    length : int, optional
        If `arbitrary` is a scalar, broadcast into a Column of
        the given length.

    Returns
    -------
    A Column of the appropriate type and size.

    Notes
    -----
    Currently support inputs are:

    * ``Column``
    * ``Series``
    * ``Index``
    * Scalars (can be broadcasted to a specified `length`)
    * Objects exposing ``__cuda_array_interface__`` (e.g., numba device arrays)
    * Objects exposing ``__array_interface__``(e.g., numpy arrays)
    * pyarrow array
    * pandas.Categorical objects
    * range objects
    """
    if not (
        dtype is None
        or isinstance(
            dtype,
            (
                np.dtype,
                CategoricalDtype,
                IntervalDtype,
                ListDtype,
                StructDtype,
                DecimalDtype,
            ),
        )
        or is_pandas_nullable_extension_dtype(dtype)
    ):
        raise ValueError(f"dtype must be None or a dtype object, got {dtype}")

    if isinstance(arbitrary, (range, pd.RangeIndex, cudf.RangeIndex)):
        if not isinstance(arbitrary, range):
            arbitrary = range(arbitrary.start, arbitrary.stop, arbitrary.step)
        column = ColumnBase.from_range(arbitrary)
        if cudf.get_option("default_integer_bitwidth") and dtype is None:
            dtype = np.dtype(
                f"i{cudf.get_option('default_integer_bitwidth') // 8}"
            )
        if dtype is not None:
            return column.astype(dtype)
        return column
    elif isinstance(arbitrary, (ColumnBase, cudf.Series, cudf.Index)):
        # Ignoring nan_as_null per the docstring
        if isinstance(arbitrary, cudf.Series):
            arbitrary = arbitrary._column
        elif isinstance(arbitrary, cudf.Index):
            arbitrary = arbitrary._column
        if dtype is not None:
            return arbitrary.astype(dtype)
        return arbitrary
    elif hasattr(arbitrary, "__cuda_array_interface__"):
        column = ColumnBase.from_cuda_array_interface(arbitrary)
        if nan_as_null is not False:
            column = column.nans_to_nulls()
        if dtype is not None:
            column = column.astype(dtype)
        return column
    elif isinstance(arbitrary, (pa.Array, pa.ChunkedArray)):
        if isinstance(arbitrary, pa.NullArray) and dtype is None:
            dtype = np.dtype("object")
        elif is_arrow_null_dtype(dtype):
            if arbitrary.null_count != len(arbitrary):
                raise ValueError(
                    f"dtype {dtype} can only be used with all-null data."
                )
            arbitrary = pa.nulls(len(arbitrary))
        column = ColumnBase.from_arrow(arbitrary)
        if nan_as_null is not False:
            column = column.nans_to_nulls()
        if dtype is not None:
            column = column.astype(dtype)
        return column
    elif isinstance(
        arbitrary, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)
    ):
        if isinstance(arbitrary.dtype, (pd.SparseDtype, pd.PeriodDtype)):
            raise NotImplementedError(
                f"cuDF does not yet support {type(arbitrary.dtype).__name__}"
            )
        elif (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(arbitrary, (pd.DatetimeIndex, pd.TimedeltaIndex))
            and arbitrary.freq is not None
        ):
            raise NotImplementedError("freq is not implemented yet")
        elif isinstance(arbitrary.dtype, pd.IntervalDtype) and isinstance(
            arbitrary.dtype.subtype, pd.DatetimeTZDtype
        ):
            raise NotImplementedError(
                "cuDF does not yet support Intervals with timezone-aware datetimes"
            )
        elif isinstance(
            arbitrary.dtype,
            (pd.CategoricalDtype, pd.IntervalDtype, pd.DatetimeTZDtype),
        ):
            if isinstance(arbitrary.dtype, pd.DatetimeTZDtype):
                new_tz = get_compatible_timezone(arbitrary.dtype)
                arbitrary = arbitrary.astype(new_tz)
            if isinstance(arbitrary.dtype, pd.CategoricalDtype):
                if isinstance(
                    arbitrary.dtype.categories.dtype, pd.DatetimeTZDtype
                ):
                    # pa.array(arbitrary, from_pandas=True) drops the tz from
                    # the dictionary value type; cast back to a tz-aware
                    # dictionary type to restore the categories' timezone.
                    # pyarrow bug: https://github.com/apache/arrow/issues/49875
                    new_tz = get_compatible_timezone(
                        arbitrary.dtype.categories.dtype
                    )
                    pa_arbitrary = pa.array(arbitrary, from_pandas=True)
                    arbitrary = pa_arbitrary.cast(
                        pa.dictionary(
                            pa_arbitrary.type.index_type,
                            pa.timestamp(new_tz.unit, str(new_tz.tz)),
                            ordered=pa_arbitrary.type.ordered,
                        )
                    )
                elif dtype is None:
                    # Going through Arrow below erases pandas extension dtypes
                    # of the categories, if any, so always along pass the exact type
                    dtype = CategoricalDtype(
                        categories=arbitrary.dtype.categories,
                        ordered=arbitrary.dtype.ordered,
                    )
            result = as_column(
                pa.array(arbitrary, from_pandas=True),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
            return result
        elif is_pandas_nullable_extension_dtype(arbitrary.dtype):
            if isinstance(arbitrary.dtype, pd.ArrowDtype) and pa.types.is_date(
                arbitrary.dtype.pyarrow_dtype
            ):
                # TODO: See if we can centralize this check with other
                # dtype conversion functions
                raise NotImplementedError(
                    f"cuDF does not yet support {arbitrary.dtype}"
                )
            if isinstance(arbitrary, (pd.Series, pd.Index)):
                # pandas arrays define __arrow_array__ for better
                # pyarrow.array conversion
                arbitrary = arbitrary.array
            if dtype is None:
                dtype = arbitrary.dtype
            return as_column(
                pa.array(arbitrary, from_pandas=True),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
        elif isinstance(
            arbitrary.dtype, pd.api.extensions.ExtensionDtype
        ) and not isinstance(arbitrary, pd.arrays.NumpyExtensionArray):
            raise NotImplementedError(
                "Custom pandas ExtensionDtypes are not supported"
            )
        elif arbitrary.dtype.kind in "fiubmM":
            # numpy dtype like
            if isinstance(arbitrary, pd.arrays.NumpyExtensionArray):
                arbitrary = np.array(arbitrary)
            arb_dtype = np.dtype(arbitrary.dtype)
            if arb_dtype.kind == "f" and arb_dtype.itemsize == 2:
                raise TypeError("Unsupported type float16")
            elif arb_dtype.kind in "mM":
                # not supported by cupy
                arbitrary = np.asarray(arbitrary)
            else:
                arbitrary = cp.asarray(arbitrary)
                column = ColumnBase.from_cuda_array_interface(arbitrary)
                if nan_as_null is not False:
                    column = column.nans_to_nulls()
                if dtype is not None:
                    column = column.astype(dtype)
                return column
            return as_column(
                arbitrary, nan_as_null=nan_as_null, dtype=dtype, length=length
            )
        elif arbitrary.dtype.kind == "O":
            pyarrow_array = None
            if isinstance(arbitrary, pd.arrays.NumpyExtensionArray):
                # infer_dtype does not handle NumpyExtensionArray
                arbitrary = np.array(arbitrary, dtype=object)
            inferred_dtype = infer_dtype(
                arbitrary,
                skipna=(
                    not cudf.get_option("mode.pandas_compatible")
                    and nan_as_null is not False
                ),
            )
            if inferred_dtype in ("mixed-integer", "mixed-integer-float"):
                raise MixedTypeError("Cannot create column with mixed types")
            elif dtype is None and inferred_dtype not in (
                "mixed",
                "decimal",
                "string",
                "empty",
                "boolean",
            ):
                raise MixedTypeError(
                    f"Cannot convert a {inferred_dtype} of object type"
                )
            elif inferred_dtype == "boolean":
                if cudf.get_option("mode.pandas_compatible"):
                    if dtype.kind != "b" or pd.isna(arbitrary).any():  # type: ignore[union-attr]  # (dtype is required for boolean compatibility)
                        raise MixedTypeError(
                            f"Cannot have mixed values with {inferred_dtype}"
                        )
                elif nan_as_null is False:
                    raise MixedTypeError(
                        f"Cannot have a {inferred_dtype} type with dtype=object"
                    )
            elif nan_as_null is False and inferred_dtype not in (
                "decimal",
                "empty",
                "string",
            ):
                if inferred_dtype == "floating":
                    raise MixedTypeError(
                        f"Cannot have a {inferred_dtype} type with dtype=object"
                    )
                try:
                    pyarrow_array = pa.array(arbitrary, from_pandas=False)
                except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError):
                    # Decimal can hold float("nan")
                    # All np.nan is not restricted by type
                    raise MixedTypeError(
                        f"Cannot have NaN with {inferred_dtype}"
                    )

            if pyarrow_array is None:
                pyarrow_array = pa.array(
                    arbitrary,
                    from_pandas=True,
                )
            if (
                cudf.get_option("mode.pandas_compatible")
                and inferred_dtype == "mixed"
                and not (
                    pa.types.is_list(pyarrow_array.type)
                    or pa.types.is_struct(pyarrow_array.type)
                    or pa.types.is_string(pyarrow_array.type)
                    or pa.types.is_null(pyarrow_array.type)
                )
            ):
                raise MixedTypeError("Cannot create column with mixed types")

            if dtype is None and (
                inferred_dtype in {"string", "empty"}
                or (
                    inferred_dtype == "mixed"
                    and pa.types.is_string(pyarrow_array.type)
                    and isinstance(arbitrary, (pd.Series, pd.Index))
                )
            ):
                # The Series/Index restriction in (2) is load-bearing. Consider:
                #
                #   pd.Series(["a", None], dtype=object)
                #     -> user explicitly chose object; preserve it
                #        (test_grouping_grouper, test_to_numpy depend on this).
                #
                #   pd.arrays.NumpyExtensionArray(np.array(["a", None], dtype=object))
                #     -> by this point ``arbitrary`` has been unwrapped to a raw
                #        np.ndarray whose ``dtype=object`` is incidental (numpy's
                #        default for arrays containing None), not user intent.
                #        We want it to fall through to string inference, matching
                #        pandas (test_from_numpyextensionarray_string_object_pandas_compat_mode).
                dtype = arbitrary.dtype
            return as_column(
                pyarrow_array,
                dtype=dtype,
                nan_as_null=nan_as_null,
                length=length,
            )
        else:
            raise NotImplementedError(
                f"{type(arbitrary).__name__} with "
                f"{type(arbitrary.dtype).__name__} is not supported."
            )
    elif is_scalar(arbitrary) and not isinstance(arbitrary, memoryview):
        if length is None:
            length = 1
        elif length < 0:
            raise ValueError(f"{length=} must be >=0.")

        if is_arrow_null_dtype(dtype):
            if is_na_like(arbitrary):
                return column_empty(length, dtype=dtype)
            pa.scalar(arbitrary, type=dtype.pyarrow_dtype)  # type: ignore[union-attr]  # (arrow null dtype has pyarrow_dtype)

        pa_type = None
        if isinstance(arbitrary, pd.Interval) or _is_categorical_dtype(dtype):
            return as_column(
                pd.Series([arbitrary] * length),
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=length,
            )
        elif (
            nan_as_null is True
            and isinstance(arbitrary, (np.floating, float))
            and np.isnan(arbitrary)
        ):
            if dtype is None:
                dtype = getattr(arbitrary, "dtype", np.dtype(np.float64))
            arbitrary = None
            pa_type = cudf_dtype_to_pa_type(dtype)
            dtype = None
        elif arbitrary is pd.NA or arbitrary is None:
            arbitrary = None
            if dtype is not None:
                pa_type = cudf_dtype_to_pa_type(dtype)
                if not cudf.get_option("mode.pandas_compatible"):
                    dtype = None
            else:
                raise ValueError(
                    "Need to pass dtype when passing pd.NA or None"
                )
        elif arbitrary is pd.NaT:
            # NaT.to_numpy() defaults to ns resolution, but pandas 3.0
            # defaults NaT to seconds resolution. Use the dtype's kind
            # if provided to distinguish datetime vs timedelta.
            if dtype is not None and dtype.kind == "m":
                arbitrary = arbitrary.to_numpy("timedelta64[s]")
            else:
                arbitrary = arbitrary.to_numpy("datetime64[s]")
        elif isinstance(arbitrary, (pd.Timestamp, pd.Timedelta)):
            arbitrary = arbitrary.to_numpy()
        elif isinstance(arbitrary, (np.datetime64, np.timedelta64)):
            unit = np.datetime_data(arbitrary.dtype)[0]
            if unit not in {"s", "ms", "us", "ns"}:
                arbitrary = arbitrary.astype(
                    np.dtype(f"{arbitrary.dtype.kind}8[s]")
                )

        pa_scalar = pa.scalar(arbitrary, type=pa_type)  # type: ignore[type-var]  # (pyarrow accepts normalized scalar inputs)
        if length == 0:
            if dtype is None:
                dtype = cudf_dtype_from_pa_type(pa_scalar.type)  # type: ignore[arg-type]  # (pyarrow scalar exposes a DataType)
            return column_empty(length, dtype=dtype)
        else:
            plc_col = plc.Column.from_scalar(
                pa_scalar_to_plc_scalar(pa_scalar), length
            )
            col = ColumnBase.create(
                plc_col,
                cudf_dtype_from_pa_type(pa_scalar.type),  # type: ignore[arg-type]  # (pyarrow scalar exposes a DataType)
            )
            if dtype is not None:
                col = col.astype(dtype)
            return col
    elif hasattr(arbitrary, "__array_interface__"):
        desc = arbitrary.__array_interface__
        check_invalid_array(desc["shape"], np.dtype(desc["typestr"]))

        # CUDF assumes values are always contiguous
        arbitrary = np.asarray(arbitrary, order="C")

        if arbitrary.ndim == 0:
            # TODO: Or treat as scalar?
            arbitrary = arbitrary[np.newaxis]

        if arbitrary.dtype.kind == "O":
            is_na = pd.isna(arbitrary)
            if is_na.any():
                if is_na.all():
                    # Avoid pyarrow converting np.ndarray[object] of all NaNs to float
                    raise MixedTypeError(
                        "Cannot have all NaN values with object dtype."
                    )
                arbitrary = pa.array(arbitrary)
            else:
                # Let pandas potentially infer object type
                # e.g. np.array([pd.Timestamp(...)], dtype=object) -> datetime64
                arbitrary = pd.Series(arbitrary)
            return as_column(arbitrary, dtype=dtype, nan_as_null=nan_as_null)
        elif arbitrary.dtype.kind in "SU":
            result_column = ColumnBase.from_arrow(pa.array(arbitrary))
            if dtype is not None:
                result_column = result_column.astype(dtype)
            return result_column
        elif arbitrary.dtype.kind in "biuf":
            if not arbitrary.dtype.isnative:
                # Not supported by pylibcudf
                arbitrary = arbitrary.astype(arbitrary.dtype.newbyteorder("="))
            result_column = ColumnBase.create(
                plc.Column.from_array_interface(arbitrary), arbitrary.dtype
            )
            if nan_as_null is not False:
                result_column = result_column.nans_to_nulls()
            if dtype is not None:
                result_column = result_column.astype(dtype)
            return result_column
        elif arbitrary.dtype.kind in "mM":
            time_unit = np.datetime_data(arbitrary.dtype)[0]
            if time_unit in ("D", "W", "M", "Y"):
                # TODO: Raise in these cases instead of downcasting to s?
                new_type = f"{arbitrary.dtype.type.__name__}[s]"
                arbitrary = arbitrary.astype(new_type)
            elif time_unit == "generic":
                # TODO: This should probably be in cudf.dtype
                raise TypeError(
                    f"{arbitrary.dtype.type.__name__} must have a unit specified"
                )

            is_nat = np.isnat(arbitrary)
            mask = None
            if is_nat.any():
                if nan_as_null is not False:
                    # Convert NaT to NA, which pyarrow does by default
                    return as_column(
                        pa.array(arbitrary),
                        dtype=dtype,
                        nan_as_null=nan_as_null,
                    )
                # Consider NaT as NA in the mask
                # but maintain NaT as a value
                mask = plc.Column.from_array_interface(~is_nat)
            plc_column = plc.Column.from_array_interface(arbitrary)
            if mask is not None:
                plc_column = plc_column.with_mask(
                    *plc.transform.bools_to_mask(mask)
                )
            result_column = ColumnBase.create(plc_column, arbitrary.dtype)
            if dtype is not None:
                result_column = result_column.astype(dtype)
            return result_column
        else:
            raise NotImplementedError(f"{arbitrary.dtype} not supported")
    else:
        # Try to convert to memoryview
        try:
            view = memoryview(arbitrary)
            return as_column(
                np.asarray(view), dtype=dtype, nan_as_null=nan_as_null
            )
        except TypeError:
            pass

        # Memoryview failed, check if it's iterable
        if not isinstance(arbitrary, (Iterable, Sequence)):
            raise TypeError(
                f"{type(arbitrary).__name__} must be an iterable or sequence."
            )
        elif isinstance(arbitrary, Iterator):
            arbitrary = list(arbitrary)

    # Start of arbitrary that's not handed above but dtype provided
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise NotImplementedError(
            "Use `tz_localize()` to construct timezone aware data."
        )
    elif isinstance(dtype, DecimalDtype):
        # Arrow throws a type error if the input is of
        # mixed-precision and cannot fit into the provided
        # decimal type properly, see:
        # https://github.com/apache/arrow/pull/9948
        # Hence we should let the exception propagate to
        # the user.
        if isinstance(dtype, cudf.Decimal128Dtype):
            pa_type = pa.decimal128(
                precision=dtype.precision, scale=dtype.scale
            )
            column_class = cudf.core.column.DecimalColumn
        elif isinstance(dtype, cudf.Decimal64Dtype):
            pa_type = pa.decimal64(
                precision=dtype.precision, scale=dtype.scale
            )
            column_class = cudf.core.column.DecimalColumn
        elif isinstance(dtype, cudf.Decimal32Dtype):
            pa_type = pa.decimal32(
                precision=dtype.precision, scale=dtype.scale
            )
            column_class = cudf.core.column.DecimalColumn
        else:
            raise NotImplementedError(f"{dtype} not implemented")
        data = pa.array(arbitrary, type=pa_type)
        return column_class.from_arrow(data)
    elif isinstance(
        dtype,
        (
            pd.CategoricalDtype,
            CategoricalDtype,
            pd.IntervalDtype,
            IntervalDtype,
        ),
    ) or dtype in {
        "category",
        "interval",
        "str",
        str,
        np.str_,
        object,
        np.dtype(object),
    }:
        if isinstance(dtype, (CategoricalDtype, IntervalDtype)):
            dtype = dtype.to_pandas()
            if isinstance(dtype, pd.IntervalDtype):
                # pd.Series(arbitrary) might be already inferred as IntervalDtype
                ser = pd.Series(arbitrary).astype(dtype)
            else:
                ser = pd.Series(
                    arbitrary,
                    dtype=pd.CategoricalDtype(ordered=dtype.ordered),  # type: ignore[union-attr]  # (categorical dtype converted from cudf dtype)
                )
                if dtype.categories is not None:  # type: ignore[union-attr]  # (categorical dtype converted from cudf dtype)
                    ser = ser.cat.set_categories(
                        dtype.categories,  # type: ignore[union-attr]  # (categorical dtype converted from cudf dtype)
                        ordered=dtype.ordered,  # type: ignore[union-attr]  # (categorical dtype converted from cudf dtype)
                    )
        else:
            ser = pd.Series(arbitrary, dtype=dtype)
        return as_column(ser, nan_as_null=nan_as_null)
    elif isinstance(dtype, (StructDtype, ListDtype)):
        try:
            data = pa.array(arbitrary, type=dtype.to_arrow())  # type: ignore[arg-type]  # (cudf nested dtype converts to Arrow type)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            if isinstance(dtype, ListDtype):
                # e.g. test_cudf_list_struct_write
                return cudf.core.column.ListColumn.from_sequences(arbitrary)
            raise
        return as_column(data, nan_as_null=nan_as_null)

    from_pandas = nan_as_null is None or nan_as_null
    if dtype is not None:
        if is_arrow_null_dtype(dtype):
            arbitrary = pa.array(
                arbitrary,
                type=dtype.pyarrow_dtype,  # type: ignore[union-attr]  # (arrow null dtype has pyarrow_dtype)
                from_pandas=True,
            )
            return as_column(arbitrary, nan_as_null=nan_as_null, dtype=dtype)
        try:
            arbitrary = pa.array(
                arbitrary,
                type=cudf_dtype_to_pa_type(dtype),
                from_pandas=from_pandas,
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError) as err:
            if err.args[0].startswith("Could not convert <NA>"):
                # nan_as_null=False, but we want to allow pd.NA values
                arbitrary = pa.array(
                    arbitrary,
                    type=cudf_dtype_to_pa_type(dtype),
                    from_pandas=True,
                )
            else:
                if isinstance(
                    dtype,
                    (
                        CategoricalDtype,
                        DecimalDtype,
                        IntervalDtype,
                        ListDtype,
                        StructDtype,
                    ),
                ):
                    dtype = dtype.to_pandas()
                arbitrary = pd.Series(arbitrary, dtype=dtype)
        return as_column(arbitrary, nan_as_null=nan_as_null, dtype=dtype)
    else:
        for element in arbitrary:
            # Carve-outs that cannot be parsed by pyarrow/pandas
            if is_column_like(element):
                # e.g. test_nested_series_from_sequence_data
                return cudf.core.column.ListColumn.from_sequences(arbitrary)
            elif isinstance(element, cp.ndarray):
                # e.g. test_series_from_cupy_scalars
                return as_column(
                    cp.array(arbitrary),
                    dtype=dtype,
                    nan_as_null=nan_as_null,
                    length=length,
                )
            elif (
                isinstance(
                    element,
                    (datetime.datetime, datetime.timedelta, pd.Interval),
                )
                or element is pd.NaT
            ):
                # datetime.datetime/timedelta cover their pd.Timestamp/
                # pd.Timedelta subclasses; routing stdlib datetimes through
                # pandas keeps mixed datetime+non-datetime inputs on the
                # object-dtype path (MixedTypeError) instead of silently
                # coercing them.
                # TODO: Remove this after
                # https://github.com/apache/arrow/issues/26492
                # is fixed.
                # Note: pd.Interval also requires pandas Series conversion
                # because PyArrow cannot infer interval type from raw list
                return as_column(
                    pd.Series(arbitrary),
                    dtype=dtype,
                    nan_as_null=nan_as_null,
                    length=length,
                )
            elif not any(element is na for na in (None, pd.NA, np.nan)):
                # Might have NA + element like above, but short-circuit if
                # an element pyarrow/pandas might be able to parse
                break
        try:
            arbitrary = pa.array(arbitrary, from_pandas=from_pandas)
            if (
                cudf.get_option("mode.pandas_compatible")
                and pa.types.is_integer(arbitrary.type)
                and arbitrary.null_count > 0
            ):
                # TODO: Need to re-visit this cast and fill_null
                # calls while addressing the following issue:
                # https://github.com/rapidsai/cudf/issues/14149
                arbitrary = arbitrary.cast(pa.float64())
                arbitrary = pc.fill_null(arbitrary, np.nan)
            if (
                cudf.get_option("default_integer_bitwidth")
                and pa.types.is_integer(arbitrary.type)
            ) or (
                cudf.get_option("default_float_bitwidth")
                and pa.types.is_floating(arbitrary.type)
            ):
                dtype = _maybe_convert_to_default_type(
                    np.dtype(arbitrary.type.to_pandas_dtype())
                )
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            arbitrary = pd.Series(arbitrary)
            if (
                cudf.get_option("default_integer_bitwidth")
                and arbitrary.dtype.kind in set("iu")
            ) or (
                cudf.get_option("default_float_bitwidth")
                and arbitrary.dtype.kind == "f"
            ):
                dtype = _maybe_convert_to_default_type(arbitrary.dtype)
        return as_column(arbitrary, nan_as_null=nan_as_null, dtype=dtype)


def serialize_columns(columns: list[ColumnBase]) -> tuple[list[dict], list]:
    """
    Return the headers and frames resulting
    from serializing a list of Column

    Parameters
    ----------
    columns : list
        list of Columns to serialize

    Returns
    -------
    headers : list
        list of header metadata for each Column
    frames : list
        list of frames
    """
    headers: list[dict[Any, Any]] = []
    frames = []

    if len(columns) > 0:
        header_columns: list[tuple[dict, list]] = [
            c.device_serialize() for c in columns
        ]
        headers, column_frames = zip(*header_columns, strict=True)
        for f in column_frames:
            frames.extend(f)

    return headers, frames


def deserialize_columns(headers: list[dict], frames: list) -> list[ColumnBase]:
    """
    Construct a list of Columns from a list of headers
    and frames.
    """
    columns = []

    for meta in headers:
        col_frame_count = meta["frame_count"]
        col_typ = Serializable._name_type_map[meta["type-serialized-name"]]
        colobj = col_typ.deserialize(meta, frames[:col_frame_count])
        columns.append(colobj)
        # Advance frames
        frames = frames[col_frame_count:]

    return columns


def concat_columns(objs: Sequence[ColumnBase]) -> ColumnBase:
    """Concatenate a sequence of columns."""
    if len(objs) == 0:
        return column_empty(0, dtype=np.dtype(np.float64))

    # If all columns are `NumericalColumn` with different dtypes,
    # we cast them to a common dtype.
    # Notice, we can always cast pure null columns
    not_null_col_dtypes = [o.dtype for o in objs if o.null_count != len(o)]
    if len(not_null_col_dtypes) and all(
        is_dtype_obj_numeric(dtype, include_decimal=False)
        and dtype.kind == "M"
        for dtype in not_null_col_dtypes
    ):
        common_dtype = find_common_type(not_null_col_dtypes)
        # Cast all columns to the common dtype
        objs = [obj.astype(common_dtype) for obj in objs]

    # Find the first non-null column:
    head = next((obj for obj in objs if obj.null_count != len(obj)), objs[0])

    replacement_cols: dict[int, ColumnBase] = {}
    for i, obj in enumerate(objs):
        # Check that all columns are the same type:
        if not is_dtype_equal(obj.dtype, head.dtype):
            # if all null, cast to appropriate dtype
            if obj.is_all_null:
                replacement_cols[i] = column_empty(
                    row_count=len(obj), dtype=head.dtype
                )
            else:
                raise ValueError("All columns must be the same type")
    if replacement_cols:
        if len(replacement_cols) == len(objs):
            new_objs: Sequence[ColumnBase] = list(replacement_cols.values())
        else:
            new_objs = list(objs)
            for idx, col in replacement_cols.items():
                new_objs[idx] = col
    else:
        new_objs = objs

    # TODO: This logic should be generalized to a dispatch to
    # ColumnBase._concat so that all subclasses can override necessary
    # behavior. However, at the moment it's not clear what that API should look
    # like, so CategoricalColumn simply implements a minimal working API.
    if all(isinstance(o.dtype, CategoricalDtype) for o in new_objs):
        return cudf.core.column.categorical.CategoricalColumn._concat(
            cast(
                "MutableSequence[cudf.core.column.categorical.CategoricalColumn]",
                new_objs,
            )
        )

    newsize = sum(map(len, new_objs))
    if newsize > np.iinfo(SIZE_TYPE_DTYPE).max:
        raise MemoryError(
            f"Result of concat cannot have size > {SIZE_TYPE_DTYPE}_MAX"
        )
    elif newsize == 0:
        return column_empty(0, head.dtype)

    # Filter out inputs that have 0 length, then concatenate.
    objs_with_len = [o for o in new_objs if len(o)]
    with access_columns(  # type: ignore[assignment]
        *objs_with_len, mode="read", scope="internal"
    ) as objs_with_len:
        result = ColumnBase.create(
            plc.concatenate.concatenate(
                [col.plc_column for col in objs_with_len]
            ),
            objs_with_len[0].dtype,
        )
    # Preserve cached `_local_time` for DatetimeTZColumn when every input has
    # it. The stored UTC values produced by `tz_localize` can be inaccurate
    # near DST transitions (the transition table is not globally sorted), so
    # `_local_time` is the authoritative source for producing correct pandas
    # output.
    if isinstance(result, cudf.core.column.datetime.DatetimeTZColumn) and all(
        isinstance(o, cudf.core.column.datetime.DatetimeTZColumn)
        and "_local_time" in o.__dict__
        for o in objs_with_len
    ):
        result._local_time = concat_columns(
            [o._local_time for o in objs_with_len]  # type: ignore[attr-defined]
        )  # type: ignore[assignment]
    return result


def _rank_out_dtype(
    orig_dtype: "DtypeObj",
    method: plc.aggregation.RankMethod,
    pct: bool,
) -> "DtypeObj":
    """Determine the output dtype for rank() to match pandas 3 semantics.

    Pandas 3 rank() output dtypes by input dtype:
      Float64 / Int64 / UInt64 (nullable numpy-backed):
        average or pct=True → Float64,  otherwise → UInt64
      float64[pyarrow] / int64[pyarrow] (Arrow numeric):
        average or pct=True → double[pyarrow], otherwise → uint64[pyarrow]
      string[pyarrow] (StringDtype with pyarrow storage):
        always → Float64
      str / string[python] / float64 / … (non-nullable or non-numeric ext):
        → float64 (unchanged)
    """
    is_average = method == plc.aggregation.RankMethod.AVERAGE
    if isinstance(orig_dtype, pd.ArrowDtype):
        if pa.types.is_floating(
            orig_dtype.pyarrow_dtype
        ) or pa.types.is_integer(orig_dtype.pyarrow_dtype):
            if is_average or pct:
                return pd.ArrowDtype(pa.float64())
            else:
                return pd.ArrowDtype(pa.uint64())
        # Non-numeric Arrow types (e.g. string): fall through to float64
    elif isinstance(orig_dtype, pd.StringDtype):
        # string[pyarrow] uses pyarrow storage + pd.NA → Float64
        # str / string[python] → float64 (no change)
        if (
            orig_dtype.na_value is pd.NA
            and getattr(orig_dtype, "storage", None) == "pyarrow"
        ):
            return pd.Float64Dtype()
    elif is_pandas_nullable_numpy_dtype(orig_dtype) and not isinstance(
        orig_dtype, (pd.StringDtype, pd.BooleanDtype)
    ):
        # Float64, Int64, UInt64, etc.
        if is_average or pct:
            return pd.Float64Dtype()
        else:
            return pd.UInt64Dtype()
    return np.dtype("float64")
