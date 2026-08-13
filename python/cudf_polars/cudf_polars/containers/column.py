# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.exceptions import InvalidOperationError

import pylibcudf as plc
from pylibcudf.strings.convert.convert_booleans import from_booleans
from pylibcudf.strings.convert.convert_floats import from_floats, is_float, to_floats
from pylibcudf.strings.convert.convert_integers import (
    from_integers,
    is_integer,
    to_integers,
)

from cudf_polars.containers import DataType
from cudf_polars.containers.datatype import _dtype_from_header, _dtype_to_header
from cudf_polars.utils import conversion
from cudf_polars.utils.dtypes import is_order_preserving_cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from polars import Series as pl_Series

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.typing import (
        ColumnHeader,
        ColumnOptions,
        DeserializedColumnOptions,
        Slice,
    )

__all__: list[str] = ["Column"]


# Float64 has 53-bit significand precision (52 fraction bits + 1 implicit leading bit),
# giving 15-17 significant decimal digits. Use 17 as upper bound when casting to
# intermediate decimal to preserve all float64 precision before rounding.
# https://en.wikipedia.org/wiki/Double-precision_floating-point_format
_FLOAT64_DECIMAL_PRECISION = 17


class Column:
    """An immutable column with sortedness metadata."""

    obj: plc.Column
    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    is_scalar: bool
    # Optional name, only ever set by evaluation of NamedExpr nodes
    # The internal evaluation should not care about the name.
    name: str | None
    dtype: DataType

    def __init__(
        self,
        column: plc.Column,
        dtype: DataType,
        *,
        is_sorted: plc.types.Sorted = plc.types.Sorted.NO,
        order: plc.types.Order = plc.types.Order.ASCENDING,
        null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
        name: str | None = None,
    ):
        self.obj = column
        self.is_scalar = self.size == 1
        self.name = name
        self.dtype = dtype
        self.set_sorted(is_sorted=is_sorted, order=order, null_order=null_order)

    @classmethod
    def deserialize(
        cls,
        header: ColumnHeader,
        frames: tuple[memoryview[bytes], plc.gpumemoryview],
        stream: Stream,
    ) -> Self:
        """
        Create a Column from a serialized representation returned by `.serialize()`.

        Parameters
        ----------
        header
            The (unpickled) metadata required to reconstruct the object.
        frames
            Two-tuple of frames (a memoryview and a gpumemoryview).
        stream
            CUDA stream used for device memory operations and kernel launches
            on this column. The caller is responsible for ensuring that
            the data in ``frames`` is valid on ``stream``.

        Returns
        -------
        Column
            The deserialized Column.
        """
        packed_metadata, packed_gpu_data = frames
        (plc_column,) = plc.contiguous_split.unpack_from_memoryviews(
            packed_metadata, packed_gpu_data, stream
        ).columns()
        return cls(plc_column, **cls.deserialize_ctor_kwargs(header["column_kwargs"]))

    @staticmethod
    def deserialize_ctor_kwargs(
        column_kwargs: ColumnOptions,
    ) -> DeserializedColumnOptions:
        """Deserialize the constructor kwargs for a Column."""
        return {
            "is_sorted": column_kwargs["is_sorted"],
            "order": column_kwargs["order"],
            "null_order": column_kwargs["null_order"],
            "name": column_kwargs["name"],
            "dtype": DataType(_dtype_from_header(column_kwargs["dtype"])),
        }

    def serialize(
        self,
        stream: Stream,
    ) -> tuple[ColumnHeader, tuple[memoryview[bytes], plc.gpumemoryview]]:
        """
        Serialize the Column into header and frames.

        Follows the Dask serialization scheme with a picklable header (dict) and
        a tuple of frames (in this case a contiguous host and device buffer).

        To enable dask support, dask serializers must be registered

            >>> from cudf_polars.streaming.dask_serialize import register
            >>> register()

        Returns
        -------
        header
            A dict containing any picklable metadata required to reconstruct the object.
        frames
            Two-tuple of frames suitable for passing to `plc.contiguous_split.unpack_from_memoryviews`
        """
        packed = plc.contiguous_split.pack(plc.Table([self.obj]), stream=stream)
        header: ColumnHeader = {
            "column_kwargs": self.serialize_ctor_kwargs(),
            "frame_count": 2,
        }
        return header, packed.release()

    def serialize_ctor_kwargs(self) -> ColumnOptions:
        """Serialize the constructor kwargs for self."""
        return {
            "is_sorted": self.is_sorted,
            "order": self.order,
            "null_order": self.null_order,
            "name": self.name,
            "dtype": _dtype_to_header(self.dtype.polars_type),
        }

    def obj_scalar(self, stream: Stream) -> plc.Scalar:
        """
        A copy of the column object as a pylibcudf Scalar.

        Parameters
        ----------
        stream
            CUDA stream used for device memory operations and kernel launches.
            ``self.obj`` must be valid on this stream, and the result will be
            valid on this stream.

        Returns
        -------
        pylibcudf Scalar object.

        Raises
        ------
        ValueError
            If the column is not length-1.
        """
        if not self.is_scalar:
            raise ValueError(f"Cannot convert a column of length {self.size} to scalar")
        return plc.copying.get_element(self.obj, 0, stream=stream)

    def rename(self, name: str | None, /) -> Self:
        """
        Return a shallow copy with a new name.

        Parameters
        ----------
        name
            New name

        Returns
        -------
        Shallow copy of self with new name set.
        """
        new = self.copy()
        new.name = name
        return new

    def sorted_like(self, like: Column, /) -> Self:
        """
        Return a shallow copy with sortedness from like.

        Parameters
        ----------
        like
            The column to copy sortedness metadata from.

        Returns
        -------
        Shallow copy of self with metadata set.

        See Also
        --------
        set_sorted, copy_metadata
        """
        return type(self)(
            self.obj,
            name=self.name,
            dtype=self.dtype,
            is_sorted=like.is_sorted,
            order=like.order,
            null_order=like.null_order,
        )

    def check_sorted(
        self,
        *,
        order: plc.types.Order,
        null_order: plc.types.NullOrder,
        stream: Stream,
    ) -> bool:
        """
        Check if the column is sorted.

        Parameters
        ----------
        order
            The requested sort order.
        null_order
            Where nulls sort to.
        stream
            CUDA stream used for device memory operations and kernel launches
            on this Column. The data in ``self.obj`` must be valid on this stream.

        Returns
        -------
        True if the column is sorted, false otherwise.

        Notes
        -----
        If the sortedness flag is not set, this launches a kernel to
        check sortedness.
        """
        if self.size <= 1 or self.size == self.null_count:
            return True
        if self.is_sorted == plc.types.Sorted.YES:
            return self.order == order and (
                self.null_count == 0 or self.null_order == null_order
            )
        if plc.sorting.is_sorted(
            plc.Table([self.obj]), [order], [null_order], stream=stream
        ):
            self.set_sorted(
                is_sorted=plc.types.Sorted.YES,
                order=order,
                null_order=null_order,
            )
            return True
        return False

    def astype(self, dtype: DataType, stream: Stream, *, strict: bool = True) -> Column:
        """
        Cast the column to as the requested dtype.

        Parameters
        ----------
        dtype
            Datatype to cast to.
        stream
            CUDA stream used for device memory operations and kernel launches
            on this Column. The data in ``self.obj`` must be valid on this stream.
        strict
            If True, raise an error if the cast is unsupported.
            If False, return nulls for unsupported casts.

        Returns
        -------
        Column of requested type.

        Raises
        ------
        RuntimeError
            If the cast is unsupported.

        Notes
        -----
        This only produces a copy if the requested dtype doesn't match
        the current one.
        """
        plc_dtype = dtype.plc_type
        if self.obj.type() == plc_dtype:
            return self

        if (
            plc_dtype.id() == plc.TypeId.STRING
            or self.obj.type().id() == plc.TypeId.STRING
        ):
            return Column(
                self._handle_string_cast(plc_dtype, stream=stream, strict=strict),
                dtype=dtype,
                name=self.name,
            )
        elif plc.traits.is_integral_not_bool(
            self.obj.type()
        ) and plc.traits.is_timestamp(plc_dtype):
            upcasted = plc.unary.cast(
                self.obj, plc.DataType(plc.TypeId.INT64), stream=stream
            )
            plc_col = plc.column.Column(
                plc_dtype,
                upcasted.size(),
                upcasted.data(),
                upcasted.null_mask(),
                upcasted.null_count(),
                upcasted.offset(),
                upcasted.children(),
            )
            return Column(plc_col, dtype=dtype, name=self.name).sorted_like(self)
        elif plc.traits.is_integral_not_bool(plc_dtype) and plc.traits.is_timestamp(
            self.obj.type()
        ):
            plc_col = plc.column.Column(
                plc.DataType(
                    plc.TypeId.INT32
                    if self.obj.type().id() == plc.TypeId.TIMESTAMP_DAYS
                    else plc.TypeId.INT64
                ),
                self.obj.size(),
                self.obj.data(),
                self.obj.null_mask(),
                self.obj.null_count(),
                self.obj.offset(),
                self.obj.children(),
            )
            return Column(
                plc.unary.cast(plc_col, plc_dtype, stream=stream),
                dtype=dtype,
                name=self.name,
            ).sorted_like(self)
        elif plc.traits.is_integral_not_bool(plc_dtype) and plc.traits.is_duration(
            self.obj.type()
        ):
            # A duration is stored as an integer tick count, so casting to that
            # integer type is a no-op reinterpret of the same bytes. Relabel the
            # column instead of launching a cast kernel.
            rep = plc.DataType(
                plc.TypeId.INT32
                if self.obj.type().id() == plc.TypeId.DURATION_DAYS
                else plc.TypeId.INT64
            )
            plc_col = plc.column.Column(
                rep,
                self.obj.size(),
                self.obj.data(),
                self.obj.null_mask(),
                self.obj.null_count(),
                self.obj.offset(),
                self.obj.children(),
            )
            if rep.id() != plc_dtype.id():
                plc_col = plc.unary.cast(plc_col, plc_dtype, stream=stream)
            return Column(plc_col, dtype=dtype, name=self.name).sorted_like(self)
        elif plc.traits.is_floating_point(
            self.obj.type()
        ) and plc.traits.is_fixed_point(plc_dtype):
            # cudf::cast from float to decimal truncates instead of rounding.
            # Polars expects rounding, so cast to higher precision decimal first to
            # preserve float64's significant digits, then round_decimal to target scale.
            target_scale = -plc_dtype.scale()
            return Column(
                plc.round.round_decimal(
                    plc.unary.cast(
                        self.obj,
                        plc.DataType(
                            plc.TypeId.DECIMAL128,
                            scale=-max(target_scale, _FLOAT64_DECIMAL_PRECISION),
                        ),
                        stream=stream,
                    ),
                    decimal_places=target_scale,
                    # Note polars uses banker's rounding
                    # See crates/polars-ops/src/series/ops/round.rs (RoundMethod::HalfEven)
                    round_method=plc.round.RoundingMethod.HALF_EVEN,
                    stream=stream,
                ),
                dtype=dtype,
                name=self.name,
            )
        else:
            result = Column(
                plc.unary.cast(self.obj, plc_dtype, stream=stream),
                dtype=dtype,
                name=self.name,
            )
            if is_order_preserving_cast(self.obj.type(), plc_dtype):
                return result.sorted_like(self)
            return result

    def _handle_string_cast(
        self, dtype: plc.DataType, stream: Stream, *, strict: bool
    ) -> plc.Column:
        if dtype.id() == plc.TypeId.STRING:
            if plc.traits.is_floating_point(self.obj.type()):
                return from_floats(self.obj, stream=stream)
            elif plc.traits.is_integral_not_bool(self.obj.type()):
                return from_integers(self.obj, stream=stream)
            elif plc.traits.is_boolean(self.obj.type()):
                return from_booleans(
                    self.obj,
                    plc.Scalar.from_py("true", dtype, stream=stream),
                    plc.Scalar.from_py("false", dtype, stream=stream),
                    stream=stream,
                )
            else:
                raise InvalidOperationError(
                    f"Unsupported casting from {self.dtype.id()} to {dtype.id()}."
                )

        type_checker: Callable[[plc.Column, Stream], plc.Column]
        type_caster: Callable[[plc.Column, plc.DataType, Stream], plc.Column]
        if plc.traits.is_floating_point(dtype):
            type_checker = is_float
            type_caster = to_floats
        elif plc.traits.is_integral_not_bool(dtype):
            # is_integer has a second optional int_type: plc.DataType | None = None argument
            # we do not use
            # unused-ignore for if RMM is missing
            type_checker = is_integer  # type: ignore[assignment,unused-ignore]
            type_caster = to_integers
        else:
            raise InvalidOperationError(
                f"Unsupported casting from {self.dtype.id()} to {dtype.id()}."
            )

        castable = type_checker(self.obj, stream=stream)  # type: ignore[call-arg]
        if not plc.reduce.reduce(
            castable,
            plc.aggregation.all(),
            plc.DataType(plc.TypeId.BOOL8),
            stream=stream,
        ).to_py(stream=stream):
            if strict:
                raise InvalidOperationError(
                    f"Conversion from {self.dtype.id()} to {dtype.id()} failed."
                )
            else:
                values = self.obj.with_mask(
                    *plc.transform.bools_to_mask(castable, stream=stream)
                )
        else:
            values = self.obj
        return type_caster(values, dtype, stream=stream)

    def copy_metadata(self, from_: pl_Series, /) -> Self:
        """
        Copy metadata from a host series onto self.

        Parameters
        ----------
        from_
            Polars series to copy metadata from

        Returns
        -------
        Self with metadata set.

        See Also
        --------
        set_sorted, sorted_like
        """
        self.name = from_.name
        if len(from_) <= 1:
            return self
        ascending = from_.flags["SORTED_ASC"]
        descending = from_.flags["SORTED_DESC"]
        if ascending or descending:
            has_null_first = from_.item(0) is None
            has_null_last = from_.item(-1) is None
            order = (
                plc.types.Order.ASCENDING if ascending else plc.types.Order.DESCENDING
            )
            null_order = plc.types.NullOrder.BEFORE
            if (descending and has_null_first) or (ascending and has_null_last):
                null_order = plc.types.NullOrder.AFTER
            return self.set_sorted(
                is_sorted=plc.types.Sorted.YES,
                order=order,
                null_order=null_order,
            )
        return self

    def set_sorted(
        self,
        *,
        is_sorted: plc.types.Sorted,
        order: plc.types.Order,
        null_order: plc.types.NullOrder,
    ) -> Self:
        """
        Modify sortedness metadata in place.

        Parameters
        ----------
        is_sorted
            Is the column sorted
        order
            The order if sorted
        null_order
            Where nulls sort, if sorted

        Returns
        -------
        Self with metadata set.
        """
        if self.size <= 1:
            is_sorted = plc.types.Sorted.YES
        self.is_sorted = is_sorted
        self.order = order
        self.null_order = null_order
        return self

    def copy(self) -> Self:
        """
        A shallow copy of the column.

        Returns
        -------
        New column sharing data with self.
        """
        return type(self)(
            self.obj,
            is_sorted=self.is_sorted,
            order=self.order,
            null_order=self.null_order,
            name=self.name,
            dtype=self.dtype,
        )

    def mask_nans(self, stream: Stream) -> Self:
        """Return a shallow copy of self with nans masked out."""
        if plc.traits.is_floating_point(self.obj.type()):
            old_count = self.null_count
            mask, new_count = plc.transform.nans_to_nulls(self.obj, stream=stream)
            result = type(self)(self.obj.with_mask(mask, new_count), self.dtype)
            if old_count == new_count:
                return result.sorted_like(self)
            return result
        return self.copy()

    def nan_count(self, stream: Stream) -> int:
        """
        Return the number of NaN values in the column.

        Parameters
        ----------
        stream
            CUDA stream used for device memory operations and kernel launches.
            ``self.obj`` must be valid on this stream, and the result will be
            valid on this stream.

        Returns
        -------
        Number of NaN values in the column.
        """
        result: int
        if self.size > 0 and plc.traits.is_floating_point(self.obj.type()):
            # See https://github.com/rapidsai/cudf/issues/20202 for we type ignore
            result = plc.reduce.reduce(  # type: ignore[assignment]
                plc.unary.is_nan(self.obj, stream=stream),
                plc.aggregation.sum(),
                plc.types.SIZE_TYPE,
                stream=stream,
            ).to_py(stream=stream)
        else:
            result = 0
        return result

    @property
    def size(self) -> int:
        """Return the size of the column."""
        return self.obj.size()

    @property
    def null_count(self) -> int:
        """Return the number of Null values in the column."""
        return self.obj.null_count()

    def slice(self, zlice: Slice | None, stream: Stream) -> Self:
        """
        Slice a column.

        Parameters
        ----------
        zlice
            optional, tuple of start and length, negative values of start
            treated as for python indexing. If not provided, returns self.
        stream
            CUDA stream used for device memory operations and kernel launches
            on this Column. The data in ``self.obj`` must be valid on this stream.

        Returns
        -------
        New column (if zlice is not None) otherwise self (if it is)
        """
        if zlice is None:
            return self
        (table,) = plc.copying.slice(
            plc.Table([self.obj]),
            conversion.from_polars_slice(zlice, num_rows=self.size),
            stream=stream,
        )
        (column,) = table.columns()
        return type(self)(column, name=self.name, dtype=self.dtype).sorted_like(self)
