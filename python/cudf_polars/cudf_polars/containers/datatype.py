# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A datatype, preserving polars metadata."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal, assert_never, cast

import polars as pl

import pylibcudf as plc

if TYPE_CHECKING:
    from cudf_polars.typing import (
        DataTypeHeader,
        PolarsDataType,
    )

__all__ = ["DataType"]

SCALAR_NAME_TO_POLARS_TYPE_MAP: dict[str, pl.DataType] = {
    "Boolean": pl.Boolean(),
    "Int8": pl.Int8(),
    "Int16": pl.Int16(),
    "Int32": pl.Int32(),
    "Int64": pl.Int64(),
    "Object": pl.Object(),
    "UInt8": pl.UInt8(),
    "UInt16": pl.UInt16(),
    "UInt32": pl.UInt32(),
    "UInt64": pl.UInt64(),
    "Float32": pl.Float32(),
    "Float64": pl.Float64(),
    "String": pl.String(),
    "Null": pl.Null(),
    "Date": pl.Date(),
    "Time": pl.Time(),
}


def _dtype_to_header(dtype: pl.DataType) -> DataTypeHeader:
    name = type(dtype).__name__
    if name in SCALAR_NAME_TO_POLARS_TYPE_MAP:
        return {"kind": "scalar", "name": name}
    if isinstance(dtype, pl.Decimal):
        return {
            "kind": "decimal",
            "precision": dtype.precision,
            "scale": dtype.scale,
        }
    if isinstance(dtype, pl.Datetime):
        return {
            "kind": "datetime",
            "time_unit": dtype.time_unit,
            "time_zone": dtype.time_zone,
        }
    if isinstance(dtype, pl.Duration):
        return {"kind": "duration", "time_unit": dtype.time_unit}
    if isinstance(dtype, pl.List):
        # isinstance narrows dtype to pl.List, but .inner returns DataTypeClass | DataType
        return {
            "kind": "list",
            "inner": _dtype_to_header(cast("pl.DataType", dtype.inner)),
        }
    if isinstance(dtype, pl.Struct):
        # isinstance narrows dtype to pl.Struct, but field.dtype returns DataTypeClass | DataType
        return {
            "kind": "struct",
            "fields": [
                {
                    "name": f.name,
                    "dtype": _dtype_to_header(cast("pl.DataType", f.dtype)),
                }
                for f in dtype.fields
            ],
        }
    raise NotImplementedError(f"Unsupported dtype {dtype!r}")


def _dtype_from_header(header: DataTypeHeader) -> pl.DataType:
    if header["kind"] == "scalar":
        name = header["name"]
        try:
            return SCALAR_NAME_TO_POLARS_TYPE_MAP[name]
        except KeyError as err:
            raise NotImplementedError(f"Unknown scalar dtype name: {name}") from err
    if header["kind"] == "decimal":
        return pl.Decimal(header["precision"], header["scale"])
    if header["kind"] == "datetime":
        return pl.Datetime(
            time_unit=cast("Literal['ns', 'us', 'ms']", header["time_unit"]),
            time_zone=header["time_zone"],
        )
    if header["kind"] == "duration":
        return pl.Duration(
            time_unit=cast("Literal['ns', 'us', 'ms']", header["time_unit"])
        )
    if header["kind"] == "list":
        return pl.List(_dtype_from_header(header["inner"]))
    if header["kind"] == "struct":
        return pl.Struct(
            [
                pl.Field(f["name"], _dtype_from_header(f["dtype"]))
                for f in header["fields"]
            ]
        )
    raise NotImplementedError(f"Unsupported kind {header['kind']!r}")


@cache
def _from_polars(dtype: pl.DataType) -> plc.DataType:
    """
    Convert a polars datatype to a pylibcudf one.

    Parameters
    ----------
    dtype
        Polars dtype to convert

    Returns
    -------
    Matching pylibcudf DataType object.

    Raises
    ------
    NotImplementedError
        For unsupported conversions.
    """
    if isinstance(dtype, pl.Boolean):
        return plc.DataType(plc.TypeId.BOOL8)
    elif isinstance(dtype, pl.Int8):
        return plc.DataType(plc.TypeId.INT8)
    elif isinstance(dtype, pl.Int16):
        return plc.DataType(plc.TypeId.INT16)
    elif isinstance(dtype, pl.Int32):
        return plc.DataType(plc.TypeId.INT32)
    elif isinstance(dtype, pl.Int64):
        return plc.DataType(plc.TypeId.INT64)
    if isinstance(dtype, pl.UInt8):
        return plc.DataType(plc.TypeId.UINT8)
    elif isinstance(dtype, pl.UInt16):
        return plc.DataType(plc.TypeId.UINT16)
    elif isinstance(dtype, pl.UInt32):
        return plc.DataType(plc.TypeId.UINT32)
    elif isinstance(dtype, pl.UInt64):
        return plc.DataType(plc.TypeId.UINT64)
    elif isinstance(dtype, pl.Float32):
        return plc.DataType(plc.TypeId.FLOAT32)
    elif isinstance(dtype, pl.Float64):
        return plc.DataType(plc.TypeId.FLOAT64)
    elif isinstance(dtype, pl.Date):
        return plc.DataType(plc.TypeId.TIMESTAMP_DAYS)
    elif isinstance(dtype, pl.Time):
        raise NotImplementedError("Time of day dtype not implemented")
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.TIMESTAMP_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.TIMESTAMP_NANOSECONDS)
        assert dtype.time_unit is not None  # pragma: no cover
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.Duration):
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.DURATION_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.DURATION_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.DURATION_NANOSECONDS)
        assert dtype.time_unit is not None  # pragma: no cover
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.String):
        return plc.DataType(plc.TypeId.STRING)
    elif isinstance(dtype, pl.Decimal):
        return plc.DataType(plc.TypeId.DECIMAL128, scale=-dtype.scale)
    elif isinstance(dtype, pl.Null):
        # TODO: Hopefully
        return plc.DataType(plc.TypeId.EMPTY)
    elif isinstance(dtype, pl.List):
        # Recurse to catch unsupported inner types
        _ = _from_polars(dtype.inner)
        return plc.DataType(plc.TypeId.LIST)
    elif isinstance(dtype, pl.Struct):
        # Recurse to catch unsupported field types
        for field in dtype.fields:
            _ = _from_polars(field.dtype)
        return plc.DataType(plc.TypeId.STRUCT)
    else:
        raise NotImplementedError(f"{dtype=} conversion not supported")


class DataType:
    """A datatype, preserving polars metadata."""

    polars_type: pl.datatypes.DataType
    plc_type: plc.DataType

    def __init__(self, polars_dtype: PolarsDataType) -> None:
        # Convert DataTypeClass to DataType instance if needed
        # polars allows both pl.Int64 (class) and pl.Int64() (instance)
        if isinstance(polars_dtype, type):
            polars_dtype = polars_dtype()
        # After conversion, it's guaranteed to be a DataType instance
        self.polars_type = cast("pl.DataType", polars_dtype)
        self.plc_type = _from_polars(self.polars_type)

    def id(self) -> plc.TypeId:
        """The pylibcudf.TypeId of this DataType."""
        return self.plc_type.id()

    @property
    def children(self) -> list[DataType]:
        """The children types of this DataType."""
        # Type checker doesn't narrow polars_type through plc_type.id() checks
        if self.plc_type.id() == plc.TypeId.STRUCT:
            # field.dtype returns DataTypeClass | DataType, need to cast to DataType
            return [
                DataType(cast("pl.DataType", field.dtype))
                for field in cast("pl.Struct", self.polars_type).fields
            ]
        elif self.plc_type.id() == plc.TypeId.LIST:
            # .inner returns DataTypeClass | DataType, need to cast to DataType
            return [
                DataType(cast("pl.DataType", cast("pl.List", self.polars_type).inner))
            ]
        return []

    def scale(self) -> int:
        """The scale of this DataType."""
        return self.plc_type.scale()

    @staticmethod
    def common_decimal_dtype(left: DataType, right: DataType) -> DataType:
        """Return a common decimal DataType for the two inputs."""
        if not (
            plc.traits.is_fixed_point(left.plc_type)
            and plc.traits.is_fixed_point(right.plc_type)
        ):
            raise ValueError("Both inputs required to be decimal types.")
        return DataType(pl.Decimal(38, abs(min(left.scale(), right.scale()))))

    def __eq__(self, other: object) -> bool:
        """Equality of DataTypes."""
        if not isinstance(other, DataType):
            return False
        return self.polars_type == other.polars_type

    def __hash__(self) -> int:
        """Hash of the DataType."""
        return hash(self.polars_type)

    def __repr__(self) -> str:
        """Representation of the DataType."""
        return f"<DataType(polars={self.polars_type}, plc={self.id()!r})>"
