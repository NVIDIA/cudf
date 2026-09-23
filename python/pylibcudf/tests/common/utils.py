# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pyarrow.orc import write_table as orc_write_table
from pyarrow.parquet import write_table as pq_write_table

import pylibcudf as plc
from pylibcudf.io.types import CompressionType


def _map_string_view_to_string(dtype: pa.DataType) -> pa.DataType:
    """Recursively convert string_view -> string in a pyarrow type."""
    if pa.types.is_string_view(dtype):
        return pa.string()
    if pa.types.is_list(dtype):
        value_field = dtype.value_field
        return pa.list_(
            pa.field(
                value_field.name,
                _map_string_view_to_string(value_field.type),
                nullable=value_field.nullable,
                metadata=value_field.metadata,
            )
        )
    if pa.types.is_struct(dtype):
        return pa.struct(
            [
                pa.field(
                    f.name,
                    _map_string_view_to_string(f.type),
                    nullable=f.nullable,
                )
                for f in dtype
            ]
        )
    return dtype


def _normalize_string_view(obj):
    """
    Normalize string_view types to string.

    PyArrow 21+ may produce string_view in places where previous versions
    produced string. This helper ensures comparisons are not affected.
    """
    if isinstance(obj, pa.Table):
        new_schema = pa.schema(
            [
                pa.field(
                    f.name,
                    _map_string_view_to_string(f.type),
                    nullable=f.nullable,
                    metadata=f.metadata,
                )
                for f in obj.schema
            ],
            metadata=obj.schema.metadata,
        )
        return obj.cast(new_schema)
    elif isinstance(obj, pa.ChunkedArray):
        new_type = _map_string_view_to_string(obj.type)
        if new_type != obj.type:
            return obj.cast(new_type)
        return obj
    elif isinstance(obj, pa.Array):
        new_type = _map_string_view_to_string(obj.type)
        if new_type != obj.type:
            return obj.cast(new_type)
        return obj
    return obj


def synchronize_stream(stream=None):
    """Synchronize a stream, handling both explicit streams and None (default stream).

    Parameters
    ----------
    stream : Stream or None
        The stream to synchronize. If None, synchronizes the default stream.
    """
    if stream is None:
        plc.utils.DEFAULT_STREAM.synchronize()
    else:
        stream.synchronize()


def metadata_from_arrow_type(
    pa_type: pa.Array,
    name: str = "",
) -> plc.interop.ColumnMetadata | None:
    metadata = plc.interop.ColumnMetadata(name)
    if pa.types.is_list(pa_type):
        child_meta = [plc.interop.ColumnMetadata("offsets")]
        for i in range(pa_type.num_fields):
            field_meta = metadata_from_arrow_type(
                pa_type.field(i).type, pa_type.field(i).name
            )
            child_meta.append(field_meta)
        metadata = plc.interop.ColumnMetadata(name, children_meta=child_meta)
    elif pa.types.is_struct(pa_type):
        child_meta = []
        for i in range(pa_type.num_fields):
            field_meta = metadata_from_arrow_type(
                pa_type.field(i).type, pa_type.field(i).name
            )
            child_meta.append(field_meta)
        metadata = plc.interop.ColumnMetadata(
            name,
            # libcudf does not store field names, so just match pyarrow's.
            children_meta=child_meta,
        )
    return metadata


def assert_column_eq(
    lhs: pa.Array | plc.Column,
    rhs: pa.Array | plc.Column,
    check_field_nullability=False,
) -> None:
    """Verify that a pylibcudf array and PyArrow array are equal.

    Parameters
    ----------
    lhs: Union[pa.Array, plc.Column]
        The array with the expected values
    rhs: Union[pa.Array, plc.Column]
        The array to check
    check_field_nullability:
        For list/struct dtypes, whether to check if the nullable attributes
        on child fields are equal.

        Useful for checking roundtripping of lossy formats like JSON that may not
        preserve this information. Also, our Arrow interop functions make different
        choices by default than pyarrow field constructors since the interop functions
        may make data-dependent choices.
    """
    # Nested types require children metadata to be passed to the conversion function.
    if isinstance(lhs, (pa.Array, pa.ChunkedArray)) and isinstance(
        rhs, plc.Column
    ):
        rhs = rhs.to_arrow(metadata=metadata_from_arrow_type(lhs.type))
    elif isinstance(lhs, plc.Column) and isinstance(
        rhs, (pa.Array, pa.ChunkedArray)
    ):
        lhs = lhs.to_arrow(metadata=metadata_from_arrow_type(rhs.type))
    else:
        raise ValueError(
            "One of the inputs must be a Column and the other an Array"
        )

    if isinstance(lhs, pa.ChunkedArray):
        lhs = lhs.combine_chunks()
    if isinstance(rhs, pa.ChunkedArray):
        rhs = rhs.combine_chunks()

    # Normalize string_view -> string for comparison.
    # PyArrow 21+ may produce string_view where previous versions produced string.
    lhs = _normalize_string_view(lhs)
    rhs = _normalize_string_view(rhs)

    def _make_fields_nullable(typ):
        new_fields = []
        for i in range(typ.num_fields):
            child_field = typ.field(i)
            if not child_field.nullable:
                child_type = child_field.type
                if isinstance(child_field.type, (pa.StructType, pa.ListType)):
                    child_type = _make_fields_nullable(child_type)
                new_fields.append(
                    pa.field(child_field.name, child_type, nullable=True)
                )
            else:
                new_fields.append(child_field)

        if isinstance(typ, pa.StructType):
            return pa.struct(new_fields)
        elif isinstance(typ, pa.ListType):
            return pa.list_(new_fields[0])
        return typ

    def _contains_type(parent_typ, typ_checker):
        """
        Check whether the parent or one of the children
        satisfies the typ_checker.
        """
        if typ_checker(parent_typ):
            return True
        if pa.types.is_nested(parent_typ):
            for i in range(parent_typ.num_fields):
                if _contains_type(parent_typ.field(i).type, typ_checker):
                    return True
        return False

    if not check_field_nullability:
        rhs_type = _make_fields_nullable(rhs.type)
        rhs = rhs.cast(rhs_type)

        lhs_type = _make_fields_nullable(lhs.type)
        lhs = lhs.cast(lhs_type)

    assert lhs.type == rhs.type, f"{lhs.type} != {rhs.type}"
    if _contains_type(lhs.type, pa.types.is_floating) and _contains_type(
        rhs.type, pa.types.is_floating
    ):
        # Flatten nested arrays to liststo do comparisons if nested
        # This is so we can do approximate comparisons
        # for floats in numpy
        def _flatten_arrays(arr):
            if pa.types.is_nested(arr.type):
                flattened = arr.flatten()
                flat_arrs = []
                if isinstance(flattened, list):
                    for flat_arr in flattened:
                        flat_arrs += _flatten_arrays(flat_arr)
                else:
                    flat_arrs = [flattened]
            else:
                flat_arrs = [arr]
            return flat_arrs

        if isinstance(lhs, (pa.ListArray, pa.StructArray)):
            lhs = _flatten_arrays(lhs)
            rhs = _flatten_arrays(rhs)
        else:
            # Just a regular doublearray
            lhs = [lhs]
            rhs = [rhs]

        def _is_supported_for_pc_is_nan(arr_type):
            if pa.types.is_floating(arr_type):
                return False
            elif pa.types.is_list(arr_type):
                return _is_supported_for_pc_is_nan(arr_type.value_type)
            return True

        for lh_arr, rh_arr in zip(lhs, rhs, strict=True):
            # pc.is_nan does not support nested list
            # with float (eg. list<list<float>>)
            if not _is_supported_for_pc_is_nan(lh_arr.type):
                continue

            # Check NaNs positions match
            # and then filter out nans
            lhs_nans = pc.is_nan(lh_arr)
            rhs_nans = pc.is_nan(rh_arr)
            if not lhs_nans.equals(rhs_nans):
                raise_array_not_equal(lhs_nans, rhs_nans)

            if pc.any(lhs_nans) or pc.any(rhs_nans):
                # masks must be equal at this point
                mask = pc.fill_null(pc.invert(lhs_nans), True)
                lh_arr = lh_arr.filter(mask)
                rh_arr = rh_arr.filter(mask)

            np.testing.assert_array_almost_equal(lh_arr, rh_arr)
    else:
        if not lhs.equals(rhs):
            raise_array_not_equal(lhs, rhs)


def raise_array_not_equal(lhs: pa.Array, rhs: pa.Array) -> None:
    try:
        left = lhs.to_numpy()
        right = rhs.to_numpy()
        is_ndarray = True
    except pa.ArrowInvalid:
        left = lhs.to_pylist()
        right = rhs.to_pylist()
        is_ndarray = False

    if is_ndarray:
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right

    # If we get here, they're not equal according to lhs.equals,
    # but *are* equal according to the numpy/python assertion machinery.
    # So we'll raise an AssertionError with a nice error message.
    raise AssertionError(f"Arrays are not equal. {left} != {right}")


def assert_table_eq(pa_table: pa.Table, plc_table: plc.Table) -> None:
    """Verify that a pylibcudf table and PyArrow table are equal."""
    assert plc_table.shape() == pa_table.shape

    for plc_col, pa_col in zip(
        plc_table.columns(), pa_table.columns, strict=True
    ):
        assert_column_eq(pa_col, plc_col)


def assert_table_and_meta_eq(
    pa_table: pa.Table,
    plc_table_w_meta: plc.io.types.TableWithMetadata,
    check_field_nullability=True,
    check_types_if_empty=True,
    check_names=True,
) -> None:
    """Verify that the pylibcudf TableWithMetadata and PyArrow table are equal"""

    plc_table = plc_table_w_meta.tbl

    assert plc_table.shape() == pa_table.shape, (
        f"{plc_table.shape()} is not equal to {pa_table.shape}"
    )

    if not check_types_if_empty and plc_table.num_rows() == 0:
        return

    for plc_col, pa_col in zip(
        plc_table.columns(), pa_table.columns, strict=True
    ):
        assert_column_eq(pa_col, plc_col, check_field_nullability)

    # Check column name equality
    if check_names:
        assert plc_table_w_meta.column_names() == pa_table.column_names, (
            f"{plc_table_w_meta.column_names()} != {pa_table.column_names}"
        )


def cudf_raises(expected_exception: BaseException, *args, **kwargs):
    # A simple wrapper around pytest.raises that defaults to looking for cudf exceptions
    match = kwargs.get("match", None)
    if match is None:
        kwargs["match"] = "CUDF failure at"
    return pytest.raises(expected_exception, *args, **kwargs)


def is_string(plc_dtype: plc.DataType):
    return plc_dtype.id() == plc.TypeId.STRING


def nesting_level(typ) -> tuple[int, int]:
    """Return list and struct nesting of a pyarrow type."""
    if isinstance(typ, pa.ListType):
        list_, struct = nesting_level(typ.value_type)
        return list_ + 1, struct
    elif isinstance(typ, pa.StructType):
        lists, structs = map(
            max, zip(*(nesting_level(t.type) for t in typ), strict=True)
        )
        return lists, structs + 1
    else:
        return 0, 0


def is_nested_struct(typ):
    return nesting_level(typ)[1] > 1


def is_nested_list(typ):
    return nesting_level(typ)[0] > 1


def _convert_types(pa_table, input_pred, result_type):
    """
    Useful little helper for testing the dtypes option in I/O readers.

    Returns a tuple containing the pylibcudf dtypes and the new pyarrow schema based on
    the data in the table.

    Parameters
    ----------
    pa_table : pyarrow.Table
        The table from which to extract the dtypes
    input_pred : function
        Predicate that evaluates to true for types to replace
    result_type : pa.DataType
        The type to cast to
    """
    dtypes = []
    new_fields = []
    for i in range(len(pa_table.schema)):
        field = pa_table.schema.field(i)
        child_types = []

        plc_type = plc.DataType.from_arrow(field.type)
        if input_pred(field.type):
            plc_type = plc.DataType.from_arrow(result_type)
            field = field.with_type(result_type)

        dtypes.append((field.name, plc_type, child_types))

        new_fields.append(field)
    return dtypes, new_fields


def write_source_str(source, input_str):
    """
    Write a string to the source
    (useful for testing CSV/JSON I/O)
    """
    if not isinstance(source, io.IOBase):
        with open(source, "w") as source_f:
            source_f.write(input_str)
    else:
        if isinstance(source, io.BytesIO):
            input_str = input_str.encode("utf-8")
        source.write(input_str)
        source.seek(0)


def sink_to_str(sink):
    """
    Takes a sink (e.g. StringIO/BytesIO, filepath, etc.)
    and reads in the contents into a string (str not bytes)
    for comparison
    """
    if isinstance(sink, (str, os.PathLike)):
        with open(sink, "r", encoding="utf-8") as f:
            str_result = f.read()
    elif isinstance(sink, io.BytesIO):
        sink.seek(0)
        str_result = sink.read().decode("utf-8")
    else:
        sink.seek(0)
        str_result = sink.read()
    return str_result


def make_source(path_or_buf, pa_table, format, **kwargs):
    """
    Write a pyarrow Table to a specific format using pandas
    by dispatching to the appropriate to_* call.
    The caller is responsible for making sure that no arguments
    unsupported by pandas are passed in.
    """
    df = pa_table.to_pandas()
    mode = "w"
    if "compression" in kwargs:
        kwargs["compression"] = COMPRESSION_TYPE_TO_PANDAS[
            kwargs["compression"]
        ]
        if kwargs["compression"] is not None and format != "json":
            # pandas json method only supports mode="w"/"a"
            mode = "wb"
    if format == "json":
        df.to_json(path_or_buf, mode=mode, **kwargs)
    elif format == "csv":
        df.to_csv(path_or_buf, mode=mode, **kwargs)
    elif format == "parquet":
        # The conversion to pandas is lossy (doesn't preserve
        # nested types) so we
        # will just use pyarrow directly to write this
        pq_write_table(
            pa_table,
            pa.PythonFile(path_or_buf)
            if isinstance(path_or_buf, io.IOBase)
            else path_or_buf,
        )
    elif format == "orc":
        # The conversion to pandas is lossy (doesn't preserve
        # nested types) so we
        # will just use pyarrow directly to write this
        orc_write_table(
            pa_table,
            pa.PythonFile(path_or_buf)
            if isinstance(path_or_buf, io.IOBase)
            else path_or_buf,
        )
    if isinstance(path_or_buf, io.IOBase):
        path_or_buf.seek(0)
    return path_or_buf


def get_bytes_from_source(source):
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            return f.read()
    elif isinstance(source, io.BytesIO):
        return source.getbuffer()
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")


NUMERIC_PA_TYPES = [pa.int64(), pa.float64(), pa.uint64()]
STRING_PA_TYPES = [pa.string()]
BOOL_PA_TYPES = [pa.bool_()]
LIST_PA_TYPES = [
    pa.list_(pa.int64()),
    # Nested case
    pa.list_(pa.list_(pa.int64())),
]

# We must explicitly specify this type via a field to ensure we don't include
# nullability accidentally.
DEFAULT_STRUCT_TESTING_TYPE = pa.struct(
    [pa.field("v", pa.int64(), nullable=False)]
)
NESTED_STRUCT_TESTING_TYPE = pa.struct(
    [
        pa.field("a", pa.int64(), nullable=False),
        pa.field(
            "b_struct",
            pa.struct([pa.field("b", pa.float64(), nullable=False)]),
            nullable=False,
        ),
    ]
)

DEFAULT_PA_STRUCT_TESTING_TYPES = [
    DEFAULT_STRUCT_TESTING_TYPE,
    NESTED_STRUCT_TESTING_TYPE,
]

NON_NESTED_PA_TYPES = NUMERIC_PA_TYPES + STRING_PA_TYPES + BOOL_PA_TYPES

DEFAULT_PA_TYPES = (
    NON_NESTED_PA_TYPES + LIST_PA_TYPES + DEFAULT_PA_STRUCT_TESTING_TYPES
)

# Map pylibcudf compression types to pandas ones
# Not all compression types map cleanly, read the comments to learn more!
# If a compression type is unsupported, it maps to False.

COMPRESSION_TYPE_TO_PANDAS = {
    CompressionType.NONE: None,
    # Users of this dict will have to special case
    # AUTO
    CompressionType.AUTO: None,
    CompressionType.GZIP: "gzip",
    CompressionType.BZIP2: "bz2",
    CompressionType.ZIP: "zip",
    CompressionType.XZ: "xz",
    CompressionType.ZSTD: "zstd",
    # Unsupported
    CompressionType.ZLIB: False,
    CompressionType.LZ4: False,
    CompressionType.LZO: False,
    # These only work for parquet
    CompressionType.SNAPPY: "snappy",
    CompressionType.BROTLI: "brotli",
}
ALL_PA_TYPES = DEFAULT_PA_TYPES
