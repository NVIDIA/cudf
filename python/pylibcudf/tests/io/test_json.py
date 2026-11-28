# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io

import pandas as pd
import pyarrow as pa
import pytest
from utils import (
    assert_table_and_meta_eq,
    make_source,
    sink_to_str,
    synchronize_stream,
    write_source_str,
)

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc
from pylibcudf.io.types import CompressionType

# Shared kwargs to pass to make_source
_COMMON_JSON_SOURCE_KWARGS = {"format": "json", "orient": "records"}


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("rows_per_chunk", [8, 100])
@pytest.mark.parametrize("lines", [True, False])
def test_write_json_basic(
    table_data, source_or_sink, lines, rows_per_chunk, stream
):
    plc_table_w_meta, pa_table = table_data
    sink = source_or_sink

    options = (
        plc.io.json.JsonWriterOptions.builder(
            plc.io.SinkInfo([sink]), plc_table_w_meta.tbl
        )
        .metadata(plc_table_w_meta)
        .lines(lines)
        .build()
    )

    options.set_rows_per_chunk(rows_per_chunk)

    plc.io.json.write_json(options, stream)

    synchronize_stream(stream)

    exp = pa_table.to_pandas()

    # Convert everything to string to make
    # comparisons easier
    str_result = sink_to_str(sink)

    pd_result = exp.to_json(orient="records", lines=lines)

    assert str_result == pd_result


@pytest.mark.parametrize("include_nulls", [True, False])
@pytest.mark.parametrize("na_rep", ["null", "awef", ""])
def test_write_json_nulls(na_rep, include_nulls):
    names = ["a", "b"]
    pa_tbl = pa.Table.from_arrays(
        [pa.array([1.0, 2.0, None]), pa.array([True, None, False])],
        names=names,
    )
    plc_tbl = plc.Table.from_arrow(pa_tbl)
    plc_tbl_w_meta = plc.io.types.TableWithMetadata(
        plc_tbl, column_names=[(name, []) for name in names]
    )

    sink = io.StringIO()

    options = (
        plc.io.json.JsonWriterOptions.builder(
            plc.io.SinkInfo([sink]), plc_tbl_w_meta.tbl
        )
        .metadata(plc_tbl_w_meta)
        .na_rep(na_rep)
        .include_nulls(include_nulls)
        .build()
    )

    plc.io.json.write_json(options)

    synchronize_stream()

    exp = pa_tbl.to_pandas()

    # Convert everything to string to make
    # comparisons easier
    str_result = sink_to_str(sink)
    pd_result = exp.to_json(orient="records")

    if not include_nulls:
        # No equivalent in pandas, so we just
        # sanity check by making sure na_rep
        # doesn't appear in the output

        # don't quote null
        for name in names:
            assert f'{{"{name}":{na_rep}}}' not in str_result
        return

    # pandas doesn't suppport na_rep
    # let's just manually do str.replace
    pd_result = pd_result.replace("null", na_rep)

    assert str_result == pd_result


@pytest.mark.parametrize("true_value", ["True", "correct"])
@pytest.mark.parametrize("false_value", ["False", "wrong"])
def test_write_json_bool_opts(true_value, false_value):
    names = ["a"]
    pa_tbl = pa.Table.from_arrays([pa.array([True, None, False])], names=names)
    plc_tbl = plc.Table.from_arrow(pa_tbl)
    plc_tbl_w_meta = plc.io.types.TableWithMetadata(
        plc_tbl, column_names=[(name, []) for name in names]
    )

    sink = io.StringIO()

    options = (
        plc.io.json.JsonWriterOptions.builder(
            plc.io.SinkInfo([sink]), plc_tbl_w_meta.tbl
        )
        .metadata(plc_tbl_w_meta)
        .na_rep("null")
        .include_nulls(True)
        .build()
    )

    options.set_true_value(true_value)
    options.set_false_value(false_value)

    plc.io.json.write_json(options)

    synchronize_stream()

    exp = pa_tbl.to_pandas()

    # Convert everything to string to make
    # comparisons easier
    str_result = sink_to_str(sink)
    pd_result = exp.to_json(orient="records")

    # pandas doesn't suppport na_rep
    # let's just manually do str.replace
    if true_value != "true":
        pd_result = pd_result.replace("true", true_value)
    if false_value != "false":
        pd_result = pd_result.replace("false", false_value)

    assert str_result == pd_result


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("lines", [True, False])
@pytest.mark.parametrize("source_strategy", ["inline", "set_source"])
def test_read_json_basic(
    table_data,
    source_or_sink,
    lines,
    text_compression_type,
    stream,
    source_strategy,
):
    compression_type = text_compression_type

    # can't compress non-binary data with pandas
    if isinstance(source_or_sink, io.StringIO):
        compression_type = CompressionType.NONE

    _, pa_table = table_data

    source = make_source(
        source_or_sink,
        pa_table,
        lines=lines,
        compression=compression_type,
        **_COMMON_JSON_SOURCE_KWARGS,
    )

    if isinstance(source, io.IOBase):
        source.seek(0)

    source_info = plc.io.SourceInfo([source])
    options = (
        plc.io.json.JsonReaderOptions.builder(
            source_info
            if source_strategy == "inline"
            else plc.io.SourceInfo([])
        )
        .compression(compression_type)
        .lines(lines)
        .build()
    )

    if source_strategy == "set_source":
        options.set_source(source_info)

    res = plc.io.json.read_json(options, stream)

    # Adjustments to correct for the fact orient=records is lossy
    #  and doesn't
    # 1) preserve colnames when zero rows in table
    # 2) preserve struct nullability
    # 3) differentiate int64/uint64
    if len(pa_table) == 0:
        pa_table = pa.table([])

    new_fields = []
    for i in range(len(pa_table.schema)):
        curr_field = pa_table.schema.field(i)
        if curr_field.type == pa.uint64():
            try:
                curr_field = curr_field.with_type(pa.int64())
            except OverflowError:
                # There will be no confusion, values are too large
                # for int64 anyways
                pass
        new_fields.append(curr_field)

    pa_table = pa_table.cast(pa.schema(new_fields))

    # Convert non-nullable struct fields to nullable fields
    # since nullable=False cannot roundtrip through orient='records'
    # JSON format
    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)


def test_read_json_dtypes(table_data, source_or_sink):
    # Simple test for dtypes where we read in
    # all numeric data as floats
    _, pa_table = table_data
    source = make_source(
        source_or_sink,
        pa_table,
        lines=True,
        **_COMMON_JSON_SOURCE_KWARGS,
    )

    dtypes = []
    new_fields = []
    for i in range(len(pa_table.schema)):
        field = pa_table.schema.field(i)
        child_types = []

        def get_child_types(typ):
            typ_child_types = []
            for i in range(typ.num_fields):
                curr_field = typ.field(i)
                typ_child_types.append(
                    (
                        curr_field.name,
                        curr_field.type,
                        get_child_types(curr_field.type),
                    )
                )
            return typ_child_types

        plc_type = plc.DataType.from_arrow(field.type)
        if pa.types.is_integer(field.type) or pa.types.is_unsigned_integer(
            field.type
        ):
            plc_type = plc.DataType.from_arrow(pa.float64())
            field = field.with_type(pa.float64())

        dtypes.append((field.name, plc_type, child_types))

        new_fields.append(field)

    new_schema = pa.schema(new_fields)

    options = (
        plc.io.json.JsonReaderOptions.builder(plc.io.SourceInfo([source]))
        .lines(True)
        .build()
    )
    options.set_dtypes(dtypes)

    res = plc.io.json.read_json(options)
    new_table = pa_table.cast(new_schema)

    # orient=records is lossy
    # and doesn't preserve column names when there's zero rows in the table
    if len(new_table) == 0:
        new_table = pa.table([])

    assert_table_and_meta_eq(new_table, res, check_field_nullability=False)


@pytest.mark.parametrize("chunk_size", [10, 15, 20])
def test_read_json_lines_byte_range(source_or_sink, chunk_size):
    source = source_or_sink
    if isinstance(source_or_sink, io.StringIO):
        pytest.skip("byte_range doesn't work on StringIO")

    json_str = "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]\n"
    write_source_str(source, json_str)

    tbls_w_meta = []
    for chunk_start in range(0, len(json_str.encode("utf-8")), chunk_size):
        tbls_w_meta.append(
            plc.io.json.read_json(
                (
                    plc.io.json.JsonReaderOptions.builder(
                        plc.io.SourceInfo([source])
                    )
                    .lines(True)
                    .byte_range_offset(chunk_start)
                    .byte_range_size(chunk_start + chunk_size)
                    .build()
                )
            )
        )

    if isinstance(source, io.IOBase):
        source.seek(0)
    exp = pd.read_json(source, orient="records", lines=True)

    # TODO: can do this operation using pylibcudf
    tbls = []
    for tbl_w_meta in tbls_w_meta:
        if tbl_w_meta.tbl.num_rows() > 0:
            tbls.append(tbl_w_meta.tbl.to_arrow())
    full_tbl = pa.concat_tables(tbls)

    full_tbl_plc = plc.io.TableWithMetadata(
        plc.Table.from_arrow(full_tbl),
        tbls_w_meta[0].column_names(include_children=True),
    )
    assert_table_and_meta_eq(pa.Table.from_pandas(exp), full_tbl_plc)


@pytest.mark.parametrize("keep_quotes", [True, False])
def test_read_json_lines_keep_quotes(keep_quotes, source_or_sink):
    source = source_or_sink

    json_bytes = '["a", "b", "c"]\n'
    write_source_str(source, json_bytes)

    tbl_w_meta = plc.io.json.read_json(
        (
            plc.io.json.JsonReaderOptions.builder(plc.io.SourceInfo([source]))
            .lines(True)
            .keep_quotes(keep_quotes)
            .build()
        )
    )

    template = "{0}"
    if keep_quotes:
        template = '"{0}"'

    exp = pa.Table.from_arrays(
        [
            [template.format("a")],
            [template.format("b")],
            [template.format("c")],
        ],
        names=["0", "1", "2"],
    )

    assert_table_and_meta_eq(exp, tbl_w_meta)


@pytest.mark.parametrize(
    "recovery_mode", [opt for opt in plc.io.types.JSONRecoveryMode]
)
def test_read_json_lines_recovery_mode(recovery_mode, source_or_sink):
    source = source_or_sink

    json_str = '{"a":1,"b":10}\n{"a":2,"b":11}\nabc\n{"a":3,"b":12}\n'
    write_source_str(source, json_str)

    options = (
        plc.io.json.JsonReaderOptions.builder(plc.io.SourceInfo([source]))
        .lines(True)
        .recovery_mode(recovery_mode)
        .build()
    )

    if recovery_mode == plc.io.types.JSONRecoveryMode.FAIL:
        with pytest.raises(RuntimeError):
            plc.io.json.read_json(options)
    else:
        # Recover case (bad values replaced with nulls)
        tbl_w_meta = plc.io.json.read_json(options)
        exp = pa.Table.from_arrays(
            [[1, 2, None, 3], [10, 11, None, 12]], names=["a", "b"]
        )
        assert_table_and_meta_eq(exp, tbl_w_meta)


@pytest.mark.parametrize("trailing_null", [False, True])
def test_read_json_string_column(trailing_null):
    # Test the string column overload specifically
    # Create a string column with JSON strings
    json_strings = [
        '{"name": "Alice", "age": 30}',
        '{"name": "Bob", "age": 25}',
        '{"name": "Charlie", "age": 35}',
    ]

    # An unused separator after the only valid row must not reach the JSON reader.
    if trailing_null:
        json_strings = [*json_strings[:1], None]
    string_col = plc.Column.from_arrow(pa.array(json_strings))

    # Create separator and narep scalars
    separator = plc.Scalar.from_py("invalid JSON" if trailing_null else "\n")
    narep = (
        plc.Scalar.from_py(None, plc.DataType(plc.TypeId.STRING))
        if trailing_null
        else plc.Scalar.from_py("null")
    )

    # Read JSON from string column
    got = plc.io.json.read_json_from_string_column(
        string_col, separator, narep
    )

    # Expected result
    expect = pa.Table.from_arrays(
        [["Alice", "Bob", "Charlie"], [30, 25, 35]], names=["name", "age"]
    )

    if trailing_null:
        expect = expect.slice(0, 1)
    assert_table_and_meta_eq(expect, got)


@pytest.mark.parametrize("num_buffers", [1, 2])
@pytest.mark.parametrize("stream", [None, Stream()])
def test_read_json_from_device_buffers(table_data, num_buffers, stream):
    _, pa_table = table_data

    json_str = pa_table.to_pandas().to_json(orient="records", lines=True)
    buf = DeviceBuffer.to_device(
        json_str.encode("utf-8"), plc.utils._get_stream(stream)
    )

    synchronize_stream(stream)

    options = (
        plc.io.json.JsonReaderOptions.builder(
            plc.io.SourceInfo([buf] * num_buffers)
        )
        .lines(True)
        .build()
    )
    result = plc.io.json.read_json(options, stream)

    expected = (
        pa_table
        if num_buffers == 1
        else pa.concat_tables([pa_table] * num_buffers)
    )

    if len(expected) == 0:
        expected = pa.table([])
    else:
        new_fields = []
        for field in expected.schema:
            if field.type == pa.uint64():
                field = field.with_type(pa.int64())
            new_fields.append(field)
        expected = expected.cast(pa.schema(new_fields))

    assert_table_and_meta_eq(expected, result, check_field_nullability=False)


def test_utf8_escaped_json_writer(tmp_path):
    arrow_table = pa.table({"a": ["C𝞵𝓓𝒻"]})
    plc_table = plc.Table.from_arrow(arrow_table)

    path = tmp_path / "out.json"

    options = (
        plc.io.json.JsonWriterOptions.builder(
            plc.io.SinkInfo([path]), plc_table
        )
        .utf8_escaped(False)
        .build()
    )
    plc.io.json.write_json(options)

    synchronize_stream()

    output_string = path.read_text(encoding="utf-8").strip()

    assert output_string == '[{"0":"C𝞵𝓓𝒻"}]'


# TODO: Add tests for these!
# Tests were not added in the initial PR porting the JSON reader to pylibcudf
# to save time (and since there are no existing tests for these in Python cuDF)
# mixed_types_as_string = mixed_types_as_string,
# prune_columns = prune_columns,
