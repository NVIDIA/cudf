# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
import io
import pathlib
import subprocess
import sys
import textwrap

import fastavro
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing.dataset_generator import rand_dataframe


def cudf_from_avro_util(schema: dict, records: list) -> cudf.DataFrame:
    schema = [] if schema is None else fastavro.parse_schema(schema)
    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records)
    buffer.seek(0)
    return cudf.read_avro(buffer)


@pytest.fixture(
    params=[
        ("boolean", "bool"),
        ("int", "int32"),
        ("long", "int64"),
        ("float", "float32"),
        ("double", "float64"),
        ("bytes", "str"),
        ("string", "str"),
    ]
)
def avro_type_params(request):
    return request.param


@pytest.fixture(params=[True, False])
def nullable(request):
    return request.param


@pytest.fixture(params=[True, False])
def prepend_null(request):
    return request.param


@pytest.mark.parametrize("namespace", [None, "root_ns"])
def test_can_detect_dtype_from_avro_type(
    avro_type_params, namespace, nullable
):
    avro_type, expected_dtype = avro_type_params
    avro_type = avro_type if not nullable else ["null", avro_type]

    schema = fastavro.parse_schema(
        {
            "type": "record",
            "name": "test",
            "namespace": namespace,
            "fields": [{"name": "prop", "type": avro_type}],
        }
    )

    actual = cudf_from_avro_util(schema, [])

    expected = cudf.DataFrame(
        {"prop": cudf.Series(None, None, expected_dtype)}
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize("namespace", [None, "root_ns"])
def test_can_detect_dtype_from_avro_type_nested(
    avro_type_params, namespace, nullable
):
    avro_type, expected_dtype = avro_type_params
    avro_type = avro_type if not nullable else ["null", avro_type]

    schema_leaf = {
        "name": "leaf",
        "type": "record",
        "fields": [{"name": "prop3", "type": avro_type}],
    }

    schema_child = {
        "name": "child",
        "type": "record",
        "fields": [{"name": "prop2", "type": schema_leaf}],
    }

    schema_root = {
        "name": "root",
        "type": "record",
        "namespace": namespace,
        "fields": [{"name": "prop1", "type": schema_child}],
    }

    actual = cudf_from_avro_util(schema_root, [])

    col_name = "{ns}child.{ns}leaf.prop3".format(
        ns="" if namespace is None else namespace + "."
    )

    expected = cudf.DataFrame(
        {col_name: cudf.Series(None, None, expected_dtype)}
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "avro_type, cudf_type, avro_val, cudf_val",
    [
        ("boolean", "bool", True, True),
        ("boolean", "bool", False, False),
        ("int", "int32", 1234, 1234),
        ("long", "int64", 1234, 1234),
        ("float", "float32", 12.34, 12.34),
        ("double", "float64", 12.34, 12.34),
        ("string", "str", "heyϴ", "heyϴ"),
    ],
)
def test_can_parse_single_value(avro_type, cudf_type, avro_val, cudf_val):
    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [{"name": "prop", "type": ["null", avro_type]}],
    }

    records = [
        {"prop": avro_val},
    ]

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame(
        {"prop": cudf.Series(data=[cudf_val], dtype=cudf_type)}
    )

    assert_eq(expected, actual)


def test_can_parse_single_null(avro_type_params):
    avro_type, expected_dtype = avro_type_params
    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [{"name": "prop", "type": ["null", avro_type]}],
    }

    records = [{"prop": None}]

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame(
        {"prop": cudf.Series(data=[None], dtype=expected_dtype)}
    )

    assert_eq(expected, actual)


def test_can_parse_no_data(avro_type_params):
    avro_type, expected_dtype = avro_type_params
    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [{"name": "prop", "type": ["null", avro_type]}],
    }

    records = []

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame(
        {"prop": cudf.Series(data=[], dtype=expected_dtype)}
    )

    assert_eq(expected, actual)


@pytest.mark.xfail(
    reason="cudf avro reader is unable to parse zero-field metadata."
)
def test_can_parse_no_fields(avro_type_params):
    _avro_type, _expected_dtype = avro_type_params
    schema_root = {
        "name": "root",
        "type": "record",
        "fields": [],
    }

    records = []

    actual = cudf_from_avro_util(schema_root, records)

    expected = cudf.DataFrame()

    assert_eq(expected, actual)


def test_can_parse_no_schema():
    schema_root = None
    records = []
    actual = cudf_from_avro_util(schema_root, records)
    expected = cudf.DataFrame()
    assert_eq(expected, actual)


@pytest.mark.parametrize("rows", [0, 20])
@pytest.mark.parametrize("codec", ["null", "deflate", "snappy"])
def test_avro_decompression(set_decomp_env_vars, rows, codec):
    schema = {
        "name": "root",
        "type": "record",
        "fields": [
            {"name": "0", "type": "int"},
            {"name": "1", "type": "string"},
        ],
    }

    # N.B. rand_dataframe() is brutally slow for some reason.  Switching to
    #      np.random() speeds things up by a factor of 10.
    #      See also: https://github.com/rapidsai/cudf/issues/13128
    df = rand_dataframe(
        [
            {"dtype": "int32", "null_frequency": 0, "cardinality": 1000},
            {
                "dtype": "str",
                "null_frequency": 0,
                "cardinality": 100,
                "max_string_length": 10,
            },
        ],
        rows,
        seed=0,
        use_threads=False,
    )
    expected_df = cudf.DataFrame.from_arrow(df)

    records = df.to_pandas().to_dict(orient="records")

    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records, codec=codec)
    buffer.seek(0)
    got_df = cudf.read_avro(buffer)

    assert_eq(expected_df, got_df)


@pytest.mark.parametrize("namespace", [None, "root_ns"])
def test_can_detect_dtypes_from_avro_logical_type(
    namespace,
    nullable,
    prepend_null,
):
    logical_type = "date"
    primitive_type = "int"
    expected_dtype = "datetime64[s]"
    avro_type = [{"logicalType": logical_type, "type": primitive_type}]
    if nullable:
        if prepend_null:
            avro_type.insert(0, "null")
        else:
            avro_type.append("null")

    schema = fastavro.parse_schema(
        {
            "type": "record",
            "name": "test",
            "namespace": namespace,
            "fields": [{"name": "prop", "type": avro_type}],
        }
    )

    actual = cudf_from_avro_util(schema, [])

    expected = cudf.DataFrame(
        {"prop": cudf.Series(None, None, expected_dtype)}
    )

    assert_eq(expected, actual)


def get_days_from_epoch(date: datetime.date | None) -> int | None:
    if date is None:
        return None
    return (date - datetime.date(1970, 1, 1)).days


@pytest.mark.parametrize("namespace", [None, "root_ns"])
def test_can_parse_avro_date_logical_type(namespace, nullable, prepend_null):
    avro_type = {"logicalType": "date", "type": "int"}
    if nullable:
        if prepend_null:
            avro_type = ["null", avro_type]
        else:
            avro_type = [avro_type, "null"]

    schema_dict = {
        "type": "record",
        "name": "test",
        "fields": [
            {"name": "o_date", "type": avro_type},
        ],
    }

    if namespace:
        schema_dict["namespace"] = namespace

    schema = fastavro.parse_schema(schema_dict)

    # Insert some None values in no particular order.  These will get converted
    # into avro "nulls" by the fastavro writer (or filtered out if we're not
    # nullable).  The first and last dates are epoch min/max values, the rest
    # are arbitrarily chosen.
    dates = [
        None,
        datetime.date(1970, 1, 1),
        datetime.date(1970, 1, 2),
        datetime.date(1981, 10, 25),
        None,
        None,
        datetime.date(2012, 5, 18),
        None,
        datetime.date(2019, 9, 3),
        None,
        datetime.date(9999, 12, 31),
    ]

    if not nullable:
        dates = [date for date in dates if date is not None]

    days_from_epoch = [get_days_from_epoch(date) for date in dates]

    records = [{"o_date": day} for day in days_from_epoch]

    actual = cudf_from_avro_util(schema, records)

    expected = cudf.DataFrame(
        {"o_date": cudf.Series(dates, dtype="datetime64[s]")}
    )

    assert_eq(expected, actual)


def test_alltypes_plain_avro():
    # During development of the logical type support, the Java avro tests were
    # triggering CUDA kernel crashes (null pointer dereferences).  We were able
    # to replicate the behavior in a C++ test case, and then subsequently came
    # up with this Python unit test to also trigger the problematic code path.
    #
    # So, unlike the other tests, this test is inherently reactive in nature,
    # added simply to verify we fixed the problematic code path that was
    # causing CUDA kernel crashes.
    #
    # See https://github.com/rapidsai/cudf/pull/12788#issuecomment-1468822875
    # for more information.
    relpath = "../../../../java/src/test/resources/alltypes_plain.avro"
    path = pathlib.Path(__file__).parent.parent.joinpath(relpath).resolve()
    assert path.is_file(), path
    path = str(path)

    with open(path, "rb") as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]

    # For reference:
    #
    # >>> from pprint import pprint
    # >>> pprint(reader.writer_schema)
    # {'fields': [{'name': 'id', 'type': ['int', 'null']},
    #             {'name': 'bool_col', 'type': ['boolean', 'null']},
    #             {'name': 'tinyint_col', 'type': ['int', 'null']},
    #             {'name': 'smallint_col', 'type': ['int', 'null']},
    #             {'name': 'int_col', 'type': ['int', 'null']},
    #             {'name': 'bigint_col', 'type': ['long', 'null']},
    #             {'name': 'float_col', 'type': ['float', 'null']},
    #             {'name': 'double_col', 'type': ['double', 'null']},
    #             {'name': 'date_string_col', 'type': ['bytes', 'null']},
    #             {'name': 'string_col', 'type': ['bytes', 'null']},
    #             {'name': 'timestamp_col',
    #              'type': [{'logicalType': 'timestamp-micros',
    #                        'type': 'long'},
    #                       'null']}],
    #  'name': 'topLevelRecord',
    #  'type': 'record'}
    #
    # >>> pprint(records[0])
    # {'bigint_col': 0,
    #  'bool_col': True,
    #  'date_string_col': b'03/01/09',
    #  'double_col': 0.0,
    #  'float_col': 0.0,
    #  'id': 4,
    #  'int_col': 0,
    #  'smallint_col': 0,
    #  'string_col': b'0',
    #  'timestamp_col': datetime.datetime(2009, 3, 1, 0, 0,
    #                                     tzinfo=datetime.timezone.utc),
    #  'tinyint_col': 0}

    # Nothing particularly special about these columns, other than them being
    # the ones that @davidwendt used to coerce the crash.
    columns = ["bool_col", "int_col", "timestamp_col"]

    # This next line would trigger the fatal CUDA kernel crash.
    actual = cudf.read_avro(path, columns=columns)

    # If we get here, we haven't crashed, obviously.  Verify the returned data
    # frame meets our expectations.  We need to fiddle with the dtypes of the
    # expected data frame in order to correctly match the schema definition and
    # our corresponding read_avro()-returned data frame.

    data = [{column: row[column] for column in columns} for row in records]

    expected = pd.DataFrame(data)

    # The fastavro.reader supports the `'logicalType': 'timestamp-micros'` used
    # by the 'timestamp_col' column, which is converted into Python
    # datetime.datetime() objects (see output of pprint(records[0]) above).
    # cudf reads this column as int64 (raw microseconds) because the logical
    # type appears inside a union schema (['timestamp-micros', 'null']), so we
    # convert the expected datetime values to int64 microseconds to match.
    expected["timestamp_col"] = (
        expected["timestamp_col"]
        .dt.tz_convert(None)
        .astype("datetime64[us]")
        .astype("int64")
    )

    # Furthermore, we need to force the 'int_col' into an int32, per the schema
    # definition.  (It ends up as an int64 due to cudf.DataFrame() defaulting
    # all Python int values to int64 sans a dtype= override.)
    expected["int_col"] = expected["int_col"].astype("int32")

    assert_eq(actual, expected)


def multiblock_testname_ids(param):
    (total_rows, num_rows, skip_rows, sync_interval) = param
    return f"{total_rows=}-{num_rows=}-{skip_rows=}-{sync_interval=}"


# The following values are used to test various boundary conditions associated
# with multiblock avro files.  Each tuple consists of four values: total number
# of rows to generate, number of rows to limit the result set to, number of
# rows to skip, and number of rows per block.  If the total number of rows and
# number of rows (i.e. first and second tuple elements) are equal, it means
# that all rows will be returned.  If the rows per block also equals the first
# two numbers, it means that a single block will be used.
@pytest.fixture(
    ids=multiblock_testname_ids,
    params=[
        (10, 10, 9, 9),
        (10, 10, 9, 5),
        (10, 10, 9, 3),
        (10, 10, 9, 2),
        (10, 10, 9, 10),
        (10, 10, 8, 2),
        (10, 10, 5, 5),
        (10, 10, 2, 9),
        (10, 10, 2, 2),
        (10, 10, 1, 9),
        (10, 10, 1, 5),
        (10, 10, 1, 2),
        (10, 10, 1, 10),
        (10, 10, 10, 9),
        (10, 10, 10, 5),
        (10, 10, 10, 2),
        (10, 10, 10, 10),
        (10, 10, 0, 9),
        (10, 10, 0, 5),
        (10, 10, 0, 2),
        (10, 10, 0, 10),
        (100, 100, 99, 10),
        (100, 100, 90, 90),
        (100, 100, 90, 89),
        (100, 100, 90, 88),
        (100, 100, 90, 87),
        (100, 100, 90, 5),
        (100, 100, 89, 90),
        (100, 100, 87, 90),
        (100, 100, 50, 7),
        (100, 100, 50, 31),
        (10, 1, 8, 9),
        (100, 1, 99, 10),
        (100, 1, 98, 10),
        (100, 1, 97, 10),
        (100, 3, 90, 87),
        (100, 4, 90, 5),
        (100, 2, 89, 90),
        (100, 9, 87, 90),
        (100, 20, 50, 7),
        (100, 10, 50, 31),
        (100, 20, 50, 31),
        (100, 30, 50, 31),
        (256, 256, 0, 256),
        (256, 256, 0, 32),
        (256, 256, 0, 31),
        (256, 256, 0, 33),
        (256, 256, 31, 32),
        (256, 256, 32, 31),
        (256, 256, 31, 33),
        (512, 512, 0, 32),
        (512, 512, 0, 31),
        (512, 512, 0, 33),
        (512, 512, 31, 32),
        (512, 512, 32, 31),
        (512, 512, 31, 33),
        (1024, 1024, 0, 1),
        (1024, 1024, 0, 3),
        (1024, 1024, 0, 7),
        (1024, 1024, 0, 8),
        (1024, 1024, 0, 9),
        (1024, 1024, 0, 15),
        (1024, 1024, 0, 16),
        (1024, 1024, 0, 17),
        (1024, 1024, 0, 32),
        (1024, 1024, 0, 31),
        (1024, 1024, 0, 33),
        (1024, 1024, 31, 32),
        (1024, 1024, 32, 31),
        (1024, 1024, 31, 33),
        (2048, 2048, 0, 31),
        (2048, 2048, 0, 32),
        (2048, 2048, 0, 33),
        (2048, 2048, 0, 2048),
    ],
)
def total_rows_and_num_rows_and_skip_rows_and_rows_per_block(request):
    return request.param


# N.B. The float32 and float64 types are chosen specifically to exercise
#      the only path in the avro reader GPU code that can process multiple
#      rows in parallel (via warp-level parallelism).  See the logic around
#      the line `if (cur + min_row_size * rows_remaining == end)` in
#      gpuDecodeAvroColumnData().
@pytest.mark.parametrize("dtype", ["str", "float32", "float64"])
@pytest.mark.parametrize(
    "use_sync_interval",
    [True, False],
    ids=["use_sync_interval", "ignore_sync_interval"],
)
@pytest.mark.parametrize("codec", ["null", "deflate", "snappy"])
def test_avro_reader_multiblock(
    dtype,
    codec,
    use_sync_interval,
    total_rows_and_num_rows_and_skip_rows_and_rows_per_block,
):
    (
        total_rows,
        num_rows,
        skip_rows,
        rows_per_block,
    ) = total_rows_and_num_rows_and_skip_rows_and_rows_per_block

    assert total_rows >= num_rows
    assert rows_per_block <= total_rows

    limit_rows = num_rows != total_rows
    if limit_rows:
        assert total_rows >= num_rows + skip_rows

    if dtype == "str":
        avro_type = "string"

        # Generate a list of strings, each of which is a 6-digit number, padded
        # with leading zeros.  This data set was very useful during development
        # of the multiblock avro reader logic, as you get implicit feedback as
        # to what may have gone wrong when the test fails, based on the
        # expected vs actual values.
        values = [f"{i:0>6}" for i in range(0, total_rows)]

        # Strings are encoded in avro with a zigzag-encoded length prefix, and
        # then the string data.  As all of our strings are fixed at length 6,
        # we only need one byte to encode the length prefix (0xc).  Thus, our
        # bytes per row is 6 + 1 = 7.
        bytes_per_row = len(values[0]) + 1
        assert bytes_per_row == 7, bytes_per_row
    else:
        assert dtype in ("float32", "float64")
        avro_type = "float" if dtype == "float32" else "double"
        rng = np.random.default_rng(seed=0)
        # We don't use rand_dataframe() here, because it increases the
        # execution time of each test by a factor of 10 or more (it appears
        # to use a very costly approach to generating random data).
        # See also: https://github.com/rapidsai/cudf/issues/13128
        values = rng.random(total_rows).astype(dtype)
        bytes_per_row = values.dtype.itemsize

    # The sync_interval is the number of bytes between sync blocks.  We know
    # how many bytes we need per row, so we can calculate the number of bytes
    # per block by multiplying the number of rows per block by the bytes per
    # row.  This is the sync interval.
    total_bytes_per_block = rows_per_block * bytes_per_row
    sync_interval = total_bytes_per_block

    source_df = cudf.DataFrame({"0": pd.Series(values)})

    if limit_rows:
        expected_df = source_df[skip_rows : skip_rows + num_rows].reset_index(
            drop=True
        )
    else:
        expected_df = source_df[skip_rows:].reset_index(drop=True)

    records = source_df.to_pandas().to_dict(orient="records")

    schema = {
        "name": "root",
        "type": "record",
        "fields": [
            {"name": "0", "type": avro_type},
        ],
    }

    if use_sync_interval:
        kwds = {"sync_interval": sync_interval}
    else:
        kwds = {}

    kwds["codec"] = codec

    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records, **kwds)
    buffer.seek(0)

    if not limit_rows:
        # Explicitly set num_rows to None if we want to read all rows.  This
        # ensures we exercise the logic behind a read_avro() call where the
        # caller doesn't specify the number of rows desired (which will be the
        # most common use case).
        num_rows = None
    actual_df = cudf.read_avro(buffer, skiprows=skip_rows, num_rows=num_rows)

    assert_eq(expected_df, actual_df)


def test_avro_reader_no_hang_on_truncated_schema(datadir):
    path = datadir / "avro" / "hang_input.avro"
    assert path.is_file(), path

    script = textwrap.dedent(
        f"""
        import cudf
        try:
            cudf.read_avro({str(path)!r})
        except Exception:
            pass
        """
    )

    timeout_s = 10
    try:
        subprocess.run(
            [sys.executable, "-c", script],
            timeout=timeout_s,
            check=False,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"cudf.read_avro hung on malformed input {path.name!r} "
            f"(no completion within {timeout_s}s)"
        )
