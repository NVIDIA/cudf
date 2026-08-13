# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import decimal
import glob
import hashlib
import math
import os
import pathlib
import string
from contextlib import contextmanager
from io import BytesIO
from string import ascii_letters

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.core import get_fs_token_paths
from packaging import version
from pyarrow import parquet as pq

import cudf
from cudf.io.parquet import (
    ParquetDatasetWriter,
    ParquetWriter,
    is_supported_read_parquet,
    is_supported_write_parquet,
    merge_parquet_filemetadata,
)
from cudf.testing import (
    assert_arrow_table_equal,
    assert_eq,
    dataset_generator as dg,
)
from cudf.testing._utils import TIMEDELTA_TYPES, set_random_null_mask_inplace


@contextmanager
def _hide_pyarrow_parquet_cpu_warnings(engine):
    if engine == "pyarrow":
        with pytest.warns(
            UserWarning,
            match="Using CPU via PyArrow to read Parquet dataset. This option "
            "is both inefficient and unstable!",
        ):
            yield
    else:
        yield


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "parquet"


@pytest.fixture
def simple_pdf():
    nrows = 10
    rng = np.random.default_rng(seed=0)
    types = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        # "uint32", pandas promotes uint32 to int64
        # https://issues.apache.org/jira/browse/ARROW-9215
        "uint64",
        "float32",
        "float64",
    ]
    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.DataFrame(
        {
            f"col_{typ}": rng.integers(0, nrows, nrows).astype(typ)
            for typ in types
        },
        # Need to ensure that this index is not a RangeIndex to get the
        # expected round-tripping behavior from Parquet reader/writer.
        index=pd.Index(list(range(nrows))),
    )
    # Delete the name of the column index, and rename the row index
    test_pdf.columns.name = None
    test_pdf.index.name = "test_index"

    return test_pdf


@pytest.fixture
def simple_gdf(simple_pdf):
    return cudf.DataFrame(simple_pdf)


def build_pdf(num_columns, day_resolution_timestamps):
    rng = np.random.default_rng(seed=0)
    types = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        # "uint32", pandas promotes uint32 to int64
        # https://issues.apache.org/jira/browse/ARROW-9215
        "uint64",
        "float32",
        "float64",
        "datetime64[ms]",
        "datetime64[us]",
    ]
    nrows = num_columns

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.DataFrame(
        {
            f"col_{typ}": rng.integers(0, nrows, nrows).astype(typ)
            for typ in types
        },
        # Need to ensure that this index is not a RangeIndex to get the
        # expected round-tripping behavior from Parquet reader/writer.
        index=pd.Index(list(range(nrows))),
    )
    # Delete the name of the column index, and rename the row index
    test_pdf.columns.name = None
    test_pdf.index.name = "test_index"

    # make datetime64's a little more interesting by increasing the range of
    # dates note that pandas will convert these to ns timestamps, so care is
    # taken to avoid overflowing a ns timestamp. There is also the ability to
    # request timestamps be whole days only via `day_resolution_timestamps`.
    for t in [
        {
            "name": "datetime64[ms]",
            "nsDivisor": 1000000,
            "dayModulus": 86400000,
        },
        {
            "name": "datetime64[us]",
            "nsDivisor": 1000,
            "dayModulus": 86400000000,
        },
    ]:
        data = [
            rng.integers(0, (0x7FFFFFFFFFFFFFFF / t["nsDivisor"]))
            for i in range(nrows)
        ]
        if day_resolution_timestamps:
            data = [int(d / t["dayModulus"]) * t["dayModulus"] for d in data]
        test_pdf["col_" + t["name"]] = pd.Series(
            np.asarray(data, dtype=t["name"])
        )

    # Create non-numeric categorical data otherwise parquet may typecast it
    data = [ascii_letters[rng.integers(0, 52)] for i in range(nrows)]
    test_pdf["col_category"] = pd.Series(
        data,
        dtype=pd.CategoricalDtype(
            categories=pd.Index(
                list(dict.fromkeys(data)),
                dtype=pd.StringDtype(na_value=np.nan),
            )
        ),
    )

    # Create non-numeric str data
    data = [ascii_letters[rng.integers(0, 52)] for i in range(nrows)]
    test_pdf["col_str"] = pd.Series(
        data, dtype=pd.StringDtype(na_value=np.nan)
    )

    return test_pdf


@pytest.fixture(params=[0, 10])
def pdf(request):
    return build_pdf(request.param, False)


@pytest.fixture(params=[0, 10])
def pdf_day_timestamps(request):
    return build_pdf(request.param, True)


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame(pdf)


@pytest.fixture
def gdf_day_timestamps(pdf_day_timestamps):
    return cudf.DataFrame(pdf_day_timestamps)


@pytest.fixture
def parquet_path_or_buf(datadir):
    fname = datadir / "spark_timestamp.snappy.parquet"
    try:
        with open(fname, "rb") as f:
            buffer = BytesIO(f.read())
    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".parquet file is not found")
        raise excpr

    def _make_parquet_path_or_buf(src):
        if src == "filepath":
            return str(fname)
        if src == "pathobj":
            return fname
        if src == "bytes_io":
            return buffer
        if src == "bytes":
            return buffer.getvalue()
        if src == "url":
            return fname.as_uri()

        raise ValueError("Invalid source type")

    yield _make_parquet_path_or_buf


@pytest.fixture(scope="module")
def large_int64_gdf():
    return cudf.DataFrame(pd.DataFrame({"col": range(0, 1 << 20)}))


@pytest.fixture(params=["pyarrow", "cudf"])
def engine(request):
    return request.param


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize(
    "compression", ["snappy", "gzip", "brotli", None, np.str_("snappy")]
)
@pytest.mark.parametrize(
    "columns",
    [
        ["col_int8"],
        ["col_category"],
        ["col_int32", "col_float32"],
        ["col_int16", "col_float64", "col_int8"],
        None,
    ],
)
def test_parquet_reader_basic(tmp_path, pdf, columns, engine, compression):
    parquet_file = tmp_path / f"{compression}_test.parquet"
    pdf.to_parquet(parquet_file, engine="pyarrow", compression=compression)
    expect = pd.read_parquet(parquet_file, columns=columns)
    got = cudf.read_parquet(parquet_file, engine=engine, columns=columns)

    # pandas returns category objects whereas cuDF returns hashes
    if engine == "cudf":
        if "col_category" in expect.columns:
            expect = expect.drop(columns=["col_category"])
        if "col_category" in got.columns:
            got = got.drop(columns=["col_category"])

    # As of pandas 3.0, empty default type of object isn't
    # necessarily equivalent to cuDF's empty default type of
    # pandas.StringDtype
    if len(pdf.index) == 0:
        expect.index = expect.index.astype(got.index.dtype)
    expect.columns = expect.columns.astype(got.columns.dtype)

    if "col_category" in expect.columns:
        # pyarrow read_parquet reader returns category[object] instead of category[pd.StringDtype]
        casted_categories = expect["col_category"].dtype.categories.astype(
            got["col_category"].dtype.categories.dtype
        )
        expect["col_category"] = expect["col_category"].cat.set_categories(
            casted_categories
        )
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_parquet_reader_empty_pandas_dataframe(tmp_path):
    df = pd.DataFrame()
    fname = tmp_path / "test_pq_reader_empty_pandas_dataframe.parquet"
    df.to_parquet(fname)
    assert os.path.exists(fname)
    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname, engine="cudf")
    expect = expect.reset_index(drop=True)
    got = got.reset_index(drop=True)

    assert_eq(expect, got)


@pytest.mark.parametrize("has_null", [False, True])
def test_parquet_reader_strings(tmp_path, has_null):
    df = pd.DataFrame(
        [(1, "aaa", 9.0), (2, "bbb", 8.0), (3, "ccc", 7.0)],
        columns=pd.Index(list("abc")),
    )
    if has_null:
        df.at[1, "b"] = None
    fname = tmp_path / "test_pq_reader_strings.parquet"
    df.to_parquet(fname)
    assert os.path.exists(fname)

    gdf = cudf.read_parquet(fname, engine="cudf")
    assert_eq(gdf["b"], df["b"])


@pytest.mark.parametrize("columns", [None, ["b"]])
@pytest.mark.parametrize("index_col", ["b", "Nameless", None])
def test_parquet_reader_index_col(tmp_path, index_col, columns):
    df = pd.DataFrame({"a": range(3), "b": range(3, 6), "c": range(6, 9)})

    if index_col is None:
        # No index column
        df.reset_index(drop=True, inplace=True)
    elif index_col == "Nameless":
        # Index column but no name
        df.set_index("a", inplace=True)
        df.index.name = None
    else:
        # Index column as normal
        df.set_index(index_col, inplace=True)

    fname = tmp_path / "test_pq_reader_index_col.parquet"

    # PANDAS' PyArrow backend always writes the index unless disabled
    df.to_parquet(fname, index=(index_col is not None))
    assert os.path.exists(fname)

    pdf = pd.read_parquet(fname, columns=columns)
    gdf = cudf.read_parquet(fname, engine="cudf", columns=columns)
    # pyarrow read_parquet reader returns Index[object] instead of Index[pd.StringDtype]
    pdf.columns = pdf.columns.astype(gdf.columns.dtype)

    assert_eq(pdf, gdf, check_categorical=False)


@pytest.mark.parametrize("pandas_compat", [True, False])
@pytest.mark.parametrize("columns", [["a"], ["d"], ["a", "d"], None])
def test_parquet_reader_pandas_metadata(tmp_path, columns, pandas_compat):
    df = pd.DataFrame(
        {
            "a": range(6, 9),
            "b": range(3, 6),
            "c": range(6, 9),
            "d": ["abc", "def", "xyz"],
        }
    )
    df.set_index("b", inplace=True)

    fname = tmp_path / "test_pq_reader_pandas_metadata.parquet"
    df.to_parquet(fname)
    assert os.path.exists(fname)

    # PANDAS `read_parquet()` and PyArrow `read_pandas()` always includes index
    # Instead, directly use PyArrow to optionally omit the index
    expect = pa.parquet.read_table(
        fname, columns=columns, use_pandas_metadata=pandas_compat
    ).to_pandas()
    got = cudf.read_parquet(
        fname, columns=columns, use_pandas_metadata=pandas_compat
    )
    assert got.index.name == "b"
    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize("pandas_compat", [True, False])
@pytest.mark.parametrize("as_bytes", [True, False])
def test_parquet_range_index_pandas_metadata(
    tmp_path, pandas_compat, as_bytes
):
    df = pd.DataFrame(
        {"a": range(6, 9), "b": ["abc", "def", "xyz"]},
        index=pd.RangeIndex(3, 6, 1, name="c"),
    )

    fname = tmp_path / "test_parquet_range_index_pandas_metadata.parquet"
    df.to_parquet(fname)
    assert os.path.exists(fname)

    # PANDAS `read_parquet()` and PyArrow `read_pandas()` always includes index
    # Instead, directly use PyArrow to optionally omit the index
    expect = pa.parquet.read_table(
        fname, use_pandas_metadata=pandas_compat
    ).to_pandas()
    if as_bytes:
        # Make sure we can handle RangeIndex parsing
        # in pandas when the input is `bytes`
        with open(fname, "rb") as f:
            got = cudf.read_parquet(
                f.read(), use_pandas_metadata=pandas_compat
            )
    else:
        got = cudf.read_parquet(fname, use_pandas_metadata=pandas_compat)

    assert_eq(expect, got)


def test_parquet_read_metadata(tmp_path, pdf):
    def num_row_groups(rows, group_size):
        return max(1, (rows + (group_size - 1)) // group_size)

    fname = tmp_path / "metadata.parquet"
    row_group_size = 5
    pdf.to_parquet(fname, compression="snappy", row_group_size=row_group_size)

    (
        num_rows,
        row_groups,
        col_names,
        num_columns,
        _,  # rowgroup_metadata
    ) = cudf.io.read_parquet_metadata(fname)

    assert num_columns == len(pdf.columns)
    assert num_rows == len(pdf.index)
    assert row_groups == num_row_groups(num_rows, row_group_size)
    for a, b in zip(col_names, pdf.columns, strict=True):
        assert a == b


def test_parquet_read_filtered(set_decomp_env_vars, tmp_path):
    # Generate data
    fname = tmp_path / "filtered.parquet"
    rng = np.random.default_rng(0)
    dg.generate(
        fname,
        dg.Parameters(
            num_rows=100,
            column_parameters=[
                dg.ColumnParameters(
                    cardinality=40,
                    null_frequency=0.05,
                    generator=lambda: [
                        "".join(rng.choice(list(string.ascii_letters), 5))
                        for _ in range(10)
                    ],
                    is_sorted=False,
                ),
                dg.ColumnParameters(
                    40,
                    0.2,
                    lambda: rng.integers(0, 100, size=10),
                    True,
                ),
            ],
            seed=42,
        ),
        format={"name": "parquet", "row_group_size": 10},
        use_threads=False,
    )

    # Get dataframes to compare
    df = cudf.read_parquet(fname)
    df_filtered = cudf.read_parquet(fname, filters=[("1", ">", 60)])
    # PyArrow's read_table function does row-group-level filtering in addition
    # to applying given filters once the table has been read into memory.
    # Because of this, we aren't using PyArrow as a reference for testing our
    # row-group selection method since the only way to only select row groups
    # with PyArrow is with the method we use and intend to test.
    tbl_filtered = pq.read_table(fname, filters=[("1", ">", 60)])

    assert_eq(cudf.io.read_parquet_metadata(fname)[1], 10)
    assert len(df_filtered) < len(df)
    assert len(tbl_filtered) <= len(df_filtered)


def test_parquet_read_filtered_everything(tmp_path):
    # Generate data
    fname = tmp_path / "filtered_everything.parquet"
    df = pd.DataFrame({"x": range(10), "y": list("aabbccddee")})
    df.to_parquet(fname, row_group_size=2)

    # Check filter
    df_filtered = cudf.read_parquet(fname, filters=[("x", "==", 12)])
    assert len(df_filtered) == 0
    assert df_filtered["x"].dtype == "int64"
    assert df_filtered["y"].dtype == pd.StringDtype(na_value=np.nan)


def test_parquet_read_filtered_multiple_files(tmp_path):
    # Generate data
    fname_0 = tmp_path / "filtered_multiple_files_0.parquet"
    df = pd.DataFrame({"x": range(10), "y": list("aabbccddee")})
    df.to_parquet(fname_0, row_group_size=2)
    fname_1 = tmp_path / "filtered_multiple_files_1.parquet"
    df = pd.DataFrame({"x": range(10), "y": list("aaccccddee")})
    df.to_parquet(fname_1, row_group_size=2)
    fname_2 = tmp_path / "filtered_multiple_files_2.parquet"
    df = pd.DataFrame(
        {"x": [0, 1, 9, 9, 4, 5, 6, 7, 8, 9], "y": list("aabbzzddee")}
    )
    df.to_parquet(fname_2, row_group_size=2)

    # Check filter
    filtered_df = cudf.read_parquet(
        [fname_0, fname_1, fname_2], filters=[("x", "==", 2)]
    )
    assert_eq(
        filtered_df,
        cudf.DataFrame({"x": [2, 2], "y": list("bc")}, index=[2, 2]),
    )


@pytest.mark.parametrize(
    "predicate,expected_len",
    [
        ([[("x", "==", 0)], [("z", "==", 0)]], 2),
        ([("x", "==", 0), ("z", "==", 0)], 0),
        ([("x", "==", 0), ("z", "!=", 0)], 1),
        ([("y", "==", "c"), ("x", ">", 8)], 0),
        ([("y", "==", "c"), ("x", ">=", 5)], 1),
        ([[("y", "==", "c")], [("x", "<", 3)]], 5),
        ([[("x", "not in", (0, 9)), ("z", "not in", (4, 5))]], 6),
        ([[("y", "==", "c")], [("x", "in", (0, 9)), ("z", "in", (0, 9))]], 4),
        ([[("x", "==", 0)], [("x", "==", 1)], [("x", "==", 2)]], 3),
        ([[("x", "==", 0), ("z", "==", 9), ("y", "==", "a")]], 1),
    ],
)
def test_parquet_read_filtered_complex_predicate(
    tmp_path, predicate, expected_len
):
    # Generate data
    fname = tmp_path / "filtered_complex_predicate.parquet"
    df = pd.DataFrame(
        {
            "x": range(10),
            "y": list("aabbccddee"),
            "z": reversed(range(10)),
        }
    )
    df.to_parquet(fname, row_group_size=2)

    # Check filters
    df_filtered = cudf.read_parquet(fname, filters=predicate)
    assert_eq(cudf.io.read_parquet_metadata(fname)[1], 10 / 2)
    assert_eq(len(df_filtered), expected_len)


@pytest.mark.parametrize("row_group_size", [1, 5])
def test_parquet_read_row_groups(tmp_path, pdf, row_group_size):
    if "col_category" in pdf.columns:
        pdf = pdf.drop(columns=["col_category"])
    fname = tmp_path / "row_group.parquet"
    pdf.to_parquet(fname, compression="gzip", row_group_size=row_group_size)

    _num_rows, row_groups, _col_names, _, _ = cudf.io.read_parquet_metadata(
        fname
    )

    gdf = [cudf.read_parquet(fname, row_groups=[i]) for i in range(row_groups)]
    gdf = cudf.concat(gdf)
    assert_eq(pdf.reset_index(drop=True), gdf.reset_index(drop=True))

    # first half rows come from the first source, rest from the second
    gdf = cudf.read_parquet(
        [fname, fname],
        row_groups=[
            list(range(row_groups // 2)),
            list(range(row_groups // 2, row_groups)),
        ],
    )
    assert_eq(pdf.reset_index(drop=True), gdf.reset_index(drop=True))


@pytest.mark.parametrize("row_group_size", [1, 5])
def test_parquet_read_row_groups_non_contiguous(tmp_path, pdf, row_group_size):
    fname = tmp_path / "row_group.parquet"
    pdf.to_parquet(fname, compression="gzip", row_group_size=row_group_size)

    _num_rows, row_groups, _col_names, _, _ = cudf.io.read_parquet_metadata(
        fname
    )

    # alternate rows between the two sources
    gdf = cudf.read_parquet(
        [fname, fname],
        row_groups=[
            list(range(0, row_groups, 2)),
            list(range(1, row_groups, 2)),
        ],
    )

    ref_df = [
        cudf.read_parquet(fname, row_groups=i)
        for i in list(range(0, row_groups, 2)) + list(range(1, row_groups, 2))
    ]
    ref_df = cudf.concat(ref_df)

    assert_eq(ref_df, gdf)


def test_parquet_reader_spark_timestamps(datadir):
    fname = datadir / "spark_timestamp.snappy.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_spark_decimals(datadir):
    fname = datadir / "spark_decimal.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


@pytest.mark.parametrize("columns", [["a"], ["b", "a"], None])
def test_parquet_reader_decimal128(datadir, columns):
    fname = datadir / "nested_decimal128_file.parquet"
    got = cudf.read_parquet(fname, columns=columns)
    expect = cudf.read_parquet(fname, columns=columns)

    assert_eq(expect, got)


def test_parquet_reader_microsecond_timestamps(datadir):
    fname = datadir / "usec_timestamp.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_mixedcompression(datadir):
    fname = datadir / "mixed_compression.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_select_columns(datadir):
    fname = datadir / "nested_column_map.parquet"

    expect = cudf.read_parquet(fname).to_pandas()[["value"]]
    got = cudf.read_parquet(fname, columns=["value"])

    assert_eq(expect, got)


@pytest.mark.parametrize("ignore_missing_columns", [True, False])
def test_parquet_reader_select_nonexistent_columns(ignore_missing_columns):
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "c": ["a", "b", "c", "d", "e", "f"],
        }
    )
    buf = BytesIO()
    df.to_parquet(buf)

    if ignore_missing_columns:
        expect = cudf.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        got = cudf.read_parquet(
            buf,
            columns=["a", "d"],
            ignore_missing_columns=ignore_missing_columns,
        )
        assert_eq(expect, got)
    else:
        with pytest.raises(
            ValueError,
            match=r"Encountered non-existent column '[^']+' in the selected path",
        ):
            cudf.read_parquet(
                buf,
                columns=["a", "d"],
                ignore_missing_columns=ignore_missing_columns,
            )


def test_parquet_reader_invalids(tmp_path):
    test_pdf = cudf.DataFrame(
        {"a": np.array([1, np.nan, np.nan, 2])}, index=cudf.Index([0, 1, 2, 3])
    )

    fname = tmp_path / "invalids.parquet"
    test_pdf.to_parquet(fname, engine="pyarrow")

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got.to_pandas(nullable=True))


def test_parquet_reader_filenotfound(tmpdir):
    with pytest.raises(FileNotFoundError):
        cudf.read_parquet("TestMissingFile.parquet")

    with pytest.raises(FileNotFoundError):
        cudf.read_parquet(tmpdir.mkdir("cudf_parquet"))


def test_parquet_reader_local_filepath():
    fname = "~/TestLocalFile.parquet"
    if not os.path.isfile(fname):
        pytest.skip("Local .parquet file is not found")

    cudf.read_parquet(fname)


@pytest.mark.parametrize(
    "src", ["filepath", "pathobj", "bytes_io", "bytes", "url"]
)
def test_parquet_reader_filepath_or_buffer(parquet_path_or_buf, src):
    expect = pd.read_parquet(parquet_path_or_buf("filepath"))
    got = cudf.read_parquet(parquet_path_or_buf(src))

    assert_eq(expect, got)


def test_parquet_reader_file_types(parquet_path_or_buf):
    expect = cudf.read_parquet(parquet_path_or_buf("filepath"))
    fs, _, paths = get_fs_token_paths(parquet_path_or_buf("filepath"))

    # Pass open fsspec file
    with fs.open(paths[0], mode="rb") as fil:
        got1 = cudf.read_parquet(fil)
    assert_eq(expect, got1)

    # Pass path only
    got2 = cudf.read_parquet(paths[0])
    assert_eq(expect, got2)


def create_parquet_source(df, src_type, fname):
    if src_type == "filepath":
        df.to_parquet(fname, engine="pyarrow")
        return str(fname)
    if src_type == "pathobj":
        df.to_parquet(fname, engine="pyarrow")
        return fname
    if src_type == "bytes_io":
        buffer = BytesIO()
        df.to_parquet(buffer, engine="pyarrow")
        return buffer
    if src_type == "bytes":
        buffer = BytesIO()
        df.to_parquet(buffer, engine="pyarrow")
        return buffer.getvalue()
    if src_type == "url":
        df.to_parquet(fname, engine="pyarrow")
        return pathlib.Path(fname).as_uri()


@pytest.mark.parametrize(
    "src", ["filepath", "pathobj", "bytes_io", "bytes", "url"]
)
def test_parquet_reader_multiple_files(tmp_path, src):
    test_pdf1 = pd.DataFrame(
        {
            "a": np.concatenate(
                [np.arange(10, dtype="float64"), np.full(10, np.nan)]
            )
        },
        index=pd.Index(np.arange(20)),
    )
    test_pdf2 = pd.DataFrame(
        {"a": np.arange(10, dtype="float64")}, index=pd.Index(np.arange(10))
    )
    expect = pd.concat([test_pdf1, test_pdf2])

    src1 = create_parquet_source(test_pdf1, src, tmp_path / "multi1.parquet")
    src2 = create_parquet_source(test_pdf2, src, tmp_path / "multi2.parquet")
    got = cudf.read_parquet([src1, src2])

    assert_eq(expect, got)


@pytest.mark.parametrize("empty_df_indices", [[0], [0, 1], [2, 3]])
def test_parquet_reader_multiple_files_some_empty(tmp_path, empty_df_indices):
    df = pd.DataFrame({"a": [1, 2, 3]})
    for idx in range(5):
        if idx in empty_df_indices:
            df = df.iloc[:0]
        df.to_parquet(tmp_path / f"df_{idx}.parquet")

    got = cudf.read_parquet([tmp_path])
    expected = pd.read_parquet(tmp_path)

    assert_eq(expected, got)


def test_parquet_reader_reordered_columns(tmp_path):
    src = pd.DataFrame(
        {"name": ["cow", None, "duck", "fish", None], "id": [0, 1, 2, 3, 4]}
    )
    fname = tmp_path / "test_parquet_reader_reordered_columns.parquet"
    src.to_parquet(fname)
    assert os.path.exists(fname)
    expect = pd.DataFrame(
        {"id": [0, 1, 2, 3, 4], "name": ["cow", None, "duck", "fish", None]}
    )
    got = cudf.read_parquet(fname, columns=["id", "name"])
    assert_eq(expect, got, check_dtype=False)


def test_parquet_reader_reordered_columns_mixed(tmp_path):
    src = pd.DataFrame(
        {
            "name": ["cow", None, "duck", "fish", None],
            "list0": [
                [[1, 2], [3, 4]],
                None,
                [[5, 6], None],
                [[1]],
                [[5], [6, None, 8]],
            ],
            "id": [0, 1, 2, 3, 4],
            "list1": [
                [[1, 2], [3, 4]],
                [[0, 0]],
                [[5, 6], [10, 12]],
                [[1]],
                [[5], [6, 8]],
            ],
        }
    )
    fname = tmp_path / "test_parquet_reader_reordered_columns.parquet"
    src.to_parquet(fname)
    assert os.path.exists(fname)
    expect = pd.DataFrame(
        {
            "list1": [
                [[1, 2], [3, 4]],
                [[0, 0]],
                [[5, 6], [10, 12]],
                [[1]],
                [[5], [6, 8]],
            ],
            "id": [0, 1, 2, 3, 4],
            "list0": [
                [[1, 2], [3, 4]],
                None,
                [[5, 6], None],
                [[1]],
                [[5], [6, None, 8]],
            ],
            "name": ["cow", None, "duck", "fish", None],
        }
    )
    got = cudf.read_parquet(fname, columns=["list1", "id", "list0", "name"])
    assert_eq(expect, got, check_dtype=False)


def test_parquet_reader_list_basic(tmp_path):
    expect = pd.DataFrame({"a": [[[1, 2], [3, 4]], None, [[5, 6], None]]})
    fname = tmp_path / "test_parquet_reader_list_basic.parquet"
    expect.to_parquet(fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_eq(expect, got)


def test_parquet_reader_list_table(tmp_path):
    expect = pd.DataFrame(
        {
            "a": [[[1, 2], [3, 4]], None, [[5, 6], None]],
            "b": [[None, None], None, [None, None]],
            "c": [[[1, 2, 3]], [[None]], [[], None]],
            "d": [[[]], [[None]], [[1, 2, 3], None]],
            "e": [[["cows"]], [["dogs"]], [["cats", "birds", "owls"], None]],
        }
    )
    fname = tmp_path / "test_parquet_reader_list_table.parquet"
    expect.to_parquet(fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_arrow_table_equal(pa.Table.from_pandas(expect), got.to_arrow())


def int_gen(first_val, i):
    """
    Returns an integer based on an absolute index and a starting value. Used
    as input to `list_gen`.
    """
    return int(i + first_val)


strings = [
    "cats",
    "dogs",
    "cows",
    "birds",
    "fish",
    "sheep",
    "owls",
    "bears",
    "ants",
]


def string_gen(first_val, i):
    """
    Returns a string based on an absolute index and a starting value. Used as
    input to `list_gen`.
    """
    return strings[int_gen(first_val, i) % len(strings)]


def list_row_gen(
    gen, first_val, list_size, lists_per_row, include_validity=False
):
    """
    Generate a single row for a List<List<>> column based on input parameters.

    Parameters
    ----------
    gen : A callable which generates an individual leaf element based on an
        absolute index.
    first_val : Generate the column as if it had started at 'first_val'
        instead of 0.
    list_size : Size of each generated list.
    lists_per_row : Number of lists to generate per row.
    include_validity : Whether or not to include nulls as part of the
        column. If true, it will add a selection of nulls at both the
        topmost row level and at the leaf level.

    Returns
    -------
    The generated list column.
    """

    def L(list_size, first_val):
        return [
            (gen(first_val, i) if i % 2 == 0 else None)
            if include_validity
            else (gen(first_val, i))
            for i in range(list_size)
        ]

    return [
        (L(list_size, first_val + (list_size * i)) if i % 2 == 0 else None)
        if include_validity
        else L(list_size, first_val + (list_size * i))
        for i in range(lists_per_row)
    ]


def list_gen(gen, num_rows, lists_per_row, list_size, include_validity=False):
    """
    Generate a list column based on input parameters.

    Parameters
    ----------
    gen : A callable which generates an individual leaf element based on an
        absolute index.
    num_rows : Number of rows to generate.
    lists_per_row : Number of lists to generate per row.
    list_size : Size of each generated list.
    include_validity : Whether or not to include nulls as part of the
        column. If true, it will add a selection of nulls at both the
        topmost row level and at the leaf level.

    Returns
    -------
    The generated list column.
    """

    def L(list_size, first_val):
        return [
            (gen(first_val, i) if i % 2 == 0 else None)
            if include_validity
            else (gen(first_val, i))
            for i in range(list_size)
        ]

    def R(first_val, lists_per_row, list_size):
        return [
            L(list_size, first_val + (list_size * i))
            for i in range(lists_per_row)
        ]

    return [
        (
            R(
                lists_per_row * list_size * i,
                lists_per_row,
                list_size,
            )
            if i % 2 == 0
            else None
        )
        if include_validity
        else R(
            lists_per_row * list_size * i,
            lists_per_row,
            list_size,
        )
        for i in range(num_rows)
    ]


def test_parquet_reader_list_large(tmp_path):
    expect = pd.DataFrame({"a": list_gen(int_gen, 64, 40, 25)})
    fname = tmp_path / "test_parquet_reader_list_large.parquet"
    expect.to_parquet(fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_eq(expect, got, check_dtype=False)


def test_parquet_reader_list_validity(tmp_path):
    expect = pd.DataFrame(
        {"a": list_gen(int_gen, 64, 40, 25, include_validity=True)}
    )
    fname = tmp_path / "test_parquet_reader_list_validity.parquet"
    expect.to_parquet(fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_eq(expect, got, check_dtype=False)


def test_parquet_reader_list_large_mixed(tmp_path):
    expect = pd.DataFrame(
        {
            "a": list_gen(string_gen, 64, 40, 25),
            "b": list_gen(int_gen, 64, 40, 25),
            "c": list_gen(int_gen, 64, 40, 25, include_validity=True),
            "d": list_gen(string_gen, 64, 40, 25, include_validity=True),
        }
    )
    fname = tmp_path / "test_parquet_reader_list_large_mixed.parquet"
    expect.to_parquet(fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_arrow_table_equal(pa.Table.from_pandas(expect), got.to_arrow())


def test_parquet_reader_list_large_multi_rowgroup(tmp_path):
    # > 40 row groups
    num_rows = 10000
    num_docs = num_rows / 2
    num_categories = 100
    row_group_size = 100

    rng = np.random.default_rng(0)

    # generate a random pairing of doc: category
    documents = cudf.DataFrame(
        {
            "document_id": rng.integers(num_docs, size=num_rows),
            "category_id": rng.integers(num_categories, size=num_rows),
        }
    )

    # group categories by document_id to create a list column
    expect = documents.groupby("document_id").agg({"category_id": ["collect"]})
    expect.columns = expect.columns.get_level_values(0)
    expect.reset_index(inplace=True)

    # round trip the dataframe to/from parquet
    fname = tmp_path / "test_parquet_reader_list_large_multi_rowgroup.parquet"
    expect.to_pandas().to_parquet(fname, row_group_size=row_group_size)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_list_large_multi_rowgroup_nulls(tmp_path):
    # 25 row groups
    num_rows = 2500
    row_group_size = 100

    expect = cudf.DataFrame(
        {"a": list_gen(int_gen, num_rows, 3, 2, include_validity=True)}
    )

    # round trip the dataframe to/from parquet
    fname = (
        tmp_path
        / "test_parquet_reader_list_large_multi_rowgroup_nulls.parquet"
    )
    expect.to_pandas().to_parquet(fname, row_group_size=row_group_size)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_eq(expect, got)


def struct_gen(gen, skip_rows, num_rows, include_validity=False):
    """
    Generate a struct column based on input parameters.

    Parameters
    ----------
    gen : A array of callables which generate an individual row based on an
        absolute index.
    skip_rows : Generate the column as if it had started at 'skip_rows'
        instead of 0. The intent here is to emulate the skip_rows
        parameter of the parquet reader.
    num_fields : Number of fields in the struct.
    include_validity : Whether or not to include nulls as part of the
        column. If true, it will add a selection of nulls at both the
        field level and at the value level.

    Returns
    -------
    The generated struct column.
    """

    def R(first_val, num_fields):
        return {
            "col" + str(f): (
                gen[f](first_val, first_val) if f % 4 != 0 else None
            )
            if include_validity
            else (gen[f](first_val, first_val))
            for f in range(len(gen))
        }

    return [
        (R((i + skip_rows), len(gen)) if (i + skip_rows) % 4 != 0 else None)
        if include_validity
        else R((i + skip_rows), len(gen))
        for i in range(num_rows)
    ]


@pytest.mark.parametrize(
    "data",
    [
        # struct
        [
            {"a": 1, "b": 2},
            {"a": 10, "b": 20},
            {"a": None, "b": 22},
            {"a": None, "b": None},
            {"a": 15, "b": None},
        ],
        # struct-of-list
        [
            {"a": 1, "b": 2, "c": [1, 2, 3]},
            {"a": 10, "b": 20, "c": [4, 5]},
            {"a": None, "b": 22, "c": [6]},
            {"a": None, "b": None, "c": None},
            {"a": 15, "b": None, "c": [-1, -2]},
            None,
            {"a": 100, "b": 200, "c": [-10, None, -20]},
        ],
        # list-of-struct
        [
            [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 4, "b": 5}],
            None,
            [{"a": 10, "b": 20}],
            [{"a": 100, "b": 200}, {"a": None, "b": 300}, None],
        ],
        # struct-of-struct
        [
            {"a": 1, "b": {"inner_a": 10, "inner_b": 20}, "c": 2},
            {"a": 3, "b": {"inner_a": 30, "inner_b": 40}, "c": 4},
            {"a": 5, "b": {"inner_a": 50, "inner_b": None}, "c": 6},
            {"a": 7, "b": None, "c": 8},
            {"a": None, "b": {"inner_a": None, "inner_b": None}, "c": None},
            None,
            {"a": None, "b": {"inner_a": None, "inner_b": 100}, "c": 10},
        ],
    ],
)
def test_parquet_reader_struct_basic(tmp_path, data):
    expect = pa.Table.from_pydict({"struct": data})
    fname = tmp_path / "test_parquet_reader_struct_basic.parquet"
    pa.parquet.write_table(expect, fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_arrow_table_equal(expect, got.to_arrow())


def select_columns_params():
    dfs = [
        # struct
        (
            [
                {"a": 1, "b": 2},
                {"a": 10, "b": 20},
                {"a": None, "b": 22},
                {"a": None, "b": None},
                {"a": 15, "b": None},
            ],
            [["struct"], ["struct.a"], ["struct.b"], ["c"]],
        ),
        # struct-of-list
        (
            [
                {"a": 1, "b": 2, "c": [1, 2, 3]},
                {"a": 10, "b": 20, "c": [4, 5]},
                {"a": None, "b": 22, "c": [6]},
                {"a": None, "b": None, "c": None},
                {"a": 15, "b": None, "c": [-1, -2]},
                None,
                {"a": 100, "b": 200, "c": [-10, None, -20]},
            ],
            [
                ["struct"],
                ["struct.c"],
                ["struct.c.list"],
                ["struct.c.list.item"],
                ["struct.b", "struct.c"],
                ["struct.b", "struct.d", "struct.c"],
            ],
        ),
        # list-of-struct
        (
            [
                [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 4, "b": 5}],
                None,
                [{"a": 10, "b": 20}],
                [{"a": 100, "b": 200}, {"a": None, "b": 300}, None],
            ],
            [
                ["struct"],
                ["struct.list"],
                ["struct.list.item"],
                ["struct.list.item.a", "struct.list.item.b"],
                ["struct.list.item.c"],
            ],
        ),
        # struct with "." in field names
        (
            [
                {"a.b": 1, "b.a": 2},
                {"a.b": 10, "b.a": 20},
                {"a.b": None, "b.a": 22},
                {"a.b": None, "b.a": None},
                {"a.b": 15, "b.a": None},
            ],
            [["struct"], ["struct.a"], ["struct.b.a"]],
        ),
    ]
    for df_col_pair in dfs:
        for cols in df_col_pair[1]:
            yield df_col_pair[0], cols


@pytest.mark.parametrize("data, columns", select_columns_params())
def test_parquet_reader_struct_select_columns(data, columns):
    table = pa.Table.from_pydict({"struct": data})
    buff = BytesIO()

    pa.parquet.write_table(table, buff)

    expect = pq.ParquetFile(buff).read(columns=columns)
    got = cudf.read_parquet(buff, columns=columns)
    assert_arrow_table_equal(expect, got.to_arrow())


def nonexistent_select_columns_params():
    dfs = [
        # columns that don't exist
        (
            [
                {"a": 1, "b": 2},
                {"a": 10, "b": 20},
                {"a": None, "b": 22},
                {"a": None, "b": None},
                {"a": 15, "b": None},
            ],
            [["c"]],  # top-level missing
        ),
        (
            [
                {"a": 1, "b": 2, "c": [1, 2, 3]},
                {"a": 10, "b": 20, "c": [4, 5]},
                {"a": None, "b": 22, "c": [6]},
                {"a": None, "b": None, "c": None},
                {"a": 15, "b": None, "c": [-1, -2]},
                None,
                {"a": 100, "b": 200, "c": [-10, None, -20]},
            ],
            [
                ["struct.b", "struct.d", "struct.c"],  # 'struct.d' missing
            ],
        ),
        (
            [
                [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 4, "b": 5}],
                None,
                [{"a": 10, "b": 20}],
                [{"a": 100, "b": 200}, {"a": None, "b": 300}, None],
            ],
            [
                ["struct.list.element.c"],  # subfield 'c' missing
            ],
        ),
    ]

    for df_col_pair in dfs:
        for cols in df_col_pair[1]:
            yield df_col_pair[0], cols


@pytest.mark.parametrize("data, columns", nonexistent_select_columns_params())
def test_parquet_reader_struct_select_columns_nonexistent_error(data, columns):
    table = pa.Table.from_pydict({"struct": data})
    buff = BytesIO()
    pa.parquet.write_table(table, buff)

    with pytest.raises(
        ValueError,
        match=r"Encountered non-existent column '[^']+' in the selected path",
    ):
        cudf.read_parquet(buff, columns=columns, ignore_missing_columns=False)


def test_parquet_reader_struct_los_large(tmp_path):
    num_rows = 256
    list_size = 64
    data = [
        struct_gen([string_gen, int_gen, string_gen], 0, list_size, False)
        if i % 2 == 0
        else None
        for i in range(num_rows)
    ]
    expect = pa.Table.from_pydict({"los": data})
    fname = tmp_path / "test_parquet_reader_struct_los_large.parquet"
    pa.parquet.write_table(expect, fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_arrow_table_equal(expect, got.to_arrow())


@pytest.mark.parametrize(
    "params", [[3, 4, 32, False], [3, 4, 32, True], [50, 10, 64, True]]
)
def test_parquet_reader_struct_sol_table(tmp_path, params):
    # Struct<List<List>>
    lists_per_row = params[0]
    list_size = params[1]
    num_rows = params[2]
    include_validity = params[3]

    def list_gen_wrapped(x, y):
        return list_row_gen(
            int_gen, x * list_size * lists_per_row, list_size, lists_per_row
        )

    def string_list_gen_wrapped(x, y):
        return list_row_gen(
            string_gen,
            x * list_size * lists_per_row,
            list_size,
            lists_per_row,
            include_validity,
        )

    data = struct_gen(
        [int_gen, string_gen, list_gen_wrapped, string_list_gen_wrapped],
        0,
        num_rows,
        include_validity,
    )
    expect = pa.Table.from_pydict({"sol": data})
    fname = tmp_path / "test_parquet_reader_struct_sol_table.parquet"
    pa.parquet.write_table(expect, fname)
    assert os.path.exists(fname)
    got = cudf.read_parquet(fname)
    assert_arrow_table_equal(expect, got.to_arrow())


def test_parquet_reader_v2(tmp_path, simple_pdf):
    pdf_fname = tmp_path / "pdfv2.parquet"
    simple_pdf.to_parquet(pdf_fname, data_page_version="2.0")
    assert_eq(cudf.read_parquet(pdf_fname), simple_pdf)

    cudf.from_pandas(simple_pdf).to_parquet(pdf_fname, header_version="2.0")
    assert_eq(cudf.read_parquet(pdf_fname), simple_pdf)


def test_parquet_delta_byte_array(datadir):
    fname = datadir / "delta_byte_arr.parquet"
    assert_eq(cudf.read_parquet(fname), pd.read_parquet(fname))


# values chosen to exercise:
#    1 - header only, no bitpacked values
#    2 - one bitpacked value
#   23 - one partially filled miniblock
#   32 - almost full miniblock
#   33 - one full miniblock
#   34 - one full miniblock plus one value in new miniblock
#  128 - almost full block
#  129 - one full block
#  130 - one full block plus one value in new block
# 129 * 3 - multiple blocks
@pytest.fixture(params=[1, 2, 23, 32, 33, 34, 128, 129, 130, 129 * 3])
def delta_num_rows(request):
    return request.param


@pytest.mark.parametrize("add_nulls", [True, False])
def test_delta_binary(
    delta_num_rows, add_nulls, signed_integer_types_as_str, tmp_path
):
    null_frequency = 0.25 if add_nulls else 0

    # Create a pandas dataframe with random data of mixed types
    arrow_table = dg.rand_dataframe(
        dtypes_meta=[
            {
                "dtype": signed_integer_types_as_str,
                "null_frequency": null_frequency,
                "cardinality": delta_num_rows,
            },
        ],
        rows=delta_num_rows,
        seed=0,
        use_threads=False,
    )
    # Roundabout conversion to pandas to preserve nulls/data types
    cudf_table = cudf.DataFrame.from_arrow(arrow_table)
    test_pdf = cudf_table.to_pandas(nullable=True)
    pdf_fname = tmp_path / "pdfv2.parquet"
    test_pdf.to_parquet(
        pdf_fname,
        version="2.6",
        column_encoding="DELTA_BINARY_PACKED",
        data_page_version="2.0",
        data_page_size=64 * 1024,
        engine="pyarrow",
        use_dictionary=False,
    )
    cdf = cudf.read_parquet(pdf_fname)
    pcdf = cudf.from_pandas(test_pdf)
    # integer columns can be pandas Int64 vs numpy int64 from source
    assert_eq(cdf, pcdf, check_dtype=False)

    # Write back out with cudf and make sure pyarrow can read it
    cudf_fname = tmp_path / "cudfv2.parquet"
    pcdf.to_parquet(
        cudf_fname,
        compression=None,
        header_version="2.0",
        use_dictionary=False,
    )

    cdf2 = cudf.from_pandas(pd.read_parquet(cudf_fname))
    # integer columns can be pandas Int64 vs numpy int64 from source
    assert_eq(cdf2, cdf, check_dtype=False)


@pytest.mark.parametrize("add_nulls", [True, False])
@pytest.mark.parametrize("max_string_length", [12, 48, 96, 128])
@pytest.mark.parametrize(
    "str_encoding", ["DELTA_BYTE_ARRAY", "DELTA_LENGTH_BYTE_ARRAY"]
)
def test_delta_byte_array_roundtrip(
    delta_num_rows, add_nulls, max_string_length, str_encoding, tmp_path
):
    null_frequency = 0.25 if add_nulls else 0

    # Create a pandas dataframe with random data of mixed lengths
    test_pdf = dg.rand_dataframe(
        dtypes_meta=[
            {
                "dtype": "str",
                "null_frequency": null_frequency,
                "cardinality": delta_num_rows,
                "max_string_length": max_string_length,
            },
        ],
        rows=delta_num_rows,
        seed=0,
        use_threads=False,
    ).to_pandas()

    pdf_fname = tmp_path / "pdfdeltaba.parquet"
    test_pdf.to_parquet(
        pdf_fname,
        version="2.6",
        column_encoding=str_encoding,
        data_page_version="2.0",
        data_page_size=64 * 1024,
        engine="pyarrow",
        use_dictionary=False,
    )
    cdf = cudf.read_parquet(pdf_fname)
    pcdf = cudf.from_pandas(test_pdf)
    assert_eq(cdf, pcdf)

    # Write back out with cudf and make sure pyarrow can read it
    cudf_fname = tmp_path / "cdfdeltaba.parquet"
    pcdf.to_parquet(
        cudf_fname,
        compression="snappy",
        header_version="2.0",
        use_dictionary=False,
    )
    cdf2 = cudf.from_pandas(pd.read_parquet(cudf_fname))
    # string columns can result in StringDtype vs object with nulls
    assert_eq(cdf2, cdf, check_dtype=not add_nulls)


@pytest.mark.parametrize("add_nulls", [True, False])
@pytest.mark.parametrize(
    "str_encoding", ["DELTA_BYTE_ARRAY", "DELTA_LENGTH_BYTE_ARRAY"]
)
def test_delta_struct_list(tmp_path, delta_num_rows, add_nulls, str_encoding):
    # Struct<List<List>>
    lists_per_row = 3
    list_size = 4
    num_rows = delta_num_rows
    include_validity = add_nulls

    def list_gen_wrapped(x, y):
        return list_row_gen(
            int_gen, x * list_size * lists_per_row, list_size, lists_per_row
        )

    def string_list_gen_wrapped(x, y):
        return list_row_gen(
            string_gen,
            x * list_size * lists_per_row,
            list_size,
            lists_per_row,
            include_validity,
        )

    data = struct_gen(
        [int_gen, string_gen, list_gen_wrapped, string_list_gen_wrapped],
        0,
        num_rows,
        include_validity,
    )
    test_pdf = pa.Table.from_pydict({"sol": data}).to_pandas()
    pdf_fname = tmp_path / "pdfdeltaba.parquet"
    test_pdf.to_parquet(
        pdf_fname,
        version="2.6",
        column_encoding={
            "sol.col0": "DELTA_BINARY_PACKED",
            "sol.col1": str_encoding,
            "sol.col2.list.element.list.element": "DELTA_BINARY_PACKED",
            "sol.col3.list.element.list.element": str_encoding,
        },
        data_page_version="2.0",
        data_page_size=64 * 1024,
        engine="pyarrow",
        use_dictionary=False,
    )
    # sanity check to verify file is written properly
    assert_eq(test_pdf, pd.read_parquet(pdf_fname))
    cdf = cudf.read_parquet(pdf_fname)
    pcdf = cudf.from_pandas(test_pdf)
    assert_eq(cdf, pcdf)

    # Write back out with cudf and make sure pyarrow can read it
    cudf_fname = tmp_path / "cdfdeltaba.parquet"
    pcdf.to_parquet(
        cudf_fname,
        compression="snappy",
        header_version="2.0",
        use_dictionary=False,
    )
    cdf2 = cudf.from_pandas(pd.read_parquet(cudf_fname))
    # string columns can result in StringDtype vs object with nulls
    if delta_num_rows == 1:
        assert_eq(cdf2, cdf.to_pandas(arrow_type=True), check_dtype=False)
    else:
        assert_eq(cdf2, cdf, check_dtype=not add_nulls)


@pytest.mark.parametrize(
    "data",
    [
        # Structs
        {
            "being": [
                None,
                {"human?": True, "Deets": {"Name": "Carrot", "Age": 27}},
                {"human?": None, "Deets": {"Name": "Angua", "Age": 25}},
                {"human?": False, "Deets": {"Name": "Cheery", "Age": 31}},
                {"human?": False, "Deets": None},
                {"human?": None, "Deets": {"Name": "Mr", "Age": None}},
            ]
        },
        # List of Structs
        {
            "family": [
                [None, {"human?": True, "deets": {"weight": 2.4, "age": 27}}],
                [
                    {"human?": None, "deets": {"weight": 5.3, "age": 25}},
                    {"human?": False, "deets": {"weight": 8.0, "age": 31}},
                    {"human?": False, "deets": None},
                ],
                [],
                [{"human?": None, "deets": {"weight": 6.9, "age": None}}],
            ]
        },
        # Struct of Lists
        {
            "Real estate records": [
                None,
                {
                    "Status": "NRI",
                    "Ownerships": {
                        "land_unit": [None, 2, None],
                        "flats": [[1, 2, 3], [], [4, 5], [], [0, 6, 0]],
                    },
                },
                {
                    "Status": None,
                    "Ownerships": {
                        "land_unit": [4, 5],
                        "flats": [[7, 8], []],
                    },
                },
                {
                    "Status": "RI",
                    "Ownerships": {"land_unit": None, "flats": [[]]},
                },
                {"Status": "RI", "Ownerships": None},
                {
                    "Status": None,
                    "Ownerships": {
                        "land_unit": [7, 8, 9],
                        "flats": [[], [], []],
                    },
                },
            ]
        },
    ],
)
def test_parquet_reader_nested_v2(tmp_path, data):
    expect = pd.DataFrame(data)
    pdf_fname = tmp_path / "pdfv2.parquet"
    expect.to_parquet(pdf_fname, data_page_version="2.0")
    assert_eq(cudf.read_parquet(pdf_fname), expect)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_parquet_writer_cpu_pyarrow(
    tmp_path, pdf_day_timestamps, gdf_day_timestamps
):
    pdf_fname = tmp_path / "pdf.parquet"
    gdf_fname = tmp_path / "gdf.parquet"

    if len(pdf_day_timestamps) == 0:
        pdf_day_timestamps = pdf_day_timestamps.reset_index(drop=True)
        gdf_day_timestamps = pdf_day_timestamps.reset_index(drop=True)

    pdf_day_timestamps.to_parquet(pdf_fname)
    gdf_day_timestamps.to_parquet(gdf_fname, engine="pyarrow")

    assert os.path.exists(pdf_fname)
    assert os.path.exists(gdf_fname)

    expect = pa.parquet.read_pandas(pdf_fname)
    got = pa.parquet.read_pandas(gdf_fname)

    assert_eq(expect, got)

    def clone_field(table, name, datatype):
        f = table.schema.field(name)
        return pa.field(f.name, datatype, f.nullable, f.metadata)

    # Pandas uses a datetime64[ns] while we use a datetime64[ms]
    for t in [expect, got]:
        for t_col in ["col_datetime64[ms]", "col_datetime64[us]"]:
            idx = t.schema.get_field_index(t_col)
            field = clone_field(t, t_col, pa.timestamp("ms"))
            t = t.set_column(idx, field, t.column(idx).cast(field.type))
            t = t.replace_schema_metadata()

    assert_eq(expect, got)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_parquet_writer_int96_timestamps(tmp_path, pdf, gdf):
    gdf_fname = tmp_path / "gdf.parquet"

    if len(pdf) == 0:
        pdf = pdf.reset_index(drop=True)
        gdf = gdf.reset_index(drop=True)

    if "col_category" in pdf.columns:
        pdf = pdf.drop(columns=["col_category"])
    if "col_category" in gdf.columns:
        gdf = gdf.drop(columns=["col_category"])

    assert_eq(pdf, gdf)

    # Write out the gdf using the GPU accelerated writer with INT96 timestamps
    gdf.to_parquet(
        gdf_fname,
        index=None,
        int96_timestamps=True,
    )

    assert os.path.exists(gdf_fname)

    expect = pdf
    got = pd.read_parquet(gdf_fname)

    # verify INT96 timestamps were converted back to the same data.
    assert_eq(expect, got, check_categorical=False, check_dtype=False)


def test_multifile_parquet_folder(tmp_path):
    test_pdf1 = pd.DataFrame(
        {
            "a": np.concatenate(
                [np.arange(10, dtype="float64"), np.full(10, np.nan)]
            )
        },
        index=pd.Index(np.arange(20)),
    )
    test_pdf2 = pd.DataFrame(
        {"a": np.arange(10, dtype="float64")}, index=pd.Index(np.arange(10))
    )
    expect = pd.concat([test_pdf1, test_pdf2])

    par_dir = tmp_path / "multi_part"
    par_dir.mkdir()

    create_parquet_source(test_pdf1, "filepath", par_dir / "multi1.parquet")
    create_parquet_source(test_pdf2, "filepath", par_dir / "multi2.parquet")

    got1 = cudf.read_parquet(tmp_path / "multi_part/*.parquet")
    assert_eq(expect, got1)

    got2 = cudf.read_parquet(tmp_path / "multi_part")
    assert_eq(expect, got2)


# Validates the metadata return path of the parquet writer
def test_parquet_writer_return_metadata(tmp_path, simple_gdf):
    gdf_fname = tmp_path / "data1.parquet"

    # Write out the gdf using the GPU accelerated writer
    df_metadata = simple_gdf.to_parquet(
        gdf_fname, index=None, metadata_file_path="test/data1.parquet"
    )
    # Verify that we got a valid parquet signature in the initial metadata blob
    assert df_metadata.tobytes()[0:4] == b"PAR1"

    df_metadata_list1 = [df_metadata]
    df_metadata_list2 = [df_metadata, df_metadata]
    merged_metadata1 = merge_parquet_filemetadata(df_metadata_list1)
    merged_metadata2 = merge_parquet_filemetadata(df_metadata_list2)

    # Verify that we got a valid parquet signature in the final metadata blob
    assert merged_metadata1.tobytes()[0:4] == b"PAR1"
    assert merged_metadata2.tobytes()[0:4] == b"PAR1"

    # Make sure aggregation is combining metadata correctly
    fmd1 = pa.parquet.ParquetFile(BytesIO(merged_metadata1.tobytes())).metadata
    fmd2 = pa.parquet.ParquetFile(BytesIO(merged_metadata2.tobytes())).metadata
    assert fmd2.num_columns == fmd1.num_columns
    assert fmd2.num_rows == 2 * fmd1.num_rows
    assert fmd2.num_row_groups == 2 * fmd1.num_row_groups


# Validates the integrity of the GPU accelerated parquet writer.
def test_parquet_writer_gpu_none_index(tmp_path, simple_pdf, simple_gdf):
    gdf_fname = tmp_path / "gdf.parquet"
    pdf_fname = tmp_path / "pdf.parquet"

    assert_eq(simple_pdf, simple_gdf)

    # Write out the gdf using the GPU accelerated writer
    simple_gdf.to_parquet(gdf_fname, index=None)
    simple_pdf.to_parquet(pdf_fname, index=None)

    assert os.path.exists(gdf_fname)
    assert os.path.exists(pdf_fname)

    expect = pd.read_parquet(pdf_fname)
    got = pd.read_parquet(gdf_fname)

    assert_eq(expect, got, check_categorical=False)


def test_parquet_writer_gpu_true_index(tmp_path, simple_pdf, simple_gdf):
    gdf_fname = tmp_path / "gdf.parquet"
    pdf_fname = tmp_path / "pdf.parquet"

    assert_eq(simple_pdf, simple_gdf)

    # Write out the gdf using the GPU accelerated writer
    simple_gdf.to_parquet(gdf_fname, index=True)
    simple_pdf.to_parquet(pdf_fname, index=True)

    assert os.path.exists(gdf_fname)
    assert os.path.exists(pdf_fname)

    expect = pd.read_parquet(pdf_fname)
    got = pd.read_parquet(gdf_fname)

    assert_eq(expect, got, check_categorical=False)


def test_parquet_writer_gpu_false_index(tmp_path, simple_pdf, simple_gdf):
    gdf_fname = tmp_path / "gdf.parquet"
    pdf_fname = tmp_path / "pdf.parquet"

    assert_eq(simple_pdf, simple_gdf)

    # Write out the gdf using the GPU accelerated writer
    simple_gdf.to_parquet(gdf_fname, index=False)
    simple_pdf.to_parquet(pdf_fname, index=False)

    assert os.path.exists(gdf_fname)
    assert os.path.exists(pdf_fname)

    expect = pd.read_parquet(pdf_fname)
    got = pd.read_parquet(gdf_fname)

    assert_eq(expect, got, check_categorical=False)


def test_parquet_writer_gpu_multi_index(tmp_path, simple_pdf, simple_gdf):
    gdf_fname = tmp_path / "gdf.parquet"
    pdf_fname = tmp_path / "pdf.parquet"

    simple_pdf = simple_pdf.set_index(["col_bool", "col_int8"])
    simple_gdf = simple_gdf.set_index(["col_bool", "col_int8"])

    assert_eq(simple_pdf, simple_gdf)

    # Write out the gdf using the GPU accelerated writer
    simple_gdf.to_parquet(gdf_fname, index=None)
    simple_pdf.to_parquet(pdf_fname, index=None)

    assert os.path.exists(gdf_fname)
    assert os.path.exists(pdf_fname)

    expect = pd.read_parquet(pdf_fname)
    got = pd.read_parquet(gdf_fname)

    assert_eq(expect, got, check_categorical=False)


def test_parquet_writer_gpu_chunked(tmp_path, simple_pdf, simple_gdf):
    gdf_fname = tmp_path / "gdf.parquet"

    writer = ParquetWriter(gdf_fname)
    writer.write_table(simple_gdf)
    writer.write_table(simple_gdf)
    writer.close()

    assert_eq(pd.read_parquet(gdf_fname), pd.concat([simple_pdf, simple_pdf]))


def test_parquet_writer_gpu_chunked_context(tmp_path, simple_pdf, simple_gdf):
    gdf_fname = tmp_path / "gdf.parquet"

    with ParquetWriter(gdf_fname) as writer:
        writer.write_table(simple_gdf)
        writer.write_table(simple_gdf)

    got = pd.read_parquet(gdf_fname)
    expect = pd.concat([simple_pdf, simple_pdf])
    assert_eq(got, expect)


def test_parquet_write_bytes_io(simple_gdf):
    output = BytesIO()
    simple_gdf.to_parquet(output)
    assert_eq(cudf.read_parquet(output), simple_gdf)


@pytest.mark.parametrize("store_schema", [True, False])
def test_parquet_writer_bytes_io(simple_gdf, store_schema):
    output = BytesIO()

    writer = ParquetWriter(output, store_schema=store_schema)
    writer.write_table(simple_gdf)
    writer.write_table(simple_gdf)
    writer.close()

    assert_eq(cudf.read_parquet(output), cudf.concat([simple_gdf, simple_gdf]))


@pytest.mark.parametrize(
    "row_group_size_kwargs",
    [
        {"row_group_size_bytes": 4 * 1024},
        {"row_group_size_rows": 5000},
    ],
)
def test_parquet_writer_row_group_size(tmp_path, row_group_size_kwargs):
    # Check that row_group_size options are exposed in Python
    # See https://github.com/rapidsai/cudf/issues/10978

    size = 20000
    gdf = cudf.DataFrame({"a": range(size), "b": [1] * size})

    fname = tmp_path / "gdf.parquet"
    with ParquetWriter(fname, **row_group_size_kwargs) as writer:
        writer.write_table(gdf)

    # Simple check for multiple row-groups
    nrows, nrow_groups, columns, _, _ = cudf.io.parquet.read_parquet_metadata(
        fname
    )
    assert nrows == size
    assert nrow_groups > 1
    assert columns == ["a", "b"]

    # Know the specific row-group count for row_group_size_rows
    if "row_group_size_rows" in row_group_size_kwargs:
        assert (
            nrow_groups == size // row_group_size_kwargs["row_group_size_rows"]
        )

    assert_eq(cudf.read_parquet(fname), gdf)


def test_parquet_writer_column_index(tmp_path):
    # Simple test for presence of indices. validity is checked
    # in libcudf tests.
    # Write 2 files, one with column index set, one without.
    # Make sure the former is larger in size.

    size = 20000
    gdf = cudf.DataFrame({"a": range(size), "b": [1] * size})

    fname = tmp_path / "gdf.parquet"
    with ParquetWriter(fname, statistics="ROWGROUP") as writer:
        writer.write_table(gdf)
    s1 = os.path.getsize(fname)

    fname = tmp_path / "gdfi.parquet"
    with ParquetWriter(fname, statistics="COLUMN") as writer:
        writer.write_table(gdf)
    s2 = os.path.getsize(fname)
    assert s2 > s1


@pytest.mark.parametrize(
    "max_page_size_kwargs",
    [
        {"max_page_size_bytes": 4 * 1024},
        {"max_page_size_rows": 5000},
    ],
)
def test_parquet_writer_max_page_size(tmp_path, max_page_size_kwargs):
    # Check that max_page_size options are exposed in Python
    # Since we don't have access to page metadata, instead check that
    # file written with more pages will be slightly larger

    size = 20000
    gdf = cudf.DataFrame({"a": range(size), "b": [1] * size})

    fname = tmp_path / "gdf.parquet"
    with ParquetWriter(fname, **max_page_size_kwargs) as writer:
        writer.write_table(gdf)
    s1 = os.path.getsize(fname)

    assert_eq(cudf.read_parquet(fname), gdf)

    fname = tmp_path / "gdf0.parquet"
    with ParquetWriter(fname) as writer:
        writer.write_table(gdf)
    s2 = os.path.getsize(fname)

    assert_eq(cudf.read_parquet(fname), gdf)
    assert s1 > s2


@pytest.mark.parametrize("use_dict", [False, True])
@pytest.mark.parametrize("max_dict_size", [0, 1048576])
def test_parquet_writer_dictionary_setting(use_dict, max_dict_size):
    # Simple test for checking the validity of dictionary encoding setting
    # and behavior of ParquetWriter in cudf.
    # Write a table with repetitive data with varying dictionary settings.
    # Make sure the written columns are dictionary-encoded accordingly.

    # Table with repetitive data
    table = cudf.DataFrame(
        {
            "int32": cudf.Series([1024] * 1024, dtype="int64"),
        }
    )

    # Write to Parquet using ParquetWriter
    buffer = BytesIO()
    writer = ParquetWriter(
        buffer,
        use_dictionary=use_dict,
        max_dictionary_size=max_dict_size,
    )
    writer.write_table(table)
    writer.close()

    # Read encodings from parquet file
    got = pq.ParquetFile(buffer)
    encodings = got.metadata.row_group(0).column(0).encodings

    # Check for `PLAIN_DICTIONARY` encoding if dictionary encoding enabled
    # and dictionary page limit > 0
    if use_dict is True and max_dict_size > 0:
        assert "PLAIN_DICTIONARY" in encodings
    else:
        assert "PLAIN_DICTIONARY" not in encodings


@pytest.mark.parametrize("filename", ["myfile.parquet", None])
@pytest.mark.parametrize("cols", [["b"], ["c", "b"]])
def test_parquet_partitioned(tmpdir_factory, cols, filename):
    rng = np.random.default_rng(seed=0)
    # Checks that write_to_dataset is wrapping to_parquet
    # as expected
    gdf_dir = str(tmpdir_factory.mktemp("gdf_dir"))
    pdf_dir = str(tmpdir_factory.mktemp("pdf_dir"))
    size = 100
    pdf = pd.DataFrame(
        {
            "a": np.arange(0, stop=size, dtype="int64"),
            "b": rng.choice(list("abcd"), size=size),
            "c": rng.choice(np.arange(4), size=size),
        }
    )
    pdf.to_parquet(pdf_dir, index=False, partition_cols=cols)
    gdf = cudf.from_pandas(pdf)
    gdf.to_parquet(
        gdf_dir, index=False, partition_cols=cols, partition_file_name=filename
    )

    # Read back with pandas to compare
    expect_pd = pd.read_parquet(pdf_dir)
    got_pd = pd.read_parquet(gdf_dir)
    assert_eq(expect_pd, got_pd)

    # Check that cudf and pd return the same read
    got_cudf = cudf.read_parquet(gdf_dir)
    if isinstance(got_pd["c"].dtype, pd.CategoricalDtype):
        # Work-around for pandas bug:
        # https://github.com/pandas-dev/pandas/issues/53345
        got_pd["c"] = got_pd["c"].astype(
            pd.CategoricalDtype(
                categories=got_pd["c"].dtype.categories.astype("int64"),
                ordered=got_pd["c"].dtype.ordered,
            )
        )
    assert_eq(got_pd, got_cudf)

    # If filename is specified, check that it is correct
    if filename:
        for _, _, files in os.walk(gdf_dir):
            for fn in files:
                assert fn == filename


@pytest.mark.parametrize("kwargs", [{"nrows": 1}, {"skip_rows": 1}])
def test_parquet_partitioned_notimplemented(tmpdir_factory, kwargs):
    rng = np.random.default_rng(seed=0)
    # Checks that write_to_dataset is wrapping to_parquet
    # as expected
    pdf_dir = str(tmpdir_factory.mktemp("pdf_dir"))
    size = 100
    pdf = pd.DataFrame(
        {
            "a": np.arange(0, stop=size, dtype="int64"),
            "b": rng.choice(list("abcd"), size=size),
            "c": rng.choice(np.arange(4), size=size),
        }
    )
    pdf.to_parquet(pdf_dir, index=False, partition_cols=["b"])

    with pytest.raises(NotImplementedError):
        cudf.read_parquet(pdf_dir, **kwargs)


@pytest.mark.parametrize("return_meta", [True, False])
def test_parquet_writer_chunked_partitioned(tmpdir_factory, return_meta):
    pdf_dir = str(tmpdir_factory.mktemp("pdf_dir"))
    gdf_dir = str(tmpdir_factory.mktemp("gdf_dir"))

    df1 = cudf.DataFrame({"a": [1, 1, 2, 2, 1], "b": [9, 8, 7, 6, 5]})
    df2 = cudf.DataFrame({"a": [1, 3, 3, 1, 3], "b": [4, 3, 2, 1, 0]})

    cw = ParquetDatasetWriter(gdf_dir, partition_cols=["a"], index=False)
    cw.write_table(df1)
    cw.write_table(df2)
    meta_byte_array = cw.close(return_metadata=return_meta)
    pdf = cudf.concat([df1, df2]).to_pandas()
    pdf.to_parquet(pdf_dir, index=False, partition_cols=["a"])

    if return_meta:
        fmd = pq.ParquetFile(BytesIO(meta_byte_array)).metadata
        assert fmd.num_rows == len(pdf)
        assert fmd.num_row_groups == 4
        files = {
            os.path.join(directory, files[0])
            for directory, _, files in os.walk(gdf_dir)
            if files
        }
        meta_files = {
            os.path.join(gdf_dir, fmd.row_group(i).column(c).file_path)
            for i in range(fmd.num_row_groups)
            for c in range(fmd.row_group(i).num_columns)
        }
        assert files == meta_files

    # Read back with pandas to compare
    expect_pd = pd.read_parquet(pdf_dir)
    got_pd = pd.read_parquet(gdf_dir)
    assert_eq(expect_pd, got_pd)

    # Check that cudf and pd return the same read
    got_cudf = cudf.read_parquet(gdf_dir)

    # Work-around for pandas bug:
    # https://github.com/pandas-dev/pandas/issues/53345
    got_pd["a"] = got_pd["a"].astype(
        pd.CategoricalDtype(
            categories=got_pd["a"].dtype.categories.astype("int64"),
            ordered=got_pd["a"].dtype.ordered,
        )
    )
    assert_eq(got_pd, got_cudf)


@pytest.mark.parametrize(
    "max_file_size,max_file_size_in_bytes",
    [("500KB", 500000), ("MB", 1000000)],
)
def test_parquet_writer_chunked_max_file_size(
    tmpdir_factory, max_file_size, max_file_size_in_bytes
):
    pdf_dir = str(tmpdir_factory.mktemp("pdf_dir"))
    gdf_dir = str(tmpdir_factory.mktemp("gdf_dir"))

    df1 = cudf.DataFrame({"a": [1, 1, 2, 2, 1] * 1000, "b": range(0, 5000)})
    df2 = cudf.DataFrame(
        {"a": [1, 3, 3, 1, 3] * 1000, "b": range(5000, 10000)}
    )

    cw = ParquetDatasetWriter(
        gdf_dir,
        partition_cols=["a"],
        max_file_size=max_file_size,
        file_name_prefix="sample",
    )
    cw.write_table(df1)
    cw.write_table(df2)
    cw.close()
    pdf = cudf.concat([df1, df2]).to_pandas()
    pdf.to_parquet(pdf_dir, index=False, partition_cols=["a"])

    expect_pd = pd.read_parquet(pdf_dir)
    got_pd = pd.read_parquet(gdf_dir)

    assert_eq(
        expect_pd.sort_values(["b"]).reset_index(drop=True),
        got_pd.sort_values(["b"]).reset_index(drop=True),
    )

    # Check that cudf and pd return the same read
    got_cudf = cudf.read_parquet(gdf_dir)

    # Work-around for pandas bug:
    # https://github.com/pandas-dev/pandas/issues/53345
    got_pd["a"] = got_pd["a"].astype(
        pd.CategoricalDtype(
            categories=got_pd["a"].dtype.categories.astype("int64"),
            ordered=got_pd["a"].dtype.ordered,
        )
    )
    assert_eq(
        got_pd.sort_values(["b"]).reset_index(drop=True),
        got_cudf.sort_values(["b"]).reset_index(drop=True),
    )

    all_files = glob.glob(gdf_dir + "/**/*.parquet", recursive=True)
    for each_file in all_files:
        # Validate file sizes with some extra 1000
        # bytes buffer to spare
        assert os.path.getsize(each_file) <= (max_file_size_in_bytes), (
            "File exceeded max_file_size"
        )


def test_parquet_writer_chunked_max_file_size_error():
    with pytest.raises(
        ValueError,
        match="file_name_prefix cannot be None if max_file_size is passed",
    ):
        ParquetDatasetWriter("sample", partition_cols=["a"], max_file_size=100)


@pytest.mark.parametrize(
    "path_names,partition_offsets,match",
    [
        (
            ["a.parquet", "b.parquet"],
            None,
            "partition info is required",
        ),
        (
            ["a.parquet", "b.parquet"],
            [0, 1],
            "same size",
        ),
    ],
)
def test_write_parquet_partitions_info_validation(
    simple_gdf, tmp_path, path_names, partition_offsets, match
):
    paths = [str(tmp_path / path_name) for path_name in path_names]
    with pytest.raises(ValueError, match=match):
        simple_gdf.to_parquet(paths, partition_offsets=partition_offsets)


def test_parquet_writer_chunked_partitioned_context(tmpdir_factory):
    pdf_dir = str(tmpdir_factory.mktemp("pdf_dir"))
    gdf_dir = str(tmpdir_factory.mktemp("gdf_dir"))

    df1 = cudf.DataFrame({"a": [1, 1, 2, 2, 1], "b": [9, 8, 7, 6, 5]})
    df2 = cudf.DataFrame({"a": [1, 3, 3, 1, 3], "b": [4, 3, 2, 1, 0]})

    with ParquetDatasetWriter(
        gdf_dir, partition_cols=["a"], index=False
    ) as cw:
        cw.write_table(df1)
        cw.write_table(df2)

    pdf = cudf.concat([df1, df2]).to_pandas()
    pdf.to_parquet(pdf_dir, index=False, partition_cols=["a"])

    # Read back with pandas to compare
    expect_pd = pd.read_parquet(pdf_dir)
    got_pd = pd.read_parquet(gdf_dir)
    assert_eq(expect_pd, got_pd)

    # Check that cudf and pd return the same read
    got_cudf = cudf.read_parquet(gdf_dir)

    # Work-around for pandas bug:
    # https://github.com/pandas-dev/pandas/issues/53345
    got_pd["a"] = got_pd["a"].astype(
        pd.CategoricalDtype(
            categories=got_pd["a"].dtype.categories.astype("int64"),
            ordered=got_pd["a"].dtype.ordered,
        )
    )
    assert_eq(got_pd, got_cudf)


@pytest.mark.parametrize("cols", [None, ["b"]])
@pytest.mark.parametrize("store_schema", [True, False])
def test_parquet_write_to_dataset(tmpdir_factory, cols, store_schema):
    rng = np.random.default_rng(seed=0)
    dir1 = tmpdir_factory.mktemp("dir1")
    dir2 = tmpdir_factory.mktemp("dir2")
    if cols is None:
        dir1 = dir1.join("file.pq")
        dir2 = dir2.join("file.pq")
    dir1 = str(dir1)
    dir2 = str(dir2)

    size = 100
    gdf = cudf.DataFrame(
        {
            "a": np.arange(0, stop=size),
            "b": rng.choice(np.arange(4), size=size),
        }
    )
    gdf.to_parquet(dir1, partition_cols=cols, store_schema=store_schema)
    cudf.io.write_to_dataset(gdf, dir2, partition_cols=cols)

    # Read back with cudf
    expect = cudf.read_parquet(dir1)
    got = cudf.read_parquet(dir2)
    assert_eq(expect, got)

    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, 3]),
            "b": cudf.Series([1, 2, 3]),
            "c": cudf.Series(["a", "b", "c"], dtype="category"),
        }
    )
    with pytest.raises(ValueError):
        gdf.to_parquet(dir1, partition_cols=cols, store_schema=store_schema)


@pytest.mark.parametrize(
    "pfilters",
    [[("b", "==", "b")], [("b", "==", "a"), ("c", "==", 1)]],
)
@pytest.mark.parametrize("selection", ["directory", "files", "row-groups"])
@pytest.mark.parametrize("use_cat", [True, False])
def test_read_parquet_partitioned_filtered(
    tmp_path, pfilters, selection, use_cat
):
    rng = np.random.default_rng(2)
    path = str(tmp_path)
    size = 100
    df = cudf.DataFrame(
        {
            "a": np.arange(0, stop=size, dtype="int64"),
            "b": rng.choice(list("abcd"), size=size),
            "c": rng.choice(np.arange(4), size=size),
        }
    )
    df.to_parquet(path, partition_cols=["c", "b"])

    if selection == "files":
        # Pass in a list of paths
        fs = get_fs_token_paths(path)[0]
        read_path = fs.find(path)
        row_groups = None
    elif selection == "row-groups":
        # Pass in a list of paths AND row-group ids
        fs = get_fs_token_paths(path)[0]
        read_path = fs.find(path)
        row_groups = [[0] for p in read_path]
    else:
        # Pass in a directory path
        # (row-group selection not allowed in this case)
        read_path = path
        row_groups = None

    # Filter on partitioned columns
    expect = pd.read_parquet(read_path, filters=pfilters)
    got = cudf.read_parquet(
        read_path,
        filters=pfilters,
        row_groups=row_groups,
        categorical_partitions=use_cat,
    )

    expect["c"] = expect["c"].astype(int)
    if use_cat:
        assert got.dtypes["c"] == "category"
        got["c"] = got["c"].astype(int)
    else:
        # Check that we didn't get categorical
        # columns, but convert back to categorical
        # for comparison with pandas
        assert got.dtypes["b"] == pd.StringDtype(na_value=np.nan)
        got["b"] = got["b"].astype(expect["b"].dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "filters", [[("a", "==", 10)], [[("a", "==", 10)], [("c", "==", 1)]]]
)
def test_read_parquet_partitioned_filtered_other(tmp_path, filters):
    rng = np.random.default_rng(2)
    path = str(tmp_path)
    size = 10
    df = cudf.DataFrame(
        {
            "a": np.arange(0, stop=size, dtype="int64"),
            "b": rng.choice(list("abcd"), size=size),
            "c": rng.choice(np.arange(4), size=size),
        }
    )
    df.to_parquet(path, partition_cols=["c", "b"])
    got = cudf.read_parquet(path, filters=filters)
    expect = pd.read_parquet(path, filters=filters)

    # Work-around for pandas bug:
    # https://github.com/pandas-dev/pandas/issues/53345
    expect["c"] = expect["c"].astype(
        pd.CategoricalDtype(
            categories=expect["c"].dtype.categories.astype("int64"),
            ordered=expect["c"].dtype.ordered,
        )
    )
    assert_eq(expect, got)


def test_parquet_writer_chunked_metadata(tmp_path, simple_pdf, simple_gdf):
    gdf_fname = tmp_path / "gdf.parquet"
    test_path = "test/path"

    writer = ParquetWriter(gdf_fname)
    writer.write_table(simple_gdf)
    writer.write_table(simple_gdf)
    meta_byte_array = writer.close(metadata_file_path=test_path)
    fmd = pq.ParquetFile(BytesIO(meta_byte_array)).metadata

    assert fmd.num_rows == 2 * len(simple_gdf)
    assert fmd.num_row_groups == 2

    for r in range(fmd.num_row_groups):
        for c in range(fmd.num_columns):
            assert fmd.row_group(r).column(c).file_path == test_path


def test_write_read_cudf(tmp_path, pdf):
    file_path = tmp_path / "cudf.parquet"
    if "col_category" in pdf.columns:
        pdf = pdf.drop(columns=["col_category"])

    gdf = cudf.from_pandas(pdf)
    gdf.to_parquet(file_path)
    gdf = cudf.read_parquet(file_path)

    assert_eq(gdf, pdf, check_index_type=not pdf.empty)


def test_write_cudf_read_pandas_pyarrow(tmp_path, pdf):
    cudf_path = tmp_path / "cudf.parquet"
    pandas_path = tmp_path / "pandas.parquet"

    if "col_category" in pdf.columns:
        pdf = pdf.drop(columns=["col_category"])

    df = cudf.from_pandas(pdf)

    df.to_parquet(cudf_path)
    pdf.to_parquet(pandas_path)

    cudf_res = pd.read_parquet(cudf_path)
    pd_res = pd.read_parquet(pandas_path)

    assert_eq(pd_res, cudf_res, check_index_type=not pdf.empty)

    cudf_res = pa.parquet.read_table(
        cudf_path, use_pandas_metadata=True
    ).to_pandas()
    pd_res = pa.parquet.read_table(
        pandas_path, use_pandas_metadata=True
    ).to_pandas()

    assert_eq(cudf_res, pd_res, check_index_type=not pdf.empty)


def test_parquet_writer_criteo(tmp_path):
    # To run this test, download the day 0 of criteo dataset from
    # http://labs.criteo.com/2013/12/download-terabyte-click-logs/
    # and place the uncompressed dataset in the home directory
    fname = os.path.expanduser("~/day_0")
    if not os.path.isfile(fname):
        pytest.skip("Local criteo day 0 tsv file is not found")

    cudf_path = tmp_path / "cudf.parquet"

    cont_names = ["I" + str(x) for x in range(1, 14)]
    cat_names = ["C" + str(x) for x in range(1, 27)]
    cols = ["label", *cont_names, *cat_names]

    df = cudf.read_csv(fname, sep="\t", names=cols, byte_range=(0, 1000000000))
    df = df.drop(columns=cont_names)

    df.to_parquet(cudf_path)


def test_trailing_nans(datadir, tmp_path):
    fname = "trailing_nans.parquet"
    file_path = datadir / fname
    cu_df = cudf.read_parquet(file_path)

    tmp_file_path = tmp_path / fname
    cu_df.to_parquet(tmp_file_path)

    pd.read_parquet(tmp_file_path)


def test_parquet_writer_sliced(tmp_path):
    cudf_path = tmp_path / "cudf.parquet"

    df = pd.DataFrame()
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df = cudf.from_pandas(df)

    df_select = df.iloc[1:3]

    df_select.to_parquet(cudf_path)
    assert_eq(cudf.read_parquet(cudf_path), df_select)


def test_parquet_writer_list_basic(tmp_path):
    expect = pd.DataFrame({"a": [[[1, 2], [3, 4]], None, [[5, 6], None]]})
    fname = tmp_path / "test_parquet_writer_list_basic.parquet"

    gdf = cudf.from_pandas(expect)

    gdf.to_parquet(fname)
    assert os.path.exists(fname)

    got = pd.read_parquet(fname)
    assert_eq(expect, got)


def test_parquet_writer_list_large(tmp_path):
    gdf = cudf.DataFrame({"a": list_gen(int_gen, 128, 40, 25)})
    fname = tmp_path / "test_parquet_writer_list_large.parquet"

    gdf.to_parquet(fname)
    assert os.path.exists(fname)

    got = pd.read_parquet(fname)
    assert_arrow_table_equal(gdf.to_arrow(), pa.Table.from_pandas(got))


def test_parquet_writer_list_large_mixed(tmp_path):
    expect = pd.DataFrame(
        {
            "a": list_gen(string_gen, 64, 40, 25),
            "b": list_gen(int_gen, 64, 40, 25),
            "c": list_gen(int_gen, 64, 40, 25, include_validity=True),
            "d": list_gen(string_gen, 64, 40, 25, include_validity=True),
        }
    )
    fname = tmp_path / "test_parquet_writer_list_large_mixed.parquet"
    gdf = cudf.from_pandas(expect)

    gdf.to_parquet(fname)
    assert os.path.exists(fname)

    got = cudf.read_parquet(fname)
    assert_eq(gdf, got)


@pytest.mark.parametrize("store_schema", [True, False])
def test_parquet_writer_list_chunked(tmp_path, store_schema):
    table1 = cudf.DataFrame(
        {
            "a": list_gen(string_gen, 64, 40, 25),
            "b": list_gen(int_gen, 64, 40, 25),
            "c": list_gen(int_gen, 64, 40, 25, include_validity=True),
            "d": list_gen(string_gen, 64, 40, 25, include_validity=True),
        }
    )
    table2 = cudf.DataFrame(
        {
            "a": list_gen(string_gen, 64, 40, 25),
            "b": list_gen(int_gen, 64, 40, 25),
            "c": list_gen(int_gen, 64, 40, 25, include_validity=True),
            "d": list_gen(string_gen, 64, 40, 25, include_validity=True),
        }
    )
    fname = tmp_path / "test_parquet_writer_list_chunked.parquet"
    expect = cudf.concat([table1, table2])
    expect = expect.reset_index(drop=True)

    with ParquetWriter(fname, store_schema=store_schema) as writer:
        writer.write_table(table1)
        writer.write_table(table2)

    assert os.path.exists(fname)
    got = pq.read_table(fname)
    # compare with pyarrow since pandas doesn't
    # have a list or struct dtype
    assert_arrow_table_equal(expect.to_arrow(), got)


def test_parquet_nullable_boolean(tmp_path, engine):
    pandas_path = tmp_path / "pandas_bools.parquet"

    expected = pd.DataFrame(
        {
            "a": pd.Series(
                [True, False, None, True, False], dtype=pd.BooleanDtype()
            )
        }
    )

    expected.to_parquet(pandas_path)
    with _hide_pyarrow_parquet_cpu_warnings(engine):
        result = cudf.read_parquet(pandas_path, engine=engine)
    if engine == "cudf":
        # TODO: Preserve BooleanDtype from the parquet metadata?
        result["a"] = result["a"].astype(pd.BooleanDtype())

    assert_eq(result, expected)


def run_parquet_index(pdf, index):
    pandas_buffer = BytesIO()
    cudf_buffer = BytesIO()

    gdf = cudf.from_pandas(pdf)

    pdf.to_parquet(pandas_buffer, index=index)
    gdf.to_parquet(cudf_buffer, index=index)

    expected = pd.read_parquet(cudf_buffer)
    actual = cudf.read_parquet(pandas_buffer)

    if expected.empty and actual.empty:
        # We return RangeIndex columns compared
        # to pandas' Index[object] columns
        actual.columns = expected.columns

    assert_eq(expected, actual, check_index_type=True)

    expected = pd.read_parquet(pandas_buffer)
    actual = cudf.read_parquet(cudf_buffer)

    if expected.empty and actual.empty:
        # We return RangeIndex columns compared
        # to pandas' Index[object] columns
        actual.columns = expected.columns

    assert_eq(
        expected,
        actual,
        check_index_type=True,
    )


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(index=[1, 2, 3]),
        pd.DataFrame({"a": [1, 2, 3]}, index=[0.43534, 345, 0.34534]),
        pd.DataFrame(
            {"b": [11, 22, 33], "c": ["a", "b", "c"]},
            index=pd.Index(["a", "b", "c"], name="custom name"),
        ),
        pd.DataFrame(
            {"a": [10, 11, 12], "b": [99, 88, 77]},
            index=pd.RangeIndex(12, 17, 2),
        ),
        pd.DataFrame(
            {"b": [99, 88, 77]},
            index=pd.RangeIndex(22, 27, 2, name="hello index"),
        ),
        pd.DataFrame(index=pd.Index(["a", "b", "c"], name="custom name")),
        pd.DataFrame(
            {"a": ["a", "bb", "cc"], "b": [10, 21, 32]},
            index=pd.MultiIndex.from_tuples([[1, 2], [10, 11], [15, 16]]),
        ),
        pd.DataFrame(
            {"a": ["a", "bb", "cc"], "b": [10, 21, 32]},
            index=pd.MultiIndex.from_tuples(
                [[1, 2], [10, 11], [15, 16]], names=["first", "second"]
            ),
        ),
    ],
)
@pytest.mark.parametrize("index", [None, True, False])
def test_parquet_index(pdf, index):
    run_parquet_index(pdf, index)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(
            None,
            marks=pytest.mark.xfail(
                reason="https://github.com/apache/arrow/issues/40743"
            ),
        ),
        True,
    ],
)
def test_parquet_index_empty(index):
    pdf = pd.DataFrame(index=pd.RangeIndex(0, 10, 1))
    run_parquet_index(pdf, index)


def test_parquet_no_index_empty():
    pdf = pd.DataFrame(index=pd.RangeIndex(0, 10, 1))
    run_parquet_index(pdf, index=False)


def test_parquet_allnull_str(tmp_path, engine):
    pandas_path = tmp_path / "pandas_allnulls.parquet"

    pdf = pd.DataFrame(
        {"a": pd.Series([None, None, None, None, None], dtype="str")}
    )
    expected_gdf = cudf.DataFrame(
        {"a": cudf.Series([None, None, None, None, None], dtype="str")}
    )

    pdf.to_parquet(pandas_path)
    with _hide_pyarrow_parquet_cpu_warnings(engine):
        actual_gdf = cudf.read_parquet(pandas_path, engine=engine)

    assert_eq(actual_gdf, expected_gdf)


def normalized_equals(value1, value2):
    if value1 is pd.NA or value1 is pd.NaT:
        value1 = None
    if value2 is pd.NA or value2 is pd.NaT:
        value2 = None
    if isinstance(value1, np.datetime64):
        value1 = pd.Timestamp(value1).to_pydatetime()
    if isinstance(value2, np.datetime64):
        value2 = pd.Timestamp(value2).to_pydatetime()
    if isinstance(value1, pd.Timestamp):
        value1 = value1.to_pydatetime()
    if isinstance(value2, pd.Timestamp):
        value2 = value2.to_pydatetime()
    if isinstance(value1, datetime.datetime):
        value1 = value1.replace(tzinfo=None)
    if isinstance(value2, datetime.datetime):
        value2 = value2.replace(tzinfo=None)
    if isinstance(value1, pd.Timedelta):
        unit = "ms" if value1.unit == "s" else value1.unit
        value2 = pd.Timedelta(value2, unit=unit)

    # if one is datetime then both values are datetimes now
    if isinstance(value1, datetime.datetime):
        return value1 == value2

    # Compare integers with floats now
    if isinstance(value1, float) or isinstance(value2, float):
        return math.isclose(value1, value2)

    return value1 == value2


@pytest.mark.parametrize("add_nulls", [True, False])
@pytest.mark.parametrize("store_schema", [True, False])
def test_parquet_writer_statistics(tmp_path, pdf, add_nulls, store_schema):
    file_path = tmp_path / "cudf.parquet"
    if "col_category" in pdf.columns:
        pdf = pdf.drop(columns=["col_category", "col_bool"])

    if not add_nulls:
        # Timedelta types convert NaT to None when reading from parquet into
        # pandas which interferes with series.max()/min()
        for t in TIMEDELTA_TYPES:
            pdf["col_" + t] = pd.Series(np.arange(len(pdf.index))).astype(t)
        # pyarrow can't read values with non-zero nanoseconds
        pdf["col_timedelta64[ns]"] = pdf["col_timedelta64[ns]"] * 1000

    gdf = cudf.from_pandas(pdf)
    if add_nulls:
        for col in gdf:
            set_random_null_mask_inplace(gdf[col])
    gdf.to_parquet(file_path, index=False, store_schema=store_schema)

    # Read back from pyarrow
    pq_file = pq.ParquetFile(file_path)
    # verify each row group's statistics
    for rg in range(0, pq_file.num_row_groups):
        pd_slice = pq_file.read_row_group(rg).to_pandas()

        # statistics are per-column. So need to verify independently
        for i, col in enumerate(pd_slice):
            stats = pq_file.metadata.row_group(rg).column(i).statistics

            actual_min = pd_slice[col].min()
            stats_min = stats.min
            assert normalized_equals(actual_min, stats_min)

            actual_max = pd_slice[col].max()
            stats_max = stats.max
            assert normalized_equals(actual_max, stats_max)

            assert stats.null_count == pd_slice[col].isna().sum()
            assert stats.num_values == pd_slice[col].count()


def test_parquet_writer_list_statistics(tmp_path):
    df = pd.DataFrame(
        {
            "a": list_gen(string_gen, 64, 40, 25),
            "b": list_gen(int_gen, 64, 40, 25),
            "c": list_gen(int_gen, 64, 40, 25, include_validity=True),
            "d": list_gen(string_gen, 64, 40, 25, include_validity=True),
        }
    )
    fname = tmp_path / "test_parquet_writer_list_statistics.parquet"
    gdf = cudf.from_pandas(df)

    gdf.to_parquet(fname)
    assert os.path.exists(fname)

    # Read back from pyarrow
    pq_file = pq.ParquetFile(fname)
    # verify each row group's statistics
    for rg in range(0, pq_file.num_row_groups):
        pd_slice = pq_file.read_row_group(rg).to_pandas()

        # statistics are per-column. So need to verify independently
        for i, col in enumerate(pd_slice):
            stats = pq_file.metadata.row_group(rg).column(i).statistics

            actual_min = pd_slice[col].explode().explode().dropna().min()
            stats_min = stats.min
            assert normalized_equals(actual_min, stats_min)

            actual_max = pd_slice[col].explode().explode().dropna().max()
            stats_max = stats.max
            assert normalized_equals(actual_max, stats_max)


@pytest.mark.parametrize(
    "data",
    [
        # Structs
        {
            "being": [
                None,
                {"human?": True, "Deets": {"Name": "Carrot", "Age": 27}},
                {"human?": None, "Deets": {"Name": "Angua", "Age": 25}},
                {"human?": False, "Deets": {"Name": "Cheery", "Age": 31}},
                {"human?": False, "Deets": None},
                {"human?": None, "Deets": {"Name": "Mr", "Age": None}},
            ]
        },
        # List of Structs
        {
            "family": [
                [None, {"human?": True, "deets": {"weight": 2.4, "age": 27}}],
                [
                    {"human?": None, "deets": {"weight": 5.3, "age": 25}},
                    {"human?": False, "deets": {"weight": 8.0, "age": 31}},
                    {"human?": False, "deets": None},
                ],
                [],
                [{"human?": None, "deets": {"weight": 6.9, "age": None}}],
            ]
        },
        # Struct of Lists
        {
            "Real estate records": [
                None,
                {
                    "Status": "NRI",
                    "Ownerships": {
                        "land_unit": [None, 2, None],
                        "flats": [[1, 2, 3], [], [4, 5], [], [0, 6, 0]],
                    },
                },
                {
                    "Status": None,
                    "Ownerships": {
                        "land_unit": [4, 5],
                        "flats": [[7, 8], []],
                    },
                },
                {
                    "Status": "RI",
                    "Ownerships": {"land_unit": None, "flats": [[]]},
                },
                {"Status": "RI", "Ownerships": None},
                {
                    "Status": None,
                    "Ownerships": {
                        "land_unit": [7, 8, 9],
                        "flats": [[], [], []],
                    },
                },
            ]
        },
    ],
)
def test_parquet_writer_nested(tmp_path, data):
    expect = pd.DataFrame(data)
    gdf = cudf.from_pandas(expect)

    fname = tmp_path / "test_parquet_writer_nested.parquet"
    gdf.to_parquet(fname)
    assert os.path.exists(fname)

    got = pd.read_parquet(fname)

    # Normalize expect via a pandas parquet round-trip so that nested
    # nullable integer lists (stored as Python lists with None) match
    # the representation pyarrow produces when reading parquet back
    # (numpy float arrays with NaN, since numpy int can't hold NaN).
    pd_fname = tmp_path / "test_parquet_writer_nested_pd.parquet"
    expect.to_parquet(pd_fname)
    expect = pd.read_parquet(pd_fname)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "decimal_type",
    [
        pytest.param(
            cudf.Decimal32Dtype,
            marks=pytest.mark.xfail(
                reason="https://github.com/apache/arrow/issues/45570 from pandas.read_parquet",
                condition=version.parse(pa.__version__) < version.parse("20"),
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype,
            marks=pytest.mark.xfail(
                reason="https://github.com/apache/arrow/issues/45570 from pandas.read_parquet",
                condition=version.parse(pa.__version__) < version.parse("20"),
            ),
        ),
        cudf.Decimal128Dtype,
    ],
)
@pytest.mark.parametrize("data", [[1, 2, 3], [0.00, 0.01, None, 0.5]])
def test_parquet_writer_decimal(decimal_type, data):
    gdf = cudf.DataFrame({"val": data})

    gdf["dec_val"] = gdf["val"].astype(decimal_type(7, 2))

    buff = BytesIO()
    gdf.to_parquet(buff)

    got = pd.read_parquet(buff, dtype_backend="numpy_nullable")
    assert_eq(gdf["val"].to_pandas(nullable=True), got["val"])
    assert_eq(gdf["dec_val"].to_pandas(), got["dec_val"])


def test_parquet_writer_column_validation():
    cudf_parquet = BytesIO()
    pandas_parquet = BytesIO()
    df = cudf.DataFrame({1: [1, 2, 3], "a": ["a", "b", "c"]})
    pdf = df.to_pandas()

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.warns(UserWarning):
            df.to_parquet(cudf_parquet)

    with pytest.warns(UserWarning):
        pdf.to_parquet(pandas_parquet)

    assert_eq(
        pd.read_parquet(cudf_parquet),
        cudf.read_parquet(pandas_parquet),
    )
    assert_eq(
        cudf.read_parquet(cudf_parquet),
        pd.read_parquet(pandas_parquet),
    )

    with cudf.option_context("mode.pandas_compatible", False):
        with pytest.raises(ValueError):
            df.to_parquet(cudf_parquet)


def test_parquet_writer_nulls_pandas_read(tmp_path, pdf):
    if "col_bool" in pdf.columns:
        pdf.drop(columns="col_bool", inplace=True)
    if "col_category" in pdf.columns:
        pdf.drop(columns="col_category", inplace=True)
    gdf = cudf.from_pandas(pdf)

    num_rows = len(gdf)

    if num_rows > 0:
        rng = np.random.default_rng(0)
        mask = rng.choice([True, False], size=num_rows)
        gdf.loc[mask, :] = None

    fname = tmp_path / "test_parquet_writer_nulls_pandas_read.parquet"
    gdf.to_parquet(fname)
    assert os.path.exists(fname)

    got = pd.read_parquet(fname)
    nullable = num_rows > 0

    if nullable:
        gdf = gdf.drop(columns="col_datetime64[ms]")
        gdf = gdf.drop(columns="col_datetime64[us]")
        got = got.drop(columns="col_datetime64[ms]")
        got = got.drop(columns="col_datetime64[us]")
    if len(got.index) == 0:
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        got.index = got.index.astype(gdf.index.dtype)

    assert_eq(gdf.to_pandas(nullable=nullable), got)


@pytest.mark.parametrize(
    "decimal_type",
    [
        pytest.param(
            cudf.Decimal32Dtype,
            marks=pytest.mark.xfail(
                reason="Requires https://github.com/apache/arrow/pull/45583",
                condition=version.parse(pa.__version__) < version.parse("20"),
            ),
        ),
        pytest.param(
            cudf.Decimal64Dtype,
            marks=pytest.mark.xfail(
                reason="Requires https://github.com/apache/arrow/pull/45583",
                condition=version.parse(pa.__version__) < version.parse("20"),
            ),
        ),
        cudf.Decimal128Dtype,
    ],
)
def test_parquet_decimal_precision(tmp_path, decimal_type):
    df = cudf.DataFrame({"val": ["3.5", "4.2"]}).astype(decimal_type(5, 2))
    assert df.val.dtype.precision == 5

    fname = tmp_path / "decimal_test.parquet"
    df.to_parquet(fname)
    df = cudf.read_parquet(fname)
    assert df.val.dtype.precision == 5


def test_parquet_decimal_precision_empty(tmp_path):
    df = (
        cudf.DataFrame({"val": ["3.5", "4.2"]})
        .astype(cudf.Decimal128Dtype(5, 2))
        .iloc[:0]
    )
    assert df.val.dtype.precision == 5

    fname = tmp_path / "decimal_test.parquet"
    df.to_parquet(fname)
    df = cudf.read_parquet(fname)
    assert df.val.dtype.precision == 5


def test_parquet_reader_brotli(datadir):
    fname = datadir / "brotli_int16.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname).to_pandas(nullable=True)

    assert_eq(expect, got)


def test_parquet_reader_one_level_list(datadir):
    fname = datadir / "one_level_list.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_binary_decimal(datadir):
    fname = datadir / "binary_decimal.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname).to_pandas()

    assert_eq(expect, got)


def test_parquet_reader_fixed_bin(datadir):
    fname = datadir / "fixed_len_byte_array.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_fixed_len_with_dict(tmp_path):
    def flba(i):
        hasher = hashlib.sha256()
        hasher.update(i.to_bytes(4, "little"))
        return hasher.digest()

    # use pyarrow to write table of fixed_len_byte_array
    num_rows = 200
    data = pa.array([flba(i) for i in range(num_rows)], type=pa.binary(32))
    padf = pa.Table.from_arrays([data], names=["flba"])
    padf_fname = tmp_path / "padf.parquet"
    pq.write_table(padf, padf_fname, use_dictionary=True)

    expect = pd.read_parquet(padf_fname)
    got = cudf.read_parquet(padf_fname)
    assert_eq(expect, got)


def test_parquet_flba_round_trip(tmp_path):
    def flba(i):
        hasher = hashlib.sha256()
        hasher.update(i.to_bytes(4, "little"))
        return hasher.digest()

    # use pyarrow to write table of fixed_len_byte_array
    num_rows = 200
    data = pa.array([flba(i) for i in range(num_rows)], type=pa.binary(32))
    padf = pa.Table.from_arrays([data], names=["flba"])
    padf_fname = tmp_path / "padf.parquet"
    pq.write_table(padf, padf_fname)

    # round trip data with cudf
    cdf = cudf.read_parquet(padf_fname)
    cdf_fname = tmp_path / "cdf.parquet"
    cdf.to_parquet(cdf_fname, column_type_length={"flba": 32})

    # now read back in with pyarrow to test it was written properly by cudf
    padf2 = pq.read_table(padf_fname)
    padf3 = pq.read_table(cdf_fname)
    assert_eq(padf2, padf3)
    assert_eq(padf2.schema[0].type, padf3.schema[0].type)


@pytest.mark.parametrize(
    "encoding",
    [
        "PLAIN",
        "DICTIONARY",
        "DELTA_BINARY_PACKED",
        "BYTE_STREAM_SPLIT",
        "USE_DEFAULT",
    ],
)
def test_per_column_encoding_option(encoding):
    pdf = pd.DataFrame({"ilist": [[1, 2, 3, 1, 2, 3]], "i1": [1]})
    cdf = cudf.from_pandas(pdf)
    buffer = BytesIO()
    cdf.to_parquet(
        buffer,
        column_encoding={"ilist.list.element": encoding},
    )
    # DICTIONARY and USE_DEFAULT should both result in a PLAIN_DICTIONARY encoding in parquet
    encoding_name = (
        "PLAIN_DICTIONARY"
        if encoding == "DICTIONARY" or encoding == "USE_DEFAULT"
        else encoding
    )
    pf = pq.ParquetFile(buffer)
    fmd = pf.metadata
    assert encoding_name in fmd.row_group(0).column(0).encodings


@pytest.mark.parametrize("compression", ["SNAPPY", "ZSTD"])
def test_per_column_compression_option(set_decomp_env_vars, compression):
    pdf = pd.DataFrame(
        {"ilist": [[1, 2, 3, 1, 2, 3]], "i1": [[1, 2, 3, 1, 2, 3]]}
    )
    cdf = cudf.from_pandas(pdf)
    buffer = BytesIO()
    cdf.to_parquet(
        buffer,
        compression=compression,
        skip_compression={"ilist.list.element"},
        use_dictionary=False,  # to make sure that data is compressible
    )

    pf = pq.ParquetFile(buffer)
    fmd = pf.metadata
    assert fmd.row_group(0).column(0).compression == "UNCOMPRESSED"
    assert fmd.row_group(0).column(1).compression == compression


@pytest.mark.parametrize(
    "encoding",
    ["DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY"],
)
def test_per_column_options_string_col(tmp_path, encoding):
    pdf = pd.DataFrame({"s": ["a string"], "i1": [1]})
    cdf = cudf.from_pandas(pdf)
    fname = tmp_path / "strcol.parquet"
    cdf.to_parquet(
        fname,
        column_encoding={"s": encoding},
        compression="SNAPPY",
    )
    pf = pq.ParquetFile(fname)
    fmd = pf.metadata
    assert encoding in fmd.row_group(0).column(0).encodings


def test_parquet_bss_round_trip(tmp_path):
    num_rows = 200

    def flba(i):
        hasher = hashlib.sha256()
        hasher.update(i.to_bytes(4, "little"))
        return hasher.digest()

    # use pyarrow to write table of types that support BYTE_STREAM_SPLIT encoding
    rows_per_rowgroup = 5000
    fixed_data = pa.array(
        [flba(i) for i in range(num_rows)], type=pa.binary(32)
    )
    i32_data = pa.array(list(range(num_rows)), type=pa.int32())
    i64_data = pa.array(list(range(num_rows)), type=pa.int64())
    f32_data = pa.array([float(i) for i in range(num_rows)], type=pa.float32())
    f64_data = pa.array([float(i) for i in range(num_rows)], type=pa.float64())
    padf = pa.Table.from_arrays(
        [fixed_data, i32_data, i64_data, f32_data, f64_data],
        names=["flba", "i32", "i64", "f32", "f64"],
    )
    padf_fname = tmp_path / "padf.parquet"
    pq.write_table(
        padf,
        padf_fname,
        column_encoding="BYTE_STREAM_SPLIT",
        use_dictionary=False,
        row_group_size=rows_per_rowgroup,
    )

    # round trip data with cudf
    cdf = cudf.read_parquet(padf_fname)
    cdf_fname = tmp_path / "cdf.parquet"
    cdf.to_parquet(
        cdf_fname,
        column_type_length={"flba": 32},
        column_encoding={
            "flba": "BYTE_STREAM_SPLIT",
            "i32": "BYTE_STREAM_SPLIT",
            "i64": "BYTE_STREAM_SPLIT",
            "f32": "BYTE_STREAM_SPLIT",
            "f64": "BYTE_STREAM_SPLIT",
        },
        row_group_size_rows=rows_per_rowgroup,
    )

    # now read back in with pyarrow to test it was written properly by cudf
    padf2 = pq.read_table(padf_fname)
    padf3 = pq.read_table(cdf_fname)
    assert_eq(padf2, padf3)
    assert_eq(padf2.schema[0].type, padf3.schema[0].type)


def test_parquet_reader_rle_boolean(datadir):
    fname = datadir / "rle_boolean_encoding.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


# testing a specific bug-fix/edge case.
# specifically:  int a parquet file containing a particular way of representing
#                a list column in a schema, the cudf reader was confusing
#                nesting information between a list column and a subsequent
#                string column, ultimately causing a crash.
def test_parquet_reader_one_level_list2(datadir):
    # we are reading in a file containing binary types, but cudf returns
    # those as strings. so we have to massage the pandas data to get
    # them to compare correctly.
    def postprocess(val):
        if isinstance(val, bytes):
            return val.decode()
        elif isinstance(val, np.ndarray):
            return np.array([v.decode() for v in val])
        else:
            return val

    fname = datadir / "one_level_list2.parquet"

    expect = pd.read_parquet(fname)
    expect = expect.map(postprocess)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got, check_dtype=False)


# testing a specific bug-fix/edge case.
# specifically:  in a parquet file containing a particular way of representing
#                a list column in a schema, the cudf reader was confusing
#                nesting information and building a list of list of int instead
#                of a list of int
def test_parquet_reader_one_level_list3(datadir):
    fname = datadir / "one_level_list3.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got, check_dtype=True)


@pytest.mark.parametrize("size_bytes", [4_000_000, 1_000_000, 600_000])
@pytest.mark.parametrize("size_rows", [1_000_000, 100_000, 10_000])
def test_to_parquet_row_group_size(
    tmp_path, large_int64_gdf, size_bytes, size_rows
):
    fname = tmp_path / "row_group_size.parquet"
    large_int64_gdf.to_parquet(
        fname, row_group_size_bytes=size_bytes, row_group_size_rows=size_rows
    )

    num_rows, row_groups, _col_names, _, _ = cudf.io.read_parquet_metadata(
        fname
    )
    # 8 bytes per row, as the column is int64
    expected_num_rows = max(
        math.ceil(num_rows / size_rows), math.ceil(8 * num_rows / size_bytes)
    )
    assert expected_num_rows == row_groups


@pytest.mark.parametrize("size_rows", [500_000, 100_000, 10_000])
def test_parquet_row_group_metadata(tmp_path, large_int64_gdf, size_rows):
    fname = tmp_path / "row_group_size.parquet"
    large_int64_gdf.to_parquet(fname, row_group_size_rows=size_rows)

    # read file metadata from parquet
    (
        num_rows,
        row_groups,
        _,  # col_names
        _,  # num_columns
        row_group_metadata,
    ) = cudf.io.read_parquet_metadata(fname)

    # length(RowGroupsMetaData) == number of row groups
    assert len(row_group_metadata) == row_groups
    # sum of rows in row groups == total rows
    assert num_rows == sum(
        [row_group["num_rows"] for row_group in row_group_metadata]
    )


def test_parquet_reader_decimal_columns():
    df = cudf.DataFrame(
        {
            "col1": cudf.Series([1, 2, 3], dtype=cudf.Decimal128Dtype(10, 2)),
            "col2": [10, 11, 12],
            "col3": [12, 13, 14],
            "col4": ["a", "b", "c"],
        }
    )
    buffer = BytesIO()
    df.to_parquet(buffer)

    actual = cudf.read_parquet(buffer, columns=["col3", "col2", "col1"])
    expected = pd.read_parquet(buffer, columns=["col3", "col2", "col1"])

    assert_eq(actual, expected)


def test_parquet_reader_zstd_compression(datadir):
    fname = datadir / "spark_zstd.parquet"
    try:
        df = cudf.read_parquet(fname)
        pdf = pd.read_parquet(fname)
        assert_eq(df, pdf)
    except RuntimeError:
        pytest.skip(reason="zstd support is not enabled")


def test_read_parquet_multiple_files(tmp_path):
    df_1_path = tmp_path / "df_1.parquet"
    df_2_path = tmp_path / "df_2.parquet"
    df_1 = cudf.DataFrame({"id": range(100), "a": [1] * 100})
    df_1.to_parquet(df_1_path)

    df_2 = cudf.DataFrame({"id": range(200, 2200), "a": [2] * 2000})
    df_2.to_parquet(df_2_path)

    expected = pd.read_parquet([df_1_path, df_2_path])
    actual = cudf.read_parquet([df_1_path, df_2_path])
    assert_eq(expected, actual)

    expected = pd.read_parquet([df_2_path, df_1_path])
    actual = cudf.read_parquet([df_2_path, df_1_path])
    assert_eq(expected, actual)


@pytest.mark.parametrize("index", [True, False, None])
@pytest.mark.parametrize("columns", [None, [], ["b", "a"]])
def test_parquet_columns_and_index_param(index, columns):
    buffer = BytesIO()
    df = cudf.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    df.to_parquet(buffer, index=index)

    expected = pd.read_parquet(buffer, columns=columns)
    got = cudf.read_parquet(buffer, columns=columns)
    if columns == []:
        # cuDF returns RangeIndex columns compared
        # to pandas' Index[object] columns
        got.columns = expected.columns

    assert_eq(expected, got, check_index_type=True)


@pytest.mark.parametrize("columns", [None, ["b", "a"]])
def test_parquet_columns_and_range_index(columns):
    buffer = BytesIO()
    df = cudf.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"]}, index=pd.RangeIndex(2, 5)
    )
    df.to_parquet(buffer)

    expected = pd.read_parquet(buffer, columns=columns)
    got = cudf.read_parquet(buffer, columns=columns)

    assert_eq(expected, got, check_index_type=True)


def test_parquet_nested_struct_list():
    buffer = BytesIO()
    data = {
        "payload": {
            "Domain": {
                "Name": "abc",
                "Id": {"Name": "host", "Value": "127.0.0.8"},
            },
            "StreamId": "12345678",
            "Duration": 10,
            "Offset": 12,
            "Resource": [{"Name": "ZoneName", "Value": "RAPIDS"}],
        }
    }
    df = cudf.DataFrame({"a": cudf.Series(data)})

    df.to_parquet(buffer)
    expected = pd.read_parquet(buffer)
    actual = cudf.read_parquet(buffer)
    assert_eq(expected, actual)
    assert_eq(actual.a.dtype, df.a.dtype)


def test_parquet_writer_zstd():
    size = 12345
    rng = np.random.default_rng(seed=0)
    expected = cudf.DataFrame(
        {
            "a": np.arange(0, stop=size, dtype="float64"),
            "b": rng.choice(list("abcd"), size=size),
            "c": rng.choice(np.arange(4), size=size),
        }
    )

    buff = BytesIO()
    try:
        expected.to_parquet(buff, compression="ZSTD")
    except RuntimeError:
        pytest.mark.xfail(reason="Newer nvCOMP version is required")
    else:
        got = pd.read_parquet(buff)
        assert_eq(expected, got)


@pytest.mark.parametrize("store_schema", [True, False])
def test_parquet_writer_time_delta_physical_type(store_schema):
    df = cudf.DataFrame(
        {
            "s": cudf.Series([1], dtype="timedelta64[s]"),
            "ms": cudf.Series([2], dtype="timedelta64[ms]"),
            "us": cudf.Series([3], dtype="timedelta64[us]"),
            # 4K because Pandas/pyarrow don't support non-zero nanoseconds
            # in Parquet files
            "ns": cudf.Series([4000], dtype="timedelta64[ns]"),
        }
    )
    buffer = BytesIO()
    df.to_parquet(buffer, store_schema=store_schema)

    got = pd.read_parquet(buffer)

    if store_schema:
        expected = pd.DataFrame(
            {
                "s": ["0 days 00:00:01"],
                "ms": ["0 days 00:00:00.002000"],
                "us": ["0 days 00:00:00.000003"],
                "ns": ["0 days 00:00:00.000004"],
            },
            dtype="str",
        )
    else:
        expected = pd.DataFrame(
            {
                "s": ["00:00:01"],
                "ms": ["00:00:00.002000"],
                "us": ["00:00:00.000003"],
                "ns": ["00:00:00.000004"],
            },
            dtype="str",
        )
    assert_eq(got.astype("str"), expected)


@pytest.mark.parametrize("store_schema", [True, False])
def test_parquet_roundtrip_time_delta(store_schema):
    num_rows = 123
    rng = np.random.default_rng(0)
    df = cudf.DataFrame(
        {
            "s": cudf.Series(
                rng.integers(0, 200000, size=num_rows),
                dtype="timedelta64[s]",
            ),
            "ms": cudf.Series(
                rng.integers(0, 200000, size=num_rows),
                dtype="timedelta64[ms]",
            ),
            "us": cudf.Series(
                rng.integers(0, 200000, size=num_rows),
                dtype="timedelta64[us]",
            ),
            "ns": cudf.Series(
                rng.integers(0, 200000, size=num_rows),
                dtype="timedelta64[ns]",
            ),
        }
    )
    buffer = BytesIO()
    df.to_parquet(buffer, store_schema=store_schema)
    # `check_dtype` cannot be removed here as timedelta64[s] will change to `timedelta[ms]`
    assert_eq(df, cudf.read_parquet(buffer), check_dtype=False)
    if store_schema:
        assert_eq(df, pd.read_parquet(buffer))


def test_parquet_reader_malformed_file(datadir):
    fname = datadir / "nested-unsigned-malformed.parquet"

    # expect a failure when reading the whole file
    with pytest.raises(RuntimeError):
        cudf.read_parquet(fname)


def test_parquet_reader_unsupported_page_encoding(datadir):
    fname = datadir / "delta_encoding.parquet"

    # expect a failure when reading the whole file
    with pytest.raises(RuntimeError):
        cudf.read_parquet(fname)


def test_parquet_reader_detect_bad_dictionary(datadir):
    fname = datadir / "bad_dict.parquet"

    # expect a failure when reading the whole file
    with pytest.raises(RuntimeError):
        cudf.read_parquet(fname)


@pytest.mark.parametrize("data", [{"a": [1, 2, 3, 4]}, {"b": [1, None, 2, 3]}])
@pytest.mark.parametrize("force_nullable_schema", [True, False])
def test_parquet_writer_schema_nullability(data, force_nullable_schema):
    df = cudf.DataFrame(data)
    file_obj = BytesIO()

    df.to_parquet(file_obj, force_nullable_schema=force_nullable_schema)

    assert pa.parquet.read_schema(file_obj).field(0).nullable == (
        force_nullable_schema or df.isnull().any().any()
    )


def test_parquet_read_filter_and_project():
    # Filter on columns that are not included
    # in the current column projection

    with BytesIO() as buffer:
        # Write parquet data
        df = cudf.DataFrame(
            {
                "a": [1, 2, 3, 4, 5] * 10,
                "b": [0, 1, 2, 3, 4] * 10,
                "c": range(50),
                "d": [6, 7] * 25,
                "e": [8, 9] * 25,
            }
        )
        df.to_parquet(buffer)

        # Read back with filter and projection
        columns = ["b"]
        filters = [[("a", "==", 5), ("c", ">", 20)]]
        got = cudf.read_parquet(buffer, columns=columns, filters=filters)

    # Check result
    expected = df[(df.a == 5) & (df.c > 20)][columns].reset_index(drop=True)
    assert_eq(got, expected)


def test_parquet_reader_multiindex():
    expected = pd.DataFrame(
        {"A": [1, 2, 3]},
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)]),
    )
    file_obj = BytesIO()
    expected.to_parquet(file_obj, engine="pyarrow")
    with pytest.warns(UserWarning):
        actual = cudf.read_parquet(file_obj, engine="pyarrow")
    assert_eq(actual, expected)


def test_parquet_reader_engine_error():
    with pytest.raises(ValueError):
        cudf.read_parquet(BytesIO(), engine="abc")


@pytest.mark.skipif(
    not is_supported_read_parquet("LZ4")
    or not is_supported_write_parquet("LZ4"),
    reason="LZ4 compression not supported for Parquet",
)
def test_parquet_roundtrip_lz4():
    gdf = cudf.DataFrame({"ints": [1, 2] * 5001})

    buffer = BytesIO()
    gdf.to_parquet(buffer, compression="LZ4")

    got = cudf.read_parquet(buffer)
    assert_eq(gdf, got)


@pytest.mark.skipif(
    not is_supported_write_parquet("LZ4"),
    reason="LZ4 compression not supported for Parquet writing",
)
def test_parquet_write_lz4():
    gdf = cudf.DataFrame({"ints": [1, 2] * 5001})

    buffer = BytesIO()
    gdf.to_parquet(buffer, compression="LZ4")

    got = pd.read_parquet(buffer)
    assert_eq(gdf, got)


def test_parquet_reader_zstd_huff_tables(datadir):
    # Ensure that this zstd-compressed file does not overrun buffers. The
    # problem was fixed in nvcomp 3.0.6.
    # See https://github.com/rapidsai/cudf/issues/15096
    fname = datadir / "zstd_huff_tables_bug.parquet"

    expected = pa.parquet.read_table(fname).to_pandas()
    actual = cudf.read_parquet(fname)
    assert_eq(actual, expected)


def test_parquet_reader_roundtrip_with_arrow_schema():
    # Ensure that the nested types are faithfully being roundtripped
    # across Parquet with arrow schema which is used to faithfully
    # round trip duration types (timedelta64) across Parquet read and write.
    pdf = pd.DataFrame(
        {
            "s": pd.Series([None, None, None], dtype="timedelta64[s]"),
            "ms": pd.Series([1234, None, 32442], dtype="timedelta64[ms]"),
            "us": pd.Series([None, 3456, None], dtype="timedelta64[us]"),
            "ns": pd.Series([1234, 3456, 32442], dtype="timedelta64[ns]"),
            "duration_list": list(
                [
                    [
                        datetime.timedelta(minutes=7, seconds=4),
                        datetime.timedelta(minutes=7),
                    ],
                    [
                        None,
                        None,
                    ],
                    [
                        datetime.timedelta(minutes=7, seconds=4),
                        None,
                    ],
                ]
            ),
            "int64": pd.Series([1234, 123, 4123], dtype="int64"),
            "list": list([[1, 2], [1, 2], [1, 2]]),
            "datetime": pd.Series([1234, 123, 4123], dtype="datetime64[ms]"),
            "map": pd.Series(["cat", "dog", "lion"]).map(
                {"cat": "kitten", "dog": "puppy", "lion": "cub"}
            ),
        }
    )

    # Write parquet with arrow for now (to write arrow:schema)
    buffer = BytesIO()
    pdf.to_parquet(buffer, engine="pyarrow")

    # Read parquet with arrow schema
    got = cudf.read_parquet(buffer)
    # Convert to cudf table for an apple to apple comparison
    expected = cudf.from_pandas(pdf)

    # Check results for reader with schema
    assert_eq(expected, got)

    # Reset buffer
    buffer = BytesIO()

    # Write to buffer with cudf
    expected.to_parquet(buffer, store_schema=True)

    # Read parquet with arrow schema
    got = cudf.read_parquet(buffer)
    # Convert to cudf table for an apple to apple comparison
    expected = cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "data",
    [
        # struct
        [
            {"a": 1, "b": 2},
            {"a": 10, "b": 20},
            {"a": None, "b": 22},
            {"a": None, "b": None},
            {"a": 15, "b": None},
        ],
        # struct-of-list
        [
            {"a": 1, "b": 2, "c": [1, 2, 3]},
            {"a": 10, "b": 20, "c": [4, 5]},
            {"a": None, "b": 22, "c": [6]},
            {"a": None, "b": None, "c": None},
            {"a": 15, "b": None, "c": [-1, -2]},
            None,
            {"a": 100, "b": 200, "c": [-10, None, -20]},
        ],
        # list-of-struct
        [
            [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 4, "b": 5}],
            None,
            [{"a": 10, "b": 20}],
            [{"a": 100, "b": 200}, {"a": None, "b": 300}, None],
        ],
        # struct-of-struct
        [
            {"a": 1, "b": {"inner_a": 10, "inner_b": 20}, "c": 2},
            {"a": 3, "b": {"inner_a": 30, "inner_b": 40}, "c": 4},
            {"a": 5, "b": {"inner_a": 50, "inner_b": None}, "c": 6},
            {"a": 7, "b": None, "c": 8},
            {"a": None, "b": {"inner_a": None, "inner_b": None}, "c": None},
            None,
            {"a": None, "b": {"inner_a": None, "inner_b": 100}, "c": 10},
        ],
        # struct-with-mixed-types
        [
            {
                "struct": {
                    "payload": {
                        "Domain": {
                            "Name": "abc",
                            "Id": {"Name": "host", "Value": "127.0.0.8"},
                            "Duration": datetime.timedelta(minutes=12),
                        },
                        "StreamId": "12345678",
                        "Duration": datetime.timedelta(minutes=4),
                        "Offset": None,
                        "Resource": [
                            {
                                "Name": "ZoneName",
                                "Value": "RAPIDS",
                                "Duration": datetime.timedelta(seconds=1),
                            }
                        ],
                    }
                }
            }
        ],
    ],
)
def test_parquet_reader_roundtrip_structs_with_arrow_schema(tmp_path, data):
    # Ensure that the structs with duration types are faithfully being
    # roundtripped across Parquet with arrow schema
    pdf = pd.DataFrame({"struct": pd.Series(data)})

    buffer = BytesIO()
    pdf.to_parquet(buffer, engine="pyarrow")

    # Read parquet with arrow schema
    got = cudf.read_parquet(buffer)
    # Convert to cudf table for an apple to apple comparison
    expected = cudf.from_pandas(pdf)

    # Check results
    assert_eq(expected, got)

    # Reset buffer
    buffer = BytesIO()

    # Write to buffer with cudf
    expected.to_parquet(buffer, store_schema=True)

    # Read parquet with arrow schema
    got = cudf.read_parquet(buffer)
    # Convert to cudf table for an apple to apple comparison
    expected = cudf.from_pandas(pdf)

    # Check results
    assert_eq(expected, got)


@pytest.mark.parametrize("index", [None, True, False])
@pytest.mark.xfail(
    condition=version.parse(pa.__version__) < version.parse("20"),
    reason="Requires https://github.com/apache/arrow/pull/45583",
)
def test_parquet_writer_roundtrip_with_arrow_schema(index):
    # Ensure that the concrete and nested types are faithfully being roundtripped
    # across Parquet with arrow schema
    expected = cudf.DataFrame(
        {
            "s": cudf.Series([None, None, None], dtype="timedelta64[s]"),
            "us": cudf.Series([None, 3456, None], dtype="timedelta64[us]"),
            "duration_list": list(
                [
                    [
                        datetime.timedelta(minutes=7, seconds=4),
                        datetime.timedelta(minutes=7),
                    ],
                    [
                        None,
                        None,
                    ],
                    [
                        datetime.timedelta(minutes=7, seconds=4),
                        None,
                    ],
                ]
            ),
            "int64": cudf.Series([-1234, 123, 4123], dtype="int64"),
            "uint32": cudf.Series([1234, 123, 4123], dtype="uint32"),
            "list": list([[1, 2], [1, 2], [1, 2]]),
            "bool": cudf.Series([True, None, False], dtype=bool),
            "fixed32": cudf.Series([0.00, 1.0, None]).astype(
                cudf.Decimal32Dtype(7, 2)
            ),
            "fixed64": cudf.Series([0.00, 1.0, None]).astype(
                cudf.Decimal64Dtype(7, 2)
            ),
            "fixed128": cudf.Series([0.00, 1.0, None]).astype(
                cudf.Decimal128Dtype(7, 2)
            ),
            "datetime": cudf.Series([1234, 123, 4123], dtype="datetime64[ms]"),
            "map": cudf.Series(["cat", "dog", "lion"]).map(
                {"cat": "kitten", "dog": "puppy", "lion": "cub"}
            ),
        }
    )

    # Write to Parquet with arrow schema for faithful roundtrip
    buffer = BytesIO()
    expected.to_parquet(buffer, store_schema=True, index=index)

    # Read parquet with pyarrow, pandas and cudf readers
    got = cudf.DataFrame.from_arrow(pq.read_table(buffer))
    got2 = cudf.DataFrame(pd.read_parquet(buffer))
    got3 = cudf.read_parquet(buffer)

    # drop the index column for comparison: __index_level_0__
    if index:
        got.drop(columns="__index_level_0__", inplace=True)
        got2.drop(columns="__index_level_0__", inplace=True)

    # Check results
    assert_eq(expected, got)
    assert_eq(expected, got2)
    assert_eq(expected, got3)


def test_parquet_writer_int96_timestamps_and_arrow_schema():
    df = cudf.DataFrame(
        {
            "timestamp": cudf.Series(
                [1234, 123, 4123], dtype="datetime64[ms]"
            ),
        }
    )

    # Output buffer
    buffer = BytesIO()

    # Writing out parquet with both INT96 timestamps and arrow_schema
    # enabled should throw an exception.
    with pytest.raises(RuntimeError):
        df.to_parquet(buffer, int96_timestamps=True, store_schema=True)


@pytest.mark.parametrize(
    "data",
    [
        # struct
        [
            {"a": 1, "b": 2},
            {"a": 10, "b": 20},
            {"a": None, "b": 22},
            {"a": None, "b": None},
            {"a": 15, "b": None},
        ],
        # struct-of-list
        [
            {"a": 1, "b": 2, "c": [1, 2, 3]},
            {"a": 10, "b": 20, "c": [4, 5]},
            {"a": None, "b": 22, "c": [6]},
            {"a": None, "b": None, "c": None},
            {"a": 15, "b": None, "c": [-1, -2]},
            None,
            {"a": 100, "b": 200, "c": [-10, None, -20]},
        ],
        # list-of-struct
        [
            [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 4, "b": 5}],
            None,
            [{"a": 10, "b": 20}],
            [{"a": 100, "b": 200}, {"a": None, "b": 300}, None],
        ],
        # struct-of-struct
        [
            {"a": 1, "b": {"inner_a": 10, "inner_b": 20}, "c": 2},
            {"a": 3, "b": {"inner_a": 30, "inner_b": 40}, "c": 4},
            {"a": 5, "b": {"inner_a": 50, "inner_b": None}, "c": 6},
            {"a": 7, "b": None, "c": 8},
            {"a": None, "b": {"inner_a": None, "inner_b": None}, "c": None},
            None,
            {"a": None, "b": {"inner_a": None, "inner_b": 100}, "c": 10},
        ],
        # struct-with-mixed-types
        [
            {
                "struct": {
                    "payload": {
                        "Domain": {
                            "Name": "abc",
                            "Id": {"Name": "host", "Value": "127.0.0.8"},
                            "Duration": datetime.timedelta(minutes=12),
                        },
                        "StreamId": "12345678",
                        "Duration": datetime.timedelta(minutes=4),
                        "Offset": None,
                        "Resource": [
                            {
                                "Name": "ZoneName",
                                "Value": "RAPIDS",
                                "Duration": datetime.timedelta(seconds=1),
                            }
                        ],
                    }
                }
            }
        ],
    ],
)
@pytest.mark.parametrize("index", [None, True, False])
def test_parquet_writer_roundtrip_structs_with_arrow_schema(data, index):
    # Ensure that the structs are faithfully being roundtripped across
    # Parquet with arrow schema
    pa_expected = pa.Table.from_pydict({"struct": data})

    expected = cudf.DataFrame.from_arrow(pa_expected)

    # Write expected data frame to Parquet with arrow schema
    buffer = BytesIO()
    expected.to_parquet(buffer, store_schema=True, index=index)

    # Read Parquet with pyarrow
    pa_got = pq.read_table(buffer)

    # drop the index column for comparison: __index_level_0__
    if index:
        pa_got = pa_got.drop(columns="__index_level_0__")

    # Check results
    assert_eq(pa_expected, pa_got)

    # Convert to cuDF table and also read Parquet with cuDF reader
    got = cudf.DataFrame.from_arrow(pa_got)
    got2 = cudf.read_parquet(buffer)

    # Check results
    assert_eq(expected, got)
    assert_eq(expected, got2)


@pytest.mark.parametrize("chunk_read_limit", [0, 240, 1024000000])
@pytest.mark.parametrize("pass_read_limit", [0, 240, 1024000000])
@pytest.mark.parametrize("use_pandas_metadata", [True, False])
@pytest.mark.parametrize("row_groups", [[[0]], None, [[0, 1]]])
def test_parquet_chunked_reader(
    chunk_read_limit, pass_read_limit, use_pandas_metadata, row_groups
):
    df = pd.DataFrame(
        {"a": [1, 2, 3, None] * 1000, "b": ["av", "qw", None, "xyz"] * 1000}
    )
    buffer = BytesIO()
    df.to_parquet(buffer, row_group_size=1000)
    with cudf.option_context("io.parquet.low_memory", True):
        actual = cudf.read_parquet(
            [buffer],
            _chunk_read_limit=chunk_read_limit,
            _pass_read_limit=pass_read_limit,
            use_pandas_metadata=use_pandas_metadata,
            row_groups=row_groups,
        )
    expected = cudf.read_parquet(
        buffer, use_pandas_metadata=use_pandas_metadata, row_groups=row_groups
    )
    assert_eq(expected, actual)


@pytest.mark.parametrize("chunk_read_limit", [256, 2560])
@pytest.mark.parametrize("pass_read_limit", [256, 2560])
@pytest.mark.parametrize("num_rows", [49, 291])
@pytest.mark.parametrize("skip_rows", [412, 601])
@pytest.mark.parametrize("data_size", [100, 200])
def test_parquet_chunked_reader_structs(
    chunk_read_limit, pass_read_limit, num_rows, skip_rows, data_size
):
    data = [
        {
            "a": "g",
            "b": {
                "b_a": 10,
                "b_b": {"b_b_b": None, "b_b_a": 2},
            },
            "c": None,
        },
        {"a": None, "b": {"b_a": None, "b_b": None}, "c": [15, 16]},
        {"a": "j", "b": None, "c": [8, 10]},
        {"a": None, "b": {"b_a": None, "b_b": None}, "c": None},
        None,
        {
            "a": None,
            "b": {"b_a": None, "b_b": {"b_b_b": 1}},
            "c": [18, 19],
        },
        {"a": None, "b": None, "c": None},
    ] * data_size

    pa_struct = pa.Table.from_pydict({"struct": data})
    df = cudf.DataFrame.from_arrow(pa_struct)
    buffer = BytesIO()
    df.to_parquet(buffer, row_group_size_rows=7000, max_page_size_rows=100)

    # Number of rows to read
    nrows = num_rows if skip_rows + num_rows < len(df) else len(df) - skip_rows

    with cudf.option_context("io.parquet.low_memory", True):
        actual = cudf.read_parquet(
            [buffer],
            _chunk_read_limit=chunk_read_limit,
            _pass_read_limit=pass_read_limit,
            nrows=nrows,
            skip_rows=skip_rows,
        ).reset_index(drop=True)
    expected = cudf.read_parquet(
        buffer, nrows=nrows, skip_rows=skip_rows
    ).reset_index(drop=True)
    assert_eq(expected, actual)


@pytest.mark.parametrize("chunk_read_limit", [0, 24, 10240000])
@pytest.mark.parametrize("pass_read_limit", [0, 24, 10240000])
@pytest.mark.parametrize("num_rows", [47, 97, None])
@pytest.mark.parametrize(
    "str_encoding",
    [
        "PLAIN",
        "DELTA_BYTE_ARRAY",
        "DELTA_LENGTH_BYTE_ARRAY",
    ],
)
def test_parquet_chunked_reader_string_decoders(
    chunk_read_limit,
    pass_read_limit,
    num_rows,
    str_encoding,
):
    df = pd.DataFrame(
        {
            "i64": [1, 2, 3, None] * 100,
            "str": ["av", "qw", "asd", "xyz"] * 100,
            "list": list(
                [["ad", "cd"], ["asd", "fd"], None, ["asd", None]] * 100
            ),
        }
    )
    buffer = BytesIO()
    # Write 4 Parquet row groups with string column encoded
    df.to_parquet(
        buffer,
        row_group_size=100,
        use_dictionary=False,
        column_encoding={"str": str_encoding},
    )

    # Number of rows to read
    nrows = num_rows if num_rows is not None else len(df)

    # Check with num_rows specified
    with cudf.option_context("io.parquet.low_memory", True):
        actual = cudf.read_parquet(
            [buffer],
            _chunk_read_limit=chunk_read_limit,
            _pass_read_limit=pass_read_limit,
            nrows=nrows,
        )
    expected = cudf.read_parquet(
        buffer,
        nrows=nrows,
    )
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "nrows, skip_rows",
    [
        (0, 0),
        (99, 101),
        (988, 61),
        (99, 1011),
        (101, 1601),
        (99, 1901),
    ],
)
@pytest.mark.parametrize(
    "row_group_size_rows, page_size_rows",
    [
        (1000, 1000),  # 1 RG, 1 page per RG
        (1000, 100),  # 1 RG, multiple pages per RG
        (100, 100),  # multiple RGs, 1 page per RG
        (100, 10),  # multiple RGs, multiple pages per RG
    ],
)
@pytest.mark.parametrize(
    "chunk_read_limit, pass_read_limit",
    [
        (256, 256),  # small chunk and pass read limits
        (0, 1024),  # zero chunk and small pass read limit
        (256, 0),  # small chunk and zero pass read limit
        (256000, 256000),  # large chunk and pass read limits
    ],
)
def test_chunked_parquet_reader_nrows_skiprows(
    nrows,
    skip_rows,
    row_group_size_rows,
    page_size_rows,
    chunk_read_limit,
    pass_read_limit,
):
    df = cudf.DataFrame(
        {
            "a": list(
                [
                    ["cat", "lion", "deer"],
                    ["bear", "ibex", None],
                    ["tiger", None, "bull"],
                    [None, "wolf", "fox"],
                ]
            )
            * 500,
            "b": ["av", "qw", None, "xyz"] * 500,
        }
    )
    expected = df[skip_rows : skip_rows + nrows]
    buffer = BytesIO()
    df.to_parquet(
        buffer,
        row_group_size_rows=row_group_size_rows,
        max_page_size_rows=page_size_rows,
    )
    got = cudf.read_parquet(buffer, nrows=nrows, skip_rows=skip_rows)
    assert_eq(expected, got)

    # Check for chunked parquet reader
    with cudf.option_context("io.parquet.low_memory", True):
        got = cudf.read_parquet(
            [buffer],
            _chunk_read_limit=chunk_read_limit,
            _pass_read_limit=pass_read_limit,
            nrows=nrows,
            skip_rows=skip_rows,
        ).reset_index(drop=True)
        # Reset index for comparison
        expected = expected.reset_index(drop=True)
        assert_eq(expected, got)


@pytest.mark.parametrize(
    "column_encoding",
    [
        "DELTA_BINARY_PACKED",
        "DELTA_LENGTH_BYTE_ARRAY",
        "DELTA_BYTE_ARRAY",
    ],
)
def test_chunked_parquet_reader_delta_decode_level_bounds(
    column_encoding, tmp_path
):
    n_rows = 2000

    if column_encoding == "DELTA_BINARY_PACKED":
        # Nullable int32 column
        data = list(range(n_rows))
        validity = [None if i % 4 == 3 else data[i] for i in range(n_rows)]
        table = pa.table({"col": pa.array(validity, type=pa.int32())})
    else:
        # Nullable string column for DELTA_LENGTH_BYTE_ARRAY / DELTA_BYTE_ARRAY
        strings = [None if i % 4 == 3 else f"str_{i}" for i in range(n_rows)]
        table = pa.table({"col": pa.array(strings, type=pa.string())})

    fname = tmp_path / "delta_level_bounds.parquet"
    pq.write_table(
        table,
        fname,
        version="2.6",
        data_page_version="2.0",
        data_page_size=4 * 1024,  # small pages -> many pages per RG
        row_group_size=1000,
        column_encoding=column_encoding,
        use_dictionary=False,
    )

    skip = 101
    nrows = 99

    expected = cudf.read_parquet(fname, nrows=nrows, skip_rows=skip)

    with cudf.option_context("io.parquet.low_memory", True):
        got = cudf.read_parquet(
            [fname],
            _chunk_read_limit=256,
            _pass_read_limit=256,
            nrows=nrows,
            skip_rows=skip,
        ).reset_index(drop=True)
        expected = expected.reset_index(drop=True)
        assert_eq(expected, got)


def test_parquet_reader_pandas_compatibility():
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4] * 10000, "b": ["av", "qw", "hi", "xyz"] * 10000}
    )
    buffer = BytesIO()
    df.to_parquet(buffer)
    with cudf.option_context("io.parquet.low_memory", True):
        expected = cudf.read_parquet(buffer)
    assert_eq(expected, df)


@pytest.mark.parametrize("store_schema", [True, False])
def test_parquet_reader_with_mismatched_tables(store_schema):
    # cuDF tables with mixed types
    df1 = cudf.DataFrame(
        {
            "i32": cudf.Series([None, None, None], dtype="int32"),
            "i64": cudf.Series([1234, 467, 123], dtype="int64"),
            "list": list([[1, 2], None, [None, 6]]),
            "time": cudf.Series([1234, 123, 4123], dtype="datetime64[ms]"),
            "str": ["vfd", None, "ghu"],
            "d_list": list(
                [
                    [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2)],
                    [None, pd.Timedelta(minutes=3)],
                    [pd.Timedelta(minutes=8), None],
                ]
            ),
        }
    )

    df2 = cudf.DataFrame(
        {
            "str": ["abc", "def", "ghi"],
            "i64": cudf.Series([None, 65, 98], dtype="int64"),
            "times": cudf.Series([1234, None, 4123], dtype="datetime64[us]"),
            "list": list([[7, 8], [9, 10], [11, 12]]),
            "d_list": list(
                [
                    [pd.Timedelta(minutes=4), None],
                    None,
                    [pd.Timedelta(minutes=6), None],
                ]
            ),
        }
    )

    # IO buffers
    buf1 = BytesIO()
    buf2 = BytesIO()

    # Write Parquet with and without arrow schema
    df1.to_parquet(buf1, store_schema=store_schema)
    df2.to_parquet(buf2, store_schema=store_schema)

    # Read mismatched Parquet files
    got = cudf.read_parquet(
        [buf1, buf2],
        columns=["list", "D_List", "Str"],
        filters=[("I64", ">", 20)],
        allow_mismatched_pq_schemas=True,
        case_sensitive_names=False,
    )

    # Construct the expected table
    expected = cudf.concat(
        [
            df1[df1["i64"] > 20][["list", "d_list", "str"]],
            df2[df2["i64"] > 20][["list", "d_list", "str"]],
        ]
    ).reset_index(drop=True)

    # Read with chunked reader (filter columns not supported)
    with cudf.option_context("io.parquet.low_memory", True):
        got_chunked = cudf.read_parquet(
            [buf1, buf2],
            columns=["list", "d_list", "str"],
            _chunk_read_limit=240,
            _pass_read_limit=240,
            allow_mismatched_pq_schemas=True,
        )

    # Construct the expected table without filter columns
    expected_chunked = cudf.concat(
        [df1[["list", "d_list", "str"]], df2[["list", "d_list", "str"]]]
    ).reset_index(drop=True)

    # Check results
    assert_eq(expected, got)
    assert_eq(expected_chunked, got_chunked)


def test_parquet_reader_with_mismatched_structs():
    data1 = [
        {
            "a": 1,
            "b": {
                "a_a": 10,
                "b_b": {"b_b_b": 1, "b_b_a": 2},
            },
            "c": 2,
        },
        {
            "a": 3,
            "b": {"b_a": 30, "b_b": {"b_b_a": 210}},
            "c": 4,
        },
        {"a": 5, "b": {"b_a": 50, "b_b": None}, "c": 6},
        {"a": 7, "b": None, "c": 8},
        {"a": 5, "b": {"b_a": None, "b_b": None}, "c": None},
    ]

    data2 = [
        {"a": 1, "b": {"b_b": {"b_b_a": None}}},
        {"a": 5, "b": {"b_b": None}},
        {"a": 7, "b": {"b_b": {"b_b_b": 1, "b_b_a": 0}}},
        {"a": None, "b": {"b_b": None}},
        None,
    ]

    # cuDF tables from struct data
    df1 = cudf.DataFrame.from_arrow(pa.Table.from_pydict({"struct": data1}))
    df2 = cudf.DataFrame.from_arrow(pa.Table.from_pydict({"struct": data2}))

    # Buffers
    buf1 = BytesIO()
    buf2 = BytesIO()

    # Write to parquet
    df1.to_parquet(buf1)
    df2.to_parquet(buf2)

    # Read the struct.b.inner_b.inner_inner_a column from parquet
    got = cudf.read_parquet(
        [buf1, buf2],
        columns=["struct.b.b_b.b_b_a"],
        allow_mismatched_pq_schemas=True,
    )
    got = (
        cudf.Series(got["struct"])
        .struct.field("b")
        .struct.field("b_b")
        .struct.field("b_b_a")
    )

    # Read with chunked reader
    with cudf.option_context("io.parquet.low_memory", True):
        got_chunked = cudf.read_parquet(
            [buf1, buf2],
            columns=["struct.b.b_b.b_b_a"],
            _chunk_read_limit=240,
            _pass_read_limit=240,
            allow_mismatched_pq_schemas=True,
        )
    got_chunked = (
        cudf.Series(got_chunked["struct"])
        .struct.field("b")
        .struct.field("b_b")
        .struct.field("b_b_a")
    )

    # Construct the expected series
    expected = cudf.concat(
        [
            cudf.Series(df1["struct"])
            .struct.field("b")
            .struct.field("b_b")
            .struct.field("b_b_a"),
            cudf.Series(df2["struct"])
            .struct.field("b")
            .struct.field("b_b")
            .struct.field("b_b_a"),
        ]
    ).reset_index(drop=True)

    # Check results
    assert_eq(expected, got)
    assert_eq(expected, got_chunked)


def test_parquet_reader_with_mismatched_schemas_error():
    df1 = cudf.DataFrame(
        {
            "millis": cudf.Series([123, 3454, 123], dtype="timedelta64[ms]"),
            "i64": cudf.Series([123, 3454, 123], dtype="int64"),
            "i32": cudf.Series([123, 3454, 123], dtype="int32"),
        }
    )
    df2 = cudf.DataFrame(
        {
            "i64": cudf.Series([123, 3454, 123], dtype="int64"),
            "millis": cudf.Series([123, 3454, 123], dtype="timedelta64[ms]"),
        }
    )

    buf1 = BytesIO()
    buf2 = BytesIO()

    df1.to_parquet(buf1, store_schema=True)
    df2.to_parquet(buf2, store_schema=False)

    with pytest.raises(
        ValueError,
        match=(
            r"Encountered mismatching data type or schema across the "
            r"Parquet sources for column '[^']+'"
        ),
    ):
        cudf.read_parquet(
            [buf1, buf2], columns=["millis"], allow_mismatched_pq_schemas=True
        )

    data1 = [
        {"a": 1, "b": {"b_a": 1, "b_b": 6}},
        {"a": 3, "b": {"b_a": None, "b_b": 2}},
    ]
    data2 = [
        {"b": {"b_a": 1}, "c": "str"},
        {"b": {"b_a": None}, "c": None},
    ]

    # cuDF tables from struct data
    df1 = cudf.DataFrame.from_arrow(pa.Table.from_pydict({"struct": data1}))
    df2 = cudf.DataFrame.from_arrow(pa.Table.from_pydict({"struct": data2}))

    # Buffers
    buf1 = BytesIO()
    buf2 = BytesIO()

    # Write to parquet
    df1.to_parquet(buf1)
    df2.to_parquet(buf2)

    with pytest.raises(
        ValueError,
        match=(
            r"Encountered mismatching number of children across Parquet "
            r"sources for column '[^']+'"
        ),
    ):
        cudf.read_parquet(
            [buf1, buf2],
            columns=["struct.b"],
            allow_mismatched_pq_schemas=True,
        )

    with pytest.raises(
        ValueError,
        match=(
            r"Encountered missing nested column '[^']+' across Parquet "
            r"sources for column '[^']+'"
        ),
    ):
        cudf.read_parquet(
            [buf1, buf2],
            columns=["struct.b.b_b"],
            allow_mismatched_pq_schemas=True,
        )


@pytest.mark.xfail(
    condition=version.parse(pa.__version__) < version.parse("20"),
    reason="Requires https://github.com/apache/arrow/pull/45583",
)
def test_parquet_roundtrip_zero_rows_no_column_mask():
    expected = cudf.DataFrame._from_data(
        {
            "int": cudf.core.column.column_empty(0, np.dtype(np.int64)),
            "float": cudf.core.column.column_empty(0, np.dtype(np.float64)),
            "datetime": cudf.core.column.column_empty(
                0, np.dtype("datetime64[ns]")
            ),
            "timedelta": cudf.core.column.column_empty(
                0, np.dtype("timedelta64[ns]")
            ),
            "bool": cudf.core.column.column_empty(0, np.dtype(np.bool_)),
            "decimal": cudf.core.column.column_empty(
                0, cudf.Decimal64Dtype(1)
            ),
            "struct": cudf.core.column.column_empty(
                0, cudf.StructDtype({"a": "int64"})
            ),
            "list": cudf.core.column.column_empty(
                0, cudf.ListDtype("float64")
            ),
        }
    )
    with BytesIO() as bio:
        expected.to_parquet(bio)
        result = cudf.read_parquet(bio)
    assert_eq(result, expected)


def test_parquet_reader_mismatched_nullability():
    # Ensure that we can faithfully read the tables with mismatched nullabilities
    df1 = cudf.DataFrame(
        {
            "timedelta": cudf.Series([12, 54, 1231], dtype="timedelta64[ms]"),
            "duration_list": list(
                [
                    [
                        [
                            [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2)],
                            None,
                            [pd.Timedelta(minutes=8), None],
                        ],
                        None,
                    ],
                    None,
                    [
                        [
                            [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2)],
                            [pd.Timedelta(minutes=5), pd.Timedelta(minutes=3)],
                            [pd.Timedelta(minutes=8), pd.Timedelta(minutes=4)],
                        ]
                    ],
                ]
            ),
            "int64": cudf.Series([1234, None, 4123], dtype="int64"),
            "int32": cudf.Series([1234, 123, 4123], dtype="int32"),
            "list": list([[1, 2], [1, 2], [1, 2]]),
            "datetime": cudf.Series([1234, 123, 4123], dtype="datetime64[ms]"),
            "string": cudf.Series(["kitten", "puppy", "cub"]),
        }
    )

    df2 = cudf.DataFrame(
        {
            "timedelta": cudf.Series(
                [None, None, None], dtype="timedelta64[ms]"
            ),
            "duration_list": list(
                [
                    [
                        [
                            [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2)],
                            [pd.Timedelta(minutes=8), pd.Timedelta(minutes=1)],
                        ],
                    ],
                    [
                        [
                            [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2)],
                            [pd.Timedelta(minutes=5), pd.Timedelta(minutes=3)],
                            [pd.Timedelta(minutes=8), pd.Timedelta(minutes=4)],
                        ]
                    ],
                    [
                        [
                            [pd.Timedelta(minutes=1), pd.Timedelta(minutes=2)],
                            [pd.Timedelta(minutes=5), pd.Timedelta(minutes=3)],
                            [pd.Timedelta(minutes=8), pd.Timedelta(minutes=4)],
                        ]
                    ],
                ]
            ),
            "int64": cudf.Series([1234, 123, 4123], dtype="int64"),
            "int32": cudf.Series([1234, None, 4123], dtype="int32"),
            "list": list([[1, 2], None, [1, 2]]),
            "datetime": cudf.Series(
                [1234, None, 4123], dtype="datetime64[ms]"
            ),
            "string": cudf.Series(["kitten", None, "cub"]),
        }
    )

    # Write tables to parquet with arrow schema for compatibility for duration column(s)
    fname1 = BytesIO()
    df1.to_parquet(fname1, store_schema=True)
    fname2 = BytesIO()
    df2.to_parquet(fname2, store_schema=True)

    # Read tables back with cudf and arrow in either order and compare
    assert_eq(
        cudf.read_parquet([fname1, fname2]),
        cudf.concat([df1, df2]).reset_index(drop=True),
    )
    assert_eq(
        cudf.read_parquet([fname2, fname1]),
        cudf.concat([df2, df1]).reset_index(drop=True),
    )


def test_parquet_reader_mismatched_nullability_structs(tmp_path):
    data1 = [
        {
            "a": "a",
            "b": {
                "b_a": 10,
                "b_b": {"b_b_b": 1, "b_b_a": 12},
            },
            "c": [1, 2],
        },
        {
            "a": "b",
            "b": {
                "b_a": 30,
                "b_b": {"b_b_b": 2, "b_b_a": 2},
            },
            "c": [3, 4],
        },
        {
            "a": "c",
            "b": {
                "b_a": 50,
                "b_b": {"b_b_b": 4, "b_b_a": 5},
            },
            "c": [5, 6],
        },
        {
            "a": "d",
            "b": {
                "b_a": 135,
                "b_b": {"b_b_b": 12, "b_b_a": 32},
            },
            "c": [7, 8],
        },
        {
            "a": "e",
            "b": {
                "b_a": 1,
                "b_b": {"b_b_b": 1, "b_b_a": 5},
            },
            "c": [9, 10],
        },
        {
            "a": "f",
            "b": {
                "b_a": 32,
                "b_b": {"b_b_b": 1, "b_b_a": 6},
            },
            "c": [11, 12],
        },
    ]

    data2 = [
        {
            "a": "g",
            "b": {
                "b_a": 10,
                "b_b": {"b_b_b": None, "b_b_a": 2},
            },
            "c": None,
        },
        {"a": None, "b": {"b_a": None, "b_b": None}, "c": [15, 16]},
        {"a": "j", "b": None, "c": [8, 10]},
        {"a": None, "b": {"b_a": None, "b_b": None}, "c": None},
        None,
        {
            "a": None,
            "b": {"b_a": None, "b_b": {"b_b_b": 1}},
            "c": [18, 19],
        },
        {"a": None, "b": None, "c": None},
    ]

    pa_table1 = pa.Table.from_pydict({"struct": data1})
    df1 = cudf.DataFrame.from_arrow(pa_table1)

    pa_table2 = pa.Table.from_pydict({"struct": data2})
    df2 = cudf.DataFrame.from_arrow(pa_table2)

    # Write tables to parquet
    buf1 = BytesIO()
    df1.to_parquet(buf1)
    buf2 = BytesIO()
    df2.to_parquet(buf2)

    # Read tables back with cudf and compare with expected.
    assert_eq(
        cudf.read_parquet([buf1, buf2]),
        cudf.concat([df1, df2]).reset_index(drop=True),
    )
    assert_eq(
        cudf.read_parquet([buf2, buf1]),
        cudf.concat([df2, df1]).reset_index(drop=True),
    )


@pytest.mark.skipif(
    pa.__version__ == "19.0.0",
    reason="https://github.com/apache/arrow/issues/45283, https://github.com/rapidsai/cudf/issues/17806",
)
@pytest.mark.parametrize(
    "stats_fname,bloom_filter_fname",
    [
        (
            "mixed_card_ndv_100_chunk_stats.snappy.parquet",
            "mixed_card_ndv_100_bf_fpp0.1_nostats.snappy.parquet",
        ),
        (
            "mixed_card_ndv_500_chunk_stats.snappy.parquet",
            "mixed_card_ndv_500_bf_fpp0.1_nostats.snappy.parquet",
        ),
    ],
)
@pytest.mark.parametrize(
    "predicate,expected_len",
    [
        ([[("str", "==", "FINDME")], [("fp64", "==", float(500))]], 2),
        ([("fixed_pt", "==", decimal.Decimal(float(500)))], 2),
        ([[("ui32", "==", np.uint32(500)), ("str", "==", "FINDME")]], 2),
        ([[("str", "==", "FINDME")], [("ui32", ">=", np.uint32(0))]], 1000),
        (
            [
                ("str", "!=", "FINDME"),
                ("fixed_pt", "==", decimal.Decimal(float(500))),
            ],
            0,
        ),
    ],
)
def test_parquet_bloom_filters(
    datadir, stats_fname, bloom_filter_fname, predicate, expected_len
):
    fname_stats = datadir / stats_fname
    fname_bf = datadir / bloom_filter_fname
    df_stats = cudf.read_parquet(fname_stats, filters=predicate).reset_index(
        drop=True
    )
    df_bf = cudf.read_parquet(fname_bf, filters=predicate).reset_index(
        drop=True
    )

    # Check if tables equal
    assert_eq(
        df_stats,
        df_bf,
    )

    # Check for table length
    assert_eq(
        len(df_stats),
        expected_len,
    )


@pytest.fixture(params=["cuda", "pool", "cuda_async"])
def memory_resource(request):
    import rmm

    current_mr = rmm.mr.get_current_device_resource()

    kind = request.param
    if kind == "cuda":
        mr = rmm.mr.CudaMemoryResource()
    elif kind == "pool":
        base = rmm.mr.CudaMemoryResource()
        free, _ = rmm.mr.available_device_memory()
        size = int(round(free * 0.5 / 256) * 256)
        mr = rmm.mr.PoolMemoryResource(base, size, size)
    elif kind == "cuda_async":
        mr = rmm.mr.CudaAsyncMemoryResource()

    rmm.mr.set_current_device_resource(mr)

    try:
        yield mr
    finally:
        rmm.mr.set_current_device_resource(current_mr)


@pytest.mark.parametrize("columns", [["r_reason_desc"], None])
def test_parquet_bloom_filters_alignment(datadir, columns, memory_resource):
    fname = datadir / "bloom_filter_alignment.parquet"
    filters = [("r_reason_desc", "==", "Did not like the color")]

    # Read expected table using pyarrow
    expected = pq.read_table(fname, columns=columns, filters=filters)

    # Read with cudf using the memory resource from fixture
    read = cudf.read_parquet(
        fname, columns=columns, filters=filters
    ).to_arrow()

    assert_eq(expected, read)


def test_parquet_reader_unsupported_compression(datadir):
    fname = datadir / "hadoop_lz4_compressed.parquet"

    with pytest.raises(
        RuntimeError,
        match="Unsupported Parquet compression type: LZ4",
    ):
        cudf.read_parquet(fname)


def test_parquet_reader_empty_compressed_page(datadir):
    fname = datadir / "empty_datapage_v2.parquet"

    df = cudf.DataFrame({"value": cudf.Series([None], dtype="float32")})
    assert_eq(cudf.read_parquet(fname), df)


@pytest.mark.parametrize("compression", ["brotli", "gzip", "snappy", "zstd"])
def test_parquet_decompression(
    set_decomp_env_vars, pdf_day_timestamps, compression
):
    # Skip if the compression type is not supported for reading
    if not is_supported_read_parquet(compression.upper()):
        pytest.skip(
            f"{compression} compression not supported for Parquet reading in this configuration"
        )

    # PANDAS returns category objects whereas cuDF returns hashes
    expect = pdf_day_timestamps.drop(columns=["col_category"])

    # Write the DataFrame to a Parquet file
    buffer = BytesIO()
    expect.to_parquet(buffer, compression=compression)

    # Read the Parquet file back into a DataFrame
    got = cudf.read_parquet(buffer)

    # As of pandas 3.0, empty default type of object isn't
    # necessarily equivalent to cuDF's empty default type of
    # pandas.StringDtype
    if len(pdf_day_timestamps.index) == 0:
        expect.index = expect.index.astype(got.index.dtype)
    assert_eq(expect, got)


def test_parquet_long_list(tmp_path):
    # This test generates int and string list columns, where each has a row that is very large.
    # When generated by the cudf writer these long rows are contained on a single page,
    # but when generated by pyarrow they span several pages.
    # This checks that the parquet reader works properly for long rows spanning several pages.

    # Generate ranges
    small_int_range = range(1, 10)
    large_int_range = range(1, 100000)

    # Create int lists
    small_int_list = list(small_int_range)
    large_int_list = list(large_int_range)

    # Create string lists from the int ranges
    small_string_list = list(map(str, small_int_range))
    large_string_list = list(map(str, large_int_range))

    # Create arrays (columns) of these lists
    # The long rows start offset from the beginning of the page, span several pages,
    # and have another row following it on the last page.
    list_array_int = pa.array([small_int_list, large_int_list, small_int_list])
    list_array_string = pa.array(
        [small_string_list, large_string_list, small_string_list]
    )

    # Generate the table containing the columns
    generated_table = pa.Table.from_arrays(
        [list_array_string, list_array_int],
        names=["string_column", "int_column"],
    )

    # Write the table to a parquet file using pyarrow
    file_name = tmp_path / "long_row_list_test.pq"
    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
    pq.write_table(
        generated_table,
        file_name,
        use_dictionary=False,
        data_page_size=65536,
        version="1.0",
        write_page_index=False,
    )

    # Make sure that the cudf reader matches the pandas reader for this data
    actual = cudf.read_parquet(file_name)
    expected = pd.read_parquet(file_name)
    assert_arrow_table_equal(pa.Table.from_pandas(expected), actual.to_arrow())


@pytest.mark.parametrize(
    "index",
    [range(1, 11), list(range(1, 11)), range(1, 11, 2)],
    ids=["RangeIndex", "IntIndex", "StridedRange"],
)
@pytest.mark.parametrize("write_index", [False, True, None])
@pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "empty"])
def test_dataframe_parquet_roundtrip(index, write_index, empty):
    if empty:
        data = {}
    else:
        data = {"a": [i * 2 for i in index]}
    df = cudf.DataFrame(data=data, index=index)
    pf = pd.DataFrame(data=data, index=index)
    gpu_buf = BytesIO()
    cpu_buf = BytesIO()

    df.to_parquet(gpu_buf, index=write_index)
    pf.to_parquet(cpu_buf, index=write_index)
    gpu_table = pq.read_table(gpu_buf)
    cpu_table = pq.read_table(cpu_buf)
    metadata_equal = (
        gpu_table.schema.pandas_metadata == cpu_table.schema.pandas_metadata
    )
    assert metadata_equal

    gpu_read = cudf.read_parquet(gpu_buf)
    cpu_read = cudf.read_parquet(cpu_buf)
    assert_eq(gpu_read, cpu_read)


def test_read_many_colchunks_with_threadpool():
    nrows = 1000
    ncols = 100
    table_data = {}
    for i in range(ncols):
        table_data[f"col_{i:04d}"] = [f"row_{j}" for j in range(nrows)]
    expected = pa.table(table_data)
    buffer = BytesIO()
    pq.write_table(expected, buffer, row_group_size=100, write_page_index=True)

    assert_eq(expected, cudf.read_parquet(buffer))


def test_parquet_decode_column_index_thrift_bool_list(datadir):
    fname = datadir / "column_index_thrift_bool_list.parquet"
    cudf.read_parquet(fname)


def test_read_parquet_case_insensitive():
    df = cudf.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "a": ["abc", "def", "ghi", "jkl", "mno"],
        }
    )
    buf = BytesIO()
    df.to_parquet(buf)

    result = cudf.read_parquet(
        buf,
        filters=[("A", "<=", 3)],
        columns=["a", "b"],
        case_sensitive_names=False,
    )
    expected = cudf.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
    assert_eq(result, expected)

    result = cudf.read_parquet(
        buf,
        filters=[("A", "<=", 3)],
        columns=["b"],
        case_sensitive_names=False,
    )
    expected = cudf.DataFrame({"B": [10, 20, 30]})
    assert_eq(result, expected)

    result = cudf.read_parquet(
        buf,
        filters=[("a", "<=", 3)],
        columns=["B"],
        case_sensitive_names=False,
    )
    expected = cudf.DataFrame({"B": [10, 20, 30]})
    assert_eq(result, expected)

    result = cudf.read_parquet(
        buf,
        filters=[("a", "<=", 3)],
        columns=["b"],
        case_sensitive_names=False,
    )
    expected = cudf.DataFrame({"B": [10, 20, 30]})
    assert_eq(result, expected)

    result = cudf.read_parquet(
        buf,
        filters=[("a", "<=", "ghi")],
        columns=["B"],
        case_sensitive_names=True,
    )
    expected = cudf.DataFrame({"B": [10, 20, 30]})
    assert_eq(result, expected)


def test_read_parquet_case_insensitive_error():
    df = cudf.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "a": ["abc", "def", "ghi", "jkl", "mno"],
        }
    )
    buf = BytesIO()
    df.to_parquet(buf)

    with pytest.raises(UserWarning):
        cudf.read_parquet(
            buf,
            filters=[("a", "<=", "ghi")],
            columns=["B"],
            case_sensitive_names=False,
        )


def test_read_parquet_case_insensitive_structs():
    struct_data = [
        {"a": 1, "b": {"a_a": 10, "b_b": 20}, "c": 2},
        {"a": 3, "b": {"a_a": 30, "b_b": 40}, "c": 4},
        {"a": 5, "b": {"a_a": 50, "b_b": None}, "c": 6},
        {"a": 7, "b": None, "c": 8},
    ]
    table = pa.Table.from_pydict(
        {"struct": struct_data, "id": [10, 20, 30, 40]}
    )
    buf = BytesIO()
    pq.write_table(table, buf)

    result = cudf.read_parquet(
        buf,
        columns=["STRUCT.b.A_a"],
        case_sensitive_names=False,
    )
    expected = cudf.read_parquet(
        buf,
        columns=["struct.b.a_a"],
    )
    assert_eq(result, expected)

    result = cudf.read_parquet(
        buf,
        columns=["STRUCT.A"],
        filters=[("iD", "<=", 20)],
        case_sensitive_names=False,
    )
    expected = cudf.read_parquet(
        buf,
        columns=["struct.a"],
        filters=[("id", "<=", 20)],
    )
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "host_decompression",
    ["OFF", "ON"],
)
def test_read_parquet_snappy_malformed_copy_elem(
    datadir, monkeypatch, host_decompression
):
    monkeypatch.setenv("LIBCUDF_NVCOMP_POLICY", "OFF")
    monkeypatch.setenv("LIBCUDF_HOST_DECOMPRESSION", host_decompression)
    fname = datadir / "invalid-snappy.parquet"
    match_string = (
        "Error during decompression"
        if host_decompression == "OFF"
        else "Snappy Decompression failed"
    )
    with pytest.raises(RuntimeError, match=match_string):
        cudf.read_parquet(fname)


@pytest.mark.parametrize(
    "filename",
    ["one_null_row_dict.parquet", "one_null_row_nodict.parquet"],
)
def test_parquet_reader_one_null_row(datadir, filename):
    cudf.read_parquet(datadir / filename)
