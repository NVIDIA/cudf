# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import gzip
import os
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
    expect_warning_if,
)


@pytest.fixture(params=["auto", "cudf", "pandas"])
def engine(request):
    return request.param


@pytest.fixture(params=[True, False])
def lines(request):
    return request.param


@pytest.fixture(params=[0, 10])
def pdf(request):
    rng = np.random.default_rng(seed=0)
    types = NUMERIC_TYPES + DATETIME_TYPES + ["bool"]
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.DataFrame(
        {
            f"col_{typ}": rng.integers(0, nrows, nrows).astype(typ)
            for typ in types
        }
    )
    test_pdf.index.name = "test_index"
    return test_pdf


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame(pdf)


@pytest.fixture(params=[0, 10])
def gdf_writer_types(request):
    # datetime64[us], datetime64[ns] are unsupported due to a bug in parser
    types = [
        *NUMERIC_TYPES,
        "datetime64[s]",
        "datetime64[ms]",
        *TIMEDELTA_TYPES,
        "bool",
        "str",
    ]
    typer = {"col_" + val: val for val in types}
    ncols = len(types)
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = cudf.DataFrame(
        np.ones((nrows, ncols)),
        columns=pd.Index(typer.keys()),
    )

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype(typer)

    return test_pdf


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("index", [True, False])
# tests limited to compressions formats supported by pandas and cudf: bz2, gzip, zip, zstd
@pytest.mark.parametrize("compression", ["bz2", "gzip", "zip", "zstd", None])
@pytest.mark.parametrize("orient", ["columns", "records", "table", "split"])
def test_json_reader(index, compression, orient, pdf, tmp_path):
    skip_reason = f"{index=} is not valid with {orient=}"
    if index is False and orient != "split":
        pytest.skip(skip_reason)
    if index is True and orient not in ("split", "table", "index", "columns"):
        pytest.skip(skip_reason)
    path_df = tmp_path / "test_df.json"
    path_series = tmp_path / "test_series.json"
    pdf.to_json(
        path_df,
        index=index,
        compression=compression,
        orient=orient,
        date_format="iso",
    )
    pdf["col_int32"].to_json(
        path_series,
        index=index,
        compression=compression,
        orient=orient,
        date_format="iso",
    )
    expect_df = pd.read_json(path_df, orient=orient, compression=compression)
    got_df = cudf.read_json(path_df, orient=orient, compression=compression)
    if len(expect_df) == 0:
        expect_df = expect_df.reset_index(drop=True)
    if len(got_df) == 0:
        got_df = got_df.reset_index(drop=True)

    assert_eq(expect_df, got_df, check_categorical=False)

    # Only these orients are allowed for Series, but isn't enforced by Pandas
    if orient in ("split", "records", "index"):
        expect_series = pd.read_json(
            path_series, orient=orient, compression=compression, typ="series"
        )
        got_series = cudf.read_json(
            path_series, orient=orient, compression=compression, typ="series"
        )
        if len(expect_series) == 0:
            expect_series = expect_series.reset_index(drop=True)
        if len(got_df) == 0:
            got_series = got_series.reset_index(drop=True)

        assert_eq(expect_series, got_series)


@pytest.mark.filterwarnings("ignore:Can't infer compression")
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_writer(tmp_path, pdf, gdf):
    pdf_df_fname = tmp_path / "pdf_df.json"
    gdf_df_fname = tmp_path / "gdf_df.json"

    pdf.to_json(pdf_df_fname, date_format="iso")
    gdf.to_json(gdf_df_fname, date_format="iso")

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    expect_df = pd.read_json(pdf_df_fname)
    got_df = pd.read_json(gdf_df_fname)

    assert_eq(expect_df, got_df)

    for column in pdf.columns:
        pdf_series_fname = tmp_path / f"{column}_pdf_series.json"
        gdf_series_fname = tmp_path / f"{column}_gdf_series.json"

        pdf[column].to_json(pdf_series_fname, date_format="iso")
        gdf[column].to_json(gdf_series_fname, date_format="iso")

        assert os.path.exists(pdf_series_fname)
        assert os.path.exists(gdf_series_fname)

        expect_series = pd.read_json(pdf_series_fname, typ="series")
        got_series = pd.read_json(gdf_series_fname, typ="series")

        assert_eq(expect_series, got_series)

        # Make sure results align for regular strings, not just files
        pdf_string = pdf[column].to_json(date_format="iso")
        gdf_string = pdf[column].to_json(date_format="iso")
        assert_eq(pdf_string, gdf_string)


def test_cudf_json_writer(pdf, lines):
    # removing datetime column because pandas doesn't support it
    for col_name in pdf.columns:
        if "datetime" in col_name:
            pdf.drop(col_name, axis=1, inplace=True)
    gdf = cudf.DataFrame(pdf)
    pdf_string = pdf.to_json(orient="records", lines=lines)
    gdf_string = gdf.to_json(orient="records", lines=lines, engine="cudf")

    assert_eq(pdf_string, gdf_string)

    gdf_string = gdf.to_json(
        orient="records", lines=lines, engine="cudf", rows_per_chunk=8
    )

    assert_eq(pdf_string, gdf_string)


@pytest.mark.parametrize("force_ascii", [True, False])
def test_cudf_json_writer_force_ascii(force_ascii):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["4", "5", "\U0001f331"]})
    gdf = cudf.DataFrame(pdf)

    pdf_string = pdf.to_json(
        orient="records", lines=True, force_ascii=force_ascii
    )
    gdf_string = gdf.to_json(
        orient="records",
        lines=True,
        engine="cudf",
        force_ascii=force_ascii,
    )

    assert_eq(pdf_string, gdf_string)


def test_cudf_json_writer_read(gdf_writer_types):
    dtypes = {
        col_name: col_name[len("col_") :]
        for col_name in gdf_writer_types.columns
    }
    gdf_string = gdf_writer_types.to_json(
        orient="records", lines=True, engine="cudf"
    )
    gdf2 = cudf.read_json(
        StringIO(gdf_string),
        lines=True,
        engine="cudf",
        dtype=dict(dtypes),
    )
    pdf2 = pd.read_json(StringIO(gdf_string), lines=True, dtype=dict(dtypes))

    # Bug in pandas https://github.com/pandas-dev/pandas/issues/28558
    if pdf2.empty:
        pdf2.reset_index(drop=True, inplace=True)

    # Pandas moved to consistent datetimes parsing format:
    # https://pandas.pydata.org/docs/dev/whatsnew/v2.0.0.html#datetimes-are-now-parsed-with-a-consistent-format
    for unit in ["s", "ms"]:
        if f"col_datetime64[{unit}]" in pdf2.columns:
            pdf2[f"col_datetime64[{unit}]"] = (
                pd.to_datetime(pdf2[f"col_datetime64[{unit}]"], format="mixed")
                .dt.tz_localize(None)
                .astype(f"datetime64[{unit}]")
            )
    assert_eq(pdf2, gdf2)


@pytest.mark.parametrize(
    "jsonl_string, expected",
    [
        # fixed width
        ("""{"a":10, "b":1.1}\n {"a":20, "b":2.1}\n""", None),
        # simple list
        ("""{"a":[1, 2, 3], "b":1.1}\n {"a":[]}\n""", None),
        # simple struct
        ("""{"a":{"c": 123 }, "b":1.1}\n {"a": {"c": 456}}\n""", None),
        # list of lists
        ("""{"a":[[], [1, 2], [3, 4]], "b":1.1}\n""", None),
        ("""{"a":[null, [1, 2], [null, 4]], "b":1.1}\n""", None),
        # list of structs
        # error ("""{"a":[null, {}], "b":1.1}\n""", None),
        (
            """{"a":[null, {"L": 123}], "b":1.0}\n {"b":1.1}\n {"b":2.1}\n""",
            None,
        ),
        (
            """{"a":[{"L": 123}, null], "b":1.0}\n {"b":1.1}\n {"b":2.1}\n""",
            None,
        ),
        # struct of lists
        (
            """{"a":{"L": [1, 2, 3]}, "b":1.1}\n {"a": {"L": [4, 5, 6]}}\n""",
            None,
        ),
        ("""{"a":{"L": [1, 2, null]}, "b":1.1}\n {"a": {"L": []}}\n""", None),
        # struct of structs
        (
            """{"a":{"L": {"M": 123}}, "b":1.1}
               {"a": {"L": {"M": 456}}}\n""",
            None,
        ),
        (
            """{"a":{"L": {"M": null}}, "b":1.1}\n {"a": {"L": {}}}\n""",
            """{"a":{"L": {}}, "b":1.1}\n {"a": {"L": {}}}\n""",
        ),
        # list of structs of lists
        ("""{"a":[{"L": [1, 2, 3]}, {"L": [4, 5, 6]}], "b":1.1}\n""", None),
        ("""{"a":[{"L": [1, 2, null]}, {"L": []}], "b":1.1}\n""", None),
        # struct of lists of structs
        ("""{"a":{"L": [{"M": 123}, {"M": 456}]}, "b":1.1}\n""", None),
        (
            """{"a":{"L": [{"M": null}, {}]}, "b":1.1}\n""",
            """{"a":{"L": [{}, {}]}, "b":1.1}\n""",
        ),
        # empty structs
        ("""{"A": null}\n {"A": {}}\n {}""", """{}\n{"A":{}}\n{}\n"""),
        (
            """{"A": {"B": null}}\n {"A": {"B": {}}}\n {"A": {}}""",
            """{"A":{}}\n{"A":{"B":{}}}\n{"A":{}}\n""",
        ),
    ],
)
def test_cudf_json_roundtrip(jsonl_string, expected):
    gdf = cudf.read_json(
        StringIO(jsonl_string),
        lines=True,
        engine="cudf",
        # dtype=dict(dtypes),
    )
    expected = jsonl_string if expected is None else expected
    gdf_string = gdf.to_json(
        orient="records", lines=True, engine="cudf", include_nulls=False
    )
    assert_eq(gdf_string, expected.replace(" ", ""))


@pytest.mark.parametrize("sink", ["string", "file"])
def test_cudf_json_writer_sinks(sink, tmp_path):
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    target = None
    if sink == "string":
        target = StringIO()
    elif sink == "file":
        target = tmp_path / "test_df.json"
    df.to_json(target, engine="cudf")
    if sink == "string":
        assert (
            target.getvalue() == '[{"a":1,"b":4},{"a":2,"b":5},{"a":3,"b":6}]'
        )
    elif sink == "file":
        assert os.path.exists(target)
        with open(target, "r") as f:
            assert f.read() == '[{"a":1,"b":4},{"a":2,"b":5},{"a":3,"b":6}]'


@pytest.fixture(params=["filepath", "pathobj", "bytes_io", "string_io", "url"])
def json_input(request, tmp_path):
    input_type = request.param
    buffer = "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]\n"
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    fname = json_dir / "test_df.json"
    if not os.path.isfile(fname):
        with open(str(fname), "w") as fp:
            fp.write(buffer)

    if input_type == "filepath":
        return str(fname)
    if input_type == "pathobj":
        return Path(fname)
    if input_type == "bytes_io":
        return BytesIO(buffer.encode())
    if input_type == "string_io":
        return StringIO(buffer)
    if input_type == "url":
        return Path(fname).as_uri()


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_lines_basic(json_input, engine):
    can_warn = isinstance(json_input, str) and not json_input.endswith(".json")
    with expect_warning_if(can_warn):
        cu_df = cudf.read_json(json_input, engine=engine, lines=True)
    # io types must seek to the beginning before you can read again
    if hasattr(json_input, "seek"):
        json_input.seek(0)
    with expect_warning_if(can_warn):
        pd_df = pd.read_json(json_input, lines=True)

    assert all(cu_df.dtypes == ["int64", "int64", "int64"])
    for cu_col, pd_col in zip(cu_df.columns, pd_df.columns, strict=True):
        assert str(cu_col) == str(pd_col)
        np.testing.assert_array_equal(pd_df[pd_col], cu_df[cu_col].to_numpy())


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_nonexistent_json_correct_error(engine):
    json_input = "doesnotexist.json"
    with pytest.raises(FileNotFoundError):
        cudf.read_json(json_input, engine=engine)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_lines_multiple(tmp_path, json_input, engine):
    if engine == "pandas":
        pytest.skip("pandas engine does not support multiple files")
    tmp_file1 = tmp_path / "MultiInputs1.json"
    tmp_file2 = tmp_path / "MultiInputs2.json"

    with expect_warning_if(
        isinstance(json_input, str) and not json_input.endswith(".json")
    ):
        pdf = pd.read_json(json_input, lines=True)
    pdf.to_json(tmp_file1, compression="infer", lines=True, orient="records")
    pdf.to_json(tmp_file2, compression="infer", lines=True, orient="records")

    cu_df = cudf.read_json([tmp_file1, tmp_file2], engine=engine, lines=True)
    pd_df = pd.concat([pdf, pdf])

    assert all(cu_df.dtypes == ["int64", "int64", "int64"])
    for cu_col, pd_col in zip(cu_df.columns, pd_df.columns, strict=True):
        assert str(cu_col) == str(pd_col)
        np.testing.assert_array_equal(pd_df[pd_col], cu_df[cu_col].to_numpy())


def test_json_read_directory(tmp_path, json_input, engine):
    if engine == "pandas":
        pytest.skip("pandas engine does not support directories")
    with expect_warning_if(
        isinstance(json_input, str) and not json_input.endswith(".json")
    ):
        pdf = pd.read_json(json_input, lines=True)
    json_dir = tmp_path / "jsons"
    json_dir.mkdir()
    pdf.to_json(
        json_dir / "MultiInputs1.json",
        compression="infer",
        lines=True,
        orient="records",
    )
    pdf.to_json(
        json_dir / "MultiInputs2.json",
        compression="infer",
        lines=True,
        orient="records",
    )
    pdf.to_json(
        json_dir / "MultiInputs3.json",
        compression="infer",
        lines=True,
        orient="records",
    )

    cu_df = cudf.read_json(json_dir, engine=engine, lines=True)
    pd_df = pd.concat([pdf, pdf, pdf])

    assert all(cu_df.dtypes == ["int64", "int64", "int64"])
    for cu_col, pd_col in zip(cu_df.columns, pd_df.columns, strict=True):
        assert str(cu_col) == str(pd_col)
        np.testing.assert_array_equal(pd_df[pd_col], cu_df[cu_col].to_numpy())


def test_json_lines_byte_range(json_input):
    # include the first row and half of the second row
    # should parse the first two rows
    will_warn = isinstance(json_input, str) and not json_input.endswith(
        ".json"
    )
    with expect_warning_if(will_warn):
        df = cudf.read_json(
            copy.deepcopy(json_input), lines=True, byte_range=(0, 15)
        )
    assert df.shape == (2, 3)

    # include half of the second row and half of the third row
    # should parse only the third row
    with expect_warning_if(will_warn):
        df = cudf.read_json(
            copy.deepcopy(json_input), lines=True, byte_range=(15, 10)
        )
    assert df.shape == (1, 3)

    # include half of the second row and entire third row
    # should parse only the third row
    with expect_warning_if(will_warn):
        df = cudf.read_json(
            copy.deepcopy(json_input), lines=True, byte_range=(15, 0)
        )
    assert df.shape == (1, 3)

    # include half of the second row till past the end of the file
    # should parse only the third row
    with expect_warning_if(will_warn):
        df = cudf.read_json(
            copy.deepcopy(json_input), lines=True, byte_range=(10, 50)
        )
    assert df.shape == (1, 3)


def test_json_lines_dtypes(json_input):
    with expect_warning_if(
        isinstance(json_input, str) and not json_input.endswith(".json")
    ):
        df = cudf.read_json(
            json_input, lines=True, dtype={1: "int", 2: "short", 0: "float"}
        )
    assert all(df.dtypes == ["float64", "int64", "int16"])


@pytest.mark.parametrize(
    "ext, out_comp, in_comp",
    [
        (".geez", "gzip", "gzip"),
        (".beez", "bz2", "bz2"),
        (".gz", "gzip", "infer"),
        (".bz2", "bz2", "infer"),
        (".data", None, "infer"),
        (".txt", None, None),
        ("", None, None),
    ],
)
def test_json_lines_compression(tmp_path, ext, out_comp, in_comp):
    fname = tmp_path / f"tmp_json_compression.{ext}"

    nrows = 20
    pd_df = pd.DataFrame(
        {
            "col1": np.arange(nrows, dtype=np.int32),
            "col2": np.arange(1, 1 + nrows, dtype=np.int32),
        }
    )
    pd_df.to_json(fname, compression=out_comp, lines=True, orient="records")

    cu_df = cudf.read_json(
        str(fname),
        compression=in_comp,
        lines=True,
        dtype={"col1": "int32", "col2": "int32"},
    )
    assert_eq(pd_df, cu_df)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_engine_selection():
    json = "[1, 2, 3]"

    # should use the cudf engine
    df = cudf.read_json(StringIO(json), lines=True)
    # column names are strings when parsing with cudf
    for col_name in df.columns:
        assert isinstance(col_name, str)

    # should use the pandas engine
    df = cudf.read_json(StringIO(json), lines=False, engine="pandas")
    # column names are ints when parsing with pandas
    for col_name in df.columns:
        assert isinstance(col_name, int)

    # should use the pandas engine
    df = cudf.read_json(StringIO(json), lines=True, engine="pandas")
    # column names are ints when parsing with pandas
    for col_name in df.columns:
        assert isinstance(col_name, int)


def test_json_bool_values():
    buffer = "[true,1]\n[false,false]\n[true,true]"
    cu_df = cudf.read_json(StringIO(buffer), lines=True)
    pd_df = pd.read_json(StringIO(buffer), lines=True)

    # types should be ['bool', 'int64']
    np.testing.assert_array_equal(pd_df.dtypes, cu_df.dtypes)
    np.testing.assert_array_equal(pd_df[0], cu_df["0"].to_numpy())
    # boolean values should be converted to 0/1
    np.testing.assert_array_equal(pd_df[1], cu_df["1"].to_numpy())

    cu_df = cudf.read_json(
        StringIO(buffer), lines=True, dtype={"0": "bool", "1": "long"}
    )
    np.testing.assert_array_equal(pd_df.dtypes, cu_df.dtypes)


def test_json_bad_protocol_string():
    test_string = StringIO('{"field": "s3://path"}')

    expect = pd.DataFrame([{"field": "s3://path"}])
    got = cudf.read_json(test_string, lines=True)

    assert_eq(expect, got)


def test_json_corner_case_with_escape_and_double_quote_char_with_pandas(
    tmp_path,
):
    fname = tmp_path / "tmp_json_escape_double_quote"

    pdf = pd.DataFrame(
        {
            "a": ['ab"cd', "\\\b", "\r\\", "'"],
            "b": ["a\tb\t", "\\", '\\"', "\t"],
            "c": ["aeiou", "try", "json", "cudf"],
        }
    )
    pdf.to_json(fname, compression="infer", lines=True, orient="records")

    df = cudf.read_json(
        fname, compression="infer", lines=True, orient="records"
    )
    pdf = pd.read_json(
        fname, compression="infer", lines=True, orient="records"
    )

    assert_eq(cudf.DataFrame(pdf), df)


def test_json_corner_case_with_escape_and_double_quote_char_with_strings():
    str_buffer = StringIO(
        """{"a":"ab\\"cd","b":"a\\tb\\t","c":"aeiou"}
           {"a":"\\\\\\b","b":"\\\\","c":"try"}
           {"a":"\\r\\\\","b":"\\\\\\"","c":"json"}
           {"a":"\'","b":"\\t","c":"cudf"}"""
    )

    df = cudf.read_json(
        str_buffer, compression="infer", lines=True, orient="records"
    )

    expected = cudf.DataFrame(
        {
            "a": ['ab"cd', "\\\b", "\r\\", "'"],
            "b": ["a\tb\t", "\\", '\\"', "\t"],
            "c": ["aeiou", "try", "json", "cudf"],
        }
    )
    assert_eq(df, expected)


def test_json_to_json_special_characters():
    df = cudf.DataFrame(
        {
            "'a'": ['ab"cd', "\\\b", "\r\\", "'"],
            "b": ["a\tb\t", "\\", '\\"', "\t"],
            "c": ["aeiou", "try", "json", "cudf"],
        }
    )

    actual = StringIO()
    df.to_json(actual, engine="cudf", lines=True, orient="records")
    expected = StringIO()
    df.to_pandas().to_json(expected, lines=True, orient="records")
    assert expected.getvalue() == actual.getvalue()


@pytest.mark.parametrize(
    "gdf,pdf",
    [
        (
            lambda: cudf.DataFrame(
                {
                    "int col": cudf.Series(
                        [1, 2, None, 2, 2323, 234, None], dtype="int64"
                    )
                }
            ),
            pd.DataFrame(
                {
                    "int col": pd.Series(
                        [1, 2, None, 2, 2323, 234, None], dtype=pd.Int64Dtype()
                    )
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "int64 col": cudf.Series(
                        [1, 2, None, 2323, None], dtype="int64"
                    ),
                    "string col": cudf.Series(
                        ["abc", "a", None, "", None], dtype="str"
                    ),
                    "float col": cudf.Series(
                        [0.234, None, 234234.2343, None, 0.0], dtype="float64"
                    ),
                    "bool col": cudf.Series(
                        [None, True, False, None, True], dtype="bool"
                    ),
                    "categorical col": cudf.Series(
                        [1, 2, 1, None, 2], dtype="category"
                    ),
                    "datetime col": cudf.Series(
                        [1231233, None, 2323234, None, 1],
                        dtype="datetime64[ns]",
                    ),
                    "timedelta col": cudf.Series(
                        [None, 34687236, 2323234, 1, None],
                        dtype="timedelta64[ns]",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "int64 col": pd.Series(
                        [1, 2, None, 2323, None], dtype=pd.Int64Dtype()
                    ),
                    "string col": pd.Series(
                        ["abc", "a", None, "", None], dtype=pd.StringDtype()
                    ),
                    "float col": pd.Series(
                        [0.234, None, 234234.2343, None, 0.0], dtype="float64"
                    ),
                    "bool col": pd.Series(
                        [None, True, False, None, True],
                        dtype=pd.BooleanDtype(),
                    ),
                    "categorical col": pd.Series(
                        [1, 2, 1, None, 2], dtype="category"
                    ),
                    "datetime col": pd.Series(
                        [1231233, None, 2323234, None, 1],
                        dtype="datetime64[ns]",
                    ),
                    "timedelta col": pd.Series(
                        [None, 34687236, 2323234, 1, None],
                        dtype="timedelta64[ns]",
                    ),
                }
            ),
        ),
    ],
)
def test_json_to_json_compare_contents(gdf, pdf):
    expected_json = pdf.to_json(
        lines=True, orient="records", date_format="iso"
    )
    with pytest.warns(UserWarning):
        actual_json = gdf().to_json(
            lines=True, orient="records", date_format="iso"
        )

    assert expected_json == actual_json


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_default_integer_bitwidth(default_integer_bitwidth, engine):
    buf = BytesIO()
    pd.DataFrame({"a": range(10)}).to_json(buf, lines=True, orient="records")
    buf.seek(0)
    df = cudf.read_json(buf, engine=engine, lines=True, orient="records")

    assert df["a"].dtype == np.dtype(f"i{default_integer_bitwidth // 8}")


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_default_integer_bitwidth_partial(default_integer_bitwidth, engine):
    buf = BytesIO()
    pd.DataFrame({"a": range(10), "b": range(10, 20)}).to_json(
        buf, lines=True, orient="records"
    )
    buf.seek(0)
    df = cudf.read_json(
        buf, engine=engine, lines=True, orient="records", dtype={"b": "i8"}
    )

    assert df["a"].dtype == np.dtype(f"i{default_integer_bitwidth // 8}")
    assert df["b"].dtype == np.dtype("i8")


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_default_integer_bitwidth_extremes(default_integer_bitwidth, engine):
    # Test that integer columns in json are _inferred_ as 32 bit columns.
    buf = StringIO(
        '{"u8":18446744073709551615, "i8":9223372036854775807}\n'
        '{"u8": 0, "i8": -9223372036854775808}'
    )
    df = cudf.read_json(buf, engine=engine, lines=True, orient="records")

    assert df["u8"].dtype == np.dtype(f"u{default_integer_bitwidth // 8}")
    assert df["i8"].dtype == np.dtype(f"i{default_integer_bitwidth // 8}")


def test_default_float_bitwidth(default_float_bitwidth):
    # Test that float columns in json are _inferred_ as 32 bit columns.
    df = cudf.read_json(
        StringIO('{"a": 1.0, "b": 2.5}\n{"a": 3.5, "b": 4.0}'),
        engine="cudf",
        lines=True,
        orient="records",
    )
    assert df["a"].dtype == np.dtype(f"f{default_float_bitwidth // 8}")
    assert df["b"].dtype == np.dtype(f"f{default_float_bitwidth // 8}")


def test_json_nested_basic():
    bytes_obj = BytesIO()
    data = {
        "c1": [{"f1": "sf11", "f2": "sf21"}, {"f1": "sf12", "f2": "sf22"}],
        "c2": [["l11", "l21"], ["l12", "l22"]],
    }
    pdf = pd.DataFrame(data)
    pdf.to_json(bytes_obj, orient="records")

    df = cudf.read_json(bytes_obj, engine="cudf", orient="records")
    bytes_obj.seek(0)
    pdf = pd.read_json(bytes_obj, orient="records")

    assert_eq(pdf, df)


@pytest.mark.parametrize(
    "data",
    [
        {
            "c1": [{"f1": "sf11", "f2": "sf21"}, {"f1": "sf12", "f2": "sf22"}],
            "c2": [["l11", "l21"], ["l12", "l22"]],
        },
        # Essential test case to handle omissions
        {
            "c1": [{"f2": "sf21"}, {"f1": "sf12"}],
            "c2": [["l11", "l21"], []],
        },
        # empty input
        {},
    ],
)
def test_json_nested_lines(data, lines):
    bio = BytesIO()
    pdf = pd.DataFrame(data)
    pdf.to_json(bio, orient="records", lines=lines)
    bio.seek(0)
    df = cudf.read_json(bio, engine="cudf", orient="records", lines=lines)
    bio.seek(0)
    pdf = pd.read_json(bio, orient="records", lines=lines)
    # In the second test-case we need to take a detour via pyarrow
    # Pandas omits "f1" in first row, so we have to enforce a common schema,
    # such that pandas would have the f1 member with null
    # Also, pyarrow chooses to select different ordering of a nested column
    # children though key-value pairs are correct.
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


def test_json_nested_data():
    json_str = (
        '[{"0":{},"2":{}},{"1":[[""],[]],"2":{"2":""}},'
        '{"0":{"a":"1"},"2":{"0":"W&RR=+I","1":""}}]'
    )
    df = cudf.read_json(StringIO(json_str), engine="cudf", orient="records")
    pdf = pd.read_json(StringIO(json_str), orient="records")
    pdf.columns = pdf.columns.astype("str")
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


def test_json_empty_types():
    json_str = """ {}
    {"a": [], "b": {}}
    {"a": []}
    {"b": {}}
    {"c": {"d": []}}
    {"e": [{}]}
    """
    df = cudf.read_json(
        StringIO(json_str), orient="records", lines=True
    ).to_pandas(arrow_type=True)
    pdf = pd.read_json(
        StringIO(json_str),
        orient="records",
        lines=True,
        dtype_backend="pyarrow",
    )

    # TODO: Remove this workaround after
    # https://github.com/pandas-dev/pandas/issues/65034 is fixed.
    pdf = pdf.astype(df.dtypes)
    assert_eq(df, pdf)


def test_json_types_data():
    # 0:<0:string,1:float>
    # 1:list<int>
    # 2:<0:bool>
    json_str = (
        '[{"0":null,"2":{}},'
        '{"1":[123],"0":{"0":"foo","1":123.4},"2":{"0":false}},'
        '{"0":{},"1":[],"2":{"0":null}}]'
    )
    df = cudf.read_json(StringIO(json_str), engine="cudf", orient="records")
    pdf = pd.read_json(StringIO(json_str), orient="records")
    pdf.columns = pdf.columns.astype("str")
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


@pytest.mark.parametrize(
    "col_type,json_str,expected_data",
    [
        # without quotes
        ("int", '[{"k": 1}, {"k": 2}, {"k": 3}, {"k": 4}]', [1, 2, 3, 4]),
        # with quotes
        ("int", '[{"k": "1"}, {"k": "2"}]', [1, 2]),
        # with quotes, mixed
        ("int", '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]', [1, 2, 3, 4]),
        # with quotes, null, mixed
        (
            "int",
            '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]',
            [1, 2, None, 4],
        ),
        # without quotes, null
        (
            "int",
            '[{"k": 1}, {"k": 2}, {"k": null}, {"k": 4}]',
            [1, 2, None, 4],
        ),
        # without quotes
        ("float", '[{"k": 1}, {"k": 2}, {"k": 3}, {"k": 4}]', [1, 2, 3, 4]),
        # with quotes
        ("float", '[{"k": "1"}, {"k": "2"}]', [1, 2]),
        # with quotes, mixed
        (
            "float",
            '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]',
            [1, 2, 3, 4],
        ),
        # with quotes, null, mixed
        (
            "float",
            '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]',
            [1, 2, None, 4],
        ),
        # with quotes, NAN
        (
            "float",
            '[{"k": "1"}, {"k": "2"}, {"k": NaN}, {"k": "4"}]',
            [1, 2, np.nan, 4],
        ),
        # without quotes
        ("str", '[{"k": 1}, {"k": 2}, {"k": 3}, {"k": 4}]', [1, 2, 3, 4]),
        # with quotes
        ("str", '[{"k": "1"}, {"k": "2"}]', [1, 2]),
        # with quotes, mixed
        ("str", '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]', [1, 2, 3, 4]),
        # with quotes, null, mixed
        (
            "str",
            '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]',
            [1, 2, None, 4],
        ),
        # without quotes, null
        (
            "str",
            '[{"k": 1}, {"k": 2}, {"k": null}, {"k": 4}]',
            [1, 2, None, 4],
        ),
    ],
)
def test_json_quoted_values_with_schema(col_type, json_str, expected_data):
    actual = cudf.read_json(
        StringIO(json_str),
        engine="cudf",
        orient="records",
        dtype={"k": col_type},
    )
    expected = cudf.DataFrame({"k": expected_data}, dtype=col_type)

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "col_type,json_str,expected_data",
    [
        # with quotes, mixed
        ("int", '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]', [1, 2, 3, 4]),
        # with quotes, null, mixed
        (
            "int",
            '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]',
            [1, 2, None, 4],
        ),
        # with quotes, mixed
        (
            "str",
            '[{"k": "1"}, {"k": "2"}, {"k": 3}, {"k": 4}]',
            ["1", "2", "3", "4"],
        ),
        # with quotes, null, mixed
        (
            "str",
            '[{"k": "1"}, {"k": "2"}, {"k": null}, {"k": 4}]',
            ["1", "2", None, "4"],
        ),
    ],
)
def test_json_quoted_values(col_type, json_str, expected_data):
    actual = cudf.read_json(
        StringIO(json_str),
        engine="cudf",
        orient="records",
        dtype={"k": col_type},
    )
    expected = cudf.DataFrame({"k": expected_data}, dtype=col_type)

    assert_eq(expected, actual)
    assert_eq(expected_data, actual.k.to_arrow().to_pylist())


@pytest.mark.parametrize(
    "keep_quotes,result",
    [
        (
            True,
            {
                "c1": [
                    {"f1": '"sf11"', "f2": '"sf21"'},
                    {"f1": '"sf12"', "f2": '"sf22"'},
                ],
                "c2": [['"l11"', '"l21"'], ['"l12"', '"l22"']],
            },
        ),
        (
            False,
            {
                "c1": [
                    {"f1": "sf11", "f2": "sf21"},
                    {"f1": "sf12", "f2": "sf22"},
                ],
                "c2": [["l11", "l21"], ["l12", "l22"]],
            },
        ),
    ],
)
def test_json_keep_quotes(keep_quotes, result):
    bytes_file = BytesIO()
    data = {
        "c1": [{"f1": "sf11", "f2": "sf21"}, {"f1": "sf12", "f2": "sf22"}],
        "c2": [["l11", "l21"], ["l12", "l22"]],
    }
    pdf = pd.DataFrame(data)
    pdf.to_json(bytes_file, orient="records", lines=True)

    actual = cudf.read_json(
        bytes_file,
        orient="records",
        lines=True,
        keep_quotes=keep_quotes,
    )
    expected = pd.DataFrame(result)

    assert_eq(actual, expected)


def test_json_dtypes_nested_data():
    # a: StructDtype({'a': StructDtype({'b': dtype('float64')}),
    #                 'b': dtype('int64')})
    # b: ListDtype(ListDtype(float64))
    actual_json_str = (
        '{"a":{"a":{"b":10.0},"b":11},"b":[[10.0,1.1],[12.0,23.0]]}\n'
        '{"a":{"a":{"b":107.0},"b":5},"b":[[10.0,11.2],[12.0,0.23]]}\n'
        '{"a":{"a":{"b":50.7},"b":2},"b":[[10.0,11.3],[12.0,2.3]]}\n'
        '{"a":{"a":{"b":1.2},"b":67},"b":[[6.0,7.0]]}\n'
        '{"a":{"a":{"b":40.1},"b":1090},"b":null}\n'
    )

    """
    In [3]: df
    Out[3]:
                                   a                             b
    0    {'a': {'b': 10.0}, 'b': 11}   [[10.0, 1.1], [12.0, 23.0]]
    1    {'a': {'b': 107.0}, 'b': 5}  [[10.0, 11.2], [12.0, 0.23]]
    2     {'a': {'b': 50.7}, 'b': 2}   [[10.0, 11.3], [12.0, 2.3]]
    3     {'a': {'b': 1.2}, 'b': 67}                  [[6.0, 7.0]]
    4  {'a': {'b': 40.1}, 'b': 1090}                          None
    """

    # a: StructDtype({'a': StructDtype({'b': dtype('int64')}),
    #                 'b': dtype('float64')})
    # b: ListDtype(ListDtype(int64))
    expected_json_str = (
        '{"a":{"a":{"b":10},"b":11.0},"b":[[10,1],[12,23]]}\n'
        '{"a":{"a":{"b":107},"b":5.0},"b":[[10,11],[12,0]]}\n'
        '{"a":{"a":{"b":50},"b":2.0},"b":[[10,11],[12,2]]}\n'
        '{"a":{"a":{"b":1},"b":67.0},"b":[[6,7]]}\n'
        '{"a":{"a":{"b":40},"b":1090.0},"b":null}\n'
    )

    """
    In [7]: df
    Out[7]:
                                  a                    b
    0    {'a': {'b': 10}, 'b': 11.0}  [[10, 1], [12, 23]]
    1    {'a': {'b': 107}, 'b': 5.0}  [[10, 11], [12, 0]]
    2     {'a': {'b': 50}, 'b': 2.0}  [[10, 11], [12, 2]]
    3     {'a': {'b': 1}, 'b': 67.0}             [[6, 7]]
    4  {'a': {'b': 40}, 'b': 1090.0}                 None
    """

    df = cudf.read_json(
        StringIO(actual_json_str),
        engine="cudf",
        orient="records",
        lines=True,
        dtype={
            "a": cudf.StructDtype(
                {
                    "a": cudf.StructDtype({"b": cudf.dtype("int64")}),
                    "b": cudf.dtype("float64"),
                }
            ),
            "b": cudf.ListDtype(cudf.ListDtype("int64")),
        },
    )

    pdf = pd.read_json(
        StringIO(expected_json_str),
        orient="records",
        lines=True,
    )

    assert_eq(df, pdf)

    pdf.columns = pdf.columns.astype("str")
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


@pytest.mark.parametrize(
    "tag, data",
    [
        (
            "normal",
            """\
{"a": 1, "b": 2}
{"a": 3, "b": 4}""",
        ),
        (
            "multiple",
            """\
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 }""",
        ),
        (
            "reordered",
            """\
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "c": 12 , "b" : [4, 5   ]}
    { "b" : [6      ],  "a": { "y" : 6}, "c": 13}
    { "c" : 14, "a": { "y" : 6}, "b" : [7      ]}
""",
        ),
        (
            "missing",
            """
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ]          }
    { "a": { "y" : 6}, "c": 13                  }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 }
""",
        ),
        (
            "dtype_mismatch",
            """\
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14.0 }""",
        ),
    ],
)
class TestNestedJsonReaderCommon:
    @pytest.mark.parametrize("chunk_size", [10, 100, 1024, 1024 * 1024])
    def test_chunked_nested_json_reader(self, tag, data, chunk_size):
        expected = cudf.read_json(StringIO(data), lines=True)

        source_size = len(data)
        chunks = []
        for chunk_start in range(0, source_size, chunk_size):
            chunks.append(
                cudf.read_json(
                    StringIO(data),
                    byte_range=[chunk_start, chunk_size],
                    lines=True,
                )
            )
        df = cudf.concat(chunks, ignore_index=True)
        assert expected.to_arrow().equals(df.to_arrow())

    def test_order_nested_json_reader(self, tag, data):
        expected = pd.read_json(StringIO(data), lines=True)
        target = cudf.read_json(StringIO(data), lines=True)
        # Using pyarrow instead of assert_eq because pandas
        # doesn't handle nested values comparisons correctly
        if tag == "dtype_mismatch":
            with pytest.raises(AssertionError):
                # pandas parses integer values in float representation
                # as integer
                assert pa.Table.from_pandas(expected).equals(target.to_arrow())
        elif tag == "missing":
            with pytest.raises(AssertionError):
                # pandas inferences integer with nulls as float64
                assert pa.Table.from_pandas(expected).equals(target.to_arrow())
        else:
            assert pa.Table.from_pandas(expected).equals(target.to_arrow())


def test_json_round_trip_gzip():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": ["abc", "def", "ghi"]})
    bio = BytesIO()
    with gzip.open(bio, mode="wb") as fo:
        with pytest.warns(UserWarning):
            df.to_json(fo, orient="records", lines=True)
    bio.seek(0)
    with gzip.open(bio, mode="rb") as fo:
        written_df = cudf.read_json(fo, orient="records", lines=True)
    assert_eq(written_df, df)

    # Testing writing from middle of the file.
    loc = bio.tell()

    with gzip.open(bio, mode="wb") as fo:
        fo.seek(loc)
        with pytest.warns(UserWarning):
            df.to_json(fo, orient="records", lines=True)
    bio.seek(loc)
    with gzip.open(bio, mode="rb") as fo:
        fo.seek(loc)
        written_df = cudf.read_json(fo, orient="records", lines=True)
    assert_eq(written_df, df)


@pytest.mark.parametrize(
    "data",
    [
        # # empty input
        # assert failing due to missing index size information
        "",
        "[]",
        "[]\n[]\n[]",
        # simple values
        """[1]\n[2]\n[3]""",
        """[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]""",
        # nulls
        """[1, 2, 3]\n[4, 5, null]\n[7, 8, 9]""",
        """[1, 2, 3]\n[4, 5, null]\n[7, 8, 9]\n[null, null, null]""",
        """[1, 2, 3]\n[4, 5, null]\n[]""",
        # missing
        """[1, 2, 3]\n[4, 5   ]\n[7, 8, 9]""",
        """[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9, 10]""",
        """[1, 2, 3]\n[4, 5, 6, {}]\n[7, 8, 9]""",
        """[1, 2, 3]\n[4, 5, 6, []]\n[7, 8, 9]""",
        """[1, 2, 3]\n[4, 5, 6, {"a": 10}]\n[7, 8, 9]""",
        """[1, 2, 3]\n[4, 5, 6, [10]]\n[7, 8, 9]""",
        # mixed
        """[1, 2, 3]\n[4, 5, {}]\n[7, 8, 9]""",
        """[1, 2, {}]\n[4, 5, 6]\n[7, 8, 9]""",
        """[1, 2, 3]\n[4, 5, [6]]\n[7, 8, 9]""",
        """[1, 2, [3]]\n[4, 5, 6]\n[7, 8, 9]""",
        # nested
        """[1, 2, [3]]\n[4, 5, [6]]\n[7, 8, [9]]""",
        """[1, 2, {"a": 3}]\n[4, 5, {"b": 6}]\n[7, 8, {"c": 9}]""",
        """[1, 2, [{"a": 3}, {"a": 3}]]
           [4, 5, [{"b": 6}, {"b": 6}, {}, {"b": 6}]]
           [7, 8, [{}]]""",
        """[1, 2, {"a": [3, 3, 3]}]
           [4, 5, {"b": [6, 6]}]
           [7, 8, {"c": 9}]""",
        """[1, 2, [{"a": 3}, {"a": null}]]
           [4, 5, [{"b": [6.0, 6, 06]}, {"b": [6]}, {}, {"b": null}]]
           [7, 8, [{}]]""",
    ],
)
def test_json_array_of_arrays(data, lines):
    data = data if lines else "[" + data.replace("\n", ",") + "]"
    pdf = pd.read_json(StringIO(data), orient="values", lines=lines)
    df = cudf.read_json(
        StringIO(data),
        engine="cudf",
        orient="values",
        lines=lines,
    )
    # if mixed with dict/list type, replace other types with None.
    if 2 in pdf.columns and any(
        pdf[2].apply(lambda x: isinstance(x, dict) or isinstance(x, list))
    ):
        pdf[2] = pdf[2].apply(
            lambda x: x if isinstance(x, dict) or isinstance(x, list) else None
        )
    # TODO: Replace string column names with integer column names
    # for values orient in cudf json reader
    pdf.rename(columns={name: str(name) for name in pdf.columns}, inplace=True)
    # assert_eq(pdf, df)
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=df.to_arrow().schema, safe=False
    )
    assert df.to_arrow().equals(pa_table_pdf)


@pytest.mark.parametrize(
    "jsonl_string",
    [
        # simple list with mixed types
        """{"a":[123, {}], "b":1.1}""",
        """{"a":[123, {"0": 123}], "b":1.0}\n {"b":1.1}\n {"b":2.1}""",
        """{"a":[{"L": 123}, 123], "b":1.0}\n {"b":1.1}\n {"b":2.1}""",
        """{"a":[123, {"0": 123}, 12.3], "b":1.0}\n {"b":1.1}\n {"b":2.1}""",
        """{"a":[123, {"0": 123}, null], "b":1.0}\n {"b":1.1}\n {"b":2.1}""",
        """{"a":["123", {"0": 123}], "b":1.0}\n {"b":1.1}\n {"b":2.1}""",
        """{"a":[{"0": 123}, "123"], "b":1.0}\n {"b":1.1}\n {"b":2.1}""",
        """{"a":["123", {"0": 123}, "123"], "b":1.0}\n {"b":1.1}""",
        """{"a":[123]}\n {"a":[{"0": 123}], "b":1.0}\n {"b":1.1}""",
        """{"a":[{"0": 123}]}\n {"a":[123], "b":1.0}\n {"b":1.1}""",
        """{"a":[{"0": 123}]}\n {"a": []}\n {"a":[123], "b":1.0}\n{"b":1.1}""",
        """{"b":1.0, "a":[{"0": 123}]}\n {"a":[123]}\n {"b":1.1}\n{"a": []}""",
        """{"a": []}\n {"a":[{"0": 123}]}\n {"a":[123], "b":1.0}\n{"b":1.1}""",
        """{"a": []}\n {"a":[123], "b":1.0}\n {"a":[{"0": 123}]}\n{"b":1.1}""",
        # nested list with mixed types
        """{"a":[123, [{"0": 123}, {}]], "b":1.0}
           {"b":1.1}
           {"a":[]}
           {"a":[123]}
           {"a":[[123], []]}""",
        """{"a":[], "b":1.0}
           {"a":[[[456]]]}
           {"a":[[123]]}
           {"a":[123]}""",
        """{"a":[123], "b":1.0}
           {"b":1.1}
           {"b":2.1}
           {"a":[[[[[[]]]]]]}""",
        """{"a":[123], "b":1.0}
           {"a":[[[[[[]]]]]]}
           {"a":[[[[[[]]]]], [[[[[]]]]]]}
           {"a":[[[[[[]]]], [[[[]]]]]]}
           {"a":[[[[[[]]], [[[]]]]]]}
           {"a":[[[[[[]], [[]]]]]]}
           {"a":[[[[[[], 123, []]]]]]}""",
        # mixed elements in multiple columns
        """{"a":[123, {"0": 123}], "b":1.0}
           {"c": ["abc"], "b":1.1}
           {"c": ["abc", []] }""",
    ],
)
def test_json_nested_mixed_types_in_list(jsonl_string):
    # utility function for this test:
    # replace list elements with None if it has dict and non-dict (ignore None)
    def _replace_in_list(list_to_replace, replace_items):
        return [
            _replace_in_list(x, replace_items)
            if isinstance(x, list)
            else None
            if x in replace_items
            else x
            for x in list_to_replace
        ]

    def _replace_with_nulls(df, replace_items):
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: (
                        _replace_in_list(x, replace_items)
                        if isinstance(x, list)
                        else x
                    )
                )
        return df

    # both json lines and json string tested.
    json_string = "[" + jsonl_string.replace("\n", ",") + "]"
    pdf = pd.read_json(
        StringIO(jsonl_string),
        orient="records",
        lines=True,
        dtype_backend="pyarrow",
    )
    pdf2 = pd.read_json(
        StringIO(json_string),
        orient="records",
        lines=False,
        dtype_backend="pyarrow",
    )
    assert_eq(pdf, pdf2)
    # replace list elements with None if it has dict and non-dict
    # in above test cases, these items are mixed with dict/list items
    # so, replace them with None.
    pdf = _replace_with_nulls(pdf, [123, "123", 12.3, "abc"])
    gdf = cudf.read_json(
        StringIO(jsonl_string),
        orient="records",
        lines=True,
    )
    gdf2 = cudf.read_json(
        StringIO(json_string),
        engine="cudf",
        orient="records",
        lines=False,
    )
    if """[{"0": 123}, {}]""" not in jsonl_string:
        # {} in pandas is represented as {"0": None} in cudf
        actual_gdf = gdf.to_pandas(arrow_type=True)
        # TODO: Remove this workaround after
        # https://github.com/pandas-dev/pandas/issues/65034 is fixed.
        pdf = pdf.astype(actual_gdf.dtypes)
        assert_eq(actual_gdf, pdf)

        actual_gdf2 = gdf2.to_pandas(arrow_type=True)
        assert_eq(actual_gdf2, pdf)
    pa_table_pdf = pa.Table.from_pandas(
        pdf, schema=gdf.to_arrow().schema, safe=False
    )
    assert gdf.to_arrow().equals(pa_table_pdf)
    assert gdf2.to_arrow().equals(pa_table_pdf)


@pytest.mark.parametrize(
    "jsonl_string",
    [
        # mixed type in list (in different order)
        """{"a":[[{"0": 123}, {}], {"1": 321}], "b":1.0}""",
        """{"a":[{"1": 321}, [{"0": 123}, {}], ], "b":1.0}""",
        """{"a":[123, [{"0": 123}, {}], {"1": 321}], "b":1.0}""",
        """{"a":[null, [{"0": 123}, {}], {"1": 321}], "b":1.0}""",
        # mixed type in struct (in different order)
        """{"a": {"b": {"0": 123}, "c": {"1": 321}}, "d":1.0}
           {"a": {"b": {"0": 123}, "c": [123, 123]}, "d":1.0}""",
        """{"a": {"b": {"0": 123}, "c": [123, 123]}, "d":1.0}
           {"a": {"b": {"0": 123}, "c": {"1": 321}}, "d":1.0}""",
        """{"a": {"b": {"0": 123}, "c": null}, "d":1.0}
           {"a": {"b": {"0": 123}, "c": {"1": 321}}, "d":1.0}
           {"a": {"b": {"0": 123}, "c": [123, 123]}, "d":1.0}""",
        """{"a": {"b": {"0": 123}, "c": 123}, "d":1.0}
           {"a": {"b": {"0": 123}, "c": {"1": 321}}, "d":1.0}
           {"a": {"b": {"0": 123}, "c": [123, 123]}, "d":1.0}""",
    ],
)
def test_json_nested_mixed_types_error(jsonl_string):
    # mixing list and struct should raise an exception
    with pytest.raises(RuntimeError):
        cudf.read_json(
            StringIO(jsonl_string),
            orient="records",
            lines=True,
        )


@pytest.mark.parametrize("on_bad_lines", ["error", "recover", "abc"])
def test_json_reader_on_bad_lines(on_bad_lines):
    json_input = StringIO(
        '{"a":1,"b":10}\n{"a":2,"b":11}\nabc\n{"a":3,"b":12}\n'
    )
    if on_bad_lines == "error":
        with pytest.raises(RuntimeError):
            cudf.read_json(
                json_input,
                lines=True,
                orient="records",
                on_bad_lines=on_bad_lines,
            )
    elif on_bad_lines == "recover":
        actual = cudf.read_json(
            json_input, lines=True, orient="records", on_bad_lines=on_bad_lines
        )
        expected = cudf.DataFrame(
            {"a": [1, 2, None, 3], "b": [10, 11, None, 12]}
        )
        assert_eq(actual, expected)
    else:
        with pytest.raises(TypeError):
            cudf.read_json(
                json_input,
                lines=True,
                orient="records",
                on_bad_lines=on_bad_lines,
            )


def test_chunked_json_reader():
    df = cudf.DataFrame(
        {
            "a": ["aaaa"] * 1_000_000,
            "b": range(1_000_000),
        }
    )
    buf = BytesIO()
    df.to_json(buf, lines=True, orient="records", engine="cudf")
    buf.seek(0)
    df = df.to_pandas()
    with cudf.option_context("io.json.low_memory", True):
        gdf = cudf.read_json(buf, lines=True)
    assert_eq(df, gdf)


# compression formats limited to those supported by both reader and writer
@pytest.mark.parametrize("compression", ["gzip", "snappy", "zstd"])
def test_roundtrip_compression(compression, tmp_path):
    expected = cudf.DataFrame({"a": [1], "b": ["2"]})
    fle = BytesIO()
    expected.to_json(fle, engine="cudf", compression=compression)
    result = cudf.read_json(fle, engine="cudf", compression=compression)
    assert_eq(result, expected)
