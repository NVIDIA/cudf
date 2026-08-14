# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import string
from functools import partial

import pytest

import polars as pl

from cudf_polars import execute_with_cudf
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_136,
    POLARS_VERSION_LT_138,
)


@pytest.fixture
def ldf(with_nulls):
    a = [
        "AbC",
        "de",
        "FGHI",
        "j",
        "kLm",
        "nOPq",
        "",
        "RsT",
        "sada",
        "uVw",
        "h",
        "Wıth ünιcοde",  # noqa: RUF001
    ]
    if with_nulls:
        a[4] = None
        a[-3] = None
    return pl.LazyFrame(
        {"a": a, "b": range(len(a)), "c": [str(i) for i in range(len(a))]}
    )


@pytest.fixture(params=[pl.Int8, pl.Int16, pl.Int32, pl.Int64])
def integer_type(request):
    return request.param


@pytest.fixture(params=[pl.Float32, pl.Float64])
def floating_type(request):
    return request.param


@pytest.fixture(params=[pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64])
def numeric_type(request):
    return request.param


@pytest.fixture
def str_to_integer_data(with_nulls):
    a = ["1", "2", "3", "4", "5", "6"]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": a})


@pytest.fixture
def str_to_float_data(with_nulls):
    a = [
        "1.1",
        "2.2",
        "3.3",
        "4.4",
        "5.5",
        "6.6",
        "inf",
        "+inf",
        "-inf",
        "Inf",
        "-Inf",
        "nan",
        "-1.234",
        "2e2",
    ]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": a})


@pytest.fixture
def str_from_integer_data(with_nulls, integer_type):
    a = [1, 2, 3, 4, 5, 6]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": pl.Series(a, dtype=integer_type)})


@pytest.fixture
def str_from_float_data(with_nulls, floating_type):
    a = [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        float("inf"),
        float("+inf"),
        float("-inf"),
        float("nan"),
    ]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": pl.Series(a, dtype=floating_type)})


slice_cases = [
    (1, 3),
    (0, 3),
    (0, 0),
    (-3, 1),
    (-100, 5),
    (1, 1),
    (100, 100),
    (-3, 4),
    (-3, 3),
]


@pytest.fixture(params=slice_cases)
def slice_column_data(ldf, request):
    start, length = request.param
    if length:
        return ldf.with_columns(
            pl.lit(start).alias("start"), pl.lit(length).alias("length")
        )
    else:
        return ldf.with_columns(pl.lit(start).alias("start"))


def test_supported_stringfunction_expression(engine: pl.GPUEngine, ldf):
    q = ldf.select(
        pl.col("a").str.starts_with("Z"),
        pl.col("a").str.ends_with("h").alias("endswith_h"),
        pl.col("a").str.to_lowercase().alias("lower"),
        pl.col("a").str.to_uppercase().alias("upper"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_unsupported_stringfunction(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.encode("hex"))

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_contains_re_non_strict(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.contains(".", strict=False))
    assert_gpu_result_equal(q, engine=engine)


def test_contains_re_non_strict_invalid(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.contains("[", strict=False))
    assert_gpu_result_equal(q, engine=engine)


def test_contains_empty_regex_with_null(engine: pl.GPUEngine):
    q = pl.LazyFrame({"a": ["abc", None, ""]}).select(pl.col("a").str.contains(""))
    assert_gpu_result_equal(q, engine=engine)


def test_contains_re_non_strict_libcudf_unsupported(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.contains("a{1000}", strict=False))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_contains_re_non_literal_raises(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.contains(pl.col("c"), literal=False))

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "substr",
    [
        "A",
        "de",
        ".*",
        "^a",
        "^A",
        "[^a-z]",
        "[a-z]{3,}",
        "^[A-Z]{2,}",
        "j|u",
    ],
)
def test_contains_regex(engine: pl.GPUEngine, ldf, substr):
    q = ldf.select(pl.col("a").str.contains(substr))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "literal", ["A", "de", "FGHI", "j", "kLm", "nOPq", "RsT", "uVw"]
)
def test_contains_literal(engine: pl.GPUEngine, ldf, literal):
    q = ldf.select(pl.col("a").str.contains(pl.lit(literal), literal=True))
    assert_gpu_result_equal(q, engine=engine)


def test_contains_column(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.contains(pl.col("a"), literal=True))
    assert_gpu_result_equal(q, engine=engine)


def test_contains_invalid(ldf):
    q = ldf.select(pl.col("a").str.contains("["))

    with pytest.raises(pl.exceptions.ComputeError):
        q.collect()
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(post_opt_callback=partial(execute_with_cudf, raise_on_fail=True))


@pytest.mark.parametrize("offset", [1, -1, 0, 100, -100])
def test_slice_scalars_offset(engine: pl.GPUEngine, ldf, offset):
    q = ldf.select(pl.col("a").str.slice(offset))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("offset,length", slice_cases)
def test_slice_scalars_length_and_offset(engine: pl.GPUEngine, ldf, offset, length):
    q = ldf.select(pl.col("a").str.slice(offset, length))
    assert_gpu_result_equal(q, engine=engine)


def test_slice_column(engine: pl.GPUEngine, slice_column_data):
    if "length" in slice_column_data.collect_schema():
        q = slice_column_data.select(
            pl.col("a").str.slice(pl.col("start"), pl.col("length"))
        )
    else:
        q = slice_column_data.select(pl.col("a").str.slice(pl.col("start")))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("name", ["head", "tail"])
def test_head_tail_expr_n(engine: pl.GPUEngine, name):
    df = pl.LazyFrame(
        {"a": ["abcdef", "hello", None, "xy", "abcd"], "n": [2, 0, 3, None, -1]}
    )
    q = df.select(getattr(pl.col("a").str, name)(pl.col("n")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.fixture
def ldf_split():
    return pl.LazyFrame({"a": ["a b", "a_b", "a_b_c", "a_b c_d", None]})


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize("by", ["_", " "])
def test_split_n(engine: pl.GPUEngine, ldf_split, n, by):
    q = ldf_split.select(pl.col("a").str.splitn(by, n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [1, 2, 10])
def test_split_exact(engine: pl.GPUEngine, ldf_split, n):
    q = ldf_split.select(pl.col("a").str.split_exact("_", n))
    assert_gpu_result_equal(q, engine=engine)


def test_split_exact_inclusive_unsupported(engine: pl.GPUEngine, ldf_split):
    q = ldf_split.select(pl.col("a").str.split_exact("_", 1, inclusive=True))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_split_exact_null_correct_children(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": ["a_b", None]})
    q = df.slice(1).select(pl.col("a").str.split_exact("_", 1))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("cache", [True, False], ids=lambda cache: f"{cache=}")
@pytest.mark.parametrize("strict", [True, False], ids=lambda strict: f"{strict=}")
@pytest.mark.parametrize("exact", [True, False], ids=lambda exact: f"{exact=}")
@pytest.mark.parametrize("format", ["%Y-%m-%d", None], ids=lambda format: f"{format=}")
@pytest.mark.parametrize(
    "values, has_invalid_row",
    [
        (["2024-01-01", "2023-12-31", "2023-06-15"], False),
        (["2024-01-01", "2023-12-31", None], False),
        (["2024-13-01", "2023-12-31", "2023-06-15"], True),
        (["2024-01-01", "foo", None], True),
        (["foo", "2023-06-15"], True),
        ([None, None, None], False),
    ],
    ids=["valid", "valid", "invalid", "invalid", "invalid", "valid"],
)
def test_to_datetime(
    engine: pl.GPUEngine, values, has_invalid_row, cache, strict, format, exact
):
    df = pl.DataFrame({"a": pl.Series(values, dtype=pl.String())})
    q = df.lazy().select(
        pl.col("a").str.strptime(
            pl.Datetime("ns"),
            format=format,
            cache=cache,
            strict=strict,
            exact=exact,
        )
    )
    if cache or not exact or (not strict and format is None):
        outcome = "translation_error"
    elif (values[0] == "foo" and format is None and strict) or (
        strict and has_invalid_row
    ):
        outcome = "collect_error"
    else:
        outcome = "success"

    if outcome == "translation_error":
        assert_ir_translation_raises(q, engine, NotImplementedError)
    elif outcome == "collect_error":
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect()
        if is_streaming_engine(engine):
            with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
                q.collect(engine=engine)
        else:
            with pytest.raises(pl.exceptions.InvalidOperationError):
                q.collect(engine=engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "target, repl",
    [("a", "a"), ("Wı", "☺"), ("FG", ""), ("doesnotexist", "blahblah")],  # noqa: RUF001
)
@pytest.mark.parametrize("n", [0, 3, -1])
def test_replace_literal(engine: pl.GPUEngine, ldf, target, repl, n):
    q = ldf.select(pl.col("a").str.replace(target, repl, literal=True, n=n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("target, repl", [("", ""), ("a", pl.col("a"))])
def test_replace_literal_unsupported(engine: pl.GPUEngine, ldf, target, repl):
    q = ldf.select(pl.col("a").str.replace(target, repl, literal=True))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_replace_re(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.replace("A", "a", literal=False))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "target,repl",
    [
        (["A", "de", "kLm", "awef"], "a"),
        (["A", "de", "kLm", "awef"], ""),
        (["A", "de", "kLm", "awef"], ["a", "b", "c", "d"]),
        (["A", "de", "kLm", "awef"], ["a", "b", "c", ""]),
        (
            pl.lit(pl.Series(["A", "de", "kLm", "awef"])),
            pl.lit(pl.Series(["a", "b", "c", "d"])),
        ),
    ],
)
def test_replace_many(engine: pl.GPUEngine, ldf, target, repl):
    q = ldf.select(pl.col("a").str.replace_many(target, repl))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "target,repl",
    [(["A", ""], ["a", "b"]), (pl.col("a").drop_nulls(), pl.col("a").drop_nulls())],
)
def test_replace_many_notimplemented(engine: pl.GPUEngine, ldf, target, repl):
    q = ldf.select(pl.col("a").str.replace_many(target, repl))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_replace_many_ascii_case(engine: pl.GPUEngine, ldf):
    q = ldf.select(
        pl.col("a").str.replace_many(["a", "b", "c"], "a", ascii_case_insensitive=True)
    )

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.skipif(POLARS_VERSION_LT_136, reason="leftmost arg added in 1.36")
def test_replace_many_leftmost(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.replace_many(["a", "b"], "x", leftmost=True))

    assert_ir_translation_raises(q, engine, NotImplementedError)


_strip_data = [
    "AbC",
    "123abc",
    "",
    " ",
    None,
    "aAaaaAAaa",
    " ab c ",
    "abc123",
    "    ",
    "\tabc\t",
    "\nabc\n",
    "\r\nabc\r\n",
    "\t\n abc \n\t",
    "!@#$%^&*()",
    "   abc!!!   ",
    "   abc\t\n!!!   ",
    "__abc__",
    "abc\n\n",
    "123abc456",
    "abcxyzabc",
]

strip_chars = [
    "a",
    "",
    " ",
    "\t",
    "\n",
    "\r\n",
    "!",
    "@#",
    "123",
    "xyz",
    "abc",
    "__",
    " \t\n",
    "abc123",
    None,
]


@pytest.fixture
def strip_ldf():
    return pl.DataFrame({"a": _strip_data}).lazy()


@pytest.fixture(params=strip_chars)
def to_strip(request):
    return request.param


def test_strip_chars(engine: pl.GPUEngine, strip_ldf, to_strip):
    q = strip_ldf.select(pl.col("a").str.strip_chars(to_strip))
    assert_gpu_result_equal(q, engine=engine)


def test_strip_chars_start(engine: pl.GPUEngine, strip_ldf, to_strip):
    q = strip_ldf.select(pl.col("a").str.strip_chars_start(to_strip))
    assert_gpu_result_equal(q, engine=engine)


def test_strip_chars_end(engine: pl.GPUEngine, strip_ldf, to_strip):
    q = strip_ldf.select(pl.col("a").str.strip_chars_end(to_strip))
    assert_gpu_result_equal(q, engine=engine)


def test_strip_chars_column(engine: pl.GPUEngine, strip_ldf):
    q = strip_ldf.select(pl.col("a").str.strip_chars(pl.col("a")))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_invalid_regex_raises():
    df = pl.LazyFrame({"a": ["abc"]})

    q = df.select(pl.col("a").str.contains(r"ab)", strict=True))

    with pytest.raises(pl.exceptions.ComputeError):
        q.collect()
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=pl.GPUEngine(executor="in-memory", raise_on_fail=True))


@pytest.mark.parametrize("pattern", ["a{1000}", "a(?i:B)", ""])
def test_unsupported_regex_raises(engine: pl.GPUEngine, pattern):
    df = pl.LazyFrame({"a": ["abc"]})

    q = df.select(pl.col("a").str.contains(pattern, strict=True))
    if pattern == "":
        assert_gpu_result_equal(q, engine=engine)
    else:
        assert_ir_translation_raises(q, engine, NotImplementedError)

    q = df.select(pl.col("a").str.count_matches(pattern))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_string_to_integer(engine: pl.GPUEngine, str_to_integer_data, integer_type):
    q = str_to_integer_data.select(pl.col("a").cast(integer_type))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("strict", [True, False])
def test_str_to_integer(engine: pl.GPUEngine, strict, integer_type):
    df = pl.LazyFrame({"a": ["1", "2", "-3", None, "0"]})
    q = df.select(pl.col("a").str.to_integer(dtype=integer_type, strict=strict))
    assert_gpu_result_equal(q, engine=engine)


def test_str_to_integer_non_strict_invalid_to_null(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": ["1", "invalid", "", None, "999999999999999999999999"]})
    q = df.select(pl.col("a").str.to_integer(dtype=pl.Int32, strict=False))
    assert_gpu_result_equal(q, engine=engine)


def test_str_to_integer_strict_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": ["1", "invalid"]})
    q = df.select(pl.col("a").str.to_integer(strict=True))
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)


def test_str_to_integer_base_unsupported(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": ["110", "101"]})
    q = df.select(pl.col("a").str.to_integer(base=2, strict=False))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_string_from_integer(engine: pl.GPUEngine, str_from_integer_data):
    q = str_from_integer_data.select(pl.col("a").cast(pl.String))
    assert_gpu_result_equal(q, engine=engine)


def test_string_to_float(engine: pl.GPUEngine, str_to_float_data, floating_type):
    q = str_to_float_data.select(pl.col("a").cast(floating_type))
    assert_gpu_result_equal(q, engine=engine)


def test_string_from_float(engine: pl.GPUEngine, request, str_from_float_data):
    if str_from_float_data.collect_schema()["a"] == pl.Float32:
        # libcudf will return a string representing the precision out to
        # a certain number of hardcoded decimal places. This results in
        # the fractional part being thrown away which causes discrepancies
        # for certain numbers. For instance, the float32 representation of
        # 1.1 is 1.100000023841858. When cast to a string, this will become
        # 1.100000024. But the float64 representation of 1.1 is
        # 1.1000000000000000888 which will result in libcudf truncating the
        # final value to 1.1.
        request.applymarker(pytest.mark.xfail(reason="libcudf truncation"))
    q = str_from_float_data.select(pl.col("a").cast(pl.String))

    # libcudf reads float('inf') -> "inf"
    # but polars reads float('inf') -> "Inf"
    q = q.select(pl.col("a").str.to_lowercase())
    assert_gpu_result_equal(q, engine=engine)


def test_string_to_numeric_invalid(engine, numeric_type):
    df = pl.LazyFrame({"a": ["a", "b", "c"]})
    q = df.select(pl.col("a").cast(numeric_type))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)


@pytest.mark.parametrize("ignore_nulls", [False, True])
@pytest.mark.parametrize("delimiter", ["", "/"])
def test_string_join(engine: pl.GPUEngine, ldf, ignore_nulls, delimiter):
    q = ldf.select(pl.col("a").str.join(delimiter, ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "fill",
    [
        0,
        1,
        2,
        5,
        999,
        -1,
        None,
    ],
)
@pytest.mark.parametrize(
    "input_strings",
    [
        ["1", "0"],
        ["123", "45"],
        ["", "0"],
        ["abc", "def"],
    ],
)
def test_string_zfill(engine: pl.GPUEngine, fill, input_strings):
    ldf = pl.LazyFrame({"a": input_strings})
    q = ldf.select(pl.col("a").str.zfill(fill))

    if fill is not None and fill < 0:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect()
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "fill",
    [5, 999],
)
def test_string_zfill_pl_129(engine: pl.GPUEngine, fill):
    ldf = pl.LazyFrame({"a": ["-1", "+2"]})
    q = ldf.select(pl.col("a").str.zfill(fill))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "fill",
    [
        0,
        1,
        2,
        5,
        999,
        -1,
        pytest.param(None, marks=pytest.mark.xfail(reason="None dtype")),
    ],
)
def test_string_zfill_column(engine: pl.GPUEngine, fill):
    ldf = pl.DataFrame(
        {
            "input_strings": ["1", "0", "123", "45", "", "0", "-1", "+2", "abc", "def"],
            "fill": [fill] * 10,
        }
    ).lazy()
    q = ldf.select(pl.col("input_strings").str.zfill(pl.col("fill")))
    if fill is not None and fill < 0:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect()
        q.collect(engine=engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


def test_string_zfill_column_null_widths(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "s": ["1", "0", "123", None, "45", "6"],
            "w": [3, None, 5, 3, None, 0],
        }
    )
    q = df.select(pl.col("s").str.zfill(pl.col("w")))
    assert_gpu_result_equal(q, engine=engine)


def test_string_zfill_forbidden_chars(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": ["Café", "345", "東京", None]})
    q = ldf.select(pl.col("a").str.zfill(3))
    # disallowed by libcudf
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)


@pytest.mark.parametrize(
    "width",
    [
        1,
        0,
        999,
        pytest.param(
            -1, marks=pytest.mark.xfail(reason="negative width fails before collect")
        ),
        pytest.param(
            None, marks=pytest.mark.xfail(reason="None width fails before collect")
        ),
    ],
)
@pytest.mark.parametrize(
    "char",
    [
        "*",
        "a",
        " ",
        pytest.param("", marks=pytest.mark.xfail(reason="length one characters only")),
        pytest.param(
            None, marks=pytest.mark.xfail(reason="None char fails before collect")
        ),
    ],
)
def test_string_pad_start(engine: pl.GPUEngine, width, char):
    df = pl.LazyFrame({"a": ["abc", "defg", "hij"]})
    q = df.select(pl.col("a").str.pad_start(width, char))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "width",
    [
        1,
        0,
        999,
        pytest.param(
            -1, marks=pytest.mark.xfail(reason="negative width fails before collect")
        ),
        pytest.param(
            None, marks=pytest.mark.xfail(reason="None width fails before collect")
        ),
    ],
)
@pytest.mark.parametrize(
    "char",
    [
        "*",
        "a",
        " ",
        pytest.param("", marks=pytest.mark.xfail(reason="length one characters only")),
        pytest.param(
            None, marks=pytest.mark.xfail(reason="None char fails before collect")
        ),
    ],
)
def test_string_pad_end(engine: pl.GPUEngine, width, char):
    df = pl.LazyFrame({"a": ["abc", "defg", "hij"]})
    q = df.select(pl.col("a").str.pad_end(width, char))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("ignore_nulls", [False, True])
@pytest.mark.parametrize("delimiter", ["", "-"])
def test_string_join_non_string_data(engine: pl.GPUEngine, ignore_nulls, delimiter):
    ldf = pl.LazyFrame({"a": [1, None, 3]})
    q = ldf.select(pl.col("a").str.join(delimiter, ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine)


def test_string_reverse(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.reverse())
    assert_gpu_result_equal(q, engine=engine)


def test_string_to_titlecase(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "quotes": [
                "'e.t. phone home'",
                "you talkin' to me?",
                "to infinity,and BEYOND!",
            ]
        }
    )
    q = df.with_columns(
        quotes_title=pl.col("quotes").str.to_titlecase(),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("tail", [1, 2, 999, -1, 0, None])
def test_string_tail(engine: pl.GPUEngine, ldf, tail):
    q = ldf.select(pl.col("a").str.tail(tail))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("head", [1, 2, 999, -1, 0, None])
def test_string_head(engine: pl.GPUEngine, ldf, head):
    q = ldf.select(pl.col("a").str.head(head))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("operation", ["head", "tail"])
@pytest.mark.parametrize("integer_type", [pl.Int32, pl.Int64])
def test_string_head_tail_by_column(engine: pl.GPUEngine, operation, integer_type):
    ldf = pl.LazyFrame(
        {
            "a": ["AbC", "de", "FGHI", None, "", "Wïth ünicode"],
            "n": pl.Series([1, -1, 0, 2, None, 999], dtype=integer_type),
        }
    )
    q = ldf.select(getattr(pl.col("a").str, operation)(pl.col("n")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("operation", ["head", "tail"])
@pytest.mark.parametrize(
    "a,n",
    [
        (
            pl.Series([], dtype=pl.String),
            pl.Series([], dtype=pl.Int64),
        ),
        (
            pl.Series([None, None], dtype=pl.String),
            pl.Series([1, -1], dtype=pl.Int64),
        ),
        (
            pl.Series(["abc", None], dtype=pl.String),
            pl.Series([None, None], dtype=pl.Int64),
        ),
    ],
)
def test_string_head_tail_by_column_shortcuts(engine: pl.GPUEngine, operation, a, n):
    ldf = pl.LazyFrame({"a": a, "n": n})
    q = ldf.select(getattr(pl.col("a").str, operation)(pl.col("n")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("ignore_nulls", [True, False])
@pytest.mark.parametrize("separator", ["*", ""])
def test_concat_horizontal(engine: pl.GPUEngine, ldf, ignore_nulls, separator):
    q = ldf.select(
        pl.concat_str(["a", "c"], separator=separator, ignore_nulls=ignore_nulls)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("ascii_case_insensitive", [True, False])
def test_contains_any(engine: pl.GPUEngine, ldf, ascii_case_insensitive):
    q = ldf.select(
        pl.col("a").str.contains_any(
            ["a", "b", "c"], ascii_case_insensitive=ascii_case_insensitive
        )
    )
    assert_gpu_result_equal(q, engine=engine)


def test_contains_any_empty(engine: pl.GPUEngine):
    # libcudf's ``contains_multiple`` asserts ``num_blocks > 0`` and crashes
    # on a zero-row input, so the expression short-circuits to an empty
    # bool column.
    ldf = pl.LazyFrame({"a": pl.Series([], dtype=pl.String)})
    q = ldf.select(pl.col("a").str.contains_any(["a", "b"]))
    assert_gpu_result_equal(q, engine=engine)


def test_count_matches(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.count_matches("a"))
    assert_gpu_result_equal(q, engine=engine)


def test_count_matches_literal_unsupported(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.count_matches("a", literal=True))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_strip_prefix(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.strip_prefix("A"))
    assert_gpu_result_equal(q, engine=engine)


def test_strip_suffix(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.strip_suffix("e"))
    assert_gpu_result_equal(q, engine=engine)


def test_strip_prefix_suffix_dupes(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": ["a", "aa", "ab", "bb", "b"]})

    q = ldf.select(pl.col("a").str.strip_prefix("a"))
    assert_gpu_result_equal(q, engine=engine)

    q = ldf.select(pl.col("a").str.strip_suffix("a"))
    assert_gpu_result_equal(q, engine=engine)

    q = ldf.select(pl.col("a").str.strip_prefix("b"))
    assert_gpu_result_equal(q, engine=engine)

    q = ldf.select(pl.col("a").str.strip_suffix("b"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.fixture
def ldf_jsonlike():
    return pl.LazyFrame(
        {"a": ['{"a":"1"}', None, '{"a":"2"}', '{"a":"2.1"}', '{"a":"true"}']}
    )


def test_json_decode(engine: pl.GPUEngine, ldf_jsonlike):
    q = ldf_jsonlike.select(pl.col("a").str.json_decode(pl.Struct({"a": pl.String()})))
    assert_gpu_result_equal(q, engine=engine)


def test_json_decode_empty(engine: pl.GPUEngine):
    # libcudf's ``read_json_from_string_column`` raises
    # "Generated token count exceeds the expected token count" on a
    # zero-row input, so the expression short-circuits to an empty
    # struct column with the requested schema.
    ldf = pl.LazyFrame({"a": pl.Series([], dtype=pl.String)})
    q = ldf.select(pl.col("a").str.json_decode(pl.Struct({"a": pl.Int64()})))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64()])
def test_json_decode_numeric_types(engine: pl.GPUEngine, dtype):
    ldf = pl.LazyFrame({"a": ['{"a": 1}', None, '{"a": 2}']})
    q = ldf.select(pl.col("a").str.json_decode(pl.Struct({"a": dtype})))
    assert_gpu_result_equal(q, engine=engine)


def test_json_decode_nested(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": ['{"a": {"b": 1}}', None]})
    q = ldf.select(
        pl.col("a").str.json_decode(pl.Struct({"a": pl.Struct({"b": pl.Int64()})}))
    )
    assert_gpu_result_equal(q, engine=engine)


def test_json_path_match(engine: pl.GPUEngine, ldf_jsonlike):
    q = ldf_jsonlike.select(pl.col("a").str.json_path_match("$.a"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.fixture
def ldf_find():
    return pl.LazyFrame(
        {
            "a": ["Crab", "Lobster", None, "Crustacean", "Cra|eb"],
            "pat": ["a[bc]", "b.t", "[aeiuo]", "(?i)A[BC]", r"\d"],
        }
    )


@pytest.mark.parametrize("pattern", ["a", "[", ""])
def test_find_literal_false_strict_false(engine: pl.GPUEngine, ldf_find, pattern):
    q = ldf_find.select(pl.col("a").str.find(pattern, literal=False, strict=False))
    assert_gpu_result_equal(q, engine=engine)


def test_find_literal_false_strict_false_libcudf_unsupported(
    engine: pl.GPUEngine, ldf_find
):
    q = ldf_find.select(pl.col("a").str.find("a{1000}", literal=False, strict=False))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("literal", [True, False])
@pytest.mark.parametrize("pattern", ["a|e", "a"])
def test_find_literal(engine: pl.GPUEngine, ldf_find, literal, pattern):
    q = ldf_find.select(pl.col("a").str.find(pattern, literal=literal))
    assert_gpu_result_equal(q, engine=engine)


def test_find_literal_false_column_unsupported(engine: pl.GPUEngine, ldf_find):
    q = ldf_find.select(pl.col("a").str.find(pl.col("pat"), literal=False))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.fixture
def ldf_extract():
    return pl.LazyFrame({"a": ["?!. 123 foo", None]})


@pytest.mark.parametrize("group_index", [1, 2])
def test_extract(engine: pl.GPUEngine, ldf_extract, group_index):
    q = ldf_extract.select(pl.col("a").str.extract(r"(\S+) (\d+) (.+)", group_index))
    assert_gpu_result_equal(q, engine=engine)


def test_extract_group_index_0_unsupported(engine: pl.GPUEngine, ldf_extract):
    q = ldf_extract.select(pl.col("a").str.extract(r"(\S+) (\d+) (.+)", 0))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_extract_groups(engine: pl.GPUEngine, ldf_extract):
    q = ldf_extract.select(pl.col("a").str.extract_groups(r"(\S+) (\d+) (.+)"))
    assert_gpu_result_equal(q, engine=engine)


def test_len_bytes(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.len_bytes())
    assert_gpu_result_equal(q, engine=engine)


def test_len_chars(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").str.len_chars())
    assert_gpu_result_equal(q, engine=engine)


def test_string_concat_empty_frame(engine: pl.GPUEngine):
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.String)})
    q = lf.select(pl.lit(", ") + pl.col("a"))

    assert_gpu_result_equal(q, engine=engine)


def test_single_column_concat_str(engine: pl.GPUEngine):
    lf = pl.LazyFrame({"c0": ["a", "b"]})
    q = lf.select(pl.concat_str(pl.col("c0")))
    assert_gpu_result_equal(q, engine=engine)


def test_concat_str_with_boolean(engine: pl.GPUEngine):
    lf = pl.LazyFrame({"c0": [True, False, None]})
    q = lf.with_columns(pl.concat_str([pl.col("c0"), pl.lit("bool")]))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="Split with literal parameter added in 1.38"
)
def test_split_regex_not_supported(engine: pl.GPUEngine):
    lf = pl.LazyFrame({"a": ["foo1bar", "baz456boo", "abc321"]})

    q = lf.select(pl.col("a").str.split(r"\d+", literal=False))

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "data",
    [
        ["a.b*c", "no_meta", "", None],
        ["(x)|[y]{z}", "^start$ end+", "back\\slash", "~#&-"],
        [string.printable],
    ],
)
def test_escape_regex(engine: pl.GPUEngine, data):
    lf = pl.LazyFrame({"a": pl.Series(data, dtype=pl.String)})
    q = lf.select(pl.col("a").str.escape_regex())
    assert_gpu_result_equal(q, engine=engine)
