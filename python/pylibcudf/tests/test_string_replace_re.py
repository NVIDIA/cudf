# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


def _make_prog(pattern):
    flags = plc.strings.regex_flags.RegexFlags.DEFAULT
    return plc.strings.regex_program.RegexProgram.create(pattern, flags)


@pytest.mark.parametrize("max_replace_count", [-1, 1])
def test_replace_re_regex_program_scalar(max_replace_count):
    arr = pa.array(["foo", "fuz", None])
    pat = "f."
    repl = "ba"
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(arr),
        _make_prog(pat),
        plc.Scalar.from_arrow(pa.scalar(repl)),
        max_replace_count=max_replace_count,
    )
    expect = pc.replace_substring_regex(
        arr,
        pat,
        repl,
        max_replacements=max_replace_count
        if max_replace_count != -1
        else None,
    )
    assert_column_eq(expect, got)


def test_replace_with_backrefs():
    arr = pa.array(["Z756", None])
    got = plc.strings.replace_re.replace_with_backrefs(
        plc.Column.from_arrow(arr),
        _make_prog("(\\d)(\\d)"),
        "V\\2\\1",
    )
    expect = pa.array(["ZV576", None])
    assert_column_eq(expect, got)


# Tests derived from cudf-spark integration tests


@pytest.fixture(scope="module")
def spark_strings():
    """Rich string array that exercises the cudf-spark regex patterns."""
    return pa.array(
        [
            "abc",
            "aabbc",
            "123abc",
            "abc123def",
            "a1b2c3",
            "boo:and:foo",
            "foo:boo:",
            "TEST",
            "TESTaaa",
            "TEST123",
            "abcd",
            "aaa",
            "bbb",
            "ccc",
            "abb",
            "ab",
            "aab",
            "a\nb",
            "a\tb",
            "a b",
            "",
            "aaabbb",
            "abcabc",
            "aa|bb",
            "foobar",
            "12345",
            "abcdef",
            "ABCDEF",
            "AbCdEf",
            None,
        ]
    )


REPLACE_REPL = "X"  # replacement string used in all replace tests


# Basic replace patterns (test_re_replace, test_regexp_replace)
@pytest.mark.parametrize(
    "pattern",
    [
        "TEST",
        "[A-Z]+",
        "a",
        "[^xyz]",
        "a|b|c",
    ],
)
def test_replace_re_basic(spark_strings, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Repetition quantifiers (test_re_replace_repetition)
@pytest.mark.parametrize(
    "pattern",
    [
        "[E]+",
        "[A]+",
        "[A-Z]+",
    ],
)
def test_replace_re_quantifiers(spark_strings, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Negated character classes (test_regexp_replace_character_set_negated)
@pytest.mark.parametrize(
    "pattern",
    [
        "[^a]",
        r"[^a\r\n]",
        r"[^\r\n]",
        r"[^\r]",
        r"[^\n]",
    ],
)
def test_replace_re_negated_classes(spark_strings, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Digit and word classes (test_regexp_replace_digit, test_regexp_replace_word)
# \D and \W are excluded from pyarrow comparison: cuDF uses ASCII semantics for
# \w/\W (only [a-zA-Z0-9_]), while pyarrow/RE2 is Unicode-aware.
@pytest.mark.parametrize(
    "pattern",
    [
        r"\d",
        r"[0-9]",
        r"[^0-9]",
        r"\w",
        r"[a-zA-Z_0-9]",
        r"[^a-zA-Z_0-9]",
    ],
)
def test_replace_re_digit_word(spark_strings, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# \D replace: cuDF ASCII semantics — non-digits include letters, spaces, punctuation
def test_replace_re_nondigit():
    arr = pa.array(["abc", "a1b2", "123", "a b", "", None])
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(arr),
        _make_prog(r"\D"),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pa.array(["XXX", "X1X2", "123", "XXX", "", None])
    assert_column_eq(expect, got)


# \W replace: cuDF ASCII semantics — non-word chars are non-[a-zA-Z0-9_]
def test_replace_re_nonword():
    arr = pa.array(["abc", "a b", "a!b", "aa|bb", "123", "", None])
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(arr),
        _make_prog(r"\W"),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pa.array(["abc", "aXb", "aXb", "aaXbb", "123", "", None])
    assert_column_eq(expect, got)


# Multi-alternation (test_regexp_replace_multi_optimization)
@pytest.mark.parametrize(
    "pattern",
    [
        "aa|bb",
        "aa|bb|cc",
        "aa|bb|cc|dd",
        "aa|bb|cc|dd|ee",
        "aa|bb|cc|dd|ee|ff",
        "(aa)|(bb)",
        "(aa)|(bb)|(cc)",
        "(aa|bb)|(cc|dd)",
    ],
)
def test_replace_re_multi_alternation(spark_strings, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)


# Non-capturing groups (test_regexp_replace)
@pytest.mark.parametrize(
    "pattern",
    [
        "(?:aa)+",
        "([^x])|([^y])",
    ],
)
def test_replace_re_noncapturing(spark_strings, pattern):
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
        plc.Scalar.from_arrow(pa.scalar(REPLACE_REPL)),
    )
    expect = pc.replace_substring_regex(spark_strings, pattern, REPLACE_REPL)
    assert_column_eq(expect, got)
