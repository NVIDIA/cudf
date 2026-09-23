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


@pytest.mark.parametrize(
    "input_strings",
    [["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]],
)
@pytest.mark.parametrize(
    "pattern",
    ["A", "de", ".*", "^a", "^A", "[^a-z]", "[a-z]{3,}", "^[A-Z]{2,}", "j|u"],
)
def test_contains_re(input_strings, pattern):
    input = pa.array(input_strings)
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(input),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(input, pattern)
    assert_column_eq(expect, got)


def test_count_re():
    pattern = "[1-9][a-z]"
    arr = pa.array(["A1a2A3a4", "A1A2A3", None])
    got = plc.strings.contains.count_re(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    expect = pc.count_substring_regex(arr, pattern)
    assert_column_eq(expect, got)


def test_match_re():
    pattern = "[1-9][a-z]"
    arr = pa.array(["1a2b", "b1a2", None])
    got = plc.strings.contains.matches_re(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    expect = pc.match_substring_regex(arr, f"^{pattern}")
    assert_column_eq(expect, got)


def test_like():
    pattern = "%a"
    arr = pa.array(["1a2aa3aaa"])
    got = plc.strings.contains.like(
        plc.Column.from_arrow(arr),
        pattern,
    )
    expect = pc.match_like(arr, pattern)
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


# Basic quantifiers (test_rlike, test_regexp, test_regexp_like)
@pytest.mark.parametrize(
    "pattern",
    [
        "a{2}",
        "a{1,3}",
        "a{1,}",
        "a[bc]d",
    ],
)
def test_contains_re_basic(spark_strings, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Alternation patterns (test_regexp_choice, test_rlike_rewrite_optimization)
@pytest.mark.parametrize(
    "pattern",
    [
        "aaa|bbb|ccc",
        "1|2|3|4|5|6",
        "[abcd]|[123]",
        "aaa|bbb",
        "aaa|(bbb|ccc)",
        ".*.*(aaa|bbb).*.*",
        "^.*(aaa|bbb|ccc)",
        "abd1a$|^ab2a",
        "[ab]+|^cd1",
    ],
)
def test_contains_re_alternation(spark_strings, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Anchor + wildcard patterns (test_rlike_rewrite_optimization)
@pytest.mark.parametrize(
    "pattern",
    [
        "^abb",
        "^.*(aaa)",
        "^(abb)(.*)",
        "abb(.*)",
        "(.*)(abb)(.*)",
        "ab(.*)cd",
        "(.*)(.*)abb",
    ],
)
def test_contains_re_anchors_wildcards(spark_strings, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Bounded repetition (test_rlike_rewrite_optimization, test_regexp)
@pytest.mark.parametrize(
    "pattern",
    [
        "ab[a-c]{3}",
        "a[a-c]{1,3}",
        "a[a-c]{1,}",
        "a[a-c]+",
        "(ab)([a-c]{1})",
        "(ab[a-c]{1})",
        "a{6}",
        "a{1,6}",
    ],
)
def test_contains_re_bounded_repetition(spark_strings, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Non-capturing groups / complex quantifiers (test_regexp_memory_ok)
@pytest.mark.parametrize(
    "pattern",
    [
        "(?:12345)+",
        "(?:aa)+",
        "abcdef",
        "(1)(2)(3)",
    ],
)
def test_contains_re_groups(spark_strings, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# Predefined character classes (test_character_classes, test_regexp_whitespace,
# test_regexp_replace_digit, test_regexp_replace_word)
@pytest.mark.parametrize(
    "pattern",
    [
        r"\d",
        r"\D",
        r"[0-9]",
        r"[^0-9]",
        r"\w",
        r"[a-zA-Z_0-9]",
        r"\s",
        r"\S",
        r"[abcd]+\s+[0-9]+",
        r"\S{3}",
        r"[^\n\r]",
    ],
)
def test_contains_re_char_classes(spark_strings, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)


# \W: cuDF uses ASCII semantics — only [^a-zA-Z0-9_] are non-word chars,
# unlike pyarrow/RE2 which is Unicode-aware. Test with hardcoded expectations.
def test_contains_re_nonword():
    arr = pa.array(
        ["abc", "abc123", "abc!", "a b", "aa|bb", "12345", "", None]
    )
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(arr),
        _make_prog(r"\W"),
    )
    # cuDF \W matches non-ASCII-word characters: space, !, |
    expect = pa.array([False, False, True, True, True, False, False, None])
    assert_column_eq(expect, got)


# Escape / character class edge cases (test_rlike_escape, test_rlike_missing_escape)
@pytest.mark.parametrize(
    "pattern",
    [
        r"a[\-]",
        r"a[+-]",
        r"a[a-b-]",
        r"[a-z]{3,}",
        r"^[A-Z]{2,}",
    ],
)
def test_contains_re_escape_edge_cases(spark_strings, pattern):
    got = plc.strings.contains.contains_re(
        plc.Column.from_arrow(spark_strings),
        _make_prog(pattern),
    )
    expect = pc.match_substring_regex(spark_strings, pattern)
    assert_column_eq(expect, got)
