# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import re

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_findall():
    arr = pa.array(["bunny", "rabbit", "hare", "dog"])
    pattern = "[ab]"
    got = plc.strings.findall.findall(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    expect = pa.array(
        [re.findall(pattern, elem) for elem in arr.to_pylist()],
        type=got.type().to_arrow(value_type=pa.string()),
    )
    assert_column_eq(expect, got)


def test_findall_one_capture():
    arr = pa.array(["abc=123 xyz=789", "", "d=4", "ab=12"])
    pattern = "[a-z]+=(\\d+)"
    got = plc.strings.findall.findall(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    expect = pa.array(
        [re.findall(pattern, elem) for elem in arr.to_pylist()],
        type=got.type().to_arrow(value_type=pa.string()),
    )
    assert_column_eq(expect, got)


def test_find_re():
    arr = pa.array(["bunny", "rabbit", "hare", "dog"])
    pattern = "[eb]"
    got = plc.strings.findall.find_re(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    expect = pa.array(
        [0, 2, 3, -1],
        type=got.type().to_arrow(value_type=pa.string()),
    )
    assert_column_eq(expect, got)
