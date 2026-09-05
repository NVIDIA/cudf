# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def norm_spaces_input_data():
    arr = ["a b", "  c  d\n", "e \t f "]
    return pa.array(arr)


@pytest.fixture(scope="module")
def norm_chars_input_data():
    arr = ["éâîô\teaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]", "[PAD]"]
    return pa.array(arr)


def test_normalize_spaces(norm_spaces_input_data):
    got = plc.nvtext.normalize.normalize_spaces(
        plc.Column.from_arrow(norm_spaces_input_data)
    )
    expect = pa.array(["a b", "c d", "e f"])
    assert_column_eq(expect, got)


@pytest.mark.parametrize("do_lower", [True, False])
def test_normalizer(norm_chars_input_data, do_lower):
    got = plc.nvtext.normalize.normalize_characters(
        plc.Column.from_arrow(norm_chars_input_data),
        plc.nvtext.normalize.CharacterNormalizer(
            do_lower,
            plc.column_factories.make_empty_column(plc.types.TypeId.STRING),
        ),
    )
    if do_lower:
        expect = pa.array(
            [
                "eaio eaio",
                "acenu",
                "acenu",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [ pad ] ",
            ]
        )
    else:
        expect = pa.array(
            [
                "éâîô eaio",
                "ĂĆĖÑÜ",
                "ACENU",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [ PAD ] ",
            ]
        )
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "flags,do_lower,expected",
    [
        (
            plc.nvtext.normalize.NormalizeFlags.STRIP_ACCENTS
            | plc.nvtext.normalize.NormalizeFlags.PAD_PUNCTUATION,
            False,
            [
                "eaio eaio",
                "ACENU",
                "ACENU",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [ PAD ] ",
            ],
        ),
        (
            plc.nvtext.normalize.NormalizeFlags.NONE,
            False,
            ["éâîô eaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]", "[PAD]"],
        ),
        (
            plc.nvtext.normalize.NormalizeFlags.NONE,
            True,
            ["eaio eaio", "acenu", "acenu", "$24.08", "[a,bb]", "[pad]"],
        ),
    ],
)
def test_normalizer_flags(norm_chars_input_data, flags, do_lower, expected):
    got = plc.nvtext.normalize.normalize_characters(
        plc.Column.from_arrow(norm_chars_input_data),
        plc.nvtext.normalize.CharacterNormalizer(
            do_lower,
            plc.column_factories.make_empty_column(plc.types.TypeId.STRING),
        ),
        int(flags),
    )
    assert_column_eq(pa.array(expected), got)


@pytest.mark.parametrize("do_lower", [True, False])
def test_normalizer_with_special_tokens(norm_chars_input_data, do_lower):
    special_tokens = pa.array(["[PAD]"])
    got = plc.nvtext.normalize.normalize_characters(
        plc.Column.from_arrow(norm_chars_input_data),
        plc.nvtext.normalize.CharacterNormalizer(
            do_lower, plc.Column.from_arrow(special_tokens)
        ),
    )
    if do_lower:
        expect = pa.array(
            [
                "eaio eaio",
                "acenu",
                "acenu",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [PAD] ",
            ]
        )
    else:
        expect = pa.array(
            [
                "éâîô eaio",
                "ĂĆĖÑÜ",
                "ACENU",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [PAD] ",
            ]
        )
    assert_column_eq(expect, got)
