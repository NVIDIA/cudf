# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unicodedata as ud

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc
from pylibcudf.nvtext.unicode_normalize import (
    UnicodeNormalizationForm,
    UnicodeNormalizer,
    normalize_unicode,
)


# ---------------------------------------------------------------------------
# Helper to build a minimal unicode_data Table from explicit row data.
# Each row is (cp_hex: str, ccc: int, decomp: str).
# ---------------------------------------------------------------------------
def _make_unicode_table(rows):
    cp_col = pa.array([r[0] for r in rows], type=pa.string())
    ccc_col = pa.array([r[1] for r in rows], type=pa.int32())
    decomp_col = pa.array([r[2] for r in rows], type=pa.string())
    tbl = pa.table({"cp": cp_col, "ccc": ccc_col, "decomp": decomp_col})
    return plc.Table.from_arrow(tbl)


# Minimal rows that cover NFD/NFC tests: é = U+00E9 → e (U+0065) + U+0301
_MINIMAL_ROWS = [
    ("00E9", 0, "0065 0301"),  # é, canonical decomposition
    ("0301", 230, ""),  # combining acute accent, CCC=230
]

# Rows needed for the NFKC ﬁ test (U+FB01 ligature)
_COMPAT_ROWS = [
    *_MINIMAL_ROWS,
    ("FB01", 0, "<compat> 0066 0069"),
]  # ﬁ ligature, compatibility only


@pytest.fixture(scope="module")
def nfc_normalizer():
    tbl = _make_unicode_table(_MINIMAL_ROWS)
    return UnicodeNormalizer(tbl, UnicodeNormalizationForm.NFC)


@pytest.fixture(scope="module")
def nfd_normalizer():
    tbl = _make_unicode_table(_MINIMAL_ROWS)
    return UnicodeNormalizer(tbl, UnicodeNormalizationForm.NFD)


@pytest.fixture(scope="module")
def nfkc_normalizer():
    tbl = _make_unicode_table(_COMPAT_ROWS)
    return UnicodeNormalizer(tbl, UnicodeNormalizationForm.NFKC)


@pytest.fixture(scope="module")
def nfkd_normalizer():
    tbl = _make_unicode_table(_COMPAT_ROWS)
    return UnicodeNormalizer(tbl, UnicodeNormalizationForm.NFKD)


# ---------------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------------


def test_null_strings(nfc_normalizer):
    input_col = plc.Column.from_arrow(
        pa.array([None, None, None], type=pa.string())
    )
    result = normalize_unicode(input_col, nfc_normalizer)
    expected = pa.array([None, None, None], type=pa.string())
    assert_column_eq(expected, result)


def test_empty_column(nfc_normalizer):
    arr = pa.array([], type=pa.string())
    result = normalize_unicode(plc.Column.from_arrow(arr), nfc_normalizer)
    assert_column_eq(arr, result)


def test_ascii_passthrough(nfc_normalizer):
    arr = pa.array(["hello", "world", "abc 123", ""])
    result = normalize_unicode(plc.Column.from_arrow(arr), nfc_normalizer)
    assert_column_eq(arr, result)


def test_nfc_compose(nfc_normalizer):
    # "é" (e + combining acute) should compose to U+00E9 (é)
    arr = pa.array(["é", "café"])
    result = normalize_unicode(plc.Column.from_arrow(arr), nfc_normalizer)
    expected = pa.array(["é", "café"])
    assert_column_eq(expected, result)


def test_nfd_decompose(nfd_normalizer):
    # U+00E9 (é) should decompose to "é"
    arr = pa.array(["é"])
    result = normalize_unicode(plc.Column.from_arrow(arr), nfd_normalizer)
    expected = pa.array(["é"])
    assert_column_eq(expected, result)


def test_nfc_stable(nfc_normalizer):
    # Already-composed strings should be unchanged by NFC
    arr = pa.array(["é", "café"])
    result = normalize_unicode(plc.Column.from_arrow(arr), nfc_normalizer)
    assert_column_eq(arr, result)


def test_nfkc_compat_ligature(nfkc_normalizer):
    # U+FB01 (ﬁ) has only a compatibility decomposition; NFKC expands it to "fi"
    arr = pa.array(["ﬁ"])
    result = normalize_unicode(plc.Column.from_arrow(arr), nfkc_normalizer)
    expected = pa.array(["fi"])
    assert_column_eq(expected, result)


def test_nfc_compat_ligature_stable(nfc_normalizer):
    # U+FB01 is NFC-stable: NFC must leave it unchanged
    arr = pa.array(["ﬁ"])
    result = normalize_unicode(plc.Column.from_arrow(arr), nfc_normalizer)
    assert_column_eq(arr, result)


def test_mixed_nulls(nfc_normalizer):
    # Null rows must not corrupt adjacent non-null rows
    arr = pa.array(["é", None, "café"])
    result = normalize_unicode(plc.Column.from_arrow(arr), nfc_normalizer)
    expected = pa.array(["é", None, "café"])
    assert_column_eq(expected, result)


# ---------------------------------------------------------------------------
# Comparison against Python's unicodedata.normalize
# ---------------------------------------------------------------------------
# Build a moderately complete unicode_data table from Python's unicodedata
# module, covering BMP codepoints (U+0080..U+FFFF) that have a non-trivial
# CCC or decomposition mapping.  This mirrors what a user would load from
# UnicodeData.txt and lets us cross-check the GPU result against the
# reference implementation.


def _build_bmp_unicode_table():
    """Return a pylibcudf Table covering non-trivial BMP codepoints."""
    cp_list, ccc_list, decomp_list = [], [], []
    for cp in range(0x0080, 0x10000):
        c = chr(cp)
        ccc = ud.combining(c)
        decomp = ud.decomposition(c)
        if ccc != 0 or decomp:
            cp_list.append(f"{cp:04X}")
            ccc_list.append(ccc)
            decomp_list.append(decomp)
    tbl = pa.table(
        {
            "cp": pa.array(cp_list, type=pa.string()),
            "ccc": pa.array(ccc_list, type=pa.int32()),
            "decomp": pa.array(decomp_list, type=pa.string()),
        }
    )
    return plc.Table.from_arrow(tbl)


@pytest.fixture(scope="module")
def bmp_nfc_normalizer():
    return UnicodeNormalizer(
        _build_bmp_unicode_table(), UnicodeNormalizationForm.NFC
    )


@pytest.fixture(scope="module")
def bmp_nfkc_normalizer():
    return UnicodeNormalizer(
        _build_bmp_unicode_table(), UnicodeNormalizationForm.NFKC
    )


@pytest.fixture(scope="module")
def bmp_nfd_normalizer():
    return UnicodeNormalizer(
        _build_bmp_unicode_table(), UnicodeNormalizationForm.NFD
    )


@pytest.fixture(scope="module")
def bmp_nfkd_normalizer():
    return UnicodeNormalizer(
        _build_bmp_unicode_table(), UnicodeNormalizationForm.NFKD
    )


@pytest.fixture(scope="module")
def comparison_strings():
    """Strings exercising a range of normalization scenarios."""
    return [
        "hello world",  # pure ASCII
        "café",  # precomposed é
        "café",  # decomposed e + combining acute
        "élève",  # é, è precomposed
        "ẛ̣",  # ẛ + combining dot below (reordering needed)
        "ﬁ ﬂ",  # ﬁ ﬂ ligatures (compat)
        "Ω",  # Ω (ohm sign, compat with U+03A9)
        "½",  # ½ vulgar fraction (compat)
        "　",  # ideographic space (compat)
        "가",  # 가 Hangul syllable (algorithmic)
        "힣",  # 힣 last Hangul syllable
        "",  # empty string
    ]


@pytest.mark.parametrize(
    "form,fixture_name",
    [
        ("NFC", "bmp_nfc_normalizer"),
        ("NFKC", "bmp_nfkc_normalizer"),
        ("NFD", "bmp_nfd_normalizer"),
        ("NFKD", "bmp_nfkd_normalizer"),
    ],
)
def test_compare_with_python_unicodedata(
    form, fixture_name, comparison_strings, request
):
    """GPU normalization must match Python's unicodedata.normalize reference."""
    normalizer = request.getfixturevalue(fixture_name)
    input_arr = pa.array(comparison_strings, type=pa.string())
    gpu_result = normalize_unicode(
        plc.Column.from_arrow(input_arr), normalizer
    )
    expected = pa.array(
        [ud.normalize(form, s) for s in comparison_strings], type=pa.string()
    )
    assert_column_eq(expected, gpu_result)
