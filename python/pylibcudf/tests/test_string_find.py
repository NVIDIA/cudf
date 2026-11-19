# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def data_col():
    pa_array = pa.array(
        [
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
            "abc123",
            "ABC123",
            "aBc123",
            "",
            " ",
            None,
            "a",
            None,
        ]
    )
    return pa_array, plc.Column.from_arrow(pa_array)


@pytest.fixture(scope="module")
def target_col():
    pa_array = pa.array(
        [
            "a",
            "B",
            "x",
            "1",
            " ",
            "a",
            None,
            None,  # find
            "a",
            "B",
            "x",
            "1",
            " ",
            "a",
            None,
            None,  # rfind
            "ab",
            "12",
            "BC",
            "",
            " ",
            "a",
            None,
            None,  # contains
            "ab",
            "ABC",
            "AB",
            "",
            " ",
            "a",
            None,
            None,  # starts_with
            "3",
            "23",
            "a23",
            "",
            " ",
            "a",
            None,
            None,  # ends_with
        ]
    )
    return pa_array, plc.Column.from_arrow(pa_array)


@pytest.fixture(params=["a", " ", "A", "Ab", "23"], scope="module")
def target_scalar(request):
    pa_scalar = pa.scalar(request.param, type=pa.string())
    return pa_scalar, plc.Scalar.from_arrow(pa_scalar)


def test_find(data_col, target_scalar):
    pa_data_col, plc_data_col = data_col
    pa_target_scalar, plc_target_scalar = target_scalar
    got = plc.strings.find.find(plc_data_col, plc_target_scalar, 0, -1)

    expected = pa.array(
        [
            elem.find(pa_target_scalar.as_py()) if elem is not None else None
            for elem in pa_data_col.to_pylist()
        ],
        type=pa.int32(),
    )

    assert_column_eq(expected, got)


def colwise_apply(pa_data_col, pa_target_col, operator):
    def handle_none(st, target):
        # Match libcudf handling of nulls
        if st is None:
            return None
        elif target is None:
            return False
        else:
            return operator(st, target)

    expected = pa.array(
        [
            handle_none(elem, target)
            for elem, target in zip(
                pa_data_col.to_pylist(), pa_target_col.to_pylist(), strict=True
            )
        ],
        type=pa.bool_(),
    )

    return expected


def test_find_column(data_col, target_col):
    pa_data_col, plc_data_col = data_col
    pa_target_col, plc_target_col = target_col
    expect = pa.array(
        [
            elem.find(target) if not (elem is None or target is None) else None
            for elem, target in zip(
                pa_data_col.to_pylist(), pa_target_col.to_pylist(), strict=True
            )
        ],
        type=pa.int32(),
    )

    got = plc.strings.find.find(plc_data_col, plc_target_col, 0)
    assert_column_eq(expect, got)


def test_rfind(data_col, target_scalar):
    pa_data_col, plc_data_col = data_col
    pa_target_scalar, plc_target_scalar = target_scalar
    py_target = pa_target_scalar.as_py()

    got = plc.strings.find.rfind(plc_data_col, plc_target_scalar, 0, -1)

    expect = pa.array(
        [
            elem.rfind(py_target)
            if not (elem is None or py_target is None)
            else None
            for elem in pa_data_col.to_pylist()
        ],
        type=pa.int32(),
    )

    assert_column_eq(expect, got)


def test_contains(data_col, target_scalar):
    pa_data_col, plc_data_col = data_col
    pa_target_scalar, plc_target_scalar = target_scalar
    py_target = pa_target_scalar.as_py()

    got = plc.strings.find.contains(plc_data_col, plc_target_scalar)
    expect = pa.array(
        [
            py_target in elem
            if not (elem is None or py_target is None)
            else None
            for elem in pa_data_col.to_pylist()
        ],
        type=pa.bool_(),
    )

    assert_column_eq(expect, got)


def test_contains_column(data_col, target_col):
    pa_data_col, plc_data_col = data_col
    pa_target_col, plc_target_col = target_col
    expect = colwise_apply(
        pa_data_col, pa_target_col, lambda st, target: target in st
    )
    got = plc.strings.find.contains(plc_data_col, plc_target_col)
    assert_column_eq(expect, got)


def test_starts_with(data_col, target_scalar):
    pa_data_col, plc_data_col = data_col
    pa_target_scalar, plc_target_scalar = target_scalar
    py_target = pa_target_scalar.as_py()
    got = plc.strings.find.starts_with(plc_data_col, plc_target_scalar)
    expect = pc.starts_with(pa_data_col, py_target)
    assert_column_eq(expect, got)


def test_starts_with_column(data_col, target_col):
    pa_data_col, plc_data_col = data_col
    pa_target_col, plc_target_col = target_col
    expect = colwise_apply(
        pa_data_col, pa_target_col, lambda st, target: st.startswith(target)
    )
    got = plc.strings.find.starts_with(plc_data_col, plc_target_col)
    assert_column_eq(expect, got)


def test_ends_with(data_col, target_scalar):
    pa_data_col, plc_data_col = data_col
    pa_target_scalar, plc_target_scalar = target_scalar
    py_target = pa_target_scalar.as_py()
    got = plc.strings.find.ends_with(plc_data_col, plc_target_scalar)
    expect = pc.ends_with(pa_data_col, py_target)
    assert_column_eq(expect, got)


def test_ends_with_column(data_col, target_col):
    pa_data_col, plc_data_col = data_col
    pa_target_col, plc_target_col = target_col
    expect = colwise_apply(
        pa_data_col, pa_target_col, lambda st, target: st.endswith(target)
    )
    got = plc.strings.find.ends_with(plc_data_col, plc_target_col)
    assert_column_eq(expect, got)
