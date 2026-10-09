# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def pa_col():
    return pa.array(["AbC", "123abc", "", " ", None])


@pytest.fixture(scope="module")
def plc_col(pa_col):
    return plc.Column.from_arrow(pa_col)


@pytest.fixture(
    scope="module",
    params=[(1, 3, 1), (0, 3, -1), (3, 2, 1), (1, 5, 5), (1, 100, 2)],
)
def pa_start_stop_step(request):
    return tuple(pa.scalar(x, type=pa.int32()) for x in request.param)


@pytest.fixture(scope="module")
def plc_start_stop_step(pa_start_stop_step):
    return tuple(x.as_py() for x in pa_start_stop_step)


@pytest.fixture(scope="module")
def pa_starts_col():
    return pa.array([0, 1, 3, -1, 100])


@pytest.fixture(scope="module")
def plc_starts_col(pa_starts_col):
    return plc.Column.from_arrow(pa_starts_col)


@pytest.fixture(scope="module")
def pa_stops_col():
    return pa.array([1, 3, 4, -1, 100])


@pytest.fixture(scope="module")
def plc_stops_col(pa_stops_col):
    return plc.Column.from_arrow(pa_stops_col)


def test_slice(pa_col, plc_col, pa_start_stop_step, plc_start_stop_step):
    pa_start, pa_stop, pa_step = pa_start_stop_step
    plc_start, plc_stop, plc_step = plc_start_stop_step

    def slice_string(st, start, stop, step):
        return st[start:stop:step] if st is not None else None

    expect = pa.array(
        [
            slice_string(x, pa_start.as_py(), pa_stop.as_py(), pa_step.as_py())
            for x in pa_col.to_pylist()
        ],
        type=pa.string(),
    )

    got = plc.strings.slice.slice_strings(
        plc_col, start=plc_start, stop=plc_stop, step=plc_step
    )

    assert_column_eq(expect, got)


def test_slice_column(
    pa_col, plc_col, pa_starts_col, plc_starts_col, pa_stops_col, plc_stops_col
):
    def slice_string(st, start, stop):
        if stop < 0:
            stop = len(st)
        return st[start:stop] if st is not None else None

    expect = pa.array(
        [
            slice_string(x, start, stop)
            for x, start, stop in zip(
                pa_col.to_pylist(),
                pa_starts_col.to_pylist(),
                pa_stops_col.to_pylist(),
                strict=True,
            )
        ],
        type=pa.string(),
    )

    got = plc.strings.slice.slice_strings(
        plc_col, plc_starts_col, plc_stops_col
    )

    assert_column_eq(expect, got)


def test_slice_invalid(plc_col, plc_starts_col, plc_stops_col):
    with pytest.raises(ValueError):
        # input cannot be None
        plc.strings.slice.slice_strings(None, plc_starts_col, plc_stops_col)
    with pytest.raises(ValueError):
        # start and stop must both be provided for column-wise slice
        plc.strings.slice.slice_strings(plc_col, plc_starts_col, None)
    with pytest.raises(ValueError):
        # can't provide step for column-wise slice
        plc.strings.slice.slice_strings(
            plc_col, plc_starts_col, plc_stops_col, 2
        )
    with pytest.raises(TypeError):
        # can't mix a Column and a non-Column
        plc.strings.slice.slice_strings(plc_col, plc_starts_col, 3)
    with pytest.raises(ValueError):
        # start/stop must be a Column or an int
        plc.strings.slice.slice_strings(plc_col, "invalid", 3)
