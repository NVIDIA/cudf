# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import rmm

import pylibcudf as plc


class MockSpan:
    """Mock object that satisfies Span protocol."""

    def __init__(self, ptr: int, size: int):
        self._ptr = ptr
        self._size = size

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size


class NotASpan:
    """Object that does not satisfy Span protocol (missing size)."""

    @property
    def ptr(self) -> int:
        return 0


def test_column_construction_with_mock_span():
    """Test Column can be constructed with mock Span objects."""
    # Create a real buffer and mock Span around it
    buf = rmm.DeviceBuffer(size=100)
    mock_data = MockSpan(ptr=buf.ptr, size=buf.size)

    # Create column with mock Span
    col = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT32),
        size=25,  # 100 bytes / 4 bytes per int32
        data=mock_data,
        mask=None,
        null_count=0,
        offset=0,
        children=[],
    )

    assert col.size() == 25
    assert col.type().id() == plc.TypeId.INT32


def test_column_construction_with_gpumemoryview():
    """Test Column can be constructed with gpumemoryview (Span-compliant)."""
    buf = rmm.DeviceBuffer(size=100)
    gmv = plc.gpumemoryview(buf)

    col = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT32),
        size=25,
        data=gmv,
        mask=None,
        null_count=0,
        offset=0,
        children=[],
    )

    assert col.size() == 25
    assert col.type().id() == plc.TypeId.INT32


def test_column_construction_with_none_data():
    """Test Column accepts None for data (e.g., struct columns)."""
    col = plc.Column(
        data_type=plc.DataType(plc.TypeId.STRUCT),
        size=0,
        data=None,
        mask=None,
        null_count=0,
        offset=0,
        children=[],
    )

    assert col.size() == 0
    assert col.type().id() == plc.TypeId.STRUCT


def test_column_construction_with_none_mask():
    """Test Column accepts None for mask (non-nullable column)."""
    buf = rmm.DeviceBuffer(size=100)
    gmv = plc.gpumemoryview(buf)

    col = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT32),
        size=25,
        data=gmv,
        mask=None,
        null_count=0,
        offset=0,
        children=[],
    )

    assert col.null_count() == 0
    assert col.null_mask() is None


def test_column_construction_with_span_mask():
    """Test Column can be constructed with Span mask."""
    data_buf = rmm.DeviceBuffer(size=100)
    data_gmv = plc.gpumemoryview(data_buf)

    # Create mask buffer (bitmask)
    mask_buf = rmm.DeviceBuffer(size=4)  # 32 bits = 32 rows
    mask_span = MockSpan(ptr=mask_buf.ptr, size=mask_buf.size)

    col = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT32),
        size=25,
        data=data_gmv,
        mask=mask_span,
        null_count=0,
        offset=0,
        children=[],
    )

    assert col.size() == 25
    assert col.null_mask() is not None


def test_column_rejects_non_span_data():
    """Test Column raises TypeError for non-Span data."""
    not_a_span = NotASpan()

    with pytest.raises(
        TypeError,
        match=r"data must satisfy Span protocol.*got NotASpan",
    ):
        plc.Column(
            data_type=plc.DataType(plc.TypeId.INT32),
            size=25,
            data=not_a_span,
            mask=None,
            null_count=0,
            offset=0,
            children=[],
        )


def test_column_rejects_non_span_mask():
    """Test Column raises TypeError for non-Span mask."""
    buf = rmm.DeviceBuffer(size=100)
    gmv = plc.gpumemoryview(buf)
    not_a_span = NotASpan()

    with pytest.raises(
        TypeError,
        match=r"mask must satisfy Span protocol.*got NotASpan",
    ):
        plc.Column(
            data_type=plc.DataType(plc.TypeId.INT32),
            size=25,
            data=gmv,
            mask=not_a_span,
            null_count=0,
            offset=0,
            children=[],
        )


def test_column_data_accessor_returns_span():
    """Test Column.data() returns the Span object."""
    buf = rmm.DeviceBuffer(size=100)
    mock_data = MockSpan(ptr=buf.ptr, size=buf.size)

    col = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT32),
        size=25,
        data=mock_data,
        mask=None,
        null_count=0,
        offset=0,
        children=[],
    )

    # data() should return the same mock_data object
    assert col.data() is mock_data


def test_column_with_mask_accepts_span():
    """Test Column.with_mask() accepts Span objects."""
    buf = rmm.DeviceBuffer(size=100)
    gmv = plc.gpumemoryview(buf)

    col = plc.Column(
        data_type=plc.DataType(plc.TypeId.INT32),
        size=25,
        data=gmv,
        mask=None,
        null_count=0,
        offset=0,
        children=[],
    )

    # Create a new mask
    mask_buf = rmm.DeviceBuffer(size=4)
    mask_span = MockSpan(ptr=mask_buf.ptr, size=mask_buf.size)

    # with_mask should accept Span
    new_col = col.with_mask(mask_span, 5)
    assert new_col.null_count() == 5
    assert new_col.null_mask() is mask_span
