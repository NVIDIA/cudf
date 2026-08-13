# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.mark.parametrize(
    "agg,dt",
    [
        (plc.aggregation.any(), pa.int8()),
        (plc.aggregation.all(), pa.uint8()),
        (plc.aggregation.sum(), pa.int16()),
        (plc.aggregation.product(), pa.uint16()),
        (plc.aggregation.min(), pa.string()),
        (plc.aggregation.max(), pa.timestamp("s")),
        (plc.aggregation.argmin(), pa.int8()),
        (plc.aggregation.argmax(), pa.uint8()),
        (plc.aggregation.mean(), pa.int32()),
        (plc.aggregation.sum_of_squares(), pa.uint32()),
        (plc.aggregation.std(), pa.float32()),
        (plc.aggregation.variance(), pa.float64()),
        (plc.aggregation.median(), pa.uint64()),
        (plc.aggregation.quantile([10]), pa.int64()),
        (plc.aggregation.count(), pa.duration("ms")),
        (plc.aggregation.nunique(), pa.decimal128(6, 1)),
        (plc.aggregation.nth_element(10), pa.float32()),
    ],
)
def test_is_valid_aggregation(agg, dt):
    plc_type = plc.DataType.from_arrow(dt)
    assert plc.reduce.is_valid_reduce_aggregation(plc_type, agg)


@pytest.mark.parametrize(
    "agg,dt",
    [
        (plc.aggregation.rank(0), pa.string()),
        (plc.aggregation.covariance(1, 1), pa.float32()),
        (plc.aggregation.correlation(0, 1), pa.float64()),
        (plc.aggregation.sum(), pa.string()),
        (plc.aggregation.sum(), pa.timestamp("ms")),
        (plc.aggregation.mean(), pa.decimal128(6, 1)),
    ],
)
def test_is_not_valid_aggregation(agg, dt):
    plc_type = plc.DataType.from_arrow(dt)
    assert not plc.reduce.is_valid_reduce_aggregation(plc_type, agg)


@pytest.fixture(params=[False, True], ids=["Column", "Table"])
def column_as_table(request):
    return request.param


@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 2, 2, 3, 3, 3], 3),
        ([1, 1, 1, 1], 1),
        ([], 0),
        ([1, 2, 3, 4, 5], 5),
    ],
)
def test_distinct_count(data, expected, column_as_table):
    arr = pa.array(data, type=pa.int32())
    col = plc.Column.from_arrow(arr)
    if column_as_table:
        result = plc.reduce.distinct_count_table(
            plc.Table([col]), plc.types.NullEquality.EQUAL
        )
    else:
        result = plc.reduce.distinct_count(
            col,
            plc.types.NullPolicy.INCLUDE,
            plc.types.NanPolicy.NAN_IS_VALID,
        )
    assert result == expected


@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 2, 2, 3, 3, 3], 3),
        ([1, 1, 1, 1], 1),
        ([], 0),
        ([1, 2, 3, 4, 5], 5),
        ([1, None, 2, None, 3], 4),
    ],
)
def test_distinct_count_with_nulls(data, expected, column_as_table):
    arr = pa.array(data, type=pa.int32())
    col = plc.Column.from_arrow(arr)
    if column_as_table:
        result = plc.reduce.distinct_count_table(
            plc.Table([col]), plc.types.NullEquality.EQUAL
        )
    else:
        result = plc.reduce.distinct_count(
            col,
            plc.types.NullPolicy.INCLUDE,
            plc.types.NanPolicy.NAN_IS_VALID,
        )
    assert result == expected


def test_distinct_count_exclude_nulls():
    arr = pa.array([1, None, 2, None, 3], type=pa.int32())
    col = plc.Column.from_arrow(arr)
    result = plc.reduce.distinct_count(
        col,
        plc.types.NullPolicy.EXCLUDE,
        plc.types.NanPolicy.NAN_IS_VALID,
    )
    assert result == 3


@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 1, 2, 2, 3, 3], 3),
        ([1, 2, 2, 3, 3, 3], 3),
        ([1, 1, 1, 1], 1),
        ([], 0),
        ([1, 2, 3, 4, 5], 5),
    ],
)
def test_unique_count(data, expected, column_as_table):
    arr = pa.array(data, type=pa.int32())
    col = plc.Column.from_arrow(arr)
    if column_as_table:
        result = plc.reduce.unique_count_table(
            plc.Table([col]), plc.types.NullEquality.EQUAL
        )
    else:
        result = plc.reduce.unique_count(
            col,
            plc.types.NullPolicy.INCLUDE,
            plc.types.NanPolicy.NAN_IS_VALID,
        )
    assert result == expected


def test_unique_count_with_nulls(column_as_table):
    arr = pa.array([1, 1, None, None, 2, 2], type=pa.int32())
    col = plc.Column.from_arrow(arr)
    if column_as_table:
        result = plc.reduce.unique_count_table(
            plc.Table([col]), plc.types.NullEquality.EQUAL
        )
    else:
        result = plc.reduce.unique_count(
            col,
            plc.types.NullPolicy.INCLUDE,
            plc.types.NanPolicy.NAN_IS_VALID,
        )
    assert result == 3


def test_unique_count_exclude_nulls():
    arr = pa.array([1, 1, None, None, 2, 2], type=pa.int32())
    col = plc.Column.from_arrow(arr)
    result = plc.reduce.unique_count(
        col,
        plc.types.NullPolicy.EXCLUDE,
        plc.types.NanPolicy.NAN_IS_VALID,
    )
    assert result == 2


def test_count_nulls_unequal():
    arr = pa.array([1, 1, None, None, 2, 2], type=pa.int64())
    col = plc.Column.from_arrow(arr)
    table = plc.Table([col])
    result = plc.reduce.unique_count_table(
        table, plc.types.NullEquality.UNEQUAL
    )
    assert result == 4


def test_distinct_count_nulls_unequal():
    arr = pa.array([1, None, 1, 2, None, 2], type=pa.int64())
    col = plc.Column.from_arrow(arr)
    table = plc.Table([col])
    result = plc.reduce.distinct_count_table(
        table, plc.types.NullEquality.UNEQUAL
    )
    assert result == 4
