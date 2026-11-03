# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

from rmm.pylibrmm.stream import Stream

import pylibcudf as plc


class _CudaStreamProto:
    """Minimal __cuda_stream__ protocol object for testing."""

    def __cuda_stream__(self):
        return (0, 0)


def test_get_stream_none():
    stream = plc.utils._get_stream(None)
    assert isinstance(stream, Stream)


def test_get_stream_stream_object():
    stream = Stream()
    result = plc.utils._get_stream(stream)
    assert result is stream


def test_get_stream_protocol_object():
    proto = _CudaStreamProto()
    result = plc.utils._get_stream(proto)
    assert isinstance(result, Stream)


@pytest.mark.parametrize("stream", [None, Stream(), _CudaStreamProto()])
def test_reduce_accepts_stream_protocol(stream):
    arr = pa.array([1, 2, 3], type=pa.int32())
    col = plc.Column.from_arrow(arr)
    agg = plc.aggregation.sum()
    dtype = plc.DataType.from_arrow(pa.int32())
    result = plc.reduce.reduce(col, agg, dtype, stream=stream)
    assert result.to_py() == 6


@pytest.mark.parametrize("stream", [None, Stream(), _CudaStreamProto()])
def test_binary_operation_accepts_stream_protocol(stream):
    lhs = plc.Column.from_arrow(pa.array([1, 2, 3], type=pa.int32()))
    rhs = plc.Column.from_arrow(pa.array([4, 5, 6], type=pa.int32()))
    dtype = plc.DataType.from_arrow(pa.int32())
    result = plc.binaryop.binary_operation(
        lhs,
        rhs,
        plc.binaryop.BinaryOperator.ADD,
        dtype,
        stream=stream,
    )
    expect = pa.array([5, 7, 9], type=pa.int32())
    assert result.to_arrow().equals(expect)


@pytest.mark.parametrize("stream", [None, Stream(), _CudaStreamProto()])
def test_gather_accepts_stream_protocol(stream):
    table = plc.Table.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    indices = plc.Column.from_arrow(pa.array([2, 0], type=pa.int32()))
    result = plc.copying.gather(
        table,
        indices,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    expected = pa.table({"a": [3, 1], "b": [6, 4]})
    got = result.to_arrow().rename_columns(expected.column_names)
    assert got.cast(expected.schema).equals(expected)
