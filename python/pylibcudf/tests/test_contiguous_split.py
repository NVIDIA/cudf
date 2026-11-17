# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import string

import pyarrow as pa
import pytest
from utils import assert_table_eq

import rmm
from rmm.pylibrmm.device_buffer import to_device
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc

param_pyarrow_tables = [
    pa.table([]),
    pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
    pa.table({"a": [1, 2, 3]}),
    pa.table({"a": [1], "b": [2], "c": [3]}),
    pa.table({"a": ["a", "bb", "ccc"]}),
    pa.table({"a": [1, 2, None], "b": [None, 3, 4]}),
    pa.table(
        {
            "a": [["a", "b"], ["cde"]],
            "b": [
                {"alpha": [1, 2], "beta": None},
                {"alpha": [3, 4], "beta": 5},
            ],
        }
    ),
]


@pytest.mark.parametrize("arrow_tbl", param_pyarrow_tables)
def test_pack_and_unpack(arrow_tbl):
    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    packed = plc.contiguous_split.pack(plc_tbl)

    res = plc.contiguous_split.unpack(packed)
    assert_table_eq(arrow_tbl, res)


@pytest.mark.parametrize("arrow_tbl", param_pyarrow_tables)
def test_pack_and_unpack_from_memoryviews(arrow_tbl):
    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    packed = plc.contiguous_split.pack(plc_tbl)

    metadata, gpudata = packed.release()

    with pytest.raises(ValueError, match="Cannot release empty"):
        packed.release()

    del packed  # `metadata` and `gpudata` will survive

    res = plc.contiguous_split.unpack_from_memoryviews(metadata, gpudata)
    assert_table_eq(arrow_tbl, res)


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("bufsize", [1024 * 1024, 3 * 1024 * 1000])
def test_chunked_pack(bufsize, stream):
    nprint = len(string.printable)
    h_table = pa.table(
        {
            "a": list(range(100_000)),
            "b": [string.printable[: i % nprint] for i in range(100_000)],
        }
    )
    temp_mr = rmm.mr.CudaMemoryResource()
    staging_buf = rmm.DeviceBuffer(size=bufsize)
    metadata, h_pack = plc.contiguous_split.ChunkedPack.create(
        plc.Table.from_arrow(h_table),
        bufsize,
        stream,
        temp_mr,
    ).pack_to_host(staging_buf)

    result = plc.contiguous_split.unpack_from_memoryviews(
        metadata,
        plc.gpumemoryview(to_device(h_pack, plc.utils._get_stream(stream))),
    )

    assert_table_eq(h_table, result)


def test_pack_and_unpack_zero_columns_with_rows():
    tbl = plc.Table([], num_rows=10)
    packed = plc.contiguous_split.pack(tbl)

    res = plc.contiguous_split.unpack(packed)
    assert res.num_columns() == 0
    assert res.num_rows() == 10


def test_pack_and_unpack_from_memoryviews_zero_columns_with_rows():
    tbl = plc.Table([], num_rows=10)
    packed = plc.contiguous_split.pack(tbl)

    metadata, gpudata = packed.release()
    del packed

    res = plc.contiguous_split.unpack_from_memoryviews(metadata, gpudata)
    assert res.num_columns() == 0
    assert res.num_rows() == 10


def test_unpack_from_memoryviews_empty_metadata_non_empty_data():
    empty_metadata = memoryview(b"")
    non_empty_data = plc.gpumemoryview(rmm.DeviceBuffer(size=64))

    with pytest.raises(
        ValueError,
        match="Expected an empty gpu_data from unpacking an empty table",
    ):
        plc.contiguous_split.unpack_from_memoryviews(
            empty_metadata, non_empty_data
        )
