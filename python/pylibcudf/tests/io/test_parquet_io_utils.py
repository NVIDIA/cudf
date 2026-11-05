# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from utils import synchronize_stream

import rmm

import pylibcudf as plc
from pylibcudf.io.parquet_io_utils import (
    IOSubmissionPolicy,
    fetch_byte_ranges_to_device,
)
from pylibcudf.io.text import ByteRangeInfo


@pytest.fixture
def source_bytes() -> list[bytes]:
    """Distinct contents of the sources to fetch from."""
    return [bytes(range(256)), bytes(reversed(range(256))), b"x" * 128]


@pytest.fixture
def source_info(tmp_path, source_bytes) -> plc.io.SourceInfo:
    paths = []
    for i, data in enumerate(source_bytes):
        path = tmp_path / f"source{i}.bin"
        path.write_bytes(data)
        paths.append(str(path))
    return plc.io.SourceInfo(paths)


def to_host(view: plc.gpumemoryview) -> bytes:
    return rmm.DeviceBuffer(ptr=view.ptr, size=view.size).tobytes()


def test_fetch_byte_ranges_to_device_single_source(
    tmp_path, source_bytes
) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(source_bytes[0])
    byte_ranges = [ByteRangeInfo(0, 16), ByteRangeInfo(100, 32)]

    views = fetch_byte_ranges_to_device(
        plc.io.SourceInfo([str(path)]),
        byte_ranges,
        IOSubmissionPolicy.SERIALIZE,
    )
    synchronize_stream()

    assert [to_host(view) for view in views] == [
        source_bytes[0][:16],
        source_bytes[0][100:132],
    ]


def test_fetch_byte_ranges_to_device_multiple_sources(
    source_info, source_bytes
) -> None:
    # Ranges are not grouped by source, and the second source has no ranges
    ranges = [
        (ByteRangeInfo(0, 16), 2),
        (ByteRangeInfo(8, 8), 0),
        (ByteRangeInfo(100, 20), 2),
        (ByteRangeInfo(200, 32), 0),
    ]

    views = fetch_byte_ranges_to_device(
        source_info,
        [byte_range for byte_range, _ in ranges],
        IOSubmissionPolicy.SERIALIZE,
        source_indices=[source for _, source in ranges],
    )
    synchronize_stream()

    assert [to_host(view) for view in views] == [
        source_bytes[source][r.offset : r.offset + r.size]
        for r, source in ranges
    ]


def test_fetch_byte_ranges_to_device_requires_source_indices(
    source_info,
) -> None:
    with pytest.raises(ValueError, match="source_indices is required"):
        fetch_byte_ranges_to_device(
            source_info,
            [ByteRangeInfo(0, 16)],
            IOSubmissionPolicy.SERIALIZE,
        )


def test_fetch_byte_ranges_to_device_source_indices_length(
    source_info,
) -> None:
    with pytest.raises(ValueError, match="one source index per byte range"):
        fetch_byte_ranges_to_device(
            source_info,
            [ByteRangeInfo(0, 16), ByteRangeInfo(16, 16)],
            IOSubmissionPolicy.SERIALIZE,
            source_indices=[0],
        )


@pytest.mark.parametrize("source", [-1, 3])
def test_fetch_byte_ranges_to_device_source_index_out_of_range(
    source_info, source
) -> None:
    with pytest.raises(ValueError, match="does not refer to one of 3 sources"):
        fetch_byte_ranges_to_device(
            source_info,
            [ByteRangeInfo(0, 16)],
            IOSubmissionPolicy.SERIALIZE,
            source_indices=[source],
        )
