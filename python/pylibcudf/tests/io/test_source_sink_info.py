# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

import pytest

import pylibcudf as plc


@pytest.fixture(params=[plc.io.SourceInfo, plc.io.SinkInfo])
def io_class(request):
    return request.param


def _skip_invalid_sinks(io_class, sink):
    """
    Skip invalid sinks for SinkInfo
    """
    if io_class is plc.io.SinkInfo and isinstance(sink, bytes):
        pytest.skip("bytes is not a valid input for SinkInfo")


@pytest.mark.parametrize(
    "source",
    [
        "a.txt",
        b"hello world",
        io.BytesIO(b"hello world"),
    ],
)
def test_source_info_ctor(io_class, source, tmp_path):
    if isinstance(source, str):
        file = tmp_path / source
        file.write_bytes("hello world".encode("utf-8"))
        source = str(file)

    _skip_invalid_sinks(io_class, source)

    io_class([source])


@pytest.mark.parametrize(
    "sources",
    [
        ["a.txt"],
        [b"hello world"],
        [io.BytesIO(b"hello world")],
        ["a.txt", "a.txt"],
        [b"hello world", b"hello there"],
        [io.BytesIO(b"hello world"), io.BytesIO(b"hello there")],
    ],
)
def test_source_info_ctor_multiple(io_class, sources, tmp_path):
    for i in range(len(sources)):
        source = sources[i]
        if isinstance(source, str):
            file = tmp_path / source
            file.write_bytes("hello world".encode("utf-8"))
            sources[i] = str(file)

        _skip_invalid_sinks(io_class, source)

    io_class(sources)


@pytest.mark.parametrize(
    "sources",
    [
        ["awef.txt", b"hello world", io.BytesIO(b"hello world")],
        [b"hello world", b"hello there", "awef.txt"],
        [
            io.BytesIO(b"hello world"),
            io.BytesIO(b"hello there"),
            b"hello world",
        ],
    ],
)
def test_source_info_ctor_mixing_invalid(io_class, sources, tmp_path):
    # Unlike the previous test
    # don't create files so that they are missing
    for i in range(len(sources)):
        source = sources[i]
        if isinstance(source, str):
            file = tmp_path / source
            file.write_bytes("hello world".encode("utf-8"))
            sources[i] = str(file)
        _skip_invalid_sinks(io_class, source)
    with pytest.raises(ValueError):
        io_class(sources)


def test_source_info_invalid():
    with pytest.raises(ValueError):
        plc.io.SourceInfo([123])


def test_filepath_source_local_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    table = plc.Table([plc.Column.from_iterable_of_py([1, 2, 3])])
    plc.io.parquet.write_parquet(
        plc.io.parquet.ParquetWriterOptions.builder(
            plc.io.SinkInfo([str(path)]), table
        ).build()
    )
    file_size = path.stat().st_size

    source = plc.io.FilepathSource(str(path), size=file_size)
    assert source.path == str(path)
    assert source.size == file_size

    source_info = plc.io.SourceInfo([source])
    read_opts = plc.io.parquet.ParquetReaderOptions.builder(
        source_info
    ).build()
    result = plc.io.parquet.read_parquet(read_opts)
    assert result.columns[0].to_arrow().to_pylist() == [1, 2, 3]


def test_filepath_source_remote_uri_without_size():
    source = plc.io.FilepathSource("s3://bucket/object.parquet")
    assert source.size is None
    plc.io.SourceInfo([source])


def test_filepath_source_mixed_sources_invalid():
    with pytest.raises(
        ValueError, match="All sources must be of the same type"
    ):
        plc.io.SourceInfo(
            [
                plc.io.FilepathSource("s3://bucket/object.parquet", size=100),
                "s3://bucket/other.parquet",
            ]
        )
