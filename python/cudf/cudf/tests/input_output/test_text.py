# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO, StringIO

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "text"


def test_read_text(datadir):
    chess_file = str(datadir) + "/chess.pgn"
    delimiter = "1."

    with open(chess_file) as f:
        content = f.read().split(delimiter)

    # Since Python split removes the delimiter and read_text does
    # not we need to add it back to the 'content'
    expected = cudf.Series(
        [
            c + delimiter if i < (len(content) - 1) else c
            for i, c in enumerate(content)
        ]
    )

    actual = cudf.read_text(chess_file, delimiter=delimiter)

    assert_eq(expected, actual)


def test_read_text_byte_range(datadir):
    chess_file = str(datadir) + "/chess.pgn"
    delimiter = "1."

    with open(chess_file, "r") as f:
        data = f.read()
        content = data.split(delimiter)

    # Since Python split removes the delimiter and read_text does
    # not we need to add it back to the 'content'
    expected = cudf.Series(
        [
            c + delimiter if i < (len(content) - 1) else c
            for i, c in enumerate(content)
        ]
    )

    byte_range_size = (len(data) // 3) + (len(data) % 3 != 0)

    actual_0 = cudf.read_text(
        chess_file,
        delimiter=delimiter,
        byte_range=[byte_range_size * 0, byte_range_size],
    )
    actual_1 = cudf.read_text(
        chess_file,
        delimiter=delimiter,
        byte_range=[byte_range_size * 1, byte_range_size],
    )
    actual_2 = cudf.read_text(
        chess_file,
        delimiter=delimiter,
        byte_range=[byte_range_size * 2, byte_range_size],
    )

    actual = cudf.concat([actual_0, actual_1, actual_2], ignore_index=True)

    assert_eq(expected, actual)


def test_read_text_byte_range_large(tmpdir):
    content = "".join(("\n" if x % 5 == 4 else "x") for x in range(0, 3000))
    delimiter = "\n"
    temp_file = str(tmpdir) + "/temp.txt"

    with open(temp_file, "w") as f:
        f.write(content)

    expected = cudf.Series(["xxxx\n" for i in range(0, 200)])

    actual = cudf.read_text(
        temp_file, delimiter=delimiter, byte_range=[1000, 1000]
    )

    assert_eq(expected, actual)


def test_read_text_in_memory(datadir):
    # Since Python split removes the delimiter and read_text does
    # not we need to add it back to the 'content'
    expected = cudf.Series(["x::", "y::", "z"])

    actual = cudf.read_text(StringIO("x::y::z"), delimiter="::")

    assert_eq(expected, actual)


def test_read_text_in_memory_bytesio(datadir):
    # GH #13734: a binary host buffer (BytesIO) should be readable just like
    # StringIO. The whole buffer is read regardless of its current position,
    # matching the other readers (e.g. read_csv).
    expected = cudf.Series(["x::", "y::", "z"])

    buf = BytesIO(b"x::y::z")
    buf.seek(0, 2)  # emulate a written-but-not-rewound buffer

    actual = cudf.read_text(buf, delimiter="::")

    assert_eq(expected, actual)


def test_read_text_in_memory_strip_delimiter(datadir):
    # Since Python split removes the delimiter and read_text does
    # not we need to add it back to the 'content'
    expected = cudf.Series(["x", "y", "z"])

    actual = cudf.read_text(
        StringIO("x::y::z"), delimiter="::", strip_delimiters=True
    )

    assert_eq(expected, actual)


def test_read_text_bgzip(datadir):
    chess_file_compressed = str(datadir) + "/chess.pgn.gz"
    chess_file = str(datadir) + "/chess.pgn"
    delimiter = "1."

    with open(chess_file) as f:
        content = f.read().split(delimiter)

    # Since Python split removes the delimiter and read_text does
    # not we need to add it back to the 'content'
    expected = cudf.Series(
        [
            c + delimiter if i < (len(content) - 1) else c
            for i, c in enumerate(content)
        ]
    )

    actual = cudf.read_text(
        chess_file_compressed, compression="bgzip", delimiter=delimiter
    )

    assert_eq(expected, actual)


def test_read_text_bgzip_offsets(datadir):
    chess_file_compressed = str(datadir) + "/chess.pgn.gz"
    chess_file = str(datadir) + "/chess.pgn"
    delimiter = "1."

    with open(chess_file) as f:
        content = f.read()[29:695].split(delimiter)

    # Since Python split removes the delimiter and read_text does
    # not we need to add it back to the 'content'
    expected = cudf.Series(
        [
            c + delimiter if i < (len(content) - 1) else c
            for i, c in enumerate(content)
        ]
    )

    actual = cudf.read_text(
        chess_file_compressed,
        compression="bgzip",
        compression_offsets=[58 * 2**16 + 2, 781 * 2**16 + 7],
        delimiter=delimiter,
    )

    assert_eq(expected, actual)
