# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cudf_polars.utils.uri import strip_file_uri


@pytest.mark.parametrize(
    "path, expected",
    [
        ("file:///abs/path.parquet", "/abs/path.parquet"),
        ("file:/abs/path.parquet", "/abs/path.parquet"),
        ("FILE:///abs/path.parquet", "/abs/path.parquet"),
        ("/abs/path.parquet", "/abs/path.parquet"),
        ("relative/path.parquet", "relative/path.parquet"),
        ("file.parquet", "file.parquet"),
        ("file:relative.parquet", "file:relative.parquet"),
        ("s3://bucket/key.parquet", "s3://bucket/key.parquet"),
        ("gs://bucket/key.parquet", "gs://bucket/key.parquet"),
        (
            "abfs://container@account/key.parquet",
            "abfs://container@account/key.parquet",
        ),
        ("http://host/file.parquet", "http://host/file.parquet"),
    ],
)
def test_strip_file_uri(path: str, expected: str) -> None:
    assert strip_file_uri(path) == expected
