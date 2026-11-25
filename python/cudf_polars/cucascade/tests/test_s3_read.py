# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet read from a local S3 server through the cache, as cudf-polars reads it."""

import polars as pl
import pylibcudf as plc
import pytest

from cudf_polars_cucascade import ScanCache, ScanCacheConfig

moto_server = pytest.importorskip("moto.server")
boto3 = pytest.importorskip("boto3")

BUCKET = "bucket"
KEY = "table/part.0.parquet"


@pytest.fixture
def s3_server(tmp_path):
    """A local S3 server holding one parquet file: (server, endpoint, local copy)."""
    path = tmp_path / "part.0.parquet"
    pl.DataFrame(
        {
            "a": range(100_000),
            "b": [f"row {i}" for i in range(100_000)],
            "c": [i / 7 for i in range(100_000)],
        }
    ).write_parquet(path, row_group_size=10_000)
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        client.create_bucket(Bucket=BUCKET)
        client.upload_file(str(path), BUCKET, KEY)
        yield server, endpoint, path
    finally:
        server.stop()


def _read(source):
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.types.SourceInfo([source])
    ).build()
    return plc.io.parquet.read_parquet(options).tbl.to_arrow()


def test_second_read_is_served_from_the_cache(s3_server):
    server, endpoint, path = s3_server
    cache = ScanCache(
        ScanCacheConfig(
            cache_bytes=256 << 20,
            endpoint=endpoint,
            region="us-east-1",
            access_key="test",
            secret_key="test",
            # Every column chunk of this small file is below the default.
            min_advise_bytes=1,
        )
    )
    expected = _read(str(path))
    source = cache.open(f"s3://{BUCKET}/{KEY}")
    assert source.nbytes() == path.stat().st_size
    assert _read(source).equals(expected)
    # With the server gone, the same read can only come from the cache.
    server.stop()
    assert _read(source).equals(expected)
