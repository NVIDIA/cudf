# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import uuid
from io import BytesIO, StringIO

import pandas as pd
import pytest
from fsspec.core import get_fs_token_paths

import cudf
from cudf.io.parquet import ParquetDatasetWriter
from cudf.testing import assert_eq


@pytest.fixture(scope="module")
def monkeymodule():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def aws_credentials(monkeymodule):
    """Mocked AWS Credentials for moto."""
    monkeymodule.setenv("AWS_ACCESS_KEY_ID", "foobar_key")
    monkeymodule.setenv("AWS_SECRET_ACCESS_KEY", "foobar_secret")
    monkeymodule.setenv("S3FS_LOGGING_LEVEL", "DEBUG")
    monkeymodule.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeymodule.setenv("AWS_SESSION_TOKEN", "foobar_session_token")
    monkeymodule.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(scope="module")
def moto_server(aws_credentials, monkeymodule):
    """
    Fixture to set up moto server in separate process
    """
    moto_server = pytest.importorskip("moto.server")
    server = moto_server.ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    url = f"http://{host}:{port}"
    monkeymodule.setenv("AWS_ENDPOINT_URL", url)
    yield url
    server.stop()


@pytest.fixture
def s3so(moto_server):
    """
    Returns s3 storage options to pass to fsspec
    """
    return {"client_kwargs": {"endpoint_url": moto_server}}


@pytest.fixture(params=["storage_options", "filesystem"])
def fs_kwargs(request, s3so):
    """
    Yield the two equivalent ways of pointing an I/O call at the mock S3:
    ``storage_options=<dict>`` and ``filesystem=<fsspec object>``.
    """
    if request.param == "storage_options":
        return {"storage_options": s3so}
    return {"filesystem": get_fs_token_paths("s3://", storage_options=s3so)[0]}


@pytest.fixture
def moto_s3_resource(moto_server):
    boto3 = pytest.importorskip("boto3")
    s3 = boto3.resource("s3", endpoint_url=moto_server)
    return s3


@pytest.fixture
def s3_bucket_public(moto_s3_resource):
    """
    Create a public S3 bucket using moto.
    """
    bucket_name = f"cudf-test-{uuid.uuid4()}"
    bucket = moto_s3_resource.Bucket(bucket_name)
    bucket.create(ACL="public-read-write")
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture(
    params=[True, False],
    ids=["kvikio=ON", "kvikio=OFF"],
)
def kvikio_remote_io(request):
    with cudf.option_context("kvikio_remote_io", request.param):
        yield request.param


@pytest.fixture
def pdf():
    return pd.DataFrame(
        {
            "Integer": [2345, 11987, 9027, 9027],
            "Float": [9.001, 8.343, 6, 2.781],
            "Integer2": [2345, 106, 2088, 789277],
            "String": ["Alpha", "Beta", "Gamma", "Delta"],
            "Boolean": [True, False, True, False],
        }
    )


@pytest.fixture
def pdf_ext():
    size = 10
    return pd.DataFrame(
        {
            "Integer": [i for i in range(size)],
            "List": [[i] for i in range(size)],
            "Struct": [{"a": i} for i in range(size)],
            "String": (["Alpha", "Beta", "Gamma", "Delta"] * (-(size // -4)))[
                :size
            ],
        }
    )


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
def test_read_csv(s3_bucket_public, fs_kwargs, pdf, bytes_per_thread):
    # Write to buffer
    fname = "test_csv_reader.csv"
    buffer = pdf.to_csv(index=False)
    s3_bucket_public.put_object(Key=fname, Body=buffer)

    # Use fsspec file object
    got = cudf.read_csv(
        f"s3://{s3_bucket_public.name}/{fname}",
        bytes_per_thread=bytes_per_thread,
        **fs_kwargs,
    )
    assert_eq(pdf, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
def test_read_csv_byte_range(s3_bucket_public, s3so, pdf, bytes_per_thread):
    # Write to buffer
    fname = "test_csv_reader_byte_range.csv"
    buffer = pdf.to_csv(index=False)
    s3_bucket_public.put_object(Key=fname, Body=buffer)

    # Use fsspec file object
    got = cudf.read_csv(
        f"s3://{s3_bucket_public.name}/{fname}",
        storage_options=s3so,
        byte_range=(74, 73),
        bytes_per_thread=bytes_per_thread,
        header=None,
        names=["Integer", "Float", "Integer2", "String", "Boolean"],
    )

    assert_eq(pdf.iloc[-2:].reset_index(drop=True), got)


@pytest.mark.parametrize("chunksize", [None, 3])
def test_write_csv(s3_bucket_public, s3so, fs_kwargs, pdf, chunksize):
    # Write to buffer
    fname = "test_csv_writer.csv"
    gdf = cudf.from_pandas(pdf)
    gdf.to_csv(
        f"s3://{s3_bucket_public.name}/{fname}",
        index=False,
        chunksize=chunksize,
        **fs_kwargs,
    )
    got = pd.read_csv(
        f"s3://{s3_bucket_public.name}/{fname}", storage_options=s3so
    )

    assert_eq(pdf, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
@pytest.mark.parametrize("columns", [None, ["Float", "String"]])
def test_read_parquet(
    s3_bucket_public,
    s3so,
    fs_kwargs,
    kvikio_remote_io,
    pdf,
    bytes_per_thread,
    columns,
):
    fname = "test_parquet_reader.parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)

    # Check direct path handling
    buffer.seek(0)
    s3_bucket_public.put_object(Key=fname, Body=buffer)
    got1 = cudf.read_parquet(
        f"s3://{s3_bucket_public.name}/{fname}",
        bytes_per_thread=bytes_per_thread,
        columns=columns,
        **fs_kwargs,
    )
    expect = pdf[columns] if columns else pdf
    assert_eq(expect, got1)

    # Check fsspec file-object handling
    buffer.seek(0)
    fs = get_fs_token_paths(
        f"s3://{s3_bucket_public.name}/{fname}", storage_options=s3so
    )[0]
    with fs.open(f"s3://{s3_bucket_public.name}/{fname}", mode="rb") as f:
        got2 = cudf.read_parquet(
            f,
            bytes_per_thread=bytes_per_thread,
            columns=columns,
        )
    assert_eq(expect, got2)


@pytest.mark.parametrize("method", ["all", "parquet"])
@pytest.mark.parametrize("blocksize", [1024 * 1024, 1024])
def test_read_parquet_prefetch_options(
    s3_bucket_public,
    s3so,
    pdf,
    method,
    blocksize,
):
    fname_1 = "test_parquet_reader_prefetch_options_1.parquet"
    buffer_1 = BytesIO()
    pdf.to_parquet(path=buffer_1)
    buffer_1.seek(0)
    s3_bucket_public.put_object(Key=fname_1, Body=buffer_1)

    fname_2 = "test_parquet_reader_prefetch_options_2.parquet"
    buffer_2 = BytesIO()
    pdf_2 = pdf.copy()
    pdf_2["Integer"] += 1
    pdf_2.to_parquet(path=buffer_2)
    buffer_2.seek(0)
    s3_bucket_public.put_object(Key=fname_2, Body=buffer_2)

    got = cudf.read_parquet(
        [
            f"s3://{s3_bucket_public.name}/{fname_1}",
            f"s3://{s3_bucket_public.name}/{fname_2}",
        ],
        storage_options=s3so,
        prefetch_options={
            "method": method,
            "blocksize": blocksize,
        },
        columns=["String", "Integer"],
    )

    expect = pd.concat([pdf, pdf_2], ignore_index=True)[["String", "Integer"]]
    assert_eq(expect, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
@pytest.mark.parametrize("columns", [None, ["List", "Struct"]])
@pytest.mark.parametrize("index", [None, "Integer"])
def test_read_parquet_ext(
    s3_bucket_public,
    s3so,
    pdf_ext,
    bytes_per_thread,
    columns,
    index,
):
    fname = "test_parquet_reader_ext.parquet"
    buffer = BytesIO()

    if index:
        pdf_ext.set_index(index).to_parquet(path=buffer)
    else:
        pdf_ext.to_parquet(path=buffer)

    # Check direct path handling
    buffer.seek(0)
    s3_bucket_public.put_object(Key=fname, Body=buffer)
    got1 = cudf.read_parquet(
        f"s3://{s3_bucket_public.name}/{fname}",
        storage_options=s3so,
        bytes_per_thread=bytes_per_thread,
        columns=columns,
    )
    if index:
        expect = (
            pdf_ext.set_index(index)[columns]
            if columns
            else pdf_ext.set_index(index)
        )
    else:
        expect = pdf_ext[columns] if columns else pdf_ext
    assert_eq(expect, got1)


def test_read_parquet_filesystem(s3_bucket_public, s3so, pdf):
    fname = "data.0.parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    fs = get_fs_token_paths("s3://", mode="rb", storage_options=s3so)[0]
    s3_bucket_public.put_object(Key=fname, Body=buffer)
    # Check that a glob pattern works
    path = f"s3://{s3_bucket_public.name}/{'data.*.parquet'}"
    got = cudf.read_parquet(path, filesystem=fs)
    assert_eq(pdf, got)


def test_read_parquet_multi_file(s3_bucket_public, s3so, pdf):
    fname_1 = "test_parquet_reader_multi_file_1.parquet"
    buffer_1 = BytesIO()
    pdf.to_parquet(path=buffer_1)
    buffer_1.seek(0)
    s3_bucket_public.put_object(Key=fname_1, Body=buffer_1)

    fname_2 = "test_parquet_reader_multi_file_2.parquet"
    buffer_2 = BytesIO()
    pdf.to_parquet(path=buffer_2)
    buffer_2.seek(0)
    s3_bucket_public.put_object(Key=fname_2, Body=buffer_2)

    got = cudf.read_parquet(
        [
            f"s3://{s3_bucket_public.name}/{fname_1}",
            f"s3://{s3_bucket_public.name}/{fname_2}",
        ],
        storage_options=s3so,
    ).reset_index(drop=True)

    expect = pd.concat([pdf, pdf], ignore_index=True)
    assert_eq(expect, got)


def test_read_parquet_filters(s3_bucket_public, s3so, pdf_ext):
    fname = "test_parquet_reader_filters.parquet"
    buffer = BytesIO()
    pdf_ext.to_parquet(path=buffer)
    buffer.seek(0)
    s3_bucket_public.put_object(Key=fname, Body=buffer)
    filters = [("String", "==", "Omega")]
    got = cudf.read_parquet(
        f"s3://{s3_bucket_public.name}/{fname}",
        storage_options=s3so,
        filters=filters,
    )

    # All row-groups should be filtered out
    assert_eq(pdf_ext.iloc[:0], got.reset_index(drop=True))


@pytest.mark.parametrize("partition_cols", [None, ["String"]])
def test_write_parquet(s3_bucket_public, s3so, fs_kwargs, pdf, partition_cols):
    fname_cudf = "test_parquet_writer_cudf"
    fname_pandas = "test_parquet_writer_pandas"
    gdf = cudf.from_pandas(pdf)

    gdf.to_parquet(
        f"s3://{s3_bucket_public.name}/{fname_cudf}",
        partition_cols=partition_cols,
        **fs_kwargs,
    )
    pdf.to_parquet(
        f"s3://{s3_bucket_public.name}/{fname_pandas}",
        partition_cols=partition_cols,
        storage_options=s3so,
    )

    got = pd.read_parquet(
        f"s3://{s3_bucket_public.name}/{fname_pandas}", storage_options=s3so
    )
    expect = cudf.read_parquet(
        f"s3://{s3_bucket_public.name}/{fname_cudf}", storage_options=s3so
    )

    assert_eq(expect, got)


def test_read_json(s3_bucket_public, fs_kwargs):
    fname = "test_json_reader.json"
    buffer = (
        '{"amount": 100, "name": "Alice"}\n'
        '{"amount": 200, "name": "Bob"}\n'
        '{"amount": 300, "name": "Charlie"}\n'
        '{"amount": 400, "name": "Dennis"}\n'
    )

    s3_bucket_public.put_object(Key=fname, Body=buffer)
    got = cudf.read_json(
        f"s3://{s3_bucket_public.name}/{fname}",
        engine="cudf",
        orient="records",
        lines=True,
        **fs_kwargs,
    )

    expect = pd.read_json(StringIO(buffer), lines=True)
    assert_eq(expect, got)


def test_read_jsonl_files_kvikio_remote(s3_bucket_public, s3so, monkeypatch):
    fname1 = "test_json_reader1.jsonl"
    fname2 = "test_json_reader2.jsonl"
    buffer1 = (
        '{"amount": 100, "name": "Alice"}\n{"amount": 200, "name": "Bob"}\n'
    )
    buffer2 = (
        '{"amount": 300, "name": "Charlie"}\n'
        '{"amount": 400, "name": "Dennis"}\n'
    )
    s3_bucket_public.put_object(Key=fname1, Body=buffer1)
    s3_bucket_public.put_object(Key=fname2, Body=buffer2)
    with cudf.option_context("kvikio_remote_io", True):
        got = cudf.read_json(
            [
                f"s3://{s3_bucket_public.name}/{fname1}",
                f"s3://{s3_bucket_public.name}/{fname2}",
            ],
            engine="cudf",
            lines=True,
            storage_options=s3so,
        )
    expected = cudf.DataFrame(
        {
            "amount": [100, 200, 300, 400],
            "name": ["Alice", "Bob", "Charlie", "Dennis"],
        }
    )
    assert_eq(expected, got)


@pytest.mark.parametrize("columns", [None, ["string1"]])
def test_read_orc(s3_bucket_public, fs_kwargs, datadir, columns):
    source_file = str(datadir / "orc" / "TestOrcFile.testSnappy.orc")
    fname = "test_orc_reader.orc"
    expect = pd.read_orc(source_file)

    with open(source_file, "rb") as f:
        buffer = f.read()

    s3_bucket_public.put_object(Key=fname, Body=buffer)
    got = cudf.read_orc(
        f"s3://{s3_bucket_public.name}/{fname}",
        columns=columns,
        **fs_kwargs,
    )

    if columns:
        expect = expect[columns]
    assert_eq(expect, got)


def test_write_orc(s3_bucket_public, fs_kwargs, pdf):
    fname = "test_orc_writer.orc"
    gdf = cudf.from_pandas(pdf)
    gdf.to_orc(f"s3://{s3_bucket_public.name}/{fname}", **fs_kwargs)

    got = pd.read_orc(f"s3://{s3_bucket_public.name}/{fname}")

    assert_eq(pdf, got)


def test_write_chunked_parquet(s3_bucket_public, s3so, fs_kwargs):
    df1 = cudf.DataFrame({"b": [10, 11, 12], "a": [1, 2, 3]})
    df2 = cudf.DataFrame({"b": [20, 30, 50], "a": [3, 2, 1]})
    dirname = "chunked_writer_directory"
    with ParquetDatasetWriter(
        f"s3://{s3_bucket_public.name}/{dirname}",
        partition_cols=["a"],
        **fs_kwargs,
    ) as cw:
        cw.write_table(df1)
        cw.write_table(df2)

    # TODO: Replace following workaround with:
    # expect = cudf.read_parquet(f"s3://{bucket}/{dirname}/",
    # storage_options=s3so)
    # after the following bug is fixed:
    # https://issues.apache.org/jira/browse/ARROW-16438

    dfs = [
        cudf.read_parquet(
            f"s3://{s3_bucket_public.name}/{file.key}", storage_options=s3so
        )
        for file in s3_bucket_public.objects.all()
    ]

    actual = cudf.concat(dfs).astype("int64")
    assert_eq(
        actual.sort_values(["b"]).reset_index(drop=True),
        cudf.concat([df1, df2]).sort_values(["b"]).reset_index(drop=True),
    )


def test_filesystem_and_storage_options_are_exclusive(
    s3_bucket_public, s3so, pdf
):
    fs = get_fs_token_paths("s3://", storage_options=s3so)[0]
    path = f"s3://{s3_bucket_public.name}/exclusive.csv"
    match = "Cannot specify storage_options"

    with pytest.raises(ValueError, match=match):
        cudf.read_csv(path, filesystem=fs, storage_options=s3so)

    with pytest.raises(ValueError, match=match):
        cudf.from_pandas(pdf).to_csv(path, filesystem=fs, storage_options=s3so)


def test_no_s3fs_on_cudf_import():
    output = subprocess.check_call(
        [
            sys.executable,
            "-c",
            "import cudf, sys; assert 'pyarrow._s3fs' not in sys.modules",
        ],
        cwd="/",
    )
    assert output == 0
