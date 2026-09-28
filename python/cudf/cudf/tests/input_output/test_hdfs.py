# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import fastavro
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq

pytest.skip(
    reason="https://github.com/NVIDIA/cudf/issues/19633",
    allow_module_level=True,
)


basedir = "/tmp/test-hdfs"
host = "localhost"  # hadoop hostname
port = 9000  # hadoop rpc port


@pytest.fixture
def hdfs(scope="module"):
    # Default Rpc port can be 8020/9000 depending on the hdfs config
    fs = pa.hdfs.connect(host=host, port=port)
    try:
        if not fs.exists(basedir):
            fs.mkdir(basedir)
    except pa.lib.ArrowIOError:
        pytest.skip("hdfs config probably incorrect")

    return fs


@pytest.fixture
def pdf(scope="module"):
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277], dtype="uint64")
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


@pytest.mark.parametrize("test_url", [False, True])
def test_read_csv(tmpdir, pdf, hdfs, test_url):
    fname = tmpdir.mkdir("csv").join("file.csv")
    # Write to local file system
    pdf.to_csv(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/test_csv_reader.csv", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/test_csv_reader.csv".format(
            host, port, basedir
        )
    else:
        hd_fpath = f"hdfs://{basedir}/test_csv_reader.csv"

    got = cudf.read_csv(hd_fpath)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/test_csv_reader.csv") as f:
        expect = pd.read_csv(f)

    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_write_csv(pdf, hdfs, test_url):
    gdf = cudf.from_pandas(pdf)
    if test_url:
        hd_fpath = "hdfs://{}:{}{}/test_csv_writer.csv".format(
            host, port, basedir
        )
    else:
        hd_fpath = f"hdfs://{basedir}/test_csv_writer.csv"

    gdf.to_csv(hd_fpath, index=False)

    assert hdfs.exists(f"{basedir}/test_csv_writer.csv")
    with hdfs.open(f"{basedir}/test_csv_writer.csv", mode="rb") as f:
        got = pd.read_csv(f, dtype=dict(pdf.dtypes))
    assert_eq(pdf, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_read_parquet(tmpdir, pdf, hdfs, test_url):
    fname = tmpdir.mkdir("parquet").join("test_parquet_reader.parquet")
    # Write to local file system
    pdf.to_parquet(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/test_parquet_reader.parquet", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/test_parquet_reader.parquet".format(
            host, port, basedir
        )
    else:
        hd_fpath = f"hdfs://{basedir}/test_parquet_reader.parquet"

    got = cudf.read_parquet(hd_fpath)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/test_parquet_reader.parquet") as f:
        expect = pd.read_parquet(f)

    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_write_parquet(pdf, hdfs, test_url):
    gdf = cudf.from_pandas(pdf)
    if test_url:
        hd_fpath = "hdfs://{}:{}{}/test_parquet_writer.parquet".format(
            host, port, basedir
        )
    else:
        hd_fpath = f"hdfs://{basedir}/test_parquet_writer.parquet"

    gdf.to_parquet(hd_fpath)

    assert hdfs.exists(f"{basedir}/test_parquet_writer.parquet")
    with hdfs.open(f"{basedir}/test_parquet_writer.parquet", mode="rb") as f:
        got = pd.read_parquet(f)

    assert_eq(pdf, got)


@pytest.mark.xfail(
    reason="Writing string columns with parition_cols is incorrect"
)
@pytest.mark.parametrize("test_url", [False, True])
def test_write_parquet_partitioned(tmpdir, pdf, hdfs, test_url):
    pdf.to_parquet(
        path=tmpdir.join("pandas_parquet_writer_partitioned.parquet"),
        index=False,
        partition_cols=["Integer", "Boolean"],
    )
    gdf = cudf.from_pandas(pdf)
    if test_url:
        hd_fpath = "hdfs://{}:{}{}/test_parquet_partitioned.parquet".format(
            host, port, basedir
        )
    else:
        hd_fpath = f"hdfs://{basedir}/test_parquet_partitioned.parquet"
    # Clear data written from previous runs
    hdfs.rm(f"{basedir}/test_parquet_partitioned.parquet", recursive=True)
    gdf.to_parquet(
        hd_fpath, index=False, partition_cols=["Integer", "Boolean"]
    )

    assert hdfs.exists(f"{basedir}/test_parquet_partitioned.parquet")
    got = pd.read_parquet(hd_fpath)
    expect = pd.read_parquet(
        tmpdir.join("pandas_parquet_writer_partitioned.parquet")
    )
    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_read_json(tmpdir, pdf, hdfs, test_url):
    fname = tmpdir.mkdir("json").join("test_json_reader.json")
    # Write to local file system
    # Sorting by col_name now as pandas sorts by col name while reading json

    pdf.sort_index(axis=1).to_json(fname, orient="records", lines=True)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/test_json_reader.json", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/test_json_reader.json".format(
            host, port, basedir
        )
    else:
        hd_fpath = f"hdfs://{basedir}/test_json_reader.json"

    got = cudf.read_json(hd_fpath, engine="cudf", orient="records", lines=True)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/test_json_reader.json") as f:
        expect = pd.read_json(f, lines=True)

    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_read_orc(datadir, hdfs, test_url):
    fname = datadir / "orc" / "TestOrcFile.testSnappy.orc"
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.orc", buffer)

    if test_url:
        hd_fpath = f"hdfs://{host}:{port}{basedir}/file.orc"
    else:
        hd_fpath = f"hdfs://{basedir}/file.orc"

    got = cudf.read_orc(hd_fpath)
    expect = pd.read_orc(buffer)
    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_write_orc(pdf, hdfs, test_url):
    # Orc writer doesn't support writing unsigned ints
    pdf["Integer2"] = pdf["Integer2"].astype("int64")
    gdf = cudf.from_pandas(pdf)
    if test_url:
        hd_fpath = "hdfs://{}:{}{}/test_orc_writer.orc".format(
            host, port, basedir
        )
    else:
        hd_fpath = f"hdfs://{basedir}/test_orc_writer.orc"

    gdf.to_orc(hd_fpath)

    assert hdfs.exists(f"{basedir}/test_orc_writer.orc")
    with hdfs.open(f"{basedir}/test_orc_writer.orc", mode="rb") as f:
        got = pd.read_orc(f)

    assert_eq(pdf, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_read_avro(datadir, hdfs, test_url):
    fname = datadir / "avro" / "example.avro"
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.avro", buffer)

    if test_url:
        hd_fpath = f"hdfs://{host}:{port}{basedir}/file.avro"
    else:
        hd_fpath = f"hdfs://{basedir}/file.avro"

    got = cudf.read_avro(hd_fpath)
    with open(fname, mode="rb") as f:
        expect = pd.DataFrame.from_records(fastavro.reader(f))

    for col in expect.columns:
        expect[col] = expect[col].astype(got[col].dtype)
    assert_eq(expect, got)


def test_storage_options(tmpdir, pdf, hdfs):
    fname = tmpdir.mkdir("csv").join("file.csv")
    # Write to local file system
    pdf.to_csv(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.csv", buffer)

    hd_fpath = f"hdfs://{basedir}/file.csv"

    storage_options = {"host": host, "port": port}

    got = cudf.read_csv(hd_fpath, storage_options=storage_options)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/file.csv") as f:
        expect = pd.read_csv(f)

    assert_eq(expect, got)


def test_storage_options_error(tmpdir, pdf, hdfs):
    fname = tmpdir.mkdir("csv").join("file.csv")
    # Write to local file system
    pdf.to_csv(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.csv", buffer)

    hd_fpath = f"hdfs://{host}:{port}{basedir}/file.avro"

    storage_options = {"host": host, "port": port}

    with pytest.raises(KeyError):
        cudf.read_csv(hd_fpath, storage_options=storage_options)
