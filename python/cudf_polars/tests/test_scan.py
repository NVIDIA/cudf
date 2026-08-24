# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime as dt
import gzip
import zlib
from decimal import Decimal
from typing import TYPE_CHECKING
from urllib.parse import quote

import numpy as np
import pytest
import zstandard as zstd
from werkzeug import Response

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import IRExecutionContext, Scan
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions, ParquetOptions
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_138,
    POLARS_VERSION_LT_139,
    POLARS_VERSION_LT_142,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_httpserver import HTTPServer
    from werkzeug import Request


NO_CHUNK_ENGINE = pl.GPUEngine(
    executor="in-memory", raise_on_fail=True, parquet_options={"chunked": False}
)


@pytest.fixture(
    params=[(None, 0), ("row-index", 0), ("index", 10)],
    ids=["no_row_index", "zero_offset_row_index", "offset_row_index"],
)
def row_index(request) -> tuple[str | None, int]:
    return request.param


@pytest.fixture(
    params=[None, 3],
    ids=["all_rows", "some_rows"],
)
def n_rows(request):
    return request.param


@pytest.fixture(scope="module")
def df():
    # TODO: more dtypes
    return pl.DataFrame(
        {
            "a": [1, 2, 3, None, 4, 5],
            "b": ["ẅ", "x", "y", "z", "123", "abcd"],
            "c": [None, None, 4, 5, -1, 0],
            "d": [
                Decimal("1.23"),
                None,
                Decimal("0.00"),
                None,
                Decimal("-5.67"),
                None,
            ],
        },
        schema={"a": pl.Int64, "b": pl.String, "c": pl.Int32, "d": pl.Decimal(15, 2)},
    )


@pytest.fixture(params=[None, ["a"], ["b", "a"]], ids=["all", "subset", "reordered"])
def columns(request, row_index):
    name, _ = row_index
    if name is not None and request.param is not None:
        return [name, *request.param]
    return request.param


@pytest.fixture(
    params=[None, pl.col("c").is_not_null()], ids=["no_mask", "c_is_not_null"]
)
def mask(request):
    return request.param


@pytest.fixture(
    params=[None, (1, 1)],
    ids=["no_slice", "slice_second"],
)
def zlice(request):
    # For use in testing that we handle
    # polars slice pushdown correctly
    return request.param


@pytest.fixture(params=["csv", "ndjson", "parquet", "chunked_parquet"])
def format(request):
    return request.param


@pytest.fixture
def scan_fn(format):
    if format == "csv":
        return pl.scan_csv
    elif format == "ndjson":
        return pl.scan_ndjson
    else:
        return pl.scan_parquet


def test_scan(
    tmp_path, df, format, scan_fn, row_index, n_rows, columns, mask, zlice, request
):
    name, offset = row_index
    is_chunked = format == "chunked_parquet"
    if is_chunked:
        format = "parquet"
    make_partitioned_source(df, tmp_path / "file", format)
    request.applymarker(
        pytest.mark.xfail(
            condition=(n_rows is not None and scan_fn is pl.scan_ndjson),
            reason="libcudf does not support n_rows",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=(zlice is not None and scan_fn is pl.scan_ndjson),
            reason="slice pushdown not supported in the libcudf JSON reader",
        )
    )
    q = scan_fn(
        tmp_path / "file",
        row_index_name=name,
        row_index_offset=offset,
        n_rows=n_rows,
    )
    engine = pl.GPUEngine(
        executor="in-memory",
        raise_on_fail=True,
        parquet_options={"chunked": is_chunked},
    )

    if zlice is not None:
        q = q.slice(*zlice)
    if mask is not None:
        q = q.filter(mask)
    if columns is not None:
        q = q.select(*columns)
    assert_gpu_result_equal(q, engine=engine)


def test_negative_slice_pushdown_raises(engine: pl.GPUEngine, tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_parquet(tmp_path / "df.parquet")
    q = pl.scan_parquet(tmp_path / "df.parquet")
    # Take the last row
    q = q.slice(-1, 1)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_parquet_prefetch_file_metadata_in_memory_raises():
    with pytest.raises(
        NotImplementedError,
        match=r"Prefetching is not supported for the in-memory executor.",
    ):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="in-memory",
                parquet_options=ParquetOptions(prefetch_file_metadata=True),
            )
        )


def test_scan_do_evaluate_missing_prefetch_metadata() -> None:
    paths = ["/some/missing/file.parquet"]
    parquet_options = ParquetOptions(prefetch_file_metadata=True)
    context = IRExecutionContext()
    schema = {"a": DataType(pl.Int64())}

    with pytest.raises(
        AssertionError,
        match=(r"Paths do not match cached parquet info."),
    ):
        Scan.do_evaluate(
            schema,
            "parquet",
            {},
            paths,
            None,
            0,
            -1,
            None,
            None,
            None,
            parquet_options,
            [],
            context=context,
        )


def test_scan_unsupported_raises(engine: pl.GPUEngine, tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_ipc(tmp_path / "df.ipc")
    q = pl.scan_ipc(tmp_path / "df.ipc")
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_ndjson_nrows_notimplemented(engine: pl.GPUEngine, tmp_path, df):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_ndjson(tmp_path / "df.jsonl")
    q = pl.scan_ndjson(tmp_path / "df.jsonl", n_rows=1)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_row_index_projected_out(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_parquet(tmp_path / "df.pq")

    q = pl.scan_parquet(tmp_path / "df.pq").with_row_index().select(pl.col("a"))

    assert_gpu_result_equal(q, engine=NO_CHUNK_ENGINE)


@pytest.mark.parametrize("chunked", [False, True])
def test_scan_parquet_pandas_index_projected_out(tmp_path, chunked):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_parquet(
        tmp_path / "pdf.pq", engine="pyarrow", index=True
    )
    q = pl.scan_parquet(tmp_path / "pdf.pq").select("b")

    engine = pl.GPUEngine(
        executor="in-memory",
        raise_on_fail=True,
        parquet_options={"chunked": chunked},
    )
    assert_gpu_result_equal(q, engine=engine)


def test_scan_csv_column_renames_projection_schema(engine: pl.GPUEngine, tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2\n3,4,5""")

    q = pl.scan_csv(
        tmp_path / "test.csv",
        with_column_names=lambda names: [f"{n}_suffix" for n in names],
        schema_overrides={
            "foo_suffix": pl.String(),
            "bar_suffix": pl.Int8(),
            "baz_suffix": pl.UInt16(),
        },
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "filename,glob",
    [
        (["test1.csv", "test2.csv"], True),
        ("test*.csv", True),
        # Make sure we don't expand glob when
        # trying to read a file like test*.csv
        # when glob=False
        ("test*.csv", False),
    ],
)
@pytest.mark.parametrize(
    "nrows_skiprows",
    [
        (None, 0),
        (1, 1),
        (3, 0),
        (4, 2),
    ],
)
def test_scan_csv_multi(engine: pl.GPUEngine, tmp_path, filename, glob, nrows_skiprows):
    n_rows, skiprows = nrows_skiprows
    with (tmp_path / "test1.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")
    with (tmp_path / "test2.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")
    with (tmp_path / "test*.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")
    if isinstance(filename, list):
        source = [tmp_path / fn for fn in filename]
    else:
        source = tmp_path / filename
    q = pl.scan_csv(source, glob=glob, n_rows=n_rows, skip_rows=skiprows)

    assert_gpu_result_equal(q, engine=engine)


def test_scan_csv_multi_differing_colnames(tmp_path):
    with (tmp_path / "test1.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2\n3,4,5""")
    with (tmp_path / "test2.csv").open("w") as f:
        f.write("""abc,def,ghi\n1,2\n3,4,5""")
    q = pl.scan_csv(
        [tmp_path / "test1.csv", tmp_path / "test2.csv"],
    )
    with pytest.raises(pl.exceptions.ComputeError):
        q.explain()


def test_scan_csv_skip_after_header_not_implemented(engine: pl.GPUEngine, tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", skip_rows_after_header=1)

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_csv_null_values_per_column_not_implemented(
    engine: pl.GPUEngine, tmp_path
):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", null_values={"foo": "1", "baz": "5"})

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_csv_comment_str_not_implemented(engine: pl.GPUEngine, tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n// 1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", comment_prefix="// ")

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_csv_comment_char(engine: pl.GPUEngine, tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n# 1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", comment_prefix="#")

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("nulls", [None, "3", ["3", "5"]])
def test_scan_csv_null_values(engine: pl.GPUEngine, tmp_path, nulls):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5\n5,,2""")

    q = pl.scan_csv(tmp_path / "test.csv", null_values=nulls)

    assert_gpu_result_equal(q, engine=engine)


def test_scan_csv_decimal_comma(engine: pl.GPUEngine, tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo|bar|baz\n1,23|2,34|3,56\n1""")

    q = pl.scan_csv(tmp_path / "test.csv", separator="|", decimal_comma=True)

    assert_gpu_result_equal(q, engine=engine)


def test_scan_csv_skip_initial_empty_rows(engine: pl.GPUEngine, tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""\n\n\n\nfoo|bar|baz\n1|2|3\n1""")

    q = pl.scan_csv(tmp_path / "test.csv", separator="|", skip_rows=1, has_header=False)

    assert_ir_translation_raises(q, engine, NotImplementedError)

    q = pl.scan_csv(tmp_path / "test.csv", separator="|", skip_rows=1)

    assert_gpu_result_equal(q, engine=engine)


def test_scan_csv_slice_end_none(engine: pl.GPUEngine, tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""c0\ntrue\nfalse""")

    q = pl.scan_csv(tmp_path / "test.csv").slice(10, None)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "schema",
    [
        # List of colnames (basicaly like names param in CSV)
        {"b": pl.String, "a": pl.Float32},
        {"a": pl.UInt64},
    ],
)
def test_scan_ndjson_schema(engine: pl.GPUEngine, df, tmp_path, schema):
    make_partitioned_source(df, tmp_path / "file", "ndjson")
    q = pl.scan_ndjson(tmp_path / "file", schema=schema)
    assert_gpu_result_equal(q, engine=engine)


def test_scan_ndjson_unsupported(engine: pl.GPUEngine, df, tmp_path):
    make_partitioned_source(df, tmp_path / "file", "ndjson")
    q = pl.scan_ndjson(tmp_path / "file", ignore_errors=True)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_parquet_nested_null_raises(engine: pl.GPUEngine, tmp_path):
    df = pl.DataFrame({"a": pl.Series([None], dtype=pl.List(pl.Null))})

    df.write_parquet(tmp_path / "file.pq")

    q = pl.scan_parquet(tmp_path / "file.pq")

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_parquet_only_row_index_raises(engine: pl.GPUEngine, df, tmp_path):
    make_partitioned_source(df, tmp_path / "file", "parquet")
    q = pl.scan_parquet(tmp_path / "file", row_index_name="index").select("index")
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("n_rows", [None, 2])
def test_scan_include_file_path(
    engine: pl.GPUEngine, request, tmp_path, format, scan_fn, df, n_rows
):
    if n_rows is not None:
        df = df.head(n_rows)
    make_partitioned_source(df, tmp_path / "file", format)

    q = scan_fn(tmp_path / "file", include_file_paths="files", n_rows=n_rows)

    if format == "ndjson":
        assert_ir_translation_raises(q, engine, NotImplementedError)
    else:
        assert_gpu_result_equal(q, engine=NO_CHUNK_ENGINE)


@pytest.fixture(
    scope="module", params=["no_slice", "skip_to_end", "skip_partial", "partial"]
)
def chunked_slice(request):
    return request.param


@pytest.fixture(scope="module")
def chunked_df(df, tmpdir_factory, chunked_slice):
    # Many small row groups so that ``pass_read_limit`` (one pass per row
    # group) and ``chunk_read_limit`` (one output chunk per page within a
    # pass) can force the libcudf chunked reader to return multiple chunks.
    df = pl.concat([df] * 100)
    path = str(tmpdir_factory.mktemp("data") / "chunked.pq")
    make_partitioned_source(df, path, "parquet", row_group_size=10)
    n_rows = len(df)
    q = pl.scan_parquet(path)
    if chunked_slice == "no_slice":
        return q
    elif chunked_slice == "skip_to_end":
        return q.slice(int(n_rows * 0.6), n_rows)
    elif chunked_slice == "skip_partial":
        return q.slice(int(n_rows * 0.6), int(n_rows * 0.2))
    else:
        return q.slice(0, int(n_rows * 0.6))


@pytest.mark.parametrize(
    "chunk_read_limit, pass_read_limit",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 1),
        (1, 2),
    ],
    ids=lambda x: f"limit_{x}",
)
@pytest.mark.parametrize(
    "filter", [None, pl.col("a") > 3], ids=["no_filters", "with_filters"]
)
def test_scan_parquet_chunked(
    chunked_df,
    chunk_read_limit,
    pass_read_limit,
    filter,
):
    if filter is None:
        q = chunked_df
    else:
        q = chunked_df.filter(filter)
    assert_gpu_result_equal(
        q,
        engine=pl.GPUEngine(
            executor="in-memory",
            raise_on_fail=True,
            parquet_options={
                "chunked": True,
                "chunk_read_limit": chunk_read_limit,
                "pass_read_limit": pass_read_limit,
            },
        ),
    )


def test_select_arbitrary_order_with_row_index_column(engine: pl.GPUEngine, tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_parquet(tmp_path / "df.parquet")
    q = pl.scan_parquet(tmp_path / "df.parquet", row_index_name="foo").select(
        [pl.col("a"), pl.col("foo")]
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "has_header,new_columns",
    [
        (True, None),
        (False, ["a", "b", "c"]),
    ],
)
def test_scan_csv_with_and_without_header(
    engine: pl.GPUEngine,
    df: pl.DataFrame,
    tmp_path: Path,
    *,
    has_header: bool,
    new_columns: list[str] | None,
    row_index: tuple[str | None, int],
    columns: list[str] | None,
    zlice: tuple[int, int] | None,
):
    path = tmp_path / "test.csv"
    make_partitioned_source(
        df, path, "csv", write_kwargs={"include_header": has_header}
    )

    name, offset = row_index

    q = pl.scan_csv(
        path,
        has_header=has_header,
        new_columns=new_columns,
        row_index_name=name,
        row_index_offset=offset,
    )

    if zlice is not None:
        q = q.slice(*zlice)
    if columns is not None:
        q = q.select(columns)

    assert_gpu_result_equal(q, engine=engine)


def test_scan_csv_without_header_and_new_column_names_raises(
    engine: pl.GPUEngine, df, tmp_path
):
    path = tmp_path / "test.csv"
    make_partitioned_source(df, path, "csv", write_kwargs={"include_header": False})
    q = pl.scan_csv(path, has_header=False)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_with_row_index(engine: pl.GPUEngine, tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    df.write_csv(tmp_path / "test-0.csv")
    df.write_csv(tmp_path / "test-1.csv")

    q = pl.scan_csv(tmp_path / "test-*.csv", row_index_name="index", row_index_offset=0)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "subdir",
    [
        "foo",
        pytest.param(
            "foo=bar",
            marks=pytest.mark.xfail(
                condition=POLARS_VERSION_LT_142,
                reason="https://github.com/pola-rs/polars/issues/27840",
                strict=True,
            ),
        ),
    ],
)
def test_scan_from_file_uri(engine: pl.GPUEngine, tmp_path: Path, subdir: str) -> None:
    target_dir = tmp_path / subdir
    target_dir.mkdir()
    path = target_dir / "out.parquet"
    df = pl.DataFrame({"a": 1})
    df.write_parquet(path)
    encoded = quote(str(path), safe="/")
    q = pl.scan_parquet(f"file://{encoded}")
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("chunked", [False, True])
def test_scan_parquet_remote(
    tmp_path: Path, df: pl.DataFrame, httpserver: HTTPServer, *, chunked: bool
) -> None:
    path = tmp_path / "foo.parquet"
    df.write_parquet(path)
    bytes_ = path.read_bytes()
    size = len(bytes_)

    def head_handler(_: Request) -> Response:
        return Response(
            status=200,
            headers={
                "Content-Type": "parquet",
                "Accept-Ranges": "bytes",
                "Content-Length": size,
            },
        )

    def get_handler(req: Request) -> Response:
        # parse bytes=200-500 for example (the actual data)
        rng = req.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start, end = map(int, req.headers["Range"][6:].split("-"))
            mv = memoryview(bytes_)[start : end + 1]
            return Response(
                mv.tobytes(),
                status=206,
                headers={
                    "Content-Type": "parquet",
                    "Accept-Ranges": "bytes",
                    "Content-Length": len(mv),
                    "Content-Range": f"bytes {start}-{end}/{size}",
                },
            )
        return Response(
            bytes_,
            status=200,
            headers={
                "Content-Type": "parquet",
                "Accept-Ranges": "bytes",
                "Content-Length": size,
            },
        )

    server_path = "/foo.parquet"
    httpserver.expect_request(server_path, method="HEAD").respond_with_handler(
        head_handler
    )
    httpserver.expect_request(server_path, method="GET").respond_with_handler(
        get_handler
    )

    q = pl.scan_parquet(httpserver.url_for(server_path))

    assert_gpu_result_equal(
        q,
        engine=pl.GPUEngine(
            executor="in-memory",
            raise_on_fail=True,
            parquet_options={"chunked": chunked},
        ),
    )


@pytest.mark.xfail(
    condition=not POLARS_VERSION_LT_139,
    reason="polars 1.39+ ndjson remote reader requires range request support",
)
def test_scan_ndjson_remote(
    engine: pl.GPUEngine,
    tmp_path: Path,
    df: pl.DataFrame,
    httpserver: HTTPServer,
) -> None:
    path = tmp_path / "foo.jsonl"
    df.write_ndjson(path)
    bytes_ = path.read_bytes()
    size = len(bytes_)

    def head_handler(_: Request) -> Response:
        return Response(
            status=200,
            headers={
                "Content-Type": "ndjson",
                "Content-Length": size,
            },
        )

    def get_handler(_: Request) -> Response:
        return Response(
            bytes_,
            status=200,
            headers={
                "Content-Type": "ndjson",
                "Content-Length": size,
            },
        )

    server_path = "/foo.jsonl"
    httpserver.expect_request(server_path, method="HEAD").respond_with_handler(
        head_handler
    )
    httpserver.expect_request(server_path, method="GET").respond_with_handler(
        get_handler
    )

    q = pl.scan_ndjson(httpserver.url_for(server_path))
    assert_gpu_result_equal(q, engine=engine)


def test_scan_parquet_with_decimal_literal_in_predicate(
    engine: pl.GPUEngine, df, tmp_path
):
    make_partitioned_source(df, tmp_path / "file", "parquet")

    q = pl.scan_parquet(tmp_path / "file").filter(
        (pl.col("d") > Decimal("1.23"))
        & (pl.lit(Decimal("2.00")).cast(pl.Decimal(15, 2)) < pl.col("d"))
    )

    assert_gpu_result_equal(q, engine=engine)


def test_scan_csv_blank_line(engine: pl.GPUEngine, tmp_path):
    data = """c0

polars"""
    fle = tmp_path / "test.csv"
    fle.write_text(data)
    q = pl.scan_csv(fle)
    assert_gpu_result_equal(q, engine=engine)


def test_hits_scan_row_index_duplicate(engine: pl.GPUEngine, request, tmp_path):
    request.applymarker(
        pytest.mark.xfail(
            condition=not POLARS_VERSION_LT_138,
            reason="polars >= 1.38 raises duplicate row_index name ahead of time",
        )
    )
    pl.DataFrame({"col": [1, 2, 3]}).write_parquet(tmp_path / "a.parquet")

    q = pl.scan_parquet(tmp_path / "*.parquet", row_index_name="index").with_row_index(
        "index"
    )

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("compression", ["gzip", "zlib", "zstd"])
@pytest.mark.parametrize("file_type", ["csv", "ndjson"])
def test_scan_compressed_file_raises(
    engine: pl.GPUEngine, tmp_path: Path, compression: str, file_type: str
):
    if file_type == "csv":
        data = b"a,b\n1,2\n3,4\n"
        scan_fn: Callable = pl.scan_csv
    else:
        data = b'{"a":1,"b":2}\n{"a":3,"b":4}\n'
        scan_fn = pl.scan_ndjson

    path = tmp_path / f"data.{file_type}"
    if compression == "gzip":
        with gzip.open(path, "wb") as f:
            f.write(data)
    elif compression == "zlib":
        with path.open("wb") as f:
            f.write(zlib.compress(data))
    else:
        cctx = zstd.ZstdCompressor()
        with path.open("wb") as f:
            f.write(cctx.compress(data))

    q = scan_fn(path)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_scan_tiny_file_not_compressed(engine: pl.GPUEngine, tmp_path):
    # code coverage for the case where we try to
    # detect compression but the file is too small
    # to have a valid signature.
    path = tmp_path / "tiny.csv"
    path.write_bytes(b"a\n")
    q = pl.scan_csv(path, has_header=False, new_columns=["a"])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="pl.LazyFrame(height=...) requires polars >= 1.38"
)
@pytest.mark.parametrize("custom_engine", [None, NO_CHUNK_ENGINE])
def test_scan_parquet_zero_width_with_limit(
    engine: pl.GPUEngine, tmp_path, custom_engine
):
    active_engine = custom_engine if custom_engine is not None else engine
    path = tmp_path / "zero_width.parquet"
    pl.LazyFrame(height=20).sink_parquet(path)
    q = pl.scan_parquet(path).head(5)
    assert_gpu_result_equal(q, engine=active_engine)


@pytest.mark.parametrize(
    "column_dtype, lit",
    [
        (pl.Datetime("us"), np.datetime64("2021-01-02")),
        (pl.Datetime("us"), pl.lit(dt.date(2021, 1, 2), dtype=pl.Date)),
        (pl.Datetime("ms"), pl.lit(dt.date(2021, 1, 2), dtype=pl.Date)),
        (pl.Datetime("ns"), pl.lit(dt.date(2021, 1, 2), dtype=pl.Date)),
        (pl.Datetime("ns"), pl.lit(dt.datetime(2021, 1, 2), dtype=pl.Datetime("us"))),
        (pl.Datetime("us"), pl.lit(dt.datetime(2021, 1, 2), dtype=pl.Datetime("ns"))),
        (pl.Datetime("ns"), pl.lit(dt.datetime(2021, 1, 2), dtype=pl.Datetime("ms"))),
        (pl.Duration("us"), pl.lit(dt.timedelta(seconds=1), dtype=pl.Duration("ns"))),
        (pl.Duration("ns"), pl.lit(dt.timedelta(seconds=1), dtype=pl.Duration("us"))),
        (pl.Int32, pl.lit(2, dtype=pl.Int64)),
        (pl.Decimal(15, 2), pl.lit(1.5, dtype=pl.Float64)),
    ],
)
@pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
def test_scan_parquet_is_between_literal_dtype_mismatch_22622(
    engine: pl.GPUEngine, tmp_path, column_dtype, lit, closed
):
    if isinstance(column_dtype, pl.Datetime):
        rows = [
            dt.datetime(2021, 1, 1),
            dt.datetime(2021, 1, 2),
            dt.datetime(2021, 1, 2, 0, 0, 0, 1),
            dt.datetime(2021, 1, 3),
        ]
        col = pl.Series("A", rows, dtype=column_dtype)
    elif isinstance(column_dtype, pl.Duration):
        col = pl.Series(
            "A",
            [
                dt.timedelta(seconds=0),
                dt.timedelta(seconds=1),
                dt.timedelta(seconds=1, microseconds=1),
                dt.timedelta(seconds=2),
            ],
            dtype=column_dtype,
        )
    elif isinstance(column_dtype, pl.Decimal):
        col = pl.Series(
            "A",
            [Decimal("1.00"), Decimal("1.50"), Decimal("1.99"), Decimal("2.00")],
            dtype=column_dtype,
        )
    else:  # integer
        col = pl.Series("A", [0, 1, 2, 3], dtype=column_dtype)

    pl.DataFrame([col]).write_parquet(tmp_path / "f.parquet")

    q = pl.scan_parquet(tmp_path / "f.parquet").filter(
        pl.col("A").is_between(lit, lit, closed=closed)
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_142,
    reason="hive::HivePartitionedDf not exposed in the logical plan before 1.42",
)
def test_scan_parquet_hive_partitioned_raises(
    engine: pl.GPUEngine, tmp_path: Path
) -> None:
    (tmp_path / "part=1").mkdir()
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(tmp_path / "part=1" / "data.parquet")
    q = pl.scan_parquet(tmp_path, hive_schema={"part": pl.Int32})
    assert_ir_translation_raises(q, engine, NotImplementedError)
