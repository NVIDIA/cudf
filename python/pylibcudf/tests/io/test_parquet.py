# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import os
import struct

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pyarrow.parquet import read_table, write_table
from utils import (
    assert_table_and_meta_eq,
    get_bytes_from_source,
    make_source,
    synchronize_stream,
)

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc
from pylibcudf.expressions import (
    ASTOperator,
    ColumnNameReference,
    ColumnReference,
    Literal,
    Operation,
)

# Shared kwargs to pass to make_source
_COMMON_PARQUET_SOURCE_KWARGS = {"format": "parquet"}


def _extract_footer_bytes_with_suffix(
    file_bytes: bytes,
) -> tuple[memoryview, memoryview]:
    """Return footer bytes with and without the parquet footer suffix."""
    parquet_suffix_size = 8  # 4-byte footer length + 4-byte magic bytes (PAR1)
    file_memoryview = memoryview(file_bytes)
    footer_size = int.from_bytes(
        file_memoryview[-parquet_suffix_size:-4], byteorder="little"
    )
    footer_start = len(file_memoryview) - parquet_suffix_size - footer_size
    footer_stop = len(file_memoryview)
    footer_with_suffix = file_memoryview[footer_start:footer_stop]
    footer_without_suffix = file_memoryview[
        footer_start : footer_stop - parquet_suffix_size
    ]
    return footer_without_suffix, footer_with_suffix


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("column_names", [None, ["col_int64", "col_bool"]])
@pytest.mark.parametrize("column_indices", [None, [2, 0]])
@pytest.mark.parametrize("source_strategy", ["inline", "set_source"])
def test_read_parquet_basic(
    table_data,
    binary_source_or_sink,
    nrows_skiprows,
    column_names,
    column_indices,
    stream,
    source_strategy,
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    source_info = plc.io.SourceInfo([source])
    options = plc.io.parquet.ParquetReaderOptions.builder(
        source_info if source_strategy == "inline" else plc.io.SourceInfo([])
    ).build()

    if source_strategy == "set_source":
        options.set_source(source_info)

    if nrows > -1:
        options.set_num_rows(nrows)
    if skiprows != 0:
        options.set_skip_rows(skiprows)
    if column_names is not None:
        options.set_column_names(column_names)
    elif column_indices is not None:
        options.set_column_indices(column_indices)

    res = plc.io.parquet.read_parquet(options, stream)

    if column_names is not None:
        pa_table = pa_table.select(column_names)
    elif column_indices is not None:
        column_names = [pa_table.column_names[idx] for idx in column_indices]
        pa_table = pa_table.select(column_names)

    # Adapt to nrows/skiprows
    pa_table = pa_table.slice(
        offset=skiprows, length=nrows if nrows > -1 else None
    )

    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)

    # No filtering done
    assert res.num_row_groups_after_stats_filter is None
    assert res.num_row_groups_after_bloom_filter is None


def test_read_parquet_column_field_ids(binary_source_or_sink):
    schema = pa.schema(
        [
            pa.field(
                "col_int64",
                pa.int64(),
                metadata={b"PARQUET:field_id": b"10"},
            ),
            pa.field(
                "col_string",
                pa.string(),
                metadata={b"PARQUET:field_id": b"20"},
            ),
            pa.field(
                "col_bool",
                pa.bool_(),
                metadata={b"PARQUET:field_id": b"30"},
            ),
        ]
    )
    pa_table = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(["a", "b", "c"], type=pa.string()),
            pa.array([True, False, True], type=pa.bool_()),
        ],
        schema=schema,
    )
    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )
    source_info = plc.io.SourceInfo([source])
    options = (
        plc.io.parquet.ParquetReaderOptions.builder(source_info)
        .column_field_ids([30, 10])
        .build()
    )

    res = plc.io.parquet.read_parquet(options)

    assert_table_and_meta_eq(
        pa_table.select(["col_bool", "col_int64"]),
        res,
        check_field_nullability=False,
    )


@pytest.mark.parametrize("if_prune_rowgroup,result", [(True, 0), (False, 1)])
def test_read_parquet_filters_metadata(tmp_path, if_prune_rowgroup, result):
    col_list = list(range(1, 10))
    min_element = min(col_list)
    max_element = max(col_list)
    tbl1 = pa.Table.from_pydict({"a": col_list})
    path1 = tmp_path / "tbl1.parquet"
    write_table(tbl1, path1)
    source = plc.io.SourceInfo([path1])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    if if_prune_rowgroup:
        # Prune the only row group since the filter aims to find elements larger than the max
        filter = Operation(
            ASTOperator.LESS,
            Literal(plc.Scalar.from_arrow(pa.scalar(max_element))),
            ColumnNameReference("a"),
        )
    else:
        # No real pruning
        filter = Operation(
            ASTOperator.LESS,
            Literal(plc.Scalar.from_arrow(pa.scalar(min_element))),
            ColumnNameReference("a"),
        )
    options.set_filter(filter)
    plc_table_w_meta = plc.io.parquet.read_parquet(options)
    assert (
        plc_table_w_meta.num_input_row_groups == 1
    )  # Input has only one rowgroup
    assert plc_table_w_meta.num_row_groups_after_stats_filter == result


@pytest.mark.parametrize(
    "pa_filters,plc_filters",
    [
        (
            pc.field("col_int64") >= 10,
            Operation(
                ASTOperator.GREATER_EQUAL,
                ColumnNameReference("col_int64"),
                Literal(plc.Scalar.from_arrow(pa.scalar(10))),
            ),
        ),
        (
            (pc.field("col_int64") >= 10) & (pc.field("col_double") < 0),
            Operation(
                ASTOperator.LOGICAL_AND,
                Operation(
                    ASTOperator.GREATER_EQUAL,
                    ColumnNameReference("col_int64"),
                    Literal(plc.Scalar.from_arrow(pa.scalar(10))),
                ),
                Operation(
                    ASTOperator.GREATER,
                    Literal(plc.Scalar.from_arrow(pa.scalar(0.0))),
                    ColumnNameReference("col_double"),
                ),
            ),
        ),
        (
            (pc.field(0) == 10),
            Operation(
                ASTOperator.EQUAL,
                ColumnReference(0),
                Literal(plc.Scalar.from_arrow(pa.scalar(10))),
            ),
        ),
    ],
)
def test_read_parquet_filters(
    table_data, binary_source_or_sink, pa_filters, plc_filters
):
    _, pa_table = table_data

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([source])
    ).build()
    options.set_filter(plc_filters)

    plc_table_w_meta = plc.io.parquet.read_parquet(options)
    exp = read_table(source, filters=pa_filters)
    assert_table_and_meta_eq(
        exp, plc_table_w_meta, check_field_nullability=False
    )


class FooSpan:
    def __init__(self, owner):
        # Keep the owning object alive
        self._data = owner

    @property
    def ptr(self):
        return self._data.ptr

    @property
    def size(self):
        return self._data.size


@pytest.mark.parametrize("num_buffers", [1, 2])
@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("column_names", [None, ["col_int64", "col_bool"]])
@pytest.mark.parametrize("column_indices", [None, [2, 0]])
@pytest.mark.parametrize("use_foo_span", [False, True])
def test_read_parquet_from_device_buffers(
    table_data,
    binary_source_or_sink,
    nrows_skiprows,
    stream,
    column_names,
    column_indices,
    num_buffers,
    use_foo_span,
):
    _, pa_table = table_data
    nrows, skiprows = nrows_skiprows

    # Load data from source
    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    # to_device() is documented as fully async on a non-default stream, and
    # the caller is responsible for keeping the source bytes alive until
    # synchronize_stream() below runs.
    # See https://github.com/rapidsai/rmm/issues/2521
    src_bytes = get_bytes_from_source(source)
    rmm_buf = DeviceBuffer.to_device(src_bytes, plc.utils._get_stream(stream))
    synchronize_stream(stream)
    buf = FooSpan(rmm_buf) if use_foo_span else rmm_buf

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([buf] * num_buffers)
    ).build()
    if nrows > -1:
        options.set_num_rows(nrows)
    if skiprows != 0:
        options.set_skip_rows(skiprows)
    if column_names is not None:
        options.set_column_names(column_names)
    elif column_indices is not None:
        options.set_column_indices(column_indices)

    res = plc.io.parquet.read_parquet(options, stream)

    expected = (
        pa_table
        if num_buffers == 1
        else pa.concat_tables([pa_table] * num_buffers)
    )
    if column_names is not None:
        expected = expected.select(column_names)
    elif column_indices is not None:
        column_names = [expected.column_names[idx] for idx in column_indices]
        expected = expected.select(column_names)

    expected = expected.slice(skiprows, nrows if nrows > -1 else None)

    assert_table_and_meta_eq(expected, res, check_field_nullability=False)


def test_read_parquet_with_pre_materialized_metadata(
    table_data: tuple[plc.io.types.TableWithMetadata, pa.Table],
    binary_source_or_sink: str | os.PathLike[str] | io.BytesIO,
) -> None:
    _, pa_table = table_data
    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )
    source_info = plc.io.SourceInfo([source])
    options = plc.io.parquet.ParquetReaderOptions.builder(source_info).build()

    parquet_metadatas = plc.io.parquet_metadata.read_parquet_footers(
        source_info
    )
    result = plc.io.parquet.read_parquet(
        options, parquet_metadatas=parquet_metadatas
    )

    assert_table_and_meta_eq(pa_table, result, check_field_nullability=False)


def test_read_parquet_with_pre_materialized_metadata_len_mismatch(
    table_data: tuple[plc.io.types.TableWithMetadata, pa.Table],
    binary_source_or_sink: str | os.PathLike[str] | io.BytesIO,
) -> None:
    _, pa_table = table_data
    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )
    source_info = plc.io.SourceInfo([source])
    options = plc.io.parquet.ParquetReaderOptions.builder(source_info).build()

    with pytest.raises(
        ValueError,
        match=r"Length of 'parquet_metadatas' \(0\) must match the number of input sources \(1\)",
    ):
        plc.io.parquet.read_parquet(options, parquet_metadatas=[])


def test_chunked_parquet_reader_with_pre_materialized_metadata(
    table_data: tuple[plc.io.types.TableWithMetadata, pa.Table],
    binary_source_or_sink: str | os.PathLike[str] | io.BytesIO,
) -> None:
    _, pa_table = table_data
    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )
    source_info = plc.io.SourceInfo([source])
    options = plc.io.parquet.ParquetReaderOptions.builder(source_info).build()
    parquet_metadatas = plc.io.parquet_metadata.read_parquet_footers(
        source_info
    )

    default_reader = plc.io.parquet.ChunkedParquetReader(
        options,
        chunk_read_limit=512,
    )
    default_chunks: list[pa.Table] = []
    while default_reader.has_next():
        default_chunks.append(default_reader.read_chunk().tbl.to_arrow())

    metadata_reader = plc.io.parquet.ChunkedParquetReader(
        options,
        chunk_read_limit=512,
        parquet_metadatas=parquet_metadatas,
    )
    metadata_chunks: list[pa.Table] = []
    while metadata_reader.has_next():
        metadata_chunks.append(metadata_reader.read_chunk().tbl.to_arrow())

    if default_chunks:
        expected = pa.concat_tables(default_chunks)
    else:
        expected = pa_table.slice(0, 0)
    if metadata_chunks:
        result = pa.concat_tables(metadata_chunks)
    else:
        result = pa_table.slice(0, 0)

    assert result.equals(expected)


def test_file_metadata_from_bytes(
    table_data: tuple[plc.io.types.TableWithMetadata, pa.Table],
    binary_source_or_sink: str | os.PathLike[str] | io.BytesIO,
) -> None:
    _, pa_table = table_data
    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )
    source_bytes = get_bytes_from_source(source)
    footer_without_suffix, footer_with_suffix = (
        _extract_footer_bytes_with_suffix(source_bytes)
    )

    metadata_from_footer_only = (
        plc.io.parquet_metadata.FileMetaData.from_bytes(footer_without_suffix)
    )
    metadata_from_footer_with_suffix = (
        plc.io.parquet_metadata.FileMetaData.from_bytes(footer_with_suffix)
    )
    assert (
        metadata_from_footer_only.version
        == metadata_from_footer_with_suffix.version
    )
    assert (
        metadata_from_footer_only.num_rows
        == metadata_from_footer_with_suffix.num_rows
    )
    assert (
        metadata_from_footer_only.created_by
        == metadata_from_footer_with_suffix.created_by
    )
    assert metadata_from_footer_only.num_rows == pa_table.num_rows


def test_file_metadata_from_bytes_empty() -> None:
    with pytest.raises(RuntimeError, match="Cannot initialize schema"):
        plc.io.parquet_metadata.FileMetaData.from_bytes(memoryview(b""))


def test_file_metadata_row_groups_and_column_chunks() -> None:
    table = pa.table(
        {
            "a": list(range(100)),
            "b": [x * 10 for x in range(100)],
        }
    )
    sink = io.BytesIO()
    write_table(table, sink, row_group_size=25)
    sink.seek(0)
    parquet_file = pq.ParquetFile(sink)
    sink.seek(0)

    source_info = plc.io.SourceInfo([sink])
    file_metadata = plc.io.parquet_metadata.read_parquet_footers(source_info)[
        0
    ]

    assert (
        len(file_metadata.row_groups) == parquet_file.metadata.num_row_groups
    )

    for rg_idx, row_group in enumerate(file_metadata.row_groups):
        pa_row_group = parquet_file.metadata.row_group(rg_idx)
        assert row_group.num_rows == pa_row_group.num_rows
        assert row_group.total_byte_size == pa_row_group.total_byte_size
        assert row_group.total_compressed_size is None or (
            row_group.total_compressed_size >= 0
        )
        assert row_group.file_offset is None or row_group.file_offset >= 0
        assert row_group.ordinal is None or row_group.ordinal == rg_idx

        assert len(row_group.columns) == pa_row_group.num_columns
        for col_idx, column_chunk in enumerate(row_group.columns):
            pa_col_chunk = pa_row_group.column(col_idx)
            meta_data = column_chunk.meta_data
            assert column_chunk.file_path == ""
            assert column_chunk.file_offset == 0
            assert isinstance(column_chunk.offset_index_offset, int)
            assert isinstance(column_chunk.offset_index_length, int)
            assert isinstance(column_chunk.column_index_offset, int)
            assert isinstance(column_chunk.column_index_length, int)
            assert isinstance(column_chunk.schema_idx, int)
            assert meta_data.num_values == pa_col_chunk.num_values
            assert (
                meta_data.total_uncompressed_size
                == pa_col_chunk.total_uncompressed_size
            )
            assert (
                meta_data.total_compressed_size
                == pa_col_chunk.total_compressed_size
            )
            assert meta_data.path_in_schema[-1] == pa_col_chunk.path_in_schema


def test_file_metadata_columnchunk_statistics() -> None:
    table = pa.table(
        {
            "a": pa.array([1, 2, 3, None], type=pa.int64()),
            "s": pa.array(["aa", "bb", "cc", None]),
        }
    )
    sink = io.BytesIO()
    write_table(table, sink, row_group_size=2)
    sink.seek(0)

    source_info = plc.io.SourceInfo([sink])
    file_metadata = plc.io.parquet_metadata.read_parquet_footers(source_info)[
        0
    ]

    expected = [
        (1, 2, 0, b"aa", b"bb"),
        (3, 3, 1, b"cc", b"cc"),
    ]
    for row_group, (
        min_a,
        max_a,
        null_count,
        min_s,
        max_s,
    ) in zip(file_metadata.row_groups, expected, strict=True):
        stats_by_name = {
            column.meta_data.path_in_schema[-1]: column.meta_data.statistics
            for column in row_group.columns
        }

        a_stats = stats_by_name["a"]
        assert a_stats.has_min_max
        assert a_stats.min_encoded == struct.pack("<q", min_a)
        assert a_stats.max_encoded == struct.pack("<q", max_a)
        assert a_stats.null_count == null_count
        assert a_stats.distinct_count is None

        s_stats = stats_by_name["s"]
        assert s_stats.has_min_max
        assert s_stats.min_encoded == min_s
        assert s_stats.max_encoded == max_s
        assert s_stats.null_count == null_count
        assert s_stats.distinct_count is None


def test_file_metadata_columnchunk_statistics_without_minmax() -> None:
    table = pa.table({"a": pa.array([1, 2], type=pa.int64())})
    sink = io.BytesIO()
    write_table(table, sink, write_statistics=False)
    sink.seek(0)

    source_info = plc.io.SourceInfo([sink])
    file_metadata = plc.io.parquet_metadata.read_parquet_footers(source_info)[
        0
    ]
    statistics = file_metadata.row_groups[0].columns[0].meta_data.statistics

    assert not statistics.has_min_max
    assert statistics.min_encoded is None
    assert statistics.max_encoded is None
    assert statistics.is_min_value_exact is None
    assert statistics.is_max_value_exact is None


def test_read_parquet_column_chunk_bounds(tmp_path) -> None:
    table_0 = pa.table(
        {
            "a": pa.array([1, 2, 3, 4], type=pa.int64()),
            "s": ["b", "a", "d", "c"],
            "ts": pa.array([0, 1, 2, 3], type=pa.timestamp("us")),
            "n": pa.array([None, 2, None, None], type=pa.int64()),
        }
    )
    table_1 = pa.table(
        {
            "a": pa.array([10, 20, 30, 40], type=pa.int64()),
            "s": ["z", "y", "x", "w"],
            "ts": pa.array([10, 20, 30, 40], type=pa.timestamp("us")),
            "n": pa.array([5, None, None, 8], type=pa.int64()),
        }
    )
    path_0 = tmp_path / "part-0.parquet"
    path_1 = tmp_path / "part-1.parquet"
    write_table(table_0, path_0, row_group_size=2)
    write_table(table_1, path_1, row_group_size=2)

    file_metadatas = plc.io.parquet_metadata.read_parquet_footers(
        plc.io.SourceInfo([path_0, path_1])
    )

    bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
        file_metadatas,
        columns=["a", "s", "ts", "n"],
    )
    columns = bounds.columns()

    assert len(columns) == 10
    assert columns[0].to_pylist() == [0, 0, 1, 1]
    assert columns[1].to_pylist() == [0, 1, 0, 1]

    a_min, a_max = columns[2], columns[3]
    assert a_min.to_pylist() == [1, 3, 10, 30]
    assert a_max.to_pylist() == [2, 4, 20, 40]

    s_min, s_max = columns[4], columns[5]
    assert s_min.to_pylist() == ["a", "c", "y", "w"]
    assert s_max.to_pylist() == ["b", "d", "z", "x"]

    ts_min, ts_max = columns[6], columns[7]
    assert ts_min.to_arrow().equals(
        pa.array([0, 2, 10, 30], type=pa.timestamp("us"))
    )
    assert ts_max.to_arrow().equals(
        pa.array([1, 3, 20, 40], type=pa.timestamp("us"))
    )

    n_min, n_max = columns[8], columns[9]
    assert n_min.to_pylist() == [2, None, 5, 8]
    assert n_max.to_pylist() == [2, None, 5, 8]


def test_read_parquet_column_chunk_bounds_without_minmax(tmp_path) -> None:
    table = pa.table({"a": pa.array([1, 2], type=pa.int64())})
    path = tmp_path / "no-stats.parquet"
    write_table(table, path, write_statistics=False)

    file_metadatas = plc.io.parquet_metadata.read_parquet_footers(
        plc.io.SourceInfo([path])
    )

    bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
        file_metadatas,
        columns=["a"],
    )
    columns = bounds.columns()

    assert len(columns) == 4
    assert columns[0].to_pylist() == [0]
    assert columns[1].to_pylist() == [0]
    min_col, max_col = columns[2], columns[3]
    assert min_col.to_pylist() == [None]
    assert max_col.to_pylist() == [None]


def test_read_parquet_column_chunk_bounds_invalid_inputs(tmp_path) -> None:
    table = pa.table({"a": pa.array([1, 2], type=pa.int64())})
    path = tmp_path / "input.parquet"
    write_table(table, path)

    file_metadatas = plc.io.parquet_metadata.read_parquet_footers(
        plc.io.SourceInfo([path])
    )

    with pytest.raises(
        ValueError, match="Parquet leaf column path not found: missing"
    ):
        plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            file_metadatas,
            columns=["missing"],
        )

    with pytest.raises(TypeError, match="columns must contain only strings"):
        plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            file_metadatas,
            columns=[1],
        )

    with pytest.raises(
        TypeError, match="columns must be a sequence of strings"
    ):
        plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            file_metadatas,
            columns="a",
        )

    with pytest.raises(
        TypeError,
        match="file_metadatas must contain only FileMetaData objects",
    ):
        plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            [object()],
            columns=["a"],
        )

    with pytest.raises(
        ValueError,
        match="without source metadata",
    ):
        plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            [],
            columns=["a"],
        )


def test_file_metadata_wrappers_not_directly_constructible() -> None:
    with pytest.raises(
        ValueError, match="SortingColumn cannot be constructed directly"
    ):
        plc.io.parquet_metadata.SortingColumn()
    with pytest.raises(
        ValueError, match="ColumnChunk cannot be constructed directly"
    ):
        plc.io.parquet_metadata.ColumnChunk()
    with pytest.raises(
        ValueError, match="ColumnChunkMetaData cannot be constructed directly"
    ):
        plc.io.parquet_metadata.ColumnChunkMetaData()
    with pytest.raises(
        ValueError,
        match="ColumnChunkStatistics cannot be constructed directly",
    ):
        plc.io.parquet_metadata.ColumnChunkStatistics()
    with pytest.raises(
        ValueError, match="RowGroup cannot be constructed directly"
    ):
        plc.io.parquet_metadata.RowGroup()


def test_file_metadata_row_group_sorting_columns(tmp_path) -> None:
    table = pa.table({"a": list(range(50)), "b": [x * 10 for x in range(50)]})
    sorting_columns = pq.SortingColumn.from_ordering(
        table.schema, [("a", "ascending")]
    )

    parquet_path = tmp_path / "sorted.parquet"
    write_table(
        table, parquet_path, row_group_size=25, sorting_columns=sorting_columns
    )

    parquet_file = pq.ParquetFile(parquet_path)
    source_info = plc.io.SourceInfo([parquet_path])
    file_metadata = plc.io.parquet_metadata.read_parquet_footers(source_info)[
        0
    ]

    for rg_idx, row_group in enumerate(file_metadata.row_groups):
        pa_sorting_columns = parquet_file.metadata.row_group(
            rg_idx
        ).sorting_columns
        assert pa_sorting_columns is not None
        assert row_group.sorting_columns is not None
        assert len(row_group.sorting_columns) == len(pa_sorting_columns)

        for sorting_column, pa_sorting_column in zip(
            row_group.sorting_columns, pa_sorting_columns, strict=True
        ):
            assert sorting_column.column_idx == pa_sorting_column.column_index
            assert sorting_column.descending == pa_sorting_column.descending
            assert sorting_column.nulls_first == pa_sorting_column.nulls_first


def test_file_metadata_columnchunk_metadata() -> None:
    table = pa.table(
        {
            "a": list(range(100)),
            "b": [x * 10 for x in range(100)],
        }
    )
    sink = io.BytesIO()
    write_table(table, sink, row_group_size=25)
    sink.seek(0)
    parquet_file = pq.ParquetFile(sink)
    sink.seek(0)

    source_info = plc.io.SourceInfo([sink])
    file_metadata = plc.io.parquet_metadata.read_parquet_footers(source_info)[
        0
    ]

    result = file_metadata.columnchunk_metadata
    assert set(result) == {"a", "b"}

    expected: dict[str, list[int]] = {"a": [], "b": []}
    for rg_idx in range(parquet_file.metadata.num_row_groups):
        pa_row_group = parquet_file.metadata.row_group(rg_idx)
        for col_idx in range(pa_row_group.num_columns):
            pa_col = pa_row_group.column(col_idx)
            expected[pa_col.path_in_schema].append(
                pa_col.total_uncompressed_size
            )

    for name, sizes in expected.items():
        assert result[name] == sizes


# TODO: Test these options
# list row_groups = None,
# ^^^ This one is not tested since it's not in pyarrow/pandas, deprecate?
# bool convert_strings_to_categories = False,
# bool use_pandas_metadata = True


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize("write_v2_headers", [True, False])
@pytest.mark.parametrize("utc_timestamps", [True, False])
@pytest.mark.parametrize("write_arrow_schema", [True, False])
@pytest.mark.parametrize(
    "partitions",
    [None, [plc.io.types.PartitionInfo(0, 10)]],
)
@pytest.mark.parametrize("column_chunks_file_paths", [None, ["tmp.parquet"]])
@pytest.mark.parametrize("row_group_size_bytes", [None, 1024])
@pytest.mark.parametrize("row_group_size_rows", [None, 1])
@pytest.mark.parametrize("max_page_size_bytes", [None, 1024])
@pytest.mark.parametrize("max_page_size_rows", [None, 1])
@pytest.mark.parametrize("max_dictionary_size", [None, 100])
def test_write_parquet(
    table_data,
    write_v2_headers,
    utc_timestamps,
    write_arrow_schema,
    partitions,
    column_chunks_file_paths,
    row_group_size_bytes,
    row_group_size_rows,
    max_page_size_bytes,
    max_page_size_rows,
    max_dictionary_size,
    stream,
):
    _, pa_table = table_data
    if len(pa_table) == 0 and partitions is not None:
        pytest.skip("https://github.com/NVIDIA/cudf/issues/17361")
    plc_table = plc.Table.from_arrow(pa_table)
    table_meta = plc.io.types.TableInputMetadata(plc_table)
    sink = plc.io.SinkInfo([io.BytesIO()])
    user_data = [{"foo": "{'bar': 'baz'}"}]
    compression = plc.io.types.CompressionType.SNAPPY
    stats_level = plc.io.types.StatisticsFreq.STATISTICS_COLUMN
    dictionary_policy = plc.io.types.DictionaryPolicy.ADAPTIVE
    options = (
        plc.io.parquet.ParquetWriterOptions.builder(sink, plc_table)
        .metadata(table_meta)
        .key_value_metadata(user_data)
        .compression(compression)
        .stats_level(stats_level)
        .write_v2_headers(write_v2_headers)
        .dictionary_policy(dictionary_policy)
        .utc_timestamps(utc_timestamps)
        .write_arrow_schema(write_arrow_schema)
        .build()
    )
    if partitions is not None:
        options.set_partitions(partitions)
    if column_chunks_file_paths is not None:
        options.set_column_chunks_file_paths(column_chunks_file_paths)
    if row_group_size_bytes is not None:
        options.set_row_group_size_bytes(row_group_size_bytes)
    if row_group_size_rows is not None:
        options.set_row_group_size_rows(row_group_size_rows)
    if max_page_size_bytes is not None:
        options.set_max_page_size_bytes(max_page_size_bytes)
    if max_page_size_rows is not None:
        options.set_max_page_size_rows(max_page_size_rows)
    if max_dictionary_size is not None:
        options.set_max_dictionary_size(max_dictionary_size)

    result = plc.io.parquet.write_parquet(options, stream)

    synchronize_stream(stream)

    assert isinstance(result, memoryview)


# cupy allocates on its own stream, so this cannot honor the injected default
# stream used by the stream-validation test pass.
@pytest.mark.uses_custom_stream
def test_write_large_list_row_group():
    import cupy as cp

    # 524.3k list<float32>[1024] rows exceed 2 GiB plain data. The writer must retain the correct
    # plain-data size for dictionary selection.
    rows = 524_300
    embedding_dim = 1024
    values = cp.zeros((rows, embedding_dim), dtype=cp.float32)
    table = plc.Table(
        [plc.Column.from_cuda_array_interface(values)],
        num_rows=rows,
    )
    metadata = plc.io.types.TableInputMetadata(table)
    sink = plc.io.SinkInfo([io.BytesIO()])
    options = (
        plc.io.parquet.ParquetWriterOptions.builder(sink, table)
        .metadata(metadata)
        .build()
    )

    result = plc.io.parquet.write_parquet(options)
    parquet_file = pq.ParquetFile(io.BytesIO(result))

    assert parquet_file.metadata.num_rows == rows
    assert parquet_file.metadata.num_row_groups == 1


@pytest.mark.parametrize("use_jit_filter", [False, True])
@pytest.mark.parametrize(
    "pa_filter,plc_filter",
    [
        (
            pc.field("col_int64") >= 10,
            Operation(
                ASTOperator.GREATER_EQUAL,
                ColumnNameReference("col_int64"),
                Literal(plc.Scalar.from_arrow(pa.scalar(10, type=pa.int64()))),
            ),
        ),
        (
            pc.field("col_str") == "foo",
            Operation(
                ASTOperator.EQUAL,
                Literal(
                    plc.Scalar.from_arrow(pa.scalar("foo", type=pa.string()))
                ),
                ColumnNameReference("col_str"),
            ),
        ),
    ],
)
def test_read_parquet_filters_jit(
    binary_source_or_sink,
    pa_filter,
    plc_filter,
    use_jit_filter,
):
    pa_table = pa.table(
        {
            "col_int64": pa.array([6, 0, 2, 2], type=pa.int64()),
            "col_str": pa.array(
                ["bar", "foo", "baz", "foo"], type=pa.string()
            ),
        }
    )

    source = make_source(
        binary_source_or_sink, pa_table, **_COMMON_PARQUET_SOURCE_KWARGS
    )

    options = (
        plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo([source])
        )
        .use_jit_filter(use_jit_filter)
        .build()
    )
    options.set_filter(plc_filter)

    assert options.is_enabled_use_jit_filter() is use_jit_filter

    got = plc.io.parquet.read_parquet(options)
    expect = read_table(source, filters=pa_filter)

    assert_table_and_meta_eq(
        expect,
        got,
        check_field_nullability=False,
    )


@pytest.fixture
def source_index_sources(tmp_path) -> list[os.PathLike[str]]:
    """Two parquet sources with differing row counts."""
    path_0 = tmp_path / "part-0.parquet"
    path_1 = tmp_path / "part-1.parquet"
    write_table(pa.table({"a": pa.array([1, 2, 3], type=pa.int64())}), path_0)
    write_table(pa.table({"a": pa.array([4, 5], type=pa.int64())}), path_1)
    return [path_0, path_1]


def test_read_parquet_prepend_source_index_column_default(
    source_index_sources,
) -> None:
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(source_index_sources)
    ).build()

    assert options.is_enabled_prepend_source_index_column() is False

    expect = pa.table({"a": pa.array([1, 2, 3, 4, 5], type=pa.int64())})
    assert_table_and_meta_eq(
        expect,
        plc.io.parquet.read_parquet(options),
        check_field_nullability=False,
    )


@pytest.mark.parametrize("use_builder", [False, True])
def test_read_parquet_prepend_source_index_column(
    source_index_sources, use_builder
) -> None:
    builder = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(source_index_sources)
    )
    if use_builder:
        options = builder.prepend_source_index_column(True).build()
    else:
        options = builder.build()
        options.enable_prepend_source_index_column(True)

    assert options.is_enabled_prepend_source_index_column() is True

    expect = pa.table(
        {
            "source_index": pa.array([0, 0, 0, 1, 1], type=pa.int32()),
            "a": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
        }
    )
    assert_table_and_meta_eq(
        expect,
        plc.io.parquet.read_parquet(options),
        check_field_nullability=False,
    )


def test_read_parquet_prepend_source_index_column_with_filter(
    source_index_sources,
) -> None:
    options = (
        plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(source_index_sources)
        )
        .prepend_source_index_column(True)
        .build()
    )
    plc_filter = Operation(
        ASTOperator.GREATER,
        ColumnNameReference("a"),
        Literal(plc.Scalar.from_arrow(pa.scalar(2, type=pa.int64()))),
    )
    options.set_filter(plc_filter)

    expect = pa.table(
        {
            "source_index": pa.array([0, 1, 1], type=pa.int32()),
            "a": pa.array([3, 4, 5], type=pa.int64()),
        }
    )
    assert_table_and_meta_eq(
        expect,
        plc.io.parquet.read_parquet(options),
        check_field_nullability=False,
    )
