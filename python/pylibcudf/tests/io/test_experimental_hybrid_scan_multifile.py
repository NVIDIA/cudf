# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io

import pyarrow as pa
import pytest
from utils import (
    extract_parquet_footer,
    synchronize_stream,
    write_hybrid_scan_parquet_bytes,
)

import rmm
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc
from pylibcudf.expressions import (
    ASTOperator,
    ColumnNameReference,
    Literal,
    Operation,
)
from pylibcudf.io.experimental import HybridScanMultifile


@pytest.fixture(scope="module")
def num_sources():
    """Number of parquet sources for multifile tests."""
    return 2


@pytest.fixture(scope="module")
def num_rows_per_source():
    """Number of rows per parquet source."""
    return 1000


@pytest.fixture(scope="module")
def num_row_groups_per_source(num_rows_per_source, row_group_size):
    """Number of row groups per parquet source."""
    return num_rows_per_source // row_group_size


@pytest.fixture(scope="module")
def multifile_parquet_tables(num_sources, num_rows_per_source):
    """Per-source PyArrow tables with disjoint value ranges in col0."""
    tables = []
    for src in range(num_sources):
        offset = src * num_rows_per_source
        data = {
            "col0": pa.array(
                list(range(offset, offset + num_rows_per_source)),
                type=pa.uint32(),
            ),
            "col1": [
                f"str_{i}" for i in range(offset, offset + num_rows_per_source)
            ],
            "col2": [
                float(i) * 1.5
                for i in range(offset, offset + num_rows_per_source)
            ],
        }
        tables.append(pa.table(data))
    return tables


@pytest.fixture(scope="module")
def multifile_parquet_bytes(multifile_parquet_tables, row_group_size):
    """Per-source parquet file bytes."""
    return [
        write_hybrid_scan_parquet_bytes(tbl, row_group_size)
        for tbl in multifile_parquet_tables
    ]


@pytest.fixture
def multifile_parquet_options(multifile_parquet_bytes):
    """ParquetReaderOptions referencing all sources.

    Note: This is function-scoped because tests may modify the options
    (e.g., by setting filters).
    """
    sources = [io.BytesIO(b) for b in multifile_parquet_bytes]
    source_info = plc.io.SourceInfo(sources)
    return plc.io.parquet.ParquetReaderOptions.builder(source_info).build()


@pytest.fixture
def multifile_hybrid_scan_reader(
    multifile_parquet_bytes, multifile_parquet_options
):
    """Create a HybridScanMultifile for the multifile parquet sources.

    Note: This is function-scoped because it depends on the function-scoped
    multifile_parquet_options fixture.
    """
    footer_mvs = [extract_parquet_footer(b) for b in multifile_parquet_bytes]
    return HybridScanMultifile(footer_mvs, multifile_parquet_options)


def test_hybrid_scan_multifile_basic(
    multifile_hybrid_scan_reader, num_sources, num_rows_per_source
):
    """Test basic HybridScanMultifile construction and metadata access."""
    metadatas = multifile_hybrid_scan_reader.parquet_metadatas()
    assert len(metadatas) == num_sources
    for metadata in metadatas:
        assert metadata.version == 2
        assert metadata.num_rows == num_rows_per_source
        assert "parquet-cpp-arrow" in metadata.created_by


def test_hybrid_scan_multifile_from_metadata(
    multifile_hybrid_scan_reader, multifile_parquet_options, num_sources
):
    """Test creating HybridScanMultifile from pre-populated metadata."""
    metadatas = multifile_hybrid_scan_reader.parquet_metadatas()
    assert all(
        isinstance(m, plc.io.parquet_metadata.FileMetaData) for m in metadatas
    )

    reader2 = HybridScanMultifile.from_parquet_metadata(
        metadatas, multifile_parquet_options
    )

    metadatas2 = reader2.parquet_metadatas()
    assert len(metadatas2) == num_sources
    assert [m.num_rows for m in metadatas] == [m.num_rows for m in metadatas2]


def test_hybrid_scan_multifile_all_row_groups(
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_sources,
    num_row_groups_per_source,
):
    """Test getting all row groups per source."""
    row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )

    assert len(row_groups) == num_sources
    for src_groups in row_groups:
        assert src_groups == list(range(num_row_groups_per_source))


def test_hybrid_scan_multifile_total_rows_in_row_groups(
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_sources,
    num_rows_per_source,
    row_group_size,
):
    """Test getting total rows across all sources."""
    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )

    total_rows = multifile_hybrid_scan_reader.total_rows_in_row_groups(
        all_row_groups
    )
    assert total_rows == num_sources * num_rows_per_source

    subset = [[0, 1]] * num_sources
    subset_rows = multifile_hybrid_scan_reader.total_rows_in_row_groups(subset)
    assert subset_rows == num_sources * row_group_size * 2


def test_hybrid_scan_multifile_reset_column_selection(
    multifile_hybrid_scan_reader,
):
    """Smoke test for reset_column_selection."""
    multifile_hybrid_scan_reader.reset_column_selection()


def test_hybrid_scan_multifile_filter_row_groups_with_byte_range(
    multifile_hybrid_scan_reader, multifile_parquet_options, num_sources
):
    """Test filtering row groups by byte range from the options."""
    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )

    filtered = multifile_hybrid_scan_reader.filter_row_groups_with_byte_range(
        all_row_groups, multifile_parquet_options
    )

    # Without a byte range restriction, all row groups should be retained
    assert len(filtered) == num_sources
    assert filtered == all_row_groups


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_multifile_filter_row_groups_with_stats(
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_sources,
    num_rows_per_source,
    num_row_groups_per_source,
    row_group_size,
    stream,
):
    """Test filtering row groups using statistics across sources."""
    # Filter: col0 >= half-way through source 0 (which is also < all of source 1)
    # This should filter out the first half of source 0's row groups, but keep all
    # of source 1's row groups (since source 1's col0 starts at num_rows_per_source).
    filter_threshold = (num_row_groups_per_source // 2) * row_group_size
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32()), stream=stream
            )
        ),
    )
    multifile_parquet_options.set_filter(filter_expression)

    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )
    filtered = multifile_hybrid_scan_reader.filter_row_groups_with_stats(
        all_row_groups, multifile_parquet_options, stream
    )

    assert len(filtered) == num_sources
    # Source 0: first half filtered out, second half retained
    assert filtered[0] == list(
        range(num_row_groups_per_source // 2, num_row_groups_per_source)
    )
    # Source 1: all retained (all values >= num_rows_per_source > threshold)
    assert filtered[1] == list(range(num_row_groups_per_source))


def test_hybrid_scan_multifile_bloom_and_dictionary_byte_ranges(
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_rows_per_source,
):
    """Test getting bloom filter and dictionary page byte ranges."""
    filter_threshold = num_rows_per_source // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    multifile_parquet_options.set_filter(filter_expression)

    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )

    bloom_ranges, bloom_sources = (
        multifile_hybrid_scan_reader.bloom_filters_byte_ranges(
            all_row_groups, multifile_parquet_options
        )
    )
    dict_ranges, dict_sources = (
        multifile_hybrid_scan_reader.dictionary_pages_byte_ranges(
            all_row_groups, multifile_parquet_options
        )
    )

    assert isinstance(bloom_ranges, list)
    assert isinstance(bloom_sources, list)
    assert len(bloom_ranges) == len(bloom_sources)
    assert isinstance(dict_ranges, list)
    assert isinstance(dict_sources, list)
    assert len(dict_ranges) == len(dict_sources)


def test_hybrid_scan_multifile_all_column_chunks_byte_ranges(
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_sources,
    num_row_groups_per_source,
    num_rows_per_source,
):
    """Test getting flattened column chunk byte ranges with source indices."""
    filter_threshold = num_rows_per_source // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    multifile_parquet_options.set_filter(filter_expression)

    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )

    ranges, source_indices = (
        multifile_hybrid_scan_reader.all_column_chunks_byte_ranges(
            all_row_groups, multifile_parquet_options
        )
    )

    # 3 columns x num_row_groups_per_source row groups x num_sources sources
    num_columns = 3
    expected_chunks = num_columns * num_row_groups_per_source * num_sources
    assert len(ranges) == expected_chunks
    assert len(source_indices) == expected_chunks

    for r in ranges:
        assert r.offset >= 0
        assert r.size > 0

    for src_idx in source_indices:
        assert 0 <= src_idx < num_sources


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_multifile_build_all_true_row_mask(
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_sources,
    num_rows_per_source,
    stream,
):
    """Test building an all-true row mask across all sources."""
    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )

    row_mask = multifile_hybrid_scan_reader.build_all_true_row_mask(
        all_row_groups, stream
    )

    assert row_mask.size() == num_sources * num_rows_per_source
    assert row_mask.type().id() == plc.types.TypeId.BOOL8


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_multifile_materialize_all_columns(
    multifile_parquet_bytes,
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_sources,
    num_rows_per_source,
    stream,
):
    """Test full workflow of single step materialization across sources."""
    # Filter: col0 >= num_rows_per_source // 10 of source 0
    filter_threshold = num_rows_per_source // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32()), stream=stream
            )
        ),
    )

    multifile_parquet_options.set_filter(filter_expression)

    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )
    filtered_row_groups = (
        multifile_hybrid_scan_reader.filter_row_groups_with_stats(
            all_row_groups, multifile_parquet_options, stream
        )
    )

    ranges, source_indices = (
        multifile_hybrid_scan_reader.all_column_chunks_byte_ranges(
            filtered_row_groups, multifile_parquet_options
        )
    )

    all_columns_data = [
        plc.gpumemoryview(
            rmm.DeviceBuffer.to_device(
                multifile_parquet_bytes[src_idx][r.offset : r.offset + r.size],
                plc.utils._get_stream(stream),
            )
        )
        for r, src_idx in zip(ranges, source_indices, strict=True)
    ]

    synchronize_stream(stream)

    result = multifile_hybrid_scan_reader.materialize_all_columns(
        filtered_row_groups,
        all_columns_data,
        multifile_parquet_options,
        stream,
    )

    synchronize_stream(stream)

    expected_kept_per_source = [
        num_rows_per_source - filter_threshold,  # source 0 partial
    ] + [num_rows_per_source] * (num_sources - 1)
    expected_total = sum(expected_kept_per_source)

    assert result.tbl.num_columns() == 3
    assert result.tbl.num_rows() == expected_total

    # Compare to regular read_parquet across all sources
    comparison_sources = [io.BytesIO(b) for b in multifile_parquet_bytes]
    comparison_options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(comparison_sources)
    ).build()
    comparison_options.set_filter(filter_expression)
    expected_result = plc.io.parquet.read_parquet(comparison_options, stream)

    synchronize_stream(stream)

    result_arrow = plc.Table(result.tbl.columns()).to_arrow()
    expected_arrow = expected_result.tbl.to_arrow()

    assert expected_arrow.equals(result_arrow)


def test_hybrid_scan_multifile_setup_page_indexes(
    multifile_parquet_bytes,
    multifile_hybrid_scan_reader,
    multifile_parquet_options,
    num_rows_per_source,
):
    """Test that page index setup enables page-level filtering.

    Mirrors the single-file test_hybrid_scan_metadata_with_page_index test.
    """
    filter_threshold = num_rows_per_source // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    multifile_parquet_options.set_filter(filter_expression)

    all_row_groups = multifile_hybrid_scan_reader.all_row_groups(
        multifile_parquet_options
    )
    assert all(len(src) > 0 for src in all_row_groups)

    # Before setup_page_indexes, page-index-dependent ops should error
    with pytest.raises(RuntimeError):
        multifile_hybrid_scan_reader.build_row_mask_with_page_index_stats(
            all_row_groups, multifile_parquet_options
        )

    # Get per-source page index byte ranges
    page_index_byte_ranges = (
        multifile_hybrid_scan_reader.page_index_byte_ranges()
    )
    assert len(page_index_byte_ranges) == len(multifile_parquet_bytes)
    for r in page_index_byte_ranges:
        assert r.size > 0

    # Fetch per-source page index bytes
    page_index_mvs = [
        memoryview(parquet_bytes)[r.offset : r.offset + r.size]
        for parquet_bytes, r in zip(
            multifile_parquet_bytes, page_index_byte_ranges, strict=True
        )
    ]

    multifile_hybrid_scan_reader.setup_page_indexes(page_index_mvs)

    # After setup, page-index-dependent ops should work
    row_mask = (
        multifile_hybrid_scan_reader.build_row_mask_with_page_index_stats(
            all_row_groups, multifile_parquet_options
        )
    )

    assert row_mask is not None
    assert row_mask.size() > 0
    assert row_mask.type().id() == plc.types.TypeId.BOOL8
