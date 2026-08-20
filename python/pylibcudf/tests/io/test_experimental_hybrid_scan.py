# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from utils import synchronize_stream

import rmm
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc
from pylibcudf.expressions import (
    ASTOperator,
    ColumnNameReference,
    Literal,
    Operation,
)
from pylibcudf.io.experimental import (
    HybridScanReader,
    UseDataPageMask,
)


@pytest.fixture(scope="module")
def num_rows() -> int:
    """Number of rows in the test table."""
    return 1000


@pytest.fixture(scope="module")
def row_group_size() -> int:
    """Row group size for parquet files."""
    return 250


@pytest.fixture(scope="module")
def num_row_groups(num_rows: int, row_group_size: int) -> int:
    """Number of row groups in the test parquet file."""
    return num_rows // row_group_size


@pytest.fixture(scope="module")
def simple_parquet_table(num_rows: int) -> pa.Table:
    """Create a simple PyArrow table for testing."""
    data = {
        "col0": pa.array(list(range(num_rows)), type=pa.uint32()),
        "col1": [f"str_{i}" for i in range(num_rows)],
        "col2": [float(i) * 1.5 for i in range(num_rows)],
    }
    return pa.table(data)


@pytest.fixture(scope="module")
def simple_parquet_bytes(
    simple_parquet_table: pa.Table, row_group_size: int
) -> bytes:
    """Create parquet bytes from the simple table."""
    buf = io.BytesIO()
    pq.write_table(
        simple_parquet_table,
        buf,
        row_group_size=row_group_size,
        use_dictionary=True,
        write_statistics=True,
        write_page_index=True,
    )
    return buf.getvalue()


@pytest.fixture
def simple_parquet_options(
    simple_parquet_bytes: bytes,
) -> plc.io.parquet.ParquetReaderOptions:
    """Create basic ParquetReaderOptions for the simple parquet file.

    Note: This is function-scoped (not module-scoped) because tests may
    modify the options (e.g., by setting filters), so each test needs
    its own independent copy.
    """
    # SourceInfo doesn't accept BytesIO, but that's fine for this test.
    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])  # type: ignore[arg-type]
    return plc.io.parquet.ParquetReaderOptions.builder(source).build()


@pytest.fixture
def simple_hybrid_scan_reader(
    simple_parquet_bytes: bytes,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
) -> HybridScanReader:
    """Create a HybridScanReader for the simple parquet file.

    Note: This is function-scoped (not module-scoped) because it depends on
    the function-scoped simple_parquet_options fixture.
    """
    # Extract footer bytes from the parquet file
    # According to Parquet file format specification:
    # https://parquet.apache.org/docs/file-format/
    PARQUET_FOOTER_SIZE_BYTES = 4  # Number of bytes encoding footer length
    PARQUET_MAGIC_BYTES = 4  # Number of bytes for "PAR1" magic number
    PARQUET_SUFFIX_BYTES = PARQUET_FOOTER_SIZE_BYTES + PARQUET_MAGIC_BYTES

    simple_parquet_mv = memoryview(simple_parquet_bytes)

    footer_size = int.from_bytes(
        simple_parquet_mv[-PARQUET_SUFFIX_BYTES:-PARQUET_MAGIC_BYTES],
        byteorder="little",
    )
    footer_start = len(simple_parquet_mv) - PARQUET_SUFFIX_BYTES - footer_size
    footer_end = len(simple_parquet_mv) - PARQUET_SUFFIX_BYTES
    footer_mv = simple_parquet_mv[footer_start:footer_end]

    return HybridScanReader(footer_mv, simple_parquet_options)


def test_hybrid_scan_reader_basic(
    simple_hybrid_scan_reader: HybridScanReader, num_rows: int
) -> None:
    """Test basic HybridScanReader construction and metadata access."""
    metadata = simple_hybrid_scan_reader.parquet_metadata()
    assert metadata.version == 2
    assert metadata.num_rows == num_rows
    assert "parquet-cpp-arrow" in metadata.created_by


def test_hybrid_scan_reader_from_metadata(
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
) -> None:
    """Test creating HybridScanReader from pre-populated metadata."""
    # Get metadata from the fixture reader
    metadata = simple_hybrid_scan_reader.parquet_metadata()
    assert isinstance(metadata, plc.io.parquet_metadata.FileMetaData)

    # Create a second reader from the metadata
    reader2 = HybridScanReader.from_parquet_metadata(
        metadata, simple_parquet_options
    )

    # Verify both readers work the same
    assert (
        simple_hybrid_scan_reader.parquet_metadata().num_rows
        == reader2.parquet_metadata().num_rows
    )


def test_hybrid_scan_all_row_groups(
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_row_groups: int,
) -> None:
    """Test getting all row groups."""
    row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )

    assert len(row_groups) == num_row_groups
    assert row_groups == list(range(num_row_groups))


def test_hybrid_scan_total_rows_in_row_groups(
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
    row_group_size: int,
) -> None:
    """Test getting total rows in specific row groups."""
    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )

    # Test all row groups
    total_rows = simple_hybrid_scan_reader.total_rows_in_row_groups(
        all_row_groups
    )
    assert total_rows == num_rows

    # Test subset of row groups
    subset_rows = simple_hybrid_scan_reader.total_rows_in_row_groups([0, 1])
    assert subset_rows == row_group_size * 2


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_filter_row_groups_with_stats(
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_row_groups: int,
    row_group_size: int,
    stream: Stream | None,
) -> None:
    """Test filtering row groups using statistics."""
    # Filter: col0 >= (num_row_groups // 2) * row_group_size
    # This should filter out the first half of row groups
    filter_threshold = (num_row_groups // 2) * row_group_size
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32()), stream=stream
            )
        ),
    )
    simple_parquet_options.set_filter(filter_expression)

    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )
    filtered = simple_hybrid_scan_reader.filter_row_groups_with_stats(
        all_row_groups, simple_parquet_options, stream
    )

    # First half of row groups should be filtered out, second half should remain
    expected_row_groups = list(range(num_row_groups // 2, num_row_groups))
    assert filtered == expected_row_groups


def test_hybrid_scan_secondary_filters_byte_ranges(
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
) -> None:
    """Test getting bloom filter and dictionary page byte ranges."""
    # Need to set a filter for secondary filters to work
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    simple_parquet_options.set_filter(filter_expression)

    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )

    bloom_ranges, dict_ranges = (
        simple_hybrid_scan_reader.secondary_filters_byte_ranges(
            all_row_groups, simple_parquet_options
        )
    )

    # These should be lists of ByteRangeInfo
    assert isinstance(bloom_ranges, list)
    assert isinstance(dict_ranges, list)


def test_hybrid_scan_column_chunk_byte_ranges(
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
    num_row_groups: int,
) -> None:
    """Test getting filter and payload column chunk byte ranges."""
    # Set filter to make col0 the filter column
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    simple_parquet_options.set_filter(filter_expression)

    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )

    # Get filter column ranges (one filter column per row group)
    filter_ranges = simple_hybrid_scan_reader.filter_column_chunks_byte_ranges(
        all_row_groups, simple_parquet_options
    )
    assert len(filter_ranges) == num_row_groups

    # Get payload column ranges (two payload columns per row group)
    num_payload_columns = 2
    payload_ranges = (
        simple_hybrid_scan_reader.payload_column_chunks_byte_ranges(
            all_row_groups, simple_parquet_options
        )
    )
    assert len(payload_ranges) == num_payload_columns * num_row_groups

    # Verify all have valid offsets and sizes
    for r in filter_ranges + payload_ranges:
        assert r.offset >= 0
        assert r.size > 0


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize(
    "use_data_page_mask", [UseDataPageMask.NO, UseDataPageMask.YES]
)
def test_hybrid_scan_materialize_columns(
    simple_parquet_bytes: bytes,
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
    stream: Stream | None,
    use_data_page_mask: UseDataPageMask,
) -> None:
    """Test full workflow of materializing filter and payload columns."""
    # Create filter: col0 >= num_rows // 10 (filter out first 10%)
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32()), stream=stream
            )
        ),
    )

    simple_parquet_options.set_filter(filter_expression)

    # Get filtered row groups
    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )
    filtered_row_groups = (
        simple_hybrid_scan_reader.filter_row_groups_with_stats(
            all_row_groups, simple_parquet_options, stream
        )
    )

    # Create initial row mask
    total_rows = simple_hybrid_scan_reader.total_rows_in_row_groups(
        filtered_row_groups
    )
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    # Get filter column data
    filter_ranges = simple_hybrid_scan_reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, simple_parquet_options
    )
    filter_data = [
        plc.gpumemoryview(
            rmm.DeviceBuffer.to_device(
                memoryview(simple_parquet_bytes)[r.offset : r.offset + r.size],
                plc.utils._get_stream(stream),
            )
        )
        for r in filter_ranges
    ]

    synchronize_stream(stream)

    # Materialize filter columns (mr is optional, defaults to None)
    filter_result = simple_hybrid_scan_reader.materialize_filter_columns(
        filtered_row_groups,
        filter_data,
        row_mask,
        use_data_page_mask,
        simple_parquet_options,
        stream,
    )

    synchronize_stream(stream)

    # Filter column should have 1 column, with rows passing the filter
    expected_result_rows = num_rows - filter_threshold
    assert filter_result.tbl.num_columns() == 1
    assert filter_result.tbl.num_rows() == expected_result_rows

    # Get payload column data
    payload_ranges = (
        simple_hybrid_scan_reader.payload_column_chunks_byte_ranges(
            filtered_row_groups, simple_parquet_options
        )
    )
    payload_data = [
        plc.gpumemoryview(
            rmm.DeviceBuffer.to_device(
                memoryview(simple_parquet_bytes)[r.offset : r.offset + r.size],
                plc.utils._get_stream(stream),
            )
        )
        for r in payload_ranges
    ]

    synchronize_stream(stream)

    # Materialize payload columns (mr is optional, defaults to None)
    payload_result = simple_hybrid_scan_reader.materialize_payload_columns(
        filtered_row_groups,
        payload_data,
        row_mask,
        use_data_page_mask,
        simple_parquet_options,
        stream,
    )

    synchronize_stream(stream)

    assert payload_result.tbl.num_columns() == 2
    assert payload_result.tbl.num_rows() == expected_result_rows

    # Verify results match regular parquet reader
    # Create a fresh BytesIO to avoid buffer position issues
    comparison_buffer = io.BytesIO(simple_parquet_bytes)
    comparison_source = plc.io.SourceInfo([comparison_buffer])
    comparison_options = plc.io.parquet.ParquetReaderOptions.builder(
        comparison_source
    ).build()
    comparison_options.set_filter(filter_expression)
    expected_result = plc.io.parquet.read_parquet(comparison_options, stream)

    synchronize_stream(stream)

    # Combine hybrid scan results
    hybrid_columns = filter_result.tbl.columns() + payload_result.tbl.columns()
    hybrid_table = plc.Table(hybrid_columns)

    # Compare just the table data (metadata structures are complex)
    expected_arrow = expected_result.tbl.to_arrow()
    hybrid_arrow = hybrid_table.to_arrow()

    assert expected_arrow.equals(hybrid_arrow)


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_single_step_materialize(
    simple_parquet_bytes: bytes,
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
    stream: Stream | None,
) -> None:
    """Test full workflow of single step materialization."""
    # Create filter: col0 >= num_rows // 10 (filter out first 10%)
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32()), stream=stream
            )
        ),
    )

    simple_parquet_options.set_filter(filter_expression)

    # Get filtered row groups
    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )

    filtered_row_groups = (
        simple_hybrid_scan_reader.filter_row_groups_with_stats(
            all_row_groups, simple_parquet_options, stream
        )
    )

    # Get filter column data
    all_columns_ranges = (
        simple_hybrid_scan_reader.all_column_chunks_byte_ranges(
            filtered_row_groups, simple_parquet_options
        )
    )

    all_columns_data = [
        plc.gpumemoryview(
            rmm.DeviceBuffer.to_device(
                memoryview(simple_parquet_bytes)[r.offset : r.offset + r.size],
                plc.utils._get_stream(stream),
            )
        )
        for r in all_columns_ranges
    ]

    synchronize_stream(stream)

    # Materialize filter columns (mr is optional, defaults to None)
    single_step_result = simple_hybrid_scan_reader.materialize_all_columns(
        filtered_row_groups,
        all_columns_data,
        simple_parquet_options,
        stream,
    )

    synchronize_stream(stream)

    assert single_step_result.tbl.num_columns() == 3
    assert single_step_result.tbl.num_rows() == num_rows - filter_threshold

    # Verify results match regular parquet reader
    # Create a fresh BytesIO to avoid buffer position issues
    comparison_buffer = io.BytesIO(simple_parquet_bytes)
    comparison_source = plc.io.SourceInfo([comparison_buffer])
    comparison_options = plc.io.parquet.ParquetReaderOptions.builder(
        comparison_source
    ).build()
    comparison_options.set_filter(filter_expression)
    expected_result = plc.io.parquet.read_parquet(comparison_options, stream)

    synchronize_stream(stream)

    result = plc.Table(single_step_result.tbl.columns()).to_arrow()
    expected = expected_result.tbl.to_arrow()

    assert expected.equals(result)


def test_hybrid_scan_has_next_table_chunk(
    simple_parquet_bytes: bytes,
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
) -> None:
    """Test has_next_table_chunk method - requires chunking to be set up first."""
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )

    simple_parquet_options.set_filter(filter_expression)

    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )
    filtered_row_groups = (
        simple_hybrid_scan_reader.filter_row_groups_with_stats(
            all_row_groups, simple_parquet_options
        )
    )

    total_rows = simple_hybrid_scan_reader.total_rows_in_row_groups(
        filtered_row_groups
    )
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    filter_ranges = simple_hybrid_scan_reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, simple_parquet_options
    )
    filter_data = [
        plc.gpumemoryview(
            rmm.DeviceBuffer.to_device(
                memoryview(simple_parquet_bytes)[r.offset : r.offset + r.size],
                plc.utils._get_stream(),
            )
        )
        for r in filter_ranges
    ]

    synchronize_stream()

    # Setup chunking first
    simple_hybrid_scan_reader.setup_chunking_for_filter_columns(
        512,  # chunk_read_limit
        0,  # pass_read_limit
        filtered_row_groups,
        row_mask,
        UseDataPageMask.NO,
        filter_data,
        simple_parquet_options,
    )

    # Now has_next_table_chunk should work
    has_next = simple_hybrid_scan_reader.has_next_table_chunk()
    assert isinstance(has_next, bool)


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_chunked_reading(
    simple_parquet_bytes: bytes,
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
    stream: Stream | None,
) -> None:
    """Test chunked reading with setup and chunk methods."""
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )

    simple_parquet_options.set_filter(filter_expression)

    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )
    filtered_row_groups = (
        simple_hybrid_scan_reader.filter_row_groups_with_stats(
            all_row_groups, simple_parquet_options, stream
        )
    )

    total_rows = simple_hybrid_scan_reader.total_rows_in_row_groups(
        filtered_row_groups
    )
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    # Get column data
    filter_ranges = simple_hybrid_scan_reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, simple_parquet_options
    )
    filter_data = [
        plc.gpumemoryview(
            rmm.DeviceBuffer.to_device(
                memoryview(simple_parquet_bytes)[r.offset : r.offset + r.size],
                plc.utils._get_stream(stream),
            )
        )
        for r in filter_ranges
    ]

    synchronize_stream(stream)

    # Setup chunking for filter columns with small chunk size
    chunk_read_limit = 512  # Small limit to force multiple chunks
    pass_read_limit = 0  # No limit

    simple_hybrid_scan_reader.setup_chunking_for_filter_columns(
        chunk_read_limit,
        pass_read_limit,
        filtered_row_groups,
        row_mask,
        UseDataPageMask.NO,
        filter_data,
        simple_parquet_options,
        stream,
    )

    # Read chunks
    chunks_read = 0
    while simple_hybrid_scan_reader.has_next_table_chunk():
        chunk_result = (
            simple_hybrid_scan_reader.materialize_filter_columns_chunk(
                row_mask
            )
        )
        assert isinstance(chunk_result, plc.io.types.TableWithMetadata)
        chunks_read += 1
        # Limit iterations to avoid infinite loop if something goes wrong
        if chunks_read > 100:
            break

    # We should have read at least one chunk
    assert chunks_read > 0


def test_hybrid_scan_construct_row_group_passes(
    simple_hybrid_scan_reader,
    simple_parquet_options,
):
    """Test construct_row_group_passes with various pass read limits."""
    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )

    # zero pass read limit => single pass with all row groups
    pass_read_limit = 0
    passes = simple_hybrid_scan_reader.construct_row_group_passes(
        all_row_groups, pass_read_limit
    )
    assert passes == [all_row_groups]

    # small pass read limit => each row group in its own pass
    pass_read_limit = 1
    passes = simple_hybrid_scan_reader.construct_row_group_passes(
        all_row_groups, pass_read_limit
    )
    assert passes == [[rg] for rg in all_row_groups]

    # Passes should flatten to all row groups
    pass_read_limit = 1024
    passes = simple_hybrid_scan_reader.construct_row_group_passes(
        all_row_groups, pass_read_limit
    )
    assert [rg for p in passes for rg in p] == all_row_groups
    assert all(passes)

    # Empty input row groups raise an error
    with pytest.raises(
        ValueError, match="Empty input row group indices encountered"
    ):
        simple_hybrid_scan_reader.construct_row_group_passes(
            [], pass_read_limit
        )


def test_hybrid_scan_metadata_with_page_index(
    simple_parquet_bytes: bytes,
    simple_hybrid_scan_reader: HybridScanReader,
    simple_parquet_options: plc.io.parquet.ParquetReaderOptions,
    num_rows: int,
) -> None:
    """Test that page index setup enables page-level filtering.

    This test mirrors the C++ page-index filter tests. It verifies that:
    1. Before setup_page_index(), filter column page pruning using page-statistics will fail
    2. After fetching page index bytes and calling setup_page_index(),
       the page index is available and page-level operations work correctly
    """
    # Set a filter for testing (not critical, but needed for page index stats)
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    simple_parquet_options.set_filter(filter_expression)

    # Get initial metadata to verify reader is working
    metadata = simple_hybrid_scan_reader.parquet_metadata()
    assert metadata.num_rows == num_rows
    assert metadata.version > 0

    # Get all row groups
    all_row_groups = simple_hybrid_scan_reader.all_row_groups(
        simple_parquet_options
    )
    assert len(all_row_groups) > 0

    # Try to use build_row_mask_with_page_index_stats BEFORE setup_page_index
    # This should raise an error because column and offset indexes are not set up yet
    try:
        simple_hybrid_scan_reader.build_row_mask_with_page_index_stats(
            all_row_groups, simple_parquet_options
        )
        # If we get here, the test should fail
        pytest.fail("Expected error when using page index before setup")
    except RuntimeError:
        # This is expected - page index not set up yet
        pass

    # Get page index byte range from the reader
    page_index_byte_range = simple_hybrid_scan_reader.page_index_byte_range()
    assert page_index_byte_range.size > 0

    # Fetch page index bytes from the parquet file
    simple_parquet_mv = memoryview(simple_parquet_bytes)
    page_index_mv = simple_parquet_mv[
        page_index_byte_range.offset : page_index_byte_range.offset
        + page_index_byte_range.size
    ]

    # Setup page index with the fetched bytes
    simple_hybrid_scan_reader.setup_page_index(page_index_mv)

    # Now try to use build_row_mask_with_page_index_stats AFTER setup_page_index
    # This should work successfully
    row_mask = simple_hybrid_scan_reader.build_row_mask_with_page_index_stats(
        all_row_groups, simple_parquet_options
    )

    # Verify the row mask was created successfully
    assert row_mask is not None
    assert row_mask.size() > 0
    assert row_mask.type().id() == plc.types.TypeId.BOOL8
