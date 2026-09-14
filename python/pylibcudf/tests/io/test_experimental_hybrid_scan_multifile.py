# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from utils import synchronize_stream

import rmm
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc
from pylibcudf.io.experimental import HybridScanMultiFile


@pytest.fixture(scope="module")
def num_rows() -> int:
    """Number of rows in each parquet source."""
    return 500


@pytest.fixture(scope="module")
def num_row_groups() -> int:
    """Number of row groups in each parquet source."""
    return 2


@pytest.fixture(scope="module")
def parquet_table(num_rows: int) -> pa.Table:
    """Create a simple PyArrow table for testing."""
    return pa.table(
        {
            "col0": pa.array(list(range(num_rows)), type=pa.uint32()),
            "col1": [float(row) * 1.5 for row in range(num_rows)],
        }
    )


@pytest.fixture(scope="module")
def parquet_bytes(
    parquet_table: pa.Table, num_rows: int, num_row_groups: int
) -> list[bytes]:
    """Create two identical parquet sources with a page index."""
    buf = io.BytesIO()
    pq.write_table(
        parquet_table,
        buf,
        row_group_size=num_rows // num_row_groups,
        # Small, non-dictionary pages so a sparse row mask prunes some of them
        data_page_size=256,
        write_batch_size=32,
        use_dictionary=False,
        write_statistics=True,
        write_page_index=True,
    )
    return [buf.getvalue()] * 2


def footer_bytes(parquet_bytes: bytes) -> memoryview:
    """Extract the footer bytes of a parquet file.

    According to Parquet file format specification:
    https://parquet.apache.org/docs/file-format/
    """
    PARQUET_FOOTER_SIZE_BYTES = 4  # Number of bytes encoding footer length
    PARQUET_MAGIC_BYTES = 4  # Number of bytes for "PAR1" magic number
    PARQUET_SUFFIX_BYTES = PARQUET_FOOTER_SIZE_BYTES + PARQUET_MAGIC_BYTES

    parquet_mv = memoryview(parquet_bytes)

    footer_size = int.from_bytes(
        parquet_mv[-PARQUET_SUFFIX_BYTES:-PARQUET_MAGIC_BYTES],
        byteorder="little",
    )
    footer_start = len(parquet_mv) - PARQUET_SUFFIX_BYTES - footer_size
    footer_end = len(parquet_mv) - PARQUET_SUFFIX_BYTES
    return parquet_mv[footer_start:footer_end]


@pytest.fixture
def parquet_options() -> plc.io.parquet.ParquetReaderOptions:
    """Create ParquetReaderOptions for the hybrid scan reader.

    The reader never reads through a datasource, so the options carry no source.

    Note: This is function-scoped (not module-scoped) because tests may modify
    the options, so each test needs its own independent copy.
    """
    return plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([])
    ).build()


@pytest.fixture
def row_groups(num_row_groups: int) -> list[list[int]]:
    """Row group indices of both parquet sources."""
    return [list(range(num_row_groups))] * 2


@pytest.fixture
def hybrid_scan_multifile_reader(
    parquet_bytes: list[bytes],
    parquet_options: plc.io.parquet.ParquetReaderOptions,
) -> HybridScanMultiFile:
    """Create a HybridScanMultiFile with the page index of both sources."""
    # Create the reader from the footer bytes of each source
    reader = HybridScanMultiFile.from_parquet_metadatas(
        [
            plc.io.parquet_metadata.FileMetaData.from_bytes(
                footer_bytes(source)
            )
            for source in parquet_bytes
        ],
        parquet_options,
    )
    # Fetch the page index of each source and set it up within the metadata
    reader.setup_page_indexes(
        [
            memoryview(source)[
                byte_range.offset : byte_range.offset + byte_range.size
            ]
            for source, byte_range in zip(
                parquet_bytes,
                reader.page_index_byte_ranges(),
                strict=True,
            )
        ]
    )
    return reader


def test_hybrid_scan_multifile_construct_directly_raises() -> None:
    """Test that a HybridScanMultiFile cannot be constructed directly."""
    with pytest.raises(ValueError, match="cannot be constructed directly"):
        HybridScanMultiFile()


def test_hybrid_scan_multifile_metadata(
    hybrid_scan_multifile_reader: HybridScanMultiFile,
    row_groups: list[list[int]],
    num_rows: int,
) -> None:
    """Test the metadata of a reader built from pre-populated metadata."""
    # One metadata object per source, in source order
    assert [
        metadata.num_rows
        for metadata in hybrid_scan_multifile_reader.parquet_metadatas()
    ] == [num_rows, num_rows]

    # Row counts are totalled across all sources
    assert (
        hybrid_scan_multifile_reader.total_rows_in_row_groups(row_groups)
        == 2 * num_rows
    )

    # Every source was written with a page index
    assert all(
        byte_range.size > 0
        for byte_range in hybrid_scan_multifile_reader.page_index_byte_ranges()
    )


def test_hybrid_scan_multifile_construct_row_group_passes(
    hybrid_scan_multifile_reader: HybridScanMultiFile,
    row_groups: list[list[int]],
) -> None:
    """Test partitioning the input row groups into passes."""
    # No read limit yields a single pass spanning all sources
    assert hybrid_scan_multifile_reader.construct_row_group_passes(
        row_groups, 0
    ) == [row_groups]

    # A tiny read limit splits the row groups across multiple passes
    assert (
        len(
            hybrid_scan_multifile_reader.construct_row_group_passes(
                row_groups, 1
            )
        )
        > 1
    )


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_multifile_materialize_payload_pages(
    parquet_bytes: list[bytes],
    hybrid_scan_multifile_reader: HybridScanMultiFile,
    parquet_options: plc.io.parquet.ParquetReaderOptions,
    row_groups: list[list[int]],
    parquet_table: pa.Table,
    num_rows: int,
    stream: Stream | None,
) -> None:
    """Test reading payload columns page by page from multiple sources."""
    # Without a filter, all selected columns are payload columns
    parquet_options.set_column_names(["col0", "col1"])

    # Select a sparse set of rows across both sources
    mask = [row % 100 == 0 for row in range(2 * num_rows)]
    row_mask = plc.Column.from_arrow(pa.array(mask), stream=stream)

    # Get the byte ranges of the payload pages surviving the row mask
    page_ranges, source_indices = (
        hybrid_scan_multifile_reader.payload_pages_byte_ranges(
            row_groups, row_mask, parquet_options, stream
        )
    )

    # Byte ranges are flattened, with one source index per byte range
    assert len(page_ranges) == len(source_indices)
    assert set(source_indices) == set(range(len(parquet_bytes)))

    # Pruned pages are reported as empty byte ranges
    empty_ranges = sum(byte_range.size == 0 for byte_range in page_ranges)
    assert 0 < empty_ranges < len(page_ranges)

    # Fetch the surviving pages from their source, passing None for pruned ones
    page_data = [
        None
        if byte_range.size == 0
        else plc.gpumemoryview(
            rmm.DeviceBuffer.to_device(
                memoryview(parquet_bytes[source])[
                    byte_range.offset : byte_range.offset + byte_range.size
                ],
                plc.utils._get_stream(stream),
            )
        )
        for byte_range, source in zip(page_ranges, source_indices, strict=True)
    ]
    synchronize_stream(stream)

    # The data page mask is inferred from the fetched page data
    hybrid_scan_multifile_reader.setup_chunking_for_payload_columns(
        1024,  # chunk_read_limit
        0,  # pass_read_limit
        row_groups,
        row_mask,
        page_data,
        parquet_options,
        stream,
    )

    # Read the output chunks, applying the row mask to each
    chunks = []
    while hybrid_scan_multifile_reader.has_next_table_chunk():
        chunks.append(
            hybrid_scan_multifile_reader.materialize_payload_columns_chunk(
                row_mask
            )
        )
    synchronize_stream(stream)

    # The chunk read limit is small enough to split the output
    assert len(chunks) > 1

    # The chunks reassemble into the masked rows of both sources
    result = pa.concat_tables([chunk.tbl.to_arrow() for chunk in chunks])
    expected = pa.concat_tables([parquet_table] * 2).filter(pa.array(mask))
    assert result.equals(
        expected.rename_columns(result.schema.names).cast(result.schema)
    )
