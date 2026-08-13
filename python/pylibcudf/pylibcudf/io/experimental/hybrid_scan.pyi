# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.io.parquet import ParquetReaderOptions
from pylibcudf.io.parquet_metadata import FileMetaData
from pylibcudf.io.text import ByteRangeInfo
from pylibcudf.io.types import TableWithMetadata
from pylibcudf.utils import CudaStreamLike

try:
    from collections.abc import Buffer
except ImportError:
    from typing_extensions import Buffer

class UseDataPageMask(IntEnum):
    YES = 1
    NO = 0

class HybridScanReader:
    def __init__(
        self, footer_bytes: Buffer, options: ParquetReaderOptions
    ) -> None: ...
    @staticmethod
    def from_parquet_metadata(
        metadata: FileMetaData, options: ParquetReaderOptions
    ) -> HybridScanReader: ...
    def parquet_metadata(self) -> FileMetaData: ...
    def page_index_byte_range(self) -> ByteRangeInfo: ...
    def setup_page_index(self, page_index_bytes: Buffer) -> None: ...
    def all_row_groups(self, options: ParquetReaderOptions) -> list[int]: ...
    def total_rows_in_row_groups(
        self, row_group_indices: list[int]
    ) -> int: ...
    def reset_column_selection(self) -> None: ...
    def filter_row_groups_with_stats(
        self,
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
    ) -> list[int]: ...
    def secondary_filters_byte_ranges(
        self, row_group_indices: list[int], options: ParquetReaderOptions
    ) -> tuple[list[ByteRangeInfo], list[ByteRangeInfo]]: ...
    def filter_row_groups_with_dictionary_pages(
        self,
        dictionary_page_data: list,
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
    ) -> list[int]: ...
    def filter_row_groups_with_bloom_filters(
        self,
        bloom_filter_data: list,
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
    ) -> list[int]: ...
    def build_row_mask_with_page_index_stats(
        self,
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def filter_column_chunks_byte_ranges(
        self, row_group_indices: list[int], options: ParquetReaderOptions
    ) -> list[ByteRangeInfo]: ...
    def materialize_filter_columns(
        self,
        row_group_indices: list[int],
        column_chunk_data: list,
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> TableWithMetadata: ...
    def payload_column_chunks_byte_ranges(
        self, row_group_indices: list[int], options: ParquetReaderOptions
    ) -> list[ByteRangeInfo]: ...
    def materialize_payload_columns(
        self,
        row_group_indices: list[int],
        column_chunk_data: list,
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> TableWithMetadata: ...
    def all_column_chunks_byte_ranges(
        self, row_group_indices: list[int], options: ParquetReaderOptions
    ) -> list[ByteRangeInfo]: ...
    def materialize_all_columns(
        self,
        row_group_indices: list[int],
        column_chunk_data: list,
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> TableWithMetadata: ...
    def setup_chunking_for_filter_columns(
        self,
        chunk_read_limit: int,
        pass_read_limit: int,
        row_group_indices: list[int],
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        column_chunk_data: list,
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> None: ...
    def materialize_filter_columns_chunk(
        self,
        row_mask: Column,
    ) -> TableWithMetadata: ...
    def setup_chunking_for_payload_columns(
        self,
        chunk_read_limit: int,
        pass_read_limit: int,
        row_group_indices: list[int],
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        column_chunk_data: list,
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> None: ...
    def materialize_payload_columns_chunk(
        self,
        row_mask: Column,
    ) -> TableWithMetadata: ...
    def construct_row_group_passes(
        self,
        row_group_indices: list[int],
        pass_read_limit: int,
    ) -> list[list[int]]: ...
    def has_next_table_chunk(self) -> bool: ...
