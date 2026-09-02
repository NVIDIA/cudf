# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.io.parquet import ParquetReaderOptions
from pylibcudf.io.parquet_metadata import FileMetaData
from pylibcudf.io.text import ByteRangeInfo
from pylibcudf.io.types import TableWithMetadata
from pylibcudf.span import Span
from pylibcudf.utils import CudaStreamLike

try:
    from collections.abc import Buffer
except ImportError:
    from typing_extensions import Buffer

class HybridScanMultifile:
    def __init__(
        self, footer_bytes: list[Buffer], options: ParquetReaderOptions
    ) -> None: ...
    @staticmethod
    def from_parquet_metadata(
        metadata: list[FileMetaData], options: ParquetReaderOptions
    ) -> HybridScanMultifile: ...
    def parquet_metadatas(self) -> list[FileMetaData]: ...
    def page_index_byte_ranges(self) -> list[ByteRangeInfo]: ...
    def setup_page_indexes(self, page_index_bytes: list[Buffer]) -> None: ...
    def all_row_groups(
        self, options: ParquetReaderOptions
    ) -> list[list[int]]: ...
    def total_rows_in_row_groups(
        self, row_group_indices: list[list[int]]
    ) -> int: ...
    def reset_column_selection(self) -> None: ...
    def filter_row_groups_with_byte_range(
        self,
        row_group_indices: list[list[int]],
        options: ParquetReaderOptions,
    ) -> list[list[int]]: ...
    def filter_row_groups_with_stats(
        self,
        row_group_indices: list[list[int]],
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
    ) -> list[list[int]]: ...
    def bloom_filters_byte_ranges(
        self,
        row_group_indices: list[list[int]],
        options: ParquetReaderOptions,
    ) -> tuple[list[ByteRangeInfo], list[int]]: ...
    def dictionary_pages_byte_ranges(
        self,
        row_group_indices: list[list[int]],
        options: ParquetReaderOptions,
    ) -> tuple[list[ByteRangeInfo], list[int]]: ...
    def build_all_true_row_mask(
        self,
        row_group_indices: list[list[int]],
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def build_row_mask_with_page_index_stats(
        self,
        row_group_indices: list[list[int]],
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def all_column_chunks_byte_ranges(
        self,
        row_group_indices: list[list[int]],
        options: ParquetReaderOptions,
    ) -> tuple[list[ByteRangeInfo], list[int]]: ...
    def materialize_all_columns(
        self,
        row_group_indices: list[list[int]],
        column_chunk_data: list[Span],
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> TableWithMetadata: ...
