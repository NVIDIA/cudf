# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

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

class HybridScanMultiFile:
    @staticmethod
    def from_parquet_metadatas(
        parquet_metadatas: Sequence[FileMetaData],
        options: ParquetReaderOptions,
    ) -> HybridScanMultiFile: ...
    def parquet_metadatas(self) -> list[FileMetaData]: ...
    def page_index_byte_ranges(self) -> list[ByteRangeInfo]: ...
    def setup_page_indexes(
        self, page_index_bytes: Sequence[Buffer]
    ) -> None: ...
    def total_rows_in_row_groups(
        self, row_group_indices: list[list[int]]
    ) -> int: ...
    def payload_pages_byte_ranges(
        self,
        row_group_indices: list[list[int]],
        row_mask: Column,
        options: ParquetReaderOptions,
        stream: CudaStreamLike | None = None,
    ) -> tuple[list[ByteRangeInfo], list[int]]: ...
    def setup_chunking_for_payload_columns(
        self,
        chunk_read_limit: int,
        pass_read_limit: int,
        row_group_indices: list[list[int]],
        row_mask: Column,
        page_data: Sequence[Span | None],
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
        row_group_indices: list[list[int]],
        pass_read_limit: int,
    ) -> list[list[list[int]]]: ...
    def has_next_table_chunk(self) -> bool: ...
