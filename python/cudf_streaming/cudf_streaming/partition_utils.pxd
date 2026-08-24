# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.contiguous_split cimport PackedColumns
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.memory.buffer_resource cimport BufferResource


cpdef object partition_and_pack(
    Table table,
    object columns_to_hash,
    int num_partitions,
    Stream stream,
    BufferResource br,
)
cpdef object split_and_pack(
    Table table,
    object splits,
    Stream stream,
    BufferResource br,
)
cpdef object unpack_and_concat(object partitions, Stream stream, BufferResource br)
cpdef object packed_data_from_cudf_packed_columns(
    PackedColumns packed_columns,
    Stream stream,
    BufferResource br,
)
