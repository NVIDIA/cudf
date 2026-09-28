# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from pylibcudf.contiguous_split cimport PackedColumns
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.memory_reservation cimport MemoryReservation


cpdef size_t partition_and_pack_cost(
    Table table,
    Stream stream,
    BufferResource br,
)
cpdef object partition_and_pack(
    Table table,
    object columns_to_hash,
    int num_partitions,
    Stream stream,
    BufferResource br,
    MemoryReservation reservation=*,
)
cpdef size_t split_and_pack_cost(
    Table table,
    Stream stream,
    BufferResource br,
)
cpdef object split_and_pack(
    Table table,
    object splits,
    Stream stream,
    BufferResource br,
    MemoryReservation reservation=*,
)
cpdef size_t unpack_and_concat_cost(object partitions)
cpdef object unpack_and_concat(
    object partitions,
    Stream stream,
    BufferResource br,
    MemoryReservation reservation=*,
)
cpdef object packed_data_from_cudf_packed_columns(
    PackedColumns packed_columns,
    Stream stream,
    BufferResource br,
)
