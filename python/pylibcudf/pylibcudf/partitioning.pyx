# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cimport pylibcudf.libcudf.types as libcudf_types
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport partitioning as cpp_partitioning
from pylibcudf.libcudf.partitioning import hash_id as HashId  # no-cython-lint
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t


__all__ = [
    "hash_partition",
    "partition",
    "round_robin_partition",
]

cpdef tuple[Table, list] hash_partition(
    Table input,
    TableOrList keys,
    int num_partitions,
    cpp_partitioning.hash_id hash_function = cpp_partitioning.hash_id.HASH_MURMUR3,
    uint32_t seed = cpp_partitioning.DEFAULT_HASH_SEED,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Partitions rows from the input table into multiple output tables.

    For details, see :cpp:func:`hash_partition`.

    Parameters
    ----------
    input : Table
        The table to partition
    keys : Table | list[int]
        Table providing keys to hash or list of indices of input columns to hash
    num_partitions : int
        The number of partitions to use
    hash_function : HashId
        Hashing function apply to key columns.
    seed : int
        Seed for hash function.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    tuple[Table, list[int]]
        An output table and a list of `num_partitions + 1` row offsets where
        partition `i` contains rows in the range `[offsets[i], offsets[i+1])`
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    cdef int c_num_partitions = num_partitions
    cdef vector[libcudf_types.size_type] columns_to_hash
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef table_view c_input = input.view()
    cdef table_view c_keys
    if TableOrList is Table:
        c_keys = keys.view()
        with nogil:
            c_result = cpp_partitioning.hash_partition(
                c_input,
                c_keys,
                c_num_partitions,
                hash_function,
                seed,
                _cs,
                mr.get_mr()
            )
    else:
        columns_to_hash = keys
        with nogil:
            c_result = cpp_partitioning.hash_partition(
                c_input,
                columns_to_hash,
                c_num_partitions,
                hash_function,
                seed,
                _cs,
                mr.get_mr()
            )
    return Table.from_libcudf(move(c_result.first), _stream, mr), list(c_result.second)


cpdef tuple[Table, list] partition(
    Table t,
    Column partition_map,
    int num_partitions,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Partitions rows of `t` according to the mapping specified by `partition_map`.

    For details, see :cpp:func:`partition`.

    Parameters
    ----------
    t : Table
        The table to partition
    partition_map : Column
        Non-nullable column of integer values that map each row
        in `t` to it's partition.
    num_partitions : int
        The total number of partitions
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    tuple[Table, list[int]]
        An output table and a list of `num_partitions + 1` row offsets where
        partition `i` contains rows in the range `[offsets[i], offsets[i+1])`
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    cdef int c_num_partitions = num_partitions

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = t.view()
    cdef column_view c_partition_map = partition_map.view()
    with nogil:
        c_result = cpp_partitioning.partition(
            c_input,
            c_partition_map,
            c_num_partitions,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result.first), _stream, mr), list(c_result.second)


cpdef tuple[Table, list] round_robin_partition(
    Table input,
    int num_partitions,
    int start_partition=0,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Round-robin partition.

    For details, see :cpp:func:`round_robin_partition`.

    Parameters
    ----------
    input : Table
        The input table to be round-robin partitioned
    num_partitions : int
        Number of partitions for the table
    start_partition : int, default 0
        Index of the 1st partition
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    tuple[Table, list[int]]
        The partitioned table and a list of `num_partitions + 1` partition offsets
        where partition `i` contains rows in the range `[offsets[i], offsets[i+1])`.
    """
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] c_result
    cdef int c_num_partitions = num_partitions
    cdef int c_start_partition = start_partition

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_partitioning.round_robin_partition(
            c_input,
            c_num_partitions,
            c_start_partition,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result.first), _stream, mr), list(c_result.second)
