# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning of cuDF tables."""

from cython.operator cimport dereference as deref
from cython.operator cimport postincrement
from libc.stdint cimport uint8_t, uint32_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.contiguous_split cimport PackedColumns
from pylibcudf.libcudf.table.table cimport table as cpp_table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer_resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.memory.packed_data cimport PackedData, cpp_PackedData


cdef extern from "<cudf_streaming/partition_utils.hpp>" nogil:
    int cpp_HASH_MURMUR3"cudf::hash_id::HASH_MURMUR3"
    uint32_t cpp_DEFAULT_HASH_SEED"cudf::DEFAULT_HASH_SEED",

    cdef unordered_map[uint32_t, cpp_PackedData] cpp_partition_and_pack \
        "cudf_streaming::partition_and_pack"(
            const table_view& table,
            const vector[size_type] &columns_to_hash,
            int num_partitions,
            int hash_function,
            uint32_t seed,
            cuda_stream_view stream,
            cpp_BufferResource* br,
        ) except +ex_handler

    cdef unordered_map[uint32_t, cpp_PackedData] cpp_split_and_pack \
        "cudf_streaming::split_and_pack"(
            const table_view& table,
            const vector[size_type] &splits,
            cuda_stream_view stream,
            cpp_BufferResource* br,
        ) except +ex_handler


cpdef object partition_and_pack(
    Table table,
    object columns_to_hash,
    int num_partitions,
    Stream stream,
    BufferResource br,
):
    """
    Partition rows from the input table into multiple packed tables.

    Parameters
    ----------
    table
        The input table to partition.
    columns_to_hash
        Indices of the input columns to use for hashing.
    num_partitions
        The number of partitions to create.
    stream
        The CUDA stream used for memory operations.
    br
        Buffer resource for memory allocations.

    Returns
    -------
    A dictionary where the keys are partition IDs and the values are packed tables.

    Raises
    ------
    IndexError
        If any index in ``columns_to_hash`` is invalid.

    See Also
    --------
    cudf_streaming.partition_utils.unpack_and_concat
    pylibcudf.partitioning.hash_partition
    pylibcudf.contiguous_split.pack
    cudf_streaming.partition_utils.split_and_pack
    """
    cdef cuda_stream_view _stream = stream.view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[size_type] _columns_to_hash = tuple(columns_to_hash)
    cdef unordered_map[uint32_t, cpp_PackedData] _ret
    cdef table_view tbl = table.view()
    with nogil:
        _ret = cpp_partition_and_pack(
            tbl,
            _columns_to_hash,
            num_partitions,
            cpp_HASH_MURMUR3,
            cpp_DEFAULT_HASH_SEED,
            _stream,
            _br,
        )
    ret = {}
    cdef unordered_map[uint32_t, cpp_PackedData].iterator it = _ret.begin()
    while(it != _ret.end()):
        ret[deref(it).first] = PackedData.from_librapidsmpf(
            make_unique[cpp_PackedData](move(deref(it).second)),
            br,
        )
        postincrement(it)
    return ret


cpdef object split_and_pack(
    Table table,
    object splits,
    Stream stream,
    BufferResource br,
):
    """
    Split rows from the input table into multiple packed tables.

    Parameters
    ----------
    table
        The input table to split and pack. The table cannot be empty (the
        split points would not be valid).
    splits
        The split points, one less than the number of result partitions.
    stream
        The CUDA stream used for memory operations.
    br
        Buffer resource for memory allocations.

    Returns
    -------
    A map of partition IDs and their packed tables.

    Raises
    ------
    IndexError
        If the splits are out of range for ``[0, len(table)]``.

    See Also
    --------
    cudf_streaming.partition_utils.unpack_and_concat
    pylibcudf.copying.split
    cudf_streaming.partition_utils.partition_and_pack
    """
    cdef cuda_stream_view _stream = stream.view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[size_type] _splits = tuple(splits)
    cdef unordered_map[uint32_t, cpp_PackedData] _ret
    cdef table_view tbl = table.view()
    with nogil:
        _ret = cpp_split_and_pack(
            tbl,
            _splits,
            _stream,
            _br,
        )
    ret = {}
    cdef unordered_map[uint32_t, cpp_PackedData].iterator it = _ret.begin()
    while(it != _ret.end()):
        ret[deref(it).first] = PackedData.from_librapidsmpf(
            make_unique[cpp_PackedData](move(deref(it).second)),
            br,
        )
        postincrement(it)
    return ret


cdef object pack(
    Table table,
    Stream stream,
    BufferResource br,
):
    """Pack a table into a single ``PackedData`` instance."""
    packed = split_and_pack(table, (), stream, br)
    return packed[0]


cdef extern from "<cudf_streaming/partition_utils.hpp>" nogil:
    cdef unique_ptr[cpp_table] cpp_unpack_and_concat \
        "cudf_streaming::unpack_and_concat"(
            vector[cpp_PackedData] partition,
            cuda_stream_view stream,
            cpp_BufferResource* br,
        ) except +ex_handler


cdef vector[cpp_PackedData] _partitions_py_to_cpp(partitions):
    cdef vector[cpp_PackedData] ret
    for part in partitions:
        if not (<PackedData?>part).c_obj:
            raise ValueError("PackedData was empty")
        ret.push_back(move(deref((<PackedData?>part).c_obj)))
    return move(ret)


cpdef object unpack_and_concat(
    object partitions,
    Stream stream,
    BufferResource br,
):
    """
    Unpack input partitions and concatenate them into a single table.

    Empty partitions are ignored.

    The unpacking of each partition is stream-ordered on that partition's own CUDA
    stream. The returned table is stream-ordered on the provided ``stream`` and
    synchronized with the unpacking.

    Notes
    -----
    The input partitions are released and left empty on return.

    Parameters
    ----------
    partitions
        Packed input tables (partitions).
    stream
        CUDA stream on which concatenation occurs and on which the resulting
        table is ordered.
    br
        Buffer resource used for memory allocations.

    Returns
    -------
    The concatenated table resulting from unpacking the input partitions.

    Raises
    ------
    ReservationError
        If the buffer resource cannot reserve enough memory to concatenate all
        partitions.

    See Also
    --------
    cudf_streaming.partition_utils.partition_and_pack
    """
    cdef cuda_stream_view _stream = stream.view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[cpp_PackedData] _partitions = _partitions_py_to_cpp(partitions)
    cdef unique_ptr[cpp_table] _ret
    with nogil:
        _ret = cpp_unpack_and_concat(
            move(_partitions),
            _stream,
            _br,
        )
    return Table.from_libcudf(move(_ret), stream, br._device_mr)

cdef extern from *:
    """
    #include <rapidsmpf/error.hpp>
    #include <rapidsmpf/memory/packed_data.hpp>
    #include <rapidsmpf/memory/buffer_resource.hpp>
    #include <rmm/device_buffer.hpp>

    std::unique_ptr<rapidsmpf::PackedData> cpp_packed_data_from_buffers(
        std::unique_ptr<std::vector<std::uint8_t>> metadata,
        std::unique_ptr<rmm::device_buffer> gpu_data,
        rmm::cuda_stream_view stream,
        rapidsmpf::BufferResource* br
    ) {
        return std::make_unique<rapidsmpf::PackedData>(
            std::move(metadata), br->move(std::move(gpu_data), stream)
        );
    }
    """
    unique_ptr[cpp_PackedData] cpp_packed_data_from_buffers(
        unique_ptr[vector[uint8_t]] metadata,
        unique_ptr[device_buffer] gpu_data,
        cuda_stream_view stream,
        cpp_BufferResource* br,
    ) except +ex_handler nogil


cpdef object packed_data_from_cudf_packed_columns(
    PackedColumns packed_columns,
    Stream stream,
    BufferResource br,
):
    """
    Construct a PackedData from a pylibcudf PackedColumns.

    Takes ownership of the metadata and GPU data from the PackedColumns
    object, leaving it empty.

    Parameters
    ----------
    packed_columns
        Packed columns from ``pylibcudf.contiguous_split.pack()``.
        Must not already be empty (already released).
    stream
        The CUDA stream on which the preceding ``pack()`` call was performed.
        Must be the same stream to ensure correct memory ordering.
    br
        Buffer resource for memory management.

    Returns
    -------
    A new PackedData instance owning the packed column data.

    Raises
    ------
    ValueError
        If the PackedColumns object is empty (already released).

    See Also
    --------
    pylibcudf.contiguous_split.pack
    cudf_streaming.partition_utils.unpack_and_concat
    """
    if packed_columns is None or stream is None or br is None:
        raise TypeError("Arguments must not be None")
    cdef cuda_stream_view _stream = stream.view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef PackedData ret = PackedData.__new__(PackedData)
    with nogil:
        if not (packed_columns.c_obj.get() != NULL and
                deref(packed_columns.c_obj).metadata and
                deref(packed_columns.c_obj).gpu_data):
            raise ValueError("Cannot release empty PackedColumns")
        ret.c_obj = cpp_packed_data_from_buffers(
            move(deref(packed_columns.c_obj).metadata),
            move(deref(packed_columns.c_obj).gpu_data),
            _stream,
            _br,
        )
    ret._br = br
    return ret
