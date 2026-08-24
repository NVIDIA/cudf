# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from libc.stdint cimport uint32_t
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport size_type

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.streaming.core.actor cimport CppActor, cpp_Actor
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context


cdef extern from "<cudf_streaming/partition.hpp>" nogil:
    int cpp_HASH_MURMUR3"cudf::hash_id::HASH_MURMUR3"
    uint32_t cpp_DEFAULT_HASH_SEED"cudf::DEFAULT_HASH_SEED",
    cdef cpp_Actor cpp_partition_and_pack \
        "cudf_streaming::actor::partition_and_pack"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            vector[size_type] columns_to_hash,
            int num_partitions,
            int hash_function,
            uint32_t seed,
        ) except +ex_handler
    cdef cpp_Actor cpp_unpack_and_concat \
        "cudf_streaming::actor::unpack_and_concat"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
        ) except +ex_handler


def partition_and_pack(
    Context ctx not None,
    Channel ch_in not None,
    Channel ch_out not None,
    object columns_to_hash not None,
    int num_partitions,
):
    """
    Asynchronously partition and pack table chunks.

    This is the streaming equivalent of
    :func:`cudf_streaming.partition_utils.partition_and_pack()`,
    operating on incoming table chunks via channels.

    Each incoming table from `ch_in` is partitioned into `num_partitions` outputs
    based on a hash of the specified columns. Each partition is then serialized
    (packed) and sent to the output channel `ch_out`.

    Parameters
    ----------
    ctx
        The streaming actor context used to create and manage the asynchronous task.
    ch_in
        Input channel that provides ``TableChunk`` objects to partition.
    ch_out
        Output channel to which packed partitions (``PartitionMapChunk`` objects)
        are sent.
    columns_to_hash
        Indices of input columns to hash when computing partition assignments.
    num_partitions
        Number of output partitions to create.

    Returns
    -------
    A streaming actor representing the asynchronous partitioning and packing operation.

    Raises
    ------
    ValueError
        If any index in ``columns_to_hash`` is invalid.

    See Also
    --------
    cudf_streaming.partition_utils.partition_and_pack
        Non-streaming variant operating on static tables.
    cudf_streaming.partition.unpack_and_concat
        The inverse operation that unpacks and concatenates packed partitions.
    """
    cdef vector[size_type] _columns_to_hash = tuple(columns_to_hash)
    cdef cpp_Actor _ret
    with nogil:
        _ret = cpp_partition_and_pack(
            ctx._handle,
            ch_in._handle,
            ch_out._handle,
            _columns_to_hash,
            num_partitions,
            cpp_HASH_MURMUR3,
            cpp_DEFAULT_HASH_SEED,
        )
    return CppActor.from_handle(
        make_unique[cpp_Actor](move(_ret)), owner = None
    )


def unpack_and_concat(
    Context ctx not None,
    Channel ch_in not None,
    Channel ch_out not None,
):
    """
    Asynchronously unpack and concatenate packed partitions.

    This is the streaming equivalent of
    :func:`cudf_streaming.partition_utils.unpack_and_concat()`,
    operating on packed partition chunks via channels.

    The function receives packed partitions from `ch_in`, deserializes them,
    concatenates the partitions belonging to the same logical table, and sends the
    resulting tables to `ch_out`. Empty partitions are automatically ignored.

    Parameters
    ----------
    ctx
        The streaming actor context used to manage asynchronous execution.
    ch_in
        Input channel providing packed partitions (``PartitionMapChunk`` or
        ``PartitionVectorChunk``).
    ch_out
        Output channel receiving the unpacked and concatenated ``TableChunk`` objects.

    Returns
    -------
    A streaming actor representing the asynchronous unpacking and concatenation
    operation.

    See Also
    --------
    cudf_streaming.partition_utils.unpack_and_concat
        Non-streaming version.
    cudf_streaming.partition.partition_and_pack
        The inverse operation that partitions and packs tables into partitions.
    """
    cdef cpp_Actor _ret
    with nogil:
        _ret = cpp_unpack_and_concat(
            ctx._handle,
            ch_in._handle,
            ch_out._handle,
        )
    return CppActor.from_handle(
        make_unique[cpp_Actor](move(_ret)), owner = None
    )
