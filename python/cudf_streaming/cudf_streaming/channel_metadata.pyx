# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel metadata types for streaming pipelines."""

from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport null_order as cpp_null_order
from pylibcudf.libcudf.types cimport order as cpp_order
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.streaming.core.message cimport Message
from cudf_streaming.table_chunk cimport TableChunk, cpp_TableChunk


cdef extern from * nogil:
    """
    #include <memory>
    #include <cudf_streaming/channel_metadata.hpp>

    static std::unique_ptr<cudf_streaming::channel_metadata>
    cpp_channel_metadata_from_message(rapidsmpf::streaming::Message msg) {
        return std::make_unique<cudf_streaming::channel_metadata>(
            msg.release<cudf_streaming::channel_metadata>()
        );
    }
    """
    unique_ptr[cpp_ChannelMetadata] cpp_channel_metadata_from_message(
        cpp_Message
    ) except +


cdef class HashScheme:
    """Hash partitioning scheme: rows distributed by hash(column_indices) % modulus."""

    def __init__(self, object column_indices, int modulus):
        """
        Parameters
        ----------
        column_indices
            Column indices to hash on.
        modulus
            Number of hash partitions.
        """
        cdef vector[int32_t] cols
        for c in column_indices:
            cols.push_back(<int32_t?>c)
        self._handle = cpp_HashScheme(cols, modulus)

    @staticmethod
    cdef HashScheme from_cpp(cpp_HashScheme scheme):
        cdef HashScheme ret = HashScheme.__new__(HashScheme)
        ret._handle = move(scheme)
        return ret

    @property
    def column_indices(self) -> tuple:
        """Column indices used for hashing."""
        return tuple(self._handle.column_indices)

    @property
    def modulus(self) -> int:
        """Number of hash partitions."""
        return self._handle.modulus

    def __eq__(self, other):
        if not isinstance(other, HashScheme):
            return NotImplemented
        return self._handle == (<HashScheme>other)._handle

    def __repr__(self):
        return f"HashScheme({self.column_indices!r}, {self.modulus})"


cdef class OrderKey:
    """A single sort key: column index, sort direction, and null placement."""

    def __init__(
        self,
        int column_index,
        cpp_order order,
        cpp_null_order null_order,
    ):
        """
        Parameters
        ----------
        column_index
            Zero-based index of the sort column.
        order
            Sort direction (ascending or descending).
        null_order
            Null placement (before or after non-null values).
        """
        self._handle = cpp_OrderKey(column_index, order, null_order)

    @staticmethod
    cdef OrderKey from_cpp(cpp_OrderKey key):
        cdef OrderKey ret = OrderKey.__new__(OrderKey)
        ret._handle = key
        return ret

    @property
    def column_index(self) -> int:
        """Zero-based index of the sort column."""
        return self._handle.column_index

    @property
    def order(self):
        """Sort direction (ascending or descending)."""
        return self._handle.order

    @property
    def null_order(self):
        """Null placement (before or after non-null values)."""
        return self._handle.null_order

    def __eq__(self, other):
        if not isinstance(other, OrderKey):
            return NotImplemented
        return self._handle == (<OrderKey>other)._handle

    def __repr__(self):
        return f"OrderKey({self.column_index}, {self.order!r}, {self.null_order!r})"


cdef class Ordering:
    """A valid ordering description for sorted/range-partitioned data."""

    def __init__(
        self,
        object keys,
        TableChunk boundaries not None,
        *,
        bint strict_boundaries = False,
    ):
        cdef vector[cpp_OrderKey] cpp_keys
        for key in keys:
            cpp_keys.push_back((<OrderKey?>key)._handle)
        self._handle = cpp_Ordering(
            move(cpp_keys), move(boundaries.release_handle()), strict_boundaries
        )

    @staticmethod
    cdef Ordering from_cpp(cpp_Ordering ordering):
        cdef Ordering ret = Ordering.__new__(Ordering)
        ret._handle = move(ordering)
        return ret

    @property
    def keys(self) -> tuple:
        """Sort keys, one per sort column."""
        cdef int i
        cdef int n = self._handle.keys.size()
        return tuple(OrderKey.from_cpp(self._handle.keys[i]) for i in range(n))

    @property
    def column_indices(self) -> tuple:
        """Column indices for the ordering keys."""
        cdef int i
        cdef int n = self._handle.keys.size()
        return tuple(self._handle.keys[i].column_index for i in range(n))

    @property
    def strict_boundaries(self) -> bool:
        """Whether chunks are strictly aligned to boundary ranges."""
        return self._handle.strict_boundaries

    @property
    def num_boundaries(self) -> int:
        """Number of boundary rows (N-1 for N partitions)."""
        return self._handle.boundaries.get().shape().first

    def get_boundaries(self, BufferResource br not None) -> TableChunk:
        """
        Return the boundary rows.

        Parameters
        ----------
        br
            Buffer resource to associate with the returned table chunk.
        """
        cdef const cpp_TableChunk* chunk = self._handle.boundaries.get()
        cdef Stream stream = Stream._from_cudaStream_t(chunk.stream().value())
        tbl = Table.from_table_view_of_arbitrary(
            chunk.table_view(), owner=self, stream=stream
        )
        return TableChunk.from_pylibcudf_table(
            tbl, stream, exclusive_view=False, br=br
        )

    def with_keys(self, object new_keys) -> Ordering:
        """Return a new ``Ordering`` with updated key column indices."""
        cdef vector[cpp_OrderKey] cpp_keys
        for key in new_keys:
            cpp_keys.push_back((<OrderKey?>key)._handle)
        return Ordering.from_cpp(self._handle.with_keys(move(cpp_keys)))

    def boundaries_aligned_with(
        self, Ordering other not None, BufferResource br not None
    ) -> bool:
        """
        Check whether boundary values are aligned with another ordering.

        Parameters
        ----------
        other
            The ordering to compare against.
        br
            Buffer resource for temporary allocations during comparison.
        """
        return self._handle.boundaries_aligned_with(other._handle, deref(br.ptr()))

    def __repr__(self):
        return (
            f"Ordering({self.keys!r}, "
            f"strict_boundaries={self.strict_boundaries})"
        )


cdef class OrderScheme:
    """Order-based partitioning scheme for sorted/range-partitioned data.

    An OrderScheme advertises that the same stream is sorted/range-partitioned
    with respect to any individual ``Ordering`` it contains. Consumers should
    inspect the ``Ordering`` they intend to use for keys, boundaries, and
    strictness.

    Parameters
    ----------
    orderings
        Non-empty sequence of ``Ordering`` objects valid for the stream.
    """

    def __init__(self, object orderings):
        cdef vector[cpp_Ordering] cpp_orderings
        for ordering in orderings:
            cpp_orderings.push_back((<Ordering?>ordering)._handle)
        self._handle = cpp_OrderScheme(move(cpp_orderings))

    @staticmethod
    cdef OrderScheme from_cpp(cpp_OrderScheme scheme):
        cdef OrderScheme ret = OrderScheme.__new__(OrderScheme)
        ret._handle = move(scheme)
        return ret

    @property
    def orderings(self) -> tuple:
        """Ordering descriptions valid for the stream."""
        cdef int i
        cdef int n = self._handle.orderings.size()
        return tuple(
            Ordering.from_cpp(self._handle.orderings[i]) for i in range(n)
        )

    def __repr__(self):
        return f"OrderScheme({self.orderings!r})"


cdef void _apply_spec(cpp_PartitioningSpec& spec, obj) except *:
    """Set *spec* in-place from a Python value."""
    if obj is None:
        spec = cpp_PartitioningSpec.none()
    elif obj == "inherit":
        spec = cpp_PartitioningSpec.inherit()
    elif isinstance(obj, HashScheme):
        spec = cpp_PartitioningSpec.from_hash((<HashScheme>obj)._handle)
    elif isinstance(obj, OrderScheme):
        spec = cpp_PartitioningSpec.from_order((<OrderScheme>obj)._handle)
    else:
        raise TypeError(
            f"Expected HashScheme, OrderScheme, None, or 'inherit', "
            f"got {type(obj).__name__}"
        )


cdef object _from_spec(const cpp_PartitioningSpec& spec):
    """Convert PartitioningSpec (by reference) to a Python object."""
    if spec.type == cpp_PartitioningSpec.cpp_Type.NONE:
        return None
    elif spec.type == cpp_PartitioningSpec.cpp_Type.INHERIT:
        return "inherit"
    elif spec.type == cpp_PartitioningSpec.cpp_Type.HASH:
        return HashScheme.from_cpp(deref(spec.hash))
    elif spec.type == cpp_PartitioningSpec.cpp_Type.ORDER:
        return OrderScheme.from_cpp(deref(spec.order))  # copies out of optional
    else:
        raise ValueError("Unknown PartitioningSpec.Type")


cdef class Partitioning:
    """
    Hierarchical partitioning metadata for a data stream.

    Parameters
    ----------
    inter_rank
        Distribution across ranks. Can be a HashScheme, OrderScheme, None,
        or 'inherit'.
    local
        Distribution within a rank. Can be a HashScheme, OrderScheme, None,
        or 'inherit'.
    """

    def __init__(self, inter_rank=None, local=None):
        _apply_spec(self._handle.inter_rank, inter_rank)
        _apply_spec(self._handle.local, local)

    @staticmethod
    cdef Partitioning from_cpp(cpp_Partitioning data):
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._handle = move(data)
        return ret

    @property
    def inter_rank(self):
        """Inter-rank partitioning spec."""
        return _from_spec(self._handle.inter_rank)

    @property
    def local(self):
        """Intra-rank (local) partitioning spec."""
        return _from_spec(self._handle.local)

    def __repr__(self):
        return f"Partitioning(inter_rank={self.inter_rank!r}, local={self.local!r})"


cdef class ChannelMetadata:
    """
    Channel-level metadata describing a data stream.

    Parameters
    ----------
    local_count
        Estimated number of chunks for this rank.
    partitioning
        How the data is partitioned (default: no partitioning).
    duplicated
        Whether data is duplicated on all workers (default: False).
    """

    def __init__(
        self,
        int local_count,
        *,
        partitioning: Partitioning | None = None,
        bint duplicated = False,
    ):
        if local_count < 0:
            raise ValueError(f"local_count must be non-negative, got {local_count}")

        cdef cpp_Partitioning part
        if partitioning is not None:
            part = (<Partitioning?>partitioning)._handle

        self._handle = make_unique[cpp_ChannelMetadata](
            local_count, part, duplicated
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef ChannelMetadata from_handle(unique_ptr[cpp_ChannelMetadata] handle):
        cdef ChannelMetadata ret = ChannelMetadata.__new__(ChannelMetadata)
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_message(Message message not None):
        """Construct by consuming a Message (message becomes empty)."""
        return ChannelMetadata.from_handle(
            cpp_channel_metadata_from_message(move(message._handle))
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this ChannelMetadata into a Message.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Empty message that will take ownership of this metadata.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message_channel_metadata(
            sequence_number, move(self.release_handle())
        )

    cdef const cpp_ChannelMetadata* handle_ptr(self) except NULL:
        """Return pointer to underlying handle, raising if released."""
        if not self._handle:
            raise ValueError("ChannelMetadata is uninitialized, has it been released?")
        return self._handle.get()

    @property
    def local_count(self) -> int:
        """Estimated number of chunks for this rank."""
        return self.handle_ptr().local_count

    @property
    def partitioning(self) -> Partitioning:
        """How the data is partitioned."""
        return Partitioning.from_cpp(self.handle_ptr().partitioning)

    @property
    def duplicated(self) -> bool:
        """Whether data is duplicated on all workers."""
        return self.handle_ptr().duplicated

    def __repr__(self):
        return (
            f"ChannelMetadata(local_count={self.local_count}, "
            f"partitioning={self.partitioning!r}, "
            f"duplicated={self.duplicated})"
        )

    cdef unique_ptr[cpp_ChannelMetadata] release_handle(self):
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)
