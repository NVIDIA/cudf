# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cython cimport no_gc_clear
from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.table cimport Table

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.memory_reservation cimport (MemoryReservation,
                                                  cpp_MemoryReservation)
from rapidsmpf.memory.packed_data cimport PackedData
# Need the header include for inline C++ code
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper  # no-cython-lint
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.context cimport Context
from rapidsmpf.streaming.core.message cimport Message, cpp_Message

from rapidsmpf.memory.buffer import MemoryType as py_MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory


cdef extern from "<cudf_streaming/table_chunk.hpp>" nogil:
    cpp_Message cpp_to_message"cudf_streaming::to_message"\
        (uint64_t sequence_number, unique_ptr[cpp_TableChunk]) except +ex_handler


cdef extern from * nogil:
    """
    namespace {
    std::unique_ptr<cudf_streaming::table_chunk>
    cpp_release_table_chunk_from_message(
        rapidsmpf::streaming::Message &&msg
    ) {
        return std::make_unique<cudf_streaming::table_chunk>(
            msg.release<cudf_streaming::table_chunk>()
        );
    }

    std::unique_ptr<cudf_streaming::table_chunk>
    cpp_from_table_view_with_owner(
        cudf::table_view view,
        rmm::cuda_stream_view stream,
        PyObject *owner,
        void(*py_deleter)(void *),
        bool exclusive_view
    ) {
        // Called holding the gil.
        // Decref is done by the deleter.
        Py_XINCREF(owner);
        return std::make_unique<cudf_streaming::table_chunk>(
            view,
            stream,
            rapidsmpf::OwningWrapper(owner, py_deleter),
            exclusive_view ?
                cudf_streaming::table_chunk::exclusive_view::YES
                : cudf_streaming::table_chunk::exclusive_view::NO
        );
    }

    std::unique_ptr<cudf_streaming::table_chunk> cpp_table_make_available(
        std::unique_ptr<cudf_streaming::table_chunk> &&table,
        rapidsmpf::MemoryReservation* reservation
    ) {
        return std::make_unique<cudf_streaming::table_chunk>(
            table->make_available(*reservation)
        );
    }

    std::unique_ptr<cudf_streaming::table_chunk> cpp_table_copy(
        std::unique_ptr<cudf_streaming::table_chunk> const& table,
        rapidsmpf::MemoryReservation* reservation
    ) {
        return std::make_unique<cudf_streaming::table_chunk>(
            table->copy(*reservation)
        );
    }
    }  // namespace
    """
    unique_ptr[cpp_TableChunk] cpp_release_table_chunk_from_message(
        cpp_Message
    ) except +ex_handler
    unique_ptr[cpp_TableChunk] cpp_from_table_view_with_owner(...) except +ex_handler
    unique_ptr[cpp_TableChunk] cpp_table_make_available(
        unique_ptr[cpp_TableChunk], cpp_MemoryReservation*
    ) except +ex_handler
    unique_ptr[cpp_TableChunk] cpp_table_copy(
        unique_ptr[cpp_TableChunk], cpp_MemoryReservation*
    ) except +ex_handler


@no_gc_clear
cdef class TableChunk:
    """
    A unit of table data in a streaming pipeline.

    Represents either an unpacked pylibcudf table, a packed (serialized) table,
    or `rapidsmpf.memory.packed_data.PackedData`.

    A TableChunk may be initially unavailable (e.g., if the data is packed or
    spilled), and can be made available (i.e., materialized to device memory)
    on demand.

    Use the factory functions `from_pylibcudf_table` and `from_message` to
    create a new table chunk.
    """
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef TableChunk from_handle(
        unique_ptr[cpp_TableChunk] handle, BufferResource br,
    ):
        """
        Construct a TableChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ TableChunk.
        br
            A BufferResource to keep alive.

        Returns
        -------
        A new TableChunk wrapping the given handle.
        """
        cdef TableChunk ret = TableChunk.__new__(TableChunk)
        ret._handle = move(handle)
        ret._br = br
        return ret

    @staticmethod
    def from_pylibcudf_table(
        Table table not None,
        Stream stream not None,
        *,
        bool_t exclusive_view,
        BufferResource br not None,
    ):
        """
        Construct a TableChunk from a pylibcudf Table.

        Parameters
        ----------
        table
            A pylibcudf Table to wrap as a TableChunk.
        stream
            The CUDA stream on which this chunk was created.
        exclusive_view
            Indicates that this TableChunk has exclusive ownership semantics for the
            underlying table view.

            When ``True``, the following guarantees must hold:
              - The pylibcudf Table is the sole representation of the table data,
                i.e. no views exist.
              - The Table object exclusively owns the table's device memory.

            These guarantees allow the TableChunk to be spillable and ensure that
            when the owner is destroyed, the underlying device memory is correctly
            freed.

        Returns
        -------
        A new TableChunk wrapping the given pylibcudf Table.

        Notes
        -----
        The returned TableChunk maintains a reference to ``table`` to ensure
        its underlying buffers remain valid for the lifetime of the chunk.
        This reference is managed by the underlying C++ object, so it
        persists even when the chunk is transferred through Channels.

        """
        cdef cuda_stream_view _stream = stream.view()
        cdef cpp_table_view view = table.view()
        return TableChunk.from_handle(
            cpp_from_table_view_with_owner(
                view,
                _stream,
                <PyObject *>table,
                py_deleter,
                exclusive_view,
            ),
            br,
        )

    @staticmethod
    def from_packed_data(PackedData pd not None, BufferResource br not None):
        """
        Construct a TableChunk from packed data.

        Parameters
        ----------
        pd
            The PackedData object

        Returns
        -------
        A new TableChunk owning the packed data.

        Notes
        -----
        This takes ownership of the data in the PackedData object, which is left empty.
        """
        return TableChunk.from_handle(make_unique[cpp_TableChunk](move(pd.c_obj)), br)

    @staticmethod
    def from_message(Message message not None, BufferResource br not None):
        """
        Construct a TableChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a TableChunk. The message is released and is empty
            after this call.

        Returns
        -------
        A new TableChunk extracted from the given message.
        """
        return TableChunk.from_handle(
            cpp_release_table_chunk_from_message(move(message._handle)),
            br,
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this TableChunk into an empty Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this TableChunk as
        its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this TableChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The original table chunk is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message(
            sequence_number, move(self.release_handle())
        )

    cdef const cpp_TableChunk* handle_ptr(self):
        """
        Return a pointer to the underlying C++ TableChunk.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the TableChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("TableChunk is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_TableChunk] release_handle(self):
        """
        Release ownership of the underlying C++ TableChunk.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the TableChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("TableChunk is uninitialized, has it been released?")
        return move(self._handle)

    @property
    def stream(self):
        """
        Return the CUDA stream on which this chunk was created.

        Returns
        -------
        Stream
            The CUDA stream.
        """
        return Stream._from_cudaStream_t(
            deref(self.handle_ptr()).stream().value()
        )

    def data_alloc_size(self, mem_type=None):
        """
        Number of bytes allocated for the data in the specified memory type.

        Parameters
        ----------
        mem_type
            The memory type to query. If None, returns the total size across
            all memory types.

        Returns
        -------
        Number of bytes allocated.
        """
        if mem_type is None:
            return sum(self.data_alloc_size(m) for m in py_MemoryType)
        return deref(self.handle_ptr()).data_alloc_size(mem_type)

    def is_available(self):
        """
        Indicates whether the underlying table data is fully available in
        device memory.

        Returns
        -------
        True if the table is already available; otherwise, False.
        """
        return deref(self.handle_ptr()).is_available()

    def make_available_cost(self):
        """
        Return the estimated cost (in bytes) of making the table available.

        Currently, only device memory usage is accounted for in this estimate.

        Returns
        -------
        The estimated cost in bytes.
        """
        return deref(self.handle_ptr()).make_available_cost()

    def make_available(self, MemoryReservation reservation not None):
        """
        Move this table chunk into a new one with its data made available.

        As part of the move, a copy or unpack operation may be performed,
        using the associated CUDA stream for execution.

        Parameters
        ----------
        reservation
            Memory reservation used for allocations, if making data available
            is needed.

        Returns
        -------
        A new table chunk with its data available on device.

        Warnings
        --------
        The original table chunk is released and must not be used after this call.
        """
        cdef cpp_MemoryReservation* res = reservation._handle.get()
        cdef unique_ptr[cpp_TableChunk] handle = self.release_handle()
        cdef unique_ptr[cpp_TableChunk] ret
        with nogil:
            ret = cpp_table_make_available(move(handle), res)
        return TableChunk.from_handle(move(ret), self._br)

    async def make_available_or_wait(
        self, Context ctx not None, *, int64_t net_memory_delta
    ):
        """
        Move this table chunk into a new one with its data made available.

        This is an asynchronous variant of :meth:`make_available`. The coroutine may
        suspend if the required device memory is not immediately available and
        resumes once a memory reservation has been granted or an error condition is
        reached.

        Parameters
        ----------
        ctx
            Streaming context used to access the memory reservation mechanism.
        net_memory_delta
            Estimated change in memory usage after the reservation is granted and
            all work using the returned table chunk has completed.

        Returns
        -------
        A new table chunk with its data available on device.

        Raises
        ------
        RuntimeError
            If shutdown occurs before the reservation can be processed.
        OverflowError
            If no progress is possible within the timeout and overbooking is
            disabled.

        Warnings
        --------
        The original table chunk is released and must not be used after this call.
        """
        return self.make_available(
            await reserve_memory(
                ctx, self.make_available_cost(), net_memory_delta=net_memory_delta
            )
        )

    def make_available_and_spill(
        self, BufferResource br not None, *, allow_overbooking
    ):
        """
        Make this table chunk available on device, spilling other data if necessary.

        Ensures that the data backing this table chunk is made available in device
        memory. If there is insufficient free memory to complete the operation, the
        buffer resource may spill other data until enough space has been freed.

        Parameters
        ----------
        br
            Buffer resource used for allocations and spill management.
        allow_overbooking
            Whether the memory reservation may temporarily exceed the current
            allocation limit.

        Returns
        -------
        A new table chunk with its data made available on device.

        Raises
        ------
        ReservationError
            If the allocation or spilling process fails to free enough memory.

        Warnings
        --------
        The original table chunk is released and must not be used after this call.

        Examples
        --------
        >>> # Make the data of an existing chunk available on device
        >>> chunk = chunk.make_available_and_spill(br, allow_overbooking=False)
        >>> chunk.table_view()
        """

        cdef MemoryReservation res = br.reserve_device_memory_and_spill(
            self.make_available_cost(),
            allow_overbooking=allow_overbooking
        )
        return self.make_available(res)

    def table_view(self):
        """
        Returns a view of the underlying pylibcudf table.

        The table must be available in device memory.

        Returns
        -------
        A view of the underlying table. The view holds a reference to this
        `TableChunk` to ensure it remains alive.

        Raises
        ------
        ValueError
            If ``self.is_available() is False``.
        """
        cdef const cpp_TableChunk* handle = self.handle_ptr()
        cdef cpp_table_view ret
        with nogil:
            ret = deref(handle).table_view()
        return Table.from_table_view_of_arbitrary(ret, owner=self, stream=self.stream)

    def is_spillable(self):
        """
        Indicates whether this chunk can be spilled.

        A chunk is considered spillable if it was created from one of the following:
          - A message (via ``.from_message()``).
          - An exclusive pylibcudf table (via
            ``.from_pylibcudf_table(..., exclusive_view=True)``).

        Both of these creation paths imply device-owning semantics, meaning the
        TableChunk owns its underlying memory and can safely be spilled to host memory.

        Returns
        -------
        True if the table chunk can be spilled, otherwise, False.
        """
        return deref(self.handle_ptr()).is_spillable()

    def copy(self, MemoryReservation reservation not None):
        """
        Create a deep copy of this table chunk.

        All buffers are allocated for the new table chunk using the provided
        memory reservation, which also determines the target memory type of
        the copy.

        Parameters
        ----------
        reservation
            Memory reservation to consume for allocating the buffers of the
            new table chunk.

        Returns
        -------
        TableChunk
            A new table chunk containing a deep copy of this chunk's data and
            metadata.
        """
        cdef unique_ptr[cpp_TableChunk] ret
        cdef cpp_MemoryReservation* res = reservation._handle.get()
        with nogil:
            ret = cpp_table_copy(self._handle, res)
        return TableChunk.from_handle(move(ret), self._br)

    def into_packed_data(self, BufferResource br not None):
        """
        Convert this table chunk to a PackedData, avoiding unnecessary copies.

        If the chunk's data is already in packed form (e.g., it arrived over the
        network or was constructed from a :class:`PackedData`), the packed data is
        moved out directly with no copy. Otherwise the table is serialized via
        ``cudf.pack()``.

        Parameters
        ----------
        br
            Buffer resource used when packing is required.

        Returns
        -------
        PackedData
            The resulting packed data.

        Raises
        ------
        ValueError
            If the data is not already packed and the table is not available
            (i.e., ``is_available() == False``).

        Warnings
        --------
        The original table chunk is released and must not be used after this call.
        """
        cdef unique_ptr[cpp_PackedData] result
        cdef cpp_BufferResource* _br = br.ptr()
        cdef unique_ptr[cpp_TableChunk] handle = self.release_handle()
        with nogil:
            result = move(deref(handle)).into_packed_data(_br)
        return PackedData.from_librapidsmpf(move(result), br)

    @property
    def shape(self):
        """Return the shape of the table in this TableChunk.

        Returns
        -------
        Tuple of shape ``(num_rows, num_columns)```.
        """
        return deref(self._handle).shape()


async def make_table_chunks_available_or_wait(
    Context ctx not None,
    chunks,
    *,
    size_t reserve_extra,
    int64_t net_memory_delta,
    allow_overbooking=None,
):
    """
    Make one or more table chunks available, waiting on a memory reservation if needed.

    This helper combines :meth:`TableChunk.make_available` with :func:`reserve_memory`.
    It computes the device-memory cost of making the provided table chunks available,
    reserves that amount (plus ``reserve_extra``), and then returns new table chunks
    whose data are available on device.

    The coroutine may suspend if the required device memory is not immediately
    available and resumes once a memory reservation has been granted or an error
    condition is reached. The behavior when the progress timeout expires depends
    on whether overbooking is allowed.

    Parameters
    ----------
    ctx
        Streaming context used to access the memory reservation mechanism.
    chunks
        A table chunk or an iterable of table chunks to make available on device.
    reserve_extra
        Additional bytes to include in the reservation beyond the aggregated
        availability cost of ``chunks``.
    net_memory_delta
        Estimated change in memory usage after the reservation is granted and all
        work using the returned table chunks has completed. This value is used as
        a heuristic to prioritize eligible requests.
    allow_overbooking
        Whether to allow overbooking if no progress is possible.
          - If ``True``, the reservation may overbook memory when no further
            progress can be made. If ``False``, the call fails when no progress
            is possible.
          - If ``None``, the behavior is determined by the configuration option
            ``"allow_overbooking_by_default"``, which is read via ``ctx.options()``.

    Returns
    -------
    A tuple containing:
      - If a chunk is provided: the new table chunk with its data available on device.
      - If multiple chunks are provided: a list of new table chunks with their data
        available on device.
      - The memory reservation used with a remaining size of @p reserve_extra.

    Raises
    ------
    RuntimeError
        If shutdown occurs before the reservation can be processed.
    OverflowError
        If no progress is possible within the timeout and overbooking is disabled.

    Warnings
    --------
    The original table chunks are released and must not be used after this call.
    """
    # Handle both single chunk and iterable of chunks
    input_chunks = chunks
    if isinstance(input_chunks, TableChunk):
        chunks = (input_chunks,)

    size = sum(chunk.make_available_cost() for chunk in chunks)
    res = await reserve_memory(
        ctx,
        size + reserve_extra,
        net_memory_delta=net_memory_delta,
        mem_type=MemoryType.DEVICE,
        allow_overbooking=allow_overbooking,
    )
    available_chunks = [chunk.make_available(res) for chunk in chunks]
    if isinstance(input_chunks, TableChunk):
        return available_chunks[0], res
    else:
        return available_chunks, res
