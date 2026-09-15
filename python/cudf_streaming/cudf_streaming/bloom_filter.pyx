# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint64_t
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport size_type

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.cancellation import (await_cpp_future, shutdown_channels)

import asyncio


cdef extern from * nogil:
    """
    namespace {
    void cpp_bloom_filter_build(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        cudf_streaming::bloom_filter& bloom_filter,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
        int32_t tag,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    bloom_filter.build(
                        std::move(ch_in),
                        std::move(ch_out),
                        tag
                    )
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    void cpp_bloom_filter_build(
        shared_ptr[cpp_Context] ctx,
        cpp_BloomFilter& bloom_filter,
        shared_ptr[cpp_Channel] ch_in,
        shared_ptr[cpp_Channel] ch_out,
        int32_t tag,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


cdef extern from * nogil:
    """
    namespace {
    void cpp_bloom_filter_apply(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        cudf_streaming::bloom_filter& bloom_filter,
        std::shared_ptr<rapidsmpf::streaming::Channel> bloom_filter_ch,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
        std::vector<cudf::size_type> keys,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    bloom_filter.apply(
                        std::move(bloom_filter_ch),
                        std::move(ch_in),
                        std::move(ch_out),
                        std::move(keys)
                    )
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    void cpp_bloom_filter_apply(
        shared_ptr[cpp_Context] ctx,
        cpp_BloomFilter& bloom_filter,
        shared_ptr[cpp_Channel] bloom_filter_ch,
        shared_ptr[cpp_Channel] ch_in,
        shared_ptr[cpp_Channel] ch_out,
        vector[size_type] keys,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


cdef class BloomFilter:
    """
    Streaming bloom filter construction and application.

    Parameters
    ----------
    ctx
        Streaming context.
    comm
        The communicator the bloom filter construction is collective over.
    seed
        Seed used for hashing values into the bloom filter.
    filter_size
        Filter storage size in bytes. Must be positive and satisfy
        ``BloomFilter.aligned_size(filter_size) == filter_size``.
    """

    def __init__(
        self,
        Context ctx not None,
        Communicator comm not None,
        uint64_t seed,
        size_t filter_size,
    ):
        self._comm = comm
        with nogil:
            self._handle = make_unique[cpp_BloomFilter](
                ctx._handle,
                comm._handle,
                seed,
                filter_size,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def comm(self):
        """
        Get the communicator used by the bloom filter.

        Returns
        -------
        The communicator.
        """
        return self._comm

    @staticmethod
    def aligned_size(size_t size):
        """
        Return the largest valid filter size no greater than a byte count.

        Parameters
        ----------
        size
            Byte count to align.

        Returns
        -------
        Largest valid filter size less than or equal to ``size``.
        """
        cdef size_t ret
        with nogil:
            ret = cpp_aligned_size(size)
        return ret

    async def build(
        self,
        Context ctx not None,
        Channel ch_in not None,
        Channel ch_out not None,
        int32_t tag,
    ):
        """
        Build a bloom filter from input table chunks.

        Parameters
        ----------
        ctx
            The current streaming context.
        ch_in
            Input channel of ``TableChunk`` objects.
        ch_out
            Output channel receiving a single bloom filter message.
        tag
            Disambiguating tag to combine filters across ranks.
        """
        # Coroutine bridging pattern: create a Python future, transfer
        # ownership to C++ via OwningWrapper (Py_INCREF here, py_deleter
        # calls Py_DECREF when the C++ task completes), then spawn the
        # C++ coroutine which resolves the future via cpp_set_py_future.
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_bloom_filter_build(
                ctx._handle,
                deref(self._handle),
                ch_in._handle,
                ch_out._handle,
                tag,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter)),
            )
        # Note: multi-rank, if we get an exception we can't cancel the
        # in-progress AllReduce, so this might still hang.
        await await_cpp_future(
            ret, on_cancel=lambda: shutdown_channels(ctx, ch_in, ch_out)
        )

    async def apply(
        self,
        Context ctx not None,
        Channel bloom_filter not None,
        Channel ch_in not None,
        Channel ch_out not None,
        keys,
    ):
        """
        Apply a bloom filter to incoming table chunks.

        Parameters
        ----------
        ctx
            The current streaming context.
        bloom_filter
            Channel containing the bloom filter (a single message).
        ch_in
            Input channel of ``TableChunk`` objects to filter.
        ch_out
            Output channel receiving filtered ``TableChunk`` objects.
        keys
            Indices selecting the key columns for hash fingerprints.
        """
        cdef vector[size_type] c_keys = tuple(keys)
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_bloom_filter_apply(
                ctx._handle,
                deref(self._handle),
                bloom_filter._handle,
                ch_in._handle,
                ch_out._handle,
                move(c_keys),
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter)),
            )
        await await_cpp_future(
            ret, on_cancel=lambda: shutdown_channels(ctx, bloom_filter, ch_in, ch_out)
        )
