# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from libcpp.memory cimport make_unique, shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.message cimport Message, cpp_Message
from rapidsmpf.streaming.core.cancellation import (
    await_cpp_future,
    shutdown_channels,
)

import asyncio


cdef extern from * nogil:
    """
    namespace {
    std::unique_ptr<cudf_streaming::cardinality_estimate>
    cpp_cardinality_estimate_from_message(
        rapidsmpf::streaming::Message msg
    ) {
        return std::make_unique<cudf_streaming::cardinality_estimate>(
            msg.release<cudf_streaming::cardinality_estimate>()
        );
    }

    void cpp_estimate_cardinality(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        cudf_streaming::cardinality_estimator& estimator,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
        std::shared_ptr<rapidsmpf::streaming::Channel> ch_sampled,
        std::vector<cudf::size_type> column_indices,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    estimator.estimate(
                        std::move(ch_in),
                        std::move(ch_out),
                        std::move(ch_sampled),
                        std::move(column_indices)
                    )
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }

    }  // namespace
    """
    unique_ptr[cpp_CardinalityEstimate] cpp_cardinality_estimate_from_message(
        cpp_Message
    ) except +ex_handler
    void cpp_estimate_cardinality(
        shared_ptr[cpp_Context] ctx,
        cpp_CardinalityEstimator& estimator,
        shared_ptr[cpp_Channel] ch_in,
        shared_ptr[cpp_Channel] ch_out,
        shared_ptr[cpp_Channel] ch_sampled,
        vector[int32_t] column_indices,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future,
    ) except +ex_handler


cdef class CardinalityEstimate:
    """Global row-count and approximate distinct-row statistics."""

    def __init__(self):
        raise ValueError("use `CardinalityEstimate.from_message`")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef CardinalityEstimate from_handle(
        unique_ptr[cpp_CardinalityEstimate] handle,
    ):
        cdef CardinalityEstimate ret = CardinalityEstimate.__new__(
            CardinalityEstimate
        )
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_message(Message message not None):
        """Construct by consuming a message."""
        return CardinalityEstimate.from_handle(
            cpp_cardinality_estimate_from_message(move(message._handle))
        )

    @property
    def row_count(self):
        """Exact global count of rows incorporated into the estimate."""
        return self._handle.get().row_count

    @property
    def distinct_count(self):
        """Approximate global distinct-row count."""
        return self._handle.get().distinct_count

cdef class CardinalityEstimator:
    """Distributed approximate distinct-row estimator."""

    def __init__(
        self,
        Context ctx not None,
        Communicator comm not None,
        int32_t tag,
        int32_t precision=14,
    ):
        self._comm = comm
        with nogil:
            self._handle = make_unique[cpp_CardinalityEstimator](
                ctx._handle,
                comm._handle,
                tag,
                precision,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def comm(self):
        """Communicator used by the estimator."""
        return self._comm

    @property
    def tag(self):
        """Collective operation identifier."""
        return self._handle.get().tag()

    @property
    def precision(self):
        """HyperLogLog precision."""
        return self._handle.get().precision()

    async def estimate(
        self,
        Context ctx not None,
        Channel ch_in not None,
        Channel ch_out not None,
        Channel ch_sampled=None,
        column_indices=(),
    ):
        """
        Estimate the global row count and distinct-row count.

        Parameters
        ----------
        ctx
            The streaming context.
        ch_in
            Channel providing TableChunks to add to the estimator.
        ch_out
            Output receiving a single CardinalityEstimate.
        ch_sampled
            If not None, input chunks are forwarded here. Must be consumed
            concurrently with ch_out if provided.
        column_indices
            Optional column indices of the input tables to use in the
            estimate. If not provided defaults to the entire table.
        """
        cdef shared_ptr[cpp_Channel] cpp_ch_sampled
        cdef vector[int32_t] cpp_column_indices = column_indices
        if ch_sampled is not None:
            cpp_ch_sampled = (<Channel>ch_sampled)._handle
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_estimate_cardinality(
                ctx._handle,
                deref(self._handle),
                ch_in._handle,
                ch_out._handle,
                cpp_ch_sampled,
                move(cpp_column_indices),
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter)),
            )
        await await_cpp_future(
            ret,
            on_cancel=lambda: shutdown_channels(
                ctx,
                ch_in,
                ch_out,
                *(() if ch_sampled is None else (ch_sampled,)),
            ),
        )
