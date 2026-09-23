# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA stream utilities."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from pylibcudf.utils import CudaStreamLike
    from rmm.pylibrmm.stream import Stream


def get_cuda_stream() -> Stream:
    """Get the default CUDA stream for the current thread."""
    return plc.utils._get_stream()


def join_cuda_streams(
    *, downstreams: Sequence[CudaStreamLike], upstreams: Sequence[CudaStreamLike]
) -> None:
    """
    Join multiple CUDA streams.

    Parameters
    ----------
    downstreams
        CUDA streams to that will be ordered after ``upstreams``.
    upstreams
        CUDA streams that will be ordered before ``downstreams``.
    """
    upstreams = list(upstreams)
    downstreams = list(downstreams)
    for downstream in downstreams:
        plc.experimental.join_streams(upstreams, downstream)


def get_joined_cuda_stream(
    get_cuda_stream: Callable[[], Stream], *, upstreams: Sequence[CudaStreamLike]
) -> Stream:
    """
    Return a CUDA stream that is joined to the given streams.

    Parameters
    ----------
    get_cuda_stream
        A zero-argument callable that returns a CUDA stream.
    upstreams
        CUDA streams that will be ordered before the returned stream.

    Returns
    -------
    CUDA stream that is joined to the given streams.
    """
    downstream = get_cuda_stream()
    join_cuda_streams(downstreams=(downstream,), upstreams=upstreams)
    return downstream


@contextlib.contextmanager
def stream_ordered_after(
    get_cuda_stream: Callable[[], Stream],
    upstreams: Sequence[CudaStreamLike],
) -> Generator[Stream, None, None]:
    """
    Get a joined CUDA stream with safe stream ordering for deallocation of inputs.

    Parameters
    ----------
    get_cuda_stream
        A zero-argument callable that returns a CUDA stream.
    upstreams
        The streams being provided to stream-ordered operations.

    Yields
    ------
    A CUDA stream that is downstream of the given streams.

    Notes
    -----
    This context manager provides two useful guarantees when working with
    objects holding references to stream-ordered objects:

    1. The stream yield upon entering the context manager is *downstream* of
       all the input streams.  This ensures that you can safely perform
       stream-ordered operations on any input using the yielded stream.
    2. The stream-ordered CUDA deallocation of the inputs happens *after* the
       context manager exits. This ensures that all stream-ordered operations
       submitted inside the context manager can complete before the memory
       referenced by the inputs is deallocated.

    Note that this does (deliberately) disconnect the dropping of the Python
    object (by its refcount dropping to 0) from the actual stream-ordered
    deallocation of the CUDA memory. This is precisely what we need to ensure
    that the inputs are valid long enough for the stream-ordered operations to
    complete.
    """
    downstream = get_joined_cuda_stream(get_cuda_stream, upstreams=upstreams)
    try:
        yield downstream
    finally:
        join_cuda_streams(downstreams=upstreams, upstreams=(downstream,))
