# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator import dereference

from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector

from pylibcudf.libcudf.scalar.scalar cimport scalar

from .scalar cimport Scalar

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport (
    DeviceMemoryResource,
    get_current_device_resource,
)

from rmm.pylibrmm.stream import DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM, Stream


import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike


# Check the environment for the variable CUDF_PER_THREAD_STREAM. If it is set,
# then set the module-scope CUDF_DEFAULT_STREAM variable here to
# rmm.pylibrmm.stream.PER_THREAD_DEFAULT_STREAM. Otherwise, it will default to
# rmm.pylibrmm.stream.DEFAULT_STREAM.
if os.getenv("CUDF_PER_THREAD_STREAM", "0") == "1":
    CUDF_DEFAULT_STREAM = PER_THREAD_DEFAULT_STREAM
else:
    CUDF_DEFAULT_STREAM = DEFAULT_STREAM

# This is a workaround for
# https://github.com/cython/cython/issues/4180
# when creating reference_wrapper[constscalar] in the constructor
ctypedef const scalar constscalar

cdef vector[reference_wrapper[const scalar]] _as_vector(list source):
    """Make a vector of reference_wrapper[const scalar] from a list of scalars."""
    cdef vector[reference_wrapper[const scalar]] c_scalars
    c_scalars.reserve(len(source))
    cdef Scalar slr
    for slr in source:
        c_scalars.push_back(
            reference_wrapper[constscalar](dereference((<Scalar?>slr).c_obj)))
    return c_scalars


cpdef Stream _get_stream(object stream: CudaStreamLike | None = None):
    if stream is None:
        return CUDF_DEFAULT_STREAM
    if isinstance(stream, Stream):
        return <Stream>stream
    return Stream(stream)  # Handles __cuda_stream__ protocol


cdef DeviceMemoryResource _get_memory_resource(DeviceMemoryResource mr = None):
    if mr is None:
        return get_current_device_resource()
    return mr
