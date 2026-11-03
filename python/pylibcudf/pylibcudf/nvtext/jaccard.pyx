# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.jaccard cimport (
    jaccard_index as cpp_jaccard_index,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["jaccard_index"]

cpdef Column jaccard_index(
    Column input1,
    Column input2,
    size_type width,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """
    Returns the Jaccard similarity between individual rows in two strings columns.

    For details, see :cpp:func:`jaccard_index`

    Parameters
    ----------
    input1 : Column
        Input strings column
    input2 : Column
        Input strings column
    width : size_type
        The ngram number to generate
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        Index calculation values
    """
    cdef column_view c_input1 = input1.view()
    cdef column_view c_input2 = input2.view()
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_jaccard_index(
            c_input1,
            c_input2,
            width,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
