# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.nvtext.unicode_normalize cimport (
    create_unicode_normalizer as cpp_create_unicode_normalizer,
    normalize_unicode as cpp_normalize_unicode,
    unicode_normalization_form as cpp_unicode_normalization_form,
)
from pylibcudf.libcudf.nvtext.unicode_normalize import \
    unicode_normalization_form as UnicodeNormalizationForm  # no-cython-lint
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["UnicodeNormalizer", "UnicodeNormalizationForm", "normalize_unicode"]


cdef class UnicodeNormalizer:
    """Normalizer object for Unicode TR15 normalization (NFD/NFC/NFKD/NFKC).

    Constructed from the three relevant columns of UnicodeData.txt loaded as a
    :class:`pylibcudf.Table`. Once built the object can be reused across
    multiple calls to :func:`normalize_unicode`.

    For details, see :cpp:class:`nvtext::unicode_normalizer`.

    Parameters
    ----------
    unicode_data : Table
        Three-column table parsed from UnicodeData.txt:
        column[0] STRING  code-point hex strings (e.g. "00C9"),
        column[1] INT32   Canonical_Combining_Class values,
        column[2] STRING  Decomposition_Mapping field.
    form : unicode_normalization_form
        Normalization form to apply (NFD, NFC, NFKD, or NFKC).
    stream : CudaStreamLike | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for internal table allocations.
    """
    def __cinit__(
        self,
        Table unicode_data,
        cpp_unicode_normalization_form form,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None,
    ):
        cdef table_view c_data = unicode_data.view()
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        cdef DeviceMemoryResource _mr = _get_memory_resource(mr)
        with nogil:
            self.c_obj = move(
                cpp_create_unicode_normalizer(c_data, form, _cs, _mr.get_mr())
            )

    __hash__ = None


cpdef Column normalize_unicode(
    Column input,
    UnicodeNormalizer normalizer,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Normalize a strings column using Unicode TR15 normalization.

    Input and output are UTF-8 encoded. Each string is normalized
    independently; null entries produce null output entries.

    For details, see :cpp:func:`nvtext::normalize_unicode`.

    Parameters
    ----------
    input : Column
        Strings column to normalize.
    normalizer : UnicodeNormalizer
        Normalizer built by :class:`UnicodeNormalizer`.
    stream : CudaStreamLike | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for the returned column.

    Returns
    -------
    Column
        New strings column of normalized UTF-8 strings.
    """
    if normalizer is None:
        raise TypeError("normalizer must not be None")
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_normalize_unicode(
            c_input,
            dereference(normalizer.c_obj.get()),
            _cs,
            _mr.get_mr(),
        )

    return Column.from_libcudf(move(c_result), _stream, _mr)
