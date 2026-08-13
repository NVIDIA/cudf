# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from pylibcudf.column cimport Column
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io cimport text as cpp_text
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "ByteRangeInfo",
    "DataChunkSource",
    "ParseOptions",
    "make_source",
    "make_source_from_bgzip_file",
    "make_source_from_file",
    "multibyte_split",
]


cdef class ByteRangeInfo:
    """Information about a byte range in a file.

    For details, see :cpp:class:`cudf::io::text::byte_range_info`

    Parameters
    ----------
    offset : int
        Offset in bytes from the start of the file
    size : int
        Size of the range in bytes
    """

    def __init__(self, size_t offset, size_t size):
        self.c_obj = byte_range_info(offset, size)

    @property
    def offset(self):
        """Get the offset in bytes."""
        return self.c_obj.offset()

    @property
    def size(self):
        """Get the size in bytes."""
        return self.c_obj.size()


cdef class ParseOptions:
    """
    Parsing options for `multibyte_split`

    Parameters
    ----------
    byte_range : list | tuple, default None
        Only rows starting inside this byte range will be
        part of the output column.

    strip_delimiters : bool, default True
        Whether delimiters at the end of rows should
        be stripped from the output column.
    """
    def __init__(
        self,
        *,
        byte_range: tuple[int, int] | list[int] | None = None,
        strip_delimiters=False,
    ):
        self.c_options = cpp_text.parse_options()
        if byte_range is not None:
            c_byte_range_offset = byte_range[0]
            c_byte_range_size = byte_range[1]
            self.c_options.byte_range = cpp_text.byte_range_info(
                c_byte_range_offset,
                c_byte_range_size
            )
        self.c_options.strip_delimiters = strip_delimiters


cdef class DataChunkSource:
    """
    Data source for `multibyte_split`

    Parameters
    ----------
    data : str
        Filename or data itself.
    """

    def __cinit__(self, str data):
        # Need to keep a reference alive for make_source
        self.data_ref = data.encode()


cpdef DataChunkSource make_source(str data):
    """
    Creates a data source capable of producing device-buffered views
    of the given string.

    Parameters
    ----------
    data : str
        The host data to be exposed as a data chunk source.

    Returns
    -------
    DataChunkSource
        The data chunk source for the provided host data.
    """
    cdef DataChunkSource dcs = DataChunkSource(data)
    with nogil:
        dcs.c_source = move(cpp_text.make_source(dcs.data_ref))
    return dcs


cpdef DataChunkSource make_source_from_file(str filename):
    """
    Creates a data source capable of producing device-buffered views of the file.

    Parameters
    ----------
    filename : str
        The filename of the file to be exposed as a data chunk source.

    Returns
    -------
    DataChunkSource
        The data chunk source for the provided filename.
    """
    cdef DataChunkSource dcs = DataChunkSource(filename)
    with nogil:
        dcs.c_source = move(cpp_text.make_source_from_file(dcs.data_ref))
    return dcs

cpdef DataChunkSource make_source_from_bgzip_file(
    str filename,
    int virtual_begin=-1,
    int virtual_end=-1,
):
    """
    Creates a data source capable of producing device-buffered views of
    a BGZIP compressed file with virtual record offsets.

    Parameters
    ----------
    filename : str
        The filename of the BGZIP-compressed file to be exposed as a data chunk source.

    virtual_begin : int
        The virtual (Tabix) offset of the first byte to be read. Its upper 48 bits
        describe the offset into the compressed file, its lower 16 bits describe the
        block-local offset.

    virtual_end : int, default None
        The virtual (Tabix) offset one past the last byte to be read

    Returns
    -------
    DataChunkSource
        The data chunk source for the provided filename.
    """
    cdef uint64_t c_virtual_begin
    cdef uint64_t c_virtual_end
    cdef DataChunkSource dcs = DataChunkSource(filename)

    if virtual_begin == -1 and virtual_end == -1:
        with nogil:
            dcs.c_source = move(cpp_text.make_source_from_bgzip_file(dcs.data_ref))
    elif virtual_begin != -1 and virtual_end != -1:
        c_virtual_begin = virtual_begin
        c_virtual_end = virtual_end
        with nogil:
            dcs.c_source = move(
                cpp_text.make_source_from_bgzip_file(
                    dcs.data_ref,
                    c_virtual_begin,
                    c_virtual_end,
                )
            )
    else:
        raise ValueError(
            "virtual_begin and virtual_end must both be None or both be int"
        )
    return dcs

cpdef Column multibyte_split(
    DataChunkSource source,
    str delimiter,
    ParseOptions options=None,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """
    Splits the source text into a strings column using a multiple byte delimiter.

    For details, see :cpp:func:`multibyte_split`

    Parameters
    ----------
    source :
        The source string.

    delimiter : str
        UTF-8 encoded string for which to find offsets in the source.

    options : ParseOptions
        The parsing options to use (including byte range).

    stream : Stream, optional
        CUDA stream for device memory operations and kernel launches

    Returns
    -------
    Column
        The strings found by splitting the source by the delimiter
        within the relevant byte range.
    """
    cdef unique_ptr[column] c_result
    cdef unique_ptr[data_chunk_source] c_source = move(source.c_source)
    cdef string c_delimiter = delimiter.encode()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if options is None:
        options = ParseOptions()

    cdef cpp_text.parse_options c_options = options.c_options

    with nogil:
        c_result = cpp_text.multibyte_split(
            dereference(c_source),
            c_delimiter,
            c_options,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
