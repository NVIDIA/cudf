# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t, uint64_t
import functools
import operator

from .types cimport DataType, size_of, type_id


cdef gpumemoryview _slice(gpumemoryview parent, uintptr_t ptr, uint64_t nbytes):
    cdef gpumemoryview v = gpumemoryview.__new__(gpumemoryview)
    v.ptr = ptr
    v.nbytes = nbytes
    v.obj = parent
    # always returns a raw byte view regardless of the source dtype.
    # TODO: Need to propagate stream from parent.cai if present
    v.cai = {"data": (ptr, parent.cai["data"][1]), "shape": (nbytes,), "typestr": "|u1", "version": 3}
    return v


__all__ = ["gpumemoryview"]


@functools.cache
def _datatype_from_dtype_desc(desc):
    mapping = {
        'u1': type_id.UINT8,
        'u2': type_id.UINT16,
        'u4': type_id.UINT32,
        'u8': type_id.UINT64,
        'i1': type_id.INT8,
        'i2': type_id.INT16,
        'i4': type_id.INT32,
        'i8': type_id.INT64,
        'f4': type_id.FLOAT32,
        'f8': type_id.FLOAT64,
        'b1': type_id.BOOL8,
        'M8[s]': type_id.TIMESTAMP_SECONDS,
        'M8[ms]': type_id.TIMESTAMP_MILLISECONDS,
        'M8[us]': type_id.TIMESTAMP_MICROSECONDS,
        'M8[ns]': type_id.TIMESTAMP_NANOSECONDS,
        'm8[s]': type_id.DURATION_SECONDS,
        'm8[ms]': type_id.DURATION_MILLISECONDS,
        'm8[us]': type_id.DURATION_MICROSECONDS,
        'm8[ns]': type_id.DURATION_NANOSECONDS,
    }
    if desc not in mapping:
        raise ValueError(f"Unsupported dtype: {desc}")
    return DataType(mapping[desc])


cdef class gpumemoryview:
    """Minimal representation of a memory buffer.

    This class aspires to be a GPU equivalent of :py:class:`memoryview` for any
    objects exposing a `CUDA Array Interface
    <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`__.
    It will be expanded to encompass more memoryview functionality over time.
    """
    # TODO: dlpack support
    def __init__(self, object obj):
        try:
            cai = obj.__cuda_array_interface__
        except AttributeError:
            raise ValueError(
                "gpumemoryview must be constructed from an object supporting "
                "the CUDA array interface"
            )
        self.obj = obj
        self.cai = cai
        # TODO: Need to respect readonly
        # TODO: Need to synchronize on stream if present in cai
        self.ptr = cai["data"][0]

        # Compute the buffer size.
        cdef size_t itemsize = size_of(
            _datatype_from_dtype_desc(
                cai["typestr"][1:]  # ignore the byteorder (the first char).
            )
        )
        self.nbytes = functools.reduce(operator.mul, cai["shape"]) * itemsize

    @property
    def __cuda_array_interface__(self):
        return self.cai

    @property
    def size(self) -> int:
        """
        Size of the memory region in bytes (Span protocol).

        This is an alias for nbytes to satisfy the Span protocol.
        """
        return self.nbytes

    def __len__(self):
        return self.cai["shape"][0]

    def byte_slice(self, s):
        """Return a byte-range sub-view of this buffer.

        Parameters
        ----------
        s : slice
            Byte-based slice.

        Returns
        -------
        gpumemoryview
            A ``|u1`` view of the requested byte range. The returned view
            holds a reference to the parent buffer, keeping it alive.

        Raises
        ------
        TypeError
            If ``s`` is not a slice.
        ValueError
            If the slice step is not 1. Out-of-range or reversed ranges
            return a zero-length view rather than raising.
        """
        if not isinstance(s, slice):
            raise TypeError(
                f"byte_slice requires a slice, not {type(s).__name__}"
            )
        start, stop, step = s.indices(self.nbytes)
        if step != 1:
            raise ValueError("byte_slice only supports step=1 slices")
        length = stop - start
        if length <= 0:
            return _slice(self, self.ptr + start, 0)
        return _slice(self, self.ptr + start, length)

    __hash__ = None
