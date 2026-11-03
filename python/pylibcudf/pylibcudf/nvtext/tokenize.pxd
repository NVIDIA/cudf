# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.tokenize cimport tokenize_vocabulary
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cdef class TokenizeVocabulary:
    cdef unique_ptr[tokenize_vocabulary] c_obj

cpdef Column tokenize_scalar(
    Column input, Scalar delimiter=*, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column tokenize_column(
    Column input, Column delimiters, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column count_tokens_scalar(
    Column input, Scalar delimiter=*, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column count_tokens_column(
    Column input, Column delimiters, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column character_tokenize(
    Column input, object stream = *, DeviceMemoryResource mr=*
)

cpdef Column detokenize(
    Column input,
    Column row_indices,
    Scalar separator=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)

cpdef Column tokenize_with_vocabulary(
    Column input,
    TokenizeVocabulary vocabulary,
    Scalar delimiter,
    size_type default_id=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)
