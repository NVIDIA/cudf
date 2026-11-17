# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.wordpiece_tokenize cimport wordpiece_vocabulary
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cdef class WordPieceVocabulary:
    cdef unique_ptr[wordpiece_vocabulary] c_obj

cpdef Column wordpiece_tokenize(
    Column input,
    WordPieceVocabulary vocabulary,
    size_type max_words_per_row,
    object stream = *,
    DeviceMemoryResource mr=*
)
