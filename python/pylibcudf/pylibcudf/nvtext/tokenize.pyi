# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.utils import CudaStreamLike

class TokenizeVocabulary:
    def __init__(
        self,
        vocab: Column,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def tokenize_scalar(
    input: Column,
    delimiter: Scalar | None = None,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def tokenize_column(
    input: Column,
    delimiters: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def count_tokens_scalar(
    input: Column,
    delimiter: Scalar | None = None,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def count_tokens_column(
    input: Column,
    delimiters: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def character_tokenize(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def detokenize(
    input: Column,
    row_indices: Column,
    separator: Scalar | None = None,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def tokenize_with_vocabulary(
    input: Column,
    vocabulary: TokenizeVocabulary,
    delimiter: Scalar,
    default_id: int = -1,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
