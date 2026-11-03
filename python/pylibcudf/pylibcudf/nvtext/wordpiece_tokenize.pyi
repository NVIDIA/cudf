# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

class WordPieceVocabulary:
    def __init__(
        self,
        vocab: Column,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def wordpiece_tokenize(
    input: Column,
    vocabulary: WordPieceVocabulary,
    max_words_per_row: int,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
