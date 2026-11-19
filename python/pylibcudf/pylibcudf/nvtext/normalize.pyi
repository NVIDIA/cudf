# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntFlag

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

class NormalizeFlags(IntFlag):
    NONE = 0
    STRIP_ACCENTS = 1
    PAD_PUNCTUATION = 2

class CharacterNormalizer:
    def __init__(
        self,
        do_lower_case: bool,
        special_tokens: Column,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def normalize_spaces(
    input: Column,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def normalize_characters(
    input: Column,
    normalizer: CharacterNormalizer,
    flags: int = ...,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
