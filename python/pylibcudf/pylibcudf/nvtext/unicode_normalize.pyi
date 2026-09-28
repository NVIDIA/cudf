# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.table import Table
from pylibcudf.typing import CudaStreamLike

class UnicodeNormalizationForm(IntEnum):
    NFD = ...
    NFC = ...
    NFKD = ...
    NFKC = ...

class UnicodeNormalizer:
    def __init__(
        self,
        unicode_data: Table,
        form: UnicodeNormalizationForm,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def normalize_unicode(
    input: Column,
    normalizer: UnicodeNormalizer,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
