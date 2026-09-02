# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.io.text import ByteRangeInfo
from pylibcudf.io.types import SourceInfo
from pylibcudf.utils import CudaStreamLike

__all__ = [
    "IOSubmissionPolicy",
    "fetch_byte_ranges_to_device",
    "fetch_page_index_to_host",
]

class IOSubmissionPolicy(IntEnum):
    SERIALIZE = 0
    INTERLEAVE = 1

def fetch_byte_ranges_to_device(
    source_info: SourceInfo,
    byte_ranges: list[ByteRangeInfo],
    policy: IOSubmissionPolicy,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> list[gpumemoryview]: ...
def fetch_page_index_to_host(
    source_info: SourceInfo,
    page_index_range: ByteRangeInfo,
) -> bytes: ...
