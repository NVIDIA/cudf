# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.utils import CudaStreamLike

class BPEMergePairs:
    def __init__(
        self,
        merge_pairs: Column,
        stream: CudaStreamLike | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def byte_pair_encoding(
    input: Column,
    merge_pairs: BPEMergePairs,
    separator: Scalar | None = None,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
