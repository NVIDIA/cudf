# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from rmm.pylibrmm.stream import Stream

class HasCudaStream(Protocol):
    def __cuda_stream__(self) -> tuple[int, int]: ...

CudaStreamLike = Stream | HasCudaStream
