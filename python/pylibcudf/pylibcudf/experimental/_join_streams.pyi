# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.utils import CudaStreamLike

def join_streams(
    streams: list[CudaStreamLike], stream: CudaStreamLike
) -> None: ...
