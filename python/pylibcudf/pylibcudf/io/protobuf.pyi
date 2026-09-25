# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.utils import CudaStreamLike

__all__ = ["decode_protobuf"]

def decode_protobuf(
    binary_input: Column,
    schema: list[tuple[int, int, int, int, int, int, bool, bool, bool]],
    default_ints: list[int],
    default_floats: list[float],
    default_bools: list[bool],
    default_strings: list[bytes],
    enum_valid_values: list[list[int]],
    enum_names: list[list[bytes]],
    fail_on_errors: bool,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
