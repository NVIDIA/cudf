# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.io.text cimport ByteRangeInfo
from pylibcudf.io.types cimport SourceInfo
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cpdef list fetch_byte_ranges_to_device(
    SourceInfo source_info,
    list byte_ranges,
    object stream=*,
    DeviceMemoryResource mr=*,
)

cpdef bytes fetch_page_index_to_host(
    SourceInfo source_info,
    ByteRangeInfo page_index_range,
)
