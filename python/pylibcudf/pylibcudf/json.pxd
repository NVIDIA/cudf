# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.json cimport get_json_object_options
from pylibcudf.scalar cimport Scalar

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class GetJsonObjectOptions:
    cdef get_json_object_options options


cpdef Column get_json_object(
    Column col,
    Scalar json_path,
    GetJsonObjectOptions options=*,
    object stream = *,
    DeviceMemoryResource mr=*,
)
