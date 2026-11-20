# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.utilities.memory_resource cimport memory_resources
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class MemoryResources(DeviceMemoryResource):
    cdef memory_resources c_obj
    cdef readonly DeviceMemoryResource output
    cdef readonly DeviceMemoryResource temporary
