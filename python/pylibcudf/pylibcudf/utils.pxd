# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector
from pylibcudf.libcudf.scalar.scalar cimport scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

cdef vector[reference_wrapper[const scalar]] _as_vector(list source)
cpdef Stream _get_stream(object stream = *)
cdef DeviceMemoryResource _get_memory_resource(DeviceMemoryResource mr = *)
cpdef void _set_up_kvikio(object nthreads = *)
