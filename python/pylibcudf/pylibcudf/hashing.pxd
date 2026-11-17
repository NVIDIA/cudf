# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table


cpdef Column murmurhash3_x86_32(
    Table input,
    uint32_t seed=*,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Table murmurhash3_x64_128(
    Table input,
    uint64_t seed=*,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column xxhash_32(
    Table input,
    uint32_t seed=*,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column xxhash_64(
    Table input,
    uint64_t seed=*,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef Column md5(Table input, object stream = *, DeviceMemoryResource mr=*)
cpdef Column sha1(Table input, object stream = *, DeviceMemoryResource mr=*)
cpdef Column sha224(Table input, object stream = *, DeviceMemoryResource mr=*)
cpdef Column sha256(Table input, object stream = *, DeviceMemoryResource mr=*)
cpdef Column sha384(Table input, object stream = *, DeviceMemoryResource mr=*)
cpdef Column sha512(Table input, object stream = *, DeviceMemoryResource mr=*)
