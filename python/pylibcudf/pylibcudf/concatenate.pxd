# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .table cimport Table

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


# There is no way to define a fused type that is a list of other objects, so we cannot
# unify the column and table paths without using runtime dispatch instead. In this case
# we choose to prioritize API consistency over performance, so we use the same function
# with a bit of runtime dispatch overhead.
cpdef concatenate(objects, object stream = *, DeviceMemoryResource mr=*)
