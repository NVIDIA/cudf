# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

class MemoryResources(DeviceMemoryResource):
    output: DeviceMemoryResource
    temporary: DeviceMemoryResource
    def __init__(
        self,
        output: DeviceMemoryResource | None = None,
        temporary: DeviceMemoryResource | None = None,
    ) -> None: ...

__all__ = ["MemoryResources"]
