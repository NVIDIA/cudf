# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cimport cython

from pylibcudf.libcudf.utilities.memory_resource cimport memory_resources
from rmm.pylibrmm.memory_resource cimport (
    DeviceMemoryResource,
    get_current_device_resource,
)

__all__ = ["MemoryResources"]


@cython.final
cdef class MemoryResources(DeviceMemoryResource):
    """Separate device memory resources for output and temporary allocations.

    Pass an instance as the ``mr`` argument of any pylibcudf function. Memory
    that is returned to the caller is allocated from ``output`` and scratch
    memory freed before the function returns is allocated from ``temporary``.
    A plain ``DeviceMemoryResource`` passed as ``mr`` is used for output
    allocations, and temporary allocations use the current device resource.

    When used as a ``DeviceMemoryResource``, for example by ``rmm``, an
    instance allocates from ``output``.

    For details, see :cpp:class:`cudf::memory_resources`.

    Parameters
    ----------
    output : DeviceMemoryResource | None
        Resource for allocations returned to the caller. Defaults to the
        current device resource.
    temporary : DeviceMemoryResource | None
        Resource for temporary allocations. Defaults to the current device
        resource.
    """
    def __cinit__(
        self,
        DeviceMemoryResource output=None,
        DeviceMemoryResource temporary=None,
    ):
        if output is None:
            output = get_current_device_resource()
        if temporary is None:
            temporary = get_current_device_resource()
        self.output = output
        self.temporary = temporary
        self.c_ref = output.c_ref
        self.c_obj = memory_resources(output.get_mr(), temporary.get_mr())

    def __repr__(self):
        return (
            f"{type(self).__name__}(output={self.output!r}, "
            f"temporary={self.temporary!r})"
        )
