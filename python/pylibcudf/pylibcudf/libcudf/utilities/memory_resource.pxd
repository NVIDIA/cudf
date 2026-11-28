# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm.memory_resource cimport device_async_resource_ref


# Cython requires types used as locals, attributes, or return values to be
# default-constructible, which cudf::memory_resources is not.
cdef extern from * nogil:
    """
    #include <cudf/utilities/memory_resource.hpp>

    #include <rmm/resource_ref.hpp>

    #include <optional>

    struct pylibcudf_memory_resources {
        std::optional<cudf::memory_resources> mrs;

        pylibcudf_memory_resources() noexcept = default;

        pylibcudf_memory_resources(
            rmm::device_async_resource_ref output_mr) noexcept
          : mrs(std::in_place, output_mr) {}

        pylibcudf_memory_resources(
            rmm::device_async_resource_ref output_mr,
            rmm::device_async_resource_ref temporary_mr) noexcept
          : mrs(std::in_place, output_mr, temporary_mr) {}

        operator cudf::memory_resources() const noexcept {
            return mrs.value();
        }

        rmm::device_async_resource_ref get_output_mr() const noexcept {
            return mrs.value().get_output_mr();
        }

        rmm::device_async_resource_ref get_temporary_mr() const noexcept {
            return mrs.value().get_temporary_mr();
        }
    };
    """
    cdef cppclass memory_resources "pylibcudf_memory_resources":
        memory_resources() noexcept
        memory_resources(device_async_resource_ref output_mr) noexcept
        memory_resources(
            device_async_resource_ref output_mr,
            device_async_resource_ref temporary_mr,
        ) noexcept
        device_async_resource_ref get_output_mr() noexcept
        device_async_resource_ref get_temporary_mr() noexcept
