# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cython requires types used as locals or return values to be
# default-constructible, which cuda::mr::resource_ref is not.
cdef extern from * nogil:
    """
    #include <cuda/memory_resource>

    #include <optional>

    struct pylibcudf_device_resource_ref {
        std::optional<cuda::mr::device_resource_ref> ref;

        pylibcudf_device_resource_ref() noexcept = default;

        template <typename T>
        pylibcudf_device_resource_ref(T const& r) noexcept
          : ref(static_cast<cuda::mr::device_resource_ref>(r)) {}

        operator cuda::mr::device_resource_ref() const noexcept {
            return ref.value();
        }
    };

    template <typename T>
    pylibcudf_device_resource_ref pylibcudf_to_device_resource_ref(
        T const& r) noexcept {
        return pylibcudf_device_resource_ref(r);
    }
    """
    cdef cppclass device_resource_ref "pylibcudf_device_resource_ref":
        device_resource_ref() noexcept

    device_resource_ref to_device_resource_ref \
        "pylibcudf_to_device_resource_ref"[T](T) noexcept


cdef extern from "<cuda/memory_resource>" namespace "cuda::mr" nogil:
    cdef cppclass any_device_resource:
        any_device_resource() except +
        any_device_resource(device_resource_ref) except +
