/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/traits.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

namespace cudf::groupby::detail::hash {
template <typename Source, cudf::aggregation::Kind k>
struct update_target_element_shmem {
  __device__ void operator()(cuda::std::byte*,
                             cudf::size_type,
                             cudf::column_device_view,
                             cudf::size_type) const
  { CUDF_UNREACHABLE("Invalid source type and aggregation combination."); }
};

template <typename Source>
  requires(cudf::is_fixed_width<Source>() &&
           cudf::has_atomic_support<device_storage_type_t<Source>>())
struct update_target_element_shmem<Source, cudf::aggregation::MIN> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceTarget =
      cudf::device_storage_type_t<cudf::detail::target_type_t<Source, cudf::aggregation::MIN>>;
    using DeviceSource = cudf::device_storage_type_t<Source>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_min(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));
  }
};

template <typename Source>
  requires(cudf::is_fixed_width<Source>() &&
           cudf::has_atomic_support<device_storage_type_t<Source>>())
struct update_target_element_shmem<Source, cudf::aggregation::MAX> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceTarget =
      cudf::device_storage_type_t<cudf::detail::target_type_t<Source, cudf::aggregation::MAX>>;
    using DeviceSource = cudf::device_storage_type_t<Source>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_max(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));
  }
};

template <typename Source>
  requires(cudf::is_fixed_width<Source>() && !cudf::is_timestamp<Source>())
struct update_target_element_shmem<Source, cudf::aggregation::SUM> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using DeviceTarget =
      cudf::device_storage_type_t<cudf::detail::target_type_t<Source, cudf::aggregation::SUM>>;
    using DeviceSource = cudf::device_storage_type_t<Source>;

    DeviceTarget* target_casted = reinterpret_cast<DeviceTarget*>(target);
    cudf::detail::atomic_add(&target_casted[target_index],
                             static_cast<DeviceTarget>(source.element<DeviceSource>(source_index)));
  }
};

template <typename Source>
  requires(cudf::detail::is_product_supported<Source>())
struct update_target_element_shmem<Source, cudf::aggregation::SUM_OF_SQUARES> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::SUM_OF_SQUARES>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto value            = static_cast<Target>(source.element<Source>(source_index));
    cudf::detail::atomic_add(&target_casted[target_index], value * value);
  }
};

template <typename Source>
  requires(cudf::detail::is_product_supported<Source>())
struct update_target_element_shmem<Source, cudf::aggregation::PRODUCT> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::PRODUCT>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_mul(&target_casted[target_index],
                             static_cast<Target>(source.element<Source>(source_index)));
  }
};

template <typename Source>
  requires(cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_VALID>())
struct update_target_element_shmem<Source, cudf::aggregation::COUNT_VALID> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    // The nullability was checked prior to this call in the `shmem_element_aggregator` functor
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_VALID>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_add(&target_casted[target_index], Target{1});
  }
};

template <typename Source>
  requires(cudf::detail::is_valid_aggregation<Source, cudf::aggregation::COUNT_ALL>())
struct update_target_element_shmem<Source, cudf::aggregation::COUNT_ALL> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::COUNT_ALL>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    cudf::detail::atomic_add(&target_casted[target_index], Target{1});
  }
};

template <typename Source>
  requires(cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMAX>() &&
           cudf::is_relationally_comparable<Source, Source>())
struct update_target_element_shmem<Source, cudf::aggregation::ARGMAX> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMAX>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto old              = cudf::detail::atomic_cas(
      &target_casted[target_index], cudf::detail::ARGMAX_SENTINEL, source_index);
    if (old != cudf::detail::ARGMAX_SENTINEL) {
      while (source.element<Source>(source_index) > source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target_casted[target_index], old, source_index);
      }
    }
  }
};

template <typename Source>
  requires(cudf::detail::is_valid_aggregation<Source, cudf::aggregation::ARGMIN>() &&
           cudf::is_relationally_comparable<Source, Source>())
struct update_target_element_shmem<Source, cudf::aggregation::ARGMIN> {
  __device__ void operator()(cuda::std::byte* target,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    using Target          = cudf::detail::target_type_t<Source, cudf::aggregation::ARGMIN>;
    Target* target_casted = reinterpret_cast<Target*>(target);
    auto old              = cudf::detail::atomic_cas(
      &target_casted[target_index], cudf::detail::ARGMIN_SENTINEL, source_index);
    if (old != cudf::detail::ARGMIN_SENTINEL) {
      while (source.element<Source>(source_index) < source.element<Source>(old)) {
        old = cudf::detail::atomic_cas(&target_casted[target_index], old, source_index);
      }
    }
  }
};

/**
 * @brief A functor that updates a single element in the target column stored in shared memory by
 * applying an aggregation operation to a corresponding element from a source column in global
 * memory.
 *
 * This functor can NOT be used for dictionary columns.
 *
 * This is a redundant copy replicating the behavior of `elementwise_aggregator` from
 * `cudf/detail/aggregation/device_aggregators.cuh`. The key difference is that this functor accepts
 * a pointer to raw bytes as the source, as `column_device_view` cannot yet be constructed from
 * shared memory.
 */
struct shmem_element_aggregator {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type target_index,
                             cudf::column_device_view source,
                             cudf::size_type source_index) const noexcept
  {
    // Check nullability for all aggregation kinds but `COUNT_ALL`
    if constexpr (k != cudf::aggregation::COUNT_ALL) {
      if (source.is_null(source_index)) { return; }
    }

    // The output for COUNT_VALID and COUNT_ALL is initialized to be all valid
    if constexpr (!(k == cudf::aggregation::COUNT_VALID or k == cudf::aggregation::COUNT_ALL)) {
      if (!target_mask[target_index]) {
        cudf::detail::atomic_max(target_mask + target_index, true);
      }
    }

    update_target_element_shmem<Source, k>{}(target, target_index, source, source_index);
  }
};
}  // namespace cudf::groupby::detail::hash
