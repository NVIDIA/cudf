/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/utility>

namespace cudf::groupby::detail::streaming {

/// Shared memory data alignment
CUDF_HOST_DEVICE size_type constexpr SHMEM_ALIGNMENT = 16;

/**
 * @brief Checks whether a source type and aggregation kind can be aggregated in shared memory.
 *
 * Streaming groupby decomposes every request into these simple aggregation kinds, which are all
 * computed with atomics on the device storage type of the aggregation target type.
 */
template <typename Source, aggregation::Kind k>
constexpr bool is_shmem_aggregation_supported()
{
  if constexpr (!cudf::detail::is_valid_aggregation<Source, k>()) {
    return false;
  } else if constexpr (k == aggregation::MIN || k == aggregation::MAX) {
    return cudf::is_fixed_width<Source>() &&
           cudf::has_atomic_support<cudf::device_storage_type_t<Source>>();
  } else if constexpr (k == aggregation::SUM) {
    return cudf::is_fixed_width<Source>() && !cudf::is_timestamp<Source>();
  } else if constexpr (k == aggregation::SUM_OF_SQUARES || k == aggregation::PRODUCT) {
    return cudf::detail::is_product_supported<Source>();
  } else {
    return k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL;
  }
}

/// Compile-time constant of `is_shmem_aggregation_supported`, usable from device code
template <typename Source, aggregation::Kind k>
inline constexpr bool is_shmem_aggregation_supported_v =
  is_shmem_aggregation_supported<Source, k>();

/// Device storage type of the aggregation target for a source type and aggregation kind
template <typename Source, aggregation::Kind k>
using shmem_target_t = cudf::device_storage_type_t<cudf::detail::target_type_t<Source, k>>;

/// Atomically combines `value` into `*address` with the binary operator of aggregation `k`
template <aggregation::Kind k, typename Target>
__device__ void atomic_combine(Target* address, Target value)
{
  if constexpr (k == aggregation::MIN) {
    cudf::detail::atomic_min(address, value);
  } else if constexpr (k == aggregation::MAX) {
    cudf::detail::atomic_max(address, value);
  } else if constexpr (k == aggregation::PRODUCT) {
    cudf::detail::atomic_mul(address, value);
  } else {
    // SUM, SUM_OF_SQUARES, COUNT_VALID and COUNT_ALL
    cudf::detail::atomic_add(address, value);
  }
}

/**
 * @brief Invokes `f.operator()<Source, k>(args...)` for the runtime source type and aggregation
 * kind.
 *
 * Only the simple aggregation kinds produced by streaming groupby are dispatched.
 */
template <typename F, typename... Ts>
__device__ void dispatch_shmem_aggregation(data_type type,
                                           aggregation::Kind kind,
                                           F&& f,
                                           Ts&&... args)
{
  auto const dispatch_kind = [&]<aggregation::Kind k>() {
    cudf::type_dispatcher(type, [&]<typename Source>() {
      if constexpr (is_shmem_aggregation_supported_v<Source, k>) {
        f.template operator()<Source, k>(cuda::std::forward<Ts>(args)...);
      } else {
        CUDF_UNREACHABLE("Unsupported shared memory aggregation.");
      }
    });
  };
  switch (kind) {
    case aggregation::SUM: return dispatch_kind.template operator()<aggregation::SUM>();
    case aggregation::PRODUCT: return dispatch_kind.template operator()<aggregation::PRODUCT>();
    case aggregation::MIN: return dispatch_kind.template operator()<aggregation::MIN>();
    case aggregation::MAX: return dispatch_kind.template operator()<aggregation::MAX>();
    case aggregation::COUNT_VALID:
      return dispatch_kind.template operator()<aggregation::COUNT_VALID>();
    case aggregation::COUNT_ALL: return dispatch_kind.template operator()<aggregation::COUNT_ALL>();
    case aggregation::SUM_OF_SQUARES:
      return dispatch_kind.template operator()<aggregation::SUM_OF_SQUARES>();
    default: CUDF_UNREACHABLE("Unsupported shared memory aggregation.");
  }
}

/**
 * @brief Initializes a shared memory target element with the identity of the aggregation.
 *
 * COUNT_VALID and COUNT_ALL are always valid; other aggregations become valid on their first
 * valid source element.
 */
struct initialize_shmem_fn {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             size_type target_index) const noexcept
  {
    using Target                                    = shmem_target_t<Source, k>;
    reinterpret_cast<Target*>(target)[target_index] = cudf::detail::get_identity<Target, k>();
    target_mask[target_index] = k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL;
  }
};

/// Aggregates a global memory source element into a shared memory target element.
struct shmem_element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if constexpr (k != aggregation::COUNT_ALL) {
      if (source.is_null(source_index)) { return; }
    }
    if constexpr (k != aggregation::COUNT_VALID && k != aggregation::COUNT_ALL) {
      if (!target_mask[target_index]) {
        cudf::detail::atomic_max(target_mask + target_index, true);
      }
    }

    using Target        = shmem_target_t<Source, k>;
    auto* const address = reinterpret_cast<Target*>(target) + target_index;
    auto const value    = [&] {
      if constexpr (k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL) {
        return Target{1};
      } else {
        auto const v =
          static_cast<Target>(source.element<cudf::device_storage_type_t<Source>>(source_index));
        if constexpr (k == aggregation::SUM_OF_SQUARES) {
          return v * v;
        } else {
          return v;
        }
      }
    }();
    atomic_combine<k>(address, value);
  }
};

/**
 * @brief Merges a partial result in shared memory into the global memory result.
 *
 * Partial results that never saw a valid source element are skipped.
 */
struct gmem_element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             cuda::std::byte const* source,
                             bool const* source_mask,
                             size_type source_index) const noexcept
  {
    if (!source_mask[source_index]) { return; }
    if constexpr (k != aggregation::COUNT_VALID && k != aggregation::COUNT_ALL) {
      if (target.is_null(target_index)) { target.set_valid(target_index); }
    }

    using Target = shmem_target_t<Source, k>;
    atomic_combine<k>(&target.element<Target>(target_index),
                      reinterpret_cast<Target const*>(source)[source_index]);
  }
};

}  // namespace cudf::groupby::detail::streaming
