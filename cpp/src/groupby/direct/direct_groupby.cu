/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/utils.hpp"
#include "groupby/hash/extract_single_pass_aggs.hpp"
#include "groupby/hash/global_memory_aggregator.cuh"
#include "groupby/hash/hash_compound_agg_finalizer.hpp"
#include "groupby/hash/output_utils.hpp"
#include "groupby/hash/shared_memory_aggregator.cuh"
#include "groupby/hash/single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/groupby/direct_groupby.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/groupby/direct_groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::detail {
namespace {

using hash::gmem_element_aggregator;
using hash::initialize_shmem;
using hash::shmem_element_aggregator;

constexpr size_type DIRECT_BLOCK_SIZE    = 256;
constexpr size_type SHMEM_ALIGNMENT      = 16;
constexpr size_type MAX_REPLICAS         = 32;
constexpr size_type REPLICA_TARGET_SLOTS = 1024;

bool is_supported_aggregation(aggregation::Kind kind)
{
  switch (kind) {
    case aggregation::SUM:
    case aggregation::MIN:
    case aggregation::MAX:
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL:
    case aggregation::MEAN: return true;
    default: return false;
  }
}

// Restricts shared memory kernel instantiations to the simple aggregations reachable from the
// supported public aggregations and to non-nested, non-dictionary element types
struct unsupported_shared_memory_type {};

template <cudf::type_id Id>
struct dispatch_shared_memory_type {
  using type = cuda::std::conditional_t<Id == cudf::type_id::DICTIONARY32 or
                                          Id == cudf::type_id::LIST or
                                          Id == cudf::type_id::STRUCT or
                                          Id == cudf::type_id::STRING,
                                        unsupported_shared_memory_type,
                                        cudf::id_to_type<Id>>;
};

template <typename Element>
struct dispatch_shared_memory_aggregation_fn {
  template <cudf::aggregation::Kind kind, typename F, typename... Ts>
  __device__ auto operator()(F&& f, Ts&&... args) const
  {
    return f.template operator()<Element, kind>(cuda::std::forward<Ts>(args)...);
  }
};

struct dispatch_shared_memory_source_fn {
  template <typename Element, typename F, typename... Ts>
  __device__ auto operator()(cudf::aggregation::Kind kind, F&& f, Ts&&... args) const
  {
    if constexpr (cuda::std::is_same_v<Element, unsupported_shared_memory_type>) {
      CUDF_UNREACHABLE("Unsupported shared memory aggregation type.");
    } else {
      auto const dispatch = dispatch_shared_memory_aggregation_fn<Element>{};
      switch (kind) {
        case cudf::aggregation::SUM:
          return dispatch.template operator()<cudf::aggregation::SUM>(
            cuda::std::forward<F>(f), cuda::std::forward<Ts>(args)...);
        case cudf::aggregation::MIN:
          return dispatch.template operator()<cudf::aggregation::MIN>(
            cuda::std::forward<F>(f), cuda::std::forward<Ts>(args)...);
        case cudf::aggregation::MAX:
          return dispatch.template operator()<cudf::aggregation::MAX>(
            cuda::std::forward<F>(f), cuda::std::forward<Ts>(args)...);
        case cudf::aggregation::COUNT_VALID:
          return dispatch.template operator()<cudf::aggregation::COUNT_VALID>(
            cuda::std::forward<F>(f), cuda::std::forward<Ts>(args)...);
        case cudf::aggregation::COUNT_ALL:
          return dispatch.template operator()<cudf::aggregation::COUNT_ALL>(
            cuda::std::forward<F>(f), cuda::std::forward<Ts>(args)...);
        default: CUDF_UNREACHABLE("Unsupported shared memory aggregation.");
      }
    }
  }
};

template <typename F, typename... Ts>
__device__ auto dispatch_shared_memory_type_and_aggregation(cudf::data_type type,
                                                            cudf::aggregation::Kind kind,
                                                            F&& f,
                                                            Ts&&... args)
{
  return cudf::type_dispatcher<dispatch_shared_memory_type>(type,
                                                            dispatch_shared_memory_source_fn{},
                                                            kind,
                                                            cuda::std::forward<F>(f),
                                                            cuda::std::forward<Ts>(args)...);
}

// Combines replicas `1..replicas-1` of a shared memory slot into replica 0
struct fold_replicas_fn {
  template <typename Source, cudf::aggregation::Kind k>
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             size_type slot,
                             size_type capacity,
                             size_type replicas) const
  {
    using Target = cudf::detail::target_type_t<Source, k>;
    if constexpr (cuda::std::is_void_v<Target>) {
      CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
    } else {
      using DeviceTarget = cudf::device_storage_type_t<Target>;
      auto* const values = reinterpret_cast<DeviceTarget*>(target);
      auto acc           = values[slot];
      auto valid         = target_mask[slot];
      for (size_type r = 1; r < replicas; ++r) {
        auto const idx = slot + r * capacity;
        valid          = valid or target_mask[idx];
        if constexpr (k == cudf::aggregation::MIN) {
          acc = values[idx] < acc ? values[idx] : acc;
        } else if constexpr (k == cudf::aggregation::MAX) {
          acc = acc < values[idx] ? values[idx] : acc;
        } else {
          acc = acc + values[idx];
        }
      }
      values[slot]      = acc;
      target_mask[slot] = valid;
    }
  }
};

// Each block accumulates `replicas` private copies of every slot in shared memory, folds them
// together, then merges the result into the global slots. Threads are spread across replicas to
// reduce same-address contention when the capacity is small.
CUDF_KERNEL void __launch_bounds__(DIRECT_BLOCK_SIZE)
  direct_shmem_aggs_kernel(std::uint32_t const* keys,
                           size_type num_rows,
                           size_type capacity,
                           size_type replicas,
                           table_device_view input_values,
                           mutable_table_device_view output_values,
                           aggregation::Kind const* agg_kinds,
                           size_type const* res_offsets,
                           size_type const* mask_offsets,
                           size_type occupancy_offset,
                           bool* occupied)
{
  extern __shared__ cuda::std::byte shmem[];

  auto const num_cols      = input_values.num_columns();
  auto const num_slots     = capacity * replicas;
  auto* const shmem_occupy = reinterpret_cast<bool*>(shmem + occupancy_offset);

  for (size_type col = 0; col < num_cols; ++col) {
    auto* const target      = shmem + res_offsets[col];
    auto* const target_mask = reinterpret_cast<bool*>(shmem + mask_offsets[col]);
    for (size_type idx = threadIdx.x; idx < num_slots; idx += blockDim.x) {
      dispatch_shared_memory_type_and_aggregation(
        output_values.column(col).type(), agg_kinds[col], initialize_shmem{}, target, target_mask, idx);
    }
  }
  for (size_type idx = threadIdx.x; idx < capacity; idx += blockDim.x) {
    shmem_occupy[idx] = false;
  }
  __syncthreads();

  auto const replica_offset = static_cast<size_type>(threadIdx.x % replicas) * capacity;
  auto const stride         = cudf::detail::grid_1d::grid_stride();
  for (auto row = cudf::detail::grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const source_idx = static_cast<size_type>(row);
    auto const key        = static_cast<size_type>(keys[source_idx]);
    shmem_occupy[key]     = true;
    auto const slot       = key + replica_offset;
    for (size_type col = 0; col < num_cols; ++col) {
      auto const source_col = input_values.column(col);
      dispatch_shared_memory_type_and_aggregation(source_col.type(),
                                                  agg_kinds[col],
                                                  shmem_element_aggregator{},
                                                  shmem + res_offsets[col],
                                                  reinterpret_cast<bool*>(shmem + mask_offsets[col]),
                                                  slot,
                                                  source_col,
                                                  source_idx);
    }
  }
  __syncthreads();

  for (size_type key = threadIdx.x; key < capacity; key += blockDim.x) {
    if (not shmem_occupy[key]) { continue; }
    for (size_type col = 0; col < num_cols; ++col) {
      auto const type   = input_values.column(col).type();
      auto* const res   = shmem + res_offsets[col];
      auto* const mask  = reinterpret_cast<bool*>(shmem + mask_offsets[col]);
      if (replicas > 1) {
        dispatch_shared_memory_type_and_aggregation(
          type, agg_kinds[col], fold_replicas_fn{}, res, mask, key, capacity, replicas);
      }
      dispatch_shared_memory_type_and_aggregation(type,
                                                  agg_kinds[col],
                                                  gmem_element_aggregator{},
                                                  output_values.column(col),
                                                  key,
                                                  input_values.column(col),
                                                  res,
                                                  mask,
                                                  key);
    }
    if (not occupied[key]) { occupied[key] = true; }
  }
}

struct direct_global_aggs_fn {
  std::uint32_t const* keys;
  bool* occupied;
  aggregation::Kind const* agg_kinds;
  table_device_view input_values;
  mutable_table_device_view output_values;

  __device__ void operator()(size_type row) const
  {
    auto const key = static_cast<size_type>(keys[row]);
    if (not occupied[key]) { occupied[key] = true; }
    for (size_type col = 0; col < input_values.num_columns(); ++col) {
      auto const source_col = input_values.column(col);
      cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                  agg_kinds[col],
                                                  cudf::detail::element_aggregator{},
                                                  output_values.column(col),
                                                  key,
                                                  source_col,
                                                  row);
    }
  }
};

struct shmem_plan {
  size_type replicas{0};
  size_type total_bytes{0};
  size_type occupancy_offset{0};
  std::vector<size_type> res_offsets;
  std::vector<size_type> mask_offsets;
};

shmem_plan make_shmem_plan(table_view const& results, size_type capacity, size_type replicas)
{
  shmem_plan plan;
  plan.replicas         = replicas;
  auto const num_slots  = static_cast<int64_t>(capacity) * replicas;
  auto const align      = [](int64_t bytes) {
    return cudf::util::round_up_safe(bytes, int64_t{SHMEM_ALIGNMENT});
  };
  int64_t bytes         = 0;
  for (auto const& col : results) {
    plan.res_offsets.push_back(static_cast<size_type>(bytes));
    bytes += align(static_cast<int64_t>(cudf::size_of(col.type())) * num_slots);
    plan.mask_offsets.push_back(static_cast<size_type>(bytes));
    bytes += align(num_slots);
  }
  plan.occupancy_offset = static_cast<size_type>(bytes);
  bytes += align(capacity);
  plan.total_bytes = static_cast<size_type>(std::min<int64_t>(bytes, std::numeric_limits<size_type>::max()));
  return plan;
}

bool is_shared_memory_compatible(table_view const& values,
                                 host_span<aggregation::Kind const> agg_kinds)
{
  for (std::size_t i = 0; i < agg_kinds.size(); ++i) {
    auto const type = values.column(i).type();
    if (is_dictionary(type) or not is_fixed_width(type)) { return false; }
    switch (agg_kinds[i]) {
      case aggregation::SUM:
      case aggregation::MIN:
      case aggregation::MAX:
      case aggregation::COUNT_VALID:
      case aggregation::COUNT_ALL: break;
      default: return false;
    }
  }
  return true;
}

int max_optin_shared_memory()
{
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  int bytes{};
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  return bytes;
}

// Returns the launch plan and grid size for the shared memory kernel, or a zero grid size if the
// slots do not fit in shared memory
std::pair<shmem_plan, size_type> plan_shared_memory_launch(table_view const& results,
                                                           size_type capacity,
                                                           size_type num_rows)
{
  auto const max_bytes = max_optin_shared_memory();
  auto replicas        = std::clamp(REPLICA_TARGET_SLOTS / std::max(capacity, 1), 1, MAX_REPLICAS);
  auto plan            = make_shmem_plan(results, capacity, replicas);
  while (plan.total_bytes > max_bytes and replicas > 1) {
    replicas /= 2;
    plan = make_shmem_plan(results, capacity, replicas);
  }
  if (plan.total_bytes > max_bytes) { return {std::move(plan), 0}; }

  CUDF_CUDA_TRY(cudaFuncSetAttribute(
    direct_shmem_aggs_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, plan.total_bytes));
  int blocks_per_sm{};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, direct_shmem_aggs_kernel, DIRECT_BLOCK_SIZE, plan.total_bytes));
  auto const max_grid   = blocks_per_sm * cudf::detail::num_multiprocessors();
  auto const num_blocks = cudf::util::div_rounding_up_safe(num_rows, DIRECT_BLOCK_SIZE);
  return {std::move(plan), std::min(max_grid, num_blocks)};
}

void run_shared_memory_aggs(column_view const& keys,
                            table_view const& values,
                            mutable_table_view results,
                            shmem_plan const& plan,
                            size_type grid_size,
                            size_type capacity,
                            aggregation::Kind const* d_agg_kinds,
                            bool* occupied,
                            cuda::stream_ref stream)
{
  auto offsets = std::vector<size_type>(plan.res_offsets);
  offsets.insert(offsets.end(), plan.mask_offsets.begin(), plan.mask_offsets.end());
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, stream, cudf::get_current_device_resource_ref());
  auto const num_cols = values.num_columns();

  auto const d_values  = table_device_view::create(values, stream);
  auto const d_results = mutable_table_device_view::create(results, stream);
  direct_shmem_aggs_kernel<<<grid_size, DIRECT_BLOCK_SIZE, plan.total_bytes, stream.get()>>>(
    keys.begin<std::uint32_t>(),
    keys.size(),
    capacity,
    plan.replicas,
    *d_values,
    *d_results,
    d_agg_kinds,
    d_offsets.data(),
    d_offsets.data() + num_cols,
    plan.occupancy_offset,
    occupied);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void run_global_memory_aggs(column_view const& keys,
                            table_view const& values,
                            mutable_table_view results,
                            aggregation::Kind const* d_agg_kinds,
                            bool* occupied,
                            cuda::stream_ref stream)
{
  auto const d_values  = table_device_view::create(values, stream);
  auto const d_results = mutable_table_device_view::create(results, stream);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    keys.size(),
    direct_global_aggs_fn{keys.begin<std::uint32_t>(), occupied, d_agg_kinds, *d_values, *d_results});
}

}  // namespace

std::pair<std::unique_ptr<column>, std::vector<aggregation_result>> direct_aggregate(
  column_view const& keys,
  std::span<aggregation_request const> requests,
  std::size_t capacity,
  direct_aggregate_path path,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(keys.type().id() == type_id::UINT32,
               "direct_aggregate keys must be of type UINT32",
               cudf::data_type_error);
  CUDF_EXPECTS(not keys.has_nulls(), "direct_aggregate keys must not contain nulls", std::invalid_argument);
  CUDF_EXPECTS(capacity <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "capacity must not exceed the maximum value of size_type",
               std::invalid_argument);
  for (auto const& request : requests) {
    CUDF_EXPECTS(request.values.size() == keys.size(),
                 "Size mismatch between request values and groupby keys.",
                 std::invalid_argument);
    CUDF_EXPECTS(is_fixed_width(request.values.type()),
                 "direct_aggregate values must be fixed-width",
                 std::invalid_argument);
    for (auto const& agg : request.aggregations) {
      CUDF_EXPECTS(is_supported_aggregation(agg->kind),
                   "Unsupported aggregation for direct_aggregate",
                   std::invalid_argument);
    }
  }
  CUDF_EXPECTS(hash::can_use_hash_groupby(requests),
               "Unsupported value type and aggregation combination for direct_aggregate",
               std::invalid_argument);

  if (keys.is_empty()) {
    auto [unique_keys, results] = cudf::groupby::groupby{table_view{{keys}}}.aggregate(requests, stream, mr);
    return {std::move(unique_keys->release().front()), std::move(results)};
  }

  auto const num_slots = static_cast<size_type>(capacity);
  auto const num_rows  = keys.size();

  cudf::detail::result_cache cache(requests.size());
  auto const single_pass_aggs   = hash::extract_single_pass_aggs(requests, stream);
  auto const& values              = std::get<0>(single_pass_aggs);
  auto const& agg_kinds           = std::get<1>(single_pass_aggs);
  auto const& aggs                = std::get<2>(single_pass_aggs);
  auto const& is_agg_intermediate = std::get<3>(single_pass_aggs);
  auto const has_compound_aggs    = std::get<4>(single_pass_aggs);
  auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, cudf::get_current_device_resource_ref());

  auto dense_results =
    hash::create_results_table(num_slots, values, agg_kinds, is_agg_intermediate, stream, mr);

  rmm::device_uvector<bool> occupied(num_slots, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(occupied.data(), 0, occupied.size(), stream.get()));

  auto const use_shared = [&]() -> std::pair<shmem_plan, size_type> {
    if (path == direct_aggregate_path::GLOBAL_MEMORY or
        not is_shared_memory_compatible(values, agg_kinds)) {
      CUDF_EXPECTS(path != direct_aggregate_path::SHARED_MEMORY,
                   "Aggregations are not compatible with the shared memory path",
                   std::invalid_argument);
      return {shmem_plan{}, 0};
    }
    auto [plan, grid_size] = plan_shared_memory_launch(dense_results->view(), num_slots, num_rows);
    if (path == direct_aggregate_path::SHARED_MEMORY) {
      CUDF_EXPECTS(grid_size > 0, "Capacity does not fit in shared memory", std::invalid_argument);
      return {std::move(plan), grid_size};
    }
    // Merging the per-block slots costs one global update per slot per block, so only use shared
    // memory when that is small relative to the input
    auto const flush_updates = static_cast<int64_t>(grid_size) * num_slots;
    if (grid_size == 0 or flush_updates > num_rows) { return {shmem_plan{}, 0}; }
    return {std::move(plan), grid_size};
  }();

  if (use_shared.second > 0) {
    run_shared_memory_aggs(keys,
                           values,
                           dense_results->mutable_view(),
                           use_shared.first,
                           use_shared.second,
                           num_slots,
                           d_agg_kinds.data(),
                           occupied.data(),
                           stream);
  } else {
    run_global_memory_aggs(
      keys, values, dense_results->mutable_view(), d_agg_kinds.data(), occupied.data(), stream);
  }

  rmm::device_uvector<std::uint32_t> unique_keys(num_slots, stream, mr);
  auto const keys_end = cudf::detail::copy_if(cuda::counting_iterator<std::uint32_t>{0},
                                              cuda::counting_iterator<std::uint32_t>{
                                                static_cast<std::uint32_t>(num_slots)},
                                              occupied.begin(),
                                              unique_keys.begin(),
                                              cuda::std::identity{},
                                              stream);
  auto const num_groups = static_cast<size_type>(cuda::std::distance(unique_keys.begin(), keys_end));
  unique_keys.resize(num_groups, stream);

  if (num_groups < num_slots) {
    auto const gather_map = device_span<size_type const>{
      reinterpret_cast<size_type const*>(unique_keys.data()), static_cast<std::size_t>(num_groups)};
    dense_results = cudf::detail::gather(dense_results->view(),
                                         gather_map,
                                         out_of_bounds_policy::DONT_CHECK,
                                         cudf::negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
  }
  hash::finalize_output(values, aggs, dense_results, &cache, stream);

  if (has_compound_aggs) {
    for (auto const& request : requests) {
      auto const finalizer =
        hash::hash_compound_agg_finalizer(request.values, &cache, nullptr, stream, mr);
      for (auto&& agg : request.aggregations) {
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }

  auto key_column = std::make_unique<column>(std::move(unique_keys), rmm::device_buffer{}, 0);
  return {std::move(key_column), extract_results(requests, cache, stream, mr)};
}

}  // namespace cudf::groupby::detail

namespace cudf::groupby {

std::pair<std::unique_ptr<column>, std::vector<aggregation_result>> direct_aggregate(
  column_view const& keys,
  std::span<aggregation_request const> requests,
  std::size_t capacity,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::direct_aggregate(
    keys, requests, capacity, detail::direct_aggregate_path::AUTO, stream, mr);
}

}  // namespace cudf::groupby
