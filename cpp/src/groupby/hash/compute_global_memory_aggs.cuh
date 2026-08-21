/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_global_memory_aggs.hpp"
#include "output_utils.hpp"
#include "single_pass_functors.cuh"

#include <cudf/detail/gather.hpp>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace cudf::groupby::detail::hash {

/**
 * @brief Compute and return an array mapping each input row to its corresponding key index in
 * the input keys table.
 *
 * @tparam SetType Type of the key hash set
 * @param row_bitmask Bitmask indicating which rows in the input keys table are valid
 * @param set_ref Key hash set
 * @param num_rows Number of rows in the input keys table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A device vector mapping each input row to its key index
 */
template <typename SetRef>
rmm::device_uvector<size_type> compute_matching_keys(bitmask_type const* row_bitmask,
                                                     SetRef set_ref,
                                                     size_type num_rows,
                                                     cuda::stream_ref stream)
{
  // Mapping from each row in the input key/value into the indices of the key.
  rmm::device_uvector<size_type> key_indices(num_rows, stream);

  // Need to set to sentinel value for rows that are null (if any).
  // The sentinel value will then be used to identify null rows instead of using the bitmask.
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(num_rows),
                    key_indices.begin(),
                    [set_ref, row_bitmask] __device__(size_type const idx) mutable {
                      if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                        return *set_ref.insert_and_find(idx).first;
                      }
                      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
                    });
  return key_indices;
}

/**
 * @brief Compute aggregations and write the results directly to a dense output table.
 *
 * The target indices in the dense output table are computed by firstly inserting all keys into the
 * hash set then extracting the unique keys and computing a transform map from the input key
 * indices to the output key indices. This incurs some overhead, but it allows us to avoid
 * allocating extra memory for the sparse table and gathering the results from the sparse table into
 * the final dense table.
 */
template <typename SetType>
std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>> compute_aggs_dense_output(
  bitmask_type const* row_bitmask,
  table_view const& values,
  SetType const& key_set,
  host_span<aggregation::Kind const> h_agg_kinds,
  device_span<aggregation::Kind const> d_agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows                = values.num_rows();
  auto [unique_keys, target_indices] = [&] {
    auto matching_keys =
      compute_matching_keys(row_bitmask, key_set.ref(cuco::op::insert_and_find), num_rows, stream);
    auto unique_keys       = extract_populated_keys(key_set, num_rows, stream, mr);
    auto key_transform_map = compute_key_transform_map(
      num_rows, unique_keys, stream, cudf::get_current_device_resource_ref());
    auto target_indices = compute_target_indices(
      matching_keys, key_transform_map, stream, cudf::get_current_device_resource_ref());
    return std::pair{std::move(unique_keys), std::move(target_indices)};
  }();

  auto const d_values = table_device_view::create(values, stream);
  auto agg_results    = create_results_table(static_cast<size_type>(unique_keys.size()),
                                          values,
                                          h_agg_kinds,
                                          is_agg_intermediate,
                                          stream,
                                          mr);
  auto d_results_ptr  = mutable_table_device_view::create(*agg_results, stream);

  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<int64_t>{0},
                     num_rows * static_cast<int64_t>(h_agg_kinds.size()),
                     compute_single_pass_aggs_dense_output_fn{
                       target_indices.begin(), d_agg_kinds.data(), *d_values, *d_results_ptr});

  return {std::move(agg_results), std::move(unique_keys)};
}

/**
 * @brief Compute aggregations and write the results to a sparse intermediate output table, then
 * generate the final output table by gathering the relevant rows from it.
 *
 * During computing the aggregations, we write the results to a sparse intermediate output table
 * using the target indices as indices of the input keys. Such input key indices are computed by
 * inserting keys into the hash set on-the-fly.
 */
template <typename SetType>
std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>> compute_aggs_sparse_output_gather(
  bitmask_type const* row_bitmask,
  table_view const& values,
  SetType const& key_set,
  host_span<aggregation::Kind const> h_agg_kinds,
  device_span<aggregation::Kind const> d_agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = values.num_rows();
  auto const d_values = table_device_view::create(values, stream);
  auto agg_results =
    create_results_table(num_rows, values, h_agg_kinds, is_agg_intermediate, stream, mr);
  auto d_results_ptr = mutable_table_device_view::create(*agg_results, stream);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    num_rows,
    compute_single_pass_aggs_sparse_output_fn{key_set.ref(cuco::op::insert_and_find),
                                              row_bitmask,
                                              d_agg_kinds.data(),
                                              *d_values,
                                              *d_results_ptr});

  auto unique_keys   = extract_populated_keys(key_set, num_rows, stream, mr);
  auto dense_results = cudf::detail::gather(agg_results->view(),
                                            unique_keys,
                                            out_of_bounds_policy::DONT_CHECK,
                                            cudf::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr);
  return {std::move(dense_results), std::move(unique_keys)};
}

template <typename SetType>
std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>> compute_global_memory_aggs(
  bitmask_type const* row_bitmask,
  table_view const& values,
  SetType const& key_set,
  host_span<aggregation::Kind const> h_agg_kinds,
  device_span<aggregation::Kind const> d_agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return h_agg_kinds.size() > GROUPBY_DENSE_OUTPUT_THRESHOLD
           ? compute_aggs_dense_output(row_bitmask,
                                       values,
                                       key_set,
                                       h_agg_kinds,
                                       d_agg_kinds,
                                       is_agg_intermediate,
                                       stream,
                                       mr)
           : compute_aggs_sparse_output_gather(row_bitmask,
                                               values,
                                               key_set,
                                               h_agg_kinds,
                                               d_agg_kinds,
                                               is_agg_intermediate,
                                               stream,
                                               mr);
}

}  // namespace cudf::groupby::detail::hash
