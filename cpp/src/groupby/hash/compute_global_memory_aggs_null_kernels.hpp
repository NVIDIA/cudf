/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "helpers.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace cudf::groupby::detail::hash {

using nullable_insert_and_find_ref = nullable_hash_set_ref_t<cuco::insert_and_find_tag>;

void launch_null_sparse_non_dictionary(nullable_insert_and_find_ref set_ref,
                                       bitmask_type const* row_bitmask,
                                       aggregation::Kind const* aggs,
                                       table_device_view const& input_values,
                                       mutable_table_device_view const& output_values,
                                       size_type num_rows,
                                       rmm::cuda_stream_view stream);

void launch_null_sparse_dictionary(nullable_insert_and_find_ref set_ref,
                                   bitmask_type const* row_bitmask,
                                   aggregation::Kind const* aggs,
                                   table_device_view const& input_values,
                                   mutable_table_device_view const& output_values,
                                   size_type num_rows,
                                   rmm::cuda_stream_view stream);

void launch_null_dense_non_dictionary(size_type const* target_indices,
                                      aggregation::Kind const* aggs,
                                      table_device_view const& input_values,
                                      mutable_table_device_view const& output_values,
                                      int64_t num_items,
                                      rmm::cuda_stream_view stream);

void launch_null_dense_dictionary(size_type const* target_indices,
                                  aggregation::Kind const* aggs,
                                  table_device_view const& input_values,
                                  mutable_table_device_view const& output_values,
                                  int64_t num_items,
                                  rmm::cuda_stream_view stream);

}  // namespace cudf::groupby::detail::hash
