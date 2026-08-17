/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"
#include "join/join_common_utils.hpp"
#include "partitioned_retrieve_kernels.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace cudf::detail {
namespace {

/**
 * @brief Returns trivial left/right index pairs for an outer join when the build side is empty.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
make_trivial_outer_indices(size_type left_start_idx,
                           size_type partition_size,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
  auto out           = cuda::zip_iterator(left_indices->begin(), right_indices->begin());
  thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   out,
                   out + partition_size,
                   cuda::proclaim_return_type<cuda::std::tuple<size_type, size_type>>(
                     [left_start_idx] __device__(auto i) {
                       return cuda::std::tuple{static_cast<size_type>(left_start_idx + i),
                                               JoinNoMatch};
                     }));
  return std::pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_join_retrieve(join_kind join,
                                             cudf::join_partition_context const& context,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(
    join == join_kind::INNER_JOIN || join == join_kind::LEFT_JOIN || join == join_kind::FULL_JOIN,
    "Unsupported join kind for partitioned retrieve");

  CUDF_EXPECTS(context.left_table_context != nullptr,
               "join_partition_context is missing left_table_context",
               std::invalid_argument);

  auto const& match_ctx     = *context.left_table_context;
  auto const left_start_idx = context.left_start_idx;
  auto const left_end_idx   = context.left_end_idx;

  CUDF_EXPECTS(match_ctx._match_counts != nullptr,
               "join_match_context is missing match counts",
               std::invalid_argument);
  CUDF_EXPECTS(left_start_idx >= 0 && left_end_idx >= left_start_idx &&
                 left_end_idx <= match_ctx._left_table.num_rows(),
               "Invalid partition bounds",
               std::invalid_argument);

  // Empty partition
  if (left_start_idx >= left_end_idx) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto const partition_size = left_end_idx - left_start_idx;

  // Trivial case: build table is empty
  if (_is_empty) {
    if (join == join_kind::INNER_JOIN) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    } else {
      return make_trivial_outer_indices(left_start_idx, partition_size, stream, mr);
    }
  }

  // Slice the left table to the partition range
  auto const left_partition_view =
    cudf::slice(match_ctx._left_table, {left_start_idx, left_end_idx})[0];

  validate_hash_join_probe(_right, left_partition_view, _has_nulls);

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left_partition_view, stream, temp_mr);

  // For FULL_JOIN, probe with LEFT_JOIN semantics (no complement here)
  bool const is_outer = (join != join_kind::INNER_JOIN);

  // launch_partitioned_retrieve reduces match counts to compute output size
  // (total = last_offset + last_count), allocates output buffers, and launches the kernel.
  auto const* partition_counts = match_ctx._match_counts->data() + left_start_idx;
  auto const n                 = static_cast<thread_index_type>(partition_size);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
    join_indices;

  auto retrieve_partition = [&](auto equality, auto d_hasher) {
    // Precompute left keys for this partition slice.
    rmm::device_uvector<probe_key_type> left_keys(n, stream, temp_mr);
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(partition_size),
                      left_keys.begin(),
                      pair_fn{d_hasher});

    auto const ref = _impl->_hash_table.ref(cuco::op::count)
                       .rebind_key_eq(equality)
                       .rebind_hash_function(_impl->_hash_table.hash_function());

    if (is_outer) {
      join_indices = launch_partitioned_retrieve<true>(
        left_keys.data(), n, partition_counts, ref, left_start_idx, stream, mr);
    } else {
      join_indices = launch_partitioned_retrieve<false>(
        left_keys.data(), n, partition_counts, ref, left_start_idx, stream, mr);
    }
  };

  dispatch_join_comparator(_right,
                           left_partition_view,
                           _preprocessed_right,
                           preprocessed_left,
                           _has_nulls,
                           _nulls_equal,
                           retrieve_partition);

  return join_indices;
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::partitioned_join_retrieve(join_kind,
                                                       cudf::join_partition_context const&,
                                                       cuda::stream_ref,
                                                       rmm::device_async_resource_ref) const;

}  // namespace cudf::detail
