/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"
#include "join/join_common_utils.hpp"
#include "size_impl.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>

namespace cudf::detail {

template <join_kind Join>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
probe_join_hash_table(
  cudf::table_view const& right_table,
  cudf::table_view const& left_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(Join == join_kind::INNER_JOIN || Join == join_kind::LEFT_JOIN ||
                Join == join_kind::FULL_JOIN);

  constexpr auto size_join = Join == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : Join;

  std::size_t const join_size = output_size
                                  ? *output_size
                                  : compute_join_output_size<size_join>(right_table,
                                                                        left_table,
                                                                        preprocessed_right,
                                                                        preprocessed_left,
                                                                        hash_table,
                                                                        has_nulls,
                                                                        compare_nulls,
                                                                        stream);

  if (join_size == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  cudf::prefetch::detail::prefetch(*left_indices, stream);
  cudf::prefetch::detail::prefetch(*right_indices, stream);

  auto const left_table_num_rows = left_table.num_rows();
  auto const out_probe_begin =
    cuda::make_transform_output_iterator(left_indices->begin(), output_fn{});
  auto const out_build_begin =
    cuda::make_transform_output_iterator(right_indices->begin(), output_fn{});

  auto retrieve_results = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    if constexpr (Join == join_kind::INNER_JOIN) {
      hash_table.retrieve(iter,
                          iter + left_table_num_rows,
                          equality,
                          hash_table.hash_function(),
                          out_probe_begin,
                          out_build_begin,
                          stream.value());
    } else {
      [[maybe_unused]] auto out_probe_end = hash_table
                                              .retrieve_outer(iter,
                                                              iter + left_table_num_rows,
                                                              equality,
                                                              hash_table.hash_function(),
                                                              out_probe_begin,
                                                              out_build_begin,
                                                              stream.value())
                                              .first;

      if constexpr (Join == join_kind::FULL_JOIN) {
        auto const actual_size = cuda::std::distance(out_probe_begin, out_probe_end);
        left_indices->resize(actual_size, stream);
        right_indices->resize(actual_size, stream);
      }
    }
  };

  dispatch_join_comparator(right_table,
                           left_table,
                           preprocessed_right,
                           preprocessed_left,
                           has_nulls,
                           compare_nulls,
                           retrieve_results);

  return std::pair(std::move(left_indices), std::move(right_indices));
}

template <typename RightOutputIterator>
void retrieve_left_join_build_indices(
  cudf::table_view const& right_table,
  cudf::table_view const& left_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  RightOutputIterator out_build_begin,
  rmm::cuda_stream_view stream)
{
  auto const left_table_num_rows = left_table.num_rows();

  auto retrieve_results = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    hash_table.retrieve_outer(iter,
                              iter + left_table_num_rows,
                              equality,
                              hash_table.hash_function(),
                              cuda::make_discard_iterator(),
                              out_build_begin,
                              stream.value());
  };

  dispatch_join_comparator(right_table,
                           left_table,
                           preprocessed_right,
                           preprocessed_left,
                           has_nulls,
                           compare_nulls,
                           retrieve_results);
}

template <typename Hasher>
template <join_kind Join>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::join_retrieve(cudf::table_view const& left,
                                 std::optional<std::size_t> output_size,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  validate_hash_join_probe(_right, left, _has_nulls);

  if constexpr (Join == join_kind::INNER_JOIN) {
    if (is_trivial_join(left, _right, Join)) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  } else {
    if (_is_empty) { return get_trivial_left_join_indices(left, stream, mr); }

    if (is_trivial_join(left, _right, Join)) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  }

  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream);

  auto join_indices = cudf::detail::probe_join_hash_table<Join>(_right,
                                                                left,
                                                                _preprocessed_right,
                                                                preprocessed_left,
                                                                _impl->_hash_table,
                                                                _has_nulls,
                                                                _nulls_equal,
                                                                output_size,
                                                                stream,
                                                                mr);

  if constexpr (Join == join_kind::FULL_JOIN) {
    return detail::finalize_full_join(
      std::move(join_indices), left.num_rows(), _right.num_rows(), stream, mr);
  } else {
    return join_indices;
  }
}

}  // namespace cudf::detail
