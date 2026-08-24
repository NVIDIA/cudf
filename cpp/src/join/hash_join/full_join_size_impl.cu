/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "retrieve_impl.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>

namespace cudf::detail {

namespace {
std::size_t compute_left_join_complement_size(cudf::device_span<size_type const> right_matches,
                                              cuda::stream_ref stream)
{
  return cudf::detail::count_if(
    right_matches.begin(),
    right_matches.end(),
    [] __device__(size_type is_matched) -> bool { return is_matched == 0; },
    stream);
}
}  // namespace

std::size_t get_full_join_size(
  cudf::table_view const& right_table,
  cudf::table_view const& left_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::size_t join_size = compute_join_output_size<join_kind::LEFT_JOIN>(right_table,
                                                                         left_table,
                                                                         preprocessed_right,
                                                                         preprocessed_left,
                                                                         hash_table,
                                                                         has_nulls,
                                                                         compare_nulls,
                                                                         stream);

  if (join_size == 0) { return right_table.num_rows(); }

  auto right_matches =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(right_table.num_rows(), stream, mr);
  auto const out_build_begin = cuda::make_transform_output_iterator(
    cuda::make_discard_iterator(),
    mark_matched_output_fn{right_matches.data(), right_table.num_rows()});

  retrieve_left_join_build_indices(right_table,
                                   left_table,
                                   preprocessed_right,
                                   preprocessed_left,
                                   hash_table,
                                   has_nulls,
                                   compare_nulls,
                                   out_build_begin,
                                   stream);

  return join_size + compute_left_join_complement_size(right_matches, stream);
}

}  // namespace cudf::detail
