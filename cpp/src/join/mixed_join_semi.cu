/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_filter_join_common_utils.cuh"
#include "mixed_join_kernels_semi.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/fill.h>

#include <optional>

namespace cudf {
namespace detail {

std::unique_ptr<rmm::device_uvector<size_type>> mixed_join_semi(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((join_type != join_kind::INNER_JOIN) and (join_type != join_kind::LEFT_JOIN) and
                 (join_type != join_kind::FULL_JOIN),
               "Inner, left, and full joins should use mixed_join.");

  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  auto const right_num_rows{right_conditional.num_rows()};
  auto const left_num_rows{left_conditional.num_rows()};
  auto const outer_num_rows{left_num_rows};

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Anti and semi return all the row indices from left
      // with a corresponding NULL from the right.
      case join_kind::LEFT_ANTI_JOIN:
        return get_trivial_left_join_indices(left_conditional, stream, mr).first;
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Anti and semi joins both return empty sets.
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) or cudf::has_nulls(right_equality) or
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& left                  = left_equality;
  auto& right                 = right_equality;
  auto left_view              = table_device_view::create(left, stream);
  auto right_view             = table_device_view::create(right, stream);
  auto left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto right_conditional_view = table_device_view::create(right_conditional, stream);

  auto const preprocessed_right =
    cudf::detail::row::equality::preprocessed_table::create(right, stream);
  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_left, preprocessed_right};
  auto const equality_left = row_comparator.equal_to<false>(has_nulls, compare_nulls);

  // Create hash table containing all keys found in right table
  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const right_nulls    = cudf::nullate::DYNAMIC{cudf::has_nulls(right)};
  auto const row_hash_right = cudf::detail::row::hash::row_hasher{preprocessed_right};

  // Since we may see multiple rows that are identical in the equality tables
  // but differ in the conditional tables, the equality comparator used for
  // insertion must account for both sets of tables. An alternative solution
  // would be to use a multimap, but that solution would store duplicates where
  // equality and conditional rows are equal, so this approach is preferable.
  // One way to make this solution even more efficient would be to only include
  // the columns of the conditional table that are used by the expression, but
  // that requires additional plumbing through the AST machinery and is out of
  // scope for now.
  auto const row_comparator_right =
    cudf::detail::row::equality::two_table_comparator{preprocessed_right, preprocessed_right};
  auto const equality_right_equality =
    row_comparator_right.equal_to<false>(right_nulls, compare_nulls);
  auto const preprocessed_right_condtional =
    cudf::detail::row::equality::preprocessed_table::create(right_conditional, stream);
  auto const row_comparator_conditional_right = cudf::detail::row::equality::two_table_comparator{
    preprocessed_right_condtional, preprocessed_right_condtional};
  auto const equality_right_conditional =
    row_comparator_conditional_right.equal_to<false>(right_nulls, compare_nulls);

  hash_set_type row_set{{static_cast<std::size_t>(right.num_rows())},
                        cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                        cuco::empty_key{JoinNoMatch},
                        {equality_right_equality, equality_right_conditional},
                        {row_hash_right.device_hasher(right_nulls)},
                        {},
                        {},
                        rmm::mr::polymorphic_allocator<char>{},
                        {stream.value()}};

  auto iter = cuda::counting_iterator<cudf::size_type>{0};

  // skip rows that are null here.
  if ((compare_nulls == null_equality::EQUAL) or (not nullable(right))) {
    row_set.insert_async(iter, iter + right_num_rows, stream.value());
  } else {
    cuda::counting_iterator<cudf::size_type> stencil(0);
    auto const [row_bitmask, _] =
      cudf::detail::bitmask_and(right, stream, cudf::get_current_device_resource_ref());
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    row_set.insert_if_async(iter, iter + right_num_rows, stencil, pred, stream.value());
  }

  detail::grid_1d const config(outer_num_rows * hash_set_type::cg_size, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block =
    parser.shmem_per_thread *
    cuco::detail::int_div_ceil(config.num_threads_per_block, hash_set_type::cg_size);

  auto const row_hash  = cudf::detail::row::hash::row_hasher{preprocessed_left};
  auto const hash_left = row_hash.device_hasher(has_nulls);

  hash_set_ref_type const row_set_ref = row_set.ref(cuco::contains).rebind_hash_function(hash_left);

  // Vector used to indicate indices from the left table which are present in output
  auto left_table_keep_mask = rmm::device_uvector<bool>(left.num_rows(), stream);

  launch_mixed_join_semi(has_nulls,
                         *left_conditional_view,
                         *right_conditional_view,
                         *left_view,
                         *right_view,
                         equality_left,
                         row_set_ref,
                         cudf::device_span<bool>(left_table_keep_mask),
                         parser.device_expression_data,
                         config,
                         shmem_size_per_block,
                         stream);

  auto gather_map = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);

  // gather_map_end will be the end of valid data in gather_map
  auto gather_map_end = cudf::detail::copy_if(
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{left.num_rows()},
    left_table_keep_mask.begin(),
    gather_map->begin(),
    [join_type] __device__(bool keep_row) -> bool {
      return keep_row == (join_type == join_kind::LEFT_SEMI_JOIN);
    },
    stream);

  gather_map->resize(cuda::std::distance(gather_map->begin(), gather_map_end), stream);
  return gather_map;
}

}  // namespace detail

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_semi_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_SEMI_JOIN,
                                 stream,
                                 mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_anti_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join_semi(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_ANTI_JOIN,
                                 stream,
                                 mr);
}

}  // namespace cudf
