/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/uninitialized_fill.h>

#include <memory>
#include <optional>
#include <utility>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Probes the equality hash table for the given join kind.
 *
 * The hash table is built on the right equality table and probed with the left equality table,
 * yielding the index pairs that the conditional predicate is subsequently applied to.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
equality_join_indices(cudf::hash_join const& hash_joiner,
                      table_view const& left_equality,
                      join_kind join_type,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  switch (join_type) {
    case join_kind::INNER_JOIN: return hash_joiner.inner_join(left_equality, {}, stream, mr);
    case join_kind::LEFT_JOIN: return hash_joiner.left_join(left_equality, {}, stream, mr);
    case join_kind::FULL_JOIN: return hash_joiner.full_join(left_equality, {}, stream, mr);
    default: CUDF_FAIL("Invalid join kind.");
  }
}

}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_join(table_view const& left_equality,
           table_view const& right_equality,
           table_view const& left_conditional,
           table_view const& right_conditional,
           ast::expression const& binary_predicate,
           null_equality compare_nulls,
           join_kind join_type,
           output_size_data_type const& output_size_data,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
               "Left semi and anti joins should use mixed_join_semi.");
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  // hash_join requires a non-empty build (right) table.
  if (right_conditional.num_rows() == 0) {
    switch (join_type) {
      case join_kind::LEFT_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left_conditional, stream, mr);
      case join_kind::INNER_JOIN:
        return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
      default: CUDF_FAIL("Invalid join kind.");
    }
  }

  // A full join is a left join plus the unmatched-right complement. Build the left-outer result and
  // append the complement with finalize_full_join rather than splitting failed pairs, which would
  // emit spurious unmatched rows for keys that also match elsewhere.
  if (join_type == join_kind::FULL_JOIN) {
    auto left_outer = mixed_join(left_equality,
                                 right_equality,
                                 left_conditional,
                                 right_conditional,
                                 binary_predicate,
                                 compare_nulls,
                                 join_kind::LEFT_JOIN,
                                 std::nullopt,
                                 stream,
                                 mr);
    return finalize_full_join(
      std::move(left_outer), left_conditional.num_rows(), right_conditional.num_rows(), stream, mr);
  }

  auto const hash_joiner = cudf::hash_join{right_equality, compare_nulls, stream};
  auto const [left_indices, right_indices] =
    equality_join_indices(hash_joiner, left_equality, join_type, stream, mr);

  auto const output_size = output_size_data.has_value()
                             ? std::optional<std::size_t>{output_size_data->first}
                             : std::nullopt;

  return detail::filter_join_indices(left_conditional,
                                     right_conditional,
                                     *left_indices,
                                     *right_indices,
                                     binary_predicate,
                                     join_type,
                                     output_size,
                                     stream,
                                     mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
compute_mixed_join_output_size(table_view const& left_equality,
                               table_view const& right_equality,
                               table_view const& left_conditional,
                               table_view const& right_conditional,
                               ast::expression const& binary_predicate,
                               null_equality compare_nulls,
                               join_kind join_type,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(join_type != join_kind::FULL_JOIN,
               "Size estimation is not available for full joins.");
  CUDF_EXPECTS(
    (join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
    "Left semi and anti join size estimation should use compute_mixed_join_output_size_semi.");
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  // hash_join requires a non-empty build (right) table.
  if (right_conditional.num_rows() == 0) {
    auto const left_num_rows = left_conditional.num_rows();
    if (join_type == join_kind::LEFT_JOIN) {
      auto counts =
        rmm::device_uvector<size_type>(static_cast<std::size_t>(left_num_rows), stream, mr);
      thrust::uninitialized_fill(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        counts.begin(),
        counts.end(),
        size_type{1});
      return {static_cast<std::size_t>(left_num_rows),
              std::make_unique<rmm::device_uvector<size_type>>(std::move(counts))};
    }
    return {0, std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  }

  auto const hash_joiner = cudf::hash_join{right_equality, compare_nulls, stream};
  auto const [left_indices, right_indices] =
    equality_join_indices(hash_joiner, left_equality, join_type, stream, mr);

  return cudf::filter_join_indices_output_size(left_conditional,
                                               right_conditional,
                                               *left_indices,
                                               *right_indices,
                                               binary_predicate,
                                               join_type,
                                               stream,
                                               mr);
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const output_size_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            join_kind::INNER_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_inner_join_size(
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
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::INNER_JOIN,
                                                stream,
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_left_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                output_size_data_type const output_size_data,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            join_kind::LEFT_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_join_size(
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
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                join_kind::LEFT_JOIN,
                                                stream,
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                output_size_data_type const output_size_data,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            join_kind::FULL_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

}  // namespace cudf
