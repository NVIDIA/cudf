/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/filter_join_indices/filter_join_indices_kernel.hpp"
#include "join/filter_join_indices/filter_join_indices_output_size_kernel.hpp"
#include "join/join_common_utils.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/dispatchers.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cub/cub.cuh>
#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <memory>
#include <optional>
#include <utility>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices(cudf::table_view const& left,
                    cudf::table_view const& right,
                    cudf::device_span<size_type const> left_indices,
                    cudf::device_span<size_type const> right_indices,
                    ast::expression const& predicate,
                    join_kind join_kind,
                    std::optional<std::size_t> output_size,
                    cuda::stream_ref stream,
                    rmm::device_async_resource_ref mr)
{
  // Validate inputs
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(join_kind == join_kind::INNER_JOIN || join_kind == join_kind::LEFT_JOIN ||
                 join_kind == join_kind::FULL_JOIN,
               "filter_join_indices only supports INNER_JOIN, LEFT_JOIN, and FULL_JOIN.",
               std::invalid_argument);

  auto make_empty_result = [&]() {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  };

  if (left_indices.empty()) { return make_empty_result(); }

  // Check if predicate may evaluate to null
  auto const has_nulls = predicate.may_evaluate_null(left, right, stream);

  // Create expression parser
  auto const parser = ast::detail::expression_parser{
    predicate, left, right, has_nulls, stream, cudf::get_current_device_resource_ref()};

  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The predicate expression must produce a Boolean output",
               std::invalid_argument);

  // Check if expression contains complex types
  auto const has_complex_type = parser.has_complex_type();

  // Create device views of tables
  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // Allocate array to store predicate evaluation results
  auto predicate_results = rmm::device_uvector<bool>(left_indices.size(), stream);

  // Configure kernel parameters with dynamic shared memory calculation
  int device_id;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));

  int shmem_limit_per_block;
  CUDF_CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

  auto const block_size =
    parser.shmem_per_thread != 0
      ? std::min(MAX_BLOCK_SIZE, shmem_limit_per_block / parser.shmem_per_thread)
      : MAX_BLOCK_SIZE;

  detail::grid_1d const config(left_indices.size(), block_size);
  auto const shmem_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Launch kernel with template dispatch based on nulls and complex types
  if (has_nulls && has_complex_type) {
    launch_filter_gather_map_kernel<true, true>(*left_table,
                                                *right_table,
                                                left_indices,
                                                right_indices,
                                                parser.device_expression_data,
                                                config,
                                                shmem_per_block,
                                                predicate_results.data(),
                                                stream);
  } else if (has_nulls && !has_complex_type) {
    launch_filter_gather_map_kernel<true, false>(*left_table,
                                                 *right_table,
                                                 left_indices,
                                                 right_indices,
                                                 parser.device_expression_data,
                                                 config,
                                                 shmem_per_block,
                                                 predicate_results.data(),
                                                 stream);
  } else if (!has_nulls && has_complex_type) {
    launch_filter_gather_map_kernel<false, true>(*left_table,
                                                 *right_table,
                                                 left_indices,
                                                 right_indices,
                                                 parser.device_expression_data,
                                                 config,
                                                 shmem_per_block,
                                                 predicate_results.data(),
                                                 stream);
  } else {
    launch_filter_gather_map_kernel<false, false>(*left_table,
                                                  *right_table,
                                                  left_indices,
                                                  right_indices,
                                                  parser.device_expression_data,
                                                  config,
                                                  shmem_per_block,
                                                  predicate_results.data(),
                                                  stream);
  }

  auto predicate_results_ptr = predicate_results.data();
  auto left_ptr              = left_indices.data();
  auto right_ptr             = right_indices.data();

  auto make_result_vectors = [&](std::size_t size) {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr)};
  };

  // Handle different join semantics
  if (join_kind == join_kind::INNER_JOIN) {
    // INNER_JOIN: only keep pairs that satisfy the predicate
    auto valid_predicate = [=] __device__(size_type i) -> bool { return predicate_results_ptr[i]; };

    auto const num_valid =
      output_size.has_value()
        ? *output_size
        : cudf::detail::count_if(
            cuda::counting_iterator<size_type>{0},
            cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
            valid_predicate,
            stream);

    if (num_valid == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(num_valid);

    auto input_iter =
      cuda::make_zip_iterator(cuda::std::tuple{left_indices.begin(), right_indices.begin()});
    auto output_iter = cuda::make_zip_iterator(
      cuda::std::tuple{filtered_left_indices->begin(), filtered_right_indices->begin()});

    cudf::detail::copy_if_async(
      input_iter,
      input_iter + left_indices.size(),
      cuda::counting_iterator<size_type>{0},
      output_iter,
      [valid_predicate] __device__(size_type idx) -> bool { return valid_predicate(idx); },
      stream);

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else if (join_kind == join_kind::LEFT_JOIN) {
    // LEFT_JOIN: Keep pairs that pass predicate, add one unmatched entry per left row with no
    // matches
    using SetType =
      cuco::static_set<size_type,
                       cuco::extent<std::size_t>,
                       cuda::thread_scope_device,
                       cuda::std::equal_to<size_type>,
                       cuco::double_hashing<1, cuco::default_hash_function<size_type>>,
                       rmm::mr::polymorphic_allocator<char>>;
    SetType filter_passing_indices{cuco::extent{static_cast<std::size_t>(left.num_rows())},
                                   cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                   cuco::empty_key{-1},
                                   {},
                                   {},
                                   {},
                                   {},
                                   {},
                                   stream.get()};

    auto predicate_func = [predicate_results_ptr] __device__(std::size_t idx) -> bool {
      return static_cast<bool>(predicate_results_ptr[idx]);
    };
    auto const num_filter_passing =
      filter_passing_indices.insert_if(left_ptr,
                                       left_ptr + left_indices.size(),
                                       cuda::counting_iterator<std::size_t>{0},
                                       predicate_func,
                                       stream.get());

    auto const num_invalid = left.num_rows() - num_filter_passing;

    auto const num_valid = [&]() -> std::size_t {
      if (output_size.has_value()) { return *output_size - num_invalid; }
      // CUB APIs are used instead of Thrust to enable 64-bit operations on index vectors of size
      // greater than integer limits
      cudf::detail::device_scalar<std::size_t> d_num_valid(stream,
                                                           cudf::get_current_device_resource_ref());
      auto const predicate_it =
        cuda::transform_iterator{predicate_results_ptr,
                                 cuda::proclaim_return_type<std::size_t>(
                                   [] __device__(auto val) -> std::size_t { return val ? 1 : 0; })};
      std::size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Sum(nullptr,
                             temp_storage_bytes,
                             predicate_it,
                             d_num_valid.data(),
                             left_indices.size(),
                             stream.get());
      rmm::device_buffer temp_storage(temp_storage_bytes, stream);
      cub::DeviceReduce::Sum(temp_storage.data(),
                             temp_storage_bytes,
                             predicate_it,
                             d_num_valid.data(),
                             left_indices.size(),
                             stream.get());
      return d_num_valid.value(stream);
    }();
    auto const result_size = num_valid + num_invalid;
    if (result_size == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(result_size);
    if (num_valid > 0) {
      auto input_iter =
        cuda::make_zip_iterator(cuda::std::tuple{left_indices.begin(), right_indices.begin()});
      auto output_iter = cuda::make_zip_iterator(
        cuda::std::tuple{filtered_left_indices->begin(), filtered_right_indices->begin()});
      auto valid_predicate = [predicate_results_ptr] __device__(auto i) -> bool {
        return predicate_results_ptr[i];
      };

      // Copy valid indices to output vector
      // CUB APIs are used instead of Thrust to enable 64-bit operations on index vectors of size
      // greater than integer limits
      cudf::detail::copy_if_async(input_iter,
                                  input_iter + left_indices.size(),
                                  cuda::counting_iterator<std::size_t>{0},
                                  output_iter,
                                  valid_predicate,
                                  stream);
    }
    if (num_invalid > 0) {
      {
        // For invalid indices, set the output pairs to be `(invalid_left_idx, JoinNoMatch)`
        // CUB APIs are used instead of Thrust to enable 64-bit operations on index vectors of size
        // greater than integer limits
        auto filter_passing_indices_ref = filter_passing_indices.ref(cuco::contains);
        auto is_unmatched_idx = [filter_passing_indices_ref] __device__(size_type idx) -> bool {
          auto is_unmatched = !filter_passing_indices_ref.contains(idx);
          return is_unmatched;
        };
        cudf::detail::copy_if_async(
          cuda::counting_iterator<std::size_t>{0},
          cuda::counting_iterator{static_cast<std::size_t>(left.num_rows())},
          filtered_left_indices->begin() + num_valid,
          is_unmatched_idx,
          stream);
      }
      cub::DeviceTransform::Fill(
        filtered_right_indices->begin() + num_valid, num_invalid, JoinNoMatch, stream.get());
    }

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else if (join_kind == join_kind::FULL_JOIN) {
    // FULL_JOIN: Optimized implementation using stream compaction
    // Strategy: Use a single scan to identify failed matches, then use stream compaction

    // First, identify failed matched pairs
    auto is_failed_matched_pair = [=] __device__(size_type i) -> bool {
      return !predicate_results_ptr[i] && left_ptr[i] != JoinNoMatch && right_ptr[i] != JoinNoMatch;
    };

    // Count failed matches for output sizing
    auto const failed_matched_count =
      output_size.has_value()
        ? *output_size - left_indices.size()
        : cudf::detail::count_if(
            cuda::counting_iterator<cudf::size_type>{0},
            cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
            is_failed_matched_pair,
            stream);
    auto const result_size = left_indices.size() + failed_matched_count;

    if (result_size == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(result_size);

    // Use two-step approach with optimized memory management
    // Step 1: Handle primary pairs
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
                      cuda::make_zip_iterator(cuda::std::tuple{filtered_left_indices->begin(),
                                                               filtered_right_indices->begin()}),
                      [=] __device__(size_type i) -> cuda::std::tuple<size_type, size_type> {
                        auto const left_idx  = left_ptr[i];
                        auto const right_idx = right_ptr[i];
                        // For FULL JOIN: preserve original unmatched rows, nullify right side of
                        // failed matches
                        auto const output_right_idx =
                          (predicate_results_ptr[i] || left_idx == JoinNoMatch) ? right_idx
                                                                                : JoinNoMatch;

                        return cuda::std::tuple{left_idx, output_right_idx};
                      });

    // Step 2: Add secondary pairs for failed matches using stream compaction
    if (failed_matched_count > 0) {
      auto secondary_iter = cuda::make_zip_iterator(
        cuda::std::tuple{filtered_left_indices->begin() + left_indices.size(),
                         filtered_right_indices->begin() + left_indices.size()});

      auto failed_match_iter = cudf::detail::make_counting_transform_iterator(
        0, [=] __device__(size_type i) -> cuda::std::tuple<size_type, size_type> {
          return cuda::std::tuple{JoinNoMatch, right_ptr[i]};
        });
      cudf::detail::copy_if_async(failed_match_iter,
                                  failed_match_iter + left_indices.size(),
                                  cuda::counting_iterator<cudf::size_type>{0},
                                  secondary_iter,
                                  is_failed_matched_pair,
                                  stream);
    }

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else {
    CUDF_FAIL("Unsupported join kind for filter_join_indices");
  }
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_output_size(cudf::table_view const& left,
                                cudf::table_view const& right,
                                cudf::device_span<size_type const> left_indices,
                                cudf::device_span<size_type const> right_indices,
                                ast::expression const& predicate,
                                join_kind join_kind,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  // Validate inputs (same constraints as filter_join_indices)
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);
  CUDF_EXPECTS(
    join_kind == join_kind::INNER_JOIN || join_kind == join_kind::LEFT_JOIN ||
      join_kind == join_kind::FULL_JOIN,
    "filter_join_indices_output_size only supports INNER_JOIN, LEFT_JOIN, and FULL_JOIN.",
    std::invalid_argument);

  auto empty_counts = [&]() {
    return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
  };

  if (left_indices.empty()) { return {0, empty_counts()}; }
  if (join_kind == join_kind::LEFT_JOIN && left.num_rows() == 0) { return {0, empty_counts()}; }

  auto const has_nulls = predicate.may_evaluate_null(left, right, stream);

  auto const parser = ast::detail::expression_parser{
    predicate, left, right, has_nulls, stream, cudf::get_current_device_resource_ref()};

  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The predicate expression must produce a Boolean output",
               std::invalid_argument);

  auto const has_complex_type = parser.has_complex_type();

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  detail::grid_1d const config(left_indices.size(), DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_per_block = parser.shmem_per_thread * DEFAULT_JOIN_BLOCK_SIZE;

  auto const counts_size = join_kind == join_kind::LEFT_JOIN
                             ? static_cast<std::size_t>(left.num_rows())
                             : left_indices.size();
  auto output_counts =
    join_kind == join_kind::LEFT_JOIN
      ? cudf::detail::make_zeroed_device_uvector_async<size_type>(counts_size, stream, mr)
      : rmm::device_uvector<size_type>(counts_size, stream, mr);

  cudf::detail::dispatch_bool(has_nulls, [&](auto has_nulls_c) {
    cudf::detail::dispatch_bool(has_complex_type, [&](auto has_complex_c) {
      launch_filter_output_size_kernel<decltype(has_nulls_c)::value,
                                       decltype(has_complex_c)::value>(
        *left_table,
        *right_table,
        left_indices,
        right_indices,
        parser.device_expression_data,
        config,
        shmem_per_block,
        join_kind,
        output_counts.data(),
        stream);
    });
  });

  if (join_kind == join_kind::LEFT_JOIN) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      output_counts.begin(),
                      output_counts.end(),
                      output_counts.begin(),
                      cuda::proclaim_return_type<size_type>(
                        [] __device__(size_type count) { return count > 0 ? count : 1; }));
  }

  std::size_t const total =
    thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   output_counts.begin(),
                   output_counts.end(),
                   std::size_t{0});

  return {total, std::make_unique<rmm::device_uvector<size_type>>(std::move(output_counts))};
}

}  // namespace detail

// Public API implementation
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices(cudf::table_view const& left,
                    cudf::table_view const& right,
                    cudf::device_span<size_type const> left_indices,
                    cudf::device_span<size_type const> right_indices,
                    ast::expression const& predicate,
                    cudf::join_kind join_kind,
                    std::optional<std::size_t> output_size,
                    cuda::stream_ref stream,
                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices(
    left, right, left_indices, right_indices, predicate, join_kind, output_size, stream, mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_output_size(cudf::table_view const& left,
                                cudf::table_view const& right,
                                cudf::device_span<size_type const> left_indices,
                                cudf::device_span<size_type const> right_indices,
                                ast::expression const& predicate,
                                cudf::join_kind join_kind,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_output_size(
    left, right, left_indices, right_indices, predicate, join_kind, stream, mr);
}

}  // namespace cudf
