/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/filter_join_indices/filter_join_indices_jit_kernel.cuh"
#include "join/jit/filter_join_kernel.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cub/device/device_transform.cuh>
#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <cuda/std/tuple>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/parser.hpp>
#include <jit/row_ir.hpp>
#include <jit/span.cuh>

#include <memory>
#include <utility>

namespace cudf {
namespace detail {

namespace {

std::vector<std::string> build_join_filter_template_params(
  std::span<transform_input const> inputs,
  std::span<std::optional<int32_t>> table_sources,
  bool has_user_data,
  bool is_null_aware)
{
  std::vector<std::string> template_params;
  template_params.emplace_back(rtcx::reflect(has_user_data));
  template_params.emplace_back(rtcx::reflect(is_null_aware));

  std::vector<std::string> accessors;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto const& input = inputs[i];
    if (auto* col = std::get_if<column_view>(&input)) {
      accessors.emplace_back(rtcx::reflect_template("cudf::jit::column_accessor",
                                                    rtcx::reflect(i),
                                                    "cudf::column_device_view_core",
                                                    cudf::type_to_name(col->type()),
                                                    rtcx::reflect(false),
                                                    rtcx::reflect(table_sources[i].value())));
    } else {
      auto& scalar = std::get<scalar_column_view>(input);
      accessors.emplace_back(rtcx::reflect_template(
        "cudf::jit::column_accessor",
        rtcx::reflect(i),
        "cudf::column_device_view_core",
        cudf::type_to_name(scalar.as_column_view().type()),
        rtcx::reflect(true),
        rtcx::reflect(0)  // scalars don't belong to a table, so just use 0 as placeholder
        ));
    }
  }

  template_params.push_back(rtcx::reflect_template("cudf::jit::type_list", accessors));

  return template_params;
}

// Build the JIT kernel for join filtering
kernel build_join_filter_kernel(std::string const& predicate_code,
                                std::span<transform_input const> inputs,
                                std::span<std::optional<int32_t>> table_sources,
                                bool is_ptx,
                                bool has_user_data,
                                bool is_null_aware,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  std::vector<std::string> ptx_output_types{"bool"};
  std::vector<std::string> ptx_input_types;

  for (auto const& input : inputs) {
    if (auto* col = std::get_if<column_view>(&input)) {
      ptx_input_types.push_back(cudf::type_to_name(col->type()));
    } else {
      auto& scalar = std::get<scalar_column_view>(input);
      ptx_input_types.push_back(cudf::type_to_name(scalar.type()));
    }
  }

  // Parse predicate code
  auto const cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(
               predicate_code,
               "GENERIC_JOIN_FILTER_OP",
               cudf::jit::build_ptx_params(ptx_output_types, ptx_input_types, has_user_data))
           : cudf::jit::parse_single_function_cuda(predicate_code, "GENERIC_JOIN_FILTER_OP");

  // Build template parameters and kernel name
  auto template_args =
    build_join_filter_template_params(inputs, table_sources, has_user_data, is_null_aware);
  auto kernel_name = rtcx::reflect_template("cudf::join::jit::filter_join_kernel", template_args);

  // Get compiled kernel
  return cudf::jit::get_udf_kernel(
    "cudf/cpp/src/join/jit/filter_join_kernel.cu", kernel_name, cuda_source);
}

// Launch the JIT kernel for join filtering
void launch_join_filter_kernel(kernel const& kernel,
                               cudf::device_span<size_type const> left_indices,
                               cudf::device_span<size_type const> right_indices,
                               std::span<transform_input const> inputs,
                               bool* predicate_results,
                               std::optional<void*> user_data,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Create device views of tables
  std::vector<column_view> column_views;
  for (auto const& input : inputs) {
    if (auto* col = std::get_if<column_view>(&input)) {
      column_views.push_back(*col);
    } else {
      auto& scalar = std::get<scalar_column_view>(input);
      column_views.push_back(scalar.as_column_view());
    }
  }

  auto [handles, device_views] =
    cudf::jit::column_views_to_device<column_device_view, column_view>(column_views, stream, mr);

  // Set up kernel parameters - use JIT-compatible span type
  cudf::size_type num_rows                         = left_indices.size();
  cudf::size_type const* left_indices_ptr          = left_indices.data();
  cudf::size_type const* right_indices_ptr         = right_indices.data();
  cudf::column_device_view_core const* columns_ptr = device_views.data();
  void* user_data_ptr                              = user_data.value_or(nullptr);

  void* args[]{&num_rows,
               &left_indices_ptr,
               &right_indices_ptr,
               &columns_ptr,
               &predicate_results,
               &user_data_ptr};

  auto cfg = kernel.max_occupancy_config(0, 0);

  kernel.launch({cfg.min_grid_size}, {cfg.block_size}, 0, stream, args);
}

// Same join semantics handling as the AST version
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
apply_join_semantics(cudf::table_view const& left,
                     cudf::table_view const& right,
                     cudf::device_span<size_type const> left_indices,
                     cudf::device_span<size_type const> right_indices,
                     rmm::device_uvector<bool> const& predicate_results,
                     join_kind join_kind,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  auto make_empty_result = [&]() {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  };

  auto make_result_vectors = [&](std::size_t size) {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr)};
  };

  auto predicate_results_ptr = predicate_results.data();
  auto left_ptr              = left_indices.data();
  auto right_ptr             = right_indices.data();

  // Handle different join semantics - same logic as AST version
  if (join_kind == join_kind::INNER_JOIN) {
    // INNER_JOIN: only keep pairs that satisfy the predicate
    auto valid_predicate = [=] __device__(size_type i) -> bool { return predicate_results_ptr[i]; };

    auto const num_valid =
      cudf::detail::count_if(cuda::counting_iterator<size_type>{0},
                             cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
                             valid_predicate,
                             stream);

    if (num_valid == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(num_valid);

    auto input_iter =
      cuda::make_zip_iterator(cuda::std::tuple{left_indices.begin(), right_indices.begin()});
    auto output_iter = cuda::make_zip_iterator(
      cuda::std::tuple{filtered_left_indices->begin(), filtered_right_indices->begin()});

    cudf::detail::copy_if(
      input_iter,
      input_iter + left_indices.size(),
      cuda::counting_iterator<size_type>{0},
      output_iter,
      [valid_predicate] __device__(size_type idx) -> bool { return valid_predicate(idx); },
      stream);

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else if (join_kind == join_kind::LEFT_JOIN) {
    // LEFT_JOIN: Keep all left rows, nullify right indices for failed predicates
    // Using same complex logic as AST version with hash set for unmatched left rows
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
                                   stream.value()};

    auto predicate_func = [predicate_results_ptr] __device__(std::size_t idx) -> bool {
      return static_cast<bool>(predicate_results_ptr[idx]);
    };
    auto const num_filter_passing =
      filter_passing_indices.insert_if(left_ptr,
                                       left_ptr + left_indices.size(),
                                       cuda::counting_iterator<std::size_t>{0},
                                       predicate_func,
                                       stream.value());

    auto const num_invalid = left.num_rows() - num_filter_passing;

    auto const num_valid = cudf::detail::count_if(
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
      [predicate_results_ptr] __device__(size_type i) -> bool { return predicate_results_ptr[i]; },
      stream);
    auto const output_size = num_valid + num_invalid;
    if (output_size == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(output_size);
    if (num_valid > 0) {
      auto input_iter =
        cuda::make_zip_iterator(cuda::std::tuple{left_indices.begin(), right_indices.begin()});
      auto output_iter = cuda::make_zip_iterator(
        cuda::std::tuple{filtered_left_indices->begin(), filtered_right_indices->begin()});
      auto valid_predicate = [predicate_results_ptr] __device__(auto i) -> bool {
        return predicate_results_ptr[i];
      };

      cudf::detail::copy_if(input_iter,
                            input_iter + left_indices.size(),
                            cuda::counting_iterator<std::size_t>{0},
                            output_iter,
                            valid_predicate,
                            stream);
    }
    if (num_invalid > 0) {
      auto filter_passing_indices_ref = filter_passing_indices.ref(cuco::contains);
      auto is_unmatched_idx = [filter_passing_indices_ref] __device__(size_type idx) -> bool {
        auto is_unmatched = !filter_passing_indices_ref.contains(idx);
        return is_unmatched;
      };
      cudf::detail::copy_if(cuda::counting_iterator<std::size_t>{0},
                            cuda::counting_iterator{static_cast<std::size_t>(left.num_rows())},
                            filtered_left_indices->begin() + num_valid,
                            is_unmatched_idx,
                            stream);

      cub::DeviceTransform::Fill(
        filtered_right_indices->begin() + num_valid, num_invalid, JoinNoMatch, stream.value());
    }

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else if (join_kind == join_kind::FULL_JOIN) {
    // FULL_JOIN: Preserve all rows, split failed matches - same as AST version
    auto is_failed_matched_pair = [=] __device__(size_type i) -> bool {
      return !predicate_results_ptr[i] && left_ptr[i] != JoinNoMatch && right_ptr[i] != JoinNoMatch;
    };

    auto const failed_matched_count =
      cudf::detail::count_if(cuda::counting_iterator<cudf::size_type>{0},
                             cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
                             is_failed_matched_pair,
                             stream);
    auto const output_size = left_indices.size() + failed_matched_count;

    if (output_size == 0) { return make_empty_result(); }

    auto [filtered_left_indices, filtered_right_indices] = make_result_vectors(output_size);

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
                      cuda::make_zip_iterator(cuda::std::tuple{filtered_left_indices->begin(),
                                                               filtered_right_indices->begin()}),
                      [=] __device__(size_type i) -> cuda::std::tuple<size_type, size_type> {
                        auto const left_idx  = left_ptr[i];
                        auto const right_idx = right_ptr[i];
                        auto const output_right_idx =
                          (predicate_results_ptr[i] || left_idx == JoinNoMatch) ? right_idx
                                                                                : JoinNoMatch;
                        return cuda::std::tuple{left_idx, output_right_idx};
                      });

    if (failed_matched_count > 0) {
      auto secondary_iter = cuda::make_zip_iterator(
        cuda::std::tuple{filtered_left_indices->begin() + left_indices.size(),
                         filtered_right_indices->begin() + left_indices.size()});

      auto failed_match_iter = cudf::detail::make_counting_transform_iterator(
        0, [=] __device__(size_type i) -> cuda::std::tuple<size_type, size_type> {
          return cuda::std::tuple{JoinNoMatch, right_ptr[i]};
        });
      cudf::detail::copy_if(failed_match_iter,
                            failed_match_iter + left_indices.size(),
                            cuda::counting_iterator<cudf::size_type>{0},
                            secondary_iter,
                            is_failed_matched_pair,
                            stream);
    }

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else {
    CUDF_FAIL("Unsupported join kind for filter_join_indices_jit");
  }
}

void validate_column_types(cudf::table_view const& table, char const* side)
{
  for (auto const& col : table) {
    CUDF_EXPECTS(
      cudf::is_fixed_width(col.type()) || col.type().id() == type_id::STRING,
      "filter_join_indices_jit does not support nested or dictionary column types in the " +
        std::string(side) + " table.",
      std::invalid_argument);
  }
}

}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        std::string const& predicate_code,
                        join_kind join_kind,
                        bool is_ptx,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs - same as AST version
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(join_kind == join_kind::INNER_JOIN || join_kind == join_kind::LEFT_JOIN ||
                 join_kind == join_kind::FULL_JOIN,
               "filter_join_indices_jit only supports INNER_JOIN, LEFT_JOIN, and FULL_JOIN.",
               std::invalid_argument);

  validate_column_types(left, "left");
  validate_column_types(right, "right");

  auto make_empty_result = [&]() {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  };

  if (left_indices.empty()) { return make_empty_result(); }

  // Compile JIT kernel
  std::vector<transform_input> inputs;
  std::vector<std::optional<int32_t>> table_sources;
  for (auto const& col : left) {
    inputs.emplace_back(col);
    table_sources.emplace_back(0);
  }
  for (auto const& col : right) {
    inputs.emplace_back(col);
    table_sources.emplace_back(1);
  }

  auto kernel = build_join_filter_kernel(predicate_code,
                                         inputs,
                                         table_sources,
                                         is_ptx,
                                         false,  // has_user_data = false for now
                                         false,
                                         stream,
                                         mr);

  // Allocate predicate results
  auto predicate_results = rmm::device_uvector<bool>(left_indices.size(), stream);

  // Launch kernel
  launch_join_filter_kernel(kernel,
                            left_indices,
                            right_indices,
                            inputs,
                            predicate_results.data(),
                            std::nullopt,  // no user data for now
                            stream,
                            mr);

  // Apply same join semantics as AST version
  return apply_join_semantics(
    left, right, left_indices, right_indices, predicate_results, join_kind, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        ast::expression const& predicate,
                        join_kind join_kind,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(join_kind == join_kind::INNER_JOIN || join_kind == join_kind::LEFT_JOIN ||
                 join_kind == join_kind::FULL_JOIN,
               "filter_join_indices_jit only supports INNER_JOIN, LEFT_JOIN, and FULL_JOIN.",
               std::invalid_argument);

  validate_column_types(left, "left");
  validate_column_types(right, "right");

  if (left_indices.empty()) {
    return std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr)};
  }

  // Convert AST predicate to JIT code
  auto filter_result = row_ir::ast_converter::filter(
    row_ir::target::CUDA, predicate, left, right, "filter_operation", stream, mr);

  auto template_args =
    build_join_filter_template_params(filter_result.inputs,
                                      filter_result.input_table_sources,
                                      filter_result.user_data.has_value(),
                                      filter_result.is_null_aware == null_aware::YES);

  auto const cuda_source =
    cudf::jit::parse_single_function_cuda(filter_result.udf, "GENERIC_JOIN_FILTER_OP");

  auto kernel_name = rtcx::reflect_template("cudf::join::jit::filter_join_kernel", template_args);
  auto kernel      = cudf::jit::get_udf_kernel(
    "cudf/cpp/src/join/jit/filter_join_kernel.cu", kernel_name, cuda_source);

  // Allocate and compute predicate results
  auto predicate_results = rmm::device_uvector<bool>(left_indices.size(), stream);
  launch_join_filter_kernel(kernel,
                            left_indices,
                            right_indices,
                            filter_result.inputs,
                            predicate_results.data(),
                            filter_result.user_data,
                            stream,
                            mr);

  return apply_join_semantics(
    left, right, left_indices, right_indices, predicate_results, join_kind, stream, mr);
}

}  // namespace detail

// Public API implementations
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        std::string const& predicate_code,
                        cudf::join_kind join_kind,
                        bool is_ptx,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_jit(
    left, right, left_indices, right_indices, predicate_code, join_kind, is_ptx, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        cudf::ast::expression const& predicate,
                        cudf::join_kind join_kind,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_jit(
    left, right, left_indices, right_indices, predicate, join_kind, stream, mr);
}

}  // namespace cudf
