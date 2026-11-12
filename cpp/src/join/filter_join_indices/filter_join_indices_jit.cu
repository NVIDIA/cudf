/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/join_common_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/errc.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cub/device/device_transform.cuh>
#include <cuco/static_set.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/reflect.hpp>
#include <jit/row_ir.hpp>
#include <jit/util.hpp>
#include <rtcx/rtcx.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cudf {

namespace {

kernel get_linked_kernel(transform_udf const& udf,
                         std::span<transform_input const> inputs,
                         std::span<std::optional<int32_t>> table_sources)
{
  auto is_null_aware = udf.is_null_aware == null_aware::YES;
  auto indexed       = udf.indexed;

  std::vector<int32_t> input_table_sources;
  std::vector<int32_t> input_table_indices;

  for (size_t i = 0; i < inputs.size(); ++i) {
    input_table_sources.push_back(table_sources[i].value_or(2));
    input_table_indices.push_back(static_cast<int32_t>(i));
  }

  auto specs      = cudf::jit::make_input_specs(inputs);
  auto reflection = cudf::jit::reflect(true,
                                       specs,
                                       std::span<transform_output_spec const>{},
                                       input_table_sources,
                                       input_table_indices);
  auto name       = rtcx::reflect_template(
    "cudf::detail::filter_join_kernel", rtcx::reflect(is_null_aware), reflection.input_accessors);
  std::array outputs{transform_output_spec{.type = type_id::BOOL8}};
  auto signature =
    cudf::jit::reflect_udf_signature(is_null_aware, specs, outputs, true, indexed ? 2 : 0);
  auto kernel_fragment = cudf::jit::get_udf_kernel_fragment(
    "cudf/cpp/src/join/jit/filter_join_kernel.cu", name, signature, indexed);
  auto adapter = cudf::get_udf_adapter(
    udf.symbol, "cudf_transform_udf", is_null_aware, specs, outputs, true, indexed ? 2 : 0);

  std::vector<rtcx::memory_fragment> fragments{
    {.data = kernel_fragment->view(), .type = rtcx::binary_type::LTO_IR, .name = nullptr}};

  if (adapter) {
    auto view = adapter->view();
    fragments.push_back(
      {.data = view.data, .type = cudf::jit::as_rtcx_binary_type(view.type), .name = nullptr});
  }

  for (auto const& fragment : udf.fragments) {
    fragments.push_back({.data = fragment.data,
                         .type = cudf::jit::as_rtcx_binary_type(fragment.type),
                         .name = nullptr});
  }

  return get_lto_linked_kernel("cudf/cpp/src/join/jit/filter_join_kernel.cu", {}, fragments);
}

// Launch the JIT kernel for join filtering
void launch_join_filter_kernel(kernel const& kernel,
                               cudf::device_span<size_type const> left_indices,
                               cudf::device_span<size_type const> right_indices,
                               std::span<transform_input const> inputs,
                               bool* predicate_results,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();

  std::vector<std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>>
    handles;

  auto host_views =
    cudf::detail::make_empty_pinned_vector<column_device_view_core>(inputs.size(), stream);
  for (auto const& input : inputs) {
    if (auto const* data = std::get_if<user_data>(&input)) {
      host_views.push_back(column_device_view_core{*data});
    } else {
      auto column = std::holds_alternative<column_view>(input)
                      ? std::get<column_view>(input)
                      : std::get<scalar_column_view>(input).as_column_view();
      auto handle = column_device_view::create(column, stream, mr.get_temporary_mr());
      host_views.push_back(*handle);
      handles.push_back(std::move(handle));
    }
  }

  cuda::device_buffer<column_device_view_core> device_views{
    stream, mr.get_temporary_mr(), inputs.size(), cuda::no_init};
  cudf::detail::cuda_memcpy<column_device_view_core>(device_views, host_views, stream);

  // Match the pointer parameters of the JIT kernel entry point.
  cudf::size_type num_rows                         = left_indices.size();
  cudf::size_type const* left_indices_ptr          = left_indices.data();
  cudf::size_type const* right_indices_ptr         = right_indices.data();
  cudf::column_device_view_core const* columns_ptr = device_views.data();

  cudf::detail::device_scalar<error_int> d_max_error(
    static_cast<error_int>(errc::SUCCESS), stream, mr.get_temporary_mr());
  auto max_error_ptr = d_max_error.data();

  void* args[]{&num_rows,
               &left_indices_ptr,
               &right_indices_ptr,
               &columns_ptr,
               &predicate_results,
               &max_error_ptr};

  auto cfg = kernel.max_occupancy_config(0, 0);

  kernel.launch({cfg.min_grid_size}, {cfg.block_size}, 0, stream, args);

  auto error = static_cast<errc>(d_max_error.value(stream));
  if (error != errc::SUCCESS) {
    throw evaluation_error(
      error, std::format("Join filter evaluation failed with error `{}`", to_string(error)));
  }
}

// Same join semantics handling as the AST version
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
apply_join_semantics(cudf::table_view const& left,
                     cudf::device_span<size_type const> left_indices,
                     cudf::device_span<size_type const> right_indices,
                     cudf::device_span<bool const> predicate_results,
                     join_kind join_kind,
                     cuda::stream_ref stream,
                     cudf::memory_resources mr)
{
  cudf::memory_resources mrs{mr};
  auto make_empty_result = [&]() {
    return std::pair{
      std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
      std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr())};
  };

  auto make_result_vectors = [&](std::size_t size) {
    return std::pair{
      std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr.get_output_mr()),
      std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr.get_output_mr())};
  };

  if (left.num_rows() == 0) { return make_empty_result(); }

  auto predicate_results_ptr = predicate_results.data();
  auto left_ptr              = left_indices.data();

  // Handle different join semantics - same logic as AST version
  if (join_kind == join_kind::INNER_JOIN) {
    // INNER_JOIN: only keep pairs that satisfy the predicate
    auto valid_predicate = [=] __device__(size_type i) -> bool { return predicate_results_ptr[i]; };

    auto const num_valid =
      cudf::detail::count_if(cuda::counting_iterator<size_type>{0},
                             cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
                             valid_predicate,
                             stream,
                             mrs.as_temporary());

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
      stream,
      mrs.as_temporary());

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
                                   rmm::mr::polymorphic_allocator<char>{mr.get_temporary_mr()},
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

    auto const num_valid = cudf::detail::count_if(
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator{static_cast<size_type>(left_indices.size())},
      [predicate_results_ptr] __device__(size_type i) -> bool { return predicate_results_ptr[i]; },
      stream,
      mrs.as_temporary());
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
                            stream,
                            mrs.as_temporary());
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
                            stream,
                            mrs.as_temporary());

      cub::DeviceTransform::Fill(
        filtered_right_indices->begin() + num_valid, num_invalid, JoinNoMatch, stream.get());
    }

    return std::pair{std::move(filtered_left_indices), std::move(filtered_right_indices)};

  } else {
    CUDF_FAIL("Unsupported join kind for filter_join_indices_jit");
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
execute_join_filter(table_view const& left,
                    table_view const& right,
                    device_span<size_type const> left_indices,
                    device_span<size_type const> right_indices,
                    transform_udf const& predicate,
                    std::span<transform_input const> inputs,
                    std::span<std::optional<int32_t>> sources,
                    join_kind kind,
                    cuda::stream_ref stream,
                    cudf::memory_resources mr)
{
  auto kernel = get_linked_kernel(predicate, inputs, sources);
  auto results =
    cuda::device_buffer<bool>(stream, mr.get_temporary_mr(), left_indices.size(), cuda::no_init);
  launch_join_filter_kernel(
    kernel, left_indices, right_indices, inputs, results.data(), stream, mr);
  auto filtered = apply_join_semantics(left,
                                       left_indices,
                                       right_indices,
                                       results,
                                       kind == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : kind,
                                       stream,
                                       mr);
  if (kind == join_kind::FULL_JOIN) {
    return detail::finalize_full_join(std::move(filtered),
                                      left.num_rows(),
                                      right.num_rows(),
                                      std::nullopt,
                                      stream,
                                      mr.get_temporary_mr());
  }
  return filtered;
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

void perform_checks(cudf::table_view const& left,
                    cudf::table_view const& right,
                    cudf::device_span<size_type const> left_indices,
                    cudf::device_span<size_type const> right_indices,
                    std::span<transform_input const> extra_inputs,
                    join_kind kind)
{
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(
    kind == join_kind::INNER_JOIN || kind == join_kind::LEFT_JOIN || kind == join_kind::FULL_JOIN,
    "filter_join_indices_jit only supports INNER_JOIN, LEFT_JOIN, and FULL_JOIN.",
    std::invalid_argument);

  validate_column_types(left, "left");
  validate_column_types(right, "right");

  for (auto const& input : extra_inputs) {
    if (auto const* column = std::get_if<column_view>(&input)) {
      CUDF_EXPECTS(static_cast<std::size_t>(column->size()) == left_indices.size(),
                   "Extra input columns must have one row per join candidate pair",
                   std::invalid_argument);
    }
  }
}

}  // anonymous namespace

namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        transform_udf const& predicate,
                        join_kind join_kind,
                        std::span<transform_input const> extra_inputs,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();

  perform_checks(left, right, left_indices, right_indices, extra_inputs, join_kind);

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

  for (auto const& input : extra_inputs) {
    inputs.push_back(input);
    table_sources.emplace_back(std::nullopt);
  }

  if (left_indices.empty()) {
    return {std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
            std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr())};
  }
  return execute_join_filter(left,
                             right,
                             left_indices,
                             right_indices,
                             predicate,
                             inputs,
                             table_sources,
                             join_kind,
                             stream,
                             mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        ast::expression const& predicate,
                        join_kind join_kind,
                        cudf::lto_fragments fragments,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();

  perform_checks(left, right, left_indices, right_indices, {}, join_kind);

  if (left_indices.empty()) {
    return std::pair{
      std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
      std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr())};
  }

  auto args     = detail::row_ir::ast_converter::filter(detail::row_ir::target::CUDA,
                                                    predicate,
                                                    fragments,
                                                    left,
                                                    right,
                                                    "cudf_transform_udf",
                                                    stream,
                                                    mr.get_temporary_mr());
  auto compiled = compile_cuda_to_lto_ir(args.udf.c_str(), {}, {}, {}, true, "cudf_row_ir");
  std::vector<lto_fragment> modules(args.lto_fragments.begin(), args.lto_fragments.end());
  modules.push_back(compiled.view());
  transform_udf udf{args.udf_expression, modules, args.is_null_aware};
  return execute_join_filter(left,
                             right,
                             left_indices,
                             right_indices,
                             udf,
                             args.inputs,
                             args.input_table_sources,
                             join_kind,
                             stream,
                             mr);
}
}  // namespace detail

// Public API implementations
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        transform_udf const& predicate,
                        cudf::join_kind join_kind,
                        std::span<transform_input const> extra_inputs,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_jit(
    left, right, left_indices, right_indices, predicate, join_kind, extra_inputs, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        cudf::ast::expression const& predicate,
                        cudf::join_kind join_kind,
                        cudf::lto_fragments fragments,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_jit(
    left, right, left_indices, right_indices, predicate, join_kind, fragments, stream, mr);
}

}  // namespace cudf
