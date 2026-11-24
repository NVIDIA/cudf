/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/filter_join_indices/filter_join_indices_jit_kernel.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/errc.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

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
#include <jit/parser.hpp>
#include <jit/reflect.hpp>
#include <jit/row_ir.hpp>
#include <jit/util.hpp>
#include <rtcx/rtcx.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cudf {
namespace detail {

namespace {

std::vector<std::string> build_join_filter_template_params(
  std::span<transform_input_spec const> input_specs,
  std::span<std::optional<int32_t>> table_sources,
  bool has_user_data,
  bool is_null_aware,
  std::variant<udf_source_type, lto_binary_type> source_type)
{
  std::vector<int32_t> input_table_sources;
  std::vector<int32_t> input_table_indices;
  for (size_t i = 0; i < input_specs.size(); ++i) {
    input_table_sources.push_back(table_sources[i].value_or(0));
    input_table_indices.push_back(static_cast<int32_t>(i));
  }

  auto reflection = cudf::jit::reflect(source_type,
                                       input_specs,
                                       std::span<transform_output_spec const>{},
                                       input_table_sources,
                                       input_table_indices);
  // Precompiled UDFs are linked against the physical type of the BOOL8 output.
  auto const result_type = std::holds_alternative<lto_binary_type>(source_type)
                             ? cudf::jit::physical_type_of(data_type{type_id::BOOL8})
                             : data_type{type_id::BOOL8};
  return {rtcx::reflect(has_user_data),
          rtcx::reflect(is_null_aware),
          type_to_name(result_type),
          reflection.input_accessors};
}

// Build the JIT kernel for join filtering
kernel build_join_filter_kernel(std::span<uint8_t const> udf,
                                std::variant<udf_source_type, lto_binary_type> udf_type,
                                std::span<transform_input const> inputs,
                                std::span<std::optional<int32_t>> table_sources,
                                bool has_user_data,
                                bool is_null_aware)
{
  CUDF_FUNC_RANGE();

  constexpr auto source_file = "cudf/cpp/src/join/jit/filter_join_kernel.cu";

  auto const input_specs = cudf::jit::make_input_specs(inputs);
  auto template_args     = build_join_filter_template_params(
    input_specs, table_sources, has_user_data, is_null_aware, udf_type);
  auto kernel_name = rtcx::reflect_template("cudf::join::jit::filter_join_kernel", template_args);

  if (auto const* binary_type = std::get_if<lto_binary_type>(&udf_type)) {
    auto const outputs = std::array{transform_output_spec{.type = type_id::BOOL8}};
    auto signature =
      cudf::jit::reflect_udf_signature(is_null_aware, has_user_data, input_specs, outputs, true);
    auto fragment = cudf::jit::get_udf_kernel_fragment(source_file, kernel_name, signature);
    return cudf::jit::get_lto_udf_kernel(
      source_file, fragment->view(), lto_binary_type::LTO_IR, kernel_name, udf, *binary_type);
  }

  auto cuda_source = cudf::jit::parse_single_function_cuda(
    std::string{reinterpret_cast<char const*>(udf.data()), udf.size()}, "GENERIC_JOIN_FILTER_OP");
  return cudf::jit::get_udf_kernel(source_file, kernel_name, cuda_source, {});
}

// Launch the JIT kernel for join filtering
void launch_join_filter_kernel(kernel const& kernel,
                               cudf::device_span<size_type const> left_indices,
                               cudf::device_span<size_type const> right_indices,
                               std::span<transform_input const> inputs,
                               bool* predicate_results,
                               std::optional<void*> user_data,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
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

  auto [handles, device_views] = cudf::jit::column_views_to_device<column_device_view, column_view>(
    column_views, stream, mr.get_temporary_mr());

  // Match the pointer parameters of the JIT kernel entry point.
  cudf::size_type num_rows                         = left_indices.size();
  cudf::size_type const* left_indices_ptr          = left_indices.data();
  cudf::size_type const* right_indices_ptr         = right_indices.data();
  cudf::column_device_view_core const* columns_ptr = device_views.data();
  void* user_data_ptr                              = user_data.value_or(nullptr);

  cudf::detail::device_scalar<int32_t> d_max_error(
    static_cast<int32_t>(errc::SUCCESS), stream, mr.get_temporary_mr());
  auto max_error_ptr = d_max_error.data();

  void* args[]{&num_rows,
               &left_indices_ptr,
               &right_indices_ptr,
               &columns_ptr,
               &predicate_results,
               &user_data_ptr,
               &max_error_ptr};

  auto cfg = kernel.max_occupancy_config(0, 0);

  kernel.launch({cfg.min_grid_size}, {cfg.block_size}, 0, stream, args);

  auto error = static_cast<errc>(d_max_error.value(stream));
  if (error != errc::SUCCESS) {
    throw evaluation_error(
      error, std::format("Join filter evaluation failed with error `{}`", to_string(error)));
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
make_empty_result(cuda::stream_ref stream, cudf::memory_resources mr)
{
  return {std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr())};
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
  auto make_result_vectors = [&](std::size_t size) {
    return std::pair{
      std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr.get_output_mr()),
      std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr.get_output_mr())};
  };

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

    if (num_valid == 0) { return make_empty_result(stream, mr); }

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
    if (output_size == 0) { return make_empty_result(stream, mr); }

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

void validate_filter_inputs(cudf::table_view const& left,
                            cudf::table_view const& right,
                            cudf::device_span<size_type const> left_indices,
                            cudf::device_span<size_type const> right_indices,
                            join_kind join_kind,
                            std::span<transform_input const> extra_inputs = {})
{
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(join_kind == join_kind::INNER_JOIN || join_kind == join_kind::LEFT_JOIN ||
                 join_kind == join_kind::FULL_JOIN,
               "filter_join_indices_jit only supports INNER_JOIN, LEFT_JOIN, and FULL_JOIN.",
               std::invalid_argument);

  validate_column_types(left, "left");
  validate_column_types(right, "right");

  for (auto const& input : extra_inputs) {
    auto const* col = std::get_if<column_view>(&input);
    CUDF_EXPECTS(col == nullptr || std::cmp_equal(col->size(), left_indices.size()),
                 "Extra input columns must have one row per index pair",
                 std::invalid_argument);
  }
}

// Evaluates the predicate on every input pair and applies the join semantics. The predicate sees
// each pair's position in the input maps, and pairs with a `JoinNoMatch` side fail without being
// evaluated, so a FULL join is the LEFT join of the same maps plus the unmatched right rows.
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
evaluate_join_filter(kernel const& kernel,
                     cudf::table_view const& left,
                     cudf::table_view const& right,
                     cudf::device_span<size_type const> left_indices,
                     cudf::device_span<size_type const> right_indices,
                     std::span<transform_input const> inputs,
                     std::optional<void*> user_data,
                     join_kind join_kind,
                     cuda::stream_ref stream,
                     cudf::memory_resources mr)
{
  auto predicate_results =
    cuda::device_buffer<bool>(stream, mr.get_temporary_mr(), left_indices.size(), cuda::no_init);
  launch_join_filter_kernel(
    kernel, left_indices, right_indices, inputs, predicate_results.data(), user_data, stream, mr);

  if (join_kind != join_kind::FULL_JOIN) {
    return apply_join_semantics(
      left, left_indices, right_indices, predicate_results, join_kind, stream, mr);
  }
  return finalize_full_join(
    apply_join_semantics(
      left, left_indices, right_indices, predicate_results, join_kind::LEFT_JOIN, stream, mr),
    left.num_rows(),
    right.num_rows(),
    std::nullopt,
    stream,
    mr.get_temporary_mr());
}

}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        std::span<uint8_t const> udf,
                        std::variant<udf_source_type, lto_binary_type> udf_type,
                        std::span<transform_input const> extra_inputs,
                        null_aware is_null_aware,
                        std::optional<void*> user_data,
                        join_kind join_kind,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();

  validate_filter_inputs(left, right, left_indices, right_indices, join_kind, extra_inputs);

  if (left_indices.empty()) { return make_empty_result(stream, mr); }

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
  // Extra columns are read at the pair index.
  for (auto const& input : extra_inputs) {
    inputs.push_back(input);
    table_sources.emplace_back(2);
  }

  auto kernel = build_join_filter_kernel(
    udf, udf_type, inputs, table_sources, user_data.has_value(), is_null_aware == null_aware::YES);

  return evaluate_join_filter(
    kernel, left, right, left_indices, right_indices, inputs, user_data, join_kind, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        ast::expression const& predicate,
                        cudf::lto_fragments fragments,
                        join_kind join_kind,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();

  validate_filter_inputs(left, right, left_indices, right_indices, join_kind);

  if (left_indices.empty()) { return make_empty_result(stream, mr); }

  // Convert AST predicate to JIT code
  auto filter_result = row_ir::ast_converter::filter(row_ir::target::CUDA,
                                                     predicate,
                                                     fragments,
                                                     left,
                                                     right,
                                                     "filter_operation",
                                                     stream,
                                                     mr.get_temporary_mr());

  auto template_args =
    build_join_filter_template_params(cudf::jit::make_input_specs(filter_result.inputs),
                                      filter_result.input_table_sources,
                                      filter_result.user_data.has_value(),
                                      filter_result.is_null_aware == null_aware::YES,
                                      udf_source_type::CUDA);

  // The generated code is compiled as it is, linked with the LTO callees it calls.
  auto const cuda_source = cudf::jit::define_operation(
    filter_result.udf, "GENERIC_JOIN_FILTER_OP", filter_result.udf_expression);

  auto kernel_name = rtcx::reflect_template("cudf::join::jit::filter_join_kernel", template_args);
  auto kernel      = cudf::jit::get_udf_kernel("cudf/cpp/src/join/jit/filter_join_kernel.cu",
                                          kernel_name,
                                          cuda_source,
                                          filter_result.lto_fragments);

  return evaluate_join_filter(kernel,
                              left,
                              right,
                              left_indices,
                              right_indices,
                              filter_result.inputs,
                              filter_result.user_data,
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
                        std::string const& predicate_code,
                        cudf::join_kind join_kind,
                        udf_source_type source_type,
                        null_aware is_null_aware,
                        std::optional<void*> user_data,
                        std::span<transform_input const> extra_inputs,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_jit(
    left,
    right,
    left_indices,
    right_indices,
    {reinterpret_cast<uint8_t const*>(predicate_code.data()), predicate_code.size()},
    source_type == udf_source_type::PTX
      ? std::variant<udf_source_type, lto_binary_type>{lto_binary_type::PTX}
      : std::variant<udf_source_type, lto_binary_type>{udf_source_type::CUDA},
    extra_inputs,
    is_null_aware,
    user_data,
    join_kind,
    stream,
    mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit_lto(cudf::table_view const& left,
                            cudf::table_view const& right,
                            cudf::device_span<size_type const> left_indices,
                            cudf::device_span<size_type const> right_indices,
                            std::span<uint8_t const> udf,
                            cudf::join_kind join_kind,
                            lto_binary_type binary_type,
                            null_aware is_null_aware,
                            std::optional<void*> user_data,
                            std::span<transform_input const> extra_inputs,
                            cuda::stream_ref stream,
                            cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_jit(left,
                                         right,
                                         left_indices,
                                         right_indices,
                                         udf,
                                         binary_type,
                                         extra_inputs,
                                         is_null_aware,
                                         user_data,
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
                        cudf::ast::expression const& predicate,
                        cudf::join_kind join_kind,
                        cudf::lto_fragments fragments,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_join_indices_jit(
    left, right, left_indices, right_indices, predicate, fragments, join_kind, stream, mr);
}

}  // namespace cudf
