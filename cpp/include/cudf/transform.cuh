/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/jit/column_accessor.cuh>
#include <cudf/detail/jit/column_device_view_wrappers.cuh>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/transform_kernel.cuh>
#include <cudf/detail/type_list.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cudf {
namespace detail {

template <typename T>
bool transform_type_matches(column_view const& input)
{
  if constexpr (is_dictionary_encoded<T>) {
    return input.type().id() == type_id::DICTIONARY32 &&
           input.child(dictionary_indices_column_index).type().id() ==
             type_to_id<typename T::index_type>() &&
           input.child(dictionary_keys_column_index).type().id() ==
             type_to_id<typename T::key_type>();
  } else {
    return input.type().id() == type_to_id<T>();
  }
}

template <typename Accessor>
void validate_transform_input(transform_input const& input)
{
  auto const as_view = std::visit(
    [](auto const& value) -> column_view {
      if constexpr (std::is_same_v<std::decay_t<decltype(value)>, scalar_column_view>) {
        return value.as_column_view();
      } else {
        return value;
      }
    },
    input);

  CUDF_EXPECTS(std::holds_alternative<scalar_column_view>(input) == Accessor::as_scalar,
               "CUDA transform input does not match its accessor's scalar specification",
               std::invalid_argument);
  CUDF_EXPECTS(transform_type_matches<typename Accessor::element_type>(as_view),
               "CUDA transform input type does not match its accessor's element type",
               std::invalid_argument);
}

template <typename... Accessor>
void validate_transform_inputs(std::span<transform_input const> inputs, type_list<Accessor...>)
{
  static_assert(
    []<std::size_t... Index>(std::index_sequence<Index...>) {
      return ((Accessor::index == static_cast<int32_t>(Index)) && ...);
    }(std::index_sequence_for<Accessor...>{}),
    "CUDA transform input accessor indices must match their positions");

  CUDF_EXPECTS(inputs.size() == sizeof...(Accessor),
               "CUDA transform input count does not match its accessor count",
               std::invalid_argument);
  (validate_transform_input<Accessor>(inputs[Accessor::index]), ...);
}

template <typename Accessor>
void validate_transform_output(transform_output const& output,
                               std::unique_ptr<column> const* string_offsets)
{
  using element_type               = typename Accessor::element_type;
  constexpr bool is_string_view    = std::is_same_v<element_type, string_view>;
  constexpr bool is_mutable_string = std::is_same_v<element_type, cuda::std::span<char>>;
  constexpr bool is_string         = is_string_view || is_mutable_string;

  constexpr auto expected_physical_type = [] {
    if constexpr (is_string) {
      return type_id::STRING;
    } else {
      return type_to_id<element_type>();
    }
  }();

  CUDF_EXPECTS(output.type.id() == expected_physical_type,
               "CUDA transform output type does not match its accessor's element type",
               std::invalid_argument);

  auto const has_offsets = string_offsets != nullptr && *string_offsets != nullptr;
  CUDF_EXPECTS(!is_string || has_offsets == is_mutable_string,
               "CUDA transform string output type does not match its offsets specification",
               std::invalid_argument);
}

template <typename... Accessor>
void validate_transform_outputs(std::span<transform_output const> outputs,
                                std::span<std::unique_ptr<column> const> string_offsets,
                                type_list<Accessor...>)
{
  CUDF_EXPECTS(outputs.size() == sizeof...(Accessor),
               "CUDA transform output count does not match its accessor count",
               std::invalid_argument);
  CUDF_EXPECTS(string_offsets.empty() || string_offsets.size() == outputs.size(),
               "CUDA transform string offsets must be empty or match the output count",
               std::invalid_argument);
  auto const offsets_at = [&](std::size_t index) {
    return string_offsets.empty() ? nullptr : &string_offsets[index];
  };
  (validate_transform_output<Accessor>(outputs[Accessor::index], offsets_at(Accessor::index)), ...);
}

template <null_aware NullAware, typename InputAccessors, typename OutputAccessors, typename UDF>
CUDF_KERNEL void cuda_transform_kernel(size_type row_size,
                                       bitmask_type const* stencil,
                                       column_device_view_core const* input_columns,
                                       mutable_column_device_view_core const* output_columns,
                                       int32_t* max_error,
                                       UDF udf)
{
  auto operation = [&]<typename Args>(size_type, Args args) {
    auto func = [&](auto... values) {
      if constexpr (cuda::std::is_void_v<decltype(udf(values...))>) {
        udf(values...);
        return errc::SUCCESS;
      } else {
        return static_cast<errc>(udf(values...));
      }
    };
    return cuda::std::apply(func, args);
  };

  cudf::detail::transform_kernel<NullAware == null_aware::YES, InputAccessors, OutputAccessors>(
    row_size, stencil, input_columns, output_columns, max_error, operation);
}

}  // namespace detail

/**
 * @brief Applies a CUDA-compiled callable to every row of one or more columns.
 *
 * This overload provides the same multi-output and null-handling behavior as `multi_transform`,
 * but compiles the callable with the application instead of compiling CUDA source at runtime.
 *
 * For a null-unaware transform, the callable receives pointers to its outputs followed by input
 * values. For a null-aware transform, output pointers and input values use
 * `cuda::std::optional`. The callable may return `void` or `cudf::errc`.
 *
 * Accessors in `InputAccessors` and `OutputAccessors` must be ordered by their zero-based `index`.
 * Input accessors use `column_device_view_core`; fixed-width output accessors use
 * `mutable_column_device_view_core`. String-view output accessors use
 * `detail::jit::mutable_vector_device_view`, while mutable string output accessors use
 * `detail::jit::mutable_strings_column_device_view` and require preallocated offsets.
 *
 * @tparam NullAware Whether the callable handles null inputs and outputs
 * @tparam InputAccessors A `cudf::detail::type_list` of input `column_accessor` types
 * @tparam OutputAccessors A `cudf::detail::type_list` of output `column_accessor` types
 * @tparam UDF CUDA device-callable type
 *
 * @param udf Device-callable transform operation
 * @param inputs Immutable views of the input columns and scalars
 * @param outputs Specifications for the output columns
 * @param string_offsets Optional preallocated offsets for string outputs
 * @param row_size Number of rows, or `std::nullopt` to infer it from column inputs
 * @param stream CUDA stream used for device memory operations and the kernel launch
 * @param mr Device memory resource used to allocate the output columns
 * @return A table containing the transformed output columns
 */
template <null_aware NullAware, typename InputAccessors, typename OutputAccessors, typename UDF>
std::unique_ptr<table> transform(
  UDF&& udf,
  std::span<transform_input const, InputAccessors::size> inputs,
  std::span<transform_output const, OutputAccessors::size> outputs,
  std::vector<std::unique_ptr<column>>&& string_offsets = {},
  std::optional<size_type> row_size                     = std::nullopt,
  rmm::cuda_stream_view stream                          = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr                     = cudf::get_current_device_resource_ref())
{
  using udf_type = std::decay_t<UDF>;
  static_assert(std::is_trivially_copyable_v<udf_type>,
                "A CUDA transform callable must be trivially copyable");

  detail::validate_transform_inputs(inputs, InputAccessors{});
  detail::validate_transform_outputs(
    outputs, std::span<std::unique_ptr<column> const>{string_offsets}, OutputAccessors{});

  auto prepared = detail::prepared_transform{
    NullAware, inputs, outputs, std::move(string_offsets), row_size, stream, mr};
  auto const [transform_size, stencil, input_columns, output_columns, max_error] =
    prepared.kernel_arguments();

  if (transform_size > 0) {
    int min_grid_size;
    int block_size;
    CUDF_CUDA_TRY(cudaOccupancyMaxPotentialBlockSizeWithFlags(
      &min_grid_size,
      &block_size,
      detail::cuda_transform_kernel<NullAware, InputAccessors, OutputAccessors, udf_type>,
      0,
      0,
      cudaOccupancyDefault));
    CUDF_EXPECTS(block_size % cudf::detail::warp_size == 0,
                 "Expected block size to be a multiple of warp size",
                 std::runtime_error);

    detail::cuda_transform_kernel<NullAware, InputAccessors, OutputAccessors>
      <<<min_grid_size, block_size, 0, stream.value()>>>(transform_size,
                                                         stencil,
                                                         input_columns,
                                                         output_columns,
                                                         max_error,
                                                         udf_type{std::forward<UDF>(udf)});
    CUDF_CHECK_CUDA(stream.value());
  }

  return std::move(prepared).finalize();
}


/**
 * @brief Applies a CUDA-compiled unary callable to every input row.
 *
 * @tparam NullAware Whether the callable handles null inputs and outputs
 * @tparam InputElement Element type passed to the callable
 * @tparam OutputElement Element type written by the callable
 * @tparam InputAsScalar Whether the input is a scalar
 * @tparam UDF CUDA device-callable type
 *
 * @return The transformed output column
 */
template <null_aware NullAware,
          typename InputElement,
          typename OutputElement,
          bool InputAsScalar = false,
          typename UDF>
std::unique_ptr<column> unary_transform(
  UDF&& udf,
  transform_input const& input,
  transform_output const& output,
  std::unique_ptr<column> string_offsets = {},
  std::optional<size_type> row_size      = std::nullopt,
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
{
  using input_accessors =
    detail::type_list<detail::jit::column_accessor<0,
                                                   column_device_view_core,
                                                   InputElement,
                                                   InputAsScalar,
                                                   0>>;
  using output_column =
    std::conditional_t<std::is_same_v<OutputElement, cuda::std::span<char>>,
                       detail::jit::mutable_strings_column_device_view,
                       std::conditional_t<std::is_same_v<OutputElement, string_view>,
                                          detail::jit::mutable_vector_device_view,
                                          mutable_column_device_view_core>>;
  using output_accessors =
    detail::type_list<detail::jit::column_accessor<0, output_column, OutputElement, false, 0>>;

  auto offsets = std::vector<std::unique_ptr<column>>{};
  if (string_offsets != nullptr) { offsets.push_back(std::move(string_offsets)); }

  auto result = transform<NullAware, input_accessors, output_accessors>(
    std::forward<UDF>(udf),
    std::span<transform_input const, 1>{&input, 1},
    std::span<transform_output const, 1>{&output, 1},
    std::move(offsets),
    row_size,
    stream,
    mr);
  auto columns = result->release();
  return std::move(columns.front());
}

/**
 * @brief Applies a CUDA-compiled binary callable to every input row.
 *
 * @tparam NullAware Whether the callable handles null inputs and outputs
 * @tparam FirstInputElement First input element type passed to the callable
 * @tparam SecondInputElement Second input element type passed to the callable
 * @tparam OutputElement Element type written by the callable
 * @tparam FirstInputAsScalar Whether the first input is a scalar
 * @tparam SecondInputAsScalar Whether the second input is a scalar
 * @tparam UDF CUDA device-callable type
 *
 * @return The transformed output column
 */
template <null_aware NullAware,
          typename FirstInputElement,
          typename SecondInputElement,
          typename OutputElement,
          bool FirstInputAsScalar  = false,
          bool SecondInputAsScalar = false,
          typename UDF>
std::unique_ptr<column> binary_transform(
  UDF&& udf,
  transform_input const& first_input,
  transform_input const& second_input,
  transform_output const& output,
  std::unique_ptr<column> string_offsets = {},
  std::optional<size_type> row_size      = std::nullopt,
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
{
  using input_accessors =
    detail::type_list<detail::jit::column_accessor<0,
                                                   column_device_view_core,
                                                   FirstInputElement,
                                                   FirstInputAsScalar,
                                                   0>,
                      detail::jit::column_accessor<1,
                                                   column_device_view_core,
                                                   SecondInputElement,
                                                   SecondInputAsScalar,
                                                   0>>;
  using output_column =
    std::conditional_t<std::is_same_v<OutputElement, cuda::std::span<char>>,
                       detail::jit::mutable_strings_column_device_view,
                       std::conditional_t<std::is_same_v<OutputElement, string_view>,
                                          detail::jit::mutable_vector_device_view,
                                          mutable_column_device_view_core>>;
  using output_accessors =
    detail::type_list<detail::jit::column_accessor<0, output_column, OutputElement, false, 0>>;

  transform_input const inputs[] = {first_input, second_input};
  auto offsets                   = std::vector<std::unique_ptr<column>>{};
  if (string_offsets != nullptr) { offsets.push_back(std::move(string_offsets)); }

  auto result = transform<NullAware, input_accessors, output_accessors>(
    std::forward<UDF>(udf),
    inputs,
    std::span<transform_output const, 1>{&output, 1},
    std::move(offsets),
    row_size,
    stream,
    mr);
  auto columns = result->release();
  return std::move(columns.front());
}

/**
 * @brief Applies a CUDA-compiled ternary callable to every input row.
 *
 * @tparam NullAware Whether the callable handles null inputs and outputs
 * @tparam FirstInputElement First input element type passed to the callable
 * @tparam SecondInputElement Second input element type passed to the callable
 * @tparam ThirdInputElement Third input element type passed to the callable
 * @tparam OutputElement Element type written by the callable
 * @tparam FirstInputAsScalar Whether the first input is a scalar
 * @tparam SecondInputAsScalar Whether the second input is a scalar
 * @tparam ThirdInputAsScalar Whether the third input is a scalar
 * @tparam UDF CUDA device-callable type
 *
 * @return The transformed output column
 */
template <null_aware NullAware,
          typename FirstInputElement,
          typename SecondInputElement,
          typename ThirdInputElement,
          typename OutputElement,
          bool FirstInputAsScalar  = false,
          bool SecondInputAsScalar = false,
          bool ThirdInputAsScalar  = false,
          typename UDF>
std::unique_ptr<column> ternary_transform(
  UDF&& udf,
  transform_input const& first_input,
  transform_input const& second_input,
  transform_input const& third_input,
  transform_output const& output,
  std::unique_ptr<column> string_offsets = {},
  std::optional<size_type> row_size      = std::nullopt,
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
{
  using input_accessors =
    detail::type_list<detail::jit::column_accessor<0,
                                                   column_device_view_core,
                                                   FirstInputElement,
                                                   FirstInputAsScalar,
                                                   0>,
                      detail::jit::column_accessor<1,
                                                   column_device_view_core,
                                                   SecondInputElement,
                                                   SecondInputAsScalar,
                                                   0>,
                      detail::jit::column_accessor<2,
                                                   column_device_view_core,
                                                   ThirdInputElement,
                                                   ThirdInputAsScalar,
                                                   0>>;
  using output_column =
    std::conditional_t<std::is_same_v<OutputElement, cuda::std::span<char>>,
                       detail::jit::mutable_strings_column_device_view,
                       std::conditional_t<std::is_same_v<OutputElement, string_view>,
                                          detail::jit::mutable_vector_device_view,
                                          mutable_column_device_view_core>>;
  using output_accessors =
    detail::type_list<detail::jit::column_accessor<0, output_column, OutputElement, false, 0>>;

  transform_input const inputs[] = {first_input, second_input, third_input};
  auto offsets                   = std::vector<std::unique_ptr<column>>{};
  if (string_offsets != nullptr) { offsets.push_back(std::move(string_offsets)); }

  auto result = transform<NullAware, input_accessors, output_accessors>(
    std::forward<UDF>(udf),
    inputs,
    std::span<transform_output const, 1>{&output, 1},
    std::move(offsets),
    row_size,
    stream,
    mr);
  auto columns = result->release();
  return std::move(columns.front());
}

}  // namespace cudf
