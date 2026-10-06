/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/transform/kernel.cuh>
#include <cudf/detail/transform/program_info.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <rtcx/rtcx.hpp>

#include <utility>
#include <vector>

/** @file @brief Compile-time descriptors and instantiation helpers for transform UDFs. */
namespace cudf {

/**
 * @brief Describes a transform input's element type and scalar-broadcasting behavior.
 * @tparam T Element type, or dictionary_element<IndexType, KeyType> for dictionary inputs
 * @tparam IsScalar Whether the input is a scalar column
 */
template <typename T, bool IsScalar = false>
struct transform_input_type {
  using element_type              = T;         ///< Device element type
  static constexpr bool is_scalar = IsScalar;  ///< Whether to broadcast element zero
};

namespace detail {

template <typename T>
struct transform_input_traits : transform_input_type<T> {};

template <typename T, bool IsScalar>
struct transform_input_traits<transform_input_type<T, IsScalar>>
  : transform_input_type<T, IsScalar> {};

// Both widths of string offsets use the same runtime offset accessor. Decimal scales are
// likewise obtained from the runtime device view, rather than being template parameters.
template <typename T>
transform_input_spec transform_element_spec()
{
  if constexpr (is_dictionary_encoded<T>) {
    return {.type     = type_id::DICTIONARY32,
            .children = {transform_element_spec<typename T::index_type>(),
                         transform_element_spec<typename T::key_type>()}};
  } else {
    static_assert(is_fixed_width<T>() || cuda::std::is_same_v<T, string_view>,
                  "Unsupported transform input element type");
    return {.type = type_to_id<T>()};
  }
}

template <typename T>
transform_input_spec transform_input_metadata()
{
  auto result      = transform_element_spec<typename transform_input_traits<T>::element_type>();
  result.is_scalar = transform_input_traits<T>::is_scalar;
  return result;
}

template <typename T>
transform_output_spec transform_output_metadata()
{
  if constexpr (cuda::std::is_same_v<T, cuda::std::span<char>>) {
    return {.type = type_id::STRING, .has_string_offsets = true};
  } else {
    static_assert(is_fixed_width<T>() || cuda::std::is_same_v<T, string_view>,
                  "Unsupported transform output element type");
    return {.type = type_to_id<T>()};
  }
}

template <typename Inputs, typename Indices>
struct transform_input_accessors;

template <typename... Inputs, std::size_t... I>
struct transform_input_accessors<type_list<Inputs...>, std::index_sequence<I...>> {
  using type = type_list<column_accessor<I,
                                         column_device_view_core,
                                         typename transform_input_traits<Inputs>::element_type,
                                         transform_input_traits<Inputs>::is_scalar,
                                         0>...>;
};

template <typename Outputs, typename Indices>
struct transform_output_accessors;

template <typename... Outputs, std::size_t... I>
struct transform_output_accessors<type_list<Outputs...>, std::index_sequence<I...>> {
  template <typename T>
  using column_type = cuda::std::conditional_t<
    cuda::std::is_same_v<T, string_view>,
    mutable_vector_device_view,
    cuda::std::conditional_t<cuda::std::is_same_v<T, cuda::std::span<char>>,
                             mutable_strings_column_device_view,
                             mutable_column_device_view_core>>;
  using type = type_list<column_accessor<I, column_type<Outputs>, Outputs, false, 0>...>;
};

template <typename Operator>
CUDF_KERNEL void instantiated_transform_entry(size_type rows,
                                              bitmask_type const* stencil,
                                              void* user_data,
                                              column_device_view_core const* inputs,
                                              mutable_column_device_view_core const* outputs,
                                              int32_t* error)
{
  using input_accessors =
    typename transform_input_accessors<typename Operator::Inputs,
                                       std::make_index_sequence<Operator::Inputs::size>>::type;
  using output_accessors =
    typename transform_output_accessors<typename Operator::Outputs,
                                        std::make_index_sequence<Operator::Outputs::size>>::type;
  transform_kernel<Operator::NullAware, Operator::UserData, input_accessors, output_accessors>(
    rows, stencil, user_data, inputs, outputs, error, typename Operator::Function{});
}

template <typename Operator, typename Inputs, typename Outputs>
struct transform_instantiation;

template <typename Operator, typename... Inputs, typename... Outputs>
struct transform_instantiation<Operator, type_list<Inputs...>, type_list<Outputs...>> {
  template <typename ProgramInfo = transform_program_metadata, typename Entry>
  static ProgramInfo make(Entry entry, std::string tag)
  {
    cudaKernel_t handle;
    CUDF_CUDA_TRY(cudaGetKernel(&handle, entry));
    return ProgramInfo{.tag        = std::move(tag),
                       .inputs     = {transform_input_metadata<Inputs>()...},
                       .outputs    = {transform_output_metadata<Outputs>()...},
                       .null_aware = Operator::NullAware,
                       .user_data  = Operator::UserData,
                       .kernel     = rtcx::kernel_ref{static_cast<CUkernel>(handle)}};
  }
};

/**
 * @brief Instantiates a UDF implementation and generates its erased kernel and dispatch metadata.
 *
 * The operator must provide `Inputs`, `Outputs`, and `Function` type aliases and static constexpr
 * bool `NullAware` and `UserData` members. Inputs are a `type_list` of element types (column
 * inputs) or `transform_input_type<T, true>` (scalar inputs). Outputs are a `type_list` of element
 * types; use `string_view` for string-view output and `cuda::std::span<char>` for preallocated
 * strings. `Function` must be default-constructible and device-invocable with output pointers
 * followed by input values. Null-aware UDFs receive optional values and pointers to optional
 * outputs. User-data UDFs additionally receive `void*` and the row index before the outputs. The
 * return type is void or convertible to cudf::errc, as with source-based transforms.
 *
 * This helper must be compiled by a CUDA compiler. Compilation instantiates cuDF's implementation;
 * calling the helper only constructs the host entry and metadata, with no runtime compilation.
 *
 * @tparam UserOperator Compile-time UDF descriptor
 * @return Erased dispatch metadata for the instantiated transform
 */
template <typename UserOperator>
[[nodiscard]] auto instantiate_transform_program(std::string tag)
{
  return transform_instantiation<
    UserOperator,
    typename UserOperator::Inputs,
    typename UserOperator::Outputs>::make(instantiated_transform_entry<UserOperator>,
                                          std::move(tag));
}

}  // namespace detail

template <typename UserOperator>
transform_program_info transform_program_info::make(std::string tag)
{
  return from_metadata(detail::instantiate_transform_program<UserOperator>(std::move(tag)));
}

}  // namespace cudf
