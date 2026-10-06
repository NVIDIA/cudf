/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/filter_join_kernel.cuh>
#include <cudf/detail/join/program_info.hpp>
#include <cudf/detail/transform/instantiate_program.cuh>
#include <cudf/join/join.hpp>

namespace cudf {
namespace detail {

template <typename Inputs, std::size_t LeftInputCount, typename Indices>
struct filter_input_accessors;

template <typename... Inputs, std::size_t LeftInputCount, std::size_t... I>
struct filter_input_accessors<type_list<Inputs...>, LeftInputCount, std::index_sequence<I...>> {
  static_assert((!transform_input_traits<Inputs>::is_scalar && ...),
                "Join filter inputs must be columns");
  static_assert((!is_dictionary_encoded<typename transform_input_traits<Inputs>::element_type> &&
                 ...),
                "Join filter inputs do not support dictionary columns");
  using type = type_list<column_accessor<I,
                                         column_device_view_core,
                                         typename transform_input_traits<Inputs>::element_type,
                                         false,
                                         (I < LeftInputCount ? 0 : 1)>...>;
};

template <typename Operator, std::size_t LeftInputCount>
CUDF_KERNEL void instantiated_filter_entry(size_type rows,
                                           size_type const* left_indices,
                                           size_type const* right_indices,
                                           column_device_view_core const* inputs,
                                           bool* results,
                                           void* user_data)
{
  static_assert(LeftInputCount <= Operator::Inputs::size, "Invalid left input count");
  static_assert(cuda::std::is_same_v<typename Operator::Outputs, type_list<bool>>,
                "Join filters require exactly one Boolean output");
  static_assert(!Operator::UserData, "Precompiled join filters do not support user data");
  using accessors =
    typename filter_input_accessors<typename Operator::Inputs,
                                    LeftInputCount,
                                    std::make_index_sequence<Operator::Inputs::size>>::type;
  filter_join_kernel<Operator::UserData, Operator::NullAware, accessors>(
    rows, left_indices, right_indices, inputs, results, user_data, typename Operator::Function{});
}

template <typename Operator, std::size_t LeftInputCount>
auto instantiate_filter_join_program(std::string tag)
{
  auto result =
    transform_instantiation<Operator, typename Operator::Inputs, typename Operator::Outputs>::
      template make<filter_join_program_metadata>(
        instantiated_filter_entry<Operator, LeftInputCount>, std::move(tag));
  result.left_input_count = LeftInputCount;
  return result;
}

}  // namespace detail

template <typename UserOperator, std::size_t LeftInputCount>
filter_join_program_info filter_join_program_info::make(std::string tag)
{
  return from_metadata(
    detail::instantiate_filter_join_program<UserOperator, LeftInputCount>(std::move(tag)));
}

}  // namespace cudf
