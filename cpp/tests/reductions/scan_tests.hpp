/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/host_vector.h>

#include <initializer_list>
#include <type_traits>

template <typename T>
struct TypeParam_to_host_type {
  using type = T;
};

template <>
struct TypeParam_to_host_type<cudf::string_view> {
  using type = std::string;
};

template <>
struct TypeParam_to_host_type<numeric::decimal32> {
  using type = numeric::decimal32::rep;
};

template <>
struct TypeParam_to_host_type<numeric::decimal64> {
  using type = numeric::decimal64::rep;
};

template <>
struct TypeParam_to_host_type<numeric::decimal128> {
  using type = numeric::decimal128::rep;
};

template <typename TypeParam, typename T>
thrust::host_vector<std::string> make_vector(std::initializer_list<T> const& init)
  requires(std::is_same_v<TypeParam, cudf::string_view>)
{ return cudf::test::make_type_param_vector<std::string, T>(init); }

template <typename TypeParam, typename T>
thrust::host_vector<typename TypeParam::rep> make_vector(std::initializer_list<T> const& init)
  requires(cudf::is_fixed_point<TypeParam>())
{ return cudf::test::make_type_param_vector<typename TypeParam::rep, T>(init); }

template <typename TypeParam, typename T>
thrust::host_vector<TypeParam> make_vector(std::initializer_list<T> const& init)
  requires(not(std::is_same_v<TypeParam, cudf::string_view> || cudf::is_fixed_point<TypeParam>()))
{ return cudf::test::make_type_param_vector<TypeParam, T>(init); }

// This is the base test feature
template <typename T>
struct BaseScanTest : public cudf::test::BaseFixture {
  using HostType = typename TypeParam_to_host_type<T>::type;

  std::unique_ptr<cudf::column> make_column(cudf::host_span<HostType const> v,
                                            cudf::host_span<bool const> b = {},
                                            numeric::scale_type scale     = numeric::scale_type{0})
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      auto col = (b.size() > 0) ? cudf::test::strings_column_wrapper(v.begin(), v.end(), b.begin())
                                : cudf::test::strings_column_wrapper(v.begin(), v.end());
      return col.release();
    } else if constexpr (cudf::is_fixed_point<T>()) {
      auto col = (b.size() > 0) ? cudf::test::fixed_point_column_wrapper<typename T::rep>(
                                    v.begin(), v.end(), b.begin(), scale)
                                : cudf::test::fixed_point_column_wrapper<typename T::rep>(
                                    v.begin(), v.end(), scale);
      return col.release();
    } else {
      auto col = (b.size() > 0)
                   ? cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end(), b.begin())
                   : cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end());
      return col.release();
    }
  }
};
