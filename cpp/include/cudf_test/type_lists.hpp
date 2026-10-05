/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/type_list_utilities.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <thrust/host_vector.h>

#include <array>
#include <tuple>
#include <type_traits>

/**
 * @file type_lists.hpp
 * @brief Provides centralized type lists for use in Google Test
 * type-parameterized tests.
 *
 * These lists should be used for consistency across tests as well as
 * future-proofing against the addition of any new types in the future.
 */
namespace CUDF_EXPORT cudf {
namespace test {
namespace detail {
template <typename TYPES, std::size_t... Indices>
constexpr std::array<cudf::type_id, sizeof...(Indices)> types_to_ids_impl(
  std::index_sequence<Indices...>)
{ return {{cudf::type_to_id<GetType<TYPES, Indices>>()...}}; }

/**
 * @brief Converts a `Types` list of types into a `std::array` of the
 * corresponding `cudf::type_id`s for each type in the list
 *
 * Example:
 * ```
 * auto array = types_to_ids<Types<int32_t, float>>();
 * array == {type_id::INT32, type_id::FLOAT};
 * ```
 *
 * @tparam TYPES List of types to convert to `type_id`s
 * @return `std::array` of `type_id`s corresponding to each type in `TYPES`
 */
template <typename TYPES>
constexpr auto types_to_ids()
{
  constexpr auto N = GetSize<TYPES>;
  return types_to_ids_impl<TYPES>(std::make_index_sequence<N>());
}

}  // namespace detail

/**
 * @brief Convert numeric values of type T to numeric vector of type TypeParam.
 *
 * This will also convert negative values to positive values if the output type is unsigned.
 *
 * @param init_list Values used to create the output vector
 * @return Vector of TypeParam with the values specified
 */
template <typename TypeParam, typename T>
std::enable_if_t<cudf::is_fixed_width<TypeParam>() && !cudf::is_timestamp_t<TypeParam>::value,
                 thrust::host_vector<TypeParam>>
make_type_param_vector(std::initializer_list<T> const& init_list)
{
  std::vector<T> input{init_list};
  std::vector<TypeParam> vec(init_list.size());
  std::transform(
    std::cbegin(input), std::cend(input), std::begin(vec), [](auto const& e) -> TypeParam {
      if constexpr (std::is_unsigned_v<TypeParam>) { return static_cast<TypeParam>(std::abs(e)); }
      return static_cast<TypeParam>(e);
    });
  return vec;
}

/**
 * @brief Convert numeric values of type T to timestamp vector
 *
 * @param init_list Values used to create the output vector
 * @return Vector of TypeParam with the values specified
 */
template <typename TypeParam, typename T>
std::enable_if_t<cudf::is_timestamp_t<TypeParam>::value, thrust::host_vector<TypeParam>>
make_type_param_vector(std::initializer_list<T> const& init_list)
{
  thrust::host_vector<TypeParam> vec(init_list.size());
  std::transform(std::cbegin(init_list), std::cend(init_list), std::begin(vec), [](auto const& e) {
    return TypeParam{typename TypeParam::duration{e}};
  });
  return vec;
}

/**
 * @brief Convert numeric values of type T to vector of std::string
 *
 * @param init_list Values used to create the output vector
 * @return Vector of TypeParam with the values specified
 */

template <typename TypeParam, typename T>
std::enable_if_t<std::is_same_v<TypeParam, std::string>, thrust::host_vector<std::string>>
make_type_param_vector(std::initializer_list<T> const& init_list)
{
  thrust::host_vector<std::string> vec(init_list.size());
  std::transform(std::cbegin(init_list), std::cend(init_list), std::begin(vec), [](auto const& e) {
    return std::to_string(e);
  });
  return vec;
}

/**
 * @brief Convert the numeric value of type T to a fixed width type of type TypeParam.
 *
 * This function is necessary because some types (such as timestamp types) are not directly
 * constructible from numeric types. This function is offered as a convenience to allow
 * implicitly constructing such objects from numeric values.
 *
 * @param init_value Value used to initialize the fixed width type
 * @return A fixed width type - [u]int32/float/duration etc. of type TypeParam with the
 *         value specified
 */
template <typename TypeParam, typename T>
std::enable_if_t<cudf::is_fixed_width<TypeParam>() && !cudf::is_timestamp_t<TypeParam>::value,
                 TypeParam>
make_type_param_scalar(T const init_value)
{ return static_cast<TypeParam>(init_value); }

/**
 * @brief Convert the timestamp value of type T to a fixed width type of type TypeParam.
 *
 * This function is necessary because some types (such as timestamp types) are not directly
 * constructible from timestamp types. This function is offered as a convenience to allow
 * implicitly constructing such objects from timestamp values.
 *
 * @param init_value Value used to initialize the fixed width type
 * @return A fixed width type - TimeStamp of type TypeParam with the
 *         value specified
 */
template <typename TypeParam, typename T>
std::enable_if_t<cudf::is_timestamp_t<TypeParam>::value, TypeParam> make_type_param_scalar(
  T const init_value)
{ return TypeParam{typename TypeParam::duration(init_value)}; }

/**
 * @brief Convert the numeric value of type T to a string type.
 *
 * This function converts the numeric value of type T to its string representation.
 *
 * @param init_value Value to convert to a string
 * @return string representation of the value
 */
template <typename TypeParam, typename T>
std::enable_if_t<std::is_same_v<TypeParam, std::string>, TypeParam> make_type_param_scalar(
  T const init_value)
{ return std::to_string(init_value); }

/**
 * @brief Type list for all integral types except type bool.
 */
using IntegralTypesNotBool =
  cudf::test::Types<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>;

/**
 * @brief Type list for all integral types.
 */
using IntegralTypes = Concat<IntegralTypesNotBool, cudf::test::Types<bool>>;

/**
 * @brief Provides a list of all floating point types supported in libcudf for
 * use in a GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all floating point types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::FloatingPointTypes);
 * ```
 */
using FloatingPointTypes = cudf::test::Types<float, double>;

/**
 * @brief Provides a list of all numeric types supported in libcudf for use in a
 * GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all numeric types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::NumericTypes);
 * ```
 */
using NumericTypes = Concat<IntegralTypes, FloatingPointTypes>;

/**
 * @brief Provides a list of all timestamp types supported in libcudf for use
 * in a GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all timestamp types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::TimestampTypes);
 * ```
 */
using TimestampTypes =
  cudf::test::Types<timestamp_D, timestamp_s, timestamp_ms, timestamp_us, timestamp_ns>;

/**
 * @brief Provides a list of all duration types supported in libcudf for use
 * in a GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all duration types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::DurationTypes);
 * ```
 */
using DurationTypes =
  cudf::test::Types<duration_D, duration_s, duration_ms, duration_us, duration_ns>;

/**
 * @brief Provides a list of all chrono types supported in libcudf for use in a GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all chrono types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::ChronoTypes);
 * ```
 */
using ChronoTypes = Concat<TimestampTypes, DurationTypes>;

/**
 * @brief Provides a list of all string types supported in libcudf for use in a
 * GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all string types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::StringTypes);
 * ```
 */
using StringTypes = cudf::test::Types<string_view>;

/**
 * @brief Provides a list of all list types supported in libcudf for use in a
 * GTest typed test.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all list types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::ListTypes);
 * ```
 */
using ListTypes = cudf::test::Types<list_view>;

/**
 * @brief Provides a list of all fixed-point element types for use in GTest
 * typed tests.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all fixed-width types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::FixedPointTypes);
 * ```
 */
using FixedPointTypes =
  cudf::test::Types<numeric::decimal32, numeric::decimal64, numeric::decimal128>;

/**
 * @brief Provides a list of all fixed-width element types for use in GTest
 * typed tests.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all fixed-width types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::FixedWidthTypes);
 * ```
 */
using FixedWidthTypes = Concat<NumericTypes, ChronoTypes, FixedPointTypes>;

/**
 * @brief Provides a list of all fixed-width element types except for the
 * fixed-point types for use in GTest typed tests.
 *
 * Certain tests written for fixed-width types don't work for fixed-point as
 * fixed-point types aren't constructible from other fixed-width types
 * because a scale needs to be specified.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all fixed-width types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::FixedWidthTypesWithoutFixedPoint);
 * ```
 */
using FixedWidthTypesWithoutFixedPoint = Concat<NumericTypes, ChronoTypes>;

/**
 * @brief Provides a list of all fixed-width element types except for the
 * chrono types for use in GTest typed tests.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all fixed-width types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::FixedWidthTypesWithoutChrono);
 * ```
 */
using FixedWidthTypesWithoutChrono = Concat<NumericTypes, FixedPointTypes>;

/**
 * @brief Provides a list of sortable types for use in GTest typed tests.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all sortable types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::ComparableTypes);
 * ```
 */
using ComparableTypes = Concat<NumericTypes, ChronoTypes, StringTypes>;

/**
 * @brief Provides a list of all compound types for use in GTest typed tests.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all compound types in libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::CompoundTypes);
 * ```
 */
using CompoundTypes =
  cudf::test::Types<cudf::string_view, cudf::dictionary32, cudf::list_view, cudf::struct_view>;

/**
 * @brief Provides a list of all types supported in libcudf for use in a GTest
 * typed test.
 *
 * @note Currently does not provide any of the "wrapped" types, e.g.,
 * category, etc.
 *
 * Example:
 * ```
 * // Invokes all typed fixture tests for all types supported by libcudf
 * TYPED_TEST_SUITE(MyTypedFixture, cudf::test::AllTypes);
 * ```
 */
using AllTypes = Concat<NumericTypes, ChronoTypes, FixedPointTypes>;

/**
 * @brief `std::array` of all `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 */
static constexpr auto all_type_ids{detail::types_to_ids<AllTypes>()};

/**
 * @brief `std::array` of all numeric `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 */
static constexpr auto numeric_type_ids{detail::types_to_ids<NumericTypes>()};

/**
 * @brief `std::array` of all timestamp `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 */
static constexpr std::array<cudf::type_id, 5> timestamp_type_ids{
  detail::types_to_ids<TimestampTypes>()};

/**
 * @brief `std::array` of all duration `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 */
static constexpr std::array<cudf::type_id, 5> duration_type_ids{
  detail::types_to_ids<DurationTypes>()};

/**
 * @brief `std::array` of all non-numeric `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 */
static constexpr std::array<cudf::type_id, 12> non_numeric_type_ids{
  cudf::type_id::EMPTY,
  cudf::type_id::TIMESTAMP_DAYS,
  cudf::type_id::TIMESTAMP_SECONDS,
  cudf::type_id::TIMESTAMP_MILLISECONDS,
  cudf::type_id::TIMESTAMP_MICROSECONDS,
  cudf::type_id::TIMESTAMP_NANOSECONDS,
  cudf::type_id::DURATION_DAYS,
  cudf::type_id::DURATION_SECONDS,
  cudf::type_id::DURATION_MILLISECONDS,
  cudf::type_id::DURATION_MICROSECONDS,
  cudf::type_id::DURATION_NANOSECONDS,
  cudf::type_id::STRING};

/**
 * @brief `std::array` of all non-fixed-width `cudf::type_id`s
 *
 * This can be used for iterating over `type_id`s for custom testing, or used in
 * GTest value-parameterized tests.
 */
static constexpr std::array<cudf::type_id, 2> non_fixed_width_type_ids{cudf::type_id::EMPTY,
                                                                       cudf::type_id::STRING};

}  // namespace test
}  // namespace CUDF_EXPORT cudf
