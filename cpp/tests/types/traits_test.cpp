/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/list_view.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <tuple>

template <typename Tuple, typename F, std::size_t... Indices>
void tuple_for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>)
{
  (void)std::initializer_list<int>{
    ((void)(f(std::get<Indices>(std::forward<Tuple>(tuple)))), int{})...};
}

template <typename F, typename... Args>
void tuple_for_each(std::tuple<Args...> const& tuple, F&& f)
{
  tuple_for_each_impl(tuple, std::forward<F>(f), std::index_sequence_for<Args...>{});
}

class TraitsTest : public ::testing::Test {};

template <typename T>
class TypedTraitsTest : public TraitsTest {};

TYPED_TEST_SUITE(TypedTraitsTest, cudf::test::AllTypes);

TEST_F(TraitsTest, NumericDataTypesAreNumeric)
{
  EXPECT_TRUE(
    std::all_of(cudf::test::numeric_type_ids.begin(),
                cudf::test::numeric_type_ids.end(),
                [](cudf::type_id type) { return cudf::is_numeric(cudf::data_type{type}); }));
}

TEST_F(TraitsTest, TimestampDataTypesAreNotNumeric)
{
  EXPECT_TRUE(
    std::none_of(cudf::test::timestamp_type_ids.begin(),
                 cudf::test::timestamp_type_ids.end(),
                 [](cudf::type_id type) { return cudf::is_numeric(cudf::data_type{type}); }));
}

TEST_F(TraitsTest, NumericDataTypesAreNotTimestamps)
{
  EXPECT_TRUE(
    std::none_of(cudf::test::numeric_type_ids.begin(),
                 cudf::test::numeric_type_ids.end(),
                 [](cudf::type_id type) { return cudf::is_timestamp(cudf::data_type{type}); }));
}

TEST_F(TraitsTest, TimestampDataTypesAreTimestamps)
{
  EXPECT_TRUE(
    std::all_of(cudf::test::timestamp_type_ids.begin(),
                cudf::test::timestamp_type_ids.end(),
                [](cudf::type_id type) { return cudf::is_timestamp(cudf::data_type{type}); }));
}

TYPED_TEST(TypedTraitsTest, RelationallyComparable)
{
  // All the test types should be comparable with themselves
  bool comparable = cudf::is_relationally_comparable<TypeParam, TypeParam>();
  EXPECT_TRUE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotRelationallyComparable)
{
  // No type should be comparable with an empty dummy type
  struct foo {};
  bool comparable = cudf::is_relationally_comparable<foo, TypeParam>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_relationally_comparable<TypeParam, foo>();
  EXPECT_FALSE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotRelationallyComparableWithList)
{
  bool comparable = cudf::is_relationally_comparable<TypeParam, cudf::list_view>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_relationally_comparable<cudf::list_view, cudf::list_view>();
  EXPECT_FALSE(comparable);
}

TYPED_TEST(TypedTraitsTest, EqualityComparable)
{
  // All the test types should be comparable with themselves
  bool comparable = cudf::is_equality_comparable<TypeParam, TypeParam>();
  EXPECT_TRUE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotEqualityComparable)
{
  // No type should be comparable with an empty dummy type
  struct foo {};
  bool comparable = cudf::is_equality_comparable<foo, TypeParam>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_equality_comparable<TypeParam, foo>();
  EXPECT_FALSE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotEqualityComparableWithList)
{
  bool comparable = cudf::is_equality_comparable<TypeParam, cudf::list_view>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_equality_comparable<cudf::list_view, cudf::list_view>();
  EXPECT_FALSE(comparable);
}

// TODO: Tests for is_compound, is_fixed_width

template <typename T>
class CvQualifiedTraitsTest : public TraitsTest {};

using DispatchedTypes = cudf::test::Concat<cudf::test::FixedWidthTypes, cudf::test::CompoundTypes>;
TYPED_TEST_SUITE(CvQualifiedTraitsTest, DispatchedTypes);

template <typename T, typename U>
void expect_same_type_category()
{
  EXPECT_EQ(cudf::is_numeric<T>(), cudf::is_numeric<U>());
  EXPECT_EQ(cudf::is_index_type<T>(), cudf::is_index_type<U>());
  EXPECT_EQ(cudf::is_signed<T>(), cudf::is_signed<U>());
  EXPECT_EQ(cudf::is_unsigned<T>(), cudf::is_unsigned<U>());
  EXPECT_EQ(cudf::is_integral<T>(), cudf::is_integral<U>());
  EXPECT_EQ(cudf::is_integral_not_bool<T>(), cudf::is_integral_not_bool<U>());
  EXPECT_EQ(cudf::is_numeric_not_bool<T>(), cudf::is_numeric_not_bool<U>());
  EXPECT_EQ(cudf::is_floating_point<T>(), cudf::is_floating_point<U>());
  EXPECT_EQ(cudf::is_byte<T>(), cudf::is_byte<U>());
  EXPECT_EQ(cudf::is_boolean<T>(), cudf::is_boolean<U>());
  EXPECT_EQ(cudf::is_timestamp<T>(), cudf::is_timestamp<U>());
  EXPECT_EQ(cudf::is_timestamp_t<T>::value, cudf::is_timestamp_t<U>::value);
  EXPECT_EQ(cudf::is_fixed_point<T>(), cudf::is_fixed_point<U>());
  EXPECT_EQ(cudf::is_duration<T>(), cudf::is_duration<U>());
  EXPECT_EQ(cudf::is_duration_t<T>::value, cudf::is_duration_t<U>::value);
  EXPECT_EQ(cudf::is_chrono<T>(), cudf::is_chrono<U>());
  EXPECT_EQ(cudf::is_rep_layout_compatible<T>(), cudf::is_rep_layout_compatible<U>());
  EXPECT_EQ(cudf::is_dictionary<T>(), cudf::is_dictionary<U>());
  EXPECT_EQ(cudf::is_fixed_width<T>(), cudf::is_fixed_width<U>());
  EXPECT_EQ(cudf::is_compound<T>(), cudf::is_compound<U>());
  EXPECT_EQ(cudf::is_nested<T>(), cudf::is_nested<U>());
}

TYPED_TEST(CvQualifiedTraitsTest, TypeCategoryIgnoresCvQualifiers)
{
  using T = TypeParam;
  expect_same_type_category<T, T const>();
  expect_same_type_category<T, T volatile>();
  expect_same_type_category<T, T const volatile>();
}

TYPED_TEST(CvQualifiedTraitsTest, DictionaryKeyIgnoresConst)
{
  using T = TypeParam;
  EXPECT_EQ(cudf::is_dictionary_key<T>(), cudf::is_dictionary_key<T const>());
}

TEST_F(TraitsTest, CvQualifiedBoolIsNotAnInteger)
{
  EXPECT_FALSE(cudf::is_index_type<bool const>());
  EXPECT_FALSE(cudf::is_integral_not_bool<bool const>());
  EXPECT_FALSE(cudf::is_numeric_not_bool<bool const>());
  EXPECT_TRUE(cudf::is_boolean<bool const>());
  EXPECT_FALSE(cudf::is_index_type<bool volatile>());
  EXPECT_FALSE(cudf::is_integral_not_bool<bool volatile>());
  EXPECT_FALSE(cudf::is_numeric_not_bool<bool volatile>());
  EXPECT_TRUE(cudf::is_boolean<bool volatile>());
}

CUDF_TEST_PROGRAM_MAIN()
