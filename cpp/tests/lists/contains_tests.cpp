/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace {
template <typename T>
auto create_scalar_search_key(T const& value)
  requires(cudf::is_numeric<T>())
{
  auto search_key = cudf::make_numeric_scalar(cudf::data_type{cudf::type_to_id<T>()});
  search_key->set_valid_async(true);
  static_cast<cudf::scalar_type_t<T>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T>
auto create_scalar_search_key(std::string const& value)
  requires(std::is_same_v<T, std::string>)
{
  return cudf::make_string_scalar(value);
}

template <typename T>
auto create_scalar_search_key(typename T::rep const& value)
  requires(cudf::is_timestamp<T>())
{
  auto search_key = cudf::make_timestamp_scalar(cudf::data_type{cudf::type_to_id<T>()});
  search_key->set_valid_async(true);
  static_cast<cudf::scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename T>
auto create_scalar_search_key(typename T::rep const& value)
  requires(cudf::is_duration<T>())
{
  auto search_key = cudf::make_duration_scalar(cudf::data_type{cudf::type_to_id<T>()});
  search_key->set_valid_async(true);
  static_cast<cudf::scalar_type_t<typename T::rep>*>(search_key.get())->set_value(value);
  return search_key;
}

template <typename... Args>
auto make_struct_scalar(Args&&... args)
{
  return cudf::struct_scalar(std::vector<cudf::column_view>{std::forward<Args>(args)...});
}

template <typename T>
auto create_null_search_key()
  requires(cudf::is_numeric<T>())
{
  auto search_key = cudf::make_numeric_scalar(cudf::data_type{cudf::type_to_id<T>()});
  search_key->set_valid_async(false);
  return search_key;
}

template <typename T>
auto create_null_search_key()
  requires(cudf::is_timestamp<T>())
{
  auto search_key = cudf::make_timestamp_scalar(cudf::data_type{cudf::type_to_id<T>()});
  search_key->set_valid_async(false);
  return search_key;
}

template <typename T>
auto create_null_search_key()
  requires(cudf::is_duration<T>())
{
  auto search_key = cudf::make_duration_scalar(cudf::data_type{cudf::type_to_id<T>()});
  search_key->set_valid_async(false);
  return search_key;
}

}  // namespace

auto constexpr X          = int32_t{0};           // Placeholder for nulls.
auto constexpr ABSENT     = cudf::size_type{-1};  // Index when key is not found in a list.
auto constexpr FIND_FIRST = cudf::lists::duplicate_find_option::FIND_FIRST;
auto constexpr FIND_LAST  = cudf::lists::duplicate_find_option::FIND_LAST;

using bools_col   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;
using indices_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

using cudf::test::iterators::all_nulls;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

using ContainsTestTypes = cudf::test::
  Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes, cudf::test::ChronoTypes>;

struct ContainsTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedContainsTest : public ContainsTest {};

TYPED_TEST_SUITE(TypedContainsTest, ContainsTestTypes);

TYPED_TEST(TypedContainsTest, ScalarKeyWithNoNulls)
{
  using T = TypeParam;

  auto const search_space_col = cudf::test::lists_column_wrapper<T, int32_t>{{0, 1, 2, 1},
                                                                             {3, 4, 5},
                                                                             {6, 7, 8},
                                                                             {9, 0, 1, 3, 1},
                                                                             {2, 3, 4},
                                                                             {5, 6, 7},
                                                                             {8, 9, 0},
                                                                             {},
                                                                             {1, 2, 1, 3},
                                                                             {}};
  auto const search_space     = cudf::lists_column_view{search_space_col};
  auto search_key_one         = create_scalar_search_key<T>(1);

  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space, *search_key_one);
    auto expected = bools_col{1, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto result   = cudf::lists::contains_nulls(search_space);
    auto expected = bools_col{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space, *search_key_one, FIND_FIRST);
    auto expected = indices_col{1, ABSENT, ABSENT, 2, ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space, *search_key_one, FIND_LAST);
    auto expected = indices_col{3, ABSENT, ABSENT, 4, ABSENT, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedContainsTest, ScalarKeyWithNullLists)
{
  // Test List columns that have NULL list rows.
  using T = TypeParam;

  auto const search_space_col = cudf::test::lists_column_wrapper<T, int32_t>{{{0, 1, 2, 1},
                                                                              {3, 4, 5},
                                                                              {6, 7, 8},
                                                                              {},
                                                                              {9, 0, 1, 3, 1},
                                                                              {2, 3, 4},
                                                                              {5, 6, 7},
                                                                              {8, 9, 0},
                                                                              {},
                                                                              {1, 2, 2, 3},
                                                                              {}},
                                                                             nulls_at({3, 10})};
  auto const search_space     = cudf::lists_column_view{search_space_col};
  auto search_key_one         = create_scalar_search_key<T>(1);
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space, *search_key_one);
    auto expected = bools_col{{1, 0, 0, X, 1, 0, 0, 0, 0, 1, X}, nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto result   = cudf::lists::contains_nulls(search_space);
    auto expected = bools_col{{0, 0, 0, X, 0, 0, 0, 0, 0, 0, X}, nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space, *search_key_one, FIND_FIRST);
    auto expected = indices_col{{1, ABSENT, ABSENT, X, 2, ABSENT, ABSENT, ABSENT, ABSENT, 0, X},
                                nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space, *search_key_one, FIND_LAST);
    auto expected = indices_col{{3, ABSENT, ABSENT, X, 4, ABSENT, ABSENT, ABSENT, ABSENT, 0, X},
                                nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedContainsTest, SlicedLists)
{
  // Test sliced List columns.
  using T = TypeParam;

  auto search_space = cudf::test::lists_column_wrapper<T, int32_t>{{{0, 1, 2, 1},
                                                                    {3, 4, 5},
                                                                    {6, 7, 8},
                                                                    {},
                                                                    {9, 0, 1, 3, 1},
                                                                    {2, 3, 4},
                                                                    {5, 6, 7},
                                                                    {8, 9, 0},
                                                                    {},
                                                                    {1, 2, 1, 3},
                                                                    {}},
                                                                   nulls_at({3, 10})};

  {
    // First Slice.
    auto sliced_column_1 = cudf::slice(search_space, {1, 8}, cudf::get_default_stream()).front();
    auto search_key_one  = create_scalar_search_key<T>(1);
    {
      // CONTAINS
      auto result          = cudf::lists::contains(sliced_column_1, *search_key_one);
      auto expected_result = bools_col{{0, 0, X, 1, 0, 0, 0}, null_at(2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // CONTAINS NULLS
      auto result          = cudf::lists::contains_nulls(sliced_column_1);
      auto expected_result = bools_col{{0, 0, X, 0, 0, 0, 0}, null_at(2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // FIND_FIRST
      auto result = cudf::lists::index_of(sliced_column_1, *search_key_one, FIND_FIRST);
      auto expected_result =
        indices_col{{ABSENT, ABSENT, 0, 2, ABSENT, ABSENT, ABSENT}, null_at(2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // FIND_LAST
      auto result = cudf::lists::index_of(sliced_column_1, *search_key_one, FIND_LAST);
      auto expected_result =
        indices_col{{ABSENT, ABSENT, 0, 4, ABSENT, ABSENT, ABSENT}, null_at(2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
  }

  {
    // Second Slice.
    auto sliced_column_2 = cudf::slice(search_space, {3, 10}, cudf::get_default_stream()).front();
    auto search_key_one  = create_scalar_search_key<T>(1);
    {
      // CONTAINS
      auto result          = cudf::lists::contains(sliced_column_2, *search_key_one);
      auto expected_result = bools_col{{X, 1, 0, 0, 0, 0, 1}, null_at(0)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // CONTAINS NULLS
      auto result          = cudf::lists::contains_nulls(sliced_column_2);
      auto expected_result = bools_col{{X, 0, 0, 0, 0, 0, 0}, null_at(0)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // FIND_FIRST
      auto result          = cudf::lists::index_of(sliced_column_2, *search_key_one, FIND_FIRST);
      auto expected_result = indices_col{{0, 2, ABSENT, ABSENT, ABSENT, ABSENT, 0}, null_at(0)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
    {
      // FIND_LAST
      auto result          = cudf::lists::index_of(sliced_column_2, *search_key_one, FIND_LAST);
      auto expected_result = indices_col{{0, 4, ABSENT, ABSENT, ABSENT, ABSENT, 2}, null_at(0)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result, result->view());
    }
  }
}

TYPED_TEST(TypedContainsTest, ScalarKeyNonNullListsWithNullValues)
{
  // Test List columns that have no NULL list rows, but NULL elements in some list rows.
  using T = TypeParam;

  auto numerals = cudf::test::fixed_width_column_wrapper<T>{
    {X, 1, 2, X, 4, 5, X, 7, 8, X, X, 1, 2, X, 1}, nulls_at({0, 3, 6, 9, 10, 13})};
  auto search_space = cudf::make_lists_column(
    8, indices_col{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(), numerals.release(), 0, {});
  // Search space: [ [x], [1,2], [x,4,5,x], [], [], [7,8,x], [x], [1,2,x,1] ]
  auto search_key_one = create_scalar_search_key<T>(1);
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), *search_key_one);
    auto expected = bools_col{0, 1, 0, 0, 0, 0, 0, 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto result   = cudf::lists::contains_nulls(search_space->view());
    auto expected = bools_col{1, 0, 1, 0, 0, 1, 1, 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_FIRST);
    auto expected = indices_col{ABSENT, 0, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_LAST);
    auto expected = indices_col{ABSENT, 0, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 3};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedContainsTest, ScalarKeysWithNullsInLists)
{
  using T = TypeParam;

  auto numerals = cudf::test::fixed_width_column_wrapper<T>{
    {X, 1, 2, X, 4, 5, X, 7, 8, X, X, 1, 2, X, 1}, nulls_at({0, 3, 6, 9, 10, 13})};
  auto input_null_mask_iter = null_at(4);

  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8);
  auto search_space = cudf::make_lists_column(8,
                                              indices_col{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
                                              numerals.release(),
                                              null_count,
                                              std::move(null_mask));

  // Search space: [ [x], [1,2], [x,4,5,x], [], x, [7,8,x], [x], [1,2,x,1] ]
  auto search_key_one = create_scalar_search_key<T>(1);
  {
    // CONTAINS.
    auto result   = cudf::lists::contains(search_space->view(), *search_key_one);
    auto expected = bools_col{{0, 1, 0, 0, X, 0, 0, 1}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS.
    auto result   = cudf::lists::contains_nulls(search_space->view());
    auto expected = bools_col{{1, 0, 1, 0, X, 1, 1, 1}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST.
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_FIRST);
    auto expected = indices_col{{ABSENT, 0, ABSENT, ABSENT, X, ABSENT, ABSENT, 0}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST.
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_LAST);
    auto expected = indices_col{{ABSENT, 0, ABSENT, ABSENT, X, ABSENT, ABSENT, 3}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TEST_F(ContainsTest, BoolScalarWithNullsInLists)
{
  using T = bool;

  auto numerals = cudf::test::fixed_width_column_wrapper<T>{
    {X, 1, 1, X, 1, 1, X, 1, 1, X, X, 1, 1, X, 1}, nulls_at({0, 3, 6, 9, 10, 13})};
  auto input_null_mask_iter = null_at(4);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8);
  auto search_space = cudf::make_lists_column(
    8,
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    numerals.release(),
    null_count,
    std::move(null_mask));

  // Search space: [ [x], [1,1], [x,1,1,x], [], x, [1,1,x], [x], [1,1,x,1] ]
  auto search_key_one = create_scalar_search_key<T>(true);
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), *search_key_one);
    auto expected = bools_col{{0, 1, 1, 0, X, 1, 0, 1}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto result   = cudf::lists::contains_nulls(search_space->view());
    auto expected = bools_col{{1, 0, 1, 0, X, 1, 1, 1}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST.
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_FIRST);
    auto expected = indices_col{{ABSENT, 0, 1, ABSENT, X, 0, ABSENT, 0}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST.
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_LAST);
    auto expected = indices_col{{ABSENT, 1, 2, ABSENT, X, 1, ABSENT, 3}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TEST_F(ContainsTest, StringScalarWithNullsInLists)
{
  using T = std::string;

  auto strings = cudf::test::strings_column_wrapper{
    {"X", "1", "2", "X", "4", "5", "X", "7", "8", "X", "X", "1", "2", "X", "1"},
    nulls_at({0, 3, 6, 9, 10, 13})};
  auto input_null_mask_iter = null_at(4);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8);
  auto search_space = cudf::make_lists_column(8,
                                              indices_col{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
                                              strings.release(),
                                              null_count,
                                              std::move(null_mask));

  // Search space: [ [x], [1,2], [x,4,5,x], [], x, [7,8,x], [x], [1,2,x,1] ]
  auto search_key_one = create_scalar_search_key<T>("1");
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), *search_key_one);
    auto expected = bools_col{{0, 1, 0, 0, X, 0, 0, 1}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto result   = cudf::lists::contains_nulls(search_space->view());
    auto expected = bools_col{{1, 0, 1, 0, X, 1, 1, 1}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST.
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_FIRST);
    auto expected = indices_col{{ABSENT, 0, ABSENT, ABSENT, X, ABSENT, ABSENT, 0}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST.
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_LAST);
    auto expected = indices_col{{ABSENT, 0, ABSENT, ABSENT, X, ABSENT, ABSENT, 3}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedContainsTest, ScalarNullSearchKey)
{
  using T = TypeParam;

  auto search_space = cudf::test::lists_column_wrapper<T, int32_t>{{{0, 1, 2},
                                                                    {3, 4, 5},
                                                                    {6, 7, 8},
                                                                    {},
                                                                    {9, 0, 1},
                                                                    {2, 3, 4},
                                                                    {5, 6, 7},
                                                                    {8, 9, 0},
                                                                    {},
                                                                    {1, 2, 3},
                                                                    {}},
                                                                   nulls_at({3, 10})}
                        .release();
  auto search_key_null = create_null_search_key<T>();
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), *search_key_null);
    auto expected = bools_col{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, all_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_null, FIND_FIRST);
    auto expected = indices_col{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, all_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_null, FIND_LAST);
    auto expected = indices_col{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, all_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TEST_F(ContainsTest, ScalarTypeRelatedExceptions)
{
  {
    // Nested types unsupported.
    auto list_of_lists = cudf::test::lists_column_wrapper<int32_t>{
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3},
       {4, 5, 6}}}.release();
    auto skey = create_scalar_search_key<int32_t>(10);
    EXPECT_THROW(cudf::lists::contains(list_of_lists->view(), *skey), cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_lists->view(), *skey, FIND_FIRST),
                 cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_lists->view(), *skey, FIND_LAST),
                 cudf::data_type_error);
  }
  {
    // Search key must match list elements in type.
    auto list_of_ints =
      cudf::test::lists_column_wrapper<int32_t>{
        {0, 1, 2},
        {3, 4, 5},
      }
        .release();
    auto skey = create_scalar_search_key<std::string>("Hello, World!");
    EXPECT_THROW(cudf::lists::contains(list_of_ints->view(), *skey), cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_ints->view(), *skey, FIND_FIRST),
                 cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_ints->view(), *skey, FIND_LAST),
                 cudf::data_type_error);
  }
}

template <typename T>
struct TypedVectorContainsTest : public ContainsTest {};

using VectorTestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(TypedVectorContainsTest, VectorTestTypes);

TYPED_TEST(TypedVectorContainsTest, VectorKeysWithNoNulls)
{
  using T = TypeParam;

  auto search_space = cudf::test::lists_column_wrapper<T, int32_t>{
    {0, 1, 2, 1},
    {3, 4, 5},
    {6, 7, 8},
    {9, 0, 1, 3, 1},
    {2, 3, 4},
    {5, 6, 7},
    {8, 9, 0},
    {},
    {1, 2, 3, 3},
    {}}.release();

  auto search_key =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_key);
    auto expected = bools_col{1, 0, 0, 1, 1, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_key, FIND_FIRST);
    auto expected = indices_col{1, ABSENT, ABSENT, 2, 0, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_key, FIND_LAST);
    auto expected = indices_col{3, ABSENT, ABSENT, 4, 0, ABSENT, ABSENT, ABSENT, 3, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedVectorContainsTest, VectorWithNullLists)
{
  // Test List columns that have NULL list rows.

  using T = TypeParam;

  auto search_space = cudf::test::lists_column_wrapper<T, int32_t>{{{0, 1, 2, 1},
                                                                    {3, 4, 5},
                                                                    {6, 7, 8},
                                                                    {},
                                                                    {9, 0, 1, 3, 1},
                                                                    {2, 3, 4},
                                                                    {5, 6, 7},
                                                                    {8, 9, 0},
                                                                    {},
                                                                    {1, 2, 3, 3},
                                                                    {}},
                                                                   nulls_at({3, 10})}
                        .release();

  auto search_keys =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2};

  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_keys);
    auto expected = bools_col{{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}, nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_FIRST);
    auto expected = indices_col{{1, ABSENT, ABSENT, X, ABSENT, 1, ABSENT, ABSENT, ABSENT, 0, X},
                                nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_LAST);
    auto expected = indices_col{{3, ABSENT, ABSENT, X, ABSENT, 1, ABSENT, ABSENT, ABSENT, 0, X},
                                nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedVectorContainsTest, VectorNonNullListsWithNullValues)
{
  // Test List columns that have no NULL list rows, but NULL elements in some list rows.
  using T = TypeParam;

  auto numerals = cudf::test::fixed_width_column_wrapper<T>{
    {X, 1, 2, X, 4, 5, X, 7, 8, X, X, 1, 2, X, 1}, nulls_at({0, 3, 6, 9, 10, 13})};

  auto search_space = cudf::make_lists_column(
    8, indices_col{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(), numerals.release(), 0, {});
  // Search space: [ [x], [1,2], [x,4,5,x], [], [], [7,8,x], [x], [1,2,x,1] ]
  auto search_keys = cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 1};
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_keys);
    auto expected = bools_col{0, 1, 0, 0, 0, 0, 0, 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_FIRST);
    auto expected = indices_col{ABSENT, 1, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_LAST);
    auto expected = indices_col{ABSENT, 1, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 3};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedVectorContainsTest, VectorWithNullsInLists)
{
  using T = TypeParam;

  auto numerals = cudf::test::fixed_width_column_wrapper<T>{
    {X, 1, 2, X, 4, 5, X, 7, 8, X, X, 1, 2, X, 1}, nulls_at({0, 3, 6, 9, 10, 13})};

  auto input_null_mask_iter = null_at(4);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8);
  auto search_space = cudf::make_lists_column(8,
                                              indices_col{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
                                              numerals.release(),
                                              null_count,
                                              std::move(null_mask));

  // Search space: [ [x], [1,2], [x,4,5,x], [], x, [7,8,x], [x], [1,2,x,1] ]

  auto search_keys = cudf::test::fixed_width_column_wrapper<T, int32_t>{1, 2, 3, 1, 2, 3, 1, 1};
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_keys);
    auto expected = bools_col{{0, 1, 0, 0, X, 0, 0, 1}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_FIRST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, ABSENT, X, ABSENT, ABSENT, 0}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_LAST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, ABSENT, X, ABSENT, ABSENT, 3}, null_at(4)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedVectorContainsTest, ListContainsVectorWithNullsInListsAndInSearchKeys)
{
  using T = TypeParam;

  auto numerals = cudf::test::fixed_width_column_wrapper<T>{
    {X, 1, 2, X, 4, 5, X, 7, 8, X, X, 1, 2, X, 1}, nulls_at({0, 3, 6, 9, 10, 13})};

  auto input_null_mask_iter = null_at(4);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8);
  auto search_space = cudf::make_lists_column(8,
                                              indices_col{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
                                              numerals.release(),
                                              null_count,
                                              std::move(null_mask));

  // Search space: [ [x], [1,2], [x,4,5,x], [], x, [7,8,x], [x], [1,2,x,1] ]

  auto search_keys =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{{1, 2, 3, X, 2, 3, 1, 1}, null_at(3)};
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_keys);
    auto expected = bools_col{{0, 1, 0, X, X, 0, 0, 1}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_FIRST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, X, X, ABSENT, ABSENT, 0}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_LAST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, X, X, ABSENT, ABSENT, 3}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TEST_F(ContainsTest, BoolKeyVectorWithNullsInListsAndInSearchKeys)
{
  using T = bool;

  auto numerals = cudf::test::fixed_width_column_wrapper<T>{
    {X, 0, 1, X, 1, 1, X, 1, 1, X, X, 0, 1, X, 1}, nulls_at({0, 3, 6, 9, 10, 13})};

  auto input_null_mask_iter = null_at(4);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8);
  auto search_space = cudf::make_lists_column(8,
                                              indices_col{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
                                              numerals.release(),
                                              null_count,
                                              std::move(null_mask));

  auto search_keys =
    cudf::test::fixed_width_column_wrapper<T, int32_t>{{0, 1, 0, X, 0, 0, 1, 1}, null_at(3)};
  // Search space: [ [x], [0,1], [x,1,1,x], [], x, [1,1,x], [x], [0,1,x,1] ]
  // Search keys : [  0,   1,     0,         x, 0,  0,       1,   1        ]
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_keys);
    auto expected = bools_col{{0, 1, 0, X, X, 0, 0, 1}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_FIRST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, X, X, ABSENT, ABSENT, 1}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_LAST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, X, X, ABSENT, ABSENT, 3}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TEST_F(ContainsTest, StringKeyVectorWithNullsInListsAndInSearchKeys)
{
  auto strings = cudf::test::strings_column_wrapper{
    {"X", "1", "2", "X", "4", "5", "X", "7", "8", "X", "X", "1", "2", "X", "1"},
    nulls_at({0, 3, 6, 9, 10, 13})};
  auto input_null_mask_iter = null_at(4);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(input_null_mask_iter, input_null_mask_iter + 8);
  auto search_space = cudf::make_lists_column(
    8,
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 3, 7, 7, 7, 10, 11, 15}.release(),
    strings.release(),
    null_count,
    std::move(null_mask));

  auto search_keys =
    cudf::test::strings_column_wrapper{{"1", "2", "3", "X", "2", "3", "1", "1"}, null_at(3)};

  // Search space: [ [x], [1,2], [x,4,5,x], [], x, [7,8,x], [x], [1,2,x,1] ]
  // Search keys:  [  1,   2,     3,         X, 2,  3,       1,   1]

  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_keys);
    auto expected = bools_col{{0, 1, 0, X, X, 0, 0, 1}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_FIRST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, X, X, ABSENT, ABSENT, 0}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_keys, FIND_LAST);
    auto expected = indices_col{{ABSENT, 1, ABSENT, X, X, ABSENT, ABSENT, 3}, nulls_at({3, 4})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TEST_F(ContainsTest, VectorTypeRelatedExceptions)
{
  {
    // Nested types unsupported.
    auto list_of_lists = cudf::test::lists_column_wrapper<int32_t>{
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3}, {4, 5, 6}},
      {{1, 2, 3},
       {4, 5, 6}}}.release();
    auto skey = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2};
    EXPECT_THROW(cudf::lists::contains(list_of_lists->view(), skey), cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_lists->view(), skey, FIND_FIRST),
                 cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_lists->view(), skey, FIND_LAST),
                 cudf::data_type_error);
  }
  {
    // Search key must match list elements in type.
    auto list_of_ints =
      cudf::test::lists_column_wrapper<int32_t>{
        {0, 1, 2},
        {3, 4, 5},
      }
        .release();
    auto skey = cudf::test::strings_column_wrapper{"Hello", "World"};
    EXPECT_THROW(cudf::lists::contains(list_of_ints->view(), skey), cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_ints->view(), skey, FIND_FIRST),
                 cudf::data_type_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_ints->view(), skey, FIND_LAST),
                 cudf::data_type_error);
  }
  {
    // Search key column size must match lists column size.
    auto list_of_ints =
      cudf::test::lists_column_wrapper<int32_t>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}.release();
    auto skey = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
    EXPECT_THROW(cudf::lists::contains(list_of_ints->view(), skey), cudf::logic_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_ints->view(), skey, FIND_FIRST), cudf::logic_error);
    EXPECT_THROW(cudf::lists::index_of(list_of_ints->view(), skey, FIND_LAST), cudf::logic_error);
  }
}

template <typename T>
struct TypedContainsNaNsTest : public ContainsTest {};

TYPED_TEST_SUITE(TypedContainsNaNsTest, cudf::test::FloatingPointTypes);

namespace {
template <typename T>
T get_nan(char const* nan_contents)
{
  return std::nan(nan_contents);
}

template <>
float get_nan<float>(char const* nan_contents)
{
  return std::nanf(nan_contents);
}
}  // namespace

TYPED_TEST(TypedContainsNaNsTest, ListWithNaNsScalar)
{
  using T = TypeParam;

  auto nan_1 = get_nan<T>("1");
  auto nan_2 = get_nan<T>("2");
  auto nan_3 = get_nan<T>("3");

  auto search_space = cudf::test::lists_column_wrapper<T>{
    {0.0, 1.0, 2.0},
    {3, 4, 5},
    {6, 7, 8},
    {9, 0, 1},
    {nan_1, 3.0, 4.0},
    {5, 6, 7},
    {8, nan_2, 0},
    {},
    {1, 2, 3},
    {}}.release();

  auto search_key_nan = create_scalar_search_key<T>(nan_3);
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), *search_key_nan);
    auto expected = bools_col{0, 0, 0, 0, 1, 0, 1, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result = cudf::lists::index_of(search_space->view(), *search_key_nan, FIND_FIRST);
    auto expected =
      indices_col{ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT, 1, ABSENT, ABSENT, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result = cudf::lists::index_of(search_space->view(), *search_key_nan, FIND_LAST);
    auto expected =
      indices_col{ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT, 1, ABSENT, ABSENT, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedContainsNaNsTest, ListWithNaNsContainsVector)
{
  // Test that different bit representations of NaN values
  // are recognized as NaN.
  // Also checks that a null handling is not broken by the
  // presence of NaN values:
  //   1. If the search key is null, null is still returned.
  //   2. If the list contains a null, and the non-null search
  //      key is not found:
  //      a) contains() returns `null`.
  //      b) index_of() returns -1.
  using T = TypeParam;

  auto nan_1 = get_nan<T>("1");
  auto nan_2 = get_nan<T>("2");
  auto nan_3 = get_nan<T>("3");

  auto search_space = cudf::test::lists_column_wrapper<T>{
    {0.0, 1.0, 2.0},
    {{3, 4, 5}, null_at(2)},  // i.e. {3, 4, ∅}.
    {6, 7, 8},
    {9, 0, 1},
    {nan_1, 3.0, 4.0},
    {5, 6, 7},
    {8, nan_2, 0},
    {},
    {1, 2, 3},
    {}}.release();

  auto search_key_values = std::vector<T>{1.0, 2.0, 3.0, nan_3, nan_3, nan_3, 0.0, nan_3, 2.0, 0.0};

  {
    // With nulls in the search key rows. (At index 2.)
    auto search_keys =
      cudf::test::fixed_width_column_wrapper<T>{
        search_key_values.begin(), search_key_values.end(), null_at(2)}
        .release();

    {
      // CONTAINS
      auto result   = cudf::lists::contains(search_space->view(), search_keys->view());
      auto expected = bools_col{{1, 0, 0, 0, 1, 0, 1, 0, 1, 0}, null_at(2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_FIRST
      auto result = cudf::lists::index_of(search_space->view(), search_keys->view(), FIND_FIRST);
      auto expected =
        indices_col{{1, ABSENT, X, ABSENT, 0, ABSENT, 2, ABSENT, 1, ABSENT}, nulls_at({2})};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_LAST
      auto result = cudf::lists::index_of(search_space->view(), search_keys->view(), FIND_LAST);
      auto expected =
        indices_col{{1, ABSENT, X, ABSENT, 0, ABSENT, 2, ABSENT, 1, ABSENT}, nulls_at({2})};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
  }
  {
    // No nulls in the search key rows.
    auto search_keys =
      cudf::test::fixed_width_column_wrapper<T>(search_key_values.begin(), search_key_values.end())
        .release();
    {
      // CONTAINS
      auto result   = cudf::lists::contains(search_space->view(), search_keys->view());
      auto expected = bools_col{1, 0, 0, 0, 1, 0, 1, 0, 1, 0};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_FIRST
      auto result   = cudf::lists::index_of(search_space->view(), search_keys->view(), FIND_FIRST);
      auto expected = indices_col{1, ABSENT, ABSENT, ABSENT, 0, ABSENT, 2, ABSENT, 1, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_LAST
      auto result   = cudf::lists::index_of(search_space->view(), search_keys->view(), FIND_LAST);
      auto expected = indices_col{1, ABSENT, ABSENT, ABSENT, 0, ABSENT, 2, ABSENT, 1, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
  }
}

template <typename T>
struct TypedContainsDecimalsTest : public ContainsTest {};

TYPED_TEST_SUITE(TypedContainsDecimalsTest, cudf::test::FixedPointTypes);

TYPED_TEST(TypedContainsDecimalsTest, ScalarKey)
{
  using T = TypeParam;

  auto const search_space = [] {
    auto const values = std::vector<typename T::rep>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
                                                     2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3};
    auto decimals     = cudf::test::fixed_point_column_wrapper<typename T::rep>{
      values.begin(), values.end(), numeric::scale_type{0}};
    auto list_offsets = indices_col{0, 3, 6, 9, 12, 15, 18, 21, 21, 24, 24};
    return cudf::make_lists_column(10, list_offsets.release(), decimals.release(), 0, {});
  }();
  auto search_key_one =
    cudf::make_fixed_point_scalar<T>(typename T::rep{1}, numeric::scale_type{0});

  // Search space: [[0,1,2], [3,4,5], [6,7,8], [9,0,1], [2,3,4], [5,6,7], [8,9,0], [], [1,2,3], []]
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), *search_key_one);
    auto expected = bools_col{1, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_FIRST);
    auto expected = indices_col{1, ABSENT, ABSENT, 2, ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), *search_key_one, FIND_LAST);
    auto expected = indices_col{1, ABSENT, ABSENT, 2, ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedContainsDecimalsTest, VectorKey)
{
  using T = TypeParam;

  auto const search_space = [] {
    auto const values = std::vector<typename T::rep>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
                                                     2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3};
    auto decimals     = cudf::test::fixed_point_column_wrapper<typename T::rep>{
      values.begin(), values.end(), numeric::scale_type{0}};
    auto list_offsets = indices_col{0, 3, 6, 9, 12, 15, 18, 21, 21, 24, 24};
    return cudf::make_lists_column(10, list_offsets.release(), decimals.release(), 0, {});
  }();

  auto search_key =
    cudf::test::fixed_point_column_wrapper<typename T::rep>{{1, 2, 3, 1, 2, 3, 1, 2, 3, 1},
                                                            numeric::scale_type{0}}
      .release();

  // Search space: [ [0,1,2], [3,4,5], [6,7,8], [9,0,1], [2,3,4], [5,6,7], [8,9,0], [], [1,2,3], []
  // ] Search keys:  [  1,       2,       3,       1,       2,       3,       1,       2,  3, 1 ]
  {
    // CONTAINS
    auto result   = cudf::lists::contains(search_space->view(), search_key->view());
    auto expected = bools_col{1, 0, 0, 1, 1, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto result   = cudf::lists::index_of(search_space->view(), search_key->view(), FIND_FIRST);
    auto expected = indices_col{1, ABSENT, ABSENT, 2, 0, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto result   = cudf::lists::index_of(search_space->view(), search_key->view(), FIND_LAST);
    auto expected = indices_col{1, ABSENT, ABSENT, 2, 0, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

template <typename T>
struct TypedStructContainsTest : public ContainsTest {};
TYPED_TEST_SUITE(TypedStructContainsTest, ContainsTestTypes);

TYPED_TEST(TypedStructContainsTest, EmptyInputTest)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists = [] {
    auto offsets = indices_col{};
    auto data    = tdata_col{};
    auto child   = cudf::test::structs_column_wrapper{{data}};
    return cudf::make_lists_column(0, offsets.release(), child.release(), 0, {});
  }();

  auto const scalar_key = [] {
    auto child = tdata_col{0};
    return make_struct_scalar(child);
  }();
  auto const column_key = [] {
    auto child = tdata_col{};
    return cudf::test::structs_column_wrapper{{child}};
  }();

  auto const result1  = cudf::lists::contains(lists->view(), scalar_key);
  auto const result2  = cudf::lists::contains(lists->view(), column_key);
  auto const expected = bools_col{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result2);
}

TYPED_TEST(TypedStructContainsTest, ScalarKeyNoNullLists)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists = [] {
    auto offsets = indices_col{0, 4, 7, 10, 15, 18, 21, 24, 24, 28, 28};
    // clang-format off
    auto data1    = tdata_col{0, 1, 2, 1,
                              3, 4, 5,
                              6, 7, 8,
                              9, 0, 1, 3, 1,
                              2, 3, 4,
                              5, 6, 7,
                              8, 9, 0,
                              1, 2, 1, 3
    };
    auto data2    = tdata_col{0, 1, 2, 3,
                              0, 1, 2,
                              0, 1, 2,
                              1, 1, 2, 2, 2,
                              0, 1, 2,
                              0, 1, 2,
                              0, 1, 2,
                              1, 0, 1, 1
    };
    // clang-format on
    auto child = cudf::test::structs_column_wrapper{{data1, data2}};
    return cudf::make_lists_column(10, offsets.release(), child.release(), 0, {});
  }();

  auto const key = [] {
    auto child1 = tdata_col{1};
    auto child2 = tdata_col{1};
    return make_struct_scalar(child1, child2);
  }();

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(lists->view(), key);
    auto const expected = bools_col{1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto const result   = cudf::lists::contains_nulls(lists->view());
    auto const expected = bools_col{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result = cudf::lists::index_of(lists->view(), key, FIND_FIRST);
    auto const expected =
      indices_col{1, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result = cudf::lists::index_of(lists->view(), key, FIND_LAST);
    auto const expected =
      indices_col{1, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedStructContainsTest, ScalarKeyWithNullLists)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists = [] {
    auto offsets = indices_col{0, 4, 7, 10, 10, 15, 18, 21, 24, 24, 28, 28};
    // clang-format off
    auto data1    = tdata_col{0, 1, 2, 1,
                              3, 4, 5,
                              6, 7, 8,
                              9, 0, 1, 3, 1,
                              2, 3, 4,
                              5, 6, 7,
                              8, 9, 0,
                              1, 2, 1, 3
    };
    auto data2    = tdata_col{0, 1, 2, 3,
                              0, 1, 2,
                              0, 1, 2,
                              1, 1, 2, 2, 2,
                              0, 1, 2,
                              0, 1, 2,
                              0, 1, 2,
                              1, 0, 1, 1
    };
    // clang-format on
    auto child               = cudf::test::structs_column_wrapper{{data1, data2}};
    auto const validity_iter = nulls_at({3, 10});
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(validity_iter, validity_iter + 11);
    return cudf::make_lists_column(
      11, offsets.release(), child.release(), null_count, std::move(null_mask));
  }();

  auto const key = [] {
    auto child1 = tdata_col{1};
    auto child2 = tdata_col{1};
    return make_struct_scalar(child1, child2);
  }();

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(lists->view(), key);
    auto const expected = bools_col{{1, 0, 0, X, 0, 0, 0, 0, 0, 1, X}, nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto const result   = cudf::lists::contains_nulls(lists->view());
    auto const expected = bools_col{{0, 0, 0, X, 0, 0, 0, 0, 0, 0, X}, nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result   = cudf::lists::index_of(lists->view(), key, FIND_FIRST);
    auto const expected = indices_col{
      {1, ABSENT, ABSENT, X, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 0, X}, nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result   = cudf::lists::index_of(lists->view(), key, FIND_LAST);
    auto const expected = indices_col{
      {1, ABSENT, ABSENT, X, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 2, X}, nulls_at({3, 10})};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedStructContainsTest, SlicedListsColumnNoNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists_original = [] {
    auto offsets = indices_col{0, 4, 7, 10, 15, 18, 21, 24, 24, 28, 28};
    // clang-format off
    auto data1    = tdata_col{0, 1, 2, 1,
                              3, 4, 5,
                              6, 7, 8,
                              9, 0, 1, 3, 1,
                              2, 3, 4,
                              5, 6, 7,
                              8, 9, 0,
                              1, 2, 1, 3
    };
    auto data2    = tdata_col{0, 1, 2, 3,
                              0, 1, 2,
                              0, 1, 2,
                              1, 1, 2, 2, 2,
                              0, 1, 2,
                              0, 1, 2,
                              0, 1, 2,
                              1, 0, 1, 1
    };
    // clang-format on
    auto child = cudf::test::structs_column_wrapper{{data1, data2}};
    return cudf::make_lists_column(10, offsets.release(), child.release(), 0, {});
  }();
  auto const lists = cudf::slice(lists_original->view(), {3, 10})[0];

  auto const key = [] {
    auto child1 = tdata_col{1};
    auto child2 = tdata_col{1};
    return make_struct_scalar(child1, child2);
  }();

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(lists, key);
    auto const expected = bools_col{0, 0, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto const result   = cudf::lists::contains_nulls(lists);
    auto const expected = bools_col{0, 0, 0, 0, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result   = cudf::lists::index_of(lists, key, FIND_FIRST);
    auto const expected = indices_col{ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result   = cudf::lists::index_of(lists, key, FIND_LAST);
    auto const expected = indices_col{ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedStructContainsTest, ScalarKeyNoNullListsWithNullStructs)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists = [] {
    auto offsets = indices_col{0, 4, 7, 10, 15, 18, 21, 24, 24, 28, 28};
    // clang-format off
    auto data1    = tdata_col{0, X, 2, 1,
                              3, 4, 5,
                              6, 7, 8,
                              X, 0, 1, 3, 1,
                              X, 3, 4,
                              5, 6, 7,
                              8, 9, 0,
                              X, 2, 1, 3
    };
    auto data2    = tdata_col{0, X, 2, 1,
                              0, 1, 2,
                              0, 1, 2,
                              X, 1, 2, 2, 2,
                              X, 1, 2,
                              0, 1, 2,
                              0, 1, 2,
                              X, 0, 1, 1
    };
    // clang-format on
    auto child = cudf::test::structs_column_wrapper{{data1, data2}, nulls_at({1, 10, 15, 24})};
    return cudf::make_lists_column(10, offsets.release(), child.release(), 0, {});
  }();

  auto const key = [] {
    auto child1 = tdata_col{1};
    auto child2 = tdata_col{1};
    return make_struct_scalar(child1, child2);
  }();

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(lists->view(), key);
    auto const expected = bools_col{1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // CONTAINS NULLS
    auto const result   = cudf::lists::contains_nulls(lists->view());
    auto const expected = bools_col{1, 0, 0, 1, 1, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result = cudf::lists::index_of(lists->view(), key, FIND_FIRST);
    auto const expected =
      indices_col{3, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result = cudf::lists::index_of(lists->view(), key, FIND_LAST);
    auto const expected =
      indices_col{3, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedStructContainsTest, ColumnKeyNoNullLists)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists = [] {
    auto offsets = indices_col{0, 4, 7, 10, 15, 18, 21, 24, 24, 28, 28};
    // clang-format off
    auto data1    = tdata_col{0, 1, 2, 1,
                              3, 4, 3,
                              6, 7, 8,
                              9, 0, 1, 3, 1,
                              2, 3, 4,
                              5, 6, 7,
                              8, 9, 0,
                              1, 2, 1, 3
    };
    auto data2    = tdata_col{0, 1, 2, 3,
                              0, 0, 0,
                              0, 1, 2,
                              1, 1, 2, 2, 2,
                              0, 1, 2,
                              0, 1, 2,
                              0, 1, 2,
                              1, 0, 1, 1
    };
    // clang-format on
    auto child = cudf::test::structs_column_wrapper{{data1, data2}};
    return cudf::make_lists_column(10, offsets.release(), child.release(), 0, {});
  }();

  auto const keys = [] {
    auto child1 = tdata_col{1, 3, 1, 1, 2, 1, 0, 0, 1, 0};
    auto child2 = tdata_col{1, 0, 1, 1, 2, 1, 0, 0, 1, 0};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(lists->view(), keys);
    auto const expected = bools_col{1, 1, 0, 0, 0, 0, 0, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result = cudf::lists::index_of(lists->view(), keys, FIND_FIRST);
    auto const expected =
      indices_col{1, 0, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 0, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result = cudf::lists::index_of(lists->view(), keys, FIND_LAST);
    auto const expected =
      indices_col{1, 2, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, 2, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedStructContainsTest, ColumnKeyWithSlicedListsNoNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists_original = [] {
    auto offsets = indices_col{0, 4, 7, 10, 15, 18, 21, 24, 24, 28, 28};
    // clang-format off
    auto data1    = tdata_col{0, 1, 2, 1,
                              3, 4, 3,
                              6, 7, 8,
                              9, 0, 1, 3, 1,
                              2, 3, 4,
                              5, 6, 7,
                              8, 9, 0,
                              1, 2, 1, 3
    };
    auto data2    = tdata_col{0, 1, 2, 3,
                              0, 0, 0,
                              0, 1, 2,
                              1, 1, 2, 2, 2,
                              0, 1, 2,
                              0, 1, 2,
                              0, 1, 2,
                              1, 0, 1, 1
    };
    // clang-format on
    auto child = cudf::test::structs_column_wrapper{{data1, data2}};
    return cudf::make_lists_column(10, offsets.release(), child.release(), 0, {});
  }();

  auto const keys_original = [] {
    auto child1 = tdata_col{1, 9, 1, 6, 2, 1, 0, 0, 1, 0};
    auto child2 = tdata_col{1, 1, 1, 1, 2, 1, 0, 0, 1, 0};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const lists = cudf::slice(lists_original->view(), {3, 7})[0];
  auto const keys  = cudf::slice(keys_original, {1, 5})[0];

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(lists, keys);
    auto const expected = bools_col{1, 0, 1, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result   = cudf::lists::index_of(lists, keys, FIND_FIRST);
    auto const expected = indices_col{0, ABSENT, 1, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result   = cudf::lists::index_of(lists, keys, FIND_LAST);
    auto const expected = indices_col{0, ABSENT, 1, ABSENT};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

TYPED_TEST(TypedStructContainsTest, ColumnKeyWithSlicedListsHavingNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists_original = [] {
    auto offsets = indices_col{0, 4, 7, 10, 10, 15, 18, 21, 24, 24, 28, 28};
    // clang-format off
    auto data1    = tdata_col{0, X, 2, 1,
                              3, 4, 5,
                              6, 7, 8,
                              X, 0, 1, 3, 1,
                              X, 3, 4,
                              5, 6, 6,
                              8, 9, 0,
                              X, 2, 1, 3
    };
    auto data2    = tdata_col{0, X, 2, 1,
                              0, 1, 2,
                              0, 1, 2,
                              X, 1, 2, 2, 2,
                              X, 1, 2,
                              0, 1, 1,
                              0, 1, 2,
                              X, 0, 1, 1
    };
    // clang-format on
    auto child = cudf::test::structs_column_wrapper{{data1, data2}, nulls_at({1, 10, 15, 24})};
    auto const validity_iter = nulls_at({3, 10});
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(validity_iter, validity_iter + 11);
    return cudf::make_lists_column(
      11, offsets.release(), child.release(), null_count, std::move(null_mask));
  }();

  auto const keys_original = [] {
    auto child1 = tdata_col{{1, X, 1, 6, X, 1, 0, 0, 1, 0, 1}, null_at(4)};
    auto child2 = tdata_col{{1, X, 1, 1, X, 1, 0, 0, 1, 0, 1}, null_at(4)};
    return cudf::test::structs_column_wrapper{{child1, child2}, null_at(1)};
  }();

  auto const lists = cudf::slice(lists_original->view(), {4, 8})[0];
  auto const keys  = cudf::slice(keys_original, {1, 5})[0];

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(lists, keys);
    auto const expected = bools_col{{X, 0, 1, 0}, null_at(0)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result   = cudf::lists::index_of(lists, keys, FIND_FIRST);
    auto const expected = indices_col{{X, ABSENT, 1, ABSENT}, null_at(0)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result   = cudf::lists::index_of(lists, keys, FIND_LAST);
    auto const expected = indices_col{{X, ABSENT, 2, ABSENT}, null_at(0)};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

template <typename T>
struct TypedListContainsTest : public ContainsTest {};
TYPED_TEST_SUITE(TypedListContainsTest, ContainsTestTypes);

TYPED_TEST(TypedListContainsTest, ScalarKeyLists)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const lists_no_nulls = lists_col{lists_col{{0, 1, 2},  // list0
                                                  {3, 4, 5},
                                                  {0, 1, 2},
                                                  {9, 0, 1, 3, 1}},
                                        lists_col{{2, 3, 4},  // list1
                                                  {3, 4, 5},
                                                  {8, 9, 0},
                                                  {}},
                                        lists_col{{0, 2, 1},  // list2
                                                  {}}};

  auto const lists_have_nulls = lists_col{lists_col{{{0, 1, 2},  // list0
                                                     {} /*NULL*/,
                                                     {0, 1, 2},
                                                     {9, 0, 1, 3, 1}},
                                                    null_at(1)},
                                          lists_col{{{} /*NULL*/,  // list1
                                                     {3, 4, 5},
                                                     {8, 9, 0},
                                                     {}},
                                                    null_at(0)},
                                          lists_col{{0, 2, 1},  // list2
                                                    {}}};

  auto const key = [] {
    auto const child = tdata_col{0, 1, 2};
    return cudf::list_scalar(child);
  }();

  auto const do_test = [&](auto const& lists, bool has_nulls) {
    {
      // CONTAINS
      auto const result   = cudf::lists::contains(cudf::lists_column_view{lists}, key);
      auto const expected = bools_col{1, 0, 0};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // CONTAINS NULLS
      auto const result   = cudf::lists::contains_nulls(cudf::lists_column_view{lists});
      auto const expected = has_nulls ? bools_col{1, 1, 0} : bools_col{0, 0, 0};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_FIRST
      auto const result   = cudf::lists::index_of(cudf::lists_column_view{lists}, key, FIND_FIRST);
      auto const expected = indices_col{0, ABSENT, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_LAST
      auto const result   = cudf::lists::index_of(cudf::lists_column_view{lists}, key, FIND_LAST);
      auto const expected = indices_col{2, ABSENT, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
  };

  do_test(lists_no_nulls, false);
  do_test(lists_have_nulls, true);
}

TYPED_TEST(TypedListContainsTest, SlicedListsColumn)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const lists_no_nulls_original = lists_col{lists_col{{0, 0, 0},  // list-2 (don't care)
                                                           {0, 1, 2},
                                                           {0, 1, 2},
                                                           {0, 0, 0}},
                                                 lists_col{{0, 0, 0},  // list-1 (don't care)
                                                           {0, 1, 2},
                                                           {0, 1, 2},
                                                           {0, 0, 0}},
                                                 lists_col{{0, 1, 2},  // list0
                                                           {3, 4, 5},
                                                           {0, 1, 2},
                                                           {9, 0, 1, 3, 1}},
                                                 lists_col{{2, 3, 4},  // list1
                                                           {3, 4, 5},
                                                           {8, 9, 0},
                                                           {}},
                                                 lists_col{{0, 2, 1},  // list2
                                                           {}},
                                                 lists_col{{0, 0, 0},  // list3 (don't care)
                                                           {0, 1, 2},
                                                           {0, 1, 2},
                                                           {0, 0, 0}},
                                                 lists_col{{0, 0, 0},  // list4 (don't care)
                                                           {0, 1, 2},
                                                           {0, 1, 2},
                                                           {0, 0, 0}}};

  auto const lists_have_nulls_original = lists_col{lists_col{{0, 0, 0},  // list-1 (don't care)
                                                             {0, 1, 2},
                                                             {0, 1, 2},
                                                             {0, 0, 0}},
                                                   lists_col{{{0, 1, 2},  // list0
                                                              {} /*NULL*/,
                                                              {0, 1, 2},
                                                              {9, 0, 1, 3, 1}},
                                                             null_at(1)},
                                                   lists_col{{{} /*NULL*/,  // list1
                                                              {3, 4, 5},
                                                              {8, 9, 0},
                                                              {}},
                                                             null_at(0)},
                                                   lists_col{{0, 2, 1},  // list2
                                                             {}},
                                                   lists_col{{0, 0, 0},  // list3 (don't care)
                                                             {0, 1, 2},
                                                             {0, 1, 2},
                                                             {0, 0, 0}}};

  auto const lists_no_nulls   = cudf::slice(lists_no_nulls_original, {2, 5})[0];
  auto const lists_have_nulls = cudf::slice(lists_have_nulls_original, {1, 4})[0];

  auto const key = [] {
    auto const child = tdata_col{0, 1, 2};
    return cudf::list_scalar(child);
  }();

  auto const do_test = [&](auto const& lists, bool has_nulls) {
    {
      // CONTAINS
      auto const result   = cudf::lists::contains(cudf::lists_column_view{lists}, key);
      auto const expected = bools_col{1, 0, 0};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // CONTAINS NULLS
      auto const result   = cudf::lists::contains_nulls(cudf::lists_column_view{lists});
      auto const expected = has_nulls ? bools_col{1, 1, 0} : bools_col{0, 0, 0};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_FIRST
      auto const result   = cudf::lists::index_of(cudf::lists_column_view{lists}, key, FIND_FIRST);
      auto const expected = indices_col{0, ABSENT, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_LAST
      auto const result   = cudf::lists::index_of(cudf::lists_column_view{lists}, key, FIND_LAST);
      auto const expected = indices_col{2, ABSENT, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
  };

  do_test(lists_no_nulls, false);
  do_test(lists_have_nulls, true);
}

TYPED_TEST(TypedListContainsTest, ColumnKeyLists)
{
  using lists_col     = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  auto constexpr null = int32_t{0};

  auto const lists_no_nulls = lists_col{lists_col{{0, 0, 2},  // list0
                                                  {3, 4, 5},
                                                  {0, 0, 2},
                                                  {9, 0, 1, 3, 1}},
                                        lists_col{{2, 3, 4},  // list1
                                                  {3, 4, 5},
                                                  {2, 3, 4},
                                                  {}},
                                        lists_col{{0, 2, 0},  // list2
                                                  {0, 2, 0},
                                                  {3, 4, 5},
                                                  {}}};

  auto const lists_have_nulls = lists_col{lists_col{{lists_col{{0, null, 2}, null_at(1)},  // list0
                                                     lists_col{} /*NULL*/,
                                                     lists_col{{0, null, 2}, null_at(1)},
                                                     lists_col{9, 0, 1, 3, 1}},
                                                    null_at(1)},
                                          lists_col{{lists_col{} /*NULL*/,  // list1
                                                     lists_col{3, 4, 5},
                                                     lists_col{2, 3, 4},
                                                     lists_col{}},
                                                    null_at(0)},
                                          lists_col{lists_col{0, 2, 1},  // list2
                                                    lists_col{{0, 2, null}, null_at(2)},
                                                    lists_col{3, 4, 5},
                                                    lists_col{}}};

  auto const key = lists_col{
    lists_col{{0, null, 2}, null_at(1)}, lists_col{2, 3, 4}, lists_col{{0, 2, null}, null_at(2)}};

  auto const do_test = [&](auto const& lists, bool has_nulls) {
    {
      // CONTAINS
      auto const result   = cudf::lists::contains(cudf::lists_column_view{lists}, key);
      auto const expected = has_nulls ? bools_col{1, 1, 1} : bools_col{0, 1, 0};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // CONTAINS NULLS
      auto const result   = cudf::lists::contains_nulls(cudf::lists_column_view{lists});
      auto const expected = has_nulls ? bools_col{1, 1, 0} : bools_col{0, 0, 0};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_FIRST
      auto const result   = cudf::lists::index_of(cudf::lists_column_view{lists}, key, FIND_FIRST);
      auto const expected = has_nulls ? indices_col{0, 2, 1} : indices_col{ABSENT, 0, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
    {
      // FIND_LAST
      auto const result   = cudf::lists::index_of(cudf::lists_column_view{lists}, key, FIND_LAST);
      auto const expected = has_nulls ? indices_col{2, 2, 1} : indices_col{ABSENT, 2, ABSENT};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
    }
  };

  do_test(lists_no_nulls, false);
  do_test(lists_have_nulls, true);
}

TYPED_TEST(TypedListContainsTest, ColumnKeyWithListsOfStructsNoNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const lists = [] {
    auto child_offsets = indices_col{0, 3, 6, 9, 14, 17, 20, 23, 23};
    // clang-format off
    auto data1 = tdata_col{0, 0, 2,
                           3, 4, 5,
                           0, 0, 2,
                           9, 0, 1, 3, 1,
                           0, 2, 0,
                           0, 0, 2,
                           3, 4, 5

    };
    auto data2 = tdata_col{10, 10, 12,
                           13, 14, 15,
                           10, 10, 12,
                           19, 10, 11, 13, 11,
                           10, 12, 10,
                           10, 10, 12,
                           13, 14, 15

    };
    // clang-format on
    auto structs = cudf::test::structs_column_wrapper{{data1, data2}};
    auto child   = cudf::make_lists_column(8, child_offsets.release(), structs.release(), 0, {});

    auto offsets = indices_col{0, 4, 8};
    return cudf::make_lists_column(2, offsets.release(), std::move(child), 0, {});
  }();

  auto const key = [] {
    auto data1       = tdata_col{0, 0, 2};
    auto data2       = tdata_col{10, 10, 12};
    auto const child = cudf::test::structs_column_wrapper{{data1, data2}};
    return cudf::list_scalar(child);
  }();

  {
    // CONTAINS
    auto const result   = cudf::lists::contains(cudf::lists_column_view{lists->view()}, key);
    auto const expected = bools_col{1, 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_FIRST
    auto const result =
      cudf::lists::index_of(cudf::lists_column_view{lists->view()}, key, FIND_FIRST);
    auto const expected = indices_col{0, 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    // FIND_LAST
    auto const result =
      cudf::lists::index_of(cudf::lists_column_view{lists->view()}, key, FIND_LAST);
    auto const expected = indices_col{2, 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}
