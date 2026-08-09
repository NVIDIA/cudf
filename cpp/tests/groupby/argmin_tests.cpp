/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_argmin_test : public cudf::test::BaseFixture {};
using K = int32_t;

TYPED_TEST_SUITE(groupby_argmin_test, cudf::test::FixedWidthTypes);

TYPED_TEST(groupby_argmin_test, basic)
{
  using V = TypeParam;
  using R = cudf::size_type;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{6, 9, 8};

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmin_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = cudf::size_type;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmin_test, zero_valid_values)
{
  using V = TypeParam;
  using R = cudf::size_type;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_argmin_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = cudf::size_type;

  if (std::is_same_v<V, bool>) return;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 4},
                                                 {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //  { 9, 6,     8, 5, 0,   7, 1,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 9, 8, 0}, {1, 1, 1, 0});

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  // TODO: explore making this a gtest parameter
  auto agg2 = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_argmin_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_argmin_string_test, basic)
{
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 5, 7});

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_argmin_string_test, zero_valid_values)
{
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::strings_column_wrapper vals({"", "", ""}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_dictionary_argmin_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_argmin_test, basic)
{
  using V = std::string;
  using R = cudf::size_type;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{    1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  cudf::test::dictionary_column_wrapper<V>  vals{"año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1, 2, 3 });
  cudf::test::fixed_width_column_wrapper<R> expect_vals({ 3, 5, 7 });
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_argmin_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_argmin_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

struct groupby_argmin_struct_test : public cudf::test::BaseFixture {};

TEST_F(groupby_argmin_struct_test, basic)
{
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const expect_keys    = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_indices = cudf::test::fixed_width_column_wrapper<cudf::size_type>{3, 5, 7};

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

TEST_F(groupby_argmin_struct_test, slice_input)
{
  constexpr int32_t dont_care{1};
  auto const keys_original = cudf::test::fixed_width_column_wrapper<int32_t>{
    dont_care, dont_care, 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, dont_care};
  auto const vals_original = [] {
    auto child1 = cudf::test::strings_column_wrapper{"dont_care",
                                                     "dont_care",
                                                     "año",
                                                     "bit",
                                                     "₹1",
                                                     "aaa",
                                                     "zit",
                                                     "bat",
                                                     "aab",
                                                     "$1",
                                                     "€1",
                                                     "wut",
                                                     "dont_care"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const keys           = cudf::slice(keys_original, {2, 12})[0];
  auto const vals           = cudf::slice(vals_original, {2, 12})[0];
  auto const expect_keys    = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_indices = cudf::test::fixed_width_column_wrapper<cudf::size_type>{3, 5, 7};

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}

TEST_F(groupby_argmin_struct_test, null_keys_and_values)
{
  constexpr int32_t null{0};
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{
    {1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4}, null_at(7)};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "" /*NULL*/, "" /*NULL*/, "$1", "€1", "wut", "" /*NULL*/};
    auto child2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{9, 8, 7, 6, 5, null, null, 2, 1, 0, null};
    return cudf::test::structs_column_wrapper{{child1, child2}, nulls_at({5, 6, 10})};
  }();

  auto const expect_keys =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4}, no_nulls()};
  auto const expect_indices =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{{3, 1, 8, null}, null_at(3)};

  auto agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_indices, std::move(agg));
}
