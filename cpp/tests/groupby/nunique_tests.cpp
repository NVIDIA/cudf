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

template <typename V>
struct groupby_nunique_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_nunique_test, cudf::test::AllTypes);

TYPED_TEST(groupby_nunique_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // clang-format off
  //                                                   {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1,        2,          3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{3,        4,          3};
  cudf::test::fixed_width_column_wrapper<R> expect_bool_vals{2,   1,          1};
  // clang-format on

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  if (std::is_same<V, bool>())
    test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
  else
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, basic_duplicates)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 3, 2, 2, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{2, 4, 1};
  cudf::test::fixed_width_column_wrapper<R> expect_bool_vals{2, 1, 1};

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  if (std::is_same<V, bool>())
    test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
  else
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys({0, 0, 0}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({0, 0, 0}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{0};

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //                                        {1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());
  // all unique values only                 {3, 6,     1, 4, 9,   2, 8,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals{2, 3, 2, 0};
  cudf::test::fixed_width_column_wrapper<R> expect_bool_vals{1, 1, 1, 0};

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  if (std::is_same<V, bool>())
    test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
  else
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, null_keys_and_values_with_duplicates)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
    {true, true, true, true, true, true, true, true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2},
                                                 {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0});

  //  { 1, 1,     2, 2, 2,    3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());
  //  { 3, 6,-    1, 4, 9,-   2*, 8,   -*}
  //  unique,     with null,  dup,     dup null
  cudf::test::fixed_width_column_wrapper<R> expect_vals{2, 3, 2, 0};
  cudf::test::fixed_width_column_wrapper<R> expect_bool_vals{1, 1, 1, 0};

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  if (std::is_same<V, bool>())
    test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
  else
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, include_nulls)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
    {true, true, true, true, true, true, true, true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2},
                                                 {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0});

  //  { 1, 1,     2, 2, 2,    3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());
  //  { 3, 6,-    1, 4, 9,-   2*, 8,   -*}
  //  unique,     with null,  dup,     dup null
  cudf::test::fixed_width_column_wrapper<R> expect_vals{3, 4, 2, 1};
  cudf::test::fixed_width_column_wrapper<R> expect_bool_vals{2, 2, 1, 1};

  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  if (std::is_same<V, bool>())
    test_single_agg(keys, vals, expect_keys, expect_bool_vals, std::move(agg));
  else
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nunique_test, dictionary)
{
  using K = int32_t;
  using V = TypeParam;
  using R = cudf::size_type;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 3, 1, 2, 2, 1, 0, 3, 2, 4, 4, 2},
                                     {true, true, true, true, true, true, true, true, false, true, true, true, true, true});
  cudf::test::dictionary_column_wrapper<V>  vals({0, 1, 2, 2, 3, 4, 0, 6, 7, 8, 9, 0, 0, 0},
                                     {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0});

  // { 1, 1,   2, 2, 2,   3, 3,   4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, cudf::test::iterators::no_nulls());
  // { 3, 6,-  1, 4, 9,-  2*, 8,  -*}
  //  unique,  with null, dup,    dup null
  cudf::test::fixed_width_column_wrapper<R> expect_fixed_vals({3, 4, 2, 1});
  cudf::test::fixed_width_column_wrapper<R> expect_bool_vals{2, 2, 1, 1};
  // clang-format on

  cudf::column_view expect_vals = (std::is_same<V, bool>()) ? cudf::column_view{expect_bool_vals}
                                                            : cudf::column_view{expect_fixed_vals};

  test_single_agg(
    keys,
    vals,
    expect_keys,
    expect_vals,
    cudf::make_nunique_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
}
