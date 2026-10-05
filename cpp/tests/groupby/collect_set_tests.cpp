/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

namespace {

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

using keys_col      = cudf::test::fixed_width_column_wrapper<int32_t, int32_t>;
using strings_col   = cudf::test::strings_column_wrapper;
using strings_lists = cudf::test::lists_column_wrapper<cudf::string_view>;
using validity_col  = std::initializer_list<bool>;

auto groupby_collect_set(cudf::column_view const& keys,
                         cudf::column_view const& values,
                         std::unique_ptr<cudf::groupby_aggregation>&& agg)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = values;
  requests[0].aggregations.emplace_back(std::move(agg));

  auto const result      = cudf::groupby::groupby(cudf::table_view({keys})).aggregate(requests);
  auto const result_keys = result.first->view();                 // <== table_view of 1 column
  auto const result_vals = result.second[0].results[0]->view();  // <== column_view

  // Sort the output columns based on the output keys.
  // This is to facilitate comparison of the output with the expected columns.
  auto keys_vals_sorted = cudf::sort_by_key(cudf::table_view{{result_keys.column(0), result_vals}},
                                            result_keys,
                                            {},
                                            {cudf::null_order::AFTER})
                            ->release();

  // After the columns were reordered, individual rows of the output values column (which are lists)
  // also need to be sorted.
  auto out_values =
    cudf::lists::sort_lists(cudf::lists_column_view{keys_vals_sorted.back()->view()},
                            cudf::order::ASCENDING,
                            cudf::null_order::AFTER);

  return std::pair(std::move(keys_vals_sorted.front()), std::move(out_values));
}

}  // namespace

struct CollectSetTest : public cudf::test::BaseFixture {
  static auto collect_set()
  { return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(); }

  static auto collect_set_null_unequal()
  {
    return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
      cudf::null_policy::INCLUDE, cudf::null_equality::UNEQUAL);
  }

  static auto collect_set_null_exclude()
  {
    return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
      cudf::null_policy::EXCLUDE);
  }
};

template <typename V>
struct CollectSetTypedTest : public cudf::test::BaseFixture {};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(CollectSetTypedTest, FixedWidthTypesNotBool);

TYPED_TEST(CollectSetTypedTest, TrivialInput)
{
  using vals_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Empty input
  {
    keys_col keys{};
    vals_col vals{};
    keys_col keys_expected{};
    lists_col vals_expected{};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // Single key input
  {
    keys_col keys{1};
    vals_col vals{10};
    keys_col keys_expected{1};
    lists_col vals_expected{lists_col{10}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // Non-repeated keys
  {
    keys_col keys{2, 1};
    vals_col vals{20, 10};
    keys_col keys_expected{1, 2};
    lists_col vals_expected{lists_col{10}, lists_col{20}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

TYPED_TEST(CollectSetTypedTest, TypicalInput)
{
  using vals_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Pre-sorted keys
  {
    keys_col keys{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    vals_col vals{10, 11, 10, 10, 20, 21, 21, 20, 30, 33, 32, 31};
    keys_col keys_expected{1, 2, 3};
    lists_col vals_expected{{10, 11}, {20, 21}, {30, 31, 32, 33}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // Expect the result keys to be sorted by sort-based groupby
  {
    keys_col keys{4, 1, 2, 4, 3, 3, 2, 1};
    vals_col vals{40, 10, 20, 40, 30, 30, 20, 11};
    keys_col keys_expected{1, 2, 3, 4};
    lists_col vals_expected{{10, 11}, {20}, {30}, {40}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

// Keys and values columns are sliced columns
TYPED_TEST(CollectSetTypedTest, SlicedColumnsInput)
{
  using vals_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  keys_col keys_original{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  vals_col vals_original{10, 11, 10, 10, 20, 21, 21, 20, 30, 33, 32, 31};
  {
    auto const keys          = cudf::slice(keys_original, {0, 4})[0];  // { 1, 1, 1, 1 }
    auto const vals          = cudf::slice(vals_original, {0, 4})[0];  // { 10, 11, 10, 10 }
    auto const keys_expected = keys_col{1};
    auto const vals_expected = lists_col{{10, 11}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
  {
    auto const keys = cudf::slice(keys_original, {2, 10})[0];  // { 1, 1, 2, 2, 2, 2, 3, 3 }
    auto const vals = cudf::slice(vals_original, {2, 10})[0];  // { 10, 10, 20, 21, 21, 20, 30, 33 }
    auto const keys_expected = keys_col{1, 2, 3};
    auto const vals_expected = lists_col{{10}, {20, 21}, {30, 33}};

    auto const [out_keys, out_lists] =
      groupby_collect_set(keys, vals, CollectSetTest::collect_set());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

TEST_F(CollectSetTest, StringInput)
{
  keys_col keys{1, 2, 3, 3, 2, 1, 2, 1, 2, 1, 1, 1, 1};
  strings_col vals{
    "String 1, first",
    "String 2, first",
    "String 3, first",
    "String 3, second",
    "String 2, second",
    "String 1, second",
    "String 2, second",  // repeated
    "String 1, second",  // repeated
    "String 2, second",  // repeated
    "String 1, second",  // repeated
    "String 1, second",  // repeated
    "String 1, second",  // repeated
    "String 1, second"   // repeated
  };
  keys_col keys_expected{1, 2, 3};
  strings_lists vals_expected{{"String 1, first", "String 1, second"},
                              {"String 2, first", "String 2, second"},
                              {"String 3, first", "String 3, second"}};

  auto const [out_keys, out_lists] = groupby_collect_set(keys, vals, CollectSetTest::collect_set());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
}

TEST_F(CollectSetTest, FloatsWithNaN)
{
  keys_col keys{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<float> vals{
    {1.0f, 1.0f, -2.3e-5f, -2.3e-5f, 2.3e5f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
    {true, true, true, true, true, true, true, true, true, true, false, false}};
  keys_col keys_expected{1};
  cudf::test::lists_column_wrapper<float> vals_expected;

  // null equal with nan unequal
  {
    vals_expected = {{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f},
                      validity_col{true, true, true, true, true, true, true, false}}};
    auto const [out_keys, out_lists] = groupby_collect_set(
      keys,
      vals,
      cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
        cudf::null_policy::INCLUDE, cudf::null_equality::EQUAL, cudf::nan_equality::UNEQUAL));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null unequal with nan unequal
  {
    vals_expected = {{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
                      validity_col{true, true, true, true, true, true, true, false, false}}};
    auto const [out_keys, out_lists] = groupby_collect_set(
      keys,
      vals,
      cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
        cudf::null_policy::INCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::UNEQUAL));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null exclude with nan unequal
  {
    vals_expected                    = {{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN}};
    auto const [out_keys, out_lists] = groupby_collect_set(
      keys,
      vals,
      cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
        cudf::null_policy::EXCLUDE, cudf::null_equality::EQUAL, cudf::nan_equality::UNEQUAL));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null equal with nan equal
  {
    vals_expected = {
      {{-2.3e-5f, 1.0f, 2.3e5f, NAN, 0.0f}, validity_col{true, true, true, true, false}}};
    auto const [out_keys, out_lists] = groupby_collect_set(
      keys,
      vals,
      cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
        cudf::null_policy::INCLUDE, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }

  // null unequal with nan equal
  {
    vals_expected                    = {{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, 0.0f, 0.0f},
                                         validity_col{true, true, true, true, false, false}}};
    auto const [out_keys, out_lists] = groupby_collect_set(
      keys,
      vals,
      cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
        cudf::null_policy::INCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
  }
}

TYPED_TEST(CollectSetTypedTest, CollectWithNulls)
{
  using vals_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Just use an arbitrary value to store null entries
  // Using this alias variable will make the code look cleaner
  constexpr int32_t null = 0;

  // Pre-sorted keys
  {
    keys_col keys{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    vals_col vals{{10, 10, null, null, 20, null, null, null, 30, 31, 30, 31},
                  {true, true, false, false, true, false, false, false, true, true, true, true}};
    keys_col keys_expected{1, 2, 3};
    lists_col vals_expected;

    // By default, nulls are consider equals, thus only one null is kept per key
    {
      vals_expected = {{{10, null}, validity_col{true, false}},
                       {{20, null}, validity_col{true, false}},
                       {{30, 31}, validity_col{true, true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are kept (nulls are put at the end of each list)
    {
      vals_expected = lists_col{{{10, null, null}, validity_col{true, false, false}},
                                {{20, null, null, null}, validity_col{true, false, false, false}},
                                {{30, 31}, validity_col{true, true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_unequal());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are excluded
    {
      vals_expected = lists_col{{10}, {20}, {30, 31}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_exclude());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }
  }

  // Expect the result keys to be sorted by sort-based groupby
  {
    keys_col keys{4, 1, 2, 4, 3, 3, 3, 3, 2, 1};
    vals_col vals{{40, 10, 20, 40, null, null, null, null, 21, null},
                  {true, true, true, true, false, false, false, false, true, false}};
    keys_col keys_expected{1, 2, 3, 4};
    lists_col vals_expected;

    // By default, nulls are consider equals, thus only one null is kept per key
    {
      vals_expected = {{{10, null}, validity_col{true, false}},
                       {{20, 21}, validity_col{true, true}},
                       {{null}, validity_col{false}},
                       {{40}, validity_col{true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are kept (nulls are put at the end of each list)
    {
      vals_expected =
        lists_col{{{10, null}, validity_col{true, false}},
                  {{20, 21}, validity_col{true, true}},
                  {{null, null, null, null}, validity_col{false, false, false, false}},
                  {{40}, validity_col{true}}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_unequal());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }

    // All nulls per key are excluded
    {
      vals_expected = lists_col{{10}, {20, 21}, {}, {40}};
      auto const [out_keys, out_lists] =
        groupby_collect_set(keys, vals, CollectSetTest::collect_set_null_exclude());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys_expected, *out_keys, verbosity);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_expected, *out_lists, verbosity);
    }
  }
}
