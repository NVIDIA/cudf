/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

using namespace cudf::test::iterators;

using key_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

template <typename T>
struct groupby_max_scan_test : public cudf::test::BaseFixture {
  using V              = T;
  using R              = V;
  using value_wrapper  = cudf::test::fixed_width_column_wrapper<V, int32_t>;
  using result_wrapper = cudf::test::fixed_width_column_wrapper<R, int32_t>;
};

TYPED_TEST_SUITE(groupby_max_scan_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(groupby_max_scan_test, basic)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys   {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  value_wrapper vals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4});

  key_wrapper expect_keys    {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
                          // {5, 8, 1, 6, 9, 0, 4, 7, 2, 3}
  result_wrapper expect_vals({5, 8, 8, 6, 9, 9, 9, 7, 7, 7});
  // clang-format on

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_max_scan_test, pre_sorted)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys   {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  value_wrapper vals({5, 8, 1, 6, 9, 0, 4, 7, 2, 3});

  key_wrapper expect_keys    {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  result_wrapper expect_vals({5, 8, 8, 6, 9, 9, 9, 7, 7, 7});
  // clang-format on

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys,
                   vals,
                   expect_keys,
                   expect_vals,
                   std::move(agg),
                   cudf::null_policy::EXCLUDE,
                   cudf::sorted::YES);
}

TYPED_TEST(groupby_max_scan_test, empty_cols)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys{};
  value_wrapper vals{};
  key_wrapper expect_keys{};
  result_wrapper expect_vals{};

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_max_scan_test, zero_valid_keys)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys({1, 2, 3}, all_nulls());
  value_wrapper vals({3, 4, 5});
  key_wrapper expect_keys{};
  result_wrapper expect_vals{};

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_max_scan_test, zero_valid_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys{1, 1, 1};
  value_wrapper vals({3, 4, 5}, all_nulls());
  key_wrapper expect_keys{1, 1, 1};
  result_wrapper expect_vals({-1, -1, -1}, all_nulls());

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_max_scan_test, null_keys_and_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys(  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4}, {true, true, true, true, true, true, true, false, true, true, true});
  value_wrapper vals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 4}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

                         //  {1, 1, 1, 2, 2, 2, 2, 3,   _, 3, 4}
  key_wrapper expect_keys(   {1, 1, 1, 2, 2, 2, 2, 3,      3, 4}, no_nulls() );
                         //  { -, 3, 6, 1, 4,  -, 9, 2, _, 8, -}
  result_wrapper expect_vals({-1, 8, 8, 6, 9, -1, 9, 7,    7, -1},
                             { 0, 1, 1, 1, 1,  0, 1, 1,    1, 0});
  // clang-format on

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

struct groupby_max_scan_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_max_scan_string_test, basic)
{
  key_wrapper keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};

  key_wrapper expect_keys{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  cudf::test::strings_column_wrapper expect_vals(
    {"año", "año", "año", "bit", "zit", "zit", "zit", "₹1", "₹1", "₹1"});

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

template <typename T>
struct GroupByMaxScanFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupByMaxScanFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupByMaxScanFixedPointTest, GroupBySortMaxScanDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys = key_wrapper{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals = fp_wrapper{{5, 6, 7, 8, 9, 0, 1, 2, 3, 4}, scale};

    //                                      {5, 8, 1, 6, 9, 0, 4, 7, 2, 3}
    auto const expect_keys     = key_wrapper{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    auto const expect_vals_max = fp_wrapper{{5, 8, 8, 6, 9, 9, 9, 7, 7, 7}, scale};
    // clang-format on

    auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
    test_single_scan(keys, vals, expect_keys, expect_vals_max, std::move(agg));
  }
}

struct groupby_max_scan_struct_test : public cudf::test::BaseFixture {};

TEST_F(groupby_max_scan_struct_test, basic)
{
  auto const keys = key_wrapper{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const expect_keys = key_wrapper{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  auto const expect_vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "año", "año", "bit", "zit", "zit", "zit", "₹1", "₹1", "₹1"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1, 1, 2, 5, 5, 5, 3, 3, 3};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_max_scan_struct_test, slice_input)
{
  constexpr int32_t dont_care{1};
  auto const keys_original =
    key_wrapper{dont_care, dont_care, 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, dont_care};
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
    auto child2 = key_wrapper{dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const keys        = cudf::slice(keys_original, {2, 12})[0];
  auto const vals        = cudf::slice(vals_original, {2, 12})[0];
  auto const expect_keys = key_wrapper{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  auto const expect_vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "año", "año", "bit", "zit", "zit", "zit", "₹1", "₹1", "₹1"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1, 1, 2, 5, 5, 5, 3, 3, 3};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_max_scan_struct_test, null_keys_and_values)
{
  constexpr int32_t null{0};
  auto const keys = key_wrapper{{1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4}, null_at(7)};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "" /*NULL*/, "" /*NULL*/, "$1", "€1", "wut", "" /*NULL*/};
    auto child2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{9, 8, 7, 6, 5, null, null, 2, 1, 0, null};
    return cudf::test::structs_column_wrapper{{child1, child2}, nulls_at({5, 6, 10})};
  }();

  auto const expect_keys = key_wrapper{{1, 1, 1, 2, 2, 2, 2, 3, 3, 4}, no_nulls()};
  auto const expect_vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "año", "" /*NULL*/, "bit", "zit", "" /*NULL*/, "zit", "₹1", "₹1", "" /*NULL*/};
    auto child2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{9, 9, null, 8, 5, null, 5, 7, 7, null};
    return cudf::test::structs_column_wrapper{{child1, child2}, nulls_at({2, 5, 9})};
  }();

  auto agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}
