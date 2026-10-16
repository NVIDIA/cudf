/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

#include <type_traits>

using key_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

template <typename T>
struct groupby_sum_scan_test : public cudf::test::BaseFixture {
  using V              = T;
  using R              = std::conditional_t<std::is_integral_v<V>, int64_t, V>;
  using value_wrapper  = cudf::test::fixed_width_column_wrapper<V, int32_t>;
  using result_wrapper = cudf::test::fixed_width_column_wrapper<R, int32_t>;
};

using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_SUITE(groupby_sum_scan_test, supported_types);

TYPED_TEST(groupby_sum_scan_test, basic)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  value_wrapper vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  key_wrapper expect_keys   {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  //                        {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  result_wrapper expect_vals{0, 3, 9, 1, 5, 10, 19, 2, 9, 17};
  // clang-format on
  auto agg = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, pre_sorted)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys  {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  value_wrapper vals{0, 3, 6, 1, 4, 5, 9, 2, 7, 8};

  key_wrapper expect_keys   {1, 1, 1, 2, 2,  2,  2, 3, 3, 3};
  result_wrapper expect_vals{0, 3, 9, 1, 5, 10, 19, 2, 9, 17};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys,
                   vals,
                   expect_keys,
                   expect_vals,
                   std::move(agg),
                   cudf::null_policy::EXCLUDE,
                   cudf::sorted::YES);
}

TYPED_TEST(groupby_sum_scan_test, empty_cols)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys{};
  value_wrapper vals{};

  key_wrapper expect_keys{};
  result_wrapper expect_vals{};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, zero_valid_keys)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  value_wrapper vals{3, 4, 5};
  key_wrapper expect_keys{};
  result_wrapper expect_vals{};

  auto agg = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, zero_valid_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys{1, 1, 1};
  value_wrapper vals({3, 4, 5}, cudf::test::iterators::all_nulls());
  key_wrapper expect_keys{1, 1, 1};
  result_wrapper expect_vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  auto agg = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_scan_test, null_keys_and_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys(  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4}, {true, true, true, true, true, true, true, false, true, true, true});
  value_wrapper vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //                         { 1, 1, 1, 2, 2,  2,  2, 3, *, 3, 4};
  key_wrapper expect_keys(   { 1, 1, 1, 2, 2,  2,  2, 3,    3, 4}, cudf::test::iterators::no_nulls());
                          // { -, 3, 6, 1, 4,  -,  9, 2, _, 8, -}
  result_wrapper expect_vals({-1, 3, 9, 1, 5, -1, 14, 2,   10, -1},
                             { 0, 1, 1, 1, 1,  0,  1, 1,    1, 0});
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

template <typename T>
struct GroupBySumScanFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupBySumScanFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupBySumScanFixedPointTest, SumScanDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys = key_wrapper{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    auto const expect_keys     = key_wrapper{1, 1, 1, 2, 2,  2,  2, 3, 3,  3};
    auto const expect_vals_sum = fp_wrapper{{0, 3, 9, 1, 5, 10, 19, 2, 9, 17}, scale};
    // clang-format on

    auto agg2 = cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
    test_single_scan(keys, vals, expect_keys, expect_vals_sum, std::move(agg2));
  }
}
