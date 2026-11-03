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

using key_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

template <typename T>
struct groupby_count_scan_test : public cudf::test::BaseFixture {
  using V              = T;
  using R              = cudf::size_type;
  using value_wrapper  = cudf::test::fixed_width_column_wrapper<V, int32_t>;
  using result_wrapper = cudf::test::fixed_width_column_wrapper<R, int32_t>;
};

TYPED_TEST_SUITE(groupby_count_scan_test, cudf::test::AllTypes);

TYPED_TEST(groupby_count_scan_test, basic)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  value_wrapper vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  key_wrapper expect_keys   {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  result_wrapper expect_vals{1, 2, 3, 1, 2, 3, 4, 1, 2, 3};
  // clang-format on

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));

  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, empty_cols)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys;
  value_wrapper vals;
  key_wrapper expect_keys;
  result_wrapper expect_vals;

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));

  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, zero_valid_keys)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  value_wrapper vals{3, 4, 5};
  key_wrapper expect_keys{};
  result_wrapper expect_vals{};

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));

  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, zero_valid_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys{1, 1, 1};
  value_wrapper vals({3, 4, 5}, cudf::test::iterators::all_nulls());
  key_wrapper expect_keys{1, 1, 1};
  result_wrapper expect_vals{0, 0, 0};

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));

  expect_vals = result_wrapper{1, 2, 3};
  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_scan_test, null_keys_and_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys(  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4}, {true, true, true, true, true, true, true, false, true, true, true});
  value_wrapper vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //                        {1, 1, 1, 2, 2, 2, 2, 3, _, 3, 4}
  key_wrapper expect_keys(  {1, 1, 1, 2, 2, 2, 2, 3,    3, 4}, cudf::test::iterators::no_nulls());
  //                        {_, 3, 6, 1, 4, _, 9, 2, 7, 8, _}
  result_wrapper expect_vals{0, 1, 2, 1, 2, 2, 3, 1,    2, 0};
  // clang-format on

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));

  expect_vals = result_wrapper{1, 2, 3, 1, 2, 3, 4, 1, 2, 1};
  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

struct groupby_count_scan_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_count_scan_string_test, basic)
{
  using R              = cudf::size_type;
  using result_wrapper = cudf::test::fixed_width_column_wrapper<R, int32_t>;

  // clang-format off
  key_wrapper keys                      {  1,   3,   3,   5,   5,   0};
  cudf::test::strings_column_wrapper vals{"1", "1", "1", "1", "1", "1"};
  key_wrapper expect_keys   {0, 1, 3, 3, 5, 5};
  result_wrapper expect_vals{1, 1, 1, 2, 1, 2};
  // clang-format on

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));

  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

template <typename T>
struct GroupByCountScanFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupByCountScanFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupByCountScanFixedPointTest, GroupByCountScan)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  using R              = cudf::size_type;
  using result_wrapper = cudf::test::fixed_width_column_wrapper<R, int32_t>;

  auto const scale       = scale_type{-1};
  auto const keys        = key_wrapper{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals        = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
  auto const expect_keys = key_wrapper{1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  auto const expect_vals = result_wrapper{1, 2, 3, 1, 2, 3, 4, 1, 2, 3};

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));

  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

struct groupby_dictionary_count_scan_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_count_scan_test, basic)
{
  using K              = int32_t;
  using R              = cudf::size_type;
  using result_wrapper = cudf::test::fixed_width_column_wrapper<R, int32_t>;

  cudf::test::strings_column_wrapper keys{"1", "3", "3", "5", "5", "0"};
  cudf::test::dictionary_column_wrapper<K> vals{1, 1, 1, 1, 1, 1};
  cudf::test::strings_column_wrapper expect_keys{"0", "1", "3", "3", "5", "5"};
  result_wrapper expect_vals{1, 1, 1, 2, 1, 2};

  auto agg1 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg1));
  auto agg2 =
    cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE);
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg2));
}
