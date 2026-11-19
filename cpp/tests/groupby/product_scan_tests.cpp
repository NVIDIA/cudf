/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

#include <cuda/std/type_traits>

#include <type_traits>

using key_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

template <typename T>
struct groupby_product_scan_test : public cudf::test::BaseFixture {
  using V              = T;
  using R              = std::conditional_t<cuda::std::is_integral_v<V>, int64_t, V>;
  using value_wrapper  = cudf::test::fixed_width_column_wrapper<V, int32_t>;
  using result_wrapper = cudf::test::fixed_width_column_wrapper<R, int32_t>;
};

using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>>;

TYPED_TEST_SUITE(groupby_product_scan_test, supported_types);

TYPED_TEST(groupby_product_scan_test, basic)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  value_wrapper vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  key_wrapper expect_keys   {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  //                        {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  result_wrapper expect_vals{0, 0, 0, 1, 4, 20, 180, 2, 14, 112};
  // clang-format on
  auto agg = cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_product_scan_test, pre_sorted)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys  {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  value_wrapper vals{0, 3, 6, 1, 4, 5, 9, 2, 7, 8};

  key_wrapper expect_keys   {1, 1, 1, 2, 2,  2,  2, 3, 3, 3};
  result_wrapper expect_vals{0, 0, 0, 1, 4, 20, 180, 2, 14, 112};
  // clang-format on

  auto agg = cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys,
                   vals,
                   expect_keys,
                   expect_vals,
                   std::move(agg),
                   cudf::null_policy::EXCLUDE,
                   cudf::sorted::YES);
}

TYPED_TEST(groupby_product_scan_test, empty_cols)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys{};
  value_wrapper vals{};

  key_wrapper expect_keys{};
  result_wrapper expect_vals{};

  auto agg = cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_product_scan_test, zero_valid_keys)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  value_wrapper vals{3, 4, 5};
  key_wrapper expect_keys{};
  result_wrapper expect_vals{};

  auto agg = cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_product_scan_test, zero_valid_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  key_wrapper keys{1, 1, 1};
  value_wrapper vals({3, 4, 5}, cudf::test::iterators::all_nulls());
  key_wrapper expect_keys{1, 1, 1};
  result_wrapper expect_vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  auto agg = cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_product_scan_test, null_keys_and_values)
{
  using value_wrapper  = typename TestFixture::value_wrapper;
  using result_wrapper = typename TestFixture::result_wrapper;

  // clang-format off
  key_wrapper keys(  {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4}, {true, true, true, true, true, true, true, false, true, true, true});
  value_wrapper vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4}, {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //                         { 1, 1, 1, 2, 2,  2,  2, 3, *, 3, 4};
  key_wrapper expect_keys(   { 1, 1, 1, 2, 2,  2,  2, 3,    3, 4}, cudf::test::iterators::no_nulls());
                          // { -, 3, 6, 1, 4,  -,  9, 2, _, 8, -}
  result_wrapper expect_vals({-1, 3, 18, 1, 4, -1, 36, 2,   16, -1},
                             { 0, 1, 1, 1, 1,  0,  1, 1,    1, 0});
  // clang-format on

  auto agg = cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(keys, vals, expect_keys, expect_vals, std::move(agg));
}
