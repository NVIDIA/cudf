/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/reductions/scan_tests.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/reduction.hpp>

#include <thrust/host_vector.h>

#include <limits>
#include <vector>

// ============== Scan Min ==============
template <typename T>
struct ScanMinTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(ScanMinTest, NumericTypesNotBool);  // other types handled below

TYPED_TEST(ScanMinTest, InclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 4, 4, 0, 0, 0, 0, 0});

  auto result = cudf::scan(
    col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMinTest, ExclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3});
  auto const identity      = std::is_floating_point_v<TypeParam>
                               ? std::numeric_limits<TypeParam>::infinity()
                               : std::numeric_limits<TypeParam>::max();
  auto const expected_vals = std::vector<TypeParam>({identity, 5, 4, 4, 0, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<TypeParam> expected(expected_vals.begin(),
                                                             expected_vals.end());

  auto result = cudf::scan(
    col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMinTest, InclusiveWithNullsExclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 4, 4, 4, 1, 1, 1, 1},
                                                                      {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(col,
                           *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMinTest, InclusiveWithNullsInclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 4, 4, 0, 0, 0, 0, 0},
                                                                      {1, 1, 1, 0, 0, 0, 0, 0});

  auto result = cudf::scan(col,
                           *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMinTest, ExclusiveWithNullsExclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  auto const identity = std::is_floating_point_v<TypeParam>
                          ? std::numeric_limits<TypeParam>::infinity()
                          : std::numeric_limits<TypeParam>::max();
  thrust::host_vector<TypeParam> expected_vals{identity, 5, 4, 4, 4, 1, 1, 1};
  thrust::host_vector<bool> validity{1, 1, 1, 0, 1, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected(
    expected_vals.begin(), expected_vals.end(), validity.begin());

  auto result = cudf::scan(col,
                           *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::EXCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// ============== Scan Max ==============
template <typename T>
struct ScanMaxTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanMaxTest, NumericTypesNotBool);  // other types handled below

TYPED_TEST(ScanMaxTest, InclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 5, 6, 6, 6, 6, 6, 6});

  auto result = cudf::scan(
    col, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMaxTest, ExclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3});
  auto const identity = std::is_floating_point_v<TypeParam>
                          ? -std::numeric_limits<TypeParam>::infinity()
                          : std::numeric_limits<TypeParam>::lowest();
  thrust::host_vector<TypeParam> expected_vals{identity, 5, 5, 6, 6, 6, 6, 6};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected(expected_vals.begin(),
                                                             expected_vals.end());

  auto result = cudf::scan(
    col, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMaxTest, InclusiveWithNullsExclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 5, 6, 6, 6, 6, 6, 6},
                                                                      {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(col,
                           *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMaxTest, InclusiveWithNullsInclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 5, 6, 0, 0, 0, 0, 0},
                                                                      {1, 1, 1, 0, 0, 0, 0, 0});

  auto result = cudf::scan(col,
                           *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// ============== Scan Sum ==============
template <typename T>
struct ScanSumTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanSumTest, NumericTypesNotBool);  // other types handled below

TYPED_TEST(ScanSumTest, InclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected(
    {5, 9, 15, 15, 16, 22, 27, 30});

  auto result = cudf::scan(
    col, *cudf::make_sum_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanSumTest, ExclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3});
  auto const zero = TypeParam{0};
  thrust::host_vector<TypeParam> expected_vals{zero, 5, 9, 15, 15, 16, 22, 27};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected(expected_vals.begin(),
                                                             expected_vals.end());

  auto result = cudf::scan(
    col, *cudf::make_sum_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanSumTest, InclusiveWithNullsExclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected(
    {5, 9, 15, 15, 16, 22, 27, 30}, {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(col,
                           *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanSumTest, InclusiveWithNullsInclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 9, 15, 0, 0, 0, 0, 0},
                                                                      {1, 1, 1, 0, 0, 0, 0, 0});

  auto result = cudf::scan(col,
                           *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// ============== Scan Product ==============
template <typename T>
struct ScanProductTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanProductTest, NumericTypesNotBool);  // other types handled below

TYPED_TEST(ScanProductTest, InclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 1, 1, 6, 5, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected(
    {5, 20, 120, 120, 120, 720, 3600, 10800});

  auto result = cudf::scan(
    col, *cudf::make_product_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanProductTest, ExclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 1, 1, 6, 5, 3});
  auto const one     = TypeParam{1};
  auto expected_vals = make_vector<TypeParam>({5, 20, 120, 120, 120, 720, 3600});
  expected_vals.insert(expected_vals.begin(), one);
  cudf::test::fixed_width_column_wrapper<TypeParam> expected(expected_vals.begin(),
                                                             expected_vals.end());

  auto result = cudf::scan(
    col, *cudf::make_product_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanProductTest, InclusiveWithNullsExclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected(
    {5, 20, 120, 120, 120, 720, 3600, 10800}, {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(col,
                           *cudf::make_product_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanProductTest, InclusiveWithNullsInclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 20, 120, 0, 0, 0, 0, 0},
                                                                      {1, 1, 1, 0, 0, 0, 0, 0});

  auto result = cudf::scan(col,
                           *cudf::make_product_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// ============== Fixed-point Min/Max/Sum ==============
template <typename T>
struct ScanMinFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanMinFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(ScanMinFixedPointTest, InclusiveNoNulls)
{
  using RepType    = typename TypeParam::rep;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const scale = numeric::scale_type{0};

  fp_wrapper col({5, 4, 6, 0, 1, 6, 5, 3}, scale);
  fp_wrapper expected({5, 4, 4, 0, 0, 0, 0, 0}, scale);

  auto result = cudf::scan(
    col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMinFixedPointTest, InclusiveWithNullsExclude)
{
  using RepType    = typename TypeParam::rep;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const scale = numeric::scale_type{0};

  fp_wrapper col({5, 4, 6, 0, 1, 6, 5, 3}, {1, 1, 1, 0, 1, 1, 1, 1}, scale);
  fp_wrapper expected({5, 4, 4, 4, 1, 1, 1, 1}, {1, 1, 1, 0, 1, 1, 1, 1}, scale);

  auto result = cudf::scan(col,
                           *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

template <typename T>
struct ScanMaxFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanMaxFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(ScanMaxFixedPointTest, InclusiveNoNulls)
{
  using RepType    = typename TypeParam::rep;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const scale = numeric::scale_type{0};

  fp_wrapper col({5, 4, 6, 0, 1, 6, 5, 3}, scale);
  fp_wrapper expected({5, 5, 6, 6, 6, 6, 6, 6}, scale);

  auto result = cudf::scan(
    col, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanMaxFixedPointTest, InclusiveWithNullsExclude)
{
  using RepType    = typename TypeParam::rep;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const scale = numeric::scale_type{0};

  fp_wrapper col({5, 4, 6, 0, 1, 6, 5, 3}, {1, 1, 1, 0, 1, 1, 1, 1}, scale);
  fp_wrapper expected({5, 5, 6, 6, 6, 6, 6, 6}, {1, 1, 1, 0, 1, 1, 1, 1}, scale);

  auto result = cudf::scan(col,
                           *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

template <typename T>
struct ScanSumFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanSumFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(ScanSumFixedPointTest, InclusiveNoNulls)
{
  using RepType    = typename TypeParam::rep;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const scale = numeric::scale_type{0};

  fp_wrapper col({5, 4, 6, 0, 1, 6, 5, 3}, scale);
  fp_wrapper expected({5, 9, 15, 15, 16, 22, 27, 30}, scale);

  auto result = cudf::scan(
    col, *cudf::make_sum_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanSumFixedPointTest, InclusiveWithNullsExclude)
{
  using RepType    = typename TypeParam::rep;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  auto const scale = numeric::scale_type{0};

  fp_wrapper col({5, 4, 6, 0, 1, 6, 5, 3}, {1, 1, 1, 0, 1, 1, 1, 1}, scale);
  fp_wrapper expected({5, 9, 15, 15, 16, 22, 27, 30}, {1, 1, 1, 0, 1, 1, 1, 1}, scale);

  auto result = cudf::scan(col,
                           *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// ============== Empty column ==============
template <typename T>
struct ScanEmptyTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanEmptyTest, cudf::test::NumericTypes);

TYPED_TEST(ScanEmptyTest, MinInclusive)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col({});
  cudf::test::fixed_width_column_wrapper<TypeParam> expected({});

  auto result = cudf::scan(
    col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanEmptyTest, MinExclusive)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col({});
  cudf::test::fixed_width_column_wrapper<TypeParam> expected({});

  auto result = cudf::scan(
    col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// ============== Leading nulls ==============
template <typename T>
struct ScanLeadingNullsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanLeadingNullsTest, cudf::test::NumericTypes);

TYPED_TEST(ScanLeadingNullsTest, MinInclusiveExclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({10, 20, 30}, {0, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 20, 20}, {0, 1, 1});

  auto result = cudf::scan(col,
                           *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(ScanLeadingNullsTest, MinInclusiveInclude)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({10, 20, 30}, {0, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({10, 20, 20}, {0, 0, 0});

  auto result = cudf::scan(col,
                           *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// ============== String Min/Max ==============
struct ScanStringTest : public cudf::test::BaseFixture {};

TEST_F(ScanStringTest, MinMaxInclusiveNoNulls)
{
  cudf::test::strings_column_wrapper col({"a", "b", "a", "c", "b", "a"});
  cudf::test::strings_column_wrapper expected_min({"a", "a", "a", "a", "a", "a"});
  cudf::test::strings_column_wrapper expected_max({"a", "b", "b", "c", "c", "c"});

  auto result_min = cudf::scan(
    col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, result_min->view());

  auto result_max = cudf::scan(
    col, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, result_max->view());
}

TEST_F(ScanStringTest, MinMaxInclusiveWithNullsExclude)
{
  cudf::test::strings_column_wrapper col({{"a", "", "a", "c", "b", "a"}, {1, 0, 1, 1, 1, 1}});
  cudf::test::strings_column_wrapper expected_min(
    {{"a", "", "a", "a", "a", "a"}, {1, 0, 1, 1, 1, 1}});
  cudf::test::strings_column_wrapper expected_max(
    {{"a", "", "a", "c", "c", "c"}, {1, 0, 1, 1, 1, 1}});

  auto result_min = cudf::scan(col,
                               *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                               cudf::scan_type::INCLUSIVE,
                               cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, result_min->view());

  auto result_max = cudf::scan(col,
                               *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                               cudf::scan_type::INCLUSIVE,
                               cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, result_max->view());
}

// ============== Scan Count ==============
struct ScanCountTest : public cudf::test::BaseFixture {};

TEST_F(ScanCountTest, InclusiveNoNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({5, 4, 6, 0, 1, 6, 5, 3});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({1, 2, 3, 4, 5, 6, 7, 8});

  auto result = cudf::scan(
    col, *cudf::make_count_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  result =
    cudf::scan(col,
               *cudf::make_count_aggregation<cudf::scan_aggregation>(cudf::null_policy::INCLUDE),
               cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(ScanCountTest, InclusiveWithNullsExclude)
{
  cudf::test::strings_column_wrapper col({"5", "4", "6", "", "1", "6", "5", "3"},
                                         {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({1, 2, 3, 4, 4, 5, 6, 7},
                                                                   {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(col,
                           *cudf::make_count_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(ScanCountTest, InclusiveWithNullsInclude)
{
  cudf::test::strings_column_wrapper col({"5", "4", "6", "", "1", "6", "5", "3"},
                                         {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({1, 2, 3, 0, 0, 0, 0, 0},
                                                                   {1, 1, 1, 0, 0, 0, 0, 0});

  auto result =
    cudf::scan(col,
               *cudf::make_count_aggregation<cudf::scan_aggregation>(cudf::null_policy::INCLUDE),
               cudf::scan_type::INCLUSIVE,
               cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(ScanCountTest, InclusiveWithOffset)
{
  cudf::test::strings_column_wrapper col({"5", "4", "6", "", "1", "6", "5", "3"},
                                         {1, 1, 1, 0, 1, 1, 1, 1});
  auto input = cudf::slice(col, {1, 7}).front();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({1, 2, 3, 3, 4, 5},
                                                                   {1, 1, 0, 1, 1, 1});

  auto result = cudf::scan(input,
                           *cudf::make_count_aggregation<cudf::scan_aggregation>(),
                           cudf::scan_type::INCLUSIVE,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

struct ScanExclusiveErrorTest : public cudf::test::BaseFixture {};

TEST_F(ScanExclusiveErrorTest, ExclusiveThrows)
{
  cudf::test::strings_column_wrapper col({"a", "b", "c"});
  EXPECT_THROW(
    cudf::scan(
      col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::scan(
      col, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::scan(
      col, *cudf::make_count_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
}

// ============== Chrono MinMax ==============
template <typename T>
struct ScanChronoTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanChronoTest, cudf::test::ChronoTypes);

TYPED_TEST(ScanChronoTest, ChronoMinMax)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected_min({5, 4, 4, 0, 1, 1, 1, 1},
                                                                          {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(
    col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_min);

  result = cudf::scan(col,
                      *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                      cudf::scan_type::INCLUSIVE,
                      cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_min);

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected_max({5, 5, 6, 0, 6, 6, 6, 6},
                                                                          {1, 1, 1, 0, 1, 1, 1, 1});
  result = cudf::scan(
    col, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_max);

  result = cudf::scan(col,
                      *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                      cudf::scan_type::INCLUSIVE,
                      cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_max);

  EXPECT_THROW(
    cudf::scan(
      col, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::scan(
      col, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
}

// ============== Duration Sum ==============
template <typename T>
struct ScanDurationTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanDurationTest, cudf::test::DurationTypes);

TYPED_TEST(ScanDurationTest, Sum)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 9, 15, 0, 16, 22, 27, 30},
                                                                      {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(
    col, *cudf::make_sum_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result = cudf::scan(col,
                      *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                      cudf::scan_type::INCLUSIVE,
                      cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  EXPECT_THROW(
    cudf::scan(
      col, *cudf::make_sum_aggregation<cudf::scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
}

// ============== Struct scan MinMax ==============
struct StructScanTest : public cudf::test::BaseFixture {};

TEST_F(StructScanTest, StructScanMinMaxNoNull)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int32_t>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;

  auto const input = [] {
    auto child1 = STRINGS_CW{"año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = INTS_CW{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return STRUCTS_CW{{child1, child2}};
  }();

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "año", "año", "aaa", "aaa", "aaa", "aaa", "$1", "$1", "$1"};
      auto child2 = INTS_CW{1, 1, 1, 4, 4, 4, 4, 8, 8, 8};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "bit", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1"};
      auto child2 = INTS_CW{1, 2, 3, 3, 3, 3, 3, 3, 3, 3};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(StructScanTest, StructScanMinMaxSlicedInput)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;
  constexpr int32_t dont_care{1};

  auto const input_original = [] {
    auto child1 = STRINGS_CW{"$dont_care",
                             "$dont_care",
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
                             "₹dont_care"};
    auto child2 = INTS_CW{dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return STRUCTS_CW{{child1, child2}};
  }();

  auto const input = cudf::slice(input_original, {2, 12})[0];

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "año", "año", "aaa", "aaa", "aaa", "aaa", "$1", "$1", "$1"};
      auto child2 = INTS_CW{1, 1, 1, 4, 4, 4, 4, 8, 8, 8};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_min_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "bit", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1"};
      auto child2 = INTS_CW{1, 2, 3, 3, 3, 3, 3, 3, 3, 3};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_max_aggregation<cudf::scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(StructScanTest, StructScanMinMaxWithNulls)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;
  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;

  auto const input = [] {
    auto child1 = STRINGS_CW{{"año",
                              "bit",
                              "",     // child null
                              "aaa",  // parent null
                              "zit",
                              "bat",
                              "aab",
                              "",    // child null
                              "€1",  // parent null
                              "wut"},
                             nulls_at({2, 7})};
    auto child2 = INTS_CW{{1,
                           2,
                           0,  // child null
                           4,  // parent null
                           5,
                           6,
                           7,
                           0,  // child null
                           9,  // parent null
                           10},
                          nulls_at({2, 7})};
    return STRUCTS_CW{{child1, child2}, nulls_at({3, 8})};
  }();

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{{"año",
                                "año",
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/},
                               nulls_at({2, 3, 4, 5, 6, 7, 8, 9})};
      auto child2 = INTS_CW{{1,
                             1,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/},
                            nulls_at({2, 3, 4, 5, 6, 7, 8, 9})};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 8})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{
        "año", "bit", "bit", "" /*NULL*/, "zit", "zit", "zit", "zit", "" /*NULL*/, "zit"};
      auto child2 = INTS_CW{1, 2, 2, 0 /*NULL*/, 5, 5, 5, 5, 0 /*NULL*/, 5};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 8})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{{"año",
                                "año",
                                "",   // child null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                ""},  // parent null
                               null_at(2)};
      auto child2 = INTS_CW{{1,
                             1,
                             0,   // child null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0},  // parent null
                            null_at(2)};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 4, 5, 6, 7, 8, 9})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_min_aggregation<cudf::scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año",
                               "bit",
                               "bit",
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/};
      auto child2 = INTS_CW{1,
                            2,
                            2,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 4, 5, 6, 7, 8, 9})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}
