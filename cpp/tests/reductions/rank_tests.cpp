/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scan_tests.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <thrust/host_vector.h>

using rank_result_col    = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
using percent_result_col = cudf::test::fixed_width_column_wrapper<double>;

auto const rank = cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::MIN);
auto const dense_rank =
  cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE);
auto const percent_rank =
  cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::MIN,
                                                      {},
                                                      cudf::null_policy::INCLUDE,
                                                      {},
                                                      cudf::rank_percentage::ONE_NORMALIZED);

auto constexpr INCLUSIVE_SCAN = cudf::scan_type::INCLUSIVE;
auto constexpr INCLUDE_NULLS  = cudf::null_policy::INCLUDE;

template <typename T>
struct TypedRankScanTest : BaseScanTest<T> {
  inline void test_ungrouped_rank_scan(cudf::column_view const& input,
                                       cudf::column_view const& expect_vals,
                                       cudf::scan_aggregation const& agg)
  {
    auto col_out = cudf::scan(input, agg, INCLUSIVE_SCAN, INCLUDE_NULLS);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals, col_out->view());
  }
};

using RankTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::FixedPointTypes,
                                     cudf::test::ChronoTypes,
                                     cudf::test::StringTypes>;

TYPED_TEST_SUITE(TypedRankScanTest, RankTypes);

TYPED_TEST(TypedRankScanTest, Rank)
{
  auto const v = [] {
    if (std::is_signed_v<TypeParam>)
      return make_vector<TypeParam>({-120, -120, -120, -16, -16, 5, 6, 6, 6, 6, 34, 113});
    return make_vector<TypeParam>({5, 5, 5, 6, 6, 9, 11, 11, 11, 11, 14, 34});
  }();
  auto const col = this->make_column(v);

  auto const expected_dense   = rank_result_col{1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6};
  auto const expected_rank    = rank_result_col{1, 1, 1, 4, 4, 6, 7, 7, 7, 7, 11, 12};
  auto const expected_percent = percent_result_col{0.0,
                                                   0.0,
                                                   0.0,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   5.0 / 11,
                                                   6.0 / 11,
                                                   6.0 / 11,
                                                   6.0 / 11,
                                                   6.0 / 11,
                                                   10.0 / 11,
                                                   11.0 / 11};
  this->test_ungrouped_rank_scan(*col, expected_dense, *dense_rank);
  this->test_ungrouped_rank_scan(*col, expected_rank, *rank);
  this->test_ungrouped_rank_scan(*col, expected_percent, *percent_rank);
}

TYPED_TEST(TypedRankScanTest, RankWithNulls)
{
  auto const v = [] {
    if (std::is_signed_v<TypeParam>)
      return make_vector<TypeParam>({-120, -120, -120, -16, -16, 5, 6, 6, 6, 6, 34, 113});
    return make_vector<TypeParam>({5, 5, 5, 6, 6, 9, 11, 11, 11, 11, 14, 34});
  }();
  auto const null_iter = cudf::test::iterators::nulls_at({3, 6, 7, 11});
  auto const b         = thrust::host_vector<bool>(null_iter, null_iter + v.size());
  auto col             = this->make_column(v, b);

  auto const expected_dense   = rank_result_col{1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8};
  auto const expected_rank    = rank_result_col{1, 1, 1, 4, 5, 6, 7, 7, 9, 9, 11, 12};
  auto const expected_percent = percent_result_col{0.0,
                                                   0.0,
                                                   0.0,
                                                   3.0 / 11,
                                                   4.0 / 11,
                                                   5.0 / 11,
                                                   6.0 / 11,
                                                   6.0 / 11,
                                                   8.0 / 11,
                                                   8.0 / 11,
                                                   10.0 / 11,
                                                   11.0 / 11};
  this->test_ungrouped_rank_scan(*col, expected_dense, *dense_rank);
  this->test_ungrouped_rank_scan(*col, expected_rank, *rank);
  this->test_ungrouped_rank_scan(*col, expected_percent, *percent_rank);
}

namespace {
template <typename TypeParam>
auto make_input_column()
{
  if constexpr (std::is_same_v<TypeParam, cudf::string_view>) {
    return cudf::test::strings_column_wrapper{
      {"0", "0", "4", "4", "4", "", "7", "7", "7", "9", "9", "9"},
      cudf::test::iterators::null_at(5)};
  } else {
    using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;
    return (std::is_signed_v<TypeParam>)
             ? fw_wrapper{{-1, -1, -4, -4, -4, 5, 7, 7, 7, 9, 9, 9},
                          cudf::test::iterators::null_at(5)}
             : fw_wrapper{{0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9}, cudf::test::iterators::null_at(5)};
  }
}

auto make_strings_column()
{
  return cudf::test::strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "", "9", "9", "10d"},
    cudf::test::iterators::null_at(8)};
}

template <typename TypeParam>
auto make_mixed_structs_column()
{
  auto col     = make_input_column<TypeParam>();
  auto strings = make_strings_column();
  return cudf::test::structs_column_wrapper{{col, strings}};
}
}  // namespace

TYPED_TEST(TypedRankScanTest, MixedStructs)
{
  auto const struct_col       = make_mixed_structs_column<TypeParam>();
  auto const expected_dense   = rank_result_col{1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8};
  auto const expected_rank    = rank_result_col{1, 1, 3, 3, 5, 6, 7, 7, 9, 10, 10, 12};
  auto const expected_percent = percent_result_col{0.0,
                                                   0.0,
                                                   2.0 / 11,
                                                   2.0 / 11,
                                                   4.0 / 11,
                                                   5.0 / 11,
                                                   6.0 / 11,
                                                   6.0 / 11,
                                                   8.0 / 11,
                                                   9.0 / 11,
                                                   9.0 / 11,
                                                   11.0 / 11};

  this->test_ungrouped_rank_scan(struct_col, expected_dense, *dense_rank);
  this->test_ungrouped_rank_scan(struct_col, expected_rank, *rank);
  this->test_ungrouped_rank_scan(struct_col, expected_percent, *percent_rank);
}

TYPED_TEST(TypedRankScanTest, NestedStructs)
{
  auto const nested_col = [&] {
    auto struct_col = [&] {
      auto col     = make_input_column<TypeParam>();
      auto strings = make_strings_column();
      return cudf::test::structs_column_wrapper{{col, strings}};
    }();
    auto col = make_input_column<TypeParam>();
    return cudf::test::structs_column_wrapper{{struct_col, col}};
  }();

  auto const flat_col = [&] {
    auto col         = make_input_column<TypeParam>();
    auto strings_col = make_strings_column();
    auto nuther_col  = make_input_column<TypeParam>();
    return cudf::test::structs_column_wrapper{{col, strings_col, nuther_col}};
  }();

  auto const dense_out      = cudf::scan(nested_col, *dense_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  auto const dense_expected = cudf::scan(flat_col, *dense_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), dense_expected->view());

  auto const rank_out      = cudf::scan(nested_col, *rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  auto const rank_expected = cudf::scan(flat_col, *rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), rank_expected->view());

  auto const percent_out = cudf::scan(nested_col, *percent_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  auto const percent_expected = cudf::scan(flat_col, *percent_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(percent_out->view(), percent_expected->view());
}

TYPED_TEST(TypedRankScanTest, StructsWithNullPushdown)
{
  auto struct_col = make_mixed_structs_column<TypeParam>().release();

  // First, verify that if the structs column has only nulls, all output rows are ranked 1.
  {
    // Null mask not pushed down to members.
    struct_col->set_null_mask(create_null_mask(struct_col->size(), cudf::mask_state::ALL_NULL),
                              struct_col->size());
    auto const expected_null_result = rank_result_col{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto const expected_percent_rank_null_result =
      percent_result_col{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto const dense_out   = cudf::scan(*struct_col, *dense_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
    auto const rank_out    = cudf::scan(*struct_col, *rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
    auto const percent_out = cudf::scan(*struct_col, *percent_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), expected_null_result);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), expected_null_result);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(percent_out->view(), expected_percent_rank_null_result);
  }

  // Next, verify that if the structs column a null mask that is NOT pushed down to members,
  // the ranks are still correct.
  {
    auto const null_iter = cudf::test::iterators::nulls_at({1, 2});
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(null_iter, null_iter + struct_col->size());
    struct_col->set_null_mask(std::move(null_mask), null_count);
    auto const expected_dense   = rank_result_col{1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9};
    auto const expected_rank    = rank_result_col{1, 2, 2, 4, 5, 6, 7, 7, 9, 10, 10, 12};
    auto const expected_percent = percent_result_col{0.0,
                                                     1.0 / 11,
                                                     1.0 / 11,
                                                     3.0 / 11,
                                                     4.0 / 11,
                                                     5.0 / 11,
                                                     6.0 / 11,
                                                     6.0 / 11,
                                                     8.0 / 11,
                                                     9.0 / 11,
                                                     9.0 / 11,
                                                     11.0 / 11};
    auto const dense_out   = cudf::scan(*struct_col, *dense_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
    auto const rank_out    = cudf::scan(*struct_col, *rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
    auto const percent_out = cudf::scan(*struct_col, *percent_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), expected_dense);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), expected_rank);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(percent_out->view(), expected_percent);
  }
}

struct RankScanTest : public cudf::test::BaseFixture {};

TEST(RankScanTest, BoolRank)
{
  auto const vals =
    cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto const expected_dense   = rank_result_col{1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  auto const expected_rank    = rank_result_col{1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  auto const expected_percent = percent_result_col{0.0,
                                                   0.0,
                                                   0.0,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11};

  auto const dense_out   = cudf::scan(vals, *dense_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  auto const rank_out    = cudf::scan(vals, *rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  auto const percent_out = cudf::scan(vals, *percent_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_dense, dense_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_rank, rank_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_percent, percent_out->view());
}

TEST(RankScanTest, BoolRankWithNull)
{
  auto const vals = cudf::test::fixed_width_column_wrapper<bool>{
    {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}, cudf::test::iterators::nulls_at({8, 9, 10, 11})};
  auto const expected_dense   = rank_result_col{1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3};
  auto const expected_rank    = rank_result_col{1, 1, 1, 4, 4, 4, 4, 4, 9, 9, 9, 9};
  auto const expected_percent = percent_result_col{0.0,
                                                   0.0,
                                                   0.0,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   3.0 / 11,
                                                   8.0 / 11,
                                                   8.0 / 11,
                                                   8.0 / 11,
                                                   8.0 / 11};

  auto nullable_dense_out   = cudf::scan(vals, *dense_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  auto nullable_rank_out    = cudf::scan(vals, *rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  auto nullable_percent_out = cudf::scan(vals, *percent_rank, INCLUSIVE_SCAN, INCLUDE_NULLS);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_dense, nullable_dense_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_rank, nullable_rank_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_percent, nullable_percent_out->view());
}

TEST(RankScanTest, ExclusiveScan)
{
  auto const vals = cudf::test::fixed_width_column_wrapper<uint32_t>{3, 4, 5};

  // Only inclusive scans are supported, so these should all raise exceptions.
  EXPECT_THROW(cudf::scan(vals, *dense_rank, cudf::scan_type::EXCLUSIVE, INCLUDE_NULLS),
               cudf::logic_error);
  EXPECT_THROW(cudf::scan(vals, *rank, cudf::scan_type::EXCLUSIVE, INCLUDE_NULLS),
               cudf::logic_error);
  EXPECT_THROW(cudf::scan(vals, *percent_rank, cudf::scan_type::EXCLUSIVE, INCLUDE_NULLS),
               cudf::logic_error);
}
