/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_test_fixtures.hpp"

#include <cudf_test/testing_main.hpp>

#include <cudf/filling.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

#include <limits>

TEST_F(DistinctJoinTest, DistinctLargeExtentOverflowPrevention)
{
  // Test that validates size_t extent can handle hash table sizes that would
  // overflow int32_t extent when load_factor < 1.0

  // Use a table size that when divided by low load factor would exceed INT32_MAX
  // if using int32_t extent, but should work fine with size_t extent
  constexpr cudf::size_type table_size = 10000000;  // 10M rows
  constexpr double load_factor         = 0.004;     // Hash table extent would be ~2.5B

  // Validate our test assumptions
  constexpr auto expected_hash_size = static_cast<size_t>(table_size / load_factor);
  static_assert(expected_hash_size > std::numeric_limits<cudf::size_type>::max(),
                "Hash table size should be significant");

  auto const init = cudf::numeric_scalar<cudf::size_type>{0};
  auto build_col  = cudf::sequence(table_size, init, cudf::numeric_scalar<cudf::size_type>{1});

  auto right_table = cudf::table_view{{build_col->view()}};
  cudf::table empty_probe_table{};

  // This should succeed with size_t extent - would have failed with int32_t extent
  // in scenarios approaching the overflow boundary
  EXPECT_NO_THROW({
    auto distinct_join = cudf::distinct_hash_join(
      right_table, cudf::null_equality::EQUAL, load_factor, cudf::get_default_stream());
    auto result = distinct_join.inner_join(empty_probe_table);
  });
}

TEST_P(SemiAntiJoinTest, AntiSemiJoinLargeExtentOverflowPrevention)
{
  auto const implementation = GetParam();
  if (use_mark_join(implementation)) { GTEST_SKIP(); }
  // Test validates size_t extent can handle bucket storage sizes that would
  // overflow int32_t extent when compute_bucket_storage_size() uses low load factors
  constexpr cudf::size_type table_size = 10000000;  // 10M rows
  constexpr double load_factor         = 0.004;     // Hash table extent would be ~2.5B

  // Compile-time validation
  constexpr auto expected_bucket_size = static_cast<size_t>(table_size / load_factor);
  static_assert(expected_bucket_size > std::numeric_limits<cudf::size_type>::max(),
                "Bucket storage size should be significant");

  // Create test tables and validate semi join operations succeed
  auto const init = cudf::numeric_scalar<cudf::size_type>{0};
  auto right_col  = cudf::sequence(table_size, init, cudf::numeric_scalar<cudf::size_type>{1});

  auto right_table = cudf::table_view{{right_col->view()}};
  cudf::table empty_left_table{};

  // Test with load factors that would cause overflow in int32_t extent
  EXPECT_NO_THROW({
    cudf::filtered_join obj(
      right_table, cudf::null_equality::EQUAL, load_factor, cudf::get_default_stream());
    auto result = obj.semi_join(
      empty_left_table, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    result = obj.anti_join(
      empty_left_table, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  });
}

INSTANTIATE_TEST_SUITE_P(JoinImpl,
                         SemiAntiJoinTest,
                         ::testing::Values(join_implementation::FILTERED_JOIN,
                                           join_implementation::MARK_JOIN,
                                           join_implementation::MARK_JOIN_PREFILTER),
                         [](auto const& info) { return test_name(info.param); });

CUDF_TEST_PROGRAM_MAIN()
