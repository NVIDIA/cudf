/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_uvector.hpp>

#include <iterator>
#include <memory>
#include <string>
#include <utility>

struct DistinctJoinTest : public cudf::test::BaseFixture {
  void compare_to_reference(
    cudf::table_view const& right_table,
    cudf::table_view const& left_table,
    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> const& result,
    cudf::table_view const& expected_table,
    cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK)
  {
    auto const& [probe_join_indices, build_join_indices] = result;

    auto build_indices_span = cudf::device_span<cudf::size_type const>{*build_join_indices};
    auto probe_indices_span = cudf::device_span<cudf::size_type const>{*probe_join_indices};

    auto build_indices_col = cudf::column_view{build_indices_span};
    auto probe_indices_col = cudf::column_view{probe_indices_span};

    auto joined_cols = cudf::gather(left_table, probe_indices_col, oob_policy)->release();
    auto right_cols  = cudf::gather(right_table, build_indices_col, oob_policy)->release();

    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(right_cols.begin()),
                       std::make_move_iterator(right_cols.end()));
    auto joined_table        = std::make_unique<cudf::table>(std::move(joined_cols));
    auto result_sort_order   = cudf::sorted_order(joined_table->view());
    auto sorted_joined_table = cudf::gather(joined_table->view(), *result_sort_order);

    auto expected_sort_order = cudf::sorted_order(expected_table);
    auto sorted_expected     = cudf::gather(expected_table, *expected_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_expected, *sorted_joined_table);
  }
};

enum class join_implementation { FILTERED_JOIN, MARK_JOIN, MARK_JOIN_PREFILTER };

struct SemiAntiJoinTest : public cudf::test::BaseFixture,
                          public ::testing::WithParamInterface<join_implementation> {};

[[nodiscard]] inline bool use_mark_join(join_implementation implementation)
{
  return implementation != join_implementation::FILTERED_JOIN;
}

inline std::string test_name(join_implementation implementation)
{
  switch (implementation) {
    case join_implementation::FILTERED_JOIN: return "FilteredJoin";
    case join_implementation::MARK_JOIN: return "MarkJoin";
    case join_implementation::MARK_JOIN_PREFILTER: return "MarkJoinPrefilter";
  }
  return "Unknown";
}
