/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/mark_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <memory>
#include <stdexcept>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using column_vector  = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

enum class join_implementation { FILTERED_JOIN, MARK_JOIN, MARK_JOIN_PREFILTER };

struct SemiAntiJoinTest : public cudf::test::BaseFixture,
                          public ::testing::WithParamInterface<join_implementation> {};

namespace {
[[nodiscard]] bool use_mark_join(join_implementation implementation)
{
  return implementation != join_implementation::FILTERED_JOIN;
}

[[nodiscard]] cudf::join_prefilter prefilter_mode(join_implementation implementation)
{
  return implementation == join_implementation::MARK_JOIN_PREFILTER ? cudf::join_prefilter::YES
                                                                    : cudf::join_prefilter::NO;
}

// Helper to perform semi/anti join via either filtered_join or mark_join
std::unique_ptr<cudf::table> left_semi_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls  = cudf::null_equality::EQUAL,
  join_implementation implementation = join_implementation::FILTERED_JOIN)
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);

  if (!use_mark_join(implementation)) {
    cudf::filtered_join obj(right_selected, compare_nulls, cudf::get_default_stream());
    auto const join_indices = obj.semi_join(
      left_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  } else {
    cudf::mark_join obj(
      left_selected, compare_nulls, prefilter_mode(implementation), cudf::get_default_stream());
    auto const join_indices = obj.semi_join(
      right_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  }
}

std::unique_ptr<cudf::table> left_anti_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls  = cudf::null_equality::EQUAL,
  join_implementation implementation = join_implementation::FILTERED_JOIN)
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);

  if (!use_mark_join(implementation)) {
    cudf::filtered_join obj(right_selected, compare_nulls, cudf::get_default_stream());
    auto const join_indices = obj.anti_join(
      left_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  } else {
    cudf::mark_join obj(
      left_selected, compare_nulls, prefilter_mode(implementation), cudf::get_default_stream());
    auto const join_indices = obj.anti_join(
      right_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
    auto indices_col  = cudf::column_view{indices_span};
    return cudf::gather(left_input, indices_col);
  }
}
}  // namespace

TEST_P(SemiAntiJoinTest, TestSimple)
{
  auto const implementation = GetParam();

  column_wrapper<int32_t> left_col0{0, 1, 2};
  column_wrapper<int32_t> right_col0{0, 1, 3};

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto result = left_semi_join(left, right, {0}, {0}, cudf::null_equality::EQUAL, implementation);
  auto sort_order = cudf::sorted_order(result->view());
  auto sorted     = cudf::gather(result->view(), *sort_order);

  column_wrapper<int32_t> expected_column{0, 1};
  auto expected = cudf::table_view{{expected_column}};
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *sorted);
}

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> get_saj_tables(
  std::vector<bool> const& left_is_human_nulls, std::vector<bool> const& right_is_human_nulls)
{
  column_wrapper<int32_t> col0_0{{99, 1, 2, 0, 2}, {false, true, true, true, true}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, left_is_human_nulls.begin()};

  auto col0_3 = cudf::test::structs_column_wrapper{
    {col0_names_col, col0_ages_col, col0_is_human_col}, {true, true, true, true, true}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, -99}, {true, true, true, true, false}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_names_col = strcol_wrapper{"Carrot Ironfoundersson",
                                       "Angua von Überwald",
                                       "Detritus",
                                       "Carrot Ironfoundersson",
                                       "Samuel Vimes"};
  auto col1_ages_col  = column_wrapper<int32_t>{{351, 25, 27, 31, 48}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, false, false, false, true}, right_is_human_nulls.begin()};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  column_vector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  return {std::make_unique<Table>(std::move(cols0)), std::make_unique<Table>(std::move(cols1))};
}

TEST_P(SemiAntiJoinTest, SemiJoinWithStructsAndNulls)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, false}, {true, false, false, true, true});

  auto result            = left_semi_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::EQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{99, 2}, {false, true}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, SemiJoinWithStructsAndNullsNotEqual)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, true}, {true, true, false, true, true});

  auto result            = left_semi_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::UNEQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{2}, {true}};
  strcol_wrapper col_gold_1({"s0"}, {true});
  column_wrapper<int32_t> col_gold_2{{1}};
  auto col_gold_3_names_col = strcol_wrapper{"Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{false}, {true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, AntiJoinWithStructsAndNulls)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, false}, {true, false, false, true, true});

  auto result            = left_anti_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::EQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{1, 2, 0}, {true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s0", "s4"}, {true, false, true});
  column_wrapper<int32_t> col_gold_2{{1, 2, 4}};
  auto col_gold_3_names_col = strcol_wrapper{"Carrot Ironfoundersson", "Detritus", "Samuel Vimes"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{27, 351, 31}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false, false}, {true, false, true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, AntiJoinWithStructsAndNullsNotEqual)
{
  auto const implementation = GetParam();
  auto tables = get_saj_tables({true, true, false, true, true}, {true, true, false, true, true});

  auto result            = left_anti_join(*tables.first,
                               *tables.second,
                                          {0, 1, 3},
                                          {0, 1, 3},
                               cudf::null_equality::UNEQUAL,
                               implementation);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{99, 1, 2, 0}, {false, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s1", "s0", "s4"}, {true, true, false, true});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4}};
  auto col_gold_3_names_col =
    strcol_wrapper{"Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes"};
  auto col_gold_3_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31}};

  auto col_gold_3_is_human_col =
    column_wrapper<bool>{{true, true, false, false}, {true, true, false, true}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_vector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(SemiAntiJoinTest, AntiJoinWithStructsAndNullsOnOneSide)
{
  auto const implementation = GetParam();
  auto constexpr null{0};
  auto left_col0 = [] {
    column_wrapper<int32_t> child1{{1, null}, cudf::test::iterators::null_at(1)};
    column_wrapper<int32_t> child2{11, 12};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();
  auto right_col0 = [] {
    column_wrapper<int32_t> child1{1, 2, 3, 4};
    column_wrapper<int32_t> child2{11, 12, 13, 14};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto result = left_anti_join(left, right, {0}, {0}, cudf::null_equality::EQUAL, implementation);
  auto expected_indices_col = column_wrapper<cudf::size_type>{1};
  auto expected             = cudf::gather(left, expected_indices_col);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
}

TEST_P(SemiAntiJoinTest, AntiJoinEmptyTables)
{
  auto const implementation = GetParam();
  cudf::table empty_right_table{};
  cudf::table empty_left_table{};
  column_wrapper<int32_t> col{0, 1, 2};
  auto nonempty_table = cudf::table_view{{col}};
  // Empty left and right tables
  {
    auto result = left_anti_join(
      empty_left_table, empty_right_table, {}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty right table
  {
    auto result = left_anti_join(
      nonempty_table, empty_right_table, {0}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{0, 1, 2};
    auto expected             = cudf::gather(nonempty_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty left table
  {
    auto result = left_anti_join(
      empty_left_table, nonempty_table, {}, {0}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
}

TEST_P(SemiAntiJoinTest, SemiJoinEmptyTables)
{
  auto const implementation = GetParam();
  cudf::table empty_right_table{};
  cudf::table empty_left_table{};
  column_wrapper<int32_t> col{0, 1, 2};
  auto nonempty_table = cudf::table_view{{col}};
  // Empty left and right tables
  {
    auto result = left_semi_join(
      empty_left_table, empty_right_table, {}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty right table
  {
    auto result = left_semi_join(
      nonempty_table, empty_right_table, {0}, {}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(nonempty_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
  // Empty left table
  {
    auto result = left_semi_join(
      empty_left_table, nonempty_table, {}, {0}, cudf::null_equality::EQUAL, implementation);
    auto expected_indices_col = column_wrapper<cudf::size_type>{};
    auto expected             = cudf::gather(empty_left_table, expected_indices_col);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, *result);
  }
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

TEST_F(SemiAntiJoinTest, PrefilterNullableColumnsNullsEqual)
{
  column_wrapper<int32_t> left_col0{{1, 2, 3, 4, 5}, {true, false, true, true, false}};
  column_wrapper<int32_t> right_col0{{2, 3, 5, 6}, {false, true, true, true}};

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto baseline = left_semi_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN);
  auto filtered = left_semi_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN_PREFILTER);

  auto baseline_order  = cudf::sorted_order(baseline->view());
  auto filtered_order  = cudf::sorted_order(filtered->view());
  auto sorted_baseline = cudf::gather(baseline->view(), *baseline_order);
  auto sorted_filtered = cudf::gather(filtered->view(), *filtered_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_baseline, *sorted_filtered);

  auto anti_baseline = left_anti_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN);
  auto anti_filtered = left_anti_join(
    left, right, {0}, {0}, cudf::null_equality::EQUAL, join_implementation::MARK_JOIN_PREFILTER);

  auto anti_baseline_order  = cudf::sorted_order(anti_baseline->view());
  auto anti_filtered_order  = cudf::sorted_order(anti_filtered->view());
  auto anti_sorted_baseline = cudf::gather(anti_baseline->view(), *anti_baseline_order);
  auto anti_sorted_filtered = cudf::gather(anti_filtered->view(), *anti_filtered_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*anti_sorted_baseline, *anti_sorted_filtered);
}

TEST_F(SemiAntiJoinTest, MarkJoinPrefilterLoadFactorOverload)
{
  column_wrapper<int32_t> left_col0{0, 1, 1, 2, 3, 5};
  column_wrapper<int32_t> right_col0{1, 3, 3, 4};

  auto left_selected  = cudf::table_view{{left_col0}};
  auto right_selected = cudf::table_view{{right_col0}};

  cudf::mark_join obj(left_selected,
                      0.5,
                      cudf::null_equality::EQUAL,
                      cudf::join_prefilter::YES,
                      cudf::get_default_stream());

  auto const join_indices = obj.semi_join(
    right_selected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
  auto indices_col  = cudf::column_view{indices_span};
  auto result       = cudf::gather(left_selected, indices_col);

  column_wrapper<int32_t> expected_column{1, 1, 3};
  auto expected = cudf::table_view{{expected_column}};

  auto result_sort_order   = cudf::sorted_order(result->view());
  auto sorted_result       = cudf::gather(result->view(), *result_sort_order);
  auto expected_sort_order = cudf::sorted_order(expected);
  auto sorted_expected     = cudf::gather(expected, *expected_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_expected, *sorted_result);
}

TEST_F(SemiAntiJoinTest, InvalidLoadFactor)
{
  column_wrapper<int32_t> keys{0, 1, 2};
  auto const table = cudf::table_view{{keys}};

  for (auto const load_factor : {-0.1, 0.0, 1.1}) {
    SCOPED_TRACE(load_factor);
    EXPECT_THROW(cudf::filtered_join(
                   table, cudf::null_equality::EQUAL, load_factor, cudf::get_default_stream()),
                 std::invalid_argument);
    EXPECT_THROW(cudf::mark_join(table, load_factor), std::invalid_argument);
  }
}

TEST_F(SemiAntiJoinTest, FilteredJoinMemoryResource)
{
  column_wrapper<int32_t> left_col0{0, 1, 2};
  column_wrapper<int32_t> right_col0{0, 1, 3};

  auto left  = cudf::table_view{{left_col0}};
  auto right = cudf::table_view{{right_col0}};

  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  cudf::filtered_join obj(right, cudf::null_equality::EQUAL, cudf::get_default_stream(), mr);

  EXPECT_GT(mr.get_bytes_counter().peak, 0);

  auto const join_indices =
    obj.semi_join(left, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
  auto indices_col  = cudf::column_view{indices_span};
  auto result       = cudf::gather(left, indices_col);

  column_wrapper<int32_t> expected_column{0, 1};
  auto expected = cudf::table_view{{expected_column}};

  auto result_sort_order   = cudf::sorted_order(result->view());
  auto sorted_result       = cudf::gather(result->view(), *result_sort_order);
  auto expected_sort_order = cudf::sorted_order(expected);
  auto sorted_expected     = cudf::gather(expected, *expected_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_expected, *sorted_result);
}

TEST_F(SemiAntiJoinTest, MarkJoinMemoryResource)
{
  column_wrapper<int32_t> left_col{0, 1, 1, 2, 3, 5};
  auto left = cudf::table_view{{left_col}};

  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  cudf::mark_join obj(
    left, cudf::null_equality::EQUAL, cudf::join_prefilter::YES, cudf::get_default_stream(), mr);

  EXPECT_GT(mr.get_bytes_counter().peak, 0);
}

std::string test_name(join_implementation implementation)
{
  switch (implementation) {
    case join_implementation::FILTERED_JOIN: return "FilteredJoin";
    case join_implementation::MARK_JOIN: return "MarkJoin";
    case join_implementation::MARK_JOIN_PREFILTER: return "MarkJoinPrefilter";
  }
  return "Unknown";
}

INSTANTIATE_TEST_SUITE_P(JoinImpl,
                         SemiAntiJoinTest,
                         ::testing::Values(join_implementation::FILTERED_JOIN,
                                           join_implementation::MARK_JOIN,
                                           join_implementation::MARK_JOIN_PREFILTER),
                         [](auto const& info) { return test_name(info.param); });
