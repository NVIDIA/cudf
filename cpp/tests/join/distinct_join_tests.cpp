/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

std::unique_ptr<rmm::device_uvector<cudf::size_type>> get_left_indices(cudf::size_type size)
{
  auto sequence = std::vector<cudf::size_type>(size);
  std::iota(sequence.begin(), sequence.end(), 0);
  auto indices = cudf::detail::make_device_uvector(
    sequence, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  return std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(indices));
}

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

TEST_F(DistinctJoinTest, IntegerInnerJoin)
{
  auto constexpr size = 2024;

  auto const init = cudf::numeric_scalar<int32_t>{0};

  auto right = cudf::sequence(size, init, cudf::numeric_scalar<int32_t>{1});
  auto left  = cudf::sequence(size, init, cudf::numeric_scalar<int32_t>{2});

  auto right_table = cudf::table_view{{right->view()}};
  auto left_table  = cudf::table_view{{left->view()}};

  auto distinct_join = cudf::distinct_hash_join{right_table};

  auto result = distinct_join.inner_join(left_table);

  auto constexpr gold_size = size / 2;
  auto gold                = cudf::sequence(gold_size, init, cudf::numeric_scalar<int32_t>{2});
  this->compare_to_reference(right_table, left_table, result, cudf::table_view{{gold->view()}});
}

TEST_F(DistinctJoinTest, MemoryResource)
{
  column_wrapper<int32_t> col0_0{{1, 2, 3, 4, 5}};
  strcol_wrapper col0_1({"s0", "s0", "s3", "s4", "s5"});
  column_wrapper<int32_t> col0_2{{9, 9, 9, 9, 9}};

  column_wrapper<int32_t> col1_0{{1, 2, 3, 4, 9}};
  strcol_wrapper col1_1({"s0", "s0", "s0", "s4", "s4"});
  column_wrapper<int32_t> col1_2{{9, 9, 9, 0, 9}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  auto distinct_join = cudf::distinct_hash_join{
    right.view(), cudf::null_equality::EQUAL, 0.5, cudf::get_default_stream(), mr};

  EXPECT_GT(mr.get_bytes_counter().peak, 0);

  auto result = distinct_join.inner_join(left.view());

  column_wrapper<int32_t> col_gold_0{{1, 2}};
  strcol_wrapper col_gold_1({"s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{9, 9}};
  column_wrapper<int32_t> col_gold_3{{1, 2}};
  strcol_wrapper col_gold_4({"s0", "s0"});
  column_wrapper<int32_t> col_gold_5{{9, 9}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(right.view(), left.view(), result, gold.view());
}

TEST_F(DistinctJoinTest, InnerJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{1, 2, 3, 4, 5}};
  strcol_wrapper col0_1({"s0", "s0", "s3", "s4", "s5"});
  column_wrapper<int32_t> col0_2{{9, 9, 9, 9, 9}};

  column_wrapper<int32_t> col1_0{{1, 2, 3, 4, 9}};
  strcol_wrapper col1_1({"s0", "s0", "s0", "s4", "s4"});
  column_wrapper<int32_t> col1_2{{9, 9, 9, 0, 9}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.inner_join(left.view());

  column_wrapper<int32_t> col_gold_0{{1, 2}};
  strcol_wrapper col_gold_1({"s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{9, 9}};
  column_wrapper<int32_t> col_gold_3{{1, 2}};
  strcol_wrapper col_gold_4({"s0", "s0"});
  column_wrapper<int32_t> col_gold_5{{9, 9}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(right.view(), left.view(), result, gold.view());
}

TEST_F(DistinctJoinTest, PrimitiveInnerJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{1, 2, 3, 4, 5}};
  column_wrapper<int32_t> col0_1({0, 0, 3, 4, 5});
  column_wrapper<int32_t> col0_2({9, 9, 9, 9, 9});

  column_wrapper<int32_t> col1_0{{1, 2, 3, 4, 9}};
  column_wrapper<int32_t> col1_1({0, 0, 0, 4, 4});
  column_wrapper<int32_t> col1_2({9, 9, 9, 0, 9});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.inner_join(left.view());

  column_wrapper<int32_t> col_gold_0{{1, 2}};
  column_wrapper<int32_t> col_gold_1({0, 0});
  column_wrapper<int32_t> col_gold_2{{9, 9}};
  column_wrapper<int32_t> col_gold_3{{1, 2}};
  column_wrapper<int32_t> col_gold_4({0, 0});
  column_wrapper<int32_t> col_gold_5{{9, 9}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(right.view(), left.view(), result, gold.view());
}

TEST_F(DistinctJoinTest, InnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{1, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{1, 2, 0, 2, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s0", "s1"});
  column_wrapper<int32_t> col1_2{{1, 1, 1, 1, 1}, {false, true, true, false, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  // Create gold table once
  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{1, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2}};
  strcol_wrapper col_gold_4({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_5{{1, 1}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  // Test with different load factors
  std::vector<double> load_factors = {0.5, 1.0};

  for (auto load_factor : load_factors) {
    auto distinct_join =
      cudf::distinct_hash_join{right.view(), cudf::null_equality::EQUAL, load_factor};
    auto result = distinct_join.inner_join(left.view());

    this->compare_to_reference(right.view(), left.view(), result, gold.view());
  }
}

TEST_F(DistinctJoinTest, PrimitiveInnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  column_wrapper<int32_t> col0_1({1, 1, 0, 4, 0}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{1, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{1, 2, 0, 2, 3}};
  column_wrapper<int32_t> col1_1({1, 0, 1, 0, 1});
  column_wrapper<int32_t> col1_2{{1, 1, 1, 1, 1}, {false, true, true, false, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  // Create gold table once
  column_wrapper<int32_t> col_gold_0{{3, 2}};
  column_wrapper<int32_t> col_gold_1({1, 0}, {true, true});
  column_wrapper<int32_t> col_gold_2{{1, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2}};
  column_wrapper<int32_t> col_gold_4({1, 0}, {true, true});
  column_wrapper<int32_t> col_gold_5{{1, 1}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  // Test with different load factors
  std::vector<double> load_factors = {0.5, 1.0};

  for (auto load_factor : load_factors) {
    auto distinct_join =
      cudf::distinct_hash_join{right.view(), cudf::null_equality::EQUAL, load_factor};
    auto result = distinct_join.inner_join(left.view());

    this->compare_to_reference(right.view(), left.view(), result, gold.view());
  }
}

TEST_F(DistinctJoinTest, InnerJoinWithStructsAndNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 4}, {true, true, true, true, false}};
  std::initializer_list<std::string> col0_names = {
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_names_col = strcol_wrapper{col0_names.begin(), col0_names.end()};
  auto col0_ages_col  = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 1, 1, 2, 0}, {true, false, true, true, true}};
  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Detritus",
                                                   "Carrot Ironfoundersson",
                                                   "Samuel Vimes"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{31, 25, 351, 27, 48}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, false, false, false, true}, {true, false, false, true, true}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table left(std::move(cols0));
  Table right(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.inner_join(left.view());

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{0, 4}, {true, false}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_wrapper<int32_t> col_gold_4{{3, 2}};
  strcol_wrapper col_gold_5({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_6{{0, -1}, {true, false}};
  auto col_gold_7_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_7_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_7_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {col_gold_7_names_col, col_gold_7_ages_col, col_gold_7_is_human_col}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  cols_gold.push_back(col_gold_6.release());
  cols_gold.push_back(col_gold_7.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(right.view(), left.view(), result, gold.view());
}

TEST_F(DistinctJoinTest, EmptyRightTableInnerJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.inner_join(left.view());

  this->compare_to_reference(right.view(), left.view(), result, right.view());
}

TEST_F(DistinctJoinTest, EmptyRightTableLeftJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.left_join(left.view());
  auto gather_map    = std::pair{get_left_indices(result->size()), std::move(result)};

  this->compare_to_reference(
    right.view(), left.view(), gather_map, left.view(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST_F(DistinctJoinTest, EmptyLeftTableInnerJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.inner_join(left.view());

  this->compare_to_reference(right.view(), left.view(), result, left.view());
}

TEST_F(DistinctJoinTest, EmptyLeftTableLeftJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table right(std::move(cols0));
  Table left(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.left_join(left.view());
  auto gather_map    = std::pair{get_left_indices(result->size()), std::move(result)};

  this->compare_to_reference(
    right.view(), left.view(), gather_map, left.view(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST_F(DistinctJoinTest, LeftJoinNoNulls)
{
  column_wrapper<int32_t> col0_0({3, 1, 2, 0, 3});
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});

  column_wrapper<int32_t> col1_0({2, 2, 0, 4, 3});
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table left(std::move(cols0));
  Table right(std::move(cols1));

  column_wrapper<int32_t> col_gold_0({3, 1, 2, 0, 3});
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col_gold_2{{-1, -1, -1, -1, 3}, {false, false, false, false, true}};
  strcol_wrapper col_gold_3{{"", "", "", "", "s1"}, {false, false, false, false, true}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.left_join(left.view());
  auto gather_map    = std::pair{get_left_indices(result->size()), std::move(result)};

  this->compare_to_reference(
    right.view(), left.view(), gather_map, gold.view(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST_F(DistinctJoinTest, PrimitiveLeftJoinNoNulls)
{
  column_wrapper<int32_t> col0_0({3, 1, 2, 0, 3});
  column_wrapper<int32_t> col0_1({0, 1, 2, 4, 1});

  column_wrapper<int32_t> col1_0({2, 2, 0, 4, 3});
  column_wrapper<int32_t> col1_1({1, 0, 1, 2, 1});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table left(std::move(cols0));
  Table right(std::move(cols1));

  column_wrapper<int32_t> col_gold_0({3, 1, 2, 0, 3});
  column_wrapper<int32_t> col_gold_1({0, 1, 2, 4, 1});
  column_wrapper<int32_t> col_gold_2{{-1, -1, -1, -1, 3}, {false, false, false, false, true}};
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 1}, {false, false, false, false, true}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.left_join(left.view());
  auto gather_map    = std::pair{get_left_indices(result->size()), std::move(result)};

  this->compare_to_reference(
    right.view(), left.view(), gather_map, gold.view(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST_F(DistinctJoinTest, LeftJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table left(std::move(cols0));
  Table right(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.left_join(left.view());
  auto gather_map    = std::pair{get_left_indices(result->size()), std::move(result)};

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 2}, {true, true, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col_gold_2{{3, -1, -1, -1, 2}, {true, false, false, false, true}};
  strcol_wrapper col_gold_3{{"s1", "", "", "", "s0"}, {true, false, false, false, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(
    right.view(), left.view(), gather_map, gold.view(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST_F(DistinctJoinTest, PrimitiveLeftJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  column_wrapper<int32_t> col0_1({1, 1, -1, 4, 0}, {true, true, false, true, true});

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1({1, 0, 1, 2, 1});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table left(std::move(cols0));
  Table right(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.left_join(left.view());
  auto gather_map    = std::pair{get_left_indices(result->size()), std::move(result)};

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 2}, {true, true, true, true, true}};
  column_wrapper<int32_t> col_gold_1({1, 1, -1, 4, 0}, {true, true, false, true, true});
  column_wrapper<int32_t> col_gold_2{{3, -1, -1, -1, 2}, {true, false, false, false, true}};
  column_wrapper<int32_t> col_gold_3({1, -1, -1, -1, 0}, {true, false, false, false, true});

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(
    right.view(), left.view(), gather_map, gold.view(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST_F(DistinctJoinTest, LeftJoinWithStructsAndNulls)
{
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};
  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};
  auto col0 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  auto col1_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Detritus", "Carrot Ironfoundersson", "Angua von Überwald"};
  auto col1_ages_col = column_wrapper<int32_t>{{48, 35, 351, 22, 25}};
  auto col1_is_human_col =
    column_wrapper<bool>{{true, true, false, false, true}, {true, true, false, true, true}};
  auto col1 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0.release());
  cols1.push_back(col1.release());

  Table left(std::move(cols0));
  Table right(std::move(cols1));

  auto distinct_join = cudf::distinct_hash_join{right.view()};
  auto result        = distinct_join.left_join(left.view());
  auto gather_map    = std::pair{get_left_indices(result->size()), std::move(result)};

  auto col0_gold_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Carrot Ironfoundersson", "Samuel Vimes", "Angua von Überwald"};
  auto col0_gold_ages_col = column_wrapper<int32_t>{{48, 351, 27, 31, 25}};
  auto col0_gold_is_human_col =
    column_wrapper<bool>{{true, false, true, false, false}, {true, false, true, true, false}};
  auto col0_gold = cudf::test::structs_column_wrapper{
    {col0_gold_names_col, col0_gold_ages_col, col0_gold_is_human_col}};

  auto col1_gold_names_col = strcol_wrapper{{
                                              "Samuel Vimes",
                                              "Detritus",
                                              "",
                                              "",
                                              "",
                                            },
                                            {true, true, false, false, false}};
  auto col1_gold_ages_col =
    column_wrapper<int32_t>{{48, 351, -1, -1, -1}, {true, true, false, false, false}};
  auto col1_gold_is_human_col =
    column_wrapper<bool>{{true, false, false, false, false}, {true, false, false, false, false}};
  auto col1_gold = cudf::test::structs_column_wrapper{
    {col1_gold_names_col, col1_gold_ages_col, col1_gold_is_human_col},
    {true, true, false, false, false}};

  CVector cols_gold;
  cols_gold.push_back(col0_gold.release());
  cols_gold.push_back(col1_gold.release());
  Table gold(std::move(cols_gold));

  this->compare_to_reference(
    right.view(), left.view(), gather_map, gold.view(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST_F(DistinctJoinTest, InvalidLoadFactor)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});

  CVector cols0;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());

  Table t0(std::move(cols0));

  // Test load factor of -0.1
  EXPECT_THROW(cudf::distinct_hash_join(t0, cudf::null_equality::EQUAL, -0.1),
               std::invalid_argument);
  // Test load factor of 0
  EXPECT_THROW(cudf::distinct_hash_join(t0, cudf::null_equality::EQUAL, 0.0),
               std::invalid_argument);

  // Test load factor > 1
  EXPECT_THROW(cudf::distinct_hash_join(t0, cudf::null_equality::EQUAL, 1.1),
               std::invalid_argument);
}

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
