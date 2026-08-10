/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/direct_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

class JoinTest : public cudf::test::BaseFixture {
  static inline cudf::table make_table()
  {
    cudf::test::fixed_width_column_wrapper<int32_t> col0{{3, 1, 2, 0, 3}};
    cudf::test::strings_column_wrapper col1{{"s0", "s1", "s2", "s4", "s1"}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2{{0, 1, 2, 4, 1}};

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(col0.release());
    columns.push_back(col1.release());
    columns.push_back(col2.release());

    return cudf::table{std::move(columns)};
  }

 public:
  cudf::table table0{make_table()};
  cudf::table table1{make_table()};
  cudf::table conditional0{make_table()};
  cudf::table conditional1{make_table()};
  cudf::ast::column_reference col_ref_left_0{0};
  cudf::ast::column_reference col_ref_right_0{0, cudf::ast::table_reference::RIGHT};
  cudf::ast::operation left_zero_eq_right_zero{
    cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0};
};

TEST_F(JoinTest, InnerJoin)
{
  cudf::inner_join(table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, StreamingHashJoin)
{
  auto const stream = cudf::test::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> right_keys{{1, 2, 3}};
  cudf::test::fixed_width_column_wrapper<int32_t> left_keys{{2, 4}};
  cudf::table_view const right{{right_keys}};
  cudf::table_view const left{{left_keys}};
  std::vector<cudf::data_type> const schema{right.column(0).type()};
  std::vector<cudf::size_type> const key_indices{0};

  cudf::streaming_hash_join joiner{schema,
                                   key_indices,
                                   right.num_rows(),
                                   /*max_num_batches=*/1,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL,
                                   /*load_factor=*/0.5,
                                   stream};
  joiner.insert(right, stream);
  [[maybe_unused]] auto result = joiner.inner_join(left, std::nullopt, stream);
}

TEST_F(JoinTest, SortMergeInnerJoin)
{
  cudf::sort_merge_join obj(
    table1, cudf::sorted::NO, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
  obj.inner_join(table0, cudf::sorted::NO, cudf::test::get_default_stream());
}

TEST_F(JoinTest, LeftJoin)
{
  cudf::left_join(table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, FullJoin)
{
  cudf::full_join(table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, LeftSemiJoin)
{
  cudf::filtered_join obj(table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
  [[maybe_unused]] auto join_result = obj.semi_join(table0, cudf::test::get_default_stream());
}

TEST_F(JoinTest, LeftAntiJoin)
{
  cudf::filtered_join obj(table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
  [[maybe_unused]] auto join_result = obj.anti_join(table0, cudf::test::get_default_stream());
}

TEST_F(JoinTest, CrossJoin) { cudf::cross_join(table0, table1, cudf::test::get_default_stream()); }

TEST_F(JoinTest, DirectInnerJoin)
{
  cudf::test::fixed_width_column_wrapper<uint32_t> left_keys{{0, 1, 2, 5}};
  cudf::test::fixed_width_column_wrapper<uint32_t> right_keys{{0, 1, 2, 3}};
  std::ignore = cudf::direct_inner_join(left_keys, right_keys, 6, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalInnerJoin)
{
  cudf::conditional_inner_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftJoin)
{
  cudf::conditional_left_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalFullJoin)
{
  cudf::conditional_full_join(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftSemiJoin)
{
  cudf::conditional_left_semi_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftAntiJoin)
{
  cudf::conditional_left_anti_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedInnerJoin)
{
  cudf::mixed_inner_join(table0,
                         table1,
                         conditional0,
                         conditional1,
                         left_zero_eq_right_zero,
                         cudf::null_equality::EQUAL,
                         std::nullopt,
                         cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftJoin)
{
  cudf::mixed_left_join(table0,
                        table1,
                        conditional0,
                        conditional1,
                        left_zero_eq_right_zero,
                        cudf::null_equality::EQUAL,
                        std::nullopt,
                        cudf::test::get_default_stream());
}

TEST_F(JoinTest, LeftJoinWithPostFilter)
{
  cudf::hash_join hash_joiner(table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
  auto hash_join_result =
    hash_joiner.left_join(table0, std::nullopt, cudf::test::get_default_stream());

  auto hash_filter_result =
    cudf::filter_join_indices(conditional0,
                              conditional1,
                              cudf::device_span<cudf::size_type const>(*hash_join_result.first),
                              cudf::device_span<cudf::size_type const>(*hash_join_result.second),
                              left_zero_eq_right_zero,
                              cudf::join_kind::LEFT_JOIN,
                              std::nullopt,
                              cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedFullJoin)
{
  cudf::mixed_full_join(table0,
                        table1,
                        conditional0,
                        conditional1,
                        left_zero_eq_right_zero,
                        cudf::null_equality::EQUAL,
                        std::nullopt,
                        cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftSemiJoin)
{
  cudf::mixed_left_semi_join(table0,
                             table1,
                             conditional0,
                             conditional1,
                             left_zero_eq_right_zero,
                             cudf::null_equality::EQUAL,
                             cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftAntiJoin)
{
  cudf::mixed_left_anti_join(table0,
                             table1,
                             conditional0,
                             conditional1,
                             left_zero_eq_right_zero,
                             cudf::null_equality::EQUAL,
                             cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedInnerJoinSize)
{
  cudf::mixed_inner_join_size(table0,
                              table1,
                              conditional0,
                              conditional1,
                              left_zero_eq_right_zero,
                              cudf::null_equality::EQUAL,
                              cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftJoinSize)
{
  cudf::mixed_left_join_size(table0,
                             table1,
                             conditional0,
                             conditional1,
                             left_zero_eq_right_zero,
                             cudf::null_equality::EQUAL,
                             cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalInnerJoinSize)
{
  cudf::conditional_inner_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftJoinSize)
{
  cudf::conditional_left_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftSemiJoinSize)
{
  cudf::conditional_left_semi_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftAntiJoinSize)
{
  cudf::conditional_left_anti_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
