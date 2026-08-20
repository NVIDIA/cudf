/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/reduction.hpp>

struct ListRankScanTest : public cudf::test::BaseFixture {
  inline void test_ungrouped_rank_scan(cudf::column_view const& input,
                                       cudf::column_view const& expect_vals,
                                       cudf::scan_aggregation const& agg,
                                       cudf::null_policy null_handling)
  {
    auto col_out = cudf::scan(input, agg, cudf::scan_type::INCLUSIVE, null_handling);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      expect_vals, col_out->view(), cudf::test::debug_output_level::ALL_ERRORS);
  }
};

TEST_F(ListRankScanTest, BasicList)
{
  using lcw      = cudf::test::lists_column_wrapper<uint64_t>;
  auto const col = lcw{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};

  auto const expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9};
  this->test_ungrouped_rank_scan(
    col,
    expected_dense_vals,
    *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
    cudf::null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, DeepList)
{
  using lcw = cudf::test::lists_column_wrapper<uint64_t>;
  lcw col{
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},
    {{7, 8}, {}},
    lcw{lcw{}, lcw{}, lcw{}},
    lcw{lcw{}},
    lcw{lcw{}},
    lcw{lcw{}},
    lcw{lcw{}, lcw{}, lcw{}},
    lcw{lcw{}, lcw{}, lcw{}},
    {lcw{10}},
    {lcw{10}},
    {{13, 14}, {15}},
    {{13, 14}, {16}},
    lcw{},
    lcw{lcw{}},
  };

  {  // Non-sliced
    auto const expected_dense_vals = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
      1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11};
    this->test_ungrouped_rank_scan(
      col,
      expected_dense_vals,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }

  {  // sliced
    auto sliced_col = cudf::slice(col, {3, 12})[0];
    auto const expected_dense_vals =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 2, 3, 3, 3, 4, 4, 5, 5};
    this->test_ungrouped_rank_scan(
      sliced_col,
      expected_dense_vals,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }
}

TEST_F(ListRankScanTest, ListOfStruct)
{
  // Constructing a list of struct of two elements
  // 0.   []                  ==
  // 1.   []                  !=
  // 2.   Null                ==
  // 3.   Null                !=
  // 4.   [Null, Null]        !=
  // 5.   [Null]              ==
  // 6.   [Null]              ==
  // 7.   [Null]              !=
  // 8.   [{Null, Null}]      !=
  // 9.   [{1,'a'}, {2,'b'}]  !=
  // 10.  [{0,'a'}, {2,'b'}]  !=
  // 11.  [{0,'a'}, {2,'c'}]  ==
  // 12.  [{0,'a'}, {2,'c'}]  !=
  // 13.  [{0,Null}]          ==
  // 14.  [{0,Null}]          !=
  // 15.  [{Null, 0}]         ==
  // 16.  [{Null, 0}]

  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>{
    {-1, -1, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 1, 2},
    {true,
     true,
     true,
     true,
     true,
     false,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     false,
     false}};
  auto col2 = cudf::test::strings_column_wrapper{
    {"x", "x", "a", "a", "b", "", "a", "b", "a", "b", "a", "c", "a", "c", "", "", "b", "b"},
    {true,
     true,
     true,
     true,
     true,
     false,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     false,
     false,
     true,
     true}};
  auto struct_col = cudf::test::structs_column_wrapper{{col1, col2},
                                                       {false,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true,
                                                        true}};

  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{
    0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};

  auto list_nullmask = std::vector<bool>{true,
                                         true,
                                         false,
                                         false,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true,
                                         true};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                       17,
                                       nullptr,
                                       static_cast<cudf::bitmask_type*>(null_mask.data()),
                                       null_count,
                                       0,
                                       {offsets, struct_col});

  {  // Non-sliced
    auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
      1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10};

    this->test_ungrouped_rank_scan(
      list_column,
      expect,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }

  {  // Sliced
    auto sliced_col = cudf::slice(list_column, {3, 15})[0];
    auto expect =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8};

    this->test_ungrouped_rank_scan(
      sliced_col,
      expect,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }
}

TEST_F(ListRankScanTest, ListOfEmptyStruct)
{
  // []
  // []
  // Null
  // Null
  // [Null, Null]
  // [Null, Null]
  // [Null, Null]
  // [Null]
  // [Null]
  // [{}]
  // [{}]
  // [{}, {}]
  // [{}, {}]

  auto struct_validity = std::vector<bool>{
    false, false, false, false, false, false, false, false, true, true, true, true, true, true};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
  auto struct_col = cudf::make_structs_column(14, {}, null_count, std::move(null_mask));

  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 2, 4, 6, 7, 8, 9, 10, 12, 14};
  auto list_nullmask = std::vector<bool>{
    true, true, false, false, true, true, true, true, true, true, true, true, true};
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    13, offsets.release(), std::move(struct_col), null_count, std::move(null_mask));

  auto expect =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6};

  this->test_ungrouped_rank_scan(
    *list_column,
    expect,
    *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
    cudf::null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, EmptyDeepList)
{
  // List<List<int>>, where all lists are empty
  // []
  // []
  // Null
  // Null

  // Internal empty list
  auto list1 = cudf::test::lists_column_wrapper<int>{};

  auto offsets       = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0};
  auto list_nullmask = std::vector<bool>{true, true, false, false};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    4, offsets.release(), list1.release(), null_count, std::move(null_mask));

  auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 2};

  this->test_ungrouped_rank_scan(
    *list_column,
    expect,
    *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
    cudf::null_policy::INCLUDE);
}
