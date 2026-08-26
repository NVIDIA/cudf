/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/lists/combine.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>

class ListTest : public cudf::test::BaseFixture {};

TEST_F(ListTest, ConcatenateRows)
{
  cudf::test::lists_column_wrapper<int> list_col_1{{0, 1}, {2, 3}, {4, 5}};
  cudf::test::lists_column_wrapper<int> list_col_2{{0, 1}, {2, 3}, {4, 5}};
  cudf::table_view lists_table({list_col_1, list_col_2});
  cudf::lists::concatenate_rows(
    lists_table, cudf::lists::concatenate_null_policy::IGNORE, cudf::test::get_default_stream());
}

TEST_F(ListTest, ConcatenateListElements)
{
  cudf::test::lists_column_wrapper<int> ll_column{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  cudf::lists::concatenate_list_elements(
    ll_column, cudf::lists::concatenate_null_policy::IGNORE, cudf::test::get_default_stream());
}

TEST_F(ListTest, ContainsNulls)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::lists::contains_nulls(list_col, cudf::test::get_default_stream());
}

TEST_F(ListTest, ContainsSearchKey)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::numeric_scalar<int32_t> search_key(2, true, cudf::test::get_default_stream());
  cudf::lists::contains(list_col, search_key, cudf::test::get_default_stream());
}

TEST_F(ListTest, ContainsSearchKeys)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::test::fixed_width_column_wrapper<int> search_keys({1, 2, 3});
  cudf::lists::contains(list_col, search_keys, cudf::test::get_default_stream());
}

TEST_F(ListTest, IndexOfSearchKey)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::numeric_scalar<int32_t> search_key(2, true, cudf::test::get_default_stream());
  cudf::lists::index_of(list_col,
                        search_key,
                        cudf::lists::duplicate_find_option::FIND_FIRST,
                        cudf::test::get_default_stream());
}

TEST_F(ListTest, IndexOfSearchKeys)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::test::fixed_width_column_wrapper<int> search_keys({1, 2, 3});
  cudf::lists::index_of(list_col,
                        search_keys,
                        cudf::lists::duplicate_find_option::FIND_FIRST,
                        cudf::test::get_default_stream());
}

TEST_F(ListTest, CountElements)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7}, {4, 5}};
  cudf::lists::count_elements(list_col, cudf::test::get_default_stream());
}

TEST_F(ListTest, ExtractListElementFromIndex)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7}, {4, 5}};
  cudf::lists::extract_list_element(list_col, -1, cudf::test::get_default_stream());
}

TEST_F(ListTest, ExtractListElementFromIndices)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7}, {4, 5}};
  cudf::test::fixed_width_column_wrapper<int> indices({-1, -2, -1});
  cudf::lists::extract_list_element(list_col, indices, cudf::test::get_default_stream());
}

TEST_F(ListTest, SegmentedGather)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<int> gather_map_list{{0}, {1, 2}, {1}};
  cudf::lists::segmented_gather(list_col,
                                gather_map_list,
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                cudf::test::get_default_stream());
}

TEST_F(ListTest, Sequences)
{
  cudf::test::fixed_width_column_wrapper<int> starts({0, 1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int> sizes({0, 1, 2, 2, 1});
  cudf::lists::sequences(starts, sizes, cudf::test::get_default_stream());
}

TEST_F(ListTest, SequencesWithSteps)
{
  cudf::test::fixed_width_column_wrapper<int> starts({0, 1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int> steps({2, 1, 1, 1, -3});
  cudf::test::fixed_width_column_wrapper<int> sizes({0, 1, 2, 2, 1});
  cudf::lists::sequences(starts, steps, sizes, cudf::test::get_default_stream());
}

TEST_F(ListTest, Reverse)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::lists::reverse(list_col, cudf::test::get_default_stream());
}

TEST_F(ListTest, SortLists)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::lists::sort_lists(
    list_col, cudf::order::DESCENDING, cudf::null_order::AFTER, cudf::test::get_default_stream());
}

TEST_F(ListTest, StableSortLists)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::lists::stable_sort_lists(
    list_col, cudf::order::DESCENDING, cudf::null_order::AFTER, cudf::test::get_default_stream());
}

TEST_F(ListTest, ApplyBooleanMask)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<bool> boolean_mask{
    {false, true}, {true, true, true, false}, {false, true}};
  cudf::lists::apply_boolean_mask(list_col, boolean_mask, cudf::test::get_default_stream());
}

TEST_F(ListTest, ApplyDeletionMask)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<bool> deletion_mask{
    {false, true}, {true, true, true, false}, {false, true}};
  cudf::lists::apply_deletion_mask(list_col, deletion_mask, cudf::test::get_default_stream());
}

TEST_F(ListTest, Distinct)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<int> boolean_mask{{0, 1}, {1, 1, 1, 0}, {0, 1}};
  cudf::lists::distinct(list_col,
                        cudf::null_equality::EQUAL,
                        cudf::nan_equality::ALL_EQUAL,
                        cudf::duplicate_keep_option::KEEP_ANY,
                        cudf::test::get_default_stream());
}

TEST_F(ListTest, DifferenceDistinct)
{
  cudf::test::lists_column_wrapper<int> list_col_a{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<int> list_col_b{{0, 1}, {1, 3, 6, 8}, {5}};
  cudf::lists::difference_distinct(list_col_a,
                                   list_col_b,
                                   cudf::null_equality::EQUAL,
                                   cudf::nan_equality::ALL_EQUAL,
                                   cudf::test::get_default_stream());
}

TEST_F(ListTest, IntersectDistinct)
{
  cudf::test::lists_column_wrapper<int> list_col_a{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<int> list_col_b{{0, 1}, {1, 3, 6, 8}, {5}};
  cudf::lists::intersect_distinct(list_col_a,
                                  list_col_b,
                                  cudf::null_equality::EQUAL,
                                  cudf::nan_equality::ALL_EQUAL,
                                  cudf::test::get_default_stream());
}

TEST_F(ListTest, UnionDistinct)
{
  cudf::test::lists_column_wrapper<int> list_col_a{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<int> list_col_b{{0, 1}, {1, 3, 6, 8}, {5}};
  cudf::lists::union_distinct(list_col_a,
                              list_col_b,
                              cudf::null_equality::EQUAL,
                              cudf::nan_equality::ALL_EQUAL,
                              cudf::test::get_default_stream());
}

TEST_F(ListTest, HaveOverlap)
{
  cudf::test::lists_column_wrapper<int> list_col_a{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  cudf::test::lists_column_wrapper<int> list_col_b{{0, 1}, {1, 3, 6, 8}, {5}};
  cudf::lists::have_overlap(list_col_a,
                            list_col_b,
                            cudf::null_equality::EQUAL,
                            cudf::nan_equality::ALL_EQUAL,
                            cudf::test::get_default_stream());
}

TEST_F(ListTest, Explode)
{
  cudf::test::fixed_width_column_wrapper<int32_t> list_col_a{100, 200, 300};
  cudf::test::lists_column_wrapper<int32_t> list_col_b{
    cudf::test::lists_column_wrapper<int32_t>{1, 2, 7},
    cudf::test::lists_column_wrapper<int32_t>{5, 6},
    cudf::test::lists_column_wrapper<int32_t>{0, 3}};
  cudf::test::strings_column_wrapper list_col_c{"string0", "string1", "string2"};
  cudf::table_view lists_table({list_col_a, list_col_b, list_col_c});
  cudf::explode(lists_table, 1, cudf::test::get_default_stream());
}

TEST_F(ListTest, ExplodePosition)
{
  cudf::test::fixed_width_column_wrapper<int32_t> list_col_a{100, 200, 300};
  cudf::test::lists_column_wrapper<int32_t> list_col_b{
    cudf::test::lists_column_wrapper<int32_t>{1, 2, 7},
    cudf::test::lists_column_wrapper<int32_t>{5, 6},
    cudf::test::lists_column_wrapper<int32_t>{0, 3}};
  cudf::test::strings_column_wrapper list_col_c{"string0", "string1", "string2"};
  cudf::table_view lists_table({list_col_a, list_col_b, list_col_c});
  cudf::explode_position(lists_table, 1, cudf::test::get_default_stream());
}

TEST_F(ListTest, ExplodeOuter)
{
  constexpr auto null = 0;
  auto valids         = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::lists_column_wrapper<int32_t> list_col_a{
    cudf::test::lists_column_wrapper<int32_t>({1, null, 7}, valids),
    cudf::test::lists_column_wrapper<int32_t>({5, null, 0, null}, valids),
    cudf::test::lists_column_wrapper<int32_t>{},
    cudf::test::lists_column_wrapper<int32_t>({0, null, 8}, valids)};
  cudf::test::fixed_width_column_wrapper<int32_t> list_col_b{100, 200, 300, 400};
  cudf::table_view lists_table({list_col_a, list_col_b});
  cudf::explode_outer(lists_table, 0, cudf::test::get_default_stream());
}

TEST_F(ListTest, ExplodeOuterPosition)
{
  constexpr auto null = 0;
  auto valids         = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::lists_column_wrapper<int32_t> list_col_a{
    cudf::test::lists_column_wrapper<int32_t>({1, null, 7}, valids),
    cudf::test::lists_column_wrapper<int32_t>({5, null, 0, null}, valids),
    cudf::test::lists_column_wrapper<int32_t>{},
    cudf::test::lists_column_wrapper<int32_t>({0, null, 8}, valids)};
  cudf::test::fixed_width_column_wrapper<int32_t> list_col_b{100, 200, 300, 400};
  cudf::table_view lists_table({list_col_a, list_col_b});
  cudf::explode_outer_position(lists_table, 0, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
