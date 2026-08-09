/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>

class SortingTest : public cudf::test::BaseFixture {};

TEST_F(SortingTest, SortedOrder)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};
  cudf::table_view const tbl{{column}};

  cudf::sorted_order(tbl, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, StableSortedOrder)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};
  cudf::table_view const tbl{{column}};

  cudf::stable_sorted_order(tbl, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, IsSorted)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};
  cudf::table_view const tbl{{column}};

  cudf::is_sorted(tbl, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, Sort)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};
  cudf::table_view const tbl{{column}};

  cudf::sort(tbl, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, SortByKey)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const values_col{10, 20, 30, 40, 50};
  cudf::table_view const values{{values_col}};
  cudf::test::fixed_width_column_wrapper<int32_t> const keys_col{10, 20, 30, 40, 50};
  cudf::table_view const keys{{keys_col}};

  cudf::sort_by_key(values, keys, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, StableSortByKey)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const values_col{10, 20, 30, 40, 50};
  cudf::table_view const values{{values_col}};
  cudf::test::fixed_width_column_wrapper<int32_t> const keys_col{10, 20, 30, 40, 50};
  cudf::table_view const keys{{keys_col}};

  cudf::stable_sort_by_key(values, keys, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, Rank)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const column{10, 20, 30, 40, 50};

  cudf::rank(column,
             cudf::rank_method::AVERAGE,
             cudf::order::ASCENDING,
             cudf::null_policy::EXCLUDE,
             cudf::null_order::AFTER,
             false,
             cudf::test::get_default_stream());
}

TEST_F(SortingTest, SegmentedSortedOrder)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const keys_col{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  cudf::table_view const keys{{keys_col}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> const segment_offsets{3, 7};

  cudf::segmented_sorted_order(keys, segment_offsets, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, StableSegmentedSortedOrder)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const keys_col{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  cudf::table_view const keys{{keys_col}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> const segment_offsets{3, 7};

  cudf::stable_segmented_sorted_order(
    keys, segment_offsets, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, SegmentedSortByKey)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const keys_col{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  cudf::table_view const keys{{keys_col}};
  cudf::test::fixed_width_column_wrapper<int32_t> const values_col{7, 6, 9, 3, 4, 5, 1, 2, 0, 4};
  cudf::table_view const values{{values_col}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> const segment_offsets{0, 3, 7, 10};

  cudf::segmented_sort_by_key(
    values, keys, segment_offsets, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, StableSegmentedSortByKey)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const keys_col{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  cudf::table_view const keys{{keys_col}};
  cudf::test::fixed_width_column_wrapper<int32_t> const values_col{7, 6, 9, 3, 4, 5, 1, 2, 0, 4};
  cudf::table_view const values{{values_col}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> const segment_offsets{0, 3, 7, 10};

  cudf::stable_segmented_sort_by_key(
    values, keys, segment_offsets, {}, {}, cudf::test::get_default_stream());
}

TEST_F(SortingTest, TopK)
{
  auto stream = cudf::test::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> const input{10, 20, 30, 40, 50};
  cudf::top_k(input, 2, cudf::order::ASCENDING, stream);
  cudf::top_k_order(input, 2, cudf::order::ASCENDING, stream);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> const offsets{0, 5};
  cudf::segmented_top_k(input, offsets, 2, cudf::order::ASCENDING, stream);
  cudf::segmented_top_k_order(input, offsets, 2, cudf::order::ASCENDING, stream);
}

CUDF_TEST_PROGRAM_MAIN()
