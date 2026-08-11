/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compression_common.hpp"
#include "io/utilities/time_utils.hpp"
#include "io_test_utils.hpp"
#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/reshape.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <cuda/iterator>

#include <src/io/parquet/parquet_gpu.hpp>
#include <src/io/parquet/stats_filter_helpers.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

using ParquetDecompressionTest = DecompressionTest<ParquetReaderTest>;

TEST_F(ParquetReaderTest, UserBounds)
{
  // trying to read more rows than there are should result in
  // receiving the properly capped # of rows
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("TooManyRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(16);
    auto result = cudf::io::read_parquet(read_opts);

    // we should only get back 4 rows
    EXPECT_EQ(result.tbl->view().column(0).size(), 4);
  }

  // trying to read past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("PastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).skip_rows(4);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows should result in empty columns
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRowsPastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .skip_rows(4)
        .num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }
}

TEST_F(ParquetReaderTest, ZeroColumnsPreservesRowCount)
{
  GTEST_SKIP() << "Zero-column / N-row parquet reads are not yet supported. See "
                  "https://github.com/rapidsai/cudf/issues/22935).";

  srand(31337);
  auto const num_rows = 8;
  auto expected       = create_random_fixed_table<int>(4, num_rows, false);

  auto filepath = temp_env->get_temp_filepath("ZeroColumns.parquet");
  cudf::io::write_parquet(
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected));

  // Project no columns: the result should be (num_rows, 0), not (0, 0).
  auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names(std::vector<std::string>{})
                     .build();
  auto result = cudf::io::read_parquet(read_opts);

  EXPECT_EQ(result.tbl->view().num_columns(), 0);
  EXPECT_EQ(result.tbl->view().num_rows(), num_rows);
}

TEST_F(ParquetReaderTest, UserBoundsWithNulls)
{
  // clang-format off
  cudf::test::fixed_width_column_wrapper<float> col{{1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8}
                                                   ,{true,true,true,false,false,false,true,true, true,true,true,true,true,true,true,true, false,false,false,false,false,false,false,false, true,true,true,true,true,true,false,false,  true,false,true,true,true,true,true,true, true,true,true,true,true,true,true,true, true,true,true,true,true,true,true,true, true,true,true,true,true,true,true,false}};
  // clang-format on
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNulls.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {1, 3}, {3, -1},
                                           {31, -1}, {32, -1}, {33, -1},
                                           {31, 5}, {32, 5}, {33, 5},
                                           {-1, 7}, {-1, 31}, {-1, 32}, {-1, 33},
                                           {62, -1}, {63, -1},
                                           {62, 2}, {63, 1}};
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsMixedTypes)
{
  constexpr int num_rows = 32 * 1024;

  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = cuda::counting_iterator<int>{0};

  // int64
  cudf::test::fixed_width_column_wrapper<int64_t> c0(values, values + num_rows, valids);

  // list<float>
  constexpr int floats_per_row = 4;
  auto c1_offset_iter          = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * floats_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets(c1_offset_iter,
                                                                     c1_offset_iter + num_rows + 1);
  cudf::test::fixed_width_column_wrapper<float> c1_floats(
    values, values + (num_rows * floats_per_row), valids);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

  auto _c1 = cudf::make_lists_column(
    num_rows, c1_offsets.release(), c1_floats.release(), null_count, std::move(null_mask));
  auto c1 = cudf::purge_nonempty_nulls(*_c1);

  // list<list<int>>
  auto c2 = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);

  // struct<list<string>, int, float>
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });
  constexpr int string_per_row  = 3;
  constexpr int num_string_rows = num_rows * string_per_row;
  cudf::test::strings_column_wrapper string_col{string_iter, string_iter + num_string_rows};
  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                  offset_iter + num_rows + 1);

  auto _c3_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 200; });
  std::vector<bool> c3_valids(num_rows);
  std::copy(_c3_valids, _c3_valids + num_rows, c3_valids.begin());
  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(valids, valids + num_rows);
  auto _c3_list                   = cudf::make_lists_column(
    num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
  auto c3_list = cudf::purge_nonempty_nulls(*_c3_list);
  cudf::test::fixed_width_column_wrapper<int> c3_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> c3_floats(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> c3_children;
  c3_children.push_back(std::move(c3_list));
  c3_children.push_back(c3_ints.release());
  c3_children.push_back(c3_floats.release());
  cudf::test::structs_column_wrapper _c3(std::move(c3_children), c3_valids);
  auto c3 = cudf::purge_nonempty_nulls(_c3);

  // write it out
  cudf::table_view tbl({c0, *c1, *c2, *c3});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsMixedTypes.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // read it back
  std::vector<std::pair<int, int>> params{
    {-1, -1}, {0, num_rows}, {1, num_rows - 1}, {num_rows - 1, 1}, {517, 22000}};
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsLarge)
{
  constexpr int num_rows = 30 * 10000;

  std::mt19937 gen(6747);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = cuda::counting_iterator<int>{0};

  cudf::test::fixed_width_column_wrapper<int> col(values, values + num_rows, valids);

  // this file will have row groups of 10,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .row_group_size_rows(10000)
      .max_page_size_rows(1000);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {16130, -1}, {19999, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {9900, 1001}, {9900, 2000}, {29999, 2}, {139997, -1},
                                           {167878, 3}, {229976, 31},
                                           {240031, 17}, {290001, 9899}, {299999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ListUserBoundsWithNullsLarge)
{
  constexpr int num_rows = 5 * 10000;
  auto colp              = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);
  cudf::column_view col  = *colp;

  // this file will have row groups of 10,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("ListUserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .row_group_size_rows(10000)
      .max_page_size_rows(1000);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {1670, -1}, {44997, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {9900, 1001}, {9900, 2000}, {29999, 2},
                                           {16567, 3}, {42976, 31},
                                           {40231, 17}, {19000, 9899}, {49999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ReorderedColumns)
{
  {
    auto a = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};
    auto b = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};

    cudf::table_view tbl{{a, b}};
    auto filepath = write_parquet_temp_file(tbl, "ReorderedColumns.parquet", {"a", "b"});

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_names({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  {
    auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
    auto b = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};

    cudf::table_view tbl{{a, b}};
    auto filepath = write_parquet_temp_file(tbl, "ReorderedColumns2.parquet", {"a", "b"});

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_names({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3, 10, 20, 30};
  auto b = cudf::test::strings_column_wrapper{{"a", "", "c", "cats", "dogs", "owls"},
                                              {true, false, true, true, false, true}};
  auto c = cudf::test::fixed_width_column_wrapper<int>{{15, 16, 17, 25, 26, 32},
                                                       {false, true, true, true, true, false}};
  auto d = cudf::test::strings_column_wrapper{"ducks", "sheep", "cows", "fish", "birds", "ants"};

  cudf::table_view tbl{{a, b, c, d}};
  auto filepath = temp_env->get_temp_filepath("ReorderedColumns3.parquet");
  cudf::io::table_input_metadata md(tbl);
  md.column_metadata[0].set_name("a");
  md.column_metadata[0].set_parquet_field_id(10);
  md.column_metadata[1].set_name("b");
  md.column_metadata[1].set_parquet_field_id(11);
  md.column_metadata[2].set_name("c");
  md.column_metadata[2].set_parquet_field_id(12);
  md.column_metadata[3].set_name("d");
  md.column_metadata[3].set_parquet_field_id(13);
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .metadata(std::move(md));
  cudf::io::write_parquet(opts);

  {
    // read them out of order using indices
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_indices({3, 0, 1, 2});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), c);
  }

  {
    // read them out of order using Parquet field IDs
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_field_ids({13, 10, 11, 12});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), c);
  }

  {
    // missing Parquet field IDs are errors when ignore_missing_columns is disabled
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_field_ids({999})
        .ignore_missing_columns(false);

    EXPECT_THROW(cudf::io::read_parquet(read_opts), std::invalid_argument);
  }

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_names({"c", "d", "a", "b"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), b);
  }

  {
    // read them out of order using names
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_names({"d", "c", "b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), a);
  }

  {
    // read them out of order using indices
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .column_indices({3, 2, 1, 0});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), a);
  }
}

TEST_F(ParquetReaderTest, SelectColumnByMissingFieldIds)
{
  auto const a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
  auto const b = cudf::test::fixed_width_column_wrapper<int>{4, 5, 6};
  cudf::table_view const table{{a, b}};
  auto const filepath = write_parquet_temp_file(table, "SelectColumnByMissingFieldIds.parquet");

  {
    auto const options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .column_field_ids({1})
                           .build();
    auto const result = cudf::io::read_parquet(options);
    EXPECT_EQ(result.tbl->num_columns(), 0);
  }

  {
    auto const options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .column_field_ids({1})
                           .ignore_missing_columns(false)
                           .build();
    EXPECT_THROW(cudf::io::read_parquet(options), std::invalid_argument);
  }
}

TEST_F(ParquetReaderTest, ColumnSelectionModesAreExclusive)
{
  auto selected_by_name = cudf::io::parquet_reader_options{};
  selected_by_name.set_column_names({});
  EXPECT_THROW(selected_by_name.set_column_indices({}), cudf::logic_error);
  EXPECT_THROW(selected_by_name.set_column_field_ids({}), cudf::logic_error);

  auto selected_by_index = cudf::io::parquet_reader_options{};
  selected_by_index.set_column_indices({});
  EXPECT_THROW(selected_by_index.set_column_field_ids({}), cudf::logic_error);
  EXPECT_THROW(selected_by_index.enable_allow_mismatched_pq_schemas(true), cudf::logic_error);

  auto selected_by_field_id = cudf::io::parquet_reader_options{};
  selected_by_field_id.set_column_field_ids({});
  EXPECT_THROW(selected_by_field_id.set_column_names({}), cudf::logic_error);

  auto mismatched_schemas_allowed = cudf::io::parquet_reader_options{};
  mismatched_schemas_allowed.enable_allow_mismatched_pq_schemas(true);
  EXPECT_THROW(mismatched_schemas_allowed.set_column_indices({}), cudf::logic_error);
}

TEST_F(ParquetReaderTest, SelectNestedColumn)
{
  // Struct<is_human:bool,
  //        Struct<weight:float,
  //               ages:int,
  //               land_unit:List<int>>,
  //               flats:List<List<int>>
  //              >
  //       >

  auto weights_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col},
                                                     {true, true, true, true, false, true}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {true, true, false, true, true, false}};

  auto struct_2 = cudf::test::structs_column_wrapper{{is_human_col, struct_1},
                                                     {false, true, true, true, true, true}}
                    .release();

  auto input = table_view({*struct_2});

  cudf::io::table_input_metadata input_metadata(input);
  input_metadata.column_metadata[0].set_name("being");
  input_metadata.column_metadata[0].set_parquet_field_id(1);
  input_metadata.column_metadata[0].child(0).set_name("human?");
  input_metadata.column_metadata[0].child(0).set_parquet_field_id(2);
  input_metadata.column_metadata[0].child(1).set_name("particulars");
  input_metadata.column_metadata[0].child(1).set_parquet_field_id(3);
  input_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  input_metadata.column_metadata[0].child(1).child(0).set_parquet_field_id(4);
  input_metadata.column_metadata[0].child(1).child(1).set_name("age");
  input_metadata.column_metadata[0].child(1).child(1).set_parquet_field_id(5);

  auto filepath = temp_env->get_temp_filepath("SelectNestedColumn.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .metadata(std::move(input_metadata));
  cudf::io::write_parquet(args);

  {  // Test selecting a single leaf from the table
    auto expect_s_1 =
      cudf::test::structs_column_wrapper{{ages_col}, {true, true, true, true, false, true}};
    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expect_s_1}, {false, true, true, true, true, true}}
        .release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");

    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .column_names({"being.particulars.age"});
    auto result = cudf::io::read_parquet(read_args);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);

    // Test selecting a single leaf by Parquet field ID
    read_args = cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
                  .column_field_ids({5});
    result = cudf::io::read_parquet(read_args);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting a non-leaf and expecting all hierarchy from that node onwards
    auto expected_s_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col},
                                                           {true, true, true, true, false, true}};

    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expected_s_1}, {false, true, true, true, true, true}}
        .release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("weight");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("age");

    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .column_names({"being.particulars"});
    auto result = cudf::io::read_parquet(read_args);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);

    // Test selecting a non-leaf by Parquet field ID
    read_args = cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
                  .column_field_ids({3});
    result = cudf::io::read_parquet(read_args);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting struct children out of order
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .column_names({"being.particulars.age", "being.particulars.weight", "being.human?"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expect_s_1 = cudf::test::structs_column_wrapper{{ages_col, weights_col},
                                                         {true, true, true, true, false, true}};

    auto expect_s_2 = cudf::test::structs_column_wrapper{{expect_s_1, is_human_col},
                                                         {false, true, true, true, true, true}}
                        .release();

    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("weight");
    expected_metadata.column_metadata[0].child(1).set_name("human?");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }
}

TEST_F(ParquetReaderTest, SelectMismatchedStructChildByFieldId)
{
  auto x_a      = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto y_a      = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30};
  auto struct_a = cudf::test::structs_column_wrapper{{x_a, y_a}, {true, true, true}}.release();
  cudf::table_view const table_a{{*struct_a}};

  auto path_a = temp_env->get_temp_filepath("SelectNestedFieldIdChildOrderA.parquet");
  cudf::io::table_input_metadata metadata_a(table_a);
  metadata_a.column_metadata[0].set_name("record").set_parquet_field_id(1);
  metadata_a.column_metadata[0].child(0).set_name("x").set_parquet_field_id(2);
  metadata_a.column_metadata[0].child(1).set_name("y").set_parquet_field_id(3);
  auto write_args_a =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{path_a}, table_a)
      .metadata(std::move(metadata_a));
  cudf::io::write_parquet(write_args_a);

  auto y_b      = cudf::test::fixed_width_column_wrapper<int32_t>{40, 50};
  auto x_b      = cudf::test::fixed_width_column_wrapper<int32_t>{4, 5};
  auto struct_b = cudf::test::structs_column_wrapper{{y_b, x_b}, {true, true}}.release();
  cudf::table_view const table_b{{*struct_b}};

  auto path_b = temp_env->get_temp_filepath("SelectNestedFieldIdChildOrderB.parquet");
  cudf::io::table_input_metadata metadata_b(table_b);
  metadata_b.column_metadata[0].set_name("record").set_parquet_field_id(1);
  metadata_b.column_metadata[0].child(0).set_name("y").set_parquet_field_id(3);
  metadata_b.column_metadata[0].child(1).set_name("x").set_parquet_field_id(2);
  auto write_args_b =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{path_b}, table_b)
      .metadata(std::move(metadata_b));
  cudf::io::write_parquet(write_args_b);

  auto expected_x = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
  auto expected_y = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40, 50};
  auto expected_struct =
    cudf::test::structs_column_wrapper{{expected_x, expected_y}, {true, true, true, true, true}}
      .release();
  cudf::table_view const expected{{*expected_struct}};

  auto const read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path_a, path_b}})
      .allow_mismatched_pq_schemas(true)
      .column_field_ids({1})
      .build();
  auto const result = cudf::io::read_parquet(read_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, DecimalRead)
{
  {
    /* We could add a dataset to include this file, but we don't want tests in cudf to have data.
       This test is a temporary test until python gains the ability to write decimal, so we're
       embedding
       a parquet file directly into the code here to prevent issues with finding the file */
    constexpr unsigned int decimals_parquet_len = 2366;
    std::array<unsigned char, decimals_parquet_len> const decimals_parquet{
      0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0xb0, 0x03, 0x15, 0xb8, 0x03, 0x2c, 0x15, 0x6a,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36, 0x02, 0x28, 0x04, 0x7f, 0x96, 0x98, 0x00,
      0x18, 0x04, 0x81, 0x69, 0x67, 0xff, 0x00, 0x00, 0x00, 0xd8, 0x01, 0xf0, 0xd7, 0x04, 0x00,
      0x00, 0x00, 0x64, 0x01, 0x03, 0x06, 0x68, 0x12, 0xdc, 0xff, 0xbd, 0x18, 0xfd, 0xff, 0x64,
      0x13, 0x80, 0x00, 0xb3, 0x5d, 0x62, 0x00, 0x90, 0x35, 0xa9, 0xff, 0xa2, 0xde, 0xe3, 0xff,
      0xe9, 0xbf, 0x96, 0xff, 0x1f, 0x8a, 0x98, 0xff, 0xb1, 0x50, 0x34, 0x00, 0x88, 0x24, 0x59,
      0x00, 0x2a, 0x33, 0xbe, 0xff, 0xd5, 0x16, 0xbc, 0xff, 0x13, 0x50, 0x8d, 0xff, 0xcb, 0x63,
      0x2d, 0x00, 0x80, 0x8f, 0xbe, 0xff, 0x82, 0x40, 0x10, 0x00, 0x84, 0x68, 0x70, 0xff, 0x9b,
      0x69, 0x78, 0x00, 0x14, 0x6c, 0x10, 0x00, 0x50, 0xd9, 0xe1, 0xff, 0xaa, 0xcd, 0x6a, 0x00,
      0xcf, 0xb1, 0x28, 0x00, 0x77, 0x57, 0x8d, 0x00, 0xee, 0x05, 0x79, 0x00, 0xf0, 0x15, 0xeb,
      0xff, 0x02, 0xe2, 0x06, 0x00, 0x87, 0x43, 0x86, 0x00, 0xf8, 0x2d, 0x2e, 0x00, 0xee, 0x2e,
      0x98, 0xff, 0x39, 0xcb, 0x4d, 0x00, 0x1e, 0x6b, 0xea, 0xff, 0x80, 0x8e, 0x6c, 0xff, 0x97,
      0x25, 0x26, 0x00, 0x4d, 0x0d, 0x0a, 0x00, 0xca, 0x64, 0x7f, 0x00, 0xf4, 0xbe, 0xa1, 0xff,
      0xe2, 0x12, 0x6c, 0xff, 0xbd, 0x77, 0xae, 0xff, 0xf9, 0x4b, 0x36, 0x00, 0xb0, 0xe3, 0x79,
      0xff, 0xa2, 0x2a, 0x29, 0x00, 0xcd, 0x06, 0xbc, 0xff, 0x2d, 0xa3, 0x7e, 0x00, 0xa9, 0x08,
      0xa1, 0xff, 0xbf, 0x81, 0xd0, 0xff, 0x4f, 0x03, 0x73, 0x00, 0xb0, 0x99, 0x0c, 0x00, 0xbd,
      0x6f, 0xf8, 0xff, 0x6b, 0x02, 0x05, 0x00, 0xc1, 0xe1, 0xba, 0xff, 0x81, 0x69, 0x67, 0xff,
      0x7f, 0x96, 0x98, 0x00, 0x15, 0x00, 0x15, 0xd0, 0x06, 0x15, 0xda, 0x06, 0x2c, 0x15, 0x6a,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36, 0x02, 0x28, 0x08, 0xff, 0x3f, 0x7a, 0x10,
      0xf3, 0x5a, 0x00, 0x00, 0x18, 0x08, 0x01, 0xc0, 0x85, 0xef, 0x0c, 0xa5, 0xff, 0xff, 0x00,
      0x00, 0x00, 0xa8, 0x03, 0xf4, 0xa7, 0x01, 0x04, 0x00, 0x00, 0x00, 0x64, 0x01, 0x03, 0x06,
      0x55, 0x6f, 0xc5, 0xe4, 0x9f, 0x1a, 0x00, 0x00, 0x47, 0x89, 0x0a, 0xe8, 0x58, 0xf0, 0xff,
      0xff, 0x63, 0xee, 0x21, 0xdd, 0xdd, 0xca, 0xff, 0xff, 0xbe, 0x6f, 0x3b, 0xaa, 0xe9, 0x3d,
      0x00, 0x00, 0xd6, 0x91, 0x2a, 0xb7, 0x08, 0x02, 0x00, 0x00, 0x75, 0x45, 0x2c, 0xd7, 0x76,
      0x0c, 0x00, 0x00, 0x54, 0x49, 0x92, 0x44, 0x9c, 0xbf, 0xff, 0xff, 0x41, 0xa9, 0x6d, 0xec,
      0x7a, 0xd0, 0xff, 0xff, 0x27, 0xa0, 0x23, 0x41, 0x44, 0xc1, 0xff, 0xff, 0x18, 0xd4, 0xe1,
      0x30, 0xd3, 0xe0, 0xff, 0xff, 0x59, 0xac, 0x14, 0xf4, 0xec, 0x58, 0x00, 0x00, 0x2c, 0x17,
      0x29, 0x57, 0x44, 0x13, 0x00, 0x00, 0xa2, 0x0d, 0x4a, 0xcc, 0x63, 0xff, 0xff, 0xff, 0x81,
      0x33, 0xbc, 0xda, 0xd5, 0xda, 0xff, 0xff, 0x4c, 0x05, 0xf4, 0x78, 0x19, 0xea, 0xff, 0xff,
      0x06, 0x71, 0x25, 0xde, 0x5a, 0xaf, 0xff, 0xff, 0x95, 0x32, 0x5f, 0x76, 0x98, 0xb3, 0xff,
      0xff, 0xf1, 0x34, 0x3c, 0xbf, 0xa8, 0xbe, 0xff, 0xff, 0x27, 0x73, 0x40, 0x0c, 0x7d, 0xcd,
      0xff, 0xff, 0x68, 0xa9, 0xc2, 0xe9, 0x2c, 0x03, 0x00, 0x00, 0x3f, 0x79, 0xd9, 0x04, 0x8c,
      0xe5, 0xff, 0xff, 0x91, 0xb4, 0x9b, 0xe3, 0x8f, 0x21, 0x00, 0x00, 0xb8, 0x20, 0xc8, 0xc2,
      0x4d, 0xa6, 0xff, 0xff, 0x47, 0xfa, 0xde, 0x36, 0x4a, 0xf3, 0xff, 0xff, 0x72, 0x80, 0x94,
      0x59, 0xdd, 0x4e, 0x00, 0x00, 0x29, 0xe4, 0xd6, 0x43, 0xb0, 0xf0, 0xff, 0xff, 0x68, 0x36,
      0xbc, 0x2d, 0xd1, 0xa9, 0xff, 0xff, 0xbc, 0xe4, 0xbe, 0xd7, 0xed, 0x1b, 0x00, 0x00, 0x02,
      0x8b, 0xcb, 0xd7, 0xed, 0x47, 0x00, 0x00, 0x3c, 0x06, 0xe4, 0xda, 0xc7, 0x47, 0x00, 0x00,
      0xf3, 0x39, 0x55, 0x28, 0x97, 0xba, 0xff, 0xff, 0x07, 0x79, 0x38, 0x4e, 0xe0, 0x21, 0x00,
      0x00, 0xde, 0xed, 0x1c, 0x23, 0x09, 0x49, 0x00, 0x00, 0x49, 0x46, 0x49, 0x5d, 0x8f, 0x34,
      0x00, 0x00, 0x38, 0x18, 0x50, 0xf6, 0xa1, 0x11, 0x00, 0x00, 0xdf, 0xb8, 0x19, 0x14, 0xd1,
      0xe1, 0xff, 0xff, 0x2c, 0x56, 0x72, 0x93, 0x64, 0x3f, 0x00, 0x00, 0x1c, 0xe0, 0xbe, 0x87,
      0x7d, 0xf9, 0xff, 0xff, 0x73, 0x0e, 0x3c, 0x01, 0x91, 0xf9, 0xff, 0xff, 0xb2, 0x37, 0x85,
      0x81, 0x5f, 0x54, 0x00, 0x00, 0x58, 0x44, 0xb0, 0x1a, 0xac, 0xbb, 0xff, 0xff, 0x36, 0xbf,
      0xbe, 0x5e, 0x22, 0xff, 0xff, 0xff, 0x06, 0x20, 0xa0, 0x23, 0x0d, 0x3b, 0x00, 0x00, 0x19,
      0xc6, 0x49, 0x0a, 0x00, 0xcf, 0xff, 0xff, 0x4f, 0xcd, 0xc6, 0x95, 0x4b, 0xf1, 0xff, 0xff,
      0xa3, 0x59, 0xaf, 0x65, 0xec, 0xe9, 0xff, 0xff, 0x58, 0xef, 0x05, 0x50, 0x63, 0xe4, 0xff,
      0xff, 0xc7, 0x6a, 0x9e, 0xf1, 0x69, 0x20, 0x00, 0x00, 0xd1, 0xb3, 0xc9, 0x14, 0xb2, 0x29,
      0x00, 0x00, 0x1d, 0x48, 0x16, 0x70, 0xf0, 0x40, 0x00, 0x00, 0x01, 0xc0, 0x85, 0xef, 0x0c,
      0xa5, 0xff, 0xff, 0xff, 0x3f, 0x7a, 0x10, 0xf3, 0x5a, 0x00, 0x00, 0x15, 0x00, 0x15, 0x90,
      0x0d, 0x15, 0x9a, 0x0d, 0x2c, 0x15, 0x6a, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36,
      0x02, 0x28, 0x10, 0x4b, 0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f,
      0xff, 0xff, 0xff, 0xff, 0x18, 0x10, 0xb4, 0xc4, 0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6,
      0x75, 0xdd, 0xc0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xc8, 0x06, 0xf4, 0x47, 0x03,
      0x04, 0x00, 0x00, 0x00, 0x64, 0x01, 0x03, 0x06, 0x05, 0x49, 0xf7, 0xfc, 0x89, 0x3d, 0x3e,
      0x20, 0x07, 0x72, 0x3e, 0xa1, 0x66, 0x81, 0x67, 0x80, 0x23, 0x78, 0x06, 0x68, 0x0e, 0x78,
      0xf5, 0x08, 0xed, 0x20, 0xcd, 0x0e, 0x7f, 0x9c, 0x70, 0xa0, 0xb9, 0x16, 0x44, 0xb2, 0x41,
      0x62, 0xba, 0x82, 0xad, 0xe1, 0x12, 0x9b, 0xa6, 0x53, 0x8d, 0x20, 0x27, 0xd5, 0x84, 0x63,
      0xb8, 0x07, 0x4b, 0x5b, 0xa4, 0x1c, 0xa4, 0x1c, 0x17, 0xbf, 0x4b, 0x00, 0x24, 0x04, 0x56,
      0xa8, 0x52, 0xaf, 0x33, 0xf7, 0xad, 0x7c, 0xc8, 0x83, 0x25, 0x13, 0xaf, 0x80, 0x25, 0x6f,
      0xbd, 0xd1, 0x15, 0x69, 0x64, 0x20, 0x7b, 0xd7, 0x33, 0xba, 0x66, 0x29, 0x8a, 0x00, 0xda,
      0x42, 0x07, 0x2c, 0x6c, 0x39, 0x76, 0x9f, 0xdc, 0x17, 0xad, 0xb6, 0x58, 0xdf, 0x5f, 0x00,
      0x18, 0x3a, 0xae, 0x1c, 0xd6, 0x5f, 0x9d, 0x78, 0x8d, 0x73, 0xdd, 0x3e, 0xd6, 0x18, 0x33,
      0x40, 0xe4, 0x36, 0xde, 0xb0, 0xb7, 0x33, 0x2a, 0x6b, 0x08, 0x03, 0x6c, 0x6d, 0x8f, 0x13,
      0x93, 0xd0, 0xd7, 0x87, 0x62, 0x63, 0x53, 0xfb, 0xd8, 0xbb, 0xc9, 0x54, 0x90, 0xd6, 0xa9,
      0x8f, 0xc8, 0x60, 0xbd, 0xec, 0x75, 0x23, 0x9a, 0x21, 0xec, 0xe4, 0x86, 0x43, 0xd7, 0xc1,
      0x88, 0xdc, 0x82, 0x00, 0x32, 0x79, 0xc9, 0x2b, 0x70, 0x85, 0xb7, 0x25, 0xa1, 0xcc, 0x7d,
      0x0b, 0x29, 0x03, 0xea, 0x80, 0xff, 0x9b, 0xf3, 0x24, 0x7f, 0xd1, 0xff, 0xf0, 0x22, 0x65,
      0x85, 0x99, 0x17, 0x63, 0xc2, 0xc0, 0xb7, 0x62, 0x05, 0xda, 0x7a, 0xa0, 0xc3, 0x2a, 0x6f,
      0x1f, 0xee, 0x1f, 0x31, 0xa8, 0x42, 0x80, 0xe4, 0xb7, 0x6c, 0xf6, 0xac, 0x47, 0xb0, 0x17,
      0x69, 0xcb, 0xff, 0x66, 0x8a, 0xd6, 0x25, 0x00, 0xf3, 0xcf, 0x0a, 0xaf, 0xf8, 0x92, 0x8a,
      0xa0, 0xdf, 0x71, 0x13, 0x8d, 0x9d, 0xff, 0x7e, 0xe0, 0x0a, 0x52, 0xf1, 0x97, 0x01, 0xa9,
      0x73, 0x27, 0xfd, 0x63, 0x58, 0x00, 0x32, 0xa6, 0xf6, 0x78, 0xb8, 0xe4, 0xfd, 0x20, 0x7c,
      0x90, 0xee, 0xad, 0x8c, 0xc9, 0x71, 0x35, 0x66, 0x71, 0x3c, 0xe0, 0xe4, 0x0b, 0xbb, 0xa0,
      0x50, 0xe9, 0xf2, 0x81, 0x1d, 0x3a, 0x95, 0x94, 0x00, 0xd5, 0x49, 0x00, 0x07, 0xdf, 0x21,
      0x53, 0x36, 0x8d, 0x9e, 0xd9, 0xa5, 0x52, 0x4d, 0x0d, 0x29, 0x74, 0xf0, 0x40, 0xbd, 0xda,
      0x63, 0x4e, 0xdd, 0x91, 0x8e, 0xa6, 0xa7, 0xf6, 0x78, 0x58, 0x3b, 0x0a, 0x5c, 0x60, 0x3c,
      0x15, 0x34, 0xf8, 0x2c, 0x21, 0xe3, 0x56, 0x1b, 0x9e, 0xd9, 0x56, 0xd3, 0x13, 0x2e, 0x80,
      0x2c, 0x36, 0xda, 0x1d, 0xc8, 0xfb, 0x52, 0xee, 0x17, 0xb3, 0x2b, 0xf3, 0xd2, 0xeb, 0x29,
      0xa0, 0x37, 0xa0, 0x12, 0xce, 0x1c, 0x50, 0x6a, 0xf4, 0x11, 0xcd, 0x96, 0x88, 0x3f, 0x43,
      0x78, 0xc0, 0x2c, 0x53, 0x6c, 0xa6, 0xdf, 0xb9, 0x9e, 0x93, 0xd4, 0x1e, 0xa9, 0x7f, 0x67,
      0xa6, 0xc1, 0x80, 0x46, 0x0f, 0x63, 0x7d, 0x15, 0xf2, 0x4c, 0xc5, 0xda, 0x11, 0x9a, 0x20,
      0x67, 0x27, 0xe8, 0x00, 0xec, 0x03, 0x1d, 0x15, 0xa7, 0x92, 0xb3, 0x1f, 0xda, 0x20, 0x92,
      0xd8, 0x00, 0xfb, 0x06, 0x80, 0xeb, 0x4b, 0x0c, 0xc1, 0x1f, 0x49, 0x40, 0x06, 0x8d, 0x8a,
      0xf8, 0x34, 0xb1, 0x0c, 0x1d, 0x20, 0xd0, 0x47, 0xe5, 0xb1, 0x7e, 0xf7, 0xe4, 0xb4, 0x7e,
      0x9c, 0x84, 0x18, 0x61, 0x32, 0x4f, 0xc0, 0xc2, 0xb2, 0xcc, 0x63, 0xf6, 0xe1, 0x16, 0xd6,
      0xd9, 0x4b, 0x74, 0x13, 0x01, 0xa1, 0xe2, 0x00, 0xb7, 0x9e, 0xc1, 0x3a, 0xc5, 0xaf, 0xe8,
      0x54, 0x07, 0x2a, 0x20, 0xfd, 0x2c, 0x6f, 0xb9, 0x80, 0x18, 0x92, 0x87, 0xa0, 0x81, 0x24,
      0x60, 0x47, 0x17, 0x4f, 0xbc, 0xbe, 0xf5, 0x03, 0x69, 0x80, 0xe3, 0x10, 0x54, 0xd6, 0x68,
      0x7d, 0x75, 0xd3, 0x0a, 0x45, 0x38, 0x9e, 0xa9, 0xfd, 0x05, 0x40, 0xd2, 0x1e, 0x6f, 0x5c,
      0x30, 0x10, 0xfe, 0x9b, 0x9f, 0x6d, 0xc0, 0x9d, 0x6c, 0x17, 0x7d, 0x00, 0x09, 0xb6, 0x8a,
      0x31, 0x8e, 0x1b, 0x6b, 0x84, 0x1e, 0x79, 0xce, 0x10, 0x55, 0x59, 0x6a, 0x40, 0x16, 0xdc,
      0x9a, 0xcf, 0x4d, 0xb0, 0x8f, 0xac, 0xe3, 0x8d, 0xee, 0xd2, 0xef, 0x01, 0x8c, 0xe0, 0x2b,
      0x24, 0xe5, 0xb4, 0xe1, 0x86, 0x72, 0x00, 0x30, 0x07, 0xce, 0x02, 0x23, 0x41, 0x33, 0x40,
      0xf0, 0x9b, 0xc2, 0x2d, 0x30, 0xec, 0x3b, 0x17, 0xb2, 0x8f, 0x64, 0x7d, 0xcd, 0x70, 0x9e,
      0x80, 0x22, 0xb5, 0xdf, 0x6d, 0x2a, 0x43, 0xd4, 0x2b, 0x5a, 0xf6, 0x96, 0xa6, 0xea, 0x91,
      0x62, 0x80, 0x39, 0xf2, 0x5a, 0x8e, 0xc0, 0xb9, 0x29, 0x99, 0x17, 0xe7, 0x35, 0x2c, 0xf6,
      0x4d, 0x18, 0x00, 0x48, 0x10, 0x85, 0xb4, 0x3f, 0x89, 0x60, 0x49, 0x6e, 0xf0, 0xcd, 0x9d,
      0x92, 0xeb, 0x96, 0x80, 0xcf, 0xf9, 0xf1, 0x46, 0x1d, 0xc0, 0x49, 0xb3, 0x36, 0x2e, 0x24,
      0xc8, 0xdb, 0x41, 0x72, 0x20, 0xf5, 0xde, 0x5c, 0xf9, 0x4a, 0x6e, 0xa0, 0x0b, 0x13, 0xfc,
      0x2d, 0x17, 0x07, 0x16, 0x5e, 0x00, 0x3c, 0x54, 0x41, 0x0e, 0xa2, 0x0d, 0xf3, 0x48, 0x12,
      0x2e, 0x7c, 0xab, 0x3c, 0x59, 0x1c, 0x40, 0xca, 0xb0, 0x71, 0xc7, 0x29, 0xf0, 0xbb, 0x9f,
      0xf4, 0x3f, 0x25, 0x49, 0xad, 0xc2, 0x8f, 0x80, 0x04, 0x38, 0x6d, 0x35, 0x02, 0xca, 0xe6,
      0x02, 0x83, 0x89, 0x4e, 0x74, 0xdb, 0x08, 0x5a, 0x80, 0x13, 0x99, 0xd4, 0x26, 0xc1, 0x27,
      0xce, 0xb0, 0x98, 0x99, 0xca, 0xf6, 0x3e, 0x50, 0x49, 0xd0, 0xbf, 0xcb, 0x6f, 0xbe, 0x5b,
      0x92, 0x63, 0xde, 0x94, 0xd3, 0x8f, 0x07, 0x06, 0x0f, 0x2b, 0x80, 0x36, 0xf1, 0x77, 0xf6,
      0x29, 0x33, 0x13, 0xa9, 0x4a, 0x55, 0x3d, 0x6c, 0xca, 0xdb, 0x4e, 0x40, 0xc4, 0x95, 0x54,
      0xf4, 0xe2, 0x8c, 0x1b, 0xa0, 0xfe, 0x30, 0x50, 0x9d, 0x62, 0xbc, 0x5c, 0x00, 0xb4, 0xc4,
      0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6, 0x75, 0xdd, 0xc0, 0x00, 0x00, 0x00, 0x01, 0x4b,
      0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f, 0xff, 0xff, 0xff, 0xff,
      0x15, 0x02, 0x19, 0x4c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68,
      0x65, 0x6d, 0x61, 0x15, 0x06, 0x00, 0x15, 0x02, 0x25, 0x02, 0x18, 0x06, 0x64, 0x65, 0x63,
      0x37, 0x70, 0x34, 0x25, 0x0a, 0x15, 0x08, 0x15, 0x0e, 0x00, 0x15, 0x04, 0x25, 0x02, 0x18,
      0x07, 0x64, 0x65, 0x63, 0x31, 0x34, 0x70, 0x35, 0x25, 0x0a, 0x15, 0x0a, 0x15, 0x1c, 0x00,
      0x15, 0x0e, 0x15, 0x20, 0x15, 0x02, 0x18, 0x08, 0x64, 0x65, 0x63, 0x33, 0x38, 0x70, 0x31,
      0x38, 0x25, 0x0a, 0x15, 0x24, 0x15, 0x4c, 0x00, 0x16, 0x6a, 0x19, 0x1c, 0x19, 0x3c, 0x26,
      0x08, 0x1c, 0x15, 0x02, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x06, 0x64, 0x65, 0x63,
      0x37, 0x70, 0x34, 0x15, 0x02, 0x16, 0x6a, 0x16, 0xf6, 0x03, 0x16, 0xfe, 0x03, 0x26, 0x08,
      0x3c, 0x36, 0x02, 0x28, 0x04, 0x7f, 0x96, 0x98, 0x00, 0x18, 0x04, 0x81, 0x69, 0x67, 0xff,
      0x00, 0x19, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0x86, 0x04,
      0x1c, 0x15, 0x04, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x07, 0x64, 0x65, 0x63, 0x31,
      0x34, 0x70, 0x35, 0x15, 0x02, 0x16, 0x6a, 0x16, 0xa6, 0x07, 0x16, 0xb0, 0x07, 0x26, 0x86,
      0x04, 0x3c, 0x36, 0x02, 0x28, 0x08, 0xff, 0x3f, 0x7a, 0x10, 0xf3, 0x5a, 0x00, 0x00, 0x18,
      0x08, 0x01, 0xc0, 0x85, 0xef, 0x0c, 0xa5, 0xff, 0xff, 0x00, 0x19, 0x1c, 0x15, 0x00, 0x15,
      0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0xb6, 0x0b, 0x1c, 0x15, 0x0e, 0x19, 0x35, 0x06,
      0x08, 0x00, 0x19, 0x18, 0x08, 0x64, 0x65, 0x63, 0x33, 0x38, 0x70, 0x31, 0x38, 0x15, 0x02,
      0x16, 0x6a, 0x16, 0x86, 0x0e, 0x16, 0x90, 0x0e, 0x26, 0xb6, 0x0b, 0x3c, 0x36, 0x02, 0x28,
      0x10, 0x4b, 0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f, 0xff, 0xff,
      0xff, 0xff, 0x18, 0x10, 0xb4, 0xc4, 0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6, 0x75, 0xdd,
      0xc0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x19, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00,
      0x00, 0x00, 0x16, 0xa2, 0x19, 0x16, 0x6a, 0x00, 0x19, 0x2c, 0x18, 0x18, 0x6f, 0x72, 0x67,
      0x2e, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2e, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x2e, 0x76,
      0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x18, 0x05, 0x33, 0x2e, 0x30, 0x2e, 0x31, 0x00, 0x18,
      0x29, 0x6f, 0x72, 0x67, 0x2e, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2e, 0x73, 0x70, 0x61,
      0x72, 0x6b, 0x2e, 0x73, 0x71, 0x6c, 0x2e, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2e,
      0x72, 0x6f, 0x77, 0x2e, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0xf4, 0x01,
      0x7b, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74,
      0x22, 0x2c, 0x22, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x73, 0x22, 0x3a, 0x5b, 0x7b, 0x22, 0x6e,
      0x61, 0x6d, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x37, 0x70, 0x34, 0x22, 0x2c, 0x22,
      0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28,
      0x37, 0x2c, 0x34, 0x29, 0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65,
      0x22, 0x3a, 0x74, 0x72, 0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74,
      0x61, 0x22, 0x3a, 0x7b, 0x7d, 0x7d, 0x2c, 0x7b, 0x22, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x3a,
      0x22, 0x64, 0x65, 0x63, 0x31, 0x34, 0x70, 0x35, 0x22, 0x2c, 0x22, 0x74, 0x79, 0x70, 0x65,
      0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28, 0x31, 0x34, 0x2c, 0x35,
      0x29, 0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x22, 0x3a, 0x74,
      0x72, 0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3a,
      0x7b, 0x7d, 0x7d, 0x2c, 0x7b, 0x22, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65,
      0x63, 0x33, 0x38, 0x70, 0x31, 0x38, 0x22, 0x2c, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a,
      0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28, 0x33, 0x38, 0x2c, 0x31, 0x38, 0x29,
      0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x22, 0x3a, 0x74, 0x72,
      0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3a, 0x7b,
      0x7d, 0x7d, 0x5d, 0x7d, 0x00, 0x18, 0x4a, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d,
      0x6d, 0x72, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x31, 0x2e, 0x31, 0x30,
      0x2e, 0x31, 0x20, 0x28, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x20, 0x61, 0x38, 0x39, 0x64, 0x66,
      0x38, 0x66, 0x39, 0x39, 0x33, 0x32, 0x62, 0x36, 0x65, 0x66, 0x36, 0x36, 0x33, 0x33, 0x64,
      0x30, 0x36, 0x30, 0x36, 0x39, 0x65, 0x35, 0x30, 0x63, 0x39, 0x62, 0x37, 0x39, 0x37, 0x30,
      0x62, 0x65, 0x62, 0x64, 0x31, 0x29, 0x19, 0x3c, 0x1c, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x1c,
      0x00, 0x00, 0x00, 0xd3, 0x02, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

    cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(decimals_parquet.data()), decimals_parquet.size()}});
    auto result = cudf::io::read_parquet(read_opts);

    auto validity = cudf::test::iterators::null_at(50);

    EXPECT_EQ(result.tbl->view().num_columns(), 3);

    std::array<int32_t, 53> col0_data{
      -2354584, -190275,  8393572,  6446515,  -5687920, -1843550, -6897687, -6780385, 3428529,
      5842056,  -4312278, -4450603, -7516141, 2974667,  -4288640, 1065090,  -9410428, 7891355,
      1076244,  -1975984, 6999466,  2666959,  9262967,  7931374,  -1370640, 451074,   8799111,
      3026424,  -6803730, 5098297,  -1414370, -9662848, 2499991,  658765,   8348874,  -6177036,
      -9694494, -5343299, 3558393,  -8789072, 2697890,  -4454707, 8299309,  -6223703, -3112513,
      7537487,  825776,   -495683,  328299,   -4529727, 0,        -9999999, 9999999};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(0).size()),
              sizeof(col0_data) / sizeof(col0_data[0]));
    cudf::test::fixed_point_column_wrapper<int32_t> col0(
      std::begin(col0_data), std::end(col0_data), validity, numeric::scale_type{-4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);

    std::array<int64_t, 53> col1_data{29274040266581,  -17210335917753, -58420730139037,
                                      68073792696254,  2236456014294,   13704555677045,
                                      -70797090469548, -52248605513407, -68976081919961,
                                      -34277313883112, 97774730521689,  21184241014572,
                                      -670882460254,   -40862944054399, -24079852370612,
                                      -88670167797498, -84007574359403, -71843004533519,
                                      -55538016554201, 3491435293032,   -29085437167297,
                                      36901882672273,  -98622066122568, -13974902998457,
                                      86712597643378,  -16835133643735, -94759096142232,
                                      30708340810940,  79086853262082,  78923696440892,
                                      -76316597208589, 37247268714759,  80303592631774,
                                      57790350050889,  19387319851064,  -33186875066145,
                                      69701203023404,  -7157433049060,  -7073790423437,
                                      92769171617714,  -75127120182184, -951893180618,
                                      64927618310150,  -53875897154023, -16168039035569,
                                      -24273449166429, -30359781249192, 35639397345991,
                                      45844829680593,  71401416837149,  0,
                                      -99999999999999, 99999999999999};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()), col1_data.size());
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      col1_data.begin(), col1_data.end(), validity, numeric::scale_type{-5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), col1);

    cudf::io::parquet_reader_options read_strict_opts = read_opts;
    read_strict_opts.set_column_names({"dec7p4", "dec14p5"});
    EXPECT_NO_THROW(cudf::io::read_parquet(read_strict_opts));
  }
  {
    // dec7p3: Decimal(precision=7, scale=3) backed by FIXED_LENGTH_BYTE_ARRAY(length = 4)
    // dec12p11: Decimal(precision=12, scale=11) backed by FIXED_LENGTH_BYTE_ARRAY(length = 6)
    // dec20p1: Decimal(precision=20, scale=1) backed by FIXED_LENGTH_BYTE_ARRAY(length = 9)
    std::array<unsigned char, 1226> const fixed_len_bytes_decimal_parquet{
      0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0xA8, 0x01, 0x15, 0xAE, 0x01, 0x2C, 0x15, 0x28,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x02, 0x28, 0x04, 0x00, 0x97, 0x45, 0x72,
      0x18, 0x04, 0x00, 0x01, 0x81, 0x3B, 0x00, 0x00, 0x00, 0x54, 0xF0, 0x53, 0x04, 0x00, 0x00,
      0x00, 0x26, 0x01, 0x03, 0x00, 0x00, 0x61, 0x10, 0xCF, 0x00, 0x0A, 0xA9, 0x08, 0x00, 0x77,
      0x58, 0x6F, 0x00, 0x6B, 0xEE, 0xA4, 0x00, 0x92, 0xF8, 0x94, 0x00, 0x2E, 0x18, 0xD4, 0x00,
      0x4F, 0x45, 0x33, 0x00, 0x97, 0x45, 0x72, 0x00, 0x0D, 0xC2, 0x75, 0x00, 0x76, 0xAA, 0xAA,
      0x00, 0x30, 0x9F, 0x86, 0x00, 0x4B, 0x9D, 0xB1, 0x00, 0x4E, 0x4B, 0x3B, 0x00, 0x01, 0x81,
      0x3B, 0x00, 0x22, 0xD4, 0x53, 0x00, 0x72, 0xC4, 0xAF, 0x00, 0x43, 0x9B, 0x72, 0x00, 0x1D,
      0x91, 0xC3, 0x00, 0x45, 0x27, 0x48, 0x15, 0x00, 0x15, 0xF4, 0x01, 0x15, 0xFA, 0x01, 0x2C,
      0x15, 0x28, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x02, 0x28, 0x06, 0x00, 0xD5,
      0xD7, 0x31, 0x99, 0xA6, 0x18, 0x06, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x00, 0x00,
      0x7A, 0xF0, 0x79, 0x04, 0x00, 0x00, 0x00, 0x24, 0x01, 0x03, 0x02, 0x00, 0x54, 0x23, 0xCF,
      0x13, 0x0A, 0x00, 0x07, 0x22, 0xB1, 0x21, 0x7E, 0x00, 0x64, 0x19, 0xD6, 0xD2, 0xA5, 0x00,
      0x61, 0x7F, 0xF6, 0xB9, 0xB0, 0x00, 0xD0, 0x7F, 0x9C, 0xA9, 0xE9, 0x00, 0x65, 0x58, 0xF0,
      0xAD, 0xFB, 0x00, 0xBC, 0x61, 0xE2, 0x03, 0xDA, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00,
      0x63, 0x4B, 0x4C, 0xFE, 0x45, 0x00, 0x7A, 0xA0, 0xD8, 0xD1, 0xC0, 0x00, 0xC0, 0x63, 0xF7,
      0x9D, 0x0A, 0x00, 0x88, 0x22, 0x0F, 0x1B, 0x25, 0x00, 0x1A, 0x80, 0x56, 0x34, 0xC7, 0x00,
      0x5F, 0x48, 0x61, 0x09, 0x7C, 0x00, 0x61, 0xEF, 0x92, 0x42, 0x2F, 0x00, 0xD5, 0xD7, 0x31,
      0x99, 0xA6, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x71, 0xDD, 0xE2, 0x22, 0x7B, 0x00,
      0x54, 0xBF, 0xAE, 0xE9, 0x3C, 0x15, 0x00, 0x15, 0xD4, 0x02, 0x15, 0xDC, 0x02, 0x2C, 0x15,
      0x28, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x04, 0x28, 0x09, 0x00, 0x7D, 0xFE,
      0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x18, 0x09, 0x00, 0x03, 0x9C, 0xCD, 0x5A, 0xAC, 0xBB,
      0xF1, 0xE3, 0x00, 0x00, 0x00, 0xAA, 0x01, 0xF0, 0xA9, 0x04, 0x00, 0x00, 0x00, 0x07, 0xBF,
      0xBF, 0x0F, 0x00, 0x7D, 0xFE, 0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x00, 0x7D, 0x9A, 0xCB,
      0xDA, 0x4B, 0x10, 0x8B, 0xAC, 0x00, 0x20, 0xBA, 0x97, 0x87, 0x2E, 0x3B, 0x4E, 0x04, 0x00,
      0x15, 0xBB, 0xC2, 0xDF, 0x2D, 0x25, 0x08, 0xB6, 0x00, 0x5C, 0x67, 0x0E, 0x36, 0x30, 0xF1,
      0xAC, 0xA4, 0x00, 0x44, 0xF1, 0x8E, 0xFB, 0x17, 0x5E, 0xE1, 0x96, 0x00, 0x64, 0x69, 0xF9,
      0x66, 0x3F, 0x11, 0xED, 0xB9, 0x00, 0x45, 0xB5, 0xDA, 0x14, 0x9C, 0xA3, 0xFA, 0x64, 0x00,
      0x26, 0x5F, 0xDE, 0xD7, 0x67, 0x95, 0xEF, 0xB1, 0x00, 0x35, 0xDB, 0x9B, 0x88, 0x46, 0xD0,
      0xA1, 0x0E, 0x00, 0x45, 0xA9, 0x92, 0x8E, 0x89, 0xD1, 0xAC, 0x4C, 0x00, 0x4C, 0xF1, 0xCB,
      0x27, 0x82, 0x3A, 0x7D, 0xB7, 0x00, 0x64, 0xD3, 0xD2, 0x2F, 0x9C, 0x83, 0x16, 0x75, 0x00,
      0x15, 0xDF, 0xC2, 0xA9, 0x63, 0xB8, 0x33, 0x65, 0x00, 0x27, 0x40, 0x28, 0x97, 0x05, 0x8E,
      0xE3, 0x46, 0x00, 0x03, 0x9C, 0xCD, 0x5A, 0xAC, 0xBB, 0xF1, 0xE3, 0x00, 0x22, 0x23, 0xF5,
      0xE8, 0x9D, 0x55, 0xD4, 0x9C, 0x00, 0x25, 0xB9, 0xD8, 0x87, 0x2D, 0xF1, 0xF2, 0x17, 0x15,
      0x02, 0x19, 0x4C, 0x48, 0x0C, 0x73, 0x70, 0x61, 0x72, 0x6B, 0x5F, 0x73, 0x63, 0x68, 0x65,
      0x6D, 0x61, 0x15, 0x06, 0x00, 0x15, 0x0E, 0x15, 0x08, 0x15, 0x02, 0x18, 0x06, 0x64, 0x65,
      0x63, 0x37, 0x70, 0x33, 0x25, 0x0A, 0x15, 0x06, 0x15, 0x0E, 0x00, 0x15, 0x0E, 0x15, 0x0C,
      0x15, 0x02, 0x18, 0x08, 0x64, 0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x25, 0x0A, 0x15,
      0x16, 0x15, 0x18, 0x00, 0x15, 0x0E, 0x15, 0x12, 0x15, 0x02, 0x18, 0x07, 0x64, 0x65, 0x63,
      0x32, 0x30, 0x70, 0x31, 0x25, 0x0A, 0x15, 0x02, 0x15, 0x28, 0x00, 0x16, 0x28, 0x19, 0x1C,
      0x19, 0x3C, 0x26, 0x08, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x06,
      0x64, 0x65, 0x63, 0x37, 0x70, 0x33, 0x15, 0x02, 0x16, 0x28, 0x16, 0xEE, 0x01, 0x16, 0xF4,
      0x01, 0x26, 0x08, 0x3C, 0x36, 0x02, 0x28, 0x04, 0x00, 0x97, 0x45, 0x72, 0x18, 0x04, 0x00,
      0x01, 0x81, 0x3B, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00,
      0x26, 0xFC, 0x01, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x08, 0x64,
      0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x15, 0x02, 0x16, 0x28, 0x16, 0xC2, 0x02, 0x16,
      0xC8, 0x02, 0x26, 0xFC, 0x01, 0x3C, 0x36, 0x02, 0x28, 0x06, 0x00, 0xD5, 0xD7, 0x31, 0x99,
      0xA6, 0x18, 0x06, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15,
      0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0xC4, 0x04, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06,
      0x08, 0x00, 0x19, 0x18, 0x07, 0x64, 0x65, 0x63, 0x32, 0x30, 0x70, 0x31, 0x15, 0x02, 0x16,
      0x28, 0x16, 0xAE, 0x03, 0x16, 0xB6, 0x03, 0x26, 0xC4, 0x04, 0x3C, 0x36, 0x04, 0x28, 0x09,
      0x00, 0x7D, 0xFE, 0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x18, 0x09, 0x00, 0x03, 0x9C, 0xCD,
      0x5A, 0xAC, 0xBB, 0xF1, 0xE3, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00,
      0x00, 0x00, 0x16, 0xDE, 0x07, 0x16, 0x28, 0x00, 0x19, 0x2C, 0x18, 0x18, 0x6F, 0x72, 0x67,
      0x2E, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2E, 0x73, 0x70, 0x61, 0x72, 0x6B, 0x2E, 0x76,
      0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x18, 0x05, 0x33, 0x2E, 0x30, 0x2E, 0x31, 0x00, 0x18,
      0x29, 0x6F, 0x72, 0x67, 0x2E, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2E, 0x73, 0x70, 0x61,
      0x72, 0x6B, 0x2E, 0x73, 0x71, 0x6C, 0x2E, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2E,
      0x72, 0x6F, 0x77, 0x2E, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0xF4, 0x01,
      0x7B, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x22, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74,
      0x22, 0x2C, 0x22, 0x66, 0x69, 0x65, 0x6C, 0x64, 0x73, 0x22, 0x3A, 0x5B, 0x7B, 0x22, 0x6E,
      0x61, 0x6D, 0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x37, 0x70, 0x33, 0x22, 0x2C, 0x22,
      0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28,
      0x37, 0x2C, 0x33, 0x29, 0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65,
      0x22, 0x3A, 0x74, 0x72, 0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74,
      0x61, 0x22, 0x3A, 0x7B, 0x7D, 0x7D, 0x2C, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A,
      0x22, 0x64, 0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x22, 0x2C, 0x22, 0x74, 0x79, 0x70,
      0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28, 0x31, 0x32, 0x2C,
      0x31, 0x31, 0x29, 0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65, 0x22,
      0x3A, 0x74, 0x72, 0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61,
      0x22, 0x3A, 0x7B, 0x7D, 0x7D, 0x2C, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x22,
      0x64, 0x65, 0x63, 0x32, 0x30, 0x70, 0x31, 0x22, 0x2C, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22,
      0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28, 0x32, 0x30, 0x2C, 0x31, 0x29,
      0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65, 0x22, 0x3A, 0x74, 0x72,
      0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3A, 0x7B,
      0x7D, 0x7D, 0x5D, 0x7D, 0x00, 0x18, 0x4A, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2D,
      0x6D, 0x72, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x20, 0x31, 0x2E, 0x31, 0x30,
      0x2E, 0x31, 0x20, 0x28, 0x62, 0x75, 0x69, 0x6C, 0x64, 0x20, 0x61, 0x38, 0x39, 0x64, 0x66,
      0x38, 0x66, 0x39, 0x39, 0x33, 0x32, 0x62, 0x36, 0x65, 0x66, 0x36, 0x36, 0x33, 0x33, 0x64,
      0x30, 0x36, 0x30, 0x36, 0x39, 0x65, 0x35, 0x30, 0x63, 0x39, 0x62, 0x37, 0x39, 0x37, 0x30,
      0x62, 0x65, 0x62, 0x64, 0x31, 0x29, 0x19, 0x3C, 0x1C, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x1C,
      0x00, 0x00, 0x00, 0xC5, 0x02, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31,
    };

    unsigned int parquet_len = 1226;

    cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(fixed_len_bytes_decimal_parquet.data()), parquet_len}});
    auto result = cudf::io::read_parquet(read_opts);
    EXPECT_EQ(result.tbl->view().num_columns(), 3);

    auto validity_c0 = cudf::test::iterators::nulls_at({19});
    std::array col0_data{6361295, 698632,  7821423, 7073444, 9631892, 3021012, 5195059,
                         9913714, 901749,  7776938, 3186566, 4955569, 5131067, 98619,
                         2282579, 7521455, 4430706, 1937859, 4532040, 0};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(0).size()), col0_data.size());
    cudf::test::fixed_point_column_wrapper<int32_t> col0(
      col0_data.begin(), col0_data.end(), validity_c0, numeric::scale_type{-3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);

    auto validity_c1 = cudf::test::iterators::nulls_at({18});
    std::array<int64_t, 20> col1_data{361378026250,
                                      30646804862,
                                      429930238629,
                                      418758703536,
                                      895494171113,
                                      435283865083,
                                      809096053722,
                                      -999999999999,
                                      426465099333,
                                      526684574144,
                                      826310892810,
                                      584686967589,
                                      113822282951,
                                      409236212092,
                                      420631167535,
                                      918438386086,
                                      -999999999999,
                                      489053889147,
                                      0,
                                      363993164092};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()), col1_data.size());
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      col1_data.begin(), col1_data.end(), validity_c1, numeric::scale_type{-11});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), col1);

    auto validity_c2 = cudf::test::iterators::nulls_at({6, 14});
    std::array<__int128_t, 20> col2_data{9078697037144433659,
                                         9050770539577117612,
                                         2358363961733893636,
                                         1566059559232276662,
                                         6658306200002735268,
                                         4967909073046397334,
                                         0,
                                         7235588493887532473,
                                         5023160741463849572,
                                         2765173712965988273,
                                         3880866513515749646,
                                         5019704400576359500,
                                         5544435986818825655,
                                         7265381725809874549,
                                         0,
                                         1576192427381240677,
                                         2828305195087094598,
                                         260308667809395171,
                                         2460080200895288476,
                                         2718441925197820439};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(2).size()), col2_data.size());
    cudf::test::fixed_point_column_wrapper<__int128_t> col2(
      col2_data.begin(), col2_data.end(), validity_c2, numeric::scale_type{-1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), col2);
  }
}

TEST_F(ParquetReaderTest, EmptyOutput)
{
  cudf::test::fixed_width_column_wrapper<int> c0;
  cudf::test::strings_column_wrapper c1;
  cudf::test::fixed_point_column_wrapper<int> c2({}, numeric::scale_type{2});
  cudf::test::lists_column_wrapper<float> _c3{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  auto c3 = cudf::empty_like(_c3);

  cudf::test::fixed_width_column_wrapper<int> sc0;
  cudf::test::strings_column_wrapper sc1;
  cudf::test::lists_column_wrapper<int> _sc2{{1, 2}};
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(sc0.release());
  struct_children.push_back(sc1.release());
  struct_children.push_back(cudf::empty_like(_sc2));
  cudf::test::structs_column_wrapper c4(std::move(struct_children));

  table_view expected({c0, c1, c2, *c3, c4});

  // set precision on the decimal column
  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[2].set_decimal_precision(1);

  auto filepath = temp_env->get_temp_filepath("EmptyOutput.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  out_args.set_metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, EmptyColumnsParam)
{
  srand(31337);
  std::vector<char> out_buffer;
  {
    auto const table = create_random_fixed_table<int>(2, 4, false);
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer}, *table);
    cudf::io::write_parquet(args);
  }

  {
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
        .column_names({});
    auto const result = cudf::io::read_parquet(read_opts);

    EXPECT_EQ(result.tbl->num_columns(), 0);
    EXPECT_EQ(result.tbl->num_rows(), 0);
  }

  {
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info{cudf::host_span<std::byte const>{
          reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
        .column_indices({});
    auto const result = cudf::io::read_parquet(read_opts);

    EXPECT_EQ(result.tbl->num_columns(), 0);
    EXPECT_EQ(result.tbl->num_rows(), 0);
  }
}

TEST_F(ParquetReaderTest, BinaryAsStrings)
{
  std::vector<char const*> strings{
    "Monday", "Wednesday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto seq_col3 = random_values<uint8_t>(num_rows);
  auto validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> string_col{strings.begin(), strings.end()};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_int_col{
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'u', 'n', 'd', 'a', 'y'}};

  auto output = table_view{{int_col, string_col, float_col, string_col, list_int_col}};
  cudf::io::table_input_metadata output_metadata(output);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_string");
  output_metadata.column_metadata[2].set_name("col_float");
  output_metadata.column_metadata[3].set_name("col_string2").set_output_as_binary(true);
  output_metadata.column_metadata[4].set_name("col_binary").set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("BinaryReadStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, output)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto expected_string = table_view{{int_col, string_col, float_col, string_col, string_col}};
  auto expected_mixed  = table_view{{int_col, string_col, float_col, list_int_col, list_int_col}};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema({{}, {}, {}, {}, {}});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    {},
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false)};

  cudf::io::parquet_reader_options mixed_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  result = cudf::io::read_parquet(mixed_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_mixed, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestedByteArray)
{
  constexpr auto num_rows = 8;

  auto seq_col0       = random_values<int>(num_rows);
  auto seq_col2       = random_values<float>(num_rows);
  auto seq_col3       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_list_int_col{
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}}};

  auto const expected = table_view{{int_col, float_col, list_list_int_col}};
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_float");
  output_metadata.column_metadata[2].set_name("col_binary").child(1).set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("NestedByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  EXPECT_EQ(fmd.schema[5].type, cudf::io::parquet::Type::BYTE_ARRAY);

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    cudf::io::reader_column_schema().add_child(
      cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, StructByteArray)
{
  constexpr auto num_rows = 100;

  auto seq_col0       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<uint8_t> int_col{seq_col0.begin(), seq_col0.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_of_int{{seq_col0.begin(), seq_col0.begin() + 50},
                                                        {seq_col0.begin() + 50, seq_col0.end()}};
  auto struct_col = cudf::test::structs_column_wrapper{{list_of_int}, validity};

  auto const expected = table_view{{struct_col}};
  EXPECT_EQ(1, expected.num_columns());
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0]
    .set_name("struct_binary")
    .child(0)
    .set_name("a")
    .set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("StructByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  std::vector<cudf::io::reader_column_schema> md{cudf::io::reader_column_schema().add_child(
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestingOptimizationTest)
{
  // test nesting levels > cudf::io::parquet::detail::max_cacheable_nesting_decode_info deep.
  constexpr cudf::size_type num_nesting_levels = 16;
  static_assert(num_nesting_levels > cudf::io::parquet::detail::max_cacheable_nesting_decode_info);
  constexpr cudf::size_type rows_per_level = 2;

  constexpr cudf::size_type num_values = (1 << num_nesting_levels) * rows_per_level;
  auto value_iter                      = cuda::counting_iterator<int>{0};
  auto validity                        = cudf::test::iterators::nulls_at_multiples_of(2);
  cudf::test::fixed_width_column_wrapper<int> values(value_iter, value_iter + num_values, validity);

  // ~256k values with num_nesting_levels = 16
  auto prev_col = values.release();
  for (int idx = 0; idx < num_nesting_levels; idx++) {
    auto const num_rows = (1 << (num_nesting_levels - idx));

    auto offsets_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i * rows_per_level; });

    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offsets_iter,
                                                                    offsets_iter + num_rows + 1);
    auto c   = cudf::make_lists_column(num_rows, offsets.release(), std::move(prev_col), 0, {});
    prev_col = std::move(c);
  }
  auto const& expect = prev_col;

  auto filepath = temp_env->get_temp_filepath("NestingDecodeCache.parquet");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table_view{{*expect}});
  cudf::io::write_parquet(opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expect, result.tbl->get_column(0));
}

TEST_F(ParquetReaderTest, SingleLevelLists)
{
  std::array<unsigned char, 214> list_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  // read single level list reproducing parquet file
  cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(list_bytes.data()), list_bytes.size()}});
  auto table = cudf::io::read_parquet(read_opts);

  auto const c0 = table.tbl->get_column(0);
  EXPECT_TRUE(c0.type().id() == cudf::type_id::LIST);

  auto const lc    = cudf::lists_column_view(c0);
  auto const child = lc.child();
  EXPECT_TRUE(child.type().id() == cudf::type_id::INT32);
}

TEST_F(ParquetReaderTest, ChunkedSingleLevelLists)
{
  std::array<unsigned char, 214> list_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto reader = cudf::io::chunked_parquet_reader(
    1L << 31,
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(list_bytes.data()), list_bytes.size()}}));
  int iterations = 0;
  while (reader.has_next() && iterations < 10) {
    auto chunk = reader.read_chunk();
  }
  EXPECT_TRUE(iterations < 10);
}

TEST_F(ParquetReaderTest, ReorderedReadMultipleFiles)
{
  constexpr auto num_rows    = 50'000;
  constexpr auto cardinality = 20'000;

  // table 1
  auto str1 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "cat " + std::to_string(i % cardinality); });
  auto cols1 = cudf::test::strings_column_wrapper(str1, str1 + num_rows);

  auto int1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % cardinality; });
  auto coli1 = cudf::test::fixed_width_column_wrapper<int>(int1, int1 + num_rows);

  auto const expected1 = table_view{{cols1, coli1}};
  auto const swapped1  = table_view{{coli1, cols1}};

  auto const filepath1 = temp_env->get_temp_filepath("LargeReorderedRead1.parquet");
  auto out_opts1 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath1}, expected1)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts1);

  // table 2
  auto str2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "dog " + std::to_string(i % cardinality); });
  auto cols2 = cudf::test::strings_column_wrapper(str2, str2 + num_rows);

  auto int2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return (i % cardinality) + cardinality; });
  auto coli2 = cudf::test::fixed_width_column_wrapper<int>(int2, int2 + num_rows);

  auto const expected2 = table_view{{cols2, coli2}};
  auto const swapped2  = table_view{{coli2, cols2}};

  auto const filepath2 = temp_env->get_temp_filepath("LargeReorderedRead2.parquet");
  auto out_opts2 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath2}, expected2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts2);

  // read in both files swapping the columns
  {
    auto read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{filepath1, filepath2}})
        .column_names({"_col1", "_col0"});
    auto result = cudf::io::read_parquet(read_opts);
    auto sliced = cudf::slice(result.tbl->view(), {0, num_rows, num_rows, 2 * num_rows});
    CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[0], swapped1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[1], swapped2);
  }

  // read in both files swapping the columns using indices
  {
    auto read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{filepath1, filepath2}})
        .column_indices({1, 0});
    auto result = cudf::io::read_parquet(read_opts);
    auto sliced = cudf::slice(result.tbl->view(), {0, num_rows, num_rows, 2 * num_rows});
    CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[0], swapped1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[1], swapped2);
  }
}

TEST_F(ParquetReaderTest, NoFilter)
{
  srand(31337);
  auto expected = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("FilterSimple.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
  EXPECT_EQ(result.metadata.num_input_row_groups, 1);
  EXPECT_FALSE(result.metadata.num_row_groups_after_stats_filter.has_value());
  EXPECT_FALSE(result.metadata.num_row_groups_after_bloom_filter.has_value());
}

TEST_F(ParquetReaderTest, FilterSimple)
{
  srand(31337);
  auto written_table = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("FilterSimple.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *written_table);
  cudf::io::write_parquet(args);

  // Filtering AST -  RAND_MAX/2 > table[0]
  auto literal_value = cudf::numeric_scalar<decltype(RAND_MAX)>(RAND_MAX / 2);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref_0     = cudf::ast::column_reference(0);
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal, col_ref_0);

  auto predicate = cudf::compute_column(*written_table, filter_expression);
  EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
    << "Predicate filter should return a boolean";
  auto expected = cudf::apply_boolean_mask(*written_table, *predicate);
  // To make sure AST filters out some elements
  EXPECT_LT(expected->num_rows(), written_table->num_rows());

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

auto create_parquet_with_stats(std::string const& filename)
{
  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = testdata::descending<int64_t>();
  auto col2 = testdata::unordered<double>();

  auto const expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_uint32");
  expected_metadata.column_metadata[0].set_parquet_field_id(10);
  expected_metadata.column_metadata[1].set_name("col_int64");
  expected_metadata.column_metadata[1].set_parquet_field_id(11);
  expected_metadata.column_metadata[2].set_name("col_double");
  expected_metadata.column_metadata[2].set_parquet_field_id(12);

  auto const filepath = temp_env->get_temp_filepath(filename);
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata))
      .row_group_size_rows(8000)
      .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(out_opts);

  std::vector<std::unique_ptr<column>> columns;
  columns.push_back(col0.release());
  columns.push_back(col1.release());
  columns.push_back(col2.release());

  return std::pair{cudf::table{std::move(columns)}, filepath};
}

TEST_F(ParquetReaderTest, FilterIdentity)
{
  auto [src, filepath] = create_parquet_with_stats("FilterIdentity.parquet");

  // Filtering AST - identity function, always true.
  auto literal_value     = cudf::numeric_scalar<bool>(true);
  auto literal           = cudf::ast::literal(literal_value);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  cudf::io::parquet_reader_options read_opts2 =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result2 = cudf::io::read_parquet(read_opts2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *result2.tbl);
}

TEST_F(ParquetReaderTest, FilterWithColumnProjection)
{
  // col_uint32 (field_id: 10), col_int64 (field_id: 11), col_double (field_id: 12)
  auto [src, filepath] = create_parquet_with_stats("FilterWithColumnProjection.parquet");
  auto val             = cudf::numeric_scalar<uint32_t>{10};
  auto lit             = cudf::ast::literal{val};

  // Prepare the predicate
  auto col_index   = cudf::ast::column_reference{0};
  auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER, lit, col_index);
  auto predicate   = cudf::compute_column(src, filter_expr);

  {  // column_name_reference in parquet filter (not present in column projection)
    auto col_ref         = cudf::ast::column_name_reference{"cOL_uint32"};
    auto read_expr       = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, lit);
    auto projected_table = cudf::table_view{{src.get_column(2)}};
    auto expected        = cudf::apply_boolean_mask(projected_table, *predicate);

    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .column_names({"col_Double"})
                       .case_sensitive_names(false)
                       .prepend_source_index_column(true)
                       .filter(read_expr);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view().select({1}), *expected);

    // Repeat but select columns using indices instead of names
    read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                  .column_indices({2})
                  .case_sensitive_names(false)
                  .filter(read_expr);
    result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);

    // Repeat but select columns using field IDs instead of names
    read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                  .column_field_ids({12})
                  .case_sensitive_names(false)
                  .filter(read_expr);
    result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
  }

  {  // column_reference in parquet filter (indices as per order of column projection)
    auto col_index2    = cudf::ast::column_reference{1};
    auto read_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index2, lit);

    auto projected_table = cudf::table_view{{src.get_column(2), src.get_column(0)}};
    auto expected        = cudf::apply_boolean_mask(projected_table, *predicate);
    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .column_names({"col_Double", "col_UInt32"})
                       .case_sensitive_names(false)
                       .filter(read_ref_expr);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*(cudf::io::read_parquet(read_opts).tbl), *expected);

    // Repeat but select columns using indices instead of names
    read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                  .column_indices({2, 0})
                  .filter(read_ref_expr);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*(cudf::io::read_parquet(read_opts).tbl), *expected);

    // Repeat but select columns using field IDs instead of names
    read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                  .column_field_ids({12, 10})
                  .filter(read_ref_expr);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*(cudf::io::read_parquet(read_opts).tbl), *expected);
  }

  // Error cases
  {  // index == 0: column_reference is not same type as literal;
     // index == 2: column_reference index is out of bounds
    for (auto const index : {0, 2}) {
      auto col_index2    = cudf::ast::column_reference{index};
      auto read_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index2, lit);
      auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                         .column_names({"col_double", "col_uint32"})
                         .filter(read_ref_expr);
      EXPECT_ANY_THROW(cudf::io::read_parquet(read_opts));

      // Repeat but select columns using indices instead of names
      read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                    .column_indices({2, 0})
                    .filter(read_ref_expr);
      EXPECT_ANY_THROW(cudf::io::read_parquet(read_opts));

      // Repeat but select columns using field IDs instead of names
      read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                    .column_field_ids({12, 10})
                    .filter(read_ref_expr);
      EXPECT_ANY_THROW(cudf::io::read_parquet(read_opts));
    }
  }
}

TEST_F(ParquetReaderTest, FilterReferenceExpression)
{
  auto [src, filepath] = create_parquet_with_stats("FilterReferenceExpression.parquet");
  // Filtering AST - 150 > table[0]
  auto literal_value = cudf::numeric_scalar<uint32_t>(150);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref_0     = cudf::ast::column_reference(0);
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal, col_ref_0);

  // Expected result
  auto predicate = cudf::compute_column(src, filter_expression);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetReaderTest, ExtendedFilterExpressions)
{
  // Create a parquet file with 3 int32 columns for col-to-col comparison testing
  auto constexpr num_rows = 20'000;
  auto col_a = cudf::test::fixed_width_column_wrapper<int32_t>(cuda::counting_iterator<int>{0},
                                                               cuda::counting_iterator{num_rows});
  auto col_b = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator{num_rows}, cuda::counting_iterator{2 * num_rows});
  auto col_c = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator{2 * num_rows}, cuda::counting_iterator{3 * num_rows});

  auto const written_table = cudf::table_view{{col_a, col_b, col_c}};

  auto const filepath = temp_env->get_temp_filepath("ExtendedFilterExpressions.parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
      .row_group_size_rows(5'000)
      .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(out_opts);

  auto col_ref_a = cudf::ast::column_reference(0);
  auto col_ref_b = cudf::ast::column_reference(1);

  // Filter: col_a < col_b
  {
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, col_ref_b);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter cannot prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              result.metadata.num_input_row_groups);
  }

  // Filter: (col_a < 150) and (col_a < col_b)
  {
    auto literal_150_value = cudf::numeric_scalar<int32_t>(150);
    auto literal_150       = cudf::ast::literal(literal_150_value);
    auto lhs    = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, literal_150);
    auto rhs    = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, col_ref_b);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, lhs, rhs);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter prunes 3 row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(), 1);
  }

  // Filter: (col_a < 150) or (col_a < col_b)
  {
    auto literal_150_value = cudf::numeric_scalar<int32_t>(150);
    auto literal_150       = cudf::ast::literal(literal_150_value);
    auto lhs    = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, literal_150);
    auto rhs    = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, col_ref_b);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, lhs, rhs);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter cannot prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              result.metadata.num_input_row_groups);
  }

  // Filter: (col_a < 10) or ((col_a + col_b < col_c) and (col_b < 1))
  {
    auto literal_10_value = cudf::numeric_scalar<int32_t>(10);
    auto literal_10       = cudf::ast::literal(literal_10_value);
    auto literal_1_value  = cudf::numeric_scalar<int32_t>(1);
    auto literal_1        = cudf::ast::literal(literal_1_value);

    auto col_ref_c = cudf::ast::column_reference(2);

    auto a_lt_10       = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, literal_10);
    auto a_plus_b      = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_a, col_ref_b);
    auto a_plus_b_lt_c = cudf::ast::operation(cudf::ast::ast_operator::LESS, a_plus_b, col_ref_c);
    auto b_lt_1        = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_b, literal_1);
    auto and_expr =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, a_plus_b_lt_c, b_lt_1);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, a_lt_10, and_expr);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter can prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(), 1);
  }

  // Filter: false AND (50 > col_a)
  {
    auto literal_50_value = cudf::numeric_scalar<int32_t>(50);
    auto literal_50       = cudf::ast::literal(literal_50_value);
    auto literal_0_value  = cudf::numeric_scalar<bool>(false);
    auto literal_0        = cudf::ast::literal(literal_0_value);
    auto a_lt_50 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal_50, col_ref_a);
    auto filter  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, literal_0, a_lt_50);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter cannot prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              result.metadata.num_input_row_groups);
  }

  // Filter: NOT(col_a NULL_EQUAL 10)
  {
    auto literal_10_value = cudf::numeric_scalar<int32_t>(10);
    auto literal_10       = cudf::ast::literal(literal_10_value);
    auto inner  = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_a, literal_10);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter cannot prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              result.metadata.num_input_row_groups);
  }

  // Filter: col_a NULL_EQUAL 10
  {
    auto literal_10_value = cudf::numeric_scalar<int32_t>(10);
    auto literal_10       = cudf::ast::literal(literal_10_value);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_a, literal_10);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter cannot prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              result.metadata.num_input_row_groups);
  }

  // Filter: NOT(col_a NULL_EQUAL 10) AND (col_a < 50)
  {
    auto literal_50_value = cudf::numeric_scalar<int32_t>(50);
    auto literal_50       = cudf::ast::literal(literal_50_value);
    auto literal_10_value = cudf::numeric_scalar<int32_t>(10);
    auto literal_10       = cudf::ast::literal(literal_10_value);
    auto null_eq = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_a, literal_10);
    auto lhs     = cudf::ast::operation(cudf::ast::ast_operator::NOT, null_eq);
    auto rhs     = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, literal_50);
    auto filter  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, lhs, rhs);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter can prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(), 1);
  }

  // Filter: NOT(col_a < col_b)
  {
    auto inner  = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_a, col_ref_b);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);

    auto predicate = cudf::compute_column(written_table, filter);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
    // Stats filter cannot prune row groups
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              result.metadata.num_input_row_groups);
  }
}

TEST_F(ParquetReaderTest, FilterNamedExpression)
{
  auto [src, filepath] = create_parquet_with_stats("NamedExpression.parquet");
  // Filtering AST - table["col_uint32"] < 150
  auto literal_value  = cudf::numeric_scalar<uint32_t>(150);
  auto literal        = cudf::ast::literal(literal_value);
  auto col_name_0     = cudf::ast::column_name_reference("col_uint32");
  auto parquet_filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
  auto col_ref_0      = cudf::ast::column_reference(0);
  auto table_filter   = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(src, table_filter);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(parquet_filter);
  auto result = cudf::io::read_parquet(read_opts);

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetReaderTest, FilterMultiple1)
{
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple1.parquet");
  auto const written_table   = src.view();

  // Filtering AST - 10000 < table[0] < 12000
  std::string const low  = "000010000";
  std::string const high = "000012000";
  auto lov               = cudf::string_scalar(low, true);
  auto hiv               = cudf::string_scalar(high, true);
  auto filter_col        = cudf::ast::column_reference(0);
  auto lo_lit            = cudf::ast::literal(lov);
  auto hi_lit            = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, hi_lit, filter_col);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_3);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_3);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple2)
{
  // multiple conditions on same column.
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple2.parquet");
  auto const written_table   = src.view();
  // 0-8000, 8001-16000, 16001-20000

  // Filtering AST
  // (table[0] >= "000010000" AND table[0] < "000012000") OR
  // (table[0] >= "000017000" AND table[0] < "000019000")
  std::string const low1  = "000010000";
  std::string const high1 = "000012000";
  auto lov                = cudf::string_scalar(low1, true);
  auto hiv                = cudf::string_scalar(high1, true);
  auto filter_col         = cudf::ast::column_reference(0);
  auto lo_lit             = cudf::ast::literal(lov);
  auto hi_lit             = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, hi_lit, filter_col);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  std::string const low2  = "000017000";
  std::string const high2 = "000019000";
  auto lov2               = cudf::string_scalar(low2, true);
  auto hiv2               = cudf::string_scalar(high2, true);
  auto lo_lit2            = cudf::ast::literal(lov2);
  auto hi_lit2            = cudf::ast::literal(hiv2);
  auto expr_4 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit2);
  auto expr_5 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit2);
  auto expr_6 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_7);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple3)
{
  // multiple conditions with reference to multiple columns.
  // index and name references mixed.
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple3.parquet");
  auto const written_table   = src.view();

  // Filtering AST - (table[0] >= 70 AND table[0] < 90) OR (table[1] >= 100 AND table[1] < 120)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1  = cudf::ast::column_reference(0);
  auto filter_col2  = cudf::ast::column_name_reference("col1");
  T constexpr low1  = 70;
  T constexpr high1 = 90;
  T constexpr low2  = 100;
  T constexpr high2 = 120;
  auto lov          = cudf::numeric_scalar(low1, true);
  auto hiv          = cudf::numeric_scalar(high1, true);
  auto lo_lit1      = cudf::ast::literal(lov);
  auto hi_lit1      = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  // expression to test
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto filter_col2_ref = cudf::ast::column_reference(1);
  auto expr_4_ref =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2_ref, lo_lit2);
  auto expr_5_ref = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2_ref, hi_lit2);
  auto expr_6_ref =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4_ref, expr_5_ref);
  auto expr_7_ref = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6_ref);
  auto predicate  = cudf::compute_column(written_table, expr_7_ref);
  auto expected   = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterSupported.parquet");
  auto const written_table   = src.view();

  // Filtering AST - ((table[0] > 70 AND table[0] <= 90) OR (table[1] >= 100 AND table[1] < 120))
  //              AND (table[1] != 110)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1       = cudf::ast::column_reference(0);
  auto filter_col2       = cudf::ast::column_reference(1);
  T constexpr low1       = 70;
  T constexpr high1      = 90;
  T constexpr low2       = 100;
  T constexpr high2      = 120;
  T constexpr skip_value = 110;
  auto lov               = cudf::numeric_scalar(low1, true);
  auto hiv               = cudf::numeric_scalar(high1, true);
  auto lo_lit1           = cudf::ast::literal(lov);
  auto hi_lit1           = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);
  auto skip_ov = cudf::numeric_scalar(skip_value, true);
  auto skip_lit = cudf::ast::literal(skip_ov);
  auto expr_8   = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, skip_lit);
  auto expr_9   = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_7, expr_8);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_9);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_9);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported2)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 4000;
  auto elements0 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 2000; });
  auto elements1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto elements2 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 500; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements0, elements0 + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<T>(elements1, elements1 + num_rows);
  auto col2 = cudf::test::fixed_width_column_wrapper<T>(elements2, elements2 + num_rows);
  auto const written_table = table_view{{col0, col1, col2}};
  auto const filepath      = temp_env->get_temp_filepath("FilterSupported2.parquet");
  {
    cudf::io::parquet_writer_options const out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(1000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto filter_col2 = cudf::ast::column_reference(2);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  auto test_expr = [&](auto& expr) {
    // Expected result
    auto predicate = cudf::compute_column(written_table, expr);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    // tests
    auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
    auto table_with_metadata = cudf::io::read_parquet(builder);
    auto result              = table_with_metadata.tbl->view();

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
  };

  // row groups min, max:
  // table[0] 0-0, 0-0, 1-1, 1-1
  // table[1] 0-0, 1-1, 2-2, 3-3
  // table[2] 0-1, 2-3, 4-5, 6-7

  // Filtering AST -   table[i] == 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, lit_value, filter_col0);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, lit_value, filter_col1);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
  // Filtering AST -   table[i] != 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col0, lit_value);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit_value);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
}

// Error types - type mismatch, invalid column name, invalid literal type, invalid operator,
// non-bool filter output type.
TEST_F(ParquetReaderTest, FilterErrors)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterErrors.parquet");
  auto const written_table   = src.view();
  auto si                    = cudf::io::source_info(filepath);

  // Filtering AST - invalid column index
  {
    auto filter_col1 = cudf::ast::column_reference(3);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
  }

  // Filtering AST - invalid column name
  {
    auto filter_col1 = cudf::ast::column_name_reference("col3");
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto builder     = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), std::invalid_argument);
  }

  // Filtering AST - incompatible literal type
  {
    auto filter_col1      = cudf::ast::column_name_reference("col0");
    auto filter_col2      = cudf::ast::column_reference(1);
    int64_t constexpr low = 100;
    auto lov              = cudf::numeric_scalar(low, true);
    auto low_lot          = cudf::ast::literal(lov);
    auto expr1    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto expr2    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, low_lot);
    auto builder1 = cudf::io::parquet_reader_options::builder(si).filter(expr1);
    EXPECT_THROW(cudf::io::read_parquet(builder1), cudf::logic_error);

    auto builder2 = cudf::io::parquet_reader_options::builder(si).filter(expr2);
    EXPECT_THROW(cudf::io::read_parquet(builder2), cudf::logic_error);
  }

  // Filtering AST - "table[0] + 110" is invalid filter expression
  {
    auto filter_col1      = cudf::ast::column_reference(0);
    T constexpr add_value = 110;
    auto add_v            = cudf::numeric_scalar(add_value, true);
    auto add_lit          = cudf::ast::literal(add_v);
    auto expr_8 = cudf::ast::operation(cudf::ast::ast_operator::ADD, filter_col1, add_lit);

    auto si      = cudf::io::source_info(filepath);
    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr_8);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);

    // Expected result throw to show that the filter expression is invalid,
    // not a limitation of the parquet predicate pushdown.
    auto predicate = cudf::compute_column(written_table, expr_8);
    EXPECT_THROW(cudf::apply_boolean_mask(written_table, *predicate), cudf::logic_error);
  }

  // Filtering AST - INT64(table[0] < 100) non-bool expression
  {
    auto filter_col1 = cudf::ast::column_reference(0);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto bool_expr   = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto cast        = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, bool_expr);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(cast);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
    EXPECT_NO_THROW(cudf::compute_column(written_table, cast));
    auto predicate = cudf::compute_column(written_table, cast);
    EXPECT_NE(predicate->view().type().id(), cudf::type_id::BOOL8);
  }
}

// Filter without stats information in file.
TEST_F(ParquetReaderTest, FilterNoStats)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 16000;
  auto elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows);
  auto const written_table = table_view{{col0}};
  auto const filepath      = temp_env->get_temp_filepath("FilterNoStats.parquet");
  {
    cudf::io::parquet_writer_options const out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000)
        .stats_level(cudf::io::statistics_freq::STATISTICS_NONE);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] > 1
  auto expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col0, lit_value);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  // tests
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

// Filter for float column with NaN values
TEST_F(ParquetReaderTest, FilterFloatNAN)
{
  constexpr auto num_rows = 24000;
  auto elements           = cudf::detail::make_counting_transform_iterator(
    0, [num_rows](auto i) { return i > num_rows / 2 ? NAN : i; });
  auto col0 = cudf::test::fixed_width_column_wrapper<float>(elements, elements + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<double>(elements, elements + num_rows);

  auto const written_table = table_view{{col0, col1}};
  auto const filepath      = temp_env->get_temp_filepath("FilterFloatNAN.parquet");
  {
    cudf::io::parquet_writer_options const out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto s0_value    = cudf::numeric_scalar<float>(NAN, true);
  auto lit0_value  = cudf::ast::literal(s0_value);
  auto s1_value    = cudf::numeric_scalar<double>(NAN, true);
  auto lit1_value  = cudf::ast::literal(s1_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] == NAN, table[1] != NAN
  auto expr_eq  = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col0, lit0_value);
  auto expr_neq = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit1_value);

  // Expected result
  auto predicate0 = cudf::compute_column(written_table, expr_eq);
  auto expected0  = cudf::apply_boolean_mask(written_table, *predicate0);
  auto predicate1 = cudf::compute_column(written_table, expr_neq);
  auto expected1  = cudf::apply_boolean_mask(written_table, *predicate1);

  // tests
  auto builder0             = cudf::io::parquet_reader_options::builder(si).filter(expr_eq);
  auto table_with_metadata0 = cudf::io::read_parquet(builder0);
  auto result0              = table_with_metadata0.tbl->view();
  auto builder1             = cudf::io::parquet_reader_options::builder(si).filter(expr_neq);
  auto table_with_metadata1 = cudf::io::read_parquet(builder1);
  auto result1              = table_with_metadata1.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected0->view(), result0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1->view(), result1);
}

TEST_F(ParquetReaderTest, RepeatedNoAnnotations)
{
  constexpr std::array<unsigned char, 662> repeated_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x04, 0x15, 0x30, 0x15, 0x30, 0x4c, 0x15, 0x0c, 0x15, 0x00, 0x12,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x0a, 0x15, 0x0a,
    0x2c, 0x15, 0x0c, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x03, 0x03, 0x88, 0xc6, 0x02,
    0x26, 0x80, 0x01, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x10, 0x19, 0x18, 0x02, 0x69, 0x64, 0x15,
    0x00, 0x16, 0x0c, 0x16, 0x78, 0x16, 0x78, 0x26, 0x54, 0x26, 0x08, 0x00, 0x00, 0x15, 0x04, 0x15,
    0x40, 0x15, 0x40, 0x4c, 0x15, 0x08, 0x15, 0x00, 0x12, 0x00, 0x00, 0xe3, 0x0c, 0x23, 0x4b, 0x01,
    0x00, 0x00, 0x00, 0xc7, 0x35, 0x3a, 0x42, 0x00, 0x00, 0x00, 0x00, 0x8e, 0x6b, 0x74, 0x84, 0x00,
    0x00, 0x00, 0x00, 0x55, 0xa1, 0xae, 0xc6, 0x00, 0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x22, 0x15,
    0x22, 0x2c, 0x15, 0x10, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x03, 0xc0, 0x03, 0x00, 0x00, 0x00, 0x03, 0x90, 0xaa, 0x02, 0x03, 0x94, 0x03, 0x26, 0xda, 0x02,
    0x1c, 0x15, 0x04, 0x19, 0x25, 0x00, 0x10, 0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e,
    0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x06, 0x6e, 0x75, 0x6d,
    0x62, 0x65, 0x72, 0x15, 0x00, 0x16, 0x10, 0x16, 0xa0, 0x01, 0x16, 0xa0, 0x01, 0x26, 0x96, 0x02,
    0x26, 0xba, 0x01, 0x00, 0x00, 0x15, 0x04, 0x15, 0x24, 0x15, 0x24, 0x4c, 0x15, 0x04, 0x15, 0x00,
    0x12, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x68, 0x6f, 0x6d, 0x65, 0x06, 0x00, 0x00, 0x00, 0x6d,
    0x6f, 0x62, 0x69, 0x6c, 0x65, 0x15, 0x00, 0x15, 0x20, 0x15, 0x20, 0x2c, 0x15, 0x10, 0x15, 0x10,
    0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0xc0, 0x03, 0x00, 0x00, 0x00,
    0x03, 0x90, 0xef, 0x01, 0x03, 0x04, 0x26, 0xcc, 0x04, 0x1c, 0x15, 0x0c, 0x19, 0x25, 0x00, 0x10,
    0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05,
    0x70, 0x68, 0x6f, 0x6e, 0x65, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x15, 0x00, 0x16, 0x10, 0x16, 0x82,
    0x01, 0x16, 0x82, 0x01, 0x26, 0x8a, 0x04, 0x26, 0xca, 0x03, 0x00, 0x00, 0x15, 0x02, 0x19, 0x6c,
    0x48, 0x04, 0x75, 0x73, 0x65, 0x72, 0x15, 0x04, 0x00, 0x15, 0x02, 0x25, 0x00, 0x18, 0x02, 0x69,
    0x64, 0x00, 0x35, 0x02, 0x18, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65,
    0x72, 0x73, 0x15, 0x02, 0x00, 0x35, 0x04, 0x18, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x15, 0x04,
    0x00, 0x15, 0x04, 0x25, 0x00, 0x18, 0x06, 0x6e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x00, 0x15, 0x0c,
    0x25, 0x02, 0x18, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x25, 0x00, 0x00, 0x16, 0x00, 0x19, 0x1c, 0x19,
    0x3c, 0x26, 0x80, 0x01, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x10, 0x19, 0x18, 0x02, 0x69, 0x64,
    0x15, 0x00, 0x16, 0x0c, 0x16, 0x78, 0x16, 0x78, 0x26, 0x54, 0x26, 0x08, 0x00, 0x00, 0x26, 0xda,
    0x02, 0x1c, 0x15, 0x04, 0x19, 0x25, 0x00, 0x10, 0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65,
    0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x06, 0x6e, 0x75,
    0x6d, 0x62, 0x65, 0x72, 0x15, 0x00, 0x16, 0x10, 0x16, 0xa0, 0x01, 0x16, 0xa0, 0x01, 0x26, 0x96,
    0x02, 0x26, 0xba, 0x01, 0x00, 0x00, 0x26, 0xcc, 0x04, 0x1c, 0x15, 0x0c, 0x19, 0x25, 0x00, 0x10,
    0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05,
    0x70, 0x68, 0x6f, 0x6e, 0x65, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x15, 0x00, 0x16, 0x10, 0x16, 0x82,
    0x01, 0x16, 0x82, 0x01, 0x26, 0x8a, 0x04, 0x26, 0xca, 0x03, 0x00, 0x00, 0x16, 0x9a, 0x03, 0x16,
    0x0c, 0x00, 0x28, 0x49, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d, 0x72, 0x73, 0x20, 0x76,
    0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x30, 0x2e, 0x33, 0x2e, 0x30, 0x20, 0x28, 0x62, 0x75,
    0x69, 0x6c, 0x64, 0x20, 0x62, 0x34, 0x35, 0x63, 0x65, 0x37, 0x63, 0x62, 0x61, 0x32, 0x31, 0x39,
    0x39, 0x66, 0x32, 0x32, 0x64, 0x39, 0x33, 0x32, 0x36, 0x39, 0x63, 0x31, 0x35, 0x30, 0x64, 0x38,
    0x61, 0x38, 0x33, 0x39, 0x31, 0x36, 0x63, 0x36, 0x39, 0x62, 0x35, 0x65, 0x29, 0x00, 0x32, 0x01,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(repeated_bytes.data()), repeated_bytes.size()}});
  auto result = cudf::io::read_parquet(read_opts);

  EXPECT_EQ(result.tbl->view().column(0).size(), 6);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  column_wrapper<int32_t> col0{1, 2, 3, 4, 5, 6};
  column_wrapper<int64_t> child0{{5555555555l, 1111111111l, 1111111111l, 2222222222l, 3333333333l}};
  cudf::test::strings_column_wrapper child1{{"-", "home", "home", "-", "mobile"},
                                            {false, true, true, false, true}};
  auto struct_col = cudf::test::structs_column_wrapper{{child0, child1}};

  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 1, 2, 5}.release();
  auto num_list_rows = list_offsets_column->size() - 1;

  auto mask = cudf::create_null_mask(6, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, 2, false);

  auto list_col = cudf::make_lists_column(
    num_list_rows, std::move(list_offsets_column), struct_col.release(), 2, std::move(mask));

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(std::move(list_col));

  auto outer_struct = cudf::test::structs_column_wrapper{{std::move(struct_children)},
                                                         {false, false, true, true, true, true}};
  table_view expected{{col0, outer_struct}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected);
}

// Regression test for https://github.com/rapidsai/cudf/issues/22541.
// Schema (single-field inner repeated group must decode as list<struct<someId>>):
//   required group root {
//     optional int32 primitive;
//     repeated group myComplex {
//       optional int32 id;
//       repeated group repeatedMessage {
//         optional int32 someId;
//       }
//     }
//   }
// Regenerate `nested_array_bytes` from the parquet-mr source file with:
//   xxd -i nested-array-struct.parquet > /tmp/blob.cpp
// (source: spark/sql/core/src/test/resources/test-data/nested-array-struct.parquet)
TEST_F(ParquetReaderTest, RepeatedNoAnnotationsSingleFieldNested)
{
  constexpr std::array<unsigned char, 775> nested_array_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x24, 0x15, 0x24, 0x2c, 0x15, 0x06, 0x15, 0x00, 0x15,
    0x06, 0x15, 0x08, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x07, 0x02, 0x00, 0x00, 0x00, 0x05,
    0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x32, 0x15, 0x32, 0x2c, 0x15, 0x06,
    0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x03, 0x2a, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00,
    0x00, 0x15, 0x00, 0x15, 0x34, 0x15, 0x34, 0x2c, 0x15, 0x06, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x3f, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x15, 0x02, 0x19, 0x6c,
    0x48, 0x29, 0x54, 0x65, 0x73, 0x74, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x41,
    0x72, 0x72, 0x61, 0x79, 0x57, 0x69, 0x74, 0x68, 0x4e, 0x65, 0x73, 0x74, 0x65, 0x64, 0x47, 0x72,
    0x6f, 0x75, 0x70, 0x41, 0x6e, 0x64, 0x41, 0x72, 0x72, 0x61, 0x79, 0x15, 0x04, 0x00, 0x15, 0x02,
    0x25, 0x02, 0x18, 0x09, 0x70, 0x72, 0x69, 0x6d, 0x69, 0x74, 0x69, 0x76, 0x65, 0x00, 0x35, 0x04,
    0x18, 0x09, 0x6d, 0x79, 0x43, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x78, 0x15, 0x04, 0x00, 0x15, 0x02,
    0x25, 0x02, 0x18, 0x02, 0x69, 0x64, 0x00, 0x35, 0x04, 0x18, 0x0f, 0x72, 0x65, 0x70, 0x65, 0x61,
    0x74, 0x65, 0x64, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x15, 0x02, 0x00, 0x15, 0x02, 0x25,
    0x02, 0x18, 0x06, 0x73, 0x6f, 0x6d, 0x65, 0x49, 0x64, 0x00, 0x16, 0x06, 0x19, 0x1c, 0x19, 0x3c,
    0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x35, 0x08, 0x06, 0x00, 0x19, 0x18, 0x09, 0x70, 0x72, 0x69,
    0x6d, 0x69, 0x74, 0x69, 0x76, 0x65, 0x15, 0x00, 0x16, 0x06, 0x16, 0x46, 0x16, 0x46, 0x26, 0x08,
    0x00, 0x00, 0x26, 0x4e, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x06, 0x00, 0x19, 0x28, 0x09, 0x6d, 0x79,
    0x43, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x78, 0x02, 0x69, 0x64, 0x15, 0x00, 0x16, 0x06, 0x16, 0x54,
    0x16, 0x54, 0x26, 0x4e, 0x00, 0x00, 0x26, 0xa2, 0x01, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x06, 0x00,
    0x19, 0x38, 0x09, 0x6d, 0x79, 0x43, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x78, 0x0f, 0x72, 0x65, 0x70,
    0x65, 0x61, 0x74, 0x65, 0x64, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x06, 0x73, 0x6f, 0x6d,
    0x65, 0x49, 0x64, 0x15, 0x00, 0x16, 0x06, 0x16, 0x56, 0x16, 0x56, 0x26, 0xa2, 0x01, 0x00, 0x00,
    0x16, 0xf0, 0x01, 0x16, 0x06, 0x00, 0x19, 0x2c, 0x18, 0x18, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65,
    0x74, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x64, 0x65, 0x73, 0x63, 0x72, 0x69, 0x70, 0x74,
    0x6f, 0x72, 0x18, 0xf8, 0x01, 0x6e, 0x61, 0x6d, 0x65, 0x3a, 0x20, 0x22, 0x41, 0x72, 0x72, 0x61,
    0x79, 0x57, 0x69, 0x74, 0x68, 0x4e, 0x65, 0x73, 0x74, 0x65, 0x64, 0x47, 0x72, 0x6f, 0x75, 0x70,
    0x41, 0x6e, 0x64, 0x41, 0x72, 0x72, 0x61, 0x79, 0x22, 0x0a, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x20,
    0x7b, 0x0a, 0x20, 0x20, 0x6e, 0x61, 0x6d, 0x65, 0x3a, 0x20, 0x22, 0x70, 0x72, 0x69, 0x6d, 0x69,
    0x74, 0x69, 0x76, 0x65, 0x22, 0x0a, 0x20, 0x20, 0x6e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x3a, 0x20,
    0x31, 0x0a, 0x20, 0x20, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x3a, 0x20, 0x4c, 0x41, 0x42, 0x45, 0x4c,
    0x5f, 0x4f, 0x50, 0x54, 0x49, 0x4f, 0x4e, 0x41, 0x4c, 0x0a, 0x20, 0x20, 0x74, 0x79, 0x70, 0x65,
    0x3a, 0x20, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x49, 0x4e, 0x54, 0x33, 0x32, 0x0a, 0x7d, 0x0a, 0x66,
    0x69, 0x65, 0x6c, 0x64, 0x20, 0x7b, 0x0a, 0x20, 0x20, 0x6e, 0x61, 0x6d, 0x65, 0x3a, 0x20, 0x22,
    0x6d, 0x79, 0x43, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x78, 0x22, 0x0a, 0x20, 0x20, 0x6e, 0x75, 0x6d,
    0x62, 0x65, 0x72, 0x3a, 0x20, 0x32, 0x0a, 0x20, 0x20, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x3a, 0x20,
    0x4c, 0x41, 0x42, 0x45, 0x4c, 0x5f, 0x52, 0x45, 0x50, 0x45, 0x41, 0x54, 0x45, 0x44, 0x0a, 0x20,
    0x20, 0x74, 0x79, 0x70, 0x65, 0x3a, 0x20, 0x54, 0x59, 0x50, 0x45, 0x5f, 0x4d, 0x45, 0x53, 0x53,
    0x41, 0x47, 0x45, 0x0a, 0x20, 0x20, 0x74, 0x79, 0x70, 0x65, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x3a,
    0x20, 0x22, 0x2e, 0x54, 0x65, 0x73, 0x74, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e,
    0x4d, 0x79, 0x43, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x78, 0x22, 0x0a, 0x7d, 0x0a, 0x00, 0x18, 0x13,
    0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x63, 0x6c,
    0x61, 0x73, 0x73, 0x18, 0x3c, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2e, 0x70, 0x72, 0x6f,
    0x74, 0x6f, 0x2e, 0x74, 0x65, 0x73, 0x74, 0x2e, 0x54, 0x65, 0x73, 0x74, 0x50, 0x72, 0x6f, 0x74,
    0x6f, 0x62, 0x75, 0x66, 0x24, 0x41, 0x72, 0x72, 0x61, 0x79, 0x57, 0x69, 0x74, 0x68, 0x4e, 0x65,
    0x73, 0x74, 0x65, 0x64, 0x47, 0x72, 0x6f, 0x75, 0x70, 0x41, 0x6e, 0x64, 0x41, 0x72, 0x72, 0x61,
    0x79, 0x00, 0x18, 0x0a, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d, 0x6d, 0x72, 0x00, 0x83,
    0x02, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(nested_array_bytes.data()), nested_array_bytes.size()}});
  auto result = cudf::io::read_parquet(read_opts);

  // 3 rows: (2, [{1, [{3}]}]), (5, [{4, [{6}]}]), (8, [{7, [{9}]}])
  auto const tv = result.tbl->view();
  ASSERT_EQ(tv.num_columns(), 2);
  EXPECT_EQ(tv.column(0).size(), 3);

  // Column 0: primitive int32 = [2, 5, 8]
  column_wrapper<int32_t> col0{2, 5, 8};

  // Column 1: list<struct<id:int32, repeatedMessage:list<struct<someId:int32>>>>
  auto const& outer_list = tv.column(1);
  ASSERT_EQ(outer_list.type().id(), cudf::type_id::LIST);
  auto const outer_child = outer_list.child(cudf::lists_column_view::child_column_index);
  ASSERT_EQ(outer_child.type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(outer_child.num_children(), 2);
  EXPECT_EQ(outer_child.child(0).type().id(), cudf::type_id::INT32);
  ASSERT_EQ(outer_child.child(1).type().id(), cudf::type_id::LIST);
  auto const inner_child = outer_child.child(1).child(cudf::lists_column_view::child_column_index);
  // Inner single-field group must be a STRUCT, not collapsed to a primitive (issue #22541).
  ASSERT_EQ(inner_child.type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(inner_child.num_children(), 1);
  EXPECT_EQ(inner_child.child(0).type().id(), cudf::type_id::INT32);

  column_wrapper<int32_t> inner_someid{3, 6, 9};
  auto inner_struct = cudf::test::structs_column_wrapper{{inner_someid}};
  auto inner_list_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3}.release();
  auto inner_list = cudf::make_lists_column(
    3, std::move(inner_list_offsets), inner_struct.release(), 0, rmm::device_buffer{});

  column_wrapper<int32_t> outer_id{1, 4, 7};
  std::vector<std::unique_ptr<cudf::column>> outer_struct_children;
  outer_struct_children.push_back(outer_id.release());
  outer_struct_children.push_back(std::move(inner_list));
  auto outer_struct_col = cudf::test::structs_column_wrapper{{std::move(outer_struct_children)}};

  auto outer_list_offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3}.release();
  auto outer_list_col = cudf::make_lists_column(
    3, std::move(outer_list_offsets), outer_struct_col.release(), 0, rmm::device_buffer{});

  // Testing for equivalence here because we only care about the outermost validity buffers.
  table_view expected{{col0, *outer_list_col}};
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.tbl->view(), expected);
}

TEST_F(ParquetReaderTest, DeltaSkipRowsWithNulls)
{
  using cudf::io::column_encoding;
  constexpr int num_rows = 10'000;
  constexpr auto seed    = 21337;

  std::mt19937 engine{seed};
  auto int32_list_nulls = make_parquet_list_col<int32_t>(engine, num_rows, 5, true);
  auto int32_list       = make_parquet_list_col<int32_t>(engine, num_rows, 5, false);
  auto int64_list_nulls = make_parquet_list_col<int64_t>(engine, num_rows, 5, true);
  auto int64_list       = make_parquet_list_col<int64_t>(engine, num_rows, 5, false);
  auto int16_list_nulls = make_parquet_list_col<int16_t>(engine, num_rows, 5, true);
  auto int16_list       = make_parquet_list_col<int16_t>(engine, num_rows, 5, false);
  auto int8_list_nulls  = make_parquet_list_col<int8_t>(engine, num_rows, 5, true);
  auto int8_list        = make_parquet_list_col<int8_t>(engine, num_rows, 5, false);

  auto str_list_nulls     = make_parquet_string_list_col(engine, num_rows, 5, 32, true);
  auto str_list           = make_parquet_string_list_col(engine, num_rows, 5, 32, false);
  auto big_str_list_nulls = make_parquet_string_list_col(engine, num_rows, 5, 256, true);
  auto big_str_list       = make_parquet_string_list_col(engine, num_rows, 5, 256, false);

  auto int32_data   = random_values<int32_t>(num_rows);
  auto int64_data   = random_values<int64_t>(num_rows);
  auto int16_data   = random_values<int16_t>(num_rows);
  auto int8_data    = random_values<int8_t>(num_rows);
  auto str_data     = string_values(engine, num_rows, 32);
  auto big_str_data = string_values(engine, num_rows, 256);

  auto const validity = random_validity(engine);
  auto const no_nulls = cudf::test::iterators::no_nulls();
  column_wrapper<int32_t> int32_nulls_col{int32_data.begin(), int32_data.end(), validity};
  column_wrapper<int32_t> int32_col{int32_data.begin(), int32_data.end(), no_nulls};
  column_wrapper<int64_t> int64_nulls_col{int64_data.begin(), int64_data.end(), validity};
  column_wrapper<int64_t> int64_col{int64_data.begin(), int64_data.end(), no_nulls};

  auto str_col = cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), no_nulls);
  auto str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), validity));
  auto big_str_col =
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), no_nulls);
  auto big_str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), validity));

  cudf::table_view tbl({int32_col,   int32_nulls_col,    *int32_list,   *int32_list_nulls,
                        int64_col,   int64_nulls_col,    *int64_list,   *int64_list_nulls,
                        *int16_list, *int16_list_nulls,  *int8_list,    *int8_list_nulls,
                        str_col,     *str_col_nulls,     *str_list,     *str_list_nulls,
                        big_str_col, *big_str_col_nulls, *big_str_list, *big_str_list_nulls,
                        str_col,     *str_col_nulls,     *str_list,     *str_list_nulls,
                        big_str_col, *big_str_col_nulls, *big_str_list, *big_str_list_nulls});

  auto const filepath = temp_env->get_temp_filepath("DeltaSkipRowsWithNulls.parquet");
  auto input_metadata = cudf::io::table_input_metadata{tbl};
  for (int i = 12; i <= 27; ++i) {
    input_metadata.column_metadata[i].set_encoding(
      i <= 19 ? column_encoding::DELTA_LENGTH_BYTE_ARRAY : column_encoding::DELTA_BYTE_ARRAY);
  }

  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .max_page_size_rows(5'000)
      .write_v2_headers(true)
      .build();
  cudf::io::write_parquet(out_opts);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{
    // skip and then read rest of file
    {-1, -1}, {1, -1}, {2, -1}, {32, -1}, {33, -1}, {128, -1}, {1000, -1},
    // no skip but read fewer rows
    {0, 1}, {0, 2}, {0, 31}, {0, 32}, {0, 33}, {0, 128}, {0, 129}, {0, 130},
    // skip and truncate
    {1, 32}, {1, 33}, {32, 32}, {33, 139},
    // cross page boundaries
    {3'000, 5'000}
  };

  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    std::vector<cudf::table_view> expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected[0]);

    // test writing the result back out as a further check of the delta writer's correctness
    std::vector<char> out_buffer;
    cudf::io::parquet_writer_options out_opts2 =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer},
                                                result.tbl->view())
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .compression(cudf::io::compression_type::NONE)
        .dictionary_policy(cudf::io::dictionary_policy::NEVER)
        .max_page_size_rows(5'000)
        .write_v2_headers(true);
    cudf::io::write_parquet(out_opts2);

    cudf::io::parquet_reader_options default_in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}});
    auto const result2 = cudf::io::read_parquet(default_in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), result2.tbl->view());
  }
}

TEST_F(ParquetReaderTest, DeltaByteArraySkipAllValid)
{
  // test that the DELTA_BYTE_ARRAY decoder can handle the case where skip rows skips all valid
  // values in a page. see #15075
  constexpr int num_rows  = 500;
  constexpr int num_valid = 150;

  auto const ones = cuda::make_constant_iterator("one");

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [num_valid](auto i) { return i < num_valid; });
  auto const col      = cudf::test::strings_column_wrapper{ones, ones + num_rows, valids};
  auto const expected = table_view({col});

  auto input_metadata = cudf::io::table_input_metadata{expected};
  input_metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::DELTA_BYTE_ARRAY);

  auto const filepath = temp_env->get_temp_filepath("DeltaByteArraySkipAllValid.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .metadata(input_metadata)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .skip_rows(num_valid + 1);
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::slice(expected, {num_valid + 1, num_rows}),
                                result.tbl->view());
}

// test that using page stats is working for full reads and various skip rows
TEST_F(ParquetReaderTest, StringsWithPageStats)
{
  constexpr int num_rows = 10'000;
  constexpr auto seed    = 21337;

  std::mt19937 engine{seed};
  auto int32_list_nulls = make_parquet_list_col<int32_t>(engine, num_rows, 5, true);
  auto int32_list       = make_parquet_list_col<int32_t>(engine, num_rows, 5, false);
  auto int64_list_nulls = make_parquet_list_col<int64_t>(engine, num_rows, 5, true);
  auto int64_list       = make_parquet_list_col<int64_t>(engine, num_rows, 5, false);
  auto int16_list_nulls = make_parquet_list_col<int16_t>(engine, num_rows, 5, true);
  auto int16_list       = make_parquet_list_col<int16_t>(engine, num_rows, 5, false);
  auto int8_list_nulls  = make_parquet_list_col<int8_t>(engine, num_rows, 5, true);
  auto int8_list        = make_parquet_list_col<int8_t>(engine, num_rows, 5, false);

  auto str_list_nulls     = make_parquet_string_list_col(engine, num_rows, 5, 32, true);
  auto str_list           = make_parquet_string_list_col(engine, num_rows, 5, 32, false);
  auto big_str_list_nulls = make_parquet_string_list_col(engine, num_rows, 5, 256, true);
  auto big_str_list       = make_parquet_string_list_col(engine, num_rows, 5, 256, false);

  auto int32_data   = random_values<int32_t>(num_rows);
  auto int64_data   = random_values<int64_t>(num_rows);
  auto int16_data   = random_values<int16_t>(num_rows);
  auto int8_data    = random_values<int8_t>(num_rows);
  auto str_data     = string_values(engine, num_rows, 32);
  auto big_str_data = string_values(engine, num_rows, 256);

  auto const validity = random_validity(engine);
  auto const no_nulls = cudf::test::iterators::no_nulls();
  column_wrapper<int32_t> int32_nulls_col{int32_data.begin(), int32_data.end(), validity};
  column_wrapper<int32_t> int32_col{int32_data.begin(), int32_data.end(), no_nulls};
  column_wrapper<int64_t> int64_nulls_col{int64_data.begin(), int64_data.end(), validity};
  column_wrapper<int64_t> int64_col{int64_data.begin(), int64_data.end(), no_nulls};

  auto str_col = cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), no_nulls);
  auto str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), validity));
  auto big_str_col =
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), no_nulls);
  auto big_str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), validity));

  cudf::table_view tbl({int32_col,   int32_nulls_col,    *int32_list,   *int32_list_nulls,
                        int64_col,   int64_nulls_col,    *int64_list,   *int64_list_nulls,
                        *int16_list, *int16_list_nulls,  *int8_list,    *int8_list_nulls,
                        str_col,     *str_col_nulls,     *str_list,     *str_list_nulls,
                        big_str_col, *big_str_col_nulls, *big_str_list, *big_str_list_nulls});

  auto const filepath = temp_env->get_temp_filepath("StringsWithPageStats.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .max_page_size_rows(5'000)
      .build();
  cudf::io::write_parquet(out_opts);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{
    // skip and then read rest of file
    {-1, -1}, {1, -1}, {2, -1}, {32, -1}, {33, -1}, {128, -1}, {1'000, -1},
    // no skip but truncate
    {0, 1'000}, {0, 6'000},
    // cross page boundaries
    {3'000, 5'000}
  };

  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    std::vector<cudf::table_view> expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected[0]);
  }
}

TEST_F(ParquetReaderTest, NumRowsPerSource)
{
  int constexpr num_rows          = 10'723;  // A prime number
  int constexpr rows_in_row_group = 500;

  // Table with single col of random int64 values
  auto const int64_data = random_values<int64_t>(num_rows);
  column_wrapper<int64_t> const int64_col{
    int64_data.begin(), int64_data.end(), cudf::test::iterators::no_nulls()};
  cudf::table_view const expected({int64_col});

  // Write to Parquet
  auto const filepath = temp_env->get_temp_filepath("NumRowsPerSource.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .row_group_size_rows(rows_in_row_group)
      .build();
  cudf::io::write_parquet(out_opts);

  // Read single data source entirely
  {
    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 1);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], num_rows);
  }

  // Read rows_to_read rows skipping rows_to_skip from single data source
  {
    auto constexpr rows_to_skip = 557;  // a prime number != rows_in_row_group
    auto constexpr rows_to_read = 7'232;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_rows(rows_to_skip)
                           .num_rows(rows_to_read)
                           .build();
    auto const result = cudf::io::read_parquet(in_opts);
    column_wrapper<int64_t> int64_col_selected{int64_data.begin() + rows_to_skip,
                                               int64_data.begin() + rows_to_skip + rows_to_read,
                                               cudf::test::iterators::no_nulls()};

    cudf::table_view const expected_selected({int64_col_selected});

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_selected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 1);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], rows_to_read);
  }

  // Filtered read from single data source
  {
    auto constexpr max_value = 100;
    auto literal_value       = cudf::numeric_scalar<int64_t>{max_value};
    auto literal             = cudf::ast::literal{literal_value};
    auto col_ref             = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal);

    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .filter(filter_expression)
                           .build();

    std::vector<int64_t> int64_data_filtered;
    int64_data_filtered.reserve(num_rows);
    std::copy_if(
      int64_data.begin(), int64_data.end(), std::back_inserter(int64_data_filtered), [=](auto val) {
        return val <= max_value;
      });
    column_wrapper<int64_t> int64_col_filtered{
      int64_data_filtered.begin(), int64_data_filtered.end(), cudf::test::iterators::no_nulls()};

    cudf::table_view expected_filtered({int64_col_filtered});

    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_filtered, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 0);
  }

  // Read two data sources skipping the first entire file completely
  {
    auto constexpr rows_to_skip = 15'723;
    auto constexpr nsources     = 2;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources})
        .skip_rows(rows_to_skip)
        .build();

    auto const result = cudf::io::read_parquet(in_opts);

    column_wrapper<int64_t> int64_col_selected{int64_data.begin() + rows_to_skip - num_rows,
                                               int64_data.end(),
                                               cudf::test::iterators::no_nulls()};

    cudf::table_view const expected_selected({int64_col_selected});

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_selected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 2);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], 0);
    EXPECT_EQ(result.metadata.num_rows_per_source[1], nsources * num_rows - rows_to_skip);
  }

  // Read ten data sources entirely
  {
    auto constexpr nsources = 10;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources}).build();
    auto const result = cudf::io::read_parquet(in_opts);

    // Initialize expected_counts
    std::vector<size_t> const expected_counts(nsources, num_rows);

    EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
    EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                           expected_counts.cend(),
                           result.metadata.num_rows_per_source.cbegin()));
  }

  // Read rows_to_read rows skipping rows_to_skip (> two sources) from ten data sources
  {
    auto constexpr rows_to_skip = 25'999;
    auto constexpr rows_to_read = 10'900;

    auto constexpr nsources = 10;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources})
        .skip_rows(rows_to_skip)
        .num_rows(rows_to_read)
        .build();

    auto const result = cudf::io::read_parquet(in_opts);

    // Initialize expected_counts
    std::vector<size_t> expected_counts(nsources, num_rows);

    // Adjust expected_counts for rows_to_skip
    int64_t counter = 0;
    for (auto& nrows : expected_counts) {
      if (counter < rows_to_skip) {
        counter += nrows;
        nrows = (counter >= rows_to_skip) ? counter - rows_to_skip : 0;
      } else {
        break;
      }
    }

    // Reset the counter
    counter = 0;

    // Adjust expected_counts for rows_to_read
    for (auto& nrows : expected_counts) {
      if (counter < rows_to_read) {
        counter += nrows;
        nrows = (counter >= rows_to_read) ? rows_to_read - counter + nrows : nrows;
      } else if (counter > rows_to_read) {
        nrows = 0;
      }
    }

    EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
    EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                           expected_counts.cend(),
                           result.metadata.num_rows_per_source.cbegin()));
  }
}

TEST_F(ParquetReaderTest, NumRowsPerSourceEmptyTable)
{
  auto const nsources = 10;

  column_wrapper<int64_t> const int64_empty_col{};
  cudf::table_view const expected_empty({int64_empty_col});

  // Write to Parquet
  auto const filepath_empty = temp_env->get_temp_filepath("NumRowsPerSourceEmpty.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath_empty}, expected_empty)
      .build();
  cudf::io::write_parquet(out_opts);

  // Read from Parquet
  std::vector<std::string> const datasources(nsources, filepath_empty);

  auto const in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources}).build();
  auto const result = cudf::io::read_parquet(in_opts);

  // Initialize expected_counts
  std::vector<size_t> const expected_counts(nsources, 0);

  EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
  EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                         expected_counts.cend(),
                         result.metadata.num_rows_per_source.cbegin()));
}

///////////////////
// metadata tests

// Test fixture for metadata tests
struct ParquetMetadataReaderTest : public cudf::test::BaseFixture {
  std::string print(cudf::io::parquet_column_schema schema, int depth = 0)
  {
    std::string child_str;
    for (auto const& child : schema.children()) {
      child_str += print(child, depth + 1);
    }
    return std::string(depth, ' ') + schema.name() + "\n" + child_str;
  }
};

static constexpr char const* parquet_metadata_size_hint_env_var =
  "LIBCUDF_PARQUET_METADATA_SIZE_HINT";

struct ParquetMetadataSizeHintTest : public ParquetReaderTest {};

TEST_F(ParquetMetadataSizeHintTest, ReadParquet)
{
  srand(31337);
  auto const expected = create_random_fixed_table<int>(2, 8, false);

  auto const filepath = temp_env->get_temp_filepath("MetadataSizeHint.parquet");
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
  cudf::io::write_parquet(write_opts);

  auto const source        = cudf::io::datasource::create(filepath);
  auto const file_size     = source->size();
  auto constexpr ender_len = sizeof(cudf::io::parquet::file_ender_s);
  auto const ender_buffer  = source->host_read(file_size - ender_len, ender_len);
  auto const ender = reinterpret_cast<cudf::io::parquet::file_ender_s const*>(ender_buffer->data());
  auto const footer_len = static_cast<size_t>(ender->footer_len);

  auto const source_info = cudf::io::source_info{filepath};

  // Test cases:
  // - 0: disable speculative reads
  // - 9: smaller than typical footer metadata
  // - footer_len + 1: larger than the footer
  // - file_size + 1: larger than the entire file
  std::vector<size_t> const hints{0, 9, footer_len + 1, file_size + 1};
  for (auto const hint : hints) {
    tmp_env_var const env(parquet_metadata_size_hint_env_var, std::to_string(hint));
    auto const read_opts = cudf::io::parquet_reader_options::builder(source_info).build();
    auto const result    = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected->view());
  }
}

TEST_F(ParquetMetadataReaderTest, Basics)
{
  auto const num_rows = 1200;

  auto ints   = random_values<int>(num_rows);
  auto floats = random_values<float>(num_rows);
  column_wrapper<int> int_col(ints.begin(), ints.end());
  column_wrapper<float> float_col(floats.begin(), floats.end());

  table_view expected({int_col, float_col});

  auto filepath =
    write_parquet_temp_file(expected, "MetadataTest.parquet", {"int_col", "float_col"});

  auto const test_parquet_metadata = [&](int num_sources) {
    auto meta =
      read_parquet_metadata(cudf::io::source_info{std::vector<std::string>(num_sources, filepath)});
    EXPECT_EQ(meta.num_rows(), num_sources * num_rows);

    auto const column_chunk_metadata = meta.columnchunk_metadata();
    // Two leaf columns
    EXPECT_EQ(column_chunk_metadata.size(), 2);
    // Check if all leaf columns have the expected number of row groups
    EXPECT_EQ(column_chunk_metadata.at("int_col").size(), meta.num_rowgroups());
    EXPECT_EQ(column_chunk_metadata.at("float_col").size(), meta.num_rowgroups());

    EXPECT_EQ(meta.num_rowgroups_per_file().size(), num_sources);
    for (auto const& num_row_groups : meta.num_rowgroups_per_file()) {
      // Each source file has only one row group
      EXPECT_EQ(num_row_groups, 1);
    }

    std::string expected_schema = R"(schema
 int_col
 float_col
)";
    EXPECT_EQ(expected_schema, print(meta.schema().root()));
    EXPECT_EQ(meta.schema().root().name(), "schema");
    EXPECT_EQ(meta.schema().root().type(), cudf::io::parquet::Type::UNDEFINED);
    ASSERT_EQ(meta.schema().root().num_children(), 2);

    EXPECT_EQ(meta.schema().root().child(0).name(), "int_col");
    EXPECT_EQ(meta.schema().root().child(1).name(), "float_col");
  };

  // Test with single file
  test_parquet_metadata(1);
  // Test with multiple files
  test_parquet_metadata(3);
}

TEST_F(ParquetMetadataReaderTest, PreMaterializedMetadata)
{
  auto const num_rows = 1200;

  auto ints   = random_values<int>(num_rows);
  auto floats = random_values<float>(num_rows);
  column_wrapper<int> int_col(ints.begin(), ints.end());
  column_wrapper<float> float_col(floats.begin(), floats.end());

  table_view input_table({int_col, float_col});
  auto filepath = temp_env->get_temp_filepath("PreMaterializedMetadata.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input_table).build();
  cudf::io::write_parquet(out_opts);

  auto const test_parquet_metadata = [&](int num_sources) {
    auto const source_info = cudf::io::source_info{std::vector<std::string>(num_sources, filepath)};
    auto datasources       = cudf::io::make_datasources(source_info);
    auto metadatas         = cudf::io::read_parquet_footers(datasources);
    EXPECT_EQ(metadatas.size(), num_sources);
    auto const rows_in_metadatas =
      std::accumulate(metadatas.begin(), metadatas.end(), 0, [](auto acc, auto const& metadata) {
        return acc + metadata.num_rows;
      });
    EXPECT_EQ(rows_in_metadatas, num_sources * num_rows);

    auto const options = cudf::io::parquet_reader_options::builder(source_info).build();
    auto const read = cudf::io::read_parquet(std::move(datasources), std::move(metadatas), options);
    auto const expected = cudf::io::read_parquet(options);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected.tbl->view(), read.tbl->view());
  };

  // Test with single file
  test_parquet_metadata(1);
  // Test with multiple files
  test_parquet_metadata(3);
}

TEST_F(ParquetMetadataReaderTest, Nested)
{
  auto const num_rows       = 1200;
  auto const lists_per_row  = 4;
  auto const num_child_rows = num_rows * lists_per_row;

  auto keys = random_values<int>(num_child_rows);
  auto vals = random_values<float>(num_child_rows);
  column_wrapper<int> keys_col(keys.begin(), keys.end());
  column_wrapper<float> vals_col(vals.begin(), vals.end());
  auto s_col = cudf::test::structs_column_wrapper({keys_col, vals_col}).release();

  std::vector<int> row_offsets(num_rows + 1);
  for (int idx = 0; idx < num_rows + 1; ++idx) {
    row_offsets[idx] = idx * lists_per_row;
  }
  column_wrapper<int> offsets(row_offsets.begin(), row_offsets.end());

  auto list_col =
    cudf::make_lists_column(num_rows, offsets.release(), std::move(s_col), 0, rmm::device_buffer{});

  table_view expected({*list_col, *list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("maps");
  expected_metadata.column_metadata[0].set_list_column_as_map();
  expected_metadata.column_metadata[1].set_name("lists");
  expected_metadata.column_metadata[1].child(1).child(0).set_name("int_field");
  expected_metadata.column_metadata[1].child(1).child(1).set_name("float_field");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  auto const column_chunk_metadata = meta.columnchunk_metadata();
  // Four leaf columns
  EXPECT_EQ(column_chunk_metadata.size(), 4);
  // Check if all leaf columns are present
  EXPECT_TRUE(column_chunk_metadata.find("maps.key_value.key") != column_chunk_metadata.end());
  EXPECT_TRUE(column_chunk_metadata.find("maps.key_value.value") != column_chunk_metadata.end());
  EXPECT_TRUE(column_chunk_metadata.find("lists.list.element.int_field") !=
              column_chunk_metadata.end());
  EXPECT_TRUE(column_chunk_metadata.find("lists.list.element.float_field") !=
              column_chunk_metadata.end());

  EXPECT_EQ(meta.num_rowgroups_per_file().size(), 1);
  EXPECT_EQ(meta.num_rowgroups_per_file()[0], meta.num_rowgroups());

  std::string expected_schema = R"(schema
 maps
  key_value
   key
   value
 lists
  list
   element
    int_field
    float_field
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type(),
            cudf::io::parquet::Type::UNDEFINED);  // struct
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  auto const& out_map_col = meta.schema().root().child(0);
  EXPECT_EQ(out_map_col.name(), "maps");
  EXPECT_EQ(out_map_col.type(), cudf::io::parquet::Type::UNDEFINED);  // map

  ASSERT_EQ(out_map_col.num_children(), 1);
  EXPECT_EQ(out_map_col.child(0).name(), "key_value");  // key_value (named in parquet writer)
  ASSERT_EQ(out_map_col.child(0).num_children(), 2);
  EXPECT_EQ(out_map_col.child(0).child(0).name(), "key");    // key (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(1).name(), "value");  // value (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(0).type(), cudf::io::parquet::Type::INT32);  // int
  EXPECT_EQ(out_map_col.child(0).child(1).type(),
            cudf::io::parquet::Type::FLOAT);  // float

  auto const& out_list_col = meta.schema().root().child(1);
  EXPECT_EQ(out_list_col.name(), "lists");
  EXPECT_EQ(out_list_col.type(), cudf::io::parquet::Type::UNDEFINED);  // list
  // TODO repetition type?
  ASSERT_EQ(out_list_col.num_children(), 1);
  EXPECT_EQ(out_list_col.child(0).name(), "list");  // list (named in parquet writer)
  ASSERT_EQ(out_list_col.child(0).num_children(), 1);

  auto const& out_list_struct_col = out_list_col.child(0).child(0);
  EXPECT_EQ(out_list_struct_col.name(), "element");  // elements (named in parquet writer)
  EXPECT_EQ(out_list_struct_col.type(),
            cudf::io::parquet::Type::UNDEFINED);  // struct
  ASSERT_EQ(out_list_struct_col.num_children(), 2);

  auto const& out_int_col = out_list_struct_col.child(0);
  EXPECT_EQ(out_int_col.name(), "int_field");
  EXPECT_EQ(out_int_col.type(), cudf::io::parquet::Type::INT32);

  auto const& out_float_col = out_list_struct_col.child(1);
  EXPECT_EQ(out_float_col.name(), "float_field");
  EXPECT_EQ(out_float_col.type(), cudf::io::parquet::Type::FLOAT);
}
TEST_F(ParquetMetadataReaderTest, CudfTypes)
{
  auto const num_rows = 4;

  auto ints    = random_values<int32_t>(num_rows);
  auto longs   = random_values<int64_t>(num_rows);
  auto floats  = random_values<float>(num_rows);
  auto doubles = random_values<double>(num_rows);
  column_wrapper<int32_t> int_col(ints.begin(), ints.end());
  column_wrapper<int64_t> long_col(longs.begin(), longs.end());
  column_wrapper<float> float_col(floats.begin(), floats.end());
  column_wrapper<double> double_col(doubles.begin(), doubles.end());

  auto list_ints = random_values<int32_t>(num_rows * 2);
  auto list_col =
    cudf::test::lists_column_wrapper<int32_t, int32_t>{{list_ints[0], list_ints[1]},
                                                       {list_ints[2], list_ints[3], list_ints[4]},
                                                       {list_ints[5]},
                                                       {list_ints[6], list_ints[7]}};

  auto struct_ints   = random_values<int32_t>(4);
  auto struct_floats = random_values<float>(4);
  column_wrapper<int32_t> struct_int_col(struct_ints.begin(), struct_ints.end());
  column_wrapper<float> struct_float_col(struct_floats.begin(), struct_floats.end());
  auto struct_col = cudf::test::structs_column_wrapper{{struct_int_col, struct_float_col}};

  table_view expected({int_col, long_col, float_col, double_col, list_col, struct_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int_col");
  expected_metadata.column_metadata[1].set_name("long_col");
  expected_metadata.column_metadata[2].set_name("float_col");
  expected_metadata.column_metadata[3].set_name("double_col");
  expected_metadata.column_metadata[4].set_name("list_col");
  expected_metadata.column_metadata[5].set_name("struct_col");
  expected_metadata.column_metadata[5].child(0).set_name("struct_int");
  expected_metadata.column_metadata[5].child(1).set_name("struct_float");

  auto filepath = temp_env->get_temp_filepath("MetadataTestCudfTypes.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta               = read_parquet_metadata(cudf::io::source_info{filepath});
  auto const& schema_root = meta.schema().root();

  ASSERT_EQ(schema_root.num_children(), 6);

  EXPECT_EQ(schema_root.child(0).name(), "int_col");
  EXPECT_EQ(schema_root.child(0).cudf_type().id(), cudf::type_id::INT32);

  EXPECT_EQ(schema_root.child(1).name(), "long_col");
  EXPECT_EQ(schema_root.child(1).cudf_type().id(), cudf::type_id::INT64);

  EXPECT_EQ(schema_root.child(2).name(), "float_col");
  EXPECT_EQ(schema_root.child(2).cudf_type().id(), cudf::type_id::FLOAT32);

  EXPECT_EQ(schema_root.child(3).name(), "double_col");
  EXPECT_EQ(schema_root.child(3).cudf_type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(schema_root.child(4).name(), "list_col");
  EXPECT_EQ(schema_root.child(4).cudf_type().id(), cudf::type_id::LIST);
  ASSERT_EQ(schema_root.child(4).num_children(), 1);
  EXPECT_EQ(schema_root.child(4).child(0).name(), "list");
  EXPECT_EQ(schema_root.child(4).child(0).cudf_type().id(), cudf::type_id::EMPTY);
  ASSERT_EQ(schema_root.child(4).child(0).num_children(), 1);
  EXPECT_EQ(schema_root.child(4).child(0).child(0).cudf_type().id(), cudf::type_id::INT32);

  EXPECT_EQ(schema_root.child(5).name(), "struct_col");
  EXPECT_EQ(schema_root.child(5).cudf_type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(schema_root.child(5).num_children(), 2);
  EXPECT_EQ(schema_root.child(5).child(0).name(), "struct_int");
  EXPECT_EQ(schema_root.child(5).child(0).cudf_type().id(), cudf::type_id::INT32);
  EXPECT_EQ(schema_root.child(5).child(1).name(), "struct_float");
  EXPECT_EQ(schema_root.child(5).child(1).cudf_type().id(), cudf::type_id::FLOAT32);
}

///////////////////////
// reader source tests

template <typename T>
struct ParquetReaderSourceTest : public ParquetReaderTest {};

TYPED_TEST_SUITE(ParquetReaderSourceTest, ByteLikeTypes);

TYPED_TEST(ParquetReaderSourceTest, BufferSourceTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(
        cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(cudf::host_span<T const>(
        reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }
}

TYPED_TEST(ParquetReaderSourceTest, BufferSourceArrayTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table, *table}));

  {
    auto spans = std::vector<cudf::host_span<T>>{
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }

  {
    auto spans = std::vector<cudf::host_span<T const>>{
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T const>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }
}

//////////////////////////////
// predicate pushdown tests

template <typename T, bool use_jit>
void filter_typed_test()
{
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterTyped.parquet");
  auto const written_table   = src.view();
  auto const col_name_0      = cudf::ast::column_name_reference("col0");
  auto const col_ref_0       = cudf::ast::column_reference(0);

  auto const test_predicate_pushdown = [&](cudf::ast::operation const& filter_expression,
                                           cudf::ast::operation const& ref_filter,
                                           cudf::size_type expected_total_row_groups,
                                           cudf::size_type expected_stats_filtered_row_groups) {
    // Expected result
    auto const predicate = cudf::compute_column(written_table, ref_filter);
    EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
      << "Predicate filter should return a boolean";
    auto const expected = cudf::apply_boolean_mask(written_table, *predicate);

    // Reading with Predicate Pushdown
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .filter(filter_expression)
        .use_jit_filter(use_jit);
    auto const result       = cudf::io::read_parquet(read_opts);
    auto const result_table = result.tbl->view();

    // Tests
    EXPECT_EQ(static_cast<int>(written_table.column(0).type().id()),
              static_cast<int>(result_table.column(0).type().id()))
      << "col0 type mismatch";

    // To make sure AST filters out some elements if row groups must be filtered
    if (expected_stats_filtered_row_groups < expected_total_row_groups) {
      EXPECT_LT(expected->num_rows(), written_table.num_rows());
    } else {
      EXPECT_LE(expected->num_rows(), written_table.num_rows());
    }
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result_table);
    EXPECT_EQ(result.metadata.num_input_row_groups, expected_total_row_groups);
    EXPECT_TRUE(result.metadata.num_row_groups_after_stats_filter.has_value());
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              expected_stats_filtered_row_groups);
    EXPECT_FALSE(result.metadata.num_row_groups_after_bloom_filter.has_value());
  };

  // The `literal_value` and stats should filter out 2 out of 4 row groups.
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 2;

    // Filtering AST
    auto literal_value = []() {
      if constexpr (cudf::is_timestamp<T>()) {
        return cudf::timestamp_scalar<T>(T(typename T::duration(10000)));  // i ∈ [0, 20,000)
      } else if constexpr (cudf::is_duration<T>()) {
        return cudf::duration_scalar<T>(T(10000));  // i ∈ [0, 20,000)
      } else if constexpr (std::is_same_v<T, cudf::string_view>) {
        return cudf::string_scalar("000010000");  // i ∈ [0-20,000)
      } else if constexpr (cudf::is_fixed_point<T>()) {
        return cudf::fixed_point_scalar<T>(typename T::rep{0},
                                           numeric::scale_type{0});  // i ∈ [-10,000, 10,000)
      } else {
        return cudf::numeric_scalar<T>(
          (100 - 100 * std::is_signed_v<T>));  // i/100 ∈ [-100, 100) or [0, 200)
      }
    }();

    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
    auto const ref_filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    test_predicate_pushdown(
      filter_expression, ref_filter, expected_total_row_groups, expected_stats_filtered_row_groups);
  }

  // The `literal_value` and stats should not filter any of the 4 row groups.
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 4;

    // Filtering AST
    auto literal_value = []() {
      if constexpr (cudf::is_timestamp<T>()) {
        return cudf::timestamp_scalar<T>(T(typename T::duration(20000)));
      } else if constexpr (cudf::is_duration<T>()) {
        return cudf::duration_scalar<T>(T(20000));
      } else if constexpr (std::is_same_v<T, cudf::string_view>) {
        return cudf::string_scalar("000020000");
      } else if constexpr (cudf::is_fixed_point<T>()) {
        return cudf::fixed_point_scalar<T>(typename T::rep{20000}, numeric::scale_type{0});
      } else {
        return cudf::numeric_scalar<T>(std::numeric_limits<T>::max());
      }
    }();

    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_name_0, literal);
    auto const ref_filter =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref_0, literal);
    test_predicate_pushdown(
      filter_expression, ref_filter, expected_total_row_groups, expected_stats_filtered_row_groups);
  }
}

template <typename T>
void filter_unary_operation_typed_test()
{
  std::mt19937 gen(0xd00dL);
  auto [src, filepath, null_count] = [&]() {
    auto constexpr num_rows            = num_ordered_rows;
    auto constexpr row_group_size_rows = num_rows / 4;
    auto _col0                         = testdata::ascending<T>().release();
    // Add nulls to col0
    [[maybe_unused]] std::bernoulli_distribution bn(0.7f);
    auto valids = cudf::detail::make_counting_transform_iterator(0, [&](int index) {
      return (index >= 2 * row_group_size_rows and index < 3 * row_group_size_rows) ? false : true;
    });
    auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);
    _col0->set_null_mask(std::move(null_mask), null_count);
    auto col0                = cudf::purge_nonempty_nulls(_col0->view());
    auto col1                = testdata::descending<T>();
    auto col2                = testdata::unordered<T>();
    auto const written_table = table_view{{col0->view(), col1, col2}};
    auto const filepath      = temp_env->get_temp_filepath("FilterUnaryOperationTyped.parquet");
    {
      cudf::io::table_input_metadata expected_metadata(written_table);
      expected_metadata.column_metadata[0].set_name("col0");
      expected_metadata.column_metadata[1].set_name("col1");
      expected_metadata.column_metadata[2].set_name("col2");

      cudf::io::parquet_writer_options const out_opts =
        cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
          .metadata(std::move(expected_metadata))
          .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
          .row_group_size_rows(row_group_size_rows);
      cudf::io::write_parquet(out_opts);
    }

    std::vector<std::unique_ptr<column>> columns;
    columns.push_back(std::move(col0));
    columns.push_back(col1.release());
    columns.push_back(col2.release());

    return std::tuple{cudf::table{std::move(columns)}, filepath, null_count};
  }();

  auto const written_table           = src.view();
  auto const test_predicate_pushdown = [&](cudf::ast::operation const& filter_expression,
                                           cudf::ast::operation const& ref_filter,
                                           cudf::size_type expected_total_row_groups,
                                           cudf::size_type expected_stats_filtered_row_groups,
                                           std::optional<cudf::size_type> expected_num_rows =
                                             std::nullopt) {
    // Expected result
    auto const predicate = cudf::compute_column(written_table, ref_filter);
    EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
      << "Predicate filter should return a boolean";
    auto const expected = cudf::apply_boolean_mask(written_table, *predicate);

    // JIT does not support nullness-dependent operators such as IS_NULL
    // Ref: https://github.com/rapidsai/cudf/issues/20177
    auto constexpr use_jit = false;

    // Reading with Predicate Pushdown
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .filter(filter_expression)
        .use_jit_filter(use_jit);
    auto const result       = cudf::io::read_parquet(read_opts);
    auto const result_table = result.tbl->view();

    // Tests
    if (expected_num_rows.has_value()) {
      EXPECT_EQ(expected->num_rows(), expected_num_rows.value());
      EXPECT_EQ(result_table.num_rows(), expected_num_rows.value());
    }
    EXPECT_EQ(static_cast<int>(written_table.column(0).type().id()),
              static_cast<int>(result_table.column(0).type().id()))
      << "col0 type mismatch";
    // To make sure AST filters out some elements if row groups must be filtered
    if (expected_stats_filtered_row_groups < expected_total_row_groups) {
      EXPECT_LT(expected->num_rows(), written_table.num_rows());
    } else {
      EXPECT_LE(expected->num_rows(), written_table.num_rows());
    }
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result_table);
    EXPECT_EQ(result.metadata.num_input_row_groups, expected_total_row_groups);
    EXPECT_TRUE(result.metadata.num_row_groups_after_stats_filter.has_value());
    EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
              expected_stats_filtered_row_groups);
    EXPECT_FALSE(result.metadata.num_row_groups_after_bloom_filter.has_value());
  };

  auto const col_name_0 = cudf::ast::column_name_reference("col0");
  auto const col_ref_0  = cudf::ast::column_reference(0);

  // Unary operation `IS_NULL` should filter all but one row group and yield exactly `null_count`
  // rows
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 1;

    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_name_0);
    auto const ref_filter = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);
    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_stats_filtered_row_groups,
                            null_count);
  }

  // Unary operation `NOT(IS_NULL)` should filter all but one row group and yield exactly `num_rows
  // - null_count` rows
  {
    auto constexpr expected_total_row_groups          = 4;
    auto constexpr expected_stats_filtered_row_groups = 3;

    auto const is_null_expr = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_name_0);
    auto const filter_expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, is_null_expr);
    auto const is_null_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);
    auto const ref_filter = cudf::ast::operation(cudf::ast::ast_operator::NOT, is_null_ref_expr);

    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_stats_filtered_row_groups,
                            num_ordered_rows - null_count);
  }

  {
    auto constexpr expected_total_row_groups = 4;

    // Filtering AST
    auto literal_value = []() {
      if constexpr (cudf::is_timestamp<T>()) {
        return cudf::timestamp_scalar<T>(T(typename T::duration(10000)));  // i ∈ [0, 20,000)
      } else if constexpr (cudf::is_duration<T>()) {
        return cudf::duration_scalar<T>(T(10000));  // i ∈ [0, 20,000)
      } else if constexpr (std::is_same_v<T, cudf::string_view>) {
        return cudf::string_scalar("000010000");  // i ∈ [0-20,000)
      } else if constexpr (cudf::is_fixed_point<T>()) {
        return cudf::fixed_point_scalar<T>(typename T::rep{0},
                                           numeric::scale_type{0});  // i ∈ [-10,000, 10,000)
      } else {
        return cudf::numeric_scalar<T>(
          (100 - 100 * std::is_signed_v<T>));  // i/100 ∈ [-100, 100) or [0, 200)
      }
    }();

    auto const literal   = cudf::ast::literal(literal_value);
    auto const expr1     = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
    auto const not_expr1 = cudf::ast::operation(cudf::ast::ast_operator::NOT, expr1);
    auto const expr2     = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_name_0);

    auto const ref_expr1 = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    auto const ref_not_expr1 = cudf::ast::operation(cudf::ast::ast_operator::NOT, ref_expr1);
    auto const ref_expr2     = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);

    // col0 < 100 AND IS_NULL(col0)
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr2);
    auto ref_filter =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ref_expr1, ref_expr2);
    auto constexpr expected_filtered_row_groups_with_unary_and = 1;
    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_filtered_row_groups_with_unary_and);

    // NOT(col0 < 100) OR IS_NULL(col0)
    filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, not_expr1, expr2);
    ref_filter =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, ref_not_expr1, ref_expr2);
    // For signed numeric types, RGs 1,2,3 pass. Otherwise, RGs 2,3 pass
    auto constexpr expected_filtered_row_groups_with_unary_or =
      (cudf::is_numeric<T>() and cudf::is_signed<T>()) ? 3 : 2;
    test_predicate_pushdown(filter_expression,
                            ref_filter,
                            expected_total_row_groups,
                            expected_filtered_row_groups_with_unary_or);
  }
}

template <typename DecimalType>
void decimal_stats_filter_test()
{
  using RepType = typename DecimalType::rep;

  auto constexpr num_input_row_groups = 3;

  auto const filepath = temp_env->get_temp_filepath("DecimalStatsFilter.parquet");

  for (auto const scale :
       {numeric::scale_type{-5}, numeric::scale_type{0}, numeric::scale_type{3}}) {
    {
      auto const rg0 = cudf::test::fixed_point_column_wrapper<RepType>(
        {RepType{100}, RepType{0}, RepType{200}}, {true, false, true}, scale);
      auto const rg1 = cudf::test::fixed_point_column_wrapper<RepType>(
        {RepType{-50}, RepType{300}, RepType{0}}, {true, true, false}, scale);
      auto const rg2 =
        cudf::test::fixed_point_column_wrapper<RepType>({RepType{-600}, RepType{-400}}, scale);
      auto const t0 = cudf::table_view{{rg0}};
      auto const t1 = cudf::table_view{{rg1}};
      auto const t2 = cudf::table_view{{rg2}};

      auto const options =
        cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
          .metadata(cudf::io::table_input_metadata(t0))
          .build();

      cudf::io::chunked_parquet_writer writer(options);
      writer.write(t0);
      writer.write(t1);
      writer.write(t2);
      writer.close();
    }

    // Verify Parquet physical type for the decimal column is as expected
    {
      auto const meta  = cudf::io::read_parquet_metadata(cudf::io::source_info{filepath});
      auto const& root = meta.schema().root();
      ASSERT_GE(root.num_children(), 1);
      auto const& col_schema = root.child(0);
      if constexpr (std::is_same_v<DecimalType, numeric::decimal32>) {
        EXPECT_EQ(col_schema.type(), cudf::io::parquet::Type::INT32);
      } else if constexpr (std::is_same_v<DecimalType, numeric::decimal64>) {
        EXPECT_EQ(col_schema.type(), cudf::io::parquet::Type::INT64);
      } else if constexpr (std::is_same_v<DecimalType, numeric::decimal128>) {
        EXPECT_EQ(col_schema.type(), cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY);
      }
    }

    // Helper function to test predicate pushdown for decimal types
    auto const test_predicate_pushdown = [&](cudf::ast::operation const& filter,
                                             cudf::size_type expected_filtered_row_groups,
                                             cudf::size_type expected_num_rows) {
      auto const options =
        cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
          .filter(filter)
          .build();

      auto const result = cudf::io::read_parquet(options);

      EXPECT_EQ(result.metadata.num_input_row_groups, num_input_row_groups);
      EXPECT_TRUE(result.metadata.num_row_groups_after_stats_filter.has_value());
      EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(),
                expected_filtered_row_groups);
      EXPECT_EQ(result.tbl->num_rows(), expected_num_rows);
    };

    auto const col_ref = cudf::ast::column_reference(0);

    // Filter: col0 >= 100 AND col0 <= 200
    {
      auto scalar_100        = cudf::fixed_point_scalar<DecimalType>(RepType{100}, scale);
      auto scalar_200        = cudf::fixed_point_scalar<DecimalType>(RepType{200}, scale);
      auto const literal_100 = cudf::ast::literal(scalar_100);
      auto const literal_200 = cudf::ast::literal(scalar_200);
      auto const col_ge_100 =
        cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal_100);
      auto const col_le_200 =
        cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal_200);
      auto const filter =
        cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, col_ge_100, col_le_200);

      // RGs 0, 1 pass
      test_predicate_pushdown(filter, 2, 2);
    }

    // Filter: col0 >= -550 AND col0 <= -450
    {
      auto scalar_neg_550        = cudf::fixed_point_scalar<DecimalType>(RepType{-550}, scale);
      auto scalar_neg_450        = cudf::fixed_point_scalar<DecimalType>(RepType{-450}, scale);
      auto const literal_neg_550 = cudf::ast::literal(scalar_neg_550);
      auto const literal_neg_450 = cudf::ast::literal(scalar_neg_450);
      auto const col_ge_neg_550 =
        cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal_neg_550);
      auto const col_le_neg_450 =
        cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal_neg_450);
      auto const filter =
        cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, col_ge_neg_550, col_le_neg_450);

      // RG 2 passes
      test_predicate_pushdown(filter, 1, 0);
    }

    // Filter: col0 >= 700 AND col0 <= 900 — matches no row groups
    {
      auto scalar_700        = cudf::fixed_point_scalar<DecimalType>(RepType{700}, scale);
      auto scalar_900        = cudf::fixed_point_scalar<DecimalType>(RepType{900}, scale);
      auto const literal_700 = cudf::ast::literal(scalar_700);
      auto const literal_900 = cudf::ast::literal(scalar_900);
      auto const col_ge_700 =
        cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal_700);
      auto const col_le_900 =
        cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal_900);
      auto const filter =
        cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, col_ge_700, col_le_900);

      test_predicate_pushdown(filter, 0, 0);
    }

    // Filter: col0 == -400
    {
      auto scalar_neg_400        = cudf::fixed_point_scalar<DecimalType>(RepType{-400}, scale);
      auto const literal_neg_400 = cudf::ast::literal(scalar_neg_400);
      auto const filter =
        cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal_neg_400);

      // RG 2 passes
      test_predicate_pushdown(filter, 1, 1);
    }

    // Filter: col0 >= -100 AND col0 <= 250
    {
      auto scalar_neg_100        = cudf::fixed_point_scalar<DecimalType>(RepType{-100}, scale);
      auto scalar_250            = cudf::fixed_point_scalar<DecimalType>(RepType{250}, scale);
      auto const literal_neg_100 = cudf::ast::literal(scalar_neg_100);
      auto const literal_250     = cudf::ast::literal(scalar_250);
      auto const col_ge =
        cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal_neg_100);
      auto const col_le =
        cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal_250);
      auto const filter =
        cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, col_ge, col_le);

      // RGs 0, 1 pass
      test_predicate_pushdown(filter, 2, 3);
    }

    // Filter: col0 == 100 OR col0 == -50
    {
      auto scalar_100           = cudf::fixed_point_scalar<DecimalType>(RepType{100}, scale);
      auto scalar_neg_50        = cudf::fixed_point_scalar<DecimalType>(RepType{-50}, scale);
      auto const literal_100    = cudf::ast::literal(scalar_100);
      auto const literal_neg_50 = cudf::ast::literal(scalar_neg_50);
      auto const eq_100 =
        cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal_100);
      auto const eq_neg_50 =
        cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal_neg_50);
      auto const filter =
        cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, eq_100, eq_neg_50);

      // RGs 0, 1 pass
      test_predicate_pushdown(filter, 2, 2);
    }

    // Large value filter only for decimal128 (overflows smaller rep types)
    if constexpr (std::is_same_v<DecimalType, numeric::decimal128>) {
      auto const big_val = (static_cast<__int128_t>(1) << 70) + 1234;
      auto scalar_val    = cudf::fixed_point_scalar<DecimalType>(big_val, scale);
      auto const lit     = cudf::ast::literal(scalar_val);
      auto const filter  = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref, lit);

      // No RGs pass
      test_predicate_pushdown(filter, 0, 0);
    }
  }
}

template <typename T>
struct ParquetPredicatePushdownTestAST : public ParquetReaderTest {};
TYPED_TEST_SUITE(ParquetPredicatePushdownTestAST, SupportedTestTypesAST);

TYPED_TEST(ParquetPredicatePushdownTestAST, FilterTyped)
{
  filter_typed_test<TypeParam, false>();
  filter_unary_operation_typed_test<TypeParam>();
  if constexpr (cudf::is_fixed_point<TypeParam>()) { decimal_stats_filter_test<TypeParam>(); }
}

template <typename T>
struct ParquetPredicatePushdownTestJIT : public ParquetReaderTest {};
TYPED_TEST_SUITE(ParquetPredicatePushdownTestJIT, SupportedTestTypesJIT);

TYPED_TEST(ParquetPredicatePushdownTestJIT, FilterTyped)
{
  filter_typed_test<TypeParam, true>();
  // JIT does not support decimals and nullness-dependent operators (IS_NULL) so we can't test:
  // `filter_unary_operation_typed_test<TypeParam>()` and `decimal_stats_filter_test<TypeParam>()`.
  // Refs: https://github.com/rapidsai/cudf/issues/20177 and
  // https://github.com/rapidsai/cudf/issues/21584
}

TEST_P(ParquetDecompressionTest, RoundTripBasic)
{
  auto const compression_type = std::get<1>(GetParam());

  srand(31337);
  // Exercises multiple rowgroups
  auto expected = create_compressible_fixed_table<int>(4, 12345, 3, true);

  // Use a host buffer for faster I/O
  std::vector<char> buffer;
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, *expected)
      .compression(compression_type);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

INSTANTIATE_TEST_CASE_P(Nvcomp,
                        ParquetDecompressionTest,
                        ::testing::Combine(::testing::Values("NVCOMP"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::LZ4,
                                                             cudf::io::compression_type::ZSTD)));

INSTANTIATE_TEST_CASE_P(DeviceInternal,
                        ParquetDecompressionTest,
                        ::testing::Combine(::testing::Values("DEVICE_INTERNAL"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY)));

INSTANTIATE_TEST_CASE_P(Host,
                        ParquetDecompressionTest,
                        ::testing::Combine(::testing::Values("HOST"),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY,
                                                             cudf::io::compression_type::ZSTD)));

//////////////////////
// wide tables tests

// The test below requires several minutes to complete with memcheck, thus it is disabled by
// default.
TEST_F(ParquetReaderTest, DISABLED_ListsWideTable)
{
  auto constexpr num_rows = 2;
  auto constexpr num_cols = 26'755;  // for slightly over 2B keys
  auto constexpr seed     = 0xceed;

  std::mt19937 engine{seed};

  auto list_list       = make_parquet_list_list_col<int32_t>(0, num_rows, 1, 1, false);
  auto list_list_nulls = make_parquet_list_list_col<int32_t>(0, num_rows, 1, 1, true);

  // switch between nullable and non-nullable
  std::vector<cudf::column_view> cols(num_cols);
  bool with_nulls = false;
  std::generate_n(cols.begin(), num_cols, [&]() {
    auto const view = with_nulls ? list_list_nulls->view() : list_list->view();
    with_nulls      = not with_nulls;
    return view;
  });

  cudf::table_view expected(cols);

  // Use a host buffer for faster I/O
  std::vector<char> buffer;
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected).build();
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}});
  auto const [result, _] = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

//////////////////////////////////////////
// row bounds and predicate pushdown tests

template <bool use_jit>
void row_bounds_and_filter_test()
{
  auto constexpr num_files                = 3;
  auto constexpr num_rows_per_file        = 100'000;
  auto constexpr num_row_groups_per_table = 5;
  auto constexpr total_num_rows           = num_rows_per_file * num_files;
  auto constexpr rows_per_row_group       = num_rows_per_file / num_row_groups_per_table;

  // Table with single col of ascending int64 values
  auto int64_data = std::vector<int64_t>(num_rows_per_file);
  std::iota(int64_data.begin(), int64_data.end(), 0);
  auto const int64_col = column_wrapper<int64_t>{
    int64_data.begin(), int64_data.end(), cudf::test::iterators::no_nulls()};
  cudf::table_view const written_table({int64_col});

  // Write to parquet
  auto const filepath = temp_env->get_temp_filepath("RowBoundsAndFilter.parquet");
  {
    cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(rows_per_row_group)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .build();
    cudf::io::write_parquet(out_opts);
  }

  // int64 data for expected table
  auto expected_int64_data = std::vector<int64_t>{};
  expected_int64_data.reserve(total_num_rows);
  for (auto i = 0; i < num_files; i++) {
    expected_int64_data.insert(expected_int64_data.end(), int64_data.cbegin(), int64_data.cend());
  }

  // Helper function to read parquet data
  auto const read_parquet_table =
    [&](auto const& filter_expression, auto rows_to_skip, auto rows_to_read) {
      auto const int64_col_row_bounded = column_wrapper<int64_t>{
        expected_int64_data.begin() + std::min(total_num_rows, rows_to_skip),
        expected_int64_data.begin() + std::min(total_num_rows, rows_to_skip + rows_to_read),
        cudf::test::iterators::no_nulls()};
      cudf::table_view const expected_row_bounded({int64_col_row_bounded});
      auto predicate = cudf::compute_column(expected_row_bounded, filter_expression);
      auto expected  = cudf::apply_boolean_mask(expected_row_bounded, *predicate);

      auto const in_opts = cudf::io::parquet_reader_options::builder(
                             cudf::io::source_info{std::vector<std::string>{num_files, filepath}})
                             .filter(filter_expression)
                             .skip_rows(rows_to_skip)
                             .num_rows(rows_to_read)
                             .use_jit_filter(use_jit)
                             .build();
      return std::tuple{cudf::io::read_parquet(in_opts), std::move(expected)};
    };

  // Filtering AST - table[0] >= 40'000
  {
    auto constexpr rows_to_skip = 30'000;
    auto constexpr rows_to_read = 40'000;

    auto literal_value = cudf::numeric_scalar<int64_t>(40'000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 3);  // RGs: {1,2,3},{},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 2);  // RGs: {2,3},{},{}
  }

  // Filtering AST - table[0] < 20'000 but skipping 30'000 rows (empty table)
  {
    auto constexpr rows_to_skip = 30'000;
    auto constexpr rows_to_read = 40'000;

    auto literal_value = cudf::numeric_scalar<int64_t>(20'000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal, col_ref_0);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 3);  // RGs: {1,2,3},{},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 0);  // RGs: {},{},{}
  }

  // Filtering AST - table[0] <= 100'000 but skipping 301'000 rows (empty table)
  {
    auto constexpr rows_to_skip = 301'000;
    auto constexpr rows_to_read = 1'000;

    auto literal_value = cudf::numeric_scalar<int64_t>(100'000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref_0, literal);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 0);  // RGs: {},{},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 0);  // RGs: {},{},{}
  }

  // Filtering AST - table[0] >= 70000 and table[0] < 120000
  {
    auto constexpr rows_to_skip = 130'000;
    auto constexpr rows_to_read = 100'000;

    // Filtering AST - table[0] < 4000
    auto literal_value  = cudf::numeric_scalar<int64_t>(70'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(100'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 6);  // RGs: {}, {1,2,3,4},{0,1}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 2);  // RGs: {},{4,5},{}
  }

  // Filtering AST - table[0] < 40000 or table[0] >= 80000
  {
    auto constexpr rows_to_skip = 120'000;
    auto constexpr rows_to_read = 190'000;  // Larger than the total number of rows in all files

    auto literal_value  = cudf::numeric_scalar<int64_t>(40'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(80'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 9);  // RGs: {},{1,2,3,4},{0,1,2,3,4}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 5);  // RGs: {},{1,4},{0,1,4}
  }

  // Filtering AST - table[0] >= 40000 and table[0] < 80000
  {
    auto constexpr rows_to_skip = 110'000;
    auto constexpr rows_to_read = 80'000;

    auto literal_value  = cudf::numeric_scalar<int64_t>(40'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(80'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 5);  // RGs: {},{0,1,2,3,4},{}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 2);  // RGs: {},{2,3},{}
  }

  // Filtering AST - table[0] < 40000 or table[0] >= 80000
  {
    auto constexpr rows_to_skip = 110'000;
    auto constexpr rows_to_read = 80'000;

    auto literal_value  = cudf::numeric_scalar<int64_t>(40'000);
    auto literal        = cudf::ast::literal(literal_value);
    auto literal_value2 = cudf::numeric_scalar<int64_t>(80'000);
    auto literal2       = cudf::ast::literal(literal_value2);

    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);

    auto const [table_with_metadata, expected] =
      read_parquet_table(filter_expression, rows_to_skip, rows_to_read);

    EXPECT_EQ(expected->num_rows(), table_with_metadata.tbl->num_rows());
    EXPECT_EQ(expected->num_columns(), table_with_metadata.tbl->num_columns());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), table_with_metadata.tbl->view());

    auto const& metadata = table_with_metadata.metadata;
    EXPECT_EQ(metadata.num_input_row_groups, 5);  // RGs: {}, {0,1,2,3,4}, {}
    EXPECT_TRUE(metadata.num_row_groups_after_stats_filter.has_value() and
                metadata.num_row_groups_after_stats_filter.value() == 3);  // RGs: {},{0,1,4},{}
  }
}

TEST_F(ParquetReaderTest, RowBoundsAndFilter) { row_bounds_and_filter_test<false>(); }

TEST_F(ParquetReaderTest, RowBoundsAndFilterJIT) { row_bounds_and_filter_test<true>(); }

//////////////////////////////////////////
// device read async tests

TEST_F(ParquetReaderTest, DeviceReadAsyncThrows)
{
  // Create a simple parquet file in memory
  auto col0           = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}};
  auto table_to_write = table_view{{col0}};

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options write_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer}, table_to_write);
  cudf::io::write_parquet(write_args);

  // Create our throwing datasource
  auto throwing_source = std::make_unique<cudf::test::ThrowingDeviceReadDatasource>(out_buffer);
  cudf::io::source_info source_info(throwing_source.get());

  // Try to read the parquet file - this should either succeed or propagate AsyncException
  // from device_read_async.
  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(source_info);
  try {
    cudf::io::read_parquet(read_args);
    // Test passes if no exception is thrown
  } catch (cudf::test::AsyncException const&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (std::exception const& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

TEST_F(ParquetReaderTest, DeviceWriteAsyncThrows)
{
  // Create a simple table to write
  auto col0           = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}};
  auto table_to_write = table_view{{col0}};

  auto throwing_sink = std::make_unique<cudf::test::ThrowingDeviceWriteDataSink>();

  cudf::io::parquet_writer_options write_args = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info{throwing_sink.get()}, table_to_write);

  // The write_parquet call should either succeed or throw AsyncException.
  try {
    cudf::io::write_parquet(write_args);
    // Test passes if no exception is thrown
  } catch (cudf::test::AsyncException const&) {
    // Test passes if AsyncException is thrown (expected test exception)
  } catch (std::exception const& e) {
    // Test fails if any other exception is thrown
    FAIL() << "Unexpected exception thrown: " << e.what();
  }
}

//////////////////////////////////////////
// row group output ordering tests
//
// See the @note on read_parquet in
// cpp/include/cudf/io/parquet.hpp for the documented behavior.

TEST_F(ParquetReaderTest, RowGroupOrderAscending)
{
  auto const f0        = make_ordered_rg_source(0, 4);
  auto const selection = std::vector<std::vector<cudf::size_type>>{{0, 1, 3}};
  auto const expected  = build_expected_ordered_table(std::vector{&f0}, selection);

  auto const opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{f0.filepath})
                      .row_groups(selection)
                      .build();
  auto const got = cudf::io::read_parquet(opts).tbl;
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, got->view());
}

TEST_F(ParquetReaderTest, RowGroupOrderNonAscending)
{
  auto const f0        = make_ordered_rg_source(0, 4);
  auto const selection = std::vector<std::vector<cudf::size_type>>{{3, 0, 2}};
  auto const expected  = build_expected_ordered_table(std::vector{&f0}, selection);

  auto const opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{f0.filepath})
                      .row_groups(selection)
                      .build();
  auto const got = cudf::io::read_parquet(opts).tbl;
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, got->view());
}

TEST_F(ParquetReaderTest, RowGroupOrderRepeatedIndices)
{
  auto const f0        = make_ordered_rg_source(0, 3);
  auto const selection = std::vector<std::vector<cudf::size_type>>{{0, 2, 0, 1, 0}};
  auto const expected  = build_expected_ordered_table(std::vector{&f0}, selection);

  auto const opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{f0.filepath})
                      .row_groups(selection)
                      .build();
  auto const got = cudf::io::read_parquet(opts).tbl;
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, got->view());
}

TEST_F(ParquetReaderTest, RowGroupOrderEmptySourceVector)
{
  auto const f0 = make_ordered_rg_source(0, 2);
  auto const f1 = make_ordered_rg_source(1, 2);
  auto const f2 = make_ordered_rg_source(2, 2);

  // Middle source contributes nothing; outer source order must be preserved.
  auto const selection = std::vector<std::vector<cudf::size_type>>{{0, 1}, {}, {0}};
  auto const expected  = build_expected_ordered_table(std::vector{&f0, &f1, &f2}, selection);

  auto const opts =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{std::vector<std::string>{f0.filepath, f1.filepath, f2.filepath}})
      .row_groups(selection)
      .build();
  auto const got = cudf::io::read_parquet(opts).tbl;
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, got->view());
}

TEST_F(ParquetReaderTest, RowGroupOrderMultiSourceMix)
{
  auto const f0 = make_ordered_rg_source(0, 4);
  auto const f1 = make_ordered_rg_source(1, 4);

  // Within each source, user-given order is preserved (non-ascending for f0).
  auto const selection = std::vector<std::vector<cudf::size_type>>{{0, 3}, {0, 1}};
  auto const expected  = build_expected_ordered_table(std::vector{&f0, &f1}, selection);

  auto const opts = cudf::io::parquet_reader_options::builder(
                      cudf::io::source_info{std::vector<std::string>{f0.filepath, f1.filepath}})
                      .row_groups(selection)
                      .build();
  auto const got = cudf::io::read_parquet(opts).tbl;
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, got->view());
}

TEST_F(ParquetReaderTest, RowGroupOrderUnsetMultiSource)
{
  auto const f0 = make_ordered_rg_source(0, 3);
  auto const f1 = make_ordered_rg_source(1, 2);

  // Without .row_groups(...): all RGs in source order, then on-disk order.
  auto const expected_order = std::vector<std::vector<cudf::size_type>>{{0, 1, 2}, {0, 1}};
  auto const expected       = build_expected_ordered_table(std::vector{&f0, &f1}, expected_order);

  auto const opts = cudf::io::parquet_reader_options::builder(
                      cudf::io::source_info{std::vector<std::string>{f0.filepath, f1.filepath}})
                      .build();
  auto const got = cudf::io::read_parquet(opts).tbl;
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, got->view());
}

TEST_F(ParquetReaderTest, RowGroupOrderStatsPushdown)
{
  // Values in f0 RGs span [0,100), [100,200), [200,300), [300,400);
  // values in f1 RGs span [10000,10100), [10100,10200), [10200,10300).
  // Predicate `col < 300` keeps every row in f0 RG{0,1,2} and prunes f0 RG3
  // and all of f1 at the row-group statistics level. Because every surviving
  // row passes the predicate, the row-level result equals the whole-RG slices,
  // letting the test focus purely on ordering of survivors.
  auto const f0 = make_ordered_rg_source(0, 4);
  auto const f1 = make_ordered_rg_source(1, 3);

  auto literal_value     = cudf::numeric_scalar<int32_t>(300);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref           = cudf::ast::column_reference(0);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);

  // Survivors must keep relative order: f0 RG0, f0 RG1, f0 RG2.
  auto const survivors = std::vector<std::vector<cudf::size_type>>{{0, 1, 2}, {}};
  auto const expected  = build_expected_ordered_table(std::vector{&f0, &f1}, survivors);

  auto const opts = cudf::io::parquet_reader_options::builder(
                      cudf::io::source_info{std::vector<std::string>{f0.filepath, f1.filepath}})
                      .filter(filter_expression)
                      .build();
  auto const result = cudf::io::read_parquet(opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, result.tbl->view());

  // Sanity: stats-based pruning actually fired and shrank the surviving set.
  ASSERT_TRUE(result.metadata.num_row_groups_after_stats_filter.has_value());
  EXPECT_EQ(result.metadata.num_row_groups_after_stats_filter.value(), 3);
}

//////////////////////////////////////////
// byte bounds tests

TEST_F(ParquetReaderTest, ByteBoundsOptions)
{
  using T             = cudf::string_view;
  auto const filepath = create_parquet_typed_with_stats<T>("ByteBounds.parquet").second;

  // Test options combinations

  // If skip_bytes is zero, we can set other options normally.
  EXPECT_NO_THROW(cudf::io::parquet_reader_options::builder(
                    cudf::io::source_info{std::vector<std::string>{2, filepath}})
                    .skip_bytes(0)
                    .num_rows(10)
                    .skip_rows(10)
                    .build());

  // Cannot set skip_bytes/num_bytes and row_groups together
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .skip_bytes(100)
                     .row_groups({{1}})
                     .build());
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .num_bytes(100)
                     .row_groups({{1}})
                     .build());

  // Cannot set skip_bytes/num_bytes and skip_rows/num_rows together
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .skip_bytes(100)
                     .num_rows(100)
                     .skip_rows(0)
                     .build());
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .skip_rows(100)
                     .num_bytes(100)
                     .num_rows(100)
                     .build());

  // skip_bytes/num_bytes only supported for single source case
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(
                     cudf::io::source_info{std::vector<std::string>{2, filepath}})
                     .skip_bytes(100)
                     .build());
  EXPECT_ANY_THROW(cudf::io::parquet_reader_options::builder(
                     cudf::io::source_info{std::vector<std::string>{2, filepath}})
                     .num_bytes(400)
                     .build());
}

TEST_F(ParquetReaderTest, ByteBoundsOnly)
{
  using T                      = cudf::string_view;
  auto const [table, filepath] = create_parquet_typed_with_stats<T>("ByteBounds.parquet");

  // Note: Currently the row groups start at the following byte offsets: 4, 75224, 150332, 225561
  // `skip_bytes` and `num_bytes` may need to be adjusted in the future if this test suddenly starts
  // failing.

  // Only read row group 0 as only it will start in [0, 1000) byte range
  {
    auto constexpr num_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .num_bytes(num_bytes)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{0}})
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Skip row group 0 as it won't start in [1000, inf) byte range
  {
    auto constexpr skip_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{1, 2, 3}})
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Only read row group 1 as only it starts in [50000, 100000) byte range
  {
    auto constexpr skip_bytes = 50000;
    auto constexpr num_bytes  = 50000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .num_bytes(num_bytes)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{1}})
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }
}

TEST_F(ParquetReaderTest, ByteBoundsAndFilters)
{
  using T                      = uint64_t;
  auto const [table, filepath] = create_parquet_typed_with_stats<T>("ByteBounds.parquet");

  // Note: Currently the row groups start at the following byte offsets: 4, 2040, 4048, 6032
  // `skip_bytes` and `num_bytes` may need to be adjusted in the future if this test suddenly starts
  // failing.

  // Only read row group 0 as only it will start in [0, 1000) byte range
  {
    auto literal_value = cudf::numeric_scalar<T>(1000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref_0, literal);

    auto constexpr num_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .num_bytes(num_bytes)
                           .filter(filter_expression)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{0}})
        .filter(filter_expression)
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Skip row group 0 using byte range and row group 1 using the filter expression
  {
    auto literal_value = cudf::numeric_scalar<T>(12000);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, literal);

    auto constexpr skip_bytes = 1000;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .filter(filter_expression)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{2, 3}})
        .filter(filter_expression)
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  // Only read row group 1 as only it starts in [1500, 3000) byte range
  {
    auto col_ref_0     = cudf::ast::column_reference(0);
    auto literal_value = cudf::numeric_scalar<T>(8000);
    auto literal       = cudf::ast::literal(literal_value);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

    auto constexpr skip_bytes = 1500;
    auto constexpr num_bytes  = 1500;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_bytes(skip_bytes)
                           .num_bytes(num_bytes)
                           .filter(filter_expression)
                           .build();
    auto const read = cudf::io::read_parquet(in_opts).tbl;

    auto const expected_in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .row_groups({{1}})
        .filter(filter_expression)
        .build();
    auto const expected = cudf::io::read_parquet(expected_in_opts).tbl;

    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }
}

TEST_F(ParquetReaderTest, TableTooLargeOverflows)
{
  using T                             = bool;
  constexpr int64_t per_file_num_rows = std::numeric_limits<cudf::size_type>::max() / 2 + 1000;
  static_assert(per_file_num_rows <= std::numeric_limits<cudf::size_type>::max(),
                "Number of rows per file should be less than size_type::max()");
  static_assert(2 * per_file_num_rows > std::numeric_limits<cudf::size_type>::max(),
                "Twice number of rows per file should be greather than size_type::max()");
  auto value  = cuda::make_constant_iterator(true);
  auto column = cudf::test::fixed_width_column_wrapper<T>(value, value + per_file_num_rows);

  auto filepath = temp_env->get_temp_filepath("TableTooLargeOverflows.parquet");
  {
    auto sink = cudf::io::sink_info{filepath};
    auto options =
      cudf::io::parquet_writer_options::builder(sink, cudf::table_view{{column}}).build();
    std::ignore = cudf::io::write_parquet(options);
  }
  std::vector<std::string> files{{filepath, filepath}};
  auto source                 = cudf::io::source_info(files);
  auto metadata               = cudf::io::read_parquet_metadata(source);
  auto const num_rows_to_read = metadata.num_rows() - 1000;
  EXPECT_EQ(metadata.num_rows(), per_file_num_rows * 2);
  auto options = cudf::io::parquet_reader_options::builder(source)
                   .num_rows(num_rows_to_read)
                   .skip_rows(10)
                   .build();

  EXPECT_THROW(cudf::io::read_parquet(options), std::overflow_error);
  auto reader = cudf::io::chunked_parquet_reader(0, 0, options);
  int64_t num_rows_read{0};
  while (reader.has_next()) {
    auto chunk = reader.read_chunk();
    num_rows_read += chunk.tbl->num_rows();
  }
  EXPECT_EQ(num_rows_read, num_rows_to_read);
}

TEST_F(ParquetReaderTest, LateBindSourceInfo)
{
  srand(31337);
  auto expected = create_random_fixed_table<int>(4, 4, false);

  auto filepath = temp_env->get_temp_filepath("LateBindSourceInfo.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{});

  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);

  read_opts.set_source(cudf::io::source_info{filepath});

  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected->view());
}

TEST_F(ParquetReaderTest, InvalidFooterMagic)
{
  auto const expected = create_random_fixed_table<int>(4, 4, false);

  std::vector<char> buffer;
  cudf::io::write_parquet(
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, *expected));

  constexpr std::array<char, 4> bad_magic{'B', 'A', 'D', '!'};
  ASSERT_GE(buffer.size(), bad_magic.size());
  for (size_t i = 0; i < bad_magic.size(); ++i) {
    buffer[buffer.size() - bad_magic.size() + i] = bad_magic[i];
  }

  auto const read_opts = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info{cudf::host_span<std::byte const>{
                             reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
                           .build();
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
}

TEST_F(ParquetReaderTest, CorruptPlainStringLengthThrows)
{
  // A PLAIN BYTE_ARRAY length that overruns the page must fail loudly instead of
  // silently decoding the corrupt tail as empty strings.
  auto const col = cudf::test::strings_column_wrapper{
    "unique_string_aaa", "unique_string_bbb", "unique_string_ccc", "unique_string_ddd"};
  auto const expected = table_view{{col}};

  std::vector<char> buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  // PLAIN BYTE_ARRAY layout is [4-byte little-endian length][string bytes]... So the 4 bytes
  // immediately preceding a known string value are that value's length prefix. Locate the third
  // string's bytes, then step back sizeof(int32_t) to land on its length prefix.
  std::string const target = "unique_string_ccc";
  auto const it = std::search(buffer.begin(), buffer.end(), target.begin(), target.end());
  ASSERT_NE(it, buffer.end());
  auto const length_pos = static_cast<size_t>(std::distance(buffer.begin(), it) - sizeof(int32_t));

  // Sanity check that we found the length prefix: it should hold the real string length.
  int32_t orig_len{};
  std::memcpy(&orig_len, buffer.data() + length_pos, sizeof(orig_len));
  ASSERT_EQ(orig_len, static_cast<int32_t>(target.size()));

  // Overwrite the length prefix with a value that runs past the end of the page. This mimics the
  // real-world corruption where a stray byte shifts the length window, producing a bogus length.
  // The offset preprocessor detects `string_offset + len > dict_size` and must raise
  // decode_error::STRING_DATA_OVERRUN rather than truncating and back-filling empty strings.
  int32_t const bad_len = 1'000'000;
  std::memcpy(buffer.data() + length_pos, &bad_len, sizeof(bad_len));

  auto const read_opts = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info{cudf::host_span<std::byte const>{
                             reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}})
                           .build();
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
}

TEST_F(ParquetReaderTest, DecimalTypeOption)
{
  auto const data = std::vector<int32_t>{1000, 2000, 3000, 4000, 5000};
  auto col        = cudf::test::fixed_point_column_wrapper<int32_t>(
    data.begin(), data.end(), numeric::scale_type{-2});

  auto expected_table = cudf::table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("DecimalTypeOption.parquet");

  // Write the table with DECIMAL32
  {
    auto options =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_table)
        .build();
    cudf::io::write_parquet(options);
  }

  // Read back with decimal_width set to DECIMAL128
  {
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .decimal_width(cudf::type_id::DECIMAL128)
                     .build();
    auto result = cudf::io::read_parquet(options);

    EXPECT_EQ(result.tbl->view().column(0).type().id(), cudf::type_id::DECIMAL128);
    EXPECT_EQ(result.tbl->view().column(0).type().scale(), -2);
  }
  // Read back with decimal_width set to DECIMAL64
  {
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .decimal_width(cudf::type_id::DECIMAL64)
                     .build();
    auto result = cudf::io::read_parquet(options);

    EXPECT_EQ(result.tbl->view().column(0).type().id(), cudf::type_id::DECIMAL64);
    EXPECT_EQ(result.tbl->view().column(0).type().scale(), -2);
  }
}

TEST_F(ParquetReaderTest, CaseInsensitiveColumnSelection)
{
  auto col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40, 50};
  cudf::table_view tbl{{col0, col1}};

  auto filepath =
    write_parquet_temp_file(tbl, "CaseInsensitiveColumnSelection.parquet", {"col0", "col1"});

  // Case-sensitive: "col0" is ignored
  {
    auto read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
    read_opts.set_column_names({"Col0"});
    // ignore_missing_columns defaults to true, so result should be empty
    auto result = cudf::io::read_parquet(read_opts);
    EXPECT_EQ(result.tbl->view().num_columns(), 0);
  }

  // Case-insensitive: "Col0" -> "col0"
  {
    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .case_sensitive_names(false)
                       .build();
    read_opts.set_column_names({"Col0"});
    auto result = cudf::io::read_parquet(read_opts);
    ASSERT_EQ(result.tbl->view().num_columns(), 1);
    // Output column name should be the file's original name "col0"
    EXPECT_EQ(result.metadata.schema_info[0].name, "col0");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);
  }

  // Case-insensitive: "COL0" -> "col0", "Col1" -> "col1"
  {
    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .case_sensitive_names(false)
                       .build();
    read_opts.set_column_names({"COL0", "Col1"});
    auto result = cudf::io::read_parquet(read_opts);
    ASSERT_EQ(result.tbl->view().num_columns(), 2);
    EXPECT_EQ(result.metadata.schema_info[0].name, "col0");
    EXPECT_EQ(result.metadata.schema_info[1].name, "col1");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), col1);
  }
}

TEST_F(ParquetReaderTest, DuplicateColumnSelection)
{
  auto col0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40, 50};
  auto col2 = cudf::test::strings_column_wrapper{"a", "b", "c", "d", "e"};
  cudf::table_view tbl{{col0, col1, col2}};

  auto filepath = write_parquet_temp_file(tbl, "DuplicateColumnSelection.parquet", {"a", "b", "A"});

  {
    auto const col_ref = cudf::ast::column_name_reference("A");
    auto scalar        = cudf::numeric_scalar<int32_t>(3, true);
    auto literal       = cudf::ast::literal(scalar);
    auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal);
    auto const predicate =
      cudf::test::fixed_width_column_wrapper<bool>{true, true, true, false, false}.release();
    auto const expected = cudf::apply_boolean_mask(tbl.select({0, 1}), predicate->view());

    auto const read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .case_sensitive_names(false)
        .column_names({"A", "b"})
        .filter(filter_expr)
        .build();
    auto const read = cudf::io::read_parquet(read_opts).tbl;
    CUDF_TEST_EXPECT_TABLES_EQUAL(read->view(), expected->view());
  }

  {
    auto const col_ref = cudf::ast::column_name_reference("A");
    auto scalar        = cudf::string_scalar("c", true);
    auto literal       = cudf::ast::literal(scalar);
    auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal);

    auto const read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .case_sensitive_names(false)
        .filter(filter_expr)
        .column_names({"A", "b"})
        .build();
    EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  }
}

template <typename T>
struct ParquetFilterPushdownTimestampPrecisions : public ParquetReaderTest {};
using SupportedTimestampTypes =
  cudf::test::Types<cudf::timestamp_ms, cudf::timestamp_us, cudf::timestamp_ns>;
TYPED_TEST_SUITE(ParquetFilterPushdownTimestampPrecisions, SupportedTimestampTypes);

TYPED_TEST(ParquetFilterPushdownTimestampPrecisions, AllPrecisions)
{
  using NativeTimestamp = TypeParam;
  using NativeDuration  = typename NativeTimestamp::duration;

  constexpr auto native_type        = cudf::type_to_id<NativeTimestamp>();
  constexpr auto num_rows           = 20000;
  constexpr auto rows_per_row_group = 5000;
  static_assert(num_rows % rows_per_row_group == 0,
                "num_rows must be a multiple of rows_per_row_group");

  constexpr auto tick_offset = int64_t{num_rows / 2} * 100;
  auto iter                  = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return NativeTimestamp{NativeDuration{i * 100 - tick_offset}}; });
  auto const ts_col =
    cudf::test::fixed_width_column_wrapper<NativeTimestamp>(iter, iter + num_rows).release();
  auto const written_table = cudf::table_view{{ts_col->view()}};

  // Write file at native precision
  auto const filepath = temp_env->get_temp_filepath("TimestampFilterAllPrec.parquet");
  {
    auto const out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .metadata(cudf::io::table_input_metadata(written_table))
        .row_group_size_rows(rows_per_row_group)
        .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP)
        .build();
    cudf::io::write_parquet(out_opts);
  }

  auto const dispatch_timestamp = [](cudf::type_id id, auto&& fn) {
    switch (id) {
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
        return fn.template operator()<cudf::timestamp_ms>();
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
        return fn.template operator()<cudf::timestamp_us>();
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
        return fn.template operator()<cudf::timestamp_ns>();
      default: CUDF_FAIL("Unexpected timestamp type");
    }
  };

  // Test filter pushdown with all output timestamp precisions
  for (auto const target_type : {cudf::type_id::TIMESTAMP_MILLISECONDS,
                                 cudf::type_id::TIMESTAMP_MICROSECONDS,
                                 cudf::type_id::TIMESTAMP_NANOSECONDS}) {
    // Compute filter threshold: value at row (num_rows / 2) in native ticks, converted to output
    // ticks. With the negative tick offset applied to the iterator, this threshold is 0
    auto const threshold_native = int64_t{num_rows / 2} * 100 - tick_offset;
    auto const scale            = static_cast<double>(cudf::io::detail::to_clockrate(target_type)) /
                       static_cast<double>(cudf::io::detail::to_clockrate(native_type));
    auto const threshold_output = static_cast<int64_t>(threshold_native * scale);

    // Build filter: ts >= threshold_output (in output precision)
    auto result = dispatch_timestamp(target_type, [&]<typename OutputTimestamp>() {
      using OutputDuration = typename OutputTimestamp::duration;
      auto scalar_val =
        cudf::timestamp_scalar<OutputTimestamp>(OutputTimestamp{OutputDuration{threshold_output}});
      auto literal = cudf::ast::literal(scalar_val);
      auto col_ref = cudf::ast::column_reference(0);
      auto filter_gte =
        cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);

      auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                         .timestamp_type(cudf::data_type{target_type})
                         .filter(filter_gte)
                         .build();
      return cudf::io::read_parquet(read_opts).tbl;
    });

    // Expect rows from written_table in range [num_rows / 2, num_rows) casted to the target
    // timestamp type
    EXPECT_EQ(result->view().column(0).type().id(), target_type);
    auto const written_table_chunk = cudf::split(written_table, {num_rows / 2}).back();
    auto const expected = cudf::cast(written_table_chunk.column(0), cudf::data_type{target_type});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(0), expected->view());
  }
}

struct ParquetStatsDecoder : cudf::io::parquet::detail::stats_caster_base {
  template <typename T>
  static T decode(std::initializer_list<uint8_t> bytes,
                  cudf::io::parquet::Type type = cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY)
  {
    return convert<T>(bytes.begin(), bytes.size(), type);
  }
};

TEST_F(ParquetReaderTest, DecodeVariableWidthDecimalStats)
{
  EXPECT_EQ(ParquetStatsDecoder::decode<int32_t>({0x64}), 100);
  EXPECT_EQ(ParquetStatsDecoder::decode<int32_t>({0x00, 0xc8}), 200);
  EXPECT_EQ(ParquetStatsDecoder::decode<int32_t>({0xfe, 0x0c}), -500);
  EXPECT_EQ(ParquetStatsDecoder::decode<int32_t>({0xff, 0xff, 0xfe, 0x0c}), -500);
  EXPECT_EQ(ParquetStatsDecoder::decode<int64_t>({0xfd, 0x44}), -700);

  using d128 = numeric::decimal128;
  EXPECT_EQ(ParquetStatsDecoder::decode<d128>({0xfd, 0x44}), d128{d128::rep{-700}});
  EXPECT_EQ(ParquetStatsDecoder::decode<d128::rep>({0xfe, 0x0c}), d128::rep{-500});
  EXPECT_EQ(ParquetStatsDecoder::decode<int32_t>({0xfe, 0x0c}, cudf::io::parquet::Type::BYTE_ARRAY),
            -500);
  EXPECT_THROW(ParquetStatsDecoder::decode<int32_t>({0xff, 0xff, 0xff, 0xfe, 0x0c}),
               cudf::logic_error);
}

TEST_F(ParquetReaderTest, SourceIndexColumn)
{
  auto constexpr num_rows = 3;
  auto col0               = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{num_rows});
  auto col1 = cudf::test::fixed_width_column_wrapper<int64_t>(
    cuda::counting_iterator<int64_t>{0}, cuda::counting_iterator<int64_t>{num_rows});
  auto table = cudf::table_view{{col0, col1}};

  auto filepath = temp_env->get_temp_filepath("SourceIndexNoFilter.parquet");
  cudf::io::write_parquet(
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table).build());

  auto const test_source_index_column = [&](auto num_sources) {
    auto const sources = std::vector<std::string>(num_sources, filepath);
    auto read_opts     = cudf::io::parquet_reader_options::builder(cudf::io::source_info(sources))
                       .prepend_source_index_column(true)
                       .build();
    auto const read = cudf::io::read_parquet(read_opts);
    EXPECT_EQ(read.tbl->num_columns(), table.num_columns() + 1);
    EXPECT_EQ(read.metadata.schema_info.front().name, "source_index");

    auto const src_index = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i / num_rows; });
    auto const expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      src_index, src_index + num_sources * num_rows);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(read.tbl->view().column(0), expected);
    // The data columns are the single source file tiled `num_sources` times
    auto const tiled = cudf::tile(table, num_sources);
    CUDF_TEST_EXPECT_TABLES_EQUAL(read.tbl->select({1, 2}), tiled->view());

    // Use filter `col0 < 2` so only rows {0, 1} from each source are read
    auto scalar  = cudf::numeric_scalar<cudf::size_type>(2);
    auto literal = cudf::ast::literal(scalar);
    auto col_ref = cudf::ast::column_reference(0);
    auto filter  = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);

    read_opts.set_filter(filter);
    auto const read_filtered = cudf::io::read_parquet(read_opts);
    auto const filtered_src_index =
      cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i / 2; });
    auto const filtered_expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      filtered_src_index, filtered_src_index + num_sources * 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(read_filtered.tbl->view().column(0), filtered_expected);
  };

  test_source_index_column(1);
  test_source_index_column(5);
}

TEST_F(ParquetReaderTest, SourceIndexSelectedRead)
{
  auto write_parquet = [](auto const& filepath, auto const& col) {
    auto constexpr rows_per_row_group = 2;
    auto const write_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath},
                                                                      cudf::table_view{{col}})
                              .row_group_size_rows(rows_per_row_group)
                              .max_page_size_rows(rows_per_row_group)
                              .max_page_fragment_size(rows_per_row_group)
                              .build();
    cudf::io::write_parquet(write_opts);
  };

  auto col0      = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  auto filepath0 = temp_env->get_temp_filepath("SourceIndexSelectedRead0.parquet");
  write_parquet(filepath0, col0);

  auto col1      = cudf::test::fixed_width_column_wrapper<int32_t>{10, 11, 12, 13};
  auto filepath1 = temp_env->get_temp_filepath("SourceIndexSelectedRead1.parquet");
  write_parquet(filepath1, col1);

  auto source = cudf::io::source_info{std::vector<std::string>{filepath0, filepath1}};

  auto expected_values = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 10, 11};
  auto expected_source = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 1, 1};
  auto expected        = cudf::table_view{{expected_source, expected_values}};

  // Test with row bounds
  {
    auto const read_opts = cudf::io::parquet_reader_options::builder(source)
                             .skip_rows(2)
                             .num_rows(4)
                             .prepend_source_index_column(true)
                             .build();
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::io::read_parquet(read_opts).tbl->view(), expected);
  }

  // Test with row group selection
  {
    auto const read_opts = cudf::io::parquet_reader_options::builder(source)
                             .row_groups({{1}, {0}})
                             .prepend_source_index_column(true)
                             .build();
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::io::read_parquet(read_opts).tbl->view(), expected);
  }
}

TEST_F(ParquetReaderTest, MismatchedSchemaFilterColumnCollision)
{
  // Different-type collision: source 0's double `price` lands on source 1's int64 `id`.
  auto const id_a    = column_wrapper<int64_t>{1, 2, 3};
  auto const price_a = column_wrapper<double>{50.0, 150.0, 75.0};
  cudf::table_view const table_a{{id_a, price_a}};
  auto const path_a =
    write_parquet_temp_file(table_a, "MismatchCollisionA.parquet", {"id", "price"}, {1, 2});

  auto const category_b = column_wrapper<cudf::string_view>{"x", "y", "z"};
  auto const id_b       = column_wrapper<int64_t>{1000, 1001, 1002};
  auto const price_b    = column_wrapper<double>{40.0, 200.0, 99.0};
  cudf::table_view const table_b{{category_b, id_b, price_b}};
  auto const path_b = write_parquet_temp_file(
    table_b, "MismatchCollisionB.parquet", {"category", "id", "price"}, {3, 1, 2});

  auto value  = cudf::numeric_scalar<double>(100.0);
  auto lit    = cudf::ast::literal(value);
  auto col    = cudf::ast::column_name_reference("price");
  auto filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col, lit);

  auto const exp_id    = column_wrapper<int64_t>{1, 3, 1000, 1002};
  auto const exp_price = column_wrapper<double>{50.0, 75.0, 40.0, 99.0};
  cudf::table_view const expected{{exp_id, exp_price}};

  {
    auto const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path_a, path_b}})
        .allow_mismatched_pq_schemas(true)
        .column_names({"id", "price"})
        .filter(filter)
        .build();
    auto const result = cudf::io::read_parquet(opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  }
  {
    auto const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path_a, path_b}})
        .allow_mismatched_pq_schemas(true)
        .column_field_ids({1, 2})
        .filter(filter)
        .build();
    auto const result = cudf::io::read_parquet(opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  }
}

TEST_F(ParquetReaderTest, MismatchedSchemaFilterSameTypeCollisionWrongResult)
{
  // Same-type collision: source 0's int64 `a` (field_id: 1) lands on source 1's int64 `b`
  // (field_id: 2).
  auto const a_a = column_wrapper<int64_t>{1, 2, 3};
  auto const b_a = column_wrapper<int64_t>{7, 8, 9};
  cudf::table_view const table_a{{a_a, b_a}};
  auto const path_a =
    write_parquet_temp_file(table_a, "MismatchSameTypeA.parquet", {"a", "b"}, {1, 2});

  auto const b_b = column_wrapper<int64_t>{1000, 2000, 3000};  // B's `b` lands at a's index
  auto const a_b = column_wrapper<int64_t>{10, 20, 30};        // B's `a` (all < 100)
  cudf::table_view const table_b{{b_b, a_b}};
  auto const path_b =
    write_parquet_temp_file(table_b, "MismatchSameTypeB.parquet", {"b", "a"}, {2, 1});

  auto value  = cudf::numeric_scalar<int64_t>(100);
  auto lit    = cudf::ast::literal(value);
  auto col    = cudf::ast::column_name_reference("a");
  auto filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col, lit);

  // All rows have a < 100, so none should be pruned.
  auto const exp_a = column_wrapper<int64_t>{1, 2, 3, 10, 20, 30};
  auto const exp_b = column_wrapper<int64_t>{7, 8, 9, 1000, 2000, 3000};
  cudf::table_view const expected{{exp_a, exp_b}};

  {
    auto const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path_a, path_b}})
        .allow_mismatched_pq_schemas(true)
        .column_names({"a", "b"})
        .filter(filter)
        .build();
    auto const result = cudf::io::read_parquet(opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  }
  {
    auto const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path_a, path_b}})
        .allow_mismatched_pq_schemas(true)
        .column_field_ids({1, 2})
        .filter(filter)
        .build();
    auto const result = cudf::io::read_parquet(opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  }
}

TEST_F(ParquetReaderTest, MismatchedSchemaFilterOnlyColumnCollision)
{
  // Filter-only column (referenced in the filter, not projected) with a different-type collision.
  auto const id_a    = column_wrapper<int64_t>{1, 2, 3};
  auto const price_a = column_wrapper<double>{10.0, 200.0, 30.0};
  cudf::table_view const table_a{{id_a, price_a}};
  auto const path_a =
    write_parquet_temp_file(table_a, "MismatchFilterOnlyA.parquet", {"id", "price"}, {10, 11});

  auto const category_b = column_wrapper<cudf::string_view>{"x", "y", "z"};
  auto const id_b       = column_wrapper<int64_t>{1000, 1001, 1002};
  auto const price_b    = column_wrapper<double>{40.0, 500.0, 60.0};
  cudf::table_view const table_b{{category_b, id_b, price_b}};
  auto const path_b = write_parquet_temp_file(
    table_b, "MismatchFilterOnlyB.parquet", {"category", "id", "price"}, {12, 10, 11});

  auto value  = cudf::numeric_scalar<double>(100.0);
  auto lit    = cudf::ast::literal(value);
  auto col    = cudf::ast::column_name_reference("price");
  auto filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col, lit);

  auto const exp_id = column_wrapper<int64_t>{1, 3, 1000, 1002};
  cudf::table_view const expected{{exp_id}};

  {
    auto const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path_a, path_b}})
        .allow_mismatched_pq_schemas(true)
        .column_names({"id"})  // `price` is filter-only
        .filter(filter)
        .build();
    auto const result = cudf::io::read_parquet(opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  }
  {
    auto const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path_a, path_b}})
        .allow_mismatched_pq_schemas(true)
        .column_field_ids({10})  // `price` is filter-only
        .filter(filter)
        .build();
    auto const result = cudf::io::read_parquet(opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  }
}

TEST_F(ParquetReaderTest, RowIndexColumn)
{
  auto constexpr num_rows = 3;
  auto col0               = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{num_rows});
  auto col1 = cudf::test::fixed_width_column_wrapper<int64_t>(
    cuda::counting_iterator<int64_t>{0}, cuda::counting_iterator<int64_t>{num_rows});
  auto table = cudf::table_view{{col0, col1}};

  auto filepath = temp_env->get_temp_filepath("RowIndexColumn.parquet");
  cudf::io::write_parquet(
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table).build());

  auto const test_row_index_column = [&](auto num_sources) {
    auto const sources = std::vector<std::string>(num_sources, filepath);
    auto read_opts     = cudf::io::parquet_reader_options::builder(cudf::io::source_info(sources))
                       .prepend_row_index_column(true)
                       .build();
    auto const read = cudf::io::read_parquet(read_opts);
    EXPECT_EQ(read.tbl->num_columns(), table.num_columns() + 1);
    EXPECT_EQ(read.metadata.schema_info.front().name, "row_index");

    // The row index restarts at zero for each source
    auto const row_index = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) -> size_t { return i % num_rows; });
    auto const expected =
      cudf::test::fixed_width_column_wrapper<size_t>(row_index, row_index + num_sources * num_rows);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(read.tbl->view().column(0), expected);

    // The data columns are the single source file tiled `num_sources` times
    auto const tiled = cudf::tile(table, num_sources);
    CUDF_TEST_EXPECT_TABLES_EQUAL(read.tbl->select({1, 2}), tiled->view());

    // Use filter `col0 < 2` so only rows {0, 1} from each source are read
    auto scalar  = cudf::numeric_scalar<cudf::size_type>(2);
    auto literal = cudf::ast::literal(scalar);
    auto col_ref = cudf::ast::column_reference(0);
    auto filter  = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);

    read_opts.set_filter(filter);
    auto const read_filtered      = cudf::io::read_parquet(read_opts);
    auto const filtered_row_index = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) -> size_t { return i % 2; });
    auto const filtered_expected = cudf::test::fixed_width_column_wrapper<size_t>(
      filtered_row_index, filtered_row_index + num_sources * 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(read_filtered.tbl->view().column(0), filtered_expected);
  };

  test_row_index_column(1);
  test_row_index_column(5);
}

TEST_F(ParquetReaderTest, SourceAndRowIndexColumns)
{
  auto constexpr num_rows    = 3;
  auto constexpr num_sources = 3;
  auto col0                  = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{num_rows});
  auto col1 = cudf::test::fixed_width_column_wrapper<int64_t>(
    cuda::counting_iterator<int64_t>{0}, cuda::counting_iterator<int64_t>{num_rows});
  auto table = cudf::table_view{{col0, col1}};

  auto filepath = temp_env->get_temp_filepath("RowIndexAndSourceIndexColumns.parquet");
  cudf::io::write_parquet(
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table).build());

  auto const sources   = std::vector<std::string>(num_sources, filepath);
  auto const read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info(sources))
                           .prepend_source_index_column(true)
                           .prepend_row_index_column(true)
                           .build();
  auto const read = cudf::io::read_parquet(read_opts);
  EXPECT_EQ(read.tbl->num_columns(), table.num_columns() + 2);
  EXPECT_EQ(read.metadata.schema_info[0].name, "source_index");
  EXPECT_EQ(read.metadata.schema_info[1].name, "row_index");

  auto const src_index = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return i / num_rows; });
  auto const expected_src_index = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    src_index, src_index + num_sources * num_rows);
  auto const row_index = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) -> size_t { return i % num_rows; });
  auto const expected_row_index =
    cudf::test::fixed_width_column_wrapper<size_t>(row_index, row_index + num_sources * num_rows);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(read.tbl->view().column(0), expected_src_index);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(read.tbl->view().column(1), expected_row_index);
}

TEST_F(ParquetReaderTest, RowIndexSelectedRead)
{
  auto write_parquet = [](auto const& filepath, auto const& col) {
    auto constexpr rows_per_row_group = 2;
    auto const write_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath},
                                                                      cudf::table_view{{col}})
                              .row_group_size_rows(rows_per_row_group)
                              .max_page_size_rows(rows_per_row_group)
                              .max_page_fragment_size(rows_per_row_group)
                              .build();
    cudf::io::write_parquet(write_opts);
  };

  auto col0      = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  auto filepath0 = temp_env->get_temp_filepath("RowIndexSelectedRead0.parquet");
  write_parquet(filepath0, col0);

  auto col1      = cudf::test::fixed_width_column_wrapper<int32_t>{10, 11, 12, 13};
  auto filepath1 = temp_env->get_temp_filepath("RowIndexSelectedRead1.parquet");
  write_parquet(filepath1, col1);

  auto source = cudf::io::source_info{std::vector<std::string>{filepath0, filepath1}};

  // The row index column contains file-local row indices
  auto expected_values    = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 10, 11};
  auto expected_source    = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 1, 1};
  auto expected_row_index = cudf::test::fixed_width_column_wrapper<size_t>{2, 3, 0, 1};
  auto expected = cudf::table_view{{expected_source, expected_row_index, expected_values}};

  // Test with row bounds
  {
    auto const read_opts = cudf::io::parquet_reader_options::builder(source)
                             .skip_rows(2)
                             .num_rows(4)
                             .prepend_source_index_column(true)
                             .prepend_row_index_column(true)
                             .build();
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::io::read_parquet(read_opts).tbl->view(), expected);
  }

  // Test with row group selection
  {
    auto const read_opts = cudf::io::parquet_reader_options::builder(source)
                             .row_groups({{1}, {0}})
                             .prepend_source_index_column(true)
                             .prepend_row_index_column(true)
                             .build();
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::io::read_parquet(read_opts).tbl->view(), expected);
  }

  // Test with out-of-order row group selection within a single source
  {
    auto const read_opts = cudf::io::parquet_reader_options::builder(
                             cudf::io::source_info{std::vector<std::string>{filepath0}})
                             .row_groups({{1, 0}})
                             .prepend_row_index_column(true)
                             .build();
    auto const expected_values    = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 0, 1};
    auto const expected_row_index = cudf::test::fixed_width_column_wrapper<size_t>{2, 3, 0, 1};
    auto const expected           = cudf::table_view{{expected_row_index, expected_values}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::io::read_parquet(read_opts).tbl->view(), expected);
  }

  // Test with a filter: `col >= 2` prunes leading rows of source 0 (values {0, 1, 2, 3}) while
  // keeping all of source 1 (values {10, 11, 12, 13}).
  {
    auto scalar  = cudf::numeric_scalar<int32_t>(2);
    auto literal = cudf::ast::literal(scalar);
    auto col_ref = cudf::ast::column_reference(0);
    auto filter  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);

    auto const read_opts = cudf::io::parquet_reader_options::builder(source)
                             .filter(filter)
                             .prepend_source_index_column(true)
                             .prepend_row_index_column(true)
                             .build();
    auto const expected_filtered_values =
      cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 10, 11, 12, 13};
    auto const expected_filtered_source =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 1, 1, 1, 1};
    auto const expected_filtered_row_index =
      cudf::test::fixed_width_column_wrapper<size_t>{2, 3, 0, 1, 2, 3};
    auto const expected_filtered = cudf::table_view{
      {expected_filtered_source, expected_filtered_row_index, expected_filtered_values}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::io::read_parquet(read_opts).tbl->view(), expected_filtered);
  }
}

TEST_F(ParquetReaderTest, MismatchedSchemaColumnValidation)
{
  // Every required column (selected or filter-referenced) must exist and be compatible across all
  // sources if mismatched schemas are allowed.
  using i64 = column_wrapper<int64_t>;
  using f64 = column_wrapper<double>;

  // When mismatched schemas are allowed, a selected column (by name or field ID) absent from any
  // sources is rejected even if ignore_missing_columns is enabled.
  {
    auto const id0 = i64{1, 2, 3};
    auto const path0 =
      write_parquet_temp_file(cudf::table_view{{id0}}, "IdOnly.parquet", {"id"}, {1});
    auto const id1    = i64{4, 5, 6};
    auto const price1 = f64{10.0, 20.0, 30.0};
    auto const path1  = write_parquet_temp_file(
      cudf::table_view{{id1, price1}}, "IdAndPrice.parquet", {"id", "price"}, {1, 2});

    // Selected column `price` is missing from either source
    {
      auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path0, path1}})
                    .allow_mismatched_pq_schemas(true)
                    .ignore_missing_columns(true)
                    .column_names({"id", "price"})
                    .build();
      EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);

      opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path1, path0}})
               .allow_mismatched_pq_schemas(true)
               .ignore_missing_columns(true)
               .column_names({"id", "price"})
               .build();
      EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);
    }

    // Filter-only column `price` is missing from either source
    {
      // Filter: `price < 100.0`
      auto value  = cudf::numeric_scalar<double>(100.0);
      auto lit    = cudf::ast::literal(value);
      auto col    = cudf::ast::column_name_reference("price");
      auto filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col, lit);

      auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path0, path1}})
                    .allow_mismatched_pq_schemas(true)
                    .ignore_missing_columns(true)
                    .column_names({"id"})
                    .filter(filter)
                    .build();
      EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);

      opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path1, path0}})
               .allow_mismatched_pq_schemas(true)
               .ignore_missing_columns(true)
               .column_names({"id"})
               .filter(filter)
               .build();
      EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);
    }

    // Selected field ID 2 (`price`) is missing from either
    {
      auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path0, path1}})
                    .allow_mismatched_pq_schemas(true)
                    .ignore_missing_columns(true)
                    .column_field_ids({1, 2})
                    .build();
      EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);

      opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path1, path0}})
               .allow_mismatched_pq_schemas(true)
               .ignore_missing_columns(true)
               .column_field_ids({1, 2})
               .build();
      EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);
    }
  }

  // Selected column `price` is present in all sources but with mismatched type (double vs int64).
  {
    auto const id0    = i64{1, 2, 3};
    auto const price0 = f64{10.0, 20.0, 30.0};
    auto const path0  = write_parquet_temp_file(
      cudf::table_view{{id0, price0}}, "Float64Price.parquet", {"id", "price"});
    auto const id1    = i64{4, 5, 6};
    auto const price1 = i64{40, 50, 60};
    auto const path1  = write_parquet_temp_file(
      cudf::table_view{{id1, price1}}, "Int64Price.parquet", {"id", "price"});

    auto const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path0, path1}})
        .allow_mismatched_pq_schemas(true)
        .column_names({"id", "price"})
        .build();
    EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);
  }
}

TEST_F(ParquetReaderTest, NestedMismatchedSchemaColumnValidation)
{
  auto const write_record = [](std::unique_ptr<cudf::column> record,
                               std::string const& filename,
                               std::vector<std::string> const& child_names) {
    cudf::table_view const table{{*record}};
    auto const path = temp_env->get_temp_filepath(filename);
    cudf::io::table_input_metadata metadata(table);
    metadata.column_metadata[0].set_name("record");
    for (size_t idx = 0; idx < child_names.size(); ++idx) {
      metadata.column_metadata[0].child(idx).set_name(child_names[idx]);
    }
    cudf::io::write_parquet(
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{path}, table)
        .metadata(std::move(metadata)));
    return path;
  };

  // Selected nested child must be present in every source
  {
    auto a0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto b1 = cudf::test::fixed_width_column_wrapper<int32_t>{4, 5, 6};
    auto const path0 =
      write_record(cudf::test::structs_column_wrapper{{a0}}.release(), "RecordA.parquet", {"a"});
    auto const path1 =
      write_record(cudf::test::structs_column_wrapper{{b1}}.release(), "RecordB.parquet", {"b"});

    auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path0, path1}})
                  .allow_mismatched_pq_schemas(true)
                  .column_names({"record.a"})
                  .build();
    EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);

    opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path1, path0}})
             .allow_mismatched_pq_schemas(true)
             .column_names({"record.a"})
             .build();
    EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);
  }

  // Selected struct must have the same children in every source
  {
    auto a0 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto a1 = cudf::test::fixed_width_column_wrapper<int32_t>{4, 5, 6};
    auto b1 = cudf::test::fixed_width_column_wrapper<int32_t>{7, 8, 9};
    auto const path0 =
      write_record(cudf::test::structs_column_wrapper{{a0}}.release(), "RecordA.parquet", {"a"});
    auto const path1 = write_record(
      cudf::test::structs_column_wrapper{{a1, b1}}.release(), "RecordAB.parquet", {"a", "b"});

    auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path0, path1}})
                  .allow_mismatched_pq_schemas(true)
                  .column_names({"record"})
                  .build();
    EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);

    opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{{path1, path0}})
             .allow_mismatched_pq_schemas(true)
             .column_names({"record"})
             .build();
    EXPECT_THROW(cudf::io::read_parquet(opts), std::invalid_argument);
  }
}
