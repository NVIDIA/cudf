/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/parquet.hpp>

// Base test fixture for chunked writer tests
struct ParquetChunkedWriterTest : public cudf::test::BaseFixture {};

// Typed test fixture for numeric type tests
template <typename T>
struct ParquetChunkedWriterNumericTypeTest : public ParquetChunkedWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TEST_F(ParquetChunkedWriterTest, SingleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedSingle.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(*table1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *table1);
}

TEST_F(ParquetChunkedWriterTest, SimpleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedSimple.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, LargeTables)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(512, 4096, true);
  auto table2 = create_random_fixed_table<int>(512, 8192, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedLarge.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  auto md = cudf::io::chunked_parquet_writer(args).write(*table1).write(*table2).close();
  ASSERT_NE(md, nullptr);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ManyTables)
{
  srand(31337);
  std::vector<std::unique_ptr<table>> tables;
  std::vector<table_view> table_views;
  constexpr int num_tables = 96;
  for (int idx = 0; idx < num_tables; idx++) {
    auto tbl = create_random_fixed_table<int>(16, 64, true);
    table_views.push_back(*tbl);
    tables.push_back(std::move(tbl));
  }

  auto expected = cudf::concatenate(table_views);

  auto filepath = temp_env->get_temp_filepath("ChunkedManyTables.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer writer(args);
  std::for_each(table_views.begin(), table_views.end(), [&writer](table_view const& tbl) {
    writer.write(tbl);
  });
  auto md = writer.close({"dummy/path"});
  ASSERT_NE(md, nullptr);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, Strings)
{
  std::vector<std::unique_ptr<cudf::column>> cols;

  std::array mask1{true, true, false, true, true, true, true};
  std::vector<char const*> h_strings1{"four", "score", "and", "seven", "years", "ago", "abcdefgh"};
  cudf::test::strings_column_wrapper strings1(h_strings1.begin(), h_strings1.end(), mask1.data());
  cols.push_back(strings1.release());
  cudf::table tbl1(std::move(cols));

  std::array mask2{false, true, true, true, true, true, true};
  std::vector<char const*> h_strings2{"ooooo", "ppppppp", "fff", "j", "cccc", "bbb", "zzzzzzzzzzz"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), mask2.data());
  cols.push_back(strings2.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedStrings.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, ListColumn)
{
  auto valids  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto valids2 = cudf::test::iterators::null_at(3);

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 (Same nullability) ====================
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col0_tbl0{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [7, 8, 9]
  // []
  // [NULL, 11]
  // NULL
  lcw col0_tbl1{{{7, 8, 9}, {}, {{10, 11}, valids}, {}}, valids2};

  // COL1 (Nullability different in different chunks, test of merging nullability in writer)
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // []
  lcw col1_tbl0{{{1, 2, 3}, valids}, {}, {4, 5}, {}};

  // [7, 8, 9]
  // []
  // [10, 11]
  // NULL
  lcw col1_tbl1{{{7, 8, 9}, {}, {10, 11}, {}}, valids2};

  // COL2 (non-nested columns to test proper schema construction)
  size_t num_rows_tbl0 = static_cast<cudf::column_view>(col0_tbl0).size();
  size_t num_rows_tbl1 = static_cast<cudf::column_view>(col0_tbl1).size();
  auto seq_col0        = random_values<int>(num_rows_tbl0);
  auto seq_col1        = random_values<int>(num_rows_tbl1);

  column_wrapper<int> col2_tbl0{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col2_tbl1{seq_col1.begin(), seq_col1.end(), valids2};

  auto tbl0 = table_view({col0_tbl0, col1_tbl0, col2_tbl0});
  auto tbl1 = table_view({col0_tbl1, col1_tbl1, col2_tbl1});

  auto expected = cudf::concatenate(std::vector<table_view>({tbl0, tbl1}));

  auto filepath = temp_env->get_temp_filepath("ChunkedLists.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(tbl0).write(tbl1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, ListOfStruct)
{
  // Table 1
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};

  auto list_offsets_column_1 =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 3, 3}.release();
  auto num_list_rows_1 = list_offsets_column_1->size() - 1;

  auto list_col_1 = cudf::make_lists_column(
    num_list_rows_1, std::move(list_offsets_column_1), struct_2_1.release(), 0, {});

  auto table_1 = table_view({*list_col_1});

  // Table 2
  auto weight_2 = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2 =
    cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {true, true, false}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {true, false, true}};
  auto is_human_2 =
    cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, {true, true, false}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};

  auto list_offsets_column_2 =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3}.release();
  auto num_list_rows_2 = list_offsets_column_2->size() - 1;

  auto list_col_2 = cudf::make_lists_column(
    num_list_rows_2, std::move(list_offsets_column_2), struct_2_2.release(), 0, {});

  auto table_2 = table_view({*list_col_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).set_nullability(false);
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedListOfStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::chunked_parquet_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ListOfStructOfStructOfListOfList)
{
  auto valids  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto valids2 = cudf::test::iterators::null_at(3);

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // Table 1 ===========================

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  lcw land_1{{{}, {{1, 2, 3}, valids}, {4, 5}, {}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  lcw flats_1{lcw{}, {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}, {}}, lcw{lcw{}}};

  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3, 1.1}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5, 31}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1, land_1, flats_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};

  auto list_offsets_column_1 =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 2, 3, 4}.release();
  auto num_list_rows_1 = list_offsets_column_1->size() - 1;

  auto list_col_1 = cudf::make_lists_column(
    num_list_rows_1, std::move(list_offsets_column_1), struct_2_1.release(), 0, {});

  auto table_1 = table_view({*list_col_1});

  // Table 2 ===========================

  // []
  // [7, 8, 9]
  lcw land_2{{}, {7, 8, 9}};

  // [[]]
  // [[], [], []]
  lcw flats_2{lcw{lcw{}}, lcw{lcw{}, lcw{}, lcw{}}};

  auto weight_2 = cudf::test::fixed_width_column_wrapper<float>{{-1.0, -1.0}};
  auto ages_2   = cudf::test::fixed_width_column_wrapper<int32_t>{{351, 351}, {true, false}};
  auto struct_1_2 =
    cudf::test::structs_column_wrapper{{weight_2, ages_2, land_2, flats_2}, {false, true}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false}, {true, false}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};

  auto list_offsets_column_2 = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2}.release();
  auto num_list_rows_2       = list_offsets_column_2->size() - 1;

  auto list_col_2 = cudf::make_lists_column(
    num_list_rows_2, std::move(list_offsets_column_2), struct_2_2.release(), 0, {});

  auto table_2 = table_view({*list_col_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).set_nullability(false);
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");
  expected_metadata.column_metadata[0].child(1).child(1).child(2).set_name("land_unit");
  expected_metadata.column_metadata[0].child(1).child(1).child(3).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("ListOfStructOfStructOfListOfList.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::chunked_parquet_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);

  // We specifically mentioned in input schema that struct_2 is non-nullable across chunked calls.
  auto result_parent_list = result.tbl->get_column(0);
  auto result_struct_2    = result_parent_list.child(cudf::lists_column_view::child_column_index);
  EXPECT_EQ(result_struct_2.nullable(), false);
}

TEST_F(ParquetChunkedWriterTest, MismatchedTypes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedTypes.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer writer(args);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
  writer.close();
}

TEST_F(ParquetChunkedWriterTest, ChunkedWriteAfterClosing)
{
  srand(31337);
  auto table = create_random_fixed_table<int>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedWriteAfterClosing.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer writer(args);
  writer.write(*table).close();
  EXPECT_THROW(writer.write(*table), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, ReadingUnclosedFile)
{
  srand(31337);
  auto table = create_random_fixed_table<int>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ReadingUnclosedFile.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer writer(args);
  writer.write(*table);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, MismatchedStructure)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(3, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedStructure.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer writer(args);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
  writer.close();
}

TEST_F(ParquetChunkedWriterTest, MismatchedStructureList)
{
  auto valids  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto valids2 = cudf::test::iterators::null_at(3);

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 (mismatched depth) ====================
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col00{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col01{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // COL2 (non-nested columns to test proper schema construction)
  size_t num_rows = static_cast<cudf::column_view>(col00).size();
  auto seq_col0   = random_values<int>(num_rows);
  auto seq_col1   = random_values<int>(num_rows);

  column_wrapper<int> col10{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col11{seq_col1.begin(), seq_col1.end(), valids2};

  auto tbl0 = table_view({col00, col10});
  auto tbl1 = table_view({col01, col11});

  auto filepath = temp_env->get_temp_filepath("ChunkedLists.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer writer(args);
  writer.write(tbl0);
  EXPECT_THROW(writer.write(tbl1), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, DifferentNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedNullable.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, DifferentNullabilityStruct)
{
  // Struct<is_human:bool (non-nullable),
  //        Struct<weight:float>,
  //               age:int
  //              > (nullable)
  //       > (non-nullable)

  // Table 1: is_human and struct_1 are non-nullable but should be nullable when read back.
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};
  auto table_1    = cudf::table_view({struct_2_1});

  // Table 2: struct_1 and is_human are nullable now so if we hadn't assumed worst case (nullable)
  // when writing table_1, we would have wrong pages for it.
  auto weight_2 = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2 =
    cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {true, true, false}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {true, false, true}};
  auto is_human_2 =
    cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, {true, true, false}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};
  auto table_2    = cudf::table_view({struct_2_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("being");
  expected_metadata.column_metadata[0].child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedNullableStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::chunked_parquet_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, false);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedNoNullable.parquet");

  cudf::io::table_input_metadata metadata(*table1);

  // In the absence of prescribed per-column nullability in metadata, the writer assumes the worst
  // and considers all columns nullable. However cudf::concatenate will not force nulls in case no
  // columns are nullable. To get the expected result, we tell the writer the nullability of all
  // columns in advance.
  for (auto& col_meta : metadata.column_metadata) {
    col_meta.set_nullability(false);
  }

  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
      .metadata(std::move(metadata));
  cudf::io::chunked_parquet_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullabilityList)
{
  srand(31337);

  auto valids  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto valids2 = cudf::test::iterators::null_at(3);

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 ====================
  // [1, 2, 3]
  // []
  // [4, 5]
  // NULL
  lcw col00{{{1, 2, 3}, {}, {4, 5}, {}}, valids2};

  // [7]
  // []
  // [8, 9, 10, 11]
  // NULL
  lcw col01{{{7}, {}, {8, 9, 10, 11}, {}}, valids2};

  // COL1 (non-nested columns to test proper schema construction)
  size_t num_rows = static_cast<cudf::column_view>(col00).size();
  auto seq_col0   = random_values<int>(num_rows);
  auto seq_col1   = random_values<int>(num_rows);

  column_wrapper<int> col10{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col11{seq_col1.begin(), seq_col1.end(), valids2};

  auto table1 = table_view({col00, col10});
  auto table2 = table_view({col01, col11});

  auto full_table = cudf::concatenate(std::vector<table_view>({table1, table2}));

  cudf::io::table_input_metadata metadata(table1);
  metadata.column_metadata[0].set_nullability(true);  // List is nullable at first (root) level
  metadata.column_metadata[0].child(1).set_nullability(
    false);  // non-nullable at second (leaf) level
  metadata.column_metadata[1].set_nullability(true);

  auto filepath = temp_env->get_temp_filepath("ChunkedListNullable.parquet");

  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
      .metadata(std::move(metadata));
  cudf::io::chunked_parquet_writer(args).write(table1).write(table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullabilityStruct)
{
  // Struct<is_human:bool (non-nullable),
  //        Struct<weight:float>,
  //               age:int
  //              > (nullable)
  //       > (non-nullable)

  // Table 1: is_human and struct_2 are non-nullable and should stay that way when read back.
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};
  auto table_1    = cudf::table_view({struct_2_1});

  auto weight_2 = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2 =
    cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {true, true, false}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {true, false, true}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};
  auto table_2    = cudf::table_view({struct_2_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("being").set_nullability(false);
  expected_metadata.column_metadata[0].child(0).set_name("human?").set_nullability(false);
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedNullableStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::chunked_parquet_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroups)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table2, *table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroups.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  {
    cudf::io::chunked_parquet_writer(args).write(*table1).write(*table2);
  }

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .row_groups({{1, 0, 1}});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroupsError)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroupsError.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(*table1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).row_groups({{0, 1}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), std::invalid_argument);
  read_opts.set_row_groups({{-1}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), std::invalid_argument);
  read_opts.set_row_groups({{0}, {0}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), std::invalid_argument);
}

TEST_F(ParquetChunkedWriterTest, RowGroupPageSizeMatch)
{
  std::vector<char> out_buffer;

  auto options = cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                   .row_group_size_bytes(128 * 1024)
                   .max_page_size_bytes(512 * 1024)
                   .row_group_size_rows(10000)
                   .max_page_size_rows(20000)
                   .build();
  EXPECT_EQ(options.get_row_group_size_bytes(), options.get_max_page_size_bytes());
  EXPECT_EQ(options.get_row_group_size_rows(), options.get_max_page_size_rows());
}

TEST_F(ParquetChunkedWriterTest, CompStats)
{
  auto table = create_random_fixed_table<int>(1, 100000, true);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_parquet_writer_options opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression_statistics(stats);
  cudf::io::chunked_parquet_writer(opts).write(*table);

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));

  auto const single_table_comp_stats = *stats;
  cudf::io::chunked_parquet_writer(opts).write(*table);

  EXPECT_EQ(stats->compression_ratio(), single_table_comp_stats.compression_ratio());
  EXPECT_EQ(stats->num_compressed_bytes(), 2 * single_table_comp_stats.num_compressed_bytes());

  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
}

TEST_F(ParquetChunkedWriterTest, CompStatsEmptyTable)
{
  auto table_no_rows = create_random_fixed_table<int>(20, 0, false);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_parquet_writer_options opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression_statistics(stats);
  cudf::io::chunked_parquet_writer(opts).write(*table_no_rows);

  expect_compression_stats_empty(stats);
}

TYPED_TEST_SUITE(ParquetChunkedWriterNumericTypeTest, SupportedTypes);

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize)
{
  // write out two 31 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  constexpr int num_els = 31;
  std::vector<std::unique_ptr<cudf::column>> cols;

  std::array<bool, num_els> mask{false, true, true, true, true, true, true, true, true, true, true,
                                 true,  true, true, true, true, true, true, true, true, true, true,

                                 true,  true, true, true, true, true, true, true, true};
  std::array<T, num_els> c1a;
  std::fill(c1a.begin(), c1a.end(), static_cast<T>(5));
  std::array<T, num_els> c1b;
  std::fill(c1b.begin(), c1b.end(), static_cast<T>(5));
  column_wrapper<T> c1a_w(c1a.begin(), c1a.end(), mask.begin());
  column_wrapper<T> c1b_w(c1b.begin(), c1b.end(), mask.begin());
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  std::array<T, num_els> c2a;
  std::fill(c2a.begin(), c2a.end(), static_cast<T>(8));
  std::array<T, num_els> c2b;
  std::fill(c2b.begin(), c2b.end(), static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a.begin(), c2a.end(), mask.begin());
  column_wrapper<T> c2b_w(c2b.begin(), c2b.end(), mask.begin());
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize2)
{
  // write out two 33 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  constexpr int num_els = 33;
  std::vector<std::unique_ptr<cudf::column>> cols;

  std::array<bool, num_els> mask{false, true, true, true, true, true, true, true, true, true, true,
                                 true,  true, true, true, true, true, true, true, true, true, true,
                                 true,  true, true, true, true, true, true, true, true, true, true};

  std::array<T, num_els> c1a;
  std::fill(c1a.begin(), c1a.end(), static_cast<T>(5));
  std::array<T, num_els> c1b;
  std::fill(c1b.begin(), c1b.end(), static_cast<T>(5));
  column_wrapper<T> c1a_w(c1a.begin(), c1a.end(), mask.begin());
  column_wrapper<T> c1b_w(c1b.begin(), c1b.end(), mask.begin());
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  std::array<T, num_els> c2a;
  std::fill(c2a.begin(), c2a.end(), static_cast<T>(8));
  std::array<T, num_els> c2b;
  std::fill(c2b.begin(), c2b.end(), static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a.begin(), c2a.end(), mask.begin());
  column_wrapper<T> c2b_w(c2b.begin(), c2b.end(), mask.begin());
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize2.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}
