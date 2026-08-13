/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/parquet.hpp>

#include <array>
#include <format>
#include <limits>
#include <random>
#include <string>

using cudf::test::iterators::no_nulls;

// Base test fixture for V2 header tests
class ParquetV2Test : public ::cudf::test::BaseFixtureWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(ParquetV2ReadWriteTest,
                         ParquetV2Test,
                         testing::Bool(),
                         testing::PrintToStringParamName());

TEST_P(ParquetV2Test, MultiColumn)
{
  constexpr auto num_rows = 50'000;
  auto const is_v2        = GetParam();

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int16_t>(num_rows);
  auto col7_vals = random_values<int32_t>(num_rows);
  auto col8_vals = random_values<int64_t>(num_rows);

  // column_wrapper<bool> col0{col0_data.begin(), col0_data.end(), no_nulls()};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), no_nulls()};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), no_nulls()};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), no_nulls()};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), no_nulls()};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), no_nulls()};

  cudf::test::fixed_point_column_wrapper<numeric::decimal32::rep> col6(
    col6_vals.begin(), col6_vals.end(), no_nulls(), numeric::scale_type{5});
  cudf::test::fixed_point_column_wrapper<numeric::decimal64::rep> col7(
    col7_vals.begin(), col7_vals.end(), no_nulls(), numeric::scale_type{-5});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col8(
    col8_vals.begin(), col8_vals.end(), no_nulls(), numeric::scale_type{-6});

  auto expected = table_view{{col1, col2, col3, col4, col5, col6, col7, col8}};

  cudf::io::table_input_metadata expected_metadata(expected);
  // expected_metadata.column_metadata[0].set_name( "bools");
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");
  expected_metadata.column_metadata[5].set_name("decimal32s").set_decimal_precision(10);
  expected_metadata.column_metadata[6].set_name("decimal64s").set_decimal_precision(20);
  expected_metadata.column_metadata[7].set_name("decimal128s").set_decimal_precision(40);

  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .compression(cudf::io::compression_type::ZSTD)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, MultiColumnWithNulls)
{
  constexpr auto num_rows = 100;
  auto const is_v2        = GetParam();

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int32_t>(num_rows);
  auto col7_vals = random_values<int64_t>(num_rows);
  auto col1_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i < 10); });
  auto col2_mask = no_nulls();
  auto col3_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i == (num_rows - 1)); });
  auto col4_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i >= 40 && i <= 60); });
  auto col5_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 80); });
  auto col6_mask = cudf::test::iterators::nulls_at_multiples_of(5);
  auto col7_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i != 55); });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), col0_mask};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), col1_mask};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), col2_mask};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), col3_mask};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), col4_mask};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), col5_mask};

  cudf::test::fixed_point_column_wrapper<numeric::decimal32::rep> col6(
    col6_vals.begin(), col6_vals.end(), col6_mask, numeric::scale_type{-2});
  cudf::test::fixed_point_column_wrapper<numeric::decimal64::rep> col7(
    col7_vals.begin(), col7_vals.end(), col7_mask, numeric::scale_type{-8});

  auto expected = table_view{{/*col0, */ col1, col2, col3, col4, col5, col6, col7}};

  cudf::io::table_input_metadata expected_metadata(expected);
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");
  expected_metadata.column_metadata[5].set_name("decimal32s").set_decimal_precision(9);
  expected_metadata.column_metadata[6].set_name("decimal64s").set_decimal_precision(20);

  auto filepath = temp_env->get_temp_filepath("MultiColumnWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .compression(cudf::io::compression_type::ZSTD)
      .metadata(expected_metadata);

  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  // TODO: Need to be able to return metadata in tree form from reader so they can be compared.
  // Unfortunately the closest thing to a hierarchical schema is column_name_info which does not
  // have any tests for it c++ or python.
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, Strings)
{
  auto const is_v2 = GetParam();

  std::vector<char const*> strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), no_nulls()};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), no_nulls()};

  auto expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");

  auto filepath = temp_env->get_temp_filepath("Strings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .compression(cudf::io::compression_type::ZSTD)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, StringsAsBinary)
{
  auto const is_v2 = GetParam();
  std::vector<char const*> unicode_strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  std::vector<char const*> ascii_strings{
    "Monday", "Wednesday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};

  column_wrapper<cudf::string_view> col0{ascii_strings.begin(), ascii_strings.end()};
  column_wrapper<cudf::string_view> col1{unicode_strings.begin(), unicode_strings.end()};
  column_wrapper<cudf::string_view> col2{ascii_strings.begin(), ascii_strings.end()};
  cudf::test::lists_column_wrapper<uint8_t> col3{{'M', 'o', 'n', 'd', 'a', 'y'},
                                                 {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'M', 'o', 'n', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'u', 'n', 'd', 'a', 'y'}};
  cudf::test::lists_column_wrapper<uint8_t> col4{
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'W', 200, 133, 'd', 'n', 200, 133, 's', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'u', 'n', 'd', 'a', 'y'}};

  auto write_tbl = table_view{{col0, col1, col2, col3, col4}};

  cudf::io::table_input_metadata expected_metadata(write_tbl);
  expected_metadata.column_metadata[0].set_name("col_single").set_output_as_binary(true);
  expected_metadata.column_metadata[1].set_name("col_string").set_output_as_binary(true);
  expected_metadata.column_metadata[2].set_name("col_another").set_output_as_binary(true);
  expected_metadata.column_metadata[3].set_name("col_binary");
  expected_metadata.column_metadata[4].set_name("col_binary2");

  auto filepath = temp_env->get_temp_filepath("BinaryStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, write_tbl)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(
        {cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().add_child(cudf::io::reader_column_schema()),
         cudf::io::reader_column_schema().add_child(cudf::io::reader_column_schema())});
  auto result   = cudf::io::read_parquet(in_opts);
  auto expected = table_view{{col3, col4, col3, col3, col4}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, SlicedTable)
{
  // This test checks for writing zero copy, offsetted views into existing cudf tables

  std::vector<char const*> strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();
  auto const is_v2    = GetParam();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity = cudf::test::iterators::nulls_at_multiples_of(3);

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  using lcw = cudf::test::lists_column_wrapper<uint64_t>;
  lcw col3{{9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}};

  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [NULL, [], NULL, [[]]]
  // NULL
  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [[[]]]
  // [NULL, [], NULL, [[]]]
  auto valids  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto valids2 = cudf::test::iterators::null_at(3);
  lcw col4{{
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
             lcw{lcw{lcw{}}},
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             lcw{lcw{lcw{}}},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
           },
           valids2};

  // Struct column
  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351, 29, 15}, {true, true, true, true, true, false, true, true}};

  auto col5 = cudf::test::structs_column_wrapper{{ages_col},
                                                 {true, true, true, true, false, true, true, true}};

  // Struct/List mixed column

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  // []
  // [7, 8, 9]
  // [10]
  // [11, 12]
  lcw land{{{}, {{1, 2, 3}, valids}, {4, 5}, {}, {}, {7, 8, 9}, {10}, {11, 12}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  // [[]]
  // [[], [], []]
  // [[10]]
  // [[13, 14], [15]]
  lcw flats{lcw{},
            {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
            {{7, 8}, {}},
            lcw{lcw{}},
            lcw{lcw{}},
            lcw{lcw{}, lcw{}, lcw{}},
            {lcw{10}},
            {{13, 14}, {15}}};

  auto struct_1 = cudf::test::structs_column_wrapper{land, flats};
  auto is_human = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, true, false, true, false}};
  auto col6 = cudf::test::structs_column_wrapper{{is_human, struct_1}};

  auto expected = table_view({col0, col1, col2, col3, col4, col5, col6});

  // auto expected_slice = expected;
  auto expected_slice = cudf::slice(expected, {2, static_cast<cudf::size_type>(num_rows) - 1});

  cudf::io::table_input_metadata expected_metadata(expected_slice);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");
  expected_metadata.column_metadata[3].set_name("col_list");
  expected_metadata.column_metadata[4].set_name("col_multi_level_list");
  expected_metadata.column_metadata[5].set_name("col_struct");
  expected_metadata.column_metadata[5].set_name("col_struct_list");
  expected_metadata.column_metadata[6].child(0).set_name("human?");
  expected_metadata.column_metadata[6].child(1).set_name("particulars");
  expected_metadata.column_metadata[6].child(1).child(0).set_name("land");
  expected_metadata.column_metadata[6].child(1).child(1).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("SlicedTable.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_slice, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, ListColumn)
{
  auto const is_v2 = GetParam();

  auto valids  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto valids2 = cudf::test::iterators::null_at(3);

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col0{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col1{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col2{{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, valids2}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // [[]]
  using dlcw = cudf::test::lists_column_wrapper<double>;
  dlcw col3{{{{1., 2., 3.}, {}, {4., 5.}, {}, {{0., 6., 0.}, valids}}, valids2},
            {{7., 8.}},
            dlcw{},
            dlcw{dlcw{}}};

  // TODO: uint16_t lists are not read properly in parquet reader
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // NULL
  // using ui16lcw = cudf::test::lists_column_wrapper<uint16_t>;
  // cudf::test::lists_column_wrapper<uint16_t> col4{
  //   {{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, valids2}, {{7, 8}}, ui16lcw{}, ui16lcw{ui16lcw{}}},
  //   valids2};

  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // NULL
  lcw col5{
    {{{{1, 2, 3}, {}, {4, 5}, {}, {{0, 6, 0}, valids}}, valids2}, {{7, 8}}, lcw{}, lcw{lcw{}}},
    valids2};

  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col6{
    {{"Monday", "Monday", "Friday"}, {}, {"Monday", "Friday"}, {}, {"Sunday", "Funday"}},
    {{"bee", "sting"}},
    strlcw{},
    strlcw{strlcw{}}};

  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [NULL, [], NULL, [[]]]
  // NULL
  lcw col7{{
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
             lcw{lcw{lcw{}}},
           },
           valids2};

  table_view expected({col0, col1, col2, col3, /* col4, */ col5, col6, col7});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_list_int_0");
  expected_metadata.column_metadata[1].set_name("col_list_list_int_1");
  expected_metadata.column_metadata[2].set_name("col_list_list_int_nullable_2");
  expected_metadata.column_metadata[3].set_name("col_list_list_nullable_double_nullable_3");
  // expected_metadata.column_metadata[0].set_name("col_list_list_uint16_4");
  expected_metadata.column_metadata[4].set_name("col_list_nullable_list_nullable_int_nullable_5");
  expected_metadata.column_metadata[5].set_name("col_list_list_string_6");
  expected_metadata.column_metadata[6].set_name("col_list_list_list_7");

  auto filepath = temp_env->get_temp_filepath("ListColumn.parquet");
  auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
                    .write_v2_headers(is_v2)
                    .page_level_compression(is_v2)
                    .metadata(expected_metadata)
                    .compression(cudf::io::compression_type::NONE);

  cudf::io::write_parquet(out_opts);

  auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result  = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, StructOfList)
{
  auto const is_v2 = GetParam();

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

  auto valids  = cudf::test::iterators::nulls_at_multiples_of(2);
  auto valids2 = cudf::test::iterators::null_at(3);

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  // []
  // [7, 8, 9]
  lcw land_unit{{{}, {{1, 2, 3}, valids}, {4, 5}, {}, {}, {7, 8, 9}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  // [[]]
  // [[], [], []]
  lcw flats{lcw{},
            {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
            {{7, 8}, {}},
            lcw{lcw{}},
            lcw{lcw{}},
            lcw{lcw{}, lcw{}, lcw{}}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col, land_unit, flats},
                                                     {true, true, true, true, false, true}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {true, true, false, true, true, false}};

  auto struct_2 = cudf::test::structs_column_wrapper{{is_human_col, struct_1},
                                                     {false, true, true, true, true, true}}
                    .release();

  auto expected = table_view({*struct_2});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("being");
  expected_metadata.column_metadata[0].child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");
  expected_metadata.column_metadata[0].child(1).child(2).set_name("land_unit");
  expected_metadata.column_metadata[0].child(1).child(3).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("StructOfList.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  auto const result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, ListOfStruct)
{
  auto const is_v2 = GetParam();

  // List<Struct<is_human:bool,
  //             Struct<weight:float,
  //                    ages:int,
  //                   >
  //            >
  //     >

  auto weight_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weight_col, ages_col},
                                                     {true, true, true, true, false, true}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {true, true, false, true, true, false}};

  auto struct_2 = cudf::test::structs_column_wrapper{{is_human_col, struct_1},
                                                     {false, true, true, true, true, true}}
                    .release();

  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 5, 5, 6}.release();
  auto num_list_rows = list_offsets_column->size() - 1;

  auto list_col = cudf::make_lists_column(
    num_list_rows, std::move(list_offsets_column), std::move(struct_2), 0, {});

  auto expected = table_view({*list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ListOfStruct.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  auto const result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, PartitionedWriteEmptyPartitions)
{
  auto const is_v2 = GetParam();

  auto source = create_random_fixed_table<int>(4, 4, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{1, 0};
  auto partition2 = cudf::io::partition_info{1, 0};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
}

TEST_P(ParquetV2Test, PartitionedWriteEmptyColumns)
{
  auto const is_v2 = GetParam();

  auto source = create_random_fixed_table<int>(0, 4, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{1, 0};
  auto partition2 = cudf::io::partition_info{1, 0};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndex)
{
  constexpr auto num_rows = 50000;
  auto const is_v2        = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return std::format("{:012d}", i); });
  auto col0 = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<uint64_t>(num_rows);
  auto col5_data = random_values<float>(num_rows);
  auto col6_data = random_values<double>(num_rows);

  auto col1 = cudf::test::fixed_width_column_wrapper<int8_t>(col1_data.begin(), col1_data.end());
  auto col2 = cudf::test::fixed_width_column_wrapper<int16_t>(col2_data.begin(), col2_data.end());
  auto col3 = cudf::test::fixed_width_column_wrapper<int32_t>(col3_data.begin(), col3_data.end());
  auto col4 = cudf::test::fixed_width_column_wrapper<uint64_t>(col4_data.begin(), col4_data.end());
  auto col5 = cudf::test::fixed_width_column_wrapper<float>(col5_data.begin(), col5_data.end());
  auto col6 = cudf::test::fixed_width_column_wrapper<double>(col6_data.begin(), col6_data.end());

  // mixed length strings
  auto str2_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return std::format("{}", i); });
  auto col7 = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows);

  auto const expected = table_view{{col0, col1, col2, col3, col4, col5, col6, col7}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndex.parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (auto const& page_loc : oi.page_locations) {
        auto const ph = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());
      ASSERT_TRUE(ci.null_counts.has_value());

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        // null_pages should always be false
        EXPECT_FALSE(ci.null_pages[p]);
        // null_counts should always be 0
        EXPECT_EQ(ci.null_counts.value()[p], 0);
        EXPECT_TRUE(compare_binary(stats.min_value.value(), ci.min_values[p], ptype, ctype) <= 0);
      }
      for (auto const& max_value : ci.max_values)
        EXPECT_TRUE(compare_binary(stats.max_value.value(), max_value, ptype, ctype) >= 0);
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexNulls)
{
  constexpr auto num_rows = 100000;
  auto const is_v2        = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return std::format("{:012d}", i); });
  auto col0 = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<uint64_t>(num_rows);
  auto col5_data = random_values<float>(num_rows);
  auto col6_data = random_values<double>(num_rows);

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // add null values for all but first column
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int8_t>(col1_data.begin(), col1_data.end(), valids);
  auto col2 =
    cudf::test::fixed_width_column_wrapper<int16_t>(col2_data.begin(), col2_data.end(), valids);
  auto col3 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col3_data.begin(), col3_data.end(), valids);
  auto col4 =
    cudf::test::fixed_width_column_wrapper<uint64_t>(col4_data.begin(), col4_data.end(), valids);
  auto col5 =
    cudf::test::fixed_width_column_wrapper<float>(col5_data.begin(), col5_data.end(), valids);
  auto col6 =
    cudf::test::fixed_width_column_wrapper<double>(col6_data.begin(), col6_data.end(), valids);

  // mixed length strings
  auto str2_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return std::format("{}", i); });
  auto col7 = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows, valids);

  auto expected = table_view{{col0, col1, col2, col3, col4, col5, col6, col7}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexNulls.parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (auto const& page_loc : oi.page_locations) {
        auto const ph = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // should be half nulls, except no nulls in column 0
      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());
      ASSERT_TRUE(stats.null_count.has_value());
      EXPECT_EQ(stats.null_count.value(), c == 0 ? 0 : num_rows / 2);
      ASSERT_TRUE(ci.null_counts.has_value());

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        EXPECT_FALSE(ci.null_pages[p]);
        if (c > 0) {  // first column has no nulls
          EXPECT_GT(ci.null_counts.value()[p], 0);
        } else {
          EXPECT_EQ(ci.null_counts.value()[p], 0);
        }
        EXPECT_TRUE(compare_binary(stats.min_value.value(), ci.min_values[p], ptype, ctype) <= 0);
      }
      for (auto const& max_value : ci.max_values) {
        EXPECT_TRUE(compare_binary(stats.max_value.value(), max_value, ptype, ctype) >= 0);
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexNullColumn)
{
  constexpr auto num_rows = 100000;
  auto const is_v2        = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return std::format("{:012d}", i); });
  auto col0 = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int32_t>(num_rows);
  auto col2_data = random_values<int32_t>(num_rows);

  // col1 is all nulls
  auto valids = cudf::test::iterators::all_nulls();
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col1_data.begin(), col1_data.end(), valids);
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end());

  // mixed length strings
  auto str2_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return std::format("{}", i); });
  auto col3 = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows);

  auto expected = table_view{{col0, col1, col2, col3}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexNullColumn.parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (auto const& page_loc : oi.page_locations) {
        auto const ph = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each non-empty page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // there should be no nulls except column 1 which is all nulls
      if (c != 1) {
        ASSERT_TRUE(stats.min_value.has_value());
        ASSERT_TRUE(stats.max_value.has_value());
      }
      ASSERT_TRUE(stats.null_count.has_value());
      EXPECT_EQ(stats.null_count.value(), c == 1 ? num_rows : 0);
      ASSERT_TRUE(ci.null_counts.has_value());

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        // check tnat null_pages is true for column 1
        if (c == 1) {
          EXPECT_TRUE(ci.null_pages[p]);
          EXPECT_GT(ci.null_counts.value()[p], 0);
        }
        if (not ci.null_pages[p]) {
          EXPECT_EQ(ci.null_counts.value()[p], 0);
          EXPECT_TRUE(compare_binary(stats.min_value.value(), ci.min_values[p], ptype, ctype) <= 0);
        }
      }
      for (size_t p = 0; p < ci.max_values.size(); p++) {
        if (not ci.null_pages[p]) {
          EXPECT_TRUE(compare_binary(stats.max_value.value(), ci.max_values[p], ptype, ctype) >= 0);
        }
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexStruct)
{
  auto const is_v2 = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  auto c0 = testdata::ascending<uint32_t>();

  auto sc0 = testdata::ascending<cudf::string_view>();
  auto sc1 = testdata::descending<int32_t>();
  auto sc2 = testdata::unordered<int64_t>();

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(sc0.release());
  struct_children.push_back(sc1.release());
  struct_children.push_back(sc2.release());
  cudf::test::structs_column_wrapper c1(std::move(struct_children));

  auto listgen = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? i / 2 : num_ordered_rows - (i / 2); });
  auto list =
    cudf::test::fixed_width_column_wrapper<int32_t>(listgen, listgen + 2 * num_ordered_rows);
  auto offgen = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>(offgen, offgen + num_ordered_rows + 1);

  auto c2 = cudf::make_lists_column(num_ordered_rows, offsets.release(), list.release(), 0, {});

  table_view expected({c0, c1, *c2});

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexStruct.parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .max_page_size_rows(page_size_for_ordered_tests);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  // hard coded schema indices.
  // TODO find a way to do this without magic
  constexpr std::array<size_t, 5> colidxs{1, 3, 4, 5, 8};
  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      size_t colidx     = colidxs[c];
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (auto const& page_loc : oi.page_locations) {
        auto const ph = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        // last column has 2 values per row
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows
                          : ph.data_page_header.num_values / (c == rg.columns.size() - 1 ? 2 : 1);
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());

      auto const ptype = fmd.schema[colidx].type;
      auto const ctype = fmd.schema[colidx].converted_type;
      for (auto const& min_value : ci.min_values) {
        EXPECT_TRUE(compare_binary(stats.min_value.value(), min_value, ptype, ctype) <= 0);
      }
      for (auto const& max_value : ci.max_values) {
        EXPECT_TRUE(compare_binary(stats.max_value.value(), max_value, ptype, ctype) >= 0);
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexStructNulls)
{
  auto const is_v2 = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  auto validity2 = cudf::test::iterators::nulls_at_multiples_of(2);
  auto validity3 = cudf::test::iterators::nulls_at_multiples_of(3);
  auto validity4 = cudf::test::iterators::nulls_at_multiples_of(4);
  auto validity5 = cudf::test::iterators::nulls_at_multiples_of(5);

  auto c0 = testdata::ascending<uint32_t>();

  auto col1_data = random_values<int32_t>(num_ordered_rows);
  auto col2_data = random_values<int32_t>(num_ordered_rows);
  auto col3_data = random_values<int32_t>(num_ordered_rows);

  // col1 is all nulls
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col1_data.begin(), col1_data.end(), validity2);
  auto col2 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end(), validity3);
  auto col3 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end(), validity4);

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(col1.release());
  struct_children.push_back(col2.release());
  struct_children.push_back(col3.release());
  auto struct_validity = std::vector<bool>(validity5, validity5 + num_ordered_rows);
  cudf::test::structs_column_wrapper c1(std::move(struct_children), struct_validity);
  table_view expected({c0, c1});

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexStructNulls.parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2)
      .max_page_size_rows(page_size_for_ordered_tests);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  // all struct columns will have num_ordered_rows / 5 nulls at level 0.
  // col1 will have num_ordered_rows / 2 nulls total
  // col2 will have num_ordered_rows / 3 nulls total
  // col3 will have num_ordered_rows / 4 nulls total
  constexpr std::array<int, 4> null_mods{0, 2, 3, 4};

  for (auto const& rg : fmd.row_groups) {
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);
      auto const ci = read_column_index(source, chunk);

      // check definition level histogram (repetition will not be present)
      if (c != 0) {
        ASSERT_TRUE(chunk.meta_data.size_statistics.has_value());
        ASSERT_TRUE(chunk.meta_data.size_statistics->definition_level_histogram.has_value());
        // there are no lists so there should be no repetition level histogram
        EXPECT_FALSE(chunk.meta_data.size_statistics->repetition_level_histogram.has_value());
        auto const& def_hist = chunk.meta_data.size_statistics->definition_level_histogram.value();
        ASSERT_TRUE(def_hist.size() == 3L);
        auto const l0_nulls    = num_ordered_rows / 5;
        auto const l1_l0_nulls = num_ordered_rows / (5 * null_mods[c]);
        auto const l1_nulls    = num_ordered_rows / null_mods[c] - l1_l0_nulls;
        auto const l2_vals     = num_ordered_rows - l1_nulls - l0_nulls;
        EXPECT_EQ(def_hist[0], l0_nulls);
        EXPECT_EQ(def_hist[1], l1_nulls);
        EXPECT_EQ(def_hist[2], l2_vals);
      } else {
        // column 0 has no lists and no nulls and no strings, so there should be no size stats
        EXPECT_FALSE(chunk.meta_data.size_statistics.has_value());
      }

      int64_t num_vals = 0;

      if (is_v2) { ASSERT_TRUE(ci.null_counts.has_value()); }
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
        // check that null counts match
        if (is_v2) { EXPECT_EQ(ci.null_counts.value()[o], ph.data_page_header_v2.num_nulls); }
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnIndexListWithNulls)
{
  auto const is_v2 = GetParam();
  auto const expected_hdr_type =
    is_v2 ? cudf::io::parquet::PageType::DATA_PAGE_V2 : cudf::io::parquet::PageType::DATA_PAGE;

  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;
  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // 4 nulls
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  // def histogram [1, 1, 2, 3]
  // rep histogram [4, 3]
  lcw col0{{{{1, 2, 3}, nulls_at({0, 2})}, {}, {4, 5}, {}}, null_at(3)};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  // def histogram [1, 3, 10]
  // rep histogram [4, 4, 6]
  lcw col1{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  // def histogram [1, 1, 2, 10]
  // rep histogram [4, 4, 6]
  lcw col2{{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, null_at(3)}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // 6 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // [[]]
  // def histogram [1, 1, 2, 2, 8]
  // rep histogram [4, 4, 6]
  using dlcw = cudf::test::lists_column_wrapper<double>;
  dlcw col3{{{{1., 2., 3.}, {}, {4., 5.}, {}, {{0., 6., 0.}, nulls_at({0, 2})}}, null_at(3)},
            {{7., 8.}},
            dlcw{},
            dlcw{dlcw{}}};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // NULL
  // def histogram [1, 1, 1, 1, 10]
  // rep histogram [4, 4, 6]
  using ui16lcw = cudf::test::lists_column_wrapper<uint16_t>;
  cudf::test::lists_column_wrapper<uint16_t> col4{
    {{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, null_at(3)}, {{7, 8}}, ui16lcw{}, ui16lcw{ui16lcw{}}},
    null_at(3)};

  // 6 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // NULL
  // def histogram [1, 1, 1, 1, 2, 8]
  // rep histogram [4, 4, 6]
  lcw col5{{{{{1, 2, 3}, {}, {4, 5}, {}, {{0, 6, 0}, nulls_at({0, 2})}}, null_at(3)},
            {{7, 8}},
            lcw{},
            lcw{lcw{}}},
           null_at(3)};

  // 4 nulls
  // def histogram [1, 3, 9]
  // rep histogram [4, 4, 5]
  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col6{
    {{"Monday", "Monday", "Friday"}, {}, {"Monday", "Friday"}, {}, {"Sunday", "Funday"}},
    {{"bee", "sting"}},
    strlcw{},
    strlcw{strlcw{}}};

  // 5 nulls
  // def histogram [1, 3, 1, 8]
  // rep histogram [4, 4, 5]
  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col7{{{"Monday", "Monday", "Friday"},
                                                            {},
                                                            {{"Monday", "Friday"}, null_at(1)},
                                                            {},
                                                            {"Sunday", "Funday"}},
                                                           {{"bee", "sting"}},
                                                           strlcw{},
                                                           strlcw{strlcw{}}};

  // 11 nulls
  // D   5   6   5  6        5  6  5      6 6
  // R   0   3   3  3        1  3  3      2 3
  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // D 2      6    6   6  6      2
  // R 0      1    2   3  3      1
  // [NULL, [[13],[14,15,16]],  NULL]
  // D 2     3   2      4
  // R 0     1   1      1
  // [NULL, [], NULL, [[]]]
  // D 0
  // R 0
  // NULL
  // def histogram [1, 0, 4, 1, 1, 4, 9]
  // rep histogram [4, 6, 2, 8]
  lcw col8{{
             {{{{1, 2, 3, 4}, nulls_at({0, 2})}}, {{{5, 6, 7}, nulls_at({0, 2})}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, nulls_at({0, 2})},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, nulls_at({0, 2})},
             lcw{lcw{lcw{}}},
           },
           null_at(3)};

  table_view expected({col0, col1, col2, col3, col4, col5, col6, col7});

  std::array<int64_t, 9> expected_null_counts{4, 4, 4, 6, 4, 6, 4, 5, 11};
  std::vector<std::vector<int64_t>> const expected_def_hists = {{1, 1, 2, 3},
                                                                {1, 3, 10},
                                                                {1, 1, 2, 10},
                                                                {1, 1, 2, 2, 8},
                                                                {1, 1, 1, 1, 10},
                                                                {1, 1, 1, 1, 2, 8},
                                                                {1, 3, 9},
                                                                {1, 3, 1, 8},
                                                                {1, 0, 4, 1, 1, 4, 9}};
  std::vector<std::vector<int64_t>> const expected_rep_hists = {{4, 3},
                                                                {4, 4, 6},
                                                                {4, 4, 6},
                                                                {4, 4, 6},
                                                                {4, 4, 6},
                                                                {4, 4, 6},
                                                                {4, 4, 5},
                                                                {4, 4, 5},
                                                                {4, 6, 2, 8}};

  auto const filepath = temp_env->get_temp_filepath("ColumnIndexListWithNulls.parquet");
  auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
                    .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
                    .write_v2_headers(is_v2)
                    .page_level_compression(is_v2)
                    .compression(cudf::io::compression_type::NONE);

  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);

  for (auto const& rg : fmd.row_groups) {
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      ASSERT_TRUE(chunk.meta_data.size_statistics.has_value());
      ASSERT_TRUE(chunk.meta_data.size_statistics->definition_level_histogram.has_value());
      ASSERT_TRUE(chunk.meta_data.size_statistics->repetition_level_histogram.has_value());
      // there is only one page, so chunk stats should match the page stats
      EXPECT_EQ(chunk.meta_data.size_statistics->definition_level_histogram.value(),
                expected_def_hists[c]);
      EXPECT_EQ(chunk.meta_data.size_statistics->repetition_level_histogram.value(),
                expected_rep_hists[c]);
      // only column 6 has string data
      if (c == 6) {
        ASSERT_TRUE(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.value(), 50L);
      } else if (c == 7) {
        ASSERT_TRUE(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.value(), 44L);
      } else {
        EXPECT_FALSE(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.has_value());
      }

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      for (auto const& page_loc : oi.page_locations) {
        auto const ph = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        // check null counts in V2 header
        if (is_v2) { EXPECT_EQ(ph.data_page_header_v2.num_nulls, expected_null_counts[c]); }
      }

      // check null counts in column chunk stats and page indexes
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);
      EXPECT_EQ(stats.null_count, expected_null_counts[c]);

      // should only be one page
      EXPECT_FALSE(ci.null_pages[0]);
      ASSERT_TRUE(ci.null_counts.has_value());
      EXPECT_EQ(ci.null_counts.value()[0], expected_null_counts[c]);

      ASSERT_TRUE(ci.definition_level_histogram.has_value());
      EXPECT_EQ(ci.definition_level_histogram.value(), expected_def_hists[c]);

      ASSERT_TRUE(ci.repetition_level_histogram.has_value());
      EXPECT_EQ(ci.repetition_level_histogram.value(), expected_rep_hists[c]);

      if (c == 6) {
        ASSERT_TRUE(oi.unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(oi.unencoded_byte_array_data_bytes.value()[0], 50L);
      } else if (c == 7) {
        ASSERT_TRUE(oi.unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(oi.unencoded_byte_array_data_bytes.value()[0], 44L);
      } else {
        EXPECT_FALSE(oi.unencoded_byte_array_data_bytes.has_value());
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckEncodings)
{
  using cudf::io::parquet::Encoding;
  constexpr auto num_rows = 100'000;
  auto const is_v2        = GetParam();

  auto const validity = cudf::test::iterators::no_nulls();
  // data should be PLAIN for v1, RLE for V2
  auto col0_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 2 == 0; });
  // data should be PLAIN for v1, DELTA_BINARY_PACKED for v2
  auto col1_data = random_values<int32_t>(num_rows);
  // data should be PLAIN_DICTIONARY for v1, PLAIN and RLE_DICTIONARY for v2
  auto col2_data = cuda::constant_iterator{1};

  cudf::test::fixed_width_column_wrapper<bool> col0{col0_data, col0_data + num_rows, validity};
  column_wrapper<int32_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int32_t> col2{col2_data, col2_data + num_rows, validity};

  auto expected = table_view{{col0, col1, col2}};

  auto const filename = is_v2 ? "CheckEncodingsV2.parquet" : "CheckEncodingsV1.parquet";
  auto filepath       = temp_env->get_temp_filepath(filename);
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(num_rows)
      .write_v2_headers(is_v2)
      .page_level_compression(is_v2);
  cudf::io::write_parquet(out_opts);

  // make sure the expected encodings are present
  auto contains = [](auto const& vec, auto const& enc) {
    return std::find(vec.begin(), vec.end(), enc) != vec.end();
  };

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;

  read_footer(source, &fmd);
  auto const& chunk0_enc = fmd.row_groups[0].columns[0].meta_data.encodings;
  auto const& chunk1_enc = fmd.row_groups[0].columns[1].meta_data.encodings;
  auto const& chunk2_enc = fmd.row_groups[0].columns[2].meta_data.encodings;
  if (is_v2) {
    // col0 should have RLE for rep/def and data
    EXPECT_TRUE(chunk0_enc.size() == 1);
    EXPECT_TRUE(contains(chunk0_enc, Encoding::RLE));
    // col1 should have RLE for rep/def and DELTA_BINARY_PACKED for data
    EXPECT_TRUE(chunk1_enc.size() == 2);
    EXPECT_TRUE(contains(chunk1_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk1_enc, Encoding::DELTA_BINARY_PACKED));
    // col2 should have RLE for rep/def, PLAIN for dict, and RLE_DICTIONARY for data
    EXPECT_TRUE(chunk2_enc.size() == 3);
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::PLAIN));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE_DICTIONARY));
  } else {
    // col0 should have RLE for rep/def and PLAIN for data
    EXPECT_TRUE(chunk0_enc.size() == 2);
    EXPECT_TRUE(contains(chunk0_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk0_enc, Encoding::PLAIN));
    // col1 should have RLE for rep/def and PLAIN for data
    EXPECT_TRUE(chunk1_enc.size() == 2);
    EXPECT_TRUE(contains(chunk1_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk1_enc, Encoding::PLAIN));
    // col2 should have RLE for rep/def and PLAIN_DICTIONARY for data and dict
    EXPECT_TRUE(chunk2_enc.size() == 2);
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::PLAIN_DICTIONARY));
  }
}

TEST_F(ParquetWriterTest, V2PageLevelCompression)
{
  constexpr auto num_rows           = 20000;
  constexpr auto max_page_size_rows = 5000;

  // Create a column with highly compressible data (constant value)
  auto compressible_data = std::vector<int64_t>(num_rows, 12345);
  auto col0 = cudf::test::fixed_width_column_wrapper<int64_t>(compressible_data.begin(),
                                                              compressible_data.end());

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("V2PageLevelCompression.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .page_level_compression(true)
      .compression(cudf::io::compression_type::SNAPPY)
      .max_page_size_rows(max_page_size_rows)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;
  read_footer(source, &fmd);

  ASSERT_TRUE(fmd.row_groups.size() > 0);
  auto const& chunk = fmd.row_groups[0].columns[0];
  EXPECT_EQ(chunk.meta_data.codec, cudf::io::parquet::Compression::SNAPPY);

  auto const oi = read_offset_index(source, chunk);
  ASSERT_TRUE(oi.page_locations.size() > 1);

  for (auto const& page_loc : oi.page_locations) {
    auto const ph = read_page_header(source, page_loc);
    EXPECT_EQ(ph.type, cudf::io::parquet::PageType::DATA_PAGE_V2);
    EXPECT_TRUE(ph.data_page_header_v2.is_compressed);
    EXPECT_LT(ph.compressed_page_size, ph.uncompressed_page_size);
  }
}

TEST_F(ParquetWriterTest, V2PageLevelCompressionMixed)
{
  constexpr auto num_rows           = 20000;
  constexpr auto max_page_size_rows = 5000;

  // Create a column with mixed data:
  // First half: random data (won't compress well)
  // Second half: constant data (will compress well)
  std::vector<int64_t> mixed_data;
  mixed_data.reserve(num_rows);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                              std::numeric_limits<int64_t>::max());
  for (int i = 0; i < num_rows / 2; i++) {
    mixed_data.push_back(dist(gen));
  }

  for (int i = 0; i < num_rows / 2; i++) {
    mixed_data.push_back(42);
  }

  auto col0 = cudf::test::fixed_width_column_wrapper<int64_t>(mixed_data.begin(), mixed_data.end());

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("V2PageLevelCompressionMixed.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .page_level_compression(true)
      .compression(cudf::io::compression_type::SNAPPY)
      .max_page_size_rows(max_page_size_rows)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::FileMetaData fmd;
  read_footer(source, &fmd);

  ASSERT_TRUE(fmd.row_groups.size() > 0);
  auto const& chunk = fmd.row_groups[0].columns[0];

  auto const oi = read_offset_index(source, chunk);
  ASSERT_TRUE(oi.page_locations.size() > 1);

  bool found_compressed = false;
  for (auto const& page_loc : oi.page_locations) {
    auto const ph = read_page_header(source, page_loc);
    EXPECT_EQ(ph.type, cudf::io::parquet::PageType::DATA_PAGE_V2);

    if (ph.data_page_header_v2.is_compressed) {
      found_compressed = true;
      EXPECT_LT(ph.compressed_page_size, ph.uncompressed_page_size);
    } else {
      EXPECT_EQ(ph.compressed_page_size, ph.uncompressed_page_size);
    }
  }

  EXPECT_TRUE(found_compressed);
}
