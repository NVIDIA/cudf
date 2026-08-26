/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>

#include <string>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class ParquetTest : public cudf::test::BaseFixture {};

template <typename... UniqPtrs>
std::vector<std::unique_ptr<cudf::column>> make_uniqueptrs_vector(UniqPtrs&&... uniqptrs)
{
  std::vector<std::unique_ptr<cudf::column>> ptrsvec;
  (ptrsvec.push_back(std::forward<UniqPtrs>(uniqptrs)), ...);
  return ptrsvec;
}

cudf::table construct_table()
{
  constexpr auto num_rows = 10;

  std::vector<size_t> zeros(num_rows, 0);
  std::vector<size_t> ones(num_rows, 1);

  cudf::test::fixed_width_column_wrapper<bool> col0(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<float> col4(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<double> col5(zeros.begin(), zeros.end());
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col6(
    ones.begin(), ones.end(), numeric::scale_type{12});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col7(
    ones.begin(), ones.end(), numeric::scale_type{-12});

  cudf::test::lists_column_wrapper<int64_t> col8{
    {1, 1}, {1, 1, 1}, {}, {1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1}, {}, {1, -1}, {}, {-1, -1}};

  cudf::test::structs_column_wrapper col9 = [&ones] {
    cudf::test::fixed_width_column_wrapper<int32_t> child_col(ones.begin(), ones.end());
    return cudf::test::structs_column_wrapper{child_col};
  }();

  cudf::test::strings_column_wrapper col10 = [] {
    std::vector<std::string> col10_data(num_rows, "rapids");
    return cudf::test::strings_column_wrapper(col10_data.begin(), col10_data.end());
  }();

  auto colsptr = make_uniqueptrs_vector(col0.release(),
                                        col1.release(),
                                        col2.release(),
                                        col3.release(),
                                        col4.release(),
                                        col5.release(),
                                        col6.release(),
                                        col7.release(),
                                        col8.release(),
                                        col9.release(),
                                        col10.release());
  return cudf::table(std::move(colsptr));
}

TEST_F(ParquetTest, ParquetWriter)
{
  auto tab      = construct_table();
  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tab);
  cudf::io::write_parquet(out_opts, cudf::test::get_default_stream());
}

TEST_F(ParquetTest, ParquetReader)
{
  auto tab      = construct_table();
  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tab);
  cudf::io::write_parquet(out_opts, cudf::test::get_default_stream());

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts, cudf::test::get_default_stream());
  auto meta   = cudf::io::read_parquet_metadata(cudf::io::source_info{filepath});
}

TEST_F(ParquetTest, ParquetReaderPredicatePushdown)
{
  auto tab      = construct_table();
  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tab);
  cudf::io::write_parquet(out_opts, cudf::test::get_default_stream());

  auto col3_ref      = cudf::ast::column_reference(3);
  auto literal_value = cudf::numeric_scalar<int32_t>(0, true, cudf::test::get_default_stream());
  auto literal       = cudf::ast::literal(literal_value);
  auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col3_ref, literal);

  auto col0_ref = cudf::ast::column_reference(0);
  auto expr2    = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, col0_ref);
  auto expr3    = cudf::ast::operation(cudf::ast::ast_operator::NOT, expr2);

  auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr3);
  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).filter(filter_expr);
  auto result = cudf::io::read_parquet(in_opts, cudf::test::get_default_stream());
}

TEST_F(ParquetTest, ChunkedOperations)
{
  auto tab      = construct_table();
  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf::io::chunked_parquet_writer_options out_opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::chunked_parquet_writer(out_opts, cudf::test::get_default_stream()).write(tab);

  auto reader = cudf::io::chunked_parquet_reader(
    1L << 31,
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}),
    cudf::test::get_default_stream());
  while (reader.has_next()) {
    auto chunk = reader.read_chunk();
  }
}

CUDF_TEST_PROGRAM_MAIN()
