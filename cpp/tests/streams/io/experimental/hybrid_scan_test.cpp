/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>

#include <string>
#include <vector>

namespace {

template <typename... UniqPtrs>
std::vector<std::unique_ptr<cudf::column>> make_unique_ptrs_vector(UniqPtrs&&... uniqptrs)
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

  auto colsptr = make_unique_ptrs_vector(col0.release(),
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
}  // namespace

class HybridScanTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanTest, DictionaryPageFiltering)
{
  auto table  = construct_table();
  auto buffer = std::vector<char>();
  cudf::io::table_input_metadata out_metadata(table);
  out_metadata.column_metadata[0].set_name("col0");
  out_metadata.column_metadata[3].set_name("col3");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, table)
      .metadata(out_metadata)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts, cudf::test::get_default_stream());

  auto col3_ref      = cudf::ast::column_name_reference("col3");
  auto literal_value = cudf::numeric_scalar<int32_t>(0, true, cudf::test::get_default_stream());
  auto literal       = cudf::ast::literal(literal_value);
  auto expr1         = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, literal, col3_ref);

  auto col0_ref = cudf::ast::column_name_reference("col0");
  auto expr2    = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, col0_ref);
  auto expr3    = cudf::ast::operation(cudf::ast::ast_operator::NOT, expr2);

  auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr3);
  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{}).filter(filter_expr);

  auto const datasource    = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()));
  auto datasource_ref      = std::ref(*datasource);
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource_ref);

  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, in_opts);

  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(datasource_ref, page_index_byte_range);
  reader->setup_page_index(*page_index_buffer);

  auto input_row_group_indices = reader->all_row_groups(in_opts);

  auto const dict_byte_ranges =
    reader->dictionary_pages_byte_ranges(input_row_group_indices, in_opts);
  auto [dict_page_buffers, dict_page_data, dict_page_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource_ref,
      dict_byte_ranges,
      cudf::io::parquet::io_submission_policy::SERIALIZE,
      cudf::test::get_default_stream());
  dict_page_tasks.get();

  auto result = reader->filter_row_groups_with_dictionary_pages(
    dict_page_data, input_row_group_indices, in_opts, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
