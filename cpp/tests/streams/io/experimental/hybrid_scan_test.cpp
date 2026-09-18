/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
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

  auto const dict_page_ranges =
    reader->dictionary_pages_byte_ranges(input_row_group_indices, in_opts);
  auto const dict_byte_ranges =
    cudf::io::parquet::experimental::dictionary_page_byte_ranges_to_read(dict_page_ranges);

  // Trim each range to exactly one dictionary page (or an empty span when the chunk has none), so
  // the reader never reads following data-page bytes as dictionary data.
  auto const stream = cudf::test::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<rmm::device_buffer> dict_page_buffers;
  std::vector<cudf::device_span<uint8_t const>> dict_page_data;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> host_reads;
  for (std::size_t i = 0; i < dict_byte_ranges.size(); ++i) {
    auto const read_size = dict_byte_ranges[i].size();
    if (read_size == 0) {
      dict_page_buffers.emplace_back();
      dict_page_data.emplace_back();
      continue;
    }
    auto host_buffer = datasource->host_read(dict_byte_ranges[i].offset(), read_size);
    auto const bytes = cudf::host_span<uint8_t const>{host_buffer->data(), host_buffer->size()};
    auto const page_size =
      (dict_page_ranges[i].extent ==
       cudf::io::parquet::experimental::dictionary_page_extent::upper_bound_if_present)
        ? cudf::io::parquet::experimental::dictionary_page_length(bytes).value_or(0)
        : static_cast<int64_t>(bytes.size());
    if (page_size == 0) {
      dict_page_buffers.emplace_back();
      dict_page_data.emplace_back();
      continue;
    }
    auto const page_bytes = static_cast<std::size_t>(page_size);
    auto device_buffer    = rmm::device_buffer{bytes.data(), page_bytes, stream, mr};
    dict_page_data.emplace_back(static_cast<uint8_t const*>(device_buffer.data()), page_bytes);
    dict_page_buffers.emplace_back(std::move(device_buffer));
    host_reads.emplace_back(std::move(host_buffer));
  }
  stream.sync();

  auto result = reader->filter_row_groups_with_dictionary_pages(
    dict_page_data, input_row_group_indices, in_opts, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
