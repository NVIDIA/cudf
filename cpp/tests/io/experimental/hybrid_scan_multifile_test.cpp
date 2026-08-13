/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"
#include "hybrid_scan_multifile_composer.hpp"
#include "tests/io/parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

/**
 * @brief Helper to test multifile hybrid scan single-shot materialization
 *
 * Writes the input table to multiple parquet sources and compares filter, payload, and all-column
 * materialization output with the regular multi-source parquet reader. The filter expression used
 * is `col0 >= 100`.
 *
 * @note The first column in the input table must be constructed with
 * `cudf::test::ascending<uint32_t>()`
 */
template <int num_sources = 2, int num_rows = num_ordered_rows>
void test_hybrid_scan_multifile(std::vector<cudf::column_view> const& columns,
                                bool case_sensitive_names = true,
                                uint32_t literal_value    = 100)
{
  auto const table = cudf::table_view{columns};
  cudf::io::table_input_metadata expected_metadata(table);
  expected_metadata.column_metadata[0].set_name("col0");

  std::vector<std::vector<char>> parquet_buffers(num_sources);
  for (auto& parquet_buffer : parquet_buffers) {
    auto out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&parquet_buffer}, table)
        .metadata(expected_metadata)
        .row_group_size_rows(num_rows)
        .max_page_size_rows(page_size_for_ordered_tests)
        .compression(cudf::io::compression_type::AUTO)
        .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
    cudf::io::write_parquet(out_opts);
  }

  auto scalar    = cudf::numeric_scalar<uint32_t>(literal_value);
  auto literal   = cudf::ast::literal(scalar);
  auto col_ref_0 = cudf::ast::column_name_reference(case_sensitive_names ? "col0" : "CoL0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto source_info  = build_source_info(parquet_buffers);

  auto const expected_options = cudf::io::parquet_reader_options::builder(source_info)
                                  .filter(filter_expression)
                                  .case_sensitive_names(case_sensitive_names)
                                  .build();
  auto const expected = cudf::io::read_parquet(expected_options, stream, mr);

  auto const [filter_table, payload_table] =
    hybrid_scan_multifile(source_info, filter_expression, {}, case_sensitive_names, stream, mr);

  auto const all_table = hybrid_scan_multifile_single_step(
    source_info, filter_expression, {}, case_sensitive_names, stream, mr);

  auto const [chunked_filter_table, chunked_payload_table] = chunked_hybrid_scan_multifile(
    source_info, filter_expression, {}, case_sensitive_names, stream, mr);

  auto const chunked_all_table = chunked_hybrid_scan_multifile_single_step(
    source_info, filter_expression, {}, case_sensitive_names, stream, mr);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), chunked_filter_table->view());

  auto payload_column_indices = std::vector<cudf::size_type>(columns.size() - 1);
  std::iota(payload_column_indices.begin(), payload_column_indices.end(), 1);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select(payload_column_indices),
                                     payload_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select(payload_column_indices),
                                     chunked_payload_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->view(), all_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->view(), chunked_all_table->view());
}

}  // namespace

struct HybridScanMultifileTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanMultifileTest, EmptyResult)
{
  std::mt19937 gen(0xc0c0a);

  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = make_list_str_column(gen, false, false);
  auto col2 = make_list_str_column(gen, false, true);
  auto col3 = make_list_str_column(gen, true, false);
  auto col4 = make_list_str_column(gen, true, true);

  auto constexpr literal_value = uint32_t(num_ordered_rows);
  test_hybrid_scan_multifile({col0, *col1, *col2, *col3, *col4}, false, literal_value);
}

TEST_F(HybridScanMultifileTest, MaterializeLists)
{
  std::mt19937 gen(0xadd);

  auto constexpr num_rows          = num_ordered_rows;
  auto constexpr lists_per_row     = 2;
  auto constexpr max_vals_per_list = 3;

  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = make_parquet_list_col<int32_t>(gen, num_rows, max_vals_per_list, true);
  auto col2 =
    make_parquet_list_list_col<int32_t>(0, num_rows, lists_per_row, max_vals_per_list, true);
  auto col3 = make_parquet_list_col<int32_t>(gen, num_rows, max_vals_per_list, false);
  auto col4 =
    make_parquet_list_list_col<int32_t>(0, num_rows, lists_per_row, max_vals_per_list, false);
  auto col5 = make_parquet_list_list_col<bool>(0, num_rows, lists_per_row, max_vals_per_list, true);

  test_hybrid_scan_multifile({col0, *col1, *col2, *col3, *col4, *col5});
}

TEST_F(HybridScanMultifileTest, MaterializeListsOfStrings)
{
  std::mt19937 gen(0xc0c0a);

  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = make_list_str_column(gen, false, false);
  auto col2 = make_list_str_column(gen, false, true);
  auto col3 = make_list_str_column(gen, true, false);
  auto col4 = make_list_str_column(gen, true, true);

  test_hybrid_scan_multifile({col0, *col1, *col2, *col3, *col4}, false);
}

TEST_F(HybridScanMultifileTest, PrependIndexColumns)
{
  using T = int32_t;
  using cudf::io::parquet::experimental::use_data_page_mask;

  // Small single-column table with sequence values [0, 10)
  auto constexpr num_rows    = 10;
  auto constexpr num_sources = 3;
  auto values                = cuda::counting_iterator<T>{0};
  cudf::test::fixed_width_column_wrapper<T> col0(values, values + num_rows);
  auto const table = cudf::table_view{{col0, col0}};

  cudf::io::table_input_metadata expected_metadata(table);
  expected_metadata.column_metadata[0].set_name("col0");
  expected_metadata.column_metadata[1].set_name("col1");

  // Write the table once and reference the same file for all sources
  auto const parquet_filepath = temp_env->get_temp_filepath("PrependIndexColumns.parquet");
  {
    auto out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{parquet_filepath}, table)
        .metadata(expected_metadata)
        .build();
    cudf::io::write_parquet(out_opts);
  }

  // Filtering AST - col0 % 2 == 0, removes odd rows (half the rows) from each source
  auto two_scalar     = cudf::numeric_scalar<T>(2);
  auto two_literal    = cudf::ast::literal(two_scalar);
  auto zero_scalar    = cudf::numeric_scalar<T>(0);
  auto zero_literal   = cudf::ast::literal(zero_scalar);
  auto col_ref_0      = cudf::ast::column_name_reference("col0");
  auto mod_expression = cudf::ast::operation(cudf::ast::ast_operator::MOD, col_ref_0, two_literal);
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, mod_expression, zero_literal);

  // Build expected table with source and row index columns
  auto const source_index =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i / 5; });
  auto const expected_source_index = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    source_index, source_index + num_sources * 5);

  auto const row_index = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) -> size_t { return (i % 5) * 2; });
  auto const expected_row_index =
    cudf::test::fixed_width_column_wrapper<size_t>(row_index, row_index + num_sources * 5);

  auto const filtered_values = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) -> T { return static_cast<T>((i % 5) * 2); });
  auto const expected_values =
    cudf::test::fixed_width_column_wrapper<T>(filtered_values, filtered_values + num_sources * 5);

  auto const expected_table =
    cudf::table_view{{expected_source_index, expected_row_index, expected_values}};

  // Hybrid scan multifile reader options
  auto const options = cudf::io::parquet_reader_options::builder()
                         .filter(filter_expression)
                         .prepend_source_index_column(true)
                         .prepend_row_index_column(true)
                         .build();

  auto const parquet_filepaths = std::vector<std::string>(num_sources, parquet_filepath);
  auto inputs                  = multifile_inputs(cudf::io::source_info(parquet_filepaths));
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const row_group_indices = reader.all_row_groups(options);
  auto row_mask                = reader.build_all_true_row_mask(row_group_indices, stream, mr);

  // Materialize filter column prepended with index columns
  auto filter_column_chunks = fetch_multisource_device_data(
    inputs, reader.filter_column_chunks_byte_ranges(row_group_indices, options), stream, mr);
  auto row_mask_view = row_mask->mutable_view();
  auto filter_result = reader.materialize_filter_columns(row_group_indices,
                                                         filter_column_chunks.flat_spans,
                                                         row_mask_view,
                                                         use_data_page_mask::NO,
                                                         options,
                                                         stream,
                                                         mr);

  ASSERT_EQ(filter_result.tbl->num_columns(), 3);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_table, filter_result.tbl->view());

  // Materialize payload column (no prepended index columns)
  auto payload_column_chunks = fetch_multisource_device_data(
    inputs, reader.payload_column_chunks_byte_ranges(row_group_indices, options), stream, mr);
  auto payload_result = reader.materialize_payload_columns(row_group_indices,
                                                           payload_column_chunks.flat_spans,
                                                           row_mask->view(),
                                                           use_data_page_mask::NO,
                                                           options,
                                                           stream,
                                                           mr);
  ASSERT_EQ(payload_result.tbl->num_columns(), 1);
  // col1 (payload) must be identical to col0 (filter) with the same row_mask
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_table.select({2}), payload_result.tbl->view());
}

TEST_F(HybridScanMultifileTest, MaterializeStructs)
{
  std::mt19937 gen(0xbaLL);

  auto constexpr num_rows = num_ordered_rows;

  auto col0 = testdata::ascending<uint32_t>();

  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto struct_valids_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 121; });
  std::vector<bool> struct_valids(num_rows);
  std::copy(struct_valids_iter, struct_valids_iter + num_rows, struct_valids.begin());

  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  auto values    = cuda::counting_iterator<int>{0};
  auto col1_list = make_list_str_column(gen, true, true);
  cudf::test::fixed_width_column_wrapper<int> col1_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> col1_floats(values, values + num_rows);
  std::vector<std::unique_ptr<cudf::column>> col1_children;
  col1_children.push_back(std::move(col1_list));
  col1_children.push_back(col1_ints.release());
  col1_children.push_back(col1_floats.release());
  cudf::test::structs_column_wrapper _col1(std::move(col1_children), struct_valids);
  auto col1 = cudf::purge_nonempty_nulls(_col1);

  auto col2_str = cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};
  auto col2_str_non_nullable =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows};
  auto col2_bool = cudf::test::fixed_width_column_wrapper<bool>(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> col2_children;
  col2_children.push_back(col2_str.release());
  col2_children.push_back(col2_str_non_nullable.release());
  col2_children.push_back(col2_bool.release());
  cudf::test::structs_column_wrapper _col2(std::move(col2_children));
  auto col2 = cudf::purge_nonempty_nulls(_col2);

  test_hybrid_scan_multifile({col0, *col1, *col2});
}
