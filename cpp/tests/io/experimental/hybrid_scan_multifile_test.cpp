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
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using cudf::io::parquet::experimental::use_data_page_mask;

/**
 * @brief Helper to test multifile hybrid scan single-shot materialization
 *
 * Writes the input table to multiple parquet sources and compares filter, payload, and all-column
 * materialization output with the regular multi-source parquet reader. The filter expression is
 * `col0 >= literal_value`.
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

  auto const [sparse_filter_table, sparse_payload_table] = chunked_sparse_hybrid_scan_multifile(
    source_info, filter_expression, {}, case_sensitive_names, stream, mr);

  auto const chunked_all_table = chunked_hybrid_scan_multifile_single_step(
    source_info, filter_expression, {}, case_sensitive_names, stream, mr);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), chunked_filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), sparse_filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunked_filter_table->view(), sparse_filter_table->view());

  auto payload_column_indices = std::vector<cudf::size_type>(columns.size() - 1);
  std::iota(payload_column_indices.begin(), payload_column_indices.end(), 1);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select(payload_column_indices),
                                     payload_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select(payload_column_indices),
                                     chunked_payload_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select(payload_column_indices),
                                     sparse_payload_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunked_payload_table->view(), sparse_payload_table->view());
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

  auto constexpr literal_value = static_cast<uint32_t>(num_ordered_rows);
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

TEST_F(HybridScanMultifileTest, PageLevelDictionaryPayloadByteReduction)
{
  auto col0 = testdata::ascending<uint32_t>();

  auto payload_values = std::vector<std::string>(num_ordered_rows);
  for (auto i = std::size_t{0}; i < payload_values.size(); ++i) {
    payload_values[i] = "dictionary value " + std::to_string(i % 8);
  }
  auto col1 = cudf::test::strings_column_wrapper(payload_values.begin(), payload_values.end());

  // A page-aligned threshold retains two of four data pages. The writer's ALWAYS dictionary policy
  // requires the page-I/O path to retain the dictionary while requesting fewer bytes than the
  // legacy full-column-chunk path.
  auto constexpr threshold = uint32_t{2 * page_size_for_ordered_tests / 100};
  test_hybrid_scan_multifile({col0, col1}, true, threshold, true);
}

TEST_F(HybridScanMultifileTest, PageLevelStringsSeparatedByPrunedPages)
{
  auto filter_values = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, [](auto i) { return (i / page_size_for_ordered_tests) % 2 == 0; });
  auto filter =
    cudf::test::fixed_width_column_wrapper<bool>(filter_values, filter_values + num_ordered_rows);

  auto payload_values = std::vector<std::string>(num_ordered_rows);
  for (auto i = std::size_t{0}; i < payload_values.size(); ++i) {
    payload_values[i] = "payload value " + std::to_string(i);
  }
  auto payload = cudf::test::strings_column_wrapper(payload_values.begin(), payload_values.end());
  auto table   = cudf::table_view{{filter, payload}};

  auto metadata = cudf::io::table_input_metadata(table);
  metadata.column_metadata[0].set_name("filter");
  metadata.column_metadata[1].set_name("payload");

  auto parquet_buffers = std::vector<std::vector<char>>(2);
  for (auto& parquet_buffer : parquet_buffers) {
    auto options =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&parquet_buffer}, table)
        .metadata(metadata)
        .row_group_size_rows(num_ordered_rows)
        .max_page_size_rows(page_size_for_ordered_tests)
        .max_page_size_bytes(64 * 1024 * 1024)
        .compression(cudf::io::compression_type::NONE)
        .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
    cudf::io::write_parquet(options);
  }

  auto const filter_ref  = cudf::ast::column_name_reference("filter");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, filter_ref);
  auto source_info       = build_source_info(parquet_buffers);
  auto const stream      = cudf::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();
  auto expected_options =
    cudf::io::parquet_reader_options::builder(source_info).filter(filter_expression).build();
  auto expected = cudf::io::read_parquet(expected_options, stream, mr);

  auto const [filter_result, payload_result] =
    page_level_chunked_hybrid_scan_multifile(source_info, filter_expression, {}, true, stream, mr);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), filter_result->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({1}), payload_result->view());
}

TEST_F(HybridScanMultifileTest, PageLevelAsymmetricSourceRowGroupOrdering)
{
  auto constexpr rows_per_group = 2 * page_size_for_ordered_tests;
  auto constexpr rows_source_0  = num_ordered_rows;
  auto constexpr rows_source_1  = num_ordered_rows;
  auto constexpr rows_per_page  = rows_per_group / 4;

  auto source_0_filter_values = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<uint32_t>(i / 100); });
  auto source_1_filter_values = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<uint32_t>((num_ordered_rows - i) / 100); });
  auto source_0_filter = cudf::test::fixed_width_column_wrapper<uint32_t>(
    source_0_filter_values, source_0_filter_values + rows_source_0);
  auto source_1_filter = cudf::test::fixed_width_column_wrapper<uint32_t>(
    source_1_filter_values, source_1_filter_values + rows_source_1);

  auto source_0_payload_values = std::vector<std::string>(rows_source_0);
  auto source_1_payload_values = std::vector<std::string>(rows_source_1);
  for (auto i = std::size_t{0}; i < source_0_payload_values.size(); ++i) {
    source_0_payload_values[i] = "source 0 dictionary value " + std::to_string(i % 8);
  }
  for (auto i = std::size_t{0}; i < source_1_payload_values.size(); ++i) {
    source_1_payload_values[i] = "source 1 dictionary value " + std::to_string(i % 8);
  }
  auto source_0_payload     = cudf::test::strings_column_wrapper(source_0_payload_values.begin(),
                                                             source_0_payload_values.end());
  auto source_1_payload     = cudf::test::strings_column_wrapper(source_1_payload_values.begin(),
                                                             source_1_payload_values.end());
  auto const source_0_table = cudf::table_view{{source_0_filter, source_0_payload}};
  auto const source_1_table = cudf::table_view{{source_1_filter, source_1_payload}};

  auto parquet_buffers    = std::vector<std::vector<char>>(2);
  auto const write_source = [&](auto const& table, auto& buffer) {
    cudf::io::table_input_metadata metadata(table);
    metadata.column_metadata[0].set_name("col0");
    auto options = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, table)
                     .metadata(metadata)
                     .row_group_size_rows(rows_per_group)
                     .max_page_size_rows(rows_per_page)
                     .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
                     .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
    cudf::io::write_parquet(options);
  };
  write_source(source_0_table, parquet_buffers[0]);
  write_source(source_1_table, parquet_buffers[1]);

  auto constexpr threshold = uint32_t{75};
  auto scalar              = cudf::numeric_scalar<uint32_t>(threshold);
  auto literal             = cudf::ast::literal(scalar);
  auto col_ref             = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);

  auto const stream      = cudf::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();
  auto const source_info = build_source_info(parquet_buffers);
  auto const expected    = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(source_info).filter(filter_expression), stream, mr);

  auto const [filter_table, payload_table] =
    page_level_chunked_hybrid_scan_multifile(source_info, filter_expression, {}, true, stream, mr);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({1}), payload_table->view());
  auto const [requested_payload_bytes, full_payload_bytes] =
    payload_byte_range_sizes(source_info, filter_expression, true, stream, mr);
  EXPECT_GT(requested_payload_bytes, 0);
  EXPECT_LT(requested_payload_bytes, full_payload_bytes);
}

TEST_F(HybridScanMultifileTest, PageLevelPlainEncodingExactCoalescedRanges)
{
  auto parquet_buffers   = make_plain_payload_parquet_buffers();
  auto const source_info = build_source_info(parquet_buffers);
  auto inputs            = multifile_inputs(source_info);

  auto scalar    = cudf::numeric_scalar<uint32_t>(0);
  auto literal   = cudf::ast::literal(scalar);
  auto col_ref_0 = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
  auto options = cudf::io::parquet_reader_options::builder()
                   .column_names({"col1"})
                   .filter(filter_expression)
                   .build();
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const row_groups = reader.all_row_groups(options);
  auto const metadatas  = reader.parquet_metadatas();
  auto selected_rows =
    std::vector<uint8_t>(reader.total_rows_in_row_groups(row_groups), uint8_t{0});
  auto expected_ranges =
    std::vector<std::vector<cudf::io::text::byte_range_info>>(metadatas.size());

  std::size_t source_row_offset = 0;
  for (std::size_t source_idx = 0; source_idx < metadatas.size(); ++source_idx) {
    auto const& metadata = metadatas[source_idx];
    ASSERT_EQ(metadata.row_groups.size(), 1);
    auto const& payload_chunk = metadata.row_groups.front().columns[1];
    EXPECT_NE(std::find(payload_chunk.meta_data.encodings.begin(),
                        payload_chunk.meta_data.encodings.end(),
                        cudf::io::parquet::Encoding::PLAIN),
              payload_chunk.meta_data.encodings.end());
    EXPECT_EQ(std::find(payload_chunk.meta_data.encodings.begin(),
                        payload_chunk.meta_data.encodings.end(),
                        cudf::io::parquet::Encoding::RLE_DICTIONARY),
              payload_chunk.meta_data.encodings.end());
    EXPECT_EQ(payload_chunk.meta_data.dictionary_page_offset, 0);
    ASSERT_TRUE(payload_chunk.offset_index.has_value());

    auto const& pages = payload_chunk.offset_index->page_locations;
    ASSERT_GE(pages.size(), 4);
    ASSERT_EQ(pages[1].offset + pages[1].compressed_page_size, pages[2].offset);
    auto const selected_begin = pages[1].first_row_index;
    auto const selected_end   = pages[3].first_row_index;
    ASSERT_GE(selected_begin, 0);
    ASSERT_LE(selected_end, metadata.row_groups.front().num_rows);
    std::fill(selected_rows.begin() + source_row_offset + selected_begin,
              selected_rows.begin() + source_row_offset + selected_end,
              uint8_t{1});

    expected_ranges[source_idx].emplace_back(
      pages[1].offset, pages[2].offset + pages[2].compressed_page_size - pages[1].offset);
    source_row_offset += metadata.row_groups.front().num_rows;
  }
  ASSERT_EQ(source_row_offset, selected_rows.size());
  auto row_mask =
    cudf::test::fixed_width_column_wrapper<bool>(selected_rows.begin(), selected_rows.end())
      .release();

  auto const stream      = cudf::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();
  auto const page_ranges = reader.payload_column_chunks_byte_ranges(
    row_groups, row_mask->view(), use_data_page_mask::YES, options, stream);
  expect_byte_ranges_equal(expected_ranges, page_ranges);

  auto page_data = fetch_multisource_device_data(inputs, page_ranges, stream, mr);
  reader.setup_chunking_for_payload_columns(256 * 1024,
                                            1024 * 1024,
                                            row_groups,
                                            row_mask->view(),
                                            use_data_page_mask::YES,
                                            page_data.per_source_spans,
                                            options,
                                            stream,
                                            mr);
  auto payload_chunks = std::vector<std::unique_ptr<cudf::table>>{};
  while (reader.has_next_table_chunk()) {
    payload_chunks.push_back(
      std::move(reader.materialize_payload_columns_chunk(row_mask->view()).tbl));
  }
  auto actual = concatenate_tables(std::move(payload_chunks), stream, mr);

  auto const full =
    cudf::io::read_parquet(cudf::io::parquet_reader_options::builder(source_info), stream, mr);
  auto const expected = cudf::apply_boolean_mask(full.tbl->select({1}), row_mask->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), actual->view());
}

TEST_F(HybridScanMultifileTest, PageLevelAllFalseMaskHasNoRanges)
{
  auto parquet_buffers   = make_plain_payload_parquet_buffers();
  auto const source_info = build_source_info(parquet_buffers);
  auto inputs            = multifile_inputs(source_info);

  auto scalar    = cudf::numeric_scalar<uint32_t>(0);
  auto literal   = cudf::ast::literal(scalar);
  auto col_ref_0 = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
  auto options = cudf::io::parquet_reader_options::builder()
                   .column_names({"col1"})
                   .filter(filter_expression)
                   .build();
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const row_groups = reader.all_row_groups(options);
  auto false_values     = cuda::make_constant_iterator(false);
  auto row_mask         = cudf::test::fixed_width_column_wrapper<bool>(
                    false_values, false_values + reader.total_rows_in_row_groups(row_groups))
                    .release();
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const page_ranges = reader.payload_column_chunks_byte_ranges(
    row_groups, row_mask->view(), use_data_page_mask::YES, options, stream);
  ASSERT_EQ(page_ranges.size(), parquet_buffers.size());
  EXPECT_TRUE(std::all_of(
    page_ranges.begin(), page_ranges.end(), [](auto const& ranges) { return ranges.empty(); }));

  auto const empty_page_data =
    std::vector<std::vector<cudf::device_span<uint8_t const>>>(parquet_buffers.size());
  reader.setup_chunking_for_payload_columns(0,
                                            0,
                                            row_groups,
                                            row_mask->view(),
                                            use_data_page_mask::YES,
                                            empty_page_data,
                                            options,
                                            stream,
                                            mr);
  ASSERT_TRUE(reader.has_next_table_chunk());
  auto const result = reader.materialize_payload_columns_chunk(row_mask->view());
  EXPECT_EQ(result.tbl->num_rows(), 0);
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.metadata.num_input_row_groups, 2);
  EXPECT_FALSE(reader.has_next_table_chunk());
}

TEST_F(HybridScanMultifileTest, PageLevelNoMaskFallbackAndPlanLifecycle)
{
  auto parquet_buffers   = make_plain_payload_parquet_buffers();
  auto const source_info = build_source_info(parquet_buffers);
  auto inputs            = multifile_inputs(source_info);

  auto scalar    = cudf::numeric_scalar<uint32_t>(0);
  auto literal   = cudf::ast::literal(scalar);
  auto col_ref_0 = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
  auto options = cudf::io::parquet_reader_options::builder()
                   .column_names({"col1"})
                   .filter(filter_expression)
                   .build();
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const row_groups  = reader.all_row_groups(options);
  auto const full_ranges = group_byte_ranges_by_source(
    reader.payload_column_chunks_byte_ranges(row_groups, options), parquet_buffers.size());
  auto true_values = cuda::make_constant_iterator(true);
  auto row_mask    = cudf::test::fixed_width_column_wrapper<bool>(
                    true_values, true_values + reader.total_rows_in_row_groups(row_groups))
                    .release();
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const planned_ranges = reader.payload_column_chunks_byte_ranges(
    row_groups, row_mask->view(), use_data_page_mask::NO, options, stream);
  expect_byte_ranges_equal(full_ranges, planned_ranges);
  EXPECT_THROW(static_cast<void>(reader.payload_column_chunks_byte_ranges(
                 row_groups, row_mask->view(), use_data_page_mask::NO, options, stream)),
               cudf::logic_error);

  auto page_data = fetch_multisource_device_data(inputs, planned_ranges, stream, mr);
  reader.setup_chunking_for_payload_columns(0,
                                            0,
                                            row_groups,
                                            row_mask->view(),
                                            use_data_page_mask::NO,
                                            page_data.per_source_spans,
                                            options,
                                            stream,
                                            mr);
  EXPECT_THROW(reader.setup_chunking_for_payload_columns(0,
                                                         0,
                                                         row_groups,
                                                         row_mask->view(),
                                                         use_data_page_mask::NO,
                                                         page_data.per_source_spans,
                                                         options,
                                                         stream,
                                                         mr),
               cudf::logic_error);
}

TEST_F(HybridScanMultifileTest, PageLevelRejectsInvalidFetchedSpans)
{
  auto parquet_buffers   = make_plain_payload_parquet_buffers();
  auto const source_info = build_source_info(parquet_buffers);
  auto inputs            = multifile_inputs(source_info);

  auto scalar    = cudf::numeric_scalar<uint32_t>(0);
  auto literal   = cudf::ast::literal(scalar);
  auto col_ref_0 = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);
  auto options = cudf::io::parquet_reader_options::builder()
                   .column_names({"col1"})
                   .filter(filter_expression)
                   .build();
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const row_groups = reader.all_row_groups(options);
  auto selected_values  = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, [](auto i) { return (i % num_ordered_rows) < num_ordered_rows / 2; });
  auto row_mask = cudf::test::fixed_width_column_wrapper<bool>(
                    selected_values, selected_values + reader.total_rows_in_row_groups(row_groups))
                    .release();
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto page_ranges = reader.payload_column_chunks_byte_ranges(
    row_groups, row_mask->view(), use_data_page_mask::YES, options, stream);
  auto page_data    = fetch_multisource_device_data(inputs, page_ranges, stream, mr);
  auto bad_count    = page_data.per_source_spans;
  auto count_source = std::find_if(
    bad_count.begin(), bad_count.end(), [](auto const& spans) { return not spans.empty(); });
  ASSERT_NE(count_source, bad_count.end());
  count_source->pop_back();
  EXPECT_THROW(
    reader.setup_chunking_for_payload_columns(
      0, 0, row_groups, row_mask->view(), use_data_page_mask::YES, bad_count, options, stream, mr),
    cudf::logic_error);

  page_ranges = reader.payload_column_chunks_byte_ranges(
    row_groups, row_mask->view(), use_data_page_mask::YES, options, stream);
  auto bad_size    = page_data.per_source_spans;
  auto size_source = std::find_if(
    bad_size.begin(), bad_size.end(), [](auto const& spans) { return not spans.empty(); });
  ASSERT_NE(size_source, bad_size.end());
  ASSERT_GT(size_source->front().size(), 1);
  size_source->front() =
    cudf::device_span<uint8_t const>{size_source->front().data(), size_source->front().size() - 1};
  EXPECT_THROW(
    reader.setup_chunking_for_payload_columns(
      0, 0, row_groups, row_mask->view(), use_data_page_mask::YES, bad_size, options, stream, mr),
    cudf::logic_error);
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

TEST_F(HybridScanMultifileTest, SparseDictionaryEncodedPages)
{
  auto constexpr num_sources         = 2;
  auto const [table, parquet_buffer] = create_parquet_with_stats<uint32_t, 1>();
  auto const payload_table           = table->view().select({2});
  auto parquet_buffers               = std::vector<std::vector<char>>(num_sources, parquet_buffer);

  auto const source_info = build_source_info(parquet_buffers);
  auto const reader_options =
    cudf::io::parquet_reader_options::builder().column_names({"col2"}).build();
  auto inputs = multifile_inputs(source_info);
  auto reader = cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans,
                                                                       reader_options};
  setup_page_indexes(reader, inputs);

  auto const row_groups = reader.all_row_groups(reader_options);
  auto row_mask_values  = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, [](auto i) { return (i / page_size_for_ordered_tests) % 2 == 0; });
  auto row_mask = cudf::test::fixed_width_column_wrapper<bool>(
    row_mask_values, row_mask_values + reader.total_rows_in_row_groups(row_groups));
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto const page_ranges =
    reader.payload_pages_byte_ranges(row_groups, row_mask, reader_options, stream);
  auto page_data = fetch_multisource_device_data(inputs, page_ranges, stream, mr);
  reader.setup_chunking_for_payload_columns(
    0, 0, row_groups, row_mask, page_data.flat_spans, reader_options, stream, mr);

  ASSERT_TRUE(reader.has_next_table_chunk());
  auto const result = reader.materialize_payload_columns_chunk(row_mask);
  EXPECT_FALSE(reader.has_next_table_chunk());

  auto const input =
    cudf::concatenate(std::vector<cudf::table_view>(num_sources, payload_table), stream, mr);
  auto const expected = cudf::apply_boolean_mask(input->view(), row_mask, stream, mr);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result.tbl->view());
}

TEST_F(HybridScanMultifileTest, SparsePayloadWithAsymmetricRowGroupOrdering)
{
  using T = uint32_t;

  auto parquet_buffers = std::vector<std::vector<char>>{};
  parquet_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));
  // Name the descending column `col0` so this source retains earlier row groups.
  parquet_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>(
    100, cudf::io::compression_type::AUTO, {"col0", "col1", "col2"}, {1, 0, 2})));

  auto const source_info = build_source_info(parquet_buffers);
  auto const stream      = cudf::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();
  auto scalar            = cudf::numeric_scalar<T>(75);
  auto literal           = cudf::ast::literal(scalar);
  auto col_ref           = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);
  auto const options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  auto const row_groups =
    reader.filter_row_groups_with_stats(reader.all_row_groups(options), options, stream);
  EXPECT_EQ(row_groups, (std::vector<std::vector<cudf::size_type>>{{1, 2, 3}, {0, 1, 2}}));

  auto const expected = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(source_info).filter(filter_expression), stream, mr);
  auto const [filter_table, payload_table] =
    chunked_sparse_hybrid_scan_multifile(source_info, filter_expression, {}, true, stream, mr);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({0}), filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected.tbl->select({1, 2}), payload_table->view());
}

TEST_F(HybridScanMultifileTest, SparsePayloadEmptyAndAllPrunedPageData)
{
  using T = uint32_t;

  // Create two sources with page indexes for sparse payload materialization.
  auto file_buffers = std::vector<std::vector<char>>{};
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1, false>()));
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1, false>()));
  auto inputs = multifile_inputs(build_source_info(file_buffers));

  auto const options = cudf::io::parquet_reader_options::builder().column_names({"col1"}).build();
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();

  // Empty row-group selection accepts an empty outer page-data span.
  {
    auto reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
    auto const row_groups =
      std::vector<std::vector<cudf::size_type>>(inputs.footer_byte_spans.size());
    auto const empty_page_data = std::vector<cudf::device_span<uint8_t const>>{};
    auto false_scalar          = cudf::numeric_scalar<bool>{false};
    auto row_mask              = cudf::make_column_from_scalar(false_scalar, 0);

    EXPECT_NO_THROW(reader->setup_chunking_for_payload_columns(
      0, 0, row_groups, row_mask->view(), empty_page_data, options, stream, mr));
    ASSERT_TRUE(reader->has_next_table_chunk());
    auto const result = reader->materialize_payload_columns_chunk(row_mask->view());
    EXPECT_EQ(result.tbl->num_rows(), 0);
    EXPECT_EQ(result.metadata.num_input_row_groups, 0);
    EXPECT_FALSE(reader->has_next_table_chunk());
  }

  // An empty outer span is invalid when the selected row groups contain pages.
  {
    auto reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
    setup_page_indexes(*reader, inputs);

    auto const row_groups      = reader->all_row_groups(options);
    auto const row_mask        = reader->build_all_true_row_mask(row_groups, stream, mr);
    auto const empty_page_data = std::vector<cudf::device_span<uint8_t const>>{};

    EXPECT_THROW(reader->setup_chunking_for_payload_columns(
                   0, 0, row_groups, row_mask->view(), empty_page_data, options, stream, mr),
                 cudf::logic_error);
  }

  // An all-false row mask prunes every page and yields an empty payload table.
  {
    auto reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
    setup_page_indexes(*reader, inputs);

    auto const row_groups = reader->all_row_groups(options);
    auto false_scalar     = cudf::numeric_scalar<bool>{false};
    auto row_mask =
      cudf::make_column_from_scalar(false_scalar, reader->total_rows_in_row_groups(row_groups));
    auto const page_ranges =
      reader->payload_pages_byte_ranges(row_groups, row_mask->view(), options, stream);

    ASSERT_FALSE(page_ranges.first.empty());
    EXPECT_TRUE(std::all_of(page_ranges.first.begin(),
                            page_ranges.first.end(),
                            [](auto const& range) { return range.is_empty(); }));
    auto const all_pruned_page_data =
      std::vector<cudf::device_span<uint8_t const>>(page_ranges.first.size());
    reader->setup_chunking_for_payload_columns(
      0, 0, row_groups, row_mask->view(), all_pruned_page_data, options, stream, mr);
    ASSERT_TRUE(reader->has_next_table_chunk());
    auto const result = reader->materialize_payload_columns_chunk(row_mask->view());
    EXPECT_EQ(result.tbl->num_rows(), 0);
    EXPECT_EQ(result.tbl->num_columns(), 1);
    // Two sources with four row groups each
    EXPECT_EQ(result.metadata.num_input_row_groups, 8);
    EXPECT_FALSE(reader->has_next_table_chunk());
  }
}
