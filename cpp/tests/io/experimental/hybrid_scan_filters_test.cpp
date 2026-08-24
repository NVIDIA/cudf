/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"
#include "tests/io/parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>

#include <src/io/parquet/parquet_gpu.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

// Base test fixture for tests
struct HybridScanFiltersTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanFiltersTest, Metadata)
{
  srand(0xf00d);
  using T = uint32_t;

  // Create a table with several row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto file_buffer                  = std::get<1>(create_parquet_with_stats<T, num_concat>());

  // Filtering AST - table[0] < 100
  auto literal_value     = cudf::numeric_scalar<T>(100, true, cudf::get_default_stream());
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("coL0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Create reader options with empty source info
  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder()
                                               .filter(filter_expression)
                                               .case_sensitive_names(false);

  // Input file buffer span
  auto const datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));
  auto datasource_ref   = std::ref(*datasource);

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource_ref);

  // Create hybrid scan reader with footer bytes
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(footer_buffer->data()),
                                   footer_buffer->size()},
    options);

  // Get Parquet file metadata from the reader
  auto parquet_metadata = reader->parquet_metadata();

  // Check that the offset and column indices are not present
  EXPECT_FALSE(parquet_metadata.row_groups[0].columns[0].offset_index.has_value());
  EXPECT_FALSE(parquet_metadata.row_groups[0].columns[0].column_index.has_value());

  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(datasource_ref, page_index_byte_range);

  // Setup page index
  reader->setup_page_index(cudf::host_span<uint8_t const>{
    static_cast<uint8_t const*>(page_index_buffer->data()), page_index_buffer->size()});

  // Get Parquet file metadata from the reader again
  parquet_metadata = reader->parquet_metadata();

  // Check that the offset and column indices are now present
  EXPECT_TRUE(parquet_metadata.row_groups[0].columns[0].offset_index.has_value());
  EXPECT_TRUE(parquet_metadata.row_groups[0].columns[0].column_index.has_value());

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);
  // Expect 4 = 20000 rows / 5000 rows per row group
  EXPECT_EQ(input_row_group_indices.size(), 4);

  // Explicitly set the row groups to read
  options.set_row_groups({{0, 1}});

  // Get all row groups from the reader again
  input_row_group_indices = reader->all_row_groups(options);
  // Expect only 2 row groups now
  EXPECT_EQ(input_row_group_indices.size(), 2);
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices), 2 * rows_per_row_group);
}

TEST_F(HybridScanFiltersTest, MultiSourceMetadata)
{
  srand(0xdede);
  using T = uint32_t;

  // Helper to test multi-source metadata
  auto const test_multisource_metadata = [&](auto num_sources) {
    auto const file_buffer = std::get<1>(create_parquet_with_stats<T, 1>());

    std::vector<std::unique_ptr<cudf::io::datasource>> datasources(num_sources);
    std::vector<std::reference_wrapper<cudf::io::datasource>> datasource_refs{};
    std::transform(datasources.begin(),
                   datasources.end(),
                   std::back_inserter(datasource_refs),
                   [&](auto& datasource) {
                     datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
                       reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));
                     return std::ref(*datasource);
                   });

    // Fetch all footers at once
    auto const footer_buffers =
      cudf::io::parquet::fetch_footers_to_host({datasource_refs.data(), datasource_refs.size()});
    ASSERT_EQ(footer_buffers.size(), num_sources);

    // Fetch all page indexes at once
    auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(footer_buffers.front()->data()),
                                     footer_buffers.front()->size()},
      cudf::io::parquet_reader_options::builder().build());
    std::vector<cudf::io::parquet::byte_range_info> page_index_byte_ranges(
      num_sources, reader->page_index_byte_range());
    auto const page_index_buffers = cudf::io::parquet::fetch_page_indexes_to_host(
      {datasource_refs.data(), datasource_refs.size()},
      {page_index_byte_ranges.data(), page_index_byte_ranges.size()});
    ASSERT_EQ(page_index_buffers.size(), num_sources);

    // Footer and page index from multi-source and single-source APIs should match
    auto const single_footer     = cudf::io::parquet::fetch_footer_to_host(datasource_refs.front());
    auto const single_page_index = cudf::io::parquet::fetch_page_index_to_host(
      datasource_refs.front(), reader->page_index_byte_range());

    auto const iter = cuda::make_zip_iterator(footer_buffers.begin(), page_index_buffers.begin());
    std::for_each(iter, iter + num_sources, [&](auto const& pair) {
      auto const& [footer_buffer, page_index_buffer] = pair;
      ASSERT_EQ(footer_buffer->size(), single_footer->size());
      EXPECT_EQ(std::memcmp(footer_buffer->data(), single_footer->data(), single_footer->size()),
                0);
      ASSERT_EQ(page_index_buffer->size(), single_page_index->size());
      EXPECT_EQ(std::memcmp(
                  page_index_buffer->data(), single_page_index->data(), single_page_index->size()),
                0);
    });
  };

  auto num_sources = 4;
  test_multisource_metadata(num_sources);
  num_sources = 32;
  test_multisource_metadata(num_sources);
}

TEST_F(HybridScanFiltersTest, ExternalMetadata)
{
  srand(0xcaffe);

  auto parquet_metadata = [&]() {
    // Create a table with several row groups each with a single page.
    auto constexpr num_concat = 1;
    auto file_buffer = std::get<1>(create_parquet_with_stats<cudf::timestamp_ms, num_concat>());
    // Input file buffer span
    auto const datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));
    auto datasource_ref   = std::ref(*datasource);

    // Fetch footer and page index bytes from the buffer.
    auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource_ref);

    auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(footer_buffer->data()),
                                     footer_buffer->size()},
      cudf::io::parquet_reader_options::builder().build());

    // Get page index byte range from the reader
    auto const page_index_byte_range = reader->page_index_byte_range();

    // Fetch page index bytes from the input buffer
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(datasource_ref, page_index_byte_range);

    // Setup page index
    reader->setup_page_index(cudf::host_span<uint8_t const>{
      static_cast<uint8_t const*>(page_index_buffer->data()), page_index_buffer->size()});

    return reader->parquet_metadata();
  }();

  // Filtering AST - 100 > table[0]
  using T = cudf::timestamp_ms;
  auto literal_value =
    cudf::timestamp_scalar<T>(T(typename T::duration(100)), true, cudf::get_default_stream());
  auto literal   = cudf::ast::literal(literal_value);
  auto col_ref_0 = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal, col_ref_0);

  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Get Parquet file metadata from the reader
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    parquet_metadata, options);

  // Get Parquet file metadata from the reader
  parquet_metadata = reader->parquet_metadata();

  // Check that the offset and column indices are present
  EXPECT_TRUE(parquet_metadata.row_groups[0].columns[0].offset_index.has_value());
  EXPECT_TRUE(parquet_metadata.row_groups[0].columns[0].column_index.has_value());

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);
  // Expect 4 = 20000 rows / 5000 rows per row group
  EXPECT_EQ(input_row_group_indices.size(), 4);

  // Explicitly set the row groups to read
  options.set_row_groups({{2, 3}});

  // Get all row groups from the reader again
  input_row_group_indices = reader->all_row_groups(options);
  // Expect only 2 row groups now
  EXPECT_EQ(input_row_group_indices.size(), 2);

  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices), 2 * rows_per_row_group);
}

TEST_F(HybridScanFiltersTest, FilterRowGroupsWithByteRanges)
{
  using T                      = cudf::string_view;
  auto const [table, filepath] = create_parquet_typed_with_stats<T>("ByteBounds.parquet");

  auto const file_size = std::filesystem::file_size(filepath);
  std::vector<char> file_buffer(file_size);
  std::ifstream file{filepath, std::ifstream::binary};
  file.read(file_buffer.data(), file_size);
  file.close();

  // Input file buffer span
  auto const datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

  // Create hybrid scan reader with footer bytes
  auto options      = cudf::io::parquet_reader_options::builder().build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(footer_buffer->data()),
                                   footer_buffer->size()},
    options);

  auto const input_row_group_indices = reader->all_row_groups(options);

  // @note: In the above parquet file, the row groups start at the following byte offsets: 4, 75224,
  // 150332, 225561. The `skip_bytes` and `num_bytes` have been chosen to have enough cushion but
  // may need to be adjusted in the future if this test suddenly starts failing.

  {
    // Start with all row groups and only read row group 0 as only it will start in [0, 1000) byte
    // range
    auto constexpr num_bytes = 1000;
    options.set_num_bytes(num_bytes);
    auto const filtered_row_group_indices =
      reader->filter_row_groups_with_byte_range(input_row_group_indices, options);
    auto const expected_row_group_indices = std::vector<cudf::size_type>{0};
    EXPECT_EQ(filtered_row_group_indices, expected_row_group_indices);
  }

  {
    // Start with all row groups and skip row group 0 as it won't start in [1000, inf) byte range
    auto skip_bytes = 1000;
    options.set_skip_bytes(skip_bytes);
    options.set_num_bytes(std::numeric_limits<size_t>::max());
    auto filtered_row_group_indices =
      reader->filter_row_groups_with_byte_range(input_row_group_indices, options);
    auto expected_row_group_indices = std::vector<cudf::size_type>{1, 2, 3};
    EXPECT_EQ(filtered_row_group_indices, expected_row_group_indices);

    // Now start with filtered row groups and only read row group 1 as only it starts in [50000,
    // 100000) byte range
    skip_bytes               = 50000;
    auto constexpr num_bytes = 50000;
    options.set_skip_bytes(skip_bytes);
    options.set_num_bytes(num_bytes);
    filtered_row_group_indices =
      reader->filter_row_groups_with_byte_range(filtered_row_group_indices, options);
    expected_row_group_indices = std::vector<cudf::size_type>{1};
    EXPECT_EQ(filtered_row_group_indices, expected_row_group_indices);
  }

  {
    // Start with all row groups and skip all row groups as [500000, inf) byte range is beyond the
    // file size
    auto constexpr skip_bytes = 500'000;
    options.set_skip_bytes(skip_bytes);
    auto const filtered_row_group_indices =
      reader->filter_row_groups_with_byte_range(input_row_group_indices, options);
    auto const expected_row_group_indices = std::vector<cudf::size_type>{};
    EXPECT_EQ(filtered_row_group_indices, expected_row_group_indices);
  }
}

TEST_F(HybridScanFiltersTest, FilterRowGroupsWithStats)
{
  srand(0xc001);
  using T = uint32_t;

  // Create a table with 4 row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto [written_table, file_buffer] = create_parquet_with_stats<T, num_concat, false>();

  // Filtering AST - table[0] < 50 and table[2] < "000010000"
  auto literal_value1     = cudf::numeric_scalar<T>(50, true, cudf::get_default_stream());
  auto literal1           = cudf::ast::literal(literal_value1);
  auto col_ref0           = cudf::ast::column_reference(0);
  auto filter_expression1 = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal1);

  auto literal_value2 = cudf::string_scalar("000010000", true, cudf::get_default_stream());
  auto literal2       = cudf::ast::literal(literal_value2);
  auto col_ref2       = cudf::ast::column_reference(2);
  auto filter_expression2 =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal2, col_ref2);

  auto filter_expression = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Input datasource
  auto const datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

  // Create hybrid scan reader with footer bytes
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(footer_buffer->data()),
                                   footer_buffer->size()},
    options);

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);
  // Expect 4 = 20000 rows / 5000 rows per row group
  EXPECT_EQ(input_row_group_indices.size(), 4);
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices), 4 * rows_per_row_group);

  auto stats_filtered_row_groups = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  // Expect 3 row groups to be filtered out with stats
  EXPECT_EQ(stats_filtered_row_groups.size(), 1);
  EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered_row_groups), rows_per_row_group);

  // Use custom input row group indices
  input_row_group_indices   = {1, 2};
  stats_filtered_row_groups = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  // Expect all row groups to be filtered out with stats
  EXPECT_EQ(stats_filtered_row_groups.size(), 0);
  EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered_row_groups), 0);
}

TEST_F(HybridScanFiltersTest, FilterRowGroupsWithComplexExpressions)
{
  srand(0xc002);
  using T = uint32_t;

  // Create a table with 4 row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto [written_table, file_buffer] = create_parquet_with_stats<T, num_concat, false>();

  auto col_ref0 = cudf::ast::column_reference(0);
  auto col_ref1 = cudf::ast::column_reference(1);

  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();

  // Input datasource
  auto const datasource    = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  // Filter: col0 < col1 (col op col, no literal)
  // Stats filter will pass this through keeping all 4 row groups
  {
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, col_ref1);
    options.set_filter(filter);

    auto input_row_group_indices = reader->all_row_groups(options);
    EXPECT_EQ(input_row_group_indices.size(), 4);

    auto stats_filtered = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    EXPECT_EQ(stats_filtered.size(), 4);
  }

  // Filter: (col0 < 50) and (col0 < col1)
  // Stats filter will prune based on col0 < 50 but pass through col0 < col1
  {
    auto literal_value = cudf::numeric_scalar<T>(50, true, cudf::get_default_stream());
    auto literal       = cudf::ast::literal(literal_value);
    auto lhs           = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal);
    auto rhs           = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, col_ref1);
    auto filter        = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, lhs, rhs);
    options.set_filter(filter);
    reader->reset_column_selection();

    auto input_row_group_indices = reader->all_row_groups(options);
    auto stats_filtered          = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    // col0 < 50 should prune row groups where min(col0) >= 50
    EXPECT_EQ(stats_filtered.size(), 1);
    EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered), rows_per_row_group);
  }

  // Filter: (col0 < 50) or (col0 < col1)
  // col0 < col1 will be passed through by stats, so the LOGICAL_OR will keep all row groups
  {
    auto literal_value = cudf::numeric_scalar<T>(50, true, cudf::get_default_stream());
    auto literal       = cudf::ast::literal(literal_value);
    auto lhs           = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal);
    auto rhs           = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, col_ref1);
    auto filter        = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, lhs, rhs);
    options.set_filter(filter);
    reader->reset_column_selection();

    auto input_row_group_indices = reader->all_row_groups(options);
    auto stats_filtered          = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    EXPECT_EQ(stats_filtered.size(), 4);
  }

  // Filter: NOT(col0 < 50)
  // Negated to col0 >= 50, stats transform: vmax >= 50. Prunes RG0 (vmax=49).
  {
    auto literal_value = cudf::numeric_scalar<T>(50, true, cudf::get_default_stream());
    auto literal       = cudf::ast::literal(literal_value);
    auto inner         = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal);
    auto filter        = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);
    options.set_filter(filter);
    reader->reset_column_selection();

    auto input_row_group_indices = reader->all_row_groups(options);
    auto stats_filtered          = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    EXPECT_EQ(stats_filtered.size(), 3);
    EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered), 3 * rows_per_row_group);
  }

  // Filter: NOT(col0 > 50)
  // Negated to col0 <= 50, stats transform: vmin <= 50. Prunes RG2 and RG3.
  {
    auto literal_value = cudf::numeric_scalar<T>(50, true, cudf::get_default_stream());
    auto literal       = cudf::ast::literal(literal_value);
    auto inner         = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref0, literal);
    auto filter        = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);
    options.set_filter(filter);
    reader->reset_column_selection();

    auto input_row_group_indices = reader->all_row_groups(options);
    auto stats_filtered          = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    EXPECT_EQ(stats_filtered.size(), 2);
    EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered), 2 * rows_per_row_group);
  }

  // Filter: NOT(col0 != 50 AND col0 != 100) becomes col0 == 50 OR col0 == 100. Prunes RG0 and RG3.
  {
    auto literal_50_value  = cudf::numeric_scalar<T>(50, true, cudf::get_default_stream());
    auto literal_50        = cudf::ast::literal(literal_50_value);
    auto literal_100_value = cudf::numeric_scalar<T>(100, true, cudf::get_default_stream());
    auto literal_100       = cudf::ast::literal(literal_100_value);
    auto ne_50  = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_ref0, literal_50);
    auto ne_100 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_ref0, literal_100);
    auto inner  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ne_50, ne_100);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);
    options.set_filter(filter);
    reader->reset_column_selection();

    auto input_row_group_indices = reader->all_row_groups(options);
    auto stats_filtered          = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    EXPECT_EQ(stats_filtered.size(), 2);
  }

  // Filter: NOT(NOT(col0 < 100) OR col0 > 150)
  // De Morgan plus double-negation returns col0 < 100 AND NOT(col0 > 150), stats transform:
  // vmin < 100 AND vmin <= 150. Prunes RG2 (vmin=100) and RG3 (vmin=150).
  {
    auto literal_100_value = cudf::numeric_scalar<T>(100, true, cudf::get_default_stream());
    auto literal_100       = cudf::ast::literal(literal_100_value);
    auto literal_150_value = cudf::numeric_scalar<T>(150, true, cudf::get_default_stream());
    auto literal_150       = cudf::ast::literal(literal_150_value);
    auto lt_100 = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal_100);
    auto not_lt = cudf::ast::operation(cudf::ast::ast_operator::NOT, lt_100);
    auto gt_150 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref0, literal_150);
    auto inner  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, not_lt, gt_150);
    auto filter = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);
    options.set_filter(filter);
    reader->reset_column_selection();

    auto input_row_group_indices = reader->all_row_groups(options);
    auto stats_filtered          = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    EXPECT_EQ(stats_filtered.size(), 2);
    EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered), 2 * rows_per_row_group);
  }
}

TEST_F(HybridScanFiltersTest, FilterColumnSelection)
{
  srand(0xc0al);
  using T = uint32_t;

  // Create a table with 4 row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto [written_table, file_buffer] = create_parquet_with_stats<T, num_concat, false>();

  // Create datasource
  auto const datasource    = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

  auto const options = cudf::io::parquet_reader_options::builder().build();
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  // Get input row group indices
  auto const input_row_group_indices = [&]() { return reader->all_row_groups(options); }();
  EXPECT_EQ(input_row_group_indices.size(), 4);

  // Helper to test filter column selection
  auto const test_filter_column_selection = [&](cudf::io::parquet_reader_options const& options) {
    reader->reset_column_selection();
    auto stats_filtered_row_groups = reader->filter_row_groups_with_stats(
      input_row_group_indices, options, cudf::get_default_stream());
    // Expect 1 remaining row group after filtering
    EXPECT_EQ(stats_filtered_row_groups.size(), 1);
  };

  auto literal_value1 = cudf::numeric_scalar<T>(50, true, cudf::get_default_stream());
  auto literal1       = cudf::ast::literal(literal_value1);
  auto col_name0      = cudf::ast::column_name_reference("col0");
  auto col_ref0       = cudf::ast::column_reference(0);

  auto literal_value2 = cudf::string_scalar("000010000", true, cudf::get_default_stream());
  auto literal2       = cudf::ast::literal(literal_value2);
  auto col_name2      = cudf::ast::column_name_reference("col2");
  auto col_ref2       = cudf::ast::column_reference(2);

  // Test columns selection by names and filter expression. Column selection is
  // irrelevant here as we can collect column names from the filter expression itself
  {
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name0, literal1);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name2, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    options.set_column_names({"col0", "col1", "col2"});
    test_filter_column_selection(options);
    options.set_column_names({"col1"});
    test_filter_column_selection(options);
    options.set_column_names({});
    test_filter_column_selection(options);

    options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    options.set_column_indices({0, 1, 2});
    test_filter_column_selection(options);
    options.set_column_indices({0, 1});
    test_filter_column_selection(options);
    options.set_column_indices({});
    test_filter_column_selection(options);
  }

  // Test column selection by name and index and filter expression. Since `col2` is referred by
  // index, it must be present in column selection (or no column selection should be specified)
  {
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name0, literal1);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal2, col_ref2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    test_filter_column_selection(options);
    options.set_column_names({"col0", "col1", "col2"});
    test_filter_column_selection(options);
    options.set_column_names({"col1"});
    EXPECT_ANY_THROW(test_filter_column_selection(options));
    options.set_column_names({});
    EXPECT_ANY_THROW(test_filter_column_selection(options));

    options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    options.set_column_indices({0, 1, 2});
    test_filter_column_selection(options);
    options.set_column_indices({2});
    EXPECT_ANY_THROW(test_filter_column_selection(options));
    options.set_column_indices({});
    EXPECT_ANY_THROW(test_filter_column_selection(options));

    // `col2` is actually in our column selection at index 0, so we can select it using the index in
    // selection
    {
      auto updated_col_ref2 = cudf::ast::column_reference(0);
      filter_expression2 =
        cudf::ast::operation(cudf::ast::ast_operator::LESS, updated_col_ref2, literal2);
      options.set_column_indices({2});
      test_filter_column_selection(options);
    }
  }

  // Test columns selection by index and filter expression. Since both columns are referred by
  // index, they must be present in the column selection at respective indices (or the filter
  // expression must be modified)
  {
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal1);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    test_filter_column_selection(options);
    options.set_column_names({"col0", "col1", "col2"});
    test_filter_column_selection(options);
    options.set_column_names({"col0", "col1"});
    EXPECT_ANY_THROW(test_filter_column_selection(options));
    options.set_column_names({"col1"});
    EXPECT_ANY_THROW(test_filter_column_selection(options));
    options.set_column_names({});
    EXPECT_ANY_THROW(test_filter_column_selection(options));

    options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    options.set_column_indices({0, 1, 2});
    test_filter_column_selection(options);
    options.set_column_indices({2});
    EXPECT_ANY_THROW(test_filter_column_selection(options));
    options.set_column_indices({});
    EXPECT_ANY_THROW(test_filter_column_selection(options));
  }

  // Both columns are in the selection, so we can select them using the correct indices in the
  // selection
  {
    auto col_ref1 = cudf::ast::column_reference(2);
    auto col_ref2 = cudf::ast::column_reference(1);
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref1, literal1);
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);

    auto options = cudf::io::parquet_reader_options::builder()
                     .filter(filter_expression)
                     .column_indices({1, 2, 0})
                     .build();
    test_filter_column_selection(options);
    options = cudf::io::parquet_reader_options::builder()
                .filter(filter_expression)
                .column_names({"col1", "col2", "col0"})
                .build();
    test_filter_column_selection(options);
  }
}

template <typename T>
struct PageFilteringWithPageIndexStats : public HybridScanFiltersTest {};

// Unsigned numeric types except booleans for columns 0 and 1 for page index stats tests
using SignedIntegralTypesNotBool =
  cudf::test::ContainedIn<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>>;
using PageFilteringTestTypes =
  cudf::test::RemoveIf<SignedIntegralTypesNotBool,
                       cudf::test::Concat<cudf::test::IntegralTypesNotBool>>;

TYPED_TEST_SUITE(PageFilteringWithPageIndexStats, PageFilteringTestTypes);

TYPED_TEST(PageFilteringWithPageIndexStats, FilterPages)
{
  using T = TypeParam;

  srand(31337);

  // A table concatenated multiple times by itself with result in a parquet file with a row group
  // per concatenation with multiple pages per row group. Since all row groups will be identical, we
  // can only prune pages based on page index stats
  auto constexpr num_concat = 2;
  auto const file_buffer    = std::get<1>(create_parquet_with_stats<T, num_concat, false>());

  // Input datasource
  auto const datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

  // Create hybrid scan reader with footer bytes
  auto options = cudf::io::parquet_reader_options::builder().build();
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Helper function to test data page filteration using page index stats
  auto const test_filter_data_pages_with_stats = [&](
                                                   cudf::ast::operation const& filter_expression,
                                                   cudf::size_type const expected_surviving_rows) {
    // Set the filter expression and reset column selection
    options.set_filter(filter_expression);
    reader->reset_column_selection();

    // Filter the data pages with page index stats
    auto const row_mask =
      reader->build_row_mask_with_page_index_stats(input_row_group_indices, options, stream, mr);

    auto const expected_num_rows = reader->total_rows_in_row_groups(input_row_group_indices);
    EXPECT_EQ(row_mask->type().id(), cudf::type_id::BOOL8);
    EXPECT_EQ(row_mask->size(), expected_num_rows);
    EXPECT_EQ(row_mask->null_count(), 0);

    // Copy the row mask to the host and count the number of surviving rows
    auto const host_row_mask = cudf::detail::make_host_vector<bool>(
      cudf::device_span<bool const>(row_mask->view().data<bool>(),
                                    static_cast<size_t>(row_mask->view().size())),
      stream);
    EXPECT_EQ(std::count(host_row_mask.begin(), host_row_mask.end(), true),
              expected_surviving_rows);
  };

  // Calling `test_filter_data_pages_with_stats` before setting up the page index should raise an
  // error
  {
    auto literal_value     = cudf::numeric_scalar<T>(T{100}, true, stream);
    auto const literal     = cudf::ast::literal(literal_value);
    auto const col_ref     = cudf::ast::column_name_reference("col0");
    auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);
    EXPECT_THROW(test_filter_data_pages_with_stats(filter_expression, 0), std::runtime_error);
  }

  // Set up the page index
  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_byte_range);
  reader->setup_page_index(cudf::host_span<uint8_t const>{
    static_cast<uint8_t const*>(page_index_buffer->data()), page_index_buffer->size()});

  // Filtering AST - table[0] < 100
  {
    auto literal_value = cudf::numeric_scalar<T>(T{100}, true, stream);
    auto const literal = cudf::ast::literal(literal_value);
    auto const col_ref = cudf::ast::column_name_reference("col0");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal, col_ref);
    // Half the pages (unsigned) or 3/4th the pages (signed) should be filtered out by the page
    // index filter
    auto constexpr expected_surviving_rows =
      (num_concat * num_ordered_rows) / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[2] >= 10000
  {
    auto literal_value = cudf::string_scalar("000010000", true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref       = cudf::ast::column_name_reference("col2");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);
    // Half the pages (unsigned) or 3/4th the pages (signed) should be filtered out by the page
    // index filter
    auto constexpr expected_surviving_rows =
      (num_concat * num_ordered_rows) / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] < 50 AND table[2] < "000010000"
  {
    auto literal_value1 = cudf::numeric_scalar<T>(T{50}, true, stream);
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000010000", true, stream);
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);
    // Only one page per num_concat per filter column should survive
    auto constexpr expected_surviving_rows = num_concat * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] > 150 OR table[2] < "000005000"
  {
    auto literal_value1 = cudf::numeric_scalar<T>(T{150}, true, stream);
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000005000", true, stream);
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);
    // Two pages (3rd and 0th from respective conditions) per num_concat per filter column should
    // survive
    auto constexpr expected_surviving_rows = 2 * num_concat * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }
}

TEST_F(HybridScanFiltersTest, OffsetIndexOnlyDataPageMask)
{
  using T                           = uint32_t;
  auto constexpr num_concat         = 2;
  auto [written_table, file_buffer] = create_parquet_with_stats<T, num_concat, false>();

  auto const datasource    = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto options             = cudf::io::parquet_reader_options::builder().build();
  auto reader = cudf::io::parquet::experimental::hybrid_scan_reader(*footer_buffer, options);

  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(*datasource, reader.page_index_byte_range());
  reader.setup_page_index(*page_index_buffer);

  auto metadata = reader.parquet_metadata();
  for (auto& row_group : metadata.row_groups) {
    for (auto& column : row_group.columns) {
      column.column_index.reset();
    }
  }

  auto offset_only_reader = cudf::io::parquet::experimental::hybrid_scan_reader(metadata, options);
  auto const selected_row_groups = offset_only_reader.all_row_groups(options);
  auto const total_rows          = offset_only_reader.total_rows_in_row_groups(selected_row_groups);

  auto row_mask_values = cudf::detail::make_counting_transform_iterator(
    0, [total_rows](auto const row) { return std::cmp_greater_equal(row, total_rows / 2); });
  auto row_mask =
    cudf::test::fixed_width_column_wrapper<bool>(row_mask_values, row_mask_values + total_rows);
  auto const row_mask_view = static_cast<cudf::column_view>(row_mask);
  auto const stream        = cudf::get_default_stream();
  auto const mr            = cudf::get_current_device_resource_ref();
  auto const byte_ranges =
    offset_only_reader.payload_column_chunks_byte_ranges(selected_row_groups, options);
  auto [column_buffers, column_data, read_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(*datasource, byte_ranges, stream, mr);
  read_tasks.get();

  // Materialization maps the row mask to pages using only offset index, then applies the row mask.
  auto const result = offset_only_reader.materialize_payload_columns(
    selected_row_groups,
    column_data,
    row_mask_view,
    cudf::io::parquet::experimental::use_data_page_mask::YES,
    options,
    stream,
    mr);
  auto const expected = cudf::apply_boolean_mask(written_table->view(), row_mask_view, stream, mr);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result.tbl->view());

  // Without offset index, data-page pruning falls back to decoding all pages.
  for (auto& row_group : metadata.row_groups) {
    for (auto& column : row_group.columns) {
      column.offset_index.reset();
    }
  }
  auto no_index_reader = cudf::io::parquet::experimental::hybrid_scan_reader(metadata, options);
  auto const no_index_row_groups = no_index_reader.all_row_groups(options);
  auto const no_index_ranges =
    no_index_reader.payload_column_chunks_byte_ranges(no_index_row_groups, options);
  auto [no_index_buffers, no_index_data, no_index_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(*datasource, no_index_ranges, stream, mr);
  no_index_tasks.get();
  auto const no_index_result = no_index_reader.materialize_payload_columns(
    no_index_row_groups,
    no_index_data,
    row_mask_view,
    cudf::io::parquet::experimental::use_data_page_mask::YES,
    options,
    stream,
    mr);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), no_index_result.tbl->view());
}

template <typename T>
struct TimestampPageFiltering : public HybridScanFiltersTest {};

using MismatchedTimestampTypes = cudf::test::Types<cudf::timestamp_us, cudf::timestamp_ns>;
TYPED_TEST_SUITE(TimestampPageFiltering, MismatchedTimestampTypes);

TYPED_TEST(TimestampPageFiltering, MismatchedPrecisions)
{
  using NativeTimestamp = cudf::timestamp_ms;
  using NativeDuration  = typename NativeTimestamp::duration;
  using NativeRep       = typename NativeDuration::rep;

  using TargetTimestamp = TypeParam;
  using TargetDuration  = typename TargetTimestamp::duration;

  srand(31337);

  // Concatenate the table twice so we get multiple row groups with identical data. This forces
  // pruning to happen at the page level rather than at the row-group level.
  auto constexpr num_concat = 2;
  auto const file_buffer    = std::get<1>(create_parquet_with_stats<NativeTimestamp, num_concat>());

  auto const datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(file_buffer.data()), file_buffer.size()));

  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);

  auto options = cudf::io::parquet_reader_options::builder()
                   .timestamp_type(cudf::data_type{cudf::type_to_id<TargetTimestamp>()})
                   .build();

  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Set up the page index before running any page-level filtering.
  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_byte_range);
  reader->setup_page_index(cudf::host_span<uint8_t const>{
    static_cast<uint8_t const*>(page_index_buffer->data()), page_index_buffer->size()});

  auto const test_filter_data_pages_with_stats = [&](
                                                   cudf::ast::operation const& filter_expression,
                                                   cudf::size_type const expected_surviving_rows) {
    auto const input_row_group_indices = reader->all_row_groups(options);

    auto const row_mask =
      reader->build_row_mask_with_page_index_stats(input_row_group_indices, options, stream, mr);

    auto const host_row_mask = cudf::detail::make_host_vector<bool>(
      cudf::device_span<bool const>(row_mask->view().data<bool>(),
                                    static_cast<size_t>(row_mask->view().size())),
      stream);
    EXPECT_EQ(std::count(host_row_mask.begin(), host_row_mask.end(), true),
              expected_surviving_rows);
  };

  auto constexpr page_size        = page_size_for_ordered_tests;
  auto constexpr page_boundary_ms = NativeRep{2 * page_size};
  auto const threshold_output =
    cuda::std::chrono::duration_cast<TargetDuration>(NativeDuration{page_boundary_ms});

  // Only the first two pages per row group should survive (values < 10000 ms).
  auto constexpr expected_surviving_rows = num_concat * 2 * page_size;

  auto literal_value =
    cudf::timestamp_scalar<TargetTimestamp>(TargetTimestamp{threshold_output}, true, stream);
  auto const literal     = cudf::ast::literal(literal_value);
  auto const col_ref     = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);
  options.set_filter(filter_expression);
  test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
}

TEST_F(HybridScanFiltersTest, FilterRowGroupsWithDictionary)
{
  srand(0xcafe);
  using T = uint32_t;

  // A table with several row groups each containing a single page per column. The data page and row
  // group stats are identical so only row groups can be pruned using stats
  auto constexpr num_concat = 1;
  auto const buffer         = std::get<1>(create_parquet_with_stats<T, num_concat>());
  auto stream               = cudf::get_default_stream();
  auto mr                   = cudf::get_current_device_resource_ref();

  // Input datasource
  auto const datasource     = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()));
  auto const datasource_ref = std::ref(*datasource);

  // Hybrid scan reader
  auto options             = cudf::io::parquet_reader_options::builder().build();
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);
  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_byte_range);
  reader->setup_page_index(*page_index_buffer);

  auto const reader_ref = std::ref(*reader);

  auto col0_ref = cudf::ast::column_name_reference("col0");
  auto col2_ref = cudf::ast::column_name_reference("col2");

  {
    // Filtering - table[0] != 1000
    auto uint_literal_value = cudf::numeric_scalar<T>(1000, true, stream);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal);
    constexpr size_t expected_row_groups = 4;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] == 1000
    auto uint_literal_value = cudf::numeric_scalar<T>(1000, true, stream);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, uint_literal, col0_ref);
    constexpr size_t expected_row_groups = 0;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] != 0100
    auto str_literal_value = cudf::string_scalar("0100", true, stream);  // in all row groups
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal);

    constexpr size_t expected_row_groups = 0;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] == 0100
    auto str_literal_value = cudf::string_scalar("0100", true, stream);  // in all row groups
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col2_ref, str_literal);

    constexpr size_t expected_row_groups = 4;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 AND table[2] == 0100
    auto uint_literal_value = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal);

    auto str_literal_value = cudf::string_scalar("0100", true, stream);
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col2_ref, str_literal);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, str_filter_expression);

    constexpr size_t expected_row_groups = 4;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering -  table[0] != 50 and table[0] != 100
    auto uint_literal_value  = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(100, true, stream);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, uint_filter_expression2);

    constexpr size_t expected_row_groups = 4;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 and table[0] == 50
    auto uint_literal_value  = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, uint_literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, uint_filter_expression2);

    constexpr size_t expected_row_groups = 1;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] != 0100 or table[2] != 0101
    auto str_literal_value  = cudf::string_scalar("0100", true, stream);  // in all row groups
    auto str_literal_value2 = cudf::string_scalar("0101", true, stream);  // in no row group
    auto str_literal        = cudf::ast::literal(str_literal_value);
    auto str_literal2       = cudf::ast::literal(str_literal_value2);
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal);
    auto str_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, str_filter_expression, str_filter_expression2);

    constexpr size_t expected_row_groups = 4;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 or table[2] != 0100
    auto uint_literal_value = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal);

    auto str_literal_value = cudf::string_scalar("0100", true, stream);
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, uint_filter_expression, str_filter_expression);

    constexpr size_t expected_row_groups = 4;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 and table[2] != 0100
    auto uint_literal_value = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal);

    auto str_literal_value = cudf::string_scalar("0100", true, stream);
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, str_filter_expression);

    constexpr size_t expected_row_groups = 0;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] == 50 or table[0] == 100 or table[0] == 150
    auto uint_literal_value  = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(100, true, stream);
    auto uint_literal_value3 = cudf::numeric_scalar<T>(150, true, stream);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_literal3       = cudf::ast::literal(uint_literal_value3);
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, uint_literal2);
    auto uint_filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, uint_literal3);
    auto composed_filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, uint_filter_expression, uint_filter_expression2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, composed_filter_expression, uint_filter_expression3);

    constexpr size_t expected_row_groups = 3;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 or table[0] != 100 or table[0] != 150
    auto uint_literal_value  = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(100, true, stream);
    auto uint_literal_value3 = cudf::numeric_scalar<T>(150, true, stream);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_literal3       = cudf::ast::literal(uint_literal_value3);
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal2);
    auto uint_filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, uint_literal3);
    auto composed_filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, uint_filter_expression, uint_filter_expression2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, composed_filter_expression, uint_filter_expression3);

    constexpr size_t expected_row_groups = 4;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] != 0100 and table[2] != 0101 and table[2] != 0150
    auto str_literal_value  = cudf::string_scalar("0100", true, stream);
    auto str_literal_value2 = cudf::string_scalar("0101", true, stream);
    auto str_literal_value3 = cudf::string_scalar("0150", true, stream);
    auto str_literal        = cudf::ast::literal(str_literal_value);
    auto str_literal2       = cudf::ast::literal(str_literal_value2);
    auto str_literal3       = cudf::ast::literal(str_literal_value3);
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal);
    auto str_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal2);
    auto str_filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal3);
    auto composed_filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, str_filter_expression, str_filter_expression2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, composed_filter_expression, str_filter_expression3);

    constexpr size_t expected_row_groups = 0;
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr).size(),
      expected_row_groups);
  }

  // Filtering - (50 == table[0]) AND (table[0] != table[2])
  {
    auto uint_literal_value = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto lhs = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, uint_literal);
    auto rhs = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, col2_ref);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, lhs, rhs);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    auto const result =
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr);
    auto const expected = std::vector<cudf::size_type>{1};
    EXPECT_EQ(result, expected);
  }

  // Filtering - NOT(table[2] == "0100")
  // Rewritten to table[2] != "0100". Every dictionary holds "0100" and nothing else, so all four
  // row groups are pruned
  {
    auto str_literal_value = cudf::string_scalar("0100", true, stream);
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto inner = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col2_ref, str_literal);
    auto const filter_expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    auto const result =
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr);
    auto const expected = std::vector<cudf::size_type>{};
    EXPECT_EQ(result, expected);

    // `NOT(col == v)` and `col != v` are the same predicate and must prune identically
    auto const not_equal =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col2_ref, str_literal);
    auto const not_equal_options =
      cudf::io::parquet_reader_options::builder().filter(not_equal).build();
    EXPECT_EQ(result,
              filter_row_groups_with_dictionaries(
                datasource_ref, reader_ref, not_equal_options, stream, mr));
  }

  // Filtering - NOT(table[0] == 50)
  // Rewritten to table[0] != 50, which prunes only when 50 is the *only* dictionary value. Row
  // group 1 holds 50..99, so nothing is pruned - negating the membership result instead would prune
  // it and drop its non-50 rows
  {
    auto uint_literal_value = cudf::numeric_scalar<T>(50, true, stream);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto inner = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, uint_literal);
    auto const filter_expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, inner);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    auto const result =
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr);
    auto const expected = std::vector<cudf::size_type>{0, 1, 2, 3};
    EXPECT_EQ(result, expected);
  }

  // Filtering - NOT(table[0] != 50) AND (table[0] NULL_EQUAL 100)
  // Rewritten to (table[0] == 50) AND NULL_EQUAL(...). NULL_EQUAL has no dictionary transform and
  // relaxes, so only the equality prunes, keeping the row group whose dictionary holds 50
  {
    auto literal_50_value  = cudf::numeric_scalar<T>(50, true, stream);
    auto literal_50        = cudf::ast::literal(literal_50_value);
    auto literal_100_value = cudf::numeric_scalar<T>(100, true, stream);
    auto literal_100       = cudf::ast::literal(literal_100_value);
    auto ne_50     = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, literal_50);
    auto not_ne_50 = cudf::ast::operation(cudf::ast::ast_operator::NOT, ne_50);
    auto null_eq_100 =
      cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col0_ref, literal_100);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, not_ne_50, null_eq_100);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    auto const result =
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr);
    auto const expected = std::vector<cudf::size_type>{1};
    EXPECT_EQ(result, expected);
  }

  // Filtering - NOT((table[0] != 50) AND (table[0] != 150))
  // De Morgan and the equality complement give (table[0] == 50) OR (table[0] == 150), keeping only
  // the row groups whose dictionaries hold 50 and 150
  {
    auto literal_50_value  = cudf::numeric_scalar<T>(50, true, stream);
    auto literal_50        = cudf::ast::literal(literal_50_value);
    auto literal_150_value = cudf::numeric_scalar<T>(150, true, stream);
    auto literal_150       = cudf::ast::literal(literal_150_value);
    auto ne_50  = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, literal_50);
    auto ne_150 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, literal_150);
    auto conjunction = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, ne_50, ne_150);
    auto const filter_expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, conjunction);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    auto const result =
      filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr);
    auto const expected = std::vector<cudf::size_type>{1, 3};
    EXPECT_EQ(result, expected);

    // The De Morgan rewrite must prune exactly like the directly spelled disjunction
    auto eq_50  = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, literal_50);
    auto eq_150 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, literal_150);
    auto const disjunction =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, eq_50, eq_150);
    auto const disjunction_options =
      cudf::io::parquet_reader_options::builder().filter(disjunction).build();
    EXPECT_EQ(result,
              filter_row_groups_with_dictionaries(
                datasource_ref, reader_ref, disjunction_options, stream, mr));
  }
}

template <typename T>
struct RowGroupFilteringWithDictTest : public HybridScanFiltersTest {};

// Booleans and fixed-point types are not supported for dictionary based filtering
using DictionaryTestTypes =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<bool>>, SupportedTestTypesJIT>;

TYPED_TEST_SUITE(RowGroupFilteringWithDictTest, DictionaryTestTypes);

TYPED_TEST(RowGroupFilteringWithDictTest, FilterFewLiteralsTyped)
{
  srand(0xace);
  using T = TypeParam;

  auto constexpr num_concat          = 1;
  auto constexpr is_constant_strings = true;
  auto constexpr is_nullable         = true;

  // Specifying ZSTD compression to explicitly test decompression of dictionary pages
  auto const buffer =
    std::get<1>(create_parquet_with_stats<T, num_concat, is_constant_strings, is_nullable>(
      100, cudf::io::compression_type::ZSTD));

  // For string tests use `col2` containing constant "0100" and for temporal types use `col1`
  // containing low cardinality descending values. For all other types use `col0`
  // containing ascending values.
  auto col_name = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_name_reference("col2");
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_name_reference("col1");
    } else {
      return cudf::ast::column_name_reference("col0");
    }
  }();

  // Same logic as above for column reference
  auto col_ref = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_reference(2);
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_reference(1);
    } else {
      return cudf::ast::column_reference(0);
    }
  }();

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Input datasource
  auto const datasource     = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()));
  auto const datasource_ref = std::ref(*datasource);

  // Hybrid scan reader
  auto options             = cudf::io::parquet_reader_options::builder().build();
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);
  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_byte_range);
  reader->setup_page_index(*page_index_buffer);

  auto const reader_ref = std::ref(*reader);

  // Filtering AST
  auto literal_value = [&]() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 100 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(100)), true, stream);  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 100 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(100), true, stream);  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0100"
      return cudf::string_scalar("0100", true, stream);  // i (0-200)
    } else {
      // table[0] == 0 or 100u
      return cudf::numeric_scalar<T>(
        (100 - 100 * std::is_signed_v<T>), true, stream);  // i/100 (-100-100/ 0-200)
    }
  }();

  // Filtering AST - col_ref == 100
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{
          0, 1, 2, 3};  // Constant string value "0100" is present in all RGs
      } else if constexpr (cudf::is_chrono<T>() or cuda::std::is_signed_v<T>) {
        return std::vector<cudf::size_type>{
          1, 2};  // Descending temporal and signed value (100) is present in RGs: 1,2
      } else {
        return std::vector<cudf::size_type>{2};  // Ascending value (100) is present in RG: 1
      }
    }();

    // Build the filter expression
    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

    // Check the results
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr),
              expected_row_groups);
  }

  // Filtering AST - col_ref != 100
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{};
      } else {
        return std::vector<cudf::size_type>{0, 1, 2, 3};
      }
    }();

    // Build the filter expression
    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_name, literal);

    // Check the results
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr),
              expected_row_groups);
  }
}

TYPED_TEST(RowGroupFilteringWithDictTest, FilterManyLiteralsTyped)
{
  srand(0xcabab);
  using T = TypeParam;

  auto constexpr num_concat          = 1;
  auto constexpr is_constant_strings = true;
  auto constexpr is_nullable         = false;

  // Specifying no compression to explicitly test uncompressed dictionary pages
  auto const buffer =
    std::get<1>(create_parquet_with_stats<T, num_concat, is_constant_strings, is_nullable>(
      100, cudf::io::compression_type::NONE));

  // For string tests use `col2` containing constant "0100" and for temporal types use `col1`
  // containing low cardinality descending values. For all other types use `col0`
  // containing ascending values.
  auto col_name = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_name_reference("col2");
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_name_reference("col1");
    } else {
      return cudf::ast::column_name_reference("col0");
    }
  }();

  // Same logic as above for column reference
  auto col_ref = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_reference(2);
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_reference(1);
    } else {
      return cudf::ast::column_reference(0);
    }
  }();

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Input datasource
  auto const datasource     = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()));
  auto const datasource_ref = std::ref(*datasource);

  // Hybrid scan reader
  auto options             = cudf::io::parquet_reader_options::builder().build();
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);
  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_byte_range);
  reader->setup_page_index(*page_index_buffer);

  auto const reader_ref = std::ref(*reader);

  // First literal value
  auto literal_value1 = [&]() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 100 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(100)), true, stream);  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 100 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(100), true, stream);  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0100"
      return cudf::string_scalar("0100", true, stream);  // i (0-200)
    } else {
      // table[0] == -100 or 100u
      return cudf::numeric_scalar<T>(
        (100 - 200 * std::is_signed_v<T>), true, stream);  // i/100 (-100-100/ 0-200)
    }
  }();

  // Second literal value
  auto literal_value2 = [&]() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 50 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(50)), true, stream);  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 50 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(50), true, stream);  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0050"
      return cudf::string_scalar("0050", true, stream);  // i (0-200)
    } else {
      // table[0] == -50 or 50u
      return cudf::numeric_scalar<T>(
        (50 - 100 * std::is_signed_v<T>), true, stream);  // i/100 (-100-100/ 0-200)
    }
  }();

  // Third literal value
  auto literal_value3 = [&]() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 25 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(25)), true, stream);  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 25 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(25), true, stream);  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0025"
      return cudf::string_scalar("0025", true, stream);  // i (0-200)
    } else {
      // table[0] == -25 or 25u
      return cudf::numeric_scalar<T>(
        (25 - 50 * std::is_signed_v<T>), true, stream);  // i/100 (-100-100/ 0-200)
    }
  }();

  // Filtering AST - col_ref == 100 or col_ref == 50 or col_ref == 25
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{
          0, 1, 2, 3};  // Constant string value present in all RGs
      } else if constexpr (cudf::is_chrono<T>()) {
        return std::vector<cudf::size_type>{
          1, 2, 3};  // Descending temporal values present in three RGs: 1,2,3
      } else if constexpr (cuda::std::is_signed_v<T>) {
        return std::vector<cudf::size_type>{0,
                                            1};  // Signed ascending values present in two RGs: 0,1
      } else {
        return std::vector<cudf::size_type>{
          0, 1, 2};  // Ascending values present in three RGs: 0,1,2
      }
    }();

    // Build the filter expression
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const literal2 = cudf::ast::literal(literal_value2);
    auto const literal3 = cudf::ast::literal(literal_value3);

    auto const filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_name, literal1);
    auto const filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal2);
    auto const filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal3);
    auto const filter_expression12 = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);
    auto const filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression12, filter_expression3);

    // Check the results
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr),
              expected_row_groups);
  }

  // Filtering AST - col_ref != 100 and col_ref != 50 and col_ref != 25
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{};
      } else {
        return std::vector<cudf::size_type>{0, 1, 2, 3};
      }
    }();

    // Build the filter expression
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const literal2 = cudf::ast::literal(literal_value2);
    auto const literal3 = cudf::ast::literal(literal_value3);

    auto const filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, literal1, col_ref);
    auto const filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_ref, literal2);
    auto const filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, literal3, col_name);
    auto const filter_expression12 = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);
    auto const filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression12, filter_expression3);

    // Check the results
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
    EXPECT_EQ(filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr),
              expected_row_groups);
  }
}

TEST_F(HybridScanFiltersTest, RowGroupPasses)
{
  auto constexpr num_rg      = 10;
  auto constexpr rows_per_rg = 1'000;

  // Create a per-row-group table (each write() call produces one row group)
  auto values = cuda::counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int32_t> col0(values, values + rows_per_rg);
  cudf::test::fixed_width_column_wrapper<double> col1(values, values + rows_per_rg);
  auto chunk_table = cudf::table_view{{col0, col1}};

  std::string parquet_filepath = temp_env->get_temp_filepath("RowGroupPassesBasic.parquet");
  {
    auto opts =
      cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{parquet_filepath})
        .build();
    auto writer = cudf::io::chunked_parquet_writer(opts);
    for (int i = 0; i < num_rg; ++i) {
      writer.write(chunk_table);
    }
    writer.close();
  }

  auto options = cudf::io::parquet_reader_options::builder().build();

  auto datasource          = cudf::io::datasource::create(parquet_filepath);
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  auto const all_row_groups = reader->all_row_groups(options);
  EXPECT_EQ(static_cast<int>(all_row_groups.size()), num_rg);

  // No pass read limit. All row groups in a single pass
  {
    auto passes = reader->construct_row_group_passes(all_row_groups, 0);
    EXPECT_EQ(passes.size(), 1);
    EXPECT_EQ(passes.front(), all_row_groups);
  }

  // Small pass limit would result in each row group in its own pass
  {
    auto passes = reader->construct_row_group_passes(all_row_groups, 1);
    EXPECT_EQ(passes.size(), all_row_groups.size());
    auto zipped = cuda::make_zip_iterator(passes.begin(), all_row_groups.begin());
    std::for_each(zipped, zipped + passes.size(), [&](auto const& iter) {
      auto const& pass      = cuda::std::get<0>(iter);
      auto const& row_group = cuda::std::get<1>(iter);
      EXPECT_EQ(pass.size(), 1);
      EXPECT_EQ(pass.front(), row_group);
    });
  }

  // All passes should cover all row groups and be consecutive
  {
    auto passes = reader->construct_row_group_passes(all_row_groups, 1'024);
    std::vector<cudf::size_type> flattened;
    for (auto const& pass : passes) {
      EXPECT_GT(pass.size(), 0);
      flattened.insert(flattened.end(), pass.begin(), pass.end());
    }
    EXPECT_EQ(flattened.size(), all_row_groups.size());
    auto zipped = cuda::make_zip_iterator(flattened.begin(), all_row_groups.begin());
    std::for_each(zipped, zipped + flattened.size(), [&](auto const& iter) {
      auto const& flattened_value = cuda::std::get<0>(iter);
      auto const& row_group       = cuda::std::get<1>(iter);
      EXPECT_EQ(flattened_value, row_group);
    });
  }
}

TEST_F(HybridScanFiltersTest, FetchByteRangesInvalidRanges)
{
  std::vector<std::byte> data(1024);
  auto const datasource =
    cudf::io::datasource::create(cudf::host_span<std::byte const>(data.data(), data.size()));
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  EXPECT_THROW(
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *datasource,
      std::vector<cudf::io::text::byte_range_info>{cudf::io::text::byte_range_info{-1, 16}},
      stream,
      mr),
    cudf::logic_error);

  EXPECT_THROW(
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *datasource,
      std::vector<cudf::io::text::byte_range_info>{cudf::io::text::byte_range_info{512, 1024}},
      stream,
      mr),
    cudf::logic_error);

  EXPECT_THROW(
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *datasource,
      std::vector<cudf::io::text::byte_range_info>{cudf::io::text::byte_range_info{0, -1}},
      stream,
      mr),
    cudf::logic_error);

  EXPECT_NO_THROW(cudf::io::parquet::fetch_byte_ranges_to_device_async(
    *datasource,
    std::vector<cudf::io::text::byte_range_info>{cudf::io::text::byte_range_info{1023, 1}},
    stream,
    mr));
}

class DictionaryFilterGapTest : public HybridScanFiltersTest,
                                public ::testing::WithParamInterface<cudf::io::compression_type> {};

TEST_P(DictionaryFilterGapTest, FilterRowGroupsWithMissingDictPages)
{
  auto const compression                = GetParam();
  auto constexpr num_rows_per_row_group = 20'000;
  // RG 0 holds a single distinct value so it is dict encoded
  // RG 1 holds all distinct values so it falls back
  auto const strings = cudf::detail::make_counting_transform_iterator(0, [](auto const i) {
    return i < num_rows_per_row_group ? std::string{"dict_value"}
                                      : "plain_value_" + std::to_string(i - num_rows_per_row_group);
  });

  auto const column =
    cudf::test::strings_column_wrapper(strings, strings + 2 * num_rows_per_row_group);
  auto const table = cudf::table_view{{column}};

  auto table_metadata = cudf::io::table_input_metadata{table};
  table_metadata.column_metadata[0].set_name("col0");

  auto const filepath = temp_env->get_temp_filepath("DictionaryFilterGapTest.parquet");
  auto const write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table)
      .metadata(std::move(table_metadata))
      .row_group_size_rows(num_rows_per_row_group)
      .dictionary_policy(cudf::io::dictionary_policy::ADAPTIVE)
      .max_dictionary_size(1024)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .compression(compression)
      .write_v2_headers(false)
      .build();
  cudf::io::write_parquet(write_opts);

  // Input datasource
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const datasource     = cudf::io::datasource::create(filepath);
  auto const datasource_ref = std::ref(*datasource);

  // Hybrid scan reader
  auto const default_options = cudf::io::parquet_reader_options::builder().build();
  auto const footer_buffer   = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    *footer_buffer, default_options);

  auto const reader_ref = std::ref(*reader);
  auto const col0_ref   = cudf::ast::column_name_reference("col0");

  // Sanity check: exactly one dictionary page byte range per row group and the second one is empty
  {
    auto literal_value = cudf::string_scalar("dict_value", true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, literal);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

    reader->reset_column_selection();
    auto const row_group_indices = reader->all_row_groups(options);
    ASSERT_EQ(row_group_indices.size(), 2);

    auto const dict_page_byte_ranges =
      reader->secondary_filters_byte_ranges(row_group_indices, options).second;
    ASSERT_EQ(dict_page_byte_ranges.size(), 2);
    EXPECT_GT(dict_page_byte_ranges[0].size(), 0);
    EXPECT_EQ(dict_page_byte_ranges[1].size(), 0);
  }

  // Filtering - col0 == "plain_value_5": row group 0 is pruned by its dictionary, row group 1
  // cannot be pruned as it has no dictionary page
  {
    auto literal_value = cudf::string_scalar("plain_value_5", true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, literal);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

    auto const expected = std::vector<cudf::size_type>{1};
    EXPECT_EQ(filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr),
              expected);
  }

  // Filtering - col0 == "dict_value": both row groups survive
  {
    auto literal_value = cudf::string_scalar("dict_value", true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col0_ref, literal);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

    auto const expected = std::vector<cudf::size_type>{0, 1};
    EXPECT_EQ(filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr),
              expected);
  }

  // Filtering - col0 != "dict_value": row group 0 is pruned as its dictionary holds only that one
  // value, row group 1 cannot be pruned
  {
    auto literal_value = cudf::string_scalar("dict_value", true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col0_ref, literal);
    auto const options =
      cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

    auto const expected = std::vector<cudf::size_type>{1};
    EXPECT_EQ(filter_row_groups_with_dictionaries(datasource_ref, reader_ref, options, stream, mr),
              expected);
  }
}

INSTANTIATE_TEST_SUITE_P(Compression,
                         DictionaryFilterGapTest,
                         ::testing::Values(cudf::io::compression_type::NONE,
                                           cudf::io::compression_type::ZSTD));
