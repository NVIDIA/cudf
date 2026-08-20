/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_inspect_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/iterator>

#include <filesystem>
#include <fstream>
#include <numeric>

/**
 * @file parquet_inspect_utils.cpp
 * @brief Definitions for utilities for `parquet_inspect` example
 */

namespace {

/**
 * @brief Compute page row counts and page row offsets and column chunk page (count) offsets for a
 * given column index
 */
[[nodiscard]] auto compute_page_row_counts_and_offsets(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::size_type col_idx,
  rmm::cuda_stream_view stream)
{
  auto const num_colchunks = metadata.row_groups.front().columns.size();

  // Row counts per page per column chunk
  auto page_row_counts = thrust::host_vector<int64_t>{};
  // Row offsets per page per column chunk
  auto page_row_offsets  = thrust::host_vector<int64_t>{};
  auto page_byte_offsets = thrust::host_vector<int64_t>{};
  // Pages (count) offsets per column chunk
  auto col_page_offsets = thrust::host_vector<cudf::size_type>{};

  page_row_offsets.push_back(0);
  col_page_offsets.push_back(0);

  // For all column chunks
  std::for_each(
    metadata.row_groups.cbegin(), metadata.row_groups.cend(), [&](auto const& row_group) {
      // Find the column chunk with the given schema index
      auto const& colchunk = row_group.columns[col_idx];

      // Compute page row counts and offsets if this column chunk has column and offset indexes
      if (colchunk.offset_index.has_value()) {
        CUDF_EXPECTS(colchunk.column_index.has_value(),
                     "Both offset and column indexes must be present");
        // Get the offset and column indexes of the column chunk
        auto const& offset_index = colchunk.offset_index.value();
        auto const& column_index = colchunk.column_index.value();

        // Number of pages in this column chunk
        auto const row_group_num_pages = offset_index.page_locations.size();

        CUDF_EXPECTS(column_index.min_values.size() == column_index.max_values.size(),
                     "page min and max values should be of same size");
        CUDF_EXPECTS(column_index.min_values.size() == row_group_num_pages,
                     "mismatch between size of min/max page values and the size of page "
                     "locations");
        // Update the cumulative number of pages in this column chunk
        col_page_offsets.push_back(col_page_offsets.back() + row_group_num_pages);

        // For all pages in this column chunk, update page row counts and offsets.
        std::for_each(
          cuda::counting_iterator<std::size_t>{0},
          cuda::counting_iterator{row_group_num_pages},
          [&](auto const page_idx) {
            int64_t const first_row_idx = offset_index.page_locations[page_idx].first_row_index;
            // For the last page, this is simply the total number of rows in the
            // column chunk
            int64_t const last_row_idx =
              (page_idx < row_group_num_pages - 1)
                ? offset_index.page_locations[page_idx + 1].first_row_index
                : row_group.num_rows;

            // Update the page row counts and offsets
            page_row_counts.push_back(last_row_idx - first_row_idx);
            page_byte_offsets.push_back(offset_index.page_locations[page_idx].offset);
            page_row_offsets.push_back(page_row_offsets.back() + page_row_counts.back());
          });
      }
    });
  return std::tuple{std::move(page_row_counts),
                    std::move(page_row_offsets),
                    std::move(page_byte_offsets),
                    std::move(col_page_offsets)};
}

/**
 * @brief Makes an INT64 index column containing elements: [0, size)
 *
 * @param num_rows Number of rows
 * @param stream CUDA stream
 *
 * @return A unique pointer to a column
 */
auto make_index_column(cudf::size_type num_rows, rmm::cuda_stream_view stream)
{
  std::vector<cudf::size_type> data(num_rows);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = rmm::device_buffer(data.data(), num_rows * sizeof(int64_t), stream);
  stream.synchronize();
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                        num_rows,
                                        std::move(buffer),
                                        rmm::device_buffer{},
                                        0);
}

/**
 * @brief Constructs a cuDF column from the host data
 *
 * @tparam T Data type
 * @param host_data Span of host data
 * @param stream CUDA stream
 *
 * @return A unique pointer to a column
 */
template <typename T>
auto make_column(cudf::host_span<T const> host_data, rmm::cuda_stream_view stream)
{
  auto device_buffer = rmm::device_buffer(host_data.data(), host_data.size() * sizeof(T), stream);
  stream.synchronize();
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<T>()},
                                        host_data.size(),
                                        std::move(device_buffer),
                                        rmm::device_buffer{},
                                        0);
}

/**
 * @brief Constructs a list column (one list per row group) for the given page-level data
 *
 * @tparam T Data type
 * @param data Span of host data
 * @param col_page_offsets Span of column page (count) offsets per row group
 * @param num_row_groups Number of row groups
 * @param num_pages_this_column Total number of pages in this column
 * @param stream CUDA stream
 *
 * @return A unique pointer to a column
 */
template <typename T>
auto make_page_data_list_column(cudf::host_span<T const> data,
                                cudf::host_span<cudf::size_type const> col_page_offsets,
                                cudf::size_type num_row_groups,
                                cudf::size_type num_pages_this_column,
                                rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(col_page_offsets.size() == num_row_groups + 1,
               "Mismatch between offsets and number of row groups");

  auto offsets_column = make_column<cudf::size_type>(col_page_offsets, stream);

  auto page_data_buffer =
    rmm::device_buffer(data.data(), num_pages_this_column * sizeof(int64_t), stream);
  stream.synchronize();

  auto page_data_column =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<int64_t>()},
                                   num_pages_this_column,
                                   std::move(page_data_buffer),
                                   rmm::device_buffer{},
                                   0);

  return cudf::make_lists_column(num_row_groups,
                                 std::move(offsets_column),
                                 std::move(page_data_column),
                                 0,
                                 rmm::device_buffer{});
}

}  // namespace

cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) {
    return rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{},
                                         rmm::percent_of_free_device_memory(50)};
  }
  return rmm::mr::cuda_async_memory_resource{};
}

std::tuple<cudf::io::parquet::FileMetaData, bool> read_parquet_file_metadata(
  std::string_view input_filepath)
{
  CUDF_FUNC_RANGE();

  auto file_buffer = std::vector<uint8_t>(std::filesystem::file_size(input_filepath));
  std::ifstream file(input_filepath.data(), std::ios::binary);
  file.read(reinterpret_cast<char*>(file_buffer.data()), file_buffer.size());
  file.close();

  auto options = cudf::io::parquet_reader_options::builder().build();

  auto datasource     = cudf::io::datasource::create(std::string(input_filepath));
  auto datasource_ref = std::ref(*datasource);

  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource_ref);
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Check and setup page index if the file contains one
  auto const has_page_index = not page_index_byte_range.is_empty();
  if (has_page_index) {
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(datasource_ref, page_index_byte_range);
    reader->setup_page_index(*page_index_buffer);
  } else {
    std::cout << "The input parquet file does not contain a page index\n";
  }

  return std::tuple{reader->parquet_metadata(), has_page_index};
}

void write_rowgroup_metadata(cudf::io::parquet::FileMetaData const& metadata,
                             std::string const& output_filepath,
                             rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto const num_row_groups = metadata.row_groups.size();

  // Compute row group row offsets, row group row counts, and row group byte offsets
  auto row_group_row_offsets = std::vector<int64_t>();
  row_group_row_offsets.reserve(num_row_groups + 1);
  row_group_row_offsets.emplace_back(0);

  // Compute row group row counts
  auto row_group_row_counts = std::vector<int64_t>();
  row_group_row_counts.reserve(num_row_groups);

  // Compute row group byte offsets
  auto row_group_byte_offsets = std::vector<int64_t>();
  row_group_byte_offsets.reserve(num_row_groups);

  std::for_each(metadata.row_groups.begin(), metadata.row_groups.end(), [&](auto const& rg) {
    row_group_row_counts.emplace_back(rg.num_rows);
    row_group_row_offsets.emplace_back(row_group_row_offsets.back() + rg.num_rows);
    // Get the file offset of this row group
    auto const row_group_file_offset = [&]() {
      if (rg.file_offset.has_value()) {
        return rg.file_offset.value();
      } else if (rg.columns.front().file_offset != 0) {
        return rg.columns.front().file_offset;
      } else {
        auto const& col_meta = rg.columns.front().meta_data;
        return col_meta.dictionary_page_offset != 0
                 ? std::min(col_meta.dictionary_page_offset, col_meta.data_page_offset)
                 : col_meta.data_page_offset;
      }
    }();
    row_group_byte_offsets.emplace_back(row_group_file_offset);
  });

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(make_index_column(num_row_groups, stream));

  auto row_offsets_buffer =
    rmm::device_buffer(row_group_row_offsets.data(), num_row_groups * sizeof(int64_t), stream);
  auto row_counts_buffer =
    rmm::device_buffer(row_group_row_counts.data(), num_row_groups * sizeof(int64_t), stream);
  auto byte_offsets_buffer =
    rmm::device_buffer(row_group_byte_offsets.data(), num_row_groups * sizeof(int64_t), stream);

  stream.synchronize();
  columns.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<int64_t>()},
                                                      num_row_groups,
                                                      std::move(row_offsets_buffer),
                                                      rmm::device_buffer{},
                                                      0));
  columns.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<int64_t>()},
                                                      num_row_groups,
                                                      std::move(row_counts_buffer),
                                                      rmm::device_buffer{},
                                                      0));
  columns.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<int64_t>()},
                                                      num_row_groups,
                                                      std::move(byte_offsets_buffer),
                                                      rmm::device_buffer{},
                                                      0));

  auto table = std::make_unique<cudf::table>(std::move(columns));

  cudf::io::table_input_metadata out_metadata(table->view());
  out_metadata.column_metadata[0].set_name("row group index");
  out_metadata.column_metadata[1].set_name("row offsets");
  out_metadata.column_metadata[2].set_name("row counts");
  out_metadata.column_metadata[3].set_name("byte offsets");

  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(output_filepath), table->view())
      .metadata(out_metadata)
      .build();
  cudf::io::write_parquet(out_opts, stream);
}

void write_page_metadata(cudf::io::parquet::FileMetaData const& metadata,
                         std::string const& output_filepath,
                         rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto const num_columns    = metadata.row_groups.front().columns.size();
  auto const num_row_groups = metadata.row_groups.size();

  auto constexpr output_cols_per_column = 3;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(make_index_column(num_row_groups, stream));

  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{num_columns},
    [&](auto const col_idx) {
      auto const [page_row_counts, page_row_offsets, page_byte_offsets, col_page_offsets] =
        compute_page_row_counts_and_offsets(metadata, col_idx, stream);

      auto const num_pages_this_column = page_row_counts.size();

      CUDF_EXPECTS(num_pages_this_column == col_page_offsets.back(),
                   "Mismatch between the number of pages and page offsets");

      columns.emplace_back(make_page_data_list_column<int64_t>(
        page_row_counts, col_page_offsets, num_row_groups, num_pages_this_column, stream));
      columns.emplace_back(make_page_data_list_column<int64_t>(
        page_row_offsets, col_page_offsets, num_row_groups, num_pages_this_column, stream));
      columns.emplace_back(make_page_data_list_column<int64_t>(
        page_byte_offsets, col_page_offsets, num_row_groups, num_pages_this_column, stream));

      stream.synchronize();
    });

  CUDF_EXPECTS(columns.size() == (num_columns * output_cols_per_column) + 1,
               "Mismatch between number of columns and number of columns in the table");

  auto table = std::make_unique<cudf::table>(std::move(columns));
  cudf::io::table_input_metadata out_metadata(table->view());
  out_metadata.column_metadata[0].set_name("row group index");

  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{num_columns},
                [&](auto const col_idx) {
                  std::string const col_name = "col" + std::to_string(col_idx);
                  out_metadata.column_metadata[1 + col_idx * output_cols_per_column].set_name(
                    col_name + " page row counts");
                  out_metadata.column_metadata[1 + col_idx * output_cols_per_column + 1].set_name(
                    col_name + " page row offsets ");
                  out_metadata.column_metadata[1 + col_idx * output_cols_per_column + 2].set_name(
                    col_name + " page byte offsets");
                });

  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(output_filepath), table->view())
      .metadata(out_metadata)
      .build();
  cudf::io::write_parquet(out_opts, stream);
}
