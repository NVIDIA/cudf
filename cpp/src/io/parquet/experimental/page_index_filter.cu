/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_helpers.hpp"
#include "io/parquet/stats_filter_helpers.hpp"
#include "io/parquet/timestamp_utils.cuh"
#include "page_index_filter_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/logger.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/stream>

#include <algorithm>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::stats_caster_base;
using metadata_base = parquet::detail::metadata;

namespace {

/**
 * @brief Converts page-level statistics of an input column to three device columns (min, max
 * values, and all-null) with the host page-row offsets that map its rows back to the input column's
 * pages.
 */
struct page_stats_caster : public stats_caster_base {
  cudf::size_type total_rows;
  std::span<metadata_base const> per_file_metadata;
  std::span<std::vector<size_type> const> row_group_indices;

  /**
   * @brief Computes host side data including page row offsets and host columns containing
   * page-level min, max, and all-null statistics for a column
   *
   * @param schema_idx Column schema index
   * @param dtype Column data type
   * @param stream CUDA stream
   * @return A tuple of page row offsets and host columns containing page-level min, max, and
   * all-null statistics
   */
  template <typename T>
  [[nodiscard]] auto compute_host_data(cudf::size_type schema_idx,
                                       cudf::data_type dtype,
                                       cuda::stream_ref stream) const
  {
    // Compute column chunk level page count offsets and page level row offsets.
    auto const [page_row_offsets, col_chunk_page_offsets] =
      compute_page_row_offsets_and_colchunk_page_offsets(
        per_file_metadata, row_group_indices, schema_idx, stream);

    CUDF_EXPECTS(page_row_offsets.back() == total_rows,
                 "The number of rows must be equal across row groups and pages within row groups");

    auto const total_pages = col_chunk_page_offsets.back();

    // Create host columns with page-level min, max, and all-null statistics. The all-null column
    // is true only when every value in the page is null, false when none are, and null when only
    // some are, which is what lets it answer both IS_NULL and IS NOT NULL.
    host_column<T> min(total_pages, stream);
    host_column<T> max(total_pages, stream);
    host_column<bool> all_null(total_pages, stream);

    // Compute timestamp scale factor for precision conversion
    auto const ts_scale = [&] {
      if constexpr (cudf::is_timestamp<T>()) {
        auto const& schema = per_file_metadata[0].schema[schema_idx];
        return parquet::detail::calc_timestamp_scale(schema.logical_type,
                                                     static_cast<int32_t>(T::period::den));
      }
      return 0;
    }();

    // Populate the host columns with page-level min, max statistics from the page index
    auto page_offset_idx = 0;
    // For all row data sources
    std::for_each(
      cuda::counting_iterator<std::size_t>{0},
      cuda::counting_iterator{row_group_indices.size()},
      [&](auto src_idx) {
        // For all column chunks in this source
        auto const& rg_indices = row_group_indices[src_idx];
        std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
          auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
          // Find colchunk_iter in row_group.columns. Guaranteed to be found as already verified
          // in compute_page_row_offsets_and_colchunk_page_offsets()
          auto colchunk_iter = std::find_if(
            row_group.columns.begin(),
            row_group.columns.end(),
            [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });

          auto const& colchunk               = *colchunk_iter;
          auto const& column_index           = colchunk.column_index.value();
          auto const num_pages_in_colchunk   = column_index.min_values.size();
          auto const page_offset_in_colchunk = col_chunk_page_offsets[page_offset_idx++];

          CUDF_EXPECTS(column_index.null_pages.size() == num_pages_in_colchunk,
                       "Number of null page flags must match the number of pages in the column "
                       "chunk",
                       std::invalid_argument);
          CUDF_EXPECTS(not column_index.null_counts.has_value() or
                         column_index.null_counts.value().size() == num_pages_in_colchunk,
                       "Number of page null counts must match the number of pages in the column "
                       "chunk",
                       std::invalid_argument);

          // For all pages in this column chunk
          std::for_each(
            cuda::counting_iterator<std::size_t>{0},
            cuda::counting_iterator{num_pages_in_colchunk},
            [&](auto page_idx) {
              auto const& min_value      = column_index.min_values[page_idx];
              auto const& max_value      = column_index.max_values[page_idx];
              auto const column_page_idx = page_offset_in_colchunk + page_idx;
              // Translate binary data to Type then to <T>
              min.set_index(column_page_idx, min_value, colchunk.meta_data.type, ts_scale);
              max.set_index(column_page_idx, max_value, colchunk.meta_data.type, ts_scale);
              // Check if the page is completely null
              if (column_index.null_pages[page_idx]) {
                all_null.val[column_page_idx] = true;
                return;
              }
              // Check if the page doesn't have a null count
              if (not column_index.null_counts.has_value()) {
                all_null.set_index(column_page_idx, std::nullopt, {});
                return;
              }
              // Use the null count to determine if the page is completely null
              auto const page_row_count =
                page_row_offsets[column_page_idx + 1] - page_row_offsets[column_page_idx];
              auto const& null_count = column_index.null_counts.value()[page_idx];
              if (null_count == 0) {
                all_null.val[column_page_idx] = false;
              } else if (null_count < page_row_count) {
                all_null.set_index(column_page_idx, std::nullopt, {});
              } else if (null_count == page_row_count) {
                all_null.val[column_page_idx] = true;
              } else {
                CUDF_FAIL("Invalid null count");
              }
            });
        });
      });

    return std::tuple{
      std::move(page_row_offsets), std::move(min), std::move(max), std::move(all_null)};
  }

  /**
   * @brief Builds a `page_statistics_input` storing compact per-column stats (min, max, all-null)
   * with the host page-row offsets that map its rows back to the input column's pages.
   *
   * @tparam T Underlying type of the column
   * @param column_index Logical filter column index used by the stats expression
   * @param schema_idx Input column schema index
   * @param dtype Input column data type
   * @param stream CUDA stream
   * @param mr Device memory resource used to allocate the returned columns' device memory
   * @return A `page_statistics_input` storing the compacted page-level stats table and host-side
   * page-row offsets for this column
   */
  template <typename T>
  [[nodiscard]] page_statistics_input operator()(size_type column_index,
                                                 size_type schema_idx,
                                                 data_type dtype,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr) const
  {
    if constexpr (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types other than strings do not have statistics");
    } else {
      // Compute page row offsets, and page-statistics (min, max and all-null) host columns.
      auto [page_row_offsets, min, max, all_null] = compute_host_data<T>(schema_idx, dtype, stream);

      std::vector<std::unique_ptr<column>> columns;
      columns.reserve(parquet::detail::stats_cols_per_column);
      columns.emplace_back(min.to_device(dtype, stream, mr));
      columns.emplace_back(max.to_device(dtype, stream, mr));
      columns.emplace_back(all_null.to_device(data_type{type_id::BOOL8}, stream, mr));

      return page_statistics_input{
        .column_index = column_index,
        .statistics   = cudf::table{std::move(columns)},
        .page_row_offsets =
          std::vector<size_type>(page_row_offsets.begin(), page_row_offsets.end()),
      };
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> aggregate_reader_metadata::build_row_mask_with_page_index_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::data_type const> output_dtypes,
  std::span<cudf::size_type const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Return if empty row group indices
  if (row_group_indices.empty()) { return cudf::make_empty_column(cudf::type_id::BOOL8); }

  // Total number of rows
  auto const total_rows = total_rows_in_row_groups(row_group_indices);
  if (total_rows == 0) { return cudf::make_empty_column(cudf::type_id::BOOL8); }

  // TODO(#22900): remove this guard once this path maps schema indices per source. It currently
  // reuses one source's schema index for every source, so it is correct only when schemas match.
  CUDF_EXPECTS(schema_idx_maps.empty(),
               "Page index statistics filtering does not support mismatched Parquet schemas yet",
               std::invalid_argument);

  CUDF_EXPECTS(std::cmp_less_equal(total_rows, std::numeric_limits<size_type>::max()),
               "Total rows in row groups exceed the cudf's column size limit. Retry with a smaller "
               "set of row groups",
               std::invalid_argument);

  auto const num_columns = output_dtypes.size();

  // Get a boolean mask indicating which columns will participate in stats based filtering
  auto const stats_columns_mask =
    parquet::detail::stats_columns_collector{filter.get(), output_dtypes}.get_stats_columns_mask();

  // Return early if no columns will participate in stats based page filtering
  if (stats_columns_mask.empty()) { return build_all_true_row_mask(row_group_indices, stream, mr); }

  // Check if we have page index available for all participating columns
  std::vector<size_type> stats_column_schemas;
  stats_column_schemas.reserve(num_columns);
  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{num_columns},
                [&](auto const col_idx) {
                  auto const& dtype = output_dtypes[col_idx];
                  if (stats_columns_mask[col_idx] and
                      (not cudf::is_compound(dtype) or dtype.id() == cudf::type_id::STRING)) {
                    stats_column_schemas.push_back(output_column_schemas[col_idx]);
                  }
                });

  // Return early if no participating columns
  if (stats_column_schemas.empty()) {
    return build_all_true_row_mask(row_group_indices, stream, mr);
  }

  // We need both column and offset indexes to be present for each participating column.
  auto const [has_column_index, has_offset_index] =
    page_index_presence(row_group_indices, stats_column_schemas);
  CUDF_EXPECTS(has_column_index and has_offset_index,
               "Filter column page pruning using page-statistics requires both column and "
               "offset indexes to be present",
               std::runtime_error);

  // Convert the filter to an expression over page statistics
  parquet::detail::stats_expression_converter const stats_expr_converter{filter.get(),
                                                                         output_dtypes};

  // Return early if statistics cannot prune any pages using the filter
  auto const stats_expr = stats_expr_converter.get_stats_expr();
  if (not stats_expr.has_value()) { return build_all_true_row_mask(row_group_indices, stream, mr); }

  page_stats_caster const stats_col{.total_rows        = static_cast<size_type>(total_rows),
                                    .per_file_metadata = per_file_metadata,
                                    .row_group_indices = row_group_indices};

  // Build page-statistics inputs for each participating column.
  std::vector<page_statistics_input> stats_inputs;
  stats_inputs.reserve(stats_column_schemas.size());
  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{num_columns},
                [&](auto const col_idx) {
                  auto const& dtype = output_dtypes[col_idx];
                  // Only participating columns and comparable types are supported
                  if (stats_columns_mask[col_idx] and
                      (not cudf::is_compound(dtype) or dtype.id() == type_id::STRING)) {
                    stats_inputs.emplace_back(cudf::type_dispatcher<dispatch_storage_type>(
                      dtype,
                      stats_col,
                      static_cast<size_type>(col_idx),
                      output_column_schemas[col_idx],
                      dtype,
                      stream,
                      cudf::get_current_device_resource_ref()));
                  }
                });

  // Return an all true row mask if no columns can participate in stats based filtering
  if (stats_inputs.empty()) { return build_all_true_row_mask(row_group_indices, stream, mr); }

  // Compute the row mask from page statistics
  return compute_row_mask_from_page_stats(stats_inputs,
                                          static_cast<size_type>(num_columns),
                                          static_cast<size_type>(total_rows),
                                          stats_expr.value(),
                                          stream,
                                          mr);
}

thrust::host_vector<bool> aggregate_reader_metadata::compute_data_page_mask(
  cudf::column_view const& row_mask,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<input_column_info const> input_columns,
  cuda::stream_ref stream) const
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(row_mask.type().id() == cudf::type_id::BOOL8,
               "Input row bitmask should be of type BOOL8");

  auto const total_rows = total_rows_in_row_groups(row_group_indices);
  CUDF_EXPECTS(std::cmp_less_equal(total_rows, std::numeric_limits<size_type>::max()),
               "Total rows in row groups exceed the cudf's column size limit. Retry with a smaller "
               "set of row groups",
               std::invalid_argument);

  CUDF_EXPECTS(
    std::cmp_equal(total_rows, row_mask.size()),
    "Encountered a mismatch in number of rows in the row group pass and the row mask size",
    std::overflow_error);

  // Return an empty vector if all rows are required
  if (are_all_rows_retained(row_mask, stream)) { return thrust::host_vector<bool>{}; }

  // Collect column schema indices from the input columns.
  auto column_schema_indices = std::vector<size_type>(input_columns.size());
  std::transform(
    input_columns.begin(), input_columns.end(), column_schema_indices.begin(), [](auto const& col) {
      return col.schema_idx;
    });

  // Mapping a row mask to data pages only requires page row locations from the offset index.
  auto const has_offset_index =
    page_index_presence(row_group_indices, column_schema_indices).second;
  if (not has_offset_index) {
    CUDF_LOG_WARN(
      "Encountered missing Parquet offset index for one or more output columns. Skipping "
      "page-index based pruning.");
    return thrust::host_vector<bool>(0);
  }

  // TODO(#22900): remove this guard once this path maps schema indices per source. It currently
  // reuses one source's schema index for every source, so it is correct only when schemas match.
  CUDF_EXPECTS(schema_idx_maps.empty(),
               "Data page masking does not support mismatched Parquet schemas yet",
               std::invalid_argument);

  // Compute page row offsets and column chunk page offsets for each column
  auto const num_columns = input_columns.size();
  std::vector<size_type> page_row_offsets;
  std::vector<size_type> col_page_offsets;
  col_page_offsets.reserve(num_columns + 1);
  col_page_offsets.push_back(0);

  size_type max_page_size = 0;

  if (num_columns <= 2) {
    std::for_each(
      column_schema_indices.begin(), column_schema_indices.end(), [&](auto const schema_idx) {
        auto [col_page_row_offsets, col_max_page_size] =
          compute_page_row_offsets(per_file_metadata, row_group_indices, schema_idx);
        page_row_offsets.insert(page_row_offsets.end(),
                                std::make_move_iterator(col_page_row_offsets.begin()),
                                std::make_move_iterator(col_page_row_offsets.end()));
        max_page_size = std::max<size_type>(max_page_size, col_max_page_size);
        col_page_offsets.emplace_back(page_row_offsets.size());
      });
  } else {
    // Using a maximum of 2 tasks to compute page row offsets for columns to avoid excessive
    // task submission overheads
    auto constexpr max_tasks         = 2;
    using task_page_row_offsets_type = std::vector<std::pair<std::vector<size_type>, size_type>>;
    std::vector<std::future<task_page_row_offsets_type>> page_row_offset_tasks{};
    page_row_offset_tasks.reserve(max_tasks);
    auto const cols_per_thread =
      cudf::util::div_rounding_up_safe<std::size_t>(num_columns, max_tasks);

    // Submit page row offset compute tasks
    std::transform(cuda::counting_iterator<int>{0},
                   cuda::counting_iterator{max_tasks},
                   std::back_inserter(page_row_offset_tasks),
                   [&](auto const tid) {
                     return cudf::detail::host_worker_pool().submit_task([&, tid = tid]() {
                       auto const start_col = std::min(tid * cols_per_thread, num_columns);
                       auto const end_col   = std::min(start_col + cols_per_thread, num_columns);
                       task_page_row_offsets_type task_page_row_offsets{};
                       task_page_row_offsets.reserve(end_col - start_col);
                       std::transform(
                         cuda::counting_iterator{start_col},
                         cuda::counting_iterator{end_col},
                         std::back_inserter(task_page_row_offsets),
                         [&](auto const col_idx) {
                           return compute_page_row_offsets(
                             per_file_metadata, row_group_indices, column_schema_indices[col_idx]);
                         });
                       return task_page_row_offsets;
                     });
                   });

    std::for_each(page_row_offset_tasks.begin(), page_row_offset_tasks.end(), [&](auto& task) {
      auto const& task_page_row_offsets = task.get();
      for (auto& [col_page_row_offsets, col_max_page_size] : task_page_row_offsets) {
        page_row_offsets.insert(page_row_offsets.end(),
                                std::make_move_iterator(col_page_row_offsets.begin()),
                                std::make_move_iterator(col_page_row_offsets.end()));
        max_page_size = std::max<size_type>(max_page_size, col_max_page_size);
        col_page_offsets.emplace_back(page_row_offsets.size());
      }
    });
  }

  auto data_page_mask = thrust::host_vector<bool>{};

  auto const row_range_mask =
    compute_row_range_selection_mask(row_mask, page_row_offsets, max_page_size, stream);
  if (row_range_mask.empty()) { return data_page_mask; }

  data_page_mask.reserve(page_row_offsets.size() - num_columns);
  // Discard results for invalid ranges. i.e. ranges starting at the last page of a column and
  // ending at the first page of the next column
  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{num_columns},
                [&](auto col_idx) {
                  auto const col_num_pages =
                    col_page_offsets[col_idx + 1] - col_page_offsets[col_idx] - 1;
                  auto const first_page_range = col_page_offsets[col_idx];
                  data_page_mask.insert(data_page_mask.end(),
                                        row_range_mask.begin() + first_page_range,
                                        row_range_mask.begin() + first_page_range + col_num_pages);
                });
  return data_page_mask;
}

}  // namespace cudf::io::parquet::experimental::detail
