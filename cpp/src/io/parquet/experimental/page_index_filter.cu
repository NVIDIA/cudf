/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_helpers.hpp"
#include "io/parquet/stats_filter_helpers.hpp"
#include "io/parquet/timestamp_utils.cuh"
#include "page_index_filter_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/logger.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/gather.h>

#include <algorithm>
#include <limits>
#include <span>

namespace cudf::io::parquet::experimental::detail {

using metadata_base     = parquet::detail::metadata;
using stats_caster_base = parquet::detail::stats_caster_base;
using string_index_pair = parquet::detail::string_index_pair;

namespace {

/**
 * @brief Converts page-level statistics of a column to 2 device columns - min, max values. Each
 * column has number of rows equal to the total rows in all row groups.
 */
struct page_stats_caster : public stats_caster_base {
  cudf::size_type total_rows;
  cudf::host_span<metadata_base const> per_file_metadata;
  std::span<std::vector<size_type> const> row_group_indices;
  bool const has_is_null_operator;

  /**
   * @brief Transforms a page-level stats column to a row-level stats column for non-string types
   *
   * @tparam T The data type of the column - must be non-compound
   * @param input_column Mutable view of input page-level device column
   * @param page_nullmask Host nullmask of the input page-level column
   * @param page_indices Device vector containing the page index for each row index
   * @param page_row_offsets Host vector row offsets of each page
   * @param dtype The data type of the column
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair containing the output data buffer and nullmask
   */
  template <typename T>
  [[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> build_data_and_nullmask(
    mutable_column_view input_column,
    bitmask_type const* page_nullmask,
    cudf::device_span<size_type const> page_indices,
    cudf::host_span<size_type const> page_row_offsets,
    cudf::data_type dtype,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const
    requires(not cudf::is_compound<T>())
  {
    // Total number of pages in the column
    size_type const total_pages = page_row_offsets.size() - 1;

    // Buffer for output data
    auto output_data = rmm::device_buffer(cudf::size_of(dtype) * total_rows, stream, mr);

    // For each row index, copy over the min/max page stat value from the corresponding page.
    thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   page_indices.begin(),
                   page_indices.end(),
                   input_column.template begin<T>(),
                   reinterpret_cast<T*>(output_data.data()));

    // Buffer for output bitmask
    auto output_nullmask = rmm::device_buffer{0, stream, mr};
    if (input_column.null_count()) {
      // Set all bits in output nullmask to valid
      output_nullmask = cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);
      // For each input page, invalidate the null mask for corresponding rows if needed.
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{total_pages},
                    [&](auto const page_idx) {
                      if (not bit_is_set(page_nullmask, page_idx)) {
                        cudf::set_null_mask(static_cast<bitmask_type*>(output_nullmask.data()),
                                            page_row_offsets[page_idx],
                                            page_row_offsets[page_idx + 1],
                                            false,
                                            stream);
                      }
                    });
    }

    return {std::move(output_data), std::move(output_nullmask)};
  }

  /**
   * @brief Builds a device column containing each page's `is_null` statistic at
   *        respectively of a column at each row index.
   *
   * @param is_null Host column storing the page-level is_null statistics
   * @param page_indices Device vector containing the page index for each row index
   * @param page_row_offsets Host vector row offsets of each page
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair containing the output data buffer and nullmask
   */
  [[nodiscard]] std::unique_ptr<column> build_is_null_device_column(
    host_column<bool> const& is_null,
    cudf::device_span<size_type const> page_indices,
    cudf::host_span<size_type const> page_row_offsets,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(
      has_is_null_operator,
      "The filter expression must have an IS_NULL operator to build is_null device column");
    auto const dtype = cudf::data_type{cudf::type_id::BOOL8};
    auto is_nullcol  = is_null.to_device(dtype, stream, cudf::get_current_device_resource_ref());
    auto [is_null_data, is_null_nullmask] =
      build_data_and_nullmask<bool>(is_nullcol->mutable_view(),
                                    is_null.null_mask.data(),
                                    page_indices,
                                    page_row_offsets,
                                    dtype,
                                    stream,
                                    mr);
    auto const is_null_nulls =
      is_nullcol->null_count()
        ? cudf::detail::null_count(
            reinterpret_cast<bitmask_type*>(is_null_nullmask.data()), 0, total_rows, stream)
        : 0;
    return std::make_unique<column>(
      dtype, total_rows, std::move(is_null_data), std::move(is_null_nullmask), is_null_nulls);
  }

  /**
   * @brief Transforms a page-level stats column to a row-level stats column for string type
   *
   * @param host_strings Host span of cudf::string_view values in the input page-level host column
   * @param host_chars Host span of string data of the input page-level host column
   * @param host_page_nullmask Nullmask of the input page-level host column
   * @param host_null_count Number of nulls in the input page-level host column
   * @param page_indices Device vector containing the page index for each row index
   * @param page_row_offsets Host vector row offsets of each page
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair containing the output data buffer and nullmask
   */
  [[nodiscard]] std::
    tuple<rmm::device_buffer, rmm::device_uvector<cudf::size_type>, rmm::device_buffer>
    build_string_data_and_nullmask(cudf::host_span<cudf::string_view const> host_strings,
                                   cudf::host_span<char const> host_chars,
                                   bitmask_type const* host_page_nullmask,
                                   size_type host_null_count,
                                   cudf::device_span<size_type const> page_indices,
                                   cudf::host_span<size_type const> page_row_offsets,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr) const
  {
    // Total number of pages in the column
    size_type const total_pages = page_row_offsets.size() - 1;

    // Construct device vectors containing page-level (input) string data, and offsets and sizes
    auto [page_str_chars, page_str_offsets, page_str_sizes] =
      host_column<cudf::string_view>::make_strings_children(
        host_strings, host_chars, stream, cudf::get_current_device_resource_ref());

    // Buffer for row-level string sizes (output).
    auto row_str_sizes = rmm::device_uvector<std::size_t>(total_rows, stream, mr);
    // Gather string sizes from page to row level
    thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   page_indices.begin(),
                   page_indices.end(),
                   page_str_sizes.begin(),
                   row_str_sizes.begin());

    // Total bytes in the output chars buffer
    auto const total_bytes = cudf::detail::reduce(row_str_sizes.begin(),
                                                  row_str_sizes.end(),
                                                  std::size_t{0},
                                                  cuda::std::plus<std::size_t>{},
                                                  stream);

    CUDF_EXPECTS(
      total_bytes <= cuda::std::numeric_limits<cudf::size_type>::max(),
      "The strings child of the page statistics column cannot exceed the column size limit");

    // page-level strings nullmask (input)
    auto const input_nullmask = host_page_nullmask;

    // Buffer for row-level strings nullmask (output)
    auto output_nullmask = rmm::device_buffer{0, stream, mr};
    if (host_null_count) {
      // Set all bits in output nullmask to valid
      output_nullmask = cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);
      // For each input page, invalidate the null mask for corresponding rows if needed.
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{total_pages},
                    [&](auto const page_idx) {
                      if (not bit_is_set(input_nullmask, page_idx)) {
                        cudf::set_null_mask(static_cast<bitmask_type*>(output_nullmask.data()),
                                            page_row_offsets[page_idx],
                                            page_row_offsets[page_idx + 1],
                                            false,
                                            stream);
                      }
                    });
    }

    // Buffer for row-level string offsets (output).
    auto row_str_offsets =
      cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(total_rows + 1, stream, mr);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           row_str_sizes.begin(),
                           row_str_sizes.end(),
                           row_str_offsets.begin() + 1);

    // Buffer for row-level string chars (output).
    auto row_str_chars = rmm::device_buffer(total_bytes, stream, mr);

    // Iterator for input (page-level) string chars
    auto const page_offsets =
      cuda::make_permutation_iterator(page_str_offsets.begin(), page_indices.begin());
    auto src_iter = cuda::transform_iterator(
      page_offsets,
      cuda::proclaim_return_type<char*>(
        [chars = page_str_chars.begin()] __device__(auto offset) { return chars + offset; }));

    // Iterator for output (row-level) string chars
    auto dst_iter =
      cuda::transform_iterator(row_str_offsets.begin(),
                               cuda::proclaim_return_type<char*>(
                                 [chars = reinterpret_cast<char*>(row_str_chars.data())] __device__(
                                   auto offset) { return chars + offset; }));

    // Iterator for string sizes
    auto size_iter = row_str_sizes.begin();

    // Gather page-level string chars to row-level string chars
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, size_iter, total_rows, stream);

    // Return row-level (output) strings children and the nullmask
    return std::tuple{
      std::move(row_str_chars), std::move(row_str_offsets), std::move(output_nullmask)};
  }

  /**
   * @brief Computes host side data including page row offsets, column chunk page offsets, and host
   * columns containing page-level min, max and (optional) all-null statistics for a column
   *
   * @param schema_idx Column schema index
   * @param dtype Column data type
   * @param stream CUDA stream
   * @return A tuple of page row offsets, column chunk page offsets, and host columns containing
   * page-level min, max and (optional) all-null statistics
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

    // Create host columns with page-level min, max and optionally all-null statistics. The
    // all-null column is true only when every value in the page is null, false when none are, and
    // null when only some are, which is what lets it answer both IS_NULL and IS NOT NULL.
    host_column<T> min(total_pages, stream);
    host_column<T> max(total_pages, stream);
    std::optional<host_column<bool>> all_null;
    if (has_is_null_operator) { all_null = host_column<bool>(total_pages, stream); }

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

          if (has_is_null_operator) {
            CUDF_EXPECTS(column_index.null_pages.size() == num_pages_in_colchunk,
                         "Number of null page flags must match the number of pages in the column "
                         "chunk",
                         std::invalid_argument);
            CUDF_EXPECTS(not column_index.null_counts.has_value() or
                           column_index.null_counts.value().size() == num_pages_in_colchunk,
                         "Number of page null counts must match the number of pages in the column "
                         "chunk",
                         std::invalid_argument);
          }

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
              if (has_is_null_operator) {
                // Check if the page is completely null
                if (column_index.null_pages[page_idx]) {
                  all_null->val[column_page_idx] = true;
                  return;
                }
                // Check if the page doesn't have a null count
                if (not column_index.null_counts.has_value()) {
                  all_null->set_index(column_page_idx, std::nullopt, {});
                  return;
                }
                // Use the null count to determine if the page is completely null
                auto const page_row_count =
                  page_row_offsets[column_page_idx + 1] - page_row_offsets[column_page_idx];
                auto const& null_count = column_index.null_counts.value()[page_idx];
                if (null_count == 0) {
                  all_null->val[column_page_idx] = false;
                } else if (null_count < page_row_count) {
                  all_null->set_index(column_page_idx, std::nullopt, {});
                } else if (null_count == page_row_count) {
                  all_null->val[column_page_idx] = true;
                } else {
                  CUDF_FAIL("Invalid null count");
                }
              }
            });
        });
      });

    return std::tuple{std::move(page_row_offsets),
                      std::move(col_chunk_page_offsets),
                      std::move(min),
                      std::move(max),
                      std::move(all_null)};
  }

  /**
   * @brief Builds three device columns storing the corresponding page-level statistics
   *        (min, max, is_null) respectively of a column at each row index
   *
   * @tparam T underlying type of the column
   * @param schema_idx Column schema index
   * @param dtype Column data type
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A tuple of device columns with min, max and optionally is_null value from page
   * statistics for each row
   */
  template <typename T>
  [[nodiscard]] std::
    tuple<std::unique_ptr<column>, std::unique_ptr<column>, std::optional<std::unique_ptr<column>>>
    operator()(cudf::size_type schema_idx,
               cudf::data_type dtype,
               cuda::stream_ref stream,
               rmm::device_async_resource_ref mr) const
  {
    // List, Struct, Dictionary types are not supported
    if constexpr (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types other than strings do not have statistics");
    } else {
      // Compute page row offsets, column chunk page offsets, min, max and optional is_null stats
      // host columns.
      auto [page_row_offsets, col_chunk_page_offsets, min, max, is_null] =
        compute_host_data<T>(schema_idx, dtype, stream);

      // Construct a row indices mapping based on page row offsets.
      auto const page_indices = compute_page_indices_async(
        page_row_offsets, total_rows, stream, cudf::get_current_device_resource_ref());
      stream.sync();

      // For non-strings columns, directly gather the page-level column data and bitmask to the
      // row-level.
      if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
        // Move host min/max columns to device
        auto mincol = min.to_device(dtype, stream, cudf::get_current_device_resource_ref());
        auto maxcol = max.to_device(dtype, stream, cudf::get_current_device_resource_ref());

        // Convert page-level min and max columns to row-level min and max columns by gathering
        // values based on page-level row offsets
        auto [min_data, min_nullmask] = build_data_and_nullmask<T>(mincol->mutable_view(),
                                                                   min.null_mask.data(),
                                                                   page_indices,
                                                                   page_row_offsets,
                                                                   dtype,
                                                                   stream,
                                                                   mr);
        auto [max_data, max_nullmask] = build_data_and_nullmask<T>(maxcol->mutable_view(),
                                                                   max.null_mask.data(),
                                                                   page_indices,
                                                                   page_row_offsets,
                                                                   dtype,
                                                                   stream,
                                                                   mr);

        // Count nulls in min and max columns
        auto const min_nulls =
          mincol->null_count()
            ? cudf::detail::null_count(
                reinterpret_cast<bitmask_type*>(min_nullmask.data()), 0, total_rows, stream)
            : 0;
        auto const max_nulls =
          maxcol->null_count()
            ? cudf::detail::null_count(
                reinterpret_cast<bitmask_type*>(max_nullmask.data()), 0, total_rows, stream)
            : 0;
        // Return min, max and is_null device columns
        return {std::make_unique<column>(
                  dtype, total_rows, std::move(min_data), std::move(min_nullmask), min_nulls),
                std::make_unique<column>(
                  dtype, total_rows, std::move(max_data), std::move(max_nullmask), max_nulls),
                has_is_null_operator
                  ? std::make_optional(build_is_null_device_column(
                      is_null.value(), page_indices, page_row_offsets, stream, mr))
                  : std::nullopt};
      }
      // For strings columns, gather the page-level string offsets and bitmask to row-level
      // directly and gather string chars using a batched memcpy.
      else {
        auto [min_data, min_offsets, min_nullmask] =
          build_string_data_and_nullmask(min.val,
                                         min.chars,
                                         min.null_mask.data(),
                                         min.null_count,
                                         page_indices,
                                         page_row_offsets,
                                         stream,
                                         mr);
        auto [max_data, max_offsets, max_nullmask] =
          build_string_data_and_nullmask(max.val,
                                         max.chars,
                                         max.null_mask.data(),
                                         max.null_count,
                                         page_indices,
                                         page_row_offsets,
                                         stream,
                                         mr);

        // Count nulls in min and max columns
        auto const min_nulls =
          min.null_count
            ? cudf::detail::null_count(
                reinterpret_cast<bitmask_type*>(min_nullmask.data()), 0, total_rows, stream)
            : 0;
        auto const max_nulls =
          max.null_count
            ? cudf::detail::null_count(
                reinterpret_cast<bitmask_type*>(max_nullmask.data()), 0, total_rows, stream)
            : 0;

        // Return min, max and is_null device strings columns
        return {
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(min_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(min_data),
            min_nulls,
            std::move(min_nullmask)),
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(max_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(max_data),
            max_nulls,
            std::move(max_nullmask)),
          has_is_null_operator ? std::make_optional(build_is_null_device_column(
                                   is_null.value(), page_indices, page_row_offsets, stream, mr))
                               : std::nullopt};
      }
    }
  }
};

/**
 * @brief Converts page-level statistics of single column to a surviving row mask device column
 */
struct page_stats_to_row_mask_converter : public page_stats_caster {
  page_stats_to_row_mask_converter(cudf::size_type total_rows,
                                   cudf::host_span<metadata_base const> per_file_metadata,
                                   std::span<std::vector<size_type> const> row_group_indices,
                                   bool has_is_null_operator)
    : page_stats_caster{.total_rows           = total_rows,
                        .per_file_metadata    = per_file_metadata,
                        .row_group_indices    = row_group_indices,
                        .has_is_null_operator = has_is_null_operator}
  {
  }

  template <typename T>
  [[nodiscard]] std::unique_ptr<cudf::column> operator()(
    cudf::size_type schema_idx,
    cudf::data_type dtype,
    std::reference_wrapper<ast::expression const> filter,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const
  {
    // List, Struct, Dictionary types are not supported
    if constexpr (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types other than strings do not have statistics");
    } else {
      // Compute page row offsets, column chunk page offsets, min, max and optional is_null stats
      // host columns.
      auto [page_row_offsets, col_chunk_page_offsets, min, max, is_null] =
        compute_host_data<T>(schema_idx, dtype, stream);

      std::vector<std::unique_ptr<column>> columns;
      columns.emplace_back(min.to_device(dtype, stream, cudf::get_current_device_resource_ref()));
      columns.emplace_back(max.to_device(dtype, stream, cudf::get_current_device_resource_ref()));
      if (has_is_null_operator) {
        columns.emplace_back(is_null->to_device(
          cudf::data_type{cudf::type_id::BOOL8}, stream, cudf::get_current_device_resource_ref()));
      }

      auto page_stats_table = cudf::table(std::move(columns));
      // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
      parquet::detail::stats_expression_converter const stats_expr{
        filter.get(), std::span{&dtype, 1}, stream};

      // Filter the input table using AST expression and return the (BOOL8) predicate column.
      auto const page_mask = cudf::detail::compute_column(page_stats_table,
                                                          stats_expr.get_stats_expr().get(),
                                                          stream,
                                                          cudf::get_current_device_resource_ref());

      auto const page_indices = compute_page_indices_async(
        page_row_offsets, total_rows, stream, cudf::get_current_device_resource_ref());

      auto const page_mask_nullmask =
        page_mask->null_count()
          ? cudf::detail::make_host_vector(
              cudf::device_span<bitmask_type const>{
                page_mask->view().null_mask(),
                static_cast<std::size_t>(num_bitmask_words(page_mask->size()))},
              stream)
          : cudf::detail::make_empty_host_vector<bitmask_type>(0, stream);

      auto [row_mask_data, row_mask_bitmask] =
        build_data_and_nullmask<bool>(page_mask->mutable_view(),
                                      page_mask_nullmask.data(),
                                      page_indices,
                                      page_row_offsets,
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      stream,
                                      mr);

      auto const row_mask_nullcount = cudf::detail::null_count(
        reinterpret_cast<bitmask_type*>(row_mask_bitmask.data()), 0, total_rows, stream);

      return std::make_unique<column>(cudf::data_type{cudf::type_id::BOOL8},
                                      total_rows,
                                      std::move(row_mask_data),
                                      std::move(row_mask_bitmask),
                                      row_mask_nullcount);
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

  // TODO(#22900): remove this guard once this path maps schema indices per source. It currently
  // reuses one source's schema index for every source, so it is correct only when schemas match.
  CUDF_EXPECTS(schema_idx_maps.empty(),
               "Page index statistics filtering does not support mismatched Parquet schemas yet",
               std::invalid_argument);

  // Total number of rows
  auto const total_rows = total_rows_in_row_groups(row_group_indices);
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

  // Optimization for single column filter: Directly build the row mask from page statistics
  if (num_columns == 1) {
    page_stats_to_row_mask_converter const stats_col{
      static_cast<size_type>(total_rows), per_file_metadata, row_group_indices, true};
    return cudf::type_dispatcher<dispatch_storage_type>(output_dtypes.front(),
                                                        stats_col,
                                                        output_column_schemas.front(),
                                                        output_dtypes.front(),
                                                        filter,
                                                        stream,
                                                        mr);
  }

  // Convert page statistics to a table
  // where min(col[i]) = columns[i*3], max(col[i])=columns[i*3+1], is_null(col[i])=columns[i*3+2]
  // For each column, it contains total number of rows from all row groups.
  page_stats_caster const stats_col{.total_rows           = static_cast<size_type>(total_rows),
                                    .per_file_metadata    = per_file_metadata,
                                    .row_group_indices    = row_group_indices,
                                    .has_is_null_operator = true};

  std::vector<std::unique_ptr<column>> page_stats_columns;
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{num_columns},
    [&](auto col_idx) {
      auto const schema_idx = output_column_schemas[col_idx];
      auto const& dtype     = output_dtypes[col_idx];
      // Only participating columns and comparable types are supported
      if (not stats_columns_mask[col_idx] or
          (cudf::is_compound(dtype) && dtype.id() != cudf::type_id::STRING)) {
        // Placeholder for unsupported types and non-participating columns
        page_stats_columns.push_back(cudf::make_numeric_column(
          data_type{cudf::type_id::BOOL8},
          total_rows,
          rmm::device_buffer{0, stream, cudf::get_current_device_resource_ref()},
          0,
          stream,
          cudf::get_current_device_resource_ref()));
        page_stats_columns.push_back(cudf::make_numeric_column(
          data_type{cudf::type_id::BOOL8},
          total_rows,
          rmm::device_buffer{0, stream, cudf::get_current_device_resource_ref()},
          0,
          stream,
          cudf::get_current_device_resource_ref()));
        page_stats_columns.push_back(cudf::make_numeric_column(
          data_type{cudf::type_id::BOOL8},
          total_rows,
          rmm::device_buffer{0, stream, cudf::get_current_device_resource_ref()},
          0,
          stream,
          cudf::get_current_device_resource_ref()));
        return;
      }
      auto [min_col, max_col, is_null_col] = cudf::type_dispatcher<dispatch_storage_type>(
        dtype, stats_col, schema_idx, dtype, stream, cudf::get_current_device_resource_ref());
      page_stats_columns.push_back(std::move(min_col));
      page_stats_columns.push_back(std::move(max_col));
      CUDF_EXPECTS(is_null_col.has_value(), "is_null host column must be present");
      page_stats_columns.push_back(std::move(is_null_col.value()));
    });

  auto page_stats_table = cudf::table(std::move(page_stats_columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  parquet::detail::stats_expression_converter const stats_expr{filter.get(), output_dtypes, stream};

  // Filter the input table using AST expression and return the (BOOL8) predicate column.
  return cudf::detail::compute_column(
    page_stats_table, stats_expr.get_stats_expr().get(), stream, mr);
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
