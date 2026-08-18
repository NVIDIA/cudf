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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/logger.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
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
#include <numeric>
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
   * columns containing page-level min, max and (optional) is_null statistics for a column
   *
   * @param schema_idx Column schema index
   * @param dtype Column data type
   * @param stream CUDA stream
   * @return A tuple of page row offsets, column chunk page offsets, and host columns containing
   * page-level min, max and (optional) is_null statistics
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

    // Create host columns with page-level min, max and optionally is_null statistics
    host_column<T> min(total_pages, stream);
    host_column<T> max(total_pages, stream);
    std::optional<host_column<bool>> is_null;
    if (has_is_null_operator) { is_null = host_column<bool>(total_pages, stream); }

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
                  is_null->val[column_page_idx] = true;
                  return;
                }
                // Check if the page doesn't have a null count
                if (not column_index.null_counts.has_value()) {
                  is_null->set_index(column_page_idx, std::nullopt, {});
                  return;
                }
                // Use the null count to determine if the page is completely null
                auto const page_row_count =
                  page_row_offsets[column_page_idx + 1] - page_row_offsets[column_page_idx];
                auto const& null_count = column_index.null_counts.value()[page_idx];
                if (null_count == page_row_count) {
                  is_null->val[column_page_idx] = false;
                } else if (null_count > 0 and null_count < page_row_count) {
                  is_null->set_index(column_page_idx, std::nullopt, {});
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
                      std::move(is_null)};
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
      auto constexpr num_columns = 1;
      parquet::detail::stats_expression_converter const stats_expr{
        filter.get(), num_columns, has_is_null_operator, stream};

      // Filter the input table using AST expression and return the (BOOL8) predicate column.
      auto const page_mask = cudf::detail::compute_column(page_stats_table,
                                                          stats_expr.get_stats_expr().get(),
                                                          stream,
                                                          cudf::get_current_device_resource_ref());

      auto const page_indices = compute_page_indices_async(
        page_row_offsets, total_rows, stream, cudf::get_current_device_resource_ref());

      auto const page_mask_nullmask =
        page_mask->null_count()
          ? cudf::detail::make_host_vector_async(
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

/*
 * @brief Functor to build a Fenwick tree level from the previous level data
 *
 * @param tree_level_ptrs Pointers to the start of Fenwick tree level data
 * @param prev_level Previous tree level
 * @param prev_level_size Size of the previous tree level
 * @param current_level_size Size of the current tree level
 */
struct build_fenwick_tree_level_functor {
  bool** tree_level_ptrs;
  cudf::size_type prev_level;
  cudf::size_type prev_level_size;
  cudf::size_type current_level_size;

  /**
   * @brief Builds the next Fenwick tree level from the current level data
   * by ORing two elements at the current level.
   *
   * elem_current_level[idx] = elem_prev_level[idx * 2] OR elem_prev_level[idx * 2 + 1];
   *
   * @param current_level_idx Current tree level element index
   */
  __device__ void operator()(cudf::size_type current_level_idx) const noexcept
  {
    auto const prev_level_ptr = tree_level_ptrs[prev_level];
    auto current_level_ptr    = tree_level_ptrs[prev_level + 1];

    // Handle the odd-sized remaining element if prev_level_size is odd
    if (prev_level_size % 2 and current_level_idx == current_level_size - 1) {
      current_level_ptr[current_level_idx] = prev_level_ptr[prev_level_size - 1];
    } else {
      current_level_ptr[current_level_idx] =
        prev_level_ptr[(current_level_idx * 2)] or prev_level_ptr[(current_level_idx * 2) + 1];
    }
  }
};

/**
 * @brief Functor to binary search a `true` value in the Fenwick tree in range [start, end)
 *
 * @param tree_level_ptrs Pointers to the start of Fenwick tree level data
 * @param page_offsets Pointer to page offsets describing each search range i as [page_offsets[i],
 * page_offsets[i+1))
 * @param num_ranges Number of search ranges
 */
struct search_fenwick_tree_functor {
  bool** tree_level_ptrs;
  cudf::size_type const* page_offsets;
  cudf::size_type num_ranges;

  /**
   * @brief Enum class to represent which range boundary we are currently processing
   */
  enum class boundary : uint8_t {
    START = 0,
    END   = 1,
  };

  /**
   * @brief Checks if a value is a power of two
   *
   * @param value Value to check
   * @return Boolean indicating if the value is a power of two
   */
  __device__ bool inline constexpr is_power_of_two(cudf::size_type value) const noexcept
  {
    return (value & (value - 1)) == 0;
  }

  /**
   * @brief Finds the smallest power of two in the range [start, end). If no power of two is
   * found, returns a zero.
   *
   * @param start Range start
   * @param end Range end
   * @return Largest power of two in the range [start, end) or a zero if no power of two is found
   */
  __device__ cudf::size_type inline constexpr smallest_power_of_two_in_range(
    cudf::size_type start, cudf::size_type end) const noexcept
  {
    start--;
    start |= start >> 1;
    start |= start >> 2;
    start |= start >> 4;
    start |= start >> 8;
    start |= start >> 16;
    auto const result = start + 1;
    return result < end ? result : 0;
  }

  /**
   * @brief Finds the largest power of two in the range (start, end]. If no power of two is found,
   * returns a zero.
   *
   * @param start Range start
   * @param end Range end
   * @return Largest power of two in the range (start, end] or a zero if no power of two is found
   */
  __device__ size_type inline constexpr largest_power_of_two_in_range(size_type start,
                                                                      size_type end) const noexcept
  {
    auto constexpr nbits = cudf::detail::size_in_bits<size_type>() - 1;
    auto const result    = size_type{1} << (nbits - cuda::std::countl_zero<uint32_t>(end));
    return result > start ? result : 0;
  }

  /**
   * @brief Aligns a range boundary to the next power-of-two block
   *
   * @tparam Boundary Current boundary type (START or END)
   * @param start Range start
   * @param end Range end
   * @return A pair of the tree level and block size
   */
  template <boundary Boundary>
  __device__ auto inline constexpr align_range_boundary(cudf::size_type start,
                                                        cudf::size_type end) const noexcept
  {
    if constexpr (Boundary == boundary::START) {
      if (start == 0 or is_power_of_two(start)) {
        auto const block_size =
          cuda::std::max<size_type>(start & -start, largest_power_of_two_in_range(start, end));
        auto const tree_level = cuda::std::countr_zero<uint32_t>(block_size);
        return cuda::std::pair{tree_level, block_size};
      } else {
        auto const tree_level = cuda::std::countr_zero<uint32_t>(start);
        return cuda::std::pair{tree_level, size_type{1} << tree_level};
      }
    } else {
      auto block_size = end & -end;
      if (start > 0 and is_power_of_two(end)) {
        auto const next_alignment = cuda::std::max(smallest_power_of_two_in_range(start, end),
                                                   largest_power_of_two_in_range(0, end - start));
        block_size                = end - next_alignment;
      }
      return cuda::std::pair{cuda::std::countr_zero<uint32_t>(block_size), block_size};
    }
  }

  /**
   * @brief Queries the Fenwick tree for the given boundary position, tree level and block size
   *
   * @tparam Boundary Current boundary type (START or END)
   * @param boundary_pos Current boundary position
   * @param tree_level Corresponding tree level to query
   * @param block_size Alignment block size of the current boundary
   * @return Boolean indicating if a `true` value is found in the fenwick tree
   */
  template <boundary Boundary>
  __device__ bool inline constexpr query_fenwick_tree(cudf::size_type boundary_pos,
                                                      cudf::size_type tree_level,
                                                      cudf::size_type block_size) const noexcept
  {
    if constexpr (Boundary == boundary::START) {
      auto const mask_index = boundary_pos >> tree_level;
      return tree_level_ptrs[tree_level][mask_index];
    } else {
      auto const mask_index = (boundary_pos - block_size) >> tree_level;
      return tree_level_ptrs[tree_level][mask_index];
    }
  }

  /**
   * @brief Searches the Fenwick tree to find a `true` value in range [start, end)
   *
   * Algorithm: While `start` < `end`, align `start` UP and `end` DOWN to the next power-of-two
   * searchable tree block. For the two aligned blocks, query the fenwick tree at corresponding
   * levels for a `true` value (larger block first). If found, return. Else, move the boundaries
   * to their alignments.
   *
   * @param range_idx Index of the range to search
   * @return Boolean indicating if a `true` value is found in the range
   */
  __device__ bool operator()(cudf::size_type range_idx) const noexcept
  {
    // Retrieve start and end for the current range [start, end)
    size_type start = page_offsets[range_idx];
    size_type end   = page_offsets[range_idx + 1];

    // Return early if the range is empty or invalid
    if (start >= end or range_idx >= num_ranges) { return false; }

    // Binary search decomposition loop
    while (start < end) {
      // Find the largest power-of-two block that aligns `start` up
      auto const [start_tree_level, start_block_size] =
        align_range_boundary<boundary::START>(start, end);

      // Find the largest power-of-two block that aligns `end` down
      auto const [end_tree_level, end_block_size] = align_range_boundary<boundary::END>(start, end);

      // Check the larger block first to minimize the number of queries
      if (start_block_size >= end_block_size) {
        // Check the `start` side alignment block first
        if (start + start_block_size <= end) {
          if (query_fenwick_tree<boundary::START>(start, start_tree_level, start_block_size)) {
            return true;
          }
          start += start_block_size;
        }
        // Check the `end` side alignment block if it's still in range
        if (end - end_block_size >= start) {
          if (query_fenwick_tree<boundary::END>(end, end_tree_level, end_block_size)) {
            return true;
          }
          end -= end_block_size;
        }
      } else {
        // Check the `end` side alignment block first
        if (end - end_block_size >= start) {
          if (query_fenwick_tree<boundary::END>(end, end_tree_level, end_block_size)) {
            return true;
          }
          end -= end_block_size;
        }
        // Check the `start` side alignment block if it's still in range
        if (start + start_block_size <= end) {
          if (query_fenwick_tree<boundary::START>(start, start_tree_level, start_block_size)) {
            return true;
          }
          start += start_block_size;
        }
      }
    }
    return false;
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
  auto const [stats_columns_mask, has_is_null_operator] =
    parquet::detail::stats_columns_collector{filter.get(),
                                             static_cast<size_type>(output_dtypes.size())}
      .get_stats_columns_mask();

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
    page_stats_to_row_mask_converter const stats_col{static_cast<size_type>(total_rows),
                                                     per_file_metadata,
                                                     row_group_indices,
                                                     has_is_null_operator};
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
                                    .has_is_null_operator = has_is_null_operator};

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
        if (has_is_null_operator) {
          page_stats_columns.push_back(cudf::make_numeric_column(
            data_type{cudf::type_id::BOOL8},
            total_rows,
            rmm::device_buffer{0, stream, cudf::get_current_device_resource_ref()},
            0,
            stream,
            cudf::get_current_device_resource_ref()));
        }
        return;
      }
      auto [min_col, max_col, is_null_col] = cudf::type_dispatcher<dispatch_storage_type>(
        dtype, stats_col, schema_idx, dtype, stream, cudf::get_current_device_resource_ref());
      page_stats_columns.push_back(std::move(min_col));
      page_stats_columns.push_back(std::move(max_col));
      if (has_is_null_operator) {
        CUDF_EXPECTS(is_null_col.has_value(), "is_null host column must be present");
        page_stats_columns.push_back(std::move(is_null_col.value()));
      }
    });

  auto page_stats_table = cudf::table(std::move(page_stats_columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  parquet::detail::stats_expression_converter const stats_expr{
    filter.get(), static_cast<size_type>(output_dtypes.size()), has_is_null_operator, stream};

  // Filter the input table using AST expression and return the (BOOL8) predicate column.
  return cudf::detail::compute_column(
    page_stats_table, stats_expr.get_stats_expr().get(), stream, mr);
}

template <typename ColumnView>
thrust::host_vector<bool> aggregate_reader_metadata::compute_data_page_mask(
  ColumnView const& row_mask,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<input_column_info const> input_columns,
  cudf::size_type row_mask_offset,
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
    std::cmp_less_equal(row_mask_offset + total_rows, row_mask.size()),
    "Encountered a mismatch in number of rows in the row group pass and the row mask size",
    std::overflow_error);

  // Return an empty vector if all rows are invalid or all rows are required
  if (std::cmp_equal(row_mask.null_count(row_mask_offset, row_mask_offset + total_rows, stream),
                     total_rows) or
      cudf::detail::all_of(row_mask.template begin<bool>() + row_mask_offset,
                           row_mask.template begin<bool>() + row_mask_offset + total_rows,
                           cuda::std::identity{},
                           stream)) {
    return thrust::host_vector<bool>(0);
  }

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
      "Encountered missing Parquet offset index for one or more output columns. Skipping page "
      "pruning.");
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

  // Make sure all row_mask elements contain valid values even if they are nulls
  if constexpr (cuda::std::is_same_v<ColumnView, cudf::mutable_column_view>) {
    if (row_mask.nullable() and row_mask.null_count() > 0) {
      thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<cudf::size_type>(row_mask_offset),
                       cuda::counting_iterator<cudf::size_type>(row_mask_offset + total_rows),
                       [row_mask  = row_mask.template begin<bool>(),
                        null_mask = row_mask.null_mask()] __device__(auto const row_idx) {
                         if (not bit_is_set(null_mask, row_idx)) { row_mask[row_idx] = true; }
                       });
    }
  } else {
    CUDF_EXPECTS(not row_mask.nullable() or row_mask.null_count() == 0,
                 "Row mask must not contain nulls for payload columns");
  }

  auto const mr = cudf::get_current_device_resource_ref();

  // Compute fenwick tree level offsets and total size (level 1 and higher)
  auto const tree_level_offsets = compute_fenwick_tree_level_offsets(total_rows, max_page_size);
  auto const num_levels         = static_cast<cudf::size_type>(tree_level_offsets.size());
  // Buffer to store Fenwick tree levels (level 1 and higher) data
  auto tree_levels_data = rmm::device_uvector<bool>(tree_level_offsets.back(), stream, mr);

  // Pointers to each Fenwick tree level data
  auto host_tree_level_ptrs = cudf::detail::make_pinned_vector_async<bool*>(num_levels, stream);
  // Zeroth level is just the row mask itself
  host_tree_level_ptrs[0] = const_cast<bool*>(row_mask.template begin<bool>()) + row_mask_offset;
  std::for_each(cuda::counting_iterator<cudf::size_type>{1},
                cuda::counting_iterator{num_levels},
                [&](auto const level_idx) {
                  host_tree_level_ptrs[level_idx] =
                    tree_levels_data.data() + tree_level_offsets[level_idx - 1];
                });

  auto fenwick_tree_level_ptrs =
    cudf::detail::make_device_uvector_async(host_tree_level_ptrs, stream, mr);

  // Build Fenwick tree levels (zeroth level is just the row mask itself)
  auto prev_level_size = static_cast<cudf::size_type>(total_rows);
  std::for_each(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{num_levels - 1},
    [&](auto const prev_level) {
      auto const current_level_size = cudf::util::div_rounding_up_safe(prev_level_size, 2);
      thrust::for_each(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        cuda::counting_iterator<cudf::size_type>{0},
        cuda::counting_iterator{current_level_size},
        build_fenwick_tree_level_functor{
          fenwick_tree_level_ptrs.data(), prev_level, prev_level_size, current_level_size});
      prev_level_size = current_level_size;
    });

  //  Search the Fenwick tree to see if there's a surviving row in each page's row range
  auto const num_ranges = static_cast<cudf::size_type>(page_row_offsets.size() - 1);
  rmm::device_uvector<bool> device_data_page_mask(num_ranges, stream, mr);
  // Use a pinned bounce buffer to avoid pageable h2d copy
  auto pinned_page_offsets = cudf::detail::make_pinned_vector(
    cudf::host_span<cudf::size_type const>{page_row_offsets}, stream);
  auto page_offsets = cudf::detail::make_device_uvector_async(pinned_page_offsets, stream, mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{num_ranges},
    device_data_page_mask.begin(),
    search_fenwick_tree_functor{fenwick_tree_level_ptrs.data(), page_offsets.data(), num_ranges});

  //  Copy over search results to host
  auto host_results      = cudf::detail::make_pinned_vector_async(device_data_page_mask, stream);
  auto const total_pages = pinned_page_offsets.size() - num_columns;
  auto data_page_mask    = thrust::host_vector<bool>(total_pages);
  auto host_results_iter = host_results.begin();
  stream.sync();

  // Discard results for invalid ranges. i.e. ranges starting at the last page of a column and
  // ending at the first page of the next column
  auto num_pages_inserted = 0;
  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{num_columns},
                [&](auto col_idx) {
                  auto const col_num_pages =
                    col_page_offsets[col_idx + 1] - col_page_offsets[col_idx] - 1;
                  data_page_mask.insert(data_page_mask.begin() + num_pages_inserted,
                                        host_results_iter,
                                        host_results_iter + col_num_pages);
                  host_results_iter += col_num_pages + 1;
                  num_pages_inserted += col_num_pages;
                });
  return data_page_mask;
}

// Instantiate the templates with ColumnView as cudf::column_view and cudf::mutable_column_view
template thrust::host_vector<bool> aggregate_reader_metadata::compute_data_page_mask<
  cudf::column_view>(cudf::column_view const& row_mask,
                     std::span<std::vector<size_type> const> row_group_indices,
                     std::span<input_column_info const> input_columns,
                     cudf::size_type row_mask_offset,
                     cuda::stream_ref stream) const;

template thrust::host_vector<bool> aggregate_reader_metadata::compute_data_page_mask<
  cudf::mutable_column_view>(cudf::mutable_column_view const& row_mask,
                             std::span<std::vector<size_type> const> row_group_indices,
                             std::span<input_column_info const> input_columns,
                             cudf::size_type row_mask_offset,
                             cuda::stream_ref stream) const;

}  // namespace cudf::io::parquet::experimental::detail
