/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reader_impl_chunking.hpp"

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/logical.h>

#include <future>
#include <vector>

namespace cudf::io::parquet::detail {

#if defined(PREPROCESS_DEBUG)
void print_pages(cudf::detail::hostdevice_span<PageInfo> pages, rmm::cuda_stream_view stream);
#endif  // PREPROCESS_DEBUG

/**
 * @brief Generate depth remappings for repetition and definition levels.
 *
 * When dealing with columns that contain lists, we must examine incoming
 * repetition and definition level pairs to determine what range of output nesting
 * is indicated when adding new values.  This function generates the mappings of
 * the R/D levels to those start/end bounds
 *
 * @param remap Maps column schema index to the R/D remapping vectors for that column for a
 *              particular input source file
 * @param src_col_schema The source column schema to generate the new mapping for
 * @param mapped_src_col_schema Mapped column schema for src_file_idx'th file
 * @param src_file_idx The input source file index for the column schema
 * @param md File metadata information
 */
void generate_depth_remappings(
  std::map<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>>& remap,
  int const src_col_schema,
  int const mapped_src_col_schema,
  int const src_file_idx,
  aggregate_reader_metadata const& md);

/**
 * @brief Reads compressed page data to device memory.
 *
 * @param sources Dataset sources
 * @param page_data Buffers to hold compressed page data for each chunk
 * @param chunks List of column chunk descriptors
 * @param begin_chunk Index of first column chunk to read
 * @param end_chunk Index after the last column chunk to read
 * @param column_chunk_offsets File offset for all chunks
 * @param chunk_source_map Association between each column chunk and its source
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device buffers
 *
 * @return A future object for reading synchronization
 */
[[nodiscard]] std::future<void> read_column_chunks_async(
  std::vector<std::unique_ptr<datasource>> const& sources,
  cudf::host_span<rmm::device_buffer> page_data,
  cudf::detail::hostdevice_vector<ColumnChunkDesc>& chunks,
  size_t begin_chunk,
  size_t end_chunk,
  std::vector<size_t> const& column_chunk_offsets,
  std::vector<size_type> const& chunk_source_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Return the number of total pages from the given column chunks.
 *
 * @param chunks Host-device span of column chunk descriptors
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return The total number of pages
 */
[[nodiscard]] size_t count_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
                                        rmm::cuda_stream_view stream);

/**
 * @brief Count the total number of pages using page index information.
 */
[[nodiscard]] size_t count_page_headers_with_pgidx(
  cudf::detail::hostdevice_span<ColumnChunkDesc> chunks, rmm::cuda_stream_view stream);

/**
 * @brief Set fields on the pages that can be derived from page indexes.
 *
 * This replaces some preprocessing steps, such as page string size calculation.
 */
void fill_in_page_info(host_span<ColumnChunkDesc> chunks,
                       device_span<PageInfo> pages,
                       rmm::cuda_stream_view stream);

/**
 * @brief Returns a string representation of known encodings
 *
 * @param encoding Given encoding
 * @return String representation of encoding
 */
std::string encoding_to_string(Encoding encoding);

/**
 * @brief Helper function to convert an encoding bitmask to a readable string
 *
 * @param encoding_bitmask Bitmask of found unsupported encodings
 * @returns Human readable string with unsupported encodings
 */
[[nodiscard]] std::string encoding_bitmask_to_str(uint32_t encoding_bitmask);

/**
 * @brief Create a readable string for the user that will list out all unsupported encodings found.
 *
 * @param pages List of page information
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns Human readable string with unsupported encodings
 */
[[nodiscard]] std::string list_unsupported_encodings(device_span<PageInfo const> pages,
                                                     rmm::cuda_stream_view stream);

/**
 * @brief Decode the page information for a given pass.
 *
 * @param pass The struct containing pass information
 * @param unsorted_pages Device span of page information to decode
 * @param has_offset_index Boolean indicating if the offset index is available
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void decode_page_headers(pass_intermediate_data& pass,
                         device_span<PageInfo> unsorted_pages,
                         bool has_offset_index,
                         rmm::cuda_stream_view stream);

/**
 * @brief Decode page information using one exact span per logical indexed page
 *
 * Empty data spans represent masked pages and retain their logical page-index metadata.
 *
 * @param pass Struct containing pass information
 * @param unsorted_pages Device span of page information to decode
 * @param page_data Host span of page data device spans, one per logical indexed page
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void decode_page_headers(pass_intermediate_data& pass,
                         device_span<PageInfo> unsorted_pages,
                         host_span<cudf::device_span<uint8_t const> const> page_data,
                         rmm::cuda_stream_view stream);

/**
 * @brief Check if the column chunk has a string (byte array or FLBA) type
 */
__device__ constexpr inline bool is_string_chunk(ColumnChunkDesc const& chunk)
{
  auto const is_decimal =
    chunk.logical_type.has_value() and chunk.logical_type->type == LogicalType::DECIMAL;
  auto const is_binary =
    chunk.physical_type == Type::BYTE_ARRAY or chunk.physical_type == Type::FIXED_LEN_BYTE_ARRAY;
  return is_binary and not is_decimal;
}

/**
 * @brief Struct to carry info from the page indexes to the device
 */
struct page_index_info {
  int32_t num_rows;
  int32_t chunk_row;
  int32_t num_nulls;
  int32_t num_valids;
  int32_t str_bytes;
  bool has_value_info;
};

/**
 * @brief Functor to copy page_index_info into the PageInfo struct
 */
struct copy_page_info {
  device_span<page_index_info const> page_indexes;
  device_span<PageInfo> pages;

  __device__ constexpr void operator()(size_type idx)
  {
    auto& pg          = pages[idx];
    auto const& pi    = page_indexes[idx];
    pg.num_rows       = pi.num_rows;
    pg.chunk_row      = pi.chunk_row;
    pg.has_value_info = pi.has_value_info;
    pg.start_val      = 0;
    if (pg.has_value_info) {
      pg.num_nulls            = pi.num_nulls;
      pg.num_valids           = pi.num_valids;
      pg.str_bytes_from_index = pi.str_bytes;
      pg.str_bytes            = pi.str_bytes;
      pg.end_val              = pg.num_valids;
    }
  }
};

/**
 * @brief Functor to set the string dictionary index counts for a given data page
 */
struct set_str_dict_index_count {
  device_span<size_t> str_dict_index_count;
  device_span<ColumnChunkDesc const> chunks;

  __device__ constexpr inline void operator()(PageInfo const& page)
  {
    auto const& chunk = chunks[page.chunk_idx];
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0 and chunk.num_dict_pages > 0 and
        is_string_chunk(chunk)) {
      // there is only ever one dictionary page per chunk, so this is safe to do in parallel.
      str_dict_index_count[page.chunk_idx] = page.num_input_values;
    }
  }
};

/**
 * @brief Functor to set the str_dict_index pointer offsets in the ColumnChunkDesc struct
 */
struct set_str_dict_index_ptr {
  string_index_pair* const base;
  device_span<size_t const> str_dict_index_offsets;
  device_span<ColumnChunkDesc> chunks;

  __device__ constexpr inline void operator()(size_t i)
  {
    auto& chunk = chunks[i];
    if (chunk.num_dict_pages > 0 and is_string_chunk(chunk)) {
      chunk.str_dict_index = base + str_dict_index_offsets[i];
    }
  }
};

/**
 * @brief Functor to compute an estimated row count for list pages
 */
struct set_list_row_count_estimate {
  device_span<ColumnChunkDesc const> chunks;

  __device__ constexpr inline void operator()(PageInfo& page)
  {
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) { return; }
    auto const& chunk  = chunks[page.chunk_idx];
    auto const is_list = chunk.max_level[level_type::REPETITION] > 0;
    if (!is_list) { return; }

    // For LIST pages that we have not yet decoded, page.num_rows is not an accurate number.
    // so we instead estimate the number of rows as follows:
    // - each chunk stores an estimated number of bytes per row E
    // - estimate number of rows in a page = page.uncompressed_page_size / E
    //
    // it is not required that this number is accurate. we just want it to be somewhat close so that
    // we get reasonable results as we choose subpass splits.
    //
    // all other columns can use page.num_rows directly as it will be accurate.
    page.num_rows = static_cast<size_t>(static_cast<float>(page.uncompressed_page_size) /
                                        chunk.list_bytes_per_row_est);
  }
};

/**
 * @brief Functor to set the expected row count on the final page for all columns
 */
struct set_final_row_count {
  device_span<PageInfo> pages;
  device_span<ColumnChunkDesc const> chunks;

  __device__ inline void operator()(size_t i)
  {
    auto& page        = pages[i];
    auto const& chunk = chunks[page.chunk_idx];
    // only do this for the last page in each chunk
    if (i < pages.size() - 1 && (pages[i + 1].chunk_idx == page.chunk_idx)) { return; }
    size_t const page_start_row = chunk.start_row + page.chunk_row;
    size_t const chunk_last_row = chunk.start_row + chunk.num_rows;
    // Mark `is_num_rows_adjusted` to signal string decoders that the `num_rows` of this page has
    // been adjusted.
    page.is_num_rows_adjusted = page.num_rows != (chunk_last_row - page_start_row);
    page.num_rows             = chunk_last_row - page_start_row;
  }
};

/**
 * @brief Functor to set the page.num_rows for all pages if page index is available
 */
struct compute_page_num_rows_from_chunk_rows {
  device_span<PageInfo> pages;
  device_span<ColumnChunkDesc const> chunks;

  __device__ constexpr inline void operator()(size_t i)
  {
    auto& page        = pages[i];
    auto const& chunk = chunks[page.chunk_idx];
    if (i < pages.size() - 1 && (pages[i + 1].chunk_idx == page.chunk_idx)) {
      page.num_rows = pages[i + 1].chunk_row - page.chunk_row;
    } else {
      page.num_rows = chunk.num_rows - page.chunk_row;
    }
  }
};

/**
 * @brief Functor to return the column chunk index of a given page
 */
struct get_page_chunk_idx {
  __device__ constexpr inline size_type operator()(PageInfo const& page) { return page.chunk_idx; }
};

/**
 * @brief Functor to return the number of rows in a given page
 */
struct get_page_num_rows {
  __device__ constexpr inline size_type operator()(PageInfo const& page) { return page.num_rows; }
};

/**
 * @brief Struct to hold information about the input columns
 */
struct input_col_info {
  int schema_idx;
  size_type nesting_depth;
};

/**
 * @brief Converts a 1-dimensional index into page, depth and column indices used in
 * allocate_columns to compute columns sizes.
 *
 * The input index will iterate through pages, nesting depth and column indices in that order.
 */
struct reduction_indices {
  size_t const page_idx;
  size_type const depth_idx;
  size_type const col_idx;

  __device__ constexpr inline reduction_indices(size_t index_,
                                                size_type max_depth_,
                                                size_t num_pages_)
    : page_idx(index_ % num_pages_),
      depth_idx((index_ / num_pages_) % max_depth_),
      col_idx(index_ / (max_depth_ * num_pages_))
  {
  }
};

/**
 * @brief Functor that returns the size field of a PageInfo struct for a given depth, keyed by
 * schema
 */
struct get_page_nesting_size {
  input_col_info const* const input_cols;
  size_type const max_depth;
  size_t const num_pages;
  PageInfo const* const pages;

  __device__ inline size_type operator()(size_t index) const
  {
    auto const indices = reduction_indices{index, max_depth, num_pages};

    auto const& page = pages[indices.page_idx];
    if (page.src_col_schema != input_cols[indices.col_idx].schema_idx ||
        page.flags & PAGEINFO_FLAGS_DICTIONARY ||
        indices.depth_idx >= input_cols[indices.col_idx].nesting_depth) {
      return 0;
    }

    return page.nesting[indices.depth_idx].batch_size;
  }
};

/**
 * @brief Functor to compute and return the reduction key for a given page
 */
struct get_reduction_key {
  size_t const num_pages;
  __device__ constexpr inline size_t operator()(size_t index) const { return index / num_pages; }
};

/**
 * @brief Writes to the chunk_row field of the PageInfo struct
 */
struct chunk_row_output_iter {
  PageInfo* p;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  CUDF_HOST_DEVICE constexpr inline chunk_row_output_iter operator+(int i) const { return {p + i}; }

  CUDF_HOST_DEVICE constexpr inline chunk_row_output_iter& operator++()
  {
    p++;
    return *this;
  }

  __device__ constexpr inline reference operator[](int i) { return p[i].chunk_row; }
  __device__ constexpr inline reference operator*() { return p->chunk_row; }
};

/**
 * @brief Writes to the page_start_value field of the PageNestingInfo struct, keyed by schema
 */
struct start_offset_output_iterator {
  PageInfo const* pages;
  size_t cur_index;
  input_col_info const* input_cols;
  size_type max_depth;
  size_t num_pages;
  int empty               = 0;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  CUDF_HOST_DEVICE constexpr inline void operator=(start_offset_output_iterator const& other)
  {
    pages      = other.pages;
    cur_index  = other.cur_index;
    input_cols = other.input_cols;
    max_depth  = other.max_depth;
    num_pages  = other.num_pages;
  }

  CUDF_HOST_DEVICE constexpr inline start_offset_output_iterator operator+(size_t i) const
  {
    return start_offset_output_iterator{pages, cur_index + i, input_cols, max_depth, num_pages};
  }

  CUDF_HOST_DEVICE constexpr inline start_offset_output_iterator& operator++()
  {
    cur_index++;
    return *this;
  }

  __device__ inline reference operator[](size_t i) { return dereference(cur_index + i); }
  __device__ inline reference operator*() { return dereference(cur_index); }

 private:
  __device__ inline reference dereference(size_t index)
  {
    auto const indices = reduction_indices{index, max_depth, num_pages};

    PageInfo const& p = pages[indices.page_idx];
    if (p.src_col_schema != input_cols[indices.col_idx].schema_idx ||
        p.flags & PAGEINFO_FLAGS_DICTIONARY ||
        indices.depth_idx >= input_cols[indices.col_idx].nesting_depth) {
      return empty;
    }
    return p.nesting_decode[indices.depth_idx].page_start_value;
  }
};

/**
 * @brief Functor to return the number of bytes in a string page
 *
 * Note: This functor returns 0 for non-string columns and dictionary pages.
 */
struct page_to_string_size {
  ColumnChunkDesc const* chunks;

  __device__ constexpr inline size_t operator()(PageInfo const& page) const
  {
    auto const chunk = chunks[page.chunk_idx];

    if (not is_string_col(chunk) || (page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { return 0; }
    return page.str_bytes;
  }
};

/**
 * @brief Functor to access and update the str_offset field of the PageInfo struct
 */
struct page_offset_output_iter {
  PageInfo* p;

  using value_type        = size_t;
  using difference_type   = size_t;
  using pointer           = size_t*;
  using reference         = size_t&;
  using iterator_category = thrust::output_device_iterator_tag;

  CUDF_HOST_DEVICE constexpr inline page_offset_output_iter operator+(int i) const
  {
    return {p + i};
  }

  CUDF_HOST_DEVICE constexpr inline page_offset_output_iter& operator++()
  {
    p++;
    return *this;
  }

  __device__ constexpr inline reference operator[](int i) { return p[i].str_offset; }
  __device__ constexpr inline reference operator*() { return p->str_offset; }
};

/**
 * @brief Functor to update chunk_row field from pass page to subpass page
 */
struct update_subpass_chunk_row {
  device_span<PageInfo> pass_pages;
  device_span<PageInfo> subpass_pages;
  device_span<size_t> page_src_index;

  __device__ constexpr inline void operator()(size_t i)
  {
    subpass_pages[i].chunk_row = pass_pages[page_src_index[i]].chunk_row;
  }
};

/**
 * @brief Functor to update num_rows field from pass page to subpass page
 */
struct update_pass_num_rows {
  device_span<PageInfo> pass_pages;
  device_span<PageInfo> subpass_pages;
  device_span<size_t> page_src_index;

  __device__ constexpr inline void operator()(size_t i)
  {
    pass_pages[page_src_index[i]].num_rows = subpass_pages[i].num_rows;
  }
};

}  // namespace cudf::io::parquet::detail
