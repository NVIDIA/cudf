/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compact_protocol_reader.hpp"
#include "cudf/types.hpp"
#include "io/comp/decompression.hpp"
#include "reader_impl_chunking.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cuda/functional>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>

namespace cudf::io::parquet::detail {

// Percentage of the total available input read limit that should be reserved for compressed
// data vs uncompressed data.
constexpr float input_limit_compression_reserve = 0.3f;

using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;
using cudf::io::detail::decompression_info;

// Forward declarations
struct cumulative_page_info;
struct page_span;

#if defined(CHUNKING_DEBUG)
void print_cumulative_page_info(device_span<PageInfo const> d_pages,
                                device_span<ColumnChunkDesc const> d_chunks,
                                device_span<cumulative_page_info const> d_c_info,
                                cuda::stream_ref stream);
#endif  // CHUNKING_DEBUG

/**
 * @brief Return the required number of bits to store a value
 */
template <typename T = uint8_t>
[[nodiscard]] constexpr inline T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

/**
 * @brief Returns the cudf compression type and whether it is supported by the parquet writer
 */
CUDF_HOST_DEVICE cuda::std::pair<compression_type, bool> parquet_compression_support(
  Compression compression);

/**
 * @brief Returns the string name of the Parquet compression type
 */
[[nodiscard]] std::string parquet_compression_name(Compression compression);

/**
 * @brief Converts a cudf Compression enum to a parquet compression type
 */
[[nodiscard]] compression_type from_parquet_compression(Compression compression);

/**
 * @brief Find the first entry in the aggreggated_info that corresponds to the specified row
 */
size_t find_start_index(cudf::host_span<cumulative_page_info const> aggregated_info,
                        size_t start_row);

/**
 * @brief Given a current position and row index, find the next split based on the
 * specified size limit
 *
 * @returns The inclusive index within `sizes` where the next split should happen
 */
int64_t find_next_split(int64_t cur_pos,
                        size_t cur_row_index,
                        size_t cur_cumulative_size,
                        cudf::host_span<cumulative_page_info const> sizes,
                        size_t size_limit,
                        size_t min_row_count);

/**
 * @brief Converts cuDF units to Parquet units
 *
 * @return A tuple of Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, cuda::std::optional<LogicalType>> conversion_info(
  type_id column_type_id,
  type_id timestamp_type_id,
  Type physical,
  cuda::std::optional<LogicalType> logical_type);

/**
 * @brief For a set of cumulative_page_info data, adjust the size_bytes field
 * such that it reflects the worst case for all pages that span the same rows
 *
 * By doing this, we can now look at row X and know the total
 * byte cost for all pages that span row X, not just the cost up to row X itself.
 *
 * This function is asynchronous. Call stream.sync() before using the
 * results.
 */
std::pair<rmm::device_uvector<cumulative_page_info>, rmm::device_uvector<int32_t>>
adjust_cumulative_sizes(device_span<cumulative_page_info const> c_info,
                        device_span<PageInfo const> pages,
                        cuda::stream_ref stream);

/**
 * @brief Computes the next subpass within the current pass
 *
 * A subpass is a subset of the pages within the parent pass that is decompressed
 * as a batch and decoded.  Subpasses are the level at which we control memory intermediate
 * memory usage. A pass consists of >= 1 subpass.  We cannot compute all subpasses in one
 * shot because we do not know how many rows we actually have in the pages of list columns.
 * So we have to make an educated guess that fits within the memory limits, and then adjust
 * for subsequent subpasses when we see how many rows we actually receive.
 *
 * @param c_info The cumulative page size information (row count and byte size) per column
 * @param pages All of the pages in the pass
 * @param chunks All of the chunks in the pass
 * @param page_offsets Offsets into the pages array representing the first page for each column
 * @param start_row The row to start the subpass at
 * @param size_limit The size limit in bytes of the subpass
 * @param num_columns The number of columns
 * @param is_first_subpass Boolean indicating if this is the first subpass
 * @param has_offset_index Boolean indicating if offset indexes are available
 * @param stream The stream to execute cuda operations on
 * @returns A tuple containing a vector of page_span structs indicating the page indices to include
 * for each column to be processed, the total number of pages over all columns, and the total
 * expected memory usage (including scratch space)
 *
 */
std::tuple<rmm::device_uvector<page_span>, size_t, size_t> compute_next_subpass(
  device_span<cumulative_page_info const> c_info,
  device_span<PageInfo const> pages,
  device_span<ColumnChunkDesc const> chunks,
  device_span<size_type const> page_offsets,
  size_t start_row,
  size_t size_limit,
  size_t num_columns,
  bool is_first_subpass,
  bool has_offset_index,
  cuda::stream_ref stream);

/**
 * @brief Computes the page splits for a given set of pages based on row count and size limit
 *
 * This function computes the page splits based on the cumulative page information, the number of
 * rows to skip, the total number of rows, and the size limit. It returns a vector of row_range
 * structs indicating the start and end rows for each split.
 *
 * @param c_info The cumulative page size information (row count and byte size) per column
 * @param pages All of the pages in the pass
 * @param skip_rows The number of rows to skip before starting the split computation
 * @param num_rows The total number of rows to consider for splitting
 * @param size_limit The size limit in bytes for each split
 * @param stream The stream to execute cuda operations on
 * @returns A vector of row_range structs indicating the start and end rows for each split
 */
std::vector<row_range> compute_page_splits_by_row(device_span<cumulative_page_info const> c_info,
                                                  device_span<PageInfo const> pages,
                                                  size_t skip_rows,
                                                  size_t num_rows,
                                                  size_t size_limit,
                                                  cuda::stream_ref stream);

/**
 * @brief Decompresses a mix of dictionary and non-dictionary pages from a set of column chunks
 *
 * To avoid multiple calls to the decompression kernel, we batch pages by codec type, where the
 * batch can include both dictionary and non-dictionary pages. This allows us to decompress all
 * pages of a given codec type in one go.
 *
 * @param chunks List of column chunk descriptors
 * @param pass_pages List of page information for the pass
 * @param subpass_pages List of page information for the subpass
 * @param subpass_page_mask Boolean page mask indicating which subpass pages to decompress. Empty
 * span indicates all pages should be decompressed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate buffers
 *
 * @return A pair of device buffers containing the decompressed data for dictionary and
 * non-dictionary pages, respectively.
 */
[[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> decompress_page_data(
  host_span<ColumnChunkDesc const> chunks,
  host_span<PageInfo> pass_pages,
  host_span<PageInfo> subpass_pages,
  host_span<bool const> subpass_page_mask,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Detect malformed parquet input data
 *
 * We have seen cases where parquet files can be oddly malformed. This function specifically
 * detects one case in particular:
 *
 * - When you have a file containing N rows
 * - For some reason, the sum total of the number of rows over all pages for a given column
 *   is != N
 *
 * @param pages All pages to be decoded
 * @param chunks Chunk data
 * @param expected_row_count Expected row count, if applicable
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void detect_malformed_pages(device_span<PageInfo const> pages,
                            device_span<ColumnChunkDesc const> chunks,
                            std::optional<size_t> expected_row_count,
                            cuda::stream_ref stream);

/**
 * @brief Computes the per-page scratch space required for decompression.
 */
rmm::device_uvector<size_t> compute_decompression_scratch_sizes(
  device_span<ColumnChunkDesc const> chunks,
  device_span<PageInfo const> pages,
  cuda::stream_ref stream);

/**
 * @brief Computes the per-page buffer sizes required for string offsets.
 *
 * For non-dictionary, non-FLBA string columns, this computes the size needed
 * to store string offsets (uint32_t per value) for each page.
 *
 * @param chunks List of column chunk descriptors
 * @param pages List of page information
 * @param skip_rows Starting row for the pass
 * @param num_rows Number of rows to read in the pass
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for allocating the returned device_uvector
 * @returns A vector of size_t values, one for each page, indicating the size needed for string
 * offsets
 */
rmm::device_uvector<size_t> compute_string_offset_sizes(device_span<ColumnChunkDesc const> chunks,
                                                        device_span<PageInfo const> pages,
                                                        size_t skip_rows,
                                                        size_t num_rows,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr);

/**
 * @brief Computes the per-page buffer sizes required for level decode preprocessing.
 *
 * This computes the memory needed to store decoded definition and repetition levels
 * during the preprocessing step. This memory is used by preprocess_levels_kernel().
 *
 * @param chunks List of column chunk descriptors
 * @param pages List of page information
 * @param level_type_size Size in bytes for each level value
 * @param skip_rows Starting row for the pass
 * @param num_rows Number of rows to read in the pass
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for allocating the returned device_uvector
 * @returns A vector of size_t values, one for each page, indicating the size needed for level
 * decode preprocessing
 */
rmm::device_uvector<size_t> compute_level_decode_sizes(device_span<ColumnChunkDesc const> chunks,
                                                       device_span<PageInfo const> pages,
                                                       int level_type_size,
                                                       size_t skip_rows,
                                                       size_t num_rows,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr);

/**
 * @brief Computes the level decode buffer sizes for a single page.
 *
 * This is a common implementation used by both GPU (chunking estimation) and CPU
 * (actual allocation) code paths.
 *
 * @param page The page to compute sizes for
 * @param chunk The column chunk descriptor for this page
 * @param level_type_size Size in bytes for each level value
 * @param skip_rows Starting row for the pass
 * @param num_rows Number of rows to read in the pass
 * @param def_level_size[out] Size needed for definition levels
 * @param rep_level_size[out] Size needed for repetition levels
 */
template <typename PageType, typename ChunkType>
CUDF_HOST_DEVICE inline void compute_page_level_decode_sizes(PageType const& page,
                                                             ChunkType const& chunk,
                                                             int level_type_size,
                                                             size_t skip_rows,
                                                             size_t num_rows,
                                                             size_t& def_level_size,
                                                             size_t& rep_level_size);

/**
 * @brief Add the cost of decompression codec scratch space to the per-page cumulative
 * size information
 */
void include_scratch_size(device_span<size_t const> pages,
                          device_span<cumulative_page_info> c_info,
                          cuda::stream_ref stream);

/**
 * @brief Struct to store split information
 */
struct split_info {
  row_range rows;
  int64_t split_pos;
};

/**
 * @brief
 */
struct cumulative_page_info {
  size_t end_row_index;  // end row index (start_row + num_rows for the corresponding page)
  size_t size_bytes;     // cumulative size in bytes
  cudf::size_type key;   // schema index
};

/**
 * @brief Struct to store the start and end of a page's row span
 */
struct page_span {
  size_t start;
  size_t end;
};

/**
 * @brief Functor which returns the size of a span
 */
struct get_span_size {
  CUDF_HOST_DEVICE inline size_t operator()(page_span const& s) const { return s.end - s.start; }
};

/**
 * @brief Functor which returns the size of a span in an array of spans, handling out-of-bounds
 * indices
 */
struct get_span_size_by_index {
  cudf::device_span<page_span const> page_indices;
  __device__ inline size_t operator()(size_t i) const
  {
    return i >= page_indices.size() ? 0 : page_indices[i].end - page_indices[i].start;
  }
};

/**
 * @brief Functor which returns the span of page indices for a given column index
 */
struct get_page_span_by_column {
  cudf::device_span<size_type const> page_offsets;
  __device__ inline page_span operator()(size_t i) const
  {
    return {static_cast<size_t>(page_offsets[i]), static_cast<size_t>(page_offsets[i + 1])};
  }
};

/**
 * @brief Functor which returns the end row index for a cumulative_page_info
 */
struct get_page_end_row_index {
  device_span<cumulative_page_info const> c_info;
  __device__ inline size_t operator()(size_t i) const { return c_info[i].end_row_index; }
};

/**
 * @brief Functor which reduces two cumulative_page_info structs of the same key
 */
struct cumulative_page_sum {
  __device__ inline cumulative_page_info operator()(cumulative_page_info const& a,
                                                    cumulative_page_info const& b) const
  {
    return cumulative_page_info{0, a.size_bytes + b.size_bytes, a.key};
  }
};

/**
 * @brief Functor which returns the compressed data size for a chunk
 */
struct get_chunk_compressed_size {
  __device__ inline size_t operator()(ColumnChunkDesc const& chunk) const
  {
    return chunk.compressed_size;
  }
};

/**
 * @brief Functor which returns the number of rows in a page for a flat column
 */
struct flat_column_num_rows {
  ColumnChunkDesc const* chunks;
  __device__ inline size_type operator()(PageInfo const& page) const
  {
    // Ignore dictionary pages and pages belonging to any column containing repetition (lists)
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) ||
        (chunks[page.chunk_idx].max_level[level_type::REPETITION] > 0)) {
      return 0;
    }
    return page.num_rows;
  }
};

/**
 * @brief Stores basic information about pages compressed with a specific codec
 */
struct codec_stats {
  Compression compression_type  = Compression::UNCOMPRESSED;
  size_t num_pages              = 0;
  int32_t max_decompressed_size = 0;
  size_t total_decomp_size      = 0;

  enum class page_selection { DICT_PAGES, NON_DICT_PAGES };

  void add_pages(host_span<ColumnChunkDesc const> chunks,
                 host_span<PageInfo> pages,
                 page_selection selection,
                 host_span<bool const> page_mask);
};

/**
 * @brief Functor which retrieves per-page decompression information.
 */
struct get_decomp_info {
  device_span<ColumnChunkDesc const> chunks;
  __device__ inline decompression_info operator()(PageInfo const& p) const
  {
    return {parquet_compression_support(chunks[p.chunk_idx].codec).first,
            1,
            static_cast<size_t>(p.uncompressed_page_size),
            static_cast<size_t>(p.uncompressed_page_size)};
  }
};

/**
 * @brief Functor which accumulates per-page decompression information.
 */
struct decomp_sum {
  __device__ inline decompression_info operator()(decompression_info const& a,
                                                  decompression_info const& b) const
  {
    return {a.type,
            a.num_pages + b.num_pages,
            cuda::std::max(a.max_page_decompressed_size, b.max_page_decompressed_size),
            a.total_decompressed_size + b.total_decompressed_size};
  }
};

/**
 * @brief Functor which check if row count is not zero
 */
struct row_counts_nonzero {
  __device__ inline bool operator()(size_type count) const { return count > 0; }
};

/**
 * @brief Functor which checks if row count is non zero and different from expected value
 */
struct row_counts_different {
  size_type const expected;
  __device__ inline bool operator()(size_type count) const
  {
    return (count != 0) && (count != expected);
  }
};

/**
 * @brief Functor which computes the total data size for a given type of cudf column
 *
 * In the case of strings, the return size does not include the chars themselves. That
 * information is tracked separately (see PageInfo::str_bytes).
 */
struct row_size_functor {
  __device__ inline size_t validity_size(size_t num_rows, bool nullable)
  {
    // TODO: Use a util from `null_mask.cuh` instead of this calculation when available
    return nullable ? cudf::util::div_rounding_up_safe<size_type>(
                        num_rows, cudf::detail::size_in_bits<bitmask_type>()) *
                        sizeof(bitmask_type)
                    : 0;
  }

  template <typename T>
  __device__ inline size_t operator()(size_t num_rows, bool nullable)
  {
    auto const element_size = sizeof(device_storage_type_t<T>);
    return (element_size * num_rows) + validity_size(num_rows, nullable);
  }
};

template <>
__device__ inline size_t row_size_functor::operator()<list_view>(size_t num_rows, bool nullable)
{
  auto const offset_size = sizeof(size_type);
  // NOTE: Adding the + 1 offset here isn't strictly correct.  There will only be 1 extra offset
  // for the entire column, whereas this is adding an extra offset per page.  So we will get a
  // small over-estimate of the real size of the order :  # of pages * 4 bytes. It seems better
  // to overestimate size somewhat than to underestimate it and potentially generate chunks
  // that are too large.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

template <>
__device__ inline size_t row_size_functor::operator()<struct_view>(size_t num_rows, bool nullable)
{
  return validity_size(num_rows, nullable);
}

template <>
__device__ inline size_t row_size_functor::operator()<string_view>(size_t num_rows, bool nullable)
{
  // only returns the size of offsets and validity. the size of the actual string chars
  // is tracked separately.
  auto const offset_size = sizeof(size_type);
  // see note about offsets in the list_view template.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

/**
 * @brief Functor which computes the total output cudf data size for all of
 * the data in this page
 *
 * Sums across all nesting levels.
 */
struct get_page_output_size {
  __device__ cumulative_page_info operator()(PageInfo const& page) const
  {
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) {
      return cumulative_page_info{0, 0, page.src_col_schema};
    }

    // total nested size, not counting string data
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_t>([page] __device__(size_type i) {
        auto const& pni = page.nesting[i];
        return cudf::type_dispatcher(
          data_type{pni.type}, row_size_functor{}, pni.size, pni.nullable);
      }));
    return {
      0,
      thrust::reduce(thrust::seq, iter, iter + page.num_output_nesting_levels) + page.str_bytes_all,
      page.src_col_schema};
  }
};

/**
 * @brief Functor which sets the (uncompressed) size of a page
 */
struct get_page_input_size {
  __device__ inline cumulative_page_info operator()(PageInfo const& page) const
  {
    // we treat dictionary page sizes as 0 for subpasses because we have already paid the price for
    // them at the pass level.
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) { return {0, 0, page.src_col_schema}; }
    return {0, static_cast<size_t>(page.uncompressed_page_size), page.src_col_schema};
  }
};

/**
 * @brief Functor which sets the absolute row index of a page in a cumulative_page_info struct
 */
struct set_row_index {
  device_span<ColumnChunkDesc const> chunks;
  device_span<PageInfo const> pages;
  device_span<cumulative_page_info> c_info;
  size_t max_row;

  __device__ inline void operator()(size_t i)
  {
    auto const& page            = pages[i];
    auto const& chunk           = chunks[page.chunk_idx];
    size_t const page_end_row   = chunk.start_row + page.chunk_row + page.num_rows;
    size_t const chunk_last_row = chunk.start_row + chunk.num_rows;
    // These caps are necessary because in the chunked reader, we use estimations for the row
    // counts for list columns, which can result in values > than the absolute number of rows.
    // Capping at the chunk's own last row keeps these indices sorted across a chunk boundary.
    c_info[i].end_row_index = cuda::std::min(max_row, cuda::std::min(chunk_last_row, page_end_row));
  }
};

/**
 * @brief Functor which computes the effective size of all input columns by page
 *
 * For a given row, we want to find the cost of all pages for all columns involved
 * in loading up to that row.  The complication here is that not all pages are the
 * same size between columns. Example:
 *
 *              page row counts
 * Column A:    0 <----> 100 <----> 200
 * Column B:    0 <---------------> 200 <--------> 400
                          |
 * if we decide to split at row 100, we don't really know the actual amount of bytes in column B
 * at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
 * page. Essentially, a conservative over-estimate of the real size.
 */
struct page_total_size {
  cumulative_page_info const* c_info;
  size_type const* key_offsets;
  size_t num_keys;

  __device__ cumulative_page_info operator()(cumulative_page_info const& i) const
  {
    // sum sizes for each input column at this row
    size_t sum = 0;
    for (auto idx = 0; cuda::std::cmp_less(idx, num_keys); idx++) {
      auto const start = key_offsets[idx];
      auto const end   = key_offsets[idx + 1];
      auto iter        = cudf::detail::make_counting_transform_iterator(
        0, cuda::proclaim_return_type<size_t>([&] __device__(size_type i) {
          return c_info[i].end_row_index;
        }));
      // Trailing pages sharing the same end row go uncounted and the sum can fall short by their
      // size. This is acceptable as it is a soft budget that is already deliberately
      // over-estimated above.
      auto const page_index =
        thrust::lower_bound(thrust::seq, iter + start, iter + end, i.end_row_index) - iter;
      sum += c_info[page_index].size_bytes;
    }
    return {i.end_row_index, sum, i.key};
  }
};

/**
 * @brief Return the span of page indices for a given column index that spans start_row and end_row
 *
 */
template <typename RowIndexIter>
struct get_page_span {
  device_span<size_type const> page_offsets;
  device_span<ColumnChunkDesc const> chunks;
  device_span<PageInfo const> pages;
  RowIndexIter page_row_index;
  size_t const start_row;
  size_t const end_row;
  bool const is_first_subpass;
  bool const has_offset_index;

  get_page_span(device_span<size_type const> _page_offsets,
                device_span<ColumnChunkDesc const> _chunks,
                device_span<PageInfo const> _pages,
                RowIndexIter _page_row_index,
                size_t _start_row,
                size_t _end_row,
                bool _is_first_subpass,
                bool _has_offset_index)
    : page_offsets(_page_offsets),
      chunks(_chunks),
      pages(_pages),
      page_row_index(_page_row_index),
      start_row(_start_row),
      end_row(_end_row),
      is_first_subpass(_is_first_subpass),
      has_offset_index(_has_offset_index)
  {
  }

  __device__ page_span operator()(size_t column_index) const
  {
    auto const first_page_index  = page_offsets[column_index];
    auto const column_page_start = page_row_index + first_page_index;
    auto const column_page_end   = page_row_index + page_offsets[column_index + 1];
    auto const num_pages         = column_page_end - column_page_start;
    bool const is_list           = chunks[column_index].max_level[level_type::REPETITION] > 0;

    // For list columns, the row counts are estimates so we need all prefix pages to correctly
    // compute page bounds. For non-list columns, we can get an exact span of pages.
    auto start_page              = first_page_index;
    auto const update_start_page = has_offset_index or (not is_list) or (not is_first_subpass);
    if (update_start_page) {
      start_page += cuda::std::distance(
        column_page_start,
        thrust::lower_bound(thrust::seq, column_page_start, column_page_end, start_row));
    }
    if (page_row_index[start_page] == start_row and (has_offset_index or not is_list)) {
      start_page++;
    }

    auto const first_page_with_end_row =
      cuda::std::distance(
        column_page_start,
        thrust::lower_bound(thrust::seq, column_page_start, column_page_end, end_row)) +
      first_page_index;

    // Following pages that hold no rows of their own only continue a row that began earlier, so
    // they share the same end row index and a search by row cannot reach them. Read them here, as
    // at the end of a pass no later subpass is left to do it and their values would be lost.
    auto const column_end_page = first_page_index + num_pages;
    auto last_page_to_read     = first_page_with_end_row;
    while ((last_page_to_read + 1) < column_end_page) {
      auto const& page      = pages[static_cast<size_t>(last_page_to_read)];
      auto const& next_page = pages[static_cast<size_t>(last_page_to_read + 1)];
      // Stop at the chunk boundary. It is also a row boundary, so the next chunk continues nothing,
      // and it leads with a dictionary page, which holds no rows either.
      if (next_page.num_rows != 0 or next_page.chunk_idx != page.chunk_idx) { break; }
      last_page_to_read++;
    }

    // The returned span is exclusive, so it ends one past the last page to read.
    auto const end_page =
      (last_page_to_read < column_end_page) ? (last_page_to_read + 1) : last_page_to_read;

    return {static_cast<size_t>(start_page), static_cast<size_t>(end_page)};
  }
};

/**
 * @brief Copy page from appropriate source location (as defined by page_offsets) to the destination
 * location, and store the index mapping
 */
struct copy_subpass_page {
  cudf::device_span<PageInfo const> src_pages;
  cudf::device_span<PageInfo> dst_pages;
  cudf::device_span<size_t> page_src_index;
  cudf::device_span<size_t const> page_offsets;
  cudf::device_span<page_span const> page_indices;
  __device__ void operator()(size_t i) const
  {
    auto const index =
      thrust::lower_bound(thrust::seq, page_offsets.begin(), page_offsets.end(), i) -
      page_offsets.begin();
    auto const col_index = page_offsets[index] == i ? index : index - 1;
    // index within the pages for the column
    auto const col_page_index = i - page_offsets[col_index];
    auto const src_page_index = page_indices[col_index].start + col_page_index;

    dst_pages[i]      = src_pages[src_page_index];
    page_src_index[i] = src_page_index;
  }
};

/**
 * @brief Helper function to compute the number of values to process for a page.
 *
 * This is common logic used by both level decoding and string offset allocation.
 * For list pages (with repetition levels), all values must be processed because
 * row boundaries aren't known until after decoding. For non-list pages, only
 * values within the [skip_rows, skip_rows + num_rows) range need to be processed.
 *
 * @param page The page to compute for
 * @param chunk The column chunk descriptor
 * @param skip_rows Starting row for the pass
 * @param num_rows Number of rows to read in the pass
 * @return Number of values to process (0 if page is outside the row range)
 */
CUDF_HOST_DEVICE [[nodiscard]] inline size_t precompute_page_num_values_in_range(
  PageInfo const& page, ColumnChunkDesc const& chunk, size_t skip_rows, size_t num_rows)
{
  // Check if this page has lists (repetition levels)
  bool const has_repetition = chunk.max_level[level_type::REPETITION] > 0;

  if (has_repetition) {
    // Must process all values in list pages because we don't know the row boundaries
    // until we decode the repetition levels.
    return page.num_input_values;
  }

  // For non-list pages: can optimize based on skip_rows and num_rows
  size_t const page_rows      = page.num_rows;
  size_t const page_start_row = chunk.start_row + page.chunk_row;
  size_t const page_end_row   = page_start_row + page_rows;
  size_t const pass_end_row   = skip_rows + num_rows;

  // if we are totally outside the range of the input, do nothing
  if ((page_start_row >= pass_end_row) || (page_end_row <= skip_rows)) { return 0; }

  // For non-list pages: only process values within the pass range
  // If skipping rows, still need to process the first rows to (e.g.) count skipped nulls.
  size_t const pass_rows_from_page = pass_end_row - page_start_row;
  return (page_rows > pass_rows_from_page) ? pass_rows_from_page : page_rows;
}

CUDF_HOST_DEVICE inline void compute_page_level_decode_sizes(PageInfo const& page,
                                                             ColumnChunkDesc const& chunk,
                                                             int level_type_size,
                                                             size_t skip_rows,
                                                             size_t num_rows,
                                                             size_t& def_level_size,
                                                             size_t& rep_level_size)
{
  def_level_size = 0;
  rep_level_size = 0;

  // Skip dictionary pages - they don't have levels to decode
  if (page.flags & PAGEINFO_FLAGS_DICTIONARY) { return; }

  // Check if this page has lists (repetition levels)
  bool const has_repetition = chunk.max_level[level_type::REPETITION] > 0;
  bool const has_definition = chunk.max_level[level_type::DEFINITION] > 0;

  // If no levels to decode, no memory needed
  if (!has_repetition && !has_definition) { return; }

  // Determine how many values need to be decoded using the common helper
  size_t const num_to_decode =
    precompute_page_num_values_in_range(page, chunk, skip_rows, num_rows);

  // Compute space for definition levels
  def_level_size = has_definition ? num_to_decode * level_type_size : 0;

  // Compute space for repetition levels
  rep_level_size = has_repetition ? num_to_decode * level_type_size : 0;
}

/**
 * @brief Result of partitioning row groups into passes.
 */
struct row_group_pass_data {
  std::vector<std::size_t> pass_row_group_offsets;
  std::vector<std::size_t> pass_start_row_counts;
};

/**
 * @brief Partition row groups into passes based on compressed size, leaf value count,
 * and row count thresholds.
 *
 * @param row_group_sizes Span of row group size information
 * @param comp_read_limit Maximum compressed bytes per pass
 * @param skip_rows Number of leading rows to skip (affects the effective size of the first row
 * group)
 * @return A row_group_pass_data containing pass boundary offsets and cumulative row counts
 */
row_group_pass_data compute_row_group_passes(std::span<row_group_size_info const> row_group_sizes,
                                             std::size_t comp_read_limit,
                                             int64_t skip_rows);

}  // namespace cudf::io::parquet::detail
