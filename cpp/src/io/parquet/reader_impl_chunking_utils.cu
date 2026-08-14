/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/decompression.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/utilities/time_utils.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_chunking_utils.cuh"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>

namespace cudf::io::parquet::detail {

using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;
using cudf::io::detail::decompression_info;

#if defined(CHUNKING_DEBUG)
void print_cumulative_page_info(device_span<PageInfo const> d_pages,
                                device_span<ColumnChunkDesc const> d_chunks,
                                device_span<cumulative_page_info const> d_c_info,
                                rmm::cuda_stream_view stream)
{
  auto const pages  = cudf::detail::make_host_vector(d_pages, stream);
  auto const chunks = cudf::detail::make_host_vector(d_chunks, stream);
  auto const c_info = cudf::detail::make_host_vector(d_c_info, stream);

  std::cout << "------------\nCumulative sizes by page\n";

  std::vector<int> schemas(pages.size());
  auto schema_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](size_type i) { return pages[i].src_col_schema; });
  thrust::copy(thrust::seq, schema_iter, schema_iter + pages.size(), schemas.begin());
  auto last = thrust::unique(thrust::seq, schemas.begin(), schemas.end());
  schemas.resize(last - schemas.begin());
  std::cout << "Num schemas: " << schemas.size() << "\n";

  for (size_t idx = 0; idx < schemas.size(); idx++) {
    std::cout << "Schema " << schemas[idx] << ":\n";
    for (size_t pidx = 0; pidx < pages.size(); pidx++) {
      auto const& page = pages[pidx];
      if (page.flags & PAGEINFO_FLAGS_DICTIONARY || page.src_col_schema != schemas[idx]) {
        continue;
      }
      bool const is_list = chunks[page.chunk_idx].max_level[level_type::REPETITION] > 0;
      std::cout << "\tP " << (is_list ? "(L) " : "") << "{" << pidx << ", "
                << c_info[pidx].end_row_index << ", " << c_info[pidx].size_bytes << "}\n";
    }
  }
}

void print_cumulative_page_info(host_span<cumulative_page_info const> sizes,
                                std::string const& label,
                                std::optional<std::vector<row_range>> splits = std::nullopt)
{
  if (splits.has_value()) {
    std::cout << "------------\nSplits (skip_rows, num_rows)\n";
    for (size_t idx = 0; idx < splits->size(); idx++) {
      std::cout << "{" << splits.value()[idx].skip_rows << ", " << splits.value()[idx].num_rows
                << "}\n";
    }
  }

  std::cout << "------------\nCumulative sizes " << label.c_str()
            << " (index, row_index, size_bytes, page_key)\n";
  for (size_t idx = 0; idx < sizes.size(); idx++) {
    std::cout << "{" << idx << ", " << sizes[idx].end_row_index << ", " << sizes[idx].size_bytes
              << ", " << sizes[idx].key << "}";

    if (splits.has_value()) {
      // if we have a split at this row count and this is the last instance of this row count
      auto start             = cuda::make_transform_iterator(splits->begin(),
                                                 [](row_range const& i) { return i.skip_rows; });
      auto end               = start + splits->size();
      auto split             = std::find(start, end, sizes[idx].end_row_index);
      auto const split_index = [&]() -> int {
        if (split != end && ((idx == sizes.size() - 1) ||
                             (sizes[idx + 1].end_row_index > sizes[idx].end_row_index))) {
          return static_cast<int>(std::distance(start, split));
        }
        return idx == 0 ? 0 : -1;
      }();
      if (split_index >= 0) {
        std::cout << " <-- split {" << splits.value()[split_index].skip_rows << ", "
                  << splits.value()[split_index].num_rows << "}";
      }
    }
    std::cout << "\n";
  }
}
#endif  // CHUNKING_DEBUG

void codec_stats::add_pages(host_span<ColumnChunkDesc const> chunks,
                            host_span<PageInfo> pages,
                            page_selection selection,
                            host_span<bool const> page_mask)
{
  auto page_mask_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](size_t page_idx) { return page_mask.empty() ? true : page_mask[page_idx]; });

  // Zip iterator for iterating over pages and the page mask
  auto zip_iter = cuda::make_zip_iterator(pages.begin(), page_mask_iter);

  std::for_each(zip_iter, zip_iter + pages.size(), [&](auto const& item) {
    auto& [page, is_page_needed] = item;
    // If this is a V2 page, use the `is_compressed` field to determine if it's compressed.
    // For dictionary and V1 data pages, they're compressed if chunk.codec is set.
    auto const is_page_compressed = (page.flags & PAGEINFO_FLAGS_V2) ? page.is_compressed : true;
    if (is_page_needed && chunks[page.chunk_idx].codec == compression_type &&
        (page.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) ==
          (selection == page_selection::DICT_PAGES) and
        is_page_compressed) {
      ++num_pages;
      total_decomp_size += page.uncompressed_page_size;
      max_decompressed_size = std::max(max_decompressed_size, page.uncompressed_page_size);
    }
  });
}

CUDF_HOST_DEVICE cuda::std::pair<compression_type, bool> parquet_compression_support(
  Compression compression)
{
  switch (compression) {
    case Compression::BROTLI: return {compression_type::BROTLI, true};
    case Compression::GZIP: return {compression_type::GZIP, true};
    case Compression::LZ4_RAW: return {compression_type::LZ4, true};
    case Compression::LZO: return {compression_type::LZO, false};
    case Compression::SNAPPY: return {compression_type::SNAPPY, true};
    case Compression::ZSTD: return {compression_type::ZSTD, true};
    case Compression::UNCOMPRESSED: return {compression_type::NONE, true};
    default: break;
  }
  return {compression_type::NONE, false};
}

[[nodiscard]] std::string parquet_compression_name(Compression compression)
{
  switch (compression) {
    case Compression::BROTLI: return "BROTLI";
    case Compression::GZIP: return "GZIP";
    case Compression::LZ4_RAW: return "LZ4_RAW";
    case Compression::LZ4: return "LZ4";
    case Compression::LZO: return "LZO";
    case Compression::SNAPPY: return "SNAPPY";
    case Compression::ZSTD: return "ZSTD";
    case Compression::UNCOMPRESSED: return "UNCOMPRESSED";
  }
  CUDF_FAIL("Unsupported Parquet compression type");
}

compression_type from_parquet_compression(Compression compression)
{
  auto const [type, supported] = parquet_compression_support(compression);
  CUDF_EXPECTS(supported,
               "Unsupported Parquet compression type: " + parquet_compression_name(compression));
  return type;
}

size_t find_start_index(cudf::host_span<cumulative_page_info const> aggregated_info,
                        size_t start_row)
{
  auto start = cuda::make_transform_iterator(
    aggregated_info.begin(), [&](cumulative_page_info const& i) { return i.end_row_index; });
  return thrust::lower_bound(thrust::host, start, start + aggregated_info.size(), start_row) -
         start;
}

int64_t find_next_split(int64_t cur_pos,
                        size_t cur_row_index,
                        size_t cur_cumulative_size,
                        cudf::host_span<cumulative_page_info const> sizes,
                        size_t size_limit,
                        size_t min_row_count)
{
  auto const start = cuda::make_transform_iterator(
    sizes.begin(),
    [&](cumulative_page_info const& i) { return i.size_bytes - cur_cumulative_size; });
  auto const end = start + sizes.size();

  int64_t split_pos = thrust::lower_bound(thrust::seq, start + cur_pos, end, size_limit) - start;

  // if we're past the end, or if the returned bucket is > than the chunk_read_limit, move back
  // one as long as this doesn't put us before our starting point.
  if (static_cast<size_t>(split_pos) >= sizes.size() ||
      ((split_pos > cur_pos) && (sizes[split_pos].size_bytes - cur_cumulative_size > size_limit))) {
    split_pos--;
  }

  // move forward until we find the next group of pages that will actually advance our row count.
  // this guarantees that even if we cannot fit the set of rows represented by our where our cur_pos
  // is, we will still move forward instead of failing.
  while (split_pos < (static_cast<int64_t>(sizes.size()) - 1) &&
         (sizes[split_pos].end_row_index - cur_row_index < min_row_count)) {
    split_pos++;
  }

  return split_pos;
}

[[nodiscard]] std::tuple<int32_t, cuda::std::optional<LogicalType>> conversion_info(
  type_id column_type_id,
  type_id timestamp_type_id,
  Type physical,
  cuda::std::optional<LogicalType> logical_type)
{
  int32_t const clock_rate =
    is_chrono(data_type{column_type_id}) ? cudf::io::detail::to_clockrate(timestamp_type_id) : 0;

  // TODO(ets): this is leftover from the original code, but will we ever output decimal as
  // anything but fixed point?
  if (logical_type.has_value() and logical_type->type == LogicalType::DECIMAL) {
    // if decimal but not outputting as float or decimal, then convert to no logical type
    if (column_type_id != type_id::FLOAT64 and
        not cudf::is_fixed_point(data_type{column_type_id})) {
      return {clock_rate, cuda::std::nullopt};
    }
  }

  return {clock_rate, std::move(logical_type)};
}

std::pair<rmm::device_uvector<cumulative_page_info>, rmm::device_uvector<int32_t>>
adjust_cumulative_sizes(device_span<cumulative_page_info const> c_info,
                        device_span<PageInfo const> pages,
                        rmm::cuda_stream_view stream)
{
  // sort by row count
  rmm::device_uvector<cumulative_page_info> c_info_sorted(c_info.size(), stream);
  {
    rmm::device_uvector<size_t> end_row_indices(c_info.size(), stream);
    rmm::device_uvector<size_t> sorted_end_row_indices(c_info.size(), stream);
    rmm::device_uvector<size_t> indices(c_info.size(), stream);
    rmm::device_uvector<size_t> sort_order(c_info.size(), stream);

    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     indices.begin(),
                     indices.end(),
                     0);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      c_info.begin(),
                      c_info.end(),
                      end_row_indices.begin(),
                      [] __device__(auto const& c) { return c.end_row_index; });

    auto tmp_bytes = std::size_t{0};
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    tmp_bytes,
                                    end_row_indices.begin(),         // keys in
                                    sorted_end_row_indices.begin(),  // sorted keys out
                                    indices.begin(),                 // values in
                                    sort_order.begin(),              // sorted values out
                                    c_info.size(),
                                    0,
                                    sizeof(size_t) * 8,
                                    stream.value());
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    cub::DeviceRadixSort::SortPairs(tmp_stg.data(),
                                    tmp_bytes,
                                    end_row_indices.begin(),         // keys in
                                    sorted_end_row_indices.begin(),  // sorted keys out
                                    indices.begin(),                 // values in
                                    sort_order.begin(),              // sorted values out
                                    c_info.size(),
                                    0,
                                    sizeof(size_t) * 8,
                                    stream.value());

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      sort_order.begin(),
                      sort_order.end(),
                      c_info_sorted.begin(),
                      [c_info] __device__(std::size_t i) { return c_info[i]; });
  }

  // page keys grouped by split.
  rmm::device_uvector<int32_t> page_keys_by_split{c_info.size(), stream};
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    c_info_sorted.begin(),
                    c_info_sorted.end(),
                    page_keys_by_split.begin(),
                    cuda::proclaim_return_type<int>(
                      [] __device__(cumulative_page_info const& c) { return c.key; }));

  // generate key offsets (offsets to the start of each partition of keys). worst case is 1 page per
  // key
  rmm::device_uvector<size_type> key_offsets(pages.size() + 1, stream);
  auto page_keys             = make_page_key_iterator(pages);
  auto const key_offsets_end = cudf::detail::reduce_by_key(page_keys,
                                                           page_keys + pages.size(),
                                                           cuda::make_constant_iterator(1),
                                                           cuda::make_discard_iterator(),
                                                           key_offsets.begin(),
                                                           cuda::std::plus<>{},
                                                           stream)
                                 .second;

  size_t const num_unique_keys = key_offsets_end - key_offsets.begin();
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         key_offsets.begin(),
                         key_offsets.end(),
                         key_offsets.begin());

  // adjust the cumulative info such that for each row count, the size includes any pages that span
  // that row count. this is so that if we have this case:
  //              page row counts
  // Column A:    0 <----> 100 <----> 200
  // Column B:    0 <---------------> 200 <--------> 400
  //                        |
  // if we decide to split at row 100, we don't really know the actual amount of bytes in column B
  // at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
  // page.
  //
  rmm::device_uvector<cumulative_page_info> aggregated_info(c_info.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    c_info_sorted.begin(),
                    c_info_sorted.end(),
                    aggregated_info.begin(),
                    page_total_size{c_info.data(), key_offsets.data(), num_unique_keys});
  return {std::move(aggregated_info), std::move(page_keys_by_split)};
}

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
  rmm::cuda_stream_view stream)
{
  auto [aggregated_info, page_keys_by_split] = adjust_cumulative_sizes(c_info, pages, stream);

  // bring back to the cpu
  auto h_aggregated_info = cudf::detail::make_pinned_vector(aggregated_info, stream);

#if defined(CHUNKING_DEBUG)
  print_cumulative_page_info(h_aggregated_info, "adjusted");
#endif  // CHUNKING_DEBUG

  // TODO: if the user has explicitly specified skip_rows/num_rows we could be more intelligent
  // about skipping subpasses/pages that do not fall within the range of values, but only if the
  // data does not contain lists (because our row counts are only estimates in that case)

  // find the next split
  auto const start_index = find_start_index(h_aggregated_info, start_row);
  auto const cumulative_size =
    start_row == 0 || start_index == 0 ? 0 : h_aggregated_info[start_index - 1].size_bytes;
  // when choosing subpasses, we need to guarantee at least 2 rows in the included pages so that all
  // list columns have a clear start and end.
  auto const end_index =
    find_next_split(start_index, start_row, cumulative_size, h_aggregated_info, size_limit, 2);
  auto const end_row = h_aggregated_info[end_index].end_row_index;

  // for each column, collect the set of pages that spans start_row / end_row
  rmm::device_uvector<page_span> page_bounds(num_columns, stream);
  auto iter = cuda::counting_iterator{size_t{0}};
  auto page_row_index =
    cudf::detail::make_counting_transform_iterator(0, get_page_end_row_index{c_info});
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    iter,
                    iter + num_columns,
                    page_bounds.begin(),
                    get_page_span{page_offsets,
                                  chunks,
                                  page_row_index,
                                  start_row,
                                  end_row,
                                  is_first_subpass,
                                  has_offset_index});

  // total page count over all columns
  auto page_count_iter   = cuda::make_transform_iterator(page_bounds.begin(), get_span_size{});
  auto const total_pages = cudf::detail::reduce(
    page_count_iter, page_count_iter + num_columns, size_t{0}, cuda::std::plus<size_t>{}, stream);

  return {
    std::move(page_bounds), total_pages, h_aggregated_info[end_index].size_bytes - cumulative_size};
}

std::vector<row_range> compute_page_splits_by_row(device_span<cumulative_page_info const> c_info,
                                                  device_span<PageInfo const> pages,
                                                  size_t skip_rows,
                                                  size_t num_rows,
                                                  size_t size_limit,
                                                  rmm::cuda_stream_view stream)
{
  auto [aggregated_info, page_keys_by_split] = adjust_cumulative_sizes(c_info, pages, stream);

  // bring back to the cpu
  auto h_aggregated_info = cudf::detail::make_pinned_vector(aggregated_info, stream);

#if defined(CHUNKING_DEBUG)
  print_cumulative_page_info(h_aggregated_info, "adjusted");
#endif  // CHUNKING_DEBUG

  std::vector<row_range> splits;
  // note: we are working with absolute row indices so skip_rows represents the absolute min row
  // index we care about
  size_t cur_pos             = find_start_index(h_aggregated_info, skip_rows);
  size_t cur_row_index       = skip_rows;
  size_t cur_cumulative_size = 0;
  auto const max_row = std::min(skip_rows + num_rows, h_aggregated_info.back().end_row_index);
  while (cur_row_index < max_row) {
    auto const split_pos = find_next_split(
      cur_pos, cur_row_index, cur_cumulative_size, h_aggregated_info, size_limit, 1);

    auto const start_row = cur_row_index;
    cur_row_index        = std::min(max_row, h_aggregated_info[split_pos].end_row_index);
    splits.push_back({start_row, cur_row_index - start_row});
    cur_pos             = split_pos;
    cur_cumulative_size = h_aggregated_info[split_pos].size_bytes;
  }

#if defined(CHUNKING_DEBUG)
  print_cumulative_page_info(h_aggregated_info, "adjusted w/splits", splits);
#endif  // CHUNKING_DEBUG

  return splits;
}

[[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> decompress_page_data(
  host_span<ColumnChunkDesc const> chunks,
  host_span<PageInfo> pass_pages,
  host_span<PageInfo> subpass_pages,
  host_span<bool const> subpass_page_mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(
    subpass_page_mask.empty() or subpass_page_mask.size() == subpass_pages.size(),
    "Subpass page mask must either be empty or have size equal to the number of subpass pages",
    std::invalid_argument);

  std::array codecs{codec_stats{Compression::BROTLI},
                    codec_stats{Compression::GZIP},
                    codec_stats{Compression::LZ4_RAW},
                    codec_stats{Compression::SNAPPY},
                    codec_stats{Compression::ZSTD}};

  auto is_codec_supported = [&codecs](Compression codec) {
    if (codec == Compression::UNCOMPRESSED) return true;
    return std::find_if(codecs.begin(), codecs.end(), [codec](auto& cstats) {
             return codec == cstats.compression_type;
           }) != codecs.end();
  };

  for (auto const& chunk : chunks) {
    CUDF_EXPECTS(is_codec_supported(chunk.codec),
                 "Unsupported Parquet compression type: " + parquet_compression_name(chunk.codec));
  }

  size_t total_pass_decomp_size = 0;
  for (auto& codec : codecs) {
    // Use an empty span as pass page mask as we don't want to filter out dictionary pages
    codec.add_pages(chunks, pass_pages, codec_stats::page_selection::DICT_PAGES, {});
    total_pass_decomp_size += codec.total_decomp_size;
  }

  // Total number of pages to decompress, including both pass and subpass pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;
  for (auto& codec : codecs) {
    codec.add_pages(
      chunks, subpass_pages, codec_stats::page_selection::NON_DICT_PAGES, subpass_page_mask);
    // at this point, the codec contains info for both dictionary pass pages and data subpass pages
    total_decomp_size += codec.total_decomp_size;
    num_comp_pages += codec.num_pages;
  }

  // Dispatch batches of pages to decompress for each codec.
  // Buffer needs to be padded, required by `gpuDecodePageData`.
  rmm::device_buffer pass_decomp_pages(
    cudf::util::round_up_safe(total_pass_decomp_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
    stream,
    mr);
  auto const total_subpass_decomp_size = total_decomp_size - total_pass_decomp_size;
  rmm::device_buffer subpass_decomp_pages(
    cudf::util::round_up_safe(total_subpass_decomp_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
    stream,
    mr);

  auto comp_in =
    cudf::detail::make_pinned_vector_async<device_span<uint8_t const>>(num_comp_pages, stream);
  auto comp_out =
    cudf::detail::make_pinned_vector_async<device_span<uint8_t>>(num_comp_pages, stream);
  auto curr_comp_page = 0;

  // vectors to save v2 def and rep level data, if any
  auto copy_in =
    cudf::detail::make_pinned_vector_async<device_span<uint8_t const>>(num_comp_pages, stream);
  auto copy_out =
    cudf::detail::make_pinned_vector_async<device_span<uint8_t>>(num_comp_pages, stream);
  auto curr_copy_page = 0;

  auto set_parameters = [&](codec_stats& codec,
                            host_span<PageInfo> pages,
                            host_span<bool const> page_mask,
                            void* decomp_data,
                            bool select_dict_pages,
                            size_t& decomp_offset) {
    // Create a page mask iterator that defaults to true if the page_mask is empty
    auto page_mask_iter = cudf::detail::make_counting_transform_iterator(
      0, [&](size_t page_idx) { return page_mask.empty() ? true : page_mask[page_idx]; });

    for (auto page_idx = 0; std::cmp_less(page_idx, pages.size()); ++page_idx) {
      auto& page                = pages[page_idx];
      auto const is_page_needed = page_mask_iter[page_idx];
      // If this is a V2 page, use the `is_compressed` field to determine if it's compressed.
      // For dictionary and V1 data pages, they're compressed if chunk.codec is set.
      auto const is_page_compressed = (page.flags & PAGEINFO_FLAGS_V2) ? page.is_compressed : true;
      if (is_page_needed && chunks[page.chunk_idx].codec == codec.compression_type &&
          (page.flags & PAGEINFO_FLAGS_DICTIONARY) == select_dict_pages and is_page_compressed) {
        auto const dst_base = static_cast<uint8_t*>(decomp_data) + decomp_offset;
        // offset will only be non-zero for V2 pages
        auto const offset =
          page.lvl_bytes[level_type::DEFINITION] + page.lvl_bytes[level_type::REPETITION];
        // for V2 need to copy def and rep level info into place, and then offset the
        // input and output buffers. otherwise we'd have to keep both the compressed
        // and decompressed data.
        if (offset != 0) {
          copy_in[curr_copy_page]  = {page.page_data, static_cast<size_t>(offset)};
          copy_out[curr_copy_page] = {dst_base, static_cast<size_t>(offset)};
          ++curr_copy_page;
        }
        // Only decompress if the page contains data after the def/rep levels
        if (page.compressed_page_size > offset) {
          comp_in[curr_comp_page]  = {page.page_data + offset,
                                      static_cast<size_t>(page.compressed_page_size - offset)};
          comp_out[curr_comp_page] = {dst_base + offset,
                                      static_cast<size_t>(page.uncompressed_page_size - offset)};
          ++curr_comp_page;
        } else {
          // If the page wasn't included in the decompression parameters, we need to adjust the
          // page count to allocate results and perform decompression correctly
          --codec.num_pages;
          --num_comp_pages;
        }
        page.page_data = dst_base;
        decomp_offset += page.uncompressed_page_size;
      }
    }
  };

  size_t pass_decomp_offset    = 0;
  size_t subpass_decomp_offset = 0;
  for (auto& codec : codecs) {
    if (codec.num_pages == 0) { continue; }
    // Use empty span as pass page mask as we don't want to filter out dictionary pages
    set_parameters(codec, pass_pages, {}, pass_decomp_pages.data(), true, pass_decomp_offset);
    set_parameters(codec,
                   subpass_pages,
                   subpass_page_mask,
                   subpass_decomp_pages.data(),
                   false,
                   subpass_decomp_offset);
  }

  auto const d_comp_in = cudf::detail::make_device_uvector_async(
    comp_in, stream, cudf::get_current_device_resource_ref());
  auto const d_comp_out = cudf::detail::make_device_uvector_async(
    comp_out, stream, cudf::get_current_device_resource_ref());
  rmm::device_uvector<codec_exec_result> comp_res(num_comp_pages, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    comp_res.begin(),
    comp_res.end(),
    codec_exec_result{0, codec_status::FAILURE});

  int32_t start_pos = 0;
  for (auto const& codec : codecs) {
    if (codec.num_pages == 0) { continue; }
    CUDF_EXPECTS(is_supported_read_parquet(from_parquet_compression(codec.compression_type)),
                 "Unsupported compression type for Parquet reading");

    device_span<device_span<uint8_t const> const> d_comp_in_view{d_comp_in.data() + start_pos,
                                                                 codec.num_pages};
    device_span<device_span<uint8_t> const> d_comp_out_view(d_comp_out.data() + start_pos,
                                                            codec.num_pages);
    device_span<codec_exec_result> d_comp_res_view(comp_res.data() + start_pos, codec.num_pages);
    cudf::io::detail::decompress(from_parquet_compression(codec.compression_type),
                                 d_comp_in_view,
                                 d_comp_out_view,
                                 d_comp_res_view,
                                 codec.max_decompressed_size,
                                 codec.total_decomp_size,
                                 stream);

    start_pos += codec.num_pages;
  }
  // now copy the uncompressed V2 def and rep level data
  if (curr_copy_page > 0) {
    auto const d_copy_in = cudf::detail::make_device_uvector_async(
      copy_in, stream, cudf::get_current_device_resource_ref());
    auto const d_copy_out = cudf::detail::make_device_uvector_async(
      copy_out, stream, cudf::get_current_device_resource_ref());

    auto const src_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t const*>(
        [inputs = d_copy_in.data()] __device__(size_type i) { return inputs[i].data(); }));
    auto const dst_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t*>(
        [outputs = d_copy_out.data()] __device__(size_type i) { return outputs[i].data(); }));
    auto const size_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<size_t>(
        [inputs = d_copy_in.data(), outputs = d_copy_out.data()] __device__(size_type i) {
          return inputs[i].size() < outputs[i].size() ? inputs[i].size() : outputs[i].size();
        }));
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, size_iter, curr_copy_page, stream);
  }

  CUDF_EXPECTS(
    cudf::detail::all_of(comp_res.begin(),
                         comp_res.end(),
                         cuda::proclaim_return_type<bool>([] __device__(auto const& res) {
                           return res.status == codec_status::SUCCESS;
                         }),
                         stream),
    "Error during decompression");

  return {std::move(pass_decomp_pages), std::move(subpass_decomp_pages)};
}

void detect_malformed_pages(device_span<PageInfo const> pages,
                            device_span<ColumnChunkDesc const> chunks,
                            std::optional<size_t> expected_row_count,
                            rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // sum row counts for all non-dictionary, non-list columns. other columns will be indicated as 0
  rmm::device_uvector<size_type> row_counts(pages.size(),
                                            stream);  // worst case:  num keys == num pages
  auto const size_iter =
    cuda::make_transform_iterator(pages.begin(), flat_column_num_rows{chunks.data()});
  auto const row_counts_begin = row_counts.begin();
  auto page_keys              = make_page_key_iterator(pages);
  auto const row_counts_end   = cudf::detail::reduce_by_key(page_keys,
                                                          page_keys + pages.size(),
                                                          size_iter,
                                                          cuda::make_discard_iterator(),
                                                          row_counts_begin,
                                                          cuda::std::plus<>{},
                                                          stream)
                                .second;

  // make sure all non-zero row counts are the same
  rmm::device_uvector<size_type> compacted_row_counts(pages.size(), stream);
  auto const compacted_row_counts_begin = compacted_row_counts.begin();
  auto const compacted_row_counts_end   = cudf::detail::copy_if(
    row_counts_begin, row_counts_end, compacted_row_counts_begin, row_counts_nonzero{}, stream);
  if (compacted_row_counts_end != compacted_row_counts_begin) {
    auto const found_row_count = [&]() {
      auto found_row_count = cudf::detail::make_pinned_vector(
        cudf::device_span<size_type const>{compacted_row_counts.data(), 1}, stream);
      return found_row_count.front();
    }();

    // if we somehow don't match the expected row count from the row groups themselves
    if (expected_row_count.has_value()) {
      CUDF_EXPECTS(std::cmp_equal(expected_row_count.value(), found_row_count),
                   "Encountered malformed parquet page data (unexpected row count in page data)");
    }

    // all non-zero row counts must be the same
    auto const row_counts_match =
      cudf::detail::none_of(compacted_row_counts_begin,
                            compacted_row_counts_end,
                            row_counts_different{static_cast<size_type>(found_row_count)},
                            stream);
    CUDF_EXPECTS(row_counts_match,
                 "Encountered malformed parquet page data (row count mismatch in page data)");
  }
}

rmm::device_uvector<size_t> compute_decompression_scratch_sizes(
  device_span<ColumnChunkDesc const> chunks,
  device_span<PageInfo const> pages,
  rmm::cuda_stream_view stream)
{
  auto page_keys = make_page_key_iterator(pages);

  // per-codec page counts and decompression sizes
  rmm::device_uvector<decompression_info> decomp_info(pages.size(), stream);
  auto decomp_iter = cuda::make_transform_iterator(pages.begin(), get_decomp_info{chunks});
  thrust::inclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    page_keys,
    page_keys + pages.size(),
    decomp_iter,
    decomp_info.begin(),
    cuda::std::equal_to<int32_t>{},
    decomp_sum{});

  // retrieve to host so we can get compression scratch sizes
  auto temp_cost     = cudf::detail::make_pinned_vector_async<size_t>(pages.size(), stream);
  auto h_decomp_info = cudf::detail::make_pinned_vector(decomp_info, stream);
  std::transform(h_decomp_info.begin(), h_decomp_info.end(), temp_cost.begin(), [](auto const& d) {
    return cudf::io::detail::get_decompression_scratch_size(d);
  });

  rmm::device_uvector<size_t> d_temp_cost =
    cudf::detail::make_device_uvector(temp_cost, stream, cudf::get_current_device_resource_ref());

  std::array codecs{compression_type::BROTLI,
                    compression_type::GZIP,
                    compression_type::LZ4,
                    compression_type::SNAPPY,
                    compression_type::ZSTD};
  for (auto const codec : codecs) {
    if (cudf::io::detail::is_decompression_scratch_size_ex_supported(codec)) {
      auto const total_decomp_info = cudf::detail::transform_reduce(
        decomp_iter,
        decomp_iter + pages.size(),
        cuda::proclaim_return_type<decompression_info>(
          [codec] __device__(decompression_info const& d) {
            return d.type == codec ? d : decompression_info{codec, 0, 0, 0};
          }),
        decompression_info{codec, 0, 0, 0},
        decomp_sum{},
        stream);

      // Collect pages with matching codecs
      rmm::device_uvector<device_span<uint8_t const>> temp_spans(pages.size(), stream);
      auto iter = cuda::counting_iterator{size_t{0}};
      thrust::for_each(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        iter,
        iter + pages.size(),
        [pages      = pages.begin(),
         chunks     = chunks.begin(),
         temp_spans = temp_spans.begin(),
         codec] __device__(size_t i) {
          auto const& page = pages[i];
          if (parquet_compression_support(chunks[page.chunk_idx].codec).first == codec) {
            temp_spans[i] = device_span<uint8_t const>(
              page.page_data, static_cast<size_t>(page.compressed_page_size));
          } else {
            temp_spans[i] = device_span<uint8_t const>();  // Mark pages with other codecs as empty
          }
        });
      // Copy only non-null spans
      rmm::device_uvector<device_span<uint8_t const>> page_spans(pages.size(), stream);
      auto end_iter =
        cudf::detail::copy_if(temp_spans.begin(),
                              temp_spans.end(),
                              page_spans.begin(),
                              cuda::proclaim_return_type<bool>(
                                [] __device__(auto const& span) { return span.data() != nullptr; }),
                              stream);
      if (end_iter == page_spans.begin()) {
        // No pages compressed with this codec, skip
        continue;
      }
      page_spans.resize(end_iter - page_spans.begin(), stream);

      auto const total_temp_size    = get_decompression_scratch_size(total_decomp_info);
      auto const total_temp_size_ex = cudf::io::detail::get_decompression_scratch_size_ex(
        total_decomp_info.type,
        page_spans,
        total_decomp_info.max_page_decompressed_size,
        total_decomp_info.total_decompressed_size,
        stream);

      // Make use of the extended API if it provides a more accurate estimate
      if (total_temp_size_ex < total_temp_size) {
        // The new extended API provides a more accurate (smaller) estimate than the legacy API.
        // We cannot efficiently use the extended API to get per-page scratch sizes, so we adjust
        // the per-page scratch sizes to on-average reflect the better estimate. This means that
        // the scratch size might not be accurate for each page, but it will in aggregate.
        auto const adjustment_ratio = static_cast<double>(total_temp_size_ex) / total_temp_size;

        // Apply the adjustment ratio to each page's temporary cost
        thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator{size_t{0}},
                         cuda::counting_iterator{pages.size()},
                         [pages           = pages.begin(),
                          chunks          = chunks.begin(),
                          d_temp_cost_ptr = d_temp_cost.begin(),
                          adjustment_ratio,
                          codec] __device__(size_t i) {
                           auto const page_codec =
                             parquet_compression_support(chunks[pages[i].chunk_idx].codec).first;
                           // Only adjust pages that use the current compression codec
                           if (page_codec == codec) {
                             auto const cost = d_temp_cost_ptr[i];
                             // Scale down the cost and round up to ensure we don't underestimate
                             auto const adjusted =
                               static_cast<size_t>(cuda::std::ceil(cost * adjustment_ratio));
                             d_temp_cost_ptr[i] = adjusted;
                           }
                         });
      }
    }
  }
  return d_temp_cost;
}

void include_scratch_size(device_span<size_t const> temp_cost,
                          device_span<cumulative_page_info> c_info,
                          rmm::cuda_stream_view stream)
{
  auto iter = cuda::counting_iterator{size_t{0}};
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   iter,
                   iter + c_info.size(),
                   [temp_cost = temp_cost.begin(), c_info = c_info.begin()] __device__(size_t i) {
                     c_info[i].size_bytes += temp_cost[i];
                   });
}

namespace {

/**
 * @brief Functor to compute the string offset buffer size needed for a page.
 *
 * This computes the memory needed to store string offsets (uint32_t per value)
 * for non-dictionary, non-FLBA string pages.
 */
struct compute_page_string_offset_size {
  device_span<ColumnChunkDesc const> chunks;
  size_t skip_rows;
  size_t num_rows;

  __device__ size_t operator()(PageInfo const& page) const
  {
    // Mask for non-dictionary, non-delta string columns (same as used in decode kernel)
    constexpr uint32_t STRINGS_MASK_NON_DELTA_NON_DICT =
      BitOr(decode_kernel_mask::STRING,
            decode_kernel_mask::STRING_NESTED,
            decode_kernel_mask::STRING_LIST,
            decode_kernel_mask::STRING_STREAM_SPLIT,
            decode_kernel_mask::STRING_STREAM_SPLIT_NESTED,
            decode_kernel_mask::STRING_STREAM_SPLIT_LIST);

    auto const& chunk = chunks[page.chunk_idx];

    // Check if this page is a non-dictionary string page using kernel mask
    if (BitAnd(page.kernel_mask, STRINGS_MASK_NON_DELTA_NON_DICT) == 0) { return 0; }

    // Fixed length byte array: Offsets are fixed, no need to allocate offset buffer
    if (chunk.physical_type == Type::FIXED_LEN_BYTE_ARRAY) { return 0; }

    // Determine how many values need offsets
    size_t const num_values = precompute_page_num_values_in_range(page, chunk, skip_rows, num_rows);
    if (num_values == 0) { return 0; }

    // We need num_values + 1 offsets (one extra for the final offset)
    auto const num_offsets = num_values + 1;

    // Return size in bytes (uint32_t per offset)
    return num_offsets * sizeof(uint32_t);
  }
};

/**
 * @brief Functor to compute the level decode buffer size needed for a page.
 *
 * This computes the memory needed to store definition and repetition levels.
 */
struct compute_page_level_decode_size {
  cudf::device_span<ColumnChunkDesc const> chunks;
  int level_type_size;
  size_t skip_rows;
  size_t num_rows;

  __device__ size_t operator()(PageInfo const& page) const
  {
    size_t def_level_size = 0;
    size_t rep_level_size = 0;
    compute_page_level_decode_sizes(page,
                                    chunks[page.chunk_idx],
                                    level_type_size,
                                    skip_rows,
                                    num_rows,
                                    def_level_size,
                                    rep_level_size);
    return def_level_size + rep_level_size;
  }
};

}  // anonymous namespace

rmm::device_uvector<size_t> compute_string_offset_sizes(device_span<ColumnChunkDesc const> chunks,
                                                        device_span<PageInfo const> pages,
                                                        size_t skip_rows,
                                                        size_t num_rows,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<size_t> string_offset_sizes(pages.size(), stream, mr);

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    pages.begin(),
                    pages.end(),
                    string_offset_sizes.begin(),
                    compute_page_string_offset_size{chunks, skip_rows, num_rows});

  return string_offset_sizes;
}

rmm::device_uvector<size_t> compute_level_decode_sizes(device_span<ColumnChunkDesc const> chunks,
                                                       device_span<PageInfo const> pages,
                                                       int level_type_size,
                                                       size_t skip_rows,
                                                       size_t num_rows,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<size_t> level_decode_sizes(pages.size(), stream, mr);

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    pages.begin(),
                    pages.end(),
                    level_decode_sizes.begin(),
                    compute_page_level_decode_size{chunks, level_type_size, skip_rows, num_rows});

  return level_decode_sizes;
}

row_group_pass_data compute_row_group_passes(std::span<row_group_size_info const> row_group_sizes,
                                             std::size_t comp_read_limit,
                                             int64_t skip_rows)
{
  auto constexpr max_rows_per_pass =
    static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());

  row_group_pass_data result;
  result.pass_row_group_offsets.push_back(0);
  result.pass_start_row_counts.push_back(0);

  std::size_t cur_pass_byte_size       = 0;
  std::size_t cur_pass_num_leaf_values = 0;
  std::size_t cur_pass_num_rows        = 0;
  std::size_t cur_rg_start             = 0;
  std::size_t cur_row_count            = 0;

  for (std::size_t cur_rg_index = 0; cur_rg_index < row_group_sizes.size(); cur_rg_index++) {
    auto const& rgi = row_group_sizes[cur_rg_index];

    // We must use the effective size of the first row group we are reading to accurately calculate
    // the first non-zero `input_pass_start_row_count` unless we are reading only one row group
    auto row_group_rows = rgi.unadjusted_num_rows;
    if (row_group_sizes.size() > 1) {
      CUDF_EXPECTS(std::cmp_greater_equal(rgi.unadjusted_num_rows, skip_rows),
                   "Row groups must contribute non-negative effective rows",
                   std::invalid_argument);
      row_group_rows -= skip_rows;
    }

    auto const compressed_rg_size    = rgi.compressed_size;
    auto const row_group_leaf_values = rgi.max_leaf_values;

    //  Set skip_rows = 0 as it is no longer needed for subsequent row_groups
    skip_rows = 0;

    // Check if we need to create a pass boundary here?
    // Note: Here we may end up with an invalid pass (number of rows exceeding the cudf column size
    // limit) in certain edge case conditions such as:
    // 1. Number of leaf-level values plus nulls (computed by dremel decoding) exceeds the cudf
    // column size limit
    // 2. For nested lists (list<list<list<...>>>), one or more nested list(s) may have number of
    // rows (computed by dremel decoding) exceeding the cudf column size limit
    if ((cur_pass_byte_size + compressed_rg_size >= comp_read_limit) or
        (cur_pass_num_leaf_values + row_group_leaf_values >= max_rows_per_pass) or
        (cur_pass_num_rows + row_group_rows >= max_rows_per_pass)) {
      // A single row group (the current one) is larger than the read limit:
      // We always need to include at least one row group, so end the pass at the end of the current
      // row group
      if (cur_rg_start == cur_rg_index) {
        CUDF_EXPECTS(std::cmp_less_equal(rgi.unadjusted_num_rows, max_rows_per_pass),
                     "Number of rows in each row group must be smaller than the column size limit");
        result.pass_row_group_offsets.push_back(cur_rg_index + 1);
        result.pass_start_row_counts.push_back(cur_row_count + row_group_rows);
        cur_rg_start             = cur_rg_index + 1;
        cur_pass_byte_size       = 0;
        cur_pass_num_leaf_values = 0;
        cur_pass_num_rows        = 0;
      }
      // End the pass at the end of the previous row group
      else {
        result.pass_row_group_offsets.push_back(cur_rg_index);
        result.pass_start_row_counts.push_back(cur_row_count);
        cur_rg_start             = cur_rg_index;
        cur_pass_byte_size       = compressed_rg_size;
        cur_pass_num_leaf_values = row_group_leaf_values;
        cur_pass_num_rows        = row_group_rows;
      }
    } else {
      cur_pass_byte_size += compressed_rg_size;
      cur_pass_num_leaf_values += row_group_leaf_values;
      cur_pass_num_rows += row_group_rows;
    }
    cur_row_count += row_group_rows;
  }

  // Add the last pass if necessary
  if (result.pass_row_group_offsets.back() != row_group_sizes.size()) {
    result.pass_row_group_offsets.push_back(row_group_sizes.size());
    result.pass_start_row_counts.push_back(cur_row_count);
  }

  return result;
}

}  // namespace cudf::io::parquet::detail
