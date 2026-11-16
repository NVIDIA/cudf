/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "error.hpp"
#include "reader_impl_preprocess_utils.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm.lower_bound.h>
#include <cuda/std/algorithm.max.h>
#include <cuda/std/algorithm.upper_bound.h>
#include <cuda/std/cstdint>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <bitset>
#if defined(PREPROCESS_DEBUG)
#include <iostream>
#endif  // PREPROCESS_DEBUG
#include <numeric>

namespace cudf::io::parquet::detail {

#if defined(PREPROCESS_DEBUG)
void print_pages(cudf::detail::hostdevice_span<PageInfo> pages, cuda::stream_ref stream)
{
  pages.device_to_host(stream);
  auto idx = 0;
  std::for_each(pages.host_begin(), pages.host_end(), [&](PageInfo const& p) {
    if (p.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) { std::cout << "Dict"; }
    std::cout << "P(" << idx++ << ", s:" << p.src_col_schema << "): chunk_row(" << p.chunk_row
              << "), num_rows(" << p.num_rows << "), skipped_values(" << p.skipped_values
              << "), skipped_leaf_values(" << p.skipped_leaf_values << "), str_bytes("
              << p.str_bytes << ")\n";
  });
}
#endif  // PREPROCESS_DEBUG

void generate_depth_remappings(
  std::map<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>>& remap,
  int const src_col_schema,
  int const mapped_src_col_schema,
  int const src_file_idx,
  aggregate_reader_metadata const& md)
{
  // already generated for this level
  if (remap.find({src_col_schema, src_file_idx}) != remap.end()) { return; }
  auto const& schema   = md.get_schema(mapped_src_col_schema, src_file_idx);
  auto const max_depth = md.get_output_nesting_depth(src_col_schema);

  CUDF_EXPECTS(remap.find({src_col_schema, src_file_idx}) == remap.end(),
               "Attempting to remap a schema more than once");
  auto inserted =
    remap.insert(std::pair<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>>{
      {src_col_schema, src_file_idx}, {}});
  auto& depth_remap = inserted.first->second;

  std::vector<int>& rep_depth_remap = (depth_remap.first);
  rep_depth_remap.resize(schema.max_repetition_level + 1);
  std::vector<int>& def_depth_remap = (depth_remap.second);
  def_depth_remap.resize(schema.max_definition_level + 1);

  // the key:
  // for incoming level values  R/D
  // add values starting at the shallowest nesting level X has repetition level R
  // until you reach the deepest nesting level Y that corresponds to the repetition level R1
  // held by the nesting level that has definition level D
  //
  // Example: a 3 level struct with a list at the bottom
  //
  //                     R / D   Depth
  // level0              0 / 1     0
  //   level1            0 / 2     1
  //     level2          0 / 3     2
  //       list          0 / 3     3
  //         element     1 / 4     4
  //
  // incoming R/D : 0, 0  -> add values from depth 0 to 3   (def level 0 always maps to depth 0)
  // incoming R/D : 0, 1  -> add values from depth 0 to 3
  // incoming R/D : 0, 2  -> add values from depth 0 to 3
  // incoming R/D : 1, 4  -> add values from depth 4 to 4
  //
  // Note : the -validity- of values is simply checked by comparing the incoming D value against the
  // D value of the given nesting level (incoming D >= the D for the nesting level == valid,
  // otherwise NULL).  The tricky part is determining what nesting levels to add values at.
  //
  // For schemas with no repetition level (no lists), X is always 0 and Y is always max nesting
  // depth.
  //

  // compute "X" from above
  for (int s_idx = schema.max_repetition_level; s_idx >= 0; s_idx--) {
    auto find_shallowest = [&](int r) {
      int shallowest = -1;
      int cur_depth  = max_depth - 1;
      int schema_idx = mapped_src_col_schema;
      while (schema_idx > 0) {
        auto& cur_schema = md.get_schema(schema_idx, src_file_idx);
        auto const one_level_list =
          cur_schema.is_one_level_list(md.get_schema(cur_schema.parent_idx, src_file_idx));
        if (cur_schema.max_repetition_level == r) {
          // if this is a repeated field, map it one level deeper
          shallowest = cur_schema.is_stub() ? cur_depth + 1 : cur_depth;
        }
        // if it's one-level encoding list
        else if (one_level_list) {
          shallowest = cur_depth - 1;
        }
        // a one-level list also accounts for its implicit list level
        if (!cur_schema.is_stub()) { cur_depth -= one_level_list ? 2 : 1; }
        schema_idx = cur_schema.parent_idx;
      }
      return shallowest;
    };
    rep_depth_remap[s_idx] = find_shallowest(s_idx);
  }

  // compute "Y" from above
  for (int s_idx = schema.max_definition_level; s_idx >= 0; s_idx--) {
    auto find_deepest = [&](int d) {
      int schema_idx = mapped_src_col_schema;
      int r1         = 0;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx, src_file_idx);
        if (cur_schema.max_definition_level == d) {
          r1 = cur_schema.max_repetition_level;
          break;
        }
        schema_idx = cur_schema.parent_idx;
      }

      // we now know R1 from above. return the deepest nesting level that has the
      // same repetition level
      schema_idx = mapped_src_col_schema;
      int depth  = max_depth - 1;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx, src_file_idx);
        auto const one_level_list =
          cur_schema.is_one_level_list(md.get_schema(cur_schema.parent_idx, src_file_idx));
        if (cur_schema.max_repetition_level == r1) {
          // if this is a repeated field, map it one level deeper
          depth = cur_schema.is_stub() ? depth + 1 : depth;
          break;
        }
        // the implicit list of a one-level list has one less repetition level than its leaf
        if (one_level_list and cur_schema.max_repetition_level - 1 == r1) {
          depth--;
          break;
        }
        if (!cur_schema.is_stub()) { depth -= one_level_list ? 2 : 1; }
        schema_idx = cur_schema.parent_idx;
      }
      return depth;
    };
    def_depth_remap[s_idx] = find_deepest(s_idx);
  }
}

[[nodiscard]] std::future<void> read_column_chunks_async(
  std::vector<std::unique_ptr<datasource>> const& sources,
  cudf::host_span<rmm::device_buffer> page_data,
  cudf::detail::hostdevice_vector<ColumnChunkDesc>& chunks,
  size_t begin_chunk,
  size_t end_chunk,
  std::vector<size_t> const& column_chunk_offsets,
  std::vector<size_type> const& chunk_source_map,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  // Construct per source byte ranges in chunk iteration order
  std::vector<std::vector<byte_range_info>> source_byte_ranges(sources.size());
  std::for_each(cuda::counting_iterator(begin_chunk),
                cuda::counting_iterator(end_chunk),
                [&](auto const chunk) {
                  auto const source_idx = chunk_source_map[chunk];
                  source_byte_ranges[source_idx].emplace_back(
                    static_cast<int64_t>(column_chunk_offsets[chunk]),
                    static_cast<int64_t>(chunks[chunk].compressed_size));
                });

  // Wrap datasource references
  std::vector<std::reference_wrapper<datasource>> datasource_refs;
  datasource_refs.reserve(sources.size());
  for (auto const& source : sources) {
    datasource_refs.emplace_back(std::ref(*source));
  }

  // Read byte ranges into buffers
  auto [buffers, data_per_source, read_task] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    {datasource_refs.data(), datasource_refs.size()},
    {source_byte_ranges.data(), source_byte_ranges.size()},
    cudf::io::parquet::io_submission_policy::INTERLEAVE,
    stream,
    mr);

  // Extract data pointers from returned spans
  size_t range_idx = 0;
  std::for_each(cuda::counting_iterator(begin_chunk),
                cuda::counting_iterator(end_chunk),
                [&](auto const chunk) {
                  auto const src_idx            = chunk_source_map[chunk];
                  chunks[chunk].compressed_data = data_per_source[src_idx][range_idx++].data();
                  if (range_idx == data_per_source[src_idx].size()) { range_idx = 0; }
                });

  // Transfer device buffer ownership to the page data span
  CUDF_EXPECTS(page_data.size() == buffers.size(),
               "Parquet reader encountered a mismatch in size of page data and read buffers");

  std::move(buffers.begin(), buffers.end(), page_data.begin());

  return std::move(read_task);
}

[[nodiscard]] size_t count_page_headers_with_pgidx(
  cudf::detail::hostdevice_span<ColumnChunkDesc> chunks, cuda::stream_ref stream)
{
  auto const total_pages =
    std::accumulate(chunks.host_begin(), chunks.host_end(), size_t{0}, [](size_t sum, auto& chunk) {
      CUDF_EXPECTS(chunk.h_chunk_info != nullptr, "Expected non-null column info struct");
      auto const& chunk_info = *(chunk.h_chunk_info);
      chunk.num_dict_pages   = chunk_info.has_dictionary() ? 1 : 0;
      chunk.num_data_pages   = chunk_info.pages.size();
      return sum + chunk.num_data_pages + chunk.num_dict_pages;
    });

  // Push the updated page counts to device like `compute_page_spans()`
  chunks.host_to_device_async(stream);

  return total_pages;
}

std::string encoding_to_string(Encoding encoding)
{
  switch (encoding) {
    case Encoding::PLAIN: return "PLAIN";
    case Encoding::GROUP_VAR_INT: return "GROUP_VAR_INT";
    case Encoding::PLAIN_DICTIONARY: return "PLAIN_DICTIONARY";
    case Encoding::RLE: return "RLE";
    case Encoding::BIT_PACKED: return "BIT_PACKED";
    case Encoding::DELTA_BINARY_PACKED: return "DELTA_BINARY_PACKED";
    case Encoding::DELTA_LENGTH_BYTE_ARRAY: return "DELTA_LENGTH_BYTE_ARRAY";
    case Encoding::DELTA_BYTE_ARRAY: return "DELTA_BYTE_ARRAY";
    case Encoding::RLE_DICTIONARY: return "RLE_DICTIONARY";
    case Encoding::BYTE_STREAM_SPLIT: return "BYTE_STREAM_SPLIT";
    case Encoding::NUM_ENCODINGS:
    default: return "UNKNOWN(" + std::to_string(static_cast<int>(encoding)) + ")";
  }
}

[[nodiscard]] std::string encoding_bitmask_to_str(uint32_t encoding_bitmask)
{
  std::bitset<32> bits(encoding_bitmask);
  std::string result;

  for (size_t i = 0; i < bits.size(); ++i) {
    if (bits.test(i)) {
      auto const current = static_cast<Encoding>(i);
      if (!is_supported_encoding(current)) { result.append(encoding_to_string(current) + " "); }
    }
  }
  return result;
}

[[nodiscard]] std::string list_unsupported_encodings(device_span<PageInfo const> pages,
                                                     cuda::stream_ref stream)
{
  cudf::memory_resources mr = cudf::get_current_device_resource_ref();
  auto const to_mask        = cuda::proclaim_return_type<uint32_t>([] __device__(auto const& page) {
    return is_supported_encoding(page.encoding) ? uint32_t{0} : encoding_to_mask(page.encoding);
  });
  auto const unsupported    = cudf::detail::transform_reduce(pages.begin(),
                                                          pages.end(),
                                                          to_mask,
                                                          uint32_t{0},
                                                          cuda::std::bit_or<uint32_t>(),
                                                          stream,
                                                          mr.as_temporary());
  return encoding_bitmask_to_str(unsupported);
}

cudf::detail::hostdevice_vector<PageInfo> sort_pages(device_span<PageInfo const> unsorted_pages,
                                                     device_span<ColumnChunkDesc const> chunks,
                                                     cuda::stream_ref stream,
                                                     cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();
  CUDF_FUNC_RANGE();

  // sort the pages in chunk/schema order. we use chunk.src_col_index instead of
  // chunk.src_col_schema because the user may have reordered them (reading columns, "a" and "b" but
  // returning them as "b" and "a")
  //
  // ordering of pages is by input column schema, repeated across row groups.  so
  // if we had 3 columns, each with 2 pages, and 1 row group, our schema values might look like
  //
  // 1, 1, 2, 2, 3, 3
  //
  // However, if we had more than one row group, the pattern would be
  //
  // 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3
  // ^ row group 0     |
  //                   ^ row group 1
  //
  // To process pages by key (exclusive_scan_by_key, reduce_by_key, etc), the ordering we actually
  // want is
  //
  // 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
  //
  // We also need to preserve key-relative page ordering, so we need to use a stable sort.
  cuda::device_buffer<int32_t> page_keys(stream, temp_mr, unsorted_pages.size(), cuda::no_init);
  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    unsorted_pages.begin(),
    unsorted_pages.end(),
    page_keys.begin(),
    cuda::proclaim_return_type<int32_t>([chunks = chunks.begin()] __device__(PageInfo const& page) {
      return chunks[page.chunk_idx].src_col_index;
    }));
  // we are doing this by sorting indices first and then transforming the output because nvcc
  // started generating kernels using too much shared memory when trying to sort the pages
  // directly.
  cuda::device_buffer<int32_t> sort_indices(stream, temp_mr, unsorted_pages.size(), cuda::no_init);
  thrust::sequence(
    rmm::exec_policy_nosync(stream, temp_mr), sort_indices.begin(), sort_indices.end(), 0);
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                             page_keys.begin(),
                             page_keys.end(),
                             sort_indices.begin(),
                             cuda::std::less<int>());
  auto pass_pages =
    cudf::detail::hostdevice_vector<PageInfo>(unsorted_pages.size(), stream, mr.get_output_mr());
  thrust::gather(rmm::exec_policy_nosync(stream, temp_mr),
                 sort_indices.begin(),
                 sort_indices.end(),
                 unsorted_pages.data(),
                 pass_pages.d_begin());
  stream.sync();
  return pass_pages;
}

namespace {

/**
 * @brief Construct metadata for a data page; dictionary slots remain zero-initialized.
 */
page_index_info make_page_index_info(page_info const& source, Type physical_type, size_t start_row)
{
  page_index_info page{};
  page.num_rows       = source.num_rows;
  page.chunk_row      = start_row;
  page.num_nulls      = source.num_nulls.value_or(0);
  page.num_valids     = source.num_valid.value_or(0);
  page.str_bytes      = source.var_bytes_size.value_or(0);
  page.has_value_info = source.num_nulls.has_value() and source.num_valid.has_value() and
                        (physical_type != Type::BYTE_ARRAY or source.var_bytes_size.has_value());
  return page;
}

/**
 * @brief Construct page_index_info metadata structs for each page.
 * To be applied to PageInfo later when decoding pages.
 *
 * @param[in] chunks Host span of column chunks
 * @param[in] num_pages Number of pages
 * @param[in] stream CUDA stream to use
 * @return Pinned vector of page index metadata
 */
auto construct_page_index_info(host_span<ColumnChunkDesc> chunks,
                               size_t num_pages,
                               cuda::stream_ref stream)
{
  auto page_indexes = cudf::detail::make_pinned_vector_async<page_index_info>(num_pages, stream);
  std::fill(page_indexes.begin(), page_indexes.end(), page_index_info{});

  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    auto const& chunk = chunks[c];
    CUDF_EXPECTS(chunk.h_chunk_info != nullptr, "Expected non-null column info struct");
    auto const& chunk_info = *chunk.h_chunk_info;
    size_t start_row       = 0;
    page_count += chunk.num_dict_pages;
    CUDF_EXPECTS(page_count + chunk_info.pages.size() <= num_pages,
                 "Page index metadata exceeds the number of pages");
    for (size_t p = 0; p < chunk_info.pages.size(); p++, page_count++) {
      auto const page = make_page_index_info(chunk_info.pages[p], chunk.physical_type, start_row);
      page_indexes[page_count] = page;
      start_row += page.num_rows;
    }
  }

  return page_indexes;
}

/**
 * @brief Compute the offset of each column chunk's first page in chunk-sorted page order
 *
 * @param chunks Device span of column chunks
 * @param stream CUDA stream to use
 * @param mr Memory resources used to allocate temporary and output device memory
 * @return Exclusive scan of the number of pages per chunk, with one extra entry for the total
 */
cuda::device_buffer<size_type> compute_chunk_page_offsets(device_span<ColumnChunkDesc const> chunks,
                                                          cuda::stream_ref stream,
                                                          cudf::memory_resources mr)
{
  cuda::device_buffer<size_type> chunk_page_offsets(
    stream, mr.get_output_mr(), chunks.size() + 1, cuda::no_init);
  auto iter = cuda::counting_iterator<size_t>{0};
  thrust::transform_exclusive_scan(
    rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
    iter,
    iter + chunks.size() + 1,
    chunk_page_offsets.begin(),
    cuda::proclaim_return_type<size_type>([chunks] __device__(size_t i) {
      return static_cast<size_type>(
        i >= chunks.size() ? 0 : chunks[i].num_data_pages + chunks[i].num_dict_pages);
    }),
    size_type{0},
    cuda::std::plus<size_type>{});
  return chunk_page_offsets;
}

/**
 * @brief Count the page headers of the column chunks and locate the data span of each page
 *
 * Page starts that do not fit in the fixed-size buffer are recorded by a second launch into an
 * exact-size buffer.
 *
 * @param chunks Host-device span of column chunks. Page counts are updated on host and device
 * @param stream CUDA stream to use
 * @param mr Memory resources used to allocate temporary and output device memory
 * @return Data span of each page, in chunk-sorted page order
 */
cuda::device_buffer<cudf::device_span<uint8_t const>> compute_page_spans(
  cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();

  // Maximum number of page header addresses recorded in counting phase
  constexpr size_t max_page_addrs = 1'000'000;

  // A column chunk can have at most one dictionary page and `num_values` data pages
  auto const max_num_pages = std::transform_reduce(
    chunks.host_begin(), chunks.host_end(), size_t{0}, std::plus{}, [](auto const& chunk) {
      return size_t{1} + chunk.num_values;
    });

  // Maximum number of page addresses recorded in counting phase
  auto const page_addrs_capacity = std::min<size_t>(max_num_pages, max_page_addrs);

  chunks.host_to_device_async(stream);

  // Initialize chunk page discovery progress
  cuda::device_buffer<chunk_page_info> chunk_page_infos(
    stream, temp_mr, chunks.size(), cuda::no_init);
  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    chunks.device_begin(),
    chunks.device_end(),
    chunk_page_infos.begin(),
    cuda::proclaim_return_type<chunk_page_info>([] __device__(ColumnChunkDesc const& chunk) {
      return chunk_page_info{.cursor           = chunk.compressed_data,
                             .num_values_found = 0,
                             .num_data_pages   = 0,
                             .num_dict_pages   = 0,
                             .is_done          = false};
    }));

  // Allocate device memory for page addresses
  cuda::device_buffer<cuda::std::uintptr_t> page_addrs(
    stream, temp_mr, page_addrs_capacity, cuda::no_init);
  cudf::detail::device_scalar<size_t> num_consumed_page_addrs(0, stream, temp_mr);

  kernel_error error_code(stream, mr.as_temporary());
  discover_page_headers(chunks,
                        chunk_page_infos,
                        page_addrs,
                        num_consumed_page_addrs.data(),
                        error_code.data(),
                        stream);
  // Synchronized by the error code below
  chunks.device_to_host_async(stream);

  // It's required to ignore unsupported encodings in this function
  // so that we can actually compile a list of all the unsupported encodings found
  // in the pages. That cannot be done here since we do not have the pages vector here.
  // see https://github.com/NVIDIA/cudf/pull/14453#pullrequestreview-1778346688
  if (auto const error = error_code.value_sync(stream);
      error != 0 and error != static_cast<uint32_t>(decode_error::UNSUPPORTED_ENCODING)) {
    CUDF_FAIL("Parquet header parsing failed with code(s) while counting page headers " +
              kernel_error::to_string(error));
  }

  // Compute the actual total number of pages
  auto const total_pages = std::transform_reduce(
    chunks.host_begin(), chunks.host_end(), size_t{0}, std::plus{}, [](auto const& chunk) {
      return static_cast<size_t>(chunk.num_data_pages) + static_cast<size_t>(chunk.num_dict_pages);
    });

  if (total_pages > page_addrs_capacity) {
    // Keep the recorded page addresses and record the rest
    cuda::device_buffer<cuda::std::uintptr_t> all_page_addrs(
      stream, temp_mr, total_pages, cuda::no_init);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(all_page_addrs.data(),
                                             page_addrs.data(),
                                             page_addrs_capacity * sizeof(cuda::std::uintptr_t),
                                             stream));
    page_addrs = std::move(all_page_addrs);
    num_consumed_page_addrs.set_value_async(page_addrs_capacity, stream);

    // Run second discovery phase to record  remaining page addresses
    discover_page_headers(chunks,
                          chunk_page_infos,
                          page_addrs,
                          num_consumed_page_addrs.data(),
                          error_code.data(),
                          stream);

    // Check if we have recorded all page addresses
    CUDF_EXPECTS(num_consumed_page_addrs.value(stream) == total_pages,
                 "Encountered a mismatch in the number of page headers");
  }

  // A chunk's data does not overlap other chunks, so its pages stay contiguous once sorted
  auto const sorted_page_addrs = device_span<cuda::std::uintptr_t>{page_addrs.data(), total_pages};
  thrust::sort(
    rmm::exec_policy_nosync(stream, temp_mr), sorted_page_addrs.begin(), sorted_page_addrs.end());

  auto const chunk_page_offsets = compute_chunk_page_offsets(chunks, stream, mr.as_temporary());

  cuda::device_buffer<cudf::device_span<uint8_t const>> page_spans(
    stream, mr.get_output_mr(), total_pages, cuda::no_init);

  CUDF_EXPECTS(std::cmp_less_equal(total_pages, std::numeric_limits<size_type>::max()),
               "Number of parquet pass pages overflow the int32_t limit",
               std::overflow_error);

  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(total_pages),
    page_spans.begin(),
    cuda::proclaim_return_type<cudf::device_span<uint8_t const>>(
      [chunks = chunks.device_begin(),
       chunk_page_offsets =
         device_span<size_type const>{chunk_page_offsets.data(), chunk_page_offsets.size()},
       page_addrs       = device_span<cuda::std::uintptr_t const>{sorted_page_addrs},
       chunk_page_infos = chunk_page_infos.data()] __device__(size_type page_idx) {
        auto const chunk_idx =
          cuda::std::distance(chunk_page_offsets.begin(),
                              cuda::std::upper_bound(
                                chunk_page_offsets.begin(), chunk_page_offsets.end(), page_idx)) -
          1;
        auto const page_addr =
          cuda::std::lower_bound(
            page_addrs.begin(),
            page_addrs.end(),
            reinterpret_cast<cuda::std::uintptr_t>(chunks[chunk_idx].compressed_data)) +
          (page_idx - chunk_page_offsets[chunk_idx]);
        auto const begin = reinterpret_cast<uint8_t const*>(*page_addr);
        auto const end   = page_idx + 1 < chunk_page_offsets[chunk_idx + 1]
                             ? reinterpret_cast<uint8_t const*>(*(page_addr + 1))
                             : chunk_page_infos[chunk_idx].cursor;
        return cudf::device_span<uint8_t const>{begin, static_cast<size_t>(end - begin)};
      }));
  return page_spans;
}

/**
 * @brief Decode page headers from corresponding page data spans and set up the pass pages
 *
 * @param pass Struct containing pass information
 * @param unsorted_pages Device span of page information to decode
 * @param page_data Device span of page data spans, in chunk order
 * @param page_indexes Device span of page index metadata
 * @param stream CUDA stream to use
 * @param mr Memory resources used to allocate temporary and output device memory
 */
void decode_page_headers_impl(pass_intermediate_data& pass,
                              device_span<PageInfo> unsorted_pages,
                              device_span<cudf::device_span<uint8_t const> const> page_data,
                              device_span<page_index_info const> page_indexes,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();
  CUDF_FUNC_RANGE();

  auto const chunks =
    device_span<ColumnChunkDesc const>(pass.chunks.device_ptr(), pass.chunks.size());
  auto const chunk_page_offsets = compute_chunk_page_offsets(chunks, stream, mr.as_temporary());

  kernel_error error_code(stream, mr.as_temporary());

  // Launch page header decode kernel
  decode_page_headers_from_page_data(
    chunks,
    unsorted_pages,
    page_data,
    device_span<size_type const>(chunk_page_offsets.data(), chunk_page_offsets.size()),
    page_indexes,
    error_code.data(),
    stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    if (BitAnd(error, decode_error::UNSUPPORTED_ENCODING) != 0) {
      auto const unsupported_str =
        ". With unsupported encodings found: " + list_unsupported_encodings(pass.pages, stream);
      CUDF_FAIL("Parquet header parsing failed with code(s) " + kernel_error::to_string(error) +
                unsupported_str);
    } else {
      CUDF_FAIL("Parquet header parsing failed with code(s) " + kernel_error::to_string(error));
    }
  }

  // Without a page index, compute each page's chunk-relative row index from the row counts of
  // the pages before it. Pages are in chunk order here
  if (page_indexes.empty()) {
    auto const page_keys = cuda::transform_iterator(unsorted_pages.begin(), get_page_chunk_idx{});
    auto const page_rows = cuda::transform_iterator(unsorted_pages.begin(), get_page_num_rows{});
    thrust::exclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, temp_mr),
      page_keys,
      page_keys + unsorted_pages.size(),
      page_rows,
      cuda::make_tabulate_output_iterator(set_chunk_row_fn{unsorted_pages.data()}));
  }

  // compute max bytes needed for level data
  auto const max_level_bits = std::transform_reduce(
    pass.chunks.begin(),
    pass.chunks.end(),
    int32_t{0},
    [](auto a, auto b) { return std::max(a, b); },
    [](auto const& chunk) {
      return std::max<int32_t>(chunk.level_bits[level_type::REPETITION],
                               chunk.level_bits[level_type::DEFINITION]);
    });
  pass.level_type_size =
    std::max<int32_t>(1, cudf::detail::div_rounding_up_safe(max_level_bits, 8));

  // sort the pages in chunk/schema order.
  pass.pages =
    sort_pages(unsorted_pages,
               device_span<ColumnChunkDesc const>(pass.chunks.device_ptr(), pass.chunks.size()),
               stream,
               mr);

  // compute offsets to each group of input pages.
  // page_keys:   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
  //
  // result:      0,          4,          8
  cuda::device_buffer<size_type> page_counts(stream, temp_mr, pass.pages.size() + 1, cuda::no_init);
  auto page_keys =
    make_page_key_iterator(device_span<PageInfo const>(pass.pages.device_ptr(), pass.pages.size()));
  auto const page_counts_end = cudf::detail::reduce_by_key(page_keys,
                                                           page_keys + pass.pages.size(),
                                                           cuda::make_constant_iterator(1),
                                                           cuda::make_discard_iterator(),
                                                           page_counts.begin(),
                                                           cuda::std::plus<>{},
                                                           stream,
                                                           mr)
                                 .second;
  auto const num_page_counts = page_counts_end - page_counts.begin();
  pass.page_offsets =
    cuda::device_buffer<size_type>(stream, temp_mr, num_page_counts + 1, cuda::no_init);
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         page_counts.begin(),
                         page_counts.begin() + num_page_counts + 1,
                         pass.page_offsets.begin());

  // setup dict_page for each chunk if necessary
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr),
                   pass.pages.d_begin(),
                   pass.pages.d_end(),
                   [chunks = pass.chunks.d_begin()] __device__(PageInfo const& p) {
                     if (p.flags & PAGEINFO_FLAGS_DICTIONARY) {
                       chunks[p.chunk_idx].dict_page = &p;
                     }
                   });

  pass.pages.device_to_host_async(stream);
  pass.chunks.device_to_host_async(stream);
  stream.sync();
}

}  // namespace

cuda::device_buffer<PageInfo> decode_page_headers_with_pgidx(pass_intermediate_data& pass,
                                                             cuda::stream_ref stream,
                                                             cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();

  auto const num_pages = count_page_headers_with_pgidx(pass.chunks, stream);
  // Zero out the pages as decoding may not write every byte of a `PageInfo`, and `sort_pages`
  // copies `PageInfo` as whole objects
  auto unsorted_pages =
    cudf::detail::make_zeroed_device_buffer_async<PageInfo>(num_pages, stream, mr);

  if (num_pages == 0) { return unsorted_pages; }

  // Collect page data spans and page index info from the offset index
  auto host_page_data =
    cudf::detail::make_pinned_vector_async<cudf::device_span<uint8_t const>>(num_pages, stream);
  auto page_indexes = cudf::detail::make_pinned_vector_async<page_index_info>(num_pages, stream);
  std::fill(page_indexes.begin(), page_indexes.end(), page_index_info{});
  size_t curr_page_idx = 0;

  std::for_each(pass.chunks.begin(), pass.chunks.end(), [&](auto const& chunk) {
    CUDF_EXPECTS(
      chunk.h_chunk_info and std::cmp_equal(chunk.h_chunk_info->pages.size(), chunk.num_data_pages),
      "Encountered invalid sized data page information in the page index");
    CUDF_EXPECTS(
      curr_page_idx + chunk.num_dict_pages + chunk.h_chunk_info->pages.size() <= num_pages,
      "Page index metadata exceeds the number of pages");

    // Column chunk buffer's data pointer
    auto data_ptr = const_cast<uint8_t*>(chunk.compressed_data);

    // Dictionary page, if any
    if (chunk.num_dict_pages) {
      CUDF_EXPECTS(chunk.h_chunk_info->dictionary_offset.has_value() and
                     chunk.h_chunk_info->dictionary_size.has_value(),
                   "Encountered missing dictionary page information in the page index");

      // Parquet spec requires that the dictionary page be the first column chunk page
      // if the column chunk has dictionary encoding
      // Ref: https://github.com/apache/parquet-format?tab=readme-ov-file#column-chunks
      CUDF_EXPECTS(std::cmp_less(chunk.h_chunk_info->dictionary_offset.value(),
                                 chunk.h_chunk_info->pages.front().location.offset),
                   "Encountered dictionary page located beyond the first data page");
      auto const dictionary_size = chunk.h_chunk_info->dictionary_size.value();
      CUDF_EXPECTS(dictionary_size >= 0, "Encountered invalid dictionary page size");

      host_page_data[curr_page_idx] = {data_ptr, static_cast<std::size_t>(dictionary_size)};
      ++curr_page_idx;
      data_ptr += dictionary_size;
    }

    // Data pages
    auto const num_data_pages = chunk.num_data_pages;
    size_t start_row          = 0;
    std::for_each(cuda::counting_iterator<int32_t>{0},
                  cuda::counting_iterator{num_data_pages},
                  [&](auto const page_idx) {
                    auto const page_size =
                      chunk.h_chunk_info->pages[page_idx].location.compressed_page_size;

                    CUDF_EXPECTS(page_size >= 0, "Encountered invalid data page size");

                    host_page_data[curr_page_idx] = {data_ptr, static_cast<std::size_t>(page_size)};
                    auto const index              = make_page_index_info(
                      chunk.h_chunk_info->pages[page_idx], chunk.physical_type, start_row);
                    page_indexes[curr_page_idx] = index;
                    start_row += index.num_rows;
                    ++curr_page_idx;
                    if (page_idx < num_data_pages - 1) {
                      data_ptr += chunk.h_chunk_info->pages[page_idx + 1].location.offset -
                                  chunk.h_chunk_info->pages[page_idx].location.offset;
                    }
                  });
  });

  // Check if we have data spans for all input pages
  CUDF_EXPECTS(std::cmp_equal(curr_page_idx, num_pages),
               "Expected page offsets to match total pages");

  // Copy page data spans and page index info to device
  auto const d_page_data = cudf::detail::make_device_buffer_async(host_page_data, stream, temp_mr);
  auto const d_page_indexes = cudf::detail::make_device_buffer_async(page_indexes, stream, temp_mr);

  decode_page_headers_impl(pass, unsorted_pages, d_page_data, d_page_indexes, stream, mr);

  return unsorted_pages;
}

cuda::device_buffer<PageInfo> decode_page_headers(pass_intermediate_data& pass,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr)
{
  auto const page_spans = compute_page_spans(pass.chunks, stream, mr.as_temporary());
  auto const num_pages  = page_spans.size();
  // Zero out the pages as decoding may not write every byte of a `PageInfo`, and `sort_pages`
  // copies `PageInfo` as whole objects
  auto unsorted_pages =
    cudf::detail::make_zeroed_device_buffer_async<PageInfo>(num_pages, stream, mr);
  if (num_pages > 0) { decode_page_headers_impl(pass, unsorted_pages, page_spans, {}, stream, mr); }
  return unsorted_pages;
}

void decode_sparse_page_headers(pass_intermediate_data& pass,
                                device_span<PageInfo> unsorted_pages,
                                std::span<cudf::device_span<uint8_t const> const> page_data,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  CUDF_EXPECTS(page_data.size() == unsorted_pages.size(),
               "Page span count must match the number of logical pages");
  auto const temp_mr        = mr.get_temporary_mr();
  auto const d_page_data    = cudf::detail::make_device_buffer_async(page_data, stream, temp_mr);
  auto const page_indexes   = construct_page_index_info(pass.chunks, unsorted_pages.size(), stream);
  auto const d_page_indexes = cudf::detail::make_device_buffer_async(page_indexes, stream, temp_mr);
  decode_page_headers_impl(pass, unsorted_pages, d_page_data, d_page_indexes, stream, mr);
}

}  // namespace cudf::io::parquet::detail
