/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/decompression.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/orc/reader_impl.hpp"
#include "io/orc/reader_impl_chunking.hpp"
#include "io/orc/reader_impl_helpers.hpp"
#include "io/utilities/hostdevice_span.hpp"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/utility>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <numeric>

namespace cudf::io::orc::detail {

namespace {

/**
 * @brief  Decompresses the stripe data, at stream granularity.
 *
 * Only the streams in the provided `stream_range` are decoded. That range is determined in
 * the previous steps, after splitting stripes into ranges to maintain memory usage to be
 * under data read limit.
 *
 * @param loaded_stripe_range Range of stripes that are already loaded in memory
 * @param stream_range Range of streams to be decoded
 * @param num_decode_stripes Number of stripes that the decoding streams belong to
 * @param compinfo Compression information for each stripe
 * @param compinfo_map A map to lookup compression info of streams
 * @param decompressor Block decompressor
 * @param stripe_data List of source stripe column data
 * @param stream_info List of stream to column mappings
 * @param chunks Vector of list of column chunk descriptors
 * @param row_groups Vector of list of row index descriptors
 * @param row_index_stride Distance between each row index
 * @param use_base_stride Whether to use base stride obtained from meta or use the computed value
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Device buffer to decompressed data
 */
rmm::device_buffer decompress_stripe_data(
  range const& loaded_stripe_range,
  range const& stream_range,
  std::size_t num_decode_stripes,
  cudf::detail::hostdevice_span<compressed_stream_info> compinfo,
  stream_source_map<stripe_level_comp_info> const& compinfo_map,
  orc_decompressor const& decompressor,
  host_span<rmm::device_buffer const> stripe_data,
  host_span<orc_stream_info const> stream_info,
  cudf::detail::hostdevice_2dvector<column_desc>& chunks,
  cudf::detail::hostdevice_2dvector<row_group>& row_groups,
  size_type row_index_stride,
  bool use_base_stride,
  rmm::cuda_stream_view stream)
{
  // Whether we have the comppression info precomputed.
  auto const compinfo_ready = not compinfo_map.empty();

  // Count the exact number of compressed blocks
  std::size_t num_compressed_blocks   = 0;
  std::size_t num_uncompressed_blocks = 0;
  std::size_t total_decomp_size       = 0;

  for (auto stream_idx = stream_range.begin; stream_idx < stream_range.end; ++stream_idx) {
    auto const& info = stream_info[stream_idx];

    auto& stream_comp_info = compinfo[stream_idx - stream_range.begin];
    stream_comp_info       = compressed_stream_info(
      static_cast<uint8_t const*>(
        stripe_data[info.source.stripe_idx - loaded_stripe_range.begin].data()) +
        info.dst_pos,
      info.length);
    if (compinfo_ready) {
      auto const& cached_comp_info                 = compinfo_map.at(info.source);
      stream_comp_info.num_compressed_blocks       = cached_comp_info.num_compressed_blocks;
      stream_comp_info.num_uncompressed_blocks     = cached_comp_info.num_uncompressed_blocks;
      stream_comp_info.max_uncompressed_size       = cached_comp_info.total_decomp_size;
      stream_comp_info.max_uncompressed_block_size = cached_comp_info.max_uncompressed_block_size;

      num_compressed_blocks += cached_comp_info.num_compressed_blocks;
      num_uncompressed_blocks += cached_comp_info.num_uncompressed_blocks;
      total_decomp_size += cached_comp_info.total_decomp_size;
    }
  }

  if (!compinfo_ready) {
    compinfo.host_to_device_async(stream);
    parse_compressed_stripe_data(compinfo.device_ptr(),
                                 compinfo.size(),
                                 decompressor.GetBlockSize(),
                                 decompressor.GetLog2MaxCompressionRatio(),
                                 stream);
    compinfo.device_to_host(stream);

    for (std::size_t i = 0; i < compinfo.size(); ++i) {
      num_compressed_blocks += compinfo[i].num_compressed_blocks;
      num_uncompressed_blocks += compinfo[i].num_uncompressed_blocks;
      total_decomp_size += compinfo[i].max_uncompressed_size;
    }
  }

  CUDF_EXPECTS(
    not((num_uncompressed_blocks + num_compressed_blocks > 0) and (total_decomp_size == 0)),
    "Inconsistent info on compression blocks");

  // Buffer needs to be padded.This is required by `decode_column_data_kernel`.
  rmm::device_buffer decomp_data(
    cudf::util::round_up_safe(total_decomp_size, BUFFER_PADDING_MULTIPLE), stream);

  // If total_decomp_size is zero, the input data may be just empty.
  // This is still a valid input, thus do not be panick.
  if (decomp_data.is_empty()) { return decomp_data; }

  rmm::device_uvector<device_span<uint8_t const>> inflate_in(
    num_compressed_blocks + num_uncompressed_blocks, stream);
  rmm::device_uvector<device_span<uint8_t>> inflate_out(
    num_compressed_blocks + num_uncompressed_blocks, stream);
  rmm::device_uvector<codec_exec_result> inflate_res(num_compressed_blocks, stream);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               inflate_res.begin(),
               inflate_res.end(),
               codec_exec_result{0, codec_status::FAILURE});

  // Parse again to populate the decompression input/output buffers
  std::size_t decomp_offset      = 0;
  uint32_t max_uncomp_block_size = 0;
  uint32_t start_pos             = 0;
  auto start_pos_uncomp          = (uint32_t)num_compressed_blocks;
  for (std::size_t i = 0; i < compinfo.size(); ++i) {
    auto dst_base                 = static_cast<uint8_t*>(decomp_data.data());
    compinfo[i].uncompressed_data = dst_base + decomp_offset;
    compinfo[i].dec_in_ctl        = inflate_in.data() + start_pos;
    compinfo[i].dec_out_ctl       = inflate_out.data() + start_pos;
    compinfo[i].dec_res      = {inflate_res.data() + start_pos, compinfo[i].num_compressed_blocks};
    compinfo[i].copy_in_ctl  = inflate_in.data() + start_pos_uncomp;
    compinfo[i].copy_out_ctl = inflate_out.data() + start_pos_uncomp;

    decomp_offset += compinfo[i].max_uncompressed_size;
    start_pos += compinfo[i].num_compressed_blocks;
    start_pos_uncomp += compinfo[i].num_uncompressed_blocks;
    max_uncomp_block_size =
      std::max(max_uncomp_block_size, compinfo[i].max_uncompressed_block_size);
  }

  compinfo.host_to_device_async(stream);
  parse_compressed_stripe_data(compinfo.device_ptr(),
                               compinfo.size(),
                               decompressor.GetBlockSize(),
                               decompressor.GetLog2MaxCompressionRatio(),
                               stream);

  // Value for checking whether we decompress successfully.
  // It doesn't need to be atomic as there is no race condition: we only write `true` if needed.
  cudf::detail::hostdevice_vector<bool> any_block_failure(1, stream);
  any_block_failure[0] = false;
  any_block_failure.host_to_device_async(stream);

  cudf::io::detail::decompress(
    decompressor.compression(),
    device_span<device_span<uint8_t const>>{inflate_in.data(), num_compressed_blocks},
    device_span<device_span<uint8_t>>{inflate_out.data(), num_compressed_blocks},
    inflate_res,
    max_uncomp_block_size,
    total_decomp_size,
    stream);

  // Check if any block has been failed to decompress.
  // Not using `thrust::any` or `thrust::count_if` to defer stream sync.
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<std::size_t>{0},
                   cuda::counting_iterator{inflate_res.size()},
                   [results           = inflate_res.begin(),
                    any_block_failure = any_block_failure.device_ptr()] __device__(auto const idx) {
                     if (results[idx].status != codec_status::SUCCESS) {
                       *any_block_failure = true;
                     }
                   });

  if (num_uncompressed_blocks > 0) {
    auto const inputs   = inflate_in.data() + num_compressed_blocks;
    auto const outputs  = inflate_out.data() + num_compressed_blocks;
    auto const src_iter = cuda::make_transform_iterator(
      cuda::counting_iterator<std::size_t>{0},
      cuda::proclaim_return_type<uint8_t const*>(
        [inputs] __device__(std::size_t i) { return inputs[i].data(); }));
    auto const dst_iter = cuda::make_transform_iterator(
      cuda::counting_iterator<std::size_t>{0},
      cuda::proclaim_return_type<uint8_t*>(
        [outputs] __device__(std::size_t i) { return outputs[i].data(); }));
    auto const size_iter = cuda::make_transform_iterator(
      cuda::counting_iterator<std::size_t>{0},
      cuda::proclaim_return_type<size_t>([inputs, outputs] __device__(std::size_t i) {
        return inputs[i].size() < outputs[i].size() ? inputs[i].size() : outputs[i].size();
      }));
    cudf::detail::batched_memcpy_async(
      src_iter, dst_iter, size_iter, num_uncompressed_blocks, stream);
  }

  // Copy without stream sync, thus need to wait for stream sync below to access.
  any_block_failure.device_to_host_async(stream);

  post_decompression_reassemble(compinfo.device_ptr(), compinfo.size(), stream);
  compinfo.device_to_host(stream);  // This also sync stream for `any_block_failure`.

  // We can check on host after stream synchronize
  CUDF_EXPECTS(not any_block_failure[0], "Error during decompression");

  auto const num_columns = chunks.size().second;

  // Update the stream information with the updated uncompressed info
  // TBD: We could update the value from the information we already
  // have in stream_info[], but using the gpu results also updates
  // max_uncompressed_size to the actual uncompressed size, or zero if
  // decompression failed.
  for (std::size_t i = 0; i < num_decode_stripes; ++i) {
    for (std::size_t j = 0; j < num_columns; ++j) {
      auto& chunk = chunks[i][j];
      for (int k = 0; k < CI_NUM_STREAMS; ++k) {
        if (chunk.strm_len[k] > 0 && chunk.strm_id[k] < compinfo.size()) {
          chunk.streams[k]  = compinfo[chunk.strm_id[k]].uncompressed_data;
          chunk.strm_len[k] = compinfo[chunk.strm_id[k]].max_uncompressed_size;
        }
      }
    }
  }

  if (row_groups.size().first) {
    chunks.host_to_device_async(stream);
    row_groups.host_to_device_async(stream);
    parse_row_group_index(row_groups.base_device_ptr(),
                          compinfo.device_ptr(),
                          chunks.base_device_ptr(),
                          num_columns,
                          num_decode_stripes,
                          row_index_stride,
                          use_base_stride,
                          stream);
  }

  return decomp_data;
}

/**
 * @brief Updates null mask of columns whose parent is a struct column.
 *
 * If struct column has null element, that row would be skipped while writing child column in ORC,
 * so we need to insert the missing null elements in child column. There is another behavior from
 * pyspark, where if the child column doesn't have any null elements, it will not have present
 * stream, so in that case parent null mask need to be copied to child column.
 *
 * @param chunks Vector of list of column chunk descriptors
 * @param out_buffers Output columns' device buffers
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource to use for device memory allocation
 */
void update_null_mask(cudf::detail::hostdevice_2dvector<column_desc>& chunks,
                      host_span<column_buffer> out_buffers,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  auto const num_stripes = chunks.size().first;
  auto const num_columns = chunks.size().second;
  bool is_mask_updated   = false;

  for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    if (chunks[0][col_idx].parent_validity_info.valid_map_base != nullptr) {
      if (not is_mask_updated) {
        chunks.device_to_host(stream);
        is_mask_updated = true;
      }

      auto parent_valid_map_base = chunks[0][col_idx].parent_validity_info.valid_map_base;
      auto child_valid_map_base  = out_buffers[col_idx].null_mask();
      auto child_mask_len =
        chunks[0][col_idx].column_num_rows - chunks[0][col_idx].parent_validity_info.null_count;
      auto parent_mask_len = chunks[0][col_idx].column_num_rows;

      if (child_valid_map_base != nullptr) {
        rmm::device_uvector<uint32_t> dst_idx(child_mask_len, stream);
        // Copy indexes at which the parent has valid value.
        cudf::detail::copy_if_async(
          cuda::counting_iterator<int64_t>{0},
          cuda::counting_iterator{parent_mask_len},
          dst_idx.begin(),
          [parent_valid_map_base] __device__(auto idx) -> bool {
            return bit_is_set(parent_valid_map_base, idx);
          },
          stream);

        auto merged_null_mask = cudf::detail::create_null_mask(
          parent_mask_len, mask_state::ALL_NULL, rmm::cuda_stream_view(stream), mr);
        auto merged_mask      = static_cast<bitmask_type*>(merged_null_mask.data());
        uint32_t* dst_idx_ptr = dst_idx.data();
        // Copy child valid bits from child column to valid indexes, this will merge both child
        // and parent null masks
        thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<std::size_t>{0},
                         cuda::counting_iterator<std::size_t>{0} + dst_idx.size(),
                         [child_valid_map_base, dst_idx_ptr, merged_mask] __device__(auto idx) {
                           if (bit_is_set(child_valid_map_base, idx)) {
                             cudf::set_bit(merged_mask, dst_idx_ptr[idx]);
                           };
                         });

        out_buffers[col_idx].set_null_mask(std::move(merged_null_mask));

      } else {
        // Since child column doesn't have a mask, copy parent null mask
        auto mask_size = bitmask_allocation_size_bytes(parent_mask_len);
        out_buffers[col_idx].set_null_mask(
          rmm::device_buffer(static_cast<void*>(parent_valid_map_base), mask_size, stream, mr));
      }
    }
  }

  if (is_mask_updated) {
    // Update chunks with pointers to column data which might have been changed.
    for (std::size_t stripe_idx = 0; stripe_idx < num_stripes; ++stripe_idx) {
      for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        auto& chunk          = chunks[stripe_idx][col_idx];
        chunk.valid_map_base = out_buffers[col_idx].null_mask();
      }
    }
    chunks.host_to_device_async(stream);
  }
}

/**
 * @brief Converts the stripe column data and outputs to columns.
 *
 * @param num_dicts Number of dictionary entries required
 * @param skip_rows Number of rows to offset from start
 * @param row_index_stride Distance between each row index
 * @param level Current nesting level being processed
 * @param d_tz_table Local time to UTC conversion table
 * @param chunks Vector of list of column chunk descriptors
 * @param row_groups Vector of list of row index descriptors
 * @param out_buffers Output columns' device buffers
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 */
void decode_stream_data(int64_t num_dicts,
                        int64_t skip_rows,
                        size_type row_index_stride,
                        std::size_t level,
                        table_device_view const& d_tz_table,
                        cudf::detail::hostdevice_2dvector<column_desc>& chunks,
                        cudf::detail::device_2dspan<row_group> row_groups,
                        std::vector<column_buffer>& out_buffers,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  auto const num_stripes = chunks.size().first;
  auto const num_columns = chunks.size().second;

  cuda::counting_iterator<int> col_idx_it(0);
  cuda::counting_iterator<int> stripe_idx_it(0);

  // Update chunks with pointers to column data
  std::for_each(stripe_idx_it, stripe_idx_it + num_stripes, [&](auto stripe_idx) {
    std::for_each(col_idx_it, col_idx_it + num_columns, [&](auto col_idx) {
      auto& chunk            = chunks[stripe_idx][col_idx];
      chunk.column_data_base = out_buffers[col_idx].data();
      chunk.valid_map_base   = out_buffers[col_idx].null_mask();
    });
  });

  // Allocate global dictionary for deserializing
  rmm::device_uvector<dictionary_entry> global_dict(num_dicts, stream);

  chunks.host_to_device_async(stream);
  decode_nulls_and_string_dictionaries(
    chunks.base_device_ptr(), global_dict.data(), num_columns, num_stripes, skip_rows, stream);

  if (level > 0) {
    // Update nullmasks for children if parent was a struct and had null mask
    update_null_mask(chunks, out_buffers, stream, mr);
  }

  cudf::detail::device_scalar<size_type> error_count(
    0, stream, cudf::get_current_device_resource_ref());
  decode_column_data(chunks.base_device_ptr(),
                     global_dict.data(),
                     row_groups,
                     num_columns,
                     num_stripes,
                     skip_rows,
                     d_tz_table,
                     row_groups.size().first,
                     row_index_stride,
                     level,
                     error_count.data(),
                     stream);
  chunks.device_to_host_async(stream);
  // `value` synchronizes
  auto const num_errors = error_count.value(stream);
  CUDF_EXPECTS(num_errors == 0, "ORC data decode failed");

  std::for_each(col_idx_it + 0, col_idx_it + num_columns, [&](auto col_idx) {
    out_buffers[col_idx].null_count() =
      std::accumulate(stripe_idx_it + 0,
                      stripe_idx_it + num_stripes,
                      0,
                      [&](auto null_count, auto const stripe_idx) {
                        return null_count + chunks[stripe_idx][col_idx].null_count;
                      });
  });
}

/**
 * @brief Compute the per-stripe prefix sum of null count, for each struct column in the current
 * layer.
 */
void scan_null_counts(cudf::detail::hostdevice_2dvector<column_desc> const& chunks,
                      uint32_t* d_prefix_sums,
                      rmm::cuda_stream_view stream)
{
  auto const num_stripes = chunks.size().first;
  if (num_stripes == 0) return;

  auto const num_columns = chunks.size().second;
  auto const num_struct_cols =
    std::count_if(chunks[0].begin(), chunks[0].end(), [](auto const& chunk) {
      return chunk.type_kind == STRUCT;
    });
  auto prefix_sums_to_update =
    cudf::detail::make_empty_host_vector<cuda::std::pair<size_type, uint32_t*>>(num_struct_cols,
                                                                                stream);
  for (auto col_idx = 0ul; col_idx < num_columns; ++col_idx) {
    // Null counts sums are only needed for children of struct columns
    if (chunks[0][col_idx].type_kind == STRUCT) {
      prefix_sums_to_update.push_back({col_idx, d_prefix_sums + num_stripes * col_idx});
    }
  }
  auto const d_prefix_sums_to_update = cudf::detail::make_device_uvector_async(
    prefix_sums_to_update, stream, cudf::get_current_device_resource_ref());

  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   d_prefix_sums_to_update.begin(),
                   d_prefix_sums_to_update.end(),
                   [num_stripes, chunks = chunks.device_view()] __device__(auto const& idx_psums) {
                     auto const col_idx = idx_psums.first;
                     auto const psums   = idx_psums.second;
                     thrust::transform(
                       thrust::seq,
                       cuda::counting_iterator<std::size_t>{0ul},
                       cuda::counting_iterator<std::size_t>{num_stripes},
                       psums,
                       [&](auto stripe_idx) { return chunks[stripe_idx][col_idx].null_count; });
                     thrust::inclusive_scan(thrust::seq, psums, psums + num_stripes, psums);
                   });
  // `prefix_sums_to_update` goes out of scope, copy has to be done before we return
  stream.synchronize();
}

/**
 * @brief Aggregate child metadata from parent column chunks.
 */
void aggregate_child_meta(std::size_t level,
                          column_hierarchy const& selected_columns,
                          cudf::detail::host_2dspan<column_desc> chunks,
                          cudf::detail::host_2dspan<row_group> row_groups,
                          host_span<orc_column_meta const> nested_cols,
                          host_span<column_buffer> out_buffers,
                          reader_column_meta& col_meta)
{
  auto const num_of_stripes         = chunks.size().first;
  auto const num_of_rowgroups       = row_groups.size().first;
  auto const num_child_cols         = selected_columns.levels[level + 1].size();
  auto const number_of_child_chunks = num_child_cols * num_of_stripes;
  auto& num_child_rows              = col_meta.num_child_rows;
  auto& parent_column_data          = col_meta.parent_column_data;

  // Reset the meta to store child column details.
  num_child_rows.resize(selected_columns.levels[level + 1].size());
  std::fill(num_child_rows.begin(), num_child_rows.end(), 0);
  parent_column_data.resize(number_of_child_chunks);
  col_meta.parent_column_index.resize(number_of_child_chunks);
  col_meta.child_start_row.resize(number_of_child_chunks);
  col_meta.num_child_rows_per_stripe.resize(number_of_child_chunks);
  col_meta.rwgrp_meta.resize(num_of_rowgroups * num_child_cols);

  auto child_start_row =
    cudf::detail::host_2dspan<int64_t>(col_meta.child_start_row, num_child_cols);
  auto num_child_rows_per_stripe =
    cudf::detail::host_2dspan<int64_t>(col_meta.num_child_rows_per_stripe, num_child_cols);
  auto rwgrp_meta = cudf::detail::host_2dspan<reader_column_meta::row_group_meta>(
    col_meta.rwgrp_meta, num_child_cols);

  int index = 0;  // number of child column processed

  // For each parent column, update its child column meta for each stripe.
  std::for_each(nested_cols.begin(), nested_cols.end(), [&](auto const p_col) {
    auto const parent_col_idx = col_meta.orc_col_map[level][p_col.id];

    int64_t start_row         = 0;
    auto processed_row_groups = 0;

    for (std::size_t stripe_id = 0; stripe_id < num_of_stripes; stripe_id++) {
      // Aggregate num_rows and start_row from processed parent columns per row groups
      if (num_of_rowgroups) {
        auto stripe_num_row_groups = chunks[stripe_id][parent_col_idx].num_rowgroups;
        auto processed_child_rows  = 0;

        for (std::size_t rowgroup_id = 0; rowgroup_id < stripe_num_row_groups;
             rowgroup_id++, processed_row_groups++) {
          auto const child_rows = row_groups[processed_row_groups][parent_col_idx].num_child_rows;
          for (size_type id = 0; id < p_col.num_children; id++) {
            auto const child_col_idx                                  = index + id;
            rwgrp_meta[processed_row_groups][child_col_idx].start_row = processed_child_rows;
            rwgrp_meta[processed_row_groups][child_col_idx].num_rows  = child_rows;
          }
          processed_child_rows += child_rows;
        }
      }

      // Aggregate start row, number of rows per chunk and total number of rows in a column
      auto const child_rows = chunks[stripe_id][parent_col_idx].num_child_rows;

      for (size_type id = 0; id < p_col.num_children; id++) {
        auto const child_col_idx = index + id;

        num_child_rows[child_col_idx] += child_rows;

        // The number of rows in child column should not be very large otherwise we will have
        // size overflow.
        // If that is the case, we need to set a read limit to reduce number of decoding stripes.
        CUDF_EXPECTS(num_child_rows[child_col_idx] <=
                       static_cast<int64_t>(std::numeric_limits<size_type>::max()),
                     "Number of rows in the child column exceeds column size limit.");

        num_child_rows_per_stripe[stripe_id][child_col_idx] = child_rows;
        // start row could be different for each column when there is nesting at each stripe level
        child_start_row[stripe_id][child_col_idx] = (stripe_id == 0) ? 0 : start_row;
      }
      start_row += child_rows;
    }

    // Parent column null mask and null count would be required for child column
    // to adjust its nullmask.
    auto type              = out_buffers[parent_col_idx].type.id();
    auto parent_null_count = static_cast<uint32_t>(out_buffers[parent_col_idx].null_count());
    auto parent_valid_map  = out_buffers[parent_col_idx].null_mask();
    auto num_rows          = out_buffers[parent_col_idx].size;

    for (size_type id = 0; id < p_col.num_children; id++) {
      auto const child_col_idx                    = index + id;
      col_meta.parent_column_index[child_col_idx] = parent_col_idx;
      if (type == type_id::STRUCT) {
        parent_column_data[child_col_idx] = {parent_valid_map, parent_null_count};
        // Number of rows in child will remain same as parent in case of struct column
        num_child_rows[child_col_idx] = num_rows;
      } else {
        parent_column_data[child_col_idx] = {nullptr, 0};
      }
    }
    index += p_col.num_children;
  });
}

/**
 * @brief struct to store buffer data and size of list buffer
 */
struct list_buffer_data {
  size_type* data;
  size_type size;
};

// Generates offsets for list buffer from number of elements in a row.
void generate_offsets_for_list(host_span<list_buffer_data> buff_data, rmm::cuda_stream_view stream)
{
  for (auto& list_data : buff_data) {
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           list_data.data,
                           list_data.data + list_data.size,
                           list_data.data);
  }
}

/**
 * @brief Find the splits of the input table such that each split range of rows has data size less
 * than a given `size_limit`.
 *
 * The parameter `segment_length` is to control the granularity of splits. The output ranges will
 * always have numbers of rows that are multiple of this value, except the last range that contains
 * the remaining rows.
 *
 * Similar to `find_splits`, the given limit is just a soft limit. This function will never output
 * empty ranges, even they have sizes exceed the value of `size_limit`.
 *
 * @param input The input table to find splits
 * @param segment_length Value to control granularity of the output ranges
 * @param size_limit A limit on the output size of each split range
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A vector of ranges as splits of the input
 */
std::vector<range> find_table_splits(table_view const& input,
                                     size_type segment_length,
                                     std::size_t size_limit,
                                     rmm::cuda_stream_view stream)
{
  if (size_limit == 0) {
    return std::vector<range>{range{0, static_cast<std::size_t>(input.num_rows())}};
  }

  CUDF_EXPECTS(segment_length > 0, "Invalid segment_length", std::invalid_argument);

  // `segmented_row_bit_count` requires that `segment_length` is not larger than number of rows.
  segment_length = std::min(segment_length, input.num_rows());

  auto const d_segmented_sizes = cudf::detail::segmented_row_bit_count(
    input, segment_length, stream, cudf::get_current_device_resource_ref());

  auto segmented_sizes =
    cudf::detail::hostdevice_vector<cumulative_size>(d_segmented_sizes->size(), stream);

  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{d_segmented_sizes->size()},
    segmented_sizes.d_begin(),
    [segment_length,
     num_rows = input.num_rows(),
     d_sizes  = d_segmented_sizes->view().begin<size_type>()] __device__(auto const segment_idx) {
      // Since the number of rows may not divisible by segment_length,
      // the last segment may be shorter than the others.
      auto const current_length = min(segment_length, num_rows - segment_length * segment_idx);
      auto const size = d_sizes[segment_idx] / CHAR_BIT;  // divide by CHAR_BIT to get size in bytes
      return cumulative_size{static_cast<std::size_t>(current_length),
                             static_cast<std::size_t>(size)};
    });

  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         segmented_sizes.d_begin(),
                         segmented_sizes.d_end(),
                         segmented_sizes.d_begin(),
                         cumulative_size_plus{});
  segmented_sizes.device_to_host(stream);

  return find_splits<cumulative_size>(segmented_sizes, input.num_rows(), size_limit);
}

}  // namespace

void reader_impl::decompress_and_decode_stripes(read_mode mode)
{
  if (!_file_itm_data.has_data()) { return; }

  CUDF_EXPECTS(_chunk_read_data.curr_load_stripe_range > 0, "There is not any stripe loaded.");

  auto const stripe_range =
    _chunk_read_data.decode_stripe_ranges[_chunk_read_data.curr_decode_stripe_range++];
  auto const stripe_start = stripe_range.begin;
  auto const stripe_end   = stripe_range.end;
  auto const stripe_count = stripe_range.size();

  // The start index of loaded stripes. They are different from decoding stripes.
  auto const load_stripe_range =
    _chunk_read_data.load_stripe_ranges[_chunk_read_data.curr_load_stripe_range - 1];
  auto const load_stripe_start = load_stripe_range.begin;

  auto const rows_to_skip      = _file_itm_data.rows_to_skip;
  auto const& selected_stripes = _file_itm_data.selected_stripes;

  // Number of rows to decode in this decompressing/decoding step.
  int64_t rows_to_decode = 0;
  for (auto stripe_idx = stripe_start; stripe_idx < stripe_end; ++stripe_idx) {
    auto const& stripe     = selected_stripes[stripe_idx];
    auto const stripe_rows = static_cast<int64_t>(stripe.stripe_info->numberOfRows);
    rows_to_decode += stripe_rows;
  }

  CUDF_EXPECTS(rows_to_decode > rows_to_skip, "Invalid rows_to_decode computation.");
  rows_to_decode = std::min<int64_t>(rows_to_decode - rows_to_skip, _file_itm_data.rows_to_read);

  // After this step, we no longer have any rows to skip.
  // The number of rows remains to read in the future also reduced.
  _file_itm_data.rows_to_skip = 0;
  _file_itm_data.rows_to_read -= rows_to_decode;

  // Technically, overflow here should never happen because the `load_next_stripe_data()` step
  // already handled it by splitting the loaded stripe range into multiple decode ranges.
  CUDF_EXPECTS(rows_to_decode <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
               "Number or rows to decode exceeds the column size limit.",
               std::overflow_error);

  auto const tz_table_dptr = table_device_view::create(_file_itm_data.tz_table->view(), _stream);
  auto const num_levels    = _selected_columns.num_levels();
  _out_buffers.resize(num_levels);

  // Column descriptors ('chunks').
  // Each 'chunk' of data here corresponds to an orc column, in a stripe, at a nested level.
  // Unfortunately we cannot create one hostdevice_vector to use for all levels because
  // currently we do not have a hostdevice_2dspan class.
  std::vector<cudf::detail::hostdevice_2dvector<column_desc>> lvl_chunks(num_levels);

  // For computing null count.
  auto null_count_prefix_sums = [&] {
    auto const num_total_cols = std::accumulate(
      _selected_columns.levels.begin(),
      _selected_columns.levels.end(),
      std::size_t{0},
      [](auto const& sum, auto const& cols_level) { return sum + cols_level.size(); });

    return cudf::detail::make_zeroed_device_uvector_async<uint32_t>(
      num_total_cols * stripe_count, _stream, cudf::get_current_device_resource_ref());
  }();
  std::size_t num_processed_lvl_columns      = 0;
  std::size_t num_processed_prev_lvl_columns = 0;

  // For parsing decompression data.
  // We create one hostdevice_vector that is large enough to use for all levels,
  // thus only need to allocate memory once.
  auto hd_compinfo = [&] {
    std::size_t max_num_streams{0};
    if (_metadata.per_file_metadata[0].ps.compression != NONE) {
      // Find the maximum number of streams in all levels of the decoding stripes.
      for (std::size_t level = 0; level < num_levels; ++level) {
        auto const stream_range =
          merge_selected_ranges(_file_itm_data.lvl_stripe_stream_ranges[level], stripe_range);
        max_num_streams = std::max(max_num_streams, stream_range.size());
      }
    }
    return cudf::detail::hostdevice_vector<compressed_stream_info>{max_num_streams, _stream};
  }();

  auto& col_meta = *_col_meta;
  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    auto const& stripe_stream_ranges = _file_itm_data.lvl_stripe_stream_ranges[level];
    auto const stream_range          = merge_selected_ranges(stripe_stream_ranges, stripe_range);

    auto const& columns_level = _selected_columns.levels[level];
    auto const& stream_info   = _file_itm_data.lvl_stream_info[level];
    auto const& column_types  = _file_itm_data.lvl_column_types[level];
    auto const& nested_cols   = _file_itm_data.lvl_nested_cols[level];

    auto& stripe_data = _file_itm_data.lvl_stripe_data[level];
    auto& chunks      = lvl_chunks[level];

    auto const num_lvl_columns = columns_level.size();
    chunks = cudf::detail::hostdevice_2dvector<column_desc>(stripe_count, num_lvl_columns, _stream);
    memset(chunks.base_host_ptr(), 0, chunks.size_bytes());

    bool const use_index =
      _options.use_index &&
      // Do stripes have row group index
      _metadata.is_row_grp_idx_present() &&
      // Only use if we don't have much work with complete columns & stripes
      // TODO: Consider nrows, gpu, and tune the threshold
      (rows_to_decode > _metadata.get_row_index_stride() &&
       !(_metadata.get_row_index_stride() & 7) && _metadata.get_row_index_stride() != 0 &&
       num_lvl_columns * stripe_count < 8 * 128) &&
      // Only use if first row is aligned to a stripe boundary
      // TODO: Fix logic to handle unaligned rows
      (rows_to_skip == 0);

    // 0-based counters, used across all decoding stripes in this step.
    int64_t stripe_start_row{0};
    int64_t num_dict_entries{0};
    uint32_t num_rowgroups{0};
    std::size_t local_stream_order{0};

    for (auto stripe_idx = stripe_start; stripe_idx < stripe_end; ++stripe_idx) {
      auto const& stripe       = selected_stripes[stripe_idx];
      auto const stripe_info   = stripe.stripe_info;
      auto const stripe_footer = stripe.stripe_footer;

      // Normalize stripe_idx to 0-based.
      auto const stripe_local_idx = stripe_idx - stripe_start;

      // The first parameter (`stripe_order`) must be normalized to 0-based.
      auto const total_data_size = gather_stream_info_and_column_desc(stripe_local_idx,
                                                                      level,
                                                                      stripe_info,
                                                                      stripe_footer,
                                                                      col_meta.orc_col_map[level],
                                                                      _metadata.get_types(),
                                                                      use_index,
                                                                      level == 0,
                                                                      &num_dict_entries,
                                                                      &local_stream_order,
                                                                      nullptr,  // stream_info
                                                                      &chunks);

      auto const is_stripe_data_empty = total_data_size == 0;
      CUDF_EXPECTS(not is_stripe_data_empty or stripe_info->indexLength == 0,
                   "Invalid index rowgroup stream data");

      auto const dst_base =
        static_cast<uint8_t*>(stripe_data[stripe_idx - load_stripe_start].data());
      auto const num_rows_in_stripe = static_cast<int64_t>(stripe_info->numberOfRows);

      uint32_t const rowgroup_id = num_rowgroups;
      uint32_t const stripe_num_rowgroups =
        use_index ? (num_rows_in_stripe + _metadata.get_row_index_stride() - 1) /
                      _metadata.get_row_index_stride()
                  : 0;

      // Update chunks to reference streams pointers.
      for (std::size_t col_idx = 0; col_idx < num_lvl_columns; col_idx++) {
        auto& chunk = chunks[stripe_local_idx][col_idx];
        // start row, number of rows in a each stripe and total number of rows
        // may change in lower levels of nesting
        chunk.start_row =
          (level == 0) ? stripe_start_row
                       : col_meta.child_start_row[stripe_local_idx * num_lvl_columns + col_idx];
        chunk.num_rows =
          (level == 0)
            ? num_rows_in_stripe
            : col_meta.num_child_rows_per_stripe[stripe_local_idx * num_lvl_columns + col_idx];
        chunk.column_num_rows = (level == 0) ? rows_to_decode : col_meta.num_child_rows[col_idx];
        chunk.parent_validity_info =
          (level == 0) ? column_validity_info{} : col_meta.parent_column_data[col_idx];
        chunk.parent_null_count_prefix_sums =
          (level == 0) ? nullptr
                       : null_count_prefix_sums.data() + (num_processed_prev_lvl_columns +
                                                          col_meta.parent_column_index[col_idx]) *
                                                           stripe_count;
        chunk.encoding_kind = stripe_footer->columns[columns_level[col_idx].id].kind;
        chunk.type_kind =
          _metadata.per_file_metadata[stripe.source_idx].ff.types[columns_level[col_idx].id].kind;

        // num_child_rows for a struct column will be same, for other nested types it will be
        // calculated.
        chunk.num_child_rows = (chunk.type_kind != STRUCT) ? 0 : chunk.num_rows;
        chunk.dtype_id       = column_types[col_idx].id();
        chunk.decimal_scale  = _metadata.per_file_metadata[stripe.source_idx]
                                .ff.types[columns_level[col_idx].id]
                                .scale.value_or(0);

        chunk.rowgroup_id   = rowgroup_id;
        chunk.dtype_len     = (column_types[col_idx].id() == type_id::STRING)
                                ? sizeof(string_index_pair)
                              : ((column_types[col_idx].id() == type_id::LIST) or
                             (column_types[col_idx].id() == type_id::STRUCT))
                                ? sizeof(size_type)
                                : cudf::size_of(column_types[col_idx]);
        chunk.num_rowgroups = stripe_num_rowgroups;

        if (chunk.type_kind == TIMESTAMP) {
          chunk.timestamp_type_id = _options.timestamp_type.id();
        }
        if (not is_stripe_data_empty) {
          for (int k = 0; k < CI_NUM_STREAMS; k++) {
            chunk.streams[k] =
              dst_base + stream_info[chunk.strm_id[k] + stream_range.begin].dst_pos;
          }
        }
      }

      stripe_start_row += num_rows_in_stripe;
      num_rowgroups += stripe_num_rowgroups;
    }

    if (stripe_data.empty()) { continue; }

    // Process dataset chunks into output columns.
    auto row_groups =
      cudf::detail::hostdevice_2dvector<row_group>(num_rowgroups, num_lvl_columns, _stream);
    if (level > 0 and row_groups.size().first) {
      cudf::host_span<row_group> row_groups_span(row_groups.base_host_ptr(),
                                                 num_rowgroups * num_lvl_columns);
      auto& rw_grp_meta = col_meta.rwgrp_meta;

      // Update start row and num rows per row group
      std::transform(rw_grp_meta.begin(),
                     rw_grp_meta.end(),
                     row_groups_span.begin(),
                     rw_grp_meta.begin(),
                     [&](auto meta, auto& row_grp) {
                       row_grp.num_rows  = meta.num_rows;
                       row_grp.start_row = meta.start_row;
                       return meta;
                     });
    }

    // Setup row group descriptors if using indexes.
    if (_metadata.per_file_metadata[0].ps.compression != NONE) {
      auto const compinfo =
        cudf::detail::hostdevice_span<compressed_stream_info>{hd_compinfo}.subspan(
          0, stream_range.size());
      auto decomp_data = decompress_stripe_data(load_stripe_range,
                                                stream_range,
                                                stripe_count,
                                                compinfo,
                                                _file_itm_data.compinfo_map,
                                                *_metadata.per_file_metadata[0].decompressor,
                                                stripe_data,
                                                stream_info,
                                                chunks,
                                                row_groups,
                                                _metadata.get_row_index_stride(),
                                                level == 0,
                                                _stream);

      // Just save the decompressed data and clear out the raw data to free up memory.
      stripe_data[stripe_start - load_stripe_start] = std::move(decomp_data);
      for (std::size_t i = 1; i < stripe_count; ++i) {
        stripe_data[i + stripe_start - load_stripe_start] = {};
      }

    } else {
      if (row_groups.size().first) {
        chunks.host_to_device_async(_stream);
        row_groups.host_to_device_async(_stream);
        row_groups.host_to_device_async(_stream);
        parse_row_group_index(row_groups.base_device_ptr(),
                              nullptr,
                              chunks.base_device_ptr(),
                              num_lvl_columns,
                              stripe_count,
                              _metadata.get_row_index_stride(),
                              level == 0,
                              _stream);
      }
    }

    _out_buffers[level].resize(0);

    for (std::size_t i = 0; i < column_types.size(); ++i) {
      bool is_nullable = false;
      for (std::size_t j = 0; j < stripe_count; ++j) {
        if (chunks[j][i].strm_len[CI_PRESENT] != 0) {
          is_nullable = true;
          break;
        }
      }

      auto const is_list_type = (column_types[i].id() == type_id::LIST);
      auto const n_rows       = (level == 0) ? rows_to_decode : col_meta.num_child_rows[i];

      // For list column, offset column will be always size + 1.
      _out_buffers[level].emplace_back(
        column_types[i], is_list_type ? n_rows + 1 : n_rows, is_nullable, _stream, _mr);
    }

    decode_stream_data(num_dict_entries,
                       rows_to_skip,
                       _metadata.get_row_index_stride(),
                       level,
                       *tz_table_dptr,
                       chunks,
                       row_groups,
                       _out_buffers[level],
                       _stream,
                       _mr);

    if (nested_cols.size()) {
      // Extract information to process nested child columns.
      scan_null_counts(
        chunks, null_count_prefix_sums.data() + num_processed_lvl_columns * stripe_count, _stream);

      row_groups.device_to_host(_stream);
      aggregate_child_meta(
        level, _selected_columns, chunks, row_groups, nested_cols, _out_buffers[level], col_meta);

      // ORC stores number of elements at each row, so we need to generate offsets from that
      std::vector<list_buffer_data> buff_data;
      std::for_each(
        _out_buffers[level].begin(), _out_buffers[level].end(), [&buff_data](auto& out_buffer) {
          if (out_buffer.type.id() == type_id::LIST) {
            auto data = static_cast<size_type*>(out_buffer.data());
            buff_data.emplace_back(list_buffer_data{data, out_buffer.size});
          }
        });

      if (not buff_data.empty()) { generate_offsets_for_list(buff_data, _stream); }
    }
    num_processed_prev_lvl_columns = num_processed_lvl_columns;
    num_processed_lvl_columns += num_lvl_columns;
  }  // end loop level

  // Now generate a table from the decoded result.
  std::vector<std::unique_ptr<column>> out_columns;
  _out_metadata = get_meta_with_user_data();
  std::transform(
    _selected_columns.levels[0].begin(),
    _selected_columns.levels[0].end(),
    std::back_inserter(out_columns),
    [&](auto const& orc_col_meta) {
      _out_metadata.schema_info.emplace_back("");
      auto col_buffer = assemble_buffer(
        orc_col_meta.id, 0, *_col_meta, _metadata, _selected_columns, _out_buffers, _stream, _mr);
      return make_column(col_buffer, &_out_metadata.schema_info.back(), std::nullopt, _stream);
    });
  _chunk_read_data.decoded_table = std::make_unique<table>(std::move(out_columns));

  out_columns =
    cudf::structs::detail::enforce_null_consistency(std::move(out_columns), _stream, _mr);

  // Free up temp memory used for decoding.
  for (std::size_t level = 0; level < _selected_columns.num_levels(); ++level) {
    _out_buffers[level].resize(0);

    auto& stripe_data = _file_itm_data.lvl_stripe_data[level];
    if (_metadata.per_file_metadata[0].ps.compression != orc::NONE) {
      stripe_data[stripe_start - load_stripe_start] = {};
    } else {
      for (std::size_t i = 0; i < stripe_count; ++i) {
        stripe_data[i + stripe_start - load_stripe_start] = {};
      }
    }
  }

  // Output table range is reset to start from the first position.
  _chunk_read_data.curr_output_table_range = 0;

  // Split the decoded table into ranges that be output into chunks having size within the given
  // output size limit.
  _chunk_read_data.output_table_ranges = find_table_splits(_chunk_read_data.decoded_table->view(),
                                                           _chunk_read_data.output_row_granularity,
                                                           _chunk_read_data.chunk_read_limit,
                                                           _stream);
}

}  // namespace cudf::io::orc::detail
