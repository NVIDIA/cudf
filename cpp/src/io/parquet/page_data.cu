/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "page_data.cuh"
#include "page_decode.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

namespace cudf::io::parquet::detail {

namespace cg = cooperative_groups;

namespace {

constexpr int decode_block_size = 128;
constexpr int rolling_buf_size  = decode_block_size * 2;

/**
 * @brief Kernel for computing the BYTE_STREAM_SPLIT column data stored in the pages
 *
 * This is basically the PLAIN decoder, but with a pared down set of supported data
 * types, and using output functions that piece together the individual streams.
 * Supported physical types include INT32, INT64, FLOAT, DOUBLE and FIXED_LEN_BYTE_ARRAY.
 * The latter is currently only used for large decimals. The Parquet specification also
 * has FLOAT16 and UUID types that are currently not supported. FIXED_LEN_BYTE_ARRAY data
 * that lacks a `LogicalType` annotation will be handled by the string decoder.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @param page_mask Boolean vector indicating which pages need to be decoded
 * @param error_code Error code to set if an error is encountered
 */
template <int lvl_buf_size, typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_split_page_data_kernel(PageInfo* pages,
                                device_span<ColumnChunkDesc const> chunks,
                                size_t min_row,
                                size_t num_rows,
                                cudf::device_span<bool const> page_mask,
                                kernel_error::pointer error_code)
{
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16)
    page_state_buffers_s<rolling_buf_size, rolling_buf_size, rolling_buf_size>
      state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(block);

  // Exit early if the page is pruned
  if (not page_mask.empty() and not page_mask[page_idx]) { return; }

  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  // Setup local page info
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::BYTE_STREAM_SPLIT},
                             page_processing_stage::DECODE)) {
    return;
  }

  // Must be evaluated after setup_local_page_info
  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  auto const data_len   = cuda::std::distance(s->stream.data_start, s->stream.data_end);
  auto const num_values = data_len / s->dtype_len_in;

  // Check malformed BYTE_STREAM_SPLIT pages
  if (s->dtype_len_in <= 0 or data_len <= 0) {
    cg::invoke_one(block, [&]() {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_BYTE_STREAM_SPLIT_SIZE),
                error_code);
    });
    return;
  }

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->setup.page.skipped_leaf_values;
  while (s->setup.error == 0 &&
         (s->input_value_count < s->setup.num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (warp.meta_group_rank() == 0) {
      target_pos = cuda::std::min(src_pos + 2 * (decode_block_size - warp.size()),
                                  s->nz_count + (decode_block_size - warp.size()));
    } else {
      target_pos = cuda::std::min<int32_t>(s->nz_count, src_pos + decode_block_size - warp.size());
    }
    // This needs to be here to prevent warp 1 modifying src_pos before all threads have read it
    block.sync();

    if (warp.meta_group_rank() == 0) {
      // WARP0: decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<lvl_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else {
      // WARP1..WARP3: Decode values
      Type const dtype = s->setup.col.physical_type;
      src_pos += block.thread_rank() - warp.size();

      // The position in the output column/buffer
      int dst_pos = sb->nz_idx[rolling_index<rolling_buf_size>(src_pos)];

      // for the flat hierarchy case we will be reading from the beginning of the value stream,
      // regardless of the value of first_row. so adjust our destination offset accordingly.
      // example:
      // - user has passed skip_rows = 2, so our first_row to output is 2
      // - the row values we get from nz_idx will be
      //   0, 1, 2, 3, 4 ....
      // - by shifting these values by first_row, the sequence becomes
      //   -2, -1, 0, 1, 2 ...
      // - so we will end up ignoring the first two input rows, and input rows 2..n will
      //   get written to the output starting at position 0.
      //
      if (!has_repetition) { dst_pos -= s->setup.first_row; }

      // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
      // before first_row) in the flat hierarchy case.
      if (src_pos < target_pos && dst_pos >= 0) {
        // src_pos represents the logical row position we want to read from. But in the case of
        // nested hierarchies, there is no 1:1 mapping of rows to values.  So our true read position
        // has to take into account the # of values we have to skip in the page to get to the
        // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
        uint32_t val_src_pos = src_pos + skipped_leaf_values;

        // nesting level that is storing actual leaf values
        int leaf_level_index = s->setup.col.max_nesting_depth - 1;

        uint32_t dtype_len = s->dtype_len;
        uint8_t const* src = s->stream.data_start + val_src_pos;
        uint8_t* dst =
          nesting_info_base[leaf_level_index].data_out + static_cast<size_t>(dst_pos) * dtype_len;
        auto const is_decimal = s->setup.col.logical_type.has_value() and
                                s->setup.col.logical_type->type == LogicalType::DECIMAL;

        // Note: non-decimal FIXED_LEN_BYTE_ARRAY will be handled in the string reader
        if (is_decimal) {
          switch (dtype) {
            case Type::INT32: gpuOutputByteStreamSplit<int32_t>(dst, src, num_values); break;
            case Type::INT64: gpuOutputByteStreamSplit<int64_t>(dst, src, num_values); break;
            case Type::FIXED_LEN_BYTE_ARRAY:
              if (s->dtype_len_in <= sizeof(int32_t)) {
                gpuOutputSplitFixedLenByteArrayAsInt(
                  reinterpret_cast<int32_t*>(dst), src, num_values, s->dtype_len_in);
                break;
              } else if (s->dtype_len_in <= sizeof(int64_t)) {
                gpuOutputSplitFixedLenByteArrayAsInt(
                  reinterpret_cast<int64_t*>(dst), src, num_values, s->dtype_len_in);
                break;
              } else if (s->dtype_len_in <= sizeof(__int128_t)) {
                gpuOutputSplitFixedLenByteArrayAsInt(
                  reinterpret_cast<__int128_t*>(dst), src, num_values, s->dtype_len_in);
                break;
              }
              // unsupported decimal precision
              [[fallthrough]];

            default: s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
          }
        } else if (dtype_len == 8) {
          if (s->dtype_len_in == 4) {
            // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
            // TIME_MILLIS is the only duration type stored as int32:
            // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
            gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
            // zero out most significant bytes
            memset(dst + 4, 0, 4);
          } else if (s->ts_scale) {
            gpuOutputSplitInt64Timestamp(
              reinterpret_cast<int64_t*>(dst), src, num_values, s->ts_scale);
          } else {
            gpuOutputByteStreamSplit<int64_t>(dst, src, num_values);
          }
        } else if (dtype_len == 4) {
          gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
        } else {
          s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
        }
      }
      // Only the first thread in the warp 1 updates src_pos
      if (warp.meta_group_rank() == 1 and warp.thread_rank() == 0) { s->src_pos = target_pos; }
    }
    block.sync();
  }

  // Zero-fill null positions after decoding valid values
  if (has_repetition) {
    int const leaf_level_index = s->setup.col.max_nesting_depth - 1;
    auto const& ni             = s->nesting_info[leaf_level_index];
    if (ni.valid_map != nullptr) {
      int const num_values = ni.valid_map_offset - init_valid_map_offset;
      zero_fill_null_positions_shared<decode_block_size>(
        s, s->dtype_len, init_valid_map_offset, num_values, static_cast<int>(block.thread_rank()));
    }
  }

  if (s->setup.error != 0) {
    cg::invoke_one(block, [&]() { set_error(s->setup.error, error_code); });
  }
}

/**
 * @brief Kernel for computing the column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk. If necessary, additional
 * conversion will be performed to translate from the Parquet datatype to
 * desired output datatype (ex. 32-bit to 16-bit, string to hash).
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @param page_mask Boolean vector indicating which pages need to be decoded
 * @param error_code Error code to set if an error is encountered
 */
template <int lvl_buf_size, typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_page_data(PageInfo* pages,
                   device_span<ColumnChunkDesc const> chunks,
                   size_t min_row,
                   size_t num_rows,
                   cudf::device_span<bool const> page_mask,
                   kernel_error::pointer error_code)
{
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16)
    page_state_buffers_s<rolling_buf_size, rolling_buf_size, rolling_buf_size>
      state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(block);
  int out_warp_id;

  // Exit early if the page is pruned
  if (not page_mask.empty() and not page_mask[page_idx]) { return; }

  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  // Setup local page info
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::GENERAL},
                             page_processing_stage::DECODE)) {
    return;
  }

  // Must be evaluated after setup_local_page_info
  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  if (s->stream.dict_base) {
    out_warp_id = (s->stream.dict_bits > 0) ? 2 : 1;
  } else {
    switch (s->setup.col.physical_type) {
      case Type::BOOLEAN: [[fallthrough]];
      case Type::BYTE_ARRAY: [[fallthrough]];
      case Type::FIXED_LEN_BYTE_ARRAY: out_warp_id = 2; break;
      default: out_warp_id = 1;
    }
  }

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  auto const is_decimal = s->setup.col.logical_type.has_value() and
                          s->setup.col.logical_type->type == LogicalType::DECIMAL;
  Type const dtype = s->setup.col.physical_type;

  auto const first_out_thread_id = out_warp_id * warp.size();
  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->setup.page.skipped_leaf_values;
  while (s->setup.error == 0 &&
         (s->input_value_count < s->setup.num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (warp.meta_group_rank() < out_warp_id) {
      target_pos = cuda::std::min<int32_t>(src_pos + 2 * (decode_block_size - first_out_thread_id),
                                           s->nz_count + (decode_block_size - first_out_thread_id));
    } else {
      target_pos =
        cuda::std::min<int32_t>(s->nz_count, src_pos + decode_block_size - first_out_thread_id);
      if (out_warp_id > 1) { target_pos = cuda::std::min<int32_t>(target_pos, s->dict_pos); }
    }
    // this needs to be here to prevent warp 3 modifying src_pos before all threads have read it
    block.sync();
    if (warp.meta_group_rank() == 0) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<lvl_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() < out_warp_id) {
      // skipped_leaf_values will always be 0 for flat hierarchies.
      uint32_t src_target_pos = target_pos + skipped_leaf_values;

      // WARP1: Decode dictionary indices, booleans or string positions
      // NOTE: racecheck complains of a RAW error involving the s->dict_pos assignment below.
      // This is likely a false positive in practice, but could be solved by wrapping the next
      // 9 lines in `if (s->dict_pos < src_target_pos) {}`. If that change is made here, it will
      // be needed in the other DecodeXXX kernels.
      if (s->stream.dict_base) {
        src_target_pos =
          decode_dictionary_indices<is_calc_sizes_only::NO>(s, sb, src_target_pos, warp).first;
      } else if (s->setup.col.physical_type == Type::BOOLEAN) {
        src_target_pos = decode_rle_booleans(s, sb, src_target_pos, warp);
      } else if (s->setup.col.physical_type == Type::BYTE_ARRAY or
                 s->setup.col.physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
        initialize_string_descriptors<is_calc_sizes_only::NO>(s, sb, src_target_pos, warp);
      }
      if (warp.thread_rank() == 0) { s->dict_pos = src_target_pos; }
    } else {
      // WARP1..WARP3: Decode values
      src_pos += block.thread_rank() - first_out_thread_id;

      // the position in the output column/buffer
      int dst_pos = sb->nz_idx[rolling_index<rolling_buf_size>(src_pos)];

      // for the flat hierarchy case we will be reading from the beginning of the value stream,
      // regardless of the value of first_row. so adjust our destination offset accordingly.
      // example:
      // - user has passed skip_rows = 2, so our first_row to output is 2
      // - the row values we get from nz_idx will be
      //   0, 1, 2, 3, 4 ....
      // - by shifting these values by first_row, the sequence becomes
      //   -2, -1, 0, 1, 2 ...
      // - so we will end up ignoring the first two input rows, and input rows 2..n will
      //   get written to the output starting at position 0.
      //
      if (!has_repetition) { dst_pos -= s->setup.first_row; }

      // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
      // before first_row) in the flat hierarchy case.
      if (src_pos < target_pos && dst_pos >= 0) {
        // src_pos represents the logical row position we want to read from. But in the case of
        // nested hierarchies, there is no 1:1 mapping of rows to values.  So our true read position
        // has to take into account the # of values we have to skip in the page to get to the
        // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
        uint32_t val_src_pos = src_pos + skipped_leaf_values;

        // nesting level that is storing actual leaf values
        int const leaf_level_index = s->setup.col.max_nesting_depth - 1;

        uint32_t const dtype_len = s->dtype_len;
        void* dst =
          nesting_info_base[leaf_level_index].data_out + static_cast<size_t>(dst_pos) * dtype_len;

        if (dtype == Type::BYTE_ARRAY) {
          if (is_decimal) {
            auto const [ptr, len]        = gpuGetStringData(s, sb, val_src_pos);
            auto const decimal_precision = s->setup.col.logical_type->precision();
            if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
              gpuOutputByteArrayAsInt(ptr, len, static_cast<int32_t*>(dst));
            } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
              gpuOutputByteArrayAsInt(ptr, len, static_cast<int64_t*>(dst));
            } else {
              gpuOutputByteArrayAsInt(ptr, len, static_cast<__int128_t*>(dst));
            }
          } else {
            gpuOutputString(s, sb, val_src_pos, dst);
          }
        } else if (dtype == Type::BOOLEAN) {
          read_boolean(sb, val_src_pos, static_cast<uint8_t*>(dst));
        } else if (is_decimal) {
          switch (dtype) {
            case Type::INT32:
              read_fixed_width_value_fast(s, sb, val_src_pos, static_cast<uint32_t*>(dst));
              break;
            case Type::INT64:
              read_fixed_width_value_fast(s, sb, val_src_pos, static_cast<uint2*>(dst));
              break;
            default:
              if (s->dtype_len_in <= sizeof(int32_t)) {
                read_fixed_width_byte_array_as_int(s, sb, val_src_pos, static_cast<int32_t*>(dst));
              } else if (s->dtype_len_in <= sizeof(int64_t)) {
                read_fixed_width_byte_array_as_int(s, sb, val_src_pos, static_cast<int64_t*>(dst));
              } else {
                read_fixed_width_byte_array_as_int(
                  s, sb, val_src_pos, static_cast<__int128_t*>(dst));
              }
              break;
          }
        } else if (dtype == Type::FIXED_LEN_BYTE_ARRAY) {
          gpuOutputString(s, sb, val_src_pos, dst);
        } else if (dtype == Type::INT96) {
          read_int96_timestamp(s, sb, val_src_pos, static_cast<int64_t*>(dst));
        } else if (dtype_len == 8) {
          if (s->dtype_len_in == 4) {
            // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
            // TIME_MILLIS is the only duration type stored as int32:
            // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
            auto const dst_ptr = static_cast<uint32_t*>(dst);
            read_fixed_width_value_fast(s, sb, val_src_pos, dst_ptr);
            // zero out most significant bytes
            cuda::std::memset(dst_ptr + 1, 0, sizeof(int32_t));
          } else if (s->ts_scale) {
            read_int64_timestamp(s, sb, val_src_pos, static_cast<int64_t*>(dst));
          } else {
            read_fixed_width_value_fast(s, sb, val_src_pos, static_cast<uint2*>(dst));
          }
        } else if (dtype_len == 4) {
          read_fixed_width_value_fast(s, sb, val_src_pos, static_cast<uint32_t*>(dst));
        } else {
          read_nbyte_fixed_width_value(s, sb, val_src_pos, static_cast<uint8_t*>(dst), dtype_len);
        }
      }

      if (block.thread_rank() == first_out_thread_id) { s->src_pos = target_pos; }
    }
    __syncthreads();
  }

  // Zero-fill null positions after decoding valid values
  auto const is_string =
    ((dtype == Type::BYTE_ARRAY) && !is_decimal) || (dtype == Type::FIXED_LEN_BYTE_ARRAY);
  if (is_string || has_repetition) {
    auto const& ni = s->nesting_info[s->setup.col.max_nesting_depth - 1];
    if (ni.valid_map != nullptr) {
      int const num_values = ni.valid_map_offset - init_valid_map_offset;
      zero_fill_null_positions_shared<decode_block_size>(
        s, s->dtype_len, init_valid_map_offset, num_values, static_cast<int>(block.thread_rank()));
    }
  }

  if (s->setup.error != 0) {
    cg::invoke_one(block, [&]() { set_error(s->setup.error, error_code); });
  }
}

struct mask_tform {
  __device__ uint32_t operator()(PageInfo const& p) { return static_cast<uint32_t>(p.kernel_mask); }
};

}  // anonymous namespace

uint32_t get_aggregated_decode_kernel_mask(cudf::detail::hostdevice_span<PageInfo const> pages,
                                           rmm::cuda_stream_view stream)
{
  // determine which kernels to invoke
  return cudf::detail::transform_reduce(pages.device_begin(),
                                        pages.device_end(),
                                        mask_tform{},
                                        uint32_t{0},
                                        cuda::std::bit_or<uint32_t>{},
                                        stream);
}

/**
 * @copydoc cudf::io::parquet::detail::decode_page_data
 */
void decode_page_data(cudf::detail::hostdevice_span<PageInfo> pages,
                      cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                      size_t num_rows,
                      size_t min_row,
                      int level_type_size,
                      cudf::device_span<bool const> page_mask,
                      kernel_error::pointer error_code,
                      rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_page_data<rolling_buf_size, uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_page_data<rolling_buf_size, uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * @copydoc cudf::io::parquet::detail::decode_split_page_data
 */
void decode_split_page_data(cudf::detail::hostdevice_span<PageInfo> pages,
                            cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                            size_t num_rows,
                            size_t min_row,
                            int level_type_size,
                            cudf::device_span<bool const> page_mask,
                            kernel_error::pointer error_code,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_split_page_data_kernel<rolling_buf_size, uint8_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(
        pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_split_page_data_kernel<rolling_buf_size, uint16_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(
        pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

void write_final_offsets(host_span<size_type const> offsets,
                         host_span<size_type* const> buff_addrs,
                         rmm::cuda_stream_view stream)
{
  // Copy offsets to device and create an iterator
  auto d_src_data = cudf::detail::make_device_uvector_async(
    offsets, stream, cudf::get_current_device_resource_ref());
  // Iterator for the source (scalar) data
  auto src_iter = thrust::make_transform_iterator(
    cuda::counting_iterator<std::size_t>{0},
    cuda::proclaim_return_type<cudf::size_type*>(
      [src = d_src_data.begin()] __device__(std::size_t i) { return src + i; }));

  // Copy buffer addresses to device and create an iterator
  auto d_dst_addrs = cudf::detail::make_device_uvector_async(
    buff_addrs, stream, cudf::get_current_device_resource_ref());
  // size_iter is simply a constant iterator of sizeof(size_type) bytes.
  auto size_iter = cuda::make_constant_iterator(sizeof(size_type));

  // Copy offsets to buffers in batched manner.
  cudf::detail::batched_memcpy_async(
    src_iter, d_dst_addrs.begin(), size_iter, offsets.size(), stream);
}

}  // namespace cudf::io::parquet::detail
