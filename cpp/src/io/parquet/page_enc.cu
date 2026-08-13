/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delta_enc.cuh"
#include "io/parquet/parquet_gpu.hpp"
#include "io/utilities/block_utils.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda/iterator>
#include <cuda/std/chrono>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

#include <bitset>

namespace cudf::io::parquet::detail {

namespace {

using ::cudf::detail::device_2dspan;

using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;

constexpr int encode_block_size = 128;
constexpr int rle_buffer_size   = 2 * encode_block_size;
constexpr int num_encode_warps  = encode_block_size / cudf::detail::warp_size;

__device__ constexpr int rolling_idx(int pos) { return rolling_index<rle_buffer_size>(pos); }

// max V1 header size
// also valid for dict page header (V1 or V2)
constexpr int MAX_V1_HDR_SIZE = util::round_up_unsafe(27, 8);

// max V2 header size
constexpr int MAX_V2_HDR_SIZE = util::round_up_unsafe(49, 8);

// do not truncate statistics
constexpr int32_t NO_TRUNC_STATS = 0;

// minimum scratch space required for encoding statistics
constexpr size_t MIN_STATS_SCRATCH_SIZE = sizeof(__int128_t);

// mask to determine lane id
constexpr uint32_t WARP_MASK = cudf::detail::warp_size - 1;

// currently 64k - 1
constexpr uint32_t MAX_GRID_Y_SIZE = (1 << 16) - 1;

// space needed for RLE length field
constexpr int RLE_LENGTH_FIELD_LEN = 4;

struct frag_init_state_s {
  parquet_column_device_view col;
  PageFragment frag;
};

template <int rle_buf_size>
struct page_enc_state_s {
  uint8_t* cur;          //!< current output ptr
  uint8_t* rle_out;      //!< current RLE write ptr
  uint8_t* rle_len_pos;  //!< position to write RLE length (for V2 boolean data)
  uint32_t rle_run;      //!< current RLE run
  uint32_t run_val;      //!< current RLE run value
  uint32_t rle_pos;      //!< RLE encoder positions
  uint32_t rle_numvals;  //!< RLE input value count
  uint32_t rle_lit_count;
  uint32_t rle_rpt_count;
  uint32_t page_start_val;
  uint32_t chunk_start_val;
  uint32_t rpt_map[num_encode_warps];
  EncPage page;
  EncColumnChunk ck;
  parquet_column_device_view col;
  uint32_t vals[rle_buf_size];
};

using rle_page_enc_state_s = page_enc_state_s<rle_buffer_size>;

/**
 * @brief Returns the size of the type in the Parquet file.
 */
__device__ constexpr uint32_t physical_type_len(Type physical_type, type_id id, int type_length)
{
  if (physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    return id == type_id::DECIMAL128 ? sizeof(__int128_t) : type_length;
  }
  switch (physical_type) {
    case Type::INT96: return 12u;
    case Type::INT64:
    case Type::DOUBLE: return sizeof(int64_t);
    case Type::BOOLEAN: return 1u;
    default: return sizeof(int32_t);
  }
}

__device__ constexpr uint32_t max_RLE_page_size(uint8_t value_bit_width, uint32_t num_values)
{
  if (value_bit_width == 0) return 0;

  // Run length = 4, max(rle/bitpack header) = 5. bitpacking worst case is one byte every 8 values
  // (because bitpacked runs are a multiple of 8). Don't need to round up the last term since that
  // overhead is accounted for in the '5'.
  // TODO: this formula does not take into account the data for RLE runs. The worst realistic case
  // is repeated runs of 8 bitpacked, 2 RLE values. In this case, the formula would be
  //   0.8 * (num_values * bw / 8 + num_values / 8) + 0.2 * (num_values / 2 * (1 + (bw+7)/8))
  // for bw < 8 the above value will be larger than below, but in testing it seems like for low
  // bitwidths it's hard to get the pathological 8:2 split.
  // If the encoder starts printing the data corruption warning, then this will need to be
  // revisited.
  return 4 + 5 + util::div_rounding_up_unsafe(num_values * value_bit_width, 8) + (num_values / 8);
}

// subtract b from a, but return 0 if this would underflow
__device__ constexpr size_t underflow_safe_subtract(size_t a, size_t b)
{
  if (b > a) { return 0; }
  return a - b;
}

void __device__ init_frag_state(frag_init_state_s* const s,
                                uint32_t fragment_size,
                                int part_end_row)
{
  // frag.num_rows = fragment_size except for the last fragment in partition which can be
  // smaller. num_rows is fixed but fragment size could be larger if the data is strings or
  // nested.
  s->frag.num_rows = cuda::std::min<size_type>(fragment_size, part_end_row - s->frag.start_row);
  s->frag.num_dict_vals      = 0;
  s->frag.fragment_data_size = 0;
  s->frag.dict_data_size     = 0;

  s->frag.start_value_idx  = row_to_value_idx(s->frag.start_row, s->col);
  auto const end_value_idx = row_to_value_idx(s->frag.start_row + s->frag.num_rows, s->col);
  s->frag.num_leaf_values  = end_value_idx - s->frag.start_value_idx;

  if (s->col.level_offsets != nullptr) {
    // For nested schemas, the number of values in a fragment is not directly related to the
    // number of encoded data elements or the number of rows.  It is simply the number of
    // repetition/definition values which together encode validity and nesting information.
    auto const first_level_val_idx = s->col.level_offsets[s->frag.start_row];
    auto const last_level_val_idx  = s->col.level_offsets[s->frag.start_row + s->frag.num_rows];
    s->frag.num_values             = last_level_val_idx - first_level_val_idx;
  } else {
    s->frag.num_values = s->frag.num_rows;
  }
}

template <int block_size>
void __device__ calculate_frag_size(frag_init_state_s* const s, int t)
{
  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;

  auto const physical_type   = s->col.physical_type;
  auto const leaf_type       = s->col.leaf_column->type().id();
  auto const dtype_len       = physical_type_len(physical_type, leaf_type, s->col.type_length);
  auto const nvals           = s->frag.num_leaf_values;
  auto const start_value_idx = s->frag.start_value_idx;

  uint32_t num_valid = 0;
  uint32_t len       = 0;
  for (uint32_t i = 0; i < nvals; i += block_size) {
    auto const val_idx  = start_value_idx + i + t;
    auto const is_valid = i + t < nvals && val_idx < s->col.leaf_column->size() &&
                          s->col.leaf_column->is_valid(val_idx);
    if (is_valid) {
      num_valid++;
      len += dtype_len;
      if (physical_type == Type::BYTE_ARRAY) {
        switch (leaf_type) {
          case type_id::STRING: {
            auto str = s->col.leaf_column->element<string_view>(val_idx);
            len += str.size_bytes();
          } break;
          case type_id::LIST: {
            auto list_element =
              get_element<statistics::byte_array_view>(*s->col.leaf_column, val_idx);
            len += list_element.size_bytes();
          } break;
          default: CUDF_UNREACHABLE("Unsupported data type for leaf column");
        }
      }
    }
  }

  auto const total_len = block_reduce(reduce_storage).Sum(len);
  __syncthreads();
  auto const total_valid = block_reduce(reduce_storage).Sum(num_valid);

  if (t == 0) {
    s->frag.fragment_data_size = total_len;
    s->frag.num_valid          = total_valid;
  }

  __syncthreads();
  // page fragment size must fit in a 32-bit signed integer
  if (s->frag.fragment_data_size >
      static_cast<uint32_t>(cuda::std::numeric_limits<int32_t>::max())) {
    // TODO need to propagate this error back to the host
    CUDF_UNREACHABLE("page fragment size exceeds maximum for i32");
  }
}

/**
 * @brief Determine the correct page encoding for the given page parameters.
 *
 * This is only used by the plain and dictionary encoders. Delta encoders will set the page
 * encoding directly.
 */
Encoding __device__ determine_encoding(PageType page_type,
                                       Type physical_type,
                                       bool use_dictionary,
                                       bool write_v2_headers,
                                       bool is_split_stream)
{
  if (is_split_stream) { return Encoding::BYTE_STREAM_SPLIT; }
  // NOTE: For dictionary encoding, parquet v2 recommends using PLAIN in dictionary page and
  // RLE_DICTIONARY in data page, but parquet v1 uses PLAIN_DICTIONARY in both dictionary and
  // data pages (actual encoding is identical).
  switch (page_type) {
    case PageType::DATA_PAGE: return use_dictionary ? Encoding::PLAIN_DICTIONARY : Encoding::PLAIN;
    case PageType::DATA_PAGE_V2:
      return physical_type == Type::BOOLEAN ? Encoding::RLE
             : use_dictionary               ? Encoding::RLE_DICTIONARY
                                            : Encoding::PLAIN;
    case PageType::DICTIONARY_PAGE:
      return write_v2_headers ? Encoding::PLAIN : Encoding::PLAIN_DICTIONARY;
    default: CUDF_UNREACHABLE("unsupported page type");
  }
}

/**
 * @brief Generate level histogram for a page.
 *
 * For definition levels, the histogram values h(0)...h(max_def-1) represent nulls at
 * various levels of the hierarchy, and h(max_def) is the number of non-null values (num_valid).
 * If the leaf level is nullable, then num_leaf_values is h(max_def-1) + h(max_def),
 * and h(max_def-1) is num_leaf_values - num_valid. h(0) is derivable as num_values -
 * sum(h(1)..h(max_def)).
 *
 * For repetition levels, h(0) equals the number of rows. Here we can calculate
 * h(1)..h(max_rep-1), set h(0) directly, and then obtain h(max_rep) in the same way as
 * for the definition levels.
 *
 * @param hist Pointer to the histogram (size is max_level + 1)
 * @param s Page encode state
 * @param lvl_data Pointer to the global repetition or definition level data
 * @param lvl_end Last element of the histogram to encode (exclusive)
 */
template <int block_size, typename state_buf>
void __device__
generate_page_histogram(uint32_t* hist, state_buf const* s, uint8_t const* lvl_data, int lvl_end)
{
  using block_reduce = cub::BlockReduce<int, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  auto const t                  = threadIdx.x;
  auto const page_first_val_idx = s->col.level_offsets[s->page.start_row];
  auto const col_last_val_idx   = s->col.level_offsets[s->col.num_rows];

  // h(0) is always derivable, so start at 1
  for (int lvl = 1; lvl < lvl_end; lvl++) {
    int nval_in_level = 0;
    for (int i = 0; i < s->page.num_values; i += block_size) {
      auto const lidx = i + t;
      auto const gidx = page_first_val_idx + lidx;
      if (lidx < s->page.num_values && gidx < col_last_val_idx && lvl_data[gidx] == lvl) {
        nval_in_level++;
      }
    }
    __syncthreads();
    auto const lvl_sum = block_reduce(temp_storage).Sum(nval_in_level);
    if (t == 0) { hist[lvl] = lvl_sum; }
  }
}

/**
 * @brief Generate definition level histogram for a block of values.
 *
 * This is used when the max repetition level is 0 (no lists) and the definition
 * level data is not calculated in advance for the entire column.
 *
 * @param hist Pointer to the histogram (size is max_def_level + 1)
 * @param s Page encode state
 * @param nrows Number of rows to process
 * @param rle_numvals Index (relative to start of page) of the first level value
 * @param maxlvl Last element of the histogram to encode (exclusive)
 */
template <int block_size>
void __device__ generate_def_level_histogram(uint32_t* hist,
                                             rle_page_enc_state_s const* s,
                                             uint32_t nrows,
                                             uint32_t rle_numvals,
                                             uint32_t maxlvl)
{
  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const t = threadIdx.x;

  // Do a block sum for each level rather than each thread trying an atomicAdd.
  // This way is much faster.
  auto const mylvl = s->vals[rolling_index<rle_buffer_size>(rle_numvals + t)];
  // We can start at 1 because hist[0] can be derived.
  for (uint32_t lvl = 1; lvl < maxlvl; lvl++) {
    uint32_t const is_yes = t < nrows and mylvl == lvl;
    auto const lvl_sum    = block_reduce(temp_storage).Sum(is_yes);
    if (t == 0) { hist[lvl] += lvl_sum; }
    __syncthreads();
  }
}

// operator to use with warp_reduce. stolen from cuda::std::plus
struct BitwiseOr {
  /// Binary OR operator, returns <tt>a | b</tt>
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(T const& a, T const& b) const
  {
    return a | b;
  }
};

// PT is the parquet physical type (INT32 or INT64).
// I is the column type from the input table.
template <Type PT, typename I>
__device__ uint8_t const* delta_encode(page_enc_state_s<0>* s, uint64_t* buffer, void* temp_space)
{
  using output_type = cuda::std::conditional_t<PT == Type::INT32, int32_t, int64_t>;
  __shared__ delta_binary_packer<output_type> packer;

  auto const t = threadIdx.x;
  if (t == 0) {
    packer.init(s->cur, s->page.num_valid, reinterpret_cast<output_type*>(buffer), temp_space);
  }
  __syncthreads();

  // TODO(ets): in the plain encoder the scaling is a little different for INT32 than INT64.
  // might need to modify this if there's a big performance hit in the 32-bit case.
  int32_t const scale = s->col.ts_scale == 0 ? 1 : s->col.ts_scale;
  for (uint32_t cur_val_idx = 0; cur_val_idx < s->page.num_leaf_values;) {
    uint32_t const nvals = min(s->page.num_leaf_values - cur_val_idx, delta::block_size);

    size_type const val_idx_in_block = cur_val_idx + t;
    size_type const val_idx          = s->page_start_val + val_idx_in_block;

    bool const is_valid =
      (val_idx < s->col.leaf_column->size() && val_idx_in_block < s->page.num_leaf_values)
        ? s->col.leaf_column->is_valid(val_idx)
        : false;

    cur_val_idx += nvals;

    output_type v = is_valid ? s->col.leaf_column->element<I>(val_idx) : 0;
    if (scale < 0) {
      v /= -scale;
    } else {
      v *= scale;
    }
    packer.add_value(v, is_valid);
  }

  return packer.flush();
}

/**
 * @brief Sets `s->cur` to point to the start of encoded page data.
 *
 * For V1 headers, this will be immediately after the repetition and definition level data. For V2,
 * it will be at the next properly aligned location after the level data. The padding in V2 is
 * needed for compressors that require aligned input.
 */
template <typename state_type>
inline void __device__ set_page_data_start(state_type* s)
{
  s->cur = s->page.page_data + s->page.max_hdr_size;
  switch (s->page.page_type) {
    case PageType::DATA_PAGE:
      s->cur += s->page.level_bytes();
      if (s->col.num_def_level_bits() != 0) { s->cur += RLE_LENGTH_FIELD_LEN; }
      if (s->col.num_rep_level_bits() != 0) { s->cur += RLE_LENGTH_FIELD_LEN; }
      break;
    case PageType::DATA_PAGE_V2: s->cur += s->page.max_lvl_size; break;
  }
}

}  // anonymous namespace

// blockDim {512,1,1}
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  gpuInitRowGroupFragments(device_2dspan<PageFragment> frag,
                           device_span<parquet_column_device_view const> col_desc,
                           device_span<partition_info const> partitions,
                           device_span<int const> part_frag_offset,
                           uint32_t fragment_size)
{
  __shared__ __align__(16) frag_init_state_s state_g;

  frag_init_state_s* const s          = &state_g;
  auto const t                        = threadIdx.x;
  auto const num_fragments_per_column = frag.size().second;

  if (t == 0) { s->col = col_desc[blockIdx.x]; }
  __syncthreads();

  for (uint32_t frag_y = blockIdx.y; frag_y < num_fragments_per_column; frag_y += gridDim.y) {
    if (t == 0) {
      // Find which partition this fragment came from
      auto it =
        thrust::upper_bound(thrust::seq, part_frag_offset.begin(), part_frag_offset.end(), frag_y);
      int const p            = it - part_frag_offset.begin() - 1;
      int const part_end_row = partitions[p].start_row + partitions[p].num_rows;
      s->frag.start_row = (frag_y - part_frag_offset[p]) * fragment_size + partitions[p].start_row;
      s->frag.chunk     = frag[blockIdx.x][frag_y].chunk;
      init_frag_state(s, fragment_size, part_end_row);
    }
    __syncthreads();

    calculate_frag_size<block_size>(s, t);
    __syncthreads();
    if (t == 0) { frag[blockIdx.x][frag_y] = s->frag; }
  }
}

// blockDim {512,1,1}
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  gpuCalculatePageFragments(device_span<PageFragment> frag,
                            device_span<size_type const> column_frag_sizes)
{
  __shared__ __align__(16) frag_init_state_s state_g;

  EncColumnChunk* const ck_g = frag[blockIdx.x].chunk;
  frag_init_state_s* const s = &state_g;
  uint32_t const t           = threadIdx.x;
  auto const fragment_size   = column_frag_sizes[ck_g->col_desc_id];

  if (t == 0) { s->col = *ck_g->col_desc; }
  __syncthreads();

  if (t == 0) {
    int const part_end_row = ck_g->start_row + ck_g->num_rows;
    s->frag.start_row      = ck_g->start_row + (blockIdx.x - ck_g->first_fragment) * fragment_size;
    s->frag.chunk          = ck_g;
    init_frag_state(s, fragment_size, part_end_row);
  }
  __syncthreads();

  calculate_frag_size<block_size>(s, t);
  if (t == 0) { frag[blockIdx.x] = s->frag; }
}

// blockDim {128,1,1}
CUDF_KERNEL void __launch_bounds__(128)
  gpuInitFragmentStats(device_span<statistics_group> groups,
                       device_span<PageFragment const> fragments)
{
  uint32_t const lane_id = threadIdx.x & WARP_MASK;
  uint32_t const frag_id = blockIdx.x * 4 + (threadIdx.x / cudf::detail::warp_size);
  if (frag_id < fragments.size()) {
    if (lane_id == 0) {
      statistics_group g;
      auto* const ck_g = fragments[frag_id].chunk;
      g.col            = ck_g->col_desc;
      g.start_row      = fragments[frag_id].start_value_idx;
      g.num_rows       = fragments[frag_id].num_leaf_values;
      g.non_leaf_nulls = fragments[frag_id].num_values - g.num_rows;
      groups[frag_id]  = g;
    }
  }
}

// given a column chunk, determine which data encoding to use
__device__ encode_kernel_mask data_encoding_for_col(EncColumnChunk const* chunk,
                                                    parquet_column_device_view const* col_desc,
                                                    bool write_v2_headers)
{
  // first check for dictionary (boolean always uses dict encoder)
  if (chunk->use_dictionary or col_desc->physical_type == Type::BOOLEAN) {
    return encode_kernel_mask::DICTIONARY;
  }

  // next check for user requested encoding, but skip if user requested dictionary encoding
  // (if we could use the requested dict encoding, we'd have returned above)
  if (col_desc->requested_encoding != column_encoding::USE_DEFAULT and
      col_desc->requested_encoding != column_encoding::DICTIONARY) {
    switch (col_desc->requested_encoding) {
      case column_encoding::PLAIN: return encode_kernel_mask::PLAIN;
      case column_encoding::DELTA_BINARY_PACKED: return encode_kernel_mask::DELTA_BINARY;
      case column_encoding::DELTA_LENGTH_BYTE_ARRAY: return encode_kernel_mask::DELTA_LENGTH_BA;
      case column_encoding::DELTA_BYTE_ARRAY: return encode_kernel_mask::DELTA_BYTE_ARRAY;
      case column_encoding::BYTE_STREAM_SPLIT: return encode_kernel_mask::BYTE_STREAM_SPLIT;
    }
  }

  // Select a fallback encoding. For V1, we always choose PLAIN. For V2 we'll use
  // DELTA_BINARY_PACKED for INT32 and INT64, and DELTA_LENGTH_BYTE_ARRAY for
  // BYTE_ARRAY. Everything else will still fall back to PLAIN.
  if (write_v2_headers) {
    switch (col_desc->physical_type) {
      case Type::INT32:
      case Type::INT64: return encode_kernel_mask::DELTA_BINARY;
      case Type::BYTE_ARRAY: return encode_kernel_mask::DELTA_LENGTH_BA;
    }
  }

  return encode_kernel_mask::PLAIN;
}

__device__ size_t delta_data_len(Type physical_type,
                                 cudf::type_id type_id,
                                 uint32_t num_values,
                                 size_t page_size,
                                 encode_kernel_mask encoding)
{
  // dtype_len_out is for the lengths, rather than the char data, so pass sizeof(int32_t)
  auto const dtype_len_out = physical_type_len(physical_type, type_id, sizeof(int32_t));
  auto const dtype_len     = [&]() -> uint32_t {
    if (physical_type == Type::INT32) { return int32_logical_len(type_id); }
    if (physical_type == Type::INT96) { return sizeof(int64_t); }
    return dtype_len_out;
  }();

  auto const vals_per_block = delta::block_size;
  size_t const num_blocks   = util::div_rounding_up_unsafe(num_values, vals_per_block);
  // need max dtype_len + 1 bytes for min_delta (because we only encode 7 bits per byte)
  // one byte per mini block for the bitwidth
  auto const mini_block_header_size = dtype_len + 1 + delta::num_mini_blocks;
  // each encoded value can be at most sizeof(type) * 8 + 1 bits
  auto const max_bits = dtype_len * 8 + 1;
  // each data block will then be max_bits * values per block. vals_per_block is guaranteed to be
  // divisible by 128 (via static assert on delta::block_size), but do safe division anyway.
  auto const bytes_per_block = cudf::util::div_rounding_up_unsafe(max_bits * vals_per_block, 8);
  auto const block_size      = mini_block_header_size + bytes_per_block;
  // the number of DELTA_BINARY_PACKED blocks to encode
  auto const num_dbp_blocks = encoding == encode_kernel_mask::DELTA_BYTE_ARRAY ? 2 : 1;

  // delta header is 2 bytes for the block_size, 1 byte for number of mini-blocks,
  // max 5 bytes for number of values, and max dtype_len + 1 for first value.
  // TODO: if we ever allow configurable block sizes then this calculation will need to be
  // modified.
  auto const header_size = 2 + 1 + 5 + dtype_len + 1;

  // The above is just a size estimate for a DELTA_BINARY_PACKED data page. For BYTE_ARRAY
  // data we also need to add size of the char data. `page_size` that is passed in is the
  // plain encoded size (i.e. num_values * sizeof(size_type) + char_data_len), so the char
  // data len is `page_size` minus the first term. For FIXED_LEN_BYTE_ARRAY there are no
  // lengths, so just use `page_size`.
  // `num_dbp_blocks` takes into account the two delta binary blocks for DELTA_BYTE_ARRAY.
  size_t char_data_len = 0;
  if (physical_type == Type::BYTE_ARRAY) {
    char_data_len = page_size - num_values * sizeof(size_type);
  } else if (physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    char_data_len = page_size;
  }

  return header_size + num_blocks * num_dbp_blocks * block_size + char_data_len;
}

// blockDim {128,1,1}
CUDF_KERNEL void __launch_bounds__(128)
  gpuInitPages(device_2dspan<EncColumnChunk> chunks,
               device_span<EncPage> pages,
               device_span<size_type> page_sizes,
               device_span<size_type const> comp_page_sizes,
               device_span<parquet_column_device_view const> col_desc,
               statistics_merge_group* page_grstats,
               statistics_merge_group* chunk_grstats,
               int32_t num_columns,
               size_t max_page_size_bytes,
               size_type max_page_size_rows,
               uint32_t page_align,
               bool write_v2_headers)
{
  // TODO: All writing seems to be done by thread 0. Could be replaced by thrust foreach
  __shared__ __align__(8) parquet_column_device_view col_g;
  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) PageFragment frag_g;
  __shared__ __align__(8) EncPage page_g;
  __shared__ __align__(8) statistics_merge_group pagestats_g;

  uint32_t const t          = threadIdx.x;
  auto const data_page_type = write_v2_headers ? PageType::DATA_PAGE_V2 : PageType::DATA_PAGE;

  // Max page header size excluding statistics
  auto const max_data_page_hdr_size = write_v2_headers ? MAX_V2_HDR_SIZE : MAX_V1_HDR_SIZE;

  if (t == 0) {
    col_g  = col_desc[blockIdx.x];
    ck_g   = chunks[blockIdx.y][blockIdx.x];
    page_g = {};
  }
  __syncthreads();

  // if writing delta encoded values, we're going to need to know the data length to get a guess
  // at the worst case number of bytes needed to encode.
  auto const physical_type = col_g.physical_type;
  auto const type_id       = col_g.leaf_column->type().id();

  // figure out kernel encoding to use for data pages
  auto const column_data_encoding = data_encoding_for_col(&ck_g, &col_g, write_v2_headers);
  auto const is_use_delta         = column_data_encoding == encode_kernel_mask::DELTA_BINARY or
                            column_data_encoding == encode_kernel_mask::DELTA_LENGTH_BA or
                            column_data_encoding == encode_kernel_mask::DELTA_BYTE_ARRAY;

  if (t < 32) {
    uint32_t fragments_in_chunk  = 0;
    uint32_t rows_in_page        = 0;
    uint32_t values_in_page      = 0;
    uint32_t leaf_values_in_page = 0;
    uint32_t num_valid           = 0;
    size_t page_size             = 0;
    size_t var_bytes_size        = 0;
    uint32_t num_pages           = 0;
    uint32_t num_rows            = 0;
    uint32_t page_start          = 0;
    uint32_t page_offset         = ck_g.ck_stat_size;
    uint32_t comp_page_offset    = ck_g.ck_stat_size;
    uint32_t page_headers_size   = 0;
    uint32_t max_page_data_size  = 0;
    uint32_t cur_row             = ck_g.start_row;
    uint32_t ck_max_stats_len    = 0;
    uint32_t max_stats_len       = 0;

    if (!t) {
      pagestats_g.col_dtype   = col_g.leaf_column->type();
      pagestats_g.stats_dtype = col_g.stats_dtype;
      pagestats_g.start_chunk = ck_g.first_fragment;
      pagestats_g.num_chunks  = 0;
    }
    if (ck_g.use_dictionary) {
      if (!t) {
        page_g.page_data       = ck_g.uncompressed_bfr + page_offset;
        page_g.compressed_data = ck_g.compressed_bfr + comp_page_offset;
        page_g.num_fragments   = 0;
        page_g.page_type       = PageType::DICTIONARY_PAGE;
        page_g.chunk           = &chunks[blockIdx.y][blockIdx.x];
        page_g.chunk_id        = blockIdx.y * num_columns + blockIdx.x;
        page_g.hdr_size        = 0;
        page_g.def_lvl_bytes   = 0;
        page_g.rep_lvl_bytes   = 0;
        page_g.max_lvl_size    = 0;
        page_g.comp_data_size  = 0;
        page_g.is_compressed   = false;
        page_g.max_hdr_size    = MAX_V1_HDR_SIZE;
        page_g.max_data_size   = ck_g.uniq_data_size;
        page_g.data_size       = ck_g.uniq_data_size;
        page_g.start_row       = cur_row;
        page_g.num_rows        = ck_g.num_dict_entries;
        page_g.num_leaf_values = ck_g.num_dict_entries;
        page_g.num_values      = ck_g.num_dict_entries;  // TODO: shouldn't matter for dict page
        page_g.dict_rle_bits   = ck_g.dict_rle_bits;     // TODO: shouldn't matter for dict page
        page_offset +=
          util::round_up_unsafe(page_g.max_hdr_size + page_g.max_data_size, page_align);
        if (not comp_page_sizes.empty()) {
          comp_page_offset += page_g.max_hdr_size + comp_page_sizes[ck_g.first_page];
          page_g.comp_data_size = comp_page_sizes[ck_g.first_page + num_pages];
        }
        page_headers_size += page_g.max_hdr_size;
        max_page_data_size = max(max_page_data_size, page_g.max_data_size);
      }
      __syncwarp();
      if (t == 0) {
        if (not pages.empty()) {
          page_g.kernel_mask     = encode_kernel_mask::PLAIN;
          pages[ck_g.first_page] = page_g;
        }
        if (not page_sizes.empty()) { page_sizes[ck_g.first_page] = page_g.max_data_size; }
        if (page_grstats) { page_grstats[ck_g.first_page] = pagestats_g; }
      }
      num_pages = 1;
    }
    __syncwarp();

    // page padding needed for RLE encoded boolean data
    auto const rle_pad =
      write_v2_headers && col_g.physical_type == Type::BOOLEAN ? RLE_LENGTH_FIELD_LEN : 0;

    // This loop goes over one page fragment at a time and adds it to page.
    // When page size crosses a particular limit, then it moves on to the next page and then next
    // page fragment gets added to that one.

    // This doesn't actually deal with data. It's agnostic. It only cares about number of rows and
    // page size.
    do {
      uint32_t minmax_len = 0;
      __syncwarp();
      if (num_rows < ck_g.num_rows) {
        if (t == 0) { frag_g = ck_g.fragments[fragments_in_chunk]; }
        if (!t && ck_g.stats) {
          if (col_g.stats_dtype == dtype_string) {
            minmax_len = max(ck_g.stats[fragments_in_chunk].min_value.str_val.length,
                             ck_g.stats[fragments_in_chunk].max_value.str_val.length);
          } else if (col_g.stats_dtype == dtype_byte_array) {
            minmax_len = max(ck_g.stats[fragments_in_chunk].min_value.byte_val.length,
                             ck_g.stats[fragments_in_chunk].max_value.byte_val.length);
          }
        }
      } else if (!t) {
        frag_g.fragment_data_size = 0;
        frag_g.num_rows           = 0;
      }
      __syncwarp();
      uint32_t fragment_data_size =
        (ck_g.use_dictionary)
          ? frag_g.num_leaf_values * util::div_rounding_up_unsafe(ck_g.dict_rle_bits, 8)
          : frag_g.fragment_data_size;

      // page fragment size must fit in a 32-bit signed integer
      if (fragment_data_size > cuda::std::numeric_limits<int32_t>::max()) {
        CUDF_UNREACHABLE("page fragment size exceeds maximum for i32");
      }

      // TODO (dm): this convoluted logic to limit page size needs refactoring
      size_t this_max_page_size = (values_in_page * 2 >= ck_g.num_values)   ? 256 * 1024
                                  : (values_in_page * 3 >= ck_g.num_values) ? 384 * 1024
                                                                            : 512 * 1024;

      // override this_max_page_size if the requested size is smaller
      this_max_page_size = min(this_max_page_size, max_page_size_bytes);

      // subtract size of rep and def level vectors and RLE length field
      auto num_vals      = values_in_page + frag_g.num_values;
      this_max_page_size = underflow_safe_subtract(
        this_max_page_size,
        max_RLE_page_size(col_g.num_def_level_bits(), num_vals) +
          max_RLE_page_size(col_g.num_rep_level_bits(), num_vals) + rle_pad);

      // checks to see when we need to close the current page and start a new one
      auto const is_last_chunk          = num_rows >= ck_g.num_rows;
      auto const is_page_bytes_exceeded = page_size + fragment_data_size > this_max_page_size;
      auto const is_page_rows_exceeded  = rows_in_page + frag_g.num_rows > max_page_size_rows;
      // only check for limit overflow if there's already at least one fragment for this page
      auto const is_page_too_big =
        values_in_page > 0 && (is_page_bytes_exceeded || is_page_rows_exceeded);

      if (is_last_chunk || is_page_too_big) {
        if (ck_g.use_dictionary) {
          // Additional byte to store entry bit width
          page_size = 1 + max_RLE_page_size(ck_g.dict_rle_bits, values_in_page);
        }
        if (!t) {
          page_g.num_fragments  = fragments_in_chunk - page_start;
          page_g.chunk          = &chunks[blockIdx.y][blockIdx.x];
          page_g.chunk_id       = blockIdx.y * num_columns + blockIdx.x;
          page_g.page_type      = data_page_type;
          page_g.hdr_size       = 0;
          page_g.def_lvl_bytes  = 0;
          page_g.rep_lvl_bytes  = 0;
          page_g.max_lvl_size   = 0;
          page_g.data_size      = 0;
          page_g.comp_data_size = 0;
          page_g.is_compressed  = false;
          page_g.dict_rle_bits =
            ck_g.dict_rle_bits;  // Conservatively set to the chunk-wide bit width
          page_g.max_hdr_size = max_data_page_hdr_size;  // Max size excluding statistics
          if (ck_g.stats) {
            uint32_t stats_hdr_len = 16;
            if (col_g.stats_dtype == dtype_string || col_g.stats_dtype == dtype_byte_array) {
              stats_hdr_len += 5 * 3 + 2 * max_stats_len;
            } else {
              stats_hdr_len += ((col_g.stats_dtype >= dtype_int64) ? 10 : 5) * 3;
            }
            page_g.max_hdr_size += stats_hdr_len;
          }
          page_g.max_hdr_size = util::round_up_unsafe(page_g.max_hdr_size, page_align);
          page_g.page_data    = ck_g.uncompressed_bfr + page_offset;
          if (not comp_page_sizes.empty()) {
            page_g.compressed_data = ck_g.compressed_bfr + comp_page_offset;
            page_g.comp_data_size  = comp_page_sizes[ck_g.first_page + num_pages];
          }
          page_g.start_row          = cur_row;
          page_g.num_rows           = rows_in_page;
          page_g.num_leaf_values    = leaf_values_in_page;
          page_g.num_values         = values_in_page;
          page_g.num_valid          = num_valid;
          auto const def_level_size = max_RLE_page_size(col_g.num_def_level_bits(), values_in_page);
          auto const rep_level_size = max_RLE_page_size(col_g.num_rep_level_bits(), values_in_page);
          if (write_v2_headers) {
            page_g.max_lvl_size =
              util::round_up_unsafe(def_level_size + rep_level_size, page_align);
          }
          // get a different bound if using delta encoding
          if (is_use_delta) {
            auto const delta_len = delta_data_len(
              physical_type, type_id, page_g.num_leaf_values, page_size, column_data_encoding);
            page_size = max(page_size, delta_len);
          }
          auto const max_data_size =
            page_size + rle_pad +
            (write_v2_headers ? page_g.max_lvl_size : def_level_size + rep_level_size);
          // page size must fit in 32-bit signed integer
          if (max_data_size > cuda::std::numeric_limits<int32_t>::max()) {
            CUDF_UNREACHABLE("page size exceeds maximum for i32");
          }
          // if byte_array then save the variable bytes size
          if (ck_g.col_desc->physical_type == Type::BYTE_ARRAY) {
            // Page size is the sum of frag sizes, and frag sizes for strings includes the
            // 4-byte length indicator, so subtract that.
            page_g.var_bytes_size = var_bytes_size;
          }

          page_g.kernel_mask      = column_data_encoding;
          page_g.max_data_size    = static_cast<uint32_t>(max_data_size);
          pagestats_g.start_chunk = ck_g.first_fragment + page_start;
          pagestats_g.num_chunks  = page_g.num_fragments;
          page_offset +=
            util::round_up_unsafe(page_g.max_hdr_size + page_g.max_data_size, page_align);
          // if encoding delta_byte_array, need to allocate some space for scratch data.
          // if there are leaf nulls, we need space for a mapping array:
          //   sizeof(size_type) * num_leaf_values
          // we always need prefix lengths: sizeof(size_type) * num_valid
          if (page_g.kernel_mask == encode_kernel_mask::DELTA_BYTE_ARRAY) {
            // scratch needs to be aligned to a size_type boundary
            auto const pg_end = reinterpret_cast<uintptr_t>(ck_g.uncompressed_bfr + page_offset);
            auto scratch      = util::round_up_unsafe(pg_end, sizeof(size_type));
            if (page_g.num_valid != page_g.num_leaf_values) {
              scratch += sizeof(size_type) * page_g.num_leaf_values;
            }
            scratch += sizeof(size_type) * page_g.num_valid;
            page_offset =
              cuda::std::distance(ck_g.uncompressed_bfr, reinterpret_cast<uint8_t*>(scratch));
          }
          if (not comp_page_sizes.empty()) {
            // V2 does not include level data in compressed size estimate
            comp_page_offset += page_g.max_hdr_size + page_g.max_lvl_size +
                                comp_page_sizes[ck_g.first_page + num_pages];
          }
          page_headers_size += page_g.max_hdr_size;
          max_page_data_size = max(max_page_data_size, page_g.max_data_size);
          cur_row += rows_in_page;
          ck_max_stats_len = max(ck_max_stats_len, max_stats_len);
        }
        __syncwarp();
        if (t == 0) {
          if (not pages.empty()) {
            // need space for the chunk histograms plus data page histograms
            auto const num_histograms = num_pages - ck_g.num_dict_pages();
            if (ck_g.def_histogram_data != nullptr && col_g.max_def_level > 0) {
              page_g.def_histogram =
                ck_g.def_histogram_data + num_histograms * (col_g.max_def_level + 1);
            }
            if (ck_g.rep_histogram_data != nullptr && col_g.max_rep_level > 0) {
              page_g.rep_histogram =
                ck_g.rep_histogram_data + num_histograms * (col_g.max_rep_level + 1);
            }
            pages[ck_g.first_page + num_pages] = page_g;
          }
          // page_sizes should be the number of bytes to be compressed, so don't include level
          // data for V2.
          if (not page_sizes.empty()) {
            page_sizes[ck_g.first_page + num_pages] = page_g.max_data_size - page_g.max_lvl_size;
          }
          if (page_grstats) { page_grstats[ck_g.first_page + num_pages] = pagestats_g; }
        }

        num_pages++;
        page_size           = 0;
        var_bytes_size      = 0;
        rows_in_page        = 0;
        values_in_page      = 0;
        leaf_values_in_page = 0;
        num_valid           = 0;
        page_start          = fragments_in_chunk;
        max_stats_len       = 0;
      }
      max_stats_len = max(max_stats_len, minmax_len);
      page_size += fragment_data_size;
      // fragment_data_size includes the length indicator...remove it
      var_bytes_size += frag_g.fragment_data_size - frag_g.num_valid * sizeof(size_type);
      rows_in_page += frag_g.num_rows;
      values_in_page += frag_g.num_values;
      leaf_values_in_page += frag_g.num_leaf_values;
      num_valid += frag_g.num_valid;
      num_rows += frag_g.num_rows;
      fragments_in_chunk++;
    } while (frag_g.num_rows != 0);
    __syncwarp();
    if (!t) {
      if (ck_g.ck_stat_size == 0 && ck_g.stats) {
        uint32_t ck_stat_size = util::round_up_unsafe(48 + 2 * ck_max_stats_len, page_align);
        page_offset += ck_stat_size;
        comp_page_offset += ck_stat_size;
        ck_g.ck_stat_size = ck_stat_size;
      }
      ck_g.num_pages          = num_pages;
      ck_g.bfr_size           = page_offset;
      ck_g.page_headers_size  = page_headers_size;
      ck_g.max_page_data_size = max_page_data_size;
      if (not comp_page_sizes.empty()) { ck_g.compressed_size = comp_page_offset; }
      pagestats_g.start_chunk = ck_g.first_page + ck_g.use_dictionary;  // Exclude dictionary
      pagestats_g.num_chunks  = num_pages - ck_g.use_dictionary;
    }
  }
  __syncthreads();
  if (t == 0) {
    if (not pages.empty()) ck_g.pages = &pages[ck_g.first_page];
    chunks[blockIdx.y][blockIdx.x] = ck_g;
    if (chunk_grstats) chunk_grstats[blockIdx.y * num_columns + blockIdx.x] = pagestats_g;
  }
}

/**
 * @brief Mask table representing how many consecutive repeats are needed to code a repeat run
 *[nbits-1]
 */
static __device__ __constant__ uint32_t kRleRunMask[24] = {
  0x00ff'ffff, 0x0fff, 0x00ff, 0x3f, 0x0f, 0x0f, 0x7, 0x7, 0x3, 0x3, 0x3, 0x3,
  0x1,         0x1,    0x1,    0x1,  0x1,  0x1,  0x1, 0x1, 0x1, 0x1, 0x1, 0x1};

/**
 * @brief Variable-length encode an integer
 */
inline __device__ uint8_t* VlqEncode(uint8_t* p, uint32_t v)
{
  while (v > 0x7f) {
    *p++ = (v | 0x80);
    v >>= 7;
  }
  *p++ = v;
  return p;
}

/**
 * @brief Pack literal values in output bitstream (1,2,3,4,5,6,8,10,12,16,20 or 24 bits per value)
 */
inline __device__ void PackLiteralsShuffle(
  uint8_t* dst, uint32_t v, uint32_t count, uint32_t w, uint32_t t)
{
  constexpr uint32_t MASK2T = 1;  // mask for 2 thread leader
  constexpr uint32_t MASK4T = 3;  // mask for 4 thread leader
  constexpr uint32_t MASK8T = 7;  // mask for 8 thread leader
  uint64_t v64;

  if (t > (count | 0x1f)) { return; }

  switch (w) {
    case 1:
      v |= shuffle_xor(v, 1) << 1;  // grab bit 1 from neighbor
      v |= shuffle_xor(v, 2) << 2;  // grab bits 2-3 from 2 lanes over
      v |= shuffle_xor(v, 4) << 4;  // grab bits 4-7 from 4 lanes over
      // sub-warp leader writes the combined bits
      if (t < count && !(t & MASK8T)) { dst[(t * w) >> 3] = v; }
      return;
    case 2:
      v |= shuffle_xor(v, 1) << 2;
      v |= shuffle_xor(v, 2) << 4;
      if (t < count && !(t & MASK4T)) { dst[(t * w) >> 3] = v; }
      return;
    case 3:
      v |= shuffle_xor(v, 1) << 3;
      v |= shuffle_xor(v, 2) << 6;
      v |= shuffle_xor(v, 4) << 12;
      if (t < count && !(t & MASK8T)) {
        dst[(t >> 3) * 3 + 0] = v;
        dst[(t >> 3) * 3 + 1] = v >> 8;
        dst[(t >> 3) * 3 + 2] = v >> 16;
      }
      return;
    case 4:
      v |= shuffle_xor(v, 1) << 4;
      if (t < count && !(t & MASK2T)) { dst[(t * w) >> 3] = v; }
      return;
    case 5:
      v |= shuffle_xor(v, 1) << 5;
      v |= shuffle_xor(v, 2) << 10;
      v64 = static_cast<uint64_t>(shuffle_xor(v, 4)) << 20 | v;
      if (t < count && !(t & MASK8T)) {
        dst[(t >> 3) * 5 + 0] = v64;
        dst[(t >> 3) * 5 + 1] = v64 >> 8;
        dst[(t >> 3) * 5 + 2] = v64 >> 16;
        dst[(t >> 3) * 5 + 3] = v64 >> 24;
        dst[(t >> 3) * 5 + 4] = v64 >> 32;
      }
      return;
    case 6:
      v |= shuffle_xor(v, 1) << 6;
      v |= shuffle_xor(v, 2) << 12;
      if (t < count && !(t & MASK4T)) {
        dst[(t >> 2) * 3 + 0] = v;
        dst[(t >> 2) * 3 + 1] = v >> 8;
        dst[(t >> 2) * 3 + 2] = v >> 16;
      }
      return;
    case 8:
      if (t < count) { dst[t] = v; }
      return;
    case 10:
      v |= shuffle_xor(v, 1) << 10;
      v64 = static_cast<uint64_t>(shuffle_xor(v, 2)) << 20 | v;
      if (t < count && !(t & MASK4T)) {
        dst[(t >> 2) * 5 + 0] = v64;
        dst[(t >> 2) * 5 + 1] = v64 >> 8;
        dst[(t >> 2) * 5 + 2] = v64 >> 16;
        dst[(t >> 2) * 5 + 3] = v64 >> 24;
        dst[(t >> 2) * 5 + 4] = v64 >> 32;
      }
      return;
    case 12:
      v |= shuffle_xor(v, 1) << 12;
      if (t < count && !(t & MASK2T)) {
        dst[(t >> 1) * 3 + 0] = v;
        dst[(t >> 1) * 3 + 1] = v >> 8;
        dst[(t >> 1) * 3 + 2] = v >> 16;
      }
      return;
    case 16:
      if (t < count) {
        dst[t * 2 + 0] = v;
        dst[t * 2 + 1] = v >> 8;
      }
      return;
    case 20:
      v64 = static_cast<uint64_t>(shuffle_xor(v, 1)) << 20 | v;
      if (t < count && !(t & MASK2T)) {
        dst[(t >> 1) * 5 + 0] = v64;
        dst[(t >> 1) * 5 + 1] = v64 >> 8;
        dst[(t >> 1) * 5 + 2] = v64 >> 16;
        dst[(t >> 1) * 5 + 3] = v64 >> 24;
        dst[(t >> 1) * 5 + 4] = v64 >> 32;
      }
      return;
    case 24:
      if (t < count) {
        dst[t * 3 + 0] = v;
        dst[t * 3 + 1] = v >> 8;
        dst[t * 3 + 2] = v >> 16;
      }
      return;

    default: CUDF_UNREACHABLE("Unsupported bit width");
  }
}

/**
 * @brief Pack literals of arbitrary bit-length in output bitstream.
 */
inline __device__ void PackLiteralsRoundRobin(
  uint8_t* dst, uint32_t v, uint32_t count, uint32_t w, uint32_t t)
{
  // Scratch space to temporarily write to. Needed because we will use atomics to write 32 bit
  // words but the destination mem may not be a multiple of 4 bytes.
  // TODO (dm): This assumes blockdim = 128. Reduce magic numbers.
  constexpr uint32_t NUM_THREADS  = 128;  // this needs to match gpuEncodePages block_size parameter
  constexpr uint32_t NUM_BYTES    = (NUM_THREADS * MAX_DICT_BITS) >> 3;
  constexpr uint32_t SCRATCH_SIZE = NUM_BYTES / sizeof(uint32_t);
  __shared__ uint32_t scratch[SCRATCH_SIZE];
  for (uint32_t i = t; i < SCRATCH_SIZE; i += NUM_THREADS) {
    scratch[i] = 0;
  }
  __syncthreads();

  if (t <= count) {
    // shift symbol left by up to 31 bits
    uint64_t v64 = v;
    v64 <<= (t * w) & 0x1f;

    // Copy 64 bit word into two 32 bit words while following C++ strict aliasing rules.
    uint32_t v32[2];
    memcpy(&v32, &v64, sizeof(uint64_t));

    // Atomically write result to scratch
    if (v32[0]) { atomicOr(scratch + ((t * w) >> 5), v32[0]); }
    if (v32[1]) { atomicOr(scratch + ((t * w) >> 5) + 1, v32[1]); }
  }
  __syncthreads();

  // Copy scratch data to final destination
  auto available_bytes = (count * w + 7) / 8;

  auto scratch_bytes = reinterpret_cast<char*>(&scratch[0]);
  for (uint32_t i = t; i < available_bytes; i += NUM_THREADS) {
    dst[i] = scratch_bytes[i];
  }
  __syncthreads();
}

/**
 * @brief Pack literal values in output bitstream
 */
inline __device__ void PackLiterals(
  uint8_t* dst, uint32_t v, uint32_t count, uint32_t w, uint32_t t)
{
  if (w > 24) { CUDF_UNREACHABLE("Unsupported bit width"); }
  switch (w) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 8:
    case 10:
    case 12:
    case 16:
    case 20:
    case 24:
      // bit widths that lie on easy boundaries can be handled either directly
      // (8, 16, 24) or through fast shuffle operations.
      PackLiteralsShuffle(dst, v, count, w, t);
      return;
    default:
      // bit packing that uses atomics, but can handle arbitrary bit widths up to 24.
      PackLiteralsRoundRobin(dst, v, count, w, t);
  }
}

/**
 * @brief RLE encoder
 *
 * @param[in,out] s Page encode state
 * @param[in] numvals Total count of input values
 * @param[in] nbits number of bits per symbol (1..16)
 * @param[in] flush nonzero if last batch in block
 * @param[in] t thread id (0..127)
 */
static __device__ void RleEncode(
  rle_page_enc_state_s* s, uint32_t numvals, uint32_t nbits, uint32_t flush, uint32_t t)
{
  using cudf::detail::warp_size;
  auto const lane_id = t % warp_size;
  auto const warp_id = t / warp_size;

  uint32_t rle_pos = s->rle_pos;
  uint32_t rle_run = s->rle_run;

  while (rle_pos < numvals || (flush && rle_run)) {
    uint32_t pos = rle_pos + t;
    if (rle_run > 0 && !(rle_run & 1)) {
      // Currently in a long repeat run
      uint32_t mask = ballot(pos < numvals && s->vals[rolling_idx(pos)] == s->run_val);
      uint32_t rle_rpt_count, max_rpt_count;
      if (lane_id == 0) { s->rpt_map[warp_id] = mask; }
      __syncthreads();
      if (t < warp_size) {
        uint32_t c32 = ballot(t >= 4 || s->rpt_map[t] != 0xffff'ffffu);
        if (t == 0) {
          uint32_t last_idx = __ffs(c32) - 1;
          s->rle_rpt_count =
            last_idx * warp_size + ((last_idx < 4) ? __ffs(~s->rpt_map[last_idx]) - 1 : 0);
        }
      }
      __syncthreads();
      max_rpt_count = min(numvals - rle_pos, encode_block_size);
      rle_rpt_count = s->rle_rpt_count;
      rle_run += rle_rpt_count << 1;
      rle_pos += rle_rpt_count;
      if (rle_rpt_count < max_rpt_count || (flush && rle_pos == numvals)) {
        if (t == 0) {
          uint32_t const run_val = s->run_val;
          uint8_t* dst           = VlqEncode(s->rle_out, rle_run);
          *dst++                 = run_val;
          if (nbits > 8) { *dst++ = run_val >> 8; }
          if (nbits > 16) { *dst++ = run_val >> 16; }
          s->rle_out = dst;
        }
        rle_run = 0;
      }
    } else {
      // New run or in a literal run
      uint32_t v0      = s->vals[rolling_idx(pos)];
      uint32_t v1      = s->vals[rolling_idx(pos + 1)];
      uint32_t mask    = ballot(pos + 1 < numvals && v0 == v1);
      uint32_t maxvals = min(numvals - rle_pos, encode_block_size);
      uint32_t rle_lit_count, rle_rpt_count;
      if (lane_id == 0) { s->rpt_map[warp_id] = mask; }
      __syncthreads();
      if (t < warp_size) {
        // Repeat run can only start on a multiple of 8 values
        uint32_t idx8        = (t * 8) / warp_size;
        uint32_t pos8        = (t * 8) % warp_size;
        uint32_t m0          = (idx8 < 4) ? s->rpt_map[idx8] : 0;
        uint32_t m1          = (idx8 < 3) ? s->rpt_map[idx8 + 1] : 0;
        uint32_t needed_mask = kRleRunMask[nbits - 1];
        mask                 = ballot((__funnelshift_r(m0, m1, pos8) & needed_mask) == needed_mask);
        if (!t) {
          uint32_t rle_run_start = (mask != 0) ? min((__ffs(mask) - 1) * 8, maxvals) : maxvals;
          uint32_t rpt_len       = 0;
          if (rle_run_start < maxvals) {
            uint32_t idx_cur = rle_run_start / warp_size;
            uint32_t idx_ofs = rle_run_start % warp_size;
            while (idx_cur < 4) {
              m0   = (idx_cur < 4) ? s->rpt_map[idx_cur] : 0;
              m1   = (idx_cur < 3) ? s->rpt_map[idx_cur + 1] : 0;
              mask = ~__funnelshift_r(m0, m1, idx_ofs);
              if (mask != 0) {
                rpt_len += __ffs(mask) - 1;
                break;
              }
              rpt_len += warp_size;
              idx_cur++;
            }
          }
          s->rle_lit_count = rle_run_start;
          s->rle_rpt_count = min(rpt_len, maxvals - rle_run_start);
        }
      }
      __syncthreads();
      rle_lit_count = s->rle_lit_count;
      rle_rpt_count = s->rle_rpt_count;
      if (rle_lit_count != 0 || (rle_run != 0 && rle_rpt_count != 0)) {
        uint32_t lit_div8;
        bool need_more_data = false;
        if (!flush && rle_pos + rle_lit_count == numvals) {
          // Wait for more data
          rle_lit_count -= min(rle_lit_count, 24);
          need_more_data = true;
        }
        if (rle_lit_count != 0) {
          lit_div8 = (rle_lit_count + ((flush && rle_pos + rle_lit_count == numvals) ? 7 : 0)) >> 3;
          if (rle_run + lit_div8 * 2 > 0x7f) {
            lit_div8      = 0x3f - (rle_run >> 1);  // Limit to fixed 1-byte header (504 literals)
            rle_rpt_count = 0;                      // Defer repeat run
          }
          if (lit_div8 != 0) {
            uint8_t* dst = s->rle_out + 1 + (rle_run >> 1) * nbits;
            PackLiterals(dst, (rle_pos + t < numvals) ? v0 : 0, lit_div8 * 8, nbits, t);
            rle_run = (rle_run + lit_div8 * 2) | 1;
            rle_pos = min(rle_pos + lit_div8 * 8, numvals);
          }
        }
        if (rle_run >= ((rle_rpt_count != 0 || (flush && rle_pos == numvals)) ? 0x03 : 0x7f)) {
          __syncthreads();
          // Complete literal run
          if (!t) {
            uint8_t* dst = s->rle_out;
            dst[0]       = rle_run;  // At most 0x7f
            dst += 1 + nbits * (rle_run >> 1);
            s->rle_out = dst;
          }
          rle_run = 0;
        }
        if (need_more_data) { break; }
      }
      // Start a repeat run
      if (rle_rpt_count != 0) {
        if (t == s->rle_lit_count) { s->run_val = v0; }
        rle_run = rle_rpt_count * 2;
        rle_pos += rle_rpt_count;
        if (rle_pos + 1 == numvals && !flush) { break; }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (!t) {
    s->rle_run     = rle_run;
    s->rle_pos     = rle_pos;
    s->rle_numvals = numvals;
  }
}

/**
 * @brief PLAIN bool encoder
 *
 * @param[in,out] s Page encode state
 * @param[in] numvals Total count of input values
 * @param[in] flush nonzero if last batch in block
 * @param[in] t thread id (0..127)
 */
static __device__ void PlainBoolEncode(rle_page_enc_state_s* s,
                                       uint32_t numvals,
                                       uint32_t flush,
                                       uint32_t t)
{
  uint32_t rle_pos = s->rle_pos;
  uint8_t* dst     = s->rle_out;

  while (rle_pos < numvals) {
    uint32_t pos    = rle_pos + t;
    uint32_t v      = (pos < numvals) ? s->vals[rolling_idx(pos)] : 0;
    uint32_t n      = min(numvals - rle_pos, 128);
    uint32_t nbytes = (n + ((flush) ? 7 : 0)) >> 3;
    if (!nbytes) { break; }
    v |= shuffle_xor(v, 1) << 1;
    v |= shuffle_xor(v, 2) << 2;
    v |= shuffle_xor(v, 4) << 4;
    if (t < n && !(t & 7)) { dst[t >> 3] = v; }
    rle_pos = min(rle_pos + nbytes * 8, numvals);
    dst += nbytes;
  }
  __syncthreads();
  if (!t) {
    s->rle_pos     = rle_pos;
    s->rle_numvals = numvals;
    s->rle_out     = dst;
  }
}

/**
 * @brief Determines the difference between the Proleptic Gregorian Calendar epoch (1970-01-01
 * 00:00:00 UTC) and the Julian date epoch (-4713-11-24 12:00:00 UTC).
 *
 * @return The difference between two epochs in `cuda::std::chrono::duration` format with a period
 * of hours.
 */
__device__ constexpr auto julian_calendar_epoch_diff()
{
  using namespace cuda::std::chrono;
  using namespace cuda::std::chrono_literals;
  return sys_days{January / 1 / 1970} - (sys_days{November / 24 / -4713} + 12h);
}

/**
 * @brief Converts number `v` of periods of type `PeriodT` into a pair with nanoseconds since
 * midnight and number of Julian days. Does not deal with time zones. Used by INT96 code.
 *
 * @tparam PeriodT a ratio representing the tick period in duration
 * @param v count of ticks since epoch
 * @return A pair of (nanoseconds, days) where nanoseconds is the number of nanoseconds
 * elapsed in the day and days is the number of days from Julian epoch.
 */
template <typename PeriodT>
__device__ auto julian_days_with_time(int64_t v)
{
  using namespace cuda::std::chrono;
  auto const dur_total             = duration<int64_t, PeriodT>{v};
  auto const dur_days              = floor<days>(dur_total);
  auto const dur_time_of_day       = dur_total - dur_days;
  auto const dur_time_of_day_nanos = duration_cast<nanoseconds>(dur_time_of_day);
  auto const julian_days           = dur_days + ceil<days>(julian_calendar_epoch_diff());
  return cuda::std::pair{dur_time_of_day_nanos, julian_days};
}

// this has been split out into its own kernel because of the amount of shared memory required
// for the state buffer. encode kernels that don't use the RLE buffer can get started while
// the level data is encoded.
// blockDim(128, 1, 1)
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size, 8)
  gpuEncodePageLevels(device_span<EncPage> pages,
                      bool write_v2_headers,
                      encode_kernel_mask kernel_mask)
{
  __shared__ __align__(8) rle_page_enc_state_s state_g;

  auto* const s    = &state_g;
  uint32_t const t = threadIdx.x;

  if (t == 0) {
    state_g = rle_page_enc_state_s{};
    s->page = pages[blockIdx.x];
    s->ck   = *s->page.chunk;
    s->col  = *s->ck.col_desc;
    s->cur  = s->page.page_data + s->page.max_hdr_size;
    // init V2 info
    s->page.def_lvl_bytes = 0;
    s->page.rep_lvl_bytes = 0;
    s->page.num_nulls     = 0;
    s->rle_len_pos        = nullptr;
  }
  __syncthreads();

  if (BitAnd(s->page.kernel_mask, kernel_mask) == 0) { return; }

  auto const is_v2 = s->page.page_type == PageType::DATA_PAGE_V2;

  // Encode Repetition and Definition levels
  if (s->page.page_type != PageType::DICTIONARY_PAGE &&
      (s->col.num_def_level_bits()) != 0 &&  // This means max definition level is not 0 (nullable)
      (s->col.num_rep_level_bits()) == 0     // This means there are no repetition levels (non-list)
  ) {
    // Calculate definition levels from validity
    uint32_t def_lvl_bits = s->col.num_def_level_bits();
    if (def_lvl_bits != 0) {
      if (!t) {
        s->rle_run     = 0;
        s->rle_pos     = 0;
        s->rle_numvals = 0;
        s->rle_out     = s->cur;
        if (not is_v2) {
          s->rle_out += 4;  // save space for length
        }
      }
      __syncthreads();
      while (s->rle_numvals < s->page.num_rows) {
        uint32_t rle_numvals = s->rle_numvals;
        uint32_t nrows       = min(s->page.num_rows - rle_numvals, 128);
        auto row             = s->page.start_row + rle_numvals + t;
        // Definition level encodes validity. Checks the valid map and if it is valid, then sets the
        // def_lvl accordingly and sets it in s->vals which is then given to RleEncode to encode
        uint32_t def_lvl = [&]() {
          bool within_bounds = rle_numvals + t < s->page.num_rows && row < s->col.num_rows;
          if (not within_bounds) { return 0u; }
          uint32_t def       = 0;
          size_type l        = 0;
          bool is_col_struct = false;
          auto col           = *s->col.parent_column;
          do {
            // If col not nullable then it does not contribute to def levels
            if (s->col.nullability[l]) {
              if (col.is_valid(row)) {
                ++def;
              } else {
                // We have found the shallowest level at which this row is null
                break;
              }
            }
            is_col_struct = (col.type().id() == type_id::STRUCT);
            if (is_col_struct) {
              row += col.offset();
              col = col.child(0);
              ++l;
            }
          } while (is_col_struct);
          return def;
        }();
        s->vals[rolling_idx(rle_numvals + t)] = def_lvl;
        __syncthreads();
        // if max_def <= 1, then the histogram is trivial to calculate
        if (s->page.def_histogram != nullptr and s->col.max_def_level > 1) {
          // Only calculate up to max_def_level...the last entry is num_valid and will be filled
          // in later.
          generate_def_level_histogram<block_size>(
            s->page.def_histogram, s, nrows, rle_numvals, s->col.max_def_level);
        }
        rle_numvals += nrows;
        RleEncode(s, rle_numvals, def_lvl_bits, (rle_numvals == s->page.num_rows), t);
        __syncthreads();
      }
      if (t < 32) {
        uint8_t* const cur     = s->cur;
        uint8_t* const rle_out = s->rle_out;
        // V2 does not write the RLE length field
        uint32_t const rle_bytes =
          static_cast<uint32_t>(rle_out - cur) - (is_v2 ? 0 : RLE_LENGTH_FIELD_LEN);
        if (not is_v2 && t < RLE_LENGTH_FIELD_LEN) { cur[t] = rle_bytes >> (t * 8); }
        __syncwarp();
        if (t == 0) {
          s->cur                = rle_out;
          s->page.def_lvl_bytes = rle_bytes;
        }
      }
    }
  } else if (s->page.page_type != PageType::DICTIONARY_PAGE &&
             s->col.num_rep_level_bits() != 0  // This means there ARE repetition levels (has list)
  ) {
    auto encode_levels = [&](uint8_t const* lvl_val_data, uint32_t nbits, uint32_t& lvl_bytes) {
      // For list types, the repetition and definition levels are pre-calculated. We just need to
      // encode and write them now.
      if (!t) {
        s->rle_run     = 0;
        s->rle_pos     = 0;
        s->rle_numvals = 0;
        s->rle_out     = s->cur;
        if (not is_v2) {
          s->rle_out += 4;  // save space for length
        }
      }
      __syncthreads();
      size_type page_first_val_idx = s->col.level_offsets[s->page.start_row];
      size_type col_last_val_idx   = s->col.level_offsets[s->col.num_rows];
      while (s->rle_numvals < s->page.num_values) {
        uint32_t rle_numvals = s->rle_numvals;
        uint32_t nvals       = min(s->page.num_values - rle_numvals, 128);
        uint32_t idx         = page_first_val_idx + rle_numvals + t;
        uint32_t lvl_val =
          (rle_numvals + t < s->page.num_values && idx < col_last_val_idx) ? lvl_val_data[idx] : 0;
        s->vals[rolling_idx(rle_numvals + t)] = lvl_val;
        __syncthreads();
        rle_numvals += nvals;
        RleEncode(s, rle_numvals, nbits, (rle_numvals == s->page.num_values), t);
        __syncthreads();
      }
      if (t < 32) {
        uint8_t* const cur     = s->cur;
        uint8_t* const rle_out = s->rle_out;
        // V2 does not write the RLE length field
        uint32_t const rle_bytes =
          static_cast<uint32_t>(rle_out - cur) - (is_v2 ? 0 : RLE_LENGTH_FIELD_LEN);
        if (not is_v2 && t < RLE_LENGTH_FIELD_LEN) { cur[t] = rle_bytes >> (t * 8); }
        __syncwarp();
        if (t == 0) {
          s->cur    = rle_out;
          lvl_bytes = rle_bytes;
        }
      }
    };
    encode_levels(s->col.rep_values, s->col.num_rep_level_bits(), s->page.rep_lvl_bytes);
    __syncthreads();
    encode_levels(s->col.def_values, s->col.num_def_level_bits(), s->page.def_lvl_bytes);
  }

  if (t == 0) { pages[blockIdx.x] = s->page; }
}

template <int block_size, typename state_buf>
__device__ void finish_page_encode(state_buf* s,
                                   uint8_t const* end_ptr,
                                   device_span<EncPage> pages,
                                   device_span<device_span<uint8_t const>> comp_in,
                                   device_span<device_span<uint8_t>> comp_out,
                                   device_span<codec_exec_result> comp_results,
                                   bool write_v2_headers)
{
  auto const t = threadIdx.x;

  // returns sum of histogram values from [1..max_level)
  auto histogram_sum = [](uint32_t* const hist, int max_level) {
    auto const hist_start = hist + 1;
    auto const hist_end   = hist + max_level;
    return thrust::reduce(thrust::seq, hist_start, hist_end, 0U);
  };

  // this will be true if max_rep > 0 (i.e. there are lists)
  if (s->page.rep_histogram != nullptr) {
    // for repetition we get hist[0] from num_rows, and can derive hist[max_rep_level]
    if (s->col.max_rep_level > 1) {
      generate_page_histogram<block_size>(
        s->page.rep_histogram, s, s->col.rep_values, s->col.max_rep_level);
    }

    if (t == 0) {
      // rep_hist[0] is num_rows, we have rep_hist[1..max_rep_level) calculated, so
      // rep_hist[max_rep_level] is num_values minus the sum of the preceding values.
      s->page.rep_histogram[0] = s->page.num_rows;
      s->page.rep_histogram[s->col.max_rep_level] =
        s->page.num_values - s->page.num_rows -
        histogram_sum(s->page.rep_histogram, s->col.max_rep_level);
    }
    __syncthreads();

    if (s->page.def_histogram != nullptr) {
      // For definition, we know `hist[max_def_level] = num_valid`. If the leaf level is
      // nullable, then `hist[max_def_level - 1] = num_leaf_values - num_valid`. Finally,
      // hist[0] can be derived as `num_values - sum(hist[1]..hist[max_def_level])`.
      bool const is_leaf_nullable = s->col.leaf_column->nullable();
      auto const last_lvl = is_leaf_nullable ? s->col.max_def_level - 1 : s->col.max_def_level;
      if (last_lvl > 1) {
        generate_page_histogram<block_size>(s->page.def_histogram, s, s->col.def_values, last_lvl);
      }

      if (t == 0) {
        s->page.def_histogram[s->col.max_def_level] = s->page.num_valid;
        if (is_leaf_nullable) {
          s->page.def_histogram[last_lvl] = s->page.num_leaf_values - s->page.num_valid;
        }
        s->page.def_histogram[0] = s->page.num_values - s->page.num_leaf_values -
                                   histogram_sum(s->page.def_histogram, last_lvl);
      }
    }
  } else if (s->page.def_histogram != nullptr) {
    // finish off what was started in generate_def_level_histogram
    if (t == 0) {
      // `hist[max_def_level] = num_valid`, and the values for hist[1..max_def_level) are known
      s->page.def_histogram[s->col.max_def_level] = s->page.num_valid;
      s->page.def_histogram[0]                    = s->page.num_values - s->page.num_valid -
                                 histogram_sum(s->page.def_histogram, s->col.max_def_level);
    }
  }

  if (t == 0) {
    // only need num_nulls for v2 data page headers
    if (write_v2_headers) { s->page.num_nulls = s->page.num_values - s->page.num_valid; }
    uint8_t const* const base   = s->page.page_data + s->page.max_hdr_size;
    auto const actual_data_size = static_cast<uint32_t>(end_ptr - base);
    if (actual_data_size > s->page.max_data_size) {
      // FIXME(ets): this needs to do error propagation back to the host
      CUDF_UNREACHABLE("detected possible page data corruption");
    }
    if (s->page.is_v2()) {
      auto const d_base = base + s->page.max_lvl_size;
      s->page.data_size = static_cast<uint32_t>(end_ptr - d_base) + s->page.level_bytes();
    } else {
      s->page.data_size = actual_data_size;
    }
    if (not comp_in.empty()) {
      auto const c_base            = base + s->page.max_lvl_size;
      auto const bytes_to_compress = static_cast<uint32_t>(end_ptr - c_base);
      comp_in[blockIdx.x]          = {c_base, bytes_to_compress};
      comp_out[blockIdx.x] = {s->page.compressed_data + s->page.max_hdr_size + s->page.max_lvl_size,
                              s->page.comp_data_size};
    }
    pages[blockIdx.x] = s->page;
    if (not comp_results.empty()) {
      auto const status = s->col.skip_compression ? codec_status::SKIPPED : codec_status::FAILURE;
      comp_results[blockIdx.x]   = {0, status};
      pages[blockIdx.x].comp_res = &comp_results[blockIdx.x];
    }
  }

  // copy uncompressed bytes over
  if (s->page.is_v2() and not comp_in.empty()) {
    uint8_t* const src = s->page.page_data + s->page.max_hdr_size;
    uint8_t* const dst = s->page.compressed_data + s->page.max_hdr_size;
    for (int i = t; i < s->page.level_bytes(); i += block_size) {
      dst[i] = src[i];
    }
  }
}

// Encode a fixed-width data type int `dst`. `dst` points to the first byte
// of the result. `stride` is 1 for PLAIN encoding and num_values for
// BYTE_STREAM_SPLIT.
template <typename T>
__device__ inline void encode_value(uint8_t* dst, T src, size_type stride)
{
  T v = src;
  for (int i = 0; i < sizeof(T); i++) {
    dst[i * stride] = v;
    v >>= 8;
  }
}

// PLAIN page data encoder
// blockDim(128, 1, 1)
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size, 8)
  gpuEncodePages(device_span<EncPage> pages,
                 device_span<device_span<uint8_t const>> comp_in,
                 device_span<device_span<uint8_t>> comp_out,
                 device_span<codec_exec_result> comp_results,
                 bool write_v2_headers,
                 bool is_split_stream)
{
  __shared__ __align__(8) page_enc_state_s<0> state_g;
  using block_scan = cub::BlockScan<uint32_t, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;

  auto* const s = &state_g;
  uint32_t t    = threadIdx.x;

  if (t == 0) {
    state_g        = page_enc_state_s<0>{};
    s->page        = pages[blockIdx.x];
    s->ck          = *s->page.chunk;
    s->col         = *s->ck.col_desc;
    s->rle_len_pos = nullptr;
    // get s->cur back to where it was at the end of encoding the rep and def level data
    set_page_data_start(s);
  }
  __syncthreads();

  auto const allowed_mask =
    is_split_stream ? encode_kernel_mask::BYTE_STREAM_SPLIT : encode_kernel_mask::PLAIN;
  if (BitAnd(s->page.kernel_mask, allowed_mask) == 0) { return; }

  // Encode data values
  __syncthreads();
  auto const physical_type = s->col.physical_type;
  auto const type_id       = s->col.leaf_column->type().id();
  auto const dtype_len_out = physical_type_len(physical_type, type_id, s->col.type_length);
  auto const dtype_len_in  = [&]() -> uint32_t {
    if (physical_type == Type::INT32) { return int32_logical_len(type_id); }
    if (physical_type == Type::INT96) { return sizeof(int64_t); }
    return dtype_len_out;
  }();

  if (t == 0) {
    uint8_t* dst     = s->cur;
    s->rle_run       = 0;
    s->rle_pos       = 0;
    s->rle_numvals   = 0;
    s->rle_out       = dst;
    s->page.encoding = determine_encoding(
      s->page.page_type, physical_type, s->ck.use_dictionary, write_v2_headers, is_split_stream);
    s->page_start_val  = row_to_value_idx(s->page.start_row, s->col);
    s->chunk_start_val = row_to_value_idx(s->ck.start_row, s->col);
  }
  __syncthreads();

  auto const stride = is_split_stream ? s->page.num_valid : 1;

  for (uint32_t cur_val_idx = 0; cur_val_idx < s->page.num_leaf_values;) {
    uint32_t nvals = min(s->page.num_leaf_values - cur_val_idx, block_size);
    uint32_t len, pos;

    auto [is_valid, val_idx] = [&]() {
      uint32_t val_idx;
      uint32_t is_valid;

      size_type const val_idx_in_block = cur_val_idx + t;
      if (s->page.page_type == PageType::DICTIONARY_PAGE) {
        val_idx  = val_idx_in_block;
        is_valid = (val_idx < s->page.num_leaf_values);
        if (is_valid) { val_idx = s->ck.dict_data[val_idx]; }
      } else {
        size_type const val_idx_in_leaf_col = s->page_start_val + val_idx_in_block;

        is_valid = (val_idx_in_leaf_col < s->col.leaf_column->size() &&
                    val_idx_in_block < s->page.num_leaf_values)
                     ? s->col.leaf_column->is_valid(val_idx_in_leaf_col)
                     : 0;
        val_idx  = val_idx_in_leaf_col;
      }
      return cuda::std::make_tuple(is_valid, val_idx);
    }();

    cur_val_idx += nvals;

    // Non-dictionary encoding
    uint8_t* dst = s->cur;

    if (is_valid) {
      len = dtype_len_out;
      if (physical_type == Type::BYTE_ARRAY) {
        if (type_id == type_id::STRING) {
          len += s->col.leaf_column->element<string_view>(val_idx).size_bytes();
        } else if (s->col.output_as_byte_array && type_id == type_id::LIST) {
          len +=
            get_element<statistics::byte_array_view>(*s->col.leaf_column, val_idx).size_bytes();
        }
      }
    } else {
      len = 0;
    }
    uint32_t total_len = 0;
    block_scan(scan_storage).ExclusiveSum(len, pos, total_len);
    __syncthreads();

    // if BYTE_STREAM_SPLIT, then translate byte positions to indexes
    if (is_split_stream) {
      pos /= dtype_len_out;
      total_len /= dtype_len_out;
    }

    if (t == 0) { s->cur = dst + total_len; }
    if (is_valid) {
      switch (physical_type) {
        case Type::INT32: [[fallthrough]];
        case Type::FLOAT: {
          auto const v = [dtype_len = dtype_len_in,
                          idx       = val_idx,
                          col       = s->col.leaf_column,
                          scale     = s->col.ts_scale == 0 ? 1 : s->col.ts_scale]() -> int32_t {
            switch (dtype_len) {
              case 8: return col->element<int64_t>(idx) * scale;
              case 4: return col->element<int32_t>(idx) * scale;
              case 2: return col->element<int16_t>(idx) * scale;
              default: return col->element<int8_t>(idx) * scale;
            }
          }();

          encode_value(dst + pos, v, stride);
        } break;
        case Type::DOUBLE:
        case Type::INT64: {
          auto v           = s->col.leaf_column->element<int64_t>(val_idx);
          int32_t ts_scale = s->col.ts_scale;
          if (ts_scale != 0) {
            if (ts_scale < 0) {
              v /= -ts_scale;
            } else {
              v *= ts_scale;
            }
          }
          encode_value(dst + pos, v, stride);
        } break;
        case Type::INT96: {
          // only PLAIN encoding is supported
          int64_t v        = s->col.leaf_column->element<int64_t>(val_idx);
          int32_t ts_scale = s->col.ts_scale;
          if (ts_scale != 0) {
            if (ts_scale < 0) {
              v /= -ts_scale;
            } else {
              v *= ts_scale;
            }
          }

          auto const [last_day_nanos, julian_days] = [&] {
            using namespace cuda::std::chrono;
            switch (s->col.leaf_column->type().id()) {
              case type_id::TIMESTAMP_SECONDS:
              case type_id::TIMESTAMP_MILLISECONDS: {
                return julian_days_with_time<cuda::std::milli>(v);
              } break;
              case type_id::TIMESTAMP_MICROSECONDS:
              case type_id::TIMESTAMP_NANOSECONDS: {
                return julian_days_with_time<cuda::std::micro>(v);
              } break;
            }
            return julian_days_with_time<cuda::std::nano>(0);
          }();

          // the 12 bytes of fixed length data.
          v = last_day_nanos.count();
          encode_value(dst + pos, v, 1);
          uint32_t w = julian_days.count();
          encode_value(dst + pos + 8, w, 1);
        } break;

        case Type::BYTE_ARRAY: {
          // only PLAIN encoding is supported
          auto const bytes = [](cudf::type_id const type_id,
                                column_device_view const* leaf_column,
                                uint32_t const val_idx) -> void const* {
            switch (type_id) {
              case type_id::STRING:
                return reinterpret_cast<void const*>(
                  leaf_column->element<string_view>(val_idx).data());
              case type_id::LIST:
                return reinterpret_cast<void const*>(
                  get_element<statistics::byte_array_view>(*(leaf_column), val_idx).data());
              default: CUDF_UNREACHABLE("invalid type id for byte array writing!");
            }
          }(type_id, s->col.leaf_column, val_idx);
          uint32_t v = len - 4;  // string length
          encode_value(dst + pos, v, 1);
          if (v != 0) memcpy(dst + pos + 4, bytes, v);
        } break;
        case Type::FIXED_LEN_BYTE_ARRAY: {
          if (type_id == type_id::DECIMAL128) {
            // When using FIXED_LEN_BYTE_ARRAY for decimals, the rep is encoded in big-endian
            auto const v = s->col.leaf_column->element<numeric::decimal128>(val_idx).value();
            auto const v_char_ptr = reinterpret_cast<char const*>(&v);
            if (is_split_stream) {
              for (int i = dtype_len_out - 1; i >= 0; i--, pos += stride) {
                dst[pos] = v_char_ptr[i];
              }
            } else {
              thrust::copy(thrust::seq,
                           cuda::std::make_reverse_iterator(v_char_ptr + sizeof(v)),
                           cuda::std::make_reverse_iterator(v_char_ptr),
                           dst + pos);
            }
          } else {
            auto const elem =
              get_element<statistics::byte_array_view>(*(s->col.leaf_column), val_idx);
            if (len != 0 and elem.data() != nullptr) {
              if (is_split_stream) {
                auto const v_char_ptr = reinterpret_cast<uint8_t const*>(elem.data());
                for (int i = 0; i < dtype_len_out; i++, pos += stride) {
                  dst[pos] = v_char_ptr[i];
                }
              } else {
                memcpy(dst + pos, elem.data(), len);
              }
            }
          }
        } break;
      }
    }
    __syncthreads();
  }

  // for BYTE_STREAM_SPLIT, s->cur now points to the end of the first stream.
  // need it to point to the end of the Nth stream.
  if (is_split_stream and t == 0) { s->cur += (dtype_len_out - 1) * s->page.num_valid; }
  __syncthreads();
  finish_page_encode<block_size>(
    s, s->cur, pages, comp_in, comp_out, comp_results, write_v2_headers);
}

// DICTIONARY page data encoder
// blockDim(128, 1, 1)
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size, 8)
  gpuEncodeDictPages(device_span<EncPage> pages,
                     device_span<device_span<uint8_t const>> comp_in,
                     device_span<device_span<uint8_t>> comp_out,
                     device_span<codec_exec_result> comp_results,
                     bool write_v2_headers)
{
  __shared__ __align__(8) rle_page_enc_state_s state_g;
  using block_scan = cub::BlockScan<uint32_t, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;

  auto* const s = &state_g;
  uint32_t t    = threadIdx.x;

  if (t == 0) {
    state_g        = rle_page_enc_state_s{};
    s->page        = pages[blockIdx.x];
    s->ck          = *s->page.chunk;
    s->col         = *s->ck.col_desc;
    s->rle_len_pos = nullptr;
    // get s->cur back to where it was at the end of encoding the rep and def level data
    set_page_data_start(s);
  }
  __syncthreads();

  if (BitAnd(s->page.kernel_mask, encode_kernel_mask::DICTIONARY) == 0) { return; }

  // Encode data values
  auto const physical_type = s->col.physical_type;
  auto const type_id       = s->col.leaf_column->type().id();
  auto const dtype_len_out = physical_type_len(physical_type, type_id, s->col.type_length);
  auto const dtype_len_in  = [&]() -> uint32_t {
    if (physical_type == Type::INT32) { return int32_logical_len(type_id); }
    if (physical_type == Type::INT96) { return sizeof(int64_t); }
    return dtype_len_out;
  }();

  // TODO assert dict_bits >= 0
  auto const dict_bits = (physical_type == Type::BOOLEAN) ? 1
                         : (s->ck.use_dictionary and s->page.page_type != PageType::DICTIONARY_PAGE)
                           ? s->page.dict_rle_bits  // Use `page.dict_rle_bits` for data pages
                           : -1;
  if (t == 0) {
    uint8_t* dst     = s->cur;
    s->rle_run       = 0;
    s->rle_pos       = 0;
    s->rle_numvals   = 0;
    s->rle_out       = dst;
    s->page.encoding = determine_encoding(
      s->page.page_type, physical_type, s->ck.use_dictionary, write_v2_headers, false);
    if (dict_bits >= 0 && physical_type != Type::BOOLEAN) {
      dst[0]     = dict_bits;
      s->rle_out = dst + 1;
    } else if (write_v2_headers && physical_type == Type::BOOLEAN) {
      // save space for RLE length. we don't know the total length yet.
      s->rle_out     = dst + RLE_LENGTH_FIELD_LEN;
      s->rle_len_pos = dst;
    }
    s->cur             = s->rle_out;
    s->page_start_val  = row_to_value_idx(s->page.start_row, s->col);
    s->chunk_start_val = row_to_value_idx(s->ck.start_row, s->col);
  }
  __syncthreads();

  for (uint32_t cur_val_idx = 0; cur_val_idx < s->page.num_leaf_values;) {
    uint32_t nvals = min(s->page.num_leaf_values - cur_val_idx, block_size);

    auto [is_valid, val_idx] = [&]() {
      size_type const val_idx_in_block    = cur_val_idx + t;
      size_type const val_idx_in_leaf_col = s->page_start_val + val_idx_in_block;

      uint32_t const is_valid = (val_idx_in_leaf_col < s->col.leaf_column->size() &&
                                 val_idx_in_block < s->page.num_leaf_values)
                                  ? s->col.leaf_column->is_valid(val_idx_in_leaf_col)
                                  : 0;
      // need to test for use_dictionary because it might be boolean
      uint32_t const val_idx =
        (s->ck.use_dictionary) ? val_idx_in_leaf_col - s->chunk_start_val : val_idx_in_leaf_col;
      return cuda::std::tuple{is_valid, val_idx};
    }();

    cur_val_idx += nvals;

    // Dictionary encoding
    if (dict_bits > 0) {
      uint32_t rle_numvals;
      uint32_t rle_numvals_in_block;
      uint32_t pos;
      block_scan(scan_storage).ExclusiveSum(is_valid, pos, rle_numvals_in_block);
      rle_numvals = s->rle_numvals;
      if (is_valid) {
        uint32_t v;
        if (physical_type == Type::BOOLEAN) {
          v = s->col.leaf_column->element<uint8_t>(val_idx);
        } else {
          v = s->ck.dict_index[val_idx];
        }
        s->vals[rolling_idx(rle_numvals + pos)] = v;
      }
      rle_numvals += rle_numvals_in_block;
      __syncthreads();
      if ((!write_v2_headers) && (physical_type == Type::BOOLEAN)) {
        PlainBoolEncode(s, rle_numvals, (cur_val_idx == s->page.num_leaf_values), t);
      } else {
        RleEncode(s, rle_numvals, dict_bits, (cur_val_idx == s->page.num_leaf_values), t);
      }
      __syncthreads();
    }
    if (t == 0) { s->cur = s->rle_out; }
    __syncthreads();
  }

  // save RLE length if necessary
  if (s->rle_len_pos != nullptr && t < 32) {
    // size doesn't include the 4 bytes for the length
    auto const rle_size = static_cast<uint32_t>(s->cur - s->rle_len_pos) - RLE_LENGTH_FIELD_LEN;
    if (t < RLE_LENGTH_FIELD_LEN) { s->rle_len_pos[t] = rle_size >> (t * 8); }
    __syncwarp();
  }

  finish_page_encode<block_size>(
    s, s->cur, pages, comp_in, comp_out, comp_results, write_v2_headers);
}

// DELTA_BINARY_PACKED page data encoder
// blockDim(128, 1, 1)
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size, 8)
  gpuEncodeDeltaBinaryPages(device_span<EncPage> pages,
                            device_span<device_span<uint8_t const>> comp_in,
                            device_span<device_span<uint8_t>> comp_out,
                            device_span<codec_exec_result> comp_results)
{
  // block of shared memory for value storage and bit packing
  __shared__ uleb128_t delta_shared[delta::buffer_size + delta::block_size];
  __shared__ __align__(8) page_enc_state_s<0> state_g;
  __shared__ union {
    typename delta_binary_packer<uleb128_t>::index_scan::TempStorage delta_index_tmp;
    typename delta_binary_packer<uleb128_t>::block_reduce::TempStorage delta_reduce_tmp;
    typename delta_binary_packer<uleb128_t>::warp_reduce::TempStorage
      delta_warp_red_tmp[delta::num_mini_blocks];
  } temp_storage;

  auto* const s = &state_g;
  uint32_t t    = threadIdx.x;

  if (t == 0) {
    state_g        = page_enc_state_s<0>{};
    s->page        = pages[blockIdx.x];
    s->ck          = *s->page.chunk;
    s->col         = *s->ck.col_desc;
    s->rle_len_pos = nullptr;
    // get s->cur back to where it was at the end of encoding the rep and def level data
    set_page_data_start(s);
  }
  __syncthreads();

  if (BitAnd(s->page.kernel_mask, encode_kernel_mask::DELTA_BINARY) == 0) { return; }

  // Encode data values
  auto const physical_type = s->col.physical_type;
  auto const type_id       = s->col.leaf_column->type().id();
  auto const dtype_len_out = physical_type_len(physical_type, type_id, s->col.type_length);
  auto const dtype_len_in  = [&]() -> uint32_t {
    if (physical_type == Type::INT32) { return int32_logical_len(type_id); }
    if (physical_type == Type::INT96) { return sizeof(int64_t); }
    return dtype_len_out;
  }();

  if (t == 0) {
    uint8_t* dst       = s->cur;
    s->rle_run         = 0;
    s->rle_pos         = 0;
    s->rle_numvals     = 0;
    s->rle_out         = dst;
    s->page.encoding   = Encoding::DELTA_BINARY_PACKED;
    s->page_start_val  = row_to_value_idx(s->page.start_row, s->col);
    s->chunk_start_val = row_to_value_idx(s->ck.start_row, s->col);
  }
  __syncthreads();

  uint8_t const* delta_ptr = nullptr;  // this will be the end of delta block pointer

  if (physical_type == Type::INT32) {
    switch (dtype_len_in) {
      case 8: {
        // only DURATIONS map to 8 bytes, so safe to just use signed here?
        delta_ptr = delta_encode<Type::INT32, int64_t>(s, delta_shared, &temp_storage);
        break;
      }
      case 4: {
        if (type_id == type_id::UINT32) {
          delta_ptr = delta_encode<Type::INT32, uint32_t>(s, delta_shared, &temp_storage);
        } else {
          delta_ptr = delta_encode<Type::INT32, int32_t>(s, delta_shared, &temp_storage);
        }
        break;
      }
      case 2: {
        if (type_id == type_id::UINT16) {
          delta_ptr = delta_encode<Type::INT32, uint16_t>(s, delta_shared, &temp_storage);
        } else {
          delta_ptr = delta_encode<Type::INT32, int16_t>(s, delta_shared, &temp_storage);
        }
        break;
      }
      case 1: {
        if (type_id == type_id::UINT8) {
          delta_ptr = delta_encode<Type::INT32, uint8_t>(s, delta_shared, &temp_storage);
        } else {
          delta_ptr = delta_encode<Type::INT32, int8_t>(s, delta_shared, &temp_storage);
        }
        break;
      }
      default: CUDF_UNREACHABLE("invalid dtype_len_in when encoding DELTA_BINARY_PACKED");
    }
  } else {
    if (type_id == type_id::UINT64) {
      delta_ptr = delta_encode<Type::INT64, uint64_t>(s, delta_shared, &temp_storage);
    } else {
      delta_ptr = delta_encode<Type::INT64, int64_t>(s, delta_shared, &temp_storage);
    }
  }

  finish_page_encode<block_size>(s, delta_ptr, pages, comp_in, comp_out, comp_results, true);
}

// DELTA_LENGTH_BYTE_ARRAY page data encoder
// blockDim(128, 1, 1)
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size, 8)
  gpuEncodeDeltaLengthByteArrayPages(device_span<EncPage> pages,
                                     device_span<device_span<uint8_t const>> comp_in,
                                     device_span<device_span<uint8_t>> comp_out,
                                     device_span<codec_exec_result> comp_results)
{
  // block of shared memory for value storage and bit packing
  __shared__ uleb128_t delta_shared[delta::buffer_size + delta::block_size];
  __shared__ __align__(8) page_enc_state_s<0> state_g;
  __shared__ delta_binary_packer<int32_t> packer;
  __shared__ uint8_t const* first_string;
  __shared__ size_type string_data_len;
  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  __shared__ union {
    typename block_reduce::TempStorage reduce_storage;
    typename delta_binary_packer<uleb128_t>::index_scan::TempStorage delta_index_tmp;
    typename delta_binary_packer<uleb128_t>::block_reduce::TempStorage delta_reduce_tmp;
    typename delta_binary_packer<uleb128_t>::warp_reduce::TempStorage
      delta_warp_red_tmp[delta::num_mini_blocks];
  } temp_storage;

  auto* const s = &state_g;
  uint32_t t    = threadIdx.x;

  if (t == 0) {
    state_g        = page_enc_state_s<0>{};
    s->page        = pages[blockIdx.x];
    s->ck          = *s->page.chunk;
    s->col         = *s->ck.col_desc;
    s->rle_len_pos = nullptr;
    // get s->cur back to where it was at the end of encoding the rep and def level data
    set_page_data_start(s);
  }
  __syncthreads();

  if (BitAnd(s->page.kernel_mask, encode_kernel_mask::DELTA_LENGTH_BA) == 0) { return; }

  // Encode data values
  if (t == 0) {
    uint8_t* dst       = s->cur;
    s->rle_run         = 0;
    s->rle_pos         = 0;
    s->rle_numvals     = 0;
    s->rle_out         = dst;
    s->page.encoding   = Encoding::DELTA_LENGTH_BYTE_ARRAY;
    s->page_start_val  = row_to_value_idx(s->page.start_row, s->col);
    s->chunk_start_val = row_to_value_idx(s->ck.start_row, s->col);
  }
  __syncthreads();

  auto const type_id = s->col.leaf_column->type().id();

  // encode the lengths as DELTA_BINARY_PACKED
  if (t == 0) {
    first_string = nullptr;
    packer.init(s->cur, s->page.num_valid, reinterpret_cast<int32_t*>(delta_shared), &temp_storage);

    // if there are valid values, find a pointer to the first valid string
    if (s->page.num_valid != 0) {
      for (uint32_t idx = 0; idx < s->page.num_leaf_values; idx++) {
        size_type const idx_in_col = s->page_start_val + idx;
        if (s->col.leaf_column->is_valid(idx_in_col)) {
          if (type_id == type_id::STRING) {
            first_string = reinterpret_cast<uint8_t const*>(
              s->col.leaf_column->element<string_view>(idx_in_col).data());
          } else if (s->col.output_as_byte_array && type_id == type_id::LIST) {
            first_string = reinterpret_cast<uint8_t const*>(
              get_element<statistics::byte_array_view>(*s->col.leaf_column, idx_in_col).data());
          }
          break;
        }
      }
    }
  }
  __syncthreads();

  uint32_t len = 0;
  for (uint32_t cur_val_idx = 0; cur_val_idx < s->page.num_leaf_values;) {
    uint32_t const nvals = min(s->page.num_leaf_values - cur_val_idx, delta::block_size);

    size_type const val_idx_in_block = cur_val_idx + t;
    size_type const val_idx          = s->page_start_val + val_idx_in_block;

    bool const is_valid =
      (val_idx < s->col.leaf_column->size() && val_idx_in_block < s->page.num_leaf_values)
        ? s->col.leaf_column->is_valid(val_idx)
        : false;

    cur_val_idx += nvals;

    int32_t v = 0;
    if (is_valid) {
      if (type_id == type_id::STRING) {
        v = s->col.leaf_column->element<string_view>(val_idx).size_bytes();
      } else if (s->col.output_as_byte_array && type_id == type_id::LIST) {
        auto const arr_size =
          get_element<statistics::byte_array_view>(*s->col.leaf_column, val_idx).size_bytes();
        // the lengths are assumed to be INT32, check for overflow
        if (arr_size > static_cast<size_t>(cuda::std::numeric_limits<int32_t>::max())) {
          CUDF_UNREACHABLE("byte array size exceeds 2GB");
        }
        v = static_cast<int32_t>(arr_size);
      }
      len += v;
    }

    packer.add_value(v, is_valid);
  }

  // string_len is only valid on thread 0
  auto const string_len = block_reduce(temp_storage.reduce_storage).Sum(len);
  if (t == 0) { string_data_len = string_len; }
  __syncthreads();

  // finish off the delta block and get the pointer to the end of the delta block
  auto const output_ptr = packer.flush();

  // now copy the char data
  memcpy_block<block_size, true>(
    output_ptr, first_string, string_data_len, cooperative_groups::this_thread_block());

  finish_page_encode<block_size>(
    s, output_ptr + string_data_len, pages, comp_in, comp_out, comp_results, true);
}

struct byte_array {
  uint8_t const* data;
  size_type length;

  // calculate the amount of overlap with a preceding array
  __device__ size_type common_prefix_length(byte_array const& preceding) const
  {
    auto const max_pref_len = min(length, preceding.length);
    size_type idx           = 0;
    while (idx < max_pref_len and data[idx] == preceding.data[idx]) {
      idx++;
    }
    return idx;
  }
};

// DELTA_BYTE_ARRAY page data encoder
// blockDim(128, 1, 1)
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size, 8)
  gpuEncodeDeltaByteArrayPages(device_span<EncPage> pages,
                               device_span<device_span<uint8_t const>> comp_in,
                               device_span<device_span<uint8_t>> comp_out,
                               device_span<codec_exec_result> comp_results)
{
  using cudf::detail::warp_size;
  // block of shared memory for value storage and bit packing
  __shared__ uleb128_t delta_shared[delta::buffer_size + delta::block_size];
  __shared__ __align__(8) page_enc_state_s<0> state_g;
  __shared__ delta_binary_packer<int32_t> packer;
  __shared__ uint8_t* scratch_data;
  __shared__ size_t avg_suffix_len;
  using block_scan   = cub::BlockScan<size_type, block_size>;
  using block_reduce = cub::BlockReduce<size_t, block_size>;
  __shared__ union {
    typename block_scan::TempStorage scan_storage;
    typename block_reduce::TempStorage reduce_storage;
    typename delta_binary_packer<uleb128_t>::index_scan::TempStorage delta_index_tmp;
    typename delta_binary_packer<uleb128_t>::block_reduce::TempStorage delta_reduce_tmp;
    typename delta_binary_packer<uleb128_t>::warp_reduce::TempStorage
      delta_warp_red_tmp[delta::num_mini_blocks];
  } temp_storage;

  auto* const s = &state_g;
  uint32_t t    = threadIdx.x;

  if (t == 0) {
    state_g        = page_enc_state_s<0>{};
    s->page        = pages[blockIdx.x];
    s->ck          = *s->page.chunk;
    s->col         = *s->ck.col_desc;
    s->rle_len_pos = nullptr;
    // get s->cur back to where it was at the end of encoding the rep and def level data
    set_page_data_start(s);
  }
  __syncthreads();

  if (BitAnd(s->page.kernel_mask, encode_kernel_mask::DELTA_BYTE_ARRAY) == 0) { return; }

  // Encode data values
  if (t == 0) {
    uint8_t* dst       = s->cur;
    s->rle_run         = 0;
    s->rle_pos         = 0;
    s->rle_numvals     = 0;
    s->rle_out         = dst;
    s->page.encoding   = Encoding::DELTA_BYTE_ARRAY;
    s->page_start_val  = row_to_value_idx(s->page.start_row, s->col);
    s->chunk_start_val = row_to_value_idx(s->ck.start_row, s->col);

    // set pointer to beginning of scratch space (aligned to size_type boundary)
    auto scratch_start =
      reinterpret_cast<uintptr_t>(s->page.page_data + s->page.max_hdr_size + s->page.max_data_size);
    scratch_start = util::round_up_unsafe(scratch_start, sizeof(size_type));
    scratch_data  = reinterpret_cast<uint8_t*>(scratch_start);
  }
  __syncthreads();

  // create offsets map (if needed)
  // We only encode valid values, and we need to know adjacent valid strings. So first we'll
  // create a mapping of leaf indexes to valid indexes:
  //
  // validity array is_valid:
  //   1 1 0 1 0 1 1 0
  //
  // exclusive scan on is_valid yields mapping of leaf index -> valid index:
  //   0 1 2 2 3 3 4 5
  //
  // Last value should equal page.num_valid. Now we need to transform that into a reverse
  // lookup that maps valid index -> leaf index (of length num_valid):
  //   0 1 3 5 6
  //
  auto const has_leaf_nulls = s->page.num_valid != s->page.num_leaf_values;

  size_type* const offsets_map =
    has_leaf_nulls ? reinterpret_cast<size_type*>(scratch_data) : nullptr;

  if (offsets_map != nullptr) {
    size_type* const forward_map = offsets_map + s->page.num_valid;

    // create the validity array
    for (int idx = t; idx < s->page.num_leaf_values; idx += block_size) {
      size_type const idx_in_col = s->page_start_val + idx;
      bool const is_valid =
        idx_in_col < s->col.leaf_column->size() and s->col.leaf_column->is_valid(idx_in_col);
      forward_map[idx] = is_valid ? 1 : 0;
    }
    __syncthreads();

    // exclusive scan to get leaf_idx -> valid_idx
    block_excl_sum<block_size>(forward_map, s->page.num_leaf_values, 0);

    // now reverse map to get valid_idx -> leaf_idx mapping
    for (int idx = t; idx < s->page.num_leaf_values; idx += block_size) {
      size_type const idx_in_col = s->page_start_val + idx;
      bool const is_valid =
        idx_in_col < s->col.leaf_column->size() and s->col.leaf_column->is_valid(idx_in_col);
      if (is_valid) { offsets_map[forward_map[idx]] = idx; }
    }
    __syncthreads();
  }

  size_type* const prefix_lengths =
    has_leaf_nulls ? offsets_map + s->page.num_valid : reinterpret_cast<size_type*>(scratch_data);

  auto const type_id = s->col.leaf_column->type().id();

  auto const byte_array_at = [type_id, s](size_type idx) -> byte_array {
    if (type_id == type_id::STRING) {
      auto const str = s->col.leaf_column->element<string_view>(idx);
      return {reinterpret_cast<uint8_t const*>(str.data()), str.size_bytes()};
    } else if (s->col.output_as_byte_array && type_id == type_id::LIST) {
      auto const str = get_element<statistics::byte_array_view>(*s->col.leaf_column, idx);
      return {reinterpret_cast<uint8_t const*>(str.data()),
              static_cast<size_type>(str.size_bytes())};
    }
    return {nullptr, 0};
  };

  // Calculate prefix lengths. The first prefix length is always 0. loop over num_valid since we
  // only encode valid values.
  // Note: calculating this on a string-per-thread basis seems bad for large strings with lots
  // of overlap. But in testing, it was found that the string copy at the end had a much larger
  // impact on performance, and doing this step on a string-per-warp basis was always slower.
  if (t == 0) { prefix_lengths[0] = 0; }
  for (int idx = t + 1; idx < s->page.num_valid; idx += block_size) {
    size_type const leaf_idx  = has_leaf_nulls ? offsets_map[idx] : idx;
    size_type const pleaf_idx = has_leaf_nulls ? offsets_map[idx - 1] : idx - 1;

    // get this string and the preceding string
    auto const current   = byte_array_at(leaf_idx + s->page_start_val);
    auto const preceding = byte_array_at(pleaf_idx + s->page_start_val);

    // calculate the amount of overlap
    prefix_lengths[idx] = current.common_prefix_length(preceding);
  }

  // encode prefix lengths
  if (t == 0) {
    packer.init(s->cur, s->page.num_valid, reinterpret_cast<int32_t*>(delta_shared), &temp_storage);
  }
  __syncthreads();

  // don't start at `t` because all threads must participate in each iteration
  for (int idx = 0; idx < s->page.num_valid; idx += block_size) {
    size_type const t_idx = idx + t;
    auto const in_range   = t_idx < s->page.num_valid;
    auto const val        = in_range ? prefix_lengths[t_idx] : 0;
    packer.add_value(val, in_range);
  }

  auto const suffix_ptr = packer.flush();
  __syncthreads();

  // encode suffix lengths
  if (t == 0) {
    packer.init(
      suffix_ptr, s->page.num_valid, reinterpret_cast<int32_t*>(delta_shared), &temp_storage);
  }
  __syncthreads();

  size_t non_zero     = 0;
  size_t suffix_bytes = 0;

  for (int idx = 0; idx < s->page.num_valid; idx += block_size) {
    size_type const t_idx = idx + t;
    auto const in_range   = t_idx < s->page.num_valid;
    int32_t val           = 0;
    if (in_range) {
      size_type const leaf_idx = has_leaf_nulls ? offsets_map[t_idx] : t_idx;
      auto const byte_arr      = byte_array_at(leaf_idx + s->page_start_val);
      val                      = byte_arr.length - prefix_lengths[t_idx];
      if (val > 0) {
        non_zero++;
        suffix_bytes += val;
      }
    }
    packer.add_value(val, in_range);
  }

  auto const strings_ptr = packer.flush();

  non_zero = block_reduce(temp_storage.reduce_storage).Sum(non_zero);
  __syncthreads();
  suffix_bytes = block_reduce(temp_storage.reduce_storage).Sum(suffix_bytes);
  if (t == 0) { avg_suffix_len = util::div_rounding_up_unsafe(suffix_bytes, non_zero); }
  __syncthreads();

  // Now copy the byte array data. For shorter suffixes (<= 64 bytes), it is faster to use
  // memcpy on a string-per-thread basis. For longer suffixes, it's better to use a parallel
  // approach. 64 was a good cutoff in testing.
  constexpr size_t suffix_cutoff = 64;

  size_t str_data_len = 0;
  if (avg_suffix_len <= suffix_cutoff) {
    for (int idx = 0; idx < s->page.num_valid; idx += block_size) {
      size_type const t_idx = idx + t;
      size_type s_len = 0, pref_len = 0, suff_len = 0;
      uint8_t const* s_ptr = nullptr;
      if (t_idx < s->page.num_valid) {
        size_type const leaf_idx = has_leaf_nulls ? offsets_map[t_idx] : t_idx;
        auto const byte_arr      = byte_array_at(leaf_idx + s->page_start_val);
        s_len                    = byte_arr.length;
        s_ptr                    = byte_arr.data;
        pref_len                 = prefix_lengths[t_idx];
        suff_len                 = byte_arr.length - pref_len;
      }

      // calculate offsets into output
      size_type s_off, total;
      block_scan(temp_storage.scan_storage)
        .ExclusiveScan(suff_len, s_off, str_data_len, cuda::std::plus(), total);

      if (t_idx < s->page.num_valid) {
        auto const dst = strings_ptr + s_off;
        memcpy(dst, s_ptr + pref_len, suff_len);
      }
      str_data_len += total;
      __syncthreads();
    }
  } else {
    int t0 = 0;  // thread 0 for each string
    for (int idx = 0; idx < s->page.num_valid; idx++) {
      // calculate ids for this string
      int const tid = (t - t0 + block_size) % block_size;

      // fetch string for this iter
      size_type const leaf_idx = has_leaf_nulls ? offsets_map[idx] : idx;
      auto const byte_arr      = byte_array_at(leaf_idx + s->page_start_val);
      size_type const pref_len = prefix_lengths[idx];
      size_type const suff_len = byte_arr.length - pref_len;

      // now copy the data
      auto const dst = strings_ptr + str_data_len;
      for (int src_idx = tid; src_idx < suff_len; src_idx += block_size) {
        dst[src_idx] = byte_arr.data[pref_len + src_idx];
      }

      str_data_len += suff_len;
      t0 = (t0 + suff_len) % block_size;
    }
  }

  finish_page_encode<block_size>(
    s, strings_ptr + str_data_len, pages, comp_in, comp_out, comp_results, true);
}

constexpr int decide_compression_warps_in_block = 4;
constexpr int decide_compression_block_size =
  decide_compression_warps_in_block * cudf::detail::warp_size;

// blockDim(decide_compression_block_size, 1, 1)
CUDF_KERNEL void __launch_bounds__(decide_compression_block_size)
  decide_compression_kernel(device_span<EncColumnChunk> chunks, bool page_level_compression)
{
  namespace cg = cooperative_groups;

  __shared__ __align__(8) EncColumnChunk ck_g[decide_compression_warps_in_block];
  using warp_reduce = cub::WarpReduce<uint32_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage[decide_compression_warps_in_block][2];

  auto const block    = cg::this_thread_block();
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto const lane_id  = warp.thread_rank();
  auto const warp_id  = warp.meta_group_rank();
  auto const chunk_id = block.group_index().x * decide_compression_warps_in_block + warp_id;

  if (chunk_id >= chunks.size()) { return; }

  if (lane_id == 0) { ck_g[warp_id] = chunks[chunk_id]; }
  warp.sync();

  // First pass: decide per-page compression for pages with PLC, collect totals for chunk decisions
  // Track PLC (page-level compression) and non-PLC contributions separately
  uint32_t uncompressed_data_size = 0;
  uint32_t plc_actual_size        = 0;
  uint32_t non_plc_uncomp_size    = 0;
  uint32_t non_plc_comp_size      = 0;
  uint32_t encodings              = 0;
  bool local_compression_error    = false;
  bool local_any_plc_compressed   = false;
  auto const num_pages            = ck_g[warp_id].num_pages;
  for (auto page_id = lane_id; page_id < num_pages; page_id += cudf::detail::warp_size) {
    auto& curr_page           = ck_g[warp_id].pages[page_id];
    auto const page_data_size = curr_page.data_size;
    uncompressed_data_size += page_data_size;

    if (auto const comp_res = curr_page.comp_res; comp_res != nullptr) {
      auto const lvl_bytes      = curr_page.is_v2() ? curr_page.level_bytes() : 0;
      auto const comp_page_size = comp_res->bytes_written + lvl_bytes;

      // For V2 pages, decide compression per-page only if page_level_compression is enabled
      if (curr_page.is_v2() and page_level_compression) {
        // Per-page decision: each V2 page decides independently
        auto const compressed_page_preferred =
          comp_res->status == codec_status::SUCCESS and comp_page_size < page_data_size;
        curr_page.is_compressed = compressed_page_preferred;
        plc_actual_size += compressed_page_preferred ? comp_page_size : page_data_size;
        if (compressed_page_preferred) { local_any_plc_compressed = true; }
      } else {
        // Chunk-level decision: V2 pages (when PLC disabled), dictionary, or V1 pages
        curr_page.is_compressed = false;  // Will be updated based on chunk decision
        non_plc_uncomp_size += page_data_size;
        non_plc_comp_size += comp_page_size;
        if (comp_res->status != codec_status::SUCCESS) { local_compression_error = true; }
      }
    } else {
      curr_page.is_compressed = false;
      if (curr_page.is_v2() and page_level_compression) {
        plc_actual_size += page_data_size;
      } else {
        non_plc_uncomp_size += page_data_size;
        non_plc_comp_size += page_data_size;
      }
    }

    // collect encoding info for the chunk metadata
    encodings |= encoding_to_mask(curr_page.encoding);
  }

  auto const has_compression_error = warp.any(local_compression_error);
  auto const any_plc_compressed    = warp.any(local_any_plc_compressed);

  uncompressed_data_size = warp_reduce(temp_storage[warp_id][0]).Sum(uncompressed_data_size);
  plc_actual_size        = warp_reduce(temp_storage[warp_id][1]).Sum(plc_actual_size);
  warp.sync();
  non_plc_uncomp_size = warp_reduce(temp_storage[warp_id][0]).Sum(non_plc_uncomp_size);
  non_plc_comp_size   = warp_reduce(temp_storage[warp_id][1]).Sum(non_plc_comp_size);
  warp.sync();
  encodings = warp_reduce(temp_storage[warp_id][0]).Reduce(encodings, BitwiseOr{});
  warp.sync();

  // Decide whether non-PLC pages should be compressed (only lane 0 has correct reduced values)
  auto non_plc_compressed = !has_compression_error and non_plc_comp_size < non_plc_uncomp_size;
  non_plc_compressed      = warp.shfl(non_plc_compressed, 0);

  // Chunk uses compression if any page is compressed (for codec in metadata)
  auto const write_compressed = any_plc_compressed or non_plc_compressed;

  if (lane_id == 0) {
    auto const non_plc_actual_size  = non_plc_compressed ? non_plc_comp_size : non_plc_uncomp_size;
    auto const compressed_data_size = plc_actual_size + non_plc_actual_size;

    chunks[chunk_id].is_compressed = write_compressed;
    chunks[chunk_id].bfr_size      = uncompressed_data_size;
    chunks[chunk_id].compressed_size =
      write_compressed ? compressed_data_size : uncompressed_data_size;

    // if there is repetition or definition level data add RLE encoding
    auto const rle_bits =
      ck_g[warp_id].col_desc->num_def_level_bits() + ck_g[warp_id].col_desc->num_rep_level_bits();
    if (rle_bits > 0) { encodings |= encoding_to_mask(Encoding::RLE); }
    chunks[chunk_id].encodings = encodings;
  }

  warp.sync();
  // Second pass: set is_compressed for pages that didn't decide in first pass.
  // Dictionary pages MUST be compressed if the chunk is compressed, because
  // dictionary page headers don't have an is_compressed field.
  // Non-PLC pages (V1 data pages, V2 with PLC disabled) follow the chunk-level decision.
  for (auto page_id = lane_id; page_id < num_pages; page_id += cudf::detail::warp_size) {
    auto& curr_page = ck_g[warp_id].pages[page_id];
    if (curr_page.is_v2() and page_level_compression) {
      // PLC pages already decided in first pass
      continue;
    }
    bool const is_dict_page = curr_page.page_type == PageType::DICTIONARY_PAGE;
    curr_page.is_compressed = is_dict_page ? write_compressed : non_plc_compressed;
  }
}

/**
 * Minimal thrift compact protocol support
 */
inline __device__ uint8_t* cpw_put_uint8(uint8_t* p, uint8_t v)
{
  *p++ = v;
  return p;
}

inline __device__ uint8_t* cpw_put_uint32(uint8_t* p, uint32_t v)
{
  while (v > 0x7f) {
    *p++ = v | 0x80;
    v >>= 7;
  }
  *p++ = v;
  return p;
}

inline __device__ uint8_t* cpw_put_uint64(uint8_t* p, uint64_t v)
{
  while (v > 0x7f) {
    *p++ = v | 0x80;
    v >>= 7;
  }
  *p++ = v;
  return p;
}

inline __device__ uint8_t* cpw_put_int32(uint8_t* p, int32_t v)
{
  int32_t s = (v < 0);
  return cpw_put_uint32(p, (v ^ -s) * 2 + s);
}

inline __device__ uint8_t* cpw_put_int64(uint8_t* p, int64_t v)
{
  int64_t s = (v < 0);
  return cpw_put_uint64(p, (v ^ -s) * 2 + s);
}

inline __device__ uint8_t* cpw_put_fldh(uint8_t* p, int f, int cur, FieldType t)
{
  auto const t_num = static_cast<uint8_t>(t);
  if (f > cur && f <= cur + 15) {
    *p++ = ((f - cur) << 4) | t_num;
    return p;
  } else {
    *p++ = t_num;
    return cpw_put_int32(p, f);
  }
}

class header_encoder {
  uint8_t* current_header_ptr;
  int current_field_index;

 public:
  inline __device__ header_encoder(uint8_t* header_start)
    : current_header_ptr(header_start), current_field_index(0)
  {
  }

  inline __device__ void field_struct_begin(int field)
  {
    current_header_ptr =
      cpw_put_fldh(current_header_ptr, field, current_field_index, FieldType::STRUCT);
    current_field_index = 0;
  }

  inline __device__ void field_struct_end(int field)
  {
    *current_header_ptr++ = 0;
    current_field_index   = field;
  }

  inline __device__ void field_list_begin(int field, size_t len, FieldType type)
  {
    current_header_ptr =
      cpw_put_fldh(current_header_ptr, field, current_field_index, FieldType::LIST);
    auto const t_num   = static_cast<uint8_t>(type);
    current_header_ptr = cpw_put_uint8(
      current_header_ptr, static_cast<uint8_t>((cuda::std::min(len, size_t{0xfu}) << 4) | t_num));
    if (len >= 0xf) { current_header_ptr = cpw_put_uint32(current_header_ptr, len); }
    current_field_index = 0;
  }

  inline __device__ void field_list_end(int field) { current_field_index = field; }

  inline __device__ void put_bool(bool value)
  {
    auto const type_byte =
      static_cast<uint8_t>(value ? FieldType::BOOLEAN_TRUE : FieldType::BOOLEAN_FALSE);
    current_header_ptr = cpw_put_uint8(current_header_ptr, type_byte);
  }

  inline __device__ void put_binary(void const* value, uint32_t length)
  {
    current_header_ptr = cpw_put_uint32(current_header_ptr, length);
    memcpy(current_header_ptr, value, length);
    current_header_ptr += length;
  }

  template <typename T>
  inline __device__ void put_int64(T value)
  {
    current_header_ptr = cpw_put_int64(current_header_ptr, static_cast<int64_t>(value));
  }

  inline __device__ void field_bool(int field, bool value)
  {
    current_header_ptr  = cpw_put_fldh(current_header_ptr,
                                      field,
                                      current_field_index,
                                      value ? FieldType::BOOLEAN_TRUE : FieldType::BOOLEAN_FALSE);
    current_field_index = field;
  }

  template <typename T>
  inline __device__ void field_int32(int field, T value)
  {
    current_header_ptr =
      cpw_put_fldh(current_header_ptr, field, current_field_index, FieldType::I32);
    current_header_ptr  = cpw_put_int32(current_header_ptr, static_cast<int32_t>(value));
    current_field_index = field;
  }

  template <typename T>
  inline __device__ void field_int64(int field, T value)
  {
    current_header_ptr =
      cpw_put_fldh(current_header_ptr, field, current_field_index, FieldType::I64);
    current_header_ptr  = cpw_put_int64(current_header_ptr, static_cast<int64_t>(value));
    current_field_index = field;
  }

  inline __device__ void field_binary(int field, void const* value, uint32_t length)
  {
    current_header_ptr =
      cpw_put_fldh(current_header_ptr, field, current_field_index, FieldType::BINARY);
    current_header_ptr = cpw_put_uint32(current_header_ptr, length);
    memcpy(current_header_ptr, value, length);
    current_header_ptr += length;
    current_field_index = field;
  }

  inline __device__ void end(uint8_t** header_end, bool termination_flag = true)
  {
    if (not termination_flag) { *current_header_ptr++ = 0; }
    *header_end = current_header_ptr;
  }

  inline __device__ uint8_t* get_ptr() { return current_header_ptr; }

  inline __device__ void set_ptr(uint8_t* ptr) { current_header_ptr = ptr; }
};

namespace {

// byteswap 128 bit integer, placing result in dst in network byte order.
// dst must point to at least 16 bytes of memory.
__device__ void byte_reverse128(__int128_t v, void* dst)
{
  auto const v_char_ptr = reinterpret_cast<unsigned char const*>(&v);
  auto const d_char_ptr = static_cast<unsigned char*>(dst);
  thrust::copy(thrust::seq,
               cuda::std::make_reverse_iterator(v_char_ptr + sizeof(v)),
               cuda::std::make_reverse_iterator(v_char_ptr),
               d_char_ptr);
}

/**
 * @brief Test to see if a span contains all valid UTF-8 characters.
 *
 * @param span device_span to test.
 * @return true if the span contains all valid UTF-8 characters.
 */
__device__ bool is_valid_utf8(device_span<unsigned char const> span)
{
  auto idx = 0;
  while (idx < span.size_bytes()) {
    // UTF-8 character should start with valid beginning bit pattern
    if (not strings::detail::is_valid_begin_utf8_char(span[idx])) { return false; }
    // subsequent elements of the character should be continuation chars
    auto const width = strings::detail::bytes_in_utf8_byte(span[idx++]);
    for (size_type i = 1; i < width && idx < span.size_bytes(); i++, idx++) {
      if (not strings::detail::is_utf8_continuation_char(span[idx])) { return false; }
    }
  }

  return true;
}

/**
 * @brief Increment part of a UTF-8 character.
 *
 * Attempt to increment the char pointed to by ptr, which is assumed to be part of a valid UTF-8
 * character. Returns true if successful, false if the increment caused an overflow, in which case
 * the data at ptr will be set to the lowest valid UTF-8 bit pattern (start or continuation).
 * Will halt execution if passed invalid UTF-8.
 */
__device__ bool increment_utf8_at(unsigned char* ptr)
{
  unsigned char elem = *ptr;
  // elem is one of (no 5 or 6 byte chars allowed):
  //  0b0vvvvvvv a 1 byte character
  //  0b10vvvvvv a continuation byte
  //  0b110vvvvv start of a 2 byte character
  //  0b1110vvvv start of a 3 byte character
  //  0b11110vvv start of a 4 byte character

  // TODO(ets): starting at 4 byte and working down.  Should probably start low and work higher.
  uint8_t mask  = 0xF8;
  uint8_t valid = 0xF0;

  while (mask != 0) {
    if ((elem & mask) == valid) {
      elem++;
      if ((elem & mask) != mask) {  // no overflow
        *ptr = elem;
        return true;
      }
      *ptr = valid;
      return false;
    }
    mask <<= 1;
    valid <<= 1;
  }

  // should not reach here since we test for valid UTF-8 higher up the call chain
  CUDF_UNREACHABLE("Trying to increment non-utf8");
}

/**
 * @brief Attempt to truncate a span of UTF-8 characters to at most truncate_length_bytes.
 *
 * If is_min is false, then the final character (or characters if there is overflow) will be
 * incremented so that the resultant UTF-8 will still be a valid maximum. scratch is only used when
 * is_min is false, and must be at least truncate_length bytes in size. If the span cannot be
 * truncated, leave it untouched and return the original length.
 *
 * @return Pair object containing a pointer to the truncated data and its length.
 */
__device__ cuda::std::pair<void const*, uint32_t> truncate_utf8(
  device_span<unsigned char const> span, bool is_min, void* scratch, int32_t truncate_length)
{
  // we know at this point that truncate_length < size_bytes, so
  // there is data at [len]. work backwards until we find
  // the start of a UTF-8 encoded character, since UTF-8 characters may be multi-byte.
  auto len = truncate_length;
  while (not strings::detail::is_begin_utf8_char(span[len]) && len > 0) {
    len--;
  }

  if (len != 0) {
    if (is_min) { return {span.data(), len}; }
    memcpy(scratch, span.data(), len);
    // increment last byte, working backwards if the byte overflows
    auto const ptr = static_cast<unsigned char*>(scratch);
    for (int32_t i = len - 1; i >= 0; i--) {
      if (increment_utf8_at(&ptr[i])) {  // true if no overflow
        return {scratch, len};
      }
    }
    // cannot increment, so fall through
  }

  // couldn't truncate, return original value
  return {span.data(), span.size_bytes()};
}

/**
 * @brief Attempt to truncate a span of binary data to at most truncate_length bytes.
 *
 * If is_min is false, then the final byte (or bytes if there is overflow) will be
 * incremented so that the resultant binary will still be a valid maximum. scratch is only used when
 * is_min is false, and must be at least truncate_length bytes in size. If the span cannot be
 * truncated, leave it untouched and return the original length.
 *
 * @return Pair object containing a pointer to the truncated data and its length.
 */
__device__ cuda::std::pair<void const*, uint32_t> truncate_binary(device_span<uint8_t const> arr,
                                                                  bool is_min,
                                                                  void* scratch,
                                                                  int32_t truncate_length)
{
  if (is_min) { return {arr.data(), truncate_length}; }
  memcpy(scratch, arr.data(), truncate_length);
  // increment last byte, working backwards if the byte overflows
  auto const ptr = static_cast<uint8_t*>(scratch);
  for (int32_t i = truncate_length - 1; i >= 0; i--) {
    ptr[i]++;
    if (ptr[i] != 0) {  // no overflow
      return {scratch, i + 1};
    }
  }

  // couldn't truncate, return original value
  return {arr.data(), arr.size_bytes()};
}

// TODO (ets): the assumption here is that string columns might have UTF-8 or plain binary,
// while binary columns are assumed to be binary and will be treated as such.  If this assumption
// is incorrect, then truncate_byte_array() and truncate_string() should just be combined into
// a single function.
/**
 * @brief Attempt to truncate a UTF-8 string to at most truncate_length bytes.
 */
__device__ cuda::std::pair<void const*, uint32_t> truncate_string(string_view const& str,
                                                                  bool is_min,
                                                                  void* scratch,
                                                                  int32_t truncate_length)
{
  if (truncate_length == NO_TRUNC_STATS or str.size_bytes() <= truncate_length) {
    return {str.data(), str.size_bytes()};
  }

  // convert char to unsigned since UTF-8 is just bytes, not chars.  can't use std::byte because
  // that can't be incremented.
  auto const span = device_span<unsigned char const>(
    reinterpret_cast<unsigned char const*>(str.data()), str.size_bytes());

  // if str is all 8-bit chars, or is actually not UTF-8, then we can just use truncate_binary()
  if (str.size_bytes() != str.length() and is_valid_utf8(span.first(truncate_length))) {
    return truncate_utf8(span, is_min, scratch, truncate_length);
  }
  return truncate_binary(span, is_min, scratch, truncate_length);
}

/**
 * @brief Attempt to truncate a binary array to at most truncate_length bytes.
 */
__device__ cuda::std::pair<void const*, uint32_t> truncate_byte_array(
  statistics::byte_array_view const& arr, bool is_min, void* scratch, int32_t truncate_length)
{
  if (truncate_length == NO_TRUNC_STATS or arr.size_bytes() <= truncate_length) {
    return {arr.data(), arr.size_bytes()};
  }

  // convert std::byte to uint8_t since bytes can't be incremented
  device_span<uint8_t const> const span{reinterpret_cast<uint8_t const*>(arr.data()),
                                        arr.size_bytes()};
  return truncate_binary(span, is_min, scratch, truncate_length);
}

/**
 * @brief Find a min or max value of the proper form to be included in Parquet statistics
 * structures.
 *
 * Given a statistics_val union and a data type, perform any transformations needed to produce a
 * valid min or max binary value.  String and byte array types will be truncated if they exceed
 * truncate_length.
 */
__device__ cuda::std::pair<void const*, uint32_t> get_extremum(statistics_val const* stats_val,
                                                               statistics_dtype dtype,
                                                               void* scratch,
                                                               bool is_min,
                                                               int32_t truncate_length)
{
  switch (dtype) {
    case dtype_bool: return {stats_val, sizeof(bool)};
    case dtype_int8:
    case dtype_int16:
    case dtype_int32:
    case dtype_date32: return {stats_val, sizeof(int32_t)};
    case dtype_float32: {
      auto const fp_scratch = static_cast<float*>(scratch);
      fp_scratch[0]         = stats_val->fp_val;
      return {scratch, sizeof(float)};
    }
    case dtype_int64:
    case dtype_decimal64:
    case dtype_timestamp64:
    case dtype_float64: return {stats_val, sizeof(int64_t)};
    case dtype_decimal128:
      byte_reverse128(stats_val->d128_val, scratch);
      return {scratch, sizeof(__int128_t)};
    case dtype_string: return truncate_string(stats_val->str_val, is_min, scratch, truncate_length);
    case dtype_byte_array:
      return truncate_byte_array(stats_val->byte_val, is_min, scratch, truncate_length);
    default: CUDF_UNREACHABLE("Invalid statistics data type");
  }
}

}  // namespace

__device__ uint8_t* EncodeStatistics(uint8_t* start,
                                     statistics_chunk const* s,
                                     statistics_dtype dtype,
                                     void* scratch)
{
  uint8_t* end;
  header_encoder encoder(start);
  encoder.field_int64(3, s->null_count);
  if (s->has_minmax) {
    auto const [max_ptr, max_size] =
      get_extremum(&s->max_value, dtype, scratch, false, NO_TRUNC_STATS);
    encoder.field_binary(5, max_ptr, max_size);
    auto const [min_ptr, min_size] =
      get_extremum(&s->min_value, dtype, scratch, true, NO_TRUNC_STATS);
    encoder.field_binary(6, min_ptr, min_size);
    // cudf min/max statistics are always exact (i.e. not truncated)
    encoder.field_bool(7, true);
    encoder.field_bool(8, true);
  }
  encoder.end(&end);
  return end;
}

// blockDim(encode_block_size, 1, 1)
CUDF_KERNEL void __launch_bounds__(encode_block_size)
  gpuEncodePageHeaders(device_span<EncPage> pages,
                       device_span<codec_exec_result const> comp_results,
                       device_span<statistics_chunk const> page_stats,
                       statistics_chunk const* chunk_stats)
{
  __align__(8) unsigned char scratch[MIN_STATS_SCRATCH_SIZE];

  auto const page_idx = cudf::detail::grid_1d::global_thread_id();
  if (page_idx >= pages.size()) { return; }

  auto page             = pages[page_idx];
  auto const chunk      = *page.chunk;
  auto const stats_type = chunk.col_desc->stats_dtype;

  uint8_t* hdr_start{};
  uint8_t* hdr_end{};
  if (chunk_stats && &pages[page_idx] == chunk.pages) {  // Is this the first page in a chunk?
    hdr_start = (chunk.is_compressed) ? chunk.compressed_bfr : chunk.uncompressed_bfr;
    hdr_end   = EncodeStatistics(hdr_start, &chunk_stats[page.chunk_id], stats_type, scratch);
    page.chunk->ck_stat_size = static_cast<uint32_t>(hdr_end - hdr_start);
  }
  // For V2 pages, compressed_page_size and uncompressed_page_size include level bytes.
  // Level bytes are stored uncompressed before the (possibly compressed) value data.
  auto const lvl_bytes = page.is_v2() ? page.level_bytes() : 0;
  uint32_t compressed_page_size{};
  uint32_t uncompressed_page_size{};
  if (page.is_compressed) {
    hdr_start            = page.compressed_data;
    compressed_page_size = lvl_bytes + comp_results[page_idx].bytes_written;
    // comp_data_size is total data size (including levels)
    page.comp_data_size    = compressed_page_size;
    uncompressed_page_size = page.data_size;
  } else {
    hdr_start = page.page_data;
    // Both sizes are the same when uncompressed
    compressed_page_size   = page.data_size;
    uncompressed_page_size = page.data_size;
  }
  header_encoder encoder(hdr_start);

  encoder.field_int32(1, page.page_type);
  encoder.field_int32(2, uncompressed_page_size);
  encoder.field_int32(3, compressed_page_size);

  if (page.page_type == PageType::DATA_PAGE) {
    // DataPageHeader
    encoder.field_struct_begin(5);
    encoder.field_int32(1, page.num_values);  // NOTE: num_values != num_rows for list types
    encoder.field_int32(2, page.encoding);    // encoding
    encoder.field_int32(3, Encoding::RLE);    // definition_level_encoding
    encoder.field_int32(4, Encoding::RLE);    // repetition_level_encoding
    // Optionally encode page-level statistics
    if (not page_stats.empty()) {
      encoder.field_struct_begin(5);
      encoder.set_ptr(
        EncodeStatistics(encoder.get_ptr(), &page_stats[page_idx], stats_type, scratch));
      encoder.field_struct_end(5);
    }
    encoder.field_struct_end(5);
  } else if (page.page_type == PageType::DATA_PAGE_V2) {
    // DataPageHeaderV2
    encoder.field_struct_begin(8);
    encoder.field_int32(1, page.num_values);
    encoder.field_int32(2, page.num_nulls);
    encoder.field_int32(3, page.num_rows);
    encoder.field_int32(4, page.encoding);
    encoder.field_int32(5, page.def_lvl_bytes);
    encoder.field_int32(6, page.rep_lvl_bytes);
    encoder.field_bool(7, page.is_compressed);
    // Optionally encode page-level statistics
    if (not page_stats.empty()) {
      encoder.field_struct_begin(8);
      encoder.set_ptr(
        EncodeStatistics(encoder.get_ptr(), &page_stats[page_idx], stats_type, scratch));
      encoder.field_struct_end(8);
    }
    encoder.field_struct_end(8);
  } else {
    // DictionaryPageHeader
    encoder.field_struct_begin(7);
    encoder.field_int32(1, chunk.num_dict_entries);  // number of values in dictionary
    encoder.field_int32(2, page.encoding);
    encoder.field_struct_end(7);
  }
  encoder.end(&hdr_end, false);
  page.hdr_size = static_cast<uint32_t>(hdr_end - hdr_start);

  pages[page_idx] = page;
}

// blockDim(1024, 1, 1)
CUDF_KERNEL void __launch_bounds__(1024) gpuGatherPages(device_span<EncColumnChunk> chunks)
{
  namespace cg = cooperative_groups;

  __shared__ __align__(8) EncColumnChunk ck_g;
  __shared__ __align__(8) EncPage page_g;

  auto const t = threadIdx.x;
  uint8_t *dst, *dst_base;
  EncPage const* first_page;
  uint32_t num_pages, uncompressed_size;

  if (t == 0) ck_g = chunks[blockIdx.x];
  __syncthreads();

  first_page = ck_g.pages;
  num_pages  = ck_g.num_pages;
  dst        = (ck_g.is_compressed) ? ck_g.compressed_bfr : ck_g.uncompressed_bfr;
  dst += ck_g.ck_stat_size;  // Skip over chunk statistics
  dst_base          = dst;
  uncompressed_size = ck_g.bfr_size;
  for (uint32_t page = 0; page < num_pages; page++) {
    uint8_t const* src;
    uint32_t hdr_len, data_len;

    if (t == 0) { page_g = first_page[page]; }
    __syncthreads();

    src = page_g.is_compressed ? page_g.compressed_data : page_g.page_data;
    // Copy page header
    hdr_len = page_g.hdr_size;
    memcpy_block<1024, true>(dst, src, hdr_len, cg::this_thread_block());
    src += page_g.max_hdr_size;
    dst += hdr_len;
    uncompressed_size += hdr_len;
    data_len = page_g.is_compressed ? page_g.comp_data_size : page_g.data_size;
    // Copy page data. For V2, the level data and page data are disjoint.
    if (page_g.is_v2()) {
      auto const lvl_len = page_g.level_bytes();
      // For V2 pages, level data is ALWAYS in the uncompressed buffer (levels are never compressed)
      auto const* lvl_src = page_g.page_data + page_g.max_hdr_size;
      memcpy_block<1024, true>(dst, lvl_src, lvl_len, cg::this_thread_block());
      src += page_g.max_lvl_size;
      dst += lvl_len;
      data_len -= lvl_len;
    }
    memcpy_block<1024, true>(dst, src, data_len, cg::this_thread_block());
    dst += data_len;
    __syncthreads();
    if (t == 0 && page == 0 && ck_g.use_dictionary) { ck_g.dictionary_size = hdr_len + data_len; }
  }
  if (t == 0) {
    chunks[blockIdx.x].bfr_size        = uncompressed_size;
    chunks[blockIdx.x].compressed_size = (dst - dst_base);
    if (ck_g.use_dictionary) { chunks[blockIdx.x].dictionary_size = ck_g.dictionary_size; }
  }
}

namespace {

/**
 * @brief Tests if statistics are comparable given the column's
 * physical and converted types
 */
__device__ bool is_comparable(Type ptype, ConvertedType ctype)
{
  switch (ptype) {
    case Type::BOOLEAN:
    case Type::INT32:
    case Type::INT64:
    case Type::FLOAT:
    case Type::DOUBLE:
    case Type::BYTE_ARRAY: return true;
    case Type::FIXED_LEN_BYTE_ARRAY:
      if (ctype == ConvertedType::DECIMAL) { return true; }
      [[fallthrough]];
    default: return false;
  }
}

/**
 * @brief Compares two values.
 * @return -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
 */
template <typename T>
constexpr __device__ int32_t compare(T& v1, T& v2)
{
  return (v1 > v2) - (v1 < v2);
}

/**
 * @brief Compares two statistics_val structs.
 * @return < 0 if v1 < v2, 0 if v1 == v2, > 0 if v1 > v2
 */
__device__ int32_t compare_values(Type ptype,
                                  ConvertedType ctype,
                                  statistics_val const& v1,
                                  statistics_val const& v2)
{
  switch (ptype) {
    case Type::BOOLEAN: return compare(v1.u_val, v2.u_val);
    case Type::INT32:
    case Type::INT64:
      switch (ctype) {
        case ConvertedType::UINT_8:
        case ConvertedType::UINT_16:
        case ConvertedType::UINT_32:
        case ConvertedType::UINT_64: return compare(v1.u_val, v2.u_val);
        default:  // assume everything else is signed
          return compare(v1.i_val, v2.i_val);
      }
    case Type::FLOAT:
    case Type::DOUBLE: return compare(v1.fp_val, v2.fp_val);
    case Type::BYTE_ARRAY: return static_cast<string_view>(v1.str_val).compare(v2.str_val);
    case Type::FIXED_LEN_BYTE_ARRAY:
      if (ctype == ConvertedType::DECIMAL) { return compare(v1.d128_val, v2.d128_val); }
  }
  // calling is_comparable() should prevent reaching here
  CUDF_UNREACHABLE("Trying to compare non-comparable type");
  return 0;
}

/**
 * @brief Determine if a set of statstistics are in ascending order.
 */
__device__ bool is_ascending(statistics_chunk const* s,
                             Type ptype,
                             ConvertedType ctype,
                             uint32_t num_pages)
{
  for (uint32_t i = 1; i < num_pages; i++) {
    if (compare_values(ptype, ctype, s[i - 1].min_value, s[i].min_value) > 0 ||
        compare_values(ptype, ctype, s[i - 1].max_value, s[i].max_value) > 0) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Determine if a set of statstistics are in descending order.
 */
__device__ bool is_descending(statistics_chunk const* s,
                              Type ptype,
                              ConvertedType ctype,
                              uint32_t num_pages)
{
  for (uint32_t i = 1; i < num_pages; i++) {
    if (compare_values(ptype, ctype, s[i - 1].min_value, s[i].min_value) < 0 ||
        compare_values(ptype, ctype, s[i - 1].max_value, s[i].max_value) < 0) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Determine the ordering of a set of statistics.
 */
__device__ BoundaryOrder calculate_boundary_order(statistics_chunk const* s,
                                                  Type ptype,
                                                  ConvertedType ctype,
                                                  uint32_t num_pages)
{
  if (not is_comparable(ptype, ctype)) { return BoundaryOrder::UNORDERED; }
  if (is_ascending(s, ptype, ctype, num_pages)) {
    return BoundaryOrder::ASCENDING;
  } else if (is_descending(s, ptype, ctype, num_pages)) {
    return BoundaryOrder::DESCENDING;
  }
  return BoundaryOrder::UNORDERED;
}

// align ptr to an 8-byte boundary. address returned will be <= ptr.
inline __device__ void* align8(void* ptr)
{
  // it's ok to round down because we have an extra 7 bytes in the buffer
  auto algn = 3 & reinterpret_cast<std::uintptr_t>(ptr);
  return static_cast<char*>(ptr) - algn;
}

struct mask_tform {
  __device__ uint32_t operator()(EncPage const& p) { return static_cast<uint32_t>(p.kernel_mask); }
};

}  // namespace

// blockDim(1, 1, 1)
CUDF_KERNEL void __launch_bounds__(1)
  gpuEncodeColumnIndexes(device_span<EncColumnChunk> chunks,
                         device_span<statistics_chunk const> column_stats,
                         int32_t column_index_truncate_length)
{
  __align__(8) unsigned char s_scratch[MIN_STATS_SCRATCH_SIZE];
  uint8_t* col_idx_end;

  if (column_stats.empty()) { return; }

  auto const ck_g                = &chunks[blockIdx.x];
  uint32_t const num_pages       = ck_g->num_pages;
  auto const& col_g              = *ck_g->col_desc;
  uint32_t const first_data_page = ck_g->use_dictionary ? 1 : 0;
  uint32_t const num_data_pages  = num_pages - first_data_page;
  uint32_t const pageidx         = ck_g->first_page;
  size_t var_bytes               = 0;

  header_encoder encoder(ck_g->column_index_blob);

  // make sure scratch is aligned properly. here column_index_size indicates
  // how much scratch space is available for this chunk, including space for
  // truncation scratch + padding for alignment.
  void* scratch =
    column_index_truncate_length < MIN_STATS_SCRATCH_SIZE
      ? s_scratch
      : align8(ck_g->column_index_blob + ck_g->column_index_size - column_index_truncate_length);

  // null_pages
  encoder.field_list_begin(1, num_data_pages, FieldType::BOOLEAN_TRUE);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    encoder.put_bool(column_stats[pageidx + page].non_nulls == 0);
  }
  encoder.field_list_end(1);
  // min_values
  encoder.field_list_begin(2, num_data_pages, FieldType::BINARY);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    auto const [min_ptr, min_size] = get_extremum(&column_stats[pageidx + page].min_value,
                                                  col_g.stats_dtype,
                                                  scratch,
                                                  true,
                                                  column_index_truncate_length);
    encoder.put_binary(min_ptr, min_size);
  }
  encoder.field_list_end(2);
  // max_values
  encoder.field_list_begin(3, num_data_pages, FieldType::BINARY);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    auto const [max_ptr, max_size] = get_extremum(&column_stats[pageidx + page].max_value,
                                                  col_g.stats_dtype,
                                                  scratch,
                                                  false,
                                                  column_index_truncate_length);
    encoder.put_binary(max_ptr, max_size);
  }
  encoder.field_list_end(3);
  // boundary_order
  encoder.field_int32(
    4,
    static_cast<int32_t>(calculate_boundary_order(&column_stats[first_data_page + pageidx],
                                                  col_g.physical_type,
                                                  col_g.converted_type,
                                                  num_pages - first_data_page)));
  // null_counts
  encoder.field_list_begin(5, num_data_pages, FieldType::I64);
  for (uint32_t page = first_data_page; page < num_pages; page++) {
    encoder.put_int64(column_stats[pageidx + page].null_count);
  }
  encoder.field_list_end(5);

  // find pointers to chunk histograms
  auto const cd          = ck_g->col_desc;
  auto const ck_def_hist = ck_g->def_histogram_data + (num_data_pages) * (cd->max_def_level + 1);
  auto const ck_rep_hist = ck_g->rep_histogram_data + (num_data_pages) * (cd->max_rep_level + 1);

  auto const page_start = ck_g->pages + first_data_page;
  auto const page_end   = ck_g->pages + ck_g->num_pages;

  // optionally encode histograms and sum var_bytes.
  if (cd->max_rep_level > REP_LVL_HIST_CUTOFF) {
    encoder.field_list_begin(6, num_data_pages * (cd->max_rep_level + 1), FieldType::I64);
    thrust::for_each(thrust::seq, page_start, page_end, [&] __device__(auto const& page) {
      for (int i = 0; i < cd->max_rep_level + 1; i++) {
        encoder.put_int64(page.rep_histogram[i]);
        ck_rep_hist[i] += page.rep_histogram[i];
      }
    });
    encoder.field_list_end(6);
  }

  if (cd->max_def_level > DEF_LVL_HIST_CUTOFF) {
    encoder.field_list_begin(7, num_data_pages * (cd->max_def_level + 1), FieldType::I64);
    thrust::for_each(thrust::seq, page_start, page_end, [&] __device__(auto const& page) {
      for (int i = 0; i < cd->max_def_level + 1; i++) {
        encoder.put_int64(page.def_histogram[i]);
        ck_def_hist[i] += page.def_histogram[i];
      }
    });
    encoder.field_list_end(7);
  }

  if (col_g.physical_type == Type::BYTE_ARRAY) {
    thrust::for_each(thrust::seq, page_start, page_end, [&] __device__(auto const& page) {
      var_bytes += page.var_bytes_size;
    });
  }

  encoder.end(&col_idx_end, false);

  // now reset column_index_size to the actual size of the encoded column index blob
  ck_g->column_index_size = static_cast<uint32_t>(col_idx_end - ck_g->column_index_blob);
  ck_g->var_bytes_size    = var_bytes;
}

void InitRowGroupFragments(device_2dspan<PageFragment> frag,
                           device_span<parquet_column_device_view const> col_desc,
                           device_span<partition_info const> partitions,
                           device_span<int const> part_frag_offset,
                           uint32_t fragment_size,
                           rmm::cuda_stream_view stream)
{
  auto const num_columns              = frag.size().first;
  auto const num_fragments_per_column = frag.size().second;
  auto const grid_y = std::min(static_cast<uint32_t>(num_fragments_per_column), MAX_GRID_Y_SIZE);
  dim3 const dim_grid(num_columns, grid_y);  // 1 threadblock per fragment
  gpuInitRowGroupFragments<512><<<dim_grid, 512, 0, stream.value()>>>(
    frag, col_desc, partitions, part_frag_offset, fragment_size);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void CalculatePageFragments(device_span<PageFragment> frag,
                            device_span<size_type const> column_frag_sizes,
                            rmm::cuda_stream_view stream)
{
  gpuCalculatePageFragments<512><<<frag.size(), 512, 0, stream.value()>>>(frag, column_frag_sizes);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void InitFragmentStatistics(device_span<statistics_group> groups,
                            device_span<PageFragment const> fragments,
                            rmm::cuda_stream_view stream)
{
  int const num_fragments = fragments.size();
  int const dim =
    util::div_rounding_up_safe(num_fragments, encode_block_size / cudf::detail::warp_size);
  gpuInitFragmentStats<<<dim, encode_block_size, 0, stream.value()>>>(groups, fragments);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void InitEncoderPages(device_2dspan<EncColumnChunk> chunks,
                      device_span<EncPage> pages,
                      device_span<size_type> page_sizes,
                      device_span<size_type const> comp_page_sizes,
                      device_span<parquet_column_device_view const> col_desc,
                      int32_t num_columns,
                      size_t max_page_size_bytes,
                      size_type max_page_size_rows,
                      uint32_t page_align,
                      bool write_v2_headers,
                      statistics_merge_group* page_grstats,
                      statistics_merge_group* chunk_grstats,
                      rmm::cuda_stream_view stream)
{
  auto num_rowgroups = chunks.size().first;
  dim3 dim_grid(num_columns, num_rowgroups);  // 1 threadblock per rowgroup
  gpuInitPages<<<dim_grid, encode_block_size, 0, stream.value()>>>(chunks,
                                                                   pages,
                                                                   page_sizes,
                                                                   comp_page_sizes,
                                                                   col_desc,
                                                                   page_grstats,
                                                                   chunk_grstats,
                                                                   num_columns,
                                                                   max_page_size_bytes,
                                                                   max_page_size_rows,
                                                                   page_align,
                                                                   write_v2_headers);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void EncodePages(device_span<EncPage> pages,
                 bool write_v2_headers,
                 device_span<device_span<uint8_t const>> comp_in,
                 device_span<device_span<uint8_t>> comp_out,
                 device_span<codec_exec_result> comp_results,
                 rmm::cuda_stream_view stream)
{
  auto num_pages = pages.size();

  // determine which kernels to invoke
  auto kernel_mask = cudf::detail::transform_reduce(
    pages.begin(), pages.end(), mask_tform{}, uint32_t{0}, cuda::std::bit_or<uint32_t>{}, stream);

  // get the number of streams we need from the pool
  int nkernels = std::bitset<32>(kernel_mask).count();
  auto streams = cudf::detail::fork_streams(stream, nkernels);

  // A page is part of one column. This is launching 1 block per page. 1 block will exclusively
  // deal with one datatype.

  int s_idx = 0;
  if (BitAnd(kernel_mask, encode_kernel_mask::PLAIN) != 0) {
    auto const strm = streams[s_idx++];
    gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, write_v2_headers, encode_kernel_mask::PLAIN);
    CUDF_CUDA_TRY(cudaGetLastError());
    gpuEncodePages<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, comp_in, comp_out, comp_results, write_v2_headers, false);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, encode_kernel_mask::BYTE_STREAM_SPLIT) != 0) {
    auto const strm = streams[s_idx++];
    gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, write_v2_headers, encode_kernel_mask::BYTE_STREAM_SPLIT);
    CUDF_CUDA_TRY(cudaGetLastError());
    gpuEncodePages<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, comp_in, comp_out, comp_results, write_v2_headers, true);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, encode_kernel_mask::DELTA_BINARY) != 0) {
    auto const strm = streams[s_idx++];
    gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, write_v2_headers, encode_kernel_mask::DELTA_BINARY);
    CUDF_CUDA_TRY(cudaGetLastError());
    gpuEncodeDeltaBinaryPages<encode_block_size>
      <<<num_pages, encode_block_size, 0, strm.value()>>>(pages, comp_in, comp_out, comp_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, encode_kernel_mask::DELTA_LENGTH_BA) != 0) {
    auto const strm = streams[s_idx++];
    gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, write_v2_headers, encode_kernel_mask::DELTA_LENGTH_BA);
    CUDF_CUDA_TRY(cudaGetLastError());
    gpuEncodeDeltaLengthByteArrayPages<encode_block_size>
      <<<num_pages, encode_block_size, 0, strm.value()>>>(pages, comp_in, comp_out, comp_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, encode_kernel_mask::DELTA_BYTE_ARRAY) != 0) {
    auto const strm = streams[s_idx++];
    gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, write_v2_headers, encode_kernel_mask::DELTA_BYTE_ARRAY);
    CUDF_CUDA_TRY(cudaGetLastError());
    gpuEncodeDeltaByteArrayPages<encode_block_size>
      <<<num_pages, encode_block_size, 0, strm.value()>>>(pages, comp_in, comp_out, comp_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, encode_kernel_mask::DICTIONARY) != 0) {
    auto const strm = streams[s_idx++];
    gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, write_v2_headers, encode_kernel_mask::DICTIONARY);
    CUDF_CUDA_TRY(cudaGetLastError());
    gpuEncodeDictPages<encode_block_size><<<num_pages, encode_block_size, 0, strm.value()>>>(
      pages, comp_in, comp_out, comp_results, write_v2_headers);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  cudf::detail::join_streams(streams, stream);
}

void decide_compression(device_span<EncColumnChunk> chunks,
                        bool page_level_compression,
                        rmm::cuda_stream_view stream)
{
  auto const num_blocks =
    util::div_rounding_up_safe<int>(chunks.size(), decide_compression_warps_in_block);
  decide_compression_kernel<<<num_blocks, decide_compression_block_size, 0, stream.value()>>>(
    chunks, page_level_compression);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void EncodePageHeaders(device_span<EncPage> pages,
                       device_span<codec_exec_result const> comp_results,
                       device_span<statistics_chunk const> page_stats,
                       statistics_chunk const* chunk_stats,
                       rmm::cuda_stream_view stream)
{
  auto const num_blocks = util::div_rounding_up_safe<int>(pages.size(), encode_block_size);
  gpuEncodePageHeaders<<<num_blocks, encode_block_size, 0, stream.value()>>>(
    pages, comp_results, page_stats, chunk_stats);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void GatherPages(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream)
{
  gpuGatherPages<<<chunks.size(), 1024, 0, stream.value()>>>(chunks);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void EncodeColumnIndexes(device_span<EncColumnChunk> chunks,
                         device_span<statistics_chunk const> column_stats,
                         int32_t column_index_truncate_length,
                         rmm::cuda_stream_view stream)
{
  gpuEncodeColumnIndexes<<<chunks.size(), 1, 0, stream.value()>>>(
    chunks, column_stats, column_index_truncate_length);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::parquet::detail
