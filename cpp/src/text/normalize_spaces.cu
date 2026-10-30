/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <nvtext/normalize.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/bit>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {
namespace cg = cooperative_groups;

constexpr int64_t block_size = 256;

/// Number of input bytes processed by each thread when normalizing spaces
constexpr int64_t bytes_per_thread = 16;
/// Number of input bytes processed by each warp when normalizing spaces
constexpr int64_t bytes_per_warp = bytes_per_thread * cudf::detail::warp_size;
/// Number of 32-bit bitmask words covering the input bytes of each warp
constexpr int64_t words_per_warp = bytes_per_warp / cudf::detail::warp_size;

__device__ bool is_whitespace(char ch) { return static_cast<uint8_t>(ch) <= ' '; }

/**
 * @brief Input bytes processed by each thread
 */
struct alignas(bytes_per_thread) spaces_chunk {
  char data[bytes_per_thread];
};

/**
 * @brief Input characters for normalizing spaces
 *
 * Byte positions are relative to `d_base` which is the input aligned down
 * to `bytes_per_thread` so each thread can use a single vector load.
 * Positions outside of [begin, end) are loaded as whitespace.
 */
struct spaces_input {
  char const* d_base;  ///< input characters aligned down to bytes_per_thread
  int64_t begin;       ///< position of the first input byte
  int64_t end;         ///< position just past the last input byte

  __device__ spaces_chunk load_chunk(int64_t pos) const
  {
    if (pos >= begin && pos + bytes_per_thread <= end) {
      return *reinterpret_cast<spaces_chunk const*>(d_base + pos);
    }
    spaces_chunk chunk;
    for (auto k = 0; k < bytes_per_thread; ++k) {
      auto const p  = pos + k;
      chunk.data[k] = (p >= begin && p < end) ? d_base[p] : ' ';
    }
    return chunk;
  }
};

/**
 * @brief Marks the whitespace bytes in each row that are not replaced with a space
 *
 * Repeated whitespace (code-point <= ' ') is replaced with a single space and
 * whitespace is trimmed from the beginning and end of each row.
 * The output space for each run of whitespace comes from the run's first byte
 * when it follows a non-whitespace byte. The exceptions are a run that begins
 * a row (the previous byte belongs to the previous row) and a run that ends a row.
 * These are marked here so the remaining checks only need the adjacent byte.
 *
 * Launched as a thread per row.
 *
 * @param input Input characters
 * @param offsets Input row offsets
 * @param first_offset Offset of the first row; only non-zero if the input has been sliced
 * @param num_rows Number of input rows
 * @param d_marks Bitmask of byte positions to mark; must be zero-initialized
 */
CUDF_KERNEL void mark_row_spaces_kernel(spaces_input input,
                                        cudf::detail::input_offsetalator offsets,
                                        int64_t first_offset,
                                        cudf::size_type num_rows,
                                        uint32_t* d_marks)
{
  auto const row = cudf::detail::grid_1d::global_thread_id();
  if (row >= num_rows) { return; }
  auto const begin = offsets[row] - first_offset + input.begin;
  auto const end   = offsets[row + 1] - first_offset + input.begin;
  if (begin == end) { return; }

  auto const d_chars = input.d_base;
  auto set_mark      = [d_marks](int64_t pos) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ref{
      d_marks[pos / cudf::detail::warp_size]};
    ref.fetch_or(1u << (pos % cudf::detail::warp_size), cuda::std::memory_order_relaxed);
  };
  if (begin > input.begin && is_whitespace(d_chars[begin]) && !is_whitespace(d_chars[begin - 1])) {
    set_mark(begin);
  }
  auto last = end;
  while (last > begin && is_whitespace(d_chars[last - 1])) {
    --last;
  }
  if (last > begin && last < end) { set_mark(last); }
}

/**
 * @brief Identifies the input bytes that are kept in the output
 *
 * A byte is kept if it is not whitespace or if it is the first byte of
 * a run of whitespace that is not marked by mark_row_spaces_kernel.
 *
 * Launched as a thread per `bytes_per_thread` input bytes.
 *
 * @param input Input characters
 * @param d_bits On input, the marks from mark_row_spaces_kernel;
 *               on output, a bit is set for each kept byte
 * @param d_warp_counts Number of kept bytes for each warp stored at warp index + 1;
 *                      the first element is set to 0
 */
CUDF_KERNEL void keep_spaces_kernel(spaces_input input, uint32_t* d_bits, int64_t* d_warp_counts)
{
  auto const tid  = cudf::detail::grid_1d::global_thread_id();
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const pos  = tid * bytes_per_thread;
  auto const word = pos / cudf::detail::warp_size;

  auto const chunk = input.load_chunk(pos);

  // whitespace state of the byte just before this thread's bytes
  bool prev_ws = warp.shfl_up(is_whitespace(chunk.data[bytes_per_thread - 1]), 1);
  if (warp.thread_rank() == 0) {
    prev_ws = (pos > input.begin && pos <= input.end) ? is_whitespace(input.d_base[pos - 1]) : true;
  }

  auto const marks = d_bits[word] >> (pos % cudf::detail::warp_size);
  uint32_t bits    = 0;
  for (auto k = 0; k < bytes_per_thread; ++k) {
    auto const ws = is_whitespace(chunk.data[k]);
    auto const p  = pos + k;
    auto const keep =
      (p >= input.begin) && (p < input.end) && (!ws || (!prev_ws && ((marks >> k) & 1) == 0));
    bits |= static_cast<uint32_t>(keep) << k;
    prev_ws = ws;
  }

  // each word is read by two threads and then replaced by the first one
  auto const upper = warp.shfl_down(bits, 1);
  if (warp.thread_rank() % 2 == 0) { d_bits[word] = bits | (upper << bytes_per_thread); }

  auto const count = cg::reduce(warp, cuda::std::popcount(bits), cg::plus<int32_t>());
  if (warp.thread_rank() == 0) { d_warp_counts[tid / cudf::detail::warp_size + 1] = count; }
  if (tid == 0) { d_warp_counts[0] = 0; }
}

/**
 * @brief Computes the output position of an input byte
 *
 * The output position is the number of kept bytes before the input byte.
 */
struct output_position_fn {
  uint32_t const* d_bits;         ///< bit set for each kept byte
  int64_t const* d_warp_offsets;  ///< number of kept bytes before each warp's input bytes
  int64_t base;                   ///< converts an input offset to a byte position

  /**
   * @brief Returns the number of kept bytes before the input offset
   *
   * @param offset Input offset
   */
  __device__ int64_t operator()(int64_t offset) const
  {
    auto const pos  = offset + base;
    auto const warp = pos / bytes_per_warp;
    auto const word = pos / cudf::detail::warp_size;
    auto const bit  = pos % cudf::detail::warp_size;
    auto result     = d_warp_offsets[warp];
    for (auto w = warp * words_per_warp; w < word; ++w) {
      result += cuda::std::popcount(d_bits[w]);
    }
    if (bit > 0) { result += cuda::std::popcount(d_bits[word] & ((1u << bit) - 1)); }
    return result;
  }
};

/**
 * @brief Writes the kept bytes to the output with whitespace replaced by a space
 *
 * Launched as a thread per `bytes_per_thread` input bytes.
 *
 * The output for each warp is assembled in shared memory and
 * then written to `d_output` at the warp's offset.
 *
 * @param input Input characters
 * @param d_bits Bit set for each kept byte
 * @param d_warp_offsets Number of kept bytes before each warp's input bytes
 * @param d_output Normalized output characters
 */
CUDF_KERNEL void write_spaces_kernel(spaces_input input,
                                     uint32_t const* d_bits,
                                     int64_t const* d_warp_offsets,
                                     char* d_output)
{
  auto const tid  = cudf::detail::grid_1d::global_thread_id();
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const pos  = tid * bytes_per_thread;

  auto const bits = (d_bits[pos / cudf::detail::warp_size] >> (pos % cudf::detail::warp_size)) &
                    ((1u << bytes_per_thread) - 1);
  auto const count = cuda::std::popcount(bits);

  auto offset           = cg::exclusive_scan(warp, count);
  auto const warp_total = warp.shfl(offset + count, cudf::detail::warp_size - 1);

  __shared__ char block_output[block_size / cudf::detail::warp_size][bytes_per_warp];
  auto warp_output = block_output[warp.meta_group_rank()];
  if (bits != 0) {
    auto const chunk = input.load_chunk(pos);
    for (auto k = 0; k < bytes_per_thread; ++k) {
      if ((bits >> k) & 1) {
        auto const ch         = chunk.data[k];
        warp_output[offset++] = is_whitespace(ch) ? ' ' : ch;
      }
    }
  }
  warp.sync();

  auto const d_warp_output = d_output + d_warp_offsets[tid / cudf::detail::warp_size];
  for (auto i = static_cast<int32_t>(warp.thread_rank()); i < warp_total;
       i += cudf::detail::warp_size) {
    d_warp_output[i] = warp_output[i];
  }
}

}  // namespace

// detail API
std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& input,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  if (chars_size == 0) {
    return std::make_unique<cudf::column>(input.parent(), stream, mr.get_output_mr());
  }

  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const temp_mr = mr.get_temporary_mr();
  auto const policy  = rmm::exec_policy_nosync(stream, temp_mr);

  auto const shift =
    static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(d_input_chars) % bytes_per_thread);
  auto const d_input = spaces_input{d_input_chars - shift, shift, shift + chars_size};

  cudf::detail::grid_1d grid{cudf::util::div_rounding_up_safe(d_input.end, bytes_per_thread),
                             block_size};
  auto const num_warps =
    static_cast<std::size_t>(grid.num_blocks * (block_size / cudf::detail::warp_size));

  // bitmask with a bit per input byte; padded to cover every warp
  auto const num_words = num_warps * words_per_warp;
  auto d_bits =
    cuda::make_buffer<uint32_t, cuda::mr::device_accessible>(stream, temp_mr, num_words, 0u);

  // mark the row boundary exceptions
  cudf::detail::grid_1d rows_grid{input.size(), block_size};
  mark_row_spaces_kernel<<<rows_grid.num_blocks,
                           rows_grid.num_threads_per_block,
                           0,
                           stream.get()>>>(
    d_input, input_offsets, first_offset, input.size(), d_bits.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // identify the kept bytes and count them for each warp
  auto d_warp_offsets = cuda::device_buffer<int64_t>{stream, temp_mr, num_warps + 1, cuda::no_init};
  keep_spaces_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    d_input, d_bits.data(), d_warp_offsets.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // convert the warp counts to warp offsets; the last element is the total
  thrust::inclusive_scan(
    policy, d_warp_offsets.begin(), d_warp_offsets.end(), d_warp_offsets.data());
  int64_t total_size = 0;
  cudf::detail::cuda_memcpy(cudf::host_span<int64_t>{&total_size, 1},
                            cudf::device_span<int64_t const>{d_warp_offsets.data() + num_warps, 1},
                            stream);

  // the output offset of each row is the number of kept bytes before the row
  auto offsets = cudf::strings::detail::create_offsets_child_column(
    total_size, input.size() + 1, stream, mr.get_output_mr());
  auto d_offsets =
    cudf::detail::offsetalator_factory::make_output_iterator(offsets->mutable_view());
  thrust::transform(policy,
                    input_offsets,
                    input_offsets + input.size() + 1,
                    d_offsets,
                    output_position_fn{d_bits.data(), d_warp_offsets.data(), shift - first_offset});

  // write the kept characters
  auto chars = rmm::device_uvector<char>(total_size, stream, mr.get_output_mr());
  write_spaces_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    d_input, d_bits.data(), d_warp_offsets.data(), chars.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& input,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_spaces(input, stream, mr);
}

}  // namespace nvtext
