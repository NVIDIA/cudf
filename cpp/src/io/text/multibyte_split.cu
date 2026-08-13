/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/output_builder.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/io/text/detail/multistate.hpp>
#include <cudf/io/text/detail/tile_state.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/utility>
#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/transform.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

namespace cudf::io::text {
namespace detail {
namespace {

using cudf::io::text::detail::multistate;

int32_t constexpr ITEMS_PER_THREAD = 64;
int32_t constexpr THREADS_PER_TILE = 128;
int32_t constexpr ITEMS_PER_TILE   = ITEMS_PER_THREAD * THREADS_PER_TILE;
int32_t constexpr TILES_PER_CHUNK  = 4096;
int32_t constexpr ITEMS_PER_CHUNK  = ITEMS_PER_TILE * TILES_PER_CHUNK;

__device__ constexpr multistate transition_init(char c, cudf::device_span<char const> delim)
{
  auto result = multistate();

  result.enqueue(0, 0);

  for (std::size_t i = 0; i < delim.size(); i++) {
    if (delim[i] == c) { result.enqueue(i, i + 1); }
  }

  return result;
}

__device__ constexpr multistate transition(char c,
                                           multistate state,
                                           cudf::device_span<char const> delim)
{
  auto result = multistate();

  result.enqueue(0, 0);

  for (uint8_t i = 0; i < state.size(); i++) {
    auto const tail = state.get_tail(i);
    if (tail < delim.size() && delim[tail] == c) { result.enqueue(state.get_head(i), tail + 1); }
  }

  return result;
}

struct PatternScan {
  using BlockScan         = cub::BlockScan<multistate, THREADS_PER_TILE>;
  using BlockScanCallback = cudf::io::text::detail::scan_tile_state_callback<multistate>;

  struct _TempStorage {
    typename BlockScan::TempStorage scan;
  };

  _TempStorage& _temp_storage;

  using TempStorage = cub::Uninitialized<_TempStorage>;

  __device__ inline PatternScan(TempStorage& temp_storage) : _temp_storage(temp_storage.Alias()) {}

  __device__ inline void Scan(cudf::size_type tile_idx,
                              cudf::io::text::detail::scan_tile_state_view<multistate> tile_state,
                              cudf::device_span<char const> delim,
                              char (&thread_data)[ITEMS_PER_THREAD],
                              multistate& thread_multistate)
  {
    thread_multistate = transition_init(thread_data[0], delim);

    for (uint32_t i = 1; i < ITEMS_PER_THREAD; i++) {
      thread_multistate = transition(thread_data[i], thread_multistate, delim);
    }

    auto prefix_callback = BlockScanCallback(tile_state, tile_idx);

    BlockScan(_temp_storage.scan)
      .ExclusiveSum(thread_multistate, thread_multistate, prefix_callback);
  }
};

// type aliases to distinguish between row offsets and character offsets
using output_offset = int64_t;
using byte_offset   = int64_t;

// multibyte_split works by splitting up inputs in to 32 inputs (bytes) per thread, and transforming
// them in to data structures called "multistates". these multistates are created by searching a
// trie, but instead of a tradition trie where the search begins at a single node at the beginning,
// we allow our search to begin anywhere within the trie tree. The position within the trie tree is
// stored as a "partial match path", which indicates "we can get from here to there by a set of
// specific transitions". By scanning together multistates, we effectively know "we can get here
// from the beginning by following the inputs". By doing this, each thread knows exactly what state
// it begins in. From there, each thread can then take deterministic action. In this case, the
// deterministic action is counting and outputting delimiter offsets when a delimiter is found.

CUDF_KERNEL void multibyte_split_init_kernel(
  cudf::size_type base_tile_idx,
  cudf::size_type num_tiles,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<output_offset> tile_output_offsets,
  cudf::io::text::detail::scan_tile_status status =
    cudf::io::text::detail::scan_tile_status::invalid)
{
  auto const thread_idx = cudf::detail::grid_1d::global_thread_id();
  if (thread_idx < num_tiles) {
    auto const tile_idx = base_tile_idx + thread_idx;
    tile_multistates.set_status(tile_idx, status);
    tile_output_offsets.set_status(tile_idx, status);
  }
}

CUDF_KERNEL __launch_bounds__(THREADS_PER_TILE) void multibyte_split_kernel(
  cudf::size_type base_tile_idx,
  byte_offset base_input_offset,
  output_offset base_output_offset,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<output_offset> tile_output_offsets,
  cudf::device_span<char const> delim,
  cudf::device_span<char const> chunk_input_chars,
  cudf::split_device_span<byte_offset> row_offsets)
{
  using InputLoad =
    cub::BlockLoad<char, THREADS_PER_TILE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using OffsetScan         = cub::BlockScan<output_offset, THREADS_PER_TILE>;
  using OffsetScanCallback = cudf::io::text::detail::scan_tile_state_callback<output_offset>;

  __shared__ union {
    typename InputLoad::TempStorage input_load;
    typename PatternScan::TempStorage pattern_scan;
    typename OffsetScan::TempStorage offset_scan;
  } temp_storage;

  auto const tile_idx          = base_tile_idx + blockIdx.x;
  auto const tile_input_offset = blockIdx.x * ITEMS_PER_TILE;
  auto const thread_input_offset =
    tile_input_offset + cudf::thread_index_type{threadIdx.x} * ITEMS_PER_THREAD;
  auto const thread_input_size =
    cuda::std::max<cudf::size_type>(chunk_input_chars.size() - thread_input_offset, 0);

  // STEP 1: Load inputs

  char thread_chars[ITEMS_PER_THREAD];

  InputLoad(temp_storage.input_load)
    .Load(chunk_input_chars.data() + tile_input_offset,
          thread_chars,
          chunk_input_chars.size() - tile_input_offset);

  // STEP 2: Scan inputs to determine absolute thread states

  multistate thread_multistate;

  __syncthreads();  // required before temp_memory re-use
  PatternScan(temp_storage.pattern_scan)
    .Scan(tile_idx, tile_multistates, delim, thread_chars, thread_multistate);

  // STEP 3: Flag matches

  output_offset thread_offset{};
  uint32_t thread_match_mask[(ITEMS_PER_THREAD + 31) / 32]{};

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_multistate       = transition(thread_chars[i], thread_multistate, delim);
    auto const thread_state = thread_multistate.max_tail();
    auto const is_match     = i < thread_input_size and thread_state == delim.size();
    thread_match_mask[i / 32] |= uint32_t{is_match} << (i % 32);
    thread_offset += output_offset{is_match};
  }

  // STEP 4: Scan flags to determine absolute thread output offset

  auto prefix_callback = OffsetScanCallback(tile_output_offsets, tile_idx);

  __syncthreads();  // required before temp_memory re-use
  OffsetScan(temp_storage.offset_scan).ExclusiveSum(thread_offset, thread_offset, prefix_callback);

  // Step 5: Assign outputs from each thread using match offsets.

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    auto const is_match = (thread_match_mask[i / 32] >> (i % 32)) & 1u;
    if (is_match) {
      auto const match_end = base_input_offset + thread_input_offset + i + 1;
      row_offsets[thread_offset - base_output_offset] = match_end;
      thread_offset++;
    }
  }
}

CUDF_KERNEL __launch_bounds__(THREADS_PER_TILE) void byte_split_kernel(
  cudf::size_type base_tile_idx,
  byte_offset base_input_offset,
  output_offset base_output_offset,
  cudf::io::text::detail::scan_tile_state_view<output_offset> tile_output_offsets,
  char delim,
  cudf::device_span<char const> chunk_input_chars,
  cudf::split_device_span<byte_offset> row_offsets)
{
  using InputLoad =
    cub::BlockLoad<char, THREADS_PER_TILE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using OffsetScan         = cub::BlockScan<output_offset, THREADS_PER_TILE>;
  using OffsetScanCallback = cudf::io::text::detail::scan_tile_state_callback<output_offset>;

  __shared__ union {
    typename InputLoad::TempStorage input_load;
    typename OffsetScan::TempStorage offset_scan;
  } temp_storage;

  auto const tile_idx          = base_tile_idx + blockIdx.x;
  auto const tile_input_offset = blockIdx.x * ITEMS_PER_TILE;
  auto const thread_input_offset =
    tile_input_offset + cudf::thread_index_type{threadIdx.x} * ITEMS_PER_THREAD;
  auto const thread_input_size =
    cuda::std::max<cudf::size_type>(chunk_input_chars.size() - thread_input_offset, 0);

  // STEP 1: Load inputs

  char thread_chars[ITEMS_PER_THREAD];

  InputLoad(temp_storage.input_load)
    .Load(chunk_input_chars.data() + tile_input_offset,
          thread_chars,
          chunk_input_chars.size() - tile_input_offset);

  // STEP 2: Flag matches

  output_offset thread_offset{};
  uint32_t thread_match_mask[(ITEMS_PER_THREAD + 31) / 32]{};

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    auto const is_match = i < thread_input_size and thread_chars[i] == delim;
    thread_match_mask[i / 32] |= uint32_t{is_match} << (i % 32);
    thread_offset += output_offset{is_match};
  }

  // STEP 3: Scan flags to determine absolute thread output offset

  auto prefix_callback = OffsetScanCallback(tile_output_offsets, tile_idx);

  __syncthreads();  // required before temp_memory re-use
  OffsetScan(temp_storage.offset_scan).ExclusiveSum(thread_offset, thread_offset, prefix_callback);

  // Step 4: Assign outputs from each thread using match offsets.

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    auto const is_match = (thread_match_mask[i / 32] >> (i % 32)) & 1u;
    if (is_match) {
      auto const match_end = base_input_offset + thread_input_offset + i + 1;
      row_offsets[thread_offset - base_output_offset] = match_end;
      thread_offset++;
    }
  }
}

}  // namespace

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source const& source,
                                              std::string_view delimiter,
                                              byte_range_info byte_range,
                                              bool strip_delimiters,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (byte_range.is_empty()) { return make_empty_column(type_id::STRING); }

  auto device_delim = cudf::string_scalar(delimiter, true, stream, mr);

  std::string sorted_delim{delimiter};
  std::sort(sorted_delim.begin(), sorted_delim.end());
  auto [_last_char, _last_char_count, max_duplicate_tokens] = std::accumulate(
    sorted_delim.begin(), sorted_delim.end(), std::make_tuple('\0', 0, 0), [](auto acc, char c) {
      if (std::get<0>(acc) != c) {
        std::get<0>(acc) = c;
        std::get<1>(acc) = 0;
      }
      std::get<1>(acc)++;
      std::get<2>(acc) = std::max(std::get<1>(acc), std::get<2>(acc));
      return acc;
    });

  CUDF_EXPECTS(max_duplicate_tokens < multistate::max_segment_count,
               "delimiter contains too many duplicate tokens to produce a deterministic result.");

  CUDF_EXPECTS(delimiter.size() < multistate::max_segment_value,
               "delimiter contains too many total tokens to produce a deterministic result.");

  auto chunk_offset = std::max<byte_offset>(0, byte_range.offset() - delimiter.size());
  std::optional<byte_offset> first_row_offset;
  if (byte_range.offset() == 0) { first_row_offset = 0; }
  std::optional<byte_offset> last_row_offset;

  auto [global_offsets, chars] = [&] {
    // must be at least 32 when using warp-reduce on partials
    // must be at least 1 more than max possible concurrent tiles
    // best when at least 32 more than max possible concurrent tiles, due to rolling `invalid`s
    auto const concurrency = 2;
    auto num_tile_states   = std::max(32, TILES_PER_CHUNK * concurrency + 32);
    auto tile_multistates =
      scan_tile_state<multistate>(num_tile_states, stream, cudf::get_current_device_resource_ref());
    auto tile_offsets = scan_tile_state<output_offset>(
      num_tile_states, stream, cudf::get_current_device_resource_ref());

    multibyte_split_init_kernel<<<TILES_PER_CHUNK,
                                  THREADS_PER_TILE,
                                  0,
                                  stream.value()>>>(  //
      -TILES_PER_CHUNK,
      TILES_PER_CHUNK,
      tile_multistates,
      tile_offsets,
      cudf::io::text::detail::scan_tile_status::oob);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto multistate_seed = multistate();
    multistate_seed.enqueue(0, 0);  // this represents the first state in the pattern.

    // Seeding the tile state with an identity value allows the 0th tile to follow the same logic as
    // the Nth tile, assuming it can look up an inclusive prefix. Without this seed, the 0th block
    // would have to follow separate logic.
    cudf::detail::device_single_thread(
      [tm = scan_tile_state_view<multistate>(tile_multistates),
       to = scan_tile_state_view<output_offset>(tile_offsets),
       multistate_seed] __device__() mutable {
        tm.set_inclusive_prefix(-1, multistate_seed);
        to.set_inclusive_prefix(-1, 0);
      },
      stream);

    auto reader               = source.create_reader();
    auto const byte_range_end = byte_range.offset() + byte_range.size();
    reader->skip_bytes(chunk_offset);
    // amortize output chunk allocations over 8 worst-case outputs. This limits the overallocation
    constexpr auto max_growth = 8;
    output_builder<byte_offset> row_offset_storage(
      ITEMS_PER_CHUNK, max_growth, stream, cudf::get_current_device_resource_ref());
    output_builder<char> char_storage(
      ITEMS_PER_CHUNK, max_growth, stream, cudf::get_current_device_resource_ref());

    auto streams = cudf::detail::fork_streams(stream, concurrency);

    cudaEvent_t last_launch_event;
    CUDF_CUDA_TRY(cudaEventCreate(&last_launch_event));

    auto& read_stream      = streams[0];
    auto& scan_stream      = streams[1];
    auto chunk             = reader->get_next_chunk(ITEMS_PER_CHUNK, read_stream);
    int64_t base_tile_idx  = 0;
    bool found_last_offset = false;
    std::swap(read_stream, scan_stream);

    while (chunk->size() > 0) {
      // if we found the last delimiter, or didn't find delimiters inside the byte range at all:
      // abort
      if (last_row_offset.has_value() or
          (not first_row_offset.has_value() and chunk_offset >= byte_range_end)) {
        break;
      }

      auto tiles_in_launch =
        cudf::util::div_rounding_up_safe(chunk->size(), static_cast<std::size_t>(ITEMS_PER_TILE));

      auto row_offsets = row_offset_storage.next_output(scan_stream);

      // reset the next chunk of tile state
      multibyte_split_init_kernel<<<tiles_in_launch,
                                    THREADS_PER_TILE,
                                    0,
                                    scan_stream.value()>>>(  //
        base_tile_idx,
        tiles_in_launch,
        tile_multistates,
        tile_offsets);

      CUDF_CUDA_TRY(cudaStreamWaitEvent(scan_stream.value(), last_launch_event));

      if (delimiter.size() == 1) {
        // the single-byte case allows for a much more efficient kernel, so we special-case it
        byte_split_kernel<<<tiles_in_launch,
                            THREADS_PER_TILE,
                            0,
                            scan_stream.value()>>>(  //
          base_tile_idx,
          chunk_offset,
          row_offset_storage.size(),
          tile_offsets,
          delimiter[0],
          *chunk,
          row_offsets);
        CUDF_CUDA_TRY(cudaGetLastError());
      } else {
        multibyte_split_kernel<<<tiles_in_launch,
                                 THREADS_PER_TILE,
                                 0,
                                 scan_stream.value()>>>(  //
          base_tile_idx,
          chunk_offset,
          row_offset_storage.size(),
          tile_multistates,
          tile_offsets,
          {device_delim.data(), static_cast<std::size_t>(device_delim.size())},
          *chunk,
          row_offsets);
        CUDF_CUDA_TRY(cudaGetLastError());
      }

      // load the next chunk
      auto next_chunk = reader->get_next_chunk(ITEMS_PER_CHUNK, read_stream);
      // while that is running, determine how many offsets we output (synchronizes)
      auto const new_offsets = [&] {
        auto const new_offsets_unclamped =
          tile_offsets.get_inclusive_prefix(base_tile_idx + tiles_in_launch - 1, scan_stream) -
          static_cast<output_offset>(row_offset_storage.size());
        // if we are not in the last chunk, we can use all offsets
        if (chunk_offset + static_cast<output_offset>(chunk->size()) < byte_range_end) {
          return new_offsets_unclamped;
        }
        // if we are in the last chunk, we need to find the first out-of-bounds offset
        auto const it      = cuda::counting_iterator<output_offset>{};
        auto const end_loc = *thrust::find_if(
          rmm::exec_policy_nosync(scan_stream, cudf::get_current_device_resource_ref()),
          it,
          it + new_offsets_unclamped,
          cuda::proclaim_return_type<bool>(
            [row_offsets, byte_range_end] __device__(output_offset i) {
              return row_offsets[i] >= byte_range_end;
            }));
        // if we had no out-of-bounds offset, we copy all offsets
        if (end_loc == new_offsets_unclamped) { return end_loc; }
        // otherwise we copy only up to (including) the first out-of-bounds delimiter
        found_last_offset = true;
        return end_loc + 1;
      }();
      row_offset_storage.advance_output(new_offsets, scan_stream);
      // determine if we found the first or last field offset for the byte range
      if (new_offsets > 0 and not first_row_offset) {
        first_row_offset = row_offset_storage.front_element(scan_stream);
      }
      if (found_last_offset) { last_row_offset = row_offset_storage.back_element(scan_stream); }
      // copy over the characters we need, if we already encountered the first field delimiter
      if (first_row_offset.has_value()) {
        auto const begin =
          chunk->data() + std::max<byte_offset>(0, *first_row_offset - chunk_offset);
        auto const sentinel = last_row_offset.value_or(std::numeric_limits<byte_offset>::max());
        auto const end =
          chunk->data() + std::min<byte_offset>(sentinel - chunk_offset, chunk->size());
        auto const output_size = end - begin;
        auto char_output       = char_storage.next_output(scan_stream);
        thrust::copy(rmm::exec_policy_nosync(scan_stream, cudf::get_current_device_resource_ref()),
                     begin,
                     end,
                     char_output.begin());
        char_storage.advance_output(output_size, scan_stream);
      }

      CUDF_CUDA_TRY(cudaEventRecord(last_launch_event, scan_stream.value()));

      std::swap(read_stream, scan_stream);
      base_tile_idx += tiles_in_launch;
      chunk_offset += chunk->size();
      chunk = std::move(next_chunk);
    }

    CUDF_CUDA_TRY(cudaEventDestroy(last_launch_event));

    cudf::detail::join_streams(streams, stream);

    auto chars          = char_storage.gather(stream, mr);
    auto global_offsets = row_offset_storage.gather(stream, mr);
    return std::pair{std::move(global_offsets), std::move(chars)};
  }();

  // if the input was empty, we didn't find a delimiter at all,
  // or the first delimiter was also the last: empty output
  if (chunk_offset == 0 or not first_row_offset.has_value() or
      first_row_offset == last_row_offset) {
    return make_empty_column(type_id::STRING);
  }

  // insert an offset at the beginning if we started at the beginning of the input
  bool const insert_begin = first_row_offset.value_or(0) == 0;
  // insert an offset at the end if we have not terminated the last row
  bool const insert_end =
    not(last_row_offset.has_value() or
        (global_offsets.size() > 0 and global_offsets.back_element(stream) == chunk_offset));
  auto const chars_bytes = chunk_offset - *first_row_offset;
  auto offsets           = cudf::strings::detail::create_offsets_child_column(
    chars_bytes, global_offsets.size() + insert_begin + insert_end, stream, mr);
  auto offsets_itr =
    cudf::detail::offsetalator_factory::make_output_iterator(offsets->mutable_view());
  auto set_offset_value = [offsets_itr, stream](size_type index, int64_t value) {
    cudf::detail::device_single_thread(
      [offsets_itr, index, value] __device__() mutable { offsets_itr[index] = value; }, stream);
  };
  if (insert_begin) { set_offset_value(0, 0); }
  if (insert_end) { set_offset_value(offsets->size() - 1, chars_bytes); }
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    global_offsets.begin(),
                    global_offsets.end(),
                    offsets_itr + insert_begin,
                    cuda::proclaim_return_type<int64_t>(
                      [baseline = *first_row_offset] __device__(byte_offset global_offset) {
                        return (global_offset - baseline);
                      }));
  auto string_count = offsets->size() - 1;
  if (strip_delimiters) {
    auto it = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<cuda::std::pair<char*, int32_t>>(
        [ofs        = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view()),
         chars      = chars.data(),
         delim_size = static_cast<size_type>(delimiter.size()),
         last_row   = static_cast<size_type>(string_count) - 1,
         insert_end] __device__(size_type row) {
          auto const begin = ofs[row];
          auto const len   = static_cast<size_type>(ofs[row + 1] - begin);
          if (row == last_row && insert_end) {
            return cuda::std::make_pair(chars + begin, len);
          } else {
            return cuda::std::make_pair(chars + begin,
                                        cuda::std::max<size_type>(0, len - delim_size));
          };
        }));
    return cudf::strings::detail::make_strings_column(it, it + string_count, stream, mr);
  } else {
    return cudf::make_strings_column(string_count, std::move(offsets), chars.release(), 0, {});
  }
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source const& source,
                                              std::string_view delimiter,
                                              parse_options options,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto result = detail::multibyte_split(
    source, delimiter, options.byte_range, options.strip_delimiters, stream, mr);

  return result;
}

}  // namespace cudf::io::text
