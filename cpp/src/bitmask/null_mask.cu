/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/numeric>
#include <cuda/stream_ref>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <numeric>
#include <span>

namespace cudf {
size_type state_null_count(mask_state state, size_type size)
{
  switch (state) {
    case mask_state::UNALLOCATED: return 0;
    case mask_state::ALL_NULL: return size;
    case mask_state::ALL_VALID: return 0;
    default: CUDF_FAIL("Invalid null mask state.", std::invalid_argument);
  }
}

// Computes required allocation size of a bitmask
std::size_t bitmask_allocation_size_bytes(size_type number_of_bits, std::size_t padding_boundary)
{
  CUDF_EXPECTS(padding_boundary > 0, "Invalid padding boundary");
  auto necessary_bytes = cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes = padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                                           necessary_bytes, padding_boundary);
  return padded_bytes;
}

// Computes number of *actual* bitmask_type elements needed
size_type num_bitmask_words(size_type number_of_bits)
{
  return cudf::util::div_rounding_up_safe<size_type>(number_of_bits,
                                                     detail::size_in_bits<bitmask_type>());
}

namespace detail {

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size,
                                    mask_state state,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  size_type mask_size{0};

  if (state != mask_state::UNALLOCATED) { mask_size = bitmask_allocation_size_bytes(size); }

  rmm::device_buffer mask(mask_size, stream, mr);

  if (state != mask_state::UNINITIALIZED) {
    uint8_t fill_value = (state == mask_state::ALL_VALID) ? 0xff : 0x00;
    CUDF_CUDA_TRY(cudaMemsetAsync(
      static_cast<bitmask_type*>(mask.data()), fill_value, mask_size, stream.get()));
  }

  return mask;
}

namespace {

/**
 * @brief Enum to specify whether to atomically bulk set terminal null mask words (SAFE) or not
 * (UNSAFE)
 */
enum class mask_set_mode : bool { SAFE = true, UNSAFE = false };

/**
 * @brief Sets a range of bits in a null mask in parallel using the threads in cooperative group
 *
 * @tparam ThreadGroup Typename of the cooperative group
 * @tparam mode Whether (SAFE) or not (UNSAFE) to atomically set terminal null mask words
 *
 * @param[out] destination Pointer to the null mask
 * @param[in]  begin_bit The starting bit index (inclusive)
 * @param[in]  end_bit The ending bit index (exclusive)
 * @param[in]  valid Whether to set/unset the bits
 * @param[in]  number_of_mask_words The total number of 32-bit words in the null mask
 * @param[in]  group Cooperative group (1d grid or thread block)
 */
template <typename ThreadGroup, mask_set_mode MODE>
__device__ void set_null_mask_impl(bitmask_type* __restrict__ destination,
                                   size_type begin_bit,
                                   size_type end_bit,
                                   bool valid,
                                   size_type number_of_mask_words,
                                   ThreadGroup const& group)
{
  auto x                            = destination + word_index(begin_bit);
  thread_index_type const last_word = word_index(end_bit) - word_index(begin_bit);
  bitmask_type fill_value           = valid ? 0xffff'ffff : 0;

  for (thread_index_type destination_word_index = group.thread_rank();
       destination_word_index < number_of_mask_words;
       destination_word_index += group.num_threads()) {
    if (destination_word_index == 0 || destination_word_index == last_word) {
      bitmask_type mask = ~bitmask_type{0};
      if (destination_word_index == 0) {
        mask = ~(set_least_significant_bits(intra_word_index(begin_bit)));
      }
      if (destination_word_index == last_word) {
        mask = mask & set_least_significant_bits(intra_word_index(end_bit));
      }
      if constexpr (MODE == mask_set_mode::SAFE) {
        // Atomic ref to the destination word. Using thread block scope as this case is only
        // encountered when ThreadGroup = cg::thread_block
        auto destination_word_ref =
          cuda::atomic_ref<bitmask_type, cuda::thread_scope_block>(x[destination_word_index]);

        auto constexpr memory_order = cuda::std::memory_order_relaxed;
        valid ? destination_word_ref.fetch_or(mask, memory_order)
              : destination_word_ref.fetch_and(~mask, memory_order);
      } else {
        x[destination_word_index] =
          valid ? x[destination_word_index] | mask : x[destination_word_index] & ~mask;
      }
    } else {
      x[destination_word_index] = fill_value;
    }
  }
}

template <mask_set_mode MODE>
CUDF_KERNEL void set_null_masks_kernel(cudf::device_span<bitmask_type*> destinations,
                                       cudf::device_span<size_type const> begin_bits,
                                       cudf::device_span<size_type const> end_bits,
                                       cudf::device_span<bool const> valids,
                                       cudf::device_span<size_type const> numbers_of_mask_words)
{
  namespace cg           = cooperative_groups;
  auto const bitmask_idx = cg::this_grid().block_rank();
  // Return early if nothing to do
  if (begin_bits[bitmask_idx] == end_bits[bitmask_idx]) { return; }
  set_null_mask_impl<cg::thread_block, MODE>(destinations[bitmask_idx],
                                             begin_bits[bitmask_idx],
                                             end_bits[bitmask_idx],
                                             valids[bitmask_idx],
                                             numbers_of_mask_words[bitmask_idx],
                                             cg::this_thread_block());
}

CUDF_KERNEL void set_null_mask_kernel(bitmask_type* destination,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      bool valid,
                                      size_type number_of_mask_words)
{
  namespace cg = cooperative_groups;
  set_null_mask_impl<cg::grid_group, mask_set_mode::UNSAFE>(
    destination, begin_bit, end_bit, valid, number_of_mask_words, cg::this_grid());
}
}  // namespace

// Set pre-allocated null masks of given bit ranges [begin_bit, end_bit) to valids, if valid==true,
// or null, otherwise;
template <mask_set_mode MODE>
void set_null_masks(cudf::host_span<bitmask_type*> bitmasks,
                    cudf::host_span<size_type const> begin_bits,
                    cudf::host_span<size_type const> end_bits,
                    cudf::host_span<bool const> valids,
                    cuda::stream_ref stream)
{
  auto const num_bitmasks = bitmasks.size();

  CUDF_EXPECTS(num_bitmasks == begin_bits.size(),
               "Number of bitmasks and begin bits must be equal.");
  CUDF_EXPECTS(num_bitmasks == end_bits.size(), "Number of bitmasks and end bits must be equal.");
  CUDF_EXPECTS(num_bitmasks == valids.size(), "Number of bitmasks and valids must be equal.");

  // Return early if no bitmasks to set
  if (num_bitmasks == 0) { return; }

  size_t average_nullmask_words     = 0;
  size_t cumulative_null_mask_words = 0;
  auto h_number_of_mask_words =
    cudf::detail::make_pinned_vector_async<size_type>(num_bitmasks, stream);
  thrust::tabulate(
    thrust::host, h_number_of_mask_words.begin(), h_number_of_mask_words.end(), [&](auto i) {
      CUDF_EXPECTS(begin_bits[i] >= 0, "Invalid range.");
      CUDF_EXPECTS(begin_bits[i] <= end_bits[i], "Invalid bit range.");
      // Return 0 if bitmask is empty
      if (begin_bits[i] == end_bits[i]) { return size_t{0}; }
      // Number of words in this bitmask
      auto const num_words =
        num_bitmask_words(end_bits[i]) - begin_bits[i] / detail::size_in_bits<bitmask_type>();
      // Handle overflow if any
      if (cuda::add_overflow<size_t>(cumulative_null_mask_words, num_words).overflow) {
        average_nullmask_words +=
          cudf::util::div_rounding_up_safe<size_t>(cumulative_null_mask_words, num_bitmasks);
        cumulative_null_mask_words = 0;
      }
      // Add to cumulative null mask words
      cumulative_null_mask_words += num_words;
      return num_words;
    });

  // Add the last cumulative null mask words to average
  average_nullmask_words +=
    cudf::util::div_rounding_up_safe<size_t>(cumulative_null_mask_words, num_bitmasks);

  // Create device vectors from host spans
  auto const mr           = cudf::get_current_device_resource_ref();
  auto destinations       = cudf::detail::make_device_uvector_async(bitmasks, stream, mr);
  auto const d_begin_bits = cudf::detail::make_device_uvector_async(begin_bits, stream, mr);
  auto const d_end_bits   = cudf::detail::make_device_uvector_async(end_bits, stream, mr);
  auto const d_valids     = cudf::detail::make_device_uvector_async(valids, stream, mr);
  auto const number_of_mask_words =
    cudf::detail::make_device_uvector_async(h_number_of_mask_words, stream, mr);

  // Compute block size using heuristic and launch kernel
  constexpr size_t max_words_per_thread  = 64;
  constexpr size_t min_threads_per_block = 128;
  auto block_size =
    std::max<size_t>(min_threads_per_block, (average_nullmask_words / max_words_per_thread));
  // Round block size to nearest (ceil) power of 2
  block_size = size_t{1} << (63 - cuda::std::countl_zero(block_size));
  // Cap block size to 1024 threads
  block_size = std::min<size_t>(block_size, 1024);

  set_null_masks_kernel<MODE><<<num_bitmasks, block_size, 0, stream.get()>>>(
    destinations, d_begin_bits, d_end_bits, d_valids, number_of_mask_words);
  CUDF_CHECK_CUDA(stream.get());
}

// Set pre-allocated null mask of given bit range [begin_bit, end_bit) to valid, if valid==true,
// or null, otherwise;
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   cuda::stream_ref stream)
{
  CUDF_EXPECTS(begin_bit >= 0 and begin_bit <= end_bit, "Invalid bit range.");
  if (begin_bit == end_bit) return;
  if (bitmask != nullptr) {
    auto number_of_mask_words =
      num_bitmask_words(end_bit) - begin_bit / detail::size_in_bits<bitmask_type>();
    cudf::detail::grid_1d config(number_of_mask_words, 256);
    set_null_mask_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      static_cast<bitmask_type*>(bitmask), begin_bit, end_bit, valid, number_of_mask_words);
    CUDF_CHECK_CUDA(stream.get());
  }
}

}  // namespace detail

// Create a device_buffer for a null mask
rmm::device_buffer create_null_mask(size_type size,
                                    mask_state state,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::create_null_mask(size, state, stream, mr);
}

// Set pre-allocated null mask of given bit range [begin_bit, end_bit) to valid, if valid==true,
// or null, otherwise;
void set_null_mask(bitmask_type* bitmask,
                   size_type begin_bit,
                   size_type end_bit,
                   bool valid,
                   cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::set_null_mask(bitmask, begin_bit, end_bit, valid, stream);
}

// Bulk set pre-allocated null masks to corresponding valid state safely handling intra-word
// aliasing in the corresponding bit ranges
void set_null_masks_safe(cudf::host_span<bitmask_type*> bitmasks,
                         cudf::host_span<size_type const> begin_bits,
                         cudf::host_span<size_type const> end_bits,
                         cudf::host_span<bool const> valids,
                         cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::set_null_masks<detail::mask_set_mode::SAFE>(
    bitmasks, begin_bits, end_bits, valids, stream);
}

// Bulk set pre-allocated null masks to corresponding valid state without handling intra-word
// aliasing in the corresponding bit ranges
void set_null_masks_unsafe(cudf::host_span<bitmask_type*> bitmasks,
                           cudf::host_span<size_type const> begin_bits,
                           cudf::host_span<size_type const> end_bits,
                           cudf::host_span<bool const> valids,
                           cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::set_null_masks<detail::mask_set_mode::UNSAFE>(
    bitmasks, begin_bits, end_bits, valids, stream);
}

namespace detail {

namespace {
/**
 * @brief Copies the bits starting at the specified offset from a source
 * bitmask into the destination bitmask.
 *
 * Bit `i` in `destination` will be equal to bit `i + offset` from `source`.
 *
 * @param destination The mask to copy into
 * @param source The mask to copy from
 * @param source_begin_bit The offset into `source` from which to begin the copy
 * @param source_end_bit   The offset into `source` till which copying is done
 * @param number_of_mask_words The number of `cudf::bitmask_type` words to copy
 */
// TODO: Also make binops test that uses offset in column_view
CUDF_KERNEL void copy_offset_bitmask(bitmask_type* __restrict__ destination,
                                     bitmask_type const* __restrict__ source,
                                     size_type source_begin_bit,
                                     size_type source_end_bit,
                                     size_type number_of_mask_words)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (thread_index_type destination_word_index = grid_1d::global_thread_id();
       destination_word_index < number_of_mask_words;
       destination_word_index += stride) {
    destination[destination_word_index] = detail::get_mask_offset_word(
      source, destination_word_index, source_begin_bit, source_end_bit);
  }
}

}  // namespace

// Create a bitmask from a specific range
rmm::device_buffer copy_bitmask(bitmask_type const* mask,
                                size_type begin_bit,
                                size_type end_bit,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(begin_bit >= 0 and begin_bit <= end_bit, "Invalid bit range.");
  rmm::device_buffer dest_mask{};
  auto num_bytes = bitmask_allocation_size_bytes(end_bit - begin_bit);
  if ((mask == nullptr) || (num_bytes == 0)) { return dest_mask; }
  if (begin_bit == 0) {
    dest_mask = rmm::device_buffer{static_cast<void const*>(mask), num_bytes, stream, mr};
  } else {
    auto number_of_mask_words = num_bitmask_words(end_bit - begin_bit);
    dest_mask                 = rmm::device_buffer{num_bytes, stream, mr};
    cudf::detail::grid_1d config(number_of_mask_words, 256);
    copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      static_cast<bitmask_type*>(dest_mask.data()), mask, begin_bit, end_bit, number_of_mask_words);
    CUDF_CHECK_CUDA(stream.get());
  }
  return dest_mask;
}

// Create a bitmask from a column view
rmm::device_buffer copy_bitmask(column_view const& view,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.nullable()) {
    null_mask =
      copy_bitmask(view.null_mask(), view.offset(), view.offset() + view.size(), stream, mr);
  }
  return null_mask;
}

namespace {

/**
 * @brief Counts the number of non-zero bits for a list of bitmasks in the range
 * `[first_bit_index, last_bit_index]`.
 *
 * Expects `0 <= first_bit_index <= last_bit_index`.
 *
 * @tparam block_size Number of threads in each thread block
 *
 * @param bitmasks Array of bitmask pointers
 * @param first_bit_index The index (inclusive) of the first bit to count
 * @param last_bit_index The index (inclusive) of the last bit to count
 * @param global_count Output array of set bit counts for each bitmask
 */
template <size_type block_size>
CUDF_KERNEL void count_set_bits_kernel(device_span<bitmask_type const* const> bitmasks,
                                       size_type first_bit_index,
                                       size_type last_bit_index,
                                       size_type* global_count)
{
  auto const bitmask_idx = blockIdx.y;
  if (bitmask_idx >= bitmasks.size()) { return; }
  auto const bitmask = bitmasks[bitmask_idx];
  if (bitmask == nullptr) { return; }

  constexpr auto word_size{detail::size_in_bits<bitmask_type>()};
  auto const first_word_index{word_index(first_bit_index)};
  auto const last_word_index{word_index(last_bit_index)};

  // Although we use a 2D grid for launching the kernel, elements of each bitmask are indexed by
  // the first dimension.
  auto const tid         = grid_1d::global_thread_id<block_size>();
  auto const stride      = grid_1d::grid_stride<block_size>();
  auto thread_word_index = tid + first_word_index;
  size_type thread_count{0};

  // First, just count the bits in all words
  while (thread_word_index <= last_word_index) {
    thread_count += cuda::std::popcount(bitmask[thread_word_index]);
    thread_word_index += stride;
  }

  // Subtract any slack bits counted from the first and last word
  // Two threads handle this -- one for first word, one for last
  if (tid < 2) {
    bool const first{tid == 0};
    bool const last{not first};

    size_type bit_index  = (first) ? first_bit_index : last_bit_index;
    size_type word_index = (first) ? first_word_index : last_word_index;

    size_type num_slack_bits = bit_index % word_size;
    if (last) { num_slack_bits = word_size - num_slack_bits - 1; }

    if (num_slack_bits > 0) {
      bitmask_type word = bitmask[word_index];
      auto slack_mask   = (first) ? set_least_significant_bits(num_slack_bits)
                                  : set_most_significant_bits(num_slack_bits);

      thread_count -= cuda::std::popcount(word & slack_mask);
    }
  }

  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto const block_count = BlockReduce(temp_storage).Sum(thread_count);

  if (threadIdx.x == 0) {
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{global_count[bitmask_idx]}.fetch_add(
      block_count, cuda::std::memory_order_relaxed);
  }
}

// Count null elements in the specified range of multiple validity bitmasks
std::vector<size_type> batch_count_set_bits(host_span<bitmask_type const* const> bitmasks,
                                            size_type start,
                                            size_type stop,
                                            cuda::stream_ref stream)
{
  CUDF_EXPECTS(start >= 0 and start <= stop, "Invalid bit range.", std::invalid_argument);

  auto const num_bitmasks      = bitmasks.size();
  auto const num_bits_to_count = stop - start;

  std::vector<size_type> output(num_bitmasks, 0);
  if (num_bitmasks == 0 || num_bits_to_count == 0) { return output; }

  auto const has_bitmask =
    std::any_of(bitmasks.begin(), bitmasks.end(), [](auto p) { return p != nullptr; });
  if (!has_bitmask) { return output; }

  auto const tmp_mr     = cudf::get_current_device_resource_ref();
  auto const d_bitmasks = cudf::detail::make_device_uvector_async(bitmasks, stream, tmp_mr);
  auto const num_words  = num_bitmask_words(num_bits_to_count);
  auto d_non_zero_count =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(num_bitmasks, stream, tmp_mr);

  constexpr size_type block_size{256};
  // We use a 2D grid to launch the kernel, where the first dimension is to access elements in each
  // single bitmask while the second dimension is to access the bitmask indices.
  auto const grid = grid_1d{num_words, block_size};
  auto const kernel_grid =
    dim3{static_cast<unsigned int>(grid.num_blocks), static_cast<unsigned int>(num_bitmasks), 1};
  count_set_bits_kernel<block_size><<<kernel_grid, block_size, 0, stream.get()>>>(
    d_bitmasks, start, stop - 1, d_non_zero_count.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // Use pinned memory to copy the result back to the host, then copy again to the output vector.
  auto h_non_zero_count = cudf::detail::make_pinned_vector<size_type>(num_bitmasks, stream);
  cudf::detail::cuda_memcpy(host_span<size_type>{h_non_zero_count.data(), num_bitmasks},
                            device_span<size_type const>{d_non_zero_count.data(), num_bitmasks},
                            stream);
  std::copy(h_non_zero_count.begin(), h_non_zero_count.end(), output.begin());

  return output;
}

}  // namespace

// Count non-zero bits in the specified range
size_type count_set_bits(bitmask_type const* bitmask,
                         size_type start,
                         size_type stop,
                         cuda::stream_ref stream)
{
  CUDF_EXPECTS(bitmask != nullptr, "Invalid bitmask.");
  auto const bitmasks = std::vector<bitmask_type const*>{bitmask};
  return detail::batch_count_set_bits(bitmasks, start, stop, stream).front();
}

// Count zero bits in the specified range
size_type count_unset_bits(bitmask_type const* bitmask,
                           size_type start,
                           size_type stop,
                           cuda::stream_ref stream)
{
  auto const num_set_bits   = detail::count_set_bits(bitmask, start, stop, stream);
  auto const total_num_bits = stop - start;
  return total_num_bits - num_set_bits;
}

// Count valid elements in the specified range of a validity bitmask
size_type valid_count(bitmask_type const* bitmask,
                      size_type start,
                      size_type stop,
                      cuda::stream_ref stream)
{
  if (bitmask == nullptr) {
    CUDF_EXPECTS(start >= 0 and start <= stop, "Invalid bit range.");
    return stop - start;
  }
  return detail::count_set_bits(bitmask, start, stop, stream);
}

// Count null elements in the specified range of a validity bitmask
size_type null_count(bitmask_type const* bitmask,
                     size_type start,
                     size_type stop,
                     cuda::stream_ref stream)
{
  auto const valid_count = detail::valid_count(bitmask, start, stop, stream);
  auto const size        = stop - start;
  return size - valid_count;
}

// Count non-zero bits in the specified ranges of a bitmask
std::vector<size_type> segmented_count_set_bits(bitmask_type const* bitmask,
                                                host_span<size_type const> indices,
                                                cuda::stream_ref stream)
{
  return detail::segmented_count_set_bits(bitmask, indices.begin(), indices.end(), stream);
}

// Count zero bits in the specified ranges of a bitmask
std::vector<size_type> segmented_count_unset_bits(bitmask_type const* bitmask,
                                                  host_span<size_type const> indices,
                                                  cuda::stream_ref stream)
{
  return segmented_count_unset_bits(bitmask, indices.begin(), indices.end(), stream);
}

// Count valid elements in the specified ranges of a validity bitmask
std::vector<size_type> segmented_valid_count(bitmask_type const* bitmask,
                                             std::span<size_type const> indices,
                                             cuda::stream_ref stream)
{
  return segmented_valid_count(bitmask, indices.begin(), indices.end(), stream);
}

// Count null elements in the specified ranges of a validity bitmask
std::vector<size_type> segmented_null_count(bitmask_type const* bitmask,
                                            std::span<size_type const> indices,
                                            cuda::stream_ref stream)
{
  return segmented_null_count(bitmask, indices.begin(), indices.end(), stream);
}

// Inplace Bitwise AND of the masks
cudf::size_type inplace_bitmask_and(device_span<bitmask_type> dest_mask,
                                    host_span<bitmask_type const* const> masks,
                                    host_span<size_type const> begin_bits,
                                    size_type mask_size,
                                    cuda::stream_ref stream)
{
  return inplace_bitmask_binop(
    [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
    dest_mask,
    masks,
    begin_bits,
    mask_size,
    stream);
}

// Bitwise AND of the masks
std::pair<rmm::device_buffer, size_type> bitmask_and(host_span<bitmask_type const* const> masks,
                                                     host_span<size_type const> begin_bits,
                                                     size_type mask_size,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  return bitmask_binop(
    [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
    masks,
    begin_bits,
    mask_size,
    stream,
    mr);
}

// Returns the bitwise AND of the null masks of all columns in the table view
std::pair<rmm::device_buffer, size_type> bitmask_and(table_view const& view,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.num_rows() == 0 or view.num_columns() == 0) {
    return std::pair(std::move(null_mask), 0);
  }

  std::vector<bitmask_type const*> masks;
  std::vector<size_type> offsets;
  for (auto&& col : view) {
    if (col.nullable()) {
      masks.push_back(col.null_mask());
      offsets.push_back(col.offset());
    }
  }

  if (masks.size() > 0) {
    return cudf::detail::bitmask_binop(
      [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
      masks,
      offsets,
      view.num_rows(),
      stream,
      mr);
  }

  return std::pair(std::move(null_mask), 0);
}

// Returns the bitwise AND of the null masks of all columns in the same segment of the input masks
std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<column_view const> colviews,
                      host_span<size_type const> segment_offsets,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::all_of(colviews.begin(),
                           colviews.end(),
                           [&](auto const& view) { return view.size() == colviews[0].size(); }),
               "All column views must have the same number of elements");

  if (colviews[0].size() == 0 or colviews.size() == 0) { return {}; }

  std::vector<bitmask_type const*> masks;
  std::vector<size_type> masks_begin_bits(colviews.size(), 0);
  for (auto colview : colviews) {
    masks.push_back(colview.null_mask());
  }

  return cudf::detail::segmented_bitmask_binop(
    [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
    masks,
    masks_begin_bits,
    colviews[0].size(),
    segment_offsets,
    stream,
    mr);
}

std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<bitmask_type const* const> masks,
                      host_span<size_type const> segment_offsets,
                      size_type mask_size_bits,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr)
{
  if (masks.size() == 0 || mask_size_bits == 0) { return {}; }

  std::vector<size_type> masks_begin_bits(masks.size(), 0);

  return cudf::detail::segmented_bitmask_binop(
    [] __device__(bitmask_type left, bitmask_type right) { return left & right; },
    masks,
    masks_begin_bits,
    mask_size_bits,
    segment_offsets,
    stream,
    mr);
}

// Returns the bitwise OR of the null masks of all columns in the table view
std::pair<rmm::device_buffer, size_type> bitmask_or(table_view const& view,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  rmm::device_buffer null_mask{0, stream, mr};
  if (view.num_rows() == 0 or view.num_columns() == 0) {
    return std::pair(std::move(null_mask), 0);
  }

  std::vector<bitmask_type const*> masks;
  std::vector<size_type> offsets;
  for (auto&& col : view) {
    if (col.nullable()) {
      masks.push_back(col.null_mask());
      offsets.push_back(col.offset());
    }
  }

  if (static_cast<size_type>(masks.size()) == view.num_columns()) {
    return cudf::detail::bitmask_binop(
      [] __device__(bitmask_type left, bitmask_type right) { return left | right; },
      masks,
      offsets,
      view.num_rows(),
      stream,
      mr);
  }

  return std::pair(std::move(null_mask), 0);
}

void set_all_valid_null_masks(column_view const& input,
                              column& output,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  if (input.nullable() && output.size() > 0) {
    auto mask = detail::create_null_mask(output.size(), mask_state::ALL_VALID, stream, mr);
    output.set_null_mask(std::move(mask), 0);

    for (size_type i = 0; i < input.num_children(); ++i) {
      set_all_valid_null_masks(input.child(i), output.child(i), stream, mr);
    }
  }
}

namespace {

template <size_type block_size>
CUDF_KERNEL void find_first_set_bit_kernel(bitmask_type const* __restrict__ bitmask,
                                           size_type start,
                                           size_type stop,
                                           size_type max,
                                           size_type* index)
{
  constexpr auto word_size = detail::size_in_bits<bitmask_type>();

  auto const tid               = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const thread_word_index = tid + word_index(start);
  auto const end_word_index    = word_index(stop);

  auto bit_index = max;
  if (thread_word_index <= end_word_index) {
    auto const mask = detail::get_mask_offset_word(bitmask, tid, start, stop);
    // returned index is 1-based; 0 means no bits were set
    auto const mask_bit_index = __ffs(mask);
    if (mask_bit_index != 0) {
      bit_index = static_cast<size_type>(tid * word_size) + mask_bit_index - 1;
    }
  }

  // using cooperative_group::reduce(block,bit_index,cg::less());
  // fails at runtime on a V100 (sm70) with driver 535
  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage ts;
  auto const out_index = BlockReduce(ts).Reduce(bit_index, cuda::minimum());

  if (threadIdx.x == 0 && out_index != max) {
    cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*(index)};
    ref.fetch_min(out_index, cuda::std::memory_order_relaxed);
  }
}
}  // namespace

size_type index_of_first_set_bit(bitmask_type const* bitmask,
                                 size_type start,
                                 size_type stop,
                                 cuda::stream_ref stream)
{
  CUDF_EXPECTS(
    start >= 0 and start <= stop and start != stop, "Invalid bit range.", std::invalid_argument);
  if (bitmask == nullptr) { return 0; }

  auto const bit_count  = stop - start;
  auto const mask_words = num_bitmask_words(bit_count);

  auto d_index =
    cudf::detail::device_scalar<size_type>(stream, cudf::get_current_device_resource_ref());
  d_index.set_value_async(bit_count, stream);  // init to no set bits found

  constexpr size_type block_size = 256;
  auto const grid                = grid_1d{mask_words + 1, block_size};
  find_first_set_bit_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      bitmask, start, stop, bit_count, d_index.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  return d_index.value(stream);
}

}  // namespace detail

// Create a bitmask from a specific range
rmm::device_buffer copy_bitmask(bitmask_type const* mask,
                                size_type begin_bit,
                                size_type end_bit,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_bitmask(mask, begin_bit, end_bit, stream, mr);
}

// Create a bitmask from a column view
rmm::device_buffer copy_bitmask(column_view const& view,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::copy_bitmask(view, stream, mr);
}

std::pair<rmm::device_buffer, size_type> bitmask_and(table_view const& view,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::bitmask_and(view, stream, mr);
}

std::pair<rmm::device_buffer, size_type> bitmask_and(host_span<bitmask_type const* const> masks,
                                                     host_span<size_type const> begin_bits,
                                                     size_type mask_size,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::bitmask_and(masks, begin_bits, mask_size, stream, mr);
}

std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<column_view const> colviews,
                      host_span<size_type const> segment_offsets,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_bitmask_and(colviews, segment_offsets, stream, mr);
}

std::pair<std::vector<std::unique_ptr<rmm::device_buffer>>, std::vector<size_type>>
segmented_bitmask_and(host_span<bitmask_type const* const> masks,
                      host_span<size_type const> segment_offsets,
                      size_type mask_size_bits,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_bitmask_and(masks, segment_offsets, mask_size_bits, stream, mr);
}

std::pair<rmm::device_buffer, size_type> bitmask_or(table_view const& view,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::bitmask_or(view, stream, mr);
}

// Count non-zero bits in the specified range
cudf::size_type null_count(bitmask_type const* bitmask,
                           size_type start,
                           size_type stop,
                           cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::null_count(bitmask, start, stop, stream);
}

std::vector<size_type> batch_null_count(host_span<bitmask_type const* const> bitmasks,
                                        size_type start,
                                        size_type stop,
                                        cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  auto counts                  = detail::batch_count_set_bits(bitmasks, start, stop, stream);
  auto const num_bits_to_count = stop - start;
  for (std::size_t i = 0; i < bitmasks.size(); ++i) {
    if (bitmasks[i] != nullptr) { counts[i] = num_bits_to_count - counts[i]; }
  }
  return counts;
}

std::vector<size_type> segmented_valid_count(bitmask_type const* bitmask,
                                             std::span<size_type const> indices,
                                             cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_valid_count(bitmask, indices, stream);
}

std::vector<size_type> segmented_null_count(bitmask_type const* bitmask,
                                            std::span<size_type const> indices,
                                            cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_null_count(bitmask, indices, stream);
}

size_type index_of_first_set_bit(bitmask_type const* bitmask,
                                 size_type start,
                                 size_type stop,
                                 cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::index_of_first_set_bit(bitmask, start, stop, stream);
}

}  // namespace cudf
