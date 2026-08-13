/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x64_128.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/minhash.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace nvtext {
namespace detail {
namespace {

constexpr cudf::thread_index_type block_size = 256;
// for potentially tuning minhash_seed_kernel independently from block_size
constexpr cudf::thread_index_type tile_size = block_size;

// Number of a/b parameter values to process per thread.
// The intermediate values are stored in shared-memory and therefore limits this count.
// This value was found to be an efficient size for both uint32 and uint64
// hash types based on benchmarks.
constexpr cuda::std::size_t params_per_thread = 16;

// Separate kernels are used to process strings above and below this value (in bytes).
constexpr cudf::size_type wide_row_threshold = 1 << 18;  // 256K
// The number of blocks per string for the above-threshold kernel processing.
constexpr cudf::size_type blocks_per_row = 64;
// The above values were determined using the redpajama and books_sample datasets

/**
 * @brief Hashing kernel launched as a thread per tile-size (block or warp)
 * for strings column
 *
 * This kernel computes the hashes for each string using the seed and the specified
 * hash function. The width is used to compute rolling substrings to hash over.
 * The hashes are stored in d_hashes to be used in the minhash_kernel.
 *
 * This kernel also counts the number of strings above the wide_row_threshold
 * and proactively initializes the output values for those strings.
 *
 * @tparam HashFunction The hash function to use for this kernel
 * @tparam hash_value_type Derived from HashFunction result_type
 *
 * @param d_strings The input strings to hash
 * @param seed The seed used for the hash function
 * @param width Width in characters used for determining substrings to hash
 * @param d_hashes The resulting hash values are stored here
 * @param threshold_count Stores the number of strings above wide_row_threshold
 * @param param_count Number of parameters (used for the proactive initialize)
 * @param d_results Final results vector (used for the proactive initialize)
 */
template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
CUDF_KERNEL void minhash_seed_kernel(cudf::column_device_view const d_strings,
                                     hash_value_type seed,
                                     cudf::size_type width,
                                     hash_value_type* d_hashes,
                                     cudf::size_type* threshold_count,
                                     cudf::size_type param_count,
                                     hash_value_type* d_results)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }

  // retrieve this string's offset to locate the output position in d_hashes
  auto const offsets = d_strings.child(cudf::strings_column_view::offsets_column_index);
  auto const offsets_itr =
    cudf::detail::input_offsetalator(offsets.head(), offsets.type(), d_strings.offset());
  auto const offset     = offsets_itr[str_idx];
  auto const size_bytes = static_cast<cudf::size_type>(offsets_itr[str_idx + 1] - offset);
  if (size_bytes == 0) { return; }

  auto const d_str    = cudf::string_view(d_strings.head<char>() + offset, size_bytes);
  auto const lane_idx = tid % tile_size;

  // hashes for this string/thread are stored here
  auto seed_hashes = d_hashes + offset - offsets_itr[0] + lane_idx;

  auto const begin  = d_str.data() + lane_idx;
  auto const end    = d_str.data() + d_str.size_bytes();
  auto const hasher = HashFunction(seed);

  for (auto itr = begin; itr < end; itr += tile_size, seed_hashes += tile_size) {
    if (cudf::strings::detail::is_utf8_continuation_char(*itr)) {
      *seed_hashes = 0;
      continue;
    }
    auto const check_str =  // used for counting 'width' characters
      cudf::string_view(itr, static_cast<cudf::size_type>(cuda::std::distance(itr, end)));
    auto const [bytes, left] = cudf::strings::detail::bytes_to_character_position(check_str, width);
    if ((itr != d_str.data()) && (left > 0)) {
      // true itr+width is past the end of the string
      *seed_hashes = 0;
      continue;
    }

    auto const hash_str = cudf::string_view(itr, bytes);
    hash_value_type hv;
    if constexpr (std::is_same_v<hash_value_type, uint32_t>) {
      hv = hasher(hash_str);
    } else {
      hv = cuda::std::get<0>(hasher(hash_str));
    }
    // disallowing hash to zero case
    *seed_hashes = cuda::std::max(hv, hash_value_type{1});
  }

  // logic appended here so an extra kernel is not required
  if (size_bytes >= wide_row_threshold) {
    if (lane_idx == 0) {
      // count the number of wide strings
      cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{*threshold_count};
      ref.fetch_add(1, cuda::std::memory_order_relaxed);
    }
    // initialize the output -- only needed for wider strings
    auto d_output = d_results + (str_idx * param_count);
    for (auto i = lane_idx; i < param_count; i += tile_size) {
      d_output[i] = cuda::std::numeric_limits<hash_value_type>::max();
    }
  }
}

/**
 * @brief Hashing kernel launched as a thread per tile-size (block or warp)
 * for a lists column
 *
 * This kernel computes the hashes for each row using the seed and the specified
 * hash function. The ngrams identifies consecutive strings to hash over in
 * sliding window formation. The hashes are stored in d_hashes and used as input
 * to the minhash_kernel.
 *
 * This kernel also counts the number of rows above the wide_row_threshold
 * and proactively initializes the output values for those rows.
 *
 * @tparam HashFunction The hash function to use for this kernel
 * @tparam hash_value_type Derived from HashFunction result_type
 *
 * @param d_input The input column to hash
 * @param seed The seed used for the hash function
 * @param ngrams Number of strings in each row to hash
 * @param d_hashes The resulting hash values are stored here
 * @param threshold_count Stores the number of rows above wide_row_threshold
 * @param param_count Number of parameters (used for the proactive initialize)
 * @param d_results Final results vector (used for the proactive initialize)
 */
template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
CUDF_KERNEL void minhash_ngrams_kernel(cudf::lists_column_device_view const d_input,
                                       hash_value_type seed,
                                       cudf::size_type ngrams,
                                       hash_value_type* d_hashes,
                                       cudf::size_type* threshold_count,
                                       cudf::size_type param_count,
                                       hash_value_type* d_results)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tid / tile_size;
  if (row_idx >= d_input.size()) { return; }
  if (d_input.is_null(row_idx)) { return; }

  // retrieve this row's offset to locate the output position in d_hashes
  auto const offsets_itr = d_input.offsets().data<cudf::size_type>() + d_input.offset();
  auto const offset      = offsets_itr[row_idx];
  auto const size_row    = offsets_itr[row_idx + 1] - offset;
  if (size_row == 0) { return; }

  auto const d_row    = cudf::list_device_view(d_input, row_idx);
  auto const lane_idx = static_cast<cudf::size_type>(tid % tile_size);

  // hashes for this row/thread are stored here
  auto seed_hashes  = d_hashes + offset - offsets_itr[0] + lane_idx;
  auto const hasher = HashFunction(seed);

  for (auto idx = lane_idx; idx < size_row; idx += tile_size, seed_hashes += tile_size) {
    if (d_row.is_null(idx)) {
      *seed_hashes = 0;
      continue;
    }

    auto next_idx = cuda::std::min(idx + ngrams, size_row - 1);
    if ((idx != 0) && ((next_idx - idx) < ngrams)) {
      *seed_hashes = 0;
      continue;
    }

    auto const first_str = d_row.element<cudf::string_view>(idx);
    auto const last_str  = d_row.element<cudf::string_view>(next_idx);
    // build super-string since adjacent strings are contiguous in memory
    auto const size = static_cast<cudf::size_type>(
      cuda::std::distance(first_str.data(), last_str.data()) + last_str.size_bytes());
    auto const hash_str = cudf::string_view(first_str.data(), size);
    hash_value_type hv;
    if constexpr (std::is_same_v<hash_value_type, uint32_t>) {
      hv = hasher(hash_str);
    } else {
      hv = cuda::std::get<0>(hasher(hash_str));
    }
    // disallowing hash to zero case
    *seed_hashes = cuda::std::max(hv, hash_value_type{1});
  }

  // logic appended here to count long rows so an extra kernel is not required
  if (size_row >= wide_row_threshold) {
    if (lane_idx == 0) {
      // count the number of wide rows
      cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{*threshold_count};
      ref.fetch_add(1, cuda::std::memory_order_relaxed);
    }
    // initialize the output -- only needed for wider rows
    auto d_output = d_results + (row_idx * param_count);
    for (auto i = lane_idx; i < param_count; i += tile_size) {
      d_output[i] = cuda::std::numeric_limits<hash_value_type>::max();
    }
  }
}

/**
 * @brief Permutation calculation kernel
 *
 * This kernel uses the hashes from the minhash_seed_kernel or minhash_ngrams_kernel
 * and the 'parameter_a' and 'parameter_b' values to compute the final output.
 * The output is the number of input rows (N) by the number of parameter values (M).
 * Each row output[i] is the calculated result for parameter_a/b[0:M].
 *
 * This kernel is launched with either blocks per row of 1 for rows
 * below the wide_row_threshold or blocks per row = blocks_per_rows
 * for rows above wide_row_threshold.
 *
 * Note that this was refactored to accommodate lists of strings which is possible
 * since there is no need here to access the characters, only the hash values.
 * The offsets and width are used to locate and count the hash values produced by
 * kernels above for each input row.
 *
 * @tparam offsets_type Type for the offsets iterator for the input column
 * @tparam hash_value_type Derived from HashFunction result_type
 * @tparam blocks_per_row Number of blocks used to process each row
 *
 * @param offsets_itr The offsets are used to address the d_hashes
 * @param indices The indices of the rows in the input column
 * @param parameter_a 1st set of parameters for the calculation result
 * @param parameter_b 2nd set of parameters for the calculation result
 * @param width Used for calculating the number of available hashes in each row
 * @param d_hashes The hash values computed in one of the hash kernels
 * @param d_results Final results vector of calculate values
 */
template <typename offsets_type, typename hash_value_type, int blocks_per_row>
CUDF_KERNEL void minhash_kernel(offsets_type offsets_itr,
                                cudf::device_span<cudf::size_type const> indices,
                                cudf::device_span<hash_value_type const> parameter_a,
                                cudf::device_span<hash_value_type const> parameter_b,
                                cudf::size_type width,
                                hash_value_type const* d_hashes,
                                hash_value_type* d_results)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  auto const idx = (tid / blocks_per_row) / block_size;
  if (idx >= indices.size()) { return; }
  auto const row_idx = indices[idx];

  auto const block      = cooperative_groups::this_thread_block();
  int const section_idx = block.group_index().x % blocks_per_row;

  auto const offset   = offsets_itr[row_idx];
  auto const row_size = static_cast<cudf::size_type>(offsets_itr[row_idx + 1] - offset);

  // number of items to process in this block;
  // last block also includes any remainder values from the row_size/blocks_per_row truncation
  // example:
  //  each section_size for string with size 588090 and blocks_per_row=64 is 9188
  //  except the last section which is 9188 + (588090 % 64) = 9246
  auto const section_size = (row_size / blocks_per_row) +
                            (section_idx < (blocks_per_row - 1) ? 0 : row_size % blocks_per_row);
  auto const section_offset = section_idx * (row_size / blocks_per_row);

  // hash values for this block/section
  auto const seed_hashes = d_hashes + offset - offsets_itr[0] + section_offset;
  // width used here as a max value since a string's char-count <= byte-count
  auto const hashes_size =
    section_idx < (blocks_per_row - 1)
      ? section_size
      : cuda::std::max(static_cast<cudf::size_type>(row_size > 0), section_size - width + 1);

  auto const init     = row_size == 0 ? 0 : cuda::std::numeric_limits<hash_value_type>::max();
  auto const lane_idx = block.thread_rank();
  auto const d_output = d_results + (row_idx * parameter_a.size());

  auto const begin = seed_hashes + lane_idx;
  auto const end   = seed_hashes + hashes_size;

  // constants used in the permutation calculations
  constexpr uint64_t mersenne_prime  = (1UL << 61) - 1;
  constexpr hash_value_type hash_max = cuda::std::numeric_limits<hash_value_type>::max();

  // found to be an efficient shared memory size for both hash types
  __shared__ hash_value_type block_values[block_size * params_per_thread];

  for (std::size_t i = 0; i < parameter_a.size(); i += params_per_thread) {
    // initialize this block's chunk of shared memory
    // each thread handles params_per_thread of values
    auto const chunk_values = block_values + (lane_idx * params_per_thread);
    thrust::uninitialized_fill(thrust::seq, chunk_values, chunk_values + params_per_thread, init);
    block.sync();

    auto const param_count =
      cuda::std::min(static_cast<cuda::std::size_t>(params_per_thread), parameter_a.size() - i);

    // each lane accumulates min hashes in its shared memory
    for (auto itr = begin; itr < end; itr += block_size) {
      auto const hv = *itr;
      // 0 is used as a skip sentinel for UTF-8 and trailing bytes
      if (hv == 0) { continue; }

      for (std::size_t param_idx = i; param_idx < (i + param_count); ++param_idx) {
        // permutation formula used by datatrove
        hash_value_type const v =
          ((hv * parameter_a[param_idx] + parameter_b[param_idx]) % mersenne_prime) & hash_max;
        auto const block_idx    = ((param_idx % params_per_thread) * block_size) + lane_idx;
        block_values[block_idx] = cuda::std::min(v, block_values[block_idx]);
      }
    }
    block.sync();

    // reduce each parameter values vector to a single min value;
    // assumes that the block_size > params_per_thread;
    // each thread reduces a block_size of parameter values (thread per parameter)
    if (lane_idx < param_count) {
      auto const values = block_values + (lane_idx * block_size);
      // cooperative groups does not have a min function and cub::BlockReduce was slower
      auto const minv =
        thrust::reduce(thrust::seq, values, values + block_size, init, cuda::minimum{});
      if constexpr (blocks_per_row > 1) {
        // accumulates mins for each block into d_output
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_block> ref{d_output[lane_idx + i]};
        ref.fetch_min(minv, cuda::std::memory_order_relaxed);
      } else {
        d_output[lane_idx + i] = minv;
      }
    }
    block.sync();
  }
}

/**
 * @brief Partition input rows by row size
 *
 * The returned index is the first row above the wide_row_threshold size.
 * The returned vector are the indices partitioned above and below the
 * wide_row_threshold size.
 *
 * @param size Number of rows in the input column
 * @param threshold_count Number of rows above wide_row_threshold
 * @param tfn Transform function returns the size of each row
 * @param stream Stream used for allocation and kernel launches
 */
template <typename transform_fn>
std::pair<cudf::size_type, rmm::device_uvector<cudf::size_type>> partition_input(
  cudf::size_type size,
  cudf::size_type threshold_count,
  transform_fn tfn,
  rmm::cuda_stream_view stream)
{
  auto indices = rmm::device_uvector<cudf::size_type>(size, stream);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   indices.begin(),
                   indices.end());
  cudf::size_type threshold_index = threshold_count < size ? size : 0;

  // if we counted a split of above/below threshold then
  // compute partitions based on the size of each string
  if ((threshold_count > 0) && (threshold_count < size)) {
    auto sizes = rmm::device_uvector<cudf::size_type>(size, stream);
    auto begin = cuda::counting_iterator<cudf::size_type>{0};
    auto end   = begin + size;
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      begin,
                      end,
                      sizes.data(),
                      tfn);
    // these 2 are slightly faster than using partition()
    thrust::sort_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        sizes.begin(),
                        sizes.end(),
                        indices.begin());
    auto const lb =
      thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          sizes.begin(),
                          sizes.end(),
                          wide_row_threshold);
    threshold_index = static_cast<cudf::size_type>(cuda::std::distance(sizes.begin(), lb));
  }
  return {threshold_index, std::move(indices)};
}

template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
std::unique_ptr<cudf::column> minhash_fn(cudf::strings_column_view const& input,
                                         hash_value_type seed,
                                         cudf::device_span<hash_value_type const> parameter_a,
                                         cudf::device_span<hash_value_type const> parameter_b,
                                         cudf::size_type width,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS(!parameter_a.empty(), "Parameters A and B cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(parameter_a.size() == parameter_b.size(),
               "Parameters A and B should have the same number of elements",
               std::invalid_argument);
  CUDF_EXPECTS(
    (static_cast<std::size_t>(input.size()) * parameter_a.size()) <
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "The number of parameters times the number of input rows exceeds the column size limit",
    std::overflow_error);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto results =
    cudf::make_numeric_column(output_type,
                              input.size() * static_cast<cudf::size_type>(parameter_a.size()),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto d_results = results->mutable_view().data<hash_value_type>();

  cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()) * block_size,
                             block_size};
  auto const hashes_size = input.chars_size(stream);
  auto d_hashes          = rmm::device_uvector<hash_value_type>(hashes_size, stream);
  auto d_threshold_count = cudf::detail::device_scalar<cudf::size_type>(
    0, stream, cudf::get_current_device_resource_ref());

  minhash_seed_kernel<HashFunction>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(*d_strings,
                                                                         seed,
                                                                         width,
                                                                         d_hashes.data(),
                                                                         d_threshold_count.data(),
                                                                         parameter_a.size(),
                                                                         d_results);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto transform_fn = [d_strings = *d_strings] __device__(auto idx) -> cudf::size_type {
    if (d_strings.is_null(idx)) { return 0; }
    return d_strings.element<cudf::string_view>(idx).size_bytes();
  };
  auto [threshold_index, indices] =
    partition_input(input.size(), d_threshold_count.value(stream), transform_fn, stream);

  auto input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  using offsets_type = decltype(input_offsets);

  // handle the strings below the threshold width
  if (threshold_index > 0) {
    auto d_indices = cudf::device_span<cudf::size_type const>(indices.data(), threshold_index);
    cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(d_indices.size()) * block_size,
                               block_size};
    minhash_kernel<offsets_type, hash_value_type, 1>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        input_offsets, d_indices, parameter_a, parameter_b, width, d_hashes.data(), d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // handle the strings above the threshold width
  if (threshold_index < input.size()) {
    auto const count = static_cast<cudf::thread_index_type>(input.size() - threshold_index);
    auto d_indices =
      cudf::device_span<cudf::size_type const>(indices.data() + threshold_index, count);
    cudf::detail::grid_1d grid{count * block_size * blocks_per_row, block_size};
    minhash_kernel<offsets_type, hash_value_type, blocks_per_row>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        input_offsets, d_indices, parameter_a, parameter_b, width, d_hashes.data(), d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return results;
}

template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
std::unique_ptr<cudf::column> minhash_ngrams_fn(
  cudf::lists_column_view const& input,
  cudf::size_type ngrams,
  hash_value_type seed,
  cudf::device_span<hash_value_type const> parameter_a,
  cudf::device_span<hash_value_type const> parameter_b,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(ngrams >= 2,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS(!parameter_a.empty(), "Parameters A and B cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(parameter_a.size() == parameter_b.size(),
               "Parameters A and B should have the same number of elements",
               std::invalid_argument);
  CUDF_EXPECTS(
    (static_cast<std::size_t>(input.size()) * parameter_a.size()) <
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "The number of parameters times the number of input rows exceeds the column size limit",
    std::overflow_error);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_input = cudf::column_device_view::create(input.parent(), stream);

  auto results =
    cudf::make_numeric_column(output_type,
                              input.size() * static_cast<cudf::size_type>(parameter_a.size()),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto d_results = results->mutable_view().data<hash_value_type>();

  cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()) * block_size,
                             block_size};
  auto const hashes_size = input.child().size();
  auto d_hashes          = rmm::device_uvector<hash_value_type>(hashes_size, stream);
  auto d_threshold_count = cudf::detail::device_scalar<cudf::size_type>(
    0, stream, cudf::get_current_device_resource_ref());

  auto d_list = cudf::lists_column_device_view(*d_input);
  minhash_ngrams_kernel<HashFunction>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(d_list,
                                                                         seed,
                                                                         ngrams,
                                                                         d_hashes.data(),
                                                                         d_threshold_count.data(),
                                                                         parameter_a.size(),
                                                                         d_results);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto sizes_fn = [d_list] __device__(auto idx) -> cudf::size_type {
    if (d_list.is_null(idx)) { return 0; }
    return cudf::list_device_view(d_list, idx).size();
  };
  auto [threshold_index, indices] =
    partition_input(input.size(), d_threshold_count.value(stream), sizes_fn, stream);

  auto input_offsets = input.offsets_begin();  // already includes input.offset()
  using offset_type  = decltype(input_offsets);

  // handle the strings below the threshold width
  if (threshold_index > 0) {
    auto d_indices = cudf::device_span<cudf::size_type const>(indices.data(), threshold_index);
    cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(d_indices.size()) * block_size,
                               block_size};
    minhash_kernel<offset_type, hash_value_type, 1>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        input_offsets, d_indices, parameter_a, parameter_b, ngrams, d_hashes.data(), d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // handle the strings above the threshold width
  if (threshold_index < input.size()) {
    auto const count = static_cast<cudf::thread_index_type>(input.size() - threshold_index);
    auto d_indices =
      cudf::device_span<cudf::size_type const>(indices.data() + threshold_index, count);
    cudf::detail::grid_1d grid{count * block_size * blocks_per_row, block_size};
    minhash_kernel<offset_type, hash_value_type, blocks_per_row>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        input_offsets, d_indices, parameter_a, parameter_b, ngrams, d_hashes.data(), d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return results;
}

std::unique_ptr<cudf::column> build_list_result(cudf::column_view const& input,
                                                std::unique_ptr<cudf::column>&& hashes,
                                                cudf::size_type seeds_size,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  // build the offsets for the output lists column
  auto const zero =
    cudf::numeric_scalar<cudf::size_type>(0, true, stream, cudf::get_current_device_resource_ref());
  auto const size = cudf::numeric_scalar<cudf::size_type>(
    seeds_size, true, stream, cudf::get_current_device_resource_ref());
  auto offsets = cudf::detail::sequence(input.size() + 1, zero, size, stream, mr);
  hashes->set_null_mask(rmm::device_buffer{}, 0);  // children have no nulls

  // build the lists column from the offsets and the hashes
  auto result = make_lists_column(input.size(),
                                  std::move(offsets),
                                  std::move(hashes),
                                  input.null_count(),
                                  cudf::detail::copy_bitmask(input, stream, mr));
  // expect this condition to be very rare
  if (input.null_count() > 0) {
    result = cudf::detail::purge_nonempty_nulls(result->view(), stream, mr);
  }
  return result;
}
}  // namespace

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      uint32_t seed,
                                      cudf::device_span<uint32_t const> parameter_a,
                                      cudf::device_span<uint32_t const> parameter_b,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes =
    detail::minhash_fn<HashFunction>(input, seed, parameter_a, parameter_b, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash_ngrams(cudf::lists_column_view const& input,
                                             cudf::size_type ngrams,
                                             uint32_t seed,
                                             cudf::device_span<uint32_t const> parameter_a,
                                             cudf::device_span<uint32_t const> parameter_b,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes        = detail::minhash_ngrams_fn<HashFunction>(
    input, ngrams, seed, parameter_a, parameter_b, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        uint64_t seed,
                                        cudf::device_span<uint64_t const> parameter_a,
                                        cudf::device_span<uint64_t const> parameter_b,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes =
    detail::minhash_fn<HashFunction>(input, seed, parameter_a, parameter_b, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash64_ngrams(cudf::lists_column_view const& input,
                                               cudf::size_type ngrams,
                                               uint64_t seed,
                                               cudf::device_span<uint64_t const> parameter_a,
                                               cudf::device_span<uint64_t const> parameter_b,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes        = detail::minhash_ngrams_fn<HashFunction>(
    input, ngrams, seed, parameter_a, parameter_b, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      uint32_t seed,
                                      cudf::device_span<uint32_t const> parameter_a,
                                      cudf::device_span<uint32_t const> parameter_b,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, seed, parameter_a, parameter_b, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash_ngrams(cudf::lists_column_view const& input,
                                             cudf::size_type ngrams,
                                             uint32_t seed,
                                             cudf::device_span<uint32_t const> parameter_a,
                                             cudf::device_span<uint32_t const> parameter_b,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)

{
  CUDF_FUNC_RANGE();
  return detail::minhash_ngrams(input, ngrams, seed, parameter_a, parameter_b, stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        uint64_t seed,
                                        cudf::device_span<uint64_t const> parameter_a,
                                        cudf::device_span<uint64_t const> parameter_b,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash64(input, seed, parameter_a, parameter_b, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash64_ngrams(cudf::lists_column_view const& input,
                                               cudf::size_type ngrams,
                                               uint64_t seed,
                                               cudf::device_span<uint64_t const> parameter_a,
                                               cudf::device_span<uint64_t const> parameter_b,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)

{
  CUDF_FUNC_RANGE();
  return detail::minhash64_ngrams(input, ngrams, seed, parameter_a, parameter_b, stream, mr);
}

}  // namespace nvtext
