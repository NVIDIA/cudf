/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <vector>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * a warp-parallel function is used.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Kernel for finding multiple targets in each row of input strings
 *
 * The d_first_bytes is sorted and unique so the d_indices and d_offsets
 * are used to map the corresponding character to its d_targets entry.
 *
 * Example
 * d_targets = ["foo", "hello", "world", "hi"]
 *  - sorted first-chars: ['f','h','h','w']
 * d_indices = [0, 3, 1, 2]
 * d_first_bytes = ['f', 'h', 'w']   (unique)
 * d_offsets = [0, 1, 3]
 * unique_count = 3
 *
 * If 'h' is found, lower_bound produces pos=1 in d_first_bytes.
 * This corresponds to d_offset[1]==1 which has two values:
 * - (d_offsets[2] - d_offsets[1]) = (3 - 1) = 2.
 * Set map_idx = d_offsets[1] = 1 and the two targets to check are sequential
 * in the d_indices array:
 * - tgt1_idx = d_indices[map_idx]   = 3 --> d_targets[3] == 'hi'
 * - tgt2_idx = d_indices[map_idx+1] = 1 --> d_targets[1] == 'hello'
 * The logic now only needs to check for either of these 2 targets.
 *
 * This kernel works in either thread-per-string or warp-per-string depending
 * on the template parameter. If tile_size==1, then this kernel executes as
 * a row-per-string. If tile_size=32, the it executes as a warp-per-string.
 * No other options are supported for now.
 *
 * @tparam tile_size Number of threads per string
 * @param d_strings Input strings
 * @param d_targets Target strings to search within input strings
 * @param d_first_bytes Sorted, unique list of first bytes of the target strings
 * @param d_indices Indices to map sorted d_first_bytes to d_targets
 * @param d_offsets Offsets to map d_indices to d_targets
 * @param unique_count Number of unique values in d_first_bytes (and d_offsets)
 * @param working_memory Global memory to use if shared-memory is too small
 * @param d_results Bool results for each target within each string row
 */
template <cudf::thread_index_type tile_size>
CUDF_KERNEL void multi_contains_kernel(column_device_view const d_strings,
                                       column_device_view const d_targets,
                                       u_char const* d_first_bytes,
                                       size_type const* d_indices,
                                       size_type const* d_offsets,
                                       size_type unique_count,
                                       bool* working_memory,
                                       cudf::device_span<bool*> d_results)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }

  // get the string for this tile
  auto const d_str = d_strings.element<string_view>(str_idx);

  namespace cg           = cooperative_groups;
  auto const tile        = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto const lane_idx    = tile.thread_rank();
  auto const num_targets = d_targets.size();

  // size of shared_bools = num_targets * block_size
  // each thread uses num_targets bools
  extern __shared__ bool shared_bools[];
  // bools for the current string
  auto bools = working_memory == nullptr
                 ? (shared_bools + (tile.meta_group_rank() * tile_size * num_targets))
                 : (working_memory + (str_idx * tile_size * num_targets));

  // initialize result: set true if target is empty, false otherwise
  for (auto target_idx = lane_idx; target_idx < num_targets; target_idx += tile_size) {
    auto const d_target = d_targets.element<string_view>(target_idx);
    if constexpr (tile_size == 1) {
      d_results[target_idx][str_idx] = d_target.empty();
    } else {
      auto const begin = bools + (target_idx * tile_size);
      thrust::uninitialized_fill(thrust::seq, begin, begin + tile_size, d_target.empty());
    }
  }
  tile.sync();

  auto const last_ptr = d_first_bytes + unique_count;
  for (size_type str_byte_idx = lane_idx; str_byte_idx < d_str.size_bytes();
       str_byte_idx += tile_size) {
    // search for byte in first_bytes array
    auto const sptr     = d_str.data() + str_byte_idx;
    auto const chr      = static_cast<u_char>(*sptr);
    auto const byte_ptr = thrust::lower_bound(thrust::seq, d_first_bytes, last_ptr, chr);
    // if not found, continue to next byte
    if ((byte_ptr == last_ptr) || (*byte_ptr != chr)) { continue; }
    // compute index of matched byte
    auto const offset_idx = static_cast<size_type>(cuda::std::distance(d_first_bytes, byte_ptr));
    auto map_idx          = d_offsets[offset_idx];
    auto const last_idx = (offset_idx + 1) < unique_count ? d_offsets[offset_idx + 1] : num_targets;
    // check for targets that begin with chr
    while (map_idx < last_idx) {
      auto const target_idx = d_indices[map_idx++];
      auto const bool_idx   = (target_idx * tile_size) + lane_idx;
      auto const found      = tile_size == 1 ? d_results[target_idx][str_idx] : bools[bool_idx];
      if (!found) {  // not found before
        auto const d_target = d_targets.element<string_view>(target_idx);
        if ((d_str.size_bytes() - str_byte_idx) >= d_target.size_bytes()) {
          // first char already checked, so just check the [1, end) chars match
          auto const tp = d_target.data();
          if (thrust::equal(thrust::seq, tp + 1, tp + d_target.size_bytes(), sptr + 1)) {
            if constexpr (tile_size == 1) {
              d_results[target_idx][str_idx] = true;
            } else {
              bools[bool_idx] = true;
            }
          }
        }
      }
    }
  }

  if constexpr (tile_size > 1) {
    tile.sync();
    // reduce the bools for each target to store in the result
    for (auto target_idx = lane_idx; target_idx < num_targets; target_idx += tile_size) {
      auto const begin = bools + (target_idx * tile_size);
      d_results[target_idx][str_idx] =
        thrust::any_of(thrust::seq, begin, begin + tile_size, cuda::std::identity{});
      // cooperative_group any() implementation was almost 3x slower than this parallel reduce
    }
  }
}
}  // namespace

std::unique_ptr<table> contains_multiple(strings_column_view const& input,
                                         strings_column_view const& targets,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    not targets.is_empty(), "Must specify at least one target string.", std::invalid_argument);
  CUDF_EXPECTS(not targets.has_nulls(), "Target strings cannot be null", std::invalid_argument);

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_targets = column_device_view::create(targets.parent(), stream);

  // copy the first byte of each target and sort them
  auto first_bytes = rmm::device_uvector<u_char>(targets.size(), stream);
  auto indices     = rmm::device_uvector<size_type>(targets.size(), stream);
  {
    auto tgt_itr = thrust::make_transform_iterator(
      d_targets->begin<string_view>(),
      cuda::proclaim_return_type<u_char>([] __device__(auto const& d_tgt) -> u_char {
        return d_tgt.empty() ? u_char{0} : static_cast<u_char>(d_tgt.data()[0]);
      }));
    auto count_itr = cuda::counting_iterator<size_type>{0};
    auto keys_out  = first_bytes.begin();
    auto vals_out  = indices.begin();
    auto num_items = targets.size();
    auto cmp_op    = cuda::std::less();
    auto sv        = stream.value();

    std::size_t tmp_bytes = 0;
    cub::DeviceMergeSort::SortPairsCopy(
      nullptr, tmp_bytes, tgt_itr, count_itr, keys_out, vals_out, num_items, cmp_op, sv);
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    cub::DeviceMergeSort::SortPairsCopy(
      tmp_stg.data(), tmp_bytes, tgt_itr, count_itr, keys_out, vals_out, num_items, cmp_op, sv);
  }

  // remove duplicates to help speed up lower_bound
  auto offsets = rmm::device_uvector<size_type>(targets.size(), stream);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   offsets.begin(),
                   offsets.end());
  auto const end =
    thrust::unique_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          first_bytes.begin(),
                          first_bytes.end(),
                          offsets.begin());
  auto const unique_count =
    static_cast<size_type>(cuda::std::distance(first_bytes.begin(), end.first));

  // create output columns
  auto const results_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    return make_numeric_column(data_type{type_id::BOOL8},
                               input.size(),
                               cudf::detail::copy_bitmask(input.parent(), stream, mr),
                               input.null_count(),
                               stream,
                               mr);
  });
  auto results = std::vector<std::unique_ptr<column>>(results_iter, results_iter + targets.size());
  auto d_results = [&] {
    auto host_results_pointer_iter =
      thrust::make_transform_iterator(results.begin(), [](auto const& results_column) {
        return results_column->mutable_view().template data<bool>();
      });
    auto host_results_pointers =
      std::vector<bool*>(host_results_pointer_iter, host_results_pointer_iter + results.size());
    return cudf::detail::make_device_uvector(host_results_pointers, stream, mr);
  }();

  constexpr cudf::thread_index_type block_size = 256;
  // calculated (benchmarked) for efficient use of shared-memory
  constexpr size_type targets_threshold = 32;

  auto d_first_bytes = first_bytes.data();
  auto d_indices     = indices.data();
  auto d_offsets     = offsets.data();

  bool const row_parallel = ((input.null_count() == input.size()) ||
                             ((input.chars_size(stream) / (input.size() - input.null_count())) <=
                              AVG_CHAR_BYTES_THRESHOLD));

  if (row_parallel) {
    // Smaller strings perform better with a row per string
    cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()), block_size};
    multi_contains_kernel<1>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(*d_strings,
                                                                           *d_targets,
                                                                           d_first_bytes,
                                                                           d_indices,
                                                                           d_offsets,
                                                                           unique_count,
                                                                           nullptr,
                                                                           d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    constexpr cudf::thread_index_type tile_size = cudf::detail::warp_size;

    auto const shared_mem_size =
      (targets.size() <= targets_threshold) ? (block_size * targets.size()) : 0;
    auto const work_mem_size =
      (targets.size() <= targets_threshold) ? 0 : tile_size * targets.size() * input.size();
    auto working_memory = rmm::device_uvector<bool>(work_mem_size, stream);

    cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()) * tile_size,
                               block_size};
    multi_contains_kernel<tile_size>
      <<<grid.num_blocks, grid.num_threads_per_block, shared_mem_size, stream.value()>>>(
        *d_strings,
        *d_targets,
        d_first_bytes,
        d_indices,
        d_offsets,
        unique_count,
        working_memory.data(),
        d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return std::make_unique<table>(std::move(results));
}

}  // namespace detail

std::unique_ptr<table> contains_multiple(strings_column_view const& strings,
                                         strings_column_view const& targets,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_multiple(strings, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
