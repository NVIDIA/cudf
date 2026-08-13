/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/edit_distance.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

constexpr cudf::size_type row_pad_size = 2;  // each row has potentially 2 extra values

struct calculate_compute_buffer_fn {
  cudf::column_device_view d_strings;
  cudf::column_device_view d_targets;
  int32_t pad{row_pad_size};
  int32_t count{3};  // number of concurrent rows for each iteration

  __device__ std::ptrdiff_t operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    if ((d_targets.size() > 1 && d_targets.is_null(idx)) ||
        (d_targets.size() == 1 && d_targets.is_null(0))) {
      return 0;
    }
    auto d_str = d_strings.element<cudf::string_view>(idx);
    auto d_tgt = d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                       : d_targets.element<cudf::string_view>(idx);
    return (cuda::std::min(d_str.length(), d_tgt.length()) + static_cast<int64_t>(pad)) *
           static_cast<int64_t>(count);
  }
};

/**
 * @brief Processes the 2 given strings by computing the matrix values
 * comparing each character of both strings
 *
 * The full matrix is computed from top-left to bottom-right with the edit-distance
 * as the final value. Only 3 diagonals are needed in memory at any iteration.
 *
 *     -  S  E  C  O  N  D
 *  -  0  1  2  3  4  5  6
 *  F  1  =  x  y  z  ↘  .
 *  I  2  x  y  z  ↘  .  .
 *  R  3  y  z  ↘  .  .  .
 *  S  4  z  ↘  .  .  .  .
 *  T  5  →  .  .  .  .  X
 *
 * Computing a diagonal only requires the previous 2 and each value
 * can be computed independently/parallel a tile at a time.
 *
 * The diagonal approach inspired by:
 * https://ashvardanian.com/posts/stringwars-on-gpus/#dynamic-programming-and-levenshtein-evaluation-order
 */
template <typename Iterator, int32_t tile_size = cudf::detail::warp_size>
__device__ void compute_distance(Iterator input1,
                                 cudf::size_type length1,
                                 Iterator input2,
                                 cudf::size_type length2,
                                 cudf::size_type* d_buffer,
                                 cudf::size_type* d_result)
{
  namespace cg        = cooperative_groups;
  auto const block    = cg::this_thread_block();
  auto const tile     = cg::tiled_partition<tile_size>(block);
  auto const lane_idx = tile.thread_rank();

  // shortcut if one of the strings is empty
  // (null rows are mapped here as well)
  if (length1 == 0 || length2 == 0) {
    if (lane_idx == 0) { *d_result = length1 == 0 ? length2 : length1; }
    return;
  }

  // setup the 3 working vectors for this string
  auto v0 = d_buffer;
  auto v1 = v0 + length1 + row_pad_size;
  auto v2 = v1 + length1 + row_pad_size;
  if (lane_idx == 0) {
    v0[0] = 0;  // first diagonal
    v1[0] = 1;  // second diagonal
    v1[1] = 1;
  }

  // utility for navigating the diagonal of the matrix of characters for the 2 strings
  auto next_itr = [](Iterator sitr, cudf::size_type length, Iterator itr, cudf::size_type offset) {
    if constexpr (cuda::std::is_pointer_v<Iterator>) {
      itr = sitr - offset;  // ASCII iterator
    } else {
      auto const pos = sitr.position() - offset;  // minimizes character counting
      itr += (pos >= 0) && (pos < length) ? (pos - itr.position()) : 0;
    }
    return itr;
  };

  // top-left of the matrix
  // includes the diagonal one after the max(length1,length2) diagonal
  for (auto idx = 0; idx < length1; ++idx, ++input1) {
    auto const n = idx + row_pad_size;  // diagonal length
    auto const a = n > length1;         // extra diagonal adjust

    auto jdx = static_cast<cudf::size_type>(lane_idx);
    auto it1 = input1;
    auto it2 = input2;

    auto tile_count = cudf::util::div_rounding_up_safe(n + 1, tile_size);
    while (tile_count--) {
      auto const offset = (jdx - 1);
      // locate the 2 characters to compare along the diagonal
      it1 = next_itr(input1, length1, it1, offset);
      it2 = next_itr(input2, length2, it2, -offset);
      if (jdx == 0) {
        if (!a) { v2[0] = n; }
      } else if (jdx < n) {
        auto const sc = v0[jdx - 1] + (*it1 != *it2);
        auto const dc = v1[jdx - 1] + 1;
        auto const ic = v1[jdx] + 1;
        v2[jdx - a]   = cuda::std::min(cuda::std::min(sc, dc), ic);
      } else if (jdx == n) {
        v2[n - a] = n;
      }
      tile.sync();
      jdx += tile_size;
    }
    cuda::std::swap(v0, v1);
    cuda::std::swap(v1, v2);
  }

  --input1;  // reset
  ++input2;  // iterators

  // bottom-right of the matrix
  for (auto idx = 1; idx < length2; ++idx, ++input2) {
    bool const fl = (length2 - idx) > length1;  // fill-last flag
    auto const n  = (fl ? length1 : (length2 - idx)) + 1;

    auto jdx = static_cast<cudf::size_type>(lane_idx);
    auto it1 = input1;
    auto it2 = input2;

    auto tile_count = cudf::util::div_rounding_up_safe(n, tile_size);
    while (tile_count--) {
      auto const offset = (jdx - 1);
      // locate the 2 characters to compare along the diagonal
      it1 = next_itr(input1, length1, it1, offset);
      it2 = next_itr(input2, length2, it2, -offset);
      if (jdx > 0 && jdx < n) {
        auto const sc = v0[jdx] + (*it1 != *it2);
        auto const dc = v1[jdx - 1] + 1;
        auto const ic = v1[jdx] + 1;
        v2[jdx - 1]   = cuda::std::min(cuda::std::min(sc, dc), ic);
      } else if (jdx == n && fl) {
        v2[n - 1] = n + idx;
      }
      tile.sync();
      jdx += tile_size;
    }
    cuda::std::swap(v0, v1);
    cuda::std::swap(v1, v2);
  }

  if (lane_idx == 0) { *d_result = v1[0]; }
}

template <int32_t tile_size = cudf::detail::warp_size>
CUDF_KERNEL void levenshtein_kernel(cudf::column_device_view d_strings,
                                    cudf::column_device_view d_targets,
                                    cudf::size_type* d_work_buffer,
                                    std::ptrdiff_t const* d_offsets,
                                    cudf::size_type* d_results)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  auto d_str1 = d_strings.is_null(str_idx) ? cudf::string_view{}
                                           : d_strings.element<cudf::string_view>(str_idx);
  auto d_str2 = [&] {  // d_targets is also allowed to have only one valid entry
    if ((d_targets.size() > 1 && d_targets.is_null(str_idx)) ||
        (d_targets.size() == 1 && d_targets.is_null(0))) {
      return cudf::string_view{};
    }
    return d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                 : d_targets.element<cudf::string_view>(str_idx);
  }();

  // compute_distance algorithm is designed such that it expects length1 <= length2
  if (d_str1.length() > d_str2.length()) { cuda::std::swap(d_str1, d_str2); }
  auto const length1 = d_str1.length();
  auto const length2 = d_str2.length();

  auto d_buffer = d_work_buffer + d_offsets[str_idx];
  auto d_result = d_results + str_idx;

  if (length1 == d_str1.size_bytes() && length2 == d_str2.size_bytes()) {
    // ASCII path
    compute_distance(d_str1.data(), length1, d_str2.data(), length2, d_buffer, d_result);
  } else {
    // use UTF8 iterator builtin to cudf::string_view
    compute_distance(d_str1.begin(), length1, d_str2.begin(), length2, d_buffer, d_result);
  }
}

}  // namespace

std::unique_ptr<cudf::column> edit_distance(cudf::strings_column_view const& input,
                                            cudf::strings_column_view const& targets,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }
  if (targets.size() != 1) {
    CUDF_EXPECTS(input.size() == targets.size(),
                 "targets.size() must equal input.size()",
                 std::invalid_argument);
  } else {
    CUDF_EXPECTS(!targets.has_nulls(), "single target must not be null", std::invalid_argument);
  }

  // create device columns from the input columns
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto d_targets = cudf::column_device_view::create(targets.parent(), stream);

  // calculate the size of the compute-buffer
  rmm::device_uvector<std::ptrdiff_t> offsets(input.size() + 1, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{input.size()},
                    offsets.begin(),
                    calculate_compute_buffer_fn{*d_strings, *d_targets});

  // get the total size of the temporary compute buffer
  // and convert sizes to offsets in-place
  auto const compute_size =
    cudf::detail::sizes_to_offsets(offsets.begin(), offsets.end(), offsets.begin(), 0, stream);
  rmm::device_uvector<cudf::size_type> compute_buffer(compute_size, stream);
  auto d_buffer = compute_buffer.data();

  auto results = cudf::make_fixed_width_column(
    output_type, input.size(), rmm::device_buffer{0, stream, mr}, 0, stream, mr);
  auto d_results = results->mutable_view().data<cudf::size_type>();

  constexpr auto block_size = 256L;
  constexpr auto tile_size  = static_cast<cudf::thread_index_type>(cudf::detail::warp_size);
  cudf::detail::grid_1d grid{input.size() * tile_size, block_size};
  levenshtein_kernel<tile_size><<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, *d_targets, d_buffer, offsets.data(), d_results);
  CUDF_CUDA_TRY(cudaGetLastError());

  return results;
}

}  // namespace detail

// external APIs

/**
 * @copydoc nvtext::edit_distance
 */
std::unique_ptr<cudf::column> edit_distance(cudf::strings_column_view const& input,
                                            cudf::strings_column_view const& targets,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::edit_distance(input, targets, stream, mr);
}

}  // namespace nvtext
