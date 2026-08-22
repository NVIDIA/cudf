/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace cudf {
namespace detail {

double checked_load_factor(double load_factor)
{
  CUDF_EXPECTS(load_factor > 0.0 && load_factor <= 1.0,
               "Invalid load factor: must be greater than 0 and less than or equal to 1.",
               std::invalid_argument);
  return load_factor;
}

VectorPair get_trivial_left_join_indices(table_view const& left,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   left_indices->begin(),
                   left_indices->end(),
                   0);
  auto right_indices =
    std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    right_indices->begin(),
    right_indices->end(),
    cudf::JoinNoMatch);
  return std::pair(std::move(left_indices), std::move(right_indices));
}

namespace {

// Predicate: right row `idx` is unmatched iff its flag slot is zero.
// We use an int32 flag (one per right row) rather than a packed bit or a byte: byte stores
// from a dense 32-wide scatter don't coalesce into full-word transactions, which costs 2–3×
// in the mark kernel for skewed left/right ratios.
struct unmatched_flag {
  size_type const* flags;
  __device__ bool operator()(size_type idx) const noexcept { return flags[idx] == 0; }
};

// Transform a selected (unmatched) right index into a (JoinNoMatch, idx) pair that is stored
// through a zip iterator over (left_out_tail, right_out_tail).
struct to_no_match_pair {
  __device__ cuda::std::tuple<size_type, size_type> operator()(size_type idx) const noexcept
  {
    return cuda::std::make_tuple(cudf::JoinNoMatch, idx);
  }
};

}  // namespace

VectorPair finalize_full_join(VectorPair&& indices,
                              size_type left_table_num_rows,
                              size_type right_table_num_rows,
                              std::optional<cudf::device_span<size_type const>> right_matches,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  auto [left_out, right_out] = std::move(indices);
  CUDF_EXPECTS(left_out->size() == right_out->size(),
               "left/right index vectors must have equal size",
               std::invalid_argument);
  auto const match_total = left_out->size();

  // Empty-left fast path: every right row is unmatched.
  if (left_table_num_rows == 0) {
    auto const tail = static_cast<std::size_t>(right_table_num_rows);
    left_out->resize(match_total + tail, stream);
    right_out->resize(match_total + tail, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     right_out->begin() + match_total,
                     right_out->end(),
                     0);
    thrust::uninitialized_fill(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      left_out->begin() + match_total,
      left_out->end(),
      cudf::JoinNoMatch);
    return std::pair(std::move(left_out), std::move(right_out));
  }

  if (right_table_num_rows == 0) { return std::pair(std::move(left_out), std::move(right_out)); }

  // Grow to the upper bound (match_total + right_table_num_rows); the complement is appended
  // into the tail. If the caller pre-reserved this capacity (see the span overload below),
  // these resizes don't reallocate.
  auto const upper = match_total + static_cast<std::size_t>(right_table_num_rows);
  left_out->resize(upper, stream);
  right_out->resize(upper, stream);

  CUDF_EXPECTS(
    !right_matches || right_matches->size() == static_cast<std::size_t>(right_table_num_rows),
    "right match flags must be absent or have one entry per right row",
    std::invalid_argument);

  // Hash joins mark right rows as part of retrieval and pass those flags here, eliminating an
  // output-sized scatter. Other join implementations use this fallback to derive the same flags
  // from their materialized right indices.
  auto computed_matches = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    right_matches ? 0 : right_table_num_rows, stream, cudf::get_current_device_resource_ref());
  if (!right_matches) {
    thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::make_constant_iterator(size_type{1}),
                       cuda::make_constant_iterator(size_type{1}) + match_total,
                       right_out->begin(),
                       right_out->begin(),
                       computed_matches.begin(),
                       valid_range<size_type>{0, right_table_num_rows});
  }
  auto const match_flags = right_matches ? right_matches->data() : computed_matches.data();

  // Fused compaction: for each unmatched right row, emit (JoinNoMatch, right_idx) into
  // (left_out_tail, right_out_tail) in a single CUB DeviceSelect pass.
  auto zip_tail =
    cuda::make_zip_iterator(left_out->data() + match_total, right_out->data() + match_total);
  auto out_iter = cuda::make_transform_output_iterator(zip_tail, to_no_match_pair{});

  auto const new_end =
    cudf::detail::copy_if(cuda::counting_iterator<size_type>{0},
                          cuda::counting_iterator<size_type>{right_table_num_rows},
                          out_iter,
                          unmatched_flag{match_flags},
                          stream);

  auto const comp_size = cuda::std::distance(out_iter, new_end);
  left_out->resize(match_total + comp_size, stream);
  right_out->resize(match_total + comp_size, stream);

  return std::pair(std::move(left_out), std::move(right_out));
}

VectorPair finalize_full_join(
  cudf::host_span<cudf::device_span<size_type const> const> left_partials,
  cudf::host_span<cudf::device_span<size_type const> const> right_partials,
  size_type left_table_num_rows,
  size_type right_table_num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(left_partials.size() == right_partials.size(),
               "left_partials and right_partials must have the same length",
               std::invalid_argument);

  std::size_t match_total = 0;
  for (std::size_t i = 0; i < left_partials.size(); ++i) {
    CUDF_EXPECTS(left_partials[i].size() == right_partials[i].size(),
                 "matching partials must have equal left/right sizes",
                 std::invalid_argument);
    match_total += left_partials[i].size();
  }

  // Pre-allocate at the upper bound so the VectorPair overload's resize-up becomes a no-op
  // (capacity is already there).
  auto const upper = match_total + static_cast<std::size_t>(right_table_num_rows);
  auto left_out    = std::make_unique<rmm::device_uvector<size_type>>(upper, stream, mr);
  auto right_out   = std::make_unique<rmm::device_uvector<size_type>>(upper, stream, mr);

  // Concatenate every partial into the head of the output via one batched memcpy.
  if (match_total > 0) {
    auto const n = left_partials.size();
    std::vector<void*> dsts;
    std::vector<void const*> srcs;
    std::vector<std::size_t> sizes;
    dsts.reserve(2 * n);
    srcs.reserve(2 * n);
    sizes.reserve(2 * n);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < n; ++i) {
      auto const sz = left_partials[i].size() * sizeof(size_type);
      dsts.push_back(left_out->data() + offset);
      srcs.push_back(left_partials[i].data());
      sizes.push_back(sz);
      dsts.push_back(right_out->data() + offset);
      srcs.push_back(right_partials[i].data());
      sizes.push_back(sz);
      offset += left_partials[i].size();
    }
    CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
      dsts.data(), srcs.data(), sizes.data(), dsts.size(), stream));
  }

  // Shrink the uvectors' logical size to match_total (capacity stays at upper bound), then
  // delegate to the VectorPair overload which resizes back up and appends the complement.
  left_out->resize(match_total, stream);
  right_out->resize(match_total, stream);

  return finalize_full_join(std::pair(std::move(left_out), std::move(right_out)),
                            left_table_num_rows,
                            right_table_num_rows,
                            std::nullopt,
                            stream,
                            mr);
}

}  // namespace detail
}  // namespace cudf
