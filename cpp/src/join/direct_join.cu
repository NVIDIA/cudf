/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/direct_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/iterator>
#include <cuda/std/iterator>

#include <cstdint>
#include <memory>
#include <utility>

namespace cudf {
namespace detail {
namespace {

// Scatters each right row index to the lookup slot addressed by its key value
struct scatter_right_index {
  size_type* lookup;
  std::uint32_t const* right_keys;

  __device__ void operator()(size_type right_idx) const
  {
    lookup[right_keys[right_idx]] = right_idx;
  }
};

// Writes the (left, right) index pair of the `out_idx`-th match, given a matched left row index
struct emit_match_pair {
  size_type* left_out;
  size_type* right_out;
  size_type const* lookup;
  std::uint32_t const* left_keys;

  __device__ void operator()(size_type out_idx, size_type left_idx) const
  {
    left_out[out_idx]  = left_idx;
    right_out[out_idx] = lookup[left_keys[left_idx]];
  }
};

// Returns true if the left row's key hits a right row in the lookup table
struct is_match {
  size_type const* lookup;
  std::uint32_t const* left_keys;

  __device__ bool operator()(size_type left_idx) const
  {
    return lookup[left_keys[left_idx]] != JoinNoMatch;
  }
};

}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
direct_inner_join(column_view const& left_keys,
                  column_view const& right_keys,
                  std::size_t capacity,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    left_keys.type().id() == type_id::UINT32 and right_keys.type().id() == type_id::UINT32,
    "direct_inner_join keys must be of type UINT32",
    cudf::data_type_error);
  CUDF_EXPECTS(not left_keys.has_nulls() and not right_keys.has_nulls(),
               "direct_inner_join keys must not contain nulls",
               std::invalid_argument);
  CUDF_EXPECTS(static_cast<std::size_t>(right_keys.size()) <= capacity,
               "capacity must be at least the number of right keys",
               std::invalid_argument);

  if (left_keys.is_empty() or right_keys.is_empty()) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  // Build: scatter each right row index to the slot addressed by its key value
  auto lookup =
    rmm::device_uvector<size_type>(capacity, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(
    cub::DeviceTransform::Fill(lookup.begin(), lookup.size(), JoinNoMatch, stream.value()));
  CUDF_CUDA_TRY(
    cub::DeviceFor::Bulk(right_keys.size(),
                         scatter_right_index{lookup.data(), right_keys.begin<std::uint32_t>()},
                         stream.value()));

  // Probe: a single pass emitting the (left index, matched right index) pairs
  auto left_indices =
    std::make_unique<rmm::device_uvector<size_type>>(left_keys.size(), stream, mr);
  auto right_indices =
    std::make_unique<rmm::device_uvector<size_type>>(left_keys.size(), stream, mr);

  auto const d_left_keys = left_keys.begin<std::uint32_t>();
  auto const out_iter    = cuda::tabulate_output_iterator{
    emit_match_pair{left_indices->data(), right_indices->data(), lookup.data(), d_left_keys}};

  auto const out_end = cudf::detail::copy_if(cuda::counting_iterator<size_type>{0},
                                             cuda::counting_iterator<size_type>{left_keys.size()},
                                             out_iter,
                                             is_match{lookup.data(), d_left_keys},
                                             stream);

  auto const num_matches = cuda::std::distance(out_iter, out_end);
  left_indices->resize(num_matches, stream);
  right_indices->resize(num_matches, stream);

  return std::pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
direct_inner_join(column_view const& left_keys,
                  column_view const& right_keys,
                  std::size_t capacity,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::direct_inner_join(left_keys, right_keys, capacity, stream, mr);
}

}  // namespace cudf
