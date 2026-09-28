/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <rmm/resource_ref.hpp>

#include <cub/block/block_scan.cuh>
#include <cuda/atomic>

namespace CUDF_EXPORT cudf {
namespace io {
namespace text {
namespace detail {

enum class scan_tile_status : uint8_t {
  oob,
  invalid,
  partial,
  inclusive,
};

template <typename T>
struct scan_tile_state_view {
  uint64_t num_tiles;
  cuda::atomic<scan_tile_status, cuda::thread_scope_device>* tile_status;
  T* tile_partial;
  T* tile_inclusive;

  __device__ inline void set_status(cudf::size_type tile_idx, scan_tile_status status)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    tile_status[offset].store(status, cuda::memory_order_relaxed);
  }

  __device__ inline void set_partial_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    cub::ThreadStore<cub::STORE_CG>(tile_partial + offset, value);
    tile_status[offset].store(scan_tile_status::partial);
  }

  __device__ inline void set_inclusive_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    cub::ThreadStore<cub::STORE_CG>(tile_inclusive + offset, value);
    tile_status[offset].store(scan_tile_status::inclusive);
  }

  __device__ inline T get_prefix(cudf::size_type tile_idx, scan_tile_status& status)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;

    while ((status = tile_status[offset].load(cuda::memory_order_relaxed)) ==
           scan_tile_status::invalid) {}

    if (status == scan_tile_status::partial) {
      return cub::ThreadLoad<cub::LOAD_CG>(tile_partial + offset);
    } else {
      return cub::ThreadLoad<cub::LOAD_CG>(tile_inclusive + offset);
    }
  }
};

template <typename T>
struct scan_tile_state {
  rmm::device_uvector<cuda::atomic<scan_tile_status, cuda::thread_scope_device>> tile_status;
  rmm::device_uvector<T> tile_state_partial;
  rmm::device_uvector<T> tile_state_inclusive;

  scan_tile_state(cudf::size_type num_tiles,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
    : tile_status(rmm::device_uvector<cuda::atomic<scan_tile_status, cuda::thread_scope_device>>(
        num_tiles, stream, mr)),
      tile_state_partial(rmm::device_uvector<T>(num_tiles, stream, mr)),
      tile_state_inclusive(rmm::device_uvector<T>(num_tiles, stream, mr))
  {
  }

  operator scan_tile_state_view<T>()
  {
    return scan_tile_state_view<T>{tile_status.size(),
                                   tile_status.data(),
                                   tile_state_partial.data(),
                                   tile_state_inclusive.data()};
  }

  inline T get_inclusive_prefix(cudf::size_type tile_idx, cuda::stream_ref stream) const
  {
    auto const offset = (tile_idx + tile_status.size()) % tile_status.size();
    return tile_state_inclusive.element(offset, stream);
  }
};

template <typename T>
struct scan_tile_state_callback {
  __device__ inline scan_tile_state_callback(scan_tile_state_view<T>& tile_state,
                                             cudf::size_type tile_idx)
    : _tile_state(tile_state), _tile_idx(tile_idx)
  {
  }

  __device__ inline T operator()(T const& block_aggregate)
  {
    T exclusive_prefix;

    if (threadIdx.x == 0) {
      _tile_state.set_partial_prefix(_tile_idx, block_aggregate);

      auto predecessor_idx    = _tile_idx - 1;
      auto predecessor_status = scan_tile_status::invalid;

      // scan partials to form prefix

      auto window_partial = _tile_state.get_prefix(predecessor_idx, predecessor_status);
      while (predecessor_status != scan_tile_status::inclusive) {
        predecessor_idx--;
        auto predecessor_prefix = _tile_state.get_prefix(predecessor_idx, predecessor_status);
        window_partial          = predecessor_prefix + window_partial;
      }
      exclusive_prefix = window_partial;

      _tile_state.set_inclusive_prefix(_tile_idx, exclusive_prefix + block_aggregate);
    }

    return exclusive_prefix;
  }

  scan_tile_state_view<T>& _tile_state;
  cudf::size_type _tile_idx;
};

}  // namespace detail
}  // namespace text
}  // namespace io
}  // namespace CUDF_EXPORT cudf
