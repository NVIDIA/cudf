/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuda/iterator>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

namespace cudf::detail {

struct precomputed_hash {
  CUDF_HOST_DEVICE constexpr precomputed_hash(hash_value_type const* hashes) : _hashes{hashes} {}

  __device__ __forceinline__ hash_value_type operator()(size_type i) const noexcept
  {
    return _hashes[i];
  }

 private:
  hash_value_type const* _hashes;
};

std::pair<rmm::device_buffer, bitmask_type const*> make_filtered_join_row_bitmask(
  table_view const& input, null_equality nulls_equal, cuda::stream_ref stream);

class filtered_join_row_is_valid {
 public:
  filtered_join_row_is_valid(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ bool operator()(size_type const& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  bitmask_type const* _row_bitmask;
};

template <int32_t CGSize, typename Iterator, typename Ref>
void filtered_join::insert_right_table(Iterator right_iter,
                                       Ref const& insert_ref,
                                       cuda::stream_ref stream)
{
  cudf::scoped_range range{"filtered_join::insert_right_table"};
  // Insert valid rows from the right table into the hash table.
  auto const grid_size              = cuco::detail::grid_size(_right.num_rows(), CGSize);
  auto const bitmask_buffer_and_ptr = make_filtered_join_row_bitmask(_right, _nulls_equal, stream);
  if (bitmask_buffer_and_ptr.second != nullptr) {
    cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
        right_iter,
        _right.num_rows(),
        cuda::counting_iterator<size_type>{0},
        filtered_join_row_is_valid{bitmask_buffer_and_ptr.second},
        insert_ref);
  } else {
    cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
        right_iter,
        _right.num_rows(),
        cuda::constant_iterator<bool>{true},
        cuda::std::identity{},
        insert_ref);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <int32_t CGSize, typename Iterator, typename Ref>
void filtered_join::query_right_table(cudf::table_view const& left,
                                      Iterator left_iter,
                                      Ref query_ref,
                                      cudf::device_span<bool> contains_map,
                                      cuda::stream_ref stream)
{
  cudf::scoped_range range{"filtered_join::query_right_table"};
  auto const grid_size              = cuco::detail::grid_size(left.num_rows(), CGSize);
  auto const bitmask_buffer_and_ptr = make_filtered_join_row_bitmask(left, _nulls_equal, stream);
  if (bitmask_buffer_and_ptr.second != nullptr) {
    cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
        left_iter,
        left.num_rows(),
        cuda::counting_iterator<size_type>{0},
        filtered_join_row_is_valid{bitmask_buffer_and_ptr.second},
        contains_map.begin(),
        query_ref);
  } else {
    cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
      <<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
        left_iter,
        left.num_rows(),
        cuda::constant_iterator<bool>{true},
        cuda::std::identity{},
        contains_map.begin(),
        query_ref);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
