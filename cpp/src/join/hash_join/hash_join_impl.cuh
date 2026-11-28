/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hash_graph.cuh"

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>
#include <cuda/std/cstdint>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cudf::detail {

template <typename Hasher>
struct hash_join<Hasher>::impl {
  impl(cuda::std::uint32_t num_buckets,
       cuda::stream_ref stream,
       cuda::mr::any_resource<cuda::mr::device_accessible> mr)
    : _mr(std::move(mr)),
      _offsets(stream, _mr, static_cast<std::size_t>(num_buckets) + 1, cuda::no_init),
      _records(stream, _mr, 0, cuda::no_init),
      _num_buckets(num_buckets)
  {
  }

  hash_graph_ref hash_graph() const { return {_offsets.data(), _records.data(), _num_buckets}; }

  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;
  cuda::device_buffer<size_type> _offsets;
  cuda::device_buffer<hash_graph_record> _records;
  cuda::std::uint32_t _num_buckets;
};

}  // namespace cudf::detail
