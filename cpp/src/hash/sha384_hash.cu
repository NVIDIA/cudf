/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sha_hash.cuh"

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

// Need alignas(16) to avoid compiler bug.
struct alignas(16) sha384_hash_state {
  uint64_t message_length = 0;
  uint32_t buffer_length  = 0;
  uint64_t hash_value[8]  = {0xcbbb9d5dc1059ed8,
                             0x629a292a367cd507,
                             0x9159015a3070dd17,
                             0x152fecd8f70e5939,
                             0x67332667ffc00b31,
                             0x8eb44a8768581511,
                             0xdb0c2e0d64f98fa7,
                             0x47b5481dbefa4fa4};
  uint8_t buffer[128];
};

struct SHA384Hash : HashBase<SHA384Hash> {
  __device__ inline SHA384Hash(char* result_location) : HashBase<SHA384Hash>(result_location) {}

  // Intermediate data type storing the hash state
  using hash_state = sha384_hash_state;
  // The word type used by this hash function
  using sha_word_type = uint64_t;
  // Number of bytes processed in each hash step
  static constexpr uint32_t message_chunk_size = 128;
  // Digest size in bytes. This is truncated from SHA-512.
  static constexpr uint32_t digest_size = 96;
  // Number of bytes used for the message length
  static constexpr uint32_t message_length_size = 16;

  __device__ inline void hash_step(hash_state& state) { sha512_hash_step(state); }

  hash_state state;
};

}  // namespace

std::unique_ptr<column> sha384(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  return sha_hash<SHA384Hash>(input, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sha384(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sha384(input, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
