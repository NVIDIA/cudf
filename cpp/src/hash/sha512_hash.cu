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
struct alignas(16) sha512_hash_state {
  uint64_t message_length = 0;
  uint32_t buffer_length  = 0;
  uint64_t hash_value[8]  = {0x6a09e667f3bcc908,
                             0xbb67ae8584caa73b,
                             0x3c6ef372fe94f82b,
                             0xa54ff53a5f1d36f1,
                             0x510e527fade682d1,
                             0x9b05688c2b3e6c1f,
                             0x1f83d9abfb41bd6b,
                             0x5be0cd19137e2179};
  uint8_t buffer[128];
};

struct SHA512Hash : HashBase<SHA512Hash> {
  __device__ inline SHA512Hash(char* result_location) : HashBase<SHA512Hash>(result_location) {}

  // Intermediate data type storing the hash state
  using hash_state = sha512_hash_state;
  // The word type used by this hash function
  using sha_word_type = uint64_t;
  // Number of bytes processed in each hash step
  static constexpr uint32_t message_chunk_size = 128;
  // Digest size in bytes
  static constexpr uint32_t digest_size = 128;
  // Number of bytes used for the message length
  static constexpr uint32_t message_length_size = 16;

  __device__ inline void hash_step(hash_state& state) { sha512_hash_step(state); }

  hash_state state;
};

}  // namespace

std::unique_ptr<column> sha512(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  return sha_hash<SHA512Hash>(input, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sha512(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sha512(input, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
