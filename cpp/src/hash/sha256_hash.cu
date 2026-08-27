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

struct sha256_hash_state {
  uint64_t message_length = 0;
  uint32_t buffer_length  = 0;
  uint32_t hash_value[8]  = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buffer[64];
};

struct SHA256Hash : HashBase<SHA256Hash> {
  __device__ inline SHA256Hash(char* result_location) : HashBase<SHA256Hash>(result_location) {}

  // Intermediate data type storing the hash state
  using hash_state = sha256_hash_state;
  // The word type used by this hash function
  using sha_word_type = uint32_t;
  // Number of bytes processed in each hash step
  static constexpr uint32_t message_chunk_size = 64;
  // Digest size in bytes
  static constexpr uint32_t digest_size = 64;
  // Number of bytes used for the message length
  static constexpr uint32_t message_length_size = 8;

  __device__ inline void hash_step(hash_state& state) { sha256_hash_step(state); }

  hash_state state;
};

}  // namespace

std::unique_ptr<column> sha256(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  return sha_hash<SHA256Hash>(input, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sha256(table_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sha256(input, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
