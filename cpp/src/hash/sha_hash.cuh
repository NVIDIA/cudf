/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>

namespace cudf {
namespace hashing {
namespace detail {

__constant__ constexpr uint32_t sha256_hash_constants[64] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

__constant__ constexpr uint64_t sha512_hash_constants[80] = {
  0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
  0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
  0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
  0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
  0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
  0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
  0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
  0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
  0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
  0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
  0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
  0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
  0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
  0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
  0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
  0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
  0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
  0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
  0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
  0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
};

/**
 * @brief A CRTP helper function
 *
 * https://www.fluentcpp.com/2017/05/19/crtp-helper/
 *
 * Does two things:
 * 1. Makes "crtp" explicit in the inheritance structure of a CRTP base class.
 * 2. Avoids having to `static_cast` in a lot of places
 *
 * @tparam T The derived class in a CRTP hierarchy
 */
template <typename T>
struct crtp {
  __device__ inline T& underlying() { return static_cast<T&>(*this); }
  __device__ inline T const& underlying() const { return static_cast<T const&>(*this); }
};

template <typename Hasher>
struct HashBase : public crtp<Hasher> {
  char* result_location;

  __device__ inline HashBase(char* result_location) : result_location(result_location) {}

  /**
   * @brief Execute SHA on input data chunks.
   *
   * This accepts arbitrary data, handles it as bytes, and calls the hash step
   * when the buffer is filled up to message_chunk_size bytes.
   */
  __device__ inline void process(uint8_t const* data, uint32_t len)
  {
    auto& state = this->underlying().state;
    state.message_length += len;

    if (state.buffer_length + len < Hasher::message_chunk_size) {
      // The buffer will not be filled by this data. We copy the new data into
      // the buffer but do not trigger a hash step yet.
      memcpy(state.buffer + state.buffer_length, data, len);
      state.buffer_length += len;
    } else {
      // The buffer will be filled by this data. Copy a chunk of the data to fill
      // the buffer and trigger a hash step.
      uint32_t copylen = Hasher::message_chunk_size - state.buffer_length;
      memcpy(state.buffer + state.buffer_length, data, copylen);
      this->underlying().hash_step(state);

      // Take buffer-sized chunks of the data and do a hash step on each chunk.
      // Check with equality here because the last chunk may be exactly the size of the buffer.
      while (len >= Hasher::message_chunk_size + copylen) {
        memcpy(state.buffer, data + copylen, Hasher::message_chunk_size);
        this->underlying().hash_step(state);
        copylen += Hasher::message_chunk_size;
      }

      // The remaining data chunk (if any) does not fill the buffer. We copy the data into
      // the buffer but do not trigger a hash step yet.
      if (len > copylen) { memcpy(state.buffer, data + copylen, len - copylen); }
      state.buffer_length = len - copylen;
    }
  }

  template <typename T>
  __device__ inline void process_fixed_width(T const& key)
  {
    uint8_t const* data    = reinterpret_cast<uint8_t const*>(&key);
    uint32_t constexpr len = sizeof(T);
    process(data, len);
  }

  /**
   * @brief Finalize SHA element processing.
   *
   * This method fills the remainder of the message buffer with zeros, appends
   * the message length (in another step of the hash, if needed), and performs
   * the final hash step.
   */
  __device__ inline void finalize()
  {
    auto& state = this->underlying().state;
    // Message length in bits.
    uint64_t const message_length_in_bits = (static_cast<uint64_t>(state.message_length)) << 3;
    // Add a one bit flag (10000000) to signal the end of the message
    uint8_t constexpr end_of_message = 0x80;
    // 1 byte for the end of the message flag
    uint32_t constexpr end_of_message_size = 1;

    thrust::fill_n(
      thrust::seq, state.buffer + state.buffer_length, end_of_message_size, end_of_message);

    // SHA-512 uses a 128-bit message length instead of a 64-bit message length
    // but this code does not support messages with lengths exceeding UINT64_MAX
    // bits. We always pad the upper 64 bits with zeros.
    uint32_t constexpr message_length_supported_size = sizeof(message_length_in_bits);

    if (state.buffer_length + Hasher::message_length_size + end_of_message_size <=
        Hasher::message_chunk_size) {
      // Fill the remainder of the buffer with zeros up to the space reserved
      // for the message length. The message length fits in this hash step.
      thrust::fill(thrust::seq,
                   state.buffer + state.buffer_length + end_of_message_size,
                   state.buffer + Hasher::message_chunk_size - message_length_supported_size,
                   0x00);
    } else {
      // Fill the remainder of the buffer with zeros. The message length doesn't
      // fit and will be processed in a subsequent hash step comprised of only
      // zeros followed by the message length.
      thrust::fill(thrust::seq,
                   state.buffer + state.buffer_length + end_of_message_size,
                   state.buffer + Hasher::message_chunk_size,
                   0x00);
      this->underlying().hash_step(state);

      // Fill the entire message with zeros up to the final bytes reserved for
      // the message length.
      thrust::fill_n(thrust::seq,
                     state.buffer,
                     Hasher::message_chunk_size - message_length_supported_size,
                     0x00);
    }

    // Convert the 64-bit message length from little-endian to big-endian.
    uint64_t const full_length_flipped = swap_endian(message_length_in_bits);
    memcpy(state.buffer + Hasher::message_chunk_size - message_length_supported_size,
           reinterpret_cast<uint8_t const*>(&full_length_flipped),
           message_length_supported_size);
    this->underlying().hash_step(state);

    // Each byte in the word generates two bytes in the hexadecimal string digest.
    // SHA-224 and SHA-384 digests are truncated because their digest does not
    // include all of the hash values.
    auto constexpr num_words_to_copy =
      Hasher::digest_size / (2 * sizeof(typename Hasher::sha_word_type));
    for (int i = 0; i < num_words_to_copy; i++) {
      // Convert word representation from big-endian to little-endian.
      typename Hasher::sha_word_type flipped = swap_endian(state.hash_value[i]);
      if constexpr (std::is_same_v<typename Hasher::sha_word_type, uint32_t>) {
        uint32ToLowercaseHexString(flipped, result_location + (8 * i));
      } else if constexpr (std::is_same_v<typename Hasher::sha_word_type, uint64_t>) {
        uint32_t low_bits = static_cast<uint32_t>(flipped);
        uint32ToLowercaseHexString(low_bits, result_location + (16 * i));
        uint32_t high_bits = static_cast<uint32_t>(flipped >> 32);
        uint32ToLowercaseHexString(high_bits, result_location + (16 * i) + 8);
      } else {
        cudf_assert(false && "Unsupported SHA word type.");
      }
    }
  };
};

template <typename Hasher>
struct HasherDispatcher {
  Hasher* hasher;
  column_device_view input_col;

  __device__ inline HasherDispatcher(Hasher* hasher, column_device_view const& input_col)
    : hasher{hasher}, input_col{input_col}
  {
  }

  template <typename Element,
            CUDF_ENABLE_IF(is_fixed_width<Element>() and not is_floating_point<Element>() and
                           not is_chrono<Element>())>
  __device__ inline void operator()(size_type row_index)
  {
    Element const& key = input_col.element<Element>(row_index);
    hasher->process_fixed_width(key);
  }

  template <typename Element, CUDF_ENABLE_IF(is_floating_point<Element>())>
  __device__ inline void operator()(size_type row_index)
  {
    Element const& key = input_col.element<Element>(row_index);
    if (isnan(key)) {
      Element nan = cuda::std::numeric_limits<Element>::quiet_NaN();
      hasher->process_fixed_width(nan);
    } else if (key == Element{0.0}) {
      hasher->process_fixed_width(Element{0.0});
    } else {
      hasher->process_fixed_width(key);
    }
  }

  template <typename Element, CUDF_ENABLE_IF(std::is_same_v<Element, string_view>)>
  __device__ inline void operator()(size_type row_index)
  {
    string_view key     = input_col.element<string_view>(row_index);
    uint8_t const* data = reinterpret_cast<uint8_t const*>(key.data());
    uint32_t const len  = static_cast<uint32_t>(key.size_bytes());
    hasher->process(data, len);
  }

  template <typename Element,
            CUDF_ENABLE_IF((not is_fixed_width<Element>() or is_chrono<Element>()) and
                           not std::is_same_v<Element, string_view>)>
  __device__ inline void operator()(size_type row_index)
  {
    (void)row_index;
    cudf_assert(false && "Unsupported type for hash function.");
  }
};

/**
 * @brief Core SHA-1 algorithm implementation
 *
 * Processes a single 512-bit chunk, updating the hash value so far.
 * This does not zero out the buffer contents.
 */
template <typename hash_state>
__device__ inline void sha1_hash_step(hash_state& state)
{
  uint32_t words[80];

  // The 512-bit message buffer fills the first 16 words.
  memcpy(&words[0], state.buffer, sizeof(words[0]) * 16);
  for (int i = 0; i < 16; i++) {
    // Convert word representation from little-endian to big-endian.
    words[i] = swap_endian(words[i]);
  }

  // The rest of the 80 words are generated from the first 16 words.
  for (int i = 16; i < 80; i++) {
    uint32_t const temp = words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16];
    words[i]            = rotate_bits_left(temp, 1);
  }

  uint32_t A = state.hash_value[0];
  uint32_t B = state.hash_value[1];
  uint32_t C = state.hash_value[2];
  uint32_t D = state.hash_value[3];
  uint32_t E = state.hash_value[4];

  for (int i = 0; i < 80; i++) {
    uint32_t F;
    uint32_t k;
    uint32_t temp;
    switch (i / 20) {
      case 0:
        F = D ^ (B & (C ^ D));
        k = 0x5a827999;
        break;
      case 1:
        F = B ^ C ^ D;
        k = 0x6ed9eba1;
        break;
      case 2:
        F = (B & C) | (B & D) | (C & D);
        k = 0x8f1bbcdc;
        break;
      case 3:
        F = B ^ C ^ D;
        k = 0xca62c1d6;
        break;
    }
    temp = rotate_bits_left(A, 5) + F + E + k + words[i];
    E    = D;
    D    = C;
    C    = rotate_bits_left(B, 30);
    B    = A;
    A    = temp;
  }

  state.hash_value[0] += A;
  state.hash_value[1] += B;
  state.hash_value[2] += C;
  state.hash_value[3] += D;
  state.hash_value[4] += E;

  state.buffer_length = 0;
}

/**
 * @brief Core SHA-256 algorithm implementation
 *
 * Processes a single 512-bit chunk, updating the hash value so far.
 * This does not zero out the buffer contents.
 */
template <typename hash_state>
__device__ inline void sha256_hash_step(hash_state& state)
{
  uint32_t words[64];

  // The 512-bit message buffer fills the first 16 words.
  memcpy(&words[0], state.buffer, sizeof(words[0]) * 16);
  for (int i = 0; i < 16; i++) {
    // Convert word representation from little-endian to big-endian.
    words[i] = swap_endian(words[i]);
  }

  // The rest of the 64 words are generated from the first 16 words.
  for (int i = 16; i < 64; i++) {
    uint32_t const s0 = rotate_bits_right(words[i - 15], 7) ^ rotate_bits_right(words[i - 15], 18) ^
                        (words[i - 15] >> 3);
    uint32_t const s1 = rotate_bits_right(words[i - 2], 17) ^ rotate_bits_right(words[i - 2], 19) ^
                        (words[i - 2] >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  uint32_t A = state.hash_value[0];
  uint32_t B = state.hash_value[1];
  uint32_t C = state.hash_value[2];
  uint32_t D = state.hash_value[3];
  uint32_t E = state.hash_value[4];
  uint32_t F = state.hash_value[5];
  uint32_t G = state.hash_value[6];
  uint32_t H = state.hash_value[7];

  for (int i = 0; i < 64; i++) {
    uint32_t const s1 =
      rotate_bits_right(E, 6) ^ rotate_bits_right(E, 11) ^ rotate_bits_right(E, 25);
    uint32_t const ch    = (E & F) ^ ((~E) & G);
    uint32_t const temp1 = H + s1 + ch + sha256_hash_constants[i] + words[i];
    uint32_t const s0 =
      rotate_bits_right(A, 2) ^ rotate_bits_right(A, 13) ^ rotate_bits_right(A, 22);
    uint32_t const maj   = (A & B) ^ (A & C) ^ (B & C);
    uint32_t const temp2 = s0 + maj;

    H = G;
    G = F;
    F = E;
    E = D + temp1;
    D = C;
    C = B;
    B = A;
    A = temp1 + temp2;
  }

  state.hash_value[0] += A;
  state.hash_value[1] += B;
  state.hash_value[2] += C;
  state.hash_value[3] += D;
  state.hash_value[4] += E;
  state.hash_value[5] += F;
  state.hash_value[6] += G;
  state.hash_value[7] += H;

  state.buffer_length = 0;
}

/**
 * @brief Core SHA-512 algorithm implementation
 *
 * Processes a single 1024-bit chunk, updating the hash value so far.
 * This does not zero out the buffer contents.
 */
template <typename hash_state>
__device__ inline void sha512_hash_step(hash_state& state)
{
  uint64_t words[80];

  // The 1024-bit message buffer fills the first 16 words.
  memcpy(&words[0], state.buffer, sizeof(words[0]) * 16);
  for (int i = 0; i < 16; i++) {
    // Convert word representation from little-endian to big-endian.
    words[i] = swap_endian(words[i]);
  }

  // The rest of the 80 words are generated from the first 16 words.
  for (int i = 16; i < 80; i++) {
    uint64_t const s0 = rotate_bits_right(words[i - 15], 1) ^ rotate_bits_right(words[i - 15], 8) ^
                        (words[i - 15] >> 7);
    uint64_t const s1 = rotate_bits_right(words[i - 2], 19) ^ rotate_bits_right(words[i - 2], 61) ^
                        (words[i - 2] >> 6);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  uint64_t A = state.hash_value[0];
  uint64_t B = state.hash_value[1];
  uint64_t C = state.hash_value[2];
  uint64_t D = state.hash_value[3];
  uint64_t E = state.hash_value[4];
  uint64_t F = state.hash_value[5];
  uint64_t G = state.hash_value[6];
  uint64_t H = state.hash_value[7];

  for (int i = 0; i < 80; i++) {
    uint64_t const s1 =
      rotate_bits_right(E, 14) ^ rotate_bits_right(E, 18) ^ rotate_bits_right(E, 41);
    uint64_t const ch    = (E & F) ^ ((~E) & G);
    uint64_t const temp1 = H + s1 + ch + sha512_hash_constants[i] + words[i];
    uint64_t const s0 =
      rotate_bits_right(A, 28) ^ rotate_bits_right(A, 34) ^ rotate_bits_right(A, 39);
    uint64_t const maj   = (A & B) ^ (A & C) ^ (B & C);
    uint64_t const temp2 = s0 + maj;

    H = G;
    G = F;
    F = E;
    E = D + temp1;
    D = C;
    C = B;
    B = A;
    A = temp1 + temp2;
  }

  state.hash_value[0] += A;
  state.hash_value[1] += B;
  state.hash_value[2] += C;
  state.hash_value[3] += D;
  state.hash_value[4] += E;
  state.hash_value[5] += F;
  state.hash_value[6] += G;
  state.hash_value[7] += H;

  state.buffer_length = 0;
}

// SHA supported leaf data type check
bool inline sha_leaf_type_check(data_type dt)
{
  return (is_fixed_width(dt) && !is_chrono(dt)) || (dt.id() == type_id::STRING);
}

/**
 * @brief Call a SHA-1 or SHA-2 hash function on a table view.
 *
 * @tparam Hasher The struct used for computing SHA hashes.
 *
 * @param input The input table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resource to use for the device memory allocation
 * @return A new column with the computed hash function result
 */
template <typename Hasher>
std::unique_ptr<column> sha_hash(table_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  if (input.num_rows() == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

  // Accepts string and fixed width columns.
  // TODO: Accept single layer list columns holding those types.
  CUDF_EXPECTS(
    std::all_of(
      input.begin(), input.end(), [](auto const& col) { return sha_leaf_type_check(col.type()); }),
    "Unsupported column type for hash function.",
    cudf::data_type_error);

  // Result column allocation and creation
  auto begin = cuda::make_constant_iterator(Hasher::digest_size);
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), stream, mr);

  auto chars   = rmm::device_uvector<char>(bytes, stream, mr);
  auto d_chars = chars.data();

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{input.num_rows()},
    [d_chars, device_input = *device_input] __device__(auto row_index) {
      Hasher hasher(d_chars + (static_cast<int64_t>(row_index) * Hasher::digest_size));
      for (auto const& col : device_input) {
        if (col.is_valid(row_index)) {
          cudf::type_dispatcher<dispatch_storage_type>(
            col.type(), HasherDispatcher(&hasher, col), row_index);
        }
      }
      hasher.finalize();
    });

  return make_strings_column(input.num_rows(), std::move(offsets_column), chars.release(), 0, {});
}

}  // namespace detail
}  // namespace hashing
}  // namespace cudf
