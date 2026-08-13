/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/utility>
#include <thrust/for_each.h>

#include <iterator>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

template <int capacity, typename hash_step_callable>
struct hash_circular_buffer {
  uint8_t storage[capacity];
  uint8_t* cur;
  int available_space{capacity};
  hash_step_callable hash_step;

  __device__ inline hash_circular_buffer(hash_step_callable hash_step)
    : cur{storage}, hash_step{hash_step}
  {
  }

  __device__ inline void put(uint8_t const* in, int size)
  {
    int copy_start = 0;
    while (size >= available_space) {
      // The buffer will be filled by this chunk of data. Copy a chunk of the
      // data to fill the buffer and trigger a hash step.
      memcpy(cur, in + copy_start, available_space);
      hash_step(storage);
      size -= available_space;
      copy_start += available_space;
      cur             = storage;
      available_space = capacity;
    }
    // The buffer will not be filled by the remaining data. That is, `size >= 0
    // && size < capacity`. We copy the remaining data into the buffer but do
    // not trigger a hash step.
    memcpy(cur, in + copy_start, size);
    cur += size;
    available_space -= size;
  }

  __device__ inline void pad(int const space_to_leave)
  {
    if (space_to_leave > available_space) {
      memset(cur, 0x00, available_space);
      hash_step(storage);
      cur             = storage;
      available_space = capacity;
    }
    memset(cur, 0x00, available_space - space_to_leave);
    cur += available_space - space_to_leave;
    available_space = space_to_leave;
  }

  __device__ inline uint8_t const& operator[](int idx) const { return storage[idx]; }
};

// Get a uint8_t pointer to a column element and its size as a pair.
template <typename Element>
auto __device__ inline get_element_pointer_and_size(Element const& element)
{
  if constexpr (is_fixed_width<Element>() && !is_chrono<Element>()) {
    return cuda::std::make_pair(reinterpret_cast<uint8_t const*>(&element), sizeof(Element));
  } else {
    CUDF_UNREACHABLE("Unsupported type.");
  }
}

template <>
auto __device__ inline get_element_pointer_and_size(string_view const& element)
{
  return cuda::std::make_pair(reinterpret_cast<uint8_t const*>(element.data()),
                              element.size_bytes());
}

// The MD5 algorithm and its hash/shift constants are officially specified in
// RFC 1321. For convenience, these values can also be found on Wikipedia:
// https://en.wikipedia.org/wiki/MD5
__constant__ constexpr uint32_t md5_shift_constants[16] = {
  7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21};

__constant__ constexpr uint32_t md5_hash_constants[64] = {
  0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
  0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
  0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
  0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
  0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
  0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
  0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
  0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

struct MD5Hasher {
  static constexpr int message_chunk_size = 64;

  __device__ inline MD5Hasher(char* result_location)
    : result_location(result_location), buffer(md5_hash_step{hash_values})
  {
  }

  __device__ inline ~MD5Hasher()
  {
    // On destruction, finalize the message buffer and write out the current
    // hexadecimal hash value to the result location.
    // Add a one byte flag 0b10000000 to signal the end of the message.
    uint8_t constexpr end_of_message = 0x80;
    // The message length is appended to the end of the last chunk processed.
    uint64_t const message_length_in_bits = message_length * 8;

    buffer.put(&end_of_message, sizeof(end_of_message));
    buffer.pad(sizeof(message_length_in_bits));
    buffer.put(reinterpret_cast<uint8_t const*>(&message_length_in_bits),
               sizeof(message_length_in_bits));

    for (int i = 0; i < 4; ++i) {
      uint32ToLowercaseHexString(hash_values[i], result_location + (8 * i));
    }
  }

  MD5Hasher(MD5Hasher const&)            = delete;
  MD5Hasher& operator=(MD5Hasher const&) = delete;
  MD5Hasher(MD5Hasher&&)                 = delete;
  MD5Hasher& operator=(MD5Hasher&&)      = delete;

  template <typename Element>
  void __device__ inline process(Element const& element)
  {
    auto const normalized_element  = normalize_nans_and_zeros(element);
    auto const [element_ptr, size] = get_element_pointer_and_size(normalized_element);
    buffer.put(element_ptr, size);
    message_length += size;
  }

  /**
   * @brief Core MD5 algorithm implementation. Processes a single 64-byte chunk,
   * updating the hash value so far. Does not zero out the buffer contents.
   */
  struct md5_hash_step {
    uint32_t (&hash_values)[4];

    void __device__ inline operator()(uint8_t const (&buffer)[message_chunk_size])
    {
      uint32_t A = hash_values[0];
      uint32_t B = hash_values[1];
      uint32_t C = hash_values[2];
      uint32_t D = hash_values[3];

      for (int j = 0; j < message_chunk_size; j++) {
        uint32_t F;
        uint32_t g;
        // No default case is needed because j < 64. j / 16 is always 0, 1, 2, or 3.
        switch (j / 16) {
          case 0:
            F = (B & C) | ((~B) & D);
            g = j;
            break;
          case 1:
            F = (D & B) | ((~D) & C);
            g = (5 * j + 1) % 16;
            break;
          case 2:
            F = B ^ C ^ D;
            g = (3 * j + 5) % 16;
            break;
          case 3:
            F = C ^ (B | (~D));
            g = (7 * j) % 16;
            break;
        }

        uint32_t buffer_element_as_int;
        memcpy(&buffer_element_as_int, &buffer[g * 4], 4);
        F = F + A + md5_hash_constants[j] + buffer_element_as_int;
        A = D;
        D = C;
        C = B;
        B = B + rotate_bits_left(F, md5_shift_constants[((j / 16) * 4) + (j % 4)]);
      }

      hash_values[0] += A;
      hash_values[1] += B;
      hash_values[2] += C;
      hash_values[3] += D;
    }
  };

  char* result_location;
  hash_circular_buffer<message_chunk_size, md5_hash_step> buffer;
  uint64_t message_length = 0;
  uint32_t hash_values[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
};

template <typename Hasher>
struct HasherDispatcher {
  Hasher* hasher;
  column_device_view const& input_col;

  __device__ inline HasherDispatcher(Hasher* hasher, column_device_view const& input_col)
    : hasher{hasher}, input_col{input_col}
  {
  }

  template <typename Element>
  void __device__ inline operator()(size_type const row_index) const
  {
    if constexpr ((is_fixed_width<Element>() && !is_chrono<Element>()) ||
                  std::is_same_v<Element, string_view>) {
      hasher->process(input_col.element<Element>(row_index));
    } else {
      (void)row_index;
      CUDF_UNREACHABLE("Unsupported type for hash function.");
    }
  }
};

template <typename Hasher>
struct ListHasherDispatcher {
  Hasher* hasher;
  column_device_view const& input_col;

  __device__ inline ListHasherDispatcher(Hasher* hasher, column_device_view const& input_col)
    : hasher{hasher}, input_col{input_col}
  {
  }

  template <typename Element>
  void __device__ inline operator()(size_type const offset_begin, size_type const offset_end) const
  {
    if constexpr ((is_fixed_width<Element>() && !is_chrono<Element>()) ||
                  std::is_same_v<Element, string_view>) {
      for (size_type i = offset_begin; i < offset_end; i++) {
        if (input_col.is_valid(i)) { hasher->process(input_col.element<Element>(i)); }
      }
    } else {
      (void)offset_begin;
      (void)offset_end;
      CUDF_UNREACHABLE("Unsupported type for hash function.");
    }
  }
};

// MD5 supported leaf data type check
inline bool md5_leaf_type_check(data_type dt)
{
  return (is_fixed_width(dt) && !is_chrono(dt)) || (dt.id() == type_id::STRING);
}

}  // namespace

std::unique_ptr<column> md5(table_view const& input,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    // Return the MD5 hash of a zero-length input.
    string_scalar const string_128bit(
      "d41d8cd98f00b204e9orig98ecf8427e", true, stream, cudf::get_current_device_resource_ref());
    return make_column_from_scalar(string_128bit, input.num_rows(), stream, mr);
  }

  // Accepts string and fixed width columns, or single layer list columns holding those types
  CUDF_EXPECTS(std::all_of(input.begin(),
                           input.end(),
                           [](auto const& col) {
                             if (col.type().id() == type_id::LIST) {
                               return md5_leaf_type_check(lists_column_view(col).child().type());
                             }
                             return md5_leaf_type_check(col.type());
                           }),
               "Unsupported column type for hash function.",
               cudf::data_type_error);

  // Digest size in bytes
  auto constexpr digest_size = 32;
  // Result column allocation and creation
  auto begin = cuda::make_constant_iterator(digest_size);
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), stream, mr);

  rmm::device_uvector<char> chars(bytes, stream, mr);
  auto d_chars = chars.data();

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{input.num_rows()},
    [d_chars, device_input = *device_input] __device__(auto row_index) {
      MD5Hasher hasher(d_chars + (static_cast<int64_t>(row_index) * digest_size));
      for (auto const& col : device_input) {
        if (col.is_valid(row_index)) {
          if (col.type().id() == type_id::LIST) {
            auto const data_col = col.child(lists_column_view::child_column_index);
            auto const offsets  = col.child(lists_column_view::offsets_column_index);
            if (data_col.type().id() == type_id::LIST) {
              CUDF_UNREACHABLE("Nested list unsupported");
            }
            auto const offset_begin = offsets.element<size_type>(row_index);
            auto const offset_end   = offsets.element<size_type>(row_index + 1);
            cudf::type_dispatcher<dispatch_storage_type>(
              data_col.type(), ListHasherDispatcher(&hasher, data_col), offset_begin, offset_end);
          } else {
            cudf::type_dispatcher<dispatch_storage_type>(
              col.type(), HasherDispatcher(&hasher, col), row_index);
          }
        }
      }
    });

  return make_strings_column(input.num_rows(), std::move(offsets_column), chars.release(), 0, {});
}

}  // namespace detail

std::unique_ptr<column> md5(table_view const& input,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::md5(input, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
