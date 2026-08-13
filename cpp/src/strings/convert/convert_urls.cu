/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cub/cub.cuh>
#include <cuda/std/algorithm>

namespace cudf {
namespace strings {
namespace detail {
namespace {
//
// This is the functor for the url_encode() method below.
// Specific requirements are documented in custrings issue #321.
// In summary it converts mostly non-ascii characters and control characters into UTF-8 hex
// characters prefixed with '%'. For example, the space character must be converted to characters
// '%20' where the '20' indicates the hex value for space in UTF-8. Likewise, multi-byte characters
// are converted to multiple hex characters. For example, the é character is converted to characters
// '%C3%A9' where 'C3A9' is the UTF-8 bytes xc3a9 for this character.
//
struct url_encoder_fn {
  column_device_view const d_strings;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  // utility to create 2-byte hex characters from single binary byte
  __device__ void byte_to_hex(uint8_t byte, char* hex)
  {
    hex[0] = '0';
    if (byte >= 16) {
      uint8_t hibyte = byte / 16;
      hex[0]         = hibyte < 10 ? '0' + hibyte : 'A' + (hibyte - 10);
      byte           = byte - (hibyte * 16);
    }
    hex[1] = byte < 10 ? '0' + byte : 'A' + (byte - 10);
  }

  __device__ bool should_not_url_encode(char ch)
  {
    return (
      (ch >= '0' && ch <= '9') ||  // these are the characters
      (ch >= 'A' && ch <= 'Z') ||  // that are not to be url encoded
      (ch >= 'a' &&
       ch <= 'z') ||  // reference: docs.python.org/3/library/urllib.parse.html#urllib.parse.quote
      (ch == '.') ||
      (ch == '_') || (ch == '~') || (ch == '-'));
  }

  // main part of the functor the performs the url-encoding
  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    string_view d_str = d_strings.element<string_view>(idx);
    //
    char* out_ptr    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    size_type nbytes = 0;
    char hex[2];  // two-byte hex max
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto ch = *itr;
      if (ch < 128) {
        if (should_not_url_encode(static_cast<char>(ch))) {
          nbytes++;
          if (out_ptr) out_ptr = copy_and_increment(out_ptr, d_str.data() + itr.byte_offset(), 1);
        } else  // url-encode everything else
        {
          nbytes += 3;
          if (out_ptr) {
            out_ptr = copy_and_increment(out_ptr, "%", 1);  // add the '%' prefix
            byte_to_hex(static_cast<uint8_t>(ch), hex);     // convert to 2 hex chars
            out_ptr = copy_and_increment(out_ptr, hex, 2);  // add them to the output
          }
        }
      } else  // these are to be utf-8 url-encoded
      {
        uint8_t char_bytes[4];  // holds utf-8 bytes for one character
        size_type char_width = from_char_utf8(ch, reinterpret_cast<char*>(char_bytes));
        nbytes += char_width * 3;  // '%' plus 2 hex chars per byte (example: é is %C3%A9)
        // process each byte in this current character
        for (size_type chidx = 0; out_ptr && (chidx < char_width); ++chidx) {
          out_ptr = copy_and_increment(out_ptr, "%", 1);  // add '%' prefix
          byte_to_hex(char_bytes[chidx], hex);            // convert to 2 hex chars
          out_ptr = copy_and_increment(out_ptr, hex, 2);  // add them to the output
        }
      }
    }
    if (!d_chars) { d_sizes[idx] = nbytes; }
  }
};

}  // namespace

//
std::unique_ptr<column> url_encode(strings_column_view const& input,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);

  auto d_column = column_device_view::create(input.parent(), stream);

  auto [offsets_column, chars] =
    make_strings_children(url_encoder_fn{*d_column}, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

// external API
std::unique_ptr<column> url_encode(strings_column_view const& input,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::url_encode(input, stream, mr);
}

namespace detail {
namespace {

// utility to convert a hex char into a single byte
__device__ constexpr uint8_t hex_char_to_byte(char ch)
{
  if (ch >= '0' && ch <= '9') return (ch - '0');
  if (ch >= 'A' && ch <= 'F') return (ch - 'A' + 10);  // in hex A=10,B=11,...,F=15
  if (ch >= 'a' && ch <= 'f') return (ch - 'a' + 10);  // same for lower case
  return 0;
}

__device__ constexpr bool is_hex_digit(char ch)
{
  return (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'F') || (ch >= 'a' && ch <= 'f');
}

__forceinline__ __device__ bool is_escape_char(char const* const ptr)
{
  return (ptr[0] == '%' && is_hex_digit(ptr[1]) && is_hex_digit(ptr[2]));
}

// helper function for converting an escaped sequence starting at `ptr` to a single byte
__forceinline__ __device__ char escaped_sequence_to_byte(char const* const ptr)
{
  return (hex_char_to_byte(ptr[1]) << 4) | hex_char_to_byte(ptr[2]);
}

/**
 * @brief Count the number of characters of each string after URL decoding.
 *
 * @tparam num_warps_per_threadblock Number of warps in a threadblock. This template argument must
 * match the launch configuration, i.e. the kernel must be launched with
 * `num_warps_per_threadblock * cudf::detail::warp_size` threads per threadblock.
 * @tparam char_block_size Number of characters which will be loaded into the shared memory at a
 * time.
 *
 * @param[in] in_strings Input string column.
 * @param[out] out_counts Number of characters in each decode URL.
 */
template <size_type num_warps_per_threadblock, size_type char_block_size>
CUDF_KERNEL void url_decode_char_counter(column_device_view const in_strings,
                                         size_type* const out_counts)
{
  constexpr int halo_size = 2;
  __shared__ char temporary_buffer[num_warps_per_threadblock][char_block_size + halo_size];
  __shared__ typename cub::WarpReduce<int8_t>::TempStorage cub_storage[num_warps_per_threadblock];

  namespace cg     = cooperative_groups;
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  auto const global_thread_id = cudf::detail::grid_1d::global_thread_id();
  auto const global_warp_id   = static_cast<size_type>(global_thread_id / cudf::detail::warp_size);
  auto const local_warp_id    = static_cast<size_type>(warp.meta_group_rank());
  auto const warp_lane        = static_cast<size_type>(warp.thread_rank());
  auto const nwarps =
    static_cast<size_type>(cudf::detail::grid_1d::grid_stride() / cudf::detail::warp_size);
  char* in_chars_shared = temporary_buffer[local_warp_id];

  // Loop through strings, and assign each string to a warp.
  for (thread_index_type tidx = global_warp_id; tidx < in_strings.size(); tidx += nwarps) {
    auto const row_idx = static_cast<size_type>(tidx);
    if (in_strings.is_null(row_idx)) {
      out_counts[row_idx] = 0;
      continue;
    }

    auto const in_string     = in_strings.element<string_view>(row_idx);
    auto const in_chars      = in_string.data();
    auto const string_length = in_string.size_bytes();
    auto const nblocks       = cudf::util::div_rounding_up_unsafe(string_length, char_block_size);
    size_type escape_char_count = 0;

    for (size_type block_idx = 0; block_idx < nblocks; block_idx++) {
      auto const string_length_block =
        cuda::std::min(char_block_size, string_length - char_block_size * block_idx);

      // Each warp collectively loads input characters of the current block to the shared memory.
      // When testing whether a location is the start of an escaped character, we need to access
      // the current location as well as the next two locations. To avoid branches, two halo cells
      // are added after the end of the block. If the cell is beyond the end of the string, 0s are
      // filled in to make sure the last two characters of the string are not the start of an
      // escaped sequence.
      for (auto char_idx = warp_lane; char_idx < string_length_block + halo_size;
           char_idx += cudf::detail::warp_size) {
        auto const in_idx         = block_idx * char_block_size + char_idx;
        in_chars_shared[char_idx] = in_idx < string_length ? in_chars[in_idx] : 0;
      }

      warp.sync();

      // `char_idx_start` represents the start character index of the current warp.
      for (size_type char_idx_start = 0; char_idx_start < string_length_block;
           char_idx_start += cudf::detail::warp_size) {
        auto const char_idx = char_idx_start + warp_lane;
        int8_t const is_ichar_escape_char =
          (char_idx < string_length_block && is_escape_char(in_chars_shared + char_idx)) ? 1 : 0;

        // Warp-wise reduction to calculate the number of escape characters.
        // All threads in the warp participate in the reduction, even if `char_idx` is beyond
        // `string_length_block`.
        int8_t const total_escape_char =
          cub::WarpReduce<int8_t>(cub_storage[local_warp_id]).Sum(is_ichar_escape_char);

        if (warp_lane == 0) { escape_char_count += total_escape_char; }

        warp.sync();
      }
    }
    // URL decoding replaces 3 bytes with 1 for each escape character.
    if (warp_lane == 0) { out_counts[row_idx] = string_length - escape_char_count * 2; }
  }
}

/**
 * @brief Decode and copy from the input string column to the output char buffer.
 *
 * @tparam num_warps_per_threadblock Number of warps in a threadblock. This template argument must
 * match the launch configuration, i.e. the kernel must be launched with
 * `num_warps_per_threadblock * cudf::detail::warp_size` threads per threadblock.
 * @tparam char_block_size Number of characters which will be loaded into the shared memory at a
 * time.
 *
 * @param[in] in_strings Input string column.
 * @param[out] out_chars Character buffer for the output string column.
 * @param[in] out_offsets Offset value of each string associated with `out_chars`.
 */
template <size_type num_warps_per_threadblock, size_type char_block_size>
CUDF_KERNEL void url_decode_char_replacer(column_device_view const in_strings,
                                          char* const out_chars,
                                          cudf::detail::input_offsetalator const out_offsets)
{
  constexpr int halo_size = 2;
  __shared__ char temporary_buffer[num_warps_per_threadblock][char_block_size + halo_size * 2];
  __shared__ typename cub::WarpScan<int8_t>::TempStorage cub_storage[num_warps_per_threadblock];
  __shared__ size_type out_idx[num_warps_per_threadblock];

  namespace cg     = cooperative_groups;
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  auto const global_thread_id = cudf::detail::grid_1d::global_thread_id();
  auto const global_warp_id   = static_cast<size_type>(global_thread_id / cudf::detail::warp_size);
  auto const local_warp_id    = static_cast<size_type>(warp.meta_group_rank());
  auto const warp_lane        = static_cast<size_type>(warp.thread_rank());
  auto const nwarps =
    static_cast<size_type>(cudf::detail::grid_1d::grid_stride() / cudf::detail::warp_size);
  char* in_chars_shared = temporary_buffer[local_warp_id];

  // Loop through strings, and assign each string to a warp
  for (thread_index_type tidx = global_warp_id; tidx < in_strings.size(); tidx += nwarps) {
    auto const row_idx = static_cast<size_type>(tidx);
    if (in_strings.is_null(row_idx)) continue;

    auto const in_string     = in_strings.element<string_view>(row_idx);
    auto const in_chars      = in_string.data();
    auto const string_length = in_string.size_bytes();
    auto out_chars_string    = out_chars + out_offsets[row_idx];
    auto const nblocks       = cudf::util::div_rounding_up_unsafe(string_length, char_block_size);

    // Use the last thread of the warp to initialize `out_idx` to 0.
    if (warp_lane == cudf::detail::warp_size - 1) { out_idx[local_warp_id] = 0; }

    for (size_type block_idx = 0; block_idx < nblocks; block_idx++) {
      auto const string_length_block =
        cuda::std::min(char_block_size, string_length - char_block_size * block_idx);

      // Each warp collectively loads input characters of the current block to shared memory.
      // Two halo cells before and after the block are added. The halo cells are used to test
      // whether the current location as well as the previous two locations are escape characters,
      // without branches.
      for (auto char_idx = warp_lane; char_idx < string_length_block + halo_size * 2;
           char_idx += cudf::detail::warp_size) {
        auto const in_idx         = block_idx * char_block_size + char_idx - halo_size;
        in_chars_shared[char_idx] = in_idx >= 0 && in_idx < string_length ? in_chars[in_idx] : 0;
      }

      warp.sync();

      // `char_idx_start` represents the start character index of the current warp.
      for (size_type char_idx_start = 0; char_idx_start < string_length_block;
           char_idx_start += cudf::detail::warp_size) {
        auto const char_idx = char_idx_start + warp_lane;
        // If the current character is part of an escape sequence starting at the previous two
        // locations, the thread with the starting location should output the escaped character, and
        // the current thread should not output a character.
        int8_t const out_size =
          (char_idx >= string_length_block || is_escape_char(in_chars_shared + char_idx) ||
           is_escape_char(in_chars_shared + char_idx + 1))
            ? 0
            : 1;

        // Warp-wise prefix sum to establish output location of the current thread.
        // All threads in the warp participate in the prefix sum, even if `char_idx` is beyond
        // `string_length_block`.
        int8_t out_offset;
        cub::WarpScan<int8_t>(cub_storage[local_warp_id]).ExclusiveSum(out_size, out_offset);

        if (out_size == 1) {
          char const* const ch_ptr = in_chars_shared + char_idx + halo_size;
          char const ch =
            is_escape_char(ch_ptr)
              ?
              // If the current location is the start of an escape sequence, load and decode.
              escaped_sequence_to_byte(ch_ptr)
              :
              // If the current location is not the start of an escape sequence, load directly.
              *ch_ptr;
          out_chars_string[out_idx[local_warp_id] + out_offset] = ch;
        }

        if (warp_lane == cudf::detail::warp_size - 1) {
          out_idx[local_warp_id] += (out_offset + out_size);
        }

        warp.sync();
      }
    }
  }
}

}  // namespace

//
std::unique_ptr<column> url_decode(strings_column_view const& strings,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  constexpr size_type num_warps_per_threadblock = 4;
  constexpr size_type threadblock_size = num_warps_per_threadblock * cudf::detail::warp_size;
  constexpr size_type char_block_size  = 256;
  auto const num_threadblocks =
    std::min(65536, cudf::util::div_rounding_up_unsafe(strings_count, num_warps_per_threadblock));

  auto const d_strings = column_device_view::create(strings.parent(), stream);

  // build offsets column by computing the output row sizes and scanning the results
  auto row_sizes = rmm::device_uvector<size_type>(strings_count, stream);
  url_decode_char_counter<num_warps_per_threadblock, char_block_size>
    <<<num_threadblocks, threadblock_size, 0, stream.value()>>>(*d_strings, row_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  // performs scan on the sizes and builds the appropriate offsets column
  auto [offsets_column, out_chars_bytes] = cudf::strings::detail::make_offsets_child_column(
    row_sizes.begin(), row_sizes.end(), stream, mr);

  // create the chars column
  rmm::device_uvector<char> chars(out_chars_bytes, stream, mr);
  auto d_out_chars = chars.data();
  auto const offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // decode and copy the characters from the input column to the output column
  url_decode_char_replacer<num_warps_per_threadblock, char_block_size>
    <<<num_threadblocks, threadblock_size, 0, stream.value()>>>(*d_strings, d_out_chars, offsets);
  CUDF_CUDA_TRY(cudaGetLastError());

  // copy null mask
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars.release(),
                             strings.null_count(),
                             std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> url_decode(strings_column_view const& input,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::url_decode(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
