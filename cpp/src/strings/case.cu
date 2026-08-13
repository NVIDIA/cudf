/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/case.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/merge.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

// table of special characters whose case equivalents have a different number of bytes;
// there are only 100 such special characters out of the 65536 possible Unicode characters
constexpr size_type NUM_SPECIAL_CHARS = 100;

__constant__ cuda::std::array<uint32_t, NUM_SPECIAL_CHARS> multi_byte_cross_table = {
  0x00c39f, /* (ß,2)->(S,1) 0x000053 */ 0x00c4b0, /* (İ,2)->(i,1) 0x000069 */
  0x00c4b1, /* (ı,2)->(I,1) 0x000049 */ 0x00c5bf, /* (ſ,2)->(S,1) 0x000053 */
  0x00c7b0, /* (ǰ,2)->(J,1) 0x00004a */ 0x00c8ba, /* (Ⱥ,2)->(ⱥ,3) 0xe2b1a5 */
  0x00c8be, /* (Ⱦ,2)->(ⱦ,3) 0xe2b1a6 */ 0x00c8bf, /* (ȿ,2)->(Ȿ,3) 0xe2b1be */
  0x00c980, /* (ɀ,2)->(Ɀ,3) 0xe2b1bf */ 0x00c990, /* (ɐ,2)->(Ɐ,3) 0xe2b1af */
  0x00c991, /* (ɑ,2)->(Ɑ,3) 0xe2b1ad */ 0x00c992, /* (ɒ,2)->(Ɒ,3) 0xe2b1b0 */
  0x00c99c, /* (ɜ,2)->(Ɜ,3) 0xea9eab */ 0x00c9a1, /* (ɡ,2)->(Ɡ,3) 0xea9eac */
  0x00c9a5, /* (ɥ,2)->(Ɥ,3) 0xea9e8d */ 0x00c9a6, /* (ɦ,2)->(Ɦ,3) 0xea9eaa */
  0x00c9aa, /* (ɪ,2)->(Ɪ,3) 0xea9eae */ 0x00c9ab, /* (ɫ,2)->(Ɫ,3) 0xe2b1a2 */
  0x00c9ac, /* (ɬ,2)->(Ɬ,3) 0xea9ead */ 0x00c9b1, /* (ɱ,2)->(Ɱ,3) 0xe2b1ae */
  0x00c9bd, /* (ɽ,2)->(Ɽ,3) 0xe2b1a4 */ 0x00ca87, /* (ʇ,2)->(Ʇ,3) 0xea9eb1 */
  0x00ca9d, /* (ʝ,2)->(Ʝ,3) 0xea9eb2 */ 0x00ca9e, /* (ʞ,2)->(Ʞ,3) 0xea9eb0 */
  0xe1b280, /* (ᲀ,3)->(В,2) 0x00d092 */ 0xe1b281, /* (ᲁ,3)->(Д,2) 0x00d094 */
  0xe1b282, /* (ᲂ,3)->(О,2) 0x00d09e */ 0xe1b283, /* (ᲃ,3)->(С,2) 0x00d0a1 */
  0xe1b284, /* (ᲄ,3)->(Т,2) 0x00d0a2 */ 0xe1b285, /* (ᲅ,3)->(Т,2) 0x00d0a2 */
  0xe1b286, /* (ᲆ,3)->(Ъ,2) 0x00d0aa */ 0xe1b287, /* (ᲇ,3)->(Ѣ,2) 0x00d1a2 */
  0xe1ba96, /* (ẖ,3)->(H,1) 0x000048 */ 0xe1ba97, /* (ẗ,3)->(T,1) 0x000054 */
  0xe1ba98, /* (ẘ,3)->(W,1) 0x000057 */ 0xe1ba99, /* (ẙ,3)->(Y,1) 0x000059 */
  0xe1ba9a, /* (ẚ,3)->(A,1) 0x000041 */ 0xe1ba9e, /* (ẞ,3)->(ß,2) 0x00c39f */
  0xe1bd90, /* (ὐ,3)->(Υ,2) 0x00cea5 */ 0xe1bd92, /* (ὒ,3)->(Υ,2) 0x00cea5 */
  0xe1bd94, /* (ὔ,3)->(Υ,2) 0x00cea5 */ 0xe1bd96, /* (ὖ,3)->(Υ,2) 0x00cea5 */
  0xe1beb3, /* (ᾳ,3)->(Α,2) 0x00ce91 */ 0xe1beb4, /* (ᾴ,3)->(Ά,2) 0x00ce86 */
  0xe1beb6, /* (ᾶ,3)->(Α,2) 0x00ce91 */ 0xe1beb7, /* (ᾷ,3)->(Α,2) 0x00ce91 */
  0xe1bebe, /* (ι,3)->(Ι,2) 0x00ce99 */ 0xe1bf83, /* (ῃ,3)->(Η,2) 0x00ce97 */
  0xe1bf84, /* (ῄ,3)->(Ή,2) 0x00ce89 */ 0xe1bf86, /* (ῆ,3)->(Η,2) 0x00ce97 */
  0xe1bf87, /* (ῇ,3)->(Η,2) 0x00ce97 */ 0xe1bf92, /* (ῒ,3)->(Ι,2) 0x00ce99 */
  0xe1bf93, /* (ΐ,3)->(Ι,2) 0x00ce99 */ 0xe1bf96, /* (ῖ,3)->(Ι,2) 0x00ce99 */
  0xe1bf97, /* (ῗ,3)->(Ι,2) 0x00ce99 */ 0xe1bfa2, /* (ῢ,3)->(Υ,2) 0x00cea5 */
  0xe1bfa3, /* (ΰ,3)->(Υ,2) 0x00cea5 */ 0xe1bfa4, /* (ῤ,3)->(Ρ,2) 0x00cea1 */
  0xe1bfa6, /* (ῦ,3)->(Υ,2) 0x00cea5 */ 0xe1bfa7, /* (ῧ,3)->(Υ,2) 0x00cea5 */
  0xe1bfb3, /* (ῳ,3)->(Ω,2) 0x00cea9 */ 0xe1bfb4, /* (ῴ,3)->(Ώ,2) 0x00ce8f */
  0xe1bfb6, /* (ῶ,3)->(Ω,2) 0x00cea9 */ 0xe1bfb7, /* (ῷ,3)->(Ω,2) 0x00cea9 */
  0xe284a6, /* (Ω,3)->(ω,2) 0x00cf89 */ 0xe284aa, /* (K,3)->(k,1) 0x00006b */
  0xe284ab, /* (Å,3)->(å,2) 0x00c3a5 */ 0xe2b1a2, /* (Ɫ,3)->(ɫ,2) 0x00c9ab */
  0xe2b1a4, /* (Ɽ,3)->(ɽ,2) 0x00c9bd */ 0xe2b1a5, /* (ⱥ,3)->(Ⱥ,2) 0x00c8ba */
  0xe2b1a6, /* (ⱦ,3)->(Ⱦ,2) 0x00c8be */ 0xe2b1ad, /* (Ɑ,3)->(ɑ,2) 0x00c991 */
  0xe2b1ae, /* (Ɱ,3)->(ɱ,2) 0x00c9b1 */ 0xe2b1af, /* (Ɐ,3)->(ɐ,2) 0x00c990 */
  0xe2b1b0, /* (Ɒ,3)->(ɒ,2) 0x00c992 */ 0xe2b1be, /* (Ȿ,3)->(ȿ,2) 0x00c8bf */
  0xe2b1bf, /* (Ɀ,3)->(ɀ,2) 0x00c980 */ 0xea9e8d, /* (Ɥ,3)->(ɥ,2) 0x00c9a5 */
  0xea9eaa, /* (Ɦ,3)->(ɦ,2) 0x00c9a6 */ 0xea9eab, /* (Ɜ,3)->(ɜ,2) 0x00c99c */
  0xea9eac, /* (Ɡ,3)->(ɡ,2) 0x00c9a1 */ 0xea9ead, /* (Ɬ,3)->(ɬ,2) 0x00c9ac */
  0xea9eae, /* (Ɪ,3)->(ɪ,2) 0x00c9aa */ 0xea9eb0, /* (Ʞ,3)->(ʞ,2) 0x00ca9e */
  0xea9eb1, /* (Ʇ,3)->(ʇ,2) 0x00ca87 */ 0xea9eb2, /* (Ʝ,3)->(ʝ,2) 0x00ca9d */
  0xefac80, /* (ﬀ,3)->(F,1) 0x000046 */ 0xefac81, /* (ﬁ,3)->(F,1) 0x000046 */
  0xefac82, /* (ﬂ,3)->(F,1) 0x000046 */ 0xefac83, /* (ﬃ,3)->(F,1) 0x000046 */
  0xefac84, /* (ﬄ,3)->(F,1) 0x000046 */ 0xefac85, /* (ﬅ,3)->(S,1) 0x000053 */
  0xefac86, /* (ﬆ,3)->(S,1) 0x000053 */ 0xefac93, /* (ﬓ,3)->(Մ,2) 0x00d584 */
  0xefac94, /* (ﬔ,3)->(Մ,2) 0x00d584 */ 0xefac95, /* (ﬕ,3)->(Մ,2) 0x00d584 */
  0xefac96, /* (ﬖ,3)->(Վ,2) 0x00d58e */ 0xefac97, /* (ﬗ,3)->(Մ,2) 0x00d584 */
};

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the warp-parallel function is used to compute the output sizes.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the strings_lengths benchmark results.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Utility functions for converting characters to upper or lower case
 */
struct convert_char_fn {
  character_flags_table_type case_flag;
  character_flags_table_type const* d_flags;
  character_cases_table_type const* d_case_table;
  special_case_mapping const* d_special_case_mapping;

  // compute size or copy the bytes representing the special case mapping for this codepoint
  __device__ size_type handle_special_case_bytes(uint32_t code_point,
                                                 detail::character_flags_table_type flag,
                                                 char* d_buffer = nullptr) const
  {
    special_case_mapping m = d_special_case_mapping[get_special_case_hash_index(code_point)];

    size_type bytes   = 0;
    auto const count  = IS_LOWER(flag) ? m.num_upper_chars : m.num_lower_chars;
    auto const* chars = IS_LOWER(flag) ? m.upper : m.lower;
    for (uint16_t idx = 0; idx < count; idx++) {
      bytes += d_buffer
                 ? detail::from_char_utf8(detail::codepoint_to_utf8(chars[idx]), d_buffer + bytes)
                 : detail::bytes_in_char_utf8(detail::codepoint_to_utf8(chars[idx]));
    }
    return bytes;
  }

  // this is called for converting any UTF-8 characters
  __device__ size_type process_character(char_utf8 chr, char* d_buffer = nullptr) const
  {
    auto const code_point = detail::utf8_to_codepoint(chr);

    detail::character_flags_table_type flag = code_point <= 0x00'FFFF ? d_flags[code_point] : 0;

    // we apply special mapping in two cases:
    // - uncased characters with the special mapping flag: always
    // - cased characters with the special mapping flag: when matching the input case_flag
    if (IS_SPECIAL(flag) && ((flag & case_flag) || !IS_UPPER_OR_LOWER(flag))) {
      return handle_special_case_bytes(code_point, case_flag, d_buffer);
    }

    char_utf8 const new_char =
      (flag & case_flag) ? detail::codepoint_to_utf8(d_case_table[code_point]) : chr;
    return (d_buffer) ? detail::from_char_utf8(new_char, d_buffer)
                      : detail::bytes_in_char_utf8(new_char);
  }

  // special function for converting ASCII-only characters
  __device__ char process_ascii(char chr) const
  {
    return (case_flag & d_flags[chr]) ? static_cast<char>(d_case_table[chr]) : chr;
  }
};

/**
 * @brief Per string logic for case conversion functions
 *
 * This can be used in calls to make_strings_children.
 */
struct base_upper_lower_fn {
  convert_char_fn converter;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  base_upper_lower_fn(convert_char_fn converter) : converter(converter) {}

  __device__ inline void process_string(string_view d_str, size_type idx) const
  {
    size_type bytes = 0;
    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (auto itr = d_str.data(); itr < (d_str.data() + d_str.size_bytes()); ++itr) {
      if (is_utf8_continuation_char(static_cast<u_char>(*itr))) continue;
      char_utf8 chr = 0;
      to_char_utf8(itr, chr);
      auto const size = converter.process_character(chr, d_buffer);
      if (d_buffer) {
        d_buffer += size;
      } else {
        bytes += size;
      }
    }
    if (!d_buffer) { d_sizes[idx] = bytes; }
  }
};

struct upper_lower_fn : public base_upper_lower_fn {
  column_device_view d_strings;

  upper_lower_fn(convert_char_fn converter, column_device_view const& d_strings)
    : base_upper_lower_fn{converter}, d_strings{d_strings}
  {
  }

  __device__ void operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str = d_strings.element<string_view>(idx);
    process_string(d_str, idx);
  }
};

// Long strings are divided into smaller strings using this value as a guide.
// Generally strings are split into sub-blocks of bytes of this size but
// care is taken to not sub-block in the middle of a multi-byte character.
constexpr size_type LS_SUB_BLOCK_SIZE = 32;

/**
 * @brief Produces sub-offsets for the chars in the given strings column
 */
struct sub_offset_fn {
  char const* d_input_chars;
  int64_t first_offset;
  int64_t last_offset;

  __device__ int64_t operator()(int64_t idx) const
  {
    auto const end = d_input_chars + last_offset;
    auto position  = (idx + 1) * LS_SUB_BLOCK_SIZE;
    auto begin     = d_input_chars + first_offset + position;
    while ((begin < end) && is_utf8_continuation_char(static_cast<u_char>(*begin))) {
      ++begin;
      ++position;
    }
    return (begin < end) ? position + first_offset : last_offset;
  }
};

/**
 * @brief Specialized case conversion for long strings
 *
 * This is needed since the offset count can exceed size_type.
 * Also, nulls are ignored since this purely builds the output chars.
 * The d_offsets are only temporary to help address the sub-blocks.
 */
struct upper_lower_ls_fn : public base_upper_lower_fn {
  convert_char_fn converter;
  char const* d_input_chars;
  int64_t* d_input_offsets;  // includes column offset

  upper_lower_ls_fn(convert_char_fn converter, char const* d_input_chars, int64_t* d_input_offsets)
    : base_upper_lower_fn{converter}, d_input_chars{d_input_chars}, d_input_offsets{d_input_offsets}
  {
  }

  // idx is row index
  __device__ void operator()(size_type idx) const
  {
    auto const offset = d_input_offsets[idx];
    auto const d_str  = string_view{d_input_chars + offset,
                                   static_cast<size_type>(d_input_offsets[idx + 1] - offset)};
    process_string(d_str, idx);
  }
};

constexpr int64_t block_size = 512;

/**
 * @brief Count output bytes in warp-parallel threads
 *
 * This executes as one warp per string and just computes the output sizes.
 */
template <int bytes_per_thread>
CUDF_KERNEL void count_bytes_kernel(convert_char_fn converter,
                                    column_device_view d_strings,
                                    size_type* d_sizes)
{
  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / cudf::detail::warp_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) {
    if (lane_idx == 0) { d_sizes[str_idx] = 0; }
    return;
  }

  auto const d_str   = d_strings.element<string_view>(str_idx);
  auto const str_ptr = d_str.data();

  // each thread processes bytes_per_thread bytes
  size_type size = 0;
  for (auto i = lane_idx * bytes_per_thread; i < d_str.size_bytes();
       i += cudf::detail::warp_size * bytes_per_thread) {
    for (auto j = i; (j < (i + bytes_per_thread)) && (j < d_str.size_bytes()); j++) {
      auto const chr = str_ptr[j];
      if (is_utf8_continuation_char(chr)) { continue; }
      char_utf8 u8 = 0;
      to_char_utf8(str_ptr + j, u8);
      size += converter.process_character(u8);
    }
  }

  auto out_size = cg::reduce(warp, size, cg::plus<size_type>());
  if (lane_idx == 0) { d_sizes[str_idx] = out_size; }
}

/**
 * @brief Counts the number of special multi-byte characters in the input
 *
 * This is used to determine if the input contains special multi-byte characters
 * that needs to be handled by the slower warp-parallel algorithm.
 */
template <int bytes_per_thread>
CUDF_KERNEL void mismatch_multibytes_kernel(char const* d_input_chars,
                                            int64_t first_offset,
                                            int64_t last_offset,
                                            int64_t* d_output)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();

  __shared__ uint32_t mb_table[NUM_SPECIAL_CHARS];
  auto const lane_idx = threadIdx.x;
  if (lane_idx < NUM_SPECIAL_CHARS) { mb_table[lane_idx] = multi_byte_cross_table[lane_idx]; }
  __syncthreads();

  auto const byte_idx = (idx * bytes_per_thread) + first_offset;

  auto count = 0;
  for (auto i = byte_idx; (i < (byte_idx + bytes_per_thread)) && (i < last_offset); i++) {
    u_char const chr = d_input_chars[i];
    if (chr < 0x080 || is_utf8_continuation_char(chr)) { continue; }
    char_utf8 utf8 = 0;
    to_char_utf8(d_input_chars + i, utf8);
    count += (thrust::binary_search(thrust::seq, mb_table, mb_table + NUM_SPECIAL_CHARS, utf8));
  }

  if (count > 0) {
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref{*d_output};
    ref.fetch_add(count, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Special kernel for processing UTF-8 characters whose
 * byte counts match their case equivalent (including ASCII)
 */
template <int bytes_per_thread>
CUDF_KERNEL void multibyte_converter_kernel(convert_char_fn converter,
                                            char const* d_input_chars,
                                            int64_t chars_size,
                                            char* d_output_chars)
{
  auto const idx      = cudf::detail::grid_1d::global_thread_id();
  auto const char_idx = (idx * bytes_per_thread);
  if (char_idx >= chars_size) { return; }

  for (auto i = char_idx; i < (char_idx + bytes_per_thread) && i < chars_size; ++i) {
    u_char const chr = d_input_chars[i];
    auto d_buffer    = d_output_chars + i;
    if (chr < 0x080) {
      *d_buffer = converter.process_ascii(chr);
      continue;
    }
    if (is_utf8_continuation_char(chr)) { continue; }
    char_utf8 utf8 = 0;
    to_char_utf8(d_input_chars + i, utf8);
    converter.process_character(utf8, d_buffer);
  }
}

/**
 * @brief Utility method for converting upper and lower case characters
 * in a strings column
 *
 * @param input Strings to convert
 * @param case_flag The character type to convert (upper, lower, or both)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with characters converted
 */
std::unique_ptr<column> convert_case(strings_column_view const& input,
                                     character_flags_table_type case_flag,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  if (input.size() == input.null_count()) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_flags   = get_character_flags_table(stream);
  auto const d_cases   = get_character_cases_table(stream);
  auto const d_special = get_special_case_mapping_table(stream);

  auto const first_offset = (input.offset() == 0) ? 0L
                                                  : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset =
    cudf::strings::detail::get_offset_value(input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size  = last_offset - first_offset;
  auto const input_chars = input.chars_begin(stream);

  convert_char_fn ccfn{case_flag, d_flags, d_cases, d_special};
  upper_lower_fn converter{ccfn, *d_strings};

  // For smaller strings, use the regular string-parallel algorithm
  if ((chars_size / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    auto [offsets, chars] = make_strings_children(converter, input.size(), stream, mr);
    return make_strings_column(input.size(),
                               std::move(offsets),
                               chars.release(),
                               input.null_count(),
                               cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  // Check if the input contains special multi-byte characters where the case equivalent
  // has a different number of bytes. There are only 100 such special characters.
  // This check incurs ~20% performance hit for smaller strings and so we only use it
  // after the threshold check above. The check makes very little impact for long strings
  // but results in a large performance gain when the input contains no special characters.
  constexpr int64_t bytes_per_thread = 4;
  cudf::detail::device_scalar<int64_t> mb_count(0, stream, cudf::get_current_device_resource_ref());
  auto const grid = cudf::detail::grid_1d(chars_size, block_size, bytes_per_thread);
  mismatch_multibytes_kernel<bytes_per_thread>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      input_chars, first_offset, last_offset, mb_count.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  if (mb_count.value(stream) == 0) {
    // optimization for the non-special case;
    // copying the input column automatically handles normalizing sliced inputs
    // and nulls properly but also does incur a wasteful chars copy too
    auto result  = std::make_unique<column>(input.parent(), stream, mr);
    auto d_chars = result->mutable_view().head<char>();
    multibyte_converter_kernel<bytes_per_thread>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        ccfn, input_chars + first_offset, chars_size, d_chars);
    CUDF_CUDA_TRY(cudaGetLastError());
    result->set_null_count(input.null_count());
    return result;
  }

  // This will use a warp-parallel algorithm to compute the output sizes for each string
  // note: tried to use segmented-reduce approach instead here and it was consistently slower
  auto [offsets, bytes] = [&] {
    rmm::device_uvector<size_type> sizes(input.size(), stream);
    constexpr thread_index_type warp_size = cudf::detail::warp_size;
    auto grid = cudf::detail::grid_1d(input.size() * warp_size, block_size);
    count_bytes_kernel<bytes_per_thread>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        ccfn, *d_strings, sizes.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    // convert sizes to offsets
    return cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  }();

  // build sub-offsets
  auto const sub_count = chars_size / LS_SUB_BLOCK_SIZE;
  auto tmp_offsets     = rmm::device_uvector<int64_t>(sub_count + input.size() + 1, stream);
  {
    rmm::device_uvector<int64_t> sub_offsets(sub_count, stream);
    auto const count_itr = cuda::counting_iterator<int64_t>{0};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      count_itr,
                      count_itr + sub_count,
                      sub_offsets.data(),
                      sub_offset_fn{input_chars, first_offset, last_offset});

    // merge them with input offsets
    auto input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    thrust::merge(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  input_offsets,
                  input_offsets + input.size() + 1,
                  sub_offsets.begin(),
                  sub_offsets.end(),
                  tmp_offsets.begin());
    stream.synchronize();  // protect against destruction of sub_offsets
  }

  // run case conversion over the new sub-strings
  auto const tmp_size = static_cast<size_type>(tmp_offsets.size()) - 1;
  upper_lower_ls_fn sub_conv{ccfn, input_chars, tmp_offsets.data()};
  auto chars = std::get<1>(make_strings_children(sub_conv, tmp_size, stream, mr));

  return make_strings_column(input.size(),
                             std::move(offsets),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> to_lower(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  character_flags_table_type case_flag = IS_UPPER(0xFF);  // convert only upper case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> to_upper(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  character_flags_table_type case_flag = IS_LOWER(0xFF);  // convert only lower case characters
  return convert_case(strings, case_flag, stream, mr);
}

//
std::unique_ptr<column> swapcase(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  // convert only upper or lower case characters
  character_flags_table_type case_flag = IS_LOWER(0xFF) | IS_UPPER(0xFF);
  return convert_case(strings, case_flag, stream, mr);
}

}  // namespace detail

// APIs

std::unique_ptr<column> to_lower(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_lower(strings, stream, mr);
}

std::unique_ptr<column> to_upper(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_upper(strings, stream, mr);
}

std::unique_ptr<column> swapcase(strings_column_view const& strings,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::swapcase(strings, stream, mr);
}

}  // namespace strings
}  // namespace cudf
