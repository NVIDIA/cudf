/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/detail/codepoint_metadata.ah"
#include "text/normalize.cuh"
#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/normalize.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/bit>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/scan.h>

#include <array>
#include <limits>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Normalize spaces in a strings column.
 *
 * Repeated whitespace (code-point <= ' ') is replaced with a single space.
 * Also, whitespace is trimmed from the beginning and end of each string.
 *
 * This functor can be called to compute the output size in bytes
 * of each string and then called again to fill in the allocated buffer.
 */
struct normalize_spaces_fn {
  cudf::column_device_view const d_strings;  // strings to normalize
  cudf::size_type* d_sizes{};                // size of each output row
  char* d_chars{};                           // output buffer for characters
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    cudf::string_view const single_space(" ", 1);
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    char* buffer     = d_chars ? d_chars + d_offsets[idx] : nullptr;
    char* optr       = buffer;  // running output pointer

    cudf::size_type nbytes = 0;  // holds the number of bytes per output string

    // create a tokenizer for this string with whitespace delimiter (default)
    characters_tokenizer tokenizer(d_str);

    // this will retrieve tokens automatically skipping runs of whitespace
    while (tokenizer.next_token()) {
      auto const token_pos = tokenizer.token_byte_positions();
      auto const token =
        cudf::string_view(d_str.data() + token_pos.first, token_pos.second - token_pos.first);
      if (optr) {
        // prepend space unless we are at the beginning
        if (optr != buffer) { optr = cudf::strings::detail::copy_string(optr, single_space); }
        // write token to output buffer
        thrust::copy_n(thrust::seq, token.data(), token.size_bytes(), optr);
        optr += token.size_bytes();
      }
      nbytes += token.size_bytes() + 1;  // token size plus a single space
    }
    // remove trailing space
    if (!d_chars) { d_sizes[idx] = (nbytes > 0) ? nbytes - 1 : 0; }
  }
};

/**
 * @brief Converts a codepoint to UTF-8 with the bytes in memory order
 *
 * The first UTF-8 byte is in the least significant byte of the result
 * so the result can be stored in memory as a UTF-8 encoded character.
 */
__device__ uint32_t cp_to_utf8(uint32_t codepoint)
{
  auto const utf8  = cudf::strings::detail::codepoint_to_utf8(codepoint);
  auto const bytes = cudf::strings::detail::bytes_in_char_utf8(utf8);
  return cuda::std::byteswap(utf8) >> (8 * (4 - bytes));
}

}  // namespace

// detail API
std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& strings,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  // create device column
  auto d_strings = cudf::column_device_view::create(strings.parent(), stream);

  // build offsets and children using the normalize_space_fn
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    normalize_spaces_fn{*d_strings}, strings.size(), stream, mr);

  return cudf::make_strings_column(strings.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   strings.null_count(),
                                   cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

/**
 * @brief Retrieve the code point metadata table.
 *
 * Build the code point metadata table in device memory
 * using the vector pieces from codepoint_metadata.ah
 */
cuda::device_buffer<codepoint_metadata_type> get_codepoint_metadata(cuda::stream_ref stream)
{
  auto table_vector = cuda::device_buffer<codepoint_metadata_type>{
    stream, cudf::get_current_device_resource_ref(), codepoint_metadata_size, cuda::no_init};
  auto table = table_vector.data();
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               table + cp_section1_end,
               table + codepoint_metadata_size,
               codepoint_metadata_default_value);
  auto dsts  = std::array<void*, 2>{table, table + cp_section2_begin};
  auto srcs  = std::array<void const*, 2>{codepoint_metadata, cp_metadata_917505_917999};
  auto sizes = std::array<std::size_t, 2>{
    cp_section1_end * sizeof(codepoint_metadata[0]),
    (cp_section2_end - cp_section2_begin + 1) * sizeof(codepoint_metadata[0])};
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_batch_async(dsts.data(), srcs.data(), sizes.data(), 2, stream));
  return table_vector;
}

/**
 * @brief Retrieve the aux code point data table.
 *
 * Build the aux code point data table in device memory
 * using the vector pieces from codepoint_metadata.ah
 */
cuda::device_buffer<aux_codepoint_data_type> get_aux_codepoint_data(cuda::stream_ref stream)
{
  auto table_vector = cuda::device_buffer<aux_codepoint_data_type>{
    stream, cudf::get_current_device_resource_ref(), aux_codepoint_data_size, cuda::no_init};
  auto table = table_vector.data();
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               table + aux_section1_end,
               table + aux_codepoint_data_size,
               aux_codepoint_default_value);
  auto dsts = std::array<void*, 4>{
    table, table + aux_section2_begin, table + aux_section3_begin, table + aux_section4_begin};
  auto srcs  = std::array<void const*, 4>{aux_codepoint_data,
                                          aux_cp_data_44032_55203,
                                          aux_cp_data_70475_71099,
                                          aux_cp_data_119134_119232};
  auto sizes = std::array<std::size_t, 4>{
    aux_section1_end * sizeof(aux_codepoint_data[0]),
    (aux_section2_end - aux_section2_begin + 1) * sizeof(aux_codepoint_data[0]),
    (aux_section3_end - aux_section3_begin + 1) * sizeof(aux_codepoint_data[0]),
    (aux_section4_end - aux_section4_begin + 1) * sizeof(aux_codepoint_data[0])};
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_batch_async(dsts.data(), srcs.data(), sizes.data(), 4, stream));
  return table_vector;
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> normalize_spaces(cudf::strings_column_view const& input,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_spaces(input, stream, mr);
}

struct character_normalizer::character_normalizer_impl {
  cuda::device_buffer<codepoint_metadata_type> cp_metadata;
  cuda::device_buffer<aux_codepoint_data_type> aux_table;
  bool do_lower_case;
  std::unique_ptr<cudf::column> special_tokens;
  rmm::device_uvector<cudf::string_view> special_tokens_view;

  cudf::device_span<cudf::string_view const> get_special_tokens() const
  {
    return special_tokens_view;
  }

  character_normalizer_impl(cuda::device_buffer<codepoint_metadata_type>&& cp_metadata,
                            cuda::device_buffer<aux_codepoint_data_type>&& aux_table,
                            bool do_lower_case,
                            std::unique_ptr<cudf::column>&& special_tokens,
                            rmm::device_uvector<cudf::string_view>&& special_tokens_view)
    : cp_metadata(std::move(cp_metadata)),
      aux_table(std::move(aux_table)),
      do_lower_case{do_lower_case},
      special_tokens{std::move(special_tokens)},
      special_tokens_view{std::move(special_tokens_view)}
  {
  }
};

character_normalizer::character_normalizer(bool do_lower_case,
                                           cudf::strings_column_view const& special_tokens,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref)
{
  auto cp_metadata = nvtext::detail::get_codepoint_metadata(stream);
  auto aux_table   = nvtext::detail::get_aux_codepoint_data(stream);
  CUDF_EXPECTS(
    !special_tokens.has_nulls(), "special tokens should not have nulls", std::invalid_argument);

  auto sorted = std::move(
    cudf::sort(cudf::table_view({special_tokens.parent()}), {}, {}, stream)->release().front());
  if (do_lower_case) {
    // lower-case the tokens so they will match the normalized input
    sorted = cudf::strings::to_lower(cudf::strings_column_view(sorted->view()), stream);
  }

  auto tokens_view = cudf::strings::detail::create_string_vector_from_column(
    cudf::strings_column_view(sorted->view()), stream, cudf::get_current_device_resource_ref());

  _impl = std::make_unique<character_normalizer_impl>(std::move(cp_metadata),
                                                      std::move(aux_table),
                                                      do_lower_case,
                                                      std::move(sorted),
                                                      std::move(tokens_view));
}

character_normalizer::~character_normalizer() {}

std::unique_ptr<character_normalizer> create_character_normalizer(
  bool do_lower_case,
  cudf::strings_column_view const& special_tokens,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<character_normalizer>(do_lower_case, special_tokens, stream, mr);
}

namespace detail {
namespace {

constexpr int64_t block_size = 256;

/// Number of input bytes searched for the closing `]` of a special token
constexpr int64_t special_token_window = 6;

// Highest codepoint that maps to a plain ASCII result via get_first_cp
constexpr uint32_t ASCII_MAX_CODEPOINT = 0x7Fu;
// The char_flags table covers exactly the Basic Multilingual Plane
constexpr uint32_t BMP_CODEPOINT_LIMIT = 0x10000u;

/**
 * @brief Normalizes the characters of the input
 *
 * The normalized result for each input byte is up to MAX_NEW_CHARS UTF-8 encoded
 * characters with each stored in its own uint32_t slot.
 * Slots with a 0 value are not part of the output.
 * All slots are 0 for bytes that are not the first byte of a UTF-8 character.
 */
struct normalize_fn {
  char const* d_chars;
  int64_t total_bytes;
  codepoint_metadata_type const* cp_metadata;
  aux_codepoint_data_type const* aux_table;
  bool do_lower_case;
  bool strip_accents;
  bool pad_punctuation;
  cudf::strings::detail::character_flags_table_type const* char_flags;
  cudf::device_span<cudf::string_view const> special_tokens;
  uint8_t const* d_matches;  ///< results of find_special_token for each byte; nullptr if none

  /**
   * @brief Normalizes the character starting at byte `idx`
   *
   * @param idx Byte position of the input characters
   * @param replacement Output slots for the normalized character
   */
  __device__ void normalize(int64_t idx, uint32_t* replacement) const
  {
    for (uint32_t k = 0; k < MAX_NEW_CHARS; ++k) {
      replacement[k] = 0;
    }
    if ((idx >= total_bytes) || !cudf::strings::detail::is_begin_utf8_char(d_chars[idx])) {
      return;
    }

    auto const cp = [utf8 = d_chars + idx] {
      cudf::char_utf8 ch_utf8 = 0;
      cudf::strings::detail::to_char_utf8(utf8, ch_utf8);
      return cudf::strings::detail::utf8_to_codepoint(ch_utf8);
    }();
    auto const metadata = cp_metadata[cp];

    if (should_remove_cp(metadata, do_lower_case, strip_accents)) { return; }

    int8_t num_new_chars = 1;
    // retrieve the normalized value for cp
    uint32_t new_cp = [this, metadata, cp] {
      if (do_lower_case || always_replace(metadata)) { return get_first_cp(metadata); }
      if (!strip_accents) { return 0u; }
      // Use the de-accented ASCII result when available; re-uppercase if needed
      auto const mapped = get_first_cp(metadata);
      if (mapped == 0 || mapped > ASCII_MAX_CODEPOINT) { return 0u; }
      auto const flag = cp < BMP_CODEPOINT_LIMIT ? char_flags[cp] : uint8_t{0};
      return cudf::strings::detail::IS_UPPER(flag) ? (mapped - 'a' + 'A') : mapped;
    }();
    replacement[0] = new_cp == 0 ? cp : new_cp;

    if (do_lower_case && is_multi_char_transform(metadata)) {
      auto const next_cps = aux_table[cp];
      replacement[1]      = static_cast<uint32_t>(next_cps >> 32);
      replacement[2]      = static_cast<uint32_t>(next_cps & 0xFFFFFFFF);
      num_new_chars       = 2 + (replacement[2] != 0);
    }

    if (should_add_spaces(metadata, do_lower_case, pad_punctuation) && (num_new_chars == 1)) {
      replacement[1] = replacement[0];
      replacement[0] = SPACE_CODE_POINT;  // add spaces around the new codepoint
      replacement[2] = SPACE_CODE_POINT;
      num_new_chars  = 3;
    }

    // convert codepoints back to UTF-8 in-place
    for (int k = 0; k < num_new_chars; ++k) {
      auto const new_cp = replacement[k];
      if (new_cp) { replacement[k] = cp_to_utf8(new_cp); }
    }
  }

  /**
   * @brief Checks for a special token beginning at byte `idx`
   *
   * A special token candidate begins with a `[` and ends with the first `]`
   * found within the next `special_token_window` bytes. The candidate is
   * built from the normalized slots between and including the `[]` characters.
   *
   * @param idx Byte position of the input characters
   * @return Number of slots from the `[` to the `]` if the candidate
   *         matches one of the `special_tokens`; 0 otherwise
   */
  __device__ uint8_t find_special_token(int64_t idx) const
  {
    // only the '[' character is normalized into '['
    if (d_chars[idx] != '[') { return 0; }

    auto const begin = static_cast<int64_t>(idx * MAX_NEW_CHARS + 1);  // slot of the '['
    auto const end =
      begin + cuda::std::min(int64_t{special_token_window}, total_bytes - idx) * MAX_NEW_CHARS;

    // only the ']' character is normalized into ']' so check for one before normalizing
    auto const last = cuda::std::min(idx + special_token_window, total_bytes);
    if (thrust::find(thrust::seq, d_chars + idx + 1, d_chars + last, ']') == d_chars + last) {
      return 0;
    }

    char candidate[special_token_window * MAX_NEW_CHARS];
    cudf::size_type size = 0;
    uint32_t slots[MAX_NEW_CHARS];
    for (auto t = begin; t < end; ++t) {
      if ((t == begin) || (t % MAX_NEW_CHARS) == 0) { normalize(t / MAX_NEW_CHARS, slots); }
      auto const value = slots[t % MAX_NEW_CHARS];
      if (t == begin && value != '[') { return 0; }
      auto const ch = static_cast<char>(value);
      if (ch != 0 && ch != ' ') { candidate[size++] = ch; }
      if (value == ']') {
        auto const token = cudf::string_view(candidate, size);
        // the binary_search expects the special_tokens to be sorted
        auto const found =
          thrust::binary_search(thrust::seq, special_tokens.begin(), special_tokens.end(), token);
        return found ? static_cast<uint8_t>(t - begin) : 0;
      }
    }
    return 0;
  }

  /**
   * @brief Undoes the padding added around the `[]` for any special tokens
   *
   * The space added after the `[` and the space added before the `]` are removed
   * for each matched special token. If `do_lower_case==true`, the characters in
   * between are also converted back to upper-case.
   *
   * Only the slots for byte `idx` are updated so this checks any `[` within the
   * previous `special_token_window` bytes since their tokens may include `idx`.
   *
   * @param idx Byte position of the input characters
   * @param replacement The normalized slots for `idx` to be updated
   * @param matches Results of find_special_token for byte positions starting at `base`
   * @param base Byte position of the first element in `matches`
   */
  __device__ void fix_special_tokens(int64_t idx,
                                     uint32_t* replacement,
                                     uint8_t const* matches,
                                     int64_t base) const
  {
    auto const first = cuda::std::max(int64_t{0}, idx - special_token_window + 1);
    for (auto i = idx; i >= first; --i) {
      auto const distance = matches[i - base];
      if (distance == 0) { continue; }
      auto const begin = static_cast<int64_t>(i * MAX_NEW_CHARS + 1);  // slot of the '['
      auto const match = begin + distance;                             // slot of the ']'
      for (uint32_t k = 0; k < MAX_NEW_CHARS; ++k) {
        auto const t = static_cast<int64_t>(idx * MAX_NEW_CHARS + k);
        if (t == begin + 1 || t == match - 1) {
          replacement[k] = 0;
        } else if (do_lower_case && t >= begin + 2 && t < match - 2) {
          auto const ch = replacement[k];
          if (ch >= 'a' && ch <= 'z') { replacement[k] = ch - 'a' + 'A'; }
        }
      }
    }
  }

  /**
   * @brief Returns true if any special token begins within the window of `idx`
   *
   * The window is the previous `special_token_window` bytes up to and including `idx`.
   * The `matches` must be 8-byte aligned and padded so the window can be read
   * using 64-bit words.
   *
   * @param idx Byte position of the input characters
   * @param matches Results of find_special_token for byte positions starting at `base`
   * @param base Byte position of the first element in `matches`
   */
  __device__ bool has_matches(int64_t idx, uint8_t const* matches, int64_t base) const
  {
    auto const offset = idx - (special_token_window - 1) - base;
    auto const words  = reinterpret_cast<uint64_t const*>(matches);
    auto const shift  = static_cast<uint32_t>(offset % 8) * 8;
    auto const lo     = words[offset / 8];
    auto const window = shift == 0 ? lo : ((lo >> shift) | (words[offset / 8 + 1] << (64 - shift)));
    constexpr uint64_t window_mask = (uint64_t{1} << (special_token_window * 8)) - 1;
    return (window & window_mask) != 0;
  }

  /**
   * @brief Computes the normalized slots for byte `idx`
   *
   * @param idx Byte position of the input characters
   * @param replacement Output slots for the normalized character
   * @param matches Results of find_special_token for byte positions starting at `base`
   * @param base Byte position of the first element in `matches`
   * @return Number of UTF-8 bytes in the output slots
   */
  __device__ int32_t operator()(int64_t idx,
                                uint32_t* replacement,
                                uint8_t const* matches,
                                int64_t base) const
  {
    normalize(idx, replacement);
    if ((idx < total_bytes) && (d_matches != nullptr) && has_matches(idx, matches, base)) {
      fix_special_tokens(idx, replacement, matches, base);
    }
    // count the non-zero bytes: __vcmpne4 sets each non-zero byte to 0xFF
    int32_t size = 0;
    for (uint32_t k = 0; k < MAX_NEW_CHARS; ++k) {
      size += __popc(__vcmpne4(replacement[k], 0)) / 8;
    }
    return size;
  }
};

/// Number of find_special_token results loaded by each block (multiple of 8)
constexpr int64_t block_matches_size = block_size + 16;

/**
 * @brief Loads the find_special_token results needed by this block into shared memory
 *
 * The results start 8 bytes before the block's first byte position so the
 * results for each thread's `special_token_window` can be read as 64-bit words.
 *
 * @param fn Normalizes each input byte
 * @param block_matches Shared memory for the results
 * @return Byte position of the first element in `block_matches`
 */
__device__ int64_t load_block_matches(normalize_fn const& fn, uint8_t* block_matches)
{
  auto const base = static_cast<int64_t>(blockIdx.x) * block_size - 8;
  if (fn.d_matches != nullptr) {
    for (auto i = static_cast<int64_t>(threadIdx.x); i < block_matches_size; i += block_size) {
      auto const pos   = base + i;
      block_matches[i] = (pos >= 0 && pos < fn.total_bytes) ? fn.d_matches[pos] : 0;
    }
    __syncthreads();
  }
  return base;
}

/**
 * @brief Computes the normalized output size for each input byte
 *
 * Launched as a thread per input byte (total_bytes).
 *
 * @param fn Normalizes each input byte
 * @param d_sizes Output size of each input byte
 * @param d_block_sizes Total output size for each block
 */
CUDF_KERNEL void normalized_sizes_kernel(normalize_fn fn, uint8_t* d_sizes, int64_t* d_block_sizes)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();

  __shared__ alignas(8) uint8_t block_matches[block_matches_size];
  auto const base = load_block_matches(fn, block_matches);

  uint32_t replacement[MAX_NEW_CHARS];
  auto const size = fn(idx, replacement, block_matches, base);
  if (idx < fn.total_bytes) { d_sizes[idx] = static_cast<uint8_t>(size); }

  // CUB is used for the block-wide collectives in these kernels since the cooperative_groups
  // equivalents require a multi-warp tile which measured slower and reports racecheck hazards
  using block_reduce = cub::BlockReduce<int32_t, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const block_total = block_reduce(temp_storage).Sum(size);
  if (threadIdx.x == 0) { d_block_sizes[blockIdx.x] = block_total; }
}

/**
 * @brief Writes the normalized output for each input byte
 *
 * Launched as a thread per input byte (total_bytes).
 *
 * The output for each block is assembled in shared memory and
 * then written to `d_output` at the block's offset.
 *
 * @param fn Normalizes each input byte
 * @param d_block_offsets Output offset for each block
 * @param d_output Normalized output characters
 */
CUDF_KERNEL void normalized_chars_kernel(normalize_fn fn,
                                         int64_t const* d_block_offsets,
                                         char* d_output)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();

  __shared__ alignas(8) uint8_t block_matches[block_matches_size];
  auto const base = load_block_matches(fn, block_matches);

  uint32_t replacement[MAX_NEW_CHARS];
  auto const size = fn(idx, replacement, block_matches, base);

  // cub::BlockScan also returns the block total needed to write the block's output
  using block_scan = cub::BlockScan<int32_t, block_size>;
  __shared__ typename block_scan::TempStorage temp_storage;
  __shared__ char block_output[block_size * MAX_NEW_CHARS * sizeof(uint32_t)];

  int32_t offset      = 0;
  int32_t block_total = 0;
  block_scan(temp_storage).ExclusiveSum(size, offset, block_total);

  // UTF-8 bytes are stored in order in each slot followed by any zero padding
  auto out = block_output + offset;
  for (uint32_t k = 0; k < MAX_NEW_CHARS; ++k) {
    for (auto v = replacement[k]; v != 0; v >>= 8) {
      *out++ = static_cast<char>(v & 0xFF);
    }
  }
  __syncthreads();

  auto const d_block_output = d_output + d_block_offsets[blockIdx.x];
  for (auto i = static_cast<int32_t>(threadIdx.x); i < block_total; i += block_size) {
    d_block_output[i] = block_output[i];
  }
}

/**
 * @brief Computes the output sizes for each row
 *
 * The input offsets are used with segmented-reduce to sum the
 * output sizes of each input byte for each output row.
 *
 * @param d_sizes The output size of each input byte
 * @param offsets These identify the row boundaries
 * @param offset Only non-zero if the input column has been sliced
 * @param size The number of output rows (sames as the number of input rows)
 * @param stream Stream used for allocating device memory and launching kernels
 * @return The sizes of each output row
 */
template <typename OffsetType>
cuda::device_buffer<cudf::size_type> compute_sizes(cudf::device_span<uint8_t const> d_sizes,
                                                   OffsetType offsets,
                                                   int64_t offset,
                                                   cudf::size_type size,
                                                   cuda::stream_ref stream)
{
  auto output_sizes = cuda::device_buffer<cudf::size_type>{
    stream, cudf::get_current_device_resource_ref(), static_cast<std::size_t>(size), cuda::no_init};

  // DeviceSegmentedReduce is used to compute the size of each output row;
  // the uint8 sizes are accumulated using the output type
  auto const env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  auto const d_in  = d_sizes.data();
  auto const d_out = output_sizes.data();
  if (offset == 0) {
    CUDF_CUDA_TRY(cub::DeviceSegmentedReduce::Sum(d_in, d_out, size, offsets, offsets + 1, env));
  } else {
    // offsets need to be normalized for segmented-reduce to work efficiently
    auto offsets_itr = cuda::transform_iterator(
      offsets,
      cuda::proclaim_return_type<int64_t>([offset] __device__(auto o) { return o - offset; }));
    CUDF_CUDA_TRY(
      cub::DeviceSegmentedReduce::Sum(d_in, d_out, size, offsets_itr, offsets_itr + 1, env));
  }

  return output_sizes;
}

}  // namespace

std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   character_normalizer const& normalizer,
                                                   bool strip_accents,
                                                   bool pad_punctuation,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  if (chars_size == 0) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }

  auto const& parameters = normalizer._impl;

  // char_flags only needed when stripping accents without lowercasing (re-uppercase logic)
  auto const char_flags = (strip_accents && !parameters->do_lower_case)
                            ? cudf::strings::detail::get_character_flags_table(stream)
                            : nullptr;

  // Special tokens only need fixing when pad_punctuation=true since otherwise
  // no spaces are inserted around the `[]` characters
  auto const special_tokens = pad_punctuation ? parameters->get_special_tokens()
                                              : cudf::device_span<cudf::string_view const>{};

  auto fn = normalize_fn{d_input_chars,
                         chars_size,
                         parameters->cp_metadata.data(),
                         parameters->aux_table.data(),
                         parameters->do_lower_case,
                         strip_accents,
                         pad_punctuation,
                         char_flags,
                         special_tokens,
                         nullptr};

  cudf::detail::grid_1d grid{chars_size, block_size};

  // locate any special tokens so their padding can be removed when normalizing
  auto d_matches =
    cuda::device_buffer<uint8_t>{stream,
                                 cudf::get_current_device_resource_ref(),
                                 static_cast<std::size_t>(special_tokens.empty() ? 0 : chars_size),
                                 cuda::no_init};
  if (!special_tokens.empty()) {
    CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
      cuda::counting_iterator<int64_t>{0},
      d_matches.data(),
      chars_size,
      [fn] __device__(int64_t idx) -> uint8_t { return fn.find_special_token(idx); },
      stream.get()));
    fn.d_matches = d_matches.data();
  }

  // compute the output size for each input byte and the total output size for each block
  auto d_block_offsets = cuda::device_buffer<int64_t>{stream,
                                                      cudf::get_current_device_resource_ref(),
                                                      static_cast<std::size_t>(grid.num_blocks),
                                                      cuda::no_init};
  auto output_sizes    = [&] {
    auto d_sizes = cuda::device_buffer<uint8_t>{stream,
                                                cudf::get_current_device_resource_ref(),
                                                static_cast<std::size_t>(chars_size),
                                                cuda::no_init};
    normalized_sizes_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      fn, d_sizes.data(), d_block_offsets.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    // Use segmented-reduce over the byte sizes to get the size of the output rows
    auto const input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    return compute_sizes(cudf::device_span<uint8_t const>(d_sizes.data(), d_sizes.size()),
                         input_offsets,
                         first_offset,
                         input.size(),
                         stream);
  }();

  // convert the block sizes to block offsets
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         d_block_offsets.data(),
                         d_block_offsets.data() + d_block_offsets.size(),
                         d_block_offsets.data());

  // convert the sizes to offsets
  auto [offsets, total_size] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.data(), output_sizes.data() + output_sizes.size(), stream, mr);

  // write the normalized characters
  auto chars = rmm::device_uvector<char>(total_size, stream, mr);
  normalized_chars_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    fn, d_block_offsets.data(), chars.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   character_normalizer const& normalizer,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_characters(input, normalizer, false, true, stream, mr);
}

std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   character_normalizer const& normalizer,
                                                   normalize_flags flags,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const strip    = static_cast<bool>(flags & normalize_flags::STRIP_ACCENTS);
  auto const tokenize = static_cast<bool>(flags & normalize_flags::PAD_PUNCTUATION);
  return detail::normalize_characters(input, normalizer, strip, tokenize, stream, mr);
}

}  // namespace nvtext
