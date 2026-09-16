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
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/normalize.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>

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

__device__ int8_t cp_to_utf8(uint32_t codepoint, char* out)
{
  auto utf8 = cudf::strings::detail::codepoint_to_utf8(codepoint);
  return cudf::strings::detail::from_char_utf8(utf8, out);
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
rmm::device_uvector<codepoint_metadata_type> get_codepoint_metadata(cuda::stream_ref stream)
{
  auto table_vector = rmm::device_uvector<codepoint_metadata_type>(codepoint_metadata_size, stream);
  auto table        = table_vector.data();
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
rmm::device_uvector<aux_codepoint_data_type> get_aux_codepoint_data(cuda::stream_ref stream)
{
  auto table_vector = rmm::device_uvector<aux_codepoint_data_type>(aux_codepoint_data_size, stream);
  auto table        = table_vector.data();
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
  rmm::device_uvector<uint32_t> cp_metadata;
  rmm::device_uvector<aux_codepoint_data_type> aux_table;
  bool do_lower_case;
  std::unique_ptr<cudf::column> special_tokens;
  rmm::device_uvector<cudf::string_view> special_tokens_view;

  cudf::device_span<cudf::string_view const> get_special_tokens() const
  {
    return special_tokens_view;
  }

  character_normalizer_impl(rmm::device_uvector<uint32_t>&& cp_metadata,
                            rmm::device_uvector<aux_codepoint_data_type>&& aux_table,
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

/**
 * @brief Kernel handles fixing up the normalized data to account for any special tokens
 *
 * This undoes the padding added around the `[]` for patterns matching the strings in the
 * special_tokens array.
 *
 * Launched as a thread per input byte (total_count).
 *
 * @param d_normalized The normalized set of UTF-8 characters; 3 uints per input byte
 * @param total_count Number of bytes represented by d_normalized; len(d_normalized)/3
 * @param special_tokens Tokens to check against
 * @param do_lower_case Whether to convert text to lower case
 */
CUDF_KERNEL void special_tokens_kernel(uint32_t* d_normalized,
                                       int64_t total_count,
                                       cudf::device_span<cudf::string_view const> special_tokens,
                                       bool do_lower_case)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= total_count) { return; }
  auto const begin = d_normalized + (idx * MAX_NEW_CHARS) + 1;
  if (*begin != '[') { return; }
  auto const end   = begin + cuda::std::min(6L, total_count - idx) * MAX_NEW_CHARS;
  auto const match = thrust::find(thrust::seq, begin, end, static_cast<uint32_t>(']'));
  if (match == end) { return; }
  char candidate[8];
  auto const ch_begin =
    cuda::transform_iterator(begin, [](auto v) { return static_cast<char>(v); });
  auto const ch_end = ch_begin + cuda::std::distance(begin, match + 1);
  auto last         = thrust::copy_if(
    thrust::seq, ch_begin, ch_end, candidate, [](auto c) { return c != 0 && c != ' '; });
  *last = 0;  // only needed for debug

  auto const size  = static_cast<cudf::size_type>(cuda::std::distance(candidate, last));
  auto const token = cudf::string_view(candidate, size);
  // the binary_search expects the special_tokens to be sorted
  if (!thrust::binary_search(thrust::seq, special_tokens.begin(), special_tokens.end(), token)) {
    return;
  }

  // fix up chars to remove the extra spaces and convert to upper-case
  *(begin + 1) = 0;  // removes space after '['
  *(match - 1) = 0;  // removes space before ']'
  if (do_lower_case) {
    auto itr = begin + 2;
    while (itr < match - 2) {
      auto ch = *itr;
      if (ch >= 'a' && ch <= 'z') { *itr = ch - 'a' + 'A'; }
      ++itr;
    }
  }
}

/**
 * @brief The normalizer kernel
 *
 * Launched as a thread per input byte (total_bytes).
 *
 * Converts the input d_chars into codepoints to lookup in the provided tables.
 * Once processed, the d_output contains 3 uints per input byte each encoded
 * as output UTF-8. Any zero values are to removed by a subsequent kernel call.
 *
 * @param d_chars The characters for the input strings column to normalize
 * @param total_bytes The number of bytes in the d_chars
 * @param cp_metadata First lookup table for codepoint metadata
 * @param aux_table Second lookup table containing possible replacement characters
 * @param do_lower_case True if the normalization includes lower-casing characters
 * @param d_output The output of the normalization (UTF-8 encoded)
 */
CUDF_KERNEL void data_normalizer_kernel(char const* d_chars,
                                        int64_t total_bytes,
                                        codepoint_metadata_type const* cp_metadata,
                                        aux_codepoint_data_type const* aux_table,
                                        bool do_lower_case,
                                        uint32_t* d_output)
{
  uint32_t replacement[MAX_NEW_CHARS] = {0};

  auto const idx = cudf::detail::grid_1d::global_thread_id();

  if ((idx < total_bytes) && cudf::strings::detail::is_begin_utf8_char(d_chars[idx])) {
    auto const cp = [utf8 = d_chars + idx] {
      cudf::char_utf8 ch_utf8 = *utf8;
      if (ch_utf8 > 0x7F) { cudf::strings::detail::to_char_utf8(utf8, ch_utf8); }
      return cudf::strings::detail::utf8_to_codepoint(ch_utf8);
    }();
    auto const metadata = cp_metadata[cp];

    if (!should_remove_cp(metadata, do_lower_case)) {
      int8_t num_new_chars = 1;
      // retrieve the normalized value for cp
      auto const new_cp = do_lower_case || always_replace(metadata) ? get_first_cp(metadata) : cp;
      replacement[0]    = new_cp == 0 ? cp : new_cp;

      if (do_lower_case && is_multi_char_transform(metadata)) {
        auto const next_cps = aux_table[cp];
        replacement[1]      = static_cast<uint32_t>(next_cps >> 32);
        replacement[2]      = static_cast<uint32_t>(next_cps & 0xFFFFFFFF);
        num_new_chars       = 2 + (replacement[2] != 0);
      }

      if (should_add_spaces(metadata, do_lower_case) && (num_new_chars == 1)) {
        replacement[1] = replacement[0];
        replacement[0] = SPACE_CODE_POINT;  // add spaces around the new codepoint
        replacement[2] = SPACE_CODE_POINT;
        num_new_chars  = 3;
      }

      // convert codepoints back to UTF-8 in-place
      for (int k = 0; k < num_new_chars; ++k) {
        auto const new_cp = replacement[k];
        if (new_cp) { cp_to_utf8(new_cp, reinterpret_cast<char*>(replacement + k)); }
      }
    }
  }

  // employ an optimized coalesced writer to output replacement as a block of transposed data
  using block_store =
    cub::BlockStore<uint32_t, 256, MAX_NEW_CHARS, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  __shared__ typename block_store::TempStorage bs_stg;
  auto block_base = d_output + blockIdx.x * blockDim.x * MAX_NEW_CHARS;
  block_store(bs_stg).Store(block_base, replacement);
}

/**
 * @brief Computes the output sizes for each row
 *
 * The input offsets are used with segmented-reduce to count the number of
 * non-zero values for each output row.
 *
 * @param d_normalized The UTF-8 encoded normalized values
 * @param offsets These identify the row boundaries
 * @param offset Only non-zero if the input column has been sliced
 * @param size The number of output rows (sames as the number of input rows)
 * @param stream Stream used for allocating device memory and launching kernels
 * @return The sizes of each output row
 */
template <typename OffsetType>
rmm::device_uvector<cudf::size_type> compute_sizes(cudf::device_span<uint32_t const> d_normalized,
                                                   OffsetType offsets,
                                                   int64_t offset,
                                                   cudf::size_type size,
                                                   cuda::stream_ref stream)
{
  auto output_sizes = rmm::device_uvector<cudf::size_type>(size, stream);

  auto d_data = d_normalized.data();

  // counts the non-zero bytes in the d_data array
  auto d_in = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<cudf::size_type>([d_data] __device__(auto idx) {
      // transform function counts number of non-zero bytes in uint32_t value
      auto tfn = [](uint32_t v) -> cudf::size_type {
        return ((v & 0xFF) > 0) + ((v & 0xFF00) > 0) + ((v & 0xFF0000) > 0) +
               ((v & 0xFF000000) > 0);
      };
      auto const begin = d_data + (static_cast<int64_t>(idx) * MAX_NEW_CHARS);
      auto const end   = begin + MAX_NEW_CHARS;
      return thrust::transform_reduce(thrust::seq, begin, end, tfn, 0, cuda::std::plus{});
    }));

  // DeviceSegmentedReduce is used to compute the size of each output row
  auto d_out = output_sizes.begin();
  auto temp  = std::size_t{0};
  if (offset == 0) {
    cub::DeviceSegmentedReduce::Sum(
      nullptr, temp, d_in, d_out, size, offsets, offsets + 1, stream.get());
    auto d_temp = rmm::device_buffer{temp, stream};
    cub::DeviceSegmentedReduce::Sum(
      d_temp.data(), temp, d_in, d_out, size, offsets, offsets + 1, stream.get());
  } else {
    // offsets need to be normalized for segmented-reduce to work efficiently
    auto offsets_itr = cuda::transform_iterator(
      offsets,
      cuda::proclaim_return_type<int64_t>([offset] __device__(auto o) { return o - offset; }));
    cub::DeviceSegmentedReduce::Sum(
      nullptr, temp, d_in, d_out, size, offsets_itr, offsets_itr + 1, stream.get());
    auto d_temp = rmm::device_buffer{temp, stream};
    cub::DeviceSegmentedReduce::Sum(
      d_temp.data(), temp, d_in, d_out, size, offsets_itr, offsets_itr + 1, stream.get());
  }

  return output_sizes;
}

// handles ranges above int32 max
template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator remove_copy_safe(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T const& value,
                                cuda::stream_ref stream)
{
  auto const copy_size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                                  static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto itr = first;
  while (itr != last) {
    auto const copy_end =
      static_cast<std::size_t>(std::distance(itr, last)) <= copy_size ? last : itr + copy_size;
    result =
      thrust::remove_copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          itr,
                          copy_end,
                          result,
                          value);
    itr = copy_end;
  }
  return result;
}

// handles ranges above int32 max
template <typename Iterator, typename T>
Iterator remove_safe(Iterator first, Iterator last, T const& value, cuda::stream_ref stream)
{
  auto const size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                             static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto result = first;
  auto itr    = first;
  while (itr != last) {
    auto end = static_cast<std::size_t>(std::distance(itr, last)) <= size ? last : itr + size;
    result   = thrust::remove(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()), itr, end, value);
    itr = end;
  }
  return result;
}
}  // namespace

std::unique_ptr<cudf::column> normalize_characters(cudf::strings_column_view const& input,
                                                   character_normalizer const& normalizer,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  if (chars_size == 0) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }

  constexpr int64_t block_size = 256;
  cudf::detail::grid_1d grid{chars_size, block_size};
  auto const max_new_char_total = cudf::util::round_up_safe(chars_size, block_size) * MAX_NEW_CHARS;

  auto const& parameters = normalizer._impl;

  auto d_normalized = rmm::device_uvector<uint32_t>(max_new_char_total, stream);
  data_normalizer_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    d_input_chars,
    chars_size,
    parameters->cp_metadata.data(),
    parameters->aux_table.data(),
    parameters->do_lower_case,
    d_normalized.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // This removes space added around any special tokens in the form of [ttt].
  // An alternate approach is to do a multi-replace of '[ ttt ]' with '[ttt]' right
  // before returning the output strings column.
  auto const special_tokens = parameters->get_special_tokens();
  if (!special_tokens.empty()) {
    special_tokens_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      d_normalized.data(), chars_size, special_tokens, parameters->do_lower_case);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Use segmented-reduce over the non-zero codepoints to get the size of the output rows
  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto output_sizes =
    compute_sizes(d_normalized, input_offsets, first_offset, input.size(), stream);

  // convert the sizes to offsets
  auto [offsets, total_size] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);

  // create output chars by calling remove_copy(0) on the bytes in d_normalized
  auto chars       = rmm::device_uvector<char>(total_size, stream, mr);
  auto const begin = reinterpret_cast<char const*>(d_normalized.begin());
  auto const end   = reinterpret_cast<char const*>(d_normalized.end());
  remove_copy_safe(begin, end, chars.data(), 0, stream);

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
  return detail::normalize_characters(input, normalizer, stream, mr);
}

}  // namespace nvtext
