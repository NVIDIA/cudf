/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Returns true for each string where all characters match the given types.
 *
 * Only the characters that match to `verify_types` are checked.
 * Returns false if no characters are checked or one character does not match `types`.
 * Returns true if at least one character is checked and all checked characters match `types`.
 */
struct char_types_fn {
  column_device_view const d_column;
  character_flags_table_type const* d_flags;
  string_character_types const types;
  string_character_types const verify_types;

  __device__ bool operator()(size_type idx) const
  {
    if (d_column.is_null(idx)) { return false; }
    auto const d_str = d_column.element<string_view>(idx);
    auto const end   = d_str.data() + d_str.size_bytes();

    bool type_matched     = !d_str.empty();  // require at least one character;
    size_type check_count = 0;               // count checked characters
    for (auto itr = d_str.data(); type_matched && (itr < end); ++itr) {
      uint8_t const chr = static_cast<uint8_t>(*itr);
      if (is_utf8_continuation_char(chr)) { continue; }
      auto u8 = static_cast<char_utf8>(chr);  // holds UTF8 value
      // using max(int8) here since max(char)=255 on ARM systems
      if (u8 > cuda::std::numeric_limits<int8_t>::max()) { to_char_utf8(itr, u8); }

      // lookup flags in table by codepoint
      auto const code_point = utf8_to_codepoint(u8);
      auto const flag       = code_point <= 0x00'FFFF ? d_flags[code_point] : 0;

      if ((verify_types & flag) ||                   // should flag be verified;
          (flag == 0 && verify_types == ALL_TYPES))  // special edge case
      {
        type_matched = (types & flag) > 0;
        ++check_count;
      }
    }

    return type_matched && (check_count > 0);
  }
};
}  // namespace

std::unique_ptr<column> all_characters_of_type(strings_column_view const& input,
                                               string_character_types types,
                                               string_character_types verify_types,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  auto d_strings = column_device_view::create(input.parent(), stream);

  // create output column
  auto results = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // get the static character types table
  auto d_flags = detail::get_character_flags_table(stream);

  // set the output values by checking the character types for each string
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    results->mutable_view().data<bool>(),
                    char_types_fn{*d_strings, d_flags, types, verify_types});

  results->set_null_count(input.null_count());
  return results;
}

namespace {

/**
 * @brief Removes individual characters from a strings column based on character type.
 *
 * Types to remove are specified by `types_to_remove` OR
 * types to not remove are specified by `types_to_keep`.
 *
 * This is called twice. The first pass calculates the size of each output string.
 * The final pass copies the results to the output strings column memory.
 */
struct filter_chars_fn {
  column_device_view const d_column;
  character_flags_table_type const* d_flags;
  string_character_types const types_to_remove;
  string_character_types const types_to_keep;
  string_view const d_replacement;  ///< optional replacement for removed characters
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Returns true if the given character should be replaced.
   */
  __device__ bool replace_char(char_utf8 ch)
  {
    auto const code_point = detail::utf8_to_codepoint(ch);
    auto const flag       = code_point <= 0x00'FFFF ? d_flags[code_point] : 0;
    if (flag == 0)  // all types pass unless specifically identified
      return (types_to_remove == ALL_TYPES);
    if (types_to_keep == ALL_TYPES)  // filter case
      return (types_to_remove & flag) != 0;
    return (types_to_keep & flag) == 0;  // keep case
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str  = d_column.element<string_view>(idx);
    auto const in_ptr = d_str.data();
    auto out_ptr      = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto nbytes       = d_str.size_bytes();

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto const char_size = bytes_in_char_utf8(*itr);
      string_view const d_newchar =
        replace_char(*itr) ? d_replacement : string_view(in_ptr + itr.byte_offset(), char_size);
      nbytes += d_newchar.size_bytes() - char_size;
      if (out_ptr) out_ptr = cudf::strings::detail::copy_string(out_ptr, d_newchar);
    }
    if (!out_ptr) { d_sizes[idx] = nbytes; }
  }
};

}  // namespace

std::unique_ptr<column> filter_characters_of_type(strings_column_view const& strings,
                                                  string_character_types types_to_remove,
                                                  string_scalar const& replacement,
                                                  string_character_types types_to_keep,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  if (types_to_remove == ALL_TYPES)
    CUDF_EXPECTS(types_to_keep != ALL_TYPES,
                 "Parameters types_to_remove and types_to_keep must not be both ALL_TYPES");
  else
    CUDF_EXPECTS(types_to_keep == ALL_TYPES,
                 "One of parameter types_to_remove and types_to_keep must be set to ALL_TYPES");

  auto const strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  cudf::string_view d_replacement(replacement.data(), replacement.size());
  filter_chars_fn filterer{*strings_column,
                           detail::get_character_flags_table(stream),
                           types_to_remove,
                           types_to_keep,
                           d_replacement};

  // copy null mask from input column
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);

  // this utility calls filterer to build the offsets and chars columns
  auto [offsets_column, chars] = make_strings_children(filterer, strings_count, stream, mr);

  // return new strings column
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars.release(),
                             strings.null_count(),
                             std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> all_characters_of_type(strings_column_view const& input,
                                               string_character_types types,
                                               string_character_types verify_types,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::all_characters_of_type(input, types, verify_types, stream, mr);
}

std::unique_ptr<column> filter_characters_of_type(strings_column_view const& input,
                                                  string_character_types types_to_remove,
                                                  string_scalar const& replacement,
                                                  string_character_types types_to_keep,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_characters_of_type(
    input, types_to_remove, replacement, types_to_keep, stream, mr);
}

}  // namespace strings
}  // namespace cudf
