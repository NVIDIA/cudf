/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

constexpr size_type NUM_SPECIAL_CHARS = 112;

// These characters have unique title-case values that are different from their
// upper-case counterparts. The lookup uses binary_search so they must be sorted here.
__constant__ cuda::std::array<uint16_t, NUM_SPECIAL_CHARS> title_case_table = {
  0x00df, 0x01c4, 0x01c6, 0x01c7, 0x01c9, 0x01ca, 0x01cc, 0x01f1, 0x01f3, 0x0587, 0x10d0, 0x10d1,
  0x10d2, 0x10d3, 0x10d4, 0x10d5, 0x10d6, 0x10d7, 0x10d8, 0x10d9, 0x10da, 0x10db, 0x10dc, 0x10dd,
  0x10de, 0x10df, 0x10e0, 0x10e1, 0x10e2, 0x10e3, 0x10e4, 0x10e5, 0x10e6, 0x10e7, 0x10e8, 0x10e9,
  0x10ea, 0x10eb, 0x10ec, 0x10ed, 0x10ee, 0x10ef, 0x10f0, 0x10f1, 0x10f2, 0x10f3, 0x10f4, 0x10f5,
  0x10f6, 0x10f7, 0x10f8, 0x10f9, 0x10fa, 0x10fd, 0x10fe, 0x10ff, 0x1f80, 0x1f81, 0x1f82, 0x1f83,
  0x1f84, 0x1f85, 0x1f86, 0x1f87, 0x1f90, 0x1f91, 0x1f92, 0x1f93, 0x1f94, 0x1f95, 0x1f96, 0x1f97,
  0x1fa0, 0x1fa1, 0x1fa2, 0x1fa3, 0x1fa4, 0x1fa5, 0x1fa6, 0x1fa7, 0x1fb2, 0x1fb3, 0x1fb4, 0x1fb7,
  0x1fc2, 0x1fc3, 0x1fc4, 0x1fc7, 0x1ff2, 0x1ff3, 0x1ff4, 0x1ff7, 0x2c5f, 0xa7c1, 0xa7c8, 0xa7ca,
  0xa7d1, 0xa7d7, 0xa7d9, 0xa7f6, 0xfb00, 0xfb01, 0xfb02, 0xfb03, 0xfb04, 0xfb05, 0xfb06, 0xfb13,
  0xfb14, 0xfb15, 0xfb16, 0xfb17};

// These are the title-case counterparts to the characters in title_case_table
__constant__ cuda::std::array<uint64_t, NUM_SPECIAL_CHARS> title_case_chars = {
  0x005300730000 /* Ss */, 0x01c500000000 /* ǅ */,   0x01c500000000 /* ǅ */,
  0x01c800000000 /* ǈ */,  0x01c800000000 /* ǈ */,   0x01cb00000000 /* ǋ */,
  0x01cb00000000 /* ǋ */,  0x01f200000000 /* ǲ */,   0x01f200000000 /* ǲ */,
  0x053505820000 /* Եւ */, 0x10d000000000 /* ა */,   0x10d100000000 /* ბ */,
  0x10d200000000 /* გ */,  0x10d300000000 /* დ */,   0x10d400000000 /* ე */,
  0x10d500000000 /* ვ */,  0x10d600000000 /* ზ */,   0x10d700000000 /* თ */,
  0x10d800000000 /* ი */,  0x10d900000000 /* კ */,   0x10da00000000 /* ლ */,
  0x10db00000000 /* მ */,  0x10dc00000000 /* ნ */,   0x10dd00000000 /* ო */,
  0x10de00000000 /* პ */,  0x10df00000000 /* ჟ */,   0x10e000000000 /* რ */,
  0x10e100000000 /* ს */,  0x10e200000000 /* ტ */,   0x10e300000000 /* უ */,
  0x10e400000000 /* ფ */,  0x10e500000000 /* ქ */,   0x10e600000000 /* ღ */,
  0x10e700000000 /* ყ */,  0x10e800000000 /* შ */,   0x10e900000000 /* ჩ */,
  0x10ea00000000 /* ც */,  0x10eb00000000 /* ძ */,   0x10ec00000000 /* წ */,
  0x10ed00000000 /* ჭ */,  0x10ee00000000 /* ხ */,   0x10ef00000000 /* ჯ */,
  0x10f000000000 /* ჰ */,  0x10f100000000 /* ჱ */,   0x10f200000000 /* ჲ */,
  0x10f300000000 /* ჳ */,  0x10f400000000 /* ჴ */,   0x10f500000000 /* ჵ */,
  0x10f600000000 /* ჶ */,  0x10f700000000 /* ჷ */,   0x10f800000000 /* ჸ */,
  0x10f900000000 /* ჹ */,  0x10fa00000000 /* ჺ */,   0x10fd00000000 /* ჽ */,
  0x10fe00000000 /* ჾ */,  0x10ff00000000 /* ჿ */,   0x1f8800000000 /* ᾈ */,
  0x1f8900000000 /* ᾉ */,  0x1f8a00000000 /* ᾊ */,   0x1f8b00000000 /* ᾋ */,
  0x1f8c00000000 /* ᾌ */,  0x1f8d00000000 /* ᾍ */,   0x1f8e00000000 /* ᾎ */,
  0x1f8f00000000 /* ᾏ */,  0x1f9800000000 /* ᾘ */,   0x1f9900000000 /* ᾙ */,
  0x1f9a00000000 /* ᾚ */,  0x1f9b00000000 /* ᾛ */,   0x1f9c00000000 /* ᾜ */,
  0x1f9d00000000 /* ᾝ */,  0x1f9e00000000 /* ᾞ */,   0x1f9f00000000 /* ᾟ */,
  0x1fa800000000 /* ᾨ */,  0x1fa900000000 /* ᾩ */,   0x1faa00000000 /* ᾪ */,
  0x1fab00000000 /* ᾫ */,  0x1fac00000000 /* ᾬ */,   0x1fad00000000 /* ᾭ */,
  0x1fae00000000 /* ᾮ */,  0x1faf00000000 /* ᾯ */,   0x1fba03450000 /* Ὰͅ */,
  0x1fbc00000000 /* ᾼ */,  0x038603450000 /* Άͅ */,   0x039103420345 /* ᾼ͂ */,
  0x1fca03450000 /* Ὴͅ */,  0x1fcc00000000 /* ῌ */,   0x038903450000 /* Ήͅ */,
  0x039703420345 /* ῌ͂ */,  0x1ffa03450000 /* Ὼͅ */,   0x1ffc00000000 /* ῼ */,
  0x038f03450000 /* Ώͅ */,  0x03a903420345 /* ῼ͂ */,   0x2c2f00000000 /* Ⱟ */,
  0xa7c000000000 /* Ꟁ */,  0xa7c700000000 /* Ꟈ */,   0xa7c900000000 /* Ꟊ */,
  0xa7d000000000 /* Ꟑ */,  0xa7d600000000 /* Ꟗ */,   0xa7d800000000 /* Ꟙ */,
  0xa7f500000000 /* Ꟶ */,  0x004600660000 /* Ff */,  0x004600690000 /* Fi */,
  0x0046006c0000 /* Fl */, 0x004600660069 /* Ffi */, 0x00460066006c /* Ffl */,
  0x005300740000 /* St */, 0x005300740000 /* St */,  0x054405760000 /* Մն */,
  0x054405650000 /* Մե */, 0x0544056b0000 /* Մի */,  0x054e05760000 /* Վն */,
  0x0544056d0000 /* Մխ */
};

// clang-format off
// These characters are already upper-case but need to be converted to title-case.
// The lookup uses binary_search so they must be sorted here.
__constant__ cuda::std::array<uint16_t,13> upper_convert = {
  0x01c4, 0x01c7, 0x01ca, 0x01f1, 0x2c5f, 0xa7c1, 0xa7c8,
  0xa7ca, 0xa7d1, 0xa7d7, 0xa7d9, 0xa7f6, 0xfb04
};
// clang-format on

using char_info = cuda::std::pair<uint32_t, detail::character_flags_table_type>;

/**
 * @brief Returns the given character's info flags.
 */
__device__ char_info get_char_info(character_flags_table_type const* d_flags, char_utf8 chr)
{
  auto const code_point = detail::utf8_to_codepoint(chr);
  auto const flag = code_point <= 0x00'FFFF ? d_flags[code_point] : character_flags_table_type{0};
  return char_info{code_point, flag};
}

/**
 * @brief Base class for capitalize and title functors.
 *
 * Utility functions here manage access to the character case and flags tables.
 * Any derived class must supply a `capitalize_next` member function.
 *
 * @tparam Derived class uses the CRTP pattern to reuse code logic.
 */
template <typename Derived>
struct base_fn {
  character_flags_table_type const* d_flags;
  character_cases_table_type const* d_case_table;
  special_case_mapping const* d_special_case_mapping;
  column_device_view const d_column;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  base_fn(column_device_view const& d_column, cuda::stream_ref stream)
    : d_flags(get_character_flags_table(stream)),
      d_case_table(get_character_cases_table(stream)),
      d_special_case_mapping(get_special_case_mapping_table(stream)),
      d_column(d_column)
  {
  }

  __device__ int32_t convert_char(char_info const& info, char* d_buffer) const
  {
    auto const code_point = info.first;
    auto const flag       = info.second;

    // first, check for the special title-case characters
    auto tc_itr = thrust::upper_bound(
      thrust::seq, title_case_table.begin(), title_case_table.end(), code_point);
    tc_itr -= (tc_itr != title_case_table.begin());
    if (*tc_itr == code_point) {
      // result is encoded with up to 3 Unicode (16-bit) characters
      auto const result = title_case_chars[cuda::std::distance(title_case_table.begin(), tc_itr)];
      auto const count  = ((result & 0x0FFFF00000000) > 0) + ((result & 0x0000FFFF0000) > 0) +
                         ((result & 0x00000000FFFF) > 0);
      size_type bytes = 0;
      for (auto i = 0; i < count; ++i) {
        auto new_cp = result >> (32 - (i * 16)) & 0x0FFFF;
        bytes += d_buffer
                   ? detail::from_char_utf8(detail::codepoint_to_utf8(new_cp), d_buffer + bytes)
                   : detail::bytes_in_char_utf8(detail::codepoint_to_utf8(new_cp));
      }
      return bytes;
    }

    if (!IS_SPECIAL(flag)) {
      auto const new_char = codepoint_to_utf8(d_case_table[code_point]);
      return d_buffer ? detail::from_char_utf8(new_char, d_buffer)
                      : detail::bytes_in_char_utf8(new_char);
    }

    special_case_mapping m = d_special_case_mapping[get_special_case_hash_index(code_point)];

    auto const count  = IS_LOWER(flag) ? m.num_upper_chars : m.num_lower_chars;
    auto const* chars = IS_LOWER(flag) ? m.upper : m.lower;
    size_type bytes   = 0;
    for (uint16_t idx = 0; idx < count; idx++) {
      bytes += d_buffer
                 ? detail::from_char_utf8(detail::codepoint_to_utf8(chars[idx]), d_buffer + bytes)
                 : detail::bytes_in_char_utf8(detail::codepoint_to_utf8(chars[idx]));
    }
    return bytes;
  }

  /**
   * @brief Operator called for each row in `d_column`.
   *
   * This logic is shared by capitalize() and title() functions.
   * The derived class must supply a `capitalize_next` member function.
   */
  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto& derived    = static_cast<Derived&>(*this);
    auto const d_str = d_column.element<string_view>(idx);
    size_type bytes  = 0;
    auto d_buffer    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    bool capitalize  = true;
    for (auto const chr : d_str) {
      auto const info = get_char_info(d_flags, chr);
      auto const flag = info.second;
      auto const is_upper_convert =
        thrust::binary_search(thrust::seq, upper_convert.begin(), upper_convert.end(), info.first);
      auto const change_case = capitalize ? (IS_LOWER(flag) || is_upper_convert) : IS_UPPER(flag);

      if (change_case) {
        auto const char_bytes = convert_char(info, d_buffer);
        bytes += char_bytes;
        d_buffer += d_buffer ? char_bytes : 0;
      } else {
        if (d_buffer) {
          d_buffer += detail::from_char_utf8(chr, d_buffer);
        } else {
          bytes += detail::bytes_in_char_utf8(chr);
        }
      }

      // capitalize the next char if this one is a delimiter
      capitalize = derived.capitalize_next(chr, flag);
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

/**
 * @brief Capitalize functor.
 *
 * This capitalizes the first character of the string and lower-cases
 * the remaining characters.
 * If a delimiter is specified, capitalization continues within the string
 * on the first eligible character after any delimiter.
 */
struct capitalize_fn : base_fn<capitalize_fn> {
  string_view const d_delimiters;

  capitalize_fn(column_device_view const& d_column,
                string_view const& d_delimiters,
                cuda::stream_ref stream)
    : base_fn(d_column, stream), d_delimiters(d_delimiters)
  {
  }

  __device__ bool capitalize_next(char_utf8 const chr, character_flags_table_type const)
  {
    return !d_delimiters.empty() && (d_delimiters.find(chr) != string_view::npos);
  }
};

/**
 * @brief Title functor.
 *
 * This capitalizes the first letter of each word.
 * The beginning of a word is identified as the first sequence_type
 * character after a non-sequence_type character.
 * Also, lower-case all other alphabetic characters.
 */
struct title_fn : base_fn<title_fn> {
  string_character_types sequence_type;

  title_fn(column_device_view const& d_column,
           string_character_types sequence_type,
           cuda::stream_ref stream)
    : base_fn(d_column, stream), sequence_type(sequence_type)
  {
  }

  __device__ bool capitalize_next(char_utf8 const, character_flags_table_type const flag)
  {
    return (flag & sequence_type) == 0;
  };
};

/**
 * @brief Functor for determining title format for each string in a column.
 *
 * The first letter of each word should be upper-case (IS_UPPER).
 * All other characters should be lower-case (IS_LOWER).
 * Non-upper/lower-case (IS_UPPER_OR_LOWER) characters delimit words.
 */
struct is_title_fn {
  character_flags_table_type const* d_flags;
  column_device_view const d_column;

  __device__ bool operator()(size_type idx)
  {
    if (d_column.is_null(idx)) { return false; }
    auto const d_str = d_column.element<string_view>(idx);

    bool at_least_one_valid    = false;  // requires one or more cased characters
    bool should_be_capitalized = true;   // current character should be upper-case
    for (auto const chr : d_str) {
      auto const flag = get_char_info(d_flags, chr).second;
      if (IS_UPPER_OR_LOWER(flag)) {
        if (should_be_capitalized == !IS_UPPER(flag)) return false;
        at_least_one_valid = true;
      }
      should_be_capitalized = !IS_UPPER_OR_LOWER(flag);
    }
    return at_least_one_valid;
  }
};

/**
 * @brief Common utility function for title() and capitalize().
 *
 * @tparam CapitalFn The specific functor.
 * @param cfn The functor instance.
 * @param input The input strings column.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for allocating the new device_buffer
 */
template <typename CapitalFn>
std::unique_ptr<column> capitalizer(CapitalFn cfn,
                                    strings_column_view const& input,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto [offsets_column, chars] = make_strings_children(cfn, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> capitalize(strings_column_view const& input,
                                   string_scalar const& delimiters,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiters.is_valid(stream), "Delimiter must be a valid string");
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  auto const d_column     = column_device_view::create(input.parent(), stream);
  auto const d_delimiters = delimiters.value(stream);
  return capitalizer(capitalize_fn{*d_column, d_delimiters, stream}, input, stream, mr);
}

std::unique_ptr<column> title(strings_column_view const& input,
                              string_character_types sequence_type,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  auto d_column = column_device_view::create(input.parent(), stream);
  return capitalizer(title_fn{*d_column, sequence_type, stream}, input, stream, mr);
}

std::unique_ptr<column> is_title(strings_column_view const& input,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::BOOL8);
  auto results  = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto d_column = column_device_view::create(input.parent(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    results->mutable_view().data<bool>(),
                    is_title_fn{get_character_flags_table(stream), *d_column});
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

std::unique_ptr<column> capitalize(strings_column_view const& input,
                                   string_scalar const& delimiter,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::capitalize(input, delimiter, stream, mr);
}

std::unique_ptr<column> title(strings_column_view const& input,
                              string_character_types sequence_type,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::title(input, sequence_type, stream, mr);
}

std::unique_ptr<column> is_title(strings_column_view const& input,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_title(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
