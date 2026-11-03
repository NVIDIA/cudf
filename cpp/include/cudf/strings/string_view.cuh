/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/export.hpp>

#ifndef __CUDA_ARCH__
#include <cudf/utilities/error.hpp>
#endif

// This is defined when compiling with NVRTC. NVRTC cannot include thrust headers at this time.
#ifndef __CUDACC_RTC__
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#endif

#include <cuda/std/functional>
#include <cuda/std/utility>

// This file should only include device code logic.
// Host-only or host/device code should be defined in the string_view.hpp header file.

namespace CUDF_EXPORT cudf {
namespace strings {
namespace detail {

/**
 * @brief Return the number of UTF-8 characters in this provided char array.
 *
 * @param str String with encoded char bytes.
 * @param bytes Number of bytes in str.
 * @return The number of characters in the array.
 */
__device__ inline size_type characters_in_string(char const* str, size_type bytes)
{
  if ((str == nullptr) || (bytes == 0)) return 0;
  auto ptr = reinterpret_cast<uint8_t const*>(str);
#ifndef __CUDACC_RTC__
  return thrust::count_if(
    thrust::seq, ptr, ptr + bytes, [](uint8_t chr) { return is_begin_utf8_char(chr); });
#else
  size_type chars = 0;
  auto const end  = ptr + bytes;
  while (ptr < end) {
    chars += is_begin_utf8_char(*ptr++);
  }
  return chars;
#endif
}

/**
 * @brief Count the bytes to a specified character position
 *
 * Returns the number of bytes and any left over position value.
 * The returned position is > 0 if the given position would read past
 * the end of the input string.
 *
 * @param d_str Input string to count bytes within
 * @param pos Character position to count to
 * @return The number of bytes and the left over non-counted position value
 */
__device__ inline cuda::std::pair<size_type, size_type> bytes_to_character_position(
  string_view d_str, size_type pos)
{
  size_type bytes    = 0;
  auto ptr           = d_str.data();
  auto const end_ptr = ptr + d_str.size_bytes();
  while ((pos > 0) && (ptr < end_ptr)) {
    auto const width = strings::detail::bytes_in_utf8_byte(static_cast<uint8_t>(*ptr));
    if (width) { --pos; }
    bytes += width;
    ++ptr;
  }
  return {bytes, pos};
}

/**
 * @brief string value for sentinel which is used in min, max reduction
 * operators
 *
 * This sentinel string value is the highest possible valid UTF-8 encoded
 * character. This serves as identity value for maximum operator on string
 * values. Also, this char pointer serves as valid device pointer of identity
 * value for minimum operator on string values.
 */
static __constant__ char max_string_sentinel[5]{"\xF7\xBF\xBF\xBF"};  // NOLINT
}  // namespace detail
}  // namespace strings

/**
 * @brief Return minimum value associated with the string type
 *
 * This function is needed to be host callable because it is called by a host
 * callable function DeviceMax::identity<string_view>()
 *
 * @return An empty string
 */
CUDF_HOST_DEVICE inline string_view string_view::min() { return {}; }

/**
 * @brief Return maximum value associated with the string type
 *
 * This function is needed to be host callable because it is called by a host
 * callable function DeviceMin::identity<string_view>()
 *
 * @return A string value which represents the highest possible valid UTF-8 encoded
 * character.
 */
CUDF_HOST_DEVICE inline string_view string_view::max()
{
  char const* psentinel{nullptr};
#if defined(__CUDA_ARCH__)
  psentinel = &cudf::strings::detail::max_string_sentinel[0];
#else
  CUDF_CUDA_TRY(
    cudaGetSymbolAddress((void**)&psentinel, cudf::strings::detail::max_string_sentinel));
#endif
  return {psentinel, 4};
}

__device__ inline size_type string_view::length() const
{
  if (_length == UNKNOWN_STRING_LENGTH)
    _length = strings::detail::characters_in_string(_data, _bytes);
  return _length;
}

// @cond
// this custom iterator knows about UTF8 encoding
__device__ inline string_view::const_iterator::const_iterator(string_view const& str, size_type pos)
  : p{str.data()}, bytes{str.size_bytes()}, char_pos{pos}, byte_pos{str.byte_offset(pos)}
{
}

__device__ inline string_view::const_iterator::const_iterator(string_view const& str,
                                                              size_type pos,
                                                              size_type offset)
  : p{str.data()}, bytes{str.size_bytes()}, char_pos{pos}, byte_pos{offset}
{
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator++()
{
  if (byte_pos < bytes) {
    // max is used to prevent an infinite loop on invalid UTF-8 data
    byte_pos +=
      cuda::std::max(1, strings::detail::bytes_in_utf8_byte(static_cast<uint8_t>(p[byte_pos])));
  }
  ++char_pos;
  return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator++(int)
{
  string_view::const_iterator tmp(*this);
  operator++();
  return tmp;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator+(
  string_view::const_iterator::difference_type offset) const
{
  const_iterator tmp(*this);
  size_type adjust = abs(offset);
  while (adjust-- > 0)
    offset > 0 ? ++tmp : --tmp;
  return tmp;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator+=(
  string_view::const_iterator::difference_type offset)
{
  size_type adjust = abs(offset);
  while (adjust-- > 0)
    offset > 0 ? operator++() : operator--();
  return *this;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator--()
{
  if (byte_pos > 0) {
    if (byte_pos == char_pos) {
      --byte_pos;
    } else {
      while (strings::detail::bytes_in_utf8_byte(static_cast<uint8_t>(p[--byte_pos])) == 0)
        ;
    }
  }
  --char_pos;
  return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator--(int)
{
  string_view::const_iterator tmp(*this);
  operator--();
  return tmp;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator-=(
  string_view::const_iterator::difference_type offset)
{
  size_type adjust = abs(offset);
  while (adjust-- > 0)
    offset > 0 ? operator--() : operator++();
  return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator-(
  string_view::const_iterator::difference_type offset) const
{
  const_iterator tmp(*this);
  size_type adjust = abs(offset);
  while (adjust-- > 0)
    offset > 0 ? --tmp : ++tmp;
  return tmp;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::move_to(
  size_type new_pos)
{
  *this += (new_pos - char_pos);  // more efficient than recounting from the start
  return *this;
}

__device__ inline bool string_view::const_iterator::operator==(
  string_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos == rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator!=(
  string_view::const_iterator const& rhs) const
{
  return (p != rhs.p) || (char_pos != rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator<(
  string_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos < rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator<=(
  string_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos <= rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator>(
  string_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos > rhs.char_pos);
}

__device__ inline bool string_view::const_iterator::operator>=(
  string_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos >= rhs.char_pos);
}

__device__ inline char_utf8 string_view::const_iterator::operator*() const
{
  char_utf8 chr = 0;
  strings::detail::to_char_utf8(p + byte_offset(), chr);
  return chr;
}

__device__ inline size_type string_view::const_iterator::position() const { return char_pos; }

__device__ inline size_type string_view::const_iterator::byte_offset() const { return byte_pos; }

__device__ inline string_view::const_iterator string_view::begin() const { return {*this, 0, 0}; }

__device__ inline string_view::const_iterator string_view::end() const
{
  return {*this, length(), size_bytes()};
}
// @endcond

__device__ inline char_utf8 string_view::operator[](size_type pos) const
{
  size_type offset = byte_offset(pos);
  if (offset >= _bytes) return 0;
  char_utf8 chr = 0;
  strings::detail::to_char_utf8(data() + offset, chr);
  return chr;
}

__device__ inline size_type string_view::byte_offset(size_type pos) const
{
  if (length() == size_bytes()) return pos;
  return cuda::std::get<0>(strings::detail::bytes_to_character_position(*this, pos));
}

__device__ inline int string_view::compare(string_view const& in) const
{
  return compare(in.data(), in.size_bytes());
}

__device__ inline int string_view::compare(char const* data, size_type bytes) const
{
  size_type const len1 = size_bytes();
  auto const* ptr1     = reinterpret_cast<unsigned char const*>(this->data());
  auto const* ptr2     = reinterpret_cast<unsigned char const*>(data);
  if ((ptr1 == ptr2) && (bytes == len1)) return 0;
  size_type idx = 0;
  for (; (idx < len1) && (idx < bytes); ++idx) {
    if (*ptr1 != *ptr2) return static_cast<int32_t>(*ptr1) - static_cast<int32_t>(*ptr2);
    ++ptr1;
    ++ptr2;
  }
  if (idx < len1) return 1;
  if (idx < bytes) return -1;
  return 0;
}

__device__ inline bool string_view::operator==(string_view const& rhs) const
{
  return (size_bytes() == rhs.size_bytes()) && (compare(rhs) == 0);
}

__device__ inline bool string_view::operator!=(string_view const& rhs) const
{
  return compare(rhs) != 0;
}

__device__ inline bool string_view::operator<(string_view const& rhs) const
{
  return compare(rhs) < 0;
}

__device__ inline bool string_view::operator>(string_view const& rhs) const
{
  return compare(rhs) > 0;
}

__device__ inline bool string_view::operator<=(string_view const& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc < 0);
}

__device__ inline bool string_view::operator>=(string_view const& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc > 0);
}

__device__ inline size_type string_view::find(string_view const& str,
                                              size_type pos,
                                              size_type count) const
{
  return find(str.data(), str.size_bytes(), pos, count);
}

template <bool forward>
__device__ inline size_type string_view::find_impl(char const* str,
                                                   size_type bytes,
                                                   size_type pos,
                                                   size_type count) const
{
  if (!str || pos < 0) { return npos; }
  if (pos > 0 && pos > length()) { return npos; }

  // use iterator to help reduce character/byte counting
  auto const itr  = begin() + pos;
  auto const spos = itr.byte_offset();
  auto const epos =
    (count >= 0) && ((pos + count) < length()) ? (itr + count).byte_offset() : size_bytes();

  auto const find_length = (epos - spos) - bytes + 1;
  auto const d_target    = string_view{str, bytes};

  auto ptr = data() + (forward ? spos : (epos - bytes));
  for (size_type idx = 0; idx < find_length; ++idx) {
    if (d_target.compare(ptr, bytes) == 0) {
      return forward ? pos : character_offset(epos - bytes - idx);
    }
    // use pos to record the current find position
    pos += strings::detail::is_begin_utf8_char(*ptr);
    forward ? ++ptr : --ptr;
  }
  return npos;
}

__device__ inline size_type string_view::find(char const* str,
                                              size_type bytes,
                                              size_type pos,
                                              size_type count) const
{
  return find_impl<true>(str, bytes, pos, count);
}

__device__ inline size_type string_view::find(char_utf8 chr, size_type pos, size_type count) const
{
  char str[sizeof(char_utf8)];  // NOLINT
  size_type chwidth = strings::detail::from_char_utf8(chr, str);
  return find(str, chwidth, pos, count);
}

__device__ inline size_type string_view::rfind(string_view const& str,
                                               size_type pos,
                                               size_type count) const
{
  return rfind(str.data(), str.size_bytes(), pos, count);
}

__device__ inline size_type string_view::rfind(char const* str,
                                               size_type bytes,
                                               size_type pos,
                                               size_type count) const
{
  return find_impl<false>(str, bytes, pos, count);
}

__device__ inline size_type string_view::rfind(char_utf8 chr, size_type pos, size_type count) const
{
  char str[sizeof(char_utf8)];  // NOLINT
  size_type chwidth = strings::detail::from_char_utf8(chr, str);
  return rfind(str, chwidth, pos, count);
}

// parameters are character position values
__device__ inline string_view string_view::substr(size_type pos, size_type count) const
{
  if (pos < 0 || pos >= length()) { return string_view{}; }
  auto const spos = begin() + pos;
  auto const epos = count >= 0 ? (spos + count) : const_iterator{*this, _length, size_bytes()};
  auto ss = string_view{data() + spos.byte_offset(), epos.byte_offset() - spos.byte_offset()};
  // this potentially saves redundant character counting downstream
  if (_length != UNKNOWN_STRING_LENGTH) { ss._length = epos.position() - spos.position(); }
  return ss;
}

__device__ inline size_type string_view::character_offset(size_type bytepos) const
{
  if (length() == size_bytes()) return bytepos;
  return strings::detail::characters_in_string(data(), bytepos);
}

}  // namespace CUDF_EXPORT cudf
