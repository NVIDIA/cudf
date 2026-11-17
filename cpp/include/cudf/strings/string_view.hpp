/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <cuda/std/iterator>

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

/**
 * @file
 * @brief Class definition for cudf::string_view.
 */

namespace CUDF_EXPORT cudf {

using char_utf8 = uint32_t;  ///< UTF-8 characters are 1-4 bytes

/**
 * @brief A non-owning, immutable view of device data that is a variable length
 * char array representing a UTF-8 string.
 *
 * @ingroup strings_classes
 *
 * The caller must maintain the device memory for the lifetime of this instance.
 *
 * This may be used to wrap a device pointer and size but any member function
 * that requires accessing the device memory must be called from a kernel.
 */
class string_view {
 public:
  /**
   * @brief Return the number of bytes in this string
   *
   * @return The number of bytes in this string
   */
  CUDF_HOST_DEVICE [[nodiscard]] inline size_type size_bytes() const { return _bytes; }
  /**
   * @brief Return the number of characters in this string
   *
   * @return The number of characters in this string
   */
  [[nodiscard]] __device__ inline size_type length() const;
  /**
   * @brief Return a pointer to the internal device array
   *
   * @return A pointer to the internal device array
   */
  CUDF_HOST_DEVICE [[nodiscard]] inline char const* data() const { return _data; }

  /**
   * @brief Return true if string has no characters
   *
   * @return true if string has no characters
   */
  CUDF_HOST_DEVICE [[nodiscard]] inline bool empty() const { return size_bytes() == 0; }

  /**
   * @brief Handy iterator for navigating through encoded characters.
   */
  class const_iterator {
    /// @cond
   public:
    using difference_type   = ptrdiff_t;
    using value_type        = char_utf8;
    using reference         = char_utf8&;
    using pointer           = char_utf8*;
    using iterator_category = cuda::std::input_iterator_tag;
    __device__ inline const_iterator(string_view const& str, size_type pos);
    const_iterator(const_iterator const& mit)        = default;
    const_iterator(const_iterator&& mit)             = default;
    const_iterator& operator=(const_iterator const&) = default;
    const_iterator& operator=(const_iterator&&)      = default;
    __device__ inline const_iterator& operator++();
    __device__ inline const_iterator operator++(int);
    __device__ inline const_iterator& operator+=(difference_type);
    __device__ inline const_iterator operator+(difference_type) const;
    __device__ inline const_iterator& operator--();
    __device__ inline const_iterator operator--(int);
    __device__ inline const_iterator& operator-=(difference_type);
    __device__ inline const_iterator operator-(difference_type) const;
    __device__ inline const_iterator& move_to(size_type);
    __device__ inline bool operator==(const_iterator const&) const;
    __device__ inline bool operator!=(const_iterator const&) const;
    __device__ inline bool operator<(const_iterator const&) const;
    __device__ inline bool operator<=(const_iterator const&) const;
    __device__ inline bool operator>(const_iterator const&) const;
    __device__ inline bool operator>=(const_iterator const&) const;
    __device__ inline char_utf8 operator*() const;
    [[nodiscard]] __device__ inline size_type position() const;
    [[nodiscard]] __device__ inline size_type byte_offset() const;

   private:
    friend class string_view;
    char const* p{};
    size_type bytes{};
    size_type char_pos{};
    size_type byte_pos{};
    __device__ inline const_iterator(string_view const& str, size_type pos, size_type offset);
    /// @endcond
  };

  /**
   * @brief Return new iterator pointing to the beginning of this string
   *
   * @return new iterator pointing to the beginning of this string
   */
  [[nodiscard]] __device__ inline const_iterator begin() const;
  /**
   * @brief Return new iterator pointing past the end of this string
   *
   * @return new iterator pointing past the end of this string
   */
  [[nodiscard]] __device__ inline const_iterator end() const;

  /**
   * @brief Return single UTF-8 character at the given character position
   *
   * @param pos Character position
   * @return UTF-8 character at the given character position
   */
  __device__ inline char_utf8 operator[](size_type pos) const;
  /**
   * @brief Return the byte offset from data() for a given character position
   *
   * @param pos Character position
   * @return Byte offset from data() for a given character position
   */
  [[nodiscard]] __device__ inline size_type byte_offset(size_type pos) const;

  /**
   * @brief Comparing target string with this string. Each character is compared
   * as a UTF-8 code-point value.
   *
   * @param str Target string to compare with this string.
   * @return 0  If they compare equal.
   *         <0 Either the value of the first character of this string that does
   *            not match is lower in the arg string, or all compared characters
   *            match but the arg string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is greater in the arg string, or all compared characters
   *            match but the arg string is longer.
   */
  [[nodiscard]] __device__ inline int compare(string_view const& str) const;
  /**
   * @brief Comparing target string with this string. Each character is compared
   * as a UTF-8 code-point value.
   *
   * @param str Target string to compare with this string.
   * @param bytes Number of bytes in str.
   * @return 0  If they compare equal.
   *         <0 Either the value of the first character of this string that does
   *            not match is lower in the arg string, or all compared characters
   *            match but the arg string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is greater in the arg string, or all compared characters
   *            match but the arg string is longer.
   */
  __device__ inline int compare(char const* str, size_type bytes) const;

  /**
   * @brief Returns true if rhs matches this string exactly.
   *
   * @param rhs Target string to compare with this string.
   * @return true if rhs matches this string exactly
   */
  __device__ inline bool operator==(string_view const& rhs) const;
  /**
   * @brief Returns true if rhs does not match this string.
   *
   * @param rhs Target string to compare with this string.
   * @return true if rhs does not match this string
   */
  __device__ inline bool operator!=(string_view const& rhs) const;
  /**
   * @brief Returns true if this string is ordered before rhs.
   *
   * @param rhs Target string to compare with this string.
   * @return true if this string is ordered before rhs
   */
  __device__ inline bool operator<(string_view const& rhs) const;
  /**
   * @brief Returns true if rhs is ordered before this string.
   *
   * @param rhs Target string to compare with this string.
   * @return true if rhs is ordered before this string
   */
  __device__ inline bool operator>(string_view const& rhs) const;
  /**
   * @brief Returns true if this string matches or is ordered before rhs.
   *
   * @param rhs Target string to compare with this string.
   * @return true if this string matches or is ordered before rhs
   */
  __device__ inline bool operator<=(string_view const& rhs) const;
  /**
   * @brief Returns true if rhs matches or is ordered before this string.
   *
   * @param rhs Target string to compare with this string.
   * @return true if rhs matches or is ordered before this string
   */
  __device__ inline bool operator>=(string_view const& rhs) const;

  /**
   * @brief Returns the character position of the first occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return npos if str is not found in this string.
   */
  [[nodiscard]] __device__ inline size_type find(string_view const& str,
                                                 size_type pos   = 0,
                                                 size_type count = -1) const;
  /**
   * @brief Returns the character position of the first occurrence where the
   * array str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target array to search within this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return npos if arg string is not found in this string.
   */
  __device__ inline size_type find(char const* str,
                                   size_type bytes,
                                   size_type pos   = 0,
                                   size_type count = -1) const;
  /**
   * @brief Returns the character position of the first occurrence where
   * character is found in this string within the character range [pos,pos+n).
   *
   * @param character Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return npos if arg string is not found in this string.
   */
  [[nodiscard]] __device__ inline size_type find(char_utf8 character,
                                                 size_type pos   = 0,
                                                 size_type count = -1) const;
  /**
   * @brief Returns the character position of the last occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return npos if arg string is not found in this string.
   */
  [[nodiscard]] __device__ inline size_type rfind(string_view const& str,
                                                  size_type pos   = 0,
                                                  size_type count = -1) const;
  /**
   * @brief Returns the character position of the last occurrence where the
   * array str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search with this string.
   * @param bytes Number of bytes in str.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return npos if arg string is not found in this string.
   */
  __device__ inline size_type rfind(char const* str,
                                    size_type bytes,
                                    size_type pos   = 0,
                                    size_type count = -1) const;
  /**
   * @brief Returns the character position of the last occurrence where
   * character is found in this string within the character range [pos,pos+n).
   *
   * @param character Single encoded character.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return npos if arg string is not found in this string.
   */
  [[nodiscard]] __device__ inline size_type rfind(char_utf8 character,
                                                  size_type pos   = 0,
                                                  size_type count = -1) const;

  /**
   * @brief Return a sub-string of this string. The original string and device
   * memory must still be maintained for the lifetime of the returned instance.
   *
   * @param start Character position to start the sub-string.
   * @param length Number of characters from start to include in the sub-string.
   * @return New instance pointing to a subset of the characters within this instance.
   */
  [[nodiscard]] __device__ inline string_view substr(size_type start, size_type length) const;

  /**
   * @brief Return minimum value associated with the string type
   *
   * This function is needed to be host callable because it is called by a host
   * callable function DeviceMax::identity<string_view>()
   *
   * @return An empty string
   */
  CUDF_HOST_DEVICE inline static string_view min();

  /**
   * @brief Return maximum value associated with the string type
   *
   * This function is needed to be host callable because it is called by a host
   * callable function DeviceMin::identity<string_view>()
   *
   * @return A string value which represents the highest possible valid UTF-8 encoded
   * character.
   */
  CUDF_HOST_DEVICE inline static string_view max();

  /**
   * @brief Default constructor represents an empty string.
   */
  CUDF_HOST_DEVICE inline string_view() : _data("") {}

  /**
   * @brief Create instance from existing device char array.
   *
   * @param data Device char array encoded in UTF8.
   * @param bytes Number of bytes in data array.
   */
  CUDF_HOST_DEVICE inline string_view(char const* data, size_type bytes)
    : _data(data), _bytes(bytes), _length(UNKNOWN_STRING_LENGTH)
  {
  }

  string_view(string_view const&) = default;  ///< Copy constructor
  string_view(string_view&&)      = default;  ///< Move constructor
  ~string_view()                  = default;
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this instance
   */
  string_view& operator=(string_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this instance (after transferring ownership)
   */
  string_view& operator=(string_view&&) = default;

  /**
   * @brief No-position value.
   *
   * Used when specifying or returning an invalid or unknown character position value.
   */
  static inline cudf::size_type const npos{-1};

 private:
  char const* _data{};          ///< Pointer to device memory contain char array for this string
  size_type _bytes{};           ///< Number of bytes in _data for this string
  mutable size_type _length{};  ///< Number of characters in this string (computed)

  /**
   * @brief The string length is initialized to this value as a place-holder
   *
   * The number of characters in a string is computed on-demand.
   */
  static inline cudf::size_type const UNKNOWN_STRING_LENGTH{-1};

  /**
   * @brief Return the character position of the given byte offset.
   *
   * @param bytepos Byte position from start of _data.
   * @return The character position for the specified byte.
   */
  [[nodiscard]] __device__ inline size_type character_offset(size_type bytepos) const;

  /**
   * @brief Common internal implementation for string_view::find and string_view::rfind.
   *
   * @tparam forward True for find and false for rfind
   *
   * @param str Target string to search with this string
   * @param bytes Number of bytes in str
   * @param pos Character position to start search within this string
   * @param count Number of characters from pos to include in the search
   * @return npos if str is not found in this string
   */
  template <bool forward>
  __device__ inline size_type find_impl(char const* str,
                                        size_type bytes,
                                        size_type pos,
                                        size_type count) const;
};

}  // namespace CUDF_EXPORT cudf
