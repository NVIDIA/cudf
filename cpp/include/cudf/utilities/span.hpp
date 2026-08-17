/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <cuda/std/span>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

/**
 * @file
 * @brief APIs for spans
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup utility_span
 * @{
 */

/// A constant used to differentiate std::span of static and dynamic extent
constexpr std::size_t dynamic_extent = cuda::std::dynamic_extent;

// ===== host_span =================================================================================

template <typename T>
struct is_host_span_supported_container : std::false_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  std::vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  thrust::host_vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  std::basic_string<T, std::char_traits<T>, Alloc>> : std::true_type {};

/**
 * @brief Host span, a non-owning view over a contiguous sequence of host-accessible elements.
 *
 * Backed by `cuda::std::span`, with additional support for constructing from cudf-supported host
 * containers and for tracking whether the underlying memory is device accessible (e.g. pinned
 * memory), which enables copy engine optimizations.
 */
template <typename T, std::size_t Extent = cudf::dynamic_extent>
struct host_span {
 private:
  using span_type = cuda::std::span<T, Extent>;  ///< The underlying span type

 public:
  using element_type    = typename span_type::element_type;     ///< Element type
  using value_type      = typename span_type::value_type;       ///< Stored value type
  using size_type       = typename span_type::size_type;        ///< Size type
  using difference_type = typename span_type::difference_type;  ///< std::ptrdiff_t
  using pointer         = typename span_type::pointer;          ///< Pointer returned by data()
  using const_pointer   = typename span_type::const_pointer;  ///< Pointer returned by data() const
  using reference       = typename span_type::reference;      ///< Reference returned by operator[]
  using const_reference = typename span_type::const_reference;  ///< Const reference to an element
  using iterator        = pointer;  ///< The type of the iterator returned by begin()

  static constexpr std::size_t extent = span_type::extent;  ///< The extent of the span

  constexpr host_span() noexcept {}  // required to compile on centos

  /**
   * @brief Constructs a span from a pointer and a size.
   *
   * @note This needs to be host-device, as it's used by a host-device function in base_2dspan
   *
   * @param data Pointer to the first element in the span
   * @param size The number of elements in the span
   */
  CUDF_HOST_DEVICE constexpr host_span(T* data, std::size_t size) : _span{data, size} {}

  /**
   * @brief Constructor from pointer, size and device-accessibility flag
   *
   * @note This needs to be host-device, as it's used by a host-device function in base_2dspan
   *
   * @param data Pointer to the first element in the span
   * @param size The number of elements in the span
   * @param is_device_accessible Whether the data is device accessible (e.g. pinned memory)
   */
  CUDF_HOST_DEVICE constexpr host_span(T* data, std::size_t size, bool is_device_accessible)
    : _span{data, size}, _is_device_accessible{is_device_accessible}
  {
  }

  /// Constructor from container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<is_host_span_supported_container<C>::value &&
                             std::is_convertible_v<
                               std::remove_pointer_t<decltype(thrust::raw_pointer_cast(  // NOLINT
                                 std::declval<C&>().data()))> (*)[],
                               T (*)[]>>* = nullptr>  // NOLINT
  constexpr host_span(C& in) : _span{thrust::raw_pointer_cast(in.data()), in.size()}
  {
  }

  /// Constructor from const container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<is_host_span_supported_container<C>::value &&
                             std::is_convertible_v<
                               std::remove_pointer_t<decltype(thrust::raw_pointer_cast(  // NOLINT
                                 std::declval<C&>().data()))> (*)[],
                               T (*)[]>>* = nullptr>  // NOLINT
  constexpr host_span(C const& in) : _span{thrust::raw_pointer_cast(in.data()), in.size()}
  {
  }

  // Copy construction to support const conversion
  /// @param other The span to copy
  template <typename OtherT,
            std::size_t OtherExtent,
            std::enable_if_t<(Extent == OtherExtent || Extent == dynamic_extent) &&
                               std::is_convertible_v<OtherT (*)[], T (*)[]>,  // NOLINT
                             void>* = nullptr>
  constexpr host_span(host_span<OtherT, OtherExtent> const& other) noexcept
    : _span{other.data(), other.size()}, _is_device_accessible{other.is_device_accessible()}
  {
  }

  // not noexcept due to undefined behavior when idx < 0 || idx >= size
  /**
   * @brief Returns a reference to the idx-th element of the sequence.
   *
   * The behavior is undefined if idx is out of range (i.e., if it is greater than or equal to
   * size()).
   *
   * @param idx the index of the element to access
   * @return A reference to the idx-th element of the sequence, i.e., `data()[idx]`
   */
  constexpr reference operator[](size_type idx) const
  {
    static_assert(sizeof(idx) >= sizeof(size_t), "index type must not be smaller than size_t");
    assert(idx < _span.size());
    return _span[idx];
  }

  // not noexcept due to undefined behavior when size = 0
  /**
   * @brief Returns a reference to the first element in the span.
   *
   * Calling front on an empty span results in undefined behavior.
   *
   * @return Reference to the first element in the span
   */
  [[nodiscard]] constexpr reference front() const { return _span.front(); }
  // not noexcept due to undefined behavior when size = 0
  /**
   * @brief Returns a reference to the last element in the span.
   *
   * Calling back on an empty span results in undefined behavior.
   *
   * @return Reference to the last element in the span
   */
  [[nodiscard]] constexpr reference back() const { return _span.back(); }

  /**
   * @brief Returns an iterator to the first element of the span.
   *
   * If the span is empty, the returned iterator will be equal to end().
   *
   * @return An iterator to the first element of the span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr iterator begin() const noexcept { return _span.data(); }
  /**
   * @brief Returns an iterator to the element following the last element of the span.
   *
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   *
   * @return An iterator to the element following the last element of the span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr iterator end() const noexcept
  {
    return _span.data() + _span.size();
  }
  /**
   * @brief Returns a pointer to the beginning of the sequence.
   *
   * @return A pointer to the first element of the span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr pointer data() const noexcept { return _span.data(); }

  /**
   * @brief Returns the number of elements in the span.
   *
   * @return The number of elements in the span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type size() const noexcept { return _span.size(); }
  /**
   * @brief Returns the size of the sequence in bytes.
   *
   * @return The size of the sequence in bytes
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type size_bytes() const noexcept
  {
    return _span.size_bytes();
  }

  /**
   * @brief Checks if the span is empty.
   *
   * @return True if the span is empty, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool empty() const noexcept { return _span.empty(); }

  /**
   * @brief Obtains a subspan consisting of the first count elements of the sequence
   *
   * @param count Number of elements from the beginning of this span to put in the subspan.
   * @return A subspan of the first count elements of the sequence
   */
  [[nodiscard]] constexpr host_span first(size_type count) const noexcept
  {
    assert(count <= _span.size());
    return host_span{_span.data(), count, _is_device_accessible};
  }

  /**
   * @brief Obtains a subspan consisting of the last count elements of the sequence
   *
   * @param count Number of elements from the end of this span to put in the subspan
   * @return A subspan of the last count elements of the sequence
   */
  [[nodiscard]] constexpr host_span last(size_type count) const noexcept
  {
    assert(count <= _span.size());
    return host_span{_span.data() + _span.size() - count, count, _is_device_accessible};
  }

  /**
   * @brief Returns whether the data is device accessible (e.g. pinned memory)
   *
   * @return true if the data is device accessible
   */
  [[nodiscard]] bool is_device_accessible() const { return _is_device_accessible; }

  /**
   * @brief Obtains a span that is a view over the `count` elements of this span starting at offset
   *
   * @param offset The offset of the first element in the subspan
   * @param count The number of elements in the subspan
   * @return A subspan of the sequence, of requested count and offset
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr host_span subspan(size_type offset,
                                                             size_type count) const noexcept
  {
    assert(offset <= _span.size() && count <= _span.size() - offset);
    return host_span{_span.data() + offset, count, _is_device_accessible};
  }

  /**
   * @brief Returns a standard span instance
   *
   * @return Standard span instance
   */
  [[nodiscard]] constexpr operator std::span<T>() const noexcept
  {
    return std::span<T>(_span.data(), _span.size());
  }

 private:
  // TODO: could be std::span once base_2dspan moves to cuda::std::mdspan and host_span no longer
  // needs to be device-usable.
  span_type _span;
  bool _is_device_accessible{false};
};

// ===== device_span ===============================================================================

/**
 * @brief Device span is an alias of cuda::std::span.
 *
 */
template <typename T, std::size_t Extent = cuda::std::dynamic_extent>
using device_span = cuda::std::span<T, Extent>;
/** @} */  // end of group

namespace detail {

/**
 * @brief Generic class for row-major 2D spans. Not compliant with STL container semantics/syntax.
 *
 * The index operator returns the corresponding row.
 */
template <typename T, template <typename, std::size_t> typename RowType>
class base_2dspan {
 public:
  using size_type =
    std::pair<size_t, size_t>;  ///< Type used to represent the dimension of the span

  constexpr base_2dspan() noexcept = default;
  /**
   * @brief Constructor from a span and number of elements in each row.
   *
   * @param flat_view The flattened 2D span
   * @param columns Number of columns
   */
  constexpr base_2dspan(RowType<T, dynamic_extent> flat_view, size_t columns)
    : _flat{flat_view}, _size{columns == 0 ? 0 : flat_view.size() / columns, columns}
  {
#ifndef __CUDA_ARCH__
    CUDF_EXPECTS(_size.first * _size.second == flat_view.size(), "Invalid 2D span size");
#endif
  }

  /**
   * @brief Returns a pointer to the beginning of the sequence.
   *
   * @return A pointer to the first element of the span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr auto data() const noexcept { return _flat.data(); }

  /**
   * @brief Returns the size in the span as pair.
   *
   * @return pair representing rows and columns size of the span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr auto size() const noexcept { return _size; }

  /**
   * @brief Returns the number of elements in the span.
   *
   * @return Number of elements in the span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr auto count() const noexcept { return _flat.size(); }

  /**
   * @brief Checks if the span is empty.
   *
   * @return True if the span is empty, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_empty() const noexcept { return count() == 0; }

  /**
   * @brief Returns a reference to the row-th element of the sequence.
   *
   * The behavior is undefined if row is out of range (i.e., if it is greater than or equal to
   * size()).
   *
   * @param row the index of the element to access
   * @return A reference to the row-th element of the sequence, i.e., `data()[row]`
   */
  CUDF_HOST_DEVICE constexpr RowType<T, dynamic_extent> operator[](std::size_t row) const
  {
    return _flat.subspan(row * _size.second, _size.second);
  }

  /**
   * @brief Returns a flattened span of the 2D span.
   *
   * @return A flattened span of the 2D span
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr RowType<T, dynamic_extent> flat_view() const
  {
    return _flat;
  }

  /**
   * @brief Construct a 2D span from another 2D span of convertible type
   *
   * @tparam OtherT Type of the other 2D span
   * @tparam OtherRowType Type of the row of the other 2D span
   * @param other The other 2D span
   */
  template <typename OtherT,
            template <typename, size_t> typename OtherRowType,
            std::enable_if_t<std::is_convertible_v<OtherRowType<OtherT, dynamic_extent>,
                                                   RowType<T, dynamic_extent>>,
                             void>* = nullptr>
  constexpr base_2dspan(base_2dspan<OtherT, OtherRowType> const& other) noexcept
    : _flat{other.flat_view()}, _size{other.size()}
  {
  }

 protected:
  RowType<T, dynamic_extent> _flat;  ///< flattened 2D span
  size_type _size{0, 0};             ///< num rows, num columns
};

/**
 * @brief Alias for the 2D span for host data.
 *
 * Index operator returns rows as `host_span`.
 */
template <class T>
using host_2dspan = base_2dspan<T, host_span>;

/**
 * @brief Alias for the 2D span for device data.
 *
 * Index operator returns rows as `device_span`.
 */
template <class T>
using device_2dspan = base_2dspan<T, device_span>;

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
