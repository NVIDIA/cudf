/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::detail {

template <typename T>
class hostdevice_span {
 public:
  using value_type = T;

  hostdevice_span()                       = default;
  ~hostdevice_span()                      = default;
  hostdevice_span(hostdevice_span const&) = default;  ///< Copy constructor
  hostdevice_span(hostdevice_span&&)      = default;  ///< Move constructor

  hostdevice_span(host_span<T> host_data, T* device_data)
    : _host_data{host_data}, _device_data{device_data}
  {
  }

  // Copy construction to support const conversion
  /// @param other The span to copy
  template <typename OtherT,
            std::enable_if_t<std::is_convertible_v<OtherT (*)[], T (*)[]>,  // NOLINT
                             void>* = nullptr>
  constexpr hostdevice_span(hostdevice_span<OtherT> const& other) noexcept
    : _host_data{host_span<OtherT>{other}}, _device_data{other.device_ptr()}
  {
  }

  /**
   * @brief Copy assignment operator.
   *
   * @return Reference to this hostdevice_span.
   */
  constexpr hostdevice_span& operator=(hostdevice_span const&) noexcept = default;

  /**
   * @brief Converts a hostdevice view into a device span.
   *
   * @tparam T The device span type.
   * @return A typed device span of the hostdevice view's data.
   */
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T (*)[], U (*)[]>,  // NOLINT
                             void>* = nullptr>
  [[nodiscard]] operator cudf::device_span<U>() const noexcept
  {
    return cudf::device_span<U>(_device_data, size());
  }

  /**
   * @brief Returns the underlying device data.
   *
   * @tparam T The type to cast to
   * @return T const* Typed pointer to underlying data
   */
  [[nodiscard]] T* device_ptr(size_t offset = 0) const noexcept { return _device_data + offset; }

  /**
   * @brief Return first element in device data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to the first element
   */
  [[nodiscard]] T* device_begin() const noexcept { return device_ptr(); }

  /**
   * @brief Return one past the last element in device_data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to one past the last element
   */
  [[nodiscard]] T* device_end() const noexcept { return device_begin() + size(); }

  /**
   * @brief Converts a hostdevice_span into a host span.
   *
   * @tparam T The host span type.
   * @return A typed host span of the hostdevice_span's data.
   */
  template <typename U,
            std::enable_if_t<std::is_convertible_v<T (*)[], U (*)[]>,  // NOLINT
                             void>* = nullptr>
  [[nodiscard]] operator host_span<U>() const noexcept
  {
    return {_host_data};
  }

  /**
   * @brief Returns the underlying host data.
   *
   * @tparam T The type to cast to
   * @return T* Typed pointer to underlying data
   */
  [[nodiscard]] T* host_ptr(size_t offset = 0) const noexcept { return _host_data.data() + offset; }

  /**
   * @brief Return first element in host data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to the first element
   */
  [[nodiscard]] T* host_begin() const noexcept { return host_ptr(); }

  /**
   * @brief Return one past the last element in host data.
   *
   * @tparam T The desired type
   * @return T const* Pointer to one past the last element
   */
  [[nodiscard]] T* host_end() const noexcept { return _host_data.end(); }

  /**
   * @brief Returns the number of elements in the view
   *
   * @return The number of elements in the view
   */
  [[nodiscard]] std::size_t size() const noexcept { return _host_data.size(); }

  /**
   * @brief Returns true if `size()` returns zero, or false otherwise
   *
   * @return True if `size()` returns zero, or false otherwise
   */
  [[nodiscard]] bool is_empty() const noexcept { return size() == 0; }

  [[nodiscard]] size_t size_bytes() const noexcept { return sizeof(T) * size(); }

  [[nodiscard]] T& operator[](size_t i) const { return _host_data[i]; }

  /**
   * @brief Obtains a `hostdevice_span` that is a view over the `count` elements of this
   * hostdevice_span starting at `offset`
   *
   * @param offset The offset of the first element in the subspan
   * @param count The number of elements in the subspan
   * @return A subspan of the sequence, of requested count and offset
   */
  [[nodiscard]] constexpr hostdevice_span<T> subspan(size_t offset, size_t count) const noexcept
  {
    return hostdevice_span<T>(_host_data.subspan(offset, count), device_ptr(offset));
  }

  void host_to_device_async(cuda::stream_ref stream) const
  {
    static_assert(not std::is_const_v<T>, "Cannot copy to const device memory");
    cudf::detail::cuda_memcpy_async<T>(device_span<T>{device_ptr(), size()}, _host_data, stream);
  }

  [[deprecated("Use host_to_device_async instead")]] void host_to_device(
    cuda::stream_ref stream) const
  {
    host_to_device_async(stream);
    stream.sync();
  }

  void device_to_host_async(cuda::stream_ref stream) const
  {
    static_assert(not std::is_const_v<T>, "Cannot copy to const host memory");
    cudf::detail::cuda_memcpy_async<T>(
      _host_data, device_span<T const>{device_ptr(), size()}, stream);
  }

  void device_to_host(cuda::stream_ref stream) const
  {
    device_to_host_async(stream);
    stream.sync();
  }

 private:
  host_span<T> _host_data;
  T* _device_data{nullptr};
};

}  // namespace cudf::detail
