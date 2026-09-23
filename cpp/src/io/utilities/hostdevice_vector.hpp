/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hostdevice_span.hpp"

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf::detail {

/**
 * @brief A helper class that wraps fixed-length device memory for the GPU, and
 * a mirror host pinned memory for the CPU.
 *
 * This abstraction allocates a specified fixed chunk of device memory that can
 * initialized upfront, or gradually initialized as required.
 * The host-side memory can be used to manipulate data on the CPU before and
 * after operating on the same data on the GPU.
 *
 * On systems with integrated memory, this class uses only pinned buffer memory
 * that is accessible by both host and device, eliminating the need for separate
 * device memory allocation and data transfers. This optimization can be controlled
 * via the LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION environment variable (AUTO by default,
 * which uses hardware detection).
 */
template <typename T>
class hostdevice_vector {
 public:
  using value_type = T;

  hostdevice_vector() : hostdevice_vector(0, cudf::get_default_stream()) {}

  explicit hostdevice_vector(size_t size, cuda::stream_ref stream)
    : keep_single_copy{cudf::io::integrated_memory_optimization::is_enabled()},
      h_data{make_pinned_vector_async<T>(size, stream)},
      d_data{keep_single_copy ? 0 : size, stream},
      _device_ptr{keep_single_copy ? h_data.data() : d_data.data()}
  {
  }

  [[nodiscard]] size_t size() const noexcept { return h_data.size(); }
  [[nodiscard]] size_t size_bytes() const noexcept { return sizeof(T) * size(); }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  [[nodiscard]] T& operator[](size_t i) { return h_data[i]; }
  [[nodiscard]] T const& operator[](size_t i) const { return h_data[i]; }

  [[nodiscard]] T* host_ptr(size_t offset = 0) { return h_data.data() + offset; }
  [[nodiscard]] T const* host_ptr(size_t offset = 0) const { return h_data.data() + offset; }

  [[nodiscard]] T* begin() { return host_ptr(); }
  [[nodiscard]] T const* begin() const { return host_ptr(); }

  [[nodiscard]] T* end() { return host_ptr(size()); }
  [[nodiscard]] T const* end() const { return host_ptr(size()); }

  [[nodiscard]] T& front() { return h_data.front(); }
  [[nodiscard]] T const& front() const { return front(); }

  [[nodiscard]] T& back() { return h_data.back(); }
  [[nodiscard]] T const& back() const { return back(); }

  [[nodiscard]] T* device_ptr(size_t offset = 0) { return _device_ptr + offset; }
  [[nodiscard]] T const* device_ptr(size_t offset = 0) const { return _device_ptr + offset; }

  [[nodiscard]] T* d_begin() { return device_ptr(); }
  [[nodiscard]] T const* d_begin() const { return device_ptr(); }

  [[nodiscard]] T* d_end() { return device_ptr(size()); }
  [[nodiscard]] T const* d_end() const { return device_ptr(size()); }

  operator cudf::host_span<T>() { return host_span<T>(host_ptr(), size(), true); }
  operator cudf::host_span<T const>() const { return host_span<T const>(host_ptr(), size(), true); }

  operator cudf::device_span<T>() { return cudf::device_span<T>(device_ptr(), size()); }
  operator cudf::device_span<T const>() const
  {
    return cudf::device_span<T const>(device_ptr(), size());
  }

  void host_to_device_async(cuda::stream_ref stream)
  {
    if (not keep_single_copy) { cuda_memcpy_async<T>(d_data, h_data, stream); }
  }

  [[deprecated("Use host_to_device_async instead")]] void host_to_device(cuda::stream_ref stream)
  {
    host_to_device_async(stream);
    stream.sync();
  }
  void device_to_host_async(cuda::stream_ref stream)
  {
    if (not keep_single_copy) { cuda_memcpy_async<T>(h_data, d_data, stream); }
  }

  void device_to_host(cuda::stream_ref stream)
  {
    device_to_host_async(stream);
    stream.sync();
  }

  /**
   * @brief Converts a hostdevice_vector into a hostdevice_span.
   *
   * @return A typed hostdevice_span of the hostdevice_vector's data
   */
  [[nodiscard]] operator hostdevice_span<T>() { return {host_span<T>{h_data}, device_ptr()}; }

  [[nodiscard]] operator hostdevice_span<T const>() const
  {
    return {host_span<T const>{h_data}, device_ptr()};
  }

 private:
  bool keep_single_copy;
  cudf::detail::host_vector<T> h_data;
  rmm::device_uvector<T> d_data;
  T* _device_ptr{};  // Device pointer for integrated memory systems
};

/**
 * @brief Wrapper around hostdevice_vector to enable two-dimensional indexing.
 *
 * Does not incur additional allocations.
 */
template <typename T>
class hostdevice_2dvector {
 public:
  hostdevice_2dvector() : hostdevice_2dvector(0, 0, cudf::get_default_stream()) {}

  hostdevice_2dvector(size_t rows, size_t columns, cuda::stream_ref stream)
    : _data{rows * columns, stream}, _size{rows, columns}
  {
  }

  operator device_2dspan<T>()
  {
    return device_2dspan<T>(device_span<T>(_data.device_ptr(), _data.size()), _size.second);
  }
  operator device_2dspan<T const>() const
  {
    return device_2dspan<T const>(device_span<T const>(_data.device_ptr(), _data.size()),
                                  _size.second);
  }

  device_2dspan<T> device_view() { return static_cast<device_2dspan<T>>(*this); }
  [[nodiscard]] device_2dspan<T const> device_view() const
  {
    return static_cast<device_2dspan<T const>>(*this);
  }

  operator host_2dspan<T>()
  {
    return host_2dspan<T>(host_span<T>(_data.host_ptr(), _data.size(), true), _size.second);
  }
  operator host_2dspan<T const>() const
  {
    return host_2dspan<T const>(host_span<T const>(_data.host_ptr(), _data.size(), true),
                                _size.second);
  }

  host_2dspan<T> host_view() { return static_cast<host_2dspan<T>>(*this); }
  [[nodiscard]] host_2dspan<T const> host_view() const
  {
    return static_cast<host_2dspan<T const>>(*this);
  }

  host_span<T> operator[](size_t row)
  {
    return host_span<T>(_data.host_ptr(), _data.size(), true)
      .subspan(row * _size.second, _size.second);
  }

  host_span<T const> operator[](size_t row) const
  {
    return host_span<T const>(_data.host_ptr(), _data.size(), true)
      .subspan(row * _size.second, _size.second);
  }

  [[nodiscard]] auto size() const noexcept { return _size; }
  [[nodiscard]] auto count() const noexcept { return _size.first * _size.second; }
  [[nodiscard]] auto is_empty() const noexcept { return count() == 0; }

  T* base_host_ptr(size_t offset = 0) { return _data.host_ptr(offset); }
  T* base_device_ptr(size_t offset = 0) { return _data.device_ptr(offset); }

  [[nodiscard]] T const* base_host_ptr(size_t offset = 0) const { return _data.host_ptr(offset); }

  [[nodiscard]] T const* base_device_ptr(size_t offset = 0) const
  {
    return _data.device_ptr(offset);
  }

  [[nodiscard]] size_t size_bytes() const noexcept { return _data.size_bytes(); }

  void host_to_device_async(cuda::stream_ref stream) { _data.host_to_device_async(stream); }
  [[deprecated("Use host_to_device_async instead")]] void host_to_device(cuda::stream_ref stream)
  {
    _data.host_to_device(stream);
  }

  void device_to_host_async(cuda::stream_ref stream) { _data.device_to_host_async(stream); }
  void device_to_host(cuda::stream_ref stream) { _data.device_to_host(stream); }

 private:
  hostdevice_vector<T> _data;
  typename host_2dspan<T>::size_type _size;
};

}  // namespace cudf::detail
