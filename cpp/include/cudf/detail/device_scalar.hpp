/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <type_traits>
#include <utility>

namespace CUDF_EXPORT cudf {
namespace detail {

template <typename T>
class device_scalar {
 public:
  static_assert(std::is_trivially_copyable_v<T>,
                "cudf::detail::device_scalar<T> requires T to be trivially copyable");
  using value_type = T;

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  ~device_scalar() = default;

  device_scalar(device_scalar&& other) noexcept      = default;
  device_scalar& operator=(device_scalar&&) noexcept = default;

  device_scalar(device_scalar const&)            = delete;
  device_scalar& operator=(device_scalar const&) = delete;

  device_scalar() = delete;

  explicit device_scalar(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : _storage{1, stream, std::move(mr)}, bounce_buffer{make_pinned_vector<T>(1, stream)}
  {
  }

  explicit device_scalar(
    T const& initial_value,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : _storage{1, stream, std::move(mr)}, bounce_buffer{make_pinned_vector<T>(1, stream)}
  {
    set_value_async(initial_value, stream);
  }

  device_scalar(device_scalar const& other,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : _storage{other._storage, stream, mr}, bounce_buffer{make_pinned_vector<T>(1, stream)}
  {
  }

  [[nodiscard]] T value(rmm::cuda_stream_view stream) const
  {
    cuda_memcpy<T>(bounce_buffer, device_span<T const>{data(), 1}, stream);
    return std::move(bounce_buffer[0]);
  }

  void set_value_async(T const& value, rmm::cuda_stream_view stream)
  {
    bounce_buffer[0] = value;
    cuda_memcpy_async<T>(device_span<T>{data(), 1}, bounce_buffer, stream);
  }

  void set_value_async(T&& value, rmm::cuda_stream_view stream)
  {
    bounce_buffer[0] = std::move(value);
    cuda_memcpy_async<T>(device_span<T>{data(), 1}, bounce_buffer, stream);
  }

  void set_value_to_zero_async(rmm::cuda_stream_view stream) { set_value_async(T{}, stream); }

  [[nodiscard]] T* data() noexcept { return _storage.data(); }

  [[nodiscard]] T const* data() const noexcept { return _storage.data(); }

 private:
  rmm::device_uvector<T> _storage;
  mutable cudf::detail::host_vector<T> bounce_buffer;
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
