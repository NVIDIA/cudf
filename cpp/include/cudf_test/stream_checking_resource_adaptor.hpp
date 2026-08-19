/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf_test/default_stream.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>
#include <iostream>

namespace cudf::test {

/**
 * @brief Resource that verifies that the default stream is not used in any allocation.
 */
class stream_checking_resource_adaptor final {
 public:
  /**
   * @brief Construct a new adaptor.
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param error_on_invalid_stream Whether to error on invalid streams
   * @param check_default_stream Whether to check for the default stream
   */
  stream_checking_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                   bool error_on_invalid_stream,
                                   bool check_default_stream)
    : upstream_{std::move(upstream)},
      error_on_invalid_stream_{error_on_invalid_stream},
      check_default_stream_{check_default_stream}
  {
  }

  stream_checking_resource_adaptor()                                                   = delete;
  ~stream_checking_resource_adaptor()                                                  = default;
  stream_checking_resource_adaptor(stream_checking_resource_adaptor const&)            = default;
  stream_checking_resource_adaptor& operator=(stream_checking_resource_adaptor const&) = default;
  stream_checking_resource_adaptor(stream_checking_resource_adaptor&&) noexcept        = default;
  stream_checking_resource_adaptor& operator=(stream_checking_resource_adaptor&&) noexcept =
    default;

  /**
   * @brief Returns the wrapped upstream resource
   *
   * @return The wrapped upstream resource
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return rmm::device_async_resource_ref{
      const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_)};
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return upstream_.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    upstream_.deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    verify_stream(rmm::cuda_stream_view{stream.get()});
    return upstream_.allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    verify_stream(rmm::cuda_stream_view{stream.get()});
    upstream_.deallocate(stream, ptr, bytes, alignment);
  }

  bool operator==(stream_checking_resource_adaptor const& other) const noexcept
  {
    return get_upstream_resource() == other.get_upstream_resource();
  }

  bool operator!=(stream_checking_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }

  friend void get_property(stream_checking_resource_adaptor const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  /**
   * @brief Throw an error if the provided stream is invalid.
   *
   * A stream is invalid if:
   * - check_default_stream_ is true and this function is passed one of CUDA's
   *   default stream specifiers, or
   * - check_default_stream_ is false and this function is passed any stream
   *   other than the result of cudf::test::get_default_stream().
   *
   * @throws `std::runtime_error` if provided an invalid stream
   */
  void verify_stream(rmm::cuda_stream_view const stream) const
  {
    auto cstream{stream.value()};
    auto const invalid_stream =
      check_default_stream_ ? ((cstream == cudaStreamDefault) || (cstream == cudaStreamLegacy) ||
                               (cstream == cudaStreamPerThread))
                            : (cstream != cudf::test::get_default_stream().value());

    if (invalid_stream) {
      if (error_on_invalid_stream_) {
        throw std::runtime_error("Attempted to perform an operation on an unexpected stream!");
      } else {
        std::cout << "Attempted to perform an operation on an unexpected stream!" << std::endl;
      }
    }
  }

  cuda::mr::any_resource<cuda::mr::device_accessible>
    upstream_;                    // the upstream resource used for satisfying allocation requests
  bool error_on_invalid_stream_;  // If true, throw an exception when the wrong stream is detected.
                                  // If false, simply print to stdout.
  bool check_default_stream_;  // If true, throw an exception when the default stream is observed.
                               // If false, throw an exception when anything other than
                               // cudf::test::get_default_stream() is observed.
};

static_assert(
  cuda::mr::resource_with<stream_checking_resource_adaptor, cuda::mr::device_accessible>,
  "stream_checking_resource_adaptor does not satisfy the cuda::mr::resource concept");

}  // namespace cudf::test
