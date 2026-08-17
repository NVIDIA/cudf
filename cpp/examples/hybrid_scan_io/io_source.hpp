/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <thrust/host_vector.h>

#include <string>

/**
 * @file io_source.hpp
 * @brief Utilities for constructing the specified IO sources from the input parquet files.
 *
 */

/**
 * @brief Available IO source types
 */
enum class io_source_type : uint8_t { FILEPATH, HOST_BUFFER, PINNED_BUFFER, DEVICE_BUFFER };

/**
 * @brief Get io source type from the string keyword argument
 *
 * @param name io source type keyword name
 * @return io source type
 */
[[nodiscard]] io_source_type get_io_source_type(std::string name);

/**
 * @brief Create and return a reference to a static pinned memory pool
 *
 * @return Reference to a static pinned memory pool
 */
rmm::host_async_resource_ref pinned_memory_resource();

/**
 * @brief Custom allocator for pinned_buffer via RMM.
 */
template <typename T>
struct pinned_allocator : public std::allocator<T> {
  pinned_allocator(rmm::host_async_resource_ref _mr, cuda::stream_ref _stream)
    : mr{_mr}, stream{_stream}
  {
  }

  T* allocate(std::size_t n)
  {
    auto ptr = mr.allocate(stream, n * sizeof(T), alignof(T));
    stream.sync();
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t n) noexcept
  {
    mr.deallocate(stream, ptr, n * sizeof(T), alignof(T));
  }

 private:
  rmm::host_async_resource_ref mr;
  cuda::stream_ref stream;
};

/**
 * @brief Class to create a cudf::io::source_info of given type from the input parquet file
 *
 */
class io_source {
 public:
  io_source(std::string_view file_path, io_source_type io_type, cuda::stream_ref stream);

  // Get the internal source info
  [[nodiscard]] cudf::io::source_info get_source_info() const { return source_info; }

 private:
  // alias for pinned vector
  template <typename T>
  using pinned_vector = thrust::host_vector<T, pinned_allocator<T>>;
  cudf::io::source_info source_info;
  std::vector<char> h_buffer;
  pinned_vector<char> pinned_buffer;
  rmm::device_uvector<std::byte> d_buffer;
};
