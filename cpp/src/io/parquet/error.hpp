/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/hostdevice_vector.hpp"

#include <cuda/stream>

#include <cstdint>
#include <sstream>

namespace cudf::io::parquet {

/**
 * @brief Specialized device scalar for use in reporting errors that occur in
 * kernel calls.
 *
 * The `kernel_error` object is created with a `cuda::stream_ref` which is used throughout
 * the object's lifetime.
 */
class kernel_error {
 public:
  using value_type = uint32_t;
  using pointer    = value_type*;

 private:
  mutable cudf::detail::hostdevice_vector<value_type> _error_code;

 public:
  /**
   * @brief Construct a new `kernel_error` with an initial value of 0.
   *
   * Note: the initial value is set asynchronously.
   *
   * @throws `rmm::bad_alloc` if allocating the device memory for `initial_value` fails.
   * @throws `rmm::cuda_error` if copying `initial_value` to device memory fails.
   *
   * @param stream CUDA stream to use
   */
  kernel_error(cuda::stream_ref stream) : _error_code(1, stream)
  {
    _error_code[0] = 0;
    _error_code.host_to_device_async(stream);
  }

  /**
   * @brief Return a pointer to the device memory for the error
   */
  [[nodiscard]] auto data() { return _error_code.device_ptr(); }

  /**
   * @brief Return the current value of the error
   *
   * @param stream The CUDA stream to synchronize with
   */
  [[nodiscard]] auto value_sync(cuda::stream_ref stream) const
  {
    _error_code.device_to_host(stream);
    return _error_code[0];
  }

  /**
   * @brief Return a hexadecimal string representation of an error code
   *
   * Returned string will have "0x" prepended.
   *
   * @param value The error code to convert to a string
   */
  [[nodiscard]] static std::string to_string(value_type value)
  {
    std::stringstream sstream;
    sstream << std::hex << value;
    return "0x" + sstream.str();
  }
};

}  // namespace cudf::io::parquet
