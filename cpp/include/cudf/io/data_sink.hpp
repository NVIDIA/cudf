/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <vector>

/**
 * @file
 * @brief Interface classes for writing output data to files, host memory, or device memory.
 */

namespace CUDF_EXPORT cudf {
//! IO interfaces
namespace io {

/**
 * @addtogroup io_datasinks
 * @{
 */

/**
 * @brief Interface class for storing the output data from the writers
 */
class data_sink {
 public:
  /**
   * @brief Create a sink from a file path
   *
   * @param[in] filepath Path to the file to use
   * @return Constructed data_sink object
   */
  static std::unique_ptr<data_sink> create(std::string const& filepath);

  /**
   * @brief Create a sink from a std::vector
   *
   * @param[in,out] buffer Pointer to the output vector
   * @return Constructed data_sink object
   */
  static std::unique_ptr<data_sink> create(std::vector<char>* buffer);

  /**
   * @brief Create a void sink (one that does no actual io)
   *
   * A useful code path for benchmarking, to eliminate physical
   * hardware randomness from profiling.
   *
   * @return Constructed data_sink object
   */
  static std::unique_ptr<data_sink> create();

  /**
   * @brief Create a wrapped custom user data sink
   *
   * @param[in] user_sink User-provided data sink (typically custom class)
   *
   * The data sink returned here is not the one passed by the user. It is an internal
   * class that wraps the user pointer.  The principle is to allow the user to declare
   * a custom sink instance and use it across multiple write() calls.
   *
   * @return Constructed data_sink object
   */
  static std::unique_ptr<data_sink> create(cudf::io::data_sink* const user_sink);

  /**
   * @brief Creates a vector of data sinks, one per element in the input vector.
   *
   * @param[in] args vector of parameters
   * @return Constructed vector of data sinks
   */
  template <typename T>
  static std::vector<std::unique_ptr<data_sink>> create(std::vector<T> const& args)
  {
    std::vector<std::unique_ptr<data_sink>> sinks;
    sinks.reserve(args.size());
    std::transform(args.cbegin(), args.cend(), std::back_inserter(sinks), [](auto const& arg) {
      return data_sink::create(arg);
    });
    return sinks;
  }

  /**
   * @brief Base class destructor
   */
  virtual ~data_sink() {};

  /**
   * @pure @brief Append the buffer content to the sink
   *
   * @param[in] data Pointer to the buffer to be written into the sink object
   * @param[in] size Number of bytes to write
   */
  virtual void host_write(void const* data, size_t size) = 0;

  /**
   * @brief Whether or not this sink supports writing from gpu memory addresses.
   *
   * Internal to some of the file format writers, we have code that does things like
   *
   * tmp_buffer = alloc_temp_buffer();
   * cudaMemcpy(tmp_buffer, device_buffer, size);
   * sink->write(tmp_buffer, size);
   *
   * In the case where the sink type is itself a memory buffered write, this ends up
   * being effectively a second memcpy.  So a useful optimization for a "smart"
   * custom data_sink is to do its own internal management of the movement
   * of data between cpu and gpu; turning the internals of the writer into simply
   *
   * sink->device_write(device_buffer, size)
   *
   * If this function returns true, the data_sink will receive calls to device_write()
   * instead of write() when possible.  However, it is still possible to receive
   * write() calls as well.
   *
   * @return If this writer supports device_write() calls
   */
  [[nodiscard]] virtual bool supports_device_write() const { return false; }

  /**
   * @brief Estimates whether a direct device write would be more optimal for the given size.
   *
   * @param size Number of bytes to write
   * @return whether the device write is expected to be more performant for the given size
   */
  [[nodiscard]] virtual bool is_device_write_preferred(size_t size) const
  {
    return supports_device_write();
  }

  /**
   * @brief Append the buffer content to the sink from a gpu address
   *
   * For optimal performance, should only be called when `is_device_write_preferred` returns `true`.
   * Data sink implementations that don't support direct device writes don't need to override
   * this function.
   *
   * @throws cudf::logic_error the object does not support direct device writes, i.e.
   * `supports_device_write` returns `false`.
   *
   * @param gpu_data Pointer to the buffer to be written into the sink object
   * @param size Number of bytes to write
   * @param stream CUDA stream to use
   */
  virtual void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("data_sink classes that support device_write must override it.");
  }

  /**
   * @brief Asynchronously append the buffer content to the sink from a gpu address
   *
   * For optimal performance, should only be called when `is_device_write_preferred` returns `true`.
   * Data sink implementations that don't support direct device writes don't need to override
   * this function.
   *
   * `gpu_data` must not be freed until this call is synchronized.
   * @code{.pseudo}
   * auto result = device_write_async(gpu_data, size, stream);
   * result.wait(); // OR result.get()
   * @endcode
   *
   * @throws cudf::logic_error the object does not support direct device writes, i.e.
   * `supports_device_write` returns `false`.
   * @throws cudf::logic_error
   *
   * @param gpu_data Pointer to the buffer to be written into the sink object
   * @param size Number of bytes to write
   * @param stream CUDA stream to use
   * @return a future that can be used to synchronize the call
   */
  virtual std::future<void> device_write_async(void const* gpu_data,
                                               size_t size,
                                               rmm::cuda_stream_view stream)
  {
    CUDF_FAIL("data_sink classes that support device_write_async must override it.");
  }

  /**
   * @pure @brief Flush the data written into the sink
   */
  virtual void flush() = 0;

  /**
   * @pure @brief Returns the total number of bytes written into this sink
   *
   * @return Total number of bytes written into this sink
   */
  virtual size_t bytes_written() = 0;
};

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
