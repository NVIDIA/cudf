/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/file_handle.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

namespace cudf {
namespace io {

/**
 * @brief Implementation class for storing data into a local file.
 */
class file_sink : public data_sink {
 public:
  explicit file_sink(std::string const& filepath)
  {
    kvikio_integration::set_up_kvikio();
    _kvikio_file = kvikio::FileHandle(filepath, "w");
    CUDF_EXPECTS(!_kvikio_file.closed(), "KvikIO did not open the file successfully.");
    CUDF_LOG_INFO("Writing a file using kvikIO, with compatibility mode %s.",
                  _kvikio_file.get_compat_mode_manager().is_compat_mode_preferred() ? "on" : "off");
  }

  // Marked as NOLINT because we are calling a virtual method in the destructor
  ~file_sink() override { flush(); }  // NOLINT

  void host_write(void const* data, size_t size) override
  {
    _kvikio_file.pwrite(data, size, _bytes_written).get();
    _bytes_written += size;
  }

  void flush() override
  {
    // kvikio::FileHandle::pwrite() makes system calls that reach the kernel buffer cache. This
    // process does not involve application buffer. Therefore calls to ::fflush() or
    // ofstream::flush() do not apply.
  }

  size_t bytes_written() override { return _bytes_written; }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  [[nodiscard]] bool is_device_write_preferred(size_t size) const override
  {
    return supports_device_write();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    if (!supports_device_write()) CUDF_FAIL("Device writes are not supported for this file.");

    size_t const offset = _bytes_written;
    _bytes_written += size;
    stream.synchronize();

    // Start the write now via the capture-initializer; only the `.get()` wait is deferred.
    return std::async(std::launch::deferred,
                      [fut = _kvikio_file.pwrite(gpu_data, size, offset)]() mutable { fut.get(); });
  }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    return device_write_async(gpu_data, size, stream).get();
  }

 private:
  size_t _bytes_written = 0;
  kvikio::FileHandle _kvikio_file;
};

/**
 * @brief Implementation class for storing data into a std::vector.
 */
class host_buffer_sink : public data_sink {
 public:
  explicit host_buffer_sink(std::vector<char>* buffer) : buffer_(buffer) {}

  // Marked as NOLINT because we are calling a virtual method in the destructor
  ~host_buffer_sink() override { flush(); }  // NOLINT

  void host_write(void const* data, size_t size) override
  {
    auto char_array = static_cast<char const*>(data);
    buffer_->insert(buffer_->end(), char_array, char_array + size);
  }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  [[nodiscard]] bool is_device_write_preferred(size_t size) const override { return true; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    device_write_async(gpu_data, size, stream).get();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    auto const current_size = buffer_->size();
    buffer_->resize(current_size + size);
    // TODO: https://github.com/rapidsai/cudf/issues/21680
    // We need to replace this with memcpy_batch_async after fixing stream
    // ordering. The issue is that buffer_->resize() can reallocate,
    // invalidating pointers from previous async copies that are still
    // in-flight when using cudaMemcpySrcAccessOrderStream. Need to ensure
    // stream ordering or pre-reserve buffer to avoid reallocation.
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      buffer_->data() + current_size, gpu_data, size, cudaMemcpyDeviceToHost, stream.value()));
    return std::async(std::launch::deferred, [stream]() -> void { stream.synchronize(); });
  }

  void flush() override {}

  size_t bytes_written() override { return buffer_->size(); }

 private:
  std::vector<char>* buffer_;
};

/**
 * @brief Implementation class for voiding data (no io performed)
 */
class void_sink : public data_sink {
 public:
  explicit void_sink() {}

  ~void_sink() override {}

  void host_write(void const* data, size_t size) override { _bytes_written += size; }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  [[nodiscard]] bool is_device_write_preferred(size_t size) const override { return true; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    _bytes_written += size;
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    _bytes_written += size;
    return std::async(std::launch::deferred, []() -> void {});
  }

  void flush() override {}

  size_t bytes_written() override { return _bytes_written; }

 private:
  size_t _bytes_written{};
};

class user_sink_wrapper : public data_sink {
 public:
  explicit user_sink_wrapper(cudf::io::data_sink* const user_sink_) : user_sink(user_sink_) {}

  ~user_sink_wrapper() override {}

  void host_write(void const* data, size_t size) override { user_sink->host_write(data, size); }

  [[nodiscard]] bool supports_device_write() const override
  {
    return user_sink->supports_device_write();
  }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(user_sink->supports_device_write(),
                 "device_write() was called on a data_sink that doesn't support it");
    user_sink->device_write(gpu_data, size, stream);
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(user_sink->supports_device_write(),
                 "device_write_async() was called on a data_sink that doesn't support it");
    return user_sink->device_write_async(gpu_data, size, stream);
  }

  [[nodiscard]] bool is_device_write_preferred(size_t size) const override
  {
    return user_sink->is_device_write_preferred(size);
  }

  void flush() override { user_sink->flush(); }

  size_t bytes_written() override { return user_sink->bytes_written(); }

 private:
  cudf::io::data_sink* const user_sink;
};

std::unique_ptr<data_sink> data_sink::create(std::string const& filepath)
{
  return std::make_unique<file_sink>(filepath);
}

std::unique_ptr<data_sink> data_sink::create(std::vector<char>* buffer)
{
  return std::make_unique<host_buffer_sink>(buffer);
}

std::unique_ptr<data_sink> data_sink::create() { return std::make_unique<void_sink>(); }

std::unique_ptr<data_sink> data_sink::create(cudf::io::data_sink* const user_sink)
{
  return std::make_unique<user_sink_wrapper>(user_sink);
}

}  // namespace io
}  // namespace cudf
