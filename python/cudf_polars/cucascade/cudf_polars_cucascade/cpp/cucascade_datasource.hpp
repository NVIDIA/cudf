/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>

#include <cuda/stream>

#include <cucascade/io/advising_reader.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>

namespace cudf_polars_cucascade {

/**
 * @brief A @c cudf::io::datasource that reads one object-store object through cuCascade's
 * prefetching cache.
 *
 * All the reading, and the advising of what a read will need to the cache, is done by
 * @c cucascade::io::advising_reader, which has no libcudf dependency. This class only gives it
 * libcudf's interface and builds libcudf's buffers. It is the one part that subclasses a libcudf
 * type, so it is built here, together with the libcudf it runs with: a subclass compiled against
 * one libcudf does not work with another.
 */
class cucascade_datasource : public cudf::io::datasource {
 public:
  explicit cucascade_datasource(std::unique_ptr<cucascade::io::advising_reader> reader);
  ~cucascade_datasource() override;

  [[nodiscard]] size_t size() const override;
  [[nodiscard]] bool supports_device_read() const override;
  [[nodiscard]] bool is_device_read_preferred(size_t size) const override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;
  std::future<std::unique_ptr<buffer>> host_read_async(size_t offset, size_t size) override;

  std::unique_ptr<buffer> device_read(size_t offset, size_t size, cuda::stream_ref stream) override;
  size_t device_read(size_t offset, size_t size, uint8_t* dst, cuda::stream_ref stream) override;
  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cuda::stream_ref stream) override;

 private:
  std::unique_ptr<cucascade::io::advising_reader> _reader;
};

}  // namespace cudf_polars_cucascade
