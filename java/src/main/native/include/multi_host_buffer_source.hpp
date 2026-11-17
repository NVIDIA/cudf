/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "jni_utils.hpp"

#include <cudf/io/datasource.hpp>

#include <vector>

namespace cudf {
namespace jni {

/**
 * @brief A custom datasource providing data from an array of host memory buffers.
 */
class multi_host_buffer_source : public cudf::io::datasource {
  std::vector<uint8_t const*> addrs_;
  std::vector<size_t> offsets_;

  size_t locate_offset_index(size_t offset);

 public:
  explicit multi_host_buffer_source(native_jlongArray const& addrs_sizes);
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  bool supports_device_read() const override { return true; }
  bool is_device_read_preferred(size_t size) const override { return true; }
  std::unique_ptr<buffer> device_read(size_t offset, size_t size, cuda::stream_ref stream) override;
  size_t device_read(size_t offset, size_t size, uint8_t* dst, cuda::stream_ref stream) override;
  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        cuda::stream_ref stream) override;
  size_t size() const override { return offsets_.back(); }
};

}  // namespace jni
}  // namespace cudf
