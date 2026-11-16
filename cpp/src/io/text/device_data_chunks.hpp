/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>

namespace cudf::io::text {

class device_span_data_chunk : public device_data_chunk {
 public:
  device_span_data_chunk(device_span<char const> data) : _data(data) {}

  [[nodiscard]] char const* data() const override { return _data.data(); }
  [[nodiscard]] std::size_t size() const override { return _data.size(); }
  operator device_span<char const>() const override { return _data; }

 private:
  device_span<char const> _data;
};

class device_buffer_data_chunk : public device_data_chunk {
 public:
  device_buffer_data_chunk(cuda::device_buffer<char>&& data) : _data(std::move(data)) {}

  device_buffer_data_chunk(device_buffer_data_chunk const&)            = delete;
  device_buffer_data_chunk& operator=(device_buffer_data_chunk const&) = delete;
  device_buffer_data_chunk(device_buffer_data_chunk&&)                 = default;
  device_buffer_data_chunk& operator=(device_buffer_data_chunk&&)      = default;

  [[nodiscard]] char const* data() const override { return _data.data(); }
  [[nodiscard]] std::size_t size() const override { return _data.size(); }
  operator device_span<char const>() const override { return _data; }

 private:
  cuda::device_buffer<char> _data;
};

}  // namespace cudf::io::text
