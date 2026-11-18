/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucascade_datasource.hpp"

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace cudf_polars_cucascade {

cucascade_datasource::cucascade_datasource(std::unique_ptr<cucascade::io::advising_reader> reader)
  : _reader(std::move(reader))
{
}

cucascade_datasource::~cucascade_datasource() = default;

size_t cucascade_datasource::size() const { return _reader->size(); }

bool cucascade_datasource::supports_device_read() const { return _reader->supports_device_read(); }

bool cucascade_datasource::is_device_read_preferred(size_t) const
{
  return _reader->supports_device_read();
}

size_t cucascade_datasource::host_read(size_t offset, size_t size, uint8_t* dst)
{
  return _reader->host_read(offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> cucascade_datasource::host_read(size_t offset,
                                                                              size_t size)
{
  std::vector<uint8_t> buf(size);
  auto const n = host_read(offset, size, buf.data());
  buf.resize(n);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

std::future<size_t> cucascade_datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  return _reader->host_read_async(offset, size, dst);
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> cucascade_datasource::host_read_async(
  size_t offset, size_t size)
{
  auto const file_size = _reader->size();
  size                 = std::min(size, file_size > offset ? file_size - offset : size_t{0});
  auto buf             = std::vector<uint8_t>(size);
  auto fut             = host_read_async(offset, size, buf.data());
  return std::async(std::launch::deferred, [s = std::move(fut), buf = std::move(buf)]() mutable {
    auto const n = s.get();
    buf.resize(n);
    return cudf::io::datasource::buffer::create(std::move(buf));
  });
}

std::unique_ptr<cudf::io::datasource::buffer> cucascade_datasource::device_read(
  size_t offset, size_t size, cuda::stream_ref stream)
{
  rmm::device_buffer buf(size, stream);
  auto n = device_read(offset, size, static_cast<uint8_t*>(buf.data()), stream);
  n      = std::min(n, size);
  buf.resize(n, stream);
  return cudf::io::datasource::buffer::create(std::move(buf));
}

size_t cucascade_datasource::device_read(size_t offset,
                                         size_t size,
                                         uint8_t* dst,
                                         cuda::stream_ref stream)
{
  auto const n = _reader->device_read_async(offset, size, dst, stream).get();
  stream.sync();
  return n;
}

std::future<size_t> cucascade_datasource::device_read_async(size_t offset,
                                                            size_t size,
                                                            uint8_t* dst,
                                                            cuda::stream_ref stream)
{
  return _reader->device_read_async(offset, size, dst, stream);
}

}  // namespace cudf_polars_cucascade
