/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "multi_host_buffer_source.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

namespace cudf {
namespace jni {

multi_host_buffer_source::multi_host_buffer_source(native_jlongArray const& addrs_sizes)
{
  if (addrs_sizes.size() % 2 != 0) {
    throw std::logic_error("addrs_sizes length not a multiple of 2");
  }
  auto count = addrs_sizes.size() / 2;
  addrs_.reserve(count);
  offsets_.reserve(count + 1);
  size_t total_size = 0;
  for (int i = 0; i < addrs_sizes.size(); i += 2) {
    addrs_.push_back(reinterpret_cast<uint8_t const*>(addrs_sizes[i]));
    offsets_.push_back(total_size);
    total_size += addrs_sizes[i + 1];
  }
  offsets_.push_back(total_size);
}

size_t multi_host_buffer_source::locate_offset_index(size_t offset)
{
  if (offset < 0 || offset >= offsets_.back()) { throw std::runtime_error("bad offset"); }
  auto start = offsets_.begin();
  auto it    = std::upper_bound(start, offsets_.end(), offset);
  return (it - start) - 1;
}

std::unique_ptr<cudf::io::datasource::buffer> multi_host_buffer_source::host_read(size_t offset,
                                                                                  size_t size)
{
  if (size == 0) { return 0; }
  if (offset < 0 || offset >= offsets_.back()) { throw std::runtime_error("bad offset"); }
  auto const end_offset = offset + size;
  if (end_offset > offsets_.back()) { throw std::runtime_error("read past end of file"); }
  auto buffer_index = locate_offset_index(offset);
  auto next_offset  = offsets_[buffer_index + 1];
  if (end_offset <= next_offset) {
    // read range hits only a single buffer, so return a zero-copy view of the data
    auto src = addrs_[buffer_index] + offset - offsets_[buffer_index];
    return std::make_unique<non_owning_buffer>(src, size);
  }
  auto buf        = std::vector<uint8_t>(size);
  auto bytes_read = host_read(offset, size, buf.data());
  if (bytes_read != size) {
    std::stringstream ss;
    ss << "Expected host read of " << size << " found " << bytes_read;
    throw std::logic_error(ss.str());
  }
  return std::make_unique<owning_buffer<std::vector<uint8_t>>>(std::move(buf));
}

size_t multi_host_buffer_source::host_read(size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) { return 0; }
  if (offset < 0 || offset >= offsets_.back()) { throw std::runtime_error("bad offset"); }
  if (offset + size > offsets_.back()) { throw std::runtime_error("read past end of file"); }
  auto buffer_index = locate_offset_index(offset);
  auto bytes_left   = size;
  while (bytes_left > 0) {
    auto next_offset   = offsets_[buffer_index + 1];
    auto buffer_left   = next_offset - offset;
    auto buffer_offset = offset - offsets_[buffer_index];
    auto src           = addrs_[buffer_index] + buffer_offset;
    auto copy_size     = std::min(buffer_left, bytes_left);
    std::memcpy(dst, src, copy_size);
    offset += copy_size;
    dst += copy_size;
    bytes_left -= copy_size;
    ++buffer_index;
  }
  return size;
}

std::unique_ptr<cudf::io::datasource::buffer> multi_host_buffer_source::device_read(
  size_t offset, size_t size, cuda::stream_ref stream)
{
  rmm::device_buffer buf(size, stream);
  auto dst        = static_cast<uint8_t*>(buf.data());
  auto bytes_read = device_read(offset, size, dst, stream);
  if (bytes_read != size) {
    std::stringstream ss;
    ss << "Expected device read of " << size << " found " << bytes_read;
    throw std::logic_error(ss.str());
  }
  return std::make_unique<owning_buffer<rmm::device_buffer>>(std::move(buf));
}

size_t multi_host_buffer_source::device_read(size_t offset,
                                             size_t size,
                                             uint8_t* dst,
                                             cuda::stream_ref stream)
{
  if (size == 0) { return 0; }
  if (offset < 0 || offset >= offsets_.back()) { throw std::runtime_error("bad offset"); }
  if (offset + size > offsets_.back()) { throw std::runtime_error("read past end of file"); }
  auto buffer_index = locate_offset_index(offset);
  auto bytes_left   = size;
  while (bytes_left > 0) {
    auto next_offset   = offsets_[buffer_index + 1];
    auto buffer_left   = next_offset - offset;
    auto buffer_offset = offset - offsets_[buffer_index];
    auto src           = addrs_[buffer_index] + buffer_offset;
    auto copy_size     = std::min(buffer_left, bytes_left);
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, copy_size, cudaMemcpyHostToDevice, stream.get()));
    offset += copy_size;
    dst += copy_size;
    bytes_left -= copy_size;
    ++buffer_index;
  }
  return size;
}

std::future<size_t> multi_host_buffer_source::device_read_async(size_t offset,
                                                                size_t size,
                                                                uint8_t* dst,
                                                                cuda::stream_ref stream)
{
  std::promise<size_t> p;
  p.set_value(device_read(offset, size, dst, stream));
  return p.get_future();
}

}  // namespace jni
}  // namespace cudf
