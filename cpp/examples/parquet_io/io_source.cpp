/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_source.hpp"

#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/stream>
#include <thrust/host_vector.h>

#include <filesystem>
#include <fstream>
#include <string>

rmm::host_async_resource_ref pinned_memory_resource()
{
  static auto mr = rmm::mr::pinned_host_memory_resource{};
  return mr;
}

io_source_type get_io_source_type(std::string name)
{
  static std::unordered_map<std::string_view, io_source_type> const map = {
    {"FILEPATH", io_source_type::FILEPATH},
    {"HOST_BUFFER", io_source_type::HOST_BUFFER},
    {"PINNED_BUFFER", io_source_type::PINNED_BUFFER},
    {"DEVICE_BUFFER", io_source_type::DEVICE_BUFFER}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) {
    return map.at(name);
  } else {
    throw std::invalid_argument(name +
                                " is not a valid io source type. Available: FILEPATH,\n"
                                "HOST_BUFFER, PINNED_BUFFER, DEVICE_BUFFER.\n\n");
  }
}

io_source::io_source(std::string_view file_path, io_source_type type, cuda::stream_ref stream)
  : pinned_buffer({pinned_memory_resource(), stream}), d_buffer{0, stream}
{
  std::string const file_name{file_path};
  auto const file_size = std::filesystem::file_size(file_name);

  // For filepath make a quick source_info and return early
  if (type == io_source_type::FILEPATH) {
    source_info = cudf::io::source_info(file_name);
    return;
  }

  std::ifstream file{file_name, std::ifstream::binary};

  // Copy file contents to the specified io source buffer
  switch (type) {
    case io_source_type::HOST_BUFFER: {
      h_buffer.resize(file_size);
      file.read(h_buffer.data(), file_size);
      source_info = cudf::io::source_info(cudf::host_span<std::byte const>(
        reinterpret_cast<std::byte const*>(h_buffer.data()), file_size));
      break;
    }
    case io_source_type::PINNED_BUFFER: {
      pinned_buffer.resize(file_size);
      file.read(pinned_buffer.data(), file_size);
      source_info = cudf::io::source_info(cudf::host_span<std::byte const>(
        reinterpret_cast<std::byte const*>(pinned_buffer.data()), file_size));
      break;
    }
    case io_source_type::DEVICE_BUFFER: {
      h_buffer.resize(file_size);
      file.read(h_buffer.data(), file_size);
      d_buffer.resize(file_size, stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_buffer.data(), h_buffer.data(), file_size, cudaMemcpyDefault, stream.get()));

      source_info = cudf::io::source_info(d_buffer);
      break;
    }
    default: {
      throw std::runtime_error("Encountered unexpected source type\n\n");
    }
  }
}
