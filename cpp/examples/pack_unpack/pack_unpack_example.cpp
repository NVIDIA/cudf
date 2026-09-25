/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Pack and unpack a table entirely on the device.
void device_pack_unpack(cudf::table_view input)
{
  cudf::packed_columns packed = cudf::pack(input);
  print_table("Device Unpacked Table", cudf::unpack(packed));
}

// Pack a table into pinned host memory, then unpack it.
void host_pack_unpack(cudf::table_view input)
{
  rmm::mr::pinned_host_memory_resource phmr;
  cudf::packed_columns packed = cudf::pack(input, cudf::get_default_stream(), phmr);
  print_table("Host Unpacked Table", cudf::unpack(packed));
}

// Pack into pinned host memory, copy the packed bytes to another host buffer
// (simulating a host-to-host transfer), then unpack the copy.
void host_pack_copy_unpack(cudf::table_view input)
{
  auto stream = cudf::get_default_stream();
  rmm::mr::pinned_host_memory_resource phmr;

  cudf::packed_columns packed = cudf::pack(input, stream, phmr);
  // pack's device->pinned copy is stream-ordered, so the CPU must wait for the stream
  // before reading the packed buffer via std::memcpy below.
  stream.sync();

  auto copied_metadata = std::make_unique<std::vector<uint8_t>>(*packed.metadata);
  std::vector<uint8_t> copied_data(packed.gpu_data->size());
  std::memcpy(copied_data.data(), packed.gpu_data->data(), packed.gpu_data->size());

  auto copied_buffer =
    std::make_unique<rmm::device_buffer>(copied_data.data(), copied_data.size(), stream, phmr);
  cudf::packed_columns copied_packed(std::move(copied_metadata), std::move(copied_buffer));

  print_table("Host Copied Unpacked Table", cudf::unpack(copied_packed));
}

int main(int argc, char** argv)
{
  std::string const mode = argc > 1 ? argv[1] : "device";

  // A device memory pool speeds up the small allocations used to build the table.
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource mr{cuda_mr, rmm::percent_of_free_device_memory(50)};
  cudf::set_current_device_resource(mr);

  auto input_table = make_table(5, 2);
  print_table("Original Table", input_table);

  if (mode == "device") {
    device_pack_unpack(input_table);
  } else if (mode == "host") {
    host_pack_unpack(input_table);
  } else if (mode == "host-copy") {
    host_pack_copy_unpack(input_table);
  } else {
    std::cerr << "Unknown mode '" << mode << "'. Use one of: device, host, host-copy.\n";
    return 1;
  }
  return 0;
}
