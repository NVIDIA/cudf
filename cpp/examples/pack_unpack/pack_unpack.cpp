/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Copies `column_data` into a device_uvector and wraps it as an int32 column with no nulls.
 */
std::unique_ptr<cudf::column> make_column_from_span(std::span<int32_t const> column_data)
{
  auto stream = cudf::get_default_stream();
  rmm::device_uvector<int32_t> device_data(column_data.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_data.data(),
                                column_data.data(),
                                column_data.size() * sizeof(int32_t),
                                cudaMemcpyDefault,
                                stream.get()));
  // Wait for the H2D copy so the caller can reuse or destroy the source span.
  stream.sync();
  return std::make_unique<cudf::column>(
    std::move(device_data), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream), 0);
}

/**
 * @brief Builds a `row_count` x `column_count` int32 table where each column holds a
 * contiguous ascending sequence starting from a distinct offset.
 */
cudf::table make_table(std::size_t row_count, std::size_t column_count)
{
  CUDF_EXPECTS(column_count > 0, "column_count must be greater than zero");
  CUDF_EXPECTS(row_count <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "row_count exceeds cudf::size_type range");

  std::vector<int32_t> column_data(row_count);
  std::vector<std::unique_ptr<cudf::column>> columns;

  int32_t current_value{0};
  for (std::size_t i = 0; i < column_count; ++i) {
    std::iota(column_data.begin(), column_data.end(), current_value);
    columns.emplace_back(make_column_from_span(column_data));
    current_value += static_cast<int32_t>(row_count);
  }
  return cudf::table{std::move(columns)};
}

/**
 * @brief Writes the given table view to stdout in CSV form, preceded by `header`.
 */
void print_table(std::string const& header, cudf::table_view const& tbl_view)
{
  std::vector<char> buffer;
  auto builder = cudf::io::csv_writer_options::builder(cudf::io::sink_info(&buffer), tbl_view);
  cudf::io::write_csv(builder.build());
  std::cout << header << ":\n" << std::string_view{buffer.data(), buffer.size()} << "\n";
}

/**
 * @brief Asserts that `actual` equals `expected` and prints the outcome. Throws on mismatch.
 */
void check_tables_equal(cudf::table_view const& expected,
                        cudf::table_view const& actual,
                        cuda::stream_ref stream = cudf::get_default_stream())
{
  auto const equal = cudf::tables_equal(expected, actual, cudf::null_equality::EQUAL, stream);
  std::cout << "Tables identical: " << std::boolalpha << equal << "\n\n";
  if (not equal) { throw std::logic_error("Table equality check failed"); }
}

/**
 * @brief Packs and unpacks a table entirely in device memory.
 */
void device_pack_unpack(cudf::table_view input)
{
  cudf::packed_columns packed = cudf::pack(input);
  auto const unpacked         = cudf::unpack(packed);
  print_table("Device Unpacked Table", unpacked);
  check_tables_equal(input, unpacked);
}

/**
 * @brief Packs a table into pinned host memory, then unpacks it.
 */
void host_pack_unpack(cudf::table_view input)
{
  rmm::mr::pinned_host_memory_resource phmr;
  cudf::packed_columns packed = cudf::pack(input, cudf::get_default_stream(), phmr);
  auto const unpacked         = cudf::unpack(packed);
  print_table("Host Unpacked Table", unpacked);
  check_tables_equal(input, unpacked);
}

/**
 * @brief Packs into pinned host memory, copies the packed bytes to another host buffer
 * (simulating a host-to-host transfer), then unpacks the copy.
 */
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
  // Ensure the async H2H copy out of `copied_data` completes before it goes out of scope.
  stream.sync();
  cudf::packed_columns copied_packed(std::move(copied_metadata), std::move(copied_buffer));

  auto const unpacked = cudf::unpack(copied_packed);
  print_table("Host Copied Unpacked Table", unpacked);
  check_tables_equal(input, unpacked, stream);
}

}  // namespace

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
    throw std::runtime_error(
      std::format("Unknown mode `{}`. Use one of: `device`, `host`, `host-copy`.", mode));
  }
  return 0;
}
