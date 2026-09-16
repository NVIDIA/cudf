/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/json/nested_json.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <tests/io/fst/common.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <string>
#include <vector>

namespace {

// pre-generate all the number strings
std::vector<std::string> _num_to_string;
std::string num_to_string(int32_t num) { return _num_to_string.at(num); }

// List of List nested.
std::string generate_list_of_lists(int32_t max_depth, int32_t max_rows, std::string elem)
{
  std::string json = "[";
  if (max_depth > 1) json += std::string(max_depth - 1, '[');
  for (int32_t row = 0; row < max_rows; ++row) {
    json += elem;
    if (row < max_rows - 1) { json += ", "; }
  }
  if (max_depth > 1) json += std::string(max_depth - 1, ']');
  json += "]";
  return json;
}

// Struct of Struct nested.
std::string generate_struct_of_structs(int32_t max_depth, int32_t max_rows, std::string elem)
{
  if (max_depth <= 0) return "{}";
  std::string json;
  for (int32_t depth = 0; depth < max_depth / 2; ++depth) {
    json += R"({"a)" + num_to_string(depth) + R"(": )";
  }
  if (max_rows == 0) json += "{}";

  for (int32_t row = 0; row < max_rows; ++row) {
    json += elem;
    if (row < max_rows - 1) {
      json += R"(, "a)" + num_to_string(max_depth / 2 - 1) + "_" + num_to_string(row) + R"(": )";
    }
  }
  if (max_depth > 0) json += std::string(max_depth / 2, '}');
  return json;
}

// Memoize the generated rows so we don't have to regenerate them.
std::map<std::tuple<int, int, int, int>, std::string> _row_cache;

std::string generate_row(
  int num_columns, int max_depth, int max_list_size, int max_struct_size, size_t max_bytes)
{
  std::string s = "{";
  std::vector<std::string> const elems{
    R"(1)", R"(-2)", R"(3.4)", R"("5")", R"("abcdefghij")", R"(true)", R"(null)"};
  for (int i = 0; i < num_columns; i++) {
    s += R"("col)" + num_to_string(i) + R"(": )";
    if (auto it = _row_cache.find({i % 2, max_depth - 2, max_struct_size, i % elems.size()});
        it != _row_cache.end()) {
      s += it->second;
    } else {
      auto r =
        (i % 2 == 0)
          ? generate_struct_of_structs(max_depth - 2, max_struct_size, elems[i % elems.size()])
          : generate_list_of_lists(max_depth - 2, max_struct_size, elems[i % elems.size()]);
      _row_cache[{i % 2, max_depth - 2, max_struct_size, i % elems.size()}] = r;
      s += r;
    }
    if (s.length() > max_bytes) break;
    if (i < num_columns - 1) s += ", ";
  }
  s += "}";
  return s;
}

std::string generate_json(int num_rows,
                          int num_columns,
                          int max_depth,
                          int max_list_size,
                          int max_struct_size,
                          size_t max_json_bytes)
{
  // std::to_string is slow, so we pre-generate all number strings we need.
  _num_to_string.clear();
  auto max_num_str =
    std::max(std::max(num_columns, max_depth), std::max(max_list_size, max_struct_size));
  for (int i = 0; i < max_num_str; i++)
    _num_to_string.emplace_back(std::to_string(i));
  _row_cache.clear();

  std::string s = "[\n";
  s.reserve(max_json_bytes + 1024);
  for (int i = 0; i < num_rows; i++) {
    s += generate_row(
      num_columns, max_depth - 2, max_list_size, max_struct_size, max_json_bytes - s.length());
    if (s.length() > max_json_bytes) break;
    if (i != num_rows - 1) s += ",\n";
  }
  s += "\n]";
  return s;
}

auto make_test_json_data(cudf::size_type string_size, cuda::stream_ref stream)
{
  // Test input
  std::string input = R"(
                      {"a":1,"b":2,"c":[3], "d": {}},
                      {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
                      {"a":1,"b":6.0,"c":[5, 7], "d": null},
                      {"a":1,"b":null,"c":null},
                      {
                        "a" : 1
                      },
                      {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}},
                      {"a": 1, "b": 8.0, "d": { "author": "Jean-Jacques Rousseau"}},)";

  cudf::size_type const repeat_times = string_size / input.size();

  auto d_input_scalar   = cudf::make_string_scalar(input, stream);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_input_scalar);
  auto d_scalar         = cudf::strings::repeat_string(d_string_scalar, repeat_times);

  auto data = const_cast<char*>(d_scalar->data());
  CUDF_CUDA_TRY(cudaMemsetAsync(data, '[', 1, stream.get()));
  CUDF_CUDA_TRY(cudaMemsetAsync(data + d_scalar->size() - 1, ']', 1, stream.get()));

  return d_scalar;
}
}  // namespace

void BM_NESTED_JSON(nvbench::state& state)
{
  auto const string_size{cudf::size_type(state.get_int64("string_size"))};
  auto const default_options = cudf::io::json_reader_options{};

  auto input = make_test_json_data(string_size, cudf::get_default_stream());
  state.add_element_count(input->size());

  // Run algorithm
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    cudf::io::json::detail::device_parse_nested_json(
      cudf::device_span<char const>{input->data(), static_cast<size_t>(input->size())},
      default_options,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(string_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_NESTED_JSON)
  .set_name("nested_json_gpu_parser")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

void BM_NESTED_JSON_DEPTH(nvbench::state& state)
{
  auto const string_size{cudf::size_type(state.get_int64("string_size"))};
  auto const depth{cudf::size_type(state.get_int64("depth"))};

  auto d_scalar = cudf::string_scalar(
    generate_json(100'000'000, 10, depth, 10, 10, string_size), true, cudf::get_default_stream());
  auto input = cudf::device_span<char const>(d_scalar.data(), d_scalar.size());

  state.add_element_count(input.size());
  auto const default_options = cudf::io::json_reader_options{};

  // Run algorithm
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    cudf::io::json::detail::device_parse_nested_json(
      input, default_options, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(string_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_NESTED_JSON_DEPTH)
  .set_name("nested_json_gpu_parser_depth")
  .add_int64_power_of_two_axis("depth", nvbench::range(1, 4, 1))
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 2));
