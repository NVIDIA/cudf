/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/table_utilities.hpp>

#include <cudf_test/nanoarrow_utils.hpp>

#include <cudf/interop.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <iterator>
#include <vector>

template <cudf::type_id data_type>
void BM_to_arrow_device(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  auto const table         = create_random_table(types, row_count{num_rows});
  int64_t const size_bytes = estimate_size(table->view());

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::to_arrow_device(table->view(), rmm::cuda_stream_view{launch.get_stream()});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <cudf::type_id data_type>
void BM_to_arrow_host(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  auto const table         = create_random_table(types, row_count{num_rows});
  int64_t const size_bytes = estimate_size(table->view());

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::to_arrow_host(table->view(), rmm::cuda_stream_view{launch.get_stream()});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <cudf::type_id data_type>
void BM_from_arrow_device(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  data_profile profile;
  profile.set_struct_depth(1);
  profile.set_list_depth(1);

  auto const table            = create_random_table(types, row_count{num_rows}, profile);
  cudf::table_view table_view = table->view();
  int64_t const size_bytes    = estimate_size(table_view);

  std::vector<cudf::column_metadata> table_metadata;

  std::transform(cuda::counting_iterator<cudf::size_type>{0},
                 cuda::counting_iterator{num_columns},
                 std::back_inserter(table_metadata),
                 [&](auto const column) {
                   cudf::column_metadata column_metadata{""};
                   column_metadata.children_meta = std::vector(
                     table->get_column(column).num_children(), cudf::column_metadata{""});
                   return column_metadata;
                 });

  cudf::unique_schema_t schema      = cudf::to_arrow_schema(table_view, table_metadata);
  cudf::unique_device_array_t input = cudf::to_arrow_device(table_view);

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::from_arrow_device_column(
      schema.get(), input.get(), rmm::cuda_stream_view{launch.get_stream()});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <cudf::type_id data_type>
void BM_from_arrow_host(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  data_profile profile;
  profile.set_struct_depth(1);
  profile.set_list_depth(1);

  auto const table            = create_random_table(types, row_count{num_rows}, profile);
  cudf::table_view table_view = table->view();
  int64_t const size_bytes    = estimate_size(table_view);

  std::vector<cudf::column_metadata> table_metadata;

  std::transform(cuda::counting_iterator<cudf::size_type>{0},
                 cuda::counting_iterator{num_columns},
                 std::back_inserter(table_metadata),
                 [&](auto const column) {
                   cudf::column_metadata column_metadata{""};
                   column_metadata.children_meta = std::vector(
                     table->get_column(column).num_children(), cudf::column_metadata{""});
                   return column_metadata;
                 });

  cudf::unique_schema_t schema      = cudf::to_arrow_schema(table_view, table_metadata);
  cudf::unique_device_array_t input = cudf::to_arrow_host(table_view);

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::from_arrow_host_column(
      schema.get(), input.get(), rmm::cuda_stream_view{launch.get_stream()});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using data_types = nvbench::enum_type_list<cudf::type_id::INT8,
                                           cudf::type_id::INT16,
                                           cudf::type_id::INT32,
                                           cudf::type_id::INT64,
                                           cudf::type_id::UINT8,
                                           cudf::type_id::UINT16,
                                           cudf::type_id::UINT32,
                                           cudf::type_id::UINT64,
                                           cudf::type_id::FLOAT32,
                                           cudf::type_id::FLOAT64,
                                           cudf::type_id::BOOL8,
                                           cudf::type_id::TIMESTAMP_SECONDS,
                                           cudf::type_id::TIMESTAMP_MILLISECONDS,
                                           cudf::type_id::TIMESTAMP_MICROSECONDS,
                                           cudf::type_id::TIMESTAMP_NANOSECONDS,
                                           cudf::type_id::DURATION_SECONDS,
                                           cudf::type_id::DURATION_MILLISECONDS,
                                           cudf::type_id::DURATION_MICROSECONDS,
                                           cudf::type_id::DURATION_NANOSECONDS,
                                           cudf::type_id::STRING,
                                           cudf::type_id::LIST,
                                           cudf::type_id::DECIMAL32,
                                           cudf::type_id::DECIMAL64,
                                           cudf::type_id::DECIMAL128,
                                           cudf::type_id::STRUCT>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::type_id,
  [](auto value) { return cudf::type_to_name(cudf::data_type{value}); },
  [](auto value) { return cudf::type_to_name(cudf::data_type{value}); })

NVBENCH_BENCH_TYPES(BM_to_arrow_host, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("to_arrow_host")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});

NVBENCH_BENCH_TYPES(BM_to_arrow_device, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("to_arrow_device")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});

NVBENCH_BENCH_TYPES(BM_from_arrow_host, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("from_arrow_host")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});

NVBENCH_BENCH_TYPES(BM_from_arrow_device, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("from_arrow_device")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});
