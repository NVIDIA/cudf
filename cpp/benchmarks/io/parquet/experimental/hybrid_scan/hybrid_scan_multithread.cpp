/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_composer.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <nvtx3/nvtx3.hpp>

#include <BS_thread_pool.hpp>
#include <nvbench/nvbench.cuh>

#include <vector>

std::string get_label(std::string const& test_name, nvbench::state const& state)
{
  auto const num_cols       = state.get_int64("num_cols");
  auto const num_reads      = state.get_int64("num_threads");
  size_t const read_size_mb = state.get_int64("total_data_size") / num_reads;
  return {test_name + ", " + std::to_string(num_cols) + " columns, " +
          std::to_string(state.get_int64("num_iterations")) + " iterations, " +
          std::to_string(state.get_int64("num_threads")) + " threads " + " (" +
          std::to_string(read_size_mb) + " MB each)"};
}

std::tuple<std::vector<cuio_source_sink_pair>, size_t, size_t> write_file_data(
  nvbench::state& state,
  std::vector<cudf::type_id> const& d_types,
  io_type io_source_type,
  bool write_page_index = false)
{
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  cudf::size_type const num_cols    = state.get_int64("num_cols");
  size_t const num_files            = state.get_int64("num_threads");
  size_t const per_file_data_size   = state.get_int64("total_data_size") / num_files;
  auto const rg_size_bytes          = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows           = state.get_int64("row_group_size_rows");

  std::vector<cuio_source_sink_pair> source_sink_vector;

  size_t total_file_size = 0;

  for (size_t i = 0; i < num_files; ++i) {
    cuio_source_sink_pair source_sink{io_source_type};

    auto const tbl = create_random_table(
      cycle_dtypes(d_types, num_cols),
      table_size_bytes{per_file_data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::SNAPPY)
        .max_page_size_rows(50000)
        .max_page_size_bytes(1024 * 1024);
    // Sentinel 0 == use cuDF default (parquet bytes default is size_t::max).
    if (rg_size_bytes > 0) write_opts.set_row_group_size_bytes(rg_size_bytes);
    if (rg_size_rows > 0) write_opts.set_row_group_size_rows(rg_size_rows);

    if (write_page_index) {
      write_opts.set_stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
    }

    cudf::io::write_parquet(write_opts);
    total_file_size += source_sink.size();

    source_sink_vector.push_back(std::move(source_sink));
  }

  return {std::move(source_sink_vector), total_file_size, num_files};
}

void BM_hybrid_scan_multithreaded_read_common(nvbench::state& state,
                                              std::vector<cudf::type_id> const& d_types,
                                              std::string const& label)
{
  size_t const data_size    = state.get_int64("total_data_size");
  auto const num_threads    = state.get_int64("num_threads");
  auto const num_iterations = state.get_int64("num_iterations");
  auto const source_type    = retrieve_io_type_enum(state.get_string("io_type"));

  std::unordered_set<hybrid_scan_filter_type> const filters = {
    hybrid_scan_filter_type::ROW_GROUPS_WITH_STATS,
    hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS,
  };

  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  BS::thread_pool threads(num_threads);

  auto [source_sink_vector, total_file_size, num_files] =
    write_file_data(state, d_types, source_type);
  std::vector<cudf::io::source_info> source_info_vector;
  std::transform(source_sink_vector.begin(),
                 source_sink_vector.end(),
                 std::back_inserter(source_info_vector),
                 [](auto& source_sink) { return source_sink.make_source_info(); });

  auto filepaths = std::vector<std::string>{};
  if (source_type == io_type::FILEPATH) {
    filepaths.reserve(source_info_vector.size());
    std::transform(source_info_vector.begin(),
                   source_info_vector.end(),
                   std::back_inserter(filepaths),
                   [](auto const& source_info) { return source_info.filepaths().front(); });
  }

  auto mem_stats_logger = cudf::memory_stats_logger();

  nvtxRangePushA(("(read) " + label).c_str());
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&, num_files = num_files](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(filepaths);

               auto read_func = [&](int index) {
                 auto const stream = streams[index % num_threads];
                 cudf::io::parquet_reader_options read_opts =
                   cudf::io::parquet_reader_options::builder(source_info_vector[index]);
                 for (int i = 0; i < num_iterations; ++i) {
                   hybrid_scan(read_opts, filters, stream, cudf::get_current_device_resource_ref());
                 }
               };

               threads.pause();
               threads.detach_sequence(decltype(num_files){0}, num_files, read_func);
               timer.start();
               threads.unpause();
               threads.wait();
               cudf::detail::join_streams(streams, cudf::get_default_stream());
               timer.stop();
             });
  nvtxRangePop();

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(num_iterations * static_cast<double>(data_size) / time,
                          "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(total_file_size, "encoded_file_size", "encoded_file_size");
}

void BM_hybrid_scan_multithreaded_read(nvbench::state& state)
{
  auto const data_type = state.get_string("data_type");
  auto label           = get_label(data_type, state);

  auto const dtypes = [&]() {
    if (data_type == "mixed") {
      return std::vector<cudf::type_id>{
        cudf::type_id::INT32, cudf::type_id::DECIMAL64, cudf::type_id::STRING};
    } else if (data_type == "fixed_width") {
      return std::vector<cudf::type_id>{cudf::type_id::INT32};
    } else if (data_type == "string") {
      return std::vector<cudf::type_id>{cudf::type_id::STRING};
    } else if (data_type == "list") {
      return std::vector<cudf::type_id>{cudf::type_id::LIST};
    } else {
      CUDF_FAIL("Invalid data type: " + data_type);
    }
  }();

  nvtxRangePushA(label.c_str());
  BM_hybrid_scan_multithreaded_read_common(state, dtypes, label);
  nvtxRangePop();
}

// mixed data types: fixed width and strings
NVBENCH_BENCH(BM_hybrid_scan_multithreaded_read)
  .set_name("hybrid_scan_multithreaded_read")
  .set_min_samples(4)
  .add_string_axis("data_type", {"mixed", "fixed_width", "string", "list"})
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("total_data_size", {512 * 1024 * 1024, 1024 * 1024 * 1024})
  .add_int64_axis("num_threads", {1, 2, 4, 8})
  .add_int64_axis("num_iterations", {1})
  .add_int64_axis("num_cols", {4})
  .add_int64_axis("run_length", {8})
  .add_string_axis("io_type", {"PINNED_BUFFER"})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});
