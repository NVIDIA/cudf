/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr std::size_t data_size = 512 << 20;

// Measures reads of dictionary-encoded low-cardinality columns with and without the direct
// Parquet-dictionary -> DICTIONARY32 transcode (`output_dict_columns`), over the column types the
// fast path accepts: flat strings and flat fixed-width INT32/INT64/TIMESTAMP_DAYS columns.
template <output_dict OutputDict>
void BM_parquet_read_dict_transcode(nvbench::state& state,
                                    nvbench::type_list<nvbench::enum_type<OutputDict>>)
{
  auto constexpr output_dict_columns = OutputDict == output_dict::YES;

  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));

  auto const data_types = std::vector<cudf::type_id>{cudf::type_id::INT32,
                                                     cudf::type_id::INT64,
                                                     cudf::type_id::TIMESTAMP_DAYS,
                                                     cudf::type_id::STRING};
  data_profile const profile =
    data_profile_builder().cardinality(cardinality).avg_run_length(run_length);
  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size}, profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  auto const write_options =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .build();
  cudf::io::write_parquet(write_options);

  cudf::io::parquet_reader_options read_options =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
      .output_dict_columns(output_dict_columns)
      .build();

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_options.get_source().filepaths());
      timer.start();
      auto const result = cudf::io::read_parquet(read_options);
      timer.stop();
      CUDF_EXPECTS(result.tbl->num_rows() == view.num_rows(),
                   "Benchmark did not read the entire table");
      CUDF_EXPECTS(result.tbl->num_columns() == view.num_columns(), "Unexpected number of columns");
    });

  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / elapsed_time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH_TYPES(BM_parquet_read_dict_transcode,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<output_dict::NO, output_dict::YES>))
  .set_name("parquet_read_dict_transcode")
  .set_type_axes_names({"output_dict_columns"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {1000})
  .add_int64_axis("run_length", {4});
