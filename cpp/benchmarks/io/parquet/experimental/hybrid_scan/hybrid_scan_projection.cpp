/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <limits>
#include <memory>
#include <tuple>
#include <utility>

// Benchmark hybrid scan payload column selection with a filter and no column projection.
void BM_hybrid_scan_projection(nvbench::state& state)
{
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));

  cuio_source_sink_pair source_sink(io_type::FILEPATH);

  constexpr cudf::size_type num_rows = 1;
  auto const tbl =
    create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                        row_count{num_rows},
                        data_profile_builder().cardinality(0).avg_run_length(1).no_validity());

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), tbl->view())
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(write_opts);

  cudf::numeric_scalar<int32_t> filter_literal{std::numeric_limits<int32_t>::min()};
  cudf::ast::tree filter_tree;
  auto const& col_ref = filter_tree.push(cudf::ast::column_name_reference("_col0"));
  auto const& lit     = filter_tree.push(cudf::ast::literal(filter_literal));
  auto const& filter_expr =
    filter_tree.push(cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, lit));

  auto const read_opts = cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
                           .filter(filter_expr)
                           .build();

  auto const source_info = source_sink.make_source_info();
  auto datasource        = std::move(cudf::io::make_datasources(source_info).front());
  auto const footer      = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const file_metadata =
    cudf::io::parquet::experimental::hybrid_scan_metadata{*footer, read_opts};
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(file_metadata);
  auto const row_groups = reader->all_row_groups(read_opts);

  state.add_element_count(num_cols, "schema_columns");
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(source_info.filepaths());
               reader->reset_column_selection();
               // Select the filter columns first, as a real read does, so the timed call covers
               // only the payload selection
               std::ignore = reader->filter_column_chunks_byte_ranges(row_groups, read_opts);

               timer.start();
               std::ignore = reader->payload_column_chunks_byte_ranges(row_groups, read_opts);
               timer.stop();
             });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_hybrid_scan_projection)
  .set_name("hybrid_scan_projection")
  .set_min_samples(4)
  .add_int64_axis("num_cols", {64, 512, 2048, 4096});
