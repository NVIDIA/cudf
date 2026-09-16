/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/parquet/parquet_common.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <limits>
#include <string>
#include <vector>

enum class projection_side : int32_t {
  FILTER           = 0,  // one or two filter column names
  PAYLOAD          = 1,  // payload set derived from the schema, minus filter columns
  PAYLOAD_EXPLICIT = 2   // payload set supplied explicitly, minus filter columns
};

// NVBENCH_DECLARE_ENUM_TYPE_STRINGS macro must be used from global namespace scope
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  projection_side,
  [](projection_side value) {
    switch (value) {
      case projection_side::FILTER: return "FILTER";
      case projection_side::PAYLOAD: return "PAYLOAD";
      case projection_side::PAYLOAD_EXPLICIT: return "PAYLOAD_EXPLICIT";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

template <projection_side Side>
void BM_hybrid_scan_projection(nvbench::state& state, nvbench::type_list<nvbench::enum_type<Side>>)
{
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));

  auto source_sink = write_named_resolution_parquet_file(num_cols, io_type::FILEPATH);

  // The deterministic column names the fixture wrote, regenerated here for the PAYLOAD_EXPLICIT
  // cell's projection.
  std::vector<std::string> column_names(num_cols);
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    column_names[i] = "col" + std::to_string(i);
  }

  cudf::numeric_scalar<int32_t> filter_literal{std::numeric_limits<int32_t>::min()};
  cudf::ast::tree filter_tree;
  auto const& col_ref = filter_tree.push(cudf::ast::column_name_reference("col0"));
  auto const& lit     = filter_tree.push(cudf::ast::literal(filter_literal));
  auto const& filter_expr =
    filter_tree.push(cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, lit));

  auto read_opts_builder =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info()).filter(filter_expr);

  // Caller-supplied payload list
  if constexpr (Side == projection_side::PAYLOAD_EXPLICIT) {
    read_opts_builder.column_names(column_names);
  }
  auto const read_opts = read_opts_builder.build();

  auto const source_info = source_sink.make_source_info();
  auto datasource        = std::move(cudf::io::make_datasources(source_info).front());
  auto const footer      = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const file_metadata =
    cudf::io::parquet::experimental::hybrid_scan_metadata{*footer, read_opts};
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(file_metadata);
  auto const row_groups = reader->all_row_groups(read_opts);

  if constexpr (Side == projection_side::PAYLOAD or Side == projection_side::PAYLOAD_EXPLICIT) {
    // Prime the filter-side column selection so payload measures the expensive branch.
    // Without this the payload cell resolves through the cheap branch.
    std::ignore = reader->filter_column_chunks_byte_ranges(row_groups, read_opts);
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(source_info.filepaths());
               reader->reset_column_selection();

               timer.start();
               if constexpr (Side == projection_side::FILTER) {
                 std::ignore = reader->filter_column_chunks_byte_ranges(row_groups, read_opts);
               } else {
                 // PAYLOAD and PAYLOAD_EXPLICIT differ only in whether the payload list is
                 // caller-supplied or derived from the schema
                 std::ignore = reader->payload_column_chunks_byte_ranges(row_groups, read_opts);
               }
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols) / time, "cols_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using projection_sides = nvbench::enum_type_list<projection_side::FILTER,
                                                 projection_side::PAYLOAD,
                                                 projection_side::PAYLOAD_EXPLICIT>;

NVBENCH_BENCH_TYPES(BM_hybrid_scan_projection, NVBENCH_TYPE_AXES(projection_sides))
  .set_name("hybrid_scan_projection")
  .set_type_axes_names({"side"})
  .set_min_samples(4)
  .add_int64_axis("num_cols", {64, 512, 2048, 4096});
