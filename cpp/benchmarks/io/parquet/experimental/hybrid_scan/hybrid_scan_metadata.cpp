/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>
#include <benchmarks/io/parquet/parquet_common.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

enum class setup_phase : int32_t {
  FOOTER_FETCH,      // copy footer bytes to host
  READER_CTOR,       // parse footer, construct reader
  PAGE_INDEX_FETCH,  // copy page-index bytes to host
  PAGE_INDEX_SETUP,  // parse page index into the reader
  ALL                // the whole path, for pairing against the naive reader
};

// NVBENCH_DECLARE_ENUM_TYPE_STRINGS macro must be used from global namespace scope
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  setup_phase,
  [](setup_phase value) {
    switch (value) {
      case setup_phase::FOOTER_FETCH: return "FOOTER_FETCH";
      case setup_phase::READER_CTOR: return "READER_CTOR";
      case setup_phase::PAGE_INDEX_FETCH: return "PAGE_INDEX_FETCH";
      case setup_phase::PAGE_INDEX_SETUP: return "PAGE_INDEX_SETUP";
      case setup_phase::ALL: return "ALL";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

namespace {

// Runs the whole setup path, timing only the step named by `Phase`, or all of it for ALL.
template <setup_phase Phase, typename Timer>
[[nodiscard]] auto run_setup_path(Timer& timer,
                                  cudf::io::datasource& datasource,
                                  cudf::io::parquet_reader_options const& options)
{
  constexpr auto all = Phase == setup_phase::ALL;

  if constexpr (Phase == setup_phase::FOOTER_FETCH or all) { timer.start(); }
  auto const footer = cudf::io::parquet::fetch_footer_to_host(datasource);
  if constexpr (Phase == setup_phase::FOOTER_FETCH) { timer.stop(); }

  if constexpr (Phase == setup_phase::READER_CTOR) { timer.start(); }
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer, options);
  if constexpr (Phase == setup_phase::READER_CTOR) { timer.stop(); }

  if constexpr (Phase == setup_phase::PAGE_INDEX_FETCH) { timer.start(); }
  auto const page_index =
    cudf::io::parquet::fetch_page_index_to_host(datasource, reader->page_index_byte_range());
  if constexpr (Phase == setup_phase::PAGE_INDEX_FETCH) { timer.stop(); }

  if constexpr (Phase == setup_phase::PAGE_INDEX_SETUP) { timer.start(); }
  reader->setup_page_index(*page_index);
  if constexpr (Phase == setup_phase::PAGE_INDEX_SETUP or all) { timer.stop(); }

  return reader;
}

}  // namespace

// Every cell runs the whole setup path and times the window named by the `phase` axis.
template <setup_phase Phase>
void BM_hybrid_scan_setup_phase(nvbench::state& state,
                                nvbench::type_list<nvbench::enum_type<Phase>>)
{
  auto const num_cols         = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_row_groups   = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const write_page_index = state.get_int64("page_index") != 0;

  if constexpr (Phase == setup_phase::PAGE_INDEX_FETCH or Phase == setup_phase::PAGE_INDEX_SETUP) {
    if (not write_page_index) {
      state.skip("Page-index phases require page_index=true");
      return;
    }
  }

  auto source_sink =
    write_mixed_dtype_parquet_file(num_cols, num_row_groups, source_type, write_page_index);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const source_info = source_sink.make_source_info();
      drop_page_cache_if_enabled(source_info.filepaths());
      auto datasource      = std::move(cudf::io::make_datasources(source_info).front());
      auto const read_opts = cudf::io::parquet_reader_options::builder(source_info).build();

      auto const reader = run_setup_path<Phase>(timer, *datasource, read_opts);
      CUDF_EXPECTS(std::cmp_equal(reader->all_row_groups(read_opts).size(), num_row_groups),
                   "Unexpected row group count");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols * num_row_groups) / time,
                          "colchunks_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

// Measure how page-index fetch and setup cost scales with file shape.
template <setup_phase Phase>
void BM_hybrid_scan_file_shape(nvbench::state& state, nvbench::type_list<nvbench::enum_type<Phase>>)
{
  auto constexpr d_type           = cudf::type_id::STRING;
  auto constexpr write_page_index = true;

  auto const source_type    = retrieve_io_type_enum(state.get_string("io_type"));
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const num_pages_per_row_group =
    static_cast<cudf::size_type>(state.get_int64("pages_per_row_group"));

  auto source_sink = write_file_shape_parquet_file(
    d_type, num_rows, num_row_groups, num_pages_per_row_group, source_type, write_page_index);

  auto const read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info()).build();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());
      auto datasource = std::move(cudf::io::make_datasources(read_opts.get_source()).front());

      auto const reader = run_setup_path<Phase>(timer, *datasource, read_opts);
      CUDF_EXPECTS(not reader->all_row_groups(read_opts).empty(),
                   "Expected at least one row group");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_row_groups * num_pages_per_row_group) / time,
                          "pages_per_sec");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using setup_phases = nvbench::enum_type_list<setup_phase::FOOTER_FETCH,
                                             setup_phase::READER_CTOR,
                                             setup_phase::PAGE_INDEX_FETCH,
                                             setup_phase::PAGE_INDEX_SETUP,
                                             setup_phase::ALL>;

using page_index_phases =
  nvbench::enum_type_list<setup_phase::PAGE_INDEX_FETCH, setup_phase::PAGE_INDEX_SETUP>;

NVBENCH_BENCH_TYPES(BM_hybrid_scan_setup_phase, NVBENCH_TYPE_AXES(setup_phases))
  .set_name("hybrid_scan_setup_phase")
  .set_type_axes_names({"phase"})
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("page_index", {true, false})
  .add_int64_axis("num_cols", {64, 256, 512})
  .add_int64_axis("num_row_groups", {10, 50});

NVBENCH_BENCH_TYPES(BM_hybrid_scan_file_shape, NVBENCH_TYPE_AXES(page_index_phases))
  .set_name("hybrid_scan_file_shape")
  .set_type_axes_names({"phase"})
  .set_min_samples(4)
  // Axes match parquet_read_file_shape so the naive and hybrid scan paths can be compared
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .add_int64_axis("num_rows", {10'000'000, 100'000'000})
  .add_int64_axis("num_row_groups", {1, 10})
  .add_int64_axis("pages_per_row_group", {1'000, 10'000});
