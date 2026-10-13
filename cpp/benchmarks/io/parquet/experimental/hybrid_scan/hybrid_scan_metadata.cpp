/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>
#include <benchmarks/io/parquet/parquet_common.hpp>

#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <utility>

// Measure how page-index setup cost scales with file shape.
template <cudf::type_id DType>
void BM_hybrid_scan_setup_page_index(nvbench::state& state,
                                     nvbench::type_list<nvbench::enum_type<DType>>)
{
  auto constexpr write_page_index = true;

  auto const source_type    = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size      = static_cast<size_t>(state.get_int64("data_size"));
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const num_pages_per_row_group =
    static_cast<cudf::size_type>(state.get_int64("pages_per_row_group"));

  auto source_sink = [&]() {
    auto const table     = create_random_table({DType}, table_size_bytes{data_size});
    auto const num_pages = num_row_groups * num_pages_per_row_group;
    // The writer produces the requested layout only for a multiple of `num_pages` rows
    auto const num_rows = table->num_rows() / num_pages * num_pages;
    return write_file_shape_parquet_file(cudf::slice(table->view(), {0, num_rows}).front(),
                                         num_row_groups,
                                         num_pages_per_row_group,
                                         source_type,
                                         write_page_index);
  }();

  auto const read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info()).build();

  auto const datasource = std::move(cudf::io::make_datasources(read_opts.get_source()).front());
  auto const footer     = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const page_index_byte_range =
    cudf::io::parquet::experimental::hybrid_scan_reader(*footer, read_opts).page_index_byte_range();
  auto const page_index =
    cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_byte_range);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               // A reader sets up its page index only once, so every sample needs a new reader
               cudf::io::parquet::experimental::hybrid_scan_reader reader(*footer, read_opts);

               timer.start();
               reader.setup_page_index(*page_index);
               timer.stop();
             });

  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

// Fixed-width types differ only in the size of their min/max values, so INT32 stands in for all
// of them. STRING adds `unencoded_byte_array_data_bytes` to the offset index
using page_index_dtypes = nvbench::enum_type_list<cudf::type_id::INT32, cudf::type_id::STRING>;

NVBENCH_BENCH_TYPES(BM_hybrid_scan_setup_page_index, NVBENCH_TYPE_AXES(page_index_dtypes))
  .set_name("hybrid_scan_setup_page_index")
  .set_type_axes_names({"dtype"})
  .set_min_samples(4)
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .add_int64_axis("data_size", {32 << 20, 128 << 20, 512 << 20})
  .add_int64_axis("num_row_groups", {1, 10})
  .add_int64_axis("pages_per_row_group", {100, 1'000, 10'000});
