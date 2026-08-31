/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <numeric>

constexpr cudf::size_type num_cols = 8;

void BM_filter_string_row_groups_with_dicts_common(nvbench::state& state,
                                                   data_profile const& table_profile,
                                                   cudf::ast::operation const& filter_expr,
                                                   double average_str_length,
                                                   cudf::size_type cardinality)
{
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto constexpr rows_per_row_group = 5'000;  //< Chosen such that it is not ignored by the writer
  auto const num_rows               = num_row_groups * rows_per_row_group;

  std::vector<char> parquet_buffer;

  // Write table to parquet
  {
    auto const table = create_random_table(
      cycle_dtypes({cudf::type_id::STRING}, num_cols), row_count{num_rows}, table_profile);

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&parquet_buffer), table->view())
        .row_group_size_rows(rows_per_row_group)
        .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .compression(cudf::io::compression_type::AUTO);
    cudf::io::write_parquet(write_opts);
  }

  auto const stream    = cudf::get_default_stream();
  auto const read_opts = cudf::io::parquet_reader_options::builder().filter(filter_expr).build();

  // Create datasource from parquet buffer
  auto const datasource = cudf::io::datasource::create(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(parquet_buffer.data()), parquet_buffer.size()));
  auto datasource_ref   = std::ref(*datasource);

  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource_ref);
  auto const reader        = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    *footer_buffer, read_opts);

  auto const parquet_metadata = reader->parquet_metadata();
  CUDF_EXPECTS(
    std::cmp_equal(parquet_metadata.row_groups.size(), num_row_groups),
    "Number of row groups written to the file must match the number of requested row groups");

  auto const page_index_byte_range = reader->page_index_byte_range();
  CUDF_EXPECTS(not page_index_byte_range.is_empty(),
               "Page index is required for dictionary page based filtering");

  // Setup page index
  auto const page_index_buffer =
    cudf::io::parquet::fetch_page_index_to_host(datasource_ref, page_index_byte_range);
  reader->setup_page_index(*page_index_buffer);

  auto input_row_group_indices = reader->all_row_groups(read_opts);
  auto dict_page_byte_ranges   = std::vector<cudf::io::text::byte_range_info>{};

  auto mem_stats_logger = cudf::memory_stats_logger();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());

      timer.start();

      // Get dictionary page byte ranges
      dict_page_byte_ranges =
        reader->dictionary_pages_byte_ranges(input_row_group_indices, read_opts);
      CUDF_EXPECTS(not dict_page_byte_ranges.empty(), "No dictionary page byte ranges found");

      // Fetch dictionary page data
      auto [dictionary_page_buffers, dictionary_page_data, read_task] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
          datasource_ref,
          dict_page_byte_ranges,
          cudf::io::parquet::io_submission_policy::SERIALIZE,
          stream,
          cudf::get_current_device_resource_ref());
      read_task.get();

      // Filter row groups with dictionary pages
      std::ignore = reader->filter_row_groups_with_dictionary_pages(
        dictionary_page_data, input_row_group_indices, read_opts, stream);

      timer.stop();
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_element_count(
    static_cast<double>(cardinality * num_row_groups * average_str_length) / time,
    "strings_per_second");
  auto const total_dict_data_size =
    std::accumulate(dict_page_byte_ranges.begin(),
                    dict_page_byte_ranges.end(),
                    std::size_t{0},
                    [](auto acc, auto const& range) { return acc + range.size(); });
  state.add_buffer_size(total_dict_data_size, "total_dict_data_size", "total_dict_data_size");
}

void BM_filter_string_rowgroups_with_dicts(nvbench::state& state)
{
  auto const min_length     = static_cast<cudf::size_type>(state.get_int64("min_length"));
  auto const max_length     = static_cast<cudf::size_type>(state.get_int64("max_length"));
  auto const cardinality    = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const is_inline_eval = static_cast<bool>(state.get_int64("is_inline"));

  auto table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_length, max_length)
      .cardinality(cardinality);

  auto col_ref = cudf::ast::column_name_reference("_col0");
  auto scalar  = cudf::string_scalar("000010000");
  auto literal = cudf::ast::literal(scalar);
  auto expr1   = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);
  auto expr2   = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_ref, literal);
  auto expr3   = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto filter_expr_few_literals =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr2);

  auto filter_expr_many_literals =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, filter_expr_few_literals, expr3);

  return BM_filter_string_row_groups_with_dicts_common(
    state,
    table_profile,
    is_inline_eval ? filter_expr_few_literals : filter_expr_many_literals,
    (static_cast<double>(min_length) + static_cast<double>(max_length)) / 2,
    cardinality);
}

NVBENCH_BENCH(BM_filter_string_rowgroups_with_dicts)
  .set_name("hybrid_scan_filter_string_rowgroups_with_dicts")
  .set_min_samples(4)
  .add_int64_axis("num_row_groups", {32, 64, 128})
  .add_int64_axis("min_length", {4})
  .add_int64_axis("max_length", {64, 128})
  .add_int64_axis("cardinality", {1'000, 10'000})
  .add_int64_axis("is_inline", {true, false});
