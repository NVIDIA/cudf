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
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <limits>
#include <numeric>

namespace {

constexpr cudf::size_type num_cols = 8;

void filter_row_groups_with_dicts_common(nvbench::state& state,
                                         cudf::type_id dtype,
                                         data_profile const& table_profile,
                                         cudf::ast::literal const& literal)
{
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const cardinality    = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const is_inline_eval = static_cast<bool>(state.get_int64("is_inline"));
  auto constexpr rows_per_row_group = 5'000;  //< Chosen such that it is not ignored by the writer
  auto const num_rows               = num_row_groups * rows_per_row_group;

  std::vector<char> parquet_buffer;

  // Write table to parquet
  {
    auto const table =
      create_random_table(cycle_dtypes({dtype}, num_cols), row_count{num_rows}, table_profile);

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&parquet_buffer), table->view())
        .row_group_size_rows(rows_per_row_group)
        .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .compression(cudf::io::compression_type::AUTO);
    cudf::io::write_parquet(write_opts);
  }

  auto col_ref = cudf::ast::column_name_reference("_col0");
  auto expr1   = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);
  auto expr2   = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_ref, literal);
  auto expr3   = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto filter_expr_few_literals =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr1, expr2);
  auto filter_expr_many_literals =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, filter_expr_few_literals, expr3);
  auto const& filter_expr = is_inline_eval ? filter_expr_few_literals : filter_expr_many_literals;

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

  // Upper bound on the dictionary entries. Real dictionaries are smaller because the generator
  // repeats each value it draws for a run of rows, and may draw the same value again
  auto const num_dict_entries =
    static_cast<std::size_t>(std::min(cardinality, rows_per_row_group) * num_row_groups);

  state.add_element_count(num_dict_entries, "num_dict_entries");

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

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  auto const total_dict_data_size =
    std::accumulate(dict_page_byte_ranges.begin(),
                    dict_page_byte_ranges.end(),
                    std::size_t{0},
                    [](auto acc, auto const& range) { return acc + range.size(); });
  state.add_buffer_size(total_dict_data_size, "total_dict_data_size", "total_dict_data_size");
}

}  // namespace

void BM_hybrid_scan_dict_page_pruning_string(nvbench::state& state)
{
  auto const min_length  = static_cast<cudf::size_type>(state.get_int64("min_length"));
  auto const max_length  = static_cast<cudf::size_type>(state.get_int64("max_length"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));

  auto table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_length, max_length)
      .cardinality(cardinality);

  auto filter_value = cudf::string_scalar("000010000");
  filter_row_groups_with_dicts_common(
    state, cudf::type_id::STRING, table_profile, cudf::ast::literal(filter_value));
}

template <cudf::type_id DType>
void BM_hybrid_scan_dict_page_pruning_fixed_width(nvbench::state& state,
                                                  nvbench::type_list<nvbench::enum_type<DType>>)
{
  using T = cudf::id_to_type<DType>;
  static_assert(cudf::is_numeric<T>() or cudf::is_chrono<T>(),
                "Filter literals are only generated for numeric and chrono types");

  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));

  // Timestamps span 1970 to 2020 as the generator overflows past 2262
  // Durations span 24 days to fit
  auto constexpr max_value = [] {
    if constexpr (cudf::is_timestamp<T>()) {
      return cuda::std::chrono::duration_cast<typename T::duration>(cudf::duration_D{50 * 365})
        .count();
    } else if constexpr (cudf::is_duration<T>()) {
      return cuda::std::chrono::duration_cast<T>(cudf::duration_D{24}).count();
    } else {
      return std::numeric_limits<T>::max();
    }
  }();

  auto table_profile =
    data_profile_builder()
      .distribution(DType, distribution_id::UNIFORM, decltype(max_value){0}, max_value)
      .cardinality(cardinality);

  // The literal lies inside the value range, so min/max statistics cannot prune any row group
  auto filter_value = [&] {
    if constexpr (cudf::is_timestamp<T>()) {
      return cudf::timestamp_scalar<T>(T{typename T::duration{max_value / 2}});
    } else if constexpr (cudf::is_duration<T>()) {
      return cudf::duration_scalar<T>(T{max_value / 2});
    } else {
      return cudf::numeric_scalar<T>(static_cast<T>(max_value / 2));
    }
  }();
  filter_row_groups_with_dicts_common(
    state, DType, table_profile, cudf::ast::literal(filter_value));
}

NVBENCH_BENCH(BM_hybrid_scan_dict_page_pruning_string)
  .set_name("hybrid_scan_dict_page_pruning_string")
  .set_min_samples(4)
  .add_int64_axis("num_row_groups", {32, 64, 128})
  .add_int64_axis("min_length", {4})
  .add_int64_axis("max_length", {64, 128})
  .add_int64_axis("cardinality", {1'000, 10'000})
  .add_int64_axis("is_inline", {true, false});

using dict_fixed_width_dtypes = nvbench::enum_type_list<cudf::type_id::INT8,
                                                        cudf::type_id::FLOAT32,
                                                        cudf::type_id::FLOAT64,
                                                        cudf::type_id::DURATION_MILLISECONDS,
                                                        cudf::type_id::TIMESTAMP_MICROSECONDS>;

NVBENCH_BENCH_TYPES(BM_hybrid_scan_dict_page_pruning_fixed_width,
                    NVBENCH_TYPE_AXES(dict_fixed_width_dtypes))
  .set_name("hybrid_scan_dict_page_pruning_fixed_width")
  .set_type_axes_names({"dtype"})
  .set_min_samples(4)
  .add_int64_axis("num_row_groups", {32, 64, 128})
  .add_int64_axis("cardinality", {1'000, 10'000})
  .add_int64_axis("is_inline", {true, false});
