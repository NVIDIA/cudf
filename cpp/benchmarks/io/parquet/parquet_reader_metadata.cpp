/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Common mixed dtypes used by all benchmarks in this file
auto const mixed_dtypes = get_type_or_group({static_cast<int32_t>(data_type::STRING),
                                             static_cast<int32_t>(data_type::INTEGRAL),
                                             static_cast<int32_t>(data_type::FLOAT),
                                             static_cast<int32_t>(data_type::DECIMAL),
                                             static_cast<int32_t>(data_type::LIST)});

namespace {

// Helper to generate and write parquet data using chunked writer
auto write_file_data(cudf::size_type num_cols,
                     cudf::size_type num_row_groups,
                     io_type source_type,
                     bool write_page_index)
{
  cuio_source_sink_pair source_sink(source_type);

  // Minimum row group size that cudf will almost always follow
  constexpr auto rows_per_row_group = 5000;
  constexpr auto min_row_groups     = 10;
  constexpr auto table_rows         = min_row_groups * rows_per_row_group;

  CUDF_EXPECTS(num_row_groups > 0 and num_row_groups % min_row_groups == 0,
               "Number of requested row groups must be non-zero and a multiple of " +
                 std::to_string(min_row_groups));

  // Create a table with the enough rows to cover min_row_groups
  auto const tbl =
    create_random_table(cycle_dtypes(mixed_dtypes, num_cols),
                        row_count{table_rows},
                        data_profile_builder().cardinality(0).avg_run_length(1).distribution(
                          cudf::type_id::LIST, distribution_id::GEOMETRIC, 0, 4));
  auto const view = tbl->view();

  auto const stats_level = write_page_index ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                            : cudf::io::statistics_freq::STATISTICS_ROWGROUP;
  auto const options =
    cudf::io::chunked_parquet_writer_options::builder(source_sink.make_sink_info())
      .row_group_size_rows(rows_per_row_group)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(stats_level)
      .build();
  auto writer = cudf::io::chunked_parquet_writer(options, cudf::get_default_stream());

  // Compute the number of times the table needs to be written to cover the requested number of row
  // groups
  auto num_writes = cudf::util::div_rounding_up_unsafe(num_row_groups, min_row_groups);
  std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                cuda::counting_iterator{num_writes},
                [&](cudf::size_type) { writer.write(view); });

  std::ignore = writer.close();

  return source_sink;
}

// Combines `operands` into a balanced AST tree using `op`: pairing adjacent operands gives a tree
// of depth ceil(log2(n)) rather than the n-deep chain a left fold would produce.
[[nodiscard]] cudf::ast::expression const* reduce_balanced(
  cudf::ast::tree& tree,
  cudf::ast::ast_operator op,
  std::vector<cudf::ast::expression const*> operands)
{
  CUDF_EXPECTS(not operands.empty(), "Cannot reduce an empty set of operands");
  while (operands.size() > 1) {
    std::vector<cudf::ast::expression const*> next;
    next.reserve((operands.size() + 1) / 2);
    for (std::size_t i = 0; i + 1 < operands.size(); i += 2) {
      next.push_back(&tree.push(cudf::ast::operation(op, *operands[i], *operands[i + 1])));
    }
    // Carry an odd trailing operand up to the next level unchanged.
    if (operands.size() % 2 == 1) { next.push_back(operands.back()); }
    operands = std::move(next);
  }
  return operands.front();
}

}  // namespace

// Benchmark to measure parquet footer read time
void BM_parquet_read_footer(nvbench::state& state)
{
  auto const num_cols         = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_row_groups   = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const write_page_index = state.get_int64("page_index") != 0;

  auto source_sink = write_file_data(num_cols, num_row_groups, source_type, write_page_index);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const source_info = source_sink.make_source_info();
      drop_page_cache_if_enabled(source_info.filepaths());
      auto sources = cudf::io::make_datasources(source_info);

      timer.start();
      auto const metadatas = cudf::io::read_parquet_footers(sources);
      timer.stop();

      // Validate metadata
      CUDF_EXPECTS(std::cmp_equal(metadatas.size(), 1), "Expected one metadata object");
      CUDF_EXPECTS(std::cmp_equal(metadatas.front().row_groups.size(), num_row_groups),
                   "Unexpected number of row groups in metadata. Got: " +
                     std::to_string(metadatas.front().row_groups.size()) +
                     " Expected: " + std::to_string(num_row_groups));
      CUDF_EXPECTS(std::cmp_equal(metadatas.front().row_groups.front().columns.size(), num_cols),
                   "Unexpected number of columns in metadata");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols * num_row_groups) / time,
                          "colchunks_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

// Benchmark to measure chunked parquet reader construction time
void BM_parquet_reader_construction(nvbench::state& state)
{
  auto const num_cols         = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_row_groups   = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const write_page_index = state.get_int64("page_index") != 0;

  auto source_sink = write_file_data(num_cols, num_row_groups, source_type, write_page_index);

  auto constexpr chunk_read_limit = 0;
  auto constexpr pass_read_limit  = 0;

  auto const read_opts = cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
                           .use_arrow_schema(false)
                           .allow_mismatched_pq_schemas(false)
                           .convert_strings_to_categories(false)
                           .build();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());
      timer.start();
      auto reader = cudf::io::chunked_parquet_reader(chunk_read_limit, pass_read_limit, read_opts);
      timer.stop();

      // Validate
      CUDF_EXPECTS(reader.has_next(), "Expected reader to have data");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols * num_row_groups) / time,
                          "colchunks_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

// Benchmark to measure parquet column selection time
void BM_parquet_column_selection(nvbench::state& state)
{
  auto const num_cols    = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));

  cuio_source_sink_pair source_sink(source_type);

  // Create a table with minimal rows (1 row is enough to create valid parquet)
  constexpr cudf::size_type num_rows = 1;
  auto const tbl                     = create_random_table(cycle_dtypes(mixed_dtypes, num_cols),
                                       row_count{num_rows},
                                       data_profile_builder().cardinality(0).avg_run_length(1));
  auto const view                    = tbl->view();

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(write_opts);

  auto constexpr chunk_read_limit = 0;
  auto constexpr pass_read_limit  = 0;

  auto const read_opts = cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
                           .use_arrow_schema(false)
                           .build();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const source_info = source_sink.make_source_info();
      drop_page_cache_if_enabled(source_info.filepaths());
      auto sources   = cudf::io::make_datasources(source_info);
      auto metadatas = cudf::io::read_parquet_footers(sources);

      // Constructing chunked parquet reader with existing datasource and metadata spends almost
      // entire time in column selection
      timer.start();
      auto reader = cudf::io::chunked_parquet_reader(
        chunk_read_limit, pass_read_limit, std::move(sources), std::move(metadatas), read_opts);
      timer.stop();

      // Validate
      CUDF_EXPECTS(reader.has_next(), "Expected reader to have data");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols) / time, "cols_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

// Benchmark Parquet filter column-name resolution during reader construction.
void BM_parquet_filter_name_resolution(nvbench::state& state)
{
  auto const num_cols       = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const case_sensitive = state.get_int64("case_sensitive") != 0;
  auto const heavy_filter   = state.get_int64("heavy_filter") != 0;
  auto const source_type    = retrieve_io_type_enum(state.get_string("io_type"));

  cuio_source_sink_pair source_sink(source_type);

  // Flat, single-row table of INT32 columns with deterministic names col0..col{n-1}. INT32 keeps
  // the filter literal trivially type-correct; name-resolution cost is independent of dtype.
  constexpr cudf::size_type num_rows = 1;
  auto const tbl =
    create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                        row_count{num_rows},
                        data_profile_builder().cardinality(0).avg_run_length(1).no_validity());
  auto const view = tbl->view();

  cudf::io::table_input_metadata input_meta(view);
  std::vector<std::string> file_names(num_cols);
  for (cudf::size_type i = 0; i < num_cols; i++) {
    file_names[i] = "col" + std::to_string(i);
    input_meta.column_metadata[i].set_name(file_names[i]);
  }

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .metadata(std::move(input_meta))
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(write_opts);

  // Query name: exact when case-sensitive, upper-cased when case-insensitive so the converter must
  // normalize on lookup.
  auto const to_query_case = [case_sensitive](std::string s) {
    if (not case_sensitive) {
      std::transform(
        s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    }
    return s;
  };

  // Always-true filter. `heavy_filter` controls how many columns it references:
  //   light: `col0 >= MIN`                                  (converter resolves 1 name reference)
  //   heavy: `(col0 >= MIN) OR (col1 >= MIN) OR ...`         (one reference per column)
  // The heavy OR tree is built balanced (depth ~log2(num_cols)) to keep visitor recursion shallow.
  cudf::numeric_scalar<int32_t> filter_literal_value{std::numeric_limits<int32_t>::min()};
  cudf::ast::tree expr;
  auto const& lit_expr      = expr.push(cudf::ast::literal(filter_literal_value));
  auto const make_predicate = [&](cudf::size_type col) -> cudf::ast::expression const& {
    auto const& col_ref =
      expr.push(cudf::ast::column_name_reference(to_query_case(file_names[col])));
    return expr.push(
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, lit_expr));
  };

  auto const num_predicates = heavy_filter ? num_cols : cudf::size_type{1};
  std::vector<cudf::ast::expression const*> predicates;
  predicates.reserve(num_predicates);
  for (cudf::size_type col = 0; col < num_predicates; col++) {
    predicates.push_back(&make_predicate(col));
  }
  // Reduce the per-column predicates into a single balanced OR tree so the AST depth stays
  // logarithmic in the column count (see reduce_balanced).
  auto const& filter_expr =
    *reduce_balanced(expr, cudf::ast::ast_operator::LOGICAL_OR, std::move(predicates));

  auto constexpr chunk_read_limit = 0;
  auto constexpr pass_read_limit  = 0;

  // No column projection is requested, so the reader reads all columns; this isolates filter name
  // resolution (named_to_reference_converter) from select_columns name scanning.
  auto read_opts = cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
                     .use_arrow_schema(false)
                     .build();
  read_opts.enable_case_sensitive_names(case_sensitive);
  read_opts.set_filter(filter_expr);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const source_info = source_sink.make_source_info();
      drop_page_cache_if_enabled(source_info.filepaths());
      auto sources   = cudf::io::make_datasources(source_info);
      auto metadatas = cudf::io::read_parquet_footers(sources);

      // Reader construction resolves all referenced filter column names
      // (named_to_reference_converter) and runs column selection. Construction throws if a
      // referenced name is missing, so successful construction is the validation; has_next() is
      // intentionally not called so per-sample timing is not perturbed by row-group filter
      // evaluation over the wide predicate.
      timer.start();
      [[maybe_unused]] auto const reader = cudf::io::chunked_parquet_reader(
        chunk_read_limit, pass_read_limit, std::move(sources), std::move(metadatas), read_opts);
      timer.stop();
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_cols) / time, "cols_per_sec");
  // Should be 0, but adding for completeness
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_parquet_read_footer)
  .set_name("parquet_read_footer")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("page_index", {true, false})
  .add_int64_axis("num_cols", {64, 256, 512})
  .add_int64_axis("num_row_groups", {10, 50});

NVBENCH_BENCH(BM_parquet_reader_construction)
  .set_name("parquet_reader_construction")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("page_index", {true, false})
  .add_int64_axis("num_cols", {64, 256, 512})
  .add_int64_axis("num_row_groups", {10, 50});

NVBENCH_BENCH(BM_parquet_column_selection)
  .set_name("parquet_column_selection")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("num_cols", {64, 512, 2048});

NVBENCH_BENCH(BM_parquet_filter_name_resolution)
  .set_name("parquet_filter_name_resolution")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH"})
  .add_int64_axis("num_cols", {64, 128, 256, 512, 1024, 1536, 2048, 4096})
  .add_int64_axis("case_sensitive", {1, 0})
  .add_int64_axis("heavy_filter", {0, 1});
