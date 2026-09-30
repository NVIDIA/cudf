/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr cudf::size_type max_filter_columns = 8;

/**
 * @brief Writes a random-data Parquet input to the given sink for the benchmark
 */
void write_benchmark_input(cudf::io::sink_info const& sink,
                           std::vector<cudf::type_id> const& dtypes,
                           cudf::size_type num_rows,
                           bool are_pages_aligned)
{
  auto const table =
    create_random_table(dtypes,
                        row_count{num_rows},
                        data_profile_builder()
                          .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, 8, 64)
                          .cardinality(0)
                          .no_validity());

  auto metadata = cudf::io::table_input_metadata(table->view());
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(dtypes.size()); ++i) {
    metadata.column_metadata[i].set_name("col" + std::to_string(i));
  }

  // Aligned mode caps every column at 5000 rows per page. Misaligned mode disables the row limit
  // and instead splits each column across pages using the byte limit depending on its physical
  // width.
  auto const max_page_size_rows = are_pages_aligned ? 5'000 : num_rows;
  auto const max_page_size_bytes =
    are_pages_aligned ? std::size_t{64} * 1024 * 1024 : std::size_t{16} * 1024;
  auto const write_options = cudf::io::parquet_writer_options::builder(sink, table->view())
                               .metadata(std::move(metadata))
                               .row_group_size_rows(num_rows)
                               .max_page_size_rows(max_page_size_rows)
                               .max_page_size_bytes(max_page_size_bytes)
                               .compression(cudf::io::compression_type::NONE)
                               .dictionary_policy(cudf::io::dictionary_policy::NEVER)
                               .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
                               .build();
  cudf::io::write_parquet(write_options);
}

/**
 * @brief Returns the sorted, unique row offsets bounding page segments across columns, along with
 * the total number of per-column page-row offsets before deduplication.
 */
[[nodiscard]] std::pair<std::vector<int64_t>, std::size_t> compute_segment_offsets(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::size_type num_rows,
  cudf::size_type num_filter_columns)
{
  auto const& row_group              = metadata.row_groups.front();
  auto segment_offsets               = std::vector<int64_t>{num_rows};
  std::size_t total_page_row_offsets = 0;
  for (cudf::size_type column_index = 0; column_index < num_filter_columns; ++column_index) {
    auto const& offset_index = row_group.columns[column_index].offset_index;
    CUDF_EXPECTS(offset_index.has_value(), "Offset index is required for benchmark input");
    total_page_row_offsets += offset_index->page_locations.size() + 1;
    std::transform(offset_index->page_locations.begin(),
                   offset_index->page_locations.end(),
                   std::back_inserter(segment_offsets),
                   [](auto const& page) { return page.first_row_index; });
  }
  std::sort(segment_offsets.begin(), segment_offsets.end());
  segment_offsets.erase(std::unique(segment_offsets.begin(), segment_offsets.end()),
                        segment_offsets.end());
  return {std::move(segment_offsets), total_page_row_offsets};
}

/**
 * @brief A filter expression tree and the scalars its literal nodes reference
 */
struct filter_expression {
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> literals;

  /// @return The root expression of the tree
  [[nodiscard]] cudf::ast::expression const& root() const { return tree.back(); }
};

using expression_ref = std::reference_wrapper<cudf::ast::expression const>;

/**
 * @brief Selects the logical operator for a combining node
 */
[[nodiscard]] cudf::ast::ast_operator logical_operator(cudf::size_type index)
{
  return index % 3 == 0 ? cudf::ast::ast_operator::LOGICAL_OR
                        : cudf::ast::ast_operator::LOGICAL_AND;
}

/**
 * @brief Pushes a literal of the specified type, keeping its scalar alive in `expression`
 */
template <typename T>
[[nodiscard]] cudf::ast::literal const& push_scalar(filter_expression& expression, T value)
{
  auto scalar         = std::make_unique<cudf::numeric_scalar<T>>(value);
  auto const& literal = expression.tree.push(cudf::ast::literal{*scalar});
  expression.literals.push_back(std::move(scalar));
  return literal;
}

/**
 * @brief Pushes a string literal, keeping its scalar alive in `expression`
 */
[[nodiscard]] cudf::ast::literal const& push_string_scalar(filter_expression& expression,
                                                           std::string value)
{
  auto scalar         = std::make_unique<cudf::string_scalar>(std::move(value));
  auto const& literal = expression.tree.push(cudf::ast::literal{*scalar});
  expression.literals.push_back(std::move(scalar));
  return literal;
}

/**
 * @brief Pushes a literal typed to match the column it will be compared against
 */
[[nodiscard]] cudf::ast::literal const& push_literal(filter_expression& expression,
                                                     cudf::type_id column_type,
                                                     int64_t value)
{
  switch (column_type) {
    case cudf::type_id::INT8: return push_scalar(expression, static_cast<int8_t>(value));
    case cudf::type_id::INT16: return push_scalar(expression, static_cast<int16_t>(value));
    case cudf::type_id::INT32: return push_scalar(expression, static_cast<int32_t>(value));
    case cudf::type_id::INT64: return push_scalar(expression, value);
    case cudf::type_id::STRING: return push_string_scalar(expression, std::to_string(value + 100));
    default: CUDF_FAIL("Unsupported page-statistics benchmark column type");
  }
}

/**
 * @brief Pushes a `col<i> < v` or `col<i> >= v` comparison against one filter column
 */
[[nodiscard]] cudf::ast::expression const& push_predicate(filter_expression& expression,
                                                          cudf::size_type predicate_index,
                                                          std::span<cudf::type_id const> dtypes)
{
  auto const num_columns  = static_cast<cudf::size_type>(dtypes.size());
  auto const column_index = predicate_index % num_columns;
  auto const& column =
    expression.tree.push(cudf::ast::column_name_reference("col" + std::to_string(column_index)));
  auto const& literal =
    push_literal(expression, dtypes[column_index], static_cast<int64_t>(predicate_index % 31) - 15);
  auto const comparison = predicate_index % 2 == 0 ? cudf::ast::ast_operator::LESS
                                                   : cudf::ast::ast_operator::GREATER_EQUAL;
  return expression.tree.push(cudf::ast::operation{comparison, column, literal});
}

/**
 * @brief Pushes a node combining two expressions under a logical operator
 */
cudf::ast::expression const& push_combination(filter_expression& expression,
                                              cudf::size_type index,
                                              cudf::ast::expression const& lhs,
                                              cudf::ast::expression const& rhs)
{
  return expression.tree.push(cudf::ast::operation{logical_operator(index), lhs, rhs});
}

/**
 * @brief Builds a filter expression where every selected column contributes a predicate
 */
[[nodiscard]] filter_expression make_filter_expression(cudf::size_type logical_depth,
                                                       std::vector<cudf::type_id> const& dtypes)
{
  auto const num_columns = static_cast<cudf::size_type>(dtypes.size());
  CUDF_EXPECTS(logical_depth > 0, "Expression depth must be positive");
  CUDF_EXPECTS(num_columns > 0, "Number of filter columns must be positive");

  auto expression    = filter_expression{};
  auto combine_index = cudf::size_type{0};

  auto level = std::vector<expression_ref>{};
  level.reserve(num_columns);
  for (cudf::size_type column_index = 0; column_index < num_columns; ++column_index) {
    level.emplace_back(push_predicate(expression, column_index, dtypes));
  }

  // Fold each level pairwise in place, carrying an odd trailing node up to the next level.
  auto built_depth = cudf::size_type{0};
  for (; level.size() > 1; ++built_depth) {
    auto folded = std::size_t{0};
    for (std::size_t index = 0; index < level.size(); index += 2, ++folded) {
      level[folded] = index + 1 == level.size()
                        ? level[index]
                        : expression_ref{push_combination(
                            expression, combine_index++, level[index], level[index + 1])};
    }
    level.erase(level.begin() + folded, level.end());
  }

  CUDF_EXPECTS(logical_depth >= built_depth,
               "Expression depth is too small to include every filter column");

  for (auto depth = built_depth; depth < logical_depth; ++depth) {
    // Read the root first, since push_predicate() appends and would otherwise become the root.
    auto const& root      = expression.root();
    auto const& predicate = push_predicate(expression, num_columns + depth, dtypes);
    push_combination(expression, depth, root, predicate);
  }

  return expression;
}

/**
 * @brief Adds an integer summary to the benchmark state
 */
void add_count_summary(nvbench::state& state,
                       std::string const& key,
                       std::string const& description,
                       std::int64_t value)
{
  auto& summary = state.add_summary(key);
  summary.set_string("name", key);
  summary.set_string("description", description);
  summary.set_int64("value", value);
}

}  // namespace

void BM_page_stats_row_mask(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_filter_columns =
    static_cast<cudf::size_type>(state.get_int64("num_filter_columns"));
  auto const expr_depth     = static_cast<cudf::size_type>(state.get_int64("expr_depth"));
  auto const page_alignment = state.get_string("page_alignment");

  CUDF_EXPECTS(num_filter_columns <= max_filter_columns,
               "Number of filter columns exceeds the maximum",
               std::invalid_argument);
  CUDF_EXPECTS(page_alignment == "aligned" or page_alignment == "misaligned",
               std::format("Unexpected page alignment mode provided: {}", page_alignment));

  std::vector<cudf::type_id> const page_stats_column_types{cudf::type_id::INT32,
                                                           cudf::type_id::STRING,
                                                           cudf::type_id::INT64,
                                                           cudf::type_id::INT8,
                                                           cudf::type_id::INT16};
  auto const dtypes = cycle_dtypes(page_stats_column_types, num_filter_columns);

  auto source_sink = cuio_source_sink_pair{io_type::FILEPATH};

  write_benchmark_input(
    source_sink.make_sink_info(), dtypes, num_rows, page_alignment == "aligned");

  auto const datasource =
    std::move(cudf::io::make_datasources(source_sink.make_source_info()).front());
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const footer_spans  = std::vector<cudf::host_span<uint8_t const>>{*footer_buffer};

  auto const filter  = make_filter_expression(expr_depth, dtypes);
  auto const options = cudf::io::parquet_reader_options::builder().filter(filter.root()).build();
  auto const reader = cudf::io::parquet::experimental::hybrid_scan_multifile{footer_spans, options};

  // Setup page indexes for page-statistics filtering
  {
    auto const page_index_ranges = reader.page_index_byte_ranges();
    CUDF_EXPECTS(page_index_ranges.size() == 1 and not page_index_ranges.front().is_empty(),
                 "Page index is required for page-statistics filtering");
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(*datasource, page_index_ranges.front());
    auto const page_index_spans = std::vector<cudf::host_span<uint8_t const>>{*page_index_buffer};
    reader.setup_page_indexes(page_index_spans);
  }

  auto const parquet_metadatas = reader.parquet_metadatas();
  auto const [segment_offsets, total_page_row_offsets] =
    compute_segment_offsets(parquet_metadatas.front(), num_rows, num_filter_columns);

  add_count_summary(state,
                    "page_row_offsets",
                    "Total page-row offsets across filter columns",
                    static_cast<std::int64_t>(total_page_row_offsets));
  add_count_summary(state,
                    "segments",
                    "Unique common page segments",
                    static_cast<std::int64_t>(segment_offsets.size() - 1));

  auto const row_groups = reader.all_row_groups(options);
  auto const stream     = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_element_count(num_rows, "rows");
  auto const mem_stats_logger = cudf::memory_stats_logger{};

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
      timer.start();
      auto const row_mask = reader.build_row_mask_with_page_index_stats(
        row_groups, options, stream, cudf::get_current_device_resource_ref());
      timer.stop();
      CUDF_EXPECTS(row_mask->type().id() == cudf::type_id::BOOL8, "Unexpected row-mask type");
      CUDF_EXPECTS(row_mask->size() == num_rows, "Unexpected row-mask size");
    });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_page_stats_row_mask)
  .set_name("row_mask_with_page_index_stats")
  .set_min_samples(4)
  .add_string_axis("page_alignment", {"aligned", "misaligned"})
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_filter_columns", {2, 4, 8})
  .add_int64_axis("expr_depth", {4, 8});
