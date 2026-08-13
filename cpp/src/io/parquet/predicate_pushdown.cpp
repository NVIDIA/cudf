/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "expression_transform_helpers.hpp"
#include "reader_impl_helpers.hpp"
#include "stats_filter_helpers.hpp"
#include "timestamp_utils.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <unordered_set>

namespace cudf::io::parquet::detail {

namespace {

/**
 * @brief Converts column chunk statistics to 2 device columns - min, max values.
 *
 * Each column's number of rows equals the total number of row groups.
 *
 */
struct row_group_stats_caster : public stats_caster_base {
  size_type total_row_groups;
  std::vector<metadata> const& per_file_metadata;
  host_span<std::vector<size_type> const> row_group_indices;
  bool has_is_null_operator;

  // Creates device columns from column statistics (min, max)
  template <typename T>
  std::
    tuple<std::unique_ptr<column>, std::unique_ptr<column>, std::optional<std::unique_ptr<column>>>
    operator()(host_span<int const> per_source_schema_indices,
               cudf::data_type dtype,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr) const
  {
    // List, Struct, Dictionary types are not supported
    if constexpr (cudf::is_compound<T>() && !std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types do not have statistics");
    } else {
      host_column<T> min(total_row_groups, stream);
      host_column<T> max(total_row_groups, stream);
      std::optional<host_column<bool>> is_null;
      if (has_is_null_operator) { is_null = host_column<bool>(total_row_groups, stream); }

      size_type stats_idx = 0;
      for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
        auto const mapped_schema_idx = per_source_schema_indices[src_idx];
        // Compute timestamp scale factor for precision conversion from the mapped source schema.
        auto const ts_scale = [&] {
          if constexpr (cudf::is_timestamp<T>()) {
            auto const& schema = per_file_metadata[src_idx].schema[mapped_schema_idx];
            return calc_timestamp_scale(schema.logical_type, static_cast<int32_t>(T::period::den));
          }
          return 0;
        }();

        for (auto const rg_idx : row_group_indices[src_idx]) {
          auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
          auto col              = std::find_if(row_group.columns.begin(),
                                  row_group.columns.end(),
                                  [mapped_schema_idx](ColumnChunk const& col) {
                                    return col.schema_idx == mapped_schema_idx;
                                  });
          if (col != std::end(row_group.columns)) {
            auto const& colchunk = *col;
            // To support deprecated min, max fields.
            auto const& min_value = colchunk.meta_data.statistics.min_value.has_value()
                                      ? colchunk.meta_data.statistics.min_value
                                      : colchunk.meta_data.statistics.min;
            auto const& max_value = colchunk.meta_data.statistics.max_value.has_value()
                                      ? colchunk.meta_data.statistics.max_value
                                      : colchunk.meta_data.statistics.max;
            // translate binary data to Type then to <T>
            min.set_index(stats_idx, min_value, colchunk.meta_data.type, ts_scale);
            max.set_index(stats_idx, max_value, colchunk.meta_data.type, ts_scale);
            // Check the nullability of this column chunk
            if (has_is_null_operator) {
              if (colchunk.meta_data.statistics.null_count.has_value()) {
                auto const& null_count = colchunk.meta_data.statistics.null_count.value();
                if (null_count == 0) {
                  is_null->val[stats_idx] = false;
                } else if (null_count < colchunk.meta_data.num_values) {
                  is_null->set_index(stats_idx, std::nullopt, {});
                } else if (null_count == colchunk.meta_data.num_values) {
                  is_null->val[stats_idx] = true;
                } else {
                  CUDF_FAIL("Invalid null count");
                }
              }
            }
          } else {
            // Marking it null, if column present in row group
            min.set_index(stats_idx, std::nullopt, {});
            max.set_index(stats_idx, std::nullopt, {});
            if (has_is_null_operator) { is_null->set_index(stats_idx, std::nullopt, {}); }
          }
          stats_idx++;
        }
      };
      return {min.to_device(dtype, stream, mr),
              max.to_device(dtype, stream, mr),
              has_is_null_operator ? std::make_optional(is_null->to_device(
                                       data_type{cudf::type_id::BOOL8}, stream, mr))
                                   : std::nullopt};
    }
  }
};

}  // namespace

bool aggregate_reader_metadata::any_row_group_stats_available(
  host_span<std::vector<size_type> const> input_row_group_indices,
  host_span<int const> filter_column_schemas) const
{
  auto const colchunk_has_stats = [](ColumnChunk const& colchunk) {
    auto const& stats = colchunk.meta_data.statistics;
    return stats.min_value.has_value() or stats.max_value.has_value() or stats.min.has_value() or
           stats.max.has_value() or stats.null_count.has_value();
  };

  // Iterate columns on the outside so a single offset caches the chunk position across sources
  for (auto const schema_idx : filter_column_schemas) {
    std::optional<size_type> colchunk_offset{std::nullopt};

    for (size_t src_idx = 0; src_idx < input_row_group_indices.size(); ++src_idx) {
      auto const& row_group_indices = input_row_group_indices[src_idx];
      if (row_group_indices.empty()) { continue; }

      auto const& first_row_group =
        per_file_metadata[src_idx].row_groups[row_group_indices.front()];
      auto const mapped_schema_idx = map_schema_index(schema_idx, static_cast<int>(src_idx));
      colchunk_offset =
        find_colchunk_iter_offset(first_row_group, mapped_schema_idx, colchunk_offset);

      if (colchunk_has_stats(first_row_group.columns[colchunk_offset.value()])) { return true; }
    }
  }
  return false;
}

std::optional<std::vector<std::vector<size_type>>> aggregate_reader_metadata::apply_stats_filters(
  host_span<std::vector<size_type> const> input_row_group_indices,
  size_type total_row_groups,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  auto mr = cudf::get_current_device_resource_ref();

  // Get a boolean mask indicating which columns can participate in stats based filtering
  auto const [stats_columns_mask, has_is_null_operator] =
    stats_columns_collector{filter.get(), static_cast<size_type>(output_dtypes.size())}
      .get_stats_columns_mask();

  // Return early if no columns will participate in stats based filtering
  if (stats_columns_mask.empty()) { return std::nullopt; }

  // Scope the stats-availability check to the columns referenced by the filter.
  std::vector<int> filter_column_schemas;
  filter_column_schemas.reserve(output_column_schemas.size());
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  stats_columns_mask.begin(),
                  std::back_inserter(filter_column_schemas),
                  std::identity{});

  // Skip building the stats table if no filter column carries usable row-group statistics.
  if (not any_row_group_stats_available(input_row_group_indices, filter_column_schemas)) {
    return std::nullopt;
  }

  // Converts Column chunk statistics to a table
  // where min(col[i]) = columns[i*2], max(col[i])=columns[i*2+1]
  // For each column, it contains #sources * #column_chunks_per_src rows
  std::vector<std::unique_ptr<column>> columns;
  row_group_stats_caster const stats_col{
    .total_row_groups     = static_cast<size_type>(total_row_groups),
    .per_file_metadata    = per_file_metadata,
    .row_group_indices    = input_row_group_indices,
    .has_is_null_operator = has_is_null_operator};

  for (size_t col_idx = 0; col_idx < output_dtypes.size(); col_idx++) {
    auto const schema_idx = output_column_schemas[col_idx];
    auto const& dtype     = output_dtypes[col_idx];
    // Only participating columns and comparable types are supported
    if (not stats_columns_mask[col_idx] or
        (cudf::is_compound(dtype) && dtype.id() != cudf::type_id::STRING)) {
      // Placeholder for unsupported types and non-participating columns
      columns.push_back(cudf::make_numeric_column(data_type{cudf::type_id::BOOL8},
                                                  total_row_groups,
                                                  rmm::device_buffer{0, stream, mr},
                                                  0,
                                                  stream,
                                                  mr));
      columns.push_back(cudf::make_numeric_column(data_type{cudf::type_id::BOOL8},
                                                  total_row_groups,
                                                  rmm::device_buffer{0, stream, mr},
                                                  0,
                                                  stream,
                                                  mr));
      if (has_is_null_operator) {
        columns.push_back(cudf::make_numeric_column(data_type{cudf::type_id::BOOL8},
                                                    total_row_groups,
                                                    rmm::device_buffer{0, stream, mr},
                                                    0,
                                                    stream,
                                                    mr));
      }
      continue;
    }
    // Map each filter column's zeroth-source schema index into every source's schema tree.
    auto const num_sources         = input_row_group_indices.size();
    auto per_source_schema_indices = std::vector<int>(num_sources);
    auto const src_iter            = cuda::counting_iterator<size_t>{0};
    std::transform(
      src_iter, src_iter + num_sources, per_source_schema_indices.begin(), [&](size_t src_idx) {
        return map_schema_index(schema_idx, static_cast<int>(src_idx));
      });
    auto [min_col, max_col, is_null_col] = cudf::type_dispatcher<dispatch_storage_type>(
      dtype, stats_col, per_source_schema_indices, dtype, stream, mr);
    columns.push_back(std::move(min_col));
    columns.push_back(std::move(max_col));
    if (has_is_null_operator) {
      CUDF_EXPECTS(is_null_col.has_value(), "is_null column must be present");
      columns.push_back(std::move(is_null_col.value()));
    }
  }
  auto stats_table = cudf::table(std::move(columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  stats_expression_converter const stats_expr{
    filter.get(), static_cast<size_type>(output_dtypes.size()), has_is_null_operator, stream};

  // Filter stats table with StatsAST expression and collect filtered row group indices
  return collect_filtered_row_group_indices(
    stats_table, stats_expr.get_stats_expr(), input_row_group_indices, stream);
}

std::pair<std::optional<std::vector<std::vector<size_type>>>, surviving_row_group_metrics>
aggregate_reader_metadata::filter_row_groups(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> input_row_group_indices,
  size_type total_row_groups,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Apply stats filtering on input row groups
  auto const stats_filtered_row_groups = apply_stats_filters(input_row_group_indices,
                                                             total_row_groups,
                                                             output_dtypes,
                                                             output_column_schemas,
                                                             filter,
                                                             stream);

  // Number of surviving row groups after applying stats filter
  auto const num_stats_filtered_row_groups =
    stats_filtered_row_groups.has_value()
      ? std::accumulate(stats_filtered_row_groups.value().cbegin(),
                        stats_filtered_row_groups.value().cend(),
                        size_type{0},
                        [](auto sum, auto const& per_file_row_groups) {
                          return sum + per_file_row_groups.size();
                        })
      : total_row_groups;

  // Span of row groups to apply bloom filtering on.
  auto const bloom_filter_input_row_groups =
    stats_filtered_row_groups.has_value()
      ? host_span<std::vector<size_type> const>(stats_filtered_row_groups.value())
      : input_row_group_indices;

  // Collect equality literals for each input table column for bloom filtering
  auto const equality_literals =
    equality_literals_collector{
      filter.get(), output_dtypes, output_column_schemas, per_file_metadata[0].schema}
      .get_literals();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> equality_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  equality_literals.begin(),
                  std::back_inserter(equality_col_schemas),
                  [](auto& eq_literals) { return not eq_literals.empty(); });

  // Return early if no column with equality predicate(s)
  if (equality_col_schemas.empty()) {
    return {stats_filtered_row_groups,
            {std::make_optional(num_stats_filtered_row_groups), std::nullopt}};
  }

  // Read a vector of bloom filter bitset device buffers for all columns with equality
  // predicate(s) across all row groups
  auto const [bloom_filter_buffers, bloom_filter_data] =
    read_bloom_filters(sources,
                       bloom_filter_input_row_groups,
                       equality_col_schemas,
                       num_stats_filtered_row_groups,
                       stream,
                       cudf::get_current_device_resource_ref());

  // No bloom filters, return early
  if (bloom_filter_data.empty()) {
    return {stats_filtered_row_groups,
            {std::make_optional(num_stats_filtered_row_groups), std::nullopt}};
  }

  // Apply bloom filtering on the output row groups from stats filter
  auto const bloom_filtered_row_groups = apply_bloom_filters(bloom_filter_data,
                                                             bloom_filter_input_row_groups,
                                                             equality_literals,
                                                             num_stats_filtered_row_groups,
                                                             output_dtypes,
                                                             equality_col_schemas,
                                                             filter,
                                                             stream);

  // Number of surviving row groups after applying bloom filter
  auto const num_bloom_filtered_row_groups =
    bloom_filtered_row_groups.has_value()
      ? std::accumulate(bloom_filtered_row_groups.value().cbegin(),
                        bloom_filtered_row_groups.value().cend(),
                        size_type{0},
                        [](auto sum, auto const& per_file_row_groups) {
                          return sum + per_file_row_groups.size();
                        })
      : num_stats_filtered_row_groups;

  // Return bloom filtered row group indices iff collected
  return {
    bloom_filtered_row_groups.has_value() ? bloom_filtered_row_groups : stats_filtered_row_groups,
    {std::make_optional(num_stats_filtered_row_groups),
     std::make_optional(num_bloom_filtered_row_groups)}};
}

}  // namespace cudf::io::parquet::detail
