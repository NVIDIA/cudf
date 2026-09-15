/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reader_impl_helpers.hpp"
#include "stats_filter_helpers.hpp"
#include "timestamp_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Converts row-group column chunk statistics to device columns.
 *
 * Each output column has one row for every selected row group.
 */
struct row_group_stats_caster : public stats_caster_base {
  using result_type = std::
    tuple<std::unique_ptr<column>, std::unique_ptr<column>, std::optional<std::unique_ptr<column>>>;

  template <typename T>
  [[nodiscard]] static constexpr bool can_use_deprecated_minmax()
  {
    if constexpr (std::is_same_v<T, string_view>) {
      return false;
    } else if constexpr (cudf::is_integral_not_bool<T>()) {
      return cudf::is_signed<T>();
    } else {
      return true;
    }
  }

  size_type total_row_groups;
  std::vector<metadata> const& per_file_metadata;
  host_span<std::vector<size_type> const> row_group_indices;
  bool has_is_null_operator;

  template <typename T>
  result_type operator()(host_span<int const> per_source_schema_indices,
                         cudf::data_type dtype,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(row_group_indices.size() == per_file_metadata.size(),
                 "Row-group indices must match parquet metadata sources",
                 std::invalid_argument);
    CUDF_EXPECTS(per_source_schema_indices.size() == per_file_metadata.size(),
                 "Per-source schema indices must match parquet metadata sources",
                 std::invalid_argument);
    auto const computed_total_row_groups =
      std::accumulate(row_group_indices.begin(),
                      row_group_indices.end(),
                      size_type{0},
                      [](auto count, auto const& source_row_group_indices) {
                        return count + static_cast<size_type>(source_row_group_indices.size());
                      });
    CUDF_EXPECTS(total_row_groups == computed_total_row_groups,
                 "Total row groups must match selected row-group indices",
                 std::invalid_argument);

    if constexpr (cudf::is_compound<T>() and not std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types do not have statistics");
    } else {
      host_column<T> min(total_row_groups, stream);
      host_column<T> max(total_row_groups, stream);
      std::optional<host_column<bool>> is_null;
      if (has_is_null_operator) { is_null = host_column<bool>(total_row_groups, stream); }

      size_type stats_idx = 0;
      for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
        auto const mapped_schema_idx = per_source_schema_indices[src_idx];
        auto const& source_metadata  = per_file_metadata[src_idx];
        CUDF_EXPECTS(mapped_schema_idx >= 0 and
                       static_cast<size_t>(mapped_schema_idx) < source_metadata.schema.size(),
                     "Mapped schema index is out of bounds",
                     std::invalid_argument);
        // Compute timestamp scale factor for precision conversion from the mapped source schema.
        auto const ts_scale = [&] {
          if constexpr (cudf::is_timestamp<T>()) {
            auto const& schema = source_metadata.schema[mapped_schema_idx];
            return calc_timestamp_scale(schema.logical_type, static_cast<int32_t>(T::period::den));
          }
          return 0;
        }();

        for (auto const rg_idx : row_group_indices[src_idx]) {
          CUDF_EXPECTS(
            rg_idx >= 0 and static_cast<size_t>(rg_idx) < source_metadata.row_groups.size(),
            "Row-group index is out of bounds",
            std::invalid_argument);
          auto const& row_group = source_metadata.row_groups[rg_idx];
          auto col              = std::find_if(row_group.columns.begin(),
                                  row_group.columns.end(),
                                  [mapped_schema_idx](ColumnChunk const& col) {
                                    return col.schema_idx == mapped_schema_idx;
                                  });
          if (col != std::end(row_group.columns)) {
            auto const& colchunk = *col;
            auto const& stats    = colchunk.meta_data.statistics;
            // Deprecated min/max use signed physical ordering, so only fall back
            // when that ordering is compatible with the output type.
            auto const* min_value = &stats.min_value;
            auto const* max_value = &stats.max_value;
            if constexpr (can_use_deprecated_minmax<T>()) {
              if (not min_value->has_value()) { min_value = &stats.min; }
              if (not max_value->has_value()) { max_value = &stats.max; }
            }
            // translate binary data to Type then to <T>
            min.set_index(stats_idx, *min_value, colchunk.meta_data.type, ts_scale);
            max.set_index(stats_idx, *max_value, colchunk.meta_data.type, ts_scale);
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
              } else {
                // Statistics without a null count say nothing about this chunk's nullability. The
                // value array is allocated uninitialized and the null mask starts out all valid, so
                // this entry has to be marked null; leaving it alone would let an uninitialized
                // byte be read as an answer.
                is_null->set_index(stats_idx, std::nullopt, {});
              }
            }
          } else {
            // Mark it null if the column chunk is absent from this row group.
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

}  // namespace cudf::io::parquet::detail
