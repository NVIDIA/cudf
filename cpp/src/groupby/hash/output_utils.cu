/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "output_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {
namespace {

/**
 * @brief Functor to create the result columns for hash-based groupby aggregations
 *
 * This functor handles the creation of appropriately typed and sized columns for each
 * aggregation, including special handling for SUM_OVERFLOW which requires a struct column.
 * For data types smaller than 4 bytes, the buffer size is adjusted to be a multiple of 4 to
 * ensure memory safety when atomic operations use 4-byte CAS loops to emulate smaller atomics.
 */
struct result_column_creator {
  size_type output_size;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  explicit result_column_creator(size_type output_size_,
                                 cuda::stream_ref stream_,
                                 rmm::device_async_resource_ref mr_)
    : output_size{output_size_}, stream{stream_}, mr{mr_}
  {
  }

  std::unique_ptr<column> operator()(column_view const& col,
                                     aggregation::Kind const& agg,
                                     bool is_agg_intermediate) const
  {
    auto const col_type =
      is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
    auto const nullable = !is_agg_intermediate && agg != aggregation::COUNT_VALID &&
                          agg != aggregation::COUNT_ALL && col.has_nulls();
    // TODO: Remove adjusted buffer size workaround once https://github.com/NVIDIA/cccl/issues/6430
    // is fixed. Use adjusted buffer size for small data types to ensure atomic operation safety.
    auto const make_uninitialized_column = [&](data_type d_type, size_type size, mask_state state) {
      auto const type_size = cudf::size_of(d_type);
      if (type_size < 4) {
        auto adjusted_size    = cudf::util::round_up_safe(size, static_cast<size_type>(4));
        auto buffer           = rmm::device_buffer(adjusted_size * type_size, stream, mr);
        auto mask             = create_null_mask(size, state, stream, mr);
        auto const null_count = state_null_count(state, size);
        return std::make_unique<column>(
          d_type, size, std::move(buffer), std::move(mask), null_count);
      }
      return make_fixed_width_column(d_type, size, state, stream, mr);
    };
    if (agg != aggregation::SUM_OVERFLOW) {
      auto const target_type = cudf::detail::target_type(col_type, agg);
      auto const mask_flag   = nullable ? mask_state::ALL_NULL : mask_state::UNALLOCATED;
      return make_uninitialized_column(target_type, output_size, mask_flag);
    }
    auto make_children = [&make_uninitialized_column, col_type](size_type size) {
      std::vector<std::unique_ptr<column>> children;
      // Create sum child column - no null mask needed, struct-level mask handles nullability
      children.push_back(make_uninitialized_column(col_type, size, mask_state::UNALLOCATED));
      // Create overflow child column (bool) - no null mask needed, only value matters
      children.push_back(
        make_uninitialized_column(data_type{type_id::BOOL8}, size, mask_state::UNALLOCATED));
      return children;
    };

    auto [null_mask, null_count] = [&]() -> std::pair<rmm::device_buffer, size_type> {
      if (output_size > 0 && nullable) {
        return {create_null_mask(output_size, mask_state::ALL_NULL, stream, mr), output_size};
      }
      return {rmm::device_buffer{}, 0};
    }();
    return create_structs_hierarchy(output_size,
                                    make_children(output_size),
                                    null_count,
                                    std::move(null_mask),
                                    stream,
                                    cudf::get_current_device_resource_ref());
  }
};

}  // anonymous namespace

std::unique_ptr<table> create_results_table(size_type output_size,
                                            table_view const& values,
                                            host_span<aggregation::Kind const> agg_kinds,
                                            std::span<int8_t const> is_agg_intermediate,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(agg_kinds.size()),
               "The number of values columns and size of agg_kinds vector must be the same.");
  CUDF_EXPECTS(
    values.num_columns() == static_cast<size_type>(is_agg_intermediate.size()),
    "The number of values columns and size of is_agg_intermediate vector must be the same.");

  auto const column_creator = result_column_creator{output_size, stream, mr};
  std::vector<std::unique_ptr<column>> output_cols;
  for (size_t i = 0; i < agg_kinds.size(); i++) {
    output_cols.emplace_back(
      column_creator(values.column(i), agg_kinds[i], static_cast<bool>(is_agg_intermediate[i])));
  }
  auto result_table = std::make_unique<table>(std::move(output_cols));
  cudf::detail::initialize_with_identity(result_table->mutable_view(), agg_kinds, stream);
  return result_table;
}

void finalize_output(table_view const& values,
                     std::vector<std::unique_ptr<aggregation>> const& aggregations,
                     std::unique_ptr<table>& agg_results,
                     cudf::detail::result_cache* cache,
                     cuda::stream_ref stream)
{
  auto result_cols       = agg_results->release();
  auto const null_counts = [&]() -> std::vector<size_type> {
    auto const has_null_masks = std::any_of(
      result_cols.begin(), result_cols.end(), [](auto const& col) { return col->nullable(); });
    if (!has_null_masks) { return {}; }

    auto null_masks =
      cudf::detail::make_pinned_vector<bitmask_type const*>(aggregations.size(), stream);
    std::ranges::transform(result_cols, null_masks.begin(), [](auto const& result) {
      return result->view().null_mask();
    });
    return cudf::batch_null_count(null_masks, 0, result_cols.front()->size(), stream);
  }();

  for (size_t i = 0; i < aggregations.size(); i++) {
    auto& result = result_cols[i];
    if (result->nullable()) { result->set_null_count(null_counts[i]); }
    cache->add_result(values.column(i), *aggregations[i], std::move(result));
  }
  agg_results.reset();  // to make sure any subsequent use will trigger exception
}

}  // namespace cudf::groupby::detail::hash
