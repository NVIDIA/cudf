/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <optional>
#include <vector>

namespace {

cudf::range_window_type to_bound(int64_t code)
{
  // 0 -> unbounded, 1 -> current_row
  return code == 0 ? cudf::range_window_type{cudf::unbounded{}}
                   : cudf::range_window_type{cudf::current_row{}};
}

}  // namespace

void bench_multi_orderby_range_rolling_sum(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_orderby_cols = static_cast<int>(state.get_int64("num_orderby_cols"));
  auto const has_nulls        = static_cast<bool>(state.get_int64("has_nulls"));
  auto const grouped          = static_cast<bool>(state.get_int64("grouped"));
  auto const preceding        = to_bound(state.get_int64("preceding"));
  auto const following        = to_bound(state.get_int64("following"));

  auto vals = [&] {
    data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<std::int32_t>(), distribution_id::UNIFORM, 0, 100);
    return create_random_column(cudf::type_to_id<std::int32_t>(), row_count{num_rows}, profile);
  }();

  auto group_keys = [&] {
    if (!grouped) { return std::unique_ptr<cudf::table>{nullptr}; }
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, num_rows);
    auto keys =
      create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
    return cudf::sort(cudf::table_view{{keys->view()}});
  }();

  // Build N order-by columns. The columns are co-sorted lexicographically, after the group keys
  // when grouped. Each individual column has lower cardinality than num_rows so peer tuples are
  // common.
  auto orderby_table = [&] {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(num_orderby_cols);
    for (int i = 0; i < num_orderby_cols; ++i) {
      data_profile profile = data_profile_builder().cardinality(0).distribution(
        cudf::type_to_id<cudf::size_type>(),
        distribution_id::UNIFORM,
        0,
        // Decrease range as column index grows so trailing columns produce frequent ties.
        std::max(num_rows / (1 << (i * 4)), 1));
      profile.set_null_probability(has_nulls ? std::optional<double>{200.0 / num_rows}
                                             : std::nullopt);
      cols.push_back(
        create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile));
    }
    // Sort lexicographically with group_keys (if any) as the most-significant prefix so the
    // orderby columns are sorted within each group.
    std::vector<cudf::column_view> sort_keys;
    std::vector<cudf::order> sort_orders;
    std::vector<cudf::null_order> sort_null_orders;
    if (group_keys) {
      sort_keys.push_back(group_keys->get_column(0).view());
      sort_orders.push_back(cudf::order::ASCENDING);
      sort_null_orders.push_back(cudf::null_order::AFTER);
    }
    for (auto const& col : cols) {
      sort_keys.push_back(col->view());
      sort_orders.push_back(cudf::order::ASCENDING);
      sort_null_orders.push_back(cudf::null_order::AFTER);
    }
    std::vector<cudf::column_view> values;
    values.reserve(cols.size());
    for (auto const& col : cols) {
      values.push_back(col->view());
    }
    return cudf::sort_by_key(
      cudf::table_view{values}, cudf::table_view{sort_keys}, sort_orders, sort_null_orders);
  }();

  std::vector<cudf::order> orders(num_orderby_cols, cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_orders(num_orderby_cols, cudf::null_order::AFTER);

  std::vector<cudf::rolling_request> requests;
  requests.push_back({vals->view(), 1, cudf::make_sum_aggregation<cudf::rolling_aggregation>()});

  auto const group_view = group_keys ? group_keys->view() : cudf::table_view{};

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto const result =
      cudf::grouped_range_rolling_window(group_view,
                                         orderby_table->view(),
                                         cudf::host_span<cudf::order const>{orders},
                                         cudf::host_span<cudf::null_order const>{null_orders},
                                         preceding,
                                         following,
                                         cudf::host_span<cudf::rolling_request const>{requests});
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_multi_orderby_range_rolling_sum)
  .set_name("multi_orderby_range_rolling_sum")
  .add_int64_power_of_two_axis("num_rows", {14, 22, 24})
  .add_int64_axis("cardinality", {100, 100'000})
  .add_int64_axis("num_orderby_cols", {2, 3})
  .add_int64_axis("has_nulls", {0, 1})
  .add_int64_axis("preceding", {0, 1})  // unbounded, current_row
  .add_int64_axis("following", {1})     // current_row
  .add_int64_axis("grouped", {1});

NVBENCH_BENCH(bench_multi_orderby_range_rolling_sum)
  .set_name("multi_orderby_range_rolling_sum_ungrouped")
  .add_int64_power_of_two_axis("num_rows", {22, 24})
  .add_int64_axis("cardinality", {0})
  .add_int64_axis("num_orderby_cols", {2, 3})
  .add_int64_axis("has_nulls", {0, 1})
  .add_int64_axis("preceding", {0, 1})
  .add_int64_axis("following", {1})
  .add_int64_axis("grouped", {0});
