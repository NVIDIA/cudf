/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

void bench_groupby_struct_keys(nvbench::state& state)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.get_int64("NumRows"))};
  cudf::size_type const n_cols{1};
  cudf::size_type const depth{static_cast<cudf::size_type>(state.get_int64("Depth"))};
  bool const nulls{static_cast<bool>(state.get_int64("Nulls"))};

  // Create columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(n_cols);
  std::generate_n(std::back_inserter(columns), n_cols, [&]() {
    auto const elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    if (!nulls) return column_wrapper(elements, elements + n_rows);
    auto valids =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 10 != 0; });
    return column_wrapper(elements, elements + n_rows, valids);
  });

  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(columns.begin(), columns.end(), std::back_inserter(cols), [](column_wrapper& col) {
    return col.release();
  });

  std::vector<std::unique_ptr<cudf::column>> child_cols = std::move(cols);
  // Add some layers
  for (int i = 0; i < depth; i++) {
    std::vector<bool> struct_validity;
    std::uniform_int_distribution<int> bool_distribution(0, 100 * (i + 1));
    std::generate_n(
      std::back_inserter(struct_validity), n_rows, [&]() { return bool_distribution(generator); });
    cudf::test::structs_column_wrapper struct_col(std::move(child_cols), struct_validity);
    child_cols = std::vector<std::unique_ptr<cudf::column>>{};
    child_cols.push_back(struct_col.release());
  }
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, 100);

  auto const keys_table = cudf::table(std::move(child_cols));
  auto const vals = create_random_column(cudf::type_to_id<int64_t>(), row_count{n_rows}, profile);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());

  // Set up nvbench default stream
  auto const mem_stats_logger = cudf::memory_stats_logger();
  auto stream                 = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(keys_table.view());
    auto const result = gb_obj.aggregate(requests);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_groupby_struct_keys)
  .set_name("groupby_struct_keys")
  .add_int64_power_of_two_axis("NumRows", {10, 16, 20})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});
