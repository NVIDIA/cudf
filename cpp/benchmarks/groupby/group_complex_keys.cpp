/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

namespace {

auto generate_int_keys(cudf::size_type num_cols,
                       cudf::size_type num_rows,
                       cudf::size_type value_key_ratio,
                       double null_probability)
{
  auto const create_column = [&] {
    auto builder =
      data_profile_builder()
        .cardinality(num_rows / value_key_ratio)
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    if (null_probability > 0) {
      builder.null_probability(null_probability);
    } else {
      builder.no_validity();
    }
    return create_random_column(
      cudf::type_to_id<int32_t>(), row_count{num_rows}, data_profile{builder});
  };
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(num_cols);
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    cols.emplace_back(create_column());
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

auto generate_mixed_types_keys(cudf::size_type num_cols,
                               cudf::size_type num_rows,
                               cudf::size_type value_key_ratio,
                               double null_probability)
{
  constexpr auto max_str_length = 50;
  constexpr auto max_list_size  = 10;
  constexpr auto nested_depth   = 2;

  auto builder = data_profile_builder()
                   .cardinality(num_rows / value_key_ratio)
                   .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, num_rows)
                   .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length)
                   .distribution(cudf::type_id::INT64, distribution_id::UNIFORM, 0, num_rows)
                   .distribution(cudf::type_id::LIST, distribution_id::UNIFORM, 0, max_list_size)
                   .list_depth(nested_depth)
                   .struct_depth(nested_depth);
  if (null_probability > 0) {
    builder.null_probability(null_probability);
  } else {
    builder.no_validity();
  }

  return create_random_table(cycle_dtypes({cudf::type_id::INT32,
                                           cudf::type_id::STRING,
                                           cudf::type_id::INT64,
                                           cudf::type_id::LIST,
                                           cudf::type_id::STRUCT},
                                          num_cols),
                             row_count{num_rows},
                             data_profile{builder});
}

auto generate_vals(cudf::size_type num_rows, double null_probability)
{
  using Type   = int64_t;
  auto builder = data_profile_builder().cardinality(0).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows);
  if (null_probability > 0) {
    builder.null_probability(null_probability);
  } else {
    builder.no_validity();
  }
  return create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, data_profile{builder});
}

template <bool is_int_keys>
void run_benchmark_complex_keys(nvbench::state& state)
{
  auto const n_cols           = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const n_rows           = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const value_key_ratio  = static_cast<cudf::size_type>(state.get_int64("value_key_ratio"));
  auto const null_probability = state.get_float64("null_probability");

  auto const keys_table = [&] {
    if constexpr (is_int_keys) {
      return generate_int_keys(n_cols, n_rows, value_key_ratio, null_probability);
    } else {
      return generate_mixed_types_keys(n_cols, n_rows, value_key_ratio, null_probability);
    }
  }();
  auto const vals = generate_vals(n_rows, null_probability);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  auto const stream           = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::groupby::groupby gb_obj(keys_table->view());
    [[maybe_unused]] auto const result = gb_obj.aggregate(requests, stream);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

void bench_groupby_int_keys(nvbench::state& state) { run_benchmark_complex_keys<true>(state); }
void bench_groupby_mixed_types_keys(nvbench::state& state)
{
  run_benchmark_complex_keys<false>(state);
}

NVBENCH_BENCH(bench_groupby_int_keys)
  .set_name("complex_int_keys")
  .add_int64_axis("num_cols", {1, 2, 4, 8, 16})
  .add_int64_power_of_two_axis("num_rows", {12, 18, 24})
  .add_int64_axis("value_key_ratio", {20, 200})
  .add_float64_axis("null_probability", {0, 0.5});

NVBENCH_BENCH(bench_groupby_mixed_types_keys)
  .set_name("complex_mixed_keys")
  .add_int64_axis("num_cols", {1, 2, 3, 4, 5})  // Not enough memory for more mixed types columns
  .add_int64_power_of_two_axis("num_rows", {12, 18, 24})
  .add_int64_axis("value_key_ratio", {20, 200})
  .add_float64_axis("null_probability", {0, 0.5});
