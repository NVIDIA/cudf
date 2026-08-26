/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_upper_bound_column(nvbench::state& state)
{
  auto const column_size = static_cast<cudf::size_type>(state.get_int64("column_size"));
  auto const nulls       = state.get_float64("nulls");
  auto const values_size = column_size;

  auto init_data  = cudf::make_fixed_width_scalar<float>(static_cast<float>(0));
  auto init_value = cudf::make_fixed_width_scalar<float>(static_cast<float>(values_size));
  auto step       = cudf::make_fixed_width_scalar<float>(static_cast<float>(-1));
  auto column     = cudf::sequence(column_size, *init_data);
  auto values     = cudf::sequence(values_size, *init_value, *step);

  // disable null bitmask if probability is exactly 0.0
  bool const no_nulls = nulls == 0.0;
  if (!no_nulls) {
    auto [column_null_mask, column_null_count] = create_random_null_mask(column->size(), nulls, 1);
    column->set_null_mask(std::move(column_null_mask), column_null_count);
    auto [values_null_mask, values_null_count] = create_random_null_mask(values->size(), nulls, 2);
    values->set_null_mask(std::move(values_null_mask), values_null_count);
  }

  auto data_table = cudf::sort(cudf::table_view({*column}));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Add memory bandwidth tracking
  state.add_element_count(column_size);
  state.add_global_memory_reads<float>(column_size);             // reading column data
  state.add_global_memory_reads<float>(values_size);             // reading values data
  state.add_global_memory_writes<cudf::size_type>(column_size);  // writing result indices
  if (!no_nulls) {
    state.add_global_memory_reads<nvbench::int8_t>(
      2L * cudf::bitmask_allocation_size_bytes(column_size));
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::upper_bound(data_table->view(),
                                    cudf::table_view({*values}),
                                    {cudf::order::ASCENDING},
                                    {cudf::null_order::BEFORE});
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_upper_bound_column)
  .set_name("upper_bound_column")
  .add_int64_axis("column_size", {1000, 10000, 100000, 1000000, 10000000, 100000000})
  .add_float64_axis("nulls", {0.0, 0.1});

static void bench_lower_bound_table(nvbench::state& state)
{
  using Type = float;

  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const column_size = static_cast<cudf::size_type>(state.get_int64("column_size"));
  auto const values_size = column_size;

  data_profile profile = data_profile_builder().cardinality(0).null_probability(0.1).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);
  auto data_table = create_random_table(
    cycle_dtypes({cudf::type_to_id<Type>()}, num_columns), row_count{column_size}, profile);
  auto values_table = create_random_table(
    cycle_dtypes({cudf::type_to_id<Type>()}, num_columns), row_count{values_size}, profile);

  std::vector<cudf::order> orders(num_columns, cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_orders(num_columns, cudf::null_order::BEFORE);
  auto sorted = cudf::sort(*data_table);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Add memory bandwidth tracking
  state.add_element_count(column_size * num_columns);
  state.add_global_memory_reads<Type>(column_size * num_columns);  // reading data table
  state.add_global_memory_reads<Type>(values_size * num_columns);  // reading values table
  state.add_global_memory_writes<cudf::size_type>(column_size);    // writing result indices
  // Add bitmask reads for null handling
  state.add_global_memory_reads<nvbench::int8_t>(2L * num_columns *
                                                 cudf::bitmask_allocation_size_bytes(column_size));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::lower_bound(sorted->view(), *values_table, orders, null_orders);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_lower_bound_table)
  .set_name("lower_bound_table")
  .add_int64_axis("num_columns", {1, 2, 4, 8})
  .add_int64_axis("column_size", {1000, 10000, 100000, 1000000, 10000000, 100000000});

static void bench_contains(nvbench::state& state)
{
  auto const column_size = static_cast<cudf::size_type>(state.get_int64("column_size"));
  auto const nulls       = state.get_float64("nulls");
  auto const values_size = column_size;

  auto init_data  = cudf::make_fixed_width_scalar<float>(static_cast<float>(0));
  auto init_value = cudf::make_fixed_width_scalar<float>(static_cast<float>(values_size));
  auto step       = cudf::make_fixed_width_scalar<float>(static_cast<float>(-1));
  auto column     = cudf::sequence(column_size, *init_data);
  auto values     = cudf::sequence(values_size, *init_value, *step);

  // disable null bitmask if probability is exactly 0.0
  bool const no_nulls = nulls == 0.0;
  if (!no_nulls) {
    auto [column_null_mask, column_null_count] = create_random_null_mask(column->size(), nulls, 1);
    column->set_null_mask(std::move(column_null_mask), column_null_count);
    auto [values_null_mask, values_null_count] = create_random_null_mask(values->size(), nulls, 2);
    values->set_null_mask(std::move(values_null_mask), values_null_count);
  }

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Add memory bandwidth tracking
  state.add_element_count(column_size);
  state.add_global_memory_reads<float>(column_size);  // reading column data
  state.add_global_memory_reads<float>(values_size);  // reading values data
  state.add_global_memory_writes<bool>(column_size);  // writing result boolean column
  if (!no_nulls) {
    state.add_global_memory_reads<nvbench::int8_t>(
      2L * cudf::bitmask_allocation_size_bytes(column_size));
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::contains(*column, *values); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_contains)
  .set_name("contains")
  .add_int64_axis("column_size", {1000, 10000, 100000, 1000000, 10000000, 100000000})
  .add_float64_axis("nulls", {0.0, 0.1});
