/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/stream_compaction/stream_compaction_common.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <limits>

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");

template <typename Type>
void nvbench_distinct(nvbench::state& state, nvbench::type_list<Type>)
{
  cudf::size_type const num_rows    = state.get_int64("NumRows");
  auto const keep                   = get_keep(state.get_string("keep"));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  auto const null_probability       = state.get_float64("null_probability");

  if (cardinality > num_rows) {
    state.skip("cardinality > num_rows");
    return;
  }

  data_profile profile = data_profile_builder()
                           .cardinality(cardinality)
                           .null_probability(null_probability)
                           .distribution(cudf::type_to_id<Type>(),
                                         distribution_id::UNIFORM,
                                         static_cast<Type>(0),
                                         std::numeric_limits<Type>::max());

  auto source_column = create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, profile);

  auto input_column = source_column->view();
  auto input_table  = cudf::table_view({input_column, input_column, input_column, input_column});

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::distinct(
      input_table, {0}, keep, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using data_type = nvbench::type_list<int32_t, int64_t>;

NVBENCH_BENCH_TYPES(nvbench_distinct, NVBENCH_TYPE_AXES(data_type))
  .set_name("distinct")
  .set_type_axes_names({"Type"})
  .add_float64_axis("null_probability", {0.01})
  .add_string_axis("keep", {"any", "first", "last", "none"})
  .add_int64_axis("cardinality", {100, 100'000, 10'000'000, 1'000'000'000})
  .add_int64_axis("NumRows", {100, 100'000, 10'000'000, 1'000'000'000});

template <typename Type>
void nvbench_distinct_list(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const size               = state.get_int64("ColumnSize");
  auto const dtype              = cudf::type_to_id<Type>();
  double const null_probability = state.get_float64("null_probability");
  auto const keep               = get_keep(state.get_string("keep"));

  auto builder = data_profile_builder().null_probability(null_probability);
  if (dtype == cudf::type_id::LIST) {
    builder.distribution(dtype, distribution_id::UNIFORM, 0, 4)
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 4)
      .list_depth(1);
  } else {
    // We're comparing distinct() on a non-nested column to that on a list column with the same
    // number of distinct rows. The max list size is 4 and the number of distinct values in the
    // list's child is 5. So the number of distinct rows in the list = 1 + 5 + 5^2 + 5^3 + 5^4 = 781
    // We want this column to also have 781 distinct values.
    builder.distribution(dtype, distribution_id::UNIFORM, 0, 781);
  }

  auto const table = create_random_table(
    {dtype}, table_size_bytes{static_cast<size_t>(size)}, data_profile{builder}, 0);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result =
      cudf::distinct(*table, {0}, keep, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(nvbench_distinct_list,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, cudf::list_view>))
  .set_name("distinct_list")
  .set_type_axes_names({"Type"})
  .add_string_axis("keep", {"any", "first", "last", "none"})
  .add_float64_axis("null_probability", {0.0, 0.1})
  .add_int64_axis("ColumnSize", {100'000'000});
