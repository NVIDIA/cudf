/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

// necessary for custom enum types
// see: https://github.com/NVIDIA/nvbench/blob/main/examples/enums.cu
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  cudf::duplicate_keep_option,
  // Callable to generate input strings:
  [](cudf::duplicate_keep_option option) {
    switch (option) {
      case cudf::duplicate_keep_option::KEEP_FIRST: return "KEEP_FIRST";
      case cudf::duplicate_keep_option::KEEP_LAST: return "KEEP_LAST";
      case cudf::duplicate_keep_option::KEEP_NONE: return "KEEP_NONE";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");

template <typename Type, cudf::duplicate_keep_option Keep>
void nvbench_unique(nvbench::state& state, nvbench::type_list<Type, nvbench::enum_type<Keep>>)
{
  // KEEP_FIRST and KEEP_ANY are equivalent for unique
  if constexpr (not std::is_same_v<Type, int32_t> and
                Keep == cudf::duplicate_keep_option::KEEP_ANY) {
    state.skip("Skip unwanted benchmarks.");
  }

  cudf::size_type const num_rows = state.get_int64("NumRows");
  auto const sorting             = state.get_int64("Sort");

  data_profile profile = data_profile_builder().cardinality(0).null_probability(0.01).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows / 100);

  auto source_column = create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, profile);

  auto input_column = source_column->view();
  auto input_table  = cudf::table_view({input_column, input_column, input_column, input_column});

  auto const run_bench = [&](auto const& input) {
    state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::unique(input, {0}, Keep, cudf::null_equality::EQUAL);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  };

  if (sorting) {
    auto const sort_order = cudf::sorted_order(input_table);
    auto const sort_table = cudf::gather(input_table, *sort_order);
    run_bench(*sort_table);
  } else {
    run_bench(input_table);
  }
}

using data_type   = nvbench::type_list<bool, int8_t, int32_t, int64_t, float, cudf::timestamp_ms>;
using keep_option = nvbench::enum_type_list<cudf::duplicate_keep_option::KEEP_FIRST,
                                            cudf::duplicate_keep_option::KEEP_LAST,
                                            cudf::duplicate_keep_option::KEEP_NONE>;

NVBENCH_BENCH_TYPES(nvbench_unique, NVBENCH_TYPE_AXES(data_type, keep_option))
  .set_name("unique")
  .set_type_axes_names({"Type", "KeepOption"})
  .add_int64_axis("NumRows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("Sort", {0, 1});

template <typename Type, cudf::duplicate_keep_option Keep>
void nvbench_unique_list(nvbench::state& state, nvbench::type_list<Type, nvbench::enum_type<Keep>>)
{
  // KEEP_FIRST and KEEP_ANY are equivalent for unique
  if constexpr (Keep == cudf::duplicate_keep_option::KEEP_ANY) {
    state.skip("Skip unwanted benchmarks.");
    return;
  }

  auto const size               = state.get_int64("ColumnSize");
  auto const dtype              = cudf::type_to_id<Type>();
  double const null_probability = state.get_float64("null_probability");
  auto const sorting            = state.get_int64("Sort");

  auto builder = data_profile_builder().null_probability(null_probability);
  if (dtype == cudf::type_id::LIST) {
    builder.distribution(dtype, distribution_id::UNIFORM, 0, 4)
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 4)
      .list_depth(1);
  } else {
    // We're comparing unique() on a non-nested column to that on a list column with the same
    // number of unique rows. The max list size is 4 and the number of unique values in the
    // list's child is 5. So the number of unique rows in the list = 1 + 5 + 5^2 + 5^3 + 5^4 = 781
    // We want this column to also have 781 unique values.
    builder.distribution(dtype, distribution_id::UNIFORM, 0, 781);
  }

  auto const input_table = create_random_table(
    {dtype}, table_size_bytes{static_cast<size_t>(size)}, data_profile{builder}, 0);

  auto const run_bench = [&](auto const& input) {
    state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::unique(input, {0}, Keep, cudf::null_equality::EQUAL);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  };

  if (sorting) {
    auto const sort_order = cudf::sorted_order(*input_table);
    auto const sort_table = cudf::gather(*input_table, *sort_order);
    run_bench(*sort_table);
  } else {
    run_bench(*input_table);
  }
}

NVBENCH_BENCH_TYPES(nvbench_unique_list,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, cudf::list_view>, keep_option))
  .set_name("unique_list")
  .set_type_axes_names({"Type", "KeepOption"})
  .add_float64_axis("null_probability", {0.0, 0.1})
  .add_int64_axis("ColumnSize", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("Sort", {0, 1});
