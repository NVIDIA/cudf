/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/strings/string_view.hpp>

#include <nvbench/nvbench.cuh>

namespace {

template <typename T>
void calculate_bandwidth(nvbench::state& state)
{
  auto const n_rows       = static_cast<cudf::size_type>(state.get_int64("rows"));
  auto const n_cols       = static_cast<cudf::size_type>(state.get_int64("columns"));
  auto const percent_true = static_cast<double>(state.get_int64("hits_%"));
  auto const is_retention = state.get_string("mask_kind") == "retention";

  auto const fraction = is_retention ? percent_true / 100.0 : (100.0 - percent_true) / 100.0;
  cudf::size_type const output_size = fraction * n_rows;
  int64_t const mask_size = sizeof(bool) * n_rows + cudf::bitmask_allocation_size_bytes(n_rows);
  int64_t const validity_bytes_in =
    (fraction >= 1.0 / 32) ? cudf::bitmask_allocation_size_bytes(n_rows) : 4 * output_size;
  int64_t const validity_bytes_out = cudf::bitmask_allocation_size_bytes(output_size);
  int64_t const column_bytes_out   = sizeof(T) * output_size;
  int64_t const column_bytes_in    = column_bytes_out;  // we only read unmasked inputs

  int64_t const bytes_read = (column_bytes_in + validity_bytes_in) * n_cols +  // reading columns
                             mask_size;  // reading boolean mask
  int64_t const bytes_written =
    (column_bytes_out + validity_bytes_out) * n_cols;  // writing columns

  state.add_element_count(n_rows * n_cols);
  state.add_global_memory_reads<nvbench::int8_t>(bytes_read);
  state.add_global_memory_writes<nvbench::int8_t>(bytes_written);
}

}  // namespace

template <typename DataType>
void apply_mask_benchmark(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const n_rows       = static_cast<cudf::size_type>(state.get_int64("rows"));
  auto const n_cols       = static_cast<cudf::size_type>(state.get_int64("columns"));
  auto const percent_true = static_cast<cudf::size_type>(state.get_int64("hits_%"));
  auto const is_retention = state.get_string("mask_kind") == "retention";

  auto const input_type = cudf::type_to_id<DataType>();
  data_profile profile  = data_profile_builder().cardinality(0).no_validity().distribution(
    input_type, distribution_id::UNIFORM, 0, 20);

  auto source_table = create_random_table(
    cycle_dtypes({input_type, cudf::type_id::STRING}, n_cols), row_count{n_rows}, profile);

  profile.set_bool_probability_true(static_cast<double>(percent_true) / 100.0);
  profile.set_null_probability(std::nullopt);  // no null mask
  auto mask = create_random_column(cudf::type_id::BOOL8, row_count{n_rows}, profile);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  calculate_bandwidth<DataType>(state);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&source_table, &mask, is_retention](nvbench::launch& launch) {
               if (is_retention) {
                 cudf::apply_boolean_mask(*source_table, mask->view());
               } else {
                 cudf::apply_deletion_mask(*source_table, mask->view());
               }
             });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  set_throughputs(state);
}

using data_type = nvbench::type_list<int32_t, int64_t, double, cudf::string_view>;
NVBENCH_BENCH_TYPES(apply_mask_benchmark, NVBENCH_TYPE_AXES(data_type))
  .set_name("apply_mask")
  .set_type_axes_names({"type"})
  .add_string_axis("mask_kind", {"retention", "deletion"})
  .add_int64_axis("columns", {1, 4, 9})
  .add_int64_axis("rows", {100'000, 1'000'000, 10'000'000})
  .add_int64_axis("hits_%", {10, 20, 50, 80, 90, 100});
