/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/edit_distance.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

static void bench_edit_distance_utf8(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const strings_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  auto input1 = strings_table->get_column(0);
  auto input2 = strings_table->get_column(0);
  auto sv1    = cudf::strings_column_view(input1.view());
  auto sv2    = cudf::strings_column_view(input2.view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.add_global_memory_reads<nvbench::int8_t>(input1.alloc_size() + input2.alloc_size());
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = nvtext::edit_distance(sv1, sv2); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_edit_distance_ascii(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  auto const max_size     = static_cast<int64_t>(num_rows) * static_cast<int64_t>(max_width);
  auto const offsets_type = max_size >= std::numeric_limits<cudf::size_type>::max()
                              ? cudf::type_id::INT64
                              : cudf::type_id::INT32;

  data_profile profile = data_profile_builder().no_validity().cardinality(0).distribution(
    offsets_type, distribution_id::NORMAL, min_width, max_width);
  data_profile ascii_profile = data_profile_builder().no_validity().cardinality(0).distribution(
    cudf::type_id::INT8, distribution_id::UNIFORM, 32, 126);  // nice ASCII range

  auto offsets    = create_random_column(offsets_type, row_count{num_rows + 1}, profile);
  offsets         = cudf::scan(offsets->view(),
                       *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                       cudf::scan_type::EXCLUSIVE);
  auto chars_size = offsets_type == cudf::type_id::INT64
                      ? dynamic_cast<cudf::numeric_scalar<int64_t>*>(
                          cudf::get_element(offsets->view(), num_rows).get())
                          ->value()
                      : static_cast<int64_t>(dynamic_cast<cudf::numeric_scalar<int32_t>*>(
                                               cudf::get_element(offsets->view(), num_rows).get())
                                               ->value());
  if (chars_size > std::numeric_limits<int32_t>::max()) {
    // to be fixed with create_ascii_string_column utility in PR 20354
    state.skip("chars size too large for this benchmark");
    return;
  }

  auto ascii_data1 = create_random_column(
    cudf::type_id::INT8, row_count{static_cast<cudf::size_type>(chars_size)}, ascii_profile);
  auto ascii_data2 = create_random_column(
    cudf::type_id::INT8, row_count{static_cast<cudf::size_type>(chars_size)}, ascii_profile);

  auto input1 = cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                  num_rows,
                                  ascii_data1->view().data<char>(),
                                  nullptr,
                                  0,
                                  0,
                                  {offsets->view()});
  auto input2 = cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                  num_rows,
                                  ascii_data2->view().data<char>(),
                                  nullptr,
                                  0,
                                  0,
                                  {offsets->view()});

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto sv1          = cudf::strings_column_view(input1);
  auto sv2          = cudf::strings_column_view(input2);
  auto offsets_size = offsets->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(chars_size + 2 * offsets_size);
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = nvtext::edit_distance(sv1, sv2); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_edit_distance(nvbench::state& state)
{
  auto const encode = state.get_string("encode");
  if (encode == "ascii") {
    bench_edit_distance_ascii(state);
  } else {
    bench_edit_distance_utf8(state);
  }
}

NVBENCH_BENCH(bench_edit_distance)
  .set_name("edit_distance")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256, 512})
  .add_int64_axis("num_rows", {262144, 524288, 1048576})
  .add_string_axis("encode", {"utf8", "ascii"});
