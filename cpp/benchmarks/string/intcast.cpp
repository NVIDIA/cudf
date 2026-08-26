/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/convert/int_cast.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

void bench_intcast(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const from_num = state.get_string("dir") == "from";
  auto const endian   = state.get_string("endian") == "little" ? cudf::strings::endian::LITTLE
                                                               : cudf::strings::endian::BIG;

  data_profile const profile = data_profile_builder()
                                 .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, 8)
                                 .no_validity();
  auto const column    = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const sv        = cudf::strings_column_view(column->view());
  auto const data_type = cudf::data_type{cudf::type_id::INT64};
  auto const numbers   = cudf::strings::cast_to_integer(sv, data_type, endian);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (from_num) {
    state.add_global_memory_reads<int64_t>(num_rows);
    state.add_global_memory_writes<int8_t>(column->alloc_size());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::strings::cast_from_integer(numbers->view(), endian);
    });
  } else {
    state.add_global_memory_reads<int8_t>(column->alloc_size());
    state.add_global_memory_writes<int64_t>(num_rows);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::strings::cast_to_integer(sv, data_type, endian);
    });
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_intcast)
  .set_name("intcast")
  .add_int64_axis("num_rows", {262144, 2097152, 8388608})
  .add_string_axis("endian", {"little", "big"})
  .add_string_axis("dir", {"to", "from"});
