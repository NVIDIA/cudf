/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <optional>

static void bench_hash(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = state.get_float64("nulls");
  // disable null bitmask if probability is exactly 0.0
  bool const no_nulls  = nulls == 0.0;
  auto const hash_name = state.get_string("hash_name");

  data_profile const profile =
    data_profile_builder().null_probability(no_nulls ? std::nullopt : std::optional<double>{nulls});
  auto const data =
    create_random_table(cycle_dtypes({cudf::type_id::INT64, cudf::type_id::STRING}, num_cols),
                        row_count{num_rows},
                        profile);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.add_global_memory_reads<nvbench::int8_t>(data->alloc_size());
  // memory written depends on used hash

  auto const mem_stats_logger = cudf::memory_stats_logger();

  if (hash_name == "murmurhash3_x86_32") {
    state.add_global_memory_writes<nvbench::uint32_t>(num_rows);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::hashing::murmurhash3_x86_32(data->view());
    });
  } else if (hash_name == "md5") {
    // md5 creates a 32-byte string
    state.add_global_memory_writes<nvbench::int8_t>(32L * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::md5(data->view()); });
  } else if (hash_name == "sha1") {
    // sha1 creates a 40-byte string
    state.add_global_memory_writes<nvbench::int8_t>(40L * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha1(data->view()); });
  } else if (hash_name == "sha224") {
    // sha224 creates a 56-byte string
    state.add_global_memory_writes<nvbench::int8_t>(56L * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha224(data->view()); });
  } else if (hash_name == "sha256") {
    // sha256 creates a 64-byte string
    state.add_global_memory_writes<nvbench::int8_t>(64L * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha256(data->view()); });
  } else if (hash_name == "sha384") {
    // sha384 creates a 96-byte string
    state.add_global_memory_writes<nvbench::int8_t>(96L * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha384(data->view()); });
  } else if (hash_name == "sha512") {
    // sha512 creates a 128-byte string
    state.add_global_memory_writes<nvbench::int8_t>(128L * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha512(data->view()); });
  } else {
    state.skip(hash_name + ": unknown hash name");
    return;
  }

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_hash)
  .set_name("hashing")
  .add_int64_axis("num_rows", {65536, 16777216})
  .add_int64_axis("num_cols", {2, 64})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_string_axis("hash_name",
                   {"murmurhash3_x86_32", "md5", "sha1", "sha224", "sha256", "sha384", "sha512"});
