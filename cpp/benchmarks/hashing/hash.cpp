/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/dictionary/encode.hpp>
#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#ifdef CUDF_ENABLE_MURMURHASH3_RTCX_EXPERIMENT
#include "hash/murmurhash3_x86_32_rtcx.hpp"
#endif

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

enum class column_type { UNKNOWN, MIXED, INT32, INT64, DOUBLE, DECIMAL128, STRING, LIST, STRUCT };

constexpr auto column_types = std::to_array<std::pair<column_type, std::string_view>>({
  {column_type::MIXED, "mixed"},
  {column_type::INT32, "int32"},
  {column_type::INT64, "int64"},
  {column_type::DOUBLE, "double"},
  {column_type::DECIMAL128, "decimal128"},
  {column_type::STRING, "string"},
  {column_type::LIST, "list"},
  {column_type::STRUCT, "struct"},
});

[[nodiscard]] constexpr column_type parse_column_type(std::string_view name)
{
  for (auto const& [type, type_name] : column_types) {
    if (type_name == name) { return type; }
  }
  return column_type::UNKNOWN;
}

[[nodiscard]] std::vector<std::string> column_type_names(
  std::initializer_list<column_type> selected_types)
{
  std::vector<std::string> result;
  result.reserve(selected_types.size());
  for (auto const selected_type : selected_types) {
    for (auto const& [type, name] : column_types) {
      if (type == selected_type) {
        result.emplace_back(name);
        break;
      }
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::string> column_type_names()
{
  std::vector<std::string> result;
  result.reserve(column_types.size());
  for (auto const& type : column_types) {
    result.emplace_back(type.second);
  }
  return result;
}

}  // namespace

static void bench_hash(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = state.get_float64("nulls");
  // disable null bitmask if probability is exactly 0.0
  bool const no_nulls  = nulls == 0.0;
  auto const hash_name = state.get_string("hash_name");
  auto const data_type = parse_column_type(state.get_string("data_type"));

  auto builder =
    data_profile_builder().null_probability(no_nulls ? std::nullopt : std::optional<double>{nulls});

  // Column types to hash. `mixed` is the historical default; the rest isolate a single type so
  // that per-type costs, such as the byte-wise decimal128 path, are visible on their own.
  auto const types = [&]() {
    switch (data_type) {
      case column_type::MIXED:
        return cycle_dtypes({cudf::type_id::INT64, cudf::type_id::STRING}, num_cols);
      case column_type::INT64: return cycle_dtypes({cudf::type_id::INT64}, num_cols);
      case column_type::INT32: return cycle_dtypes({cudf::type_id::INT32}, num_cols);
      case column_type::DOUBLE: return cycle_dtypes({cudf::type_id::FLOAT64}, num_cols);
      case column_type::DECIMAL128: return cycle_dtypes({cudf::type_id::DECIMAL128}, num_cols);
      case column_type::STRING: return cycle_dtypes({cudf::type_id::STRING}, num_cols);
      case column_type::LIST:
        builder.list_depth(1).list_type(cudf::type_id::INT64);
        return cycle_dtypes({cudf::type_id::LIST}, num_cols);
      case column_type::STRUCT: {
        auto const struct_types =
          std::vector<cudf::type_id>{cudf::type_id::INT64, cudf::type_id::FLOAT64};
        builder.struct_types(struct_types);
        return cycle_dtypes({cudf::type_id::STRUCT}, num_cols);
      }
      default: return cycle_dtypes({}, 0);
    }
  }();
  if (types.empty()) {
    state.skip(state.get_string("data_type") + ": unknown data type");
    return;
  }

  data_profile const profile = builder;
  auto const data            = create_random_table(types, row_count{num_rows}, profile);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  state.add_global_memory_reads<nvbench::int8_t>(data->alloc_size());
  // memory written depends on used hash

  auto const mem_stats_logger = cudf::memory_stats_logger();

  if (hash_name == "murmurhash3_x86_32") {
    state.add_global_memory_writes<nvbench::uint32_t>(num_rows);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::hashing::murmurhash3_x86_32(data->view());
    });
  } else if (hash_name == "spark_murmurhash3_x86_32") {
    state.add_global_memory_writes<nvbench::uint32_t>(num_rows);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::hashing::spark_murmurhash3_x86_32(data->view());
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
  .add_string_axis("data_type", column_type_names({column_type::MIXED}))
  .add_int64_axis("num_cols", {2, 64})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_string_axis("hash_name",
                   {"murmurhash3_x86_32", "md5", "sha1", "sha224", "sha256", "sha384", "sha512"});

// Register the Spark type sweep separately so the other hashers keep their historical
// mixed INT64/STRING workload.
NVBENCH_BENCH(bench_hash)
  .set_name("spark_hashing")
  .add_int64_axis("num_rows", {65536, 16777216})
  .add_string_axis("data_type", column_type_names())
  .add_int64_axis("num_cols", {2, 64})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_string_axis("hash_name", {"spark_murmurhash3_x86_32"});

NVBENCH_BENCH(bench_hash)
  .set_name("murmurhash_int32")
  .add_int64_axis("num_rows", {65536, 16777216})
  .add_string_axis("data_type", {"int32"})
  .add_int64_axis("num_cols", {1, 8})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_string_axis("hash_name", {"murmurhash3_x86_32"});

static void bench_string_murmurhash3(nvbench::state& state)
{
  auto const use_rtcx = state.get_string("implementation") == "rtcx";
#ifdef CUDF_MURMURHASH3_RTCX_ONLY
  if (!use_rtcx) {
    state.skip("CUB implementation is absent from this measurement build");
    return;
  }
#endif
#ifndef CUDF_ENABLE_MURMURHASH3_RTCX_EXPERIMENT
  if (use_rtcx) {
    state.skip("RTCX build option is disabled");
    return;
  }
#endif
  // Isolate each implementation from both the caller's environment and earlier benchmark states.
  struct scoped_opt_in {
    std::optional<std::string> previous;
    explicit scoped_opt_in(bool enabled)
    {
      if (auto const* value = std::getenv("LIBCUDF_MURMURHASH3_RTCX_ENABLED")) { previous = value; }
      setenv("LIBCUDF_MURMURHASH3_RTCX_ENABLED", enabled ? "ON" : "OFF", 1);
    }
    ~scoped_opt_in()
    {
      if (previous) {
        setenv("LIBCUDF_MURMURHASH3_RTCX_ENABLED", previous->c_str(), 1);
      } else {
        unsetenv("LIBCUDF_MURMURHASH3_RTCX_ENABLED");
      }
    }
  } opt_in{use_rtcx};
  auto const num_rows          = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols          = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const cardinality       = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const max_string_length = static_cast<cudf::size_type>(state.get_int64("max_string_length"));
  auto const nulls             = state.get_float64("nulls");

  data_profile const profile =
    data_profile_builder()
      .cardinality(cardinality)
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, max_string_length)
      .null_probability(nulls == 0.0 ? std::nullopt : std::optional<double>{nulls});
  auto const data =
    create_random_table(std::vector(num_cols, cudf::type_id::STRING), row_count{num_rows}, profile);

#ifdef CUDF_ENABLE_MURMURHASH3_RTCX_EXPERIMENT
  CUDF_EXPECTS(cudf::hashing::detail::murmurhash3_x86_32_rtcx_enabled(data->view()) == use_rtcx,
               "String benchmark selected the wrong implementation");
#endif

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<nvbench::int8_t>(data->alloc_size());
  state.add_global_memory_writes<nvbench::uint32_t>(num_rows);

  // Keep one-time RTCX planner/linker work out of the timed samples.
  auto warmup = cudf::hashing::murmurhash3_x86_32(data->view());
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

#ifdef CUDF_ENABLE_MURMURHASH3_RTCX_EXPERIMENT
  auto const cache_size = cudf::hashing::detail::murmurhash3_x86_32_rtcx_cache_size();
  CUDF_EXPECTS(!use_rtcx || cache_size > 0, "RTCX warmup did not populate the launcher cache");
#endif

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::hashing::murmurhash3_x86_32(data->view());
  });
#ifdef CUDF_ENABLE_MURMURHASH3_RTCX_EXPERIMENT
  CUDF_EXPECTS(cudf::hashing::detail::murmurhash3_x86_32_rtcx_cache_size() == cache_size,
               "Steady-state string samples unexpectedly created another launcher");
#endif
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_string_murmurhash3)
  .set_name("hashing_string")
  .add_string_axis("implementation", {"cub", "rtcx"})
  .add_int64_axis("num_rows", {65536, 16777216})
  .add_int64_axis("num_cols", {1, 8})
  .add_int64_axis("cardinality", {32, 4096})
  .add_int64_axis("max_string_length", {8, 64})
  .add_float64_axis("nulls", {0.0, 0.1});

static void bench_dictionary_string_murmurhash3(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols    = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const nulls       = state.get_float64("nulls");
  bool const no_nulls    = nulls == 0.0;

  data_profile const profile =
    data_profile_builder()
      .cardinality(cardinality)
      .null_probability(no_nulls ? std::nullopt : std::optional<double>{nulls});
  auto strings =
    create_random_table(std::vector(num_cols, cudf::type_id::STRING), row_count{num_rows}, profile);
  auto dictionary_columns = std::vector<std::unique_ptr<cudf::column>>{};
  dictionary_columns.reserve(num_cols);
  for (auto const& column : strings->view()) {
    dictionary_columns.push_back(cudf::dictionary::encode(column));
  }
  auto const data = std::make_unique<cudf::table>(std::move(dictionary_columns));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<nvbench::int8_t>(data->alloc_size());
  state.add_global_memory_writes<nvbench::uint32_t>(num_rows);

  // Keep one-time RTCX planner/linker work out of the timed samples. This is also a warm-up for
  // the legacy path, so the benchmark compares steady-state hash execution.
  auto warmup = cudf::hashing::murmurhash3_x86_32(data->view());
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::hashing::murmurhash3_x86_32(data->view());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_dictionary_string_murmurhash3)
  .set_name("hashing_dictionary_string")
  .add_int64_axis("num_rows", {65536, 16777216})
  .add_int64_axis("num_cols", {1, 8})
  .add_int64_axis("cardinality", {32, 4096})
  .add_float64_axis("nulls", {0.0, 0.1});
