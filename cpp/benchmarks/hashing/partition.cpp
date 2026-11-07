/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Append random columns using consecutive deterministic seeds.
 *
 * Each column uses the current seed, then advances it before generating the next column.
 *
 * @param columns Column vector to append to
 * @param types Column types to append
 * @param num_rows Number of rows in each column
 * @param profile Random data configuration
 * @param seed Seed for the next column
 */
void append_random_columns(std::vector<std::unique_ptr<cudf::column>>& columns,
                           std::span<cudf::type_id const> types,
                           cudf::size_type num_rows,
                           data_profile const& profile,
                           std::uint32_t& seed)
{
  for (auto const type : types) {
    columns.push_back(create_random_column(type, row_count{num_rows}, profile, seed++));
  }
}

/**
 * @brief Create a table with key columns followed by payload columns.
 *
 * Key cardinality is capped at the row count. Payload values use the full distribution for their
 * type. Every column consumes the next deterministic seed.
 *
 * @param key_types Key column types
 * @param payload_types Payload column types
 * @param num_rows Number of table rows
 * @param key_null_probability Probability that a key value is null, or `std::nullopt` to omit the
 * validity mask
 * @param payload_null_probability Probability that a payload value is null, or `std::nullopt` to
 * omit the validity mask
 * @return Generated input table
 */
std::unique_ptr<cudf::table> make_input_table(
  std::span<cudf::type_id const> key_types,
  std::span<cudf::type_id const> payload_types,
  cudf::size_type num_rows,
  std::optional<double> key_null_probability     = std::nullopt,
  std::optional<double> payload_null_probability = std::nullopt)
{
  data_profile const key_profile =
    data_profile_builder().cardinality(num_rows).avg_run_length(1).null_probability(
      key_null_probability);
  data_profile const payload_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(
      payload_null_probability);
  auto columns = std::vector<std::unique_ptr<cudf::column>>{};
  columns.reserve(key_types.size() + payload_types.size());

  auto seed = std::uint32_t{1234};
  append_random_columns(columns, key_types, num_rows, key_profile, seed);
  append_random_columns(columns, payload_types, num_rows, payload_profile, seed);
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Create a key column containing a requested fraction of one hot value.
 *
 * Non-hot rows contain high-cardinality random values. A deterministic random Boolean mask
 * distributes the hot value throughout the column instead of creating long runs.
 *
 * @param num_rows Number of rows
 * @param hot_probability Probability that a row contains the hot value
 * @param seed Seed for the next generated column
 * @return Generated INT64 key column
 */
std::unique_ptr<cudf::column> make_hot_key_column(cudf::size_type num_rows,
                                                  double hot_probability,
                                                  std::uint32_t& seed)
{
  data_profile const tail_profile =
    data_profile_builder().cardinality(num_rows).avg_run_length(1).no_validity();
  auto tail = create_random_column(cudf::type_id::INT64, row_count{num_rows}, tail_profile, seed++);

  data_profile const mask_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).no_validity().bool_probability_true(
      hot_probability);
  auto mask = create_random_column(cudf::type_id::BOOL8, row_count{num_rows}, mask_profile, seed++);

  auto const hot_key = cudf::numeric_scalar<std::int64_t>{0};
  return cudf::copy_if_else(hot_key, tail->view(), mask->view());
}

/**
 * @brief Read an integer benchmark axis after checking that it fits in `cudf::size_type`.
 *
 * @param state Benchmark state
 * @param axis_name Name of the integer axis
 * @return Axis value converted to `cudf::size_type`
 */
cudf::size_type get_size_type_axis(nvbench::state& state, std::string const& axis_name)
{
  auto const value = state.get_int64(axis_name);
  CUDF_EXPECTS(std::in_range<cudf::size_type>(value),
               axis_name + " value " + std::to_string(value) + " does not fit in cudf::size_type",
               std::out_of_range);
  return static_cast<cudf::size_type>(value);
}

/**
 * @brief Measure hash partitioning and record benchmark metrics.
 *
 * The timed region contains only the call to cudf::hash_partition. The benchmark also records
 * row count, buffer sizes, estimated memory traffic, and peak memory use.
 *
 * @param state Benchmark state
 * @param input Table to partition
 * @param num_keys Number of leading columns to hash
 * @param num_partitions Number of output partitions
 */
void run_hash_partition(nvbench::state& state,
                        cudf::table const& input,
                        cudf::size_type num_keys,
                        cudf::size_type num_partitions)
{
  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto const input_view  = input.view();
  auto const key_begin   = cuda::counting_iterator<cudf::size_type>{0};
  auto const keys        = input_view.select(key_begin, key_begin + num_keys);
  auto const input_bytes = static_cast<std::int64_t>(input.alloc_size());
  auto const key_bytes =
    std::accumulate(key_begin, key_begin + num_keys, std::int64_t{0}, [&](auto bytes, auto key) {
      return bytes + static_cast<std::int64_t>(input.get_column(key).alloc_size());
    });
  auto const output_bytes = input_bytes;
  auto const offset_bytes =
    (static_cast<std::int64_t>(num_partitions) + 1) * sizeof(cudf::size_type);

  state.add_element_count(input.num_rows(), "rows");
  state.add_buffer_size(input_bytes, "input_size", "input_size");
  state.add_buffer_size(key_bytes, "key_size", "key_size");
  state.add_buffer_size(output_bytes, "output_size", "output_size");
  state.add_buffer_size(offset_bytes, "offset_size", "offset_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes + key_bytes);
  state.add_global_memory_writes<nvbench::int8_t>(output_bytes + offset_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto output = cudf::hash_partition(input_view, keys, num_partitions);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

/**
 * @brief Measure hash partitioning across common analytic table shapes.
 *
 * The benchmark varies key type, key count, payload column count, row count, and partition count
 * independently.
 *
 * @tparam Key Key element type
 * @param state Benchmark state
 */
template <typename Key>
void bench_hash_partition_analytic(nvbench::state& state, nvbench::type_list<Key>)
{
  auto const num_rows            = get_size_type_axis(state, "num_rows");
  auto const num_keys            = get_size_type_axis(state, "num_key_columns");
  auto const num_payload_columns = get_size_type_axis(state, "num_payload_columns");
  auto const num_partitions      = get_size_type_axis(state, "num_partitions");

  auto input =
    make_input_table(std::vector<cudf::type_id>(num_keys, cudf::type_to_id<Key>()),
                     std::vector<cudf::type_id>(num_payload_columns, cudf::type_id::INT64),
                     num_rows);
  run_hash_partition(state, *input, num_keys, num_partitions);
}

/**
 * @brief Measure partition-count scaling for a fixed table.
 *
 * The input contains one INT64 key, eight INT64 payload columns, and 2^26 rows.
 *
 * @param state Benchmark state
 */
void bench_hash_partition_partition_count(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 26;
  auto const num_partitions = get_size_type_axis(state, "num_partitions");

  auto input = make_input_table(std::array{cudf::type_id::INT64},
                                std::vector<cudf::type_id>(8, cudf::type_id::INT64),
                                num_rows);
  run_hash_partition(state, *input, 1, num_partitions);
}

/**
 * @brief Measure column-count scaling while holding input bytes constant.
 *
 * Each table contains one INT64 key and only INT64 payload columns. The row count changes with the
 * column count to preserve the requested input size.
 *
 * @param state Benchmark state
 */
void bench_hash_partition_equal_input_size(nvbench::state& state)
{
  auto const target_input_bytes = static_cast<std::size_t>(state.get_int64("target_input_bytes"));
  auto const num_columns        = get_size_type_axis(state, "num_columns");
  auto const num_partitions     = get_size_type_axis(state, "num_partitions");
  auto const row_width          = static_cast<std::size_t>(num_columns) * sizeof(std::int64_t);

  if (target_input_bytes % row_width != 0) {
    state.skip("target_input_bytes must be divisible by the row width");
    return;
  }
  auto const num_rows = target_input_bytes / row_width;
  if (num_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("number of rows exceeds cudf::size_type");
    return;
  }
  auto input = make_input_table(std::array{cudf::type_id::INT64},
                                std::vector<cudf::type_id>(num_columns - 1, cudf::type_id::INT64),
                                static_cast<cudf::size_type>(num_rows));
  run_hash_partition(state, *input, 1, num_partitions);
}

/**
 * @brief Measure hash partitioning with low-cardinality and hot keys.
 *
 * The key contains at most 16 uniformly distributed values, or a hot value selected with 50% or
 * 90% probability. Each input also contains eight INT64 payload columns.
 *
 * @param state Benchmark state
 */
void bench_hash_partition_key_skew(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 23;
  auto const distribution   = state.get_string("distribution");
  auto const num_partitions = get_size_type_axis(state, "num_partitions");

  auto columns = std::vector<std::unique_ptr<cudf::column>>{};
  columns.reserve(9);
  auto seed = std::uint32_t{1234};
  if (distribution == "uniform_16_values") {
    data_profile const profile =
      data_profile_builder().cardinality(16).avg_run_length(1).no_validity();
    columns.push_back(
      create_random_column(cudf::type_id::INT64, row_count{num_rows}, profile, seed++));
  } else if (distribution == "hot_50_percent") {
    columns.push_back(make_hot_key_column(num_rows, 0.5, seed));
  } else if (distribution == "hot_90_percent") {
    columns.push_back(make_hot_key_column(num_rows, 0.9, seed));
  } else {
    state.skip("Unknown key distribution: " + distribution);
    return;
  }

  data_profile const payload_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).no_validity();
  append_random_columns(
    columns, std::vector<cudf::type_id>(8, cudf::type_id::INT64), num_rows, payload_profile, seed);
  auto input = std::make_unique<cudf::table>(std::move(columns));
  run_hash_partition(state, *input, 1, num_partitions);
}

/**
 * @brief Measure hash partitioning when keys, payload columns, or both contain nulls.
 *
 * Each selected column generates nulls with 10% probability.
 *
 * @param state Benchmark state
 */
void bench_hash_partition_nullability(nvbench::state& state)
{
  auto const num_rows               = cudf::size_type{1} << 23;
  auto const nullable               = state.get_string("nullable");
  auto const num_partitions         = get_size_type_axis(state, "num_partitions");
  constexpr double null_probability = 0.1;

  auto key_nulls     = std::optional<double>{};
  auto payload_nulls = std::optional<double>{};
  if (nullable == "keys") {
    key_nulls = null_probability;
  } else if (nullable == "payload") {
    payload_nulls = null_probability;
  } else if (nullable == "keys_and_payload") {
    key_nulls     = null_probability;
    payload_nulls = null_probability;
  } else {
    state.skip("Unknown nullability config: " + nullable);
    return;
  }

  auto input = make_input_table(std::array{cudf::type_id::INT64},
                                std::vector<cudf::type_id>(8, cudf::type_id::INT64),
                                num_rows,
                                key_nulls,
                                payload_nulls);
  run_hash_partition(state, *input, 1, num_partitions);
}

/**
 * @brief Measure fixed-width and string table layouts.
 *
 * The layouts cover heterogeneous fixed-width payloads, a string payload mixed with fixed-width
 * columns, a string key, and a composite INT64 and string key.
 *
 * @param state Benchmark state
 */
void bench_hash_partition_type_mix(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 23;
  auto const layout         = state.get_string("layout");
  auto const num_partitions = get_size_type_axis(state, "num_partitions");

  auto key_types     = std::vector<cudf::type_id>{};
  auto payload_types = std::vector<cudf::type_id>{};
  if (layout == "heterogeneous_fixed_payload") {
    key_types     = {cudf::type_id::INT64};
    payload_types = {cudf::type_id::INT8,
                     cudf::type_id::INT16,
                     cudf::type_id::INT32,
                     cudf::type_id::INT64,
                     cudf::type_id::FLOAT32,
                     cudf::type_id::FLOAT64,
                     cudf::type_id::DECIMAL128};
  } else if (layout == "hybrid_string_payload") {
    key_types     = {cudf::type_id::INT64};
    payload_types = {cudf::type_id::INT8,
                     cudf::type_id::INT16,
                     cudf::type_id::INT32,
                     cudf::type_id::INT64,
                     cudf::type_id::FLOAT32,
                     cudf::type_id::FLOAT64,
                     cudf::type_id::DECIMAL128,
                     cudf::type_id::STRING};
  } else if (layout == "string_key") {
    key_types     = {cudf::type_id::STRING};
    payload_types = std::vector<cudf::type_id>(8, cudf::type_id::INT64);
  } else if (layout == "mixed_composite_key") {
    key_types     = {cudf::type_id::INT64, cudf::type_id::STRING};
    payload_types = std::vector<cudf::type_id>(8, cudf::type_id::INT64);
  } else {
    state.skip("Unknown type layout: " + layout);
    return;
  }

  auto input = make_input_table(key_types, payload_types, num_rows);
  run_hash_partition(state, *input, static_cast<cudf::size_type>(key_types.size()), num_partitions);
}

/**
 * @brief Measure wide-table partitioning when every column is a key.
 *
 * This models DISTINCT-style partitioning and increases the hashing work with the column count.
 * It uses 2^21 rows to keep the 256-column input at 4 GiB.
 *
 * @param state Benchmark state
 */
void bench_hash_partition_all_keys_stress(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 21;
  auto const num_columns    = get_size_type_axis(state, "num_columns");
  auto const num_partitions = get_size_type_axis(state, "num_partitions");

  auto const input_bytes = sizeof(std::int64_t) * static_cast<std::size_t>(num_rows) * num_columns;
  CUDF_EXPECTS(input_bytes <= (std::size_t{1} << 32),
               "All-keys stress benchmark input exceeds 4 GiB");

  auto input =
    make_input_table(std::vector<cudf::type_id>(num_columns, cudf::type_id::INT64), {}, num_rows);
  run_hash_partition(state, *input, num_columns, num_partitions);
}

}  // namespace

NVBENCH_BENCH_TYPES(bench_hash_partition_analytic,
                    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, std::int64_t>))
  .set_name("hash_partition_analytic")
  .set_type_axes_names({"key_type"})
  .add_int64_axis("num_rows", {1 << 17, 1 << 21, 1 << 24})
  .add_int64_axis("num_key_columns", {1, 2, 3})
  .add_int64_axis("num_payload_columns", {1, 8, 16})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_partition_count)
  .set_name("hash_partition_partition_count")
  .add_int64_axis("num_partitions",
                  {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 27648, 28672, 65536});

NVBENCH_BENCH(bench_hash_partition_equal_input_size)
  .set_name("hash_partition_equal_input_size")
  .add_int64_axis("target_input_bytes",
                  {std::int64_t{256} << 20, std::int64_t{1} << 30, std::int64_t{4} << 30})
  .add_int64_axis("num_columns", {2, 8, 16, 256})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_key_skew)
  .set_name("hash_partition_key_skew")
  .add_string_axis("distribution", {"uniform_16_values", "hot_50_percent", "hot_90_percent"})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_nullability)
  .set_name("hash_partition_nullability")
  .add_string_axis("nullable", {"keys", "payload", "keys_and_payload"})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_type_mix)
  .set_name("hash_partition_type_mix")
  .add_string_axis(
    "layout",
    {"heterogeneous_fixed_payload", "hybrid_string_payload", "string_key", "mixed_composite_key"})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_all_keys_stress)
  .set_name("hash_partition_all_keys_stress")
  .add_int64_axis("num_columns", {2, 8, 16, 256})
  .add_int64_axis("num_partitions", {256, 1024});
