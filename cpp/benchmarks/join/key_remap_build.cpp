/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/table_utilities.hpp>
#include <benchmarks/join/join_common.hpp>

#include <cudf/join/key_remapping.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @brief Benchmark for key_remapping build phase with metrics computation.
 *
 * This benchmark isolates the key_remapping construction time (including metrics)
 * to measure performance across various data types and distributions.
 *
 * Cardinality distributions:
 * - ALL_UNIQUE: cardinality = num_rows (all unique keys, approximate)
 * - HIGH_UNIQUE: cardinality = num_rows / 10 (10% unique)
 * - MED_UNIQUE: cardinality = num_rows / 100 (1% unique)
 * - LOW_UNIQUE: cardinality = num_rows / 1000 (0.1% unique)
 * - SINGLE_KEY: cardinality = 1 (all duplicates)
 */

// Cardinality distributions for testing
enum class key_cardinality {
  ALL_UNIQUE,   // 100% unique keys
  HIGH_UNIQUE,  // 10% unique keys
  MED_UNIQUE,   // 1% unique keys
  LOW_UNIQUE,   // 0.1% unique keys
  SINGLE_KEY    // 1 unique key (all duplicates)
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  key_cardinality,
  [](auto value) {
    switch (value) {
      case key_cardinality::ALL_UNIQUE: return "ALL_UNIQUE";
      case key_cardinality::HIGH_UNIQUE: return "HIGH_UNIQUE";
      case key_cardinality::MED_UNIQUE: return "MED_UNIQUE";
      case key_cardinality::LOW_UNIQUE: return "LOW_UNIQUE";
      case key_cardinality::SINGLE_KEY: return "SINGLE_KEY";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

template <bool Nullable, data_type DataType, key_cardinality Card>
void nvbench_key_remap_build(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<Nullable>,
                                                nvbench::enum_type<DataType>,
                                                nvbench::enum_type<Card>>)
{
  auto const num_rows = state.get_int64("num_rows");
  auto const num_keys = state.get_int64("num_keys");
  auto dtypes         = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  // Determine cardinality (number of unique keys) based on distribution
  cudf::size_type cardinality = 0;
  if constexpr (Card == key_cardinality::ALL_UNIQUE) {
    // Request all unique keys. Note: create_random_table cannot guarantee 100% uniqueness
    // without expensive post-processing (distinct + memory reallocation), so there will be
    // some duplicates. This is acceptable for key remapping benchmarks because:
    // 1. Hash table building performance is not catastrophically affected by duplicates
    // 2. The extra work from duplicates is actually realistic (real data has duplicates)
    // 3. Avoiding the deduplication overhead keeps benchmark setup time minimal
    //
    // However, this approach would NOT work for join_on_int32.cu where duplicates cause
    // combinatorial explosion in join output, leading to OOM crashes at large scale.
    cardinality = num_rows;
  } else if constexpr (Card == key_cardinality::HIGH_UNIQUE) {
    cardinality = std::max(cudf::size_type{1}, static_cast<cudf::size_type>(num_rows / 10));
  } else if constexpr (Card == key_cardinality::MED_UNIQUE) {
    cardinality = std::max(cudf::size_type{1}, static_cast<cudf::size_type>(num_rows / 100));
  } else if constexpr (Card == key_cardinality::LOW_UNIQUE) {
    cardinality = std::max(cudf::size_type{1}, static_cast<cudf::size_type>(num_rows / 1000));
  } else if constexpr (Card == key_cardinality::SINGLE_KEY) {
    cardinality = 1;  // All rows have the same key
  }

  // Generate table with controlled cardinality using create_random_table directly
  double const null_probability = Nullable ? 0.3 : 0;
  auto const profile            = data_profile{
    data_profile_builder().null_probability(null_probability).cardinality(cardinality)};
  auto table =
    create_random_table(dtypes, row_count{static_cast<cudf::size_type>(num_rows)}, profile);

  auto const keys = table->view();

  auto const input_size = estimate_size(keys);
  state.add_element_count(input_size, "input_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_size);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::key_remapping remap(
      keys, cudf::null_equality::EQUAL, cudf::compute_metrics::YES, cudf::get_default_stream());
    // Access metrics to ensure they're computed
    [[maybe_unused]] auto dc = remap.get_distinct_count();
    [[maybe_unused]] auto mc = remap.get_max_duplicate_count();
  });
}

// Data types to test
using key_remap_datatypes =
  nvbench::enum_type_list<data_type::INT32, data_type::INT64, data_type::STRING, data_type::STRUCT>;

// Cardinality distributions
using cardinality_list = nvbench::enum_type_list<key_cardinality::ALL_UNIQUE,
                                                 key_cardinality::HIGH_UNIQUE,
                                                 key_cardinality::MED_UNIQUE,
                                                 key_cardinality::LOW_UNIQUE,
                                                 key_cardinality::SINGLE_KEY>;

// Nullable options
using nullable_list = nvbench::enum_type_list<false>;

NVBENCH_BENCH_TYPES(nvbench_key_remap_build,
                    NVBENCH_TYPE_AXES(nullable_list, key_remap_datatypes, cardinality_list))
  .set_name("key_remap_build")
  .set_type_axes_names({"Nullable", "DataType", "Cardinality"})
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_keys", {1, 2, 3});
