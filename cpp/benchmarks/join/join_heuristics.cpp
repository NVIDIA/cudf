/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>
#include <benchmarks/common/table_utilities.hpp>
#include <benchmarks/join/nvbench_helpers.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/key_remapping.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

/**
 * Benchmark for join heuristic metrics calculation.
 *
 * Compares three approaches for computing join decision metrics:
 * 1. cudf::distinct_count - for deciding if distinct join should be used
 * 2. groupby + max reduction - for getting max duplicate count
 * 3. key_remapping with metrics - computes both in a single hash table build
 *
 * The metrics help decide:
 * - distinct_count: Whether to use a distinct join optimization
 * - max_duplicate_count: Whether hash join will suffer from key collisions
 */

enum class heuristic_method : int32_t {
  DISTINCT_COUNT = 0,  // cudf::distinct_count API
  GROUPBY_MAX    = 1,  // groupby count + max reduction
  KEY_REMAPPING  = 2   // key_remapping with metrics
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  heuristic_method,
  [](heuristic_method value) {
    switch (value) {
      case heuristic_method::DISTINCT_COUNT: return "DISTINCT_COUNT";
      case heuristic_method::GROUPBY_MAX: return "GROUPBY_MAX";
      case heuristic_method::KEY_REMAPPING: return "KEY_REMAPPING";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

// Cardinality ratios - fraction of rows that are unique
enum class cardinality_ratio : int32_t {
  ALL_UNIQUE  = 0,  // Every row is unique (cardinality = num_rows)
  HIGH_UNIQUE = 1,  // 10% of rows are unique
  LOW_UNIQUE  = 2,  // 0.1% of rows are unique
  SINGLE_KEY  = 3   // All rows have the same key (cardinality = 1)
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cardinality_ratio,
  [](cardinality_ratio value) {
    switch (value) {
      case cardinality_ratio::ALL_UNIQUE: return "ALL_UNIQUE";
      case cardinality_ratio::HIGH_UNIQUE: return "HIGH_UNIQUE";
      case cardinality_ratio::LOW_UNIQUE: return "LOW_UNIQUE";
      case cardinality_ratio::SINGLE_KEY: return "SINGLE_KEY";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

using HEURISTIC_METHODS = nvbench::enum_type_list<heuristic_method::DISTINCT_COUNT,
                                                  heuristic_method::GROUPBY_MAX,
                                                  heuristic_method::KEY_REMAPPING>;

using CARDINALITY_RATIOS = nvbench::enum_type_list<cardinality_ratio::ALL_UNIQUE,
                                                   cardinality_ratio::HIGH_UNIQUE,
                                                   cardinality_ratio::LOW_UNIQUE,
                                                   cardinality_ratio::SINGLE_KEY>;

// Simplified data types for this benchmark
using HEURISTIC_DATATYPES =
  nvbench::enum_type_list<data_type::INT32, data_type::INT64, data_type::STRING, data_type::STRUCT>;

template <bool Nullable, data_type DataType, heuristic_method Method, cardinality_ratio Cardinality>
void nvbench_join_heuristics(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<Nullable>,
                                                nvbench::enum_type<DataType>,
                                                nvbench::enum_type<Method>,
                                                nvbench::enum_type<Cardinality>>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_keys = state.get_int64("num_keys");

  // Calculate cardinality based on the ratio
  cudf::size_type cardinality;
  switch (Cardinality) {
    case cardinality_ratio::ALL_UNIQUE:
      cardinality = num_rows;  // All unique
      break;
    case cardinality_ratio::HIGH_UNIQUE:
      cardinality = std::max(1, num_rows / 10);  // 10% unique
      break;
    case cardinality_ratio::LOW_UNIQUE:
      cardinality = std::max(1, num_rows / 1000);  // 0.1% unique
      break;
    case cardinality_ratio::SINGLE_KEY:
      cardinality = 1;  // Single key
      break;
  }

  // Generate input table with specified cardinality
  auto const dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  double const null_probability = Nullable ? 0.1 : 0;
  auto const profile            = data_profile{
    data_profile_builder().null_probability(null_probability).cardinality(cardinality)};
  auto table = create_random_table(dtypes, row_count{num_rows}, profile, 42);

  auto const table_view = table->view();

  // Select key columns
  std::vector<cudf::size_type> key_indices(num_keys);
  std::iota(key_indices.begin(), key_indices.end(), 0);
  auto const keys = table_view.select(key_indices);

  auto const input_size = estimate_size(keys);
  state.add_element_count(input_size, "input_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_size);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if constexpr (Method == heuristic_method::DISTINCT_COUNT) {
      // Approach 1: cudf::distinct_count
      [[maybe_unused]] auto distinct = cudf::distinct_count(keys, cudf::null_equality::EQUAL);
      // Note: This only gives distinct count, not max duplicate count
    } else if constexpr (Method == heuristic_method::GROUPBY_MAX) {
      // Approach 2: groupby count + max reduction
      // Group by keys and count occurrences
      cudf::groupby::groupby gb(keys);
      std::vector<cudf::groupby::aggregation_request> requests;

      // We need a values column for aggregation - use first key column
      cudf::groupby::aggregation_request req;
      req.values = keys.column(0);
      req.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
      requests.push_back(std::move(req));

      auto [group_keys, results] = gb.aggregate(requests);

      // Get distinct count from number of groups
      [[maybe_unused]] auto distinct_count = group_keys->num_rows();

      // Get max duplicate count via max reduction on count column
      auto const& counts_column = results[0].results[0];
      [[maybe_unused]] auto max_count =
        cudf::reduce(*counts_column,
                     *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
                     counts_column->type());
    } else if constexpr (Method == heuristic_method::KEY_REMAPPING) {
      // Approach 3: key_remapping with metrics
      cudf::key_remapping remap(
        keys, cudf::null_equality::EQUAL, cudf::compute_metrics::YES, cudf::get_default_stream());

      // Get both metrics
      [[maybe_unused]] auto distinct_count = remap.get_distinct_count();
      [[maybe_unused]] auto max_dup_count  = remap.get_max_duplicate_count();
    }
  });

  set_throughputs(state);
}

// Main benchmark
NVBENCH_BENCH_TYPES(nvbench_join_heuristics,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<false, true>,  // Nullable
                                      HEURISTIC_DATATYPES,
                                      HEURISTIC_METHODS,
                                      CARDINALITY_RATIOS))
  .set_name("join_heuristics")
  .set_type_axes_names({"Nullable", "DataType", "Method", "Cardinality"})
  .add_int64_axis("num_rows", {1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_keys", {1, 2, 3});
