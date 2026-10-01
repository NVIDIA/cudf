/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/table_utilities.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/random>
#include <thrust/shuffle.h>

#include <nvbench/nvbench.cuh>

/**
 * @brief Benchmark for join performance on INT32 keys.
 *
 * This simulates the join phase after key remapping, where complex keys have
 * been converted to simple INT32 IDs. Comparing this with key_remap_build times
 * shows what percentage of total join time is spent on metrics computation.
 *
 * Note that we use divisor-based generation instead of create_random_table primarily
 * because create_random_table cannot guarantee unique keys for ALL_UNIQUE. before
 * this change we ran into out of memory allocation errors with 100 million rows
 * and ALL_UNIQUE. ALL_UNIQUE should never cause these kinds of crashes.
 *
 * Cardinality distributions:
 * - ALL_UNIQUE: Sequential keys 0, 1, 2, ... (exactly num_rows unique keys)
 * - HIGH_UNIQUE: Each key repeated ~10 times
 * - MED_UNIQUE: Each key repeated ~100 times
 * - LOW_UNIQUE: Each key repeated ~1000 times
 * - SINGLE_KEY: All rows have the same key
 */

// Cardinality distributions (same as key_remap_build)
enum class key_cardinality { ALL_UNIQUE, HIGH_UNIQUE, MED_UNIQUE, LOW_UNIQUE, SINGLE_KEY };

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

enum class join_algo { HASH, SORT_MERGE };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  join_algo,
  [](auto value) {
    switch (value) {
      case join_algo::HASH: return "HASH";
      case join_algo::SORT_MERGE: return "SORT_MERGE";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

/**
 * @brief Generate an INT32 key column with controlled cardinality using divisor approach.
 *
 * For a given number of rows and divisor:
 * - keys[i] = i / divisor (before shuffling)
 * This creates (num_rows / divisor) unique keys, each appearing exactly divisor times
 * (except possibly the last key which may have fewer occurrences).
 *
 * The keys are generated sequentially and then shuffled to reduce data locality bias,
 * which could affect join performance. The shuffle uses a fixed seed (12345) for
 * reproducibility across benchmark runs.
 *
 * This approach provides:
 * - Exact cardinality control - CRITICAL for ALL_UNIQUE to prevent memory allocation errors from
 *   unexpected duplicates in large datasets (100M+ rows)
 * - Perfect control over duplicate counts - ensures benchmarks measure what we intend
 * - Deterministic output for reproducible benchmarks (fixed seed)
 * - Minimal generation overhead
 * - Reduced data locality bias from shuffling
 *
 * For a more realistic distribution, one could generate random data with create_random_table,
 * perform key remapping, and use the remapped INT32 IDs. However:
 * - This would add complexity and require careful stream synchronization
 * - For ALL_UNIQUE, would still need deduplication to avoid OOM crashes
 */
std::unique_ptr<cudf::column> generate_int32_keys(cudf::size_type num_rows, cudf::size_type divisor)
{
  auto stream = cudf::get_default_stream();

  // Generate sequence 0, 1, 2, ..., num_rows-1
  auto init = cudf::make_fixed_width_scalar<cudf::size_type>(0, stream);
  auto step = cudf::make_fixed_width_scalar<cudf::size_type>(1, stream);
  auto seq  = cudf::sequence(num_rows, *init, *step, stream);

  std::unique_ptr<cudf::column> result;
  if (divisor == 1) {
    // ALL_UNIQUE: use sequence directly
    result = std::move(seq);
  } else {
    // Divide each element by divisor to create duplicates
    auto divisor_scalar = cudf::make_fixed_width_scalar<cudf::size_type>(divisor, stream);
    result              = cudf::binary_operation(seq->view(),
                                    *divisor_scalar,
                                    cudf::binary_operator::DIV,
                                    cudf::data_type{cudf::type_id::INT32});
  }

  // Shuffle the generated data to reduce bias from data locality.
  // Use a fixed seed for reproducibility across benchmark runs.
  thrust::shuffle(rmm::exec_policy_nosync(stream),
                  result->mutable_view().begin<int32_t>(),
                  result->mutable_view().end<int32_t>(),
                  cuda::std::philox4x32{12345});

  return result;
}

template <key_cardinality Card, join_algo Algo>
void nvbench_join_on_int32(nvbench::state& state,
                           nvbench::type_list<nvbench::enum_type<Card>, nvbench::enum_type<Algo>>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  // Skip HASH join with SINGLE_KEY for large row counts - output is num_rows^2 which
  // requires enormous memory. Only run smallest size (10K rows = 100M output rows).
  if constexpr (Card == key_cardinality::SINGLE_KEY && Algo == join_algo::HASH) {
    if (num_rows > 10'000) {
      state.skip("HASH join with SINGLE_KEY: output size num_rows^2 exceeds memory");
      return;
    }
  }

  // Determine divisor based on cardinality (higher divisor = more duplicates)
  cudf::size_type divisor = 1;
  if constexpr (Card == key_cardinality::ALL_UNIQUE) {
    divisor = 1;  // Each key appears once
  } else if constexpr (Card == key_cardinality::HIGH_UNIQUE) {
    divisor = 10;  // Each key appears ~10 times
  } else if constexpr (Card == key_cardinality::MED_UNIQUE) {
    divisor = 100;  // Each key appears ~100 times
  } else if constexpr (Card == key_cardinality::LOW_UNIQUE) {
    divisor = 1000;  // Each key appears ~1000 times
  } else if constexpr (Card == key_cardinality::SINGLE_KEY) {
    divisor = num_rows;  // All rows have key 0
  }

  // Generate controlled INT32 key columns
  auto left_keys  = generate_int32_keys(num_rows, divisor);
  auto right_keys = generate_int32_keys(num_rows, divisor);

  // Create table views
  std::vector<cudf::column_view> left_cols  = {left_keys->view()};
  std::vector<cudf::column_view> right_cols = {right_keys->view()};
  cudf::table_view left(left_cols);
  cudf::table_view right(right_cols);

  auto const input_size = estimate_size(left) + estimate_size(right);
  state.add_element_count(input_size, "input_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_size);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if constexpr (Algo == join_algo::HASH) {
      auto result = cudf::inner_join(left, right, cudf::null_equality::EQUAL);
    } else {
      auto smj    = cudf::sort_merge_join(right, cudf::sorted::NO, cudf::null_equality::EQUAL);
      auto result = smj.inner_join(left);
    }
  });
}

// Cardinality distributions
using cardinality_list = nvbench::enum_type_list<key_cardinality::ALL_UNIQUE,
                                                 key_cardinality::HIGH_UNIQUE,
                                                 key_cardinality::MED_UNIQUE,
                                                 key_cardinality::LOW_UNIQUE,
                                                 key_cardinality::SINGLE_KEY>;

// Join algorithms
using join_algo_list = nvbench::enum_type_list<join_algo::HASH, join_algo::SORT_MERGE>;

NVBENCH_BENCH_TYPES(nvbench_join_on_int32, NVBENCH_TYPE_AXES(cardinality_list, join_algo_list))
  .set_name("join_on_int32")
  .set_type_axes_names({"Cardinality", "JoinAlgo"})
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000});
