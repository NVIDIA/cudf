/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/table/table.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using cudf::groupby::streaming_aggregation_request;

std::vector<cudf::table_view> make_batches(cudf::table_view input, cudf::size_type batch_size)
{
  std::vector<cudf::size_type> slice_indices;
  for (cudf::size_type start = 0; start < input.num_rows(); start += batch_size) {
    slice_indices.push_back(start);
    slice_indices.push_back(std::min(start + batch_size, input.num_rows()));
  }
  return cudf::slice(input, slice_indices);
}

void add_request(std::vector<streaming_aggregation_request>& requests,
                 cudf::size_type column_index,
                 std::unique_ptr<cudf::groupby_aggregation> aggregation)
{
  requests.push_back({column_index, std::move(aggregation)});
}

// Measures one complete streaming operation: construction, every batch, finalize, and teardown.
void run_streaming_groupby(nvbench::state& state,
                           std::vector<cudf::table_view> const& batches,
                           std::vector<cudf::size_type> const& key_indices,
                           std::vector<streaming_aggregation_request> const& requests,
                           cudf::size_type max_distinct_keys,
                           cudf::null_policy null_handling = cudf::null_policy::EXCLUDE)
{
  auto const stream = cudf::get_default_stream();
  stream.sync();  // Keep input construction out of both metrics.
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::groupby::streaming_groupby streaming{
      key_indices, requests, max_distinct_keys, null_handling};
    for (auto const& batch : batches) {
      streaming.aggregate(batch, stream);
    }
    [[maybe_unused]] auto const result = streaming.finalize(stream);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_streaming_groupby(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const batch_size  = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const key_order   = state.get_string("key_order");
  auto const aggs        = state.get_string("aggs");
  // Sorted runs of a single row are the same keys as the round robin order.
  if (key_order == "sorted" && batch_size / cardinality <= 1) {
    state.skip("sorted keys equal round robin keys without runs of equal keys");
    return;
  }

  auto const keys = [&] {
    if (key_order == "uniform_random") {
      data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
        cudf::type_id::INT32, distribution_id::UNIFORM, 0, cardinality - 1);
      return create_random_column(cudf::type_id::INT32, row_count{num_rows}, profile);
    }
    auto const int32_type = cudf::data_type{cudf::type_id::INT32};
    auto const sequence   = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>(0));
    if (key_order == "round_robin") {
      return cudf::binary_operation(sequence->view(),
                                    cudf::numeric_scalar<int32_t>(cardinality),
                                    cudf::binary_operator::PYMOD,
                                    int32_type);
    }
    auto const run  = std::max(1, batch_size / cardinality);
    auto const runs = cudf::binary_operation(
      sequence->view(), cudf::numeric_scalar<int32_t>(run), cudf::binary_operator::DIV, int32_type);
    return cudf::binary_operation(runs->view(),
                                  cudf::numeric_scalar<int32_t>(cardinality),
                                  cudf::binary_operator::PYMOD,
                                  int32_type);
  }();
  data_profile const int_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, 100);
  auto const int_vals =
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, int_profile);
  data_profile const double_profile =
    data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_id::FLOAT64, distribution_id::UNIFORM, 0, 100);
  auto const double_vals =
    create_random_column(cudf::type_id::FLOAT64, row_count{num_rows}, double_profile);
  auto const batches = make_batches(
    cudf::table_view({keys->view(), int_vals->view(), double_vals->view()}), batch_size);

  std::vector<streaming_aggregation_request> requests;
  add_request(requests, 1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (aggs == "mixed") {
    add_request(requests, 1, cudf::make_min_aggregation<cudf::groupby_aggregation>());
    add_request(requests, 2, cudf::make_max_aggregation<cudf::groupby_aggregation>());
    add_request(requests, 2, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  }

  state.add_element_count(num_rows);
  // Streaming groupby also requires capacity for every row in an individual batch.
  run_streaming_groupby(state, batches, {0}, requests, std::max(batch_size, cardinality));
}

// Two-column keys: quotient and remainder of a random group by key_modulus.
std::unique_ptr<cudf::table> make_cardinality_inputs(cudf::size_type num_rows,
                                                     cudf::size_type cardinality,
                                                     cudf::size_type key_modulus,
                                                     std::string const& key_format,
                                                     double null_probability,
                                                     unsigned seed)
{
  data_profile const key_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).no_validity().distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, 0, cardinality - 1);
  auto const groups =
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, key_profile, seed);

  auto const key_type = cudf::data_type{cudf::type_id::INT32};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::binary_operation(groups->view(),
                                           cudf::numeric_scalar<int32_t>{key_modulus},
                                           cudf::binary_operator::DIV,
                                           key_type));
  columns.push_back(cudf::binary_operation(groups->view(),
                                           cudf::numeric_scalar<int32_t>{key_modulus},
                                           cudf::binary_operator::PYMOD,
                                           key_type));
  auto const nulls = null_probability > 0 ? std::optional<double>{null_probability} : std::nullopt;
  // Attach independent masks after optional string conversion.
  for (unsigned column = 0; column < 2; ++column) {
    if (key_format == "string") {
      columns[column] = cudf::strings::from_integers(columns[column]->view());
    }
    auto [mask, null_count] = create_random_null_mask(num_rows, nulls, seed + 17u + 14u * column);
    columns[column]->set_null_mask(std::move(mask), null_count);
  }

  data_profile const double_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(nulls).distribution(
      cudf::type_id::FLOAT64, distribution_id::UNIFORM, -1.0, 1.0);
  columns.push_back(
    create_random_column(cudf::type_id::FLOAT64, row_count{num_rows}, double_profile, seed + 43u));
  data_profile const integer_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(nulls).distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, -1000, 1000);
  columns.push_back(
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, integer_profile, seed + 71u));
  columns.push_back(
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, integer_profile, seed + 101u));
  return std::make_unique<cudf::table>(std::move(columns));
}

void bench_streaming_groupby_cardinality(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const batch_size       = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const key_modulus      = static_cast<cudf::size_type>(state.get_int64("key_modulus"));
  auto const key_format       = state.get_string("key_format");
  auto const null_probability = state.get_float64("null_probability");
  auto const seed             = static_cast<unsigned>(state.get_int64("seed"));

  // INCLUDE admits four null patterns: both valid, only quotient valid, only remainder
  // valid, and both null. Bound all possible composite keys for off-grid configurations.
  auto const quotient_groups  = cardinality / key_modulus + (cardinality % key_modulus != 0);
  auto const remainder_groups = std::min(cardinality, key_modulus);
  auto const group_bound =
    cardinality + (null_probability > 0 ? quotient_groups + remainder_groups + 1 : 0);
  auto const largest_batch = std::min(batch_size, num_rows);
  auto const capacity      = std::max(largest_batch, std::min(num_rows, group_bound));

  auto const input =
    make_cardinality_inputs(num_rows, cardinality, key_modulus, key_format, null_probability, seed);
  std::vector<streaming_aggregation_request> requests;
  add_request(requests, 2, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  add_request(requests, 3, cudf::make_max_aggregation<cudf::groupby_aggregation>());
  add_request(requests,
              4,
              cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  state.add_element_count(num_rows);
  run_streaming_groupby(state,
                        make_batches(input->view(), batch_size),
                        {0, 1},
                        requests,
                        capacity,
                        cudf::null_policy::INCLUDE);
}

// Keys are distinct across batches, so the number of groups grows throughout the operation.
void bench_streaming_groupby_distinct_keys(nvbench::state& state)
{
  auto const groups     = static_cast<cudf::size_type>(state.get_int64("max_distinct_keys"));
  auto const batch_size = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const with_min   = state.get_string("schema") == "int8_min";

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::sequence(groups, cudf::numeric_scalar<int32_t>{0}));
  std::vector<streaming_aggregation_request> requests;
  if (with_min) {
    auto const residues = cudf::binary_operation(columns.front()->view(),
                                                 cudf::numeric_scalar<int32_t>{127},
                                                 cudf::binary_operator::PYMOD,
                                                 cudf::data_type{cudf::type_id::INT32});
    columns.push_back(cudf::binary_operation(residues->view(),
                                             cudf::numeric_scalar<int32_t>{63},
                                             cudf::binary_operator::SUB,
                                             cudf::data_type{cudf::type_id::INT8}));
    add_request(requests, 1, cudf::make_min_aggregation<cudf::groupby_aggregation>());
  }
  auto const input = cudf::table{std::move(columns)};

  state.add_element_count(groups);
  run_streaming_groupby(state, make_batches(input.view(), batch_size), {0}, requests, groups);
}

void bench_streaming_groupby_long_string(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const batch_size = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const key_bytes  = static_cast<cudf::size_type>(state.get_int64("key_bytes"));
  auto const mode       = state.get_string("key_mode");
  auto const key_format = state.get_string("key_format");
  auto const groups     = mode == "all_distinct" ? num_rows : std::min(batch_size, num_rows);
  auto const capacity   = std::max(groups, std::min(batch_size, num_rows));

  // Decimal digits never contain '#', so right padding preserves distinct keys exactly.
  // Repeated mode reuses the first batch's keys; all_distinct uses different keys in every row.
  auto integer_keys = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>{0});
  if (mode == "repeated_batch") {
    integer_keys = cudf::binary_operation(integer_keys->view(),
                                          cudf::numeric_scalar<int32_t>{batch_size},
                                          cudf::binary_operator::PYMOD,
                                          cudf::data_type{cudf::type_id::INT32});
  }
  auto const decimal_keys = cudf::strings::from_integers(integer_keys->view());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::strings::pad(cudf::strings_column_view{decimal_keys->view()},
                                       key_bytes,
                                       cudf::strings::side_type::RIGHT,
                                       "#"));
  if (key_format == "struct") {
    columns.push_back(std::move(integer_keys));
    auto keys = cudf::make_structs_column(
      num_rows, std::move(columns), 0, cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    columns.clear();
    columns.push_back(std::move(keys));
  }
  columns.push_back(
    cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>{1}, cudf::numeric_scalar<int32_t>{0}));
  auto const input = cudf::table{std::move(columns)};
  std::vector<streaming_aggregation_request> requests;
  add_request(requests, 1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  state.add_element_count(num_rows);
  run_streaming_groupby(state, make_batches(input.view(), batch_size), {0}, requests, capacity);
}

}  // namespace

NVBENCH_BENCH(bench_streaming_groupby)
  .set_name("streaming_groupby")
  .add_int64_axis("num_rows", {100'000'000})
  .add_int64_axis("batch_size", {100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("cardinality", {32, 256, 512, 65'536, 1'048'576})
  .add_string_axis("key_order", {"uniform_random", "round_robin", "sorted"})
  .add_string_axis("aggs", {"sum", "mixed"});

NVBENCH_BENCH(bench_streaming_groupby_cardinality)
  .set_name("streaming_groupby_cardinality")
  .add_int64_axis("num_rows", {100'000'000})
  .add_int64_axis("batch_size", {10'000'000})
  .add_int64_axis("cardinality", {257, 65'536})
  .add_string_axis("key_format", {"int32", "string"})
  .add_float64_axis("null_probability", {0.0, 0.1})
  .add_int64_axis("key_modulus", {97})
  .add_int64_axis("seed", {12'345});

NVBENCH_BENCH(bench_streaming_groupby_distinct_keys)
  .set_name("streaming_groupby_distinct_keys")
  .add_int64_axis("max_distinct_keys", {20'000'000, 100'000'000})
  .add_int64_axis("batch_size", {10'000'000})
  .add_string_axis("schema", {"keys_only", "int8_min"});

NVBENCH_BENCH(bench_streaming_groupby_long_string)
  .set_name("streaming_groupby_long_string")
  .add_int64_axis("num_rows", {20'000'000})
  .add_int64_axis("batch_size", {10'000'000})
  .add_int64_axis("key_bytes", {64, 256})
  .add_string_axis("key_mode", {"all_distinct", "repeated_batch"})
  .add_string_axis("key_format", {"string", "struct"});
