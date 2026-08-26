/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal128, "decimal128", "decimal128");

template <typename Type>
void groupby_max_helper(nvbench::state& state,
                        cudf::size_type num_rows,
                        cudf::size_type cardinality,
                        double null_probability)
{
  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, profile);
  }();

  auto const make_values = [&]() {
    auto builder = data_profile_builder().cardinality(0);
    if constexpr (cudf::is_fixed_point<Type>()) {
      builder.distribution(
        cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows, numeric::scale_type{0});
    } else {
      builder.distribution(cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows);
    }
    if (null_probability > 0) {
      builder.null_probability(null_probability);
    } else {
      builder.no_validity();
    }
    return create_random_column(
      cudf::type_to_id<Type>(), row_count{num_rows}, data_profile{builder});
  };

  auto const num_aggregations = state.get_int64("num_aggregations");

  auto keys_view = keys->view();

  std::vector<std::unique_ptr<cudf::column>> val_cols;
  std::vector<cudf::groupby::aggregation_request> requests;
  for (int64_t i = 0; i < num_aggregations; i++) {
    requests.emplace_back();
    val_cols.emplace_back(make_values());
    requests[i].values = val_cols.back()->view();
    requests[i].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto gb_obj       = cudf::groupby::groupby(cudf::table_view({keys_view, keys_view, keys_view}));
    auto const result = gb_obj.aggregate(requests);
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(
    static_cast<double>(num_rows * num_aggregations) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <typename Type>
void bench_groupby_max(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");

  groupby_max_helper<Type>(state, num_rows, cardinality, null_probability);
}

template <typename Type>
void bench_groupby_max_cardinality(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_aggregations = state.get_int64("num_aggregations");
  auto const is_streaming     = state.get_string("api") == "streaming";

  // TODO: streaming groupby reuses the cudf hash element_aggregator, which has
  // no decimal128 MIN/MAX/SUM specialization (no native 128-bit atomics).  The
  // stateless `normal` path falls back to sort-based aggregation, but streaming
  // has no fallback and rejects the request.  Re-enable once streaming has a
  // non-atomic aggregator path or 128-bit atomics gain hardware support.
  if (is_streaming && std::is_same_v<Type, numeric::decimal128>) {
    state.skip("streaming groupby does not support decimal128 MAX/MIN/SUM");
    return;
  }

  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, profile);
  }();

  auto const make_values = [&]() {
    auto builder = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(
      cudf::type_to_id<Type>(), row_count{num_rows}, data_profile{builder});
  };

  std::vector<std::unique_ptr<cudf::column>> val_cols;
  for (int64_t i = 0; i < num_aggregations; i++) {
    val_cols.emplace_back(make_values());
  }

  auto keys_view = keys->view();

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  if (is_streaming) {
    std::vector<cudf::column_view> all_columns = {keys_view, keys_view, keys_view};
    for (auto const& vc : val_cols) {
      all_columns.push_back(vc->view());
    }
    auto const full_table = cudf::table_view(all_columns);

    std::vector<cudf::size_type> key_indices = {0, 1, 2};
    std::vector<cudf::groupby::streaming_aggregation_request> requests;
    for (int64_t i = 0; i < num_aggregations; i++) {
      cudf::groupby::streaming_aggregation_request req;
      req.column_index = static_cast<cudf::size_type>(3 + i);
      req.aggregation  = cudf::make_max_aggregation<cudf::groupby_aggregation>();
      requests.push_back(std::move(req));
    }
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto sgb = cudf::groupby::streaming_groupby(key_indices, requests, num_rows);
      sgb.aggregate(full_table);
      auto const result = sgb.finalize();
    });
  } else {
    std::vector<cudf::groupby::aggregation_request> requests;
    for (int64_t i = 0; i < num_aggregations; i++) {
      requests.emplace_back();
      requests[i].values = val_cols[i]->view();
      requests[i].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    }
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys_view, keys_view, keys_view}));
      auto const result = gb_obj.aggregate(requests);
    });
  }

  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(
    static_cast<double>(num_rows * num_aggregations) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_groupby_max,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t, float, double>))
  .set_name("groupby_max")
  .add_int64_axis("cardinality", {0})
  .add_int64_power_of_two_axis("num_rows", {12, 18, 24})
  .add_float64_axis("null_probability", {0, 0.1, 0.9})
  .add_int64_axis("num_aggregations", {1, 2, 4, 8, 16, 32});

NVBENCH_BENCH_TYPES(bench_groupby_max_cardinality,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, numeric::decimal128>))
  .set_name("groupby_max_cardinality")
  .add_int64_axis("num_rows", {20'000'000})
  .add_int64_axis("num_aggregations", {1, 2, 3, 4, 5, 6, 7, 8})
  .add_int64_axis("cardinality", {20, 50, 100, 1'000, 10'000, 100'000, 1'000'000})
  .add_string_axis("api", {"normal", "streaming"});
