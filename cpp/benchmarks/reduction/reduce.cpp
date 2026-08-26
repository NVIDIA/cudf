/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>

template <cudf::reduce_aggregation::Kind kind>
static std::unique_ptr<cudf::reduce_aggregation> make_reduce_aggregation()
{
  switch (kind) {
    case cudf::reduce_aggregation::ARGMIN:
      return cudf::make_argmin_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MIN:
      return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::SUM:
      return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MEAN:
      return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::VARIANCE:
      return cudf::make_variance_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::STD:
      return cudf::make_std_aggregation<cudf::reduce_aggregation>();
    default: CUDF_FAIL("Unsupported reduce aggregation in this benchmark");
  }
}

template <typename DataType, cudf::reduce_aggregation::Kind kind>
static void reduction(nvbench::state& state, nvbench::type_list<DataType, nvbench::enum_type<kind>>)
{
  if (cudf::is_chrono<DataType>() && kind != cudf::aggregation::MIN) {
    state.skip("Skip chrono types for some aggregations");
    return;
  }

  auto const size      = static_cast<cudf::size_type>(state.get_int64("size"));
  auto const max_value = static_cast<cudf::size_type>(state.get_int64("max_value"));

  auto const input_type      = cudf::type_to_id<DataType>();
  data_profile const profile = data_profile_builder().no_validity().distribution(
    input_type, distribution_id::UNIFORM, 0, max_value > 0 ? max_value : 100);
  auto const input_column = create_random_column(input_type, row_count{size}, profile);

  auto const output_type = [&] {
    if (kind == cudf::aggregation::MEAN || kind == cudf::aggregation::VARIANCE ||
        kind == cudf::aggregation::STD) {
      return cudf::data_type{cudf::type_id::FLOAT64};
    }
    if (kind == cudf::aggregation::ARGMIN) { return cudf::data_type{cudf::type_id::INT32}; }
    return input_column->type();
  }();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(size);
  state.add_global_memory_reads<DataType>(size);
  state.add_global_memory_writes<DataType>(1);

  auto agg                    = make_reduce_aggregation<kind>();
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&input_column, output_type, &agg](nvbench::launch&) {
    cudf::reduce(*input_column, *agg, output_type);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  set_throughputs(state);
}

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");

using Types    = nvbench::type_list<int32_t, int64_t, double, cudf::timestamp_ms>;
using AggKinds = nvbench::enum_type_list<cudf::reduce_aggregation::ARGMIN,
                                         cudf::reduce_aggregation::MIN,
                                         cudf::reduce_aggregation::SUM,
                                         cudf::reduce_aggregation::PRODUCT,
                                         cudf::reduce_aggregation::VARIANCE,
                                         cudf::reduce_aggregation::STD,
                                         cudf::reduce_aggregation::MEAN>;

NVBENCH_BENCH_TYPES(reduction, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("reduction")
  .set_type_axes_names({"DataType", "AggKinds"})
  .add_int64_axis("size", {100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("max_value", {100, -1});
