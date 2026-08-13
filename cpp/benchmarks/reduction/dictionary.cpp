/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <nvbench/nvbench.cuh>

template <cudf::reduce_aggregation::Kind kind>
static std::unique_ptr<cudf::reduce_aggregation> make_reduce_aggregation()
{
  switch (kind) {
    case cudf::reduce_aggregation::ANY:
      return cudf::make_any_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::ALL:
      return cudf::make_all_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MIN:
      return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MAX:
      return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    case cudf::reduce_aggregation::MEAN:
      return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    default: CUDF_FAIL("Unsupported reduce aggregation in this benchmark");
  }
}

template <typename DataType, cudf::reduce_aggregation::Kind kind>
static void reduction_dictionary(nvbench::state& state,
                                 nvbench::type_list<DataType, nvbench::enum_type<kind>>)
{
  cudf::size_type const size{static_cast<cudf::size_type>(state.get_int64("size"))};

  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<long>(),
    distribution_id::UNIFORM,
    (kind == cudf::aggregation::ALL ? 1 : 0),
    (kind == cudf::aggregation::ANY ? 0 : 100));
  auto int_column = create_random_column(cudf::type_to_id<long>(), row_count{size}, profile);
  auto number_col = cudf::cast(*int_column, cudf::data_type{cudf::type_to_id<DataType>()});
  auto values     = cudf::dictionary::encode(*number_col);

  cudf::data_type output_type = [&] {
    if (kind == cudf::aggregation::ANY || kind == cudf::aggregation::ALL) {
      return cudf::data_type{cudf::type_id::BOOL8};
    }
    if (kind == cudf::aggregation::MEAN) { return cudf::data_type{cudf::type_id::FLOAT64}; }
    return cudf::data_type{cudf::type_to_id<DataType>()};
  }();

  auto agg = make_reduce_aggregation<kind>();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(size);
  state.add_global_memory_reads<DataType>(size);
  if (kind == cudf::aggregation::ANY || kind == cudf::aggregation::ALL) {
    state.add_global_memory_writes<nvbench::int8_t>(1);  // BOOL8s
  } else {
    state.add_global_memory_writes<DataType>(1);
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&values, output_type, &agg](nvbench::launch& launch) {
    cudf::reduce(*values, *agg, output_type);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  set_throughputs(state);
}

using Types    = nvbench::type_list<int32_t, float>;
using AggKinds = nvbench::enum_type_list<cudf::reduce_aggregation::ALL,
                                         cudf::reduce_aggregation::ANY,
                                         cudf::reduce_aggregation::MIN,
                                         cudf::reduce_aggregation::MAX,
                                         cudf::reduce_aggregation::MEAN>;

NVBENCH_BENCH_TYPES(reduction_dictionary, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("reduction_dictionary")
  .set_type_axes_names({"DataType", "AggKinds"})
  .add_int64_axis("size", {100'000, 1'000'000, 10'000'000, 100'000'000});
