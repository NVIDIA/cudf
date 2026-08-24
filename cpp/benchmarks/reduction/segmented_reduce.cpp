/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>

bool constexpr is_boolean_output_agg(cudf::segmented_reduce_aggregation::Kind kind)
{
  return kind == cudf::segmented_reduce_aggregation::ALL ||
         kind == cudf::segmented_reduce_aggregation::ANY;
}

bool constexpr is_float_output_agg(cudf::segmented_reduce_aggregation::Kind kind)
{
  return kind == cudf::segmented_reduce_aggregation::MEAN ||
         kind == cudf::segmented_reduce_aggregation::VARIANCE ||
         kind == cudf::segmented_reduce_aggregation::STD;
}

template <cudf::segmented_reduce_aggregation::Kind kind>
std::unique_ptr<cudf::segmented_reduce_aggregation> make_reduce_aggregation()
{
  switch (kind) {
    case cudf::segmented_reduce_aggregation::SUM:
      return cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::MIN:
      return cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::MAX:
      return cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::ALL:
      return cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::ANY:
      return cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::SUM_OF_SQUARES:
      return cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::MEAN:
      return cudf::make_mean_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::VARIANCE:
      return cudf::make_variance_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::STD:
      return cudf::make_std_aggregation<cudf::segmented_reduce_aggregation>();
    case cudf::segmented_reduce_aggregation::NUNIQUE:
      return cudf::make_nunique_aggregation<cudf::segmented_reduce_aggregation>();
    default: CUDF_FAIL("Unsupported segmented reduce aggregation in this benchmark");
  }
}

template <typename DataType>
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> make_test_data(
  nvbench::state& state)
{
  auto const column_size{cudf::size_type(state.get_int64("column_size"))};
  auto const num_segments{cudf::size_type(state.get_int64("num_segments"))};

  auto segment_length = column_size / num_segments;

  auto const dtype     = cudf::type_to_id<DataType>();
  data_profile profile = data_profile_builder().cardinality(0).no_validity().distribution(
    dtype, distribution_id::UNIFORM, 0, 100);
  auto input = create_random_column(dtype, row_count{column_size}, profile);

  auto offsets = cudf::sequence(num_segments + 1,
                                cudf::numeric_scalar<cudf::size_type>(0),
                                cudf::numeric_scalar<cudf::size_type>(segment_length));
  return std::pair(std::move(input), std::move(offsets));
}

template <typename DataType, cudf::aggregation::Kind kind>
void BM_Segmented_Reduction(nvbench::state& state,
                            nvbench::type_list<DataType, nvbench::enum_type<kind>>)
{
  auto const column_size{cudf::size_type(state.get_int64("column_size"))};
  auto const num_segments{cudf::size_type(state.get_int64("num_segments"))};

  auto [input, offsets] = make_test_data<DataType>(state);
  auto agg              = make_reduce_aggregation<kind>();

  auto const output_type = [] {
    if (is_boolean_output_agg(kind)) { return cudf::data_type{cudf::type_id::BOOL8}; }
    if (is_float_output_agg(kind)) { return cudf::data_type{cudf::type_id::FLOAT64}; }
    if (kind == cudf::segmented_reduce_aggregation::NUNIQUE) {
      return cudf::data_type{cudf::type_to_id<cudf::size_type>()};
    }
    return cudf::data_type{cudf::type_to_id<DataType>()};
  }();

  state.add_element_count(column_size);
  state.add_global_memory_reads<DataType>(column_size);
  if (is_boolean_output_agg(kind)) {
    state.add_global_memory_writes<nvbench::int8_t>(num_segments);  // BOOL8
  } else {
    state.add_global_memory_writes<DataType>(num_segments);
  }

  auto const input_view   = input->view();
  auto const offsets_view = offsets->view();
  auto const offset_span  = cudf::device_span<cudf::size_type const>{
    offsets_view.template data<cudf::size_type>(), static_cast<std::size_t>(offsets_view.size())};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(
    nvbench::exec_tag::sync, [input_view, output_type, offset_span, &agg](nvbench::launch& launch) {
      segmented_reduce(input_view, offset_span, *agg, output_type, cudf::null_policy::INCLUDE);
    });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using Types = nvbench::type_list<bool, int32_t, float, double>;
// Skip benchmarking MAX/ANY since they are covered by MIN/ALL respectively.
// Also VARIANCE includes STD calculation.
using AggKinds = nvbench::enum_type_list<cudf::aggregation::SUM,
                                         cudf::aggregation::PRODUCT,
                                         cudf::aggregation::MIN,
                                         cudf::aggregation::ALL,
                                         cudf::aggregation::MEAN,
                                         cudf::aggregation::VARIANCE,
                                         cudf::aggregation::NUNIQUE>;

NVBENCH_BENCH_TYPES(BM_Segmented_Reduction, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("segmented_reduction")
  .set_type_axes_names({"DataType", "AggregationKinds"})
  .add_int64_axis("column_size", {100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_segments", {1'000, 10'000, 100'000});
