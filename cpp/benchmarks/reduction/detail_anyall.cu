/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

template <typename DataType>
std::unique_ptr<cudf::column> make_detail_anyall_input(cudf::size_type size,
                                                       std::string const& kind_str,
                                                       std::string const& pattern,
                                                       rmm::cuda_stream_view stream)
{
  auto values = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<DataType>()},
                                              size,
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              cudf::get_current_device_resource_ref());
  using host_value_type = std::conditional_t<std::is_same_v<DataType, bool>, int8_t, DataType>;

  // "first" and "last" place the decisive value for each detail reduction at that position.
  auto const fill_value = kind_str == "any" ? host_value_type{0} : host_value_type{1};
  auto const find_value = kind_str == "any" ? host_value_type{1} : host_value_type{0};
  auto host_values      = std::vector<host_value_type>(size, fill_value);

  if (pattern == "first") {
    host_values.front() = find_value;
  } else if (pattern == "last") {
    host_values.back() = find_value;
  } else {
    CUDF_EXPECTS(pattern == "none", "Unsupported detail_anyall benchmark pattern.");
  }

  CUDF_CUDA_TRY(cudaMemcpyAsync(values->mutable_view().template begin<DataType>(),
                                host_values.data(),
                                sizeof(DataType) * size,
                                cudaMemcpyHostToDevice,
                                stream.value()));

  return values;
}

template <typename DataType>
static void detail_anyall(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const size     = static_cast<cudf::size_type>(state.get_int64("size"));
  auto const kind_str = state.get_string("kind");
  auto const pattern  = state.get_string("pattern");

  auto stream       = cudf::get_default_stream();
  auto const values = make_detail_anyall_input<DataType>(size, kind_str, pattern, stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(size);
  state.add_global_memory_reads<DataType>(size);
  state.add_global_memory_writes<nvbench::int8_t>(1);

  bool result                 = false;
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&values, &kind_str, stream, &result](nvbench::launch&) {
    auto const input = values->view();
    auto const begin = input.template begin<DataType>();
    auto const end   = input.template end<DataType>();
    auto predicate   = [] __device__(DataType value) -> bool { return value != DataType{0}; };

    result = kind_str == "any" ? cudf::detail::any_of(begin, end, predicate, stream)
                               : cudf::detail::all_of(begin, end, predicate, stream);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  set_throughputs(state);
}

using Types = nvbench::type_list<bool, int32_t>;

NVBENCH_BENCH_TYPES(detail_anyall, NVBENCH_TYPE_AXES(Types))
  .set_name("detail_anyall")
  .set_type_axes_names({"DataType"})
  .add_string_axis("kind", {"any", "all"})
  .add_string_axis("pattern", {"first", "last", "none"})
  .add_int64_axis("size", {100'000, 10'000'000, 100'000'000});
