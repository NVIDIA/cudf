/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <nvbench/nvbench.cuh>

#include <type_traits>

// Divides decimal columns whose dividends, after being rescaled by the divisor's scale, fit in 64
// bits, need more than 64 bits, or need more than 128 bits.
template <typename DecimalType>
void bench_divide_decimal(nvbench::state& state, nvbench::type_list<DecimalType>)
{
  using Rep = typename DecimalType::rep;

  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const dividend_bits = state.get_int64("dividend_bits");
  auto const intermediate =
    static_cast<cudf::decimal_intermediate>(state.get_int64("intermediate"));

  // 64: dividends up to 10^8 rescaled by 10^2 stay below 2^64.
  // 128: dividends up to 10^17 rescaled by 10^4 exceed 2^64.
  // 256: dividends up to 10^17 rescaled by 10^22 exceed 2^127.
  Rep lhs_max    = 100'000'000;
  auto rhs_scale = numeric::scale_type{-2};
  if (dividend_bits > 64) {
    if constexpr (std::is_same_v<DecimalType, numeric::decimal32>) {
      state.skip("decimal32 dividends always fit in 64 bits.");
      return;
    } else {
      lhs_max   = 100'000'000'000'000'000;
      rhs_scale = numeric::scale_type{-4};
    }
  }
  if (dividend_bits > 128) {
    if (!std::is_same_v<DecimalType, numeric::decimal128>) {
      state.skip("Only decimal128 quotients fit when dividends exceed 128 bits.");
      return;
    }
    rhs_scale = numeric::scale_type{-22};
  }
  auto const type_id = cudf::type_to_id<DecimalType>();

  data_profile const lhs_profile = data_profile_builder().no_validity().distribution(
    type_id, distribution_id::UNIFORM, Rep{1}, lhs_max, numeric::scale_type{-2});
  data_profile const rhs_profile = data_profile_builder().no_validity().distribution(
    type_id, distribution_id::UNIFORM, Rep{1}, Rep{1'000'000}, rhs_scale);

  auto const lhs = create_random_column(type_id, row_count{num_rows}, lhs_profile);
  auto const rhs = create_random_column(type_id, row_count{num_rows}, rhs_profile);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::divide_decimal(
      lhs->view(), rhs->view(), lhs->type(), cudf::rounding_method::HALF_UP, intermediate, stream);
  });

  state.add_element_count(num_rows);
  state.add_global_memory_reads<Rep>(2 * num_rows);
  state.add_global_memory_writes<Rep>(num_rows);
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using decimal_types =
  nvbench::type_list<numeric::decimal32, numeric::decimal64, numeric::decimal128>;

NVBENCH_BENCH_TYPES(bench_divide_decimal, NVBENCH_TYPE_AXES(decimal_types))
  .set_name("divide_decimal")
  .set_type_axes_names({"DecimalType"})
  .add_int64_power_of_two_axis("num_rows", {20, 24})
  .add_int64_axis("dividend_bits", {64, 128, 256})
  .add_int64_axis("intermediate", {0, 1});
