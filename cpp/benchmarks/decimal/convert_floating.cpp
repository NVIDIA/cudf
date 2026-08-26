/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <nvbench/nvbench.cuh>

#include <type_traits>

// This benchmark compares the cost of converting decimal <--> floating point
template <typename InputType, typename OutputType>
void bench_cast_decimal(nvbench::state& state, nvbench::type_list<InputType, OutputType>)
{
  static constexpr bool is_input_floating  = std::is_floating_point_v<InputType>;
  static constexpr bool is_output_floating = std::is_floating_point_v<OutputType>;

  static constexpr bool is_double =
    std::is_same_v<InputType, double> || std::is_same_v<OutputType, double>;
  static constexpr bool is_128bit = std::is_same_v<InputType, numeric::decimal128> ||
                                    std::is_same_v<OutputType, numeric::decimal128>;

  // Skip floating --> floating and decimal --> decimal
  if constexpr (is_input_floating == is_output_floating) {
    state.skip("Meaningless conversion.");
    return;
  }

  // Skip float <--> dec128
  if constexpr (!is_double && is_128bit) {
    state.skip("Ignoring float <--> dec128.");
    return;
  }

  // Get settings
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const exp_mode = state.get_int64("exp_range");

  // Exponent range: Range size is 10^6
  // These probe the edges of the float and double ranges, as well as more common values
  int const exp_min_array[] = {-307, -37, -14, -3, 8, 31, 301};
  int const exp_range_size  = 6;
  int const exp_min         = exp_min_array[exp_mode];
  int const exp_max         = exp_min + exp_range_size;

  // With exp range size of 6, decimal output (generated or casted-to) has 7 digits of precision
  int const extra_digits_precision = 1;

  // Exclude end range of double from float test
  if (!is_double && ((exp_mode == 0) || (exp_mode == 6))) {
    state.skip("Range beyond end of float tests.");
    return;
  }

  // Type IDs
  auto const input_id  = cudf::type_to_id<InputType>();
  auto const output_id = cudf::type_to_id<OutputType>();

  // Create data profile and scale
  auto const [output_scale, profile] = [&]() {
    if constexpr (is_input_floating) {
      // Range for generated floating point values
      auto get_pow10 = [](auto exp10) {
        return std::pow(static_cast<InputType>(10), static_cast<InputType>(exp10));
      };
      InputType const floating_range_min = get_pow10(exp_min);
      InputType const floating_range_max = get_pow10(exp_max);

      // With exp range size of 6, output has 7 decimal digits of precision
      auto const decimal_output_scale = exp_min - extra_digits_precision;

      // Input distribution
      data_profile const profile = data_profile_builder().distribution(
        input_id, distribution_id::NORMAL, floating_range_min, floating_range_max);

      return std::pair{decimal_output_scale, profile};

    } else {  // Generating decimals

      using decimal_rep_type = typename InputType::rep;

      // For exp range size 6 and precision 7, generates ints between 10 and 10^7,
      // with scale factor of: exp_max - 7. This matches floating point generation.
      int const digits_precision     = exp_range_size + extra_digits_precision;
      auto const decimal_input_scale = numeric::scale_type{exp_max - digits_precision};

      // Range for generated integer values
      auto get_pow10 = [](auto exp10) {
        return numeric::detail::ipow<decimal_rep_type, numeric::Radix::BASE_10>(exp10);
      };
      auto const decimal_range_min = get_pow10(digits_precision - exp_range_size);
      auto const decimal_range_max = get_pow10(digits_precision);

      // Input distribution
      data_profile const profile = data_profile_builder().distribution(input_id,
                                                                       distribution_id::NORMAL,
                                                                       decimal_range_min,
                                                                       decimal_range_max,
                                                                       decimal_input_scale);

      return std::pair{0, profile};
    }
  }();

  // Generate input data
  auto const input_col  = create_random_column(input_id, row_count{num_rows}, profile);
  auto const input_view = input_col->view();

  // Output type
  auto const output_type =
    !is_input_floating ? cudf::data_type(output_id) : cudf::data_type(output_id, output_scale);

  // Stream
  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Run benchmark
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::cast(input_view, output_type); });

  // Throughput statistics
  state.add_element_count(num_rows);
  state.add_global_memory_reads<InputType>(num_rows);
  state.add_global_memory_writes<OutputType>(num_rows);
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

// Data types
using data_types =
  nvbench::type_list<float, double, numeric::decimal32, numeric::decimal64, numeric::decimal128>;

NVBENCH_BENCH_TYPES(bench_cast_decimal, NVBENCH_TYPE_AXES(data_types, data_types))
  .set_name("decimal_floating_conversion")
  .set_type_axes_names({"InputType", "OutputType"})
  .add_int64_power_of_two_axis("num_rows", {28})
  .add_int64_axis("exp_range", nvbench::range(0, 6));
