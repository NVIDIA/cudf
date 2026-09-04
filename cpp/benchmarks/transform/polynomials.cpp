/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <random>

template <typename key_type>
static void BM_transform_polynomials(nvbench::state& state)
{
  auto const num_rows{static_cast<cudf::size_type>(state.get_int64("num_rows"))};
  auto const order{static_cast<cudf::size_type>(state.get_int64("order"))};
  auto const null_probability = state.get_float64("null_probability");

  CUDF_EXPECTS(order > 0, "Polynomial order must be greater than 0");

  data_profile profile;
  profile.set_null_probability(null_probability);
  profile.set_distribution_params(cudf::type_to_id<key_type>(),
                                  distribution_id::NORMAL,
                                  static_cast<key_type>(0),
                                  static_cast<key_type>(1));
  auto column = create_random_column(cudf::type_to_id<key_type>(), row_count{num_rows}, profile);

  std::vector<std::unique_ptr<cudf::column>> constants;

  std::transform(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{order + 1},
    std::back_inserter(constants),
    [&](int) { return create_random_column(cudf::type_to_id<key_type>(), row_count{1}, profile); });

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(num_rows);
  state.add_global_memory_writes<key_type>(num_rows);

  std::vector<cudf::transform_input> inputs{cudf::transform_input{*column}};
  std::transform(constants.begin(),
                 constants.end(),
                 std::back_inserter(inputs),
                 [](auto& col) -> cudf::transform_input { return cudf::scalar_column_view(*col); });

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // computes polynomials: (((ax + b)x + c)x + d)x + e... = ax**4 + bx**3 + cx**2 + dx + e....

    cudf::benchmark::scoped_range range{"benchmark_iteration"};

    std::string type = cudf::type_to_name(cudf::data_type{cudf::type_to_id<key_type>()});

    std::string params_decl = type + " c0";
    std::string expr        = "c0";

    for (cudf::size_type i = 1; i < order + 1; i++) {
      expr = "( " + expr + " ) * x +  c" + std::to_string(i);
      params_decl += ", " + type + " c" + std::to_string(i);
    }

    static_assert(std::is_same_v<key_type, float> || std::is_same_v<key_type, double>);

    // clang-format off
    std::string udf =
    "__device__ inline void compute_polynomial(" + type + "* out, " + type + " x, " + params_decl + ")" +
"{ "
" *out = " + expr + ";"
"}";

    // clang-format on

    cudf::transform(cudf::cuda_udf{udf.c_str(), "compute_polynomial"},
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_to_id<key_type>()},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    launch.get_stream().get_stream());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

#define TRANSFORM_POLYNOMIALS_BENCHMARK_DEFINE(name, key_type)                         \
                                                                                       \
  static void name(::nvbench::state& st) { ::BM_transform_polynomials<key_type>(st); } \
  NVBENCH_BENCH(name)                                                                  \
    .set_name(#name)                                                                   \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})         \
    .add_int64_axis("order", {1, 2, 4, 8, 16, 32})                                     \
    .add_float64_axis("null_probability", {0.01})

TRANSFORM_POLYNOMIALS_BENCHMARK_DEFINE(transform_polynomials_float32, float);

TRANSFORM_POLYNOMIALS_BENCHMARK_DEFINE(transform_polynomials_float64, double);
