/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <random>

template <typename key_type>
static void BM_binaryop_polynomials(nvbench::state& state)
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
  auto table = create_random_table({cudf::type_to_id<key_type>()}, row_count{num_rows}, profile);
  auto column_view = table->get_column(0);

  std::vector<cudf::numeric_scalar<key_type>> constants;
  {
    std::random_device random_device;
    std::mt19937 generator;
    std::uniform_real_distribution<key_type> distribution{0, 1};

    std::transform(cuda::counting_iterator<cudf::size_type>{0},
                   cuda::counting_iterator{order + 1},
                   std::back_inserter(constants),
                   [&](int) { return cudf::numeric_scalar<key_type>(distribution(generator)); });
  }

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(num_rows);
  state.add_global_memory_writes<key_type>(num_rows);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // computes polynomials: (((ax + b)x + c)x + d)x + e... = ax**4 + bx**3 + cx**2 + dx + e....
    cudf::benchmark::scoped_range range{"benchmark_iteration"};
    rmm::cuda_stream_view stream{launch.get_stream().get_stream()};
    std::vector<std::unique_ptr<cudf::column>> intermediates;

    auto result = cudf::make_column_from_scalar(constants[0], num_rows, stream);

    for (cudf::size_type i = 0; i < order; i++) {
      auto product = cudf::binary_operation(result->view(),
                                            column_view,
                                            cudf::binary_operator::MUL,
                                            cudf::data_type{cudf::type_to_id<key_type>()},
                                            stream);
      auto sum     = cudf::binary_operation(product->view(),
                                        constants[i + 1],
                                        cudf::binary_operator::ADD,
                                        cudf::data_type{cudf::type_to_id<key_type>()},
                                        stream);
      intermediates.push_back(std::move(product));
      intermediates.push_back(std::move(result));
      result = std::move(sum);
    }
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

#define BINARYOP_POLYNOMIALS_BENCHMARK_DEFINE(name, key_type)                         \
                                                                                      \
  static void name(::nvbench::state& st) { ::BM_binaryop_polynomials<key_type>(st); } \
  NVBENCH_BENCH(name)                                                                 \
    .set_name(#name)                                                                  \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})        \
    .add_int64_axis("order", {1, 2, 4, 8, 16, 32})                                    \
    .add_float64_axis("null_probability", {0.01})

BINARYOP_POLYNOMIALS_BENCHMARK_DEFINE(binaryop_polynomials_float32, float);

BINARYOP_POLYNOMIALS_BENCHMARK_DEFINE(binaryop_polynomials_float64, double);
