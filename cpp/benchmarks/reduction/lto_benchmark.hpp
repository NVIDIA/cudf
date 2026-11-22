/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf_test/column_utilities.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace reduction_benchmark {
constexpr int32_t warmups      = 4;
constexpr int32_t measurements = 5;

struct options {
  bool smoke = false;
  std::optional<cudf::size_type> rows;
  std::string csv = "reduce_lto.csv";
};

struct workload {
  std::string name;
  std::string shape;
  cudf::size_type rows;
  cudf::size_type segments;
  int32_t null_percent;
  std::string hint;
  double logical_bytes;
};

template <typename T>
std::vector<std::optional<T>> host_values(cudf::column_view view, cuda::stream_ref stream)
{
  auto [values, mask]      = cudf::test::to_host<T>(view, stream);
  constexpr auto word_bits = sizeof(cudf::bitmask_type) * CHAR_BIT;
  std::vector<std::optional<T>> result(view.size());
  for (cudf::size_type row = 0; row < view.size(); ++row) {
    if (mask.empty() || ((mask[row / word_bits] >> (row % word_bits)) & 1U)) {
      result[row] = values[row];
    }
  }
  return result;
}

template <typename T>
void expect_equal(std::vector<std::optional<T>> const& actual,
                  std::vector<std::optional<T>> const& expected)
{
  if (actual.size() != expected.size()) { throw std::runtime_error{"Output size mismatch"}; }
  for (size_t index = 0; index < actual.size(); ++index) {
    if (actual[index].has_value() != expected[index].has_value()) {
      throw std::runtime_error{"Output validity mismatch"};
    }
    if (!actual[index]) { continue; }
    if constexpr (std::is_floating_point_v<T>) {
      auto const tolerance = T{1e-9} * std::max(T{1}, std::abs(*expected[index]));
      if (!std::isfinite(*actual[index]) ||
          std::abs(*actual[index] - *expected[index]) > tolerance) {
        throw std::runtime_error{"Output value mismatch"};
      }
    } else if (*actual[index] != *expected[index]) {
      throw std::runtime_error{"Output value mismatch"};
    }
  }
}

template <typename Execute, typename Validate>
void measure(std::ostream& output,
             workload const& work,
             std::string_view engine,
             std::string_view layout,
             cuda::stream_ref stream,
             Execute&& execute,
             Validate&& validate)
{
  for (int32_t iteration = 0; iteration < warmups; ++iteration) {
    auto result = std::invoke(execute);
    stream.sync();
    if (iteration == warmups - 1) { std::invoke(validate, result); }
  }
  double samples[measurements]{};
  for (int32_t iteration = 0; iteration < measurements; ++iteration) {
    stream.sync();
    auto const start = std::chrono::steady_clock::now();
    auto result      = std::invoke(execute);
    stream.sync();
    samples[iteration] =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (iteration == measurements - 1) { std::invoke(validate, result); }
  }
  std::vector<double> sorted(std::begin(samples), std::end(samples));
  std::sort(sorted.begin(), sorted.end());
  auto const median = sorted[measurements / 2];
  output << work.name << ',' << work.shape << ',' << engine << ',' << work.rows << ','
         << work.segments << ',' << work.null_percent << ',' << work.hint << ',' << layout << ','
         << work.logical_bytes << ',' << median << ',' << sorted.front() << ',' << sorted.back()
         << ',' << work.logical_bytes / (median * 1e6);
  for (auto sample : samples) {
    output << ',' << sample;
  }
  output << '\n';
  output.flush();
}

void run_q13(options const& settings, std::ostream& output);
void run_segmented(options const& settings, std::ostream& output);
}  // namespace reduction_benchmark
