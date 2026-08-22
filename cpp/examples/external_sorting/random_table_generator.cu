/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "parquet_io.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform.h>

#include <memory>
#include <type_traits>
#include <vector>

/**
 * @file random_table_generator.cuh
 * @brief Random table and column generation utilities for external sorting example
 *
 * This header provides utilities to generate random tables and columns with various
 * data types for testing and demonstration purposes. The implementation uses thrust
 * algorithms for GPU-accelerated generation.
 */

namespace cudf {
namespace examples {

/**
 * @brief Generate a random numeric column using thrust
 *
 * Creates a column filled with random values of the specified numeric type.
 * Uses thrust::uniform_int_distribution for integral types and
 * thrust::uniform_real_distribution for floating-point types.
 *
 * @tparam T The numeric data type (int32_t, int64_t, float, double, etc.)
 * @param lower Lower bound of the random values (inclusive)
 * @param upper Upper bound of the random values (inclusive for int, exclusive for float)
 * @param num_rows Number of rows in the output column
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocations
 * @return Unique pointer to the generated column
 */
template <typename T>
std::unique_ptr<cudf::column> generate_random_numeric_column(T lower,
                                                             T upper,
                                                             cudf::size_type num_rows,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_to_id<T>()}, num_rows, cudf::mask_state::ALL_VALID, stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator(0),
                    thrust::counting_iterator(num_rows),
                    col->mutable_view().begin<T>(),
                    [lower, upper] __device__(auto idx) -> T {
                      thrust::minstd_rand engine;
                      engine.discard(idx);
                      if constexpr (std::is_integral_v<T>) {
                        thrust::uniform_int_distribution<T> dist(lower, upper);
                        return dist(engine);
                      } else {
                        thrust::uniform_real_distribution<T> dist(lower, upper);
                        return dist(engine);
                      }
                    });

  return col;
}

/**
 * @brief Generate a table with n_columns random columns and m_rows
 *
 * Creates a table with a mix of different data types to demonstrate libcudf's
 * type system and provide realistic test data. The column types cycle through:
 * INT32, FLOAT64, INT64, FLOAT32, INT16.
 *
 * Each column type uses different value ranges to create diverse datasets:
 * - INT32: Large integers with column-specific offsets
 * - FLOAT64: Double precision floats in range [0, 1000+offset]
 * - INT64: Large long integers around 1e9-1e10 range
 * - FLOAT32: Single precision floats with small ranges
 * - INT16: Short integers with moderate ranges
 *
 * @param n_columns Number of columns to generate
 * @param m_rows Number of rows in each column
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocations
 * @return Unique pointer to the generated table
 */
std::unique_ptr<cudf::table> generate_random_table(std::vector<cudf::data_type> const& types,
                                                   cudf::size_type n_columns,
                                                   cudf::size_type m_rows,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;

  for (cudf::size_type i = 0; i < n_columns; ++i) {
    auto type_idx = i % types.size();
    switch (types[type_idx].id()) {
      case cudf::type_id::INT32:
        columns.emplace_back(generate_random_numeric_column<int32_t>(
          1000000 + i * 100000, 9999999 + i * 100000, m_rows, stream, mr));
        break;
      case cudf::type_id::FLOAT64:
        columns.emplace_back(
          generate_random_numeric_column<double>(0.0, 1000.0 + i * 100, m_rows, stream, mr));
        break;
      case cudf::type_id::INT64:
        columns.emplace_back(
          generate_random_numeric_column<int64_t>(static_cast<int64_t>(1e9) + i * 1000000,
                                                  static_cast<int64_t>(1e10) + i * 1000000,
                                                  m_rows,
                                                  stream,
                                                  mr));
        break;
      case cudf::type_id::FLOAT32:
        columns.emplace_back(
          generate_random_numeric_column<float>(10.0f + i, 1000.0f + i * 50, m_rows, stream, mr));
        break;
      case cudf::type_id::INT16:
        columns.emplace_back(
          generate_random_numeric_column<int16_t>(static_cast<int16_t>(100 + i * 10),
                                                  static_cast<int16_t>(30000 + i * 1000),
                                                  m_rows,
                                                  stream,
                                                  mr));
        break;
      case cudf::type_id::TIMESTAMP_SECONDS: {
        // Generate timestamps from 2020-01-01 to 2024-12-31
        int64_t start_timestamp = 1577836800;  // 2020-01-01 00:00:00 UTC
        int64_t end_timestamp   = 1735689600;  // 2024-12-31 00:00:00 UTC
        columns.emplace_back(generate_random_numeric_column<int64_t>(
          start_timestamp + i * 86400, end_timestamp - i * 86400, m_rows, stream, mr));
        break;
      }
      case cudf::type_id::TIMESTAMP_MILLISECONDS: {
        // Generate timestamps in milliseconds from 2020-01-01 to 2024-12-31
        int64_t start_timestamp_ms = 1577836800000LL;  // 2020-01-01 00:00:00 UTC
        int64_t end_timestamp_ms   = 1735689600000LL;  // 2024-12-31 00:00:00 UTC
        columns.emplace_back(
          generate_random_numeric_column<int64_t>(start_timestamp_ms + i * 86400000LL,
                                                  end_timestamp_ms - i * 86400000LL,
                                                  m_rows,
                                                  stream,
                                                  mr));
        break;
      }
      case cudf::type_id::TIMESTAMP_MICROSECONDS: {
        // Generate timestamps in microseconds from 2020-01-01 to 2024-12-31
        int64_t start_timestamp_us = 1577836800000000LL;  // 2020-01-01 00:00:00 UTC
        int64_t end_timestamp_us   = 1735689600000000LL;  // 2024-12-31 00:00:00 UTC
        columns.emplace_back(
          generate_random_numeric_column<int64_t>(start_timestamp_us + i * 86400000000LL,
                                                  end_timestamp_us - i * 86400000000LL,
                                                  m_rows,
                                                  stream,
                                                  mr));
        break;
      }
      case cudf::type_id::TIMESTAMP_NANOSECONDS: {
        // Generate timestamps in nanoseconds from 2020-01-01 to 2024-12-31
        int64_t start_timestamp_ns = 1577836800000000000LL;  // 2020-01-01 00:00:00 UTC
        int64_t end_timestamp_ns   = 1735689600000000000LL;  // 2024-12-31 00:00:00 UTC
        columns.emplace_back(
          generate_random_numeric_column<int64_t>(start_timestamp_ns + i * 86400000000000LL,
                                                  end_timestamp_ns - i * 86400000000000LL,
                                                  m_rows,
                                                  stream,
                                                  mr));
        break;
      }
      case cudf::type_id::TIMESTAMP_DAYS: {
        // Generate timestamp days from 2020-01-01 to 2024-12-31
        int32_t start_days = 18262;  // Days since Unix epoch for 2020-01-01
        int32_t end_days   = 20088;  // Days since Unix epoch for 2024-12-31
        columns.emplace_back(generate_random_numeric_column<int32_t>(
          start_days + i, end_days - i, m_rows, stream, mr));
        break;
      }
      case cudf::type_id::DURATION_SECONDS: {
        // Generate duration in seconds (e.g., 0 to 1 year)
        int64_t max_duration = 365LL * 24 * 60 * 60;  // 1 year in seconds
        columns.emplace_back(
          generate_random_numeric_column<int64_t>(0, max_duration + i * 86400, m_rows, stream, mr));
        break;
      }
      case cudf::type_id::DURATION_MILLISECONDS: {
        // Generate duration in milliseconds (e.g., 0 to 1 year)
        int64_t max_duration_ms = 365LL * 24 * 60 * 60 * 1000;  // 1 year in ms
        columns.emplace_back(generate_random_numeric_column<int64_t>(
          0, max_duration_ms + i * 86400000LL, m_rows, stream, mr));
        break;
      }
      case cudf::type_id::DURATION_MICROSECONDS: {
        // Generate duration in microseconds (e.g., 0 to 1 year)
        int64_t max_duration_us = 365LL * 24 * 60 * 60 * 1000000;  // 1 year in us
        columns.emplace_back(generate_random_numeric_column<int64_t>(
          0, max_duration_us + i * 86400000000LL, m_rows, stream, mr));
        break;
      }
      case cudf::type_id::DURATION_NANOSECONDS: {
        // Generate duration in nanoseconds (e.g., 0 to 1 year)
        int64_t max_duration_ns = 365LL * 24 * 60 * 60 * 1000000000;  // 1 year in ns
        columns.emplace_back(generate_random_numeric_column<int64_t>(
          0, max_duration_ns + i * 86400000000000LL, m_rows, stream, mr));
        break;
      }
      case cudf::type_id::DURATION_DAYS: {
        // Generate duration in days (e.g., 0 to 365 days)
        int32_t max_duration_days = 365;  // 1 year in days
        columns.emplace_back(
          generate_random_numeric_column<int32_t>(0, max_duration_days + i, m_rows, stream, mr));
        break;
      }
      default: CUDF_FAIL("Unsupported data type");
    }
  }

  return std::make_unique<cudf::table>(std::move(columns));
}

void write_random_table(std::string const& directory,
                        int num_files,
                        cudf::size_type num_rows,
                        cudf::size_type num_cols,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  std::vector<cudf::data_type> types = {
    cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},       // Timestamp in seconds
    cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},  // Timestamp in milliseconds
    cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},          // Timestamp in days
    cudf::data_type{cudf::type_id::DURATION_SECONDS},        // Duration in seconds
    cudf::data_type{cudf::type_id::INT32},                   // Integer column for primary sorting
    cudf::data_type{cudf::type_id::FLOAT64},                 // Double precision floating point
    cudf::data_type{cudf::type_id::INT64},                   // Long integer
    cudf::data_type{cudf::type_id::FLOAT32},                 // Single precision floating point
    cudf::data_type{cudf::type_id::INT16}                    // Short integer
  };

  for (int i = 0; i < num_files; i++) {
    std::string filepath = construct_file_path(directory, i);
    auto partition       = generate_random_table(types, num_cols, num_rows, stream, mr);
    cudf::examples::write_parquet_file(filepath, partition->view(), stream);
  }
}

}  // namespace examples
}  // namespace cudf
