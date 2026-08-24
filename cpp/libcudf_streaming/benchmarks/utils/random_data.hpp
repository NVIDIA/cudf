/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <cstddef>
#include <cstdint>

/**
 * @brief The type of random data to generate.
 */
using random_data_t = std::int32_t;

/**
 * @brief The lower bound of the size of a random table.
 *
 * @param ncolumns The number of columns in the table.
 * @param nrows The number of rows in the table.
 */
std::size_t constexpr random_table_size_lower_bound(cudf::size_type ncolumns, cudf::size_type nrows)
{
  return rapidsmpf::safe_cast<std::size_t>(ncolumns) * rapidsmpf::safe_cast<std::size_t>(nrows) *
         sizeof(random_data_t);
}

/**
 * @brief Generates a random numeric device vector (std::int32_t).
 *
 * Creates a device vector with random integer values uniformly distributed in
 * the range `[min_val, max_val]`.
 *
 * @param nelem Number of elements in the generated vector.
 * @param min_val Minimum value (inclusive) for the random data.
 * @param max_val Maximum value (inclusive) for the random data.
 * @param stream CUDA stream to use for memory and kernel operations.
 * @param mr Device memory resource for allocating the device vector.
 * @return A unique pointer to the generated device vector.
 *
 * @note The function uses the specified CUDA stream for asynchronous operations.
 */
rmm::device_uvector<std::int32_t> random_device_vector(std::size_t nelem,
                                                       std::int32_t min_val,
                                                       std::int32_t max_val,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr);

/**
 * @brief Generates a random numeric column (std::int32_t).
 *
 * Creates a cuDF column with random integer values uniformly distributed in the range
 * `[min_val, max_val]`.
 *
 * @param nrows Number of rows in the generated column.
 * @param min_val Minimum value (inclusive) for the random data.
 * @param max_val Maximum value (inclusive) for the random data.
 * @param stream CUDA stream to use for memory and kernel operations.
 * @param mr Device memory resource for allocating the column.
 * @return A unique pointer to the generated cuDF column.
 *
 * @note The function uses the specified CUDA stream for asynchronous operations.
 */
std::unique_ptr<cudf::column> random_column(cudf::size_type nrows,
                                            std::int32_t min_val,
                                            std::int32_t max_val,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Generates a random numeric table (std::int32_t).
 *
 * Creates a cuDF table consisting of multiple columns with random integer values, each
 * uniformly distributed in the range `[min_val, max_val]`.
 *
 * @param ncolumns Number of columns in the generated table.
 * @param nrows Number of rows in each column of the table.
 * @param min_val Minimum value (inclusive) for the random data.
 * @param max_val Maximum value (inclusive) for the random data.
 * @param stream CUDA stream to use for memory and kernel operations.
 * @param mr Device memory resource for allocating the table.
 * @return A cuDF table containing the generated random columns.
 *
 * @note Each column in the table will have the same number of rows and data distribution.
 */
cudf::table random_table(cudf::size_type ncolumns,
                         cudf::size_type nrows,
                         std::int32_t min_val,
                         std::int32_t max_val,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr);

/**
 * @brief Fill a rapidsmpf buffer with random data (std::int32_t).
 *
 * Using buffer's CUDA stream.
 *
 * @param buffer The buffer to fill.
 * @param mr Device memory resource for allocating temporary random data.
 *
 * @throws std::invalid_argument if the memory type of `buffer` isn't supported.
 */
void random_fill(rapidsmpf::Buffer& buffer, rmm::device_async_resource_ref mr);
