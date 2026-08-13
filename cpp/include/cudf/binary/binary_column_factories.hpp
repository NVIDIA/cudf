/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>

/**
 * @file
 * @brief Factory functions for `BINARY` columns.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief Constructs a `BINARY` column from offsets and a contiguous byte buffer.
 *
 * `offsets_column` must contain either `INT32` or `INT64` values and must not
 * contain nulls. Nonnegative, monotonically nondecreasing offsets whose final
 * value does not exceed `bytes_buffer.size()` are a caller precondition.
 *
 * Empty input is canonicalized to an empty `BINARY` column with no children.
 *
 * @param num_rows Number of binary values represented by the column
 * @param offsets_column Offsets with `num_rows + 1` elements
 * @param bytes_buffer Contiguous payload bytes
 * @param null_count Number of null rows
 * @param null_mask Row validity mask
 * @return Newly constructed `BINARY` column
 */
std::unique_ptr<column> make_binary_column(size_type num_rows,
                                           std::unique_ptr<column> offsets_column,
                                           rmm::device_buffer&& bytes_buffer,
                                           size_type null_count,
                                           rmm::device_buffer&& null_mask);

/**
 * @brief Creates an empty `BINARY` column.
 */
std::unique_ptr<column> make_empty_binary_column();

}  // namespace CUDF_EXPORT cudf
