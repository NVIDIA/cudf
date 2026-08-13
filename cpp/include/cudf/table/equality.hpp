/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
namespace CUDF_EXPORT cudf {

/**
 * @brief Check if two tables are equal.
 *
 * @ingroup table_classes
 *
 * Returns true if the input tables have the same number of rows, the same number of columns,
 * matching column types, and every row in `left` compares equal to the row at the same index in
 * `right`. Null equality is controlled by `nulls_equal`. Floating point NaN values compare equal.
 *
 * @throws cudf::logic_error if the tables contain `EMPTY` types.
 *
 * @param left The first table to compare
 * @param right The second table to compare
 * @param nulls_equal Flag to denote if null elements should be considered equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return true if the tables are equal, false otherwise
 */
[[nodiscard]] bool tables_equal(table_view const& left,
                                table_view const& right,
                                null_equality nulls_equal    = null_equality::EQUAL,
                                rmm::cuda_stream_view stream = cudf::get_default_stream());

}  // namespace CUDF_EXPORT cudf
