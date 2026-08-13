/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

/**
 * @file
 * @brief API for counting consecutive groups of equivalent rows in a column
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_reduction
 * @{
 */

/**
 * @brief Count the number of consecutive groups of equivalent rows in a column.
 *
 * If `null_handling` is null_policy::EXCLUDE and `nan_handling` is  nan_policy::NAN_IS_NULL, both
 * `NaN` and `null` values are ignored. If `null_handling` is null_policy::EXCLUDE and
 * `nan_handling` is nan_policy::NAN_IS_VALID, only `null` is ignored, `NaN` is considered in count.
 *
 * `null`s are handled as equal.
 *
 * @param[in] input The column_view whose consecutive groups of equivalent rows will be counted
 * @param[in] null_handling flag to include or ignore `null` while counting
 * @param[in] nan_handling flag to consider `NaN==null` or not
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of consecutive groups of equivalent rows in the column
 */
cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Count the number of consecutive groups of equivalent rows in a table.
 *
 * @param[in] input Table whose consecutive groups of equivalent rows will be counted
 * @param[in] nulls_equal flag to denote if null elements should be considered equal
 *            nulls are not equal if null_equality::UNEQUAL.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of consecutive groups of equivalent rows in the column
 */
cudf::size_type unique_count(table_view const& input,
                             null_equality nulls_equal    = null_equality::EQUAL,
                             rmm::cuda_stream_view stream = cudf::get_default_stream());

/** @} */

}  // namespace CUDF_EXPORT cudf
