/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for computing quantiles and percentiles of columns and tables.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup column_quantiles
 * @{
 */

/**
 * @brief Computes quantiles with interpolation.
 *
 * Computes the specified quantiles by interpolating values between which they
 * lie, using the interpolation strategy specified in `interp`.
 *
 * @param[in] input           Column from which to compute quantile values
 * @param[in] q               Specified quantiles in range [0, 1]
 * @param[in] interp          Strategy used to select between values adjacent to
 *                            a specified quantile.
 * @param[in] ordered_indices Column containing the sorted order of `input`.
 *                            If the column is empty, all `input` values are
 *                            used in existing order. Indices must be in range
 *                            [0, `input.size()`), but are not required to be
 *                            unique. Values not indexed by this column will be
 *                            ignored.
 * @param[in] exact           If true, returns doubles.
 *                            If false, returns same type as input.
 * @param[in] stream          CUDA stream used for device memory operations and kernel launches
 * @param[in] mr              Device memory resource used to allocate the returned column's device
 memory
 * @returns Column of specified quantiles, with nulls for indeterminable values
 */

std::unique_ptr<column> quantile(
  column_view const& input,
  std::vector<double> const& q,
  interpolation interp               = interpolation::LINEAR,
  column_view const& ordered_indices = {},
  bool exact                         = true,
  rmm::cuda_stream_view stream       = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr  = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the rows of the input corresponding to the requested quantiles.
 *
 * Quantiles are cut points that divide the range of a dataset into continuous
 * intervals. e.g: quartiles are the three cut points that divide a dataset into
 * four equal-sized groups. See https://en.wikipedia.org/wiki/Quantile
 *
 * The indices used to gather rows are computed by interpolating between the
 * index on either side of the desired quantile. Since some columns may be
 * non-arithmetic, interpolation between rows is limited to non-arithmetic
 * strategies.
 *
 * Non-arithmetic interpolation strategies include HIGHER, LOWER, and NEAREST.
 *
 * quantiles `<= 0` correspond to row `0`. (first)
 * quantiles `>= 1` correspond to row `input.size() - 1`. (last)
 *
 * @param input           Table used to compute quantile rows
 * @param q               Desired quantiles in range [0, 1]
 * @param interp          Strategy used to select between the two rows on either
                          side of the desired quantile.
 * @param is_input_sorted Indicates if the input has been pre-sorted
 * @param column_order    The desired sort order for each column
 * @param null_precedence The desired order of null compared to other elements
 * @param stream          CUDA stream used for device memory operations and kernel launches
 * @param mr              Device memory resource used to allocate the returned table's device memory
 *
 * @returns Table of specified quantiles, with nulls for indeterminable values
 * @throws cudf::logic_error if `interp` is an arithmetic interpolation strategy
 * @throws cudf::logic_error if `input` is empty
 */
std::unique_ptr<table> quantiles(
  table_view const& input,
  std::vector<double> const& q,
  interpolation interp                           = interpolation::NEAREST,
  cudf::sorted is_input_sorted                   = sorted::NO,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Calculate approximate percentiles on an input tdigest column.
 *
 * tdigest (https://arxiv.org/pdf/1902.04023) columns are produced specifically
 * by the TDIGEST and MERGE_TDIGEST aggregations.  These columns represent
 * compressed representations of a very large input data set that can be
 * queried for quantile information.
 *
 * Produces a LIST column where each row `i` represents output from querying the
 * corresponding tdigest from `input` row `i`. The length of each output list
 * is the number of percentages specified in `percentages`.
 *
 * @param input           tdigest input data. One tdigest per row
 * @param percentiles     Desired percentiles in range [0, 1]
 * @param stream          CUDA stream used for device memory operations and kernel launches
 * @param mr              Device memory resource used to allocate the returned column's device
 * memory
 *
 * @throws cudf::logic_error if `input` is not a valid tdigest column.
 * @throws cudf::logic_error if `percentiles` is not a FLOAT64 column.
 *
 * @returns LIST Column containing requested percentile values as FLOAT64
 */
std::unique_ptr<column> percentile_approx(
  tdigest::tdigest_column_view const& input,
  column_view const& percentiles,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
