/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/quantiles.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::quantile()
 */
std::unique_ptr<column> quantile(column_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 column_view const& ordered_indices,
                                 bool exact,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::quantiles()
 */
std::unique_ptr<table> quantiles(table_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 cudf::sorted is_input_sorted,
                                 std::vector<order> const& column_order,
                                 std::vector<null_order> const& null_precedence,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::percentile_approx(tdigest_column_view const&, column_view const&,
 * rmm::device_async_resource_ref)
 */
std::unique_ptr<column> percentile_approx(tdigest::tdigest_column_view const& input,
                                          column_view const& percentiles,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
