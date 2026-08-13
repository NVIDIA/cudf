/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/groupby.hpp>

#include <memory>
#include <tuple>
#include <vector>

namespace cudf::groupby::detail::hash {

/**
 * @brief Extract single pass aggregations from the given aggregation requests.
 *
 * During extraction, compound (i.e., multi-pass) aggregations will be replaced by their
 * corresponding single pass aggregations dependencies. For example, a MEAN aggregation will be
 * replaced by a SUM and a COUNT_VALID aggregation.
 *
 * For some single-pass aggregations, we also try to reduce overhead by forcing their results
 * columns to be non-nullable. For example, a SUM aggregation needed only as the intermediate result
 * for M2 aggregation will not need to have a nullmask to avoid the extra nullmask update and null
 * count computation overhead.
 *
 * @param requests The aggregation requests
 * @param stream The CUDA stream
 *
 * @return A tuple containing:
 *         - A table_view containing the input values columns for the single-pass aggregations,
 *         - A vector of aggregation kinds corresponding to each of these values columns,
 *         - A vector of aggregation objects corresponding to each of these values columns,
 *         - A vector of binary values indicating if the corresponding result will be used only
 *           as temporary, intermediate result for computing other compound aggregations, and
 *         - A boolean value indicating if there are any compound aggregations.
 */
std::tuple<table_view,
           cudf::detail::host_vector<aggregation::Kind>,
           std::vector<std::unique_ptr<aggregation>>,
           std::vector<int8_t>,
           bool>
extract_single_pass_aggs(std::span<aggregation_request const> requests,
                         rmm::cuda_stream_view stream);

/**
 * @brief Get simple aggregations from groupby aggregation
 *
 * @param agg The groupby aggregation
 * @param values_type The data type for the aggregation
 * @return A vector of aggregation kinds
 */
std::vector<aggregation::Kind> get_simple_aggregations(groupby_aggregation const& agg,
                                                       data_type values_type);
}  // namespace cudf::groupby::detail::hash
