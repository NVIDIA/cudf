/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <string>
#include <utility>

namespace cudf {
namespace detail {

/**
 * @brief JIT implementation of filter_join_indices
 *
 * Internal implementation that provides JIT-compiled filtering of join indices.
 * This is the implementation behind the public filter_join_indices_jit function.
 *
 * @param left The left table for predicate evaluation
 * @param right The right table for predicate evaluation
 * @param left_indices Device span of row indices in left table from join
 * @param right_indices Device span of row indices in right table from join
 * @param predicate_code String containing CUDA device code for predicate
 * @param join_kind The type of join operation
 * @param is_ptx Whether predicate_code contains PTX assembly
 * @param stream CUDA stream for operations
 * @param mr Device memory resource
 *
 * @return Pair of filtered left and right index vectors
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        std::string const& predicate_code,
                        join_kind join_kind,
                        bool is_ptx,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::filter_join_indices_jit(table_view const&, table_view const&,
 *   device_span<size_type const>, device_span<size_type const>,
 *   ast::expression const&, join_kind, rmm::cuda_stream_view,
 *   rmm::device_async_resource_ref)
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        ast::expression const& predicate,
                        join_kind join_kind,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
