/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>

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
 * @param udf The predicate as CUDA source, PTX (NUL-terminated), or an LTO binary
 * @param udf_type The source language or LTO binary type of `udf`
 * @param is_null_aware Whether the predicate receives row inputs as optional values
 * @param user_data Caller-owned device data passed to the predicate
 * @param join_kind The type of join operation
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
                        std::span<uint8_t const> udf,
                        std::variant<udf_source_type, lto_binary_type> udf_type,
                        null_aware is_null_aware,
                        std::optional<void*> user_data,
                        join_kind join_kind,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr);

/**
 * @copydoc cudf::filter_join_indices_jit(table_view const&, table_view const&,
 *   device_span<size_type const>, device_span<size_type const>,
 *   ast::expression const&, join_kind, lto_fragments, cuda::stream_ref, memory_resources)
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(cudf::table_view const& left,
                        cudf::table_view const& right,
                        cudf::device_span<size_type const> left_indices,
                        cudf::device_span<size_type const> right_indices,
                        ast::expression const& predicate,
                        cudf::lto_fragments fragments,
                        join_kind join_kind,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr);

}  // namespace detail
}  // namespace cudf
