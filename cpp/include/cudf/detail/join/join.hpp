/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace cudf {
namespace detail {

constexpr int DEFAULT_JOIN_CG_SIZE = 2;

/**
 * @brief Internal `filter_join_indices` accepting a precomputed output size.
 *
 * Same semantics as `cudf::filter_join_indices`. When `output_size` is provided it is used directly
 * to size the output, skipping the internal size-counting pass. The value must equal the size that
 * the function would otherwise compute (for example the result of `filter_join_indices_output_size`
 * for the same inputs); behavior is undefined otherwise.
 *
 * @param output_size Optional precomputed number of output rows; computed internally if not
 * provided
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices(table_view const& left,
                    table_view const& right,
                    device_span<size_type const> left_indices,
                    device_span<size_type const> right_indices,
                    ast::expression const& predicate,
                    join_kind join_kind,
                    std::optional<std::size_t> output_size,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
