/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <span>
#include <vector>

namespace cudf {
//! Inner interfaces and implementations
namespace detail {
/**
 * @copydoc cudf::concatenate(std::span<column_view
 * const>,cuda::stream_ref,cudf::memory_resources)
 */
std::unique_ptr<column> concatenate(std::span<column_view const> columns_to_concat,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr);

/**
 * @copydoc cudf::concatenate(std::span<table_view
 * const>,cuda::stream_ref,cudf::memory_resources)
 */
std::unique_ptr<table> concatenate(std::span<table_view const> tables_to_concat,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr);

}  // namespace detail
}  // namespace cudf
