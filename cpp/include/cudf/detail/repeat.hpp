/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::repeat(table_view const&, column_view const&,
 * cuda::stream_ref, cudf::memory_resources)
 */
std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr);

/**
 * @copydoc cudf::repeat(table_view const&, size_type,
 * cuda::stream_ref, cudf::memory_resources)
 */
std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr);

}  // namespace detail
}  // namespace cudf
