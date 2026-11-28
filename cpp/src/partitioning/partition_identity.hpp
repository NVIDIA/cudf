// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf::detail::partition_identity {

/** @brief Whether the metadata identifies an all-null fixed-width or STRING column. */
bool is_uniform(column_view const& input);

/** @brief Construct an independent all-null column of the input type. */
std::unique_ptr<column> uniform_column(column_view const& input,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr);

/** @brief Deep-copy a table whose partition path preserves row order. */
std::unique_ptr<table> copy(table_view const& input,
                            cuda::stream_ref stream,
                            cudf::memory_resources mr);

}  // namespace cudf::detail::partition_identity
