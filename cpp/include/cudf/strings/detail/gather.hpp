/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf::strings::detail {

/**
 * @brief Gather a complete permutation of non-null STRING columns in one batch.
 *
 * The input must have no nulls and at most 65,535 columns. Non-empty columns must use INT32
 * offsets. Slices and all-valid masks are supported. The output has no null mask and follows the
 * large-strings policy, including promotion to INT64 offsets when needed. The map must be a
 * permutation of `[0, input.num_rows())`, with no repeated or omitted rows. Map values are not
 * checked. This preserves each column's character count, allowing a single batched size readback.
 *
 * @note Batching can reduce synchronization and kernel-launch overhead, but this function is not
 * guaranteed to outperform other gather or materialization paths for every workload. Performance
 * depends on row and column counts, string lengths, and permutation locality. Callers should use
 * workload-specific heuristics to select this function or an alternative path; this utility does
 * not make that selection.
 *
 * @param input Table of non-null STRING columns with INT32 offsets
 * @param gather_map Device span containing the complete permutation
 * @param stream CUDA stream used for device operations
 * @param mr Memory resources for temporary allocations and the returned table
 * @return Table containing the permuted STRING columns
 * @throw cudf::logic_error if the permutation size differs from the input row count
 * @throw cudf::logic_error if the input has more than 65,535 columns
 * @throw cudf::logic_error if any input column is not STRING or contains nulls
 * @throw cudf::logic_error if any non-empty input column does not use INT32 offsets
 */
std::unique_ptr<table> gather_non_null_permutation(table_view const& input,
                                                   device_span<size_type const> gather_map,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr);

}  // namespace cudf::strings::detail
