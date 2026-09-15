/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf::jni {

/**
 * @brief Extract a 32-bit integer column from a column of 128-bit values.
 *
 * Given a 128-bit input column, a 32-bit integer column is returned corresponding to
 * the index of which 32-bit chunk of the original 128-bit values to extract.
 * 0 corresponds to the least significant chunk, and 3 corresponds to the most
 * significant chunk.
 *
 * A null input row will result in a corresponding null output row.
 *
 * @param col       Column of 128-bit values
 * @param dtype     Integer type to use for the output column (e.g.: UINT32 or INT32)
 * @param chunk_idx Index of the 32-bit chunk to extract
 * @param stream    CUDA stream to use
 * @return          A column containing the extracted 32-bit integer values
 */
std::unique_ptr<cudf::column> extract_chunk32(cudf::column_view const& col,
                                              cudf::data_type dtype,
                                              int chunk_idx,
                                              cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Reassemble a 128-bit column from four 64-bit integer columns with overflow detection.
 *
 * The 128-bit value is reconstructed by overlapping the 64-bit values by 32-bits. The least
 * significant 32-bits of the least significant 64-bit value are used directly as the least
 * significant 32-bits of the final 128-bit value, and the remaining 32-bits are added to the next
 * most significant 64-bit value. The lower 32-bits of that sum become the next most significant
 * 32-bits in the final 128-bit value, and the remaining 32-bits are added to the next most
 * significant 64-bit input value, and so on.
 *
 * A null input row will result in a corresponding null output row.
 *
 * @param chunks_table Table of four 64-bit integer columns with the columns ordered from least
 *                     significant to most significant. The last column must be an INT64 column.
 * @param output_type  The type to use for the resulting 128-bit value column
 * @param stream       CUDA stream to use
 * @return             Table containing a boolean column and a 128-bit value column of the
 *                     requested type. The boolean value will be true if an overflow was detected
 *                     for that row's value.
 */
std::unique_ptr<cudf::table> assemble128_from_sum(
  cudf::table_view const& chunks_table,
  cudf::data_type output_type,
  cuda::stream_ref stream = cudf::get_default_stream());

}  // namespace cudf::jni
