/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream_ref>

#include <string>
#include <vector>

namespace ndsh {

/**
 * @file
 * @brief Private Vortex file writer for NDS-H benchmarks.
 */

/**
 * @brief Local-file sink, borrowed GPU table, and staging options for `write_vortex()`.
 */
struct vortex_writer_options {
  cudf::io::sink_info sink;
  cudf::table_view table;
  /// One unique, NUL-free name per column, or empty to generate `_col0`, `_col1`, etc.
  std::vector<std::string> names{};
  /// Positive staging/physical-block row bound, not a byte or total-memory bound.
  /// The original GPU input remains resident.
  cudf::size_type rows_per_chunk{16 << 20};
};

/**
 * @brief Write a GPU table to a local Vortex file using CPU compression and encoding.
 *
 * Requires Linux, the private Vortex target, and current CUDA device 0. The experimental
 * CUDA-flat layout has no cross-version compatibility guarantee.
 *
 * Accepts only integers, floats, booleans, strings, decimal32/64/128, and day-resolution
 * timestamps. Nulls, slices, and zero rows are supported; at least one column is required.
 * The sink must be one nonempty, NUL-free local path, not a buffer, custom sink, or URI.
 *
 * Stages row-bounded host Arrow chunks, compacting sliced strings on device. `stream` orders
 * staging; `mr` controls device staging buffers, not host/Vortex allocations. Slicing and string
 * compaction scratch use the current device resource. CUDA pool retention is unchanged.
 *
 * Blocks through staging and file finalization; input and resource must outlive the call.
 * Overwrites existing files without atomic replacement or fsync; errors may leave a partial file.
 *
 * @throws cudf::logic_error If options, types or the current device are unsupported
 * @throws cudf::cuda_error If a CUDA operation fails
 * @throws std::runtime_error If Vortex conversion, compression or file I/O fails
 *
 * @param options Destination, borrowed table, names and chunk size
 * @param stream CUDA stream used for cuDF staging
 * @param mr Device resource used for cuDF staging buffers (not all internal scratch allocations)
 */
void write_vortex(vortex_writer_options const& options,
                  cuda::stream_ref stream           = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace ndsh
