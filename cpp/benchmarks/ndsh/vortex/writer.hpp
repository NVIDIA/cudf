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
 * @brief Settings for `write_vortex()`.
 *
 * The table is borrowed and must remain valid until `write_vortex()` returns.
 * Only a single local-file sink is currently supported.
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
 * @brief Write a GPU-resident table to a local Vortex file.
 *
 * Requires the optional NDS-H Vortex benchmark target on Linux.
 * The current implementation requires CUDA device 0 to be current and uses Vortex's experimental
 * CUDA-flat layout. Files are tied to the pinned Vortex revision; cross-version compatibility of
 * this layout is not guaranteed.
 *
 * Supported columns are signed/unsigned integers, floating point, booleans, strings,
 * decimal32/64/128, and day-resolution timestamps. Nulls, slices and zero-row tables are supported;
 * at least one column is required. Nested, dictionary, duration and other timestamp types are not
 * supported. Only `cudf::io::sink_info` containing one nonempty, NUL-free local path is accepted;
 * buffers, custom sinks and remote URIs are not supported.
 *
 * Input is copied to host Arrow in row-bounded chunks (sliced strings are compacted on device),
 * then compressed and encoded on the CPU. This is not a GPU compression API. `stream` orders cuDF
 * staging work; `mr` controls cuDF device staging buffers, not host or Vortex allocations.
 * Scratch allocations inside cuDF slicing and string compaction use the current device resource.
 * No CUDA memory-pool retention policy is changed by this function.
 *
 * The function blocks until cuDF staging and file finalization complete. It overwrites an existing
 * file and does not guarantee durable storage (no fsync) or atomic replacement. A failed write can
 * leave a partial/invalid file. Input and the supplied resource must remain alive until return.
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
