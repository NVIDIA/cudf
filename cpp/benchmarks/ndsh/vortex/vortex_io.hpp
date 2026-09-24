/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Benchmark-private adapter between local Vortex files and owning cuDF tables.
 * Reuses a GPU reader session and delegates fixture writes to the host-staged CPU writer.
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ndsh {

/**
 * Device-0 I/O context reusing CUDA state and pinned staging.
 * Retains up to 8 GiB in CUDA's default memory pool between synchronized reads.
 * Caller stream/resource must outlive this context and returned tables.
 * Requires at least one flat, typed column; zero-row tables are supported.
 * Device 0 must be current for construction and I/O. Paths must be nonempty and NUL-free.
 */
class vortex_io {
 public:
  /** Create a Vortex CUDA session; the borrowed stream and resource control cuDF work only. */
  explicit vortex_io(cudaStream_t stream,
                     rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  ~vortex_io();
  vortex_io(vortex_io const&)            = delete;
  vortex_io& operator=(vortex_io const&) = delete;

  /**
   * Delegate to ndsh::write_vortex: write a CUDA-readable local file on CPU via host
   * Arrow chunks. Requires positive chunk_rows and the private writer's supported types.
   * Names are unique and NUL-free, one per column (or empty for generated names). chunk_rows also
   * sets the physical CUDA-flat row-block size and disables byte coalescing/layout dictionaries.
   * Partial string slices are compacted on device. Finalizes before return; failed writes may leave
   * an invalid file.
   */
  void write_vortex(std::string const& path,
                    cudf::table_view table,
                    std::vector<std::string> const& column_names,
                    cudf::size_type chunk_rows = 16 << 20) const;

  /**
   * GPU-read a local CUDA-compatible file into an owning cuDF table. An empty column
   * list selects all; otherwise scan literal top-level names in order. Unknown or
   * duplicate names are errors. Includes dictionary decode, Arrow Device import, one
   * final owning copy/concatenation, and consumer-stream completion before releasing
   * temporary inputs. On successful return, the consumer stream is idle, including
   * async frees of cuDF import scratch. Vortex release cleanup may remain queued on producer
   * streams. Zero batch_rows uses layout-derived splitting. Nonzero values request fixed row counts
   * (except the final batch), not layout-preserving caps; ranges crossing physical blocks may
   * require unsupported CUDA Chunked concatenation. Prefer zero for arbitrary files.
   * direct_io bypasses the OS page cache for data-plane reads; metadata remains buffered.
   * Peak memory includes retained Vortex batches
   * plus the owning result. batch_rows and total result rows must fit cudf::size_type.
   */
  [[nodiscard]] cudf::io::table_with_metadata read_vortex(
    std::string const& path,
    std::size_t batch_rows                  = 0,
    std::vector<std::string> const& columns = {},
    bool direct_io                          = false) const;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace ndsh
