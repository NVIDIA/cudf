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
 * Local-file I/O context; device 0 must be current for construction and I/O.
 * Retains up to 8 GiB in CUDA's default memory pool. The caller's stream/resource must
 * outlive this context and returned tables. Requires nonempty, NUL-free paths and at least
 * one flat, typed column; zero-row tables are supported.
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
   * Delegate to ndsh::write_vortex; see its type, naming, and file-completion requirements.
   * Positive chunk_rows sets the staging and physical CUDA-flat row-block size,
   * with byte coalescing and layout dictionaries disabled.
   */
  void write_vortex(std::string const& path,
                    cudf::table_view table,
                    std::vector<std::string> const& column_names,
                    cudf::size_type chunk_rows = 16 << 20) const;

  /**
   * GPU-read a local CUDA-compatible file into an owning cuDF table.
   * An empty column list selects all; otherwise names are literal, ordered, unique top-level fields.
   * Unknown names are errors. direct_io bypasses the data-page cache; metadata stays buffered.
   *
   * batch_rows=0 follows the file layout (recommended). Positive values request fixed row
   * counts, except the final batch; cross-block ranges may require unsupported CUDA Chunked
   * concatenation. Batch size and total rows must fit cudf::size_type.
   *
   * Decoded Arrow Device batches stay alive through materialization. On successful return,
   * the consumer stream is idle, including cuDF scratch frees; Vortex producer cleanup may
   * remain pending. Peak memory includes retained batches and the owning result.
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
