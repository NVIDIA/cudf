/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>

#include <cucascade/io/column_admission.hpp>
#include <cucascade/io/scan_cache.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cudf_polars_cucascade {

/**
 * @brief cuCascade's scan cache, with @ref open returning a @c cudf::io::datasource.
 *
 * Owns the pinned host cache, the REST io context, the column admission policy and the cache of
 * opened objects (all in cuCascade's @c scan_cache). Creating it pins the cache, which is the whole
 * cost of creating it.
 */
class scan_cache {
 public:
  explicit scan_cache(cucascade::io::scan_cache_config const& cfg);

  /**
   * @brief Open an object-store object for reading.
   *
   * The first open of a path resolves its size, version tag and footer with one request; later
   * opens of it need none.
   */
  [[nodiscard]] std::unique_ptr<cudf::io::datasource> open(std::string const& path);

  /** @brief Register the column chunks of one file with the admission policy. */
  void register_layout(std::string const& path,
                       std::string const& table,
                       std::vector<std::string> const& columns,
                       std::span<std::uint32_t const> column_of_chunk,
                       std::span<std::uint64_t const> offsets,
                       std::span<std::uint64_t const> sizes);

  /** @brief Per-column admission accounting. */
  [[nodiscard]] std::vector<cucascade::io::column_stats> stats() const;

  /** @brief Counters of the prefetching cache (reads, hits, misses, evictions) as text. */
  [[nodiscard]] std::string summary() const;

 private:
  cucascade::io::scan_cache _core;
};

}  // namespace cudf_polars_cucascade
