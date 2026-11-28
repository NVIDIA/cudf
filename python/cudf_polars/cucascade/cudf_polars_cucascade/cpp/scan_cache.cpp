/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scan_cache.hpp"

#include "cucascade_datasource.hpp"

#include <utility>

namespace cudf_polars_cucascade {

scan_cache::scan_cache(cucascade::io::scan_cache_config const& cfg) : _core(cfg) {}

std::unique_ptr<cudf::io::datasource> scan_cache::open(std::string const& path)
{
  return std::make_unique<cucascade_datasource>(_core.open(path));
}

void scan_cache::register_layout(std::string const& path,
                                 std::string const& table,
                                 std::vector<std::string> const& columns,
                                 std::span<std::uint32_t const> column_of_chunk,
                                 std::span<std::uint64_t const> offsets,
                                 std::span<std::uint64_t const> sizes)
{
  _core.register_layout(path, table, columns, column_of_chunk, offsets, sizes);
}

std::vector<cucascade::io::column_stats> scan_cache::stats() const { return _core.stats(); }

std::string scan_cache::summary() const { return _core.summary(); }

}  // namespace cudf_polars_cucascade
