/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Shared local-file setup and timing for Parquet/Vortex NDS-H comparisons.
 * Provides paired fixtures, projected and concurrent reads, validation, and OS page-cache control.
 */

#pragma once

#include "fixture_cache.hpp"
#include "parquet/parquet_io.hpp"
#include "utilities.hpp"
#include "vortex/vortex_io.hpp"

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <kvikio/file_utils.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ndsh {
class local_table_files {
 public:
  int64_t rows = 0;

  void write(std::string const& name, table_with_names const& table, vortex_io const& io)
  {
    CUDF_EXPECTS(!files_.contains(name), "Duplicate fixture table: " + name);
    auto const base = directory_.path() + name;
    table.to_parquet(base + ".parquet");
    io.write_vortex(base + ".vortex", table.table(), table.column_names());
    files_.emplace(name, std::make_pair(base + ".parquet", base + ".vortex"));
    rows += table.table().num_rows();
  }

  std::string const& path(std::string const& name, bool use_vortex) const
  {
    auto const& files = files_.at(name);
    return use_vortex ? files.second : files.first;
  }

  std::uintmax_t bytes(bool use_vortex) const
  {
    std::uintmax_t total = 0;
    for (auto const& [name, files] : files_) {
      total += std::filesystem::file_size(use_vortex ? files.second : files.first);
    }
    return total;
  }

  std::vector<std::string> paths(bool use_vortex) const
  {
    std::vector<std::string> result;
    result.reserve(files_.size());
    for (auto const& [name, files] : files_) {
      result.push_back(use_vortex ? files.second : files.first);
    }
    return result;
  }

 private:
  temp_directory directory_{"ndsh_local"};
  std::map<std::string, std::pair<std::string, std::string>> files_;
};

/** Read projected columns into an owning table; direct_io requires Vortex. */
inline std::unique_ptr<table_with_names> read_local_file(std::string const& path,
                                                         bool use_vortex,
                                                         vortex_io const& io,
                                                         std::vector<std::string> const& columns,
                                                         bool direct_io = false)
{
  CUDF_EXPECTS(use_vortex || !direct_io, "Direct I/O is supported only for Vortex");
  if (!use_vortex) { return read_parquet(cudf::io::source_info{path}, columns); }
  auto result = io.read_vortex(path, 0, columns, direct_io);
  std::vector<std::string> names;
  for (auto const& field : result.metadata.schema_info) {
    names.push_back(field.name);
  }
  return std::make_unique<table_with_names>(std::move(result.tbl), std::move(names));
}

// Parallel reads launch one worker per table in this small input set. read must be thread-safe
// and complete materialization before returning; future destruction joins workers on errors too.
template <typename Read>
std::vector<std::unique_ptr<table_with_names>> read_local_tables(
  std::vector<std::string> const& names,
  std::map<std::string, std::vector<std::string>> const& projections,
  Read&& read,
  bool parallel = false)
{
  std::vector<std::unique_ptr<table_with_names>> tables;
  tables.reserve(names.size());
  std::unique_ptr<cudf::ast::operation> const no_predicate;
  if (parallel) {
    int device;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    std::vector<std::future<std::unique_ptr<table_with_names>>> pending;
    pending.reserve(names.size());
    for (auto const& name : names) {
      pending.push_back(std::async(std::launch::async, [&, name, device] {
        CUDF_CUDA_TRY(cudaSetDevice(device));
        return read(name, projections.at(name), no_predicate);
      }));
    }
    for (auto& future : pending) {
      tables.push_back(future.get());
    }
  } else {
    for (auto const& name : names) {
      tables.push_back(read(name, projections.at(name), no_predicate));
    }
  }
  return tables;
}

struct local_options {
  bool use_vortex, read_only, cold, direct_io;

  local_options(nvbench::state& state, int query)
  {
    auto const format   = state.get_string("format");
    auto const workload = state.get_string("workload");
    auto const cache    = state.get_string("cache");
    auto const io       = state.get_string("io");
    auto const number   = std::to_string(query);
    CUDF_EXPECTS(cache == "warm" || cache == "cold", "Unknown cache mode");
    CUDF_EXPECTS(io == "buffered" || io == "direct", "Unknown I/O mode");
    CUDF_EXPECTS(format == "parquet" || format == "vortex", "Unknown Q" + number + " format");
    CUDF_EXPECTS(workload == "read" || workload == "q" + number,
                 "Unknown Q" + number + " workload");
    use_vortex = format == "vortex";
    read_only  = workload == "read";
    cold       = cache == "cold";
    direct_io  = io == "direct";
  }

  bool supported(nvbench::state& state) const
  {
    if (!direct_io || use_vortex) { return true; }
    state.skip("io=direct is supported only for Vortex");
    return false;
  }
};

inline void add_count(nvbench::state& state, char const* key, char const* name, int64_t value)
{
  auto& summary = state.add_summary(key);
  summary.set_string("name", name);
  summary.set_int64("value", value);
}

inline std::unique_ptr<table_with_names> take_result(std::unique_ptr<table_with_names>& result)
{
  return std::move(result);
}

inline void evict_file_pages(std::vector<std::string> const& paths)
{
  CUDF_EXPECTS(!paths.empty(), "Cold-cache benchmark requires at least one input file");
  for (auto const& path : paths) {
    kvikio::drop_file_page_cache(path);
  }
  for (auto const& path : paths) {
    auto const resident_pages = kvikio::get_page_cache_info(path).first;
    CUDF_EXPECTS(resident_pages == 0, "Input pages remain cached after eviction: " + path);
  }
}

inline void check_projection(cudf::table_view expected,
                             table_with_names const& actual,
                             std::vector<std::string> const& columns)
{
  CUDF_EXPECTS(actual.column_names() == columns &&
                 actual.table().num_columns() == expected.num_columns() &&
                 actual.table().num_rows() == expected.num_rows(),
               "Projected schema/row count mismatch");
  for (cudf::size_type i = 0; i < expected.num_columns(); ++i) {
    auto const lhs = expected.column(i);
    auto const rhs = actual.table().column(i);
    CUDF_EXPECTS(lhs.type() == rhs.type(), "Projected type mismatch");
    if (lhs.is_empty()) { continue; }
    auto equal = cudf::binary_operation(
      lhs, rhs, cudf::binary_operator::NULL_EQUALS, cudf::data_type{cudf::type_id::BOOL8});
    auto all = cudf::reduce(equal->view(),
                            *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::BOOL8});
    CUDF_EXPECTS(all->is_valid() && static_cast<cudf::numeric_scalar<bool> const&>(*all).value(),
                 "Projected values mismatch");
  }
}

// Setup only: buffered fixture reads were checked against the generated projections.
// For concurrent direct reads, also compare against a sequential read in the same mode.
inline void check_local_projection(std::string const& path,
                                   bool use_vortex,
                                   vortex_io const& io,
                                   std::vector<std::string> const& columns,
                                   table_with_names const& actual,
                                   bool direct_io = false)
{
  {
    auto expected = read_local_file(path, use_vortex, io, columns, direct_io);
    check_projection(expected->table(), actual, columns);
  }
  if (direct_io) {
    auto expected = read_local_file(path, use_vortex, io, columns, false);
    check_projection(expected->table(), actual, columns);
  }
}

// Construct after fixture generation; the logger must outlive the I/O context that borrows it.
class local_benchmark {
 public:
  local_benchmark(nvbench::state& state, local_table_files const& files, local_options options)
    : state_{state}, files_{files}, options_{options}
  {
    state_.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  }

  cuda::stream_ref const stream = cudf::get_default_stream();

  auto read(std::string const& name, std::vector<std::string> const& columns) const
  {
    return read_local_file(files_.path(name, options_.use_vortex),
                           options_.use_vortex,
                           io_,
                           columns,
                           options_.direct_io);
  }

  void check_projection(std::string const& name,
                        std::vector<std::string> const& columns,
                        table_with_names const& actual) const
  {
    check_local_projection(files_.path(name, options_.use_vortex),
                           options_.use_vortex,
                           io_,
                           columns,
                           actual,
                           options_.direct_io);
  }

  // Both callbacks return owners. Keep them alive through consumer synchronization, then release
  // them before the device sync that drains cleanup on independent Vortex producer streams.
  template <typename Read, typename Query>
  void exec(Read&& read,
            Query&& query,
            char const* file_description = "Total file size",
            char const* range_name       = nullptr)
  {
    if (!options_.cold) {
      auto inputs = read();
      CUDF_CUDA_TRY(cudaDeviceSynchronize());
    }
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
    memory_.reset_counters();
    state_.add_element_count(files_.rows, "Rows");
    state_.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
                [&](nvbench::launch&, auto& timer) {
                  if (options_.cold) { evict_file_pages(files_.paths(options_.use_vortex)); }
                  timer.start();
                  {
                    std::optional<cudf::benchmark::scoped_range> range;
                    if (range_name) { range.emplace(range_name); }
                    if (options_.read_only) {
                      auto inputs = read();
                      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
                    } else {
                      auto result = query();
                      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
                    }
                  }
                  CUDF_CUDA_TRY(cudaDeviceSynchronize());
                  timer.stop();
                });
    state_.add_buffer_size(files_.bytes(options_.use_vortex), "file_size", file_description);
    state_.add_buffer_size(memory_.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  }

 private:
  nvbench::state& state_;
  local_table_files const& files_;
  local_options const options_;
  cudf::memory_stats_logger memory_;
  vortex_io io_{stream.get()};
};

inline void check_file_projections(local_table_files const& files,
                                   std::string const& name,
                                   cudf::table_view expected,
                                   std::vector<std::string> const& columns,
                                   vortex_io const& io)
{
  for (bool use_vortex : {false, true}) {
    auto input = read_local_file(files.path(name, use_vortex), use_vortex, io, columns);
    check_projection(expected, *input, columns);
  }
}

// Setup only: the generator retains each full table until its callback has drained both readers.
// Reference dimensions are released before reading query results, just as the generated owners are.
template <typename Builder, typename Reference, typename Validate>
void make_reference_files(double scale_factor,
                          local_table_files& tables,
                          Reference& reference,
                          std::vector<std::string> const& names,
                          std::map<std::string, std::vector<std::string>> const& projections,
                          Validate&& validate)
{
  cuda::stream_ref const stream = cudf::get_default_stream();
  vortex_io io{stream.get()};
  {
    Builder builder;
    for_each_generated_table(
      scale_factor, names, [&](std::string const& name, table_with_names const& generated) {
        auto const& columns = projections.at(name);
        CUDF_EXPECTS(
          generated.table().num_columns() == static_cast<cudf::size_type>(ndsh_schema(name).size()),
          "Fixture requires full generated table: " + name);
        tables.write(name, generated, io);
        auto const projected = generated.select(columns);
        builder.add_table(name, projected, stream);
        check_file_projections(tables, name, projected, columns, io);
        CUDF_CUDA_TRY(cudaDeviceSynchronize());
      });
    reference = builder.finish();
  }
  for (bool use_vortex : {false, true}) {
    validate(
      [&](std::string const& name, std::vector<std::string> const& columns, auto const&...) {
        return read_local_file(tables.path(name, use_vortex), use_vortex, io, columns);
      },
      stream);
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace ndsh
