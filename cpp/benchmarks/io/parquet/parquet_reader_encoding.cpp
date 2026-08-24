/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Benchmarks decoding pages written with an explicitly requested column encoding. The writer's
// defaults never choose the DELTA_* encodings, so `parquet_read_decode` does not exercise their
// decode kernels; this benchmark covers them (with PLAIN as the baseline encoding).

namespace {

cudf::io::column_encoding retrieve_column_encoding_enum(std::string_view encoding_string)
{
  if (encoding_string == "PLAIN") { return cudf::io::column_encoding::PLAIN; }
  if (encoding_string == "DELTA_BINARY_PACKED") {
    return cudf::io::column_encoding::DELTA_BINARY_PACKED;
  }
  if (encoding_string == "DELTA_LENGTH_BYTE_ARRAY") {
    return cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY;
  }
  if (encoding_string == "DELTA_BYTE_ARRAY") { return cudf::io::column_encoding::DELTA_BYTE_ARRAY; }
  CUDF_FAIL("Unsupported column encoding: " + std::string(encoding_string));
}

// The writer only honours an encoding request on the schema node whose
// physical type matches, and for a LIST the encoded values live on the element
// node, not the top-level column, so this function pushes the request down to
// the leaves.
void set_encoding_recursive(cudf::io::column_in_metadata& col_meta,
                            cudf::io::column_encoding encoding)
{
  if (col_meta.num_children() == 0) {
    col_meta.set_encoding(encoding);
    return;
  }
  for (cudf::size_type i = 0; i < col_meta.num_children(); i++) {
    set_encoding_recursive(col_meta.child(i), encoding);
  }
}

data_profile make_profile(cudf::size_type cardinality, cudf::size_type run_length)
{
  return data_profile_builder().cardinality(cardinality).avg_run_length(run_length);
}

data_profile make_list_profile(cudf::size_type cardinality,
                               cudf::size_type run_length,
                               cudf::size_type nesting,
                               cudf::type_id leaf_type)
{
  return data_profile_builder()
    .cardinality(cardinality)
    .avg_run_length(run_length)
    .list_depth(nesting)
    .list_type(leaf_type);
}

std::unique_ptr<cudf::table> create_nested_table(std::vector<cudf::type_id> const& leaf_types,
                                                 size_t data_size,
                                                 cudf::size_type cardinality,
                                                 cudf::size_type run_length,
                                                 cudf::size_type nesting)
{
  auto const target_column_size = std::max<size_t>(data_size / leaf_types.size(), 1);
  std::vector<cudf::size_type> row_counts;
  row_counts.reserve(leaf_types.size());

  for (auto const leaf_type : leaf_types) {
    auto const profile = make_list_profile(0, run_length, nesting, leaf_type);
    row_counts.push_back(
      create_random_table({cudf::type_id::LIST}, table_size_bytes{target_column_size}, profile)
        ->num_rows());
  }

  auto const num_rows = *std::min_element(row_counts.cbegin(), row_counts.cend());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(leaf_types.size());
  for (std::size_t col_idx = 0; col_idx < leaf_types.size(); col_idx++) {
    // Keep the requested leaf cardinality, but avoid the top-level LIST distinct-row path when
    // small smoke runs have fewer rows than the cardinality axis. That path appends an INT32 list
    // suffix and is only type-compatible with INT32 leaves.
    auto const list_cardinality =
      cardinality == 0 ? 0 : std::min<cudf::size_type>(cardinality, num_rows - 1);
    auto const profile =
      make_list_profile(list_cardinality, run_length, nesting, leaf_types[col_idx]);
    columns.push_back(create_random_column(
      cudf::type_id::LIST, row_count{num_rows}, profile, static_cast<unsigned>(col_idx + 1)));
  }

  return std::make_unique<cudf::table>(std::move(columns));
}

void bench_read_encoding(nvbench::state& state, std::vector<cudf::type_id> const& d_types)
{
  auto const encoding    = retrieve_column_encoding_enum(state.get_string("encoding"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const nesting     = static_cast<cudf::size_type>(state.get_int64("nesting"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const leaf_types = cycle_dtypes(d_types, num_cols);
    auto const tbl =
      nesting > 0
        ? create_nested_table(leaf_types, data_size, cardinality, run_length, nesting)
        : create_random_table(
            leaf_types, table_size_bytes{data_size}, make_profile(cardinality, run_length));
    auto const view = tbl->view();

    cudf::io::table_input_metadata metadata(view);
    for (auto& col_meta : metadata.column_metadata) {
      set_encoding_recursive(col_meta, encoding);
    }

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .metadata(std::move(metadata))
        .compression(cudf::io::compression_type::NONE)
        .dictionary_policy(cudf::io::dictionary_policy::NEVER)
        .write_v2_headers(true);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, num_cols, source_sink, state);
}

}  // namespace

void BM_parquet_read_delta_binary(nvbench::state& state)
{
  bench_read_encoding(state, {cudf::type_id::INT32, cudf::type_id::INT64});
}

void BM_parquet_read_delta_string(nvbench::state& state)
{
  bench_read_encoding(state, {cudf::type_id::STRING});
}

NVBENCH_BENCH(BM_parquet_read_delta_binary)
  .set_name("parquet_read_delta_binary")
  .add_string_axis("encoding", {"PLAIN", "DELTA_BINARY_PACKED"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("nesting", {0, 1})
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_parquet_read_delta_string)
  .set_name("parquet_read_delta_string")
  .add_string_axis("encoding", {"PLAIN", "DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("nesting", {0, 1})
  .add_int64_axis("data_size", {512 << 20});
