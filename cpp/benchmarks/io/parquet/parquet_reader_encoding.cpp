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

void bench_read_encoding(nvbench::state& state, std::vector<cudf::type_id> const& d_types)
{
  auto const encoding    = retrieve_column_encoding_enum(state.get_string("encoding"));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_types, num_cols),
      table_size_bytes{data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::table_input_metadata metadata(view);
    for (auto& col_meta : metadata.column_metadata) {
      col_meta.set_encoding(encoding);
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
  .add_int64_axis("data_size", {512 << 20});

NVBENCH_BENCH(BM_parquet_read_delta_string)
  .set_name("parquet_read_delta_string")
  .add_string_axis("encoding", {"PLAIN", "DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20});
