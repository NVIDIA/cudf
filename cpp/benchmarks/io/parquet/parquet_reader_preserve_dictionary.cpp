/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Compare ordinary output, legacy string encoding, and preservation of existing dictionaries.
// The NEVER input cases measure the no-reencoding contract, including low-cardinality strings.
void parquet_preserve_dictionary(nvbench::state& state)
{
  auto const type     = static_cast<cudf::type_id>(state.get_int64("type"));
  auto const policy   = static_cast<cudf::io::dictionary_output_policy>(state.get_int64("output"));
  auto const encoding = state.get_int64("dictionary_input") != 0
                          ? cudf::io::dictionary_policy::ALWAYS
                          : cudf::io::dictionary_policy::NEVER;
  data_profile const profile = data_profile_builder().cardinality(state.get_int64("cardinality"));
  auto const input           = create_random_table({type}, row_count{1'000'000}, profile);
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  auto const write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), input->view())
      .dictionary_policy(encoding)
      .row_group_size_rows(100'000)
      .build();
  cudf::io::write_parquet(write_opts);
  auto const read_opts = cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
                           .dictionary_output_policy(policy)
                           .build();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_element_count(input->num_rows());
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto const result = cudf::io::read_parquet(read_opts);
    CUDF_EXPECTS(result.tbl->num_rows() == input->num_rows(), "Unexpected row count");
  });
}

NVBENCH_BENCH(parquet_preserve_dictionary)
  .add_int64_axis("type",
                  {static_cast<int64_t>(cudf::type_id::STRING),
                   static_cast<int64_t>(cudf::type_id::DECIMAL32),
                   static_cast<int64_t>(cudf::type_id::DECIMAL64),
                   static_cast<int64_t>(cudf::type_id::DECIMAL128)})
  .add_int64_axis("output",
                  {static_cast<int64_t>(cudf::io::dictionary_output_policy::DECODE),
                   static_cast<int64_t>(cudf::io::dictionary_output_policy::ENCODE),
                   static_cast<int64_t>(cudf::io::dictionary_output_policy::PRESERVE)})
  .add_int64_axis("dictionary_input", {0, 1})
  .add_int64_axis("cardinality", {10, 1000});
