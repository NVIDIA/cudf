/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_writer.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {

rapidsmpf::streaming::Actor write_parquet(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                          std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                          cudf::io::sink_info sink,
                                          std::vector<std::string> column_names)
{
  streaming::ShutdownAtExit c{ch_in};
  co_await ctx->executor()->schedule();
  auto builder = cudf::io::chunked_parquet_writer_options::builder(sink);
  auto msg     = co_await ch_in->receive();
  RAPIDSMPF_EXPECTS(!msg.empty(), "Writing from empty channel not supported");
  auto chunk    = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
  auto table    = chunk.table_view();
  auto metadata = cudf::io::table_input_metadata(table);
  CudaEvent event;
  auto write_stream = chunk.stream();
  RAPIDSMPF_EXPECTS(column_names.size() == metadata.column_metadata.size(),
                    "Mismatching number of column names and chunk columns");
  for (std::size_t i = 0; i < column_names.size(); i++) {
    metadata.column_metadata[i].set_name(column_names[i]);
  }
  builder      = builder.metadata(metadata);
  auto options = builder.build();
  auto writer  = cudf::io::chunked_parquet_writer(options, write_stream);
  writer.write(table);
  while (true) {
    msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    table = chunk.table_view();
    RAPIDSMPF_EXPECTS(static_cast<std::size_t>(table.num_columns()) == column_names.size(),
                      "Mismatching number of column names and chunk columns");
    cuda_stream_join(write_stream, chunk.stream(), &event);
    writer.write(table);
    cuda_stream_join(chunk.stream(), write_stream, &event);
  }
  writer.close();
}
}  // namespace rapidsmpf::ndsh
