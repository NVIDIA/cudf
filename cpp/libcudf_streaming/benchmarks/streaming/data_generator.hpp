/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../utils/random_data.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::streaming::actor {

using cudf_streaming::table_chunk;

/**
 * @brief Asynchronously generates and sends a sequence of random numeric tables.
 *
 * This is a streaming version of `rapidsmpf::random_table_generator` that operates on
 * table chunks using channels.
 *
 * It creates a specified number of cuDF tables with random `std::int32_t` values, each
 * consisting of `ncolumns` columns and `nrows` rows. The values are uniformly
 * distributed in the range [`min_val`, `max_val`]. Each generated table is wrapped
 * in a `table_chunk` and sent to the provided output channel in streaming fashion.
 *
 * @param ctx The actor context to use.
 * @param stream The CUDA stream on which to create the random tables. TODO: use a pool
 * of CUDA streams.
 * @param ch_out Output channel to which generated `table_chunk` objects are sent.
 * @param num_blocks Number of tables (chunks) to generate and send.
 * @param ncolumns Number of columns per generated table.
 * @param nrows Number of rows per column in each table.
 * @param min_val Minimum inclusive value for the generated random integers.
 * @param max_val Maximum inclusive value for the generated random integers.
 * @return A streaming actor that completes once all random tables have been generated
 * and sent, and the channel has been drained.
 */
inline Actor random_table_generator(std::shared_ptr<Context> ctx,
                                    rmm::cuda_stream_view stream,
                                    std::shared_ptr<Channel> ch_out,
                                    std::uint64_t num_blocks,
                                    cudf::size_type ncolumns,
                                    cudf::size_type nrows,
                                    std::int32_t min_val,
                                    std::int32_t max_val)
{
  ShutdownAtExit c{ch_out};
  co_await ctx->executor()->schedule();
  auto nbytes = rapidsmpf::safe_cast<std::size_t>(ncolumns) *
                rapidsmpf::safe_cast<std::size_t>(nrows) * sizeof(std::int32_t);
  for (std::uint64_t seq = 0; seq < num_blocks; ++seq) {
    auto res = ctx->br()->reserve_device_memory_and_spill(nbytes, AllowOverbooking::NO);
    co_await ch_out->send(
      to_message(seq,
                 std::make_unique<table_chunk>(
                   std::make_unique<cudf::table>(random_table(
                     ncolumns, nrows, min_val, max_val, stream, ctx->br()->device_mr())),
                   stream)));
  }
  co_await ch_out->drain(ctx->executor());
}

}  // namespace rapidsmpf::streaming::actor
