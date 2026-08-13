/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <cstddef>
#include <memory>

namespace cudf_streaming {

/**
 * @brief Filter ast expression with lifetime/stream management.
 */
struct filter {
  rmm::cuda_stream_view stream;      ///< Stream the filter's scalars are valid on.
  cudf::ast::expression& filter;     ///< Filter expression.
  rapidsmpf::OwningWrapper owner{};  ///< Owner of all objects in the filter.
};

namespace actor {
/**
 * @brief Asynchronously read parquet files into an output channel.
 *
 * @note This is a collective operation, all ranks named by the execution context's
 * communicator will participate. All ranks must specify the same set of options.
 * Behaviour is undefined if a `read_parquet` actor appears only on a subset of the ranks
 * named by the communicator, or the options differ between ranks.
 *
 * @param ctx The execution context to use.
 * @param comm Communicator for distributing files across ranks.
 * @param ch_out Channel to which `table_chunk`s are sent.
 * @param num_producers Number of concurrent producer tasks.
 * @param options Template reader options. The files within will be picked apart and used
 * to reconstruct new options for each read chunk. The options should therefore specify
 * the read options "as-if" one were reading the whole input in one go.
 * @param num_rows_per_chunk Target (maximum) number of rows any sent `table_chunk` should
 * have.
 * @param filter Optional filter expression to apply to the read.
 *
 * @return Streaming actor representing the asynchronous read.
 */
rapidsmpf::streaming::Actor read_parquet(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                         std::shared_ptr<rapidsmpf::Communicator> comm,
                                         std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                         std::size_t num_producers,
                                         cudf::io::parquet_reader_options options,
                                         // TODO: use byte count, not row count?
                                         cudf::size_type num_rows_per_chunk,
                                         std::unique_ptr<filter> filter = nullptr);
}  // namespace actor
}  // namespace cudf_streaming
