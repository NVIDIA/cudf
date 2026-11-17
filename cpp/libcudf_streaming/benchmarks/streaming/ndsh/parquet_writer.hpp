/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {

/**
 * @brief Write chunks in a channel to an output sink
 *
 * @param ctx Streaming context
 * @param ch_in Input channel of `table_chunk`s
 * @param sink Sink to write into
 * @param column_names Names of the columns to add to the parquet metadata
 *
 * @return Coroutine representing the write
 */
[[nodiscard]] rapidsmpf::streaming::Actor write_parquet(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  cudf::io::sink_info sink,
  std::vector<std::string> column_names);
}  // namespace rapidsmpf::ndsh
