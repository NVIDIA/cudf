/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {

///< @brief Description of aggregation requests on a given column
struct groupby_request {
  cudf::size_type column_idx;  ///< Index of column in input table to aggregate
  std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>>
    requests;  ///< Functions to generate aggregations to perform on the column
};

/**
 * @brief Perform a chunkwise grouped aggregation.
 *
 * @note Grouped chunks are not further grouped together.
 *
 * @param ctx Streaming context.
 * @param ch_in `table_chunk`s to aggregate
 * @param ch_out Output channel of grouped `table_chunk`s
 * @param keys Column indices of the key columns in the input channel.
 * @param requests Vector of aggregation requests referencing columns in the input
 * channel.
 * @param null_policy How nulls in the key columns are treated.
 *
 * @return Coroutine representing the completion of the aggregation.
 */
streaming::Actor chunkwise_group_by(std::shared_ptr<streaming::Context> ctx,
                                    std::shared_ptr<streaming::Channel> ch_in,
                                    std::shared_ptr<streaming::Channel> ch_out,
                                    std::vector<cudf::size_type> keys,
                                    std::vector<groupby_request> requests,
                                    cudf::null_policy null_policy

);
}  // namespace rapidsmpf::ndsh
