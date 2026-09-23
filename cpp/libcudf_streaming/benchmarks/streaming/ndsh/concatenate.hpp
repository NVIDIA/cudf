/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <memory>

namespace rapidsmpf::ndsh {

///< @brief Should the concatenation respect input ordering?
enum class ConcatOrder : bool {
  DONT_CARE,  ///< No, we don't need ordering
  LINEARIZE,  ///< Yes, maintain input ordering
};

/**
 * @brief Concatenate all table chunks from an input channel.
 *
 * @param ctx Streaming context.
 * @param ch_in Input channel of `table_chunk`s.
 * @param ch_out Output channel of concatenated chunks, contains at most one message.
 * @param order Do we care about maintaining the input ordering?
 *
 * @return Coroutine representing the concatenation.
 */
streaming::Actor concatenate(std::shared_ptr<streaming::Context> ctx,
                             std::shared_ptr<streaming::Channel> ch_in,
                             std::shared_ptr<streaming::Channel> ch_out,
                             ConcatOrder order = ConcatOrder::DONT_CARE);

}  // namespace rapidsmpf::ndsh
