/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/join/join.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/buffer>

#include <memory>

namespace cudf::detail {
/**
 * @brief State a join object caches in the join_match_context it creates
 */
struct join_match_state {
  join_match_state(void const* owner, join_kind kind, std::size_t output_size)
    : owner{owner}, kind{kind}, output_size{output_size}
  {
  }
  virtual ~join_match_state() = default;

  /**
   * @brief Creates a context that owns `state`
   */
  static join_match_context make_context(table_view const& left,
                                         std::unique_ptr<rmm::device_uvector<size_type>> counts,
                                         std::unique_ptr<join_match_state> state)
  {
    return join_match_context{left, std::move(counts), std::move(state)};
  }

  /**
   * @brief Returns the state cached in `context`, or nullptr if the context was constructed
   * without one
   *
   * @throw std::invalid_argument If the state was created by another join object or join kind
   */
  template <typename State>
  static State const* get(join_match_context const& context, void const* owner, join_kind kind)
  {
    auto const* state = context._state.get();
    CUDF_EXPECTS(state == nullptr || (state->owner == owner && state->kind == kind),
                 "Match context belongs to a different join object or join kind",
                 std::invalid_argument);
    return static_cast<State const*>(state);
  }

  void const* owner;        ///< Join object that created the context
  join_kind kind;           ///< Join kind of the context
  std::size_t output_size;  ///< Number of row index pairs for the whole left table
};

/**
 * @brief Hash join state: the matching build-side group of each left row
 */
struct hash_join_match_state final : join_match_state {
  hash_join_match_state(void const* owner,
                        join_kind kind,
                        std::size_t output_size,
                        cuda::device_buffer<size_type> groups)
    : join_match_state{owner, kind, output_size}, groups{std::move(groups)}
  {
  }

  cuda::device_buffer<size_type> groups;  ///< Matching group of each left row
};
}  // namespace cudf::detail
