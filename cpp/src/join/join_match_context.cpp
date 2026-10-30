/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "join_match_state.hpp"

// The context's own members still use the deprecated fields.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace cudf {
join_match_context::join_match_context(table_view const& left_table,
                                       std::unique_ptr<rmm::device_uvector<size_type>> match_counts)
  : _left_table{left_table}, _match_counts{std::move(match_counts)}
{
}

join_match_context::join_match_context(table_view const& left_table,
                                       std::unique_ptr<rmm::device_uvector<size_type>> match_counts,
                                       std::unique_ptr<detail::join_match_state> state)
  : _left_table{left_table}, _match_counts{std::move(match_counts)}, _state{std::move(state)}
{
}

join_match_context::join_match_context(join_match_context&&) noexcept            = default;
join_match_context& join_match_context::operator=(join_match_context&&) noexcept = default;
join_match_context::~join_match_context()                                        = default;

table_view join_match_context::left_table() const { return _left_table; }

device_span<size_type const> join_match_context::match_counts() const
{
  CUDF_EXPECTS(
    _match_counts != nullptr, "join_match_context is missing match counts", std::invalid_argument);
  return *_match_counts;
}

std::size_t join_match_context::output_size() const
{
  CUDF_EXPECTS(_state != nullptr,
               "join_match_context was not created by a join object",
               std::invalid_argument);
  return _state->output_size;
}
}  // namespace cudf

#pragma GCC diagnostic pop
