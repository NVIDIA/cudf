/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/join/asof_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <memory>

namespace cudf::detail::row {
namespace equality {
struct preprocessed_table;
}
namespace lexicographic {
struct preprocessed_table;
}
}  // namespace cudf::detail::row

namespace cudf::experimental::detail {

/**
 * @brief Implementation of an as-of join with a reusable right side.
 */
class asof_join_impl {
 public:
  asof_join_impl()                                 = delete;
  ~asof_join_impl()                                = default;
  asof_join_impl(asof_join_impl const&)            = delete;
  asof_join_impl(asof_join_impl&&)                 = delete;
  asof_join_impl& operator=(asof_join_impl const&) = delete;
  asof_join_impl& operator=(asof_join_impl&&)      = delete;

  asof_join_impl(table_view const& right_by,
                 column_view const& right_on,
                 cuda::stream_ref stream,
                 cudf::memory_resources mr);

  cuda::device_buffer<size_type> join(table_view const& left_by,
                                      column_view const& left_on,
                                      asof_join_strategy strategy,
                                      asof_join_exact_matches exact_matches,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr) const;

 private:
  cuda::device_buffer<size_type> probe(table_view const& left_by,
                                       column_view const& left_on,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr) const;

  void build_groups(cuda::stream_ref stream, cudf::memory_resources mr);

  table_view _right_by;
  column_view _right_on;
  cuda::device_buffer<size_type> _right_group_offsets;
  size_type _num_right_groups{};
  // Built once and reused by every probe. Lexicographic comparison locates a candidate group
  // with lower_bound; equality verifies an actual match, treating null grouping keys as unequal.
  // The two comparator APIs require separate preprocessing types.
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> _right_eq_preprocessed;
  std::shared_ptr<cudf::detail::row::lexicographic::preprocessed_table> _right_lex_preprocessed;
};

}  // namespace cudf::experimental::detail
