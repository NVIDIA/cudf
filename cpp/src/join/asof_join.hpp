/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/join/asof_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf::detail {

/**
 * @brief Implementation of an as-of join with a reusable right side.
 */
class asof_join {
 public:
  asof_join()                            = delete;
  ~asof_join()                           = default;
  asof_join(asof_join const&)            = delete;
  asof_join(asof_join&&)                 = delete;
  asof_join& operator=(asof_join const&) = delete;
  asof_join& operator=(asof_join&&)      = delete;

  asof_join(table_view const& right_by, column_view const& right_on, cuda::stream_ref stream);

  std::unique_ptr<rmm::device_uvector<size_type>> join(table_view const& left_by,
                                                       column_view const& left_on,
                                                       asof_join_strategy strategy,
                                                       bool allow_exact_matches,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr) const;

 private:
  void build_right_group_index(cuda::stream_ref stream);

  table_view _right_by;
  column_view _right_on;
  rmm::device_uvector<size_type> _right_group_offsets;
  size_type _num_right_groups{};
  // Built once and reused by every probe. Lexicographic comparison locates a candidate group
  // with lower_bound; equality verifies an actual match, treating null grouping keys as unequal.
  // The two comparator APIs require separate preprocessing types.
  std::shared_ptr<row::equality::preprocessed_table> _right_eq_preprocessed;
  std::shared_ptr<row::lexicographic::preprocessed_table> _right_lex_preprocessed;
};

}  // namespace cudf::detail
