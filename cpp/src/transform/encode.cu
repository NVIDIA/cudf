/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/stream>

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<table>, std::unique_ptr<column>> encode(table_view const& input_table,
                                                                  cuda::stream_ref stream,
                                                                  rmm::device_async_resource_ref mr)
{
  auto const num_cols = input_table.num_columns();

  std::vector<size_type> drop_keys(num_cols);
  std::iota(drop_keys.begin(), drop_keys.end(), 0);

  auto distinct_keys = cudf::detail::distinct(input_table,
                                              drop_keys,
                                              duplicate_keep_option::KEEP_ANY,
                                              null_equality::EQUAL,
                                              nan_equality::ALL_EQUAL,
                                              stream,
                                              mr);

  std::vector<order> column_order(num_cols, order::ASCENDING);
  std::vector<null_order> null_precedence(num_cols, null_order::AFTER);
  auto sorted_unique_keys =
    cudf::detail::sort(distinct_keys->view(), column_order, null_precedence, stream, mr);

  auto indices_column = cudf::detail::lower_bound(
    sorted_unique_keys->view(), input_table, column_order, null_precedence, stream, mr);

  return std::pair(std::move(sorted_unique_keys), std::move(indices_column));
}

}  // namespace detail

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode(input, stream, mr);
}

}  // namespace cudf
