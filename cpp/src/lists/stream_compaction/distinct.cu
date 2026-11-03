/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lists/utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <utility>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> distinct(lists_column_view const& input,
                                 null_equality nulls_equal,
                                 nan_equality nans_equal,
                                 duplicate_keep_option keep_option,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  // Algorithm:
  // - Generate labels for the child elements.
  // - Get distinct rows of the table {labels, child} using `stable_distinct`.
  // - Build the output lists column from the output distinct rows above.

  if (input.is_empty()) { return empty_like(input.parent()); }

  auto const child = input.get_sliced_child(stream);
  auto const labels =
    generate_labels(input, child.size(), stream, cudf::get_current_device_resource_ref());

  auto const distinct_table =
    cudf::detail::stable_distinct(table_view{{labels->view(), child}},  // input table
                                  std::vector<size_type>{0, 1},         // keys
                                  keep_option,
                                  nulls_equal,
                                  nans_equal,
                                  stream,
                                  mr);

  auto out_offsets =
    reconstruct_offsets(distinct_table->get_column(0).view(), input.size(), stream, mr);

  return make_lists_column(input.size(),
                           std::move(out_offsets),
                           std::move(distinct_table->release().back()),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<column> distinct(lists_column_view const& input,
                                 null_equality nulls_equal,
                                 nan_equality nans_equal,
                                 duplicate_keep_option keep_option,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(input, nulls_equal, nans_equal, keep_option, stream, mr);
}

}  // namespace cudf::lists
