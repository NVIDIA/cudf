/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/linked_column.hpp>

#include <cuda/iterator>

namespace cudf::detail {

linked_column_view::linked_column_view(column_view const& col) : linked_column_view(nullptr, col) {}

linked_column_view::linked_column_view(linked_column_view* parent, column_view const& col)
  : column_view_base(col), parent(parent)
{
  children.reserve(col.num_children());
  std::transform(
    col.child_begin(), col.child_end(), std::back_inserter(children), [&](column_view const& c) {
      return std::make_shared<linked_column_view>(this, c);
    });
}

linked_column_view::operator column_view() const
{
  auto child_it = cuda::transform_iterator(
    children.begin(), [](auto const& c) { return static_cast<column_view>(*c); });
  return column_view(this->type(),
                     this->size(),
                     this->head(),
                     this->null_mask(),
                     this->null_count(),
                     this->offset(),
                     std::vector<column_view>(child_it, child_it + children.size()));
}

LinkedColVector table_to_linked_columns(table_view const& table)
{
  auto linked_it = cuda::transform_iterator(
    table.begin(), [](auto const& c) { return std::make_shared<linked_column_view>(c); });
  return LinkedColVector(linked_it, linked_it + table.num_columns());
}

}  // namespace cudf::detail
