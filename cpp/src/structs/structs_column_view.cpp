/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {

structs_column_view::structs_column_view(column_view const& rhs) : column_view{rhs}
{
  CUDF_EXPECTS(type().id() == type_id::STRUCT, "structs_column_view only supports struct columns");
}

column_view structs_column_view::parent() const { return *this; }

column_view structs_column_view::get_sliced_child(int index, cuda::stream_ref stream) const
{
  std::vector<column_view> children;
  children.reserve(child(index).num_children());
  for (size_type i = 0; i < child(index).num_children(); i++) {
    children.push_back(child(index).child(i));
  }

  return column_view{
    child(index).type(),
    size(),
    child(index).head<uint8_t>(),
    child(index).null_mask(),
    child(index).null_count()
      ? cudf::detail::null_count(child(index).null_mask(), offset(), offset() + size(), stream)
      : 0,
    offset(),
    children};
}

}  // namespace cudf
