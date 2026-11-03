/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
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
  return structs::detail::get_sliced_child(
    *this, index, stream, cudf::get_current_device_resource_ref());
}

column_view structs::detail::get_sliced_child(structs_column_view const& input,
                                              int index,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref temp_mr)
{
  auto const child = input.child(index);
  std::vector<column_view> children;
  children.reserve(child.num_children());
  for (size_type i = 0; i < child.num_children(); i++) {
    children.push_back(child.child(i));
  }

  return column_view{
    child.type(),
    input.size(),
    child.head<uint8_t>(),
    child.null_mask(),
    child.null_count()
      ? cudf::detail::null_count(
          child.null_mask(), input.offset(), input.offset() + input.size(), stream, temp_mr)
      : 0,
    input.offset(),
    children};
}

}  // namespace cudf
