/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/lists/list_view.hpp>
#include <cudf/lists/detail/utilities.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/unary.hpp>

#include <limits>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace lists::detail {
int64_t get_offset_value(column_view const& offsets,
                         size_type index,
                         rmm::cuda_stream_view stream)
{
  if (offsets.type().id() == type_id::INT32) {
    return static_cast<int64_t>(cudf::detail::get_value<int32_t>(offsets, index, stream));
  }
  CUDF_EXPECTS(offsets.type().id() == type_id::INT64, "List offsets must be INT32 or INT64");
  return cudf::detail::get_value<int64_t>(offsets, index, stream);
}

data_type promoted_offsets_type(data_type lhs, data_type rhs)
{
  CUDF_EXPECTS((lhs.id() == type_id::INT32 || lhs.id() == type_id::INT64) &&
                 (rhs.id() == type_id::INT32 || rhs.id() == type_id::INT64),
               "List offsets must be INT32 or INT64");
  return data_type{lhs.id() == type_id::INT64 || rhs.id() == type_id::INT64 ? type_id::INT64
                                                                           : type_id::INT32};
}

std::unique_ptr<column> normalize_offsets(std::unique_ptr<column> offsets,
                                          size_type child_size,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const desired_type =
    child_size > static_cast<size_type>(std::numeric_limits<int32_t>::max())
      ? data_type{type_id::INT64}
      : data_type{type_id::INT32};
  if (offsets->type() == desired_type) { return offsets; }
  return cudf::cast(offsets->view(), desired_type, stream, mr);
}
}  // namespace lists::detail

lists_column_view::lists_column_view(column_view const& lists_column) : column_view(lists_column)
{
  CUDF_EXPECTS(type().id() == type_id::LIST, "lists_column_view only supports lists");
  if (num_children() != 0) {
    auto const offsets_type = offsets().type().id();
    CUDF_EXPECTS(offsets_type == type_id::INT32 || offsets_type == type_id::INT64,
                 "List offsets must be INT32 or INT64");
#if CUDF_SIZE_TYPE_BITS == 32
    CUDF_EXPECTS(offsets_type != type_id::INT64,
                 "INT64 list offsets require a 64-bit cudf::size_type");
#endif
  }
}

column_view lists_column_view::parent() const { return static_cast<column_view>(*this); }

column_view lists_column_view::offsets() const
{
  CUDF_EXPECTS(num_children() == 2, "lists column has an incorrect number of children");
  return column_view::child(offsets_column_index);
}

column_view lists_column_view::child() const
{
  CUDF_EXPECTS(num_children() == 2, "lists column has an incorrect number of children");
  return column_view::child(child_column_index);
}

column_view lists_column_view::get_sliced_child(rmm::cuda_stream_view stream) const
{
  // if I have a positive offset, I need to slice my child
  if (offset() > 0) {
    // theoretically this function could always do this step and be correct, but get_value<>
    // actually hits the gpu so it's best to avoid it if possible.
    size_type child_offset_start =
      static_cast<size_type>(lists::detail::get_offset_value(offsets(), offset(), stream));
    size_type child_offset_end =
      static_cast<size_type>(lists::detail::get_offset_value(offsets(), offset() + size(), stream));
    return cudf::detail::slice(child(), {child_offset_start, child_offset_end}, stream).front();
  }

  // if I don't have a positive offset, but I am shorter than my offsets() would otherwise indicate,
  // I need to do a split and return the front.
  if (size() < offsets().size() - 1) {
    size_type child_offset =
      static_cast<size_type>(lists::detail::get_offset_value(offsets(), size(), stream));
    return cudf::detail::slice(child(), {0, child_offset}, stream).front();
  }

  // otherwise just return the child directly
  return child();
}

}  // namespace cudf
