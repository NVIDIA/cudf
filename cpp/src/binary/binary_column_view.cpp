/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/binary/binary_column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/utilities/error.hpp>

namespace cudf {

binary_column_view::binary_column_view(column_view binary_column) : column_view(binary_column)
{
  CUDF_EXPECTS(type().id() == type_id::BINARY, "binary_column_view only supports BINARY");
  if (not is_empty()) {
    CUDF_EXPECTS(num_children() == 1, "non-empty binary column must have one offsets child");
    auto const offsets_view = offsets();
    CUDF_EXPECTS(offsets_view.type().id() == type_id::INT32 or
                   offsets_view.type().id() == type_id::INT64,
                 "binary offsets must have type INT32 or INT64");
    CUDF_EXPECTS(offsets_view.null_count() == 0, "binary offsets must not contain nulls");
    CUDF_EXPECTS(offsets_view.size() == size() + 1,
                 "binary offsets size must equal the row count plus one");
  }
}

column_view binary_column_view::parent() const { return static_cast<column_view>(*this); }

column_view binary_column_view::offsets() const
{
  CUDF_EXPECTS(num_children() > 0, "binary column has no children");
  return child(offsets_column_index);
}

int64_t binary_column_view::bytes_size(rmm::cuda_stream_view stream) const
{
  if (size() == 0) { return 0; }
  auto const offsets_view = offsets();
  auto const last         = offsets_view.size() - 1;
  if (offsets_view.type().id() == type_id::INT32) {
    return cudf::detail::get_value<int32_t>(offsets_view, last, stream);
  }
  CUDF_EXPECTS(offsets_view.type().id() == type_id::INT64,
               "binary offsets must have type INT32 or INT64");
  return cudf::detail::get_value<int64_t>(offsets_view, last, stream);
}

uint8_t const* binary_column_view::bytes_begin() const noexcept { return head<uint8_t>(); }

uint8_t const* binary_column_view::bytes_end(rmm::cuda_stream_view stream) const
{
  return bytes_begin() + bytes_size(stream);
}

}  // namespace cudf
