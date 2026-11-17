/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/is_element_valid.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

bool is_element_valid_sync(column_view const& col_view,
                           size_type element_index,
                           cuda::stream_ref stream)
{
  CUDF_EXPECTS(element_index >= 0 and element_index < col_view.size(), "invalid index.");
  if (!col_view.nullable()) { return true; }

  // null_mask() returns device ptr to bitmask without offset
  size_type const index = element_index + col_view.offset();

  auto const word =
    cudf::detail::make_host_vector(
      device_span<bitmask_type const>{col_view.null_mask() + word_index(index), 1}, stream)
      .front();

  return static_cast<bool>(word & (bitmask_type{1} << intra_word_index(index)));
}

}  // namespace detail
}  // namespace cudf
