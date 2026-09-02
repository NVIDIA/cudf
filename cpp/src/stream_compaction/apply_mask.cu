/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

#include <algorithm>

namespace {
// Returns true if the mask is `true` and valid (non-null) for index i
template <bool has_nulls = true>
struct retention_mask_filter {
  cudf::column_device_view mask;

  __device__ inline bool operator()(cudf::size_type i) const noexcept
  {
    if constexpr (has_nulls) {
      return mask.is_valid(i) and mask.data<bool>()[i];
    } else {
      return mask.data<bool>()[i];
    }
  }
};

// Returns true if the mask is `false` and valid (non-null) for index i
template <bool has_nulls = true>
struct deletion_mask_filter {
  cudf::column_device_view mask;

  __device__ inline bool operator()(cudf::size_type i) const noexcept
  {
    if constexpr (has_nulls) {
      return mask.is_valid(i) and not mask.data<bool>()[i];
    } else {
      return not mask.data<bool>()[i];
    }
  }
};

}  // namespace

namespace cudf {
namespace detail {
/*
 * Filters a table_view using a column_view of boolean values as a mask. Masking behavior
 * (retentions or deletions) is specified by the `mask_kind` parameter.
 *
 * calls copy_if() with the `boolean_mask_filter` functor.
 */
std::unique_ptr<table> apply_mask(table_view const& input,
                                  column_view const& boolean_mask,
                                  mask_type mask_kind,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  auto const is_retention = (mask_kind == mask_type::RETENTION);

  if (boolean_mask.is_empty()) {
    return is_retention ? empty_like(input) : std::make_unique<table>(input, stream, mr);
  }

  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8, "Mask must be Boolean type");
  CUDF_EXPECTS(input.num_rows() == 0 || input.num_rows() == boolean_mask.size(),
               "Column size mismatch");

  auto device_boolean_mask = cudf::column_device_view::create(boolean_mask, stream);

  if (boolean_mask.has_nulls()) {
    if (is_retention) {
      return detail::copy_if(input, retention_mask_filter<true>{*device_boolean_mask}, stream, mr);
    } else {
      return detail::copy_if(input, deletion_mask_filter<true>{*device_boolean_mask}, stream, mr);
    }
  } else {
    if (is_retention) {
      return detail::copy_if(input, retention_mask_filter<false>{*device_boolean_mask}, stream, mr);
    } else {
      return detail::copy_if(input, deletion_mask_filter<false>{*device_boolean_mask}, stream, mr);
    }
  }
}

}  // namespace detail

/*
 * Filters a table_view using a column_view of boolean values as a mask.
 */
std::unique_ptr<table> apply_retention_mask(table_view const& input,
                                            column_view const& retention_mask,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(input, retention_mask, detail::mask_type::RETENTION, stream, mr);
}

std::unique_ptr<table> apply_boolean_mask(table_view const& input,
                                          column_view const& boolean_mask,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(input, boolean_mask, detail::mask_type::RETENTION, stream, mr);
}

std::unique_ptr<table> apply_deletion_mask(table_view const& input,
                                           column_view const& deletion_mask,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(input, deletion_mask, detail::mask_type::DELETION, stream, mr);
}

}  // namespace cudf
