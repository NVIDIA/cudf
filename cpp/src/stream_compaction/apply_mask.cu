/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apply_mask.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/iterator.cuh>
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

namespace cudf {
namespace detail {
/*
 * Filters a table_view using a column_view of boolean values as a mask. Masking behavior
 * (retentions or deletions) is specified by the `mask_kind` parameter.
 *
 * calls copy_if() with a `mask_filter` functor.
 */
std::unique_ptr<table> apply_mask(table_view const& input,
                                  column_view const& boolean_mask,
                                  mask_type mask_kind,
                                  null_policy null_handling,
                                  cuda::stream_ref stream,
                                  cudf::memory_resources mr)
{
  if (boolean_mask.is_empty()) {
    return (mask_kind == mask_type::RETENTION)
             ? empty_like(input)
             : std::make_unique<table>(input, stream, mr.get_output_mr());
  }

  CUDF_EXPECTS(boolean_mask.type().id() == type_id::BOOL8, "Mask must be Boolean type");
  CUDF_EXPECTS(input.num_rows() == 0 || input.num_rows() == boolean_mask.size(),
               "Column size mismatch");

  auto const d_mask = cudf::column_device_view::create(boolean_mask, stream, mr.get_temporary_mr());
  auto const is_retention = (mask_kind == mask_type::RETENTION);
  auto const keep_nulls   = (null_handling == null_policy::INCLUDE);

  // Lambda to filter the table using the mask iterator
  auto const filter = [&](auto mask_iter) {
    return detail::copy_if(
      input, mask_filter<decltype(mask_iter)>{mask_iter, is_retention}, stream, mr);
  };

  if (boolean_mask.has_nulls()) {
    // Replace nulls with the value that keeps (INCLUDE) or drops (EXCLUDE) the row
    return filter(make_null_replacement_iterator<bool>(*d_mask, keep_nulls == is_retention));
  }

  // No nulls, apply the mask directly
  return filter(d_mask->data<bool>());
}

}  // namespace detail

/*
 * Filters a table_view using a column_view of boolean values as a mask.
 */
std::unique_ptr<table> apply_retention_mask(table_view const& input,
                                            column_view const& retention_mask,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  return apply_retention_mask(input, retention_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<table> apply_retention_mask(table_view const& input,
                                            column_view const& retention_mask,
                                            null_policy null_handling,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(
    input, retention_mask, detail::mask_type::RETENTION, null_handling, stream, mr);
}

std::unique_ptr<table> apply_boolean_mask(table_view const& input,
                                          column_view const& boolean_mask,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  return apply_retention_mask(input, boolean_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<table> apply_deletion_mask(table_view const& input,
                                           column_view const& deletion_mask,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  return apply_deletion_mask(input, deletion_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<table> apply_deletion_mask(table_view const& input,
                                           column_view const& deletion_mask,
                                           null_policy null_handling,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(
    input, deletion_mask, detail::mask_type::DELETION, null_handling, stream, mr);
}

}  // namespace cudf
