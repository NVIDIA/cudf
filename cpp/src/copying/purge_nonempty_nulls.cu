/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>

namespace cudf {
namespace detail {

using cudf::type_id;

namespace {

/// Check if nonempty-null checks can be skipped for a given type.
bool type_may_have_nonempty_nulls(cudf::type_id const& type)
{
  return type == type_id::STRING || type == type_id::LIST || type == type_id::STRUCT;
}

/// Check if the (STRING/LIST) column has any null rows with non-zero length.
bool has_nonempty_null_rows(cudf::column_view const& input, cuda::stream_ref stream)
{
  if (not input.has_nulls()) { return false; }  // No nulls => no dirty rows.

  if ((input.size() == input.null_count()) && (input.num_children() == 0)) { return false; }

  // Cross-reference nullmask and offsets.
  auto const type    = input.type().id();
  auto const offsets = offsetalator_factory::make_input_iterator(
    (type == type_id::STRING) ? strings_column_view{input}.offsets()
                              : lists_column_view{input}.offsets(),
    input.offset());
  auto const d_input      = cudf::column_device_view::create(input, stream);
  auto const is_dirty_row = [d_input = *d_input, offsets] __device__(size_type const& row_idx) {
    return d_input.is_null_nocheck(row_idx) && (offsets[row_idx] != offsets[row_idx + 1]);
  };

  auto const row_begin = cuda::counting_iterator<cudf::size_type>{0};
  auto const row_end   = row_begin + input.size();
  return cudf::detail::count_if(row_begin, row_end, is_dirty_row, stream) > 0;
}

}  // namespace

/**
 * @copydoc cudf::detail::has_nonempty_nulls
 */
bool has_nonempty_nulls(cudf::column_view const& input, cuda::stream_ref stream)
{
  auto const type = input.type().id();

  if (not type_may_have_nonempty_nulls(type)) { return false; }

  // For types with variable-length rows, check if any rows are "dirty".
  // A dirty row is a null row with non-zero length.
  if ((type == type_id::STRING || type == type_id::LIST) && has_nonempty_null_rows(input, stream)) {
    return true;
  }

  // For complex types, check if child columns need purging.
  if ((type == type_id::STRUCT || type == type_id::LIST) &&
      std::any_of(input.child_begin(), input.child_end(), [stream](auto const& child) {
        return cudf::detail::has_nonempty_nulls(child, stream);
      })) {
    return true;
  }

  return false;
}

std::unique_ptr<column> purge_nonempty_nulls(column_view const& input,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  // If not compound types (LIST/STRING/STRUCT/DICTIONARY) then just copy the input into output.
  if (!cudf::is_compound(input.type())) { return std::make_unique<column>(input, stream, mr); }

  // Implement via identity gather.
  auto gathered_table = cudf::detail::gather(table_view{{input}},
                                             cuda::counting_iterator<cudf::size_type>{0},
                                             cuda::counting_iterator{input.size()},
                                             out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr);
  return std::move(gathered_table->release().front());
}

}  // namespace detail

/**
 * @copydoc cudf::may_have_nonempty_nulls
 */
bool may_have_nonempty_nulls(column_view const& input)
{
  auto const type = input.type().id();

  if (not detail::type_may_have_nonempty_nulls(type)) { return false; }

  if ((type == type_id::STRING || type == type_id::LIST) && input.has_nulls()) { return true; }

  if ((type == type_id::STRUCT || type == type_id::LIST) &&
      std::any_of(input.child_begin(), input.child_end(), may_have_nonempty_nulls)) {
    return true;
  }

  return false;
}

/**
 * @copydoc cudf::has_nonempty_nulls
 */
bool has_nonempty_nulls(column_view const& input, cuda::stream_ref stream)
{
  return detail::has_nonempty_nulls(input, stream);
}

/**
 * @copydoc cudf::purge_nonempty_nulls(column_view const&, rmm::device_async_resource_ref)
 */
std::unique_ptr<cudf::column> purge_nonempty_nulls(column_view const& input,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  return detail::purge_nonempty_nulls(input, stream, mr);
}

}  // namespace cudf
