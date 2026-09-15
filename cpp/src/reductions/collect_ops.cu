/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace reduction {
namespace detail {
namespace {

/**
 * @brief Check if we need to handle nulls in the input column.
 *
 * @param input The input column
 * @param null_handling The null handling policy
 * @return A boolean value indicating if we need to handle nulls
 */
bool need_handle_nulls(column_view const& input, null_policy null_handling)
{
  return null_handling == null_policy::EXCLUDE && input.has_nulls();
}

}  // namespace

std::unique_ptr<scalar> collect_list(column_view const& col,
                                     null_policy null_handling,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  if (need_handle_nulls(col, null_handling)) {
    auto d_view             = column_device_view::create(col, stream);
    auto filter             = cudf::detail::validity_accessor(*d_view);
    auto null_purged_table  = cudf::detail::copy_if(table_view{{col}}, filter, stream, mr);
    column* null_purged_col = null_purged_table->release().front().release();
    null_purged_col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
    return std::make_unique<list_scalar>(std::move(*null_purged_col), true, stream, mr);
  } else {
    return make_list_scalar(col, stream, mr);
  }
}

std::unique_ptr<scalar> merge_lists(lists_column_view const& col,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto flatten_col = col.get_sliced_child(stream);
  return make_list_scalar(flatten_col, stream, mr);
}

std::unique_ptr<scalar> collect_set(column_view const& col,
                                    null_policy null_handling,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  // `input_as_collect_list` is the result of the input column that has been processed to obey
  // the given null handling behavior.
  [[maybe_unused]] auto const [input_as_collect_list, unused_scalar] = [&] {
    if (need_handle_nulls(col, null_handling)) {
      // Only call `collect_list` when we need to handle nulls.
      auto scalar = collect_list(col, null_handling, stream, mr);
      return std::pair(static_cast<list_scalar*>(scalar.get())->view(), std::move(scalar));
    }

    return std::pair(col, std::unique_ptr<scalar>(nullptr));
  }();

  auto distinct_table = cudf::detail::distinct(table_view{{input_as_collect_list}},
                                               std::vector<size_type>{0},
                                               duplicate_keep_option::KEEP_ANY,
                                               nulls_equal,
                                               nans_equal,
                                               stream,
                                               mr);

  return std::make_unique<list_scalar>(std::move(distinct_table->get_column(0)), true, stream, mr);
}

std::unique_ptr<scalar> merge_sets(lists_column_view const& col,
                                   null_equality nulls_equal,
                                   nan_equality nans_equal,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  auto flatten_col    = col.get_sliced_child(stream);
  auto distinct_table = cudf::detail::distinct(table_view{{flatten_col}},
                                               std::vector<size_type>{0},
                                               duplicate_keep_option::KEEP_ANY,
                                               nulls_equal,
                                               nans_equal,
                                               stream,
                                               mr);
  return std::make_unique<list_scalar>(std::move(distinct_table->get_column(0)), true, stream, mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
