/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/concatenate_masks.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/transform.h>

#include <algorithm>
#include <memory>
#include <numeric>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Merges the offsets child columns of multiple list columns into one.
 *
 * Since offsets are all relative to the start of their respective column,
 * all offsets are shifted to account for the new starting position
 *
 * @param[in] columns               Vector of lists columns to concatenate
 * @param[in] total_list_count      Total number of lists contained in the columns
 * @param[in] stream                CUDA stream used for device memory operations
 * and kernel launches.
 * @param[in] mr                    Device memory resource used to allocate the
 * returned column's device memory.
 */
std::unique_ptr<column> merge_offsets(host_span<lists_column_view const> columns,
                                      size_type total_list_count,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  // outgoing offsets
  auto merged_offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, total_list_count + 1, mask_state::UNALLOCATED, stream, mr);
  mutable_column_device_view d_merged_offsets(*merged_offsets, 0, 0);

  // merge offsets
  // TODO : this could probably be done as a single gpu operation if done as a kernel.
  size_type shift = 0;
  size_type count = 0;
  std::for_each(columns.begin(), columns.end(), [&](lists_column_view const& c) {
    if (c.size() > 0) {
      // handle sliced columns
      int const local_shift =
        shift -
        (c.offset() > 0 ? cudf::detail::get_value<int32_t>(c.offsets(), c.offset(), stream) : 0);
      column_device_view offsets(c.offsets(), nullptr, nullptr);
      thrust::transform(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        offsets.begin<int32_t>() + c.offset(),
        offsets.begin<int32_t>() + c.offset() + c.size() + 1,
        d_merged_offsets.begin<int32_t>() + count,
        [local_shift] __device__(int32_t offset) -> int32_t { return offset + local_shift; });

      shift += c.get_sliced_child(stream).size();
      count += c.size();
    }
  });

  return merged_offsets;
}

}  // namespace

/**
 * @copydoc cudf::lists::detail::concatenate
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  std::vector<lists_column_view> lists_columns;
  lists_columns.reserve(columns.size());
  std::transform(
    columns.begin(), columns.end(), std::back_inserter(lists_columns), [](column_view const& c) {
      return lists_column_view(c);
    });

  // concatenate children. also prep data needed for offset merging
  std::vector<column_view> children;
  children.reserve(columns.size());
  size_type total_list_count = 0;
  std::for_each(lists_columns.begin(),
                lists_columns.end(),
                [&total_list_count, &children, stream](lists_column_view const& l) {
                  // count total # of lists
                  total_list_count += l.size();
                  children.push_back(l.get_sliced_child(stream));
                });
  auto data = cudf::detail::concatenate(children, stream, mr);

  // merge offsets
  auto offsets = merge_offsets(lists_columns, total_list_count, stream, mr);

  // if any of the input columns have nulls, construct the output mask
  bool const has_nulls =
    std::any_of(columns.begin(), columns.end(), [](auto const& col) { return col.has_nulls(); });
  rmm::device_buffer null_mask = cudf::detail::create_null_mask(
    total_list_count, has_nulls ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED, stream, mr);
  auto null_mask_data = static_cast<bitmask_type*>(null_mask.data());
  auto const null_count =
    has_nulls ? cudf::detail::concatenate_masks(columns, null_mask_data, stream) : size_type{0};

  // assemble into outgoing list column
  return make_lists_column(
    total_list_count, std::move(offsets), std::move(data), null_count, std::move(null_mask));
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
