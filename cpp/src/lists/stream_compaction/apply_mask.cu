/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/reduction/detail/segmented_reduction.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/scan.h>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> apply_mask(lists_column_view const& input,
                                   lists_column_view const& boolean_mask,
                                   cudf::detail::mask_type mask_kind,
                                   null_policy null_handling,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr)
{
  CUDF_EXPECTS(boolean_mask.child().type().id() == type_id::BOOL8, "Mask must be of type BOOL8.");
  CUDF_EXPECTS(input.size() == boolean_mask.size(),
               "Boolean masks column must have same number of rows as input.");
  auto const num_rows = input.size();

  if (num_rows == 0) { return cudf::empty_like(input.parent()); }

  auto constexpr offset_data_type = data_type{type_id::INT32};

  auto const mask_sliced_child = boolean_mask.get_sliced_child(stream);

  auto const make_filtered_child = [&] {
    auto filtered = cudf::detail::apply_mask(cudf::table_view{{input.get_sliced_child(stream)}},
                                             mask_sliced_child,
                                             mask_kind,
                                             null_handling,
                                             stream,
                                             mr)
                      ->release();
    return std::move(filtered.front());
  };

  auto const make_output_offsets = [&] {
    auto const is_retention = (mask_kind == cudf::detail::mask_type::RETENTION);
    auto const keep_nulls   = (null_handling == null_policy::INCLUDE);
    auto const mask_child   = boolean_mask.child();
    auto const d_mask_child = column_device_view::create(mask_child, stream);
    // 1 if the mask element keeps its list element, 0 otherwise
    auto const keep_begin = cuda::make_transform_iterator(
      cudf::detail::make_null_replacement_iterator<bool>(
        *d_mask_child, keep_nulls == is_retention, mask_child.has_nulls()),
      cuda::proclaim_return_type<size_type>(
        [is_retention] __device__(bool mask) -> size_type { return mask == is_retention; }));

    auto output_offsets = cudf::make_numeric_column(
      offset_data_type, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
    auto const d_offsets = output_offsets->mutable_view().begin<size_type>();

    // Count the kept elements of each row (the mask offsets index the full mask child), then scan
    // the counts into output offsets
    cudf::reduction::detail::segmented_reduce(keep_begin,
                                              boolean_mask.offsets_begin(),
                                              boolean_mask.offsets_end(),
                                              d_offsets + 1,
                                              cuda::std::plus<size_type>{},
                                              size_type{0},
                                              stream,
                                              mr);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                           d_offsets + 1,
                           d_offsets + num_rows + 1,
                           d_offsets + 1);
    CUDF_CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(size_type), stream.get()));
    return output_offsets;
  };

  return cudf::make_lists_column(input.size(),
                                 make_output_offsets(),
                                 make_filtered_child(),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr));
}
}  // namespace detail

std::unique_ptr<column> apply_retention_mask(lists_column_view const& input,
                                             lists_column_view const& retention_mask,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  return apply_retention_mask(input, retention_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<column> apply_retention_mask(lists_column_view const& input,
                                             lists_column_view const& retention_mask,
                                             null_policy null_handling,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(
    input, retention_mask, cudf::detail::mask_type::RETENTION, null_handling, stream, mr);
}

std::unique_ptr<column> apply_boolean_mask(lists_column_view const& input,
                                           lists_column_view const& boolean_mask,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  return apply_retention_mask(input, boolean_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<column> apply_deletion_mask(lists_column_view const& input,
                                            lists_column_view const& deletion_mask,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  return apply_deletion_mask(input, deletion_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<column> apply_deletion_mask(lists_column_view const& input,
                                            lists_column_view const& deletion_mask,
                                            null_policy null_handling,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(
    input, deletion_mask, cudf::detail::mask_type::DELETION, null_handling, stream, mr);
}

}  // namespace cudf::lists
