/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stream_compaction/apply_mask.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstddef>

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
    auto const num_elements = mask_sliced_child.size();
    auto const d_mask_child = column_device_view::create(mask_sliced_child, stream);

    // kept_counts[i] = number of kept elements among the first `i` elements of the sliced child
    auto kept_counts = cuda::device_buffer<size_type>(
      stream, mr.get_temporary_mr(), static_cast<std::size_t>(num_elements) + 1, cuda::no_init);
    CUDF_CUDA_TRY(cudaMemsetAsync(kept_counts.data(), 0, sizeof(size_type), stream.get()));

    auto const is_retention = (mask_kind == cudf::detail::mask_type::RETENTION);
    auto const keep_nulls   = (null_handling == null_policy::INCLUDE);

    // Lambda to scan the kept counts using the mask iterator
    auto const scan_kept_counts = [&](auto mask_iter) {
      auto const keep_begin = cuda::make_transform_iterator(
        cuda::counting_iterator<size_type>{0},
        cudf::detail::mask_filter<decltype(mask_iter)>{mask_iter, is_retention});
      // Pass a size_type init so the scan does not accumulate in bool
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                             keep_begin,
                             keep_begin + num_elements,
                             kept_counts.data() + 1,
                             size_type{0},
                             cuda::std::plus<size_type>{});
    };

    if (mask_sliced_child.has_nulls()) {
      // Replace nulls with the value that keeps (INCLUDE) or drops (EXCLUDE) the element
      scan_kept_counts(cudf::detail::make_null_replacement_iterator<bool>(
        *d_mask_child, keep_nulls == is_retention));
    } else {
      // No nulls, apply the mask directly
      scan_kept_counts(d_mask_child->data<bool>());
    }

    // Each output offset is the kept count at the corresponding mask offset
    auto output_offsets = cudf::make_numeric_column(
      offset_data_type, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
    thrust::transform(
      rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
      boolean_mask.offsets_begin(),
      boolean_mask.offsets_end(),
      output_offsets->mutable_view().begin<size_type>(),
      cuda::proclaim_return_type<size_type>(
        [kept_counts = kept_counts.data(), mask_offsets = boolean_mask.offsets_begin()] __device__(
          size_type offset) { return kept_counts[offset - mask_offsets[0]]; }));
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
