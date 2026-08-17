/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @brief List gatherer function object.
 *
 * The iterator needed for gathering at level N+1 needs to reference the offsets
 * from level N and the "base" offsets used from level N-1.  An example of
 * the gather map needed for level N+1 (see documentation for make_gather_data for
 * the full example)
 *
 * @code{.pseudo}
 * level N-1 offsets               : [0, 2, 5, 10], gather map[0, 2]
 *
 * level N offsets                 : [0, 2, 7]
 * "base" offsets from level N-1   : [0, 5]
 *
 * desired output sequence for the level N+1 gather map
 * [0, 1, 5, 6, 7, 8, 9]
 *
 * The generation of this sequence in this functor works as follows
 *
 * step 1, generate row index sequence
 * [0, 0, 1, 1, 1, 1, 1]
 * step 2, generate row subindex sequence
 * [0, 1, 0, 1, 2, 3, 4]
 * step 3, add base offsets to get the final sequence
 * [0, 1, 5, 6, 7, 8, 9]
 * @endcode
 */
struct list_gatherer {
  using argument_type = size_type;
  using result_type   = size_type;

  size_t offset_count;
  int32_t const* base_offsets;
  int32_t const* offsets;

  list_gatherer(gather_data const& gd)
    : offset_count{gd.base_offsets.size()},
      base_offsets{gd.base_offsets.data()},
      offsets{gd.offsets->mutable_view().data<int32_t>()}
  {
  }

  __device__ result_type operator()(argument_type index)
  {
    // the "upper bound" of the span for a given offset is always offsets+1;
    int32_t const* upper_bound_start = offsets + 1;
    // "step 1" from above
    auto const bound =
      thrust::upper_bound(thrust::seq, upper_bound_start, upper_bound_start + offset_count, index);
    size_type offset_index = cuda::std::distance(upper_bound_start, bound);
    // "step 2" from above
    size_type offset_subindex = offset_index == 0 ? index : index - offsets[offset_index];
    // "step 3" from above
    return offset_subindex + base_offsets[offset_index];
  }
};

/**
 * @copydoc cudf::lists::detail::gather_list_leaf
 */
std::unique_ptr<column> gather_list_leaf(column_view const& column,
                                         gather_data const& gd,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  // gather map iterator for this level (N)
  auto gather_map_begin =
    cudf::detail::make_counting_transform_iterator(size_type{0}, list_gatherer{gd});
  size_type gather_map_size = gd.gather_map_size;

  // call the normal gather
  // note : we don't need to bother checking for out-of-bounds here since
  // our inputs at this stage aren't coming from the user.
  auto gather_table = cudf::detail::gather(cudf::table_view({column}),
                                           gather_map_begin,
                                           gather_map_begin + gather_map_size,
                                           out_of_bounds_policy::DONT_CHECK,
                                           stream,
                                           mr);
  auto leaf_column  = std::move(gather_table->release().front());

  if (column.null_count() == 0) { leaf_column->set_null_mask(rmm::device_buffer{}, 0); }

  return leaf_column;
}

/**
 * @copydoc cudf::lists::detail::gather_list_nested
 */
std::unique_ptr<column> gather_list_nested(cudf::lists_column_view const& list,
                                           gather_data& gd,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  // gather map iterator for this level (N)
  auto gather_map_begin =
    cudf::detail::make_counting_transform_iterator(size_type{0}, list_gatherer{gd});
  size_type gather_map_size = gd.gather_map_size;

  // if the gather map is empty, return an empty column
  if (gather_map_size == 0) { return empty_like(list.parent()); }

  // gather the bitmask, if relevant
  rmm::device_buffer null_mask{0, stream, mr};
  size_type null_count = list.null_count();
  if (null_count > 0) {
    auto list_cdv = column_device_view::create(list.parent(), stream);
    auto validity = cudf::detail::valid_if(
      gather_map_begin,
      gather_map_begin + gather_map_size,
      [cdv = *list_cdv] __device__(int index) { return cdv.is_valid(index); },
      stream,
      mr);
    null_mask  = std::move(validity.first);
    null_count = validity.second;
  }

  // generate gather_data for next level (N+1), potentially recycling the temporary
  // base_offsets buffer.
  gather_data child_gd = make_gather_data<false>(
    list, gather_map_begin, gather_map_size, std::move(gd.base_offsets), stream, mr);

  // the nesting case.
  if (list.child().type() == cudf::data_type{type_id::LIST}) {
    // gather children.
    auto child = gather_list_nested(list.get_sliced_child(stream), child_gd, stream, mr);

    // return the nested column
    return make_lists_column(gather_map_size,
                             std::move(child_gd.offsets),
                             std::move(child),
                             null_count,
                             std::move(null_mask));
  }

  // it's a leaf.  do a regular gather
  auto child = gather_list_leaf(list.get_sliced_child(stream), child_gd, stream, mr);

  // assemble final column
  return make_lists_column(gather_map_size,
                           std::move(child_gd.offsets),
                           std::move(child),
                           null_count,
                           std::move(null_mask));
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
