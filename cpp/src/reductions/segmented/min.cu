/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::reduction::simple::detail {

std::unique_ptr<column> string_segmented_minmax(column_view const& col,
                                                device_span<size_type const> offsets,
                                                bool is_argmin,
                                                null_policy null_handling,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  auto device_col         = cudf::column_device_view::create(col, stream);
  auto it                 = cuda::counting_iterator<cudf::size_type>{0};
  auto const num_segments = static_cast<size_type>(offsets.size()) - 1;
  auto string_comparator  = reduce_argminmax_fn<string_view>{*device_col, is_argmin, null_handling};
  auto const identity = is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL;

  auto gather_map = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, num_segments, mask_state::UNALLOCATED, stream, mr);
  auto gather_map_it = gather_map->mutable_view().begin<size_type>();

  cudf::reduction::detail::segmented_reduce(
    it, offsets.begin(), offsets.end(), gather_map_it, string_comparator, identity, stream);

  auto result = std::move(cudf::detail::gather(table_view{{col}},
                                               *gather_map,
                                               cudf::out_of_bounds_policy::NULLIFY,
                                               cudf::negative_index_policy::NOT_ALLOWED,
                                               stream,
                                               mr)
                            ->release()[0]);
  cudf::reduction::detail::segmented_update_validity(
    *result, col, offsets, null_handling, std::nullopt, stream, mr);
  return result;
}

}  // namespace cudf::reduction::simple::detail

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::column> segmented_min(
  column_view const& col,
  device_span<size_type const> offsets,
  data_type const output_dtype,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(col.type() == output_dtype,
               "segmented_min() operation requires matching output type");
  using reducer = simple::detail::same_column_type_dispatcher<op::min>;
  return cudf::type_dispatcher(
    col.type(), reducer{}, col, offsets, null_handling, init, stream, mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
