/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::lists::detail {

std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        size_type n_elements,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  auto labels = make_numeric_column(
    data_type(type_to_id<size_type>()), n_elements, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const labels_begin = labels->mutable_view().template begin<size_type>();
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels_begin, labels_begin + n_elements, stream);
  return labels;
}

std::unique_ptr<column> reconstruct_offsets(column_view const& labels,
                                            size_type n_lists,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)

{
  auto out_offsets = make_numeric_column(
    data_type{type_id::INT32}, n_lists + 1, mask_state::UNALLOCATED, stream, mr);

  auto const labels_begin  = labels.template begin<size_type>();
  auto const offsets_begin = out_offsets->mutable_view().template begin<int32_t>();
  cudf::detail::labels_to_offsets(labels_begin,
                                  labels_begin + labels.size(),
                                  offsets_begin,
                                  offsets_begin + out_offsets->size(),
                                  stream);
  return out_offsets;
}

std::unique_ptr<column> get_normalized_offsets(lists_column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return empty_like(input.offsets()); }

  auto out_offsets = make_numeric_column(
    data_type(type_id::INT32), input.size() + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    input.offsets_begin(),
                    input.offsets_end(),
                    out_offsets->mutable_view().begin<int32_t>(),
                    [d_offsets = input.offsets_begin()] __device__(auto const offset_val) {
                      // The first offset value, used for zero-normalizing offsets.
                      return offset_val - *d_offsets;
                    });
  return out_offsets;
}

}  // namespace cudf::lists::detail
