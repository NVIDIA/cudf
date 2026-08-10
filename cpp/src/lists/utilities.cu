/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <limits>

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
                                            rmm::device_async_resource_ref mr,
                                            data_type preferred_type)

{
  auto const output_type =
    preferred_type.id() == type_id::INT64 ||
        labels.size() > static_cast<size_type>(std::numeric_limits<int32_t>::max())
      ? data_type{type_id::INT64}
      : data_type{type_id::INT32};
  auto out_offsets = make_numeric_column(
    output_type, n_lists + 1, mask_state::UNALLOCATED, stream, mr);

  auto const labels_begin = labels.template begin<size_type>();
  auto offsets_view       = out_offsets->mutable_view();
  if (output_type.id() == type_id::INT32) {
    auto const offsets_begin = offsets_view.template begin<int32_t>();
    cudf::detail::labels_to_offsets(labels_begin,
                                    labels_begin + labels.size(),
                                    offsets_begin,
                                    offsets_begin + out_offsets->size(),
                                    stream);
  } else {
    auto const offsets_begin = offsets_view.template begin<int64_t>();
    cudf::detail::labels_to_offsets(labels_begin,
                                    labels_begin + labels.size(),
                                    offsets_begin,
                                    offsets_begin + out_offsets->size(),
                                    stream);
  }
  return out_offsets;
}

std::unique_ptr<column> get_normalized_offsets(lists_column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return empty_like(input.offsets()); }

  auto out_offsets = make_numeric_column(input.offsets().type(),
                                         input.size() + 1,
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    input.offsets_begin(),
                    input.offsets_end(),
                    cudf::detail::offsetalator_factory::make_output_iterator(
                      out_offsets->mutable_view()),
                    [d_offsets = input.offsets_begin()] __device__(auto const offset_val) {
                      // The first offset value, used for zero-normalizing offsets.
                      return offset_val - *d_offsets;
                    });
  return out_offsets;
}

}  // namespace cudf::lists::detail
