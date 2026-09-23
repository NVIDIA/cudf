/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lists/utilities.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/reduction/detail/histogram.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/gather.h>

namespace cudf::groupby::detail {

namespace {

std::unique_ptr<column> build_histogram(column_view const& values,
                                        cudf::device_span<size_type const> group_labels,
                                        std::optional<column_view> const& partial_counts,
                                        size_type num_groups,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be the same as that of group labels.",
               std::invalid_argument);

  // Attach group labels to the input values.
  auto const labels_cv      = column_view{data_type{type_to_id<size_type>()},
                                     static_cast<size_type>(group_labels.size()),
                                     group_labels.data(),
                                     nullptr,
                                     0};
  auto const labeled_values = table_view{{labels_cv, values}};

  // Build histogram for the labeled values.
  auto [distinct_indices, distinct_counts] =
    cudf::reduction::detail::compute_row_frequencies(labeled_values, partial_counts, stream, mr);

  // Gather the distinct rows for the output histogram.
  auto out_table = cudf::detail::gather(labeled_values,
                                        *distinct_indices,
                                        out_of_bounds_policy::DONT_CHECK,
                                        cudf::negative_index_policy::NOT_ALLOWED,
                                        stream,
                                        mr);

  // Build offsets for the output lists column containing output histograms.
  // Each list will be a histogram corresponding to one value group.
  auto out_offsets = cudf::lists::detail::reconstruct_offsets(
    out_table->get_column(0).view(), num_groups, stream, mr);

  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(std::move(out_table->release().back()));
  struct_children.emplace_back(std::move(distinct_counts));
  auto out_structs = make_structs_column(static_cast<size_type>(distinct_indices->size()),
                                         std::move(struct_children),
                                         0,
                                         {},
                                         stream,
                                         mr);

  return make_lists_column(num_groups, std::move(out_offsets), std::move(out_structs), 0, {});
}

}  // namespace

std::unique_ptr<column> group_histogram(column_view const& values,
                                        cudf::device_span<size_type const> group_labels,
                                        size_type num_groups,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  // Empty group should be handled before reaching here.
  CUDF_EXPECTS(num_groups > 0, "Group should not be empty.", std::invalid_argument);

  return build_histogram(values, group_labels, std::nullopt, num_groups, stream, mr);
}

std::unique_ptr<column> group_merge_histogram(column_view const& values,
                                              cudf::device_span<size_type const> group_offsets,
                                              size_type num_groups,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  // Empty group should be handled before reaching here.
  CUDF_EXPECTS(num_groups > 0, "Group should not be empty.", std::invalid_argument);

  // The input must be a lists column without nulls.
  CUDF_EXPECTS(!values.has_nulls(), "The input column must not have nulls.", std::invalid_argument);
  CUDF_EXPECTS(values.type().id() == type_id::LIST,
               "The input of MERGE_HISTOGRAM aggregation must be a lists column.",
               std::invalid_argument);

  // Child of the input lists column must be a structs column without nulls,
  // and its second child is a columns of integer type having no nulls.
  auto const lists_cv     = lists_column_view{values};
  auto const histogram_cv = lists_cv.get_sliced_child(stream);
  CUDF_EXPECTS(!histogram_cv.has_nulls(),
               "Child of the input lists column must not have nulls.",
               std::invalid_argument);
  CUDF_EXPECTS(histogram_cv.type().id() == type_id::STRUCT && histogram_cv.num_children() == 2,
               "The input column has invalid histograms structure.",
               std::invalid_argument);
  CUDF_EXPECTS(
    cudf::is_integral(histogram_cv.child(1).type()) && !histogram_cv.child(1).has_nulls(),
    "The input column has invalid histograms structure.",
    std::invalid_argument);

  // Concatenate the histograms corresponding to the same key values.
  // That is equivalent to creating a new lists column (view) from the input lists column
  // with new offsets gathered as below.
  auto new_offsets = rmm::device_uvector<size_type>(num_groups + 1, stream);
  thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 group_offsets.begin(),
                 group_offsets.end(),
                 lists_cv.offsets_begin(),
                 new_offsets.begin());

  // Generate labels for the new lists.
  auto key_labels = rmm::device_uvector<size_type>(histogram_cv.size(), stream);
  cudf::detail::label_segments(
    new_offsets.begin(), new_offsets.end(), key_labels.begin(), key_labels.end(), stream);

  auto const structs_cv   = structs_column_view{histogram_cv};
  auto const input_values = structs_cv.get_sliced_child(0, stream);
  auto const input_counts = structs_cv.get_sliced_child(1, stream);

  return build_histogram(input_values, key_labels, input_counts, num_groups, stream, mr);
}

}  // namespace cudf::groupby::detail
