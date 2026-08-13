/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ColumnViewJni.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <vector>

namespace cudf::jni {

std::unique_ptr<cudf::column> generate_list_offsets(cudf::column_view const& list_length,
                                                    rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(list_length.type().id() == cudf::type_id::INT32,
               "Input column does not have type INT32.");

  auto const begin_iter = list_length.template begin<cudf::size_type>();
  auto const end_iter   = list_length.template end<cudf::size_type>();

  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, list_length.size() + 1, mask_state::UNALLOCATED, stream);
  auto offsets_view = offsets_column->mutable_view();
  auto d_offsets    = offsets_view.template begin<int32_t>();

  thrust::inclusive_scan(rmm::exec_policy_nosync(stream), begin_iter, end_iter, d_offsets + 1);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(int32_t), stream));

  return offsets_column;
}

namespace {

/**
 * @brief Check if the input list has any null elements.
 *
 * @param list The input list.
 * @return The boolean result indicating if the input list has null elements.
 */
__device__ bool list_has_nulls(list_device_view list)
{
  return thrust::any_of(thrust::seq,
                        cuda::counting_iterator{cudf::size_type{0}},
                        cuda::counting_iterator{list.size()},
                        [&list](auto const idx) { return list.is_null(idx); });
}

}  // namespace

void post_process_list_overlap(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               std::unique_ptr<cudf::column> const& overlap_result,
                               rmm::cuda_stream_view stream)
{
  // If both of the input columns do not have nulls, we don't need to do anything here.
  if (!lists_column_view{lhs}.child().has_nulls() && !lists_column_view{rhs}.child().has_nulls()) {
    return;
  }

  auto const overlap_cv      = overlap_result->view();
  auto const lhs_cdv_ptr     = column_device_view::create(lhs, stream);
  auto const rhs_cdv_ptr     = column_device_view::create(rhs, stream);
  auto const overlap_cdv_ptr = column_device_view::create(overlap_cv, stream);

  // Create a new bitmask to satisfy Spark's arrays_overlap's special behavior.
  auto validity = rmm::device_uvector<bool>(overlap_cv.size(), stream);
  thrust::tabulate(
    rmm::exec_policy_nosync(stream),
    validity.begin(),
    validity.end(),
    [lhs            = cudf::lists_column_device_view{*lhs_cdv_ptr},
     rhs            = cudf::lists_column_device_view{*rhs_cdv_ptr},
     overlap_result = *overlap_cdv_ptr] __device__(auto const idx) {
      if (overlap_result.is_null(idx) || overlap_result.template element<bool>(idx)) {
        return true;
      }

      // `lhs_list` and `rhs_list` should not be null, otherwise
      // `overlap_result[idx]` is null and that has been handled above.
      auto const lhs_list = list_device_view{lhs, idx};
      auto const rhs_list = list_device_view{rhs, idx};

      // Only proceed if both lists are non-empty.
      if (lhs_list.size() == 0 || rhs_list.size() == 0) { return true; }

      // Only proceed if at least one list has nulls.
      if (!list_has_nulls(lhs_list) && !list_has_nulls(rhs_list)) { return true; }

      // Here, the input lists satisfy all the conditions below so we output a
      // null:
      //  - Both of the input lists have no non-null common element, and
      //  - They are both non-empty, and
      //  - Either of them contains null elements.
      return false;
    });

  // Create a new nullmask from the validity data.
  auto [new_null_mask, new_null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity),
                        cudf::get_default_stream(),
                        cudf::get_current_device_resource_ref());

  if (new_null_count > 0) {
    // If the `overlap_result` column is nullable, perform `bitmask_and` of its nullmask and the
    // new nullmask.
    if (overlap_cv.nullable()) {
      auto [null_mask, null_count] = cudf::bitmask_and(
        std::vector<bitmask_type const*>{overlap_cv.null_mask(),
                                         static_cast<bitmask_type const*>(new_null_mask->data())},
        std::vector<cudf::size_type>{0, 0},
        overlap_cv.size(),
        stream,
        cudf::get_current_device_resource_ref());
      overlap_result->set_null_mask(std::move(null_mask), null_count);
    } else {
      // Just set the output nullmask as the new nullmask.
      overlap_result->set_null_mask(std::move(*new_null_mask.release()), new_null_count);
    }
  }
}

std::unique_ptr<cudf::column> lists_distinct_by_key(cudf::lists_column_view const& input,
                                                    rmm::cuda_stream_view stream)
{
  if (input.is_empty()) { return empty_like(input.parent()); }

  auto const child = input.get_sliced_child(stream);

  // Generate labels for the input list elements.
  auto labels = rmm::device_uvector<cudf::size_type>(child.size(), stream);
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels.begin(), labels.end(), stream);

  // Use `cudf::duplicate_keep_option::KEEP_LAST` so this will produce the desired behavior when
  // being called in `create_map` in spark-rapids.
  // Other options comparing nulls and NaNs are set as all-equal.
  auto out_columns =
    cudf::stable_distinct(table_view{{column_view{cudf::device_span<cudf::size_type const>{labels}},
                                      child.child(0),
                                      child.child(1)}},  // input table
                          std::vector<size_type>{0, 1},  // key columns
                          cudf::duplicate_keep_option::KEEP_LAST,
                          cudf::null_equality::EQUAL,
                          cudf::nan_equality::ALL_EQUAL,
                          stream,
                          cudf::get_current_device_resource_ref())
      ->release();
  auto const out_labels = out_columns.front()->view();

  // Assemble a structs column of <out_keys, out_vals>.
  auto out_structs_members = std::vector<std::unique_ptr<cudf::column>>();
  out_structs_members.emplace_back(std::move(out_columns[1]));
  out_structs_members.emplace_back(std::move(out_columns[2]));
  auto out_structs =
    cudf::make_structs_column(out_labels.size(), std::move(out_structs_members), 0, {});

  // Assemble a lists column of structs<out_keys, out_vals>.
  auto out_offsets = make_numeric_column(
    data_type{type_to_id<size_type>()}, input.size() + 1, mask_state::UNALLOCATED, stream);
  auto const offsets_begin = out_offsets->mutable_view().template begin<size_type>();
  auto const labels_begin  = out_labels.template begin<size_type>();
  cudf::detail::labels_to_offsets(labels_begin,
                                  labels_begin + out_labels.size(),
                                  offsets_begin,
                                  offsets_begin + out_offsets->size(),
                                  stream);

  return cudf::make_lists_column(
    input.size(),
    std::move(out_offsets),
    std::move(out_structs),
    input.null_count(),
    cudf::copy_bitmask(input.parent(), stream, cudf::get_current_device_resource_ref()));
}

}  // namespace cudf::jni
