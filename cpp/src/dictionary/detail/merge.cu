/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/merge.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace dictionary {
namespace detail {

std::unique_ptr<column> merge(dictionary_column_view const& lcol,
                              dictionary_column_view const& rcol,
                              cudf::detail::index_vector const& row_order,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  auto const lcol_iter = cudf::detail::indexalator_factory::make_input_iterator(lcol.indices());
  auto const rcol_iter = cudf::detail::indexalator_factory::make_input_iterator(rcol.indices());

  // create output indices column
  auto const merged_size  = lcol.size() + rcol.size();
  auto const indices_type = get_indices_type_for_size(merged_size);
  auto indices_column =
    make_fixed_width_column(indices_type, merged_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto output_iter =
    cudf::detail::indexalator_factory::make_output_iterator(indices_column->mutable_view());

  // merge the input indices columns into the output column
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    row_order.begin(),
                    row_order.end(),
                    output_iter,
                    [lcol_iter, rcol_iter] __device__(auto const& index_pair) {
                      auto const [side, index] = index_pair;
                      return side == cudf::detail::side::LEFT ? lcol_iter[index] : rcol_iter[index];
                    });

  // build dictionary; the validity mask is updated by the caller
  return make_dictionary_column(
    std::make_unique<column>(lcol.keys(), stream, mr),
    std::move(indices_column),
    cudf::detail::create_null_mask(
      lcol.has_nulls() || rcol.has_nulls() ? static_cast<size_t>(merged_size) : 0,
      mask_state::UNINITIALIZED,
      stream,
      mr),
    lcol.null_count() + rcol.null_count());
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
