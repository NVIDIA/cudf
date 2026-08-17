/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_column_impl.cuh"
#include "sort_radix.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace detail {

/**
 * @copydoc
 * stable_sorted_order(column_view&,order,null_order,cuda::stream_ref,rmm::device_async_resource_ref)
 */
template <>
std::unique_ptr<column> sorted_order<sort_method::STABLE>(column_view const& input,
                                                          order column_order,
                                                          null_order null_precedence,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  auto sorted_indices = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view indices_view = sorted_indices->mutable_view();
  if (is_radix_sortable(input)) {
    sorted_order_radix(input, indices_view, column_order == order::ASCENDING, stream);
  } else {
    cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                                 column_sorted_order_fn<sort_method::STABLE>{},
                                                 input,
                                                 indices_view,
                                                 column_order == order::ASCENDING,
                                                 null_precedence,
                                                 stream);
  }
  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
