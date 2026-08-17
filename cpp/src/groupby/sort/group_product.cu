/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_product(column_view const& values,
                                      size_type num_groups,
                                      cudf::device_span<size_type const> group_labels,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? dictionary_column_view(values).keys().type()
                       : values.type();
  return type_dispatcher(values_type,
                         group_reduction_dispatcher<aggregation::PRODUCT>{},
                         values,
                         num_groups,
                         group_labels,
                         stream,
                         mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
