/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "update_validity.hpp"

#include <cudf/detail/null_mask.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace cudf {
namespace reduction {
namespace detail {

void segmented_update_validity(column& result,
                               column_view const& col,
                               device_span<size_type const> offsets,
                               null_policy null_handling,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto [output_null_mask, output_null_count] = cudf::detail::segmented_null_mask_reduction(
    col.null_mask(),
    offsets.begin(),
    offsets.end() - 1,
    offsets.begin() + 1,
    null_handling,
    init.has_value() ? std::optional(init.value().get().is_valid(stream)) : std::nullopt,
    stream,
    mr);
  result.set_null_mask(std::move(output_null_mask), output_null_count);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
