/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/join_common_utils.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::finalize_partitioned_full_join(
  cudf::host_span<cudf::device_span<size_type const> const> left_partials,
  cudf::host_span<cudf::device_span<size_type const> const> right_partials,
  size_type left_table_num_rows,
  size_type right_table_num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::finalize_full_join(
    left_partials, right_partials, left_table_num_rows, right_table_num_rows, stream, mr);
}

}  // namespace cudf
