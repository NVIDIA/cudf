/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby {

void streaming_groupby::aggregate(table_view const& data, cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  do_aggregate(data, stream);
}

void streaming_groupby::merge(streaming_groupby const& other, cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  do_merge(other, stream);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> streaming_groupby::finalize(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return do_finalize(stream, mr);
}

}  // namespace cudf::groupby
