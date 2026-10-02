/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/groupby.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <span>
#include <utility>

namespace cudf::groupby::detail {
struct groupby_helper;
namespace hash {

/// Compute all aggregation requests using one cached grouping and result cache.
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
  std::span<aggregation_request const> requests,
  groupby_helper& helper,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace hash
}  // namespace cudf::groupby::detail
