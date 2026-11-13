/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>
#include <utility>

namespace cudf {
namespace groupby::detail::hash {
// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
  std::span<aggregation_request const> requests,
  groupby_helper& helper,
  cuda::stream_ref stream,
  cudf::memory_resources mr);
}  // namespace groupby::detail::hash
}  // namespace cudf
