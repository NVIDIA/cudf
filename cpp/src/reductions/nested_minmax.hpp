/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf::reduction::detail {

/**
 * @brief Finds the index of a nested minimum or maximum and returns its element.
 *
 * This owns the fixed row-comparator/CUB reduction instantiation shared by the
 * nested `min` and `max` reductions. `is_min_op` remains runtime state in the
 * device operator, so the two public entry points need not emit identical
 * device kernels.
 */
std::unique_ptr<scalar> nested_minmax(column_view const& input,
                                      bool is_min_op,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

}  // namespace cudf::reduction::detail
