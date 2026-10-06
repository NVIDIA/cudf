/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/transform/column_device_view_wrappers.cuh>

namespace cudf::jit {
using cudf::detail::mutable_strings_column_device_view;
using cudf::detail::mutable_vector_device_view;
}  // namespace cudf::jit
