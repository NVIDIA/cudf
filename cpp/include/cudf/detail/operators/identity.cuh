/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Copies an input value to the output.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <typename T>
__device__ T identity(T a)
{ return a; }

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
