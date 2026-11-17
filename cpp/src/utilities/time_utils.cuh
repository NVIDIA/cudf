/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cudf::detail {

/**
 * @brief Lookup table to compute power of ten
 */
static __device__ const __constant__ int32_t powers_of_ten[10] = {
  1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

}  // namespace cudf::detail
