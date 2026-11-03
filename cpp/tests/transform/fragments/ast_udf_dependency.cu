/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdint>

// A function used by another module, without appearing in the AST itself.
extern "C" __device__ int32_t lto_increment(int32_t input) { return input + 1; }
