/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file
 * @brief Common device headers for CUDA UDF source.
 *
 * Provides cuDF value types, error codes, row operators, and CUDA standard-library utilities.
 * CUDA UDFs can include this header before compilation with `compile_cuda_to_lto_ir`.
 * Keeping these includes together also provides a stable prefix for NVRTC precompiled headers.
 */

#include <cudf/detail/row_ir/includes.hpp>
#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>

#include <cuda/std/cstring>
#include <cuda/std/optional>
#include <cuda/std/span>
