/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Stable translation unit and PCH prefix for generated ABI-only UDF forwarding code.
// UDF implementations, including cuDF builtins, are supplied as linked fragments.
#include "jit/udf_abi.cuh"

#pragma nv_hdrstop  // Only callback symbols and physical signatures vary per dispatcher.

namespace cudf_udf_abi {
#include <cudf/detail/udf_dispatch.cuh>
}  // namespace cudf_udf_abi
