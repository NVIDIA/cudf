/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "regex_ir_detail.hpp"

namespace regex_ir {

/** @brief Emit CUDA C++ source from an operation-specialized instruction IR. */
compile_result generate_cuda_source(instruction_ir const& ir, cuda_codegen_options const& options);

}  // namespace regex_ir
