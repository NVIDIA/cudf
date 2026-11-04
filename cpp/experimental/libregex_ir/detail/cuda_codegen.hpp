/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuda_codegen.hpp
 * @brief Private interface for emitting CUDA source from regex instruction graphs.
 *
 * Connects the compiler pipeline to CUDA generation and its compiled-program
 * metadata without exposing emitter implementation details.
 */

#pragma once

#include <libregex_ir/detail/detail.hpp>

namespace regex_ir {

/** @brief Emit CUDA C++ source from an operation-specialized instruction IR. */
compile_result generate_cuda_source(instruction_ir const& ir, cuda_codegen_options const& options);

}  // namespace regex_ir
