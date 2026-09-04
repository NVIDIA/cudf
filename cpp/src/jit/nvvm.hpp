/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvvm.h>

#define CUDF_FOR_EACH_NVVM_FUNCTION(DO_IT) \
  DO_IT(GetErrorString)                    \
  DO_IT(CreateProgram)                     \
  DO_IT(DestroyProgram)                    \
  DO_IT(AddModuleToProgram)                \
  DO_IT(CompileProgram)                    \
  DO_IT(GetCompiledResultSize)             \
  DO_IT(GetCompiledResult)                 \
  DO_IT(GetProgramLogSize)                 \
  DO_IT(GetProgramLog)

namespace cudf::jit {

/**
 * @brief Dispatch table for the libNVVM functions used by cuDF.
 *
 * Entries bind directly to libNVVM in static-link builds and are populated from the dynamically
 * loaded libNVVM shared library otherwise.
 */
struct nvvm_api {
#define CUDF_DECLARE_NVVM_FUNCTION(name) decltype(&::nvvm##name) name{};
  CUDF_FOR_EACH_NVVM_FUNCTION(CUDF_DECLARE_NVVM_FUNCTION)
#undef CUDF_DECLARE_NVVM_FUNCTION
};

/**
 * @brief Initializes the process-wide libNVVM dispatch table.
 *
 * This operation is idempotent and thread-safe. The cuDF context normally manages this lifecycle.
 *
 */
void initialize_nvvm();

/**
 * @brief Releases the process-wide libNVVM dispatch table.
 *
 * All references returned by get_nvvm() are invalidated. initialize_nvvm() may be called afterward
 * to load libNVVM again. This function is not thread-safe.
 */
void teardown_nvvm();

/**
 * @brief Returns the initialized process-wide libNVVM dispatch table.
 *
 * initialize_nvvm() must have completed and teardown_nvvm() must not have been called afterward.
 *
 * @return Reference valid until teardown_nvvm() is called
 * @throws std::runtime_error If libNVVM is not initialized
 */
nvvm_api const& get_nvvm();

}  // namespace cudf::jit
