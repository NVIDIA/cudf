/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvvm.h>
#include <rtcx/rtcx.hpp>

#include <cstdint>
#include <string>
#include <string_view>

#define CUDF_NVVM_FOR_EACH_FUNCTION(F) \
  F(GetErrorString)                    \
  F(Version)                           \
  F(CreateProgram)                     \
  F(DestroyProgram)                    \
  F(AddModuleToProgram)                \
  F(VerifyProgram)                     \
  F(CompileProgram)                    \
  F(GetCompiledResultSize)             \
  F(GetCompiledResult)                 \
  F(GetProgramLogSize)                 \
  F(GetProgramLog)

namespace cudf::jit {

/**
 * @brief Process-wide access to the libNVVM API.
 *
 * The API is bound directly when libcudf is configured with static libNVVM support. Otherwise,
 * libNVVM and its entry points are loaded on first use so applications that do not use NVVM-backed
 * JIT compilation do not acquire a runtime dependency on libNVVM.
 */
class nvvm_api {
 public:
#define CUDF_NVVM_DECL_MEMBER(name) decltype(nvvm##name)* name = nullptr;
  CUDF_NVVM_FOR_EACH_FUNCTION(CUDF_NVVM_DECL_MEMBER)
#undef CUDF_NVVM_DECL_MEMBER

  /** @brief Returns the process-wide libNVVM API instance. */
  static nvvm_api& get();

  nvvm_api(nvvm_api const&)            = delete;
  nvvm_api(nvvm_api&&)                 = delete;
  nvvm_api& operator=(nvvm_api const&) = delete;
  nvvm_api& operator=(nvvm_api&&)      = delete;

  ~nvvm_api();

  /** @brief Returns the loaded libNVVM version as `major.minor`. */
  [[nodiscard]] std::string version_string() const;

 private:
  explicit nvvm_api(void* handle);

  [[nodiscard]] static void* load();
  void load_symbols();

  void* handle_{};
  int version_major_{};
  int version_minor_{};
};

/**
 * @brief Compiles textual NVVM IR into an LTO IR fragment.
 *
 * @param name Logical filename used in compiler diagnostics
 * @param nvvm_ir Complete textual NVVM IR module
 * @param architecture Compute architecture used for the portable LTO fragment
 * @return Compiled LTO IR fragment
 */
rtcx::blob compile_nvvm(std::string const& name, std::string_view nvvm_ir, int32_t architecture);

}  // namespace cudf::jit
