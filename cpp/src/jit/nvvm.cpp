/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDF_STATIC_LINK_NVVM
#define CUDF_STATIC_LINK_NVVM 0
#endif

#include <jit/nvvm.hpp>

#if !CUDF_STATIC_LINK_NVVM
#include <dlfcn.h>
#endif

#include <exception>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace cudf::jit {
namespace {

#if !CUDF_STATIC_LINK_NVVM

void* load_dso()
{
  std::string names[] = {"libnvvm.so.4", "libnvvm.so"};  // NOLINT(modernize-avoid-c-arrays)
  for (auto& name : names) {
    void* handle = ::dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) { return handle; }
  }
  throw std::runtime_error(
    "Failed to load dynamic library `libnvvm.so` "
    "(tried: libnvvm.so.4, libnvvm.so)");
}

void* get_dso_symbol(void* handle, char const* name)
{
  void* symbol = ::dlsym(handle, name);
  if (symbol == nullptr) {
    throw std::runtime_error(
      std::format("Failed to load symbol `{}` from `libnvvm`, error: `{}`", name, ::dlerror()));
  }
  return symbol;
}

#endif

std::string format_options(std::span<char const* const> options)
{
  if (options.empty()) { return "  <none>"; }
  auto result = std::string{};
  for (auto option : options) {
    result += std::format("  {}\n", option == nullptr ? "<null>" : option);
  }
  result.pop_back();
  return result;
}

std::string format_nvvm_failure(nvvm_api& nvvm,
                                nvvmProgram program,
                                nvvmResult result,
                                std::string_view operation,
                                std::span<char const* const> options,
                                std::string_view name,
                                std::string_view nvvm_ir)
{
  auto log         = std::string{};
  std::size_t size = 0;
  if (program != nullptr && nvvm.GetProgramLogSize(program, &size) == NVVM_SUCCESS && size > 0) {
    log.resize(size);
    if (nvvm.GetProgramLog(program, log.data()) != NVVM_SUCCESS) {
      log = "<failed to retrieve compiler log>";
    } else if (!log.empty() && log.back() == '\0') {
      log.pop_back();
    }
  }
  char const* error = nvvm.GetErrorString(result);
  return std::format(
    R"(libNVVM compilation failed
operation: {}
result: {} ({})
libNVVM version: {}
options:
{}
input fragments:
--- {} ({} bytes) ---
{}
--- end {} ---
compiler log:
{})",
    operation,
    static_cast<int>(result),
    error == nullptr ? "unknown error" : error,
    nvvm.version_string(),
    format_options(options),
    name,
    nvvm_ir.size(),
    nvvm_ir,
    name,
    log.empty() ? "<empty>" : log);
}

}  // namespace

nvvm_api& nvvm_api::get()
{
  static nvvm_api api{load()};
  return api;
}

nvvm_api::nvvm_api(void* handle) : handle_{handle}
{
  try {
    load_symbols();
    auto result = Version(&version_major_, &version_minor_);
    if (result != NVVM_SUCCESS) {
      throw std::runtime_error(
        std::format("nvvmVersion failed with error {}", static_cast<int>(result)));
    }
  } catch (...) {
#if !CUDF_STATIC_LINK_NVVM
    static_cast<void>(::dlclose(handle_));
    handle_ = nullptr;
#endif
    throw;
  }
}

void* nvvm_api::load()
{
#if CUDF_STATIC_LINK_NVVM
  return nullptr;
#else
  return load_dso();
#endif
}

void nvvm_api::load_symbols()
{
#if CUDF_STATIC_LINK_NVVM
#define CUDF_NVVM_LOAD_FUNCTION(name) this->name = nvvm##name;
#else
#define CUDF_NVVM_LOAD_FUNCTION(name) \
  this->name = reinterpret_cast<decltype(nvvm##name)*>(get_dso_symbol(handle_, "nvvm" #name));
#endif
  CUDF_NVVM_FOR_EACH_FUNCTION(CUDF_NVVM_LOAD_FUNCTION)
#undef CUDF_NVVM_LOAD_FUNCTION
}

nvvm_api::~nvvm_api()
{
#if !CUDF_STATIC_LINK_NVVM
  static_cast<void>(::dlclose(handle_));
#endif
}

std::string nvvm_api::version_string() const
{
  return std::format("{}.{}", version_major_, version_minor_);
}

rtcx::blob compile_nvvm(std::string const& name, std::string_view nvvm_ir, int32_t architecture)
{
  auto& nvvm                     = nvvm_api::get();
  nvvmProgram program            = nullptr;
  auto architecture_option       = std::format("-arch=compute_{}", architecture);
  char const* verify_options[]   = {architecture_option.c_str()};
  char const* compiler_options[] = {architecture_option.c_str(), "-opt=3", "-gen-lto"};
  auto check =
    [&](nvvmResult result, std::string_view operation, std::span<char const* const> options) {
      if (result != NVVM_SUCCESS) {
        throw std::runtime_error(
          format_nvvm_failure(nvvm, program, result, operation, options, name, nvvm_ir));
      }
    };

  check(nvvm.CreateProgram(&program), "nvvmCreateProgram", {});
  auto output = rtcx::blob{};
  try {
    check(nvvm.AddModuleToProgram(program, nvvm_ir.data(), nvvm_ir.size(), name.c_str()),
          "nvvmAddModuleToProgram",
          {});
    check(nvvm.VerifyProgram(program, std::size(verify_options), verify_options),
          "nvvmVerifyProgram",
          verify_options);
    check(nvvm.CompileProgram(program, std::size(compiler_options), compiler_options),
          "nvvmCompileProgram",
          compiler_options);

    std::size_t result_size = 0;
    check(nvvm.GetCompiledResultSize(program, &result_size),
          "nvvmGetCompiledResultSize",
          compiler_options);
    auto result = rtcx::byte_buffer::make(result_size);
    check(nvvm.GetCompiledResult(program, reinterpret_cast<char*>(result.data())),
          "nvvmGetCompiledResult",
          compiler_options);
    output = std::make_shared<rtcx::blob_t>(rtcx::blob_t::from_buffer(std::move(result)));
  } catch (...) {
    auto original_error = std::current_exception();
    auto destroy_result = nvvm.DestroyProgram(&program);
    if (destroy_result != NVVM_SUCCESS) {
      auto cleanup_error = format_nvvm_failure(
        nvvm, program, destroy_result, "nvvmDestroyProgram", compiler_options, name, nvvm_ir);
      try {
        std::rethrow_exception(original_error);
      } catch (std::exception const& error) {
        throw std::runtime_error(
          std::format("{}\nAdditionally, NVVM cleanup failed:\n{}", error.what(), cleanup_error));
      }
    }
    std::rethrow_exception(original_error);
  }

  check(nvvm.DestroyProgram(&program), "nvvmDestroyProgram", compiler_options);
  return output;
}

}  // namespace cudf::jit
