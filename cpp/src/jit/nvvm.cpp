/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvvm.hpp"

#include <cudf/utilities/error.hpp>

#ifndef CUDF_STATIC_LINK_LIBNVVM
#define CUDF_STATIC_LINK_LIBNVVM 0
#endif

#if !CUDF_STATIC_LINK_LIBNVVM
#include <dlfcn.h>
#endif

#include <array>
#include <format>
#include <mutex>
#include <optional>
#include <string>

namespace cudf::jit {
namespace {

#if !CUDF_STATIC_LINK_LIBNVVM
void* load_symbol(void* handle, char const* name)
{
  ::dlerror();
  auto* symbol = ::dlsym(handle, name);
  if (symbol == nullptr) {
    auto const* error = ::dlerror();
    CUDF_FAIL(std::format("Failed to load symbol {} from libNVVM: {}",
                          name,
                          error == nullptr ? "unknown error" : error),
              std::runtime_error);
  }
  return symbol;
}

void* load_library()
{
  // The `4` in `libnvvm.so.4` is the libNVVM API major version, not the CUDA Toolkit major
  // version
  constexpr std::array candidates{"libnvvm.so.4", "libnvvm.so"};
  std::string last_error;
  for (auto const& candidate : candidates) {
    if (auto* handle = ::dlopen(candidate, RTLD_NOW | RTLD_LOCAL); handle != nullptr) {
      return handle;
    }
    if (auto const* error = ::dlerror(); error != nullptr) { last_error = error; }
  }

  std::string tried;
  for (auto const& candidate : candidates) {
    if (!tried.empty()) { tried += ", "; }
    tried += candidate;
  }
  CUDF_FAIL(std::format("Failed to load libNVVM (tried: {}). Last error: {}", tried, last_error),
            std::runtime_error);
}
#endif

class libnvvm {
 public:
  libnvvm();
  libnvvm(libnvvm const&)            = delete;
  libnvvm(libnvvm&&)                 = delete;
  libnvvm& operator=(libnvvm const&) = delete;
  libnvvm& operator=(libnvvm&&)      = delete;
  ~libnvvm();

  [[nodiscard]] nvvm_api const& api() const { return api_; }

 private:
#if !CUDF_STATIC_LINK_LIBNVVM
  void* handle_{};
#endif
  nvvm_api api_{};
};

static std::optional<libnvvm> library;
static std::optional<std::once_flag> init_library_flag{std::in_place};
static std::optional<std::once_flag> teardown_library_flag{std::in_place};

}  // namespace

libnvvm::libnvvm()
{
#if CUDF_STATIC_LINK_LIBNVVM
#define CUDF_BIND_NVVM_FUNCTION(name) api_.name = ::nvvm##name;
  CUDF_FOR_EACH_NVVM_FUNCTION(CUDF_BIND_NVVM_FUNCTION)
#undef CUDF_BIND_NVVM_FUNCTION
#else
  handle_ = load_library();
#define CUDF_LOAD_NVVM_FUNCTION(name) \
  api_.name = reinterpret_cast<decltype(api_.name)>(load_symbol(handle_, "nvvm" #name));
  CUDF_FOR_EACH_NVVM_FUNCTION(CUDF_LOAD_NVVM_FUNCTION)
#undef CUDF_LOAD_NVVM_FUNCTION
#endif
}

libnvvm::~libnvvm()
{
#if !CUDF_STATIC_LINK_LIBNVVM
  if (handle_ != nullptr) { ::dlclose(handle_); }
#endif
}

void initialize_nvvm()
{
  std::call_once(init_library_flag.value(), [] { library.emplace(); });
}

void teardown_nvvm()
{
  std::call_once(teardown_library_flag.value(), [] {
    library.reset();
    init_library_flag.reset();
    teardown_library_flag.reset();
    init_library_flag.emplace();
    teardown_library_flag.emplace();
  });
}

nvvm_api const& get_nvvm() { return library->api(); }

}  // namespace cudf::jit
