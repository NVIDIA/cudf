/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_error_diagnostics.hpp"

#include <cuda.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace cudf::jni {
namespace {

std::string format_cuda_version(int version)
{
  if (version <= 0) { return std::to_string(version); }
  return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10) + " (" +
         std::to_string(version) + ")";
}

template <typename Function>
Function load_function(detail::dynamic_loader const& loader, void* handle, char const* name)
{
  auto* symbol = loader.symbol(handle, name);
  Function function{};
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

template <typename GetErrorName, typename GetErrorString>
std::string format_driver_error(CUresult status,
                                GetErrorName get_error_name,
                                GetErrorString get_error_string)
{
  std::ostringstream out;
  char const* name        = nullptr;
  char const* description = nullptr;
  if (get_error_name != nullptr) { get_error_name(status, &name); }
  if (get_error_string != nullptr) { get_error_string(status, &description); }
  if (name != nullptr) {
    out << name;
  } else {
    out << "CUDA Driver API error";
  }
  out << " (" << static_cast<int>(status) << ")";
  if (description != nullptr) { out << ": " << description; }
  return out.str();
}

std::string generic_guidance()
{
  return "CUDA initialization diagnostics were inconclusive. cudaErrorInsufficientDriver can "
         "also mean that the NVIDIA driver is unavailable, no GPU is visible, or the driver is "
         "incompatible with the bundled CUDA runtime.";
}

detail::dynamic_loader const system_loader{dlopen, dlsym, dlerror, dlclose};

std::string const& cached_cuda_initialization_diagnostic() noexcept
{
  static std::string const diagnostic = [] {
    try {
      return detail::format_cuda_initialization_diagnostic(detail::probe_cuda_driver(system_loader),
                                                           CUDART_VERSION);
    } catch (...) {
      return generic_guidance();
    }
  }();
  return diagnostic;
}

}  // namespace

namespace detail {

cuda_driver_probe_result probe_cuda_driver(dynamic_loader const& loader)
{
  auto* handle = loader.open("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    auto const* error = loader.error();
    return {cuda_driver_probe_state::library_unavailable,
            error == nullptr ? "unknown dynamic loader error" : error};
  }

  struct library_closer {
    dynamic_loader const& loader;
    void* handle;
    ~library_closer() { loader.close(handle); }
  } close_library{loader, handle};

  auto const init = load_function<decltype(&cuInit)>(loader, handle, "cuInit");
  auto const get_driver_version =
    load_function<decltype(&cuDriverGetVersion)>(loader, handle, "cuDriverGetVersion");
  auto const get_device_count =
    load_function<decltype(&cuDeviceGetCount)>(loader, handle, "cuDeviceGetCount");
  auto const get_error_name =
    load_function<decltype(&cuGetErrorName)>(loader, handle, "cuGetErrorName");
  auto const get_error_string =
    load_function<decltype(&cuGetErrorString)>(loader, handle, "cuGetErrorString");

  std::vector<std::string> missing_symbols;
  if (init == nullptr) { missing_symbols.emplace_back("cuInit"); }
  if (get_driver_version == nullptr) { missing_symbols.emplace_back("cuDriverGetVersion"); }
  if (get_device_count == nullptr) { missing_symbols.emplace_back("cuDeviceGetCount"); }
  if (!missing_symbols.empty()) {
    std::ostringstream detail;
    for (std::size_t i = 0; i < missing_symbols.size(); ++i) {
      if (i != 0) { detail << ", "; }
      detail << missing_symbols[i];
    }
    return {cuda_driver_probe_state::symbols_unavailable, detail.str()};
  }

  auto const init_status = init(0);
  if (init_status != CUDA_SUCCESS) {
    return {cuda_driver_probe_state::initialization_failed,
            format_driver_error(init_status, get_error_name, get_error_string),
            static_cast<int>(init_status)};
  }

  cuda_driver_probe_result result{cuda_driver_probe_state::initialized, {}};
  int driver_version{};
  auto const driver_version_status = get_driver_version(&driver_version);
  if (driver_version_status == CUDA_SUCCESS) {
    result.driver_version = driver_version;
  } else {
    result.detail = "cuDriverGetVersion failed with " +
                    format_driver_error(driver_version_status, get_error_name, get_error_string);
  }

  int device_count{};
  auto const device_count_status = get_device_count(&device_count);
  if (device_count_status == CUDA_SUCCESS) {
    result.device_count = device_count;
  } else {
    if (!result.detail.empty()) { result.detail += "; "; }
    result.detail += "cuDeviceGetCount failed with " +
                     format_driver_error(device_count_status, get_error_name, get_error_string);
  }
  return result;
}

std::string format_cuda_initialization_diagnostic(cuda_driver_probe_result const& probe,
                                                  int runtime_version)
{
  std::ostringstream out;
  out << "CUDA initialization diagnostics: bundled CUDA runtime "
      << format_cuda_version(runtime_version) << "; ";

  switch (probe.state) {
    case cuda_driver_probe_state::library_unavailable:
      out << "libcuda.so.1 could not be loaded (" << probe.detail
          << "). The NVIDIA driver may be absent, or its library may not be mounted in or "
             "discoverable from the container. Verify GPU assignment, the NVIDIA container "
             "runtime or device plugin, and the dynamic loader configuration.";
      break;
    case cuda_driver_probe_state::symbols_unavailable:
      out << "libcuda.so.1 is missing required symbol(s): " << probe.detail
          << ". The driver installation may be incomplete or incompatible.";
      break;
    case cuda_driver_probe_state::initialization_failed:
      out << "libcuda.so.1 was loaded, but cuInit failed with " << probe.detail << ". ";
      if (probe.initialization_status == CUDA_ERROR_NO_DEVICE) {
        out << "No CUDA device is visible. Verify the container GPU assignment, device plugin or "
               "runtime class, and CUDA_VISIBLE_DEVICES.";
      } else if (probe.initialization_status == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) {
        out << "The NVIDIA kernel and user-mode driver components do not match.";
      } else if (probe.initialization_status == CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE) {
        out << "The CUDA forward-compatibility configuration is not supported by this device.";
      } else {
        out << "Verify the driver installation, GPU visibility, and driver/runtime compatibility.";
      }
      break;
    case cuda_driver_probe_state::initialized:
      out << "libcuda.so.1 loaded and the Driver API initialized";
      if (probe.driver_version.has_value()) {
        out << "; driver supports CUDA " << format_cuda_version(*probe.driver_version);
      }
      if (!probe.detail.empty()) { out << "; " << probe.detail; }
      if (probe.device_count.has_value()) {
        out << "; visible devices: " << *probe.device_count << ". ";
        if (*probe.device_count == 0) {
          out << "No CUDA device is visible. Verify the container GPU assignment, device plugin "
                 "or runtime class, and CUDA_VISIBLE_DEVICES.";
        } else if (probe.driver_version.has_value() && *probe.driver_version < runtime_version) {
          out << "The CUDA runtime rejected this older driver. Upgrade the host driver or use a "
                 "compatible cuDF CUDA artifact.";
        } else {
          out << "The CUDA runtime still rejected the accessible driver. Verify driver/runtime "
                 "compatibility and container driver-library mounts.";
        }
      } else {
        out << ". Verify GPU visibility and driver/runtime compatibility.";
      }
      break;
  }
  return out.str();
}

}  // namespace detail

std::string augment_cuda_error_message(char const* message, cudaError_t status)
{
  std::string result{message == nullptr ? "" : message};
  if (status != cudaErrorInsufficientDriver) { return result; }

  result += "\n";
  result += cached_cuda_initialization_diagnostic();
  return result;
}

}  // namespace cudf::jni
