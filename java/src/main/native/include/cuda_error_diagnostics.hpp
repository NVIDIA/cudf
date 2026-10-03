/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <optional>
#include <string>
#include <utility>

namespace cudf::jni {

namespace detail {

enum class cuda_driver_probe_state {
  library_unavailable,
  symbols_unavailable,
  initialization_failed,
  initialized
};

struct cuda_driver_probe_result {
  cuda_driver_probe_result(cuda_driver_probe_state state,
                           std::string detail,
                           int initialization_status         = 0,
                           std::optional<int> driver_version = std::nullopt,
                           std::optional<int> device_count   = std::nullopt)
    : state{state},
      detail{std::move(detail)},
      initialization_status{initialization_status},
      driver_version{driver_version},
      device_count{device_count}
  {
  }

  cuda_driver_probe_state state;
  std::string detail;
  int initialization_status{};
  std::optional<int> driver_version;
  std::optional<int> device_count;
};

struct dynamic_loader {
  void* (*open)(char const*, int);
  void* (*symbol)(void*, char const*);
  char* (*error)();
  int (*close)(void*);
};

cuda_driver_probe_result probe_cuda_driver(dynamic_loader const& loader);

std::string format_cuda_initialization_diagnostic(cuda_driver_probe_result const& probe,
                                                  int runtime_version);

}  // namespace detail

/**
 * @brief Add initialization diagnostics to a CUDA error message when appropriate.
 *
 * Only `cudaErrorInsufficientDriver` messages are changed. The diagnostic is computed once per
 * process and does not use the CUDA Runtime API, which may already be in a failed state.
 */
std::string augment_cuda_error_message(char const* message, cudaError_t status);

}  // namespace cudf::jni
