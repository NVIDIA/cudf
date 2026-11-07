/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_error_diagnostics.hpp"

#include <cuda.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace cudf::jni::detail {
namespace {

struct fake_driver {
  bool library_available{true};
  std::string missing_symbol;
  std::string loader_error{"library unavailable"};
  CUresult init_status{CUDA_SUCCESS};
  CUresult driver_version_status{CUDA_SUCCESS};
  CUresult device_count_status{CUDA_SUCCESS};
  int driver_version{12080};
  int device_count{1};
  int close_count{};
};

fake_driver* active_driver{};
int fake_library_handle;
int failures{};

class active_driver_guard {
 public:
  explicit active_driver_guard(fake_driver& driver) : previous_driver{active_driver}
  {
    active_driver = &driver;
  }
  ~active_driver_guard() { active_driver = previous_driver; }

  active_driver_guard(active_driver_guard const&)            = delete;
  active_driver_guard& operator=(active_driver_guard const&) = delete;

 private:
  fake_driver* previous_driver;
};

void expect(bool condition, char const* message)
{
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

void expect_contains(std::string const& value, char const* expected)
{
  expect(value.find(expected) != std::string::npos, expected);
}

template <typename Function>
void* function_address(Function function)
{
  void* address{};
  static_assert(sizeof(address) == sizeof(function));
  std::memcpy(&address, &function, sizeof(address));
  return address;
}

CUresult CUDAAPI fake_cu_init(unsigned int) { return active_driver->init_status; }

CUresult CUDAAPI fake_cu_driver_get_version(int* version)
{
  if (active_driver->driver_version_status == CUDA_SUCCESS) {
    *version = active_driver->driver_version;
  }
  return active_driver->driver_version_status;
}

CUresult CUDAAPI fake_cu_device_get_count(int* count)
{
  if (active_driver->device_count_status == CUDA_SUCCESS) { *count = active_driver->device_count; }
  return active_driver->device_count_status;
}

CUresult CUDAAPI fake_cu_get_error_name(CUresult error, char const** name)
{
  if (error == CUDA_ERROR_NO_DEVICE) {
    *name = "CUDA_ERROR_NO_DEVICE";
  } else if (error == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) {
    *name = "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";
  } else {
    *name = "CUDA_ERROR_UNKNOWN";
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_cu_get_error_string(CUresult, char const** description)
{
  *description = "fake driver error";
  return CUDA_SUCCESS;
}

void* fake_open(char const*, int)
{
  return active_driver->library_available ? &fake_library_handle : nullptr;
}

void* fake_symbol(void*, char const* name)
{
  if (active_driver->missing_symbol == name) { return nullptr; }
  if (std::strcmp(name, "cuInit") == 0) { return function_address(&fake_cu_init); }
  if (std::strcmp(name, "cuDriverGetVersion") == 0) {
    return function_address(&fake_cu_driver_get_version);
  }
  if (std::strcmp(name, "cuDeviceGetCount") == 0) {
    return function_address(&fake_cu_device_get_count);
  }
  if (std::strcmp(name, "cuGetErrorName") == 0) {
    return function_address(&fake_cu_get_error_name);
  }
  if (std::strcmp(name, "cuGetErrorString") == 0) {
    return function_address(&fake_cu_get_error_string);
  }
  return nullptr;
}

char* fake_error() { return active_driver->loader_error.data(); }

int fake_close(void*)
{
  ++active_driver->close_count;
  return 0;
}

dynamic_loader const fake_loader{fake_open, fake_symbol, fake_error, fake_close};

void test_unavailable_driver_library()
{
  fake_driver driver;
  active_driver_guard guard{driver};
  driver.library_available = false;

  auto const result = probe_cuda_driver(fake_loader);

  expect(result.state == cuda_driver_probe_state::library_unavailable, "library unavailable state");
  expect(result.detail == "library unavailable", "dynamic loader error");
  expect(driver.close_count == 0, "unopened library is not closed");
}

void test_missing_required_symbols()
{
  fake_driver driver;
  active_driver_guard guard{driver};
  driver.missing_symbol = "cuDeviceGetCount";

  auto const result = probe_cuda_driver(fake_loader);

  expect(result.state == cuda_driver_probe_state::symbols_unavailable, "symbols unavailable state");
  expect(result.detail == "cuDeviceGetCount", "missing symbol name");
  expect(driver.close_count == 1, "opened library is closed after missing symbol");
}

void test_driver_initialization_failure()
{
  fake_driver driver;
  active_driver_guard guard{driver};
  driver.init_status = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;

  auto const result = probe_cuda_driver(fake_loader);

  expect(result.state == cuda_driver_probe_state::initialization_failed,
         "initialization failed state");
  expect(result.initialization_status == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
         "initialization status");
  expect_contains(result.detail, "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH");
  expect(driver.close_count == 1, "opened library is closed after initialization failure");
}

void test_driver_version_and_device_count()
{
  fake_driver driver;
  active_driver_guard guard{driver};
  driver.driver_version = 12080;
  driver.device_count   = 2;

  auto const result = probe_cuda_driver(fake_loader);

  expect(result.state == cuda_driver_probe_state::initialized, "initialized state");
  expect(result.driver_version == 12080, "driver version");
  expect(result.device_count == 2, "device count");
  expect(driver.close_count == 1, "opened library is closed after successful probe");
}

void test_query_failures()
{
  fake_driver driver;
  active_driver_guard guard{driver};
  driver.driver_version_status = CUDA_ERROR_UNKNOWN;
  driver.device_count_status   = CUDA_ERROR_NO_DEVICE;

  auto const result = probe_cuda_driver(fake_loader);

  expect(result.state == cuda_driver_probe_state::initialized, "initialized query failure state");
  expect(!result.driver_version.has_value(), "driver version query failure");
  expect(!result.device_count.has_value(), "device count query failure");
  expect_contains(result.detail, "cuDriverGetVersion failed");
  expect_contains(result.detail, "cuDeviceGetCount failed");
}

void test_diagnostic_formatting()
{
  auto message = format_cuda_initialization_diagnostic(
    {cuda_driver_probe_state::library_unavailable, "not found"}, 13000);
  expect_contains(message, "bundled CUDA runtime 13.0 (13000)");
  expect_contains(message, "libcuda.so.1 could not be loaded");
  expect_contains(message, "container runtime or device plugin");

  message = format_cuda_initialization_diagnostic(
    {cuda_driver_probe_state::initialized, {}, 0, 13000, 0}, 13000);
  expect_contains(message, "visible devices: 0");
  expect_contains(message, "CUDA_VISIBLE_DEVICES");

  message = format_cuda_initialization_diagnostic(
    {cuda_driver_probe_state::initialized, {}, 0, 12080, 1}, 13000);
  expect_contains(message, "driver supports CUDA 12.8 (12080)");
  expect_contains(message, "rejected this older driver");
  expect_contains(message, "compatible cuDF CUDA artifact");

  message = format_cuda_initialization_diagnostic({cuda_driver_probe_state::initialization_failed,
                                                   "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (803)",
                                                   CUDA_ERROR_SYSTEM_DRIVER_MISMATCH},
                                                  13000);
  expect_contains(message, "kernel and user-mode driver components do not match");

  message = format_cuda_initialization_diagnostic({cuda_driver_probe_state::initialization_failed,
                                                   "CUDA_ERROR_NO_DEVICE (100)",
                                                   CUDA_ERROR_NO_DEVICE},
                                                  13000);
  expect_contains(message, "No CUDA device is visible");

  message =
    format_cuda_initialization_diagnostic({cuda_driver_probe_state::initialization_failed,
                                           "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE (804)",
                                           CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE},
                                          13000);
  expect_contains(message, "forward-compatibility configuration is not supported");
}

void test_other_cuda_errors_unchanged()
{
  expect(augment_cuda_error_message("invalid value", cudaErrorInvalidValue) == "invalid value",
         "non-initialization CUDA error message");
}

void test_insufficient_driver_diagnostic_is_cached_thread_safely()
{
  std::vector<std::string> messages(8);
  std::vector<std::thread> threads;
  threads.reserve(messages.size());
  for (std::size_t i = 0; i < messages.size(); ++i) {
    threads.emplace_back([i, &messages] {
      messages[i] = augment_cuda_error_message("original", cudaErrorInsufficientDriver);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  expect_contains(messages.front(), "original\nCUDA initialization diagnostics:");
  for (auto const& message : messages) {
    expect(message == messages.front(), "cached concurrent diagnostic");
  }
}

}  // namespace
}  // namespace cudf::jni::detail

int main()
{
  using namespace cudf::jni::detail;
  test_unavailable_driver_library();
  test_missing_required_symbols();
  test_driver_initialization_failure();
  test_driver_version_and_device_count();
  test_query_failures();
  test_diagnostic_formatting();
  test_other_cuda_errors_unchanged();
  test_insufficient_driver_diagnostic_is_cached_thread_safely();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
