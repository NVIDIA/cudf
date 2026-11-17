/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

__global__ static void kernel() { printf("The kernel ran!\n"); }

void test_cudaLaunchKernel()
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel<<<1, 1, 0, stream>>>();
  cudaError_t err{cudaDeviceSynchronize()};
  if (err != cudaSuccess) { throw std::runtime_error("Kernel failed on non-default stream!"); }
  err = cudaGetLastError();
  if (err != cudaSuccess) { throw std::runtime_error("Kernel failed on non-default stream!"); }

  try {
    kernel<<<1, 1>>>();
  } catch (std::runtime_error&) {
    return;
  }
  if (getenv("GTEST_CUDF_STREAM_MODE") == nullptr) { return; }
  throw std::runtime_error("No exception raised for kernel on default stream!");
}

int main()
{
  test_cudaLaunchKernel();
  return 0;
}
