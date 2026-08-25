/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/timestamp_utilities.cuh>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/limits>

#include <algorithm>

namespace {
template <typename T>
CUDF_KERNEL void gpu_atomic_test(T* result, T* data, size_t size)
{
  size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (; id < size; id += step) {
    cudf::detail::atomic_add(&result[0], data[id]);
    cudf::detail::atomic_min(&result[1], data[id]);
    cudf::detail::atomic_max(&result[2], data[id]);
    cudf::detail::genericAtomicOperation(&result[3], data[id], cudf::DeviceSum{});
    cudf::detail::genericAtomicOperation(&result[4], data[id], cudf::DeviceMin{});
    cudf::detail::genericAtomicOperation(&result[5], data[id], cudf::DeviceMax{});
  }
}

template <typename T, typename BinaryOp>
constexpr inline bool is_timestamp_sum()
{
  return cudf::is_timestamp<T>() && std::is_same_v<BinaryOp, cudf::DeviceSum>;
}
// Disable SUM of TIMESTAMP types
template <typename T,
          typename BinaryOp,
          std::enable_if_t<is_timestamp_sum<T, BinaryOp>()>* = nullptr>
__device__ T atomic_op(T* addr, T const& value, BinaryOp op)
{
  return {};
}

template <typename T,
          typename BinaryOp,
          std::enable_if_t<!is_timestamp_sum<T, BinaryOp>()>* = nullptr>
__device__ T atomic_op(T* addr, T const& value, BinaryOp op)
{
  T old_value = *addr;
  T assumed;

  do {
    assumed     = old_value;
    T new_value = op(old_value, value);

    old_value = cudf::detail::atomic_cas(addr, assumed, new_value);
  } while (assumed != old_value);

  return old_value;
}

template <typename T>
CUDF_KERNEL void gpu_atomicCAS_test(T* result, T* data, size_t size)
{
  size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (; id < size; id += step) {
    atomic_op(&result[0], data[id], cudf::DeviceSum{});
    atomic_op(&result[1], data[id], cudf::DeviceMin{});
    atomic_op(&result[2], data[id], cudf::DeviceMax{});
    atomic_op(&result[3], data[id], cudf::DeviceSum{});
    atomic_op(&result[4], data[id], cudf::DeviceMin{});
    atomic_op(&result[5], data[id], cudf::DeviceMax{});
  }
}

template <typename T>
std::enable_if_t<!cudf::is_timestamp<T>(), T> accumulate(cudf::host_span<T const> xs)
{
  return std::accumulate(xs.begin(), xs.end(), T{0});
}

template <typename T>
std::enable_if_t<cudf::is_timestamp<T>(), T> accumulate(cudf::host_span<T const> xs)
{
  auto ys = std::vector<typename T::rep>(xs.size());
  std::transform(
    xs.begin(), xs.end(), ys.begin(), [](T const& ts) { return ts.time_since_epoch().count(); });
  return T{typename T::duration{std::accumulate(ys.begin(), ys.end(), 0)}};
}
}  // namespace

template <typename T>
struct AtomicsTest : public cudf::test::BaseFixture {
  void atomic_test(std::vector<int> const& v_input,
                   bool is_cas_test,
                   int block_size = 0,
                   int grid_size  = 1)
  {
    size_t vec_size = v_input.size();

    // use transform from thrust::host_vector<int> instead.
    thrust::host_vector<T> v(vec_size);
    std::transform(v_input.begin(), v_input.end(), v.begin(), [](int x) {
      T t = cudf::test::make_type_param_scalar<T>(x);
      return t;
    });

    T exact[3];
    exact[0] = accumulate<T>(v);
    exact[1] = *(std::min_element(v.begin(), v.end()));
    exact[2] = *(std::max_element(v.begin(), v.end()));

    thrust::host_vector<T> result_init(9);  // +3 padding for int8 tests
    result_init[0] = cudf::test::make_type_param_scalar<T>(0);
    if constexpr (cudf::is_chrono<T>()) {
      result_init[1] = T::max();
      result_init[2] = T::min();
    } else {
      result_init[1] = std::numeric_limits<T>::max();
      result_init[2] = std::numeric_limits<T>::min();
    }
    result_init[3] = result_init[0];
    result_init[4] = result_init[1];
    result_init[5] = result_init[2];

    auto dev_data = cudf::detail::make_device_uvector(
      v, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    auto dev_result = cudf::detail::make_device_uvector(
      result_init, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    if (block_size == 0) { block_size = vec_size; }

    if (is_cas_test) {
      gpu_atomicCAS_test<<<grid_size, block_size, 0, cudf::get_default_stream().value()>>>(
        dev_result.data(), dev_data.data(), vec_size);
    } else {
      gpu_atomic_test<<<grid_size, block_size, 0, cudf::get_default_stream().value()>>>(
        dev_result.data(), dev_data.data(), vec_size);
    }

    auto host_result = cudf::detail::make_host_vector(dev_result, cudf::get_default_stream());

    CUDF_CHECK_CUDA(cudf::get_default_stream().value());

    if (!is_timestamp_sum<T, cudf::DeviceSum>()) {
      EXPECT_EQ(host_result[0], exact[0]) << "atomicAdd test failed";
    }
    EXPECT_EQ(host_result[1], exact[1]) << "atomicMin test failed";
    EXPECT_EQ(host_result[2], exact[2]) << "atomicMax test failed";
    if (!is_timestamp_sum<T, cudf::DeviceSum>()) {
      EXPECT_EQ(host_result[3], exact[0]) << "atomicAdd test(2) failed";
    }
    EXPECT_EQ(host_result[4], exact[1]) << "atomicMin test(2) failed";
    EXPECT_EQ(host_result[5], exact[2]) << "atomicMax test(2) failed";
  }
};

TYPED_TEST_SUITE(AtomicsTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOps)
{
  bool is_cas_test = false;
  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCAS)
{
  bool is_cas_test = true;
  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOpsGrid)
{
  bool is_cas_test = false;
  int block_size   = 3;
  int grid_size    = 4;

  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test, block_size, grid_size);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCASGrid)
{
  bool is_cas_test = true;
  int block_size   = 3;
  int grid_size    = 4;

  std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  this->atomic_test(input_array, is_cas_test, block_size, grid_size);

  std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
  this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for large array
TYPED_TEST(AtomicsTest, atomicOpsRandom)
{
  bool is_cas_test = false;
  int block_size   = 256;
  int grid_size    = 64;

  std::vector<int> input_array(grid_size * block_size);

  std::default_random_engine engine;
  std::uniform_int_distribution<> dist(-10, 10);
  std::generate(input_array.begin(), input_array.end(), [&]() { return dist(engine); });

  this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

TYPED_TEST(AtomicsTest, atomicCASRandom)
{
  bool is_cas_test = true;
  int block_size   = 256;
  int grid_size    = 64;

  std::vector<int> input_array(grid_size * block_size);

  std::default_random_engine engine;
  std::uniform_int_distribution<> dist(-10, 10);
  std::generate(input_array.begin(), input_array.end(), [&]() { return dist(engine); });

  this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

__global__ void test_single_atomic_add_kernel(__int128_t* target, __int128_t value)
{
  cudf::detail::atomic_add(target, value);
}

class Atomic128Test : public cudf::test::BaseFixture {
 public:
  void run_atomic_add_test(__int128_t initial_value,
                           __int128_t add_value,
                           __int128_t expected_result)
  {
    cudf::detail::device_scalar<__int128_t> d_target(initial_value, cudf::get_default_stream());
    test_single_atomic_add_kernel<<<32, 256, 0, cudf::get_default_stream().value()>>>(
      d_target.data(), add_value);
    CUDF_CHECK_CUDA(cudf::get_default_stream().value());
    __int128_t result = d_target.value(cudf::get_default_stream());
    EXPECT_EQ(result, expected_result);
  }
};

TEST_F(Atomic128Test, BasicAddition) { run_atomic_add_test(0, 1, 32 * 256 * 1); }

TEST_F(Atomic128Test, CarryPropagation)
{
  constexpr int total_threads = 32 * 256;
  __int128_t initial_value    = cuda::std::numeric_limits<int64_t>::max() - total_threads + 1;
  __int128_t expected = static_cast<__int128_t>(cuda::std::numeric_limits<int64_t>::max()) + 1;
  run_atomic_add_test(initial_value, 1, expected);
}

TEST_F(Atomic128Test, NegativeNumbers)
{
  constexpr int total_threads = 32 * 256;
  run_atomic_add_test(0, -50, -total_threads * 50);
}

TEST_F(Atomic128Test, NegativeCarryPropagation)
{
  constexpr int total_threads = 32 * 256;
  __int128_t initial_value    = cuda::std::numeric_limits<int64_t>::min() + total_threads - 1;
  __int128_t expected = static_cast<__int128_t>(cuda::std::numeric_limits<int64_t>::min()) - 1;
  run_atomic_add_test(initial_value, -1, expected);
}

CUDF_TEST_PROGRAM_MAIN()
