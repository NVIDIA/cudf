/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief overloads for CUDA atomic operations
 * @file device_atomics.cuh
 *
 * Provides the overloads for all of cudf's data types, specifically int8_t,
 * int16_t, int32_t, int64_t, float, double, cudf::timestamp_D,
 * cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool for the CUDA atomic operations
 * `atomicAdd`, `atomicMin`, `atomicMax`, `atomicCAS`.
 *
 * Also provides `cudf::detail::genericAtomicOperation` which performs an
 * atomic operation with the given binary operator.
 */

#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/atomic>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace cudf {
namespace detail {

template <typename T, typename Op>
struct genericAtomicOperationImpl;

template <typename T, typename Op>
struct genericAtomicOperationImpl {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, Op op)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

// Optimized specializations using native atomic operations
template <typename T>
struct genericAtomicOperationImpl<T, DeviceSum> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceSum op)
    requires(cuda::std::is_arithmetic_v<T> && !cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    return atomic_addr.fetch_add(update_value, cuda::memory_order_relaxed);
  }

  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceSum op)
    requires(!cuda::std::is_arithmetic_v<T> || cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, DeviceMin> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMin op)
    requires(cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    return atomic_addr.fetch_min(update_value, cuda::memory_order_relaxed);
  }

  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMin op)
    requires(!cuda::std::is_integral_v<T> || cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

template <typename T>
struct genericAtomicOperationImpl<T, DeviceMax> {
  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMax op)
    requires(cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    return atomic_addr.fetch_max(update_value, cuda::memory_order_relaxed);
  }

  __forceinline__ __device__ T operator()(T* addr, T const& update_value, DeviceMax op)
    requires(!cuda::std::is_integral_v<T> || cuda::std::is_same_v<T, bool>)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T old_value = atomic_addr.load(cuda::memory_order_relaxed);
    T new_value;

    do {
      new_value = op(old_value, update_value);
    } while (!atomic_addr.compare_exchange_weak(old_value, new_value, cuda::memory_order_relaxed));

    return old_value;
  }
};

// -----------------------------------------------------------------------
// the implementation of `typesAtomicCASImpl`
template <typename T>
struct typesAtomicCASImpl {
  __forceinline__ __device__ T operator()(T* addr, T const& compare, T const& update_value)
  {
    cuda::atomic_ref<T> atomic_addr(*addr);
    T expected = compare;
    atomic_addr.compare_exchange_strong(expected, update_value, cuda::memory_order_relaxed);
    return expected;
  }
};

/**
 * @brief Compute atomic binary operation
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes 'BinaryOp'('old', 'update_value'),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `genericAtomicOperation` are:
 * int8_t, int16_t, int32_t, int64_t, float, double
 *
 * @param address The address of old value in global or shared memory
 * @param update_value The value to be computed
 * @param op  The binary operator used for compute
 *
 * @returns The old value at `address`
 */
template <typename T, typename BinaryOp>
T __forceinline__ __device__ genericAtomicOperation(T* address, T const& update_value, BinaryOp op)
  requires(cudf::is_numeric<T>())
{
  auto fun = cudf::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

// specialization for cudf::detail::timestamp types
template <typename T, typename BinaryOp>
T __forceinline__ __device__ genericAtomicOperation(T* address, T const& update_value, BinaryOp op)
  requires(cudf::is_timestamp<T>())
{
  using R = typename T::rep;
  // Unwrap the input timestamp to its underlying duration value representation.
  // Use the underlying representation's type to apply operation for the cudf::detail::timestamp
  auto update_value_rep = update_value.time_since_epoch().count();
  auto fun              = cudf::detail::genericAtomicOperationImpl<R, BinaryOp>{};
  return T{T::duration(fun(reinterpret_cast<R*>(address), update_value_rep, op))};
}

// specialization for cudf::detail::duration types
template <typename T, typename BinaryOp>
T __forceinline__ __device__ genericAtomicOperation(T* address, T const& update_value, BinaryOp op)
  requires(cudf::is_duration<T>())
{
  using R = typename T::rep;
  // Unwrap the input duration to its underlying duration value representation.
  // Use the underlying representation's type to apply operation for the cudf::detail::duration
  auto update_value_rep = update_value.count();
  auto fun              = cudf::detail::genericAtomicOperationImpl<R, BinaryOp>{};
  return T(fun(reinterpret_cast<R*>(address), update_value_rep, op));
}

// specialization for bool types
template <typename BinaryOp>
__forceinline__ __device__ bool genericAtomicOperation(bool* address,
                                                       bool const& update_value,
                                                       BinaryOp op)
{
  using T = bool;
  // don't use underlying type to apply operation for bool
  auto fun = cudf::detail::genericAtomicOperationImpl<T, BinaryOp>{};
  return T(fun(address, update_value, op));
}

/**
 * @brief Overloads for `atomic_add`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes (old + val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_add` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 *
 * CUDA natively supports `int32_t`, `uint32_t`, `uint64_t`, `float`, `double.
 * (`double` is supported after Pascal).
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be added
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_add(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceSum{});
}

/**
 * @brief Overloads for `atomic_mul`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes (old * val), and stores the result back to memory at the same
 * address. These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomicMul` are:
 * int8_t, int16_t, int32_t, int64_t, float, double, and bool
 *
 * All types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be multiplied
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_mul(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceProduct{});
}

/**
 * @brief Overloads for `atomic_min`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes the minimum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_min` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 *
 * CUDA natively supports `int32_t`, `uint32_t`, `int64_t`, `uint64_t`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_min(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceMin{});
}

/**
 * @brief Overloads for `atomic_max`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes the maximum of old and val, and stores the result back to memory
 * at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_max` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 *
 * CUDA natively supports `int32_t`, `uint32_t`, `int64_t`, `uint64_t`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_max(T* address, T val)
{
  return cudf::detail::genericAtomicOperation(address, val, cudf::DeviceMax{});
}

/**
 * @brief Overloads for `atomic_cas`
 *
 * Reads the `old` located at the `address` in global or shared memory,
 * computes (`old` == `compare` ? `val` : `old`),
 * and stores the result back to memory at the same address.
 * These three operations are performed in one atomic transaction.
 *
 * The supported cudf types for `atomic_cas` are:
 * int8_t, int16_t, int32_t, int64_t, float, double,
 * cudf::timestamp_D, cudf::timestamp_s, cudf::timestamp_ms, cudf::timestamp_us,
 * cudf::timestamp_ns, cudf::duration_D, cudf::duration_s, cudf::duration_ms,
 * cudf::duration_us, cudf::duration_ns and bool
 * CUDA natively supports `int32_t`, `uint32_t`, `uint64_t`.
 * Other types are implemented by `atomicCAS`.
 *
 * @param address The address of old value in global or shared memory
 * @param compare The value to be compared
 * @param val The value to be computed
 *
 * @returns The old value at `address`
 */
template <typename T>
__forceinline__ __device__ T atomic_cas(T* address, T compare, T val)
{
  return cudf::detail::typesAtomicCASImpl<T>()(address, compare, val);
}

/**
 * @brief Helper function to calculate carry for 64-bit addition
 *
 * @param old_val The original 64-bit value
 * @param add_val The value being added
 * @param carry_in Carry from previous addition
 * @return 1 if carry is generated, 0 otherwise
 */
__device__ __forceinline__ uint64_t calculate_carry_64(uint64_t old_val,
                                                       uint64_t add_val,
                                                       uint64_t carry_in)
{
  // Use __uint128_t to detect overflow beyond 64-bit range
  __uint128_t sum = static_cast<__uint128_t>(old_val) + add_val + carry_in;
  return sum > cuda::std::numeric_limits<uint64_t>::max();
}

/**
 * @brief Atomic addition for __int128_t with architecture-specific optimization
 *
 * Uses native 128-bit CAS on Hopper+ GPUs for optimal performance, except for CUDA 13.0 on
 * Blackwell. Other configurations fall back to two 64-bit atomic additions with carry
 * propagation.
 *
 * @param address Pointer to the __int128_t value
 * @param val Value to add
 * @return The old value before addition
 */
__forceinline__ __device__ __int128_t atomic_add(__int128_t* address, __int128_t val)
{
  // CUDA 13.0 miscompiles the native 128-bit CAS on Blackwell; the fix is confirmed in 13.3, so
  // gate off all 13.x before 13.3. See https://github.com/rapidsai/cudf/issues/23150.
#if __CUDA_ARCH__ >= 900 && \
  !(__CUDA_ARCH__ >= 1000 && __CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 3)
  __int128_t expected, desired;

  do {
    expected = *address;
    desired  = expected + val;
  } while (atomicCAS(address, expected, desired) != expected);

  return expected;
#else
  uint64_t* const target_ptr         = reinterpret_cast<uint64_t*>(address);
  __uint128_t const add_val_unsigned = static_cast<__uint128_t>(val);

  // Split the 128-bit add value into two 64-bit parts
  auto const add_low  = static_cast<uint64_t>(add_val_unsigned);
  auto const add_high = static_cast<uint64_t>(add_val_unsigned >> 64);

  uint64_t old_parts[2];
  old_parts[0]     = atomicAdd(reinterpret_cast<unsigned long long*>(target_ptr), add_low);
  auto const carry = calculate_carry_64(old_parts[0], add_low, 0);
  old_parts[1] = atomicAdd(reinterpret_cast<unsigned long long*>(target_ptr + 1), add_high + carry);

  auto const old_val_unsigned = (static_cast<__uint128_t>(old_parts[1]) << 64) | old_parts[0];
  return static_cast<__int128_t>(old_val_unsigned);
#endif
}

}  // namespace detail
}  // namespace cudf
