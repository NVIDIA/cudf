/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief Definition of the device operators
 * @file
 */

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/fixed_point/temporary.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/algorithm.max.h>
#include <cuda/std/algorithm.min.h>
#include <cuda/std/bit>
#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Constrained min function suitable for std::is_invocable
 */
template <typename LHS, typename RHS>
CUDF_HOST_DEVICE inline auto min(LHS const& lhs, RHS const& rhs)
  requires(cuda::std::totally_ordered_with<LHS, RHS>)
{
  return cuda::std::min(lhs, rhs);
}

/**
 * @brief Constrained max function suitable for std::is_invocable
 */
template <typename LHS, typename RHS>
CUDF_HOST_DEVICE inline auto max(LHS const& lhs, RHS const& rhs)
  requires(cuda::std::totally_ordered_with<LHS, RHS>)
{
  return cuda::std::max(lhs, rhs);
}

// The NaN tests and comparisons in the IEEE 754 helpers below operate on the integer bit pattern
// rather than using floating-point instructions. On many GPUs (for example workstation and
// consumer parts) double-precision throughput is 1/32 to 1/64 of single-precision, and the extra
// FP64 instructions needed for the NaN and signed-zero checks made double min/max compute-bound:
// reductions, scans, and groupby were up to 2x slower than the memory-bound single compare they
// replace. Integer operations run at full rate on all GPUs.

/**
 * @brief Signed integer type with the same size as the floating-point type `T`
 */
template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
using float_bits_t = cuda::std::conditional_t<sizeof(T) == 4, int32_t, int64_t>;

/**
 * @brief Returns true if `value` is NaN
 *
 * The test uses integer operations on the bit pattern. This avoids double-precision instructions,
 * which have low throughput on many GPUs.
 */
template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
CUDF_HOST_DEVICE inline bool is_nan_bits(T value)
{
  using I = float_bits_t<T>;
  auto constexpr inf_bits =
    static_cast<I>(sizeof(T) == 4 ? 0x7f80'0000LL : 0x7ff0'0000'0000'0000LL);
  return (cuda::std::bit_cast<I>(value) & cuda::std::numeric_limits<I>::max()) > inf_bits;
}

/**
 * @brief Maps a non-NaN floating-point value to a signed integer with the same ordering
 *
 * Negative values have their magnitude bits inverted, so integer comparison orders the keys as
 * the values, with -0 less than +0.
 */
template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
CUDF_HOST_DEVICE inline float_bits_t<T> ordered_bits(T value)
{
  using I         = float_bits_t<T>;
  auto const bits = cuda::std::bit_cast<I>(value);
  return bits ^ ((bits >> (sizeof(I) * 8 - 1)) & cuda::std::numeric_limits<I>::max());
}

/**
 * @brief IEEE 754-2019 `minimum` of two floating-point values
 *
 * NaN propagates and -0 is less than +0, so the operation is associative and commutative and the
 * result does not depend on the evaluation order. Only integer operations are used.
 */
template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
CUDF_HOST_DEVICE inline T ieee_minimum(T lhs, T rhs)
{
  if (is_nan_bits(lhs)) { return lhs; }
  if (is_nan_bits(rhs)) { return rhs; }
  return ordered_bits(lhs) <= ordered_bits(rhs) ? lhs : rhs;
}

/**
 * @brief IEEE 754-2019 `maximum` of two floating-point values
 *
 * NaN propagates and +0 is greater than -0, so the operation is associative and commutative and
 * the result does not depend on the evaluation order. Only integer operations are used.
 */
template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
CUDF_HOST_DEVICE inline T ieee_maximum(T lhs, T rhs)
{
  if (is_nan_bits(lhs)) { return lhs; }
  if (is_nan_bits(rhs)) { return rhs; }
  return ordered_bits(lhs) >= ordered_bits(rhs) ? lhs : rhs;
}

/**
 * @brief `minimum` of two floating-point values with `nan_semantics::TOTALLY_ORDERED` ordering
 *
 * NaN is greater than every other value, so a NaN is returned only when both inputs are NaN.
 * In that case the canonical quiet NaN is returned so the result does not depend on the order
 * of the inputs. As in `ieee_minimum`, -0 is less than +0 and only integer operations are used.
 */
template <typename T>
  requires(cuda::std::is_floating_point_v<T>)
CUDF_HOST_DEVICE inline T totally_ordered_minimum(T lhs, T rhs)
{
  auto const lhs_nan = is_nan_bits(lhs);
  auto const rhs_nan = is_nan_bits(rhs);
  if (lhs_nan || rhs_nan) {
    if (lhs_nan && rhs_nan) { return cuda::std::numeric_limits<T>::quiet_NaN(); }
    return lhs_nan ? rhs : lhs;
  }
  return ordered_bits(lhs) <= ordered_bits(rhs) ? lhs : rhs;
}
}  // namespace detail

/**
 * @brief Binary `sum` operator
 */
struct DeviceSum {
  template <typename T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs) -> decltype(lhs + rhs)
    requires(not cudf::concepts::timestamp<T>)
  {
    return lhs + rhs;
  }

  template <cudf::concepts::timestamp T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{typename T::duration{0}};
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{0};
  }

  template <cudf::concepts::fixed_point T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support device operator identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support device operator identity");
#endif
    return T{};
  }
};

/**
 * @brief `count` operator - used in rolling windows
 */
struct DeviceCount {
  template <cudf::concepts::timestamp T>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs)
  {
    return T{DeviceCount{}(lhs.time_since_epoch(), rhs.time_since_epoch())};
  }

  template <typename T>
  CUDF_HOST_DEVICE inline T operator()(T const&, T const& rhs)
  {
    return rhs + T{1};
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{};
  }
};

/**
 * @brief binary `min` operator
 *
 * Floating-point values follow the IEEE 754-2019 `minimum` operation (see `ieee_minimum`).
 */
struct DeviceMin {
  template <cudf::concepts::fixed_width T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs)
    -> decltype(numeric::detail::min(lhs, rhs))
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      return cudf::detail::ieee_minimum(lhs, rhs);
    } else {
      return numeric::detail::min(lhs, rhs);
    }
  }

  template <typename T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs)
    -> decltype(cudf::detail::min(lhs, rhs))
  {
    return cudf::detail::min(lhs, rhs);
  }

  template <cudf::concepts::arithmetic T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::max();
    }
  }

  template <cudf::concepts::fixed_point T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceMin identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceMin identity");
#endif
    return cuda::std::numeric_limits<T>::max();
  }

  // identity specialized for string_view and chrono types
  // chrono types do not have std::numeric_limits specializations and should use T::max()
  // https://eel.is/c++draft/numeric.limits.general#6
  template <typename T>
    requires(cudf::concepts::string_view<T> or cudf::concepts::chrono<T>)
  CUDF_HOST_DEVICE inline static constexpr T identity()
  {
    return T::max();
  }

  template <cudf::concepts::dictionary T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return static_cast<T>(T::max_value());
  }
};

/**
 * @brief binary `min` operator with `nan_semantics::TOTALLY_ORDERED` ordering of floating-point
 * values
 *
 * NaN is greater than every other value, so it is the result only when every input is NaN (see
 * `totally_ordered_minimum`). The identity is therefore NaN instead of +infinity.
 * Non-floating-point types behave the same as `DeviceMin`.
 */
struct DeviceMinTotallyOrdered {
  template <typename T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs)
    -> decltype(DeviceMin{}(lhs, rhs))
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      return cudf::detail::totally_ordered_minimum(lhs, rhs);
    } else {
      return DeviceMin{}(lhs, rhs);
    }
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
    requires(requires { DeviceMin::identity<T>(); })
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      return cuda::std::numeric_limits<T>::quiet_NaN();
    } else {
      return DeviceMin::identity<T>();
    }
  }
};

/**
 * @brief binary `max` operator
 *
 * Floating-point values follow the IEEE 754-2019 `maximum` operation (see `ieee_maximum`).
 */
struct DeviceMax {
  template <cudf::concepts::fixed_width T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs)
    -> decltype(numeric::detail::max(lhs, rhs))
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      return cudf::detail::ieee_maximum(lhs, rhs);
    } else {
      return numeric::detail::max(lhs, rhs);
    }
  }

  template <typename T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs)
    -> decltype(cudf::detail::max(lhs, rhs))
  {
    return cudf::detail::max(lhs, rhs);
  }

  template <cudf::concepts::arithmetic T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return -cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::lowest();
    }
  }

  template <cudf::concepts::fixed_point T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceMax identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceMax identity");
#endif
    return cuda::std::numeric_limits<T>::lowest();
  }

  // identity specialized for string_view and chrono types
  // chrono types do not have std::numeric_limits specializations and should use T::min()
  // https://eel.is/c++draft/numeric.limits.general#6
  template <typename T>
    requires(cudf::concepts::string_view<T> or cudf::concepts::chrono<T>)
  CUDF_HOST_DEVICE inline static constexpr T identity()
  {
    return T::min();
  }

  template <cudf::concepts::dictionary T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return static_cast<T>(T::lowest_value());
  }
};

/**
 * @brief binary `product` operator
 */
struct DeviceProduct {
  template <typename T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs) -> decltype(lhs * rhs)
    requires(not cudf::concepts::timestamp<T>)
  {
    return lhs * rhs;
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{1};
  }

  template <cudf::concepts::fixed_point T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceProduct identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceProduct identity");
#endif
    return T{1, numeric::scale_type{0}};
  }
};

/**
 * @brief Operator for calculating Lead/Lag window function.
 */
struct DeviceLeadLag {
  size_type const row_offset;

  explicit CUDF_HOST_DEVICE inline DeviceLeadLag(size_type offset_) : row_offset(offset_) {}
};

/**
 * @brief Binary bitwise `AND` operator
 */
struct DeviceBitAnd {
  template <cuda::std::integral T>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs) const
  {
    return lhs & rhs;
  }

  template <cuda::std::integral T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    if constexpr (std::is_same_v<T, bool>) {
      return true;
    } else {
      return ~T{0};
    }
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Bitwise AND is only supported for integral types.");
#else
    CUDF_UNREACHABLE("Bitwise AND is only supported for integral types.");
#endif
    return T{};
  }
};

/**
 * @brief Binary bitwise `OR` operator
 */
struct DeviceBitOr {
  template <cuda::std::integral T>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs) const
  {
    return lhs | rhs;
  }

  template <cuda::std::integral T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{0};
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Bitwise OR is only supported for integral types.");
#else
    CUDF_UNREACHABLE("Bitwise OR is only supported for integral types.");
#endif
    return T{};
  }
};

/**
 * @brief Binary bitwise `XOR` operator
 */
struct DeviceBitXor {
  template <cuda::std::integral T>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs) const
  {
    return lhs ^ rhs;
  }

  template <cuda::std::integral T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{0};
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Bitwise XOR is only supported for integral types.");
#else
    CUDF_UNREACHABLE("Bitwise XOR is only supported for integral types.");
#endif
    return T{};
  }
};

}  // namespace cudf
