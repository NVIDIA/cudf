/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/integral_math.cuh>
#include <cudf/utilities/traits.hpp>

#include <cmath>

namespace cudf {
namespace binops {
namespace compiled {

// All binary operations
namespace ops {

struct Add {
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs + rhs)
  { return lhs + rhs; }
};

struct Sub {
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs - rhs)
  { return lhs - rhs; }
};

struct Mul {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without the following line, compilation error
           // _deps/libcudacxx-src/include/cuda/std/detail/libcxx/include/chrono(917): error:
           // identifier "cuda::std::__3::ratio<(long)86400000000l, (long)1l> ::num" is undefined in
           // device code
           (is_duration<TypeLhs>() and std::is_integral<TypeRhs>()) or
           (std::is_integral<TypeLhs>() and is_duration<TypeRhs>()) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs * rhs)
    requires(is_supported<T1, T2>())
  { return lhs * rhs; }
};

struct Div {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error on chrono:917
           (is_duration<TypeLhs>() and (std::is_integral<TypeRhs>() or is_duration<TypeRhs>())) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs / rhs)
    requires(is_supported<T1, T2>())
  { return lhs / rhs; }
};

struct TrueDiv {
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs)
    -> decltype((static_cast<double>(lhs) / static_cast<double>(rhs)))
  { return (static_cast<double>(lhs) / static_cast<double>(rhs)); }
};

struct FloorDiv {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x / y)
    requires(cuda::std::is_integral_v<cuda::std::common_type_t<TypeLhs, TypeRhs>>)
  { return cudf::detail::integral_floor_div(x, y); }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> float
    requires(cuda::std::is_same_v<cuda::std::common_type_t<TypeLhs, TypeRhs>, float>)
  { return floorf(x / y); }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(cuda::std::is_same_v<cuda::std::common_type_t<TypeLhs, TypeRhs>, double>)
  { return floor(x / y); }
};

struct Mod {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error
           //_deps/libcudacxx-src/include/cuda/std/detail/libcxx/include/chrono(1337):
           // error : expression must have integral or unscoped enum type
           (is_duration<TypeLhs>() and (std::is_integral<TypeRhs>() or is_duration<TypeRhs>()));
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs % rhs)
    requires(is_supported<T1, T2>())
  { return lhs % rhs; }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> float
    requires(std::is_same_v<float, std::common_type_t<T1, T2>>)
  { return fmodf(static_cast<float>(lhs), static_cast<float>(rhs)); }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> double
    requires(std::is_same_v<double, std::common_type_t<T1, T2>>)
  { return fmod(static_cast<double>(lhs), static_cast<double>(rhs)); }
};

struct PMod {
  // Ideally, these two specializations - one for integral types and one for non integral
  // types shouldn't be required, as std::fmod should promote integral types automatically
  // to double and call the std::fmod overload for doubles. Sadly, doing this in jitified
  // code does not work - it is having trouble deciding between float/double overloads
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    requires(std::is_integral_v<std::common_type_t<TypeLhs, TypeRhs>>)
  {
    using common_t = std::common_type_t<TypeLhs, TypeRhs>;
    auto xconv     = static_cast<common_t>(x);
    auto yconv     = static_cast<common_t>(y);
    auto rem       = xconv % yconv;
    if constexpr (std::is_signed_v<decltype(rem)>)
      if (rem < 0) rem = (rem + yconv) % yconv;
    return rem;
  }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    requires(std::is_floating_point_v<std::common_type_t<TypeLhs, TypeRhs>>)
  {
    using common_t = std::common_type_t<TypeLhs, TypeRhs>;
    auto xconv     = static_cast<common_t>(x);
    auto yconv     = static_cast<common_t>(y);
    auto rem       = std::fmod(xconv, yconv);
    if (rem < 0) rem = std::fmod(rem + yconv, yconv);
    return rem;
  }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    requires(cudf::is_fixed_point<TypeLhs>() and std::is_same_v<TypeLhs, TypeRhs>)
  {
    auto const remainder = x % y;
    return remainder.value() < 0 ? (remainder + y) % y : remainder;
  }
};

struct PyMod {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(((x % y) + y) % y)
    requires(std::is_integral_v<std::common_type_t<TypeLhs, TypeRhs>> or
             (cudf::is_fixed_point<TypeLhs>() and std::is_same_v<TypeLhs, TypeRhs>))
  { return ((x % y) + y) % y; }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(std::is_floating_point_v<std::common_type_t<TypeLhs, TypeRhs>>)
  {
    auto x1 = static_cast<double>(x);
    auto y1 = static_cast<double>(y);
    return fmod(fmod(x1, y1) + y1, y1);
  }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(((x % y) + y) % y)
    requires(is_duration<TypeLhs>())
  { return ((x % y) + y) % y; }
};

struct Pow {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(std::is_convertible_v<TypeLhs, double> and std::is_convertible_v<TypeRhs, double>)
  { return pow(static_cast<double>(x), static_cast<double>(y)); }
};

struct IntPow {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> TypeLhs
    requires(std::is_integral_v<TypeLhs> and std::is_integral_v<TypeRhs>)
  { return cudf::detail::integral_pow(x, y); }
};

struct LogBase {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(std::is_convertible_v<TypeLhs, double> and std::is_convertible_v<TypeRhs, double>)
  { return (std::log(static_cast<double>(x)) / std::log(static_cast<double>(y))); }
};

struct ATan2 {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(std::is_convertible_v<TypeLhs, double> and std::is_convertible_v<TypeRhs, double>)
  { return std::atan2(static_cast<double>(x), static_cast<double>(y)); }
};

struct ShiftLeft {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x << y)
  { return (x << y); }
};

struct ShiftRight {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x >> y)
  { return (x >> y); }
};

struct ShiftRightUnsigned {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    -> decltype(static_cast<cuda::std::make_unsigned_t<TypeLhs>>(x) >> y)
    requires(cuda::std::is_integral_v<TypeLhs> and not is_boolean<TypeLhs>())
  { return (static_cast<cuda::std::make_unsigned_t<TypeLhs>>(x) >> y); }
};

struct BitwiseAnd {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x & y)
  { return (x & y); }
};

struct BitwiseOr {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x | y)
  { return (x | y); }
};

struct BitwiseXor {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x ^ y)
  { return (x ^ y); }
};

struct LogicalAnd {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x && y)
  { return (x && y); }
};

struct LogicalOr {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x || y)
  { return (x || y); }
};

struct Equal {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x == y)
  { return (x == y); }
};

struct NotEqual {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x != y)
  { return (x != y); }
};

struct Less {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x < y)
  { return (x < y); }
};

struct Greater {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x > y)
  { return (x > y); }
};

struct LessEqual {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x <= y)
  { return (x <= y); }
};

struct GreaterEqual {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x >= y)
  { return (x >= y); }
};

struct NullEquals {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) -> decltype(x == y)
  {
    output_valid = true;
    if (lhs_valid && rhs_valid) return x == y;
    return !lhs_valid && !rhs_valid;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x == y);
};

struct NullNotEquals {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) -> decltype(x != y)
  { return !NullEquals{}(x, y, lhs_valid, rhs_valid, output_valid); }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x != y);
};

struct NullMax {
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) > static_cast<common_t>(y) ? x : y))
  {
    output_valid      = true;
    auto const x_conv = static_cast<common_t>(x);
    auto const y_conv = static_cast<common_t>(y);
    if (!lhs_valid && !rhs_valid) {
      output_valid = false;
      return common_t{};
    } else if (lhs_valid && rhs_valid) {
      return (x_conv > y_conv) ? x_conv : y_conv;
    } else if (lhs_valid)
      return x_conv;
    else
      return y_conv;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) > static_cast<common_t>(y) ? x : y));
};

struct NullMin {
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) < static_cast<common_t>(y) ? x : y))
  {
    output_valid      = true;
    auto const x_conv = static_cast<common_t>(x);
    auto const y_conv = static_cast<common_t>(y);
    if (!lhs_valid && !rhs_valid) {
      output_valid = false;
      return common_t{};
    } else if (lhs_valid && rhs_valid) {
      return (x_conv < y_conv) ? x_conv : y_conv;
    } else if (lhs_valid)
      return x_conv;
    else
      return y_conv;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) < static_cast<common_t>(y) ? x : y));
};

struct NullLogicalAnd {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) -> decltype(x && y)
  {
    bool lhs_false  = lhs_valid && !x;
    bool rhs_false  = rhs_valid && !y;
    bool both_valid = lhs_valid && rhs_valid;
    output_valid    = lhs_false || rhs_false || both_valid;
    return both_valid && !lhs_false && !rhs_false;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x && y);
};

struct NullLogicalOr {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) -> decltype(x || y)
  {
    bool lhs_true   = lhs_valid && x;
    bool rhs_true   = rhs_valid && y;
    bool both_valid = lhs_valid && rhs_valid;
    output_valid    = lhs_true || rhs_true || both_valid;
    return lhs_true || rhs_true;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x || y);
};

}  // namespace ops
}  // namespace compiled
}  // namespace binops
}  // namespace cudf
