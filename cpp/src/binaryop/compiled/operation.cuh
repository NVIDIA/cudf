/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/integral_math.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include <cmath>

namespace cudf {
namespace binops {
namespace compiled {

// All binary operations
namespace ops {

struct Add {
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }
};

struct Sub {
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs - rhs)
  {
    return lhs - rhs;
  }
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
           (is_duration<TypeLhs>() and cuda::std::is_integral<TypeRhs>()) or
           (cuda::std::is_integral<TypeLhs>() and is_duration<TypeRhs>()) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs * rhs)
    requires(is_supported<T1, T2>())
  {
    return lhs * rhs;
  }
};

struct Div {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error on chrono:917
           (is_duration<TypeLhs>() and
            (cuda::std::is_integral<TypeRhs>() or is_duration<TypeRhs>())) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs / rhs)
    requires(is_supported<T1, T2>())
  {
    return lhs / rhs;
  }
};

struct TrueDiv {
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs)
    -> decltype((static_cast<double>(lhs) / static_cast<double>(rhs)))
  {
    return (static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

struct FloorDiv {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x / y)
    requires(cuda::std::integral<cuda::std::common_type_t<TypeLhs, TypeRhs>>)
  {
    return cudf::detail::integral_floor_div(x, y);
  }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> float
    requires(cuda::std::same_as<cuda::std::common_type_t<TypeLhs, TypeRhs>, float>)
  {
    return floorf(x / y);
  }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(cuda::std::same_as<cuda::std::common_type_t<TypeLhs, TypeRhs>, double>)
  {
    return floor(x / y);
  }
};

struct Mod {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error
           //_deps/libcudacxx-src/include/cuda/std/detail/libcxx/include/chrono(1337):
           // error : expression must have integral or unscoped enum type
           (is_duration<TypeLhs>() and
            (cuda::std::is_integral<TypeRhs>() or is_duration<TypeRhs>()));
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs % rhs)
    requires(is_supported<T1, T2>())
  {
    return lhs % rhs;
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> float
    requires(cuda::std::same_as<float, std::common_type_t<T1, T2>>)
  {
    return fmodf(static_cast<float>(lhs), static_cast<float>(rhs));
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> double
    requires(cuda::std::same_as<double, std::common_type_t<T1, T2>>)
  {
    return fmod(static_cast<double>(lhs), static_cast<double>(rhs));
  }
};

struct PMod {
  // Ideally, these two specializations - one for integral types and one for non integral
  // types shouldn't be required, as std::fmod should promote integral types automatically
  // to double and call the std::fmod overload for doubles. Sadly, doing this in jitified
  // code does not work - it is having trouble deciding between float/double overloads
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    requires(cuda::std::integral<std::common_type_t<TypeLhs, TypeRhs>>)
  {
    using common_t = std::common_type_t<TypeLhs, TypeRhs>;
    auto xconv     = static_cast<common_t>(x);
    auto yconv     = static_cast<common_t>(y);
    auto rem       = xconv % yconv;
    if constexpr (cuda::std::is_signed_v<decltype(rem)>)
      if (rem < 0) rem = (rem + yconv) % yconv;
    return rem;
  }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    requires(cuda::std::floating_point<std::common_type_t<TypeLhs, TypeRhs>>)
  {
    using common_t = std::common_type_t<TypeLhs, TypeRhs>;
    auto xconv     = static_cast<common_t>(x);
    auto yconv     = static_cast<common_t>(y);
    auto rem       = std::fmod(xconv, yconv);
    if (rem < 0) rem = std::fmod(rem + yconv, yconv);
    return rem;
  }

  template <cudf::concepts::fixed_point TypeLhs, cuda::std::same_as<TypeLhs> TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
  {
    auto const remainder = x % y;
    return remainder.value() < 0 ? (remainder + y) % y : remainder;
  }
};

struct PyMod {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(((x % y) + y) % y)
    requires(cuda::std::integral<std::common_type_t<TypeLhs, TypeRhs>> or
             (cudf::concepts::fixed_point<TypeLhs> and cuda::std::same_as<TypeLhs, TypeRhs>))
  {
    return ((x % y) + y) % y;
  }

  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(cuda::std::floating_point<std::common_type_t<TypeLhs, TypeRhs>>)
  {
    auto x1 = static_cast<double>(x);
    auto y1 = static_cast<double>(y);
    return fmod(fmod(x1, y1) + y1, y1);
  }

  template <cudf::concepts::duration TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(((x % y) + y) % y)
  {
    return ((x % y) + y) % y;
  }
};

struct Pow {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(std::is_convertible_v<TypeLhs, double> and std::is_convertible_v<TypeRhs, double>)
  {
    return pow(static_cast<double>(x), static_cast<double>(y));
  }
};

struct IntPow {
  template <cuda::std::integral TypeLhs, cuda::std::integral TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> TypeLhs
  {
    return cudf::detail::integral_pow(x, y);
  }
};

struct LogBase {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(std::is_convertible_v<TypeLhs, double> and std::is_convertible_v<TypeRhs, double>)
  {
    return (std::log(static_cast<double>(x)) / std::log(static_cast<double>(y)));
  }
};

struct ATan2 {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> double
    requires(std::is_convertible_v<TypeLhs, double> and std::is_convertible_v<TypeRhs, double>)
  {
    return std::atan2(static_cast<double>(x), static_cast<double>(y));
  }
};

struct ShiftLeft {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x << y)
  {
    return (x << y);
  }
};

struct ShiftRight {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x >> y)
  {
    return (x >> y);
  }
};

struct ShiftRightUnsigned {
  template <cudf::concepts::integral_not_bool TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y)
    -> decltype(static_cast<cuda::std::make_unsigned_t<TypeLhs>>(x) >> y)
  {
    return (static_cast<cuda::std::make_unsigned_t<TypeLhs>>(x) >> y);
  }
};

struct BitwiseAnd {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x & y)
  {
    return (x & y);
  }
};

struct BitwiseOr {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x | y)
  {
    return (x | y);
  }
};

struct BitwiseXor {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x ^ y)
  {
    return (x ^ y);
  }
};

struct LogicalAnd {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x && y)
  {
    return (x && y);
  }
};

struct LogicalOr {
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x || y)
  {
    return (x || y);
  }
};

/**
 * @brief Operand types whose comparison depends on `nan_semantics`: at least one of them is
 * floating-point
 */
template <typename TypeLhs, typename TypeRhs>
concept has_floating_point_operand =
  cuda::std::floating_point<TypeLhs> or cuda::std::floating_point<TypeRhs>;

/**
 * @brief Returns true if `value` is a floating-point NaN
 */
template <typename T>
__device__ inline bool is_nan_value(T value)
{
  if constexpr (cuda::std::floating_point<T>) {
    return isnan(value);
  } else {
    return false;
  }
}

/**
 * @brief Returns `x == y` with `nan_semantics::TOTALLY_ORDERED`: NaNs are equal to each other
 */
template <typename TypeLhs, typename TypeRhs>
__device__ inline bool totally_ordered_equal(TypeLhs x, TypeRhs y)
{
  auto const x_nan = is_nan_value(x);
  auto const y_nan = is_nan_value(y);
  return (x_nan or y_nan) ? (x_nan and y_nan) : (x == y);
}

/**
 * @brief Returns `x < y` with `nan_semantics::TOTALLY_ORDERED`: NaN is greater than every other
 * value
 */
template <typename TypeLhs, typename TypeRhs>
__device__ inline bool totally_ordered_less(TypeLhs x, TypeRhs y)
{
  auto const x_nan = is_nan_value(x);
  auto const y_nan = is_nan_value(y);
  // With a NaN operand, x < y only when y is the NaN and x is not
  return (x_nan or y_nan) ? not x_nan : (x < y);
}

// The comparison operators below hold how NaN values compare. Default construction selects
// `nan_semantics::UNORDERED`; the `nan_semantics::TOTALLY_ORDERED` branch only exists for
// floating-point operands and is selected at runtime.

struct Equal {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x == y)
  {
    if constexpr (has_floating_point_operand<TypeLhs, TypeRhs>) {
      if (semantics == nan_semantics::TOTALLY_ORDERED) { return totally_ordered_equal(x, y); }
    }
    return (x == y);
  }
};

struct NotEqual {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x != y)
  {
    if constexpr (has_floating_point_operand<TypeLhs, TypeRhs>) {
      if (semantics == nan_semantics::TOTALLY_ORDERED) { return not totally_ordered_equal(x, y); }
    }
    return (x != y);
  }
};

struct Less {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x < y)
  {
    if constexpr (has_floating_point_operand<TypeLhs, TypeRhs>) {
      if (semantics == nan_semantics::TOTALLY_ORDERED) { return totally_ordered_less(x, y); }
    }
    return (x < y);
  }
};

struct Greater {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x > y)
  {
    if constexpr (has_floating_point_operand<TypeLhs, TypeRhs>) {
      if (semantics == nan_semantics::TOTALLY_ORDERED) { return totally_ordered_less(y, x); }
    }
    return (x > y);
  }
};

struct LessEqual {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x <= y)
  {
    if constexpr (has_floating_point_operand<TypeLhs, TypeRhs>) {
      if (semantics == nan_semantics::TOTALLY_ORDERED) { return not totally_ordered_less(y, x); }
    }
    return (x <= y);
  }
};

struct GreaterEqual {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x >= y)
  {
    if constexpr (has_floating_point_operand<TypeLhs, TypeRhs>) {
      if (semantics == nan_semantics::TOTALLY_ORDERED) { return not totally_ordered_less(x, y); }
    }
    return (x >= y);
  }
};

struct NullEquals {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) -> decltype(x == y)
  {
    output_valid = true;
    if (lhs_valid && rhs_valid) return Equal{semantics}(x, y);
    return !lhs_valid && !rhs_valid;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x == y);
};

struct NullNotEquals {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) -> decltype(x != y)
  {
    return !NullEquals{semantics}(x, y, lhs_valid, rhs_valid, output_valid);
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs, typename TypeRhs>
  __device__ inline auto operator()(TypeLhs x, TypeRhs y) -> decltype(x != y);
};

struct NullMax {
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
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
      return Greater{semantics}(x_conv, y_conv) ? x_conv : y_conv;
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
  nan_semantics semantics{nan_semantics::UNORDERED};  ///< How NaN values compare
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
      return Less{semantics}(x_conv, y_conv) ? x_conv : y_conv;
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

/**
 * @brief Operators that take the operands' validity and set the output validity
 */
template <typename BinaryOperator>
constexpr bool is_null_aware_op_v =
  cuda::std::is_same_v<BinaryOperator, NullEquals> or
  cuda::std::is_same_v<BinaryOperator, NullNotEquals> or
  cuda::std::is_same_v<BinaryOperator, NullLogicalAnd> or
  cuda::std::is_same_v<BinaryOperator, NullLogicalOr> or
  cuda::std::is_same_v<BinaryOperator, NullMax> or cuda::std::is_same_v<BinaryOperator, NullMin>;

}  // namespace ops
}  // namespace compiled
}  // namespace binops
}  // namespace cudf
