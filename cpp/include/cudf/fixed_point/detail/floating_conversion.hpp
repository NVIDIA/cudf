/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace numeric {
namespace detail {

/**
 * @brief Determine the number of significant bits in an integer
 *
 * @tparam T Type of input integer value. Must be either uint32_t, uint64_t, or __uint128_t
 * @param value The integer whose bits are being counted
 * @return The number of significant bits: the # of bits - # of leading zeroes
 */
template <typename T,
          CUDF_ENABLE_IF(cuda::std::is_same_v<T, uint32_t> || cuda::std::is_same_v<T, uint64_t> ||
                         cuda::std::is_same_v<T, __uint128_t>)>
CUDF_HOST_DEVICE inline constexpr int count_significant_bits(T value)
{
#ifdef __CUDA_ARCH__
  if constexpr (cuda::std::is_same_v<T, uint64_t>) {
    return 64 - __clzll(static_cast<int64_t>(value));
  } else if constexpr (cuda::std::is_same_v<T, uint32_t>) {
    return 32 - __clz(static_cast<int32_t>(value));
  } else if constexpr (cuda::std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<int64_t>(value >> 64);
    auto const low_bits  = static_cast<int64_t>(value);
    return 128 - (__clzll(high_bits) + static_cast<int>(high_bits == 0) * __clzll(low_bits));
  }
#else
  if constexpr (cuda::std::is_same_v<T, uint64_t>) {
    return 64 - cuda::std::countl_zero(value);
  } else if constexpr (cuda::std::is_same_v<T, uint32_t>) {
    return 32 - cuda::std::countl_zero(value);
  } else if constexpr (cuda::std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<uint64_t>(value >> 64);
    if (high_bits == 0) {
      return 64 - cuda::std::countl_zero(static_cast<uint64_t>(value));
    } else {
      return 128 - cuda::std::countl_zero(high_bits);
    }
  }
#endif
}

/**
 * @brief Helper struct for getting and setting the components of a floating-point value
 *
 * @tparam FloatingType Type of floating-point value
 */
template <typename FloatingType, CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
struct floating_converter {
  // This struct assumes we're working with IEEE 754 floating-point values.
  // Details on the IEEE-754 floating-point format:
  // Format: https://learn.microsoft.com/en-us/cpp/build/ieee-floating-point-representation
  // Float Visualizer: https://www.h-schmidt.net/FloatConverter/IEEE754.html
  static_assert(cuda::std::numeric_limits<FloatingType>::is_iec559, "Assumes IEEE 754");

  /// Unsigned int type with same size as floating type
  using IntegralType =
    cuda::std::conditional_t<cuda::std::is_same_v<FloatingType, float>, uint32_t, uint64_t>;

  // The high bit is the sign bit (0 for positive, 1 for negative).
  /// How many bits in the floating type
  static constexpr int num_floating_bits = sizeof(FloatingType) * CHAR_BIT;
  /// The index of the sign bit
  static constexpr int sign_bit_index = num_floating_bits - 1;
  /// The mask to select the sign bit
  static constexpr IntegralType sign_mask = (IntegralType(1) << sign_bit_index);

  // The low 23 / 52 bits (for float / double) are the mantissa.
  // The mantissa is normalized. There is an understood 1 bit to the left of the binary point.
  // The value of the mantissa is in the range [1, 2).
  /// # significand bits (includes understood bit)
  static constexpr int num_significand_bits = cuda::std::numeric_limits<FloatingType>::digits;
  /// # stored mantissa bits (-1 for understood bit)
  static constexpr int num_stored_mantissa_bits = num_significand_bits - 1;
  /// The mask for the understood bit
  static constexpr IntegralType understood_bit_mask = (IntegralType(1) << num_stored_mantissa_bits);
  /// The mask to select the mantissa
  static constexpr IntegralType mantissa_mask = understood_bit_mask - 1;

  // And in between are the bits used to store the biased power-of-2 exponent.
  /// # exponents bits (-1 for sign bit)
  static constexpr int num_exponent_bits = num_floating_bits - num_stored_mantissa_bits - 1;
  /// The mask for the exponents, unshifted
  static constexpr IntegralType unshifted_exponent_mask =
    (IntegralType(1) << num_exponent_bits) - 1;
  /// The mask to select the exponents
  static constexpr IntegralType exponent_mask = unshifted_exponent_mask << num_stored_mantissa_bits;

  // To store positive and negative exponents as unsigned values, the stored value for
  // the power-of-2 is exponent + bias. The bias is 127 for floats and 1023 for doubles.
  /// 127 / 1023 for float / double
  static constexpr int exponent_bias = cuda::std::numeric_limits<FloatingType>::max_exponent - 1;

  /**
   * @brief Reinterpret the bits of a floating-point value as an integer
   *
   * @param floating The floating-point value to cast
   * @return An integer with bits identical to the input
   */
  CUDF_HOST_DEVICE inline static IntegralType bit_cast_to_integer(FloatingType floating)
  {
    // Convert floating to integer
    return cuda::std::bit_cast<IntegralType>(floating);
  }

  /**
   * @brief Reinterpret the bits of an integer as floating-point value
   *
   * @param integer The integer to cast
   * @return A floating-point value with bits identical to the input
   */
  CUDF_HOST_DEVICE inline static FloatingType bit_cast_to_floating(IntegralType integer)
  {
    // Convert back to float
    return cuda::std::bit_cast<FloatingType>(integer);
  }

  /**
   * @brief Checks whether the bit-casted floating-point value is +/-0
   *
   * @param integer_rep The bit-casted floating value to check if is +/-0
   * @return True if is a zero, else false
   */
  CUDF_HOST_DEVICE inline static bool is_zero(IntegralType integer_rep)
  {
    // It's a zero if every non-sign bit is zero
    return ((integer_rep & ~sign_mask) == 0);
  }

  /**
   * @brief Extracts the sign bit of a bit-casted floating-point number
   *
   * @param integer_rep The bit-casted floating value to extract the exponent from
   * @return The sign bit
   */
  CUDF_HOST_DEVICE inline static bool get_is_negative(IntegralType integer_rep)
  {
    // Extract the sign bit:
    return static_cast<bool>(sign_mask & integer_rep);
  }

  /**
   * @brief Extracts the significand and exponent of a bit-casted floating-point number,
   * shifted for denormals.
   *
   * @note Zeros/inf/NaN not handled.
   *
   * @param integer_rep The bit-casted floating value to extract the exponent from
   * @return The stored base-2 exponent and significand, shifted for denormals
   */
  CUDF_HOST_DEVICE inline static cuda::std::pair<IntegralType, int> get_significand_and_pow2(
    IntegralType integer_rep)
  {
    // Extract the significand
    auto significand = (integer_rep & mantissa_mask);

    // Extract the exponent bits.
    auto const exponent_bits = integer_rep & exponent_mask;

    // Notes on special values of exponent_bits:
    // bits = exponent_mask is +/-inf or NaN, but those are handled prior to input.
    // bits = 0 is either a denormal (handled below) or a zero (handled earlier by caller).
    int floating_pow2;
    if (exponent_bits == 0) {
      // Denormal values are 2^(1 - exponent_bias) * Sum_i(B_i * 2^-i)
      // Where i is the i-th mantissa bit (counting from the LEFT, starting at 1),
      // and B_i is the value of that bit (0 or 1)
      // So e.g. for the minimum denormal, only the lowest bit is set:
      // FLT_TRUE_MIN = 2^(1 - 127) * 2^-23 = 2^-149
      // DBL_TRUE_MIN = 2^(1 - 1023) * 2^-52 = 2^-1074
      floating_pow2 = 1 - exponent_bias;

      // Line-up denormal to same (understood) bit as normal numbers
      // This is so bit-shifting starts at the same bit index
      auto const lineup_shift = num_significand_bits - count_significant_bits(significand);
      significand <<= lineup_shift;
      floating_pow2 -= lineup_shift;
    } else {
      // Extract the exponent value: shift the bits down and subtract the bias.
      auto const shifted_exponent_bits = exponent_bits >> num_stored_mantissa_bits;
      floating_pow2                    = static_cast<int>(shifted_exponent_bits) - exponent_bias;

      // Set the high bit for the understood 1/2
      significand |= understood_bit_mask;
    }

    // To convert the mantissa to an integer, we effectively applied #-mantissa-bits
    // powers of 2 to convert the fractional value to an integer, so subtract them off here
    int const pow2 = floating_pow2 - num_stored_mantissa_bits;

    return {significand, pow2};
  }

  /**
   * @brief Sets the sign bit of a floating-point number
   *
   * @param floating The floating-point value to set the sign of. Must be positive.
   * @param is_negative The sign bit to set for the floating-point number
   * @return The input floating-point value with the chosen sign
   */
  CUDF_HOST_DEVICE inline static FloatingType set_is_negative(FloatingType floating,
                                                              bool is_negative)
  {
    // Convert floating to integer
    IntegralType integer_rep = bit_cast_to_integer(floating);

    // Set the sign bit. Note that the input floating-point number must be positive (bit = 0).
    integer_rep |= (IntegralType(is_negative) << sign_bit_index);

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }

  /**
   * @brief Adds to the base-2 exponent of a floating-point number
   *
   * @note The caller must guarantee that the input is a positive (> 0) whole number.
   *
   * @param floating The floating value to add to the exponent of. Must be positive.
   * @param pow2 The power-of-2 to add to the floating-point number
   * @return The input floating-point value * 2^pow2
   */
  CUDF_HOST_DEVICE inline static FloatingType add_pow2(FloatingType floating, int pow2)
  {
    // Note that the input floating-point number is positive (& whole), so we don't have to
    // worry about the sign here; the sign will be set later in set_is_negative()

    // Convert floating to integer
    auto integer_rep = bit_cast_to_integer(floating);

    // Extract the currently stored (biased) exponent
    using SignedType   = cuda::std::make_signed_t<IntegralType>;
    auto exponent_bits = integer_rep & exponent_mask;
    auto stored_pow2   = static_cast<SignedType>(exponent_bits >> num_stored_mantissa_bits);

    // Add the additional power-of-2
    stored_pow2 += pow2;

    // Check for exponent over/under-flow.
    if (stored_pow2 <= 0) {
      // Denormal (zero handled prior to input)

      // Early out if bit shift will zero it anyway.
      // Note: We must handle this explicitly, as too-large a bit-shift is UB
      auto const bit_shift = -stored_pow2 + 1;  //+1 due to understood bit set below
      if (bit_shift > num_stored_mantissa_bits) { return 0.0; }

      // Clear the exponent bits (zero means 2^-126/2^-1022 w/ no understood bit)
      integer_rep &= (~exponent_mask);

      // The input floating-point number has an "understood" bit that we need to set
      // prior to bit-shifting. Set the understood bit.
      integer_rep |= understood_bit_mask;

      // Convert to denormal: bit shift off the low bits
      integer_rep >>= bit_shift;
    } else if (stored_pow2 >= static_cast<SignedType>(unshifted_exponent_mask)) {
      // Overflow: Set infinity
      return cuda::std::numeric_limits<FloatingType>::infinity();
    } else {
      // Normal number: Clear existing exponent bits and set new ones
      exponent_bits = static_cast<IntegralType>(stored_pow2) << num_stored_mantissa_bits;
      integer_rep &= (~exponent_mask);
      integer_rep |= exponent_bits;
    }

    // Convert back to float
    return bit_cast_to_floating(integer_rep);
  }
};

/**
 * @brief Recursively calculate a signed large power of 10 (>= 10^19) that can only be stored in an
 * 128bit integer
 *
 * @note Intended to be run at compile time.
 *
 * @tparam Pow10 The power of 10 to calculate
 * @return Returns 10^Pow10
 */
template <int Pow10>
constexpr __uint128_t large_power_of_10()
{
  // Stop at 10^19 to speed up compilation; literals can be used for smaller powers of 10.
  static_assert(Pow10 >= 19);
  if constexpr (Pow10 == 19)
    return __uint128_t(10000000000000000000ULL);
  else
    return large_power_of_10<Pow10 - 1>() * __uint128_t(10);
}

/**
 * @brief Divide by a power of 10 that fits within a 32bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param pow10 The power-of-10 of the denominator, from 0 to 9 inclusive.
 * @return Returns value / 10^pow10
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline T divide_power10_32bit(T value, int pow10)
{
  // Computing division this way is much faster than the alternatives.
  // Division is not implemented in GPU hardware, and the compiler will often implement it as a
  // multiplication of the reciprocal of the denominator, requiring a conversion to floating point.
  // Ths is especially slow for larger divides that have to use the FP64 pipeline, where threads
  // bottleneck.

  // Instead, if the compiler can see exactly what number it is dividing by, it can
  // produce much more optimal assembly, doing bit shifting, multiplies by a constant, etc.
  // For the compiler to see the value though, array lookup (with pow10 as the index)
  // is not sufficient: We have to use a switch statement. Although this introduces a branch,
  // it is still much faster than doing the divide any other way.
  // Perhaps an array can be used in C++23 with the assume attribute?

  // Since we're optimizing division this way, we have to do this for multiplication as well.
  // That's because doing them in different ways (switch, array, runtime-computation, etc.)
  // increases the register pressure on all kernels that use fixed_point types, specifically slowing
  // down some of the PYMOD and join benchmarks.

  // This is split up into separate functions for 32-, 64-, and 128-bit denominators.
  // That way we limit the templated, inlined code generation to the exponents that are
  // capable of being represented. Combining them together into a single function again
  // introduces too much pressure on the kernels that use this code, slowing down their benchmarks.
  // It also dramatically slows down the compile time.

  switch (pow10) {
    case 0: return value;
    case 1: return value / 10U;
    case 2: return value / 100U;
    case 3: return value / 1000U;
    case 4: return value / 10000U;
    case 5: return value / 100000U;
    case 6: return value / 1000000U;
    case 7: return value / 10000000U;
    case 8: return value / 100000000U;
    case 9: return value / 1000000000U;
    default: return 0;
  }
}

/**
 * @brief Divide by a power of 10 that fits within a 64bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param pow10 The power-of-10 of the denominator, from 0 to 19 inclusive.
 * @return Returns value / 10^pow10
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline T divide_power10_64bit(T value, int pow10)
{
  return value / ipow<uint64_t, Radix::BASE_10>(pow10);
}

/**
 * @brief Divide by a power of 10 that fits within a 128bit integer.
 *
 * @tparam T Type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param pow10 The power-of-10 of the denominator, from 0 to 38 inclusive.
 * @return Returns value / 10^pow10.
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline constexpr T divide_power10_128bit(T value, int pow10)
{
  return value / ipow<__uint128_t, Radix::BASE_10>(pow10);
}

/**
 * @brief Multiply by a power of 10 that fits within a 32bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param pow10 The power-of-10 of the multiplier, from 0 to 9 inclusive.
 * @return Returns value * 10^pow10
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_32bit(T value, int pow10)
{
  // See comments in divide_power10_32bit() for discussion.
  switch (pow10) {
    case 0: return value;
    case 1: return value * 10U;
    case 2: return value * 100U;
    case 3: return value * 1000U;
    case 4: return value * 10000U;
    case 5: return value * 100000U;
    case 6: return value * 1000000U;
    case 7: return value * 10000000U;
    case 8: return value * 100000000U;
    case 9: return value * 1000000000U;
    default: return 0;
  }
}

/**
 * @brief Multiply by a power of 10 that fits within a 64bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param pow10 The power-of-10 of the multiplier, from 0 to 19 inclusive.
 * @return Returns value * 10^pow10
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_64bit(T value, int pow10)
{
  return value * ipow<uint64_t, Radix::BASE_10>(pow10);
}

/**
 * @brief Multiply by a power of 10 that fits within a 128bit integer.
 *
 * @tparam T Type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param pow10 The power-of-10 of the multiplier, from 0 to 38 inclusive.
 * @return Returns value * 10^pow10.
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline constexpr T multiply_power10_128bit(T value, int pow10)
{
  return value * ipow<__uint128_t, Radix::BASE_10>(pow10);
}

/**
 * @brief Multiply an integer by a power of 10.
 *
 * @note Use this function if you have no a-priori knowledge of what pow10 might be.
 * If you do, prefer calling the bit-size-specific versions
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam T Integral type of value to be multiplied.
 * @param value The number to be multiplied.
 * @param pow10 The power-of-10 of the multiplier.
 * @return Returns value * 10^pow10
 */
template <typename Rep, typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline constexpr T multiply_power10(T value, int pow10)
{
  // Use this function if you have no knowledge of what pow10 might be
  // If you do, prefer calling the bit-size-specific versions
  if constexpr (sizeof(Rep) <= 4) {
    return multiply_power10_32bit(value, pow10);
  } else if constexpr (sizeof(Rep) <= 8) {
    return multiply_power10_64bit(value, pow10);
  } else {
    return multiply_power10_128bit(value, pow10);
  }
}

/**
 * @brief Divide an integer by a power of 10.
 *
 * @note Use this function if you have no a-priori knowledge of what pow10 might be.
 * If you do, prefer calling the bit-size-specific versions
 *
 * @tparam Rep Representation type needed for integer exponentiation
 * @tparam T Integral type of value to be divided-from.
 * @param value The number to be divided-from.
 * @param pow10 The power-of-10 of the denominator.
 * @return Returns value / 10^pow10
 */
template <typename Rep, typename T, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<T>)>
CUDF_HOST_DEVICE inline constexpr T divide_power10(T value, int pow10)
{
  // Use this function if you have no knowledge of what pow10 might be
  // If you do, prefer calling the bit-size-specific versions
  if constexpr (sizeof(Rep) <= 4) {
    return divide_power10_32bit(value, pow10);
  } else if constexpr (sizeof(Rep) <= 8) {
    return divide_power10_64bit(value, pow10);
  } else {
    return divide_power10_128bit(value, pow10);
  }
}

/**
 * @brief Perform a bit-shift left, guarding against undefined behavior
 *
 * @tparam IntegerType Type of input unsigned integer value
 * @param value The integer whose bits are being shifted
 * @param bit_shift The number of bits to shift left
 * @return The bit-shifted integer, except max value if UB would occur
 */
template <typename IntegerType, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<IntegerType>)>
CUDF_HOST_DEVICE inline IntegerType guarded_left_shift(IntegerType value, int bit_shift)
{
  // Bit shifts larger than this are undefined behavior
  constexpr int max_safe_bit_shift = cuda::std::numeric_limits<IntegerType>::digits - 1;
  return (bit_shift <= max_safe_bit_shift) ? value << bit_shift
                                           : cuda::std::numeric_limits<IntegerType>::max();
}

/**
 * @brief Perform a bit-shift right, guarding against undefined behavior
 *
 * @tparam IntegerType Type of input unsigned integer value
 * @param value The integer whose bits are being shifted
 * @param bit_shift The number of bits to shift right
 * @return The bit-shifted integer, which is zero on underflow
 */
template <typename IntegerType, CUDF_ENABLE_IF(cuda::std::is_unsigned_v<IntegerType>)>
CUDF_HOST_DEVICE inline IntegerType guarded_right_shift(IntegerType value, int bit_shift)
{
  // Bit shifts larger than this are undefined behavior
  constexpr int max_safe_bit_shift = cuda::std::numeric_limits<IntegerType>::digits - 1;
  return (bit_shift <= max_safe_bit_shift) ? value >> bit_shift : 0;
}

/**
 * @brief Helper struct with common constants needed by the floating <--> decimal conversions
 */
template <typename FloatingType>
struct shifting_constants {
  /// Whether the type is double
  static constexpr bool is_double = cuda::std::is_same_v<FloatingType, double>;

  /// Integer type that can hold the value of the significand
  using IntegerRep = cuda::std::conditional_t<is_double, uint64_t, uint32_t>;

  /// Num bits needed to hold the significand
  static constexpr auto num_significand_bits = cuda::std::numeric_limits<FloatingType>::digits;

  /// Shift data back and forth in space of a type with 2x the starting bits, to give us enough room
  using ShiftingRep = cuda::std::conditional_t<is_double, __uint128_t, uint64_t>;

  // The significand of a float / double is 24 / 53 bits
  // However, to uniquely represent each double / float as different #'s in decimal
  // you need 17 / 9 digits (from std::numeric_limits<T>::max_digits10)
  // To represent 10^17 / 10^9, you need 57 / 30 bits
  // So we need to keep track of at least this # of bits during shifting to ensure no info is lost

  // We will be alternately shifting our data back and forth by powers of 2 and 10 to convert
  // between floating and decimal (see shifting functions for details).

  // To iteratively shift back and forth, our 2's (bit-) and 10's (divide-/multiply-) shifts must
  // be of nearly the same magnitude, or else we'll over-/under-flow our shifting integer

  // 2^10 is approximately 10^3, so the largest shifts will have a 10/3 ratio
  // The difference between 2^10 and 10^3 is 1024/1000: 2.4%
  // So every time we shift by 10 bits and 3 decimal places, the 2s shift is an extra 2.4%

  // This 2.4% error compounds each time we do an iteration.
  // The min (normal) float is 2^-126.
  // Min denormal: 2^-126 * 2^-23 (mantissa bits): 2^-149 = ~1.4E-45
  // With our 10/3 shifting ratio, 149 (bit-shifts) * (3 / 10) = 44.7 (10s-shifts)
  // 10^(-44.7) = 2E-45, which is off by ~1.4x from 1.4E-45

  // Similarly, the min (normal) double is 2^-1022.
  // Min denormal: 2^-1022 * 2^-52 (mantissa bits): 2^-1074 = 4.94E-324
  // With our 10/3 shifting ratio, 1074 (bit-shifts) * (3 / 10) = 322.2 (10s-shifts)
  // 10^(-322.2) = 6.4E-323, which is off by ~13.2x from 4.94E-324

  // To account for this compounding error, we can either complicate our loop code (slow),
  // or use extra bits (in the direction we're shifting the 2s!) to compensate:
  // 4 extra bits for doubles (2^4 = 16 > 13.2x error), 1 extra for floats (2 > 1.4x error)
  /// # buffer bits to account for shifting error
  static constexpr int num_2s_shift_buffer_bits = is_double ? 4 : 1;

  // How much room do we have for shifting?
  // Float: 64-bit ShiftingRep - 31 (rep + buffer) = 33 bits. 2^33 = 8.6E9
  // Double: 128-bit ShiftingRep - 61 (rep + buffer) = 67 bits. 2^67 = 1.5E20
  // Thus for double / float we can shift up to 20 / 9 decimal places at once

  // But, we need to stick to our 10-bits / 3-decimals shift ratio to not over/under-flow.
  // To simplify our loop code, we'll keep to this ratio by instead shifting a max of
  // 18 / 9 decimal places, for double / float (60 / 30 bits)
  /// Max at-once decimal place shift
  static constexpr int max_digits_shift = is_double ? 18 : 9;
  /// Max at-once bit shift
  static constexpr int max_bits_shift = max_digits_shift * 10 / 3;

  // Pre-calculate 10^max_digits_shift. Note that 10^18 / 10^9 fits within IntegerRep
  /// 10^max_digits_shift
  static constexpr auto max_digits_shift_pow =
    multiply_power10<IntegerRep>(IntegerRep(1), max_digits_shift);
};

/**
 * @brief Add half a bit to integer rep of floating point if conversion causes truncation
 *
 * @note This fixes problems like 1.2 (value = 1.1999...) at scale -1 -> 11
 *
 * @tparam FloatingType Type of integer holding the floating-point significand
 * @param floating The floating-point number to convert
 * @param integer_rep The integer representation of the floating-point significand
 * @param pow2 The power of 2 that needs to be applied to the significand
 * @param pow10 The power of 10 that needs to be applied to the significand
 * @return integer_rep, shifted 1 and ++'d if the conversion to decimal causes truncation
 */
template <typename FloatingType, CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE cuda::std::pair<typename floating_converter<FloatingType>::IntegralType, int>
add_half_if_truncates(FloatingType floating,
                      typename floating_converter<FloatingType>::IntegralType integer_rep,
                      int pow2,
                      int pow10)
{
  // The user-supplied scale may truncate information, so we need to talk about rounding.
  // We have chosen not to round, so we want 1.23456f with scale -4 to be decimal 12345

  // But if we don't round at all, 1.2 (double) with scale -1 is 11 instead of 12!
  // Why? Because 1.2 (double) is actually stored as 1.1999999... which we truncate to 1.1
  // While correct (given our choice to truncate), this is surprising and undesirable.
  // This problem happens because 1.2 is not perfectly representable in floating point,
  // and the value 1.199999... happened to be closer to 1.2 than the next value (1.2000...1...)

  // If the scale truncates information (we didn't choose to keep exactly 1.1999...), how
  // do we make sure we store 1.2?  We'll add half an ulp! (unit in the last place)
  // Then 1.1999... becomes 1.2000...1... which truncates to 1.2.
  // And if it had been 1.2000...1..., adding half an ulp still truncates to 1.2

  // Why 1/2 an ulp? Because that's all that is needed. The reason we have this problem in the
  // first place is because the compiler rounded (e.g.) 1.2 to the nearest floating point number.
  // The distance of this rounding is at most 1/2 ulp, otherwise we'd have rounded the other way.

  // How do we add 1/2 an ulp? Just shift the bits left (updating pow2) and add 1.
  // We'll always shift up so every input to the conversion algorithm is aligned the same way.

  // If we add a full ulp we run into issues where we add too much and get the wrong result.
  // This is because (e.g.) 2^23 = 8.4E6 which is not quite 7 digits of precision.
  // So if we want 7 digits, that may "barely" truncate information; adding a 1 ulp is overkill.

  // So when does the user-supplied scale truncate info?
  // For powers > 0: When the 10s (scale) shift is larger than the corresponding bit-shift.
  // For powers < 0: When the 10s shift is less than the corresponding bit-shift.

  // Corresponding bit-shift:
  // 2^10 is approximately 10^3, but this is off by 1.024%
  // 1.024^30 is 2.03704, so this is high by one bit for every 30*3 = 90 powers of 10
  // So 10^N = 2^(10*N/3 - N/90) = 2^(299*N/90)
  // Do comparison without dividing, which loses information:
  // Note: if shift is "equal," still truncates if pow2 < 0 (shifting UP by 2s, 2^10 > 10^3)
  int const pow2_term  = 90 * pow2;
  int const pow10_term = 299 * pow10;
  bool const conversion_truncates =
    (pow10_term > pow2_term) || ((pow2_term == pow10_term) && (pow2 < 0));

  // However, don't add a half-bit if the input is a whole number!
  // This is only for errors introduced by rounding decimal fractions!
  bool const is_whole_number = (cuda::std::floor(floating) == floating);
  bool const add_half_bit    = conversion_truncates && !is_whole_number;

  // Add half a bit on truncation (shift to make room and update pow2)
  integer_rep <<= 1;
  --pow2;
  integer_rep += static_cast<decltype(integer_rep)>(add_half_bit);

  return {integer_rep, pow2};
}

/**
 * @brief Perform base-2 -> base-10 fixed-point conversion for pow10 > 0
 *
 * @tparam Rep The type of the storage for the decimal value
 * @tparam FloatingType The type of the original floating-point value we are converting from
 * @param base2_value The base-2 fixed-point value we are converting from
 * @param pow2 The number of powers of 2 to apply to convert from base-2
 * @param pow10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Magnitude of the converted-to decimal integer
 */
template <typename Rep,
          typename FloatingType,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE inline cuda::std::make_unsigned_t<Rep> shift_to_decimal_pospow(
  typename shifting_constants<FloatingType>::IntegerRep const base2_value, int pow2, int pow10)
{
  // To convert to decimal, we need to apply the input powers of 2 and 10
  // The result will be (integer) base2_value * (2^pow2) / (10^pow10)
  // Output type is ShiftingRep

  // Here pow10 > 0 and pow2 > 0, so we need to shift left by 2s and divide by 10s.
  // We'll iterate back and forth between them, shifting up by 2s
  // and down by 10s until all of the powers have been applied.

  // However the input base2_value type has virtually no spare room to shift our data
  // without over- or under-flowing and losing precision.
  // So we'll cast up to ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;
  auto shifting_rep = static_cast<ShiftingRep>(base2_value);

  // We want to start with our significand bits at the top of the shifting range,
  // so that we don't lose information we need on intermediary right-shifts.
  // Note that since we're shifting 2s up, we need num_2s_shift_buffer_bits space on the high side,
  // For all numbers this bit shift is a fixed distance, due to the understood 2^0 bit.
  // Note that shift_from is +1 due to shift in add_half_if_truncates()
  static constexpr int shift_up_to = sizeof(ShiftingRep) * 8 - Constants::num_2s_shift_buffer_bits;
  static constexpr int shift_from  = Constants::num_significand_bits + 1;
  static constexpr int max_init_shift = shift_up_to - shift_from;

  // If our total bit shift is less than this, we don't need to iterate
  using UnsignedRep = cuda::std::make_unsigned_t<Rep>;
  if (pow2 <= max_init_shift) {
    // Shift bits left, divide by 10s to apply the scale factor, and we're done.
    shifting_rep = divide_power10<ShiftingRep>(shifting_rep << pow2, pow10);
    // NOTE: Cast can overflow!
    return static_cast<UnsignedRep>(shifting_rep);
  }

  // We need to iterate. Do the combined initial shift
  shifting_rep <<= max_init_shift;
  pow2 -= max_init_shift;

  // Iterate, dividing by 10s and shifting up by 2s until we're almost done
  while (pow10 > Constants::max_digits_shift) {
    // More decimal places to shift than we have room: Divide the max number of 10s
    shifting_rep /= Constants::max_digits_shift_pow;
    pow10 -= Constants::max_digits_shift;

    // If our remaining bit shift is less than the max, we're finished iterating
    if (pow2 <= Constants::max_bits_shift) {
      // Shift bits left, divide by 10s to apply the scale factor, and we're done.
      shifting_rep = divide_power10<ShiftingRep>(shifting_rep << pow2, pow10);

      // NOTE: Cast can overflow!
      return static_cast<UnsignedRep>(shifting_rep);
    }

    // Shift the max number of bits left again
    shifting_rep <<= Constants::max_bits_shift;
    pow2 -= Constants::max_bits_shift;
  }

  // Last 10s-shift: Divide all remaining decimal places, shift all remaining bits, then bail
  // Note: This divide result may not fit in the low half of the bit range
  // But the divisor is less than the max-shift, and thus fits within 64 / 32 bits
  if constexpr (Constants::is_double) {
    shifting_rep = divide_power10_64bit(shifting_rep, pow10);
  } else {
    shifting_rep = divide_power10_32bit(shifting_rep, pow10);
  }

  // Final bit shift: Shift may be large, guard against UB
  // NOTE: This can overflow (both cast and shift)!
  return guarded_left_shift(static_cast<UnsignedRep>(shifting_rep), pow2);
}

/**
 * @brief Perform base-2 -> base-10 fixed-point conversion for pow10 < 0
 *
 * @tparam Rep The type of the storage for the decimal value
 * @tparam FloatingType The type of the original floating-point value we are converting from
 * @param base2_value The base-2 fixed-point value we are converting from
 * @param pow2 The number of powers of 2 to apply to convert from base-2
 * @param pow10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Magnitude of the converted-to decimal integer
 */
template <typename Rep,
          typename FloatingType,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE inline cuda::std::make_unsigned_t<Rep> shift_to_decimal_negpow(
  typename shifting_constants<FloatingType>::IntegerRep base2_value, int pow2, int pow10)
{
  // This is similar to shift_to_decimal_pospow(), except pow10 < 0 & pow2 < 0
  // See comments in that function for details.
  // Instead here we need to multiply by 10s and shift right by 2s

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;
  auto shifting_rep = static_cast<ShiftingRep>(base2_value);

  // Convert to using positive values so we don't have keep negating
  int pow10_mag = -pow10;
  int pow2_mag  = -pow2;

  // For performing final 10s-shift
  using UnsignedRep        = cuda::std::make_unsigned_t<Rep>;
  auto final_shifts_low10s = [&]() {
    // Last 10s-shift: multiply all remaining decimal places, shift all remaining bits, then bail
    // The multiplier is less than the max-shift, and thus fits within 64 / 32 bits
    if constexpr (Constants::is_double) {
      shifting_rep = multiply_power10_64bit(shifting_rep, pow10_mag);
    } else {
      shifting_rep = multiply_power10_32bit(shifting_rep, pow10_mag);
    }

    // Final bit shifting: Shift may be large, guard against UB
    return static_cast<UnsignedRep>(guarded_right_shift(shifting_rep, pow2_mag));
  };

  // If our total decimal shift is less than the max, we don't need to iterate
  if (pow10_mag <= Constants::max_digits_shift) { return final_shifts_low10s(); }

  // We want to start by lining up our bits to the top of the shifting range,
  // except our first operation is a multiply, so not quite that far
  // We are bit-shifting down, so we need extra bits on the low-side, which this has.
  // Note that shift_from is +1 due to shift in add_half_if_truncates()
  static constexpr int shift_up_to        = sizeof(ShiftingRep) * 8 - Constants::max_bits_shift;
  static constexpr int shift_from         = Constants::num_significand_bits + 1;
  static constexpr int num_init_bit_shift = shift_up_to - shift_from;

  // Perform initial shift
  shifting_rep <<= num_init_bit_shift;
  pow2_mag += num_init_bit_shift;

  // Iterate, multiplying by 10s and shifting down by 2s until we're almost done
  do {
    // More decimal places to shift than we have room: Multiply the max number of 10s
    shifting_rep *= Constants::max_digits_shift_pow;
    pow10_mag -= Constants::max_digits_shift;

    // If our remaining bit shift is less than the max, we're finished iterating
    if (pow2_mag <= Constants::max_bits_shift) {
      // Last bit-shift: Shift all remaining bits, apply the remaining scale, then bail
      shifting_rep >>= pow2_mag;

      // We need to convert to the output rep for the final scale-factor multiply, because if (e.g.)
      // float -> dec128 and some large pow10_mag, it might overflow the 64bit shifting rep.
      // It's not needed for pow10 > 0 because we're dividing by 10s there instead of multiplying.
      // NOTE: This can overflow! (Both multiply and cast)
      return multiply_power10<UnsignedRep>(static_cast<UnsignedRep>(shifting_rep), pow10_mag);
    }

    // More bits to shift than we have room: Shift the max number of 2s
    shifting_rep >>= Constants::max_bits_shift;
    pow2_mag -= Constants::max_bits_shift;
  } while (pow10_mag > Constants::max_digits_shift);

  // Do our final shifts
  return final_shifts_low10s();
}

/**
 * @brief Perform base-2 -> base-10 fixed-point conversion
 *
 * @tparam Rep The type of integer we are converting to, to store the decimal value
 * @tparam FloatingType The type of floating-point object we are converting from
 * @param base2_value The base-2 fixed-point value we are converting from
 * @param pow2 The number of powers of 2 to apply to convert from base-2
 * @param pow10 The number of powers of 10 to apply to reach the desired scale factor
 * @return Integer representation of the floating-point value, given the desired scale
 */
template <typename Rep,
          typename FloatingType,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE inline cuda::std::make_unsigned_t<Rep> convert_floating_to_integral_shifting(
  typename floating_converter<FloatingType>::IntegralType base2_value, int pow10, int pow2)
{
  // Apply the powers of 2 and 10 to convert to decimal.
  // The result will be base2_value * (2^pow2) / (10^pow10)

  // Note that while this code is branchy, the decimal scale factor is part of the
  // column type itself, so every thread will take the same branches on pow10.
  // Also data within a column tends to be similar, so they will often take the
  // same branches on pow2 as well.

  // NOTE: some returns here can overflow (e.g. ShiftingRep -> UnsignedRep)
  using UnsignedRep = cuda::std::make_unsigned_t<Rep>;
  if (pow10 == 0) {
    // NOTE: Left Bit-shift can overflow! As can cast! (e.g. double -> decimal32)
    // Bit shifts may be large, guard against UB
    if (pow2 >= 0) {
      return guarded_left_shift(static_cast<UnsignedRep>(base2_value), pow2);
    } else {
      return static_cast<UnsignedRep>(guarded_right_shift(base2_value, -pow2));
    }
  } else if (pow10 > 0) {
    if (pow2 <= 0) {
      // Power-2/10 shifts both downward: order doesn't matter, apply and bail.
      // Guard against shift being undefined behavior
      auto const shifted = guarded_right_shift(base2_value, -pow2);
      return static_cast<UnsignedRep>(divide_power10<decltype(shifted)>(shifted, pow10));
    }
    return shift_to_decimal_pospow<Rep, FloatingType>(base2_value, pow2, pow10);
  } else {  // pow10 < 0
    if (pow2 >= 0) {
      // Power-2/10 shifts both upward: order doesn't matter, apply and bail.
      // NOTE: Either shift, multiply, or cast (e.g. double -> decimal32) can overflow!
      auto const shifted = guarded_left_shift(static_cast<UnsignedRep>(base2_value), pow2);
      return multiply_power10<UnsignedRep>(shifted, -pow10);
    }
    return shift_to_decimal_negpow<Rep, FloatingType>(base2_value, pow2, pow10);
  }
}

/**
 * @brief Perform floating-point -> integer decimal conversion
 *
 * @tparam Rep The type of integer we are converting to, to store the decimal value
 * @tparam FloatingType The type of floating-point object we are converting from
 * @param floating The floating point value to convert
 * @param scale The desired base-10 scale factor: decimal value = returned value * 10^scale
 * @return Integer representation of the floating-point value, given the desired scale
 */
template <typename Rep,
          typename FloatingType,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE inline Rep convert_floating_to_integral(FloatingType const& floating,
                                                         scale_type const& scale)
{
  // Extract components of the floating point number
  using converter        = floating_converter<FloatingType>;
  auto const integer_rep = converter::bit_cast_to_integer(floating);
  if (converter::is_zero(integer_rep)) { return 0; }

  // Note that the significand here is an unsigned integer with sizeof(FloatingType)
  auto const is_negative                  = converter::get_is_negative(integer_rep);
  auto const [significand, floating_pow2] = converter::get_significand_and_pow2(integer_rep);

  // Add half a bit if truncating to yield expected value, see function for discussion.
  auto const pow10 = static_cast<int>(scale);
  auto const [base2_value, pow2] =
    add_half_if_truncates(floating, significand, floating_pow2, pow10);

  // Apply the powers of 2 and 10 to convert to decimal.
  auto const magnitude =
    convert_floating_to_integral_shifting<Rep, FloatingType>(base2_value, pow10, pow2);

  // Reapply the sign and return
  // NOTE: Cast can overflow!
  auto const signed_magnitude = static_cast<Rep>(magnitude);
  return is_negative ? -signed_magnitude : signed_magnitude;
}

/**
 * @brief Perform base-10 -> base-2 fixed-point conversion for pow10 > 0
 *
 * @tparam DecimalRep The decimal integer type we are converting from
 * @tparam FloatingType The type of floating point object we are converting to
 * @param decimal_rep The decimal integer to convert
 * @param pow10 The number of powers of 10 to apply to undo the scale factor
 * @return A pair of the base-2 value and the remaining powers of 2 to be applied
 */
template <typename FloatingType,
          typename DecimalRep,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE inline auto shift_to_binary_pospow(DecimalRep decimal_rep, int pow10)
{
  // This is the reverse of shift_to_decimal_pospow(), see that for more details.

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;

  // We want to start by lining up our bits to the top of the shifting range,
  // except our first operation is a multiply, so not quite that far
  // We are bit-shifting down, so we need extra bits on the low-side, which this has.
  static constexpr int shift_up_to = sizeof(ShiftingRep) * 8 - Constants::max_bits_shift;
  int const shift_from             = count_significant_bits(decimal_rep);
  int const num_init_bit_shift     = shift_up_to - shift_from;
  int pow2                         = -num_init_bit_shift;

  // Perform the initial bit shift
  ShiftingRep shifting_rep;
  if constexpr (sizeof(ShiftingRep) < sizeof(DecimalRep)) {
    // Shift within DecimalRep before dropping to the smaller ShiftingRep
    decimal_rep  = (pow2 >= 0) ? (decimal_rep >> pow2) : (decimal_rep << -pow2);
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
  } else {
    // Scale up to ShiftingRep before shifting
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
    shifting_rep = (pow2 >= 0) ? (shifting_rep >> pow2) : (shifting_rep << -pow2);
  }

  // Iterate, multiplying by 10s and shifting down by 2s until we're almost done
  while (pow10 > Constants::max_digits_shift) {
    // More decimal places to shift than we have room: Multiply the max number of 10s
    shifting_rep *= Constants::max_digits_shift_pow;
    pow10 -= Constants::max_digits_shift;

    // Then make more room by bit shifting down by the max # of 2s
    shifting_rep >>= Constants::max_bits_shift;
    pow2 += Constants::max_bits_shift;
  }

  // Last 10s-shift: multiply all remaining decimal places
  // The multiplier is less than the max-shift, and thus fits within 64 / 32 bits
  if constexpr (Constants::is_double) {
    shifting_rep = multiply_power10_64bit(shifting_rep, pow10);
  } else {
    shifting_rep = multiply_power10_32bit(shifting_rep, pow10);
  }

  // Our shifting_rep is now the integer mantissa, return it and the powers of 2
  return cuda::std::pair{shifting_rep, pow2};
}

/**
 * @brief Perform base-10 -> base-2 fixed-point conversion for pow10 < 0
 *
 * @tparam DecimalRep The decimal integer type we are converting from
 * @tparam FloatingType The type of floating point object we are converting to
 * @param decimal_rep The decimal integer to convert
 * @param pow10 The number of powers of 10 to apply to undo the scale factor
 * @return A pair of the base-2 value and the remaining powers of 2 to be applied
 */
template <typename FloatingType,
          typename DecimalRep,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE inline auto shift_to_binary_negpow(DecimalRep decimal_rep, int const pow10)
{
  // This is the reverse of shift_to_decimal_negpow(), see that for more details.

  // ShiftingRep: uint64 for float's, __uint128_t for double's
  using Constants   = shifting_constants<FloatingType>;
  using ShiftingRep = typename Constants::ShiftingRep;

  // We want to start with our significand bits at the top of the shifting range,
  // so that we lose minimal information we need on intermediary right-shifts.
  // Note that since we're shifting 2s up, we need num_2s_shift_buffer_bits space on the high side
  static constexpr int shift_up_to = sizeof(ShiftingRep) * 8 - Constants::num_2s_shift_buffer_bits;
  int const shift_from             = count_significant_bits(decimal_rep);
  int const num_init_bit_shift     = shift_up_to - shift_from;
  int pow2                         = -num_init_bit_shift;

  // Perform the initial bit shift
  ShiftingRep shifting_rep;
  if constexpr (sizeof(ShiftingRep) < sizeof(DecimalRep)) {
    // Shift within DecimalRep before dropping to the smaller ShiftingRep
    decimal_rep  = (pow2 >= 0) ? (decimal_rep >> pow2) : (decimal_rep << -pow2);
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
  } else {
    // Scale up to ShiftingRep before shifting
    shifting_rep = static_cast<ShiftingRep>(decimal_rep);
    shifting_rep = (pow2 >= 0) ? (shifting_rep >> pow2) : (shifting_rep << -pow2);
  }

  // Convert to using positive values upfront, simpler than doing later.
  int pow10_mag = -pow10;

  // Iterate, dividing by 10s and shifting up by 2s until we're almost done
  while (pow10_mag > Constants::max_digits_shift) {
    // More decimal places to shift than we have room: Divide the max number of 10s
    shifting_rep /= Constants::max_digits_shift_pow;
    pow10_mag -= Constants::max_digits_shift;

    // Then make more room by bit shifting up by the max # of 2s
    shifting_rep <<= Constants::max_bits_shift;
    pow2 -= Constants::max_bits_shift;
  }

  // Last 10s-shift: Divdie all remaining decimal places.
  // This divide result may not fit in the low half of the bit range
  // But the divisor is less than the max-shift, and thus fits within 64 / 32 bits
  if constexpr (Constants::is_double) {
    shifting_rep = divide_power10_64bit(shifting_rep, pow10_mag);
  } else {
    shifting_rep = divide_power10_32bit(shifting_rep, pow10_mag);
  }

  // Our shifting_rep is now the integer mantissa, return it and the powers of 2
  return cuda::std::pair{shifting_rep, pow2};
}

/**
 * @brief Perform integer decimal -> floating-point conversion
 *
 * @tparam FloatingType The type of floating-point object we are converting to
 * @tparam Rep The decimal integer type we are converting from
 * @param value The decimal integer to convert
 * @param scale The base-10 scale factor for the input integer
 * @return Floating-point representation of the scaled integral value
 */
template <typename FloatingType,
          typename Rep,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatingType>)>
CUDF_HOST_DEVICE inline FloatingType convert_integral_to_floating(Rep const& value,
                                                                  scale_type const& scale)
{
  // Check the sign of the input
  bool const is_negative = (value < 0);

  // Convert to unsigned for bit counting/shifting
  using UnsignedType        = cuda::std::make_unsigned_t<Rep>;
  auto const unsigned_value = [&]() -> UnsignedType {
    // Must guard against minimum value, as we can't just negate it: not representable.
    if (value == cuda::std::numeric_limits<Rep>::min()) { return static_cast<UnsignedType>(value); }

    // No abs function for 128bit types, so have to do it manually.
    if constexpr (cuda::std::is_same_v<Rep, __int128_t>) {
      return static_cast<UnsignedType>(is_negative ? -value : value);
    } else {
      return cuda::std::abs(value);
    }
  }();

  // Shift by powers of 2 and 10 to get our integer mantissa
  auto const [mantissa, pow2] = [&]() {
    auto const pow10 = static_cast<int32_t>(scale);
    if (pow10 >= 0) {
      return shift_to_binary_pospow<FloatingType>(unsigned_value, pow10);
    } else {  // pow10 < 0
      return shift_to_binary_negpow<FloatingType>(unsigned_value, pow10);
    }
  }();

  // Zero has special exponent bits, just handle it here
  if (mantissa == 0) { return FloatingType(0.0f); }

  // Cast our integer mantissa to floating point
  auto const floating = static_cast<FloatingType>(mantissa);  // IEEE-754 rounds to even

  // Apply the sign and the remaining powers of 2
  using converter      = floating_converter<FloatingType>;
  auto const magnitude = converter::add_pow2(floating, pow2);
  return converter::set_is_negative(magnitude, is_negative);
}

}  // namespace detail
}  // namespace numeric
