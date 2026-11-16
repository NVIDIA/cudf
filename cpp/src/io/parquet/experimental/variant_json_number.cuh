/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/limits>
#include <math_constants.h>

#include <cstdint>

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental::detail {

// Longest number Jackson's default `StreamReadConstraints` accepts, which is the limit Spark's
// `parse_json` runs under.
constexpr int max_number_length = 1000;

/// Lexical components of a JSON number that passed strict validation
struct number_parts {
  bool negative;
  char const* int_digits;
  int int_len;
  bool has_fraction;
  char const* frac_digits;
  int frac_len;
  bool has_exponent;
  int exp_len;
  int64_t exponent;  // saturated to +/- 2^40
};

__device__ inline number_parts split_number(char const* s, int len)
{
  number_parts p{};
  int i      = 0;
  p.negative = s[0] == '-';
  if (p.negative) { ++i; }
  p.int_digits = s + i;
  while (i < len && s[i] >= '0' && s[i] <= '9') {
    ++i;
    ++p.int_len;
  }
  if (i < len && s[i] == '.') {
    p.has_fraction = true;
    ++i;
    p.frac_digits = s + i;
    while (i < len && s[i] >= '0' && s[i] <= '9') {
      ++i;
      ++p.frac_len;
    }
  }
  if (i < len && (s[i] == 'e' || s[i] == 'E')) {
    p.has_exponent = true;
    ++i;
    bool exp_negative = false;
    if (s[i] == '+' || s[i] == '-') {
      exp_negative = s[i] == '-';
      ++i;
    }
    int64_t e = 0;
    while (i < len) {
      e = cuda::std::min<int64_t>(e * 10 + (s[i] - '0'), int64_t{1} << 40);
      ++i;
      ++p.exp_len;
    }
    p.exponent = exp_negative ? -e : e;
  }
  return p;
}

enum class scalar_kind : uint8_t {
  INVALID,
  NULL_VALUE,
  TRUE_VALUE,
  FALSE_VALUE,
  INT,
  DECIMAL,
  DOUBLE
};

struct scalar_class {
  scalar_kind kind;
  int width;  // payload bytes for INT and DECIMAL
  int scale;
  __int128_t unscaled;  // integer value for INT, unscaled value for DECIMAL
};

__device__ inline bool matches_literal(char const* s, int len, char const* literal, int literal_len)
{
  if (len != literal_len) { return false; }
  for (int i = 0; i < len; ++i) {
    if (s[i] != literal[i]) { return false; }
  }
  return true;
}

/**
 * @brief Classifies a validated non-string JSON scalar the way Spark's VariantBuilder does.
 *
 * Integers that fit in int64 use the narrowest integer type. Otherwise, numbers written without an
 * exponent whose precision and scale are at most 38 become decimals; everything else is a double.
 */
__device__ inline scalar_class classify_scalar(char const* s, int len)
{
  if (matches_literal(s, len, "null", 4)) { return {scalar_kind::NULL_VALUE, 0, 0, 0}; }
  if (matches_literal(s, len, "true", 4)) { return {scalar_kind::TRUE_VALUE, 0, 0, 0}; }
  if (matches_literal(s, len, "false", 5)) { return {scalar_kind::FALSE_VALUE, 0, 0, 0}; }
  if (len == 0 || !(s[0] == '-' || (s[0] >= '0' && s[0] <= '9'))) {
    return {scalar_kind::INVALID, 0, 0, 0};
  }
  auto const p = split_number(s, len);
  if (!p.has_fraction && !p.has_exponent && p.int_len <= 19) {
    uint64_t magnitude = 0;
    bool overflow      = false;
    for (int i = 0; i < p.int_len; ++i) {
      auto const d = static_cast<uint64_t>(p.int_digits[i] - '0');
      if (magnitude > (cuda::std::numeric_limits<uint64_t>::max() - d) / 10) { overflow = true; }
      magnitude = magnitude * 10 + d;
    }
    auto const limit = static_cast<uint64_t>(cuda::std::numeric_limits<int64_t>::max()) +
                       static_cast<uint64_t>(p.negative);
    if (!overflow && magnitude <= limit) {
      auto const v =
        p.negative ? static_cast<int64_t>(0 - magnitude) : static_cast<int64_t>(magnitude);
      auto const width = (v == static_cast<int8_t>(v))    ? 1
                         : (v == static_cast<int16_t>(v)) ? 2
                         : (v == static_cast<int32_t>(v)) ? 4
                                                          : 8;
      return {scalar_kind::INT, width, 0, v};
    }
  }
  if (!p.has_exponent) {
    // BigDecimal precision counts digits of the unscaled value without leading zeros (minimum 1)
    int total       = p.int_len + p.frac_len;
    int first_digit = total;
    for (int i = 0; i < total; ++i) {
      auto const c = i < p.int_len ? p.int_digits[i] : p.frac_digits[i - p.int_len];
      if (c != '0') {
        first_digit = i;
        break;
      }
    }
    auto const precision = cuda::std::max(1, total - first_digit);
    auto const scale     = p.frac_len;
    if (precision <= 38 && scale <= 38) {
      __int128_t unscaled = 0;
      for (int i = first_digit; i < total; ++i) {
        auto const c = i < p.int_len ? p.int_digits[i] : p.frac_digits[i - p.int_len];
        unscaled     = unscaled * 10 + (c - '0');
      }
      if (p.negative) { unscaled = -unscaled; }
      auto const width = (scale <= 9 && precision <= 9)     ? 4
                         : (scale <= 18 && precision <= 18) ? 8
                                                            : 16;
      return {scalar_kind::DECIMAL, width, scale, unscaled};
    }
  }
  return {scalar_kind::DOUBLE, 8, 0, 0};
}

CUDF_HOST_DEVICE constexpr int64_t scalar_encoded_size(scalar_class const& c)
{
  switch (c.kind) {
    case scalar_kind::INT: return 1 + c.width;
    case scalar_kind::DECIMAL: return 2 + c.width;
    case scalar_kind::DOUBLE: return 9;
    default: return 1;
  }
}

/**
 * @brief Checks Jackson's number-length constraint.
 *
 * Jackson's slow path, taken when a number starts with '0' or runs to the end of the input, counts
 * a missing fraction or exponent as -1 instead of 0.
 */
__device__ inline bool number_length_ok(number_parts const& p, bool at_input_end)
{
  if (!p.has_fraction && !p.has_exponent) { return p.int_len <= max_number_length; }
  auto const slow_path = at_input_end || p.int_digits[0] == '0';
  auto const absent    = slow_path ? -1 : 0;
  auto const length =
    p.int_len + (p.has_fraction ? p.frac_len : absent) + (p.has_exponent ? p.exp_len : absent);
  return length <= max_number_length;
}

/**
 * @brief Fixed-capacity unsigned big integer used to settle double rounding exactly.
 */
struct bigint {
  static constexpr int capacity = 112;
  uint32_t limbs[capacity];
  int size = 0;

  __device__ void set(uint64_t v)
  {
    size = 0;
    while (v != 0) {
      limbs[size++] = static_cast<uint32_t>(v);
      v >>= 32;
    }
  }

  __device__ void mul_add(uint32_t m, uint32_t a)
  {
    uint64_t carry = a;
    for (int i = 0; i < size; ++i) {
      auto const prod = static_cast<uint64_t>(limbs[i]) * m + carry;
      limbs[i]        = static_cast<uint32_t>(prod);
      carry           = prod >> 32;
    }
    if (carry != 0) { limbs[size++] = static_cast<uint32_t>(carry); }
  }

  __device__ void mul_pow5(int e)
  {
    constexpr uint32_t pow5_13 = 1'220'703'125u;
    while (e >= 13) {
      mul_add(pow5_13, 0);
      e -= 13;
    }
    uint32_t m = 1;
    while (e-- > 0) {
      m *= 5;
    }
    if (m != 1) { mul_add(m, 0); }
  }

  [[nodiscard]] __device__ int64_t bit_length() const
  {
    if (size == 0) { return 0; }
    return int64_t{32} * (size - 1) + (32 - __clz(static_cast<int>(limbs[size - 1])));
  }

  __device__ void shift_left(int64_t bits)
  {
    if (size == 0 || bits == 0) { return; }
    auto const words    = static_cast<int>(bits / 32);
    auto const rem      = static_cast<int>(bits % 32);
    int new_size        = size + words + 1;
    limbs[new_size - 1] = 0;
    for (int i = size - 1; i >= 0; --i) {
      auto const v = limbs[i];
      if (rem != 0) {
        limbs[i + words + 1] |= v >> (32 - rem);
        limbs[i + words] = v << rem;
      } else {
        limbs[i + words] = v;
      }
    }
    for (int i = 0; i < words; ++i) {
      limbs[i] = 0;
    }
    while (new_size > 0 && limbs[new_size - 1] == 0) {
      --new_size;
    }
    size = new_size;
  }
};

__device__ inline int compare_magnitude(bigint const& a, bigint const& b)
{
  if (a.size != b.size) { return a.size < b.size ? -1 : 1; }
  for (int i = a.size - 1; i >= 0; --i) {
    if (a.limbs[i] != b.limbs[i]) { return a.limbs[i] < b.limbs[i] ? -1 : 1; }
  }
  return 0;
}

/// Compares `a * 2^a_shift` with `b * 2^b_shift`; both shifts are non-negative
__device__ inline int compare_shifted(bigint& a, int64_t a_shift, bigint& b, int64_t b_shift)
{
  auto const a_bits = a.size == 0 ? 0 : a.bit_length() + a_shift;
  auto const b_bits = b.size == 0 ? 0 : b.bit_length() + b_shift;
  if (a_bits != b_bits) { return a_bits < b_bits ? -1 : 1; }
  auto const common = cuda::std::min(a_shift, b_shift);
  a.shift_left(a_shift - common);
  b.shift_left(b_shift - common);
  return compare_magnitude(a, b);
}

// Significant digits beyond this count cannot change which double a decimal rounds to, as long as
// a sticky flag records whether any of the dropped digits were nonzero.
constexpr int max_significant_digits = 800;

struct decimal_digits {
  number_parts const* parts;
  int first;  // index of the first nonzero digit
  int count;  // number of significant digits kept
  bool sticky;
  int64_t exponent;  // value = digits * 10^exponent

  [[nodiscard]] __device__ char digit(int i) const
  {
    auto const idx = first + i;
    return idx < parts->int_len ? parts->int_digits[idx] : parts->frac_digits[idx - parts->int_len];
  }
};

/// Sign of (digits * 10^exponent) - (m * 2^k)
__device__ inline int compare_decimal_to_binary(decimal_digits const& d, uint64_t m, int64_t k)
{
  bigint lhs;
  bigint rhs;
  lhs.set(0);
  for (int i = 0; i < d.count;) {
    auto const chunk = cuda::std::min(9, d.count - i);
    uint32_t value   = 0;
    uint32_t scale   = 1;
    for (int j = 0; j < chunk; ++j) {
      value = value * 10 + (d.digit(i + j) - '0');
      scale *= 10;
    }
    if (lhs.size == 0) {
      lhs.set(value);
    } else {
      lhs.mul_add(scale, value);
    }
    i += chunk;
  }
  rhs.set(m);
  int64_t lhs_shift = 0;
  int64_t rhs_shift = 0;
  if (d.exponent >= 0) {
    lhs.mul_pow5(static_cast<int>(d.exponent));
    lhs_shift = d.exponent;
  } else {
    rhs.mul_pow5(static_cast<int>(-d.exponent));
    rhs_shift = -d.exponent;
  }
  rhs_shift += k;
  auto const common = cuda::std::min(lhs_shift, rhs_shift);
  auto const c      = compare_shifted(lhs, lhs_shift - common, rhs, rhs_shift - common);
  return (c == 0 && d.sticky) ? 1 : c;
}

__device__ inline double pow10_exact(int e)
{
  constexpr double table[] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
                              1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                              1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
  return table[e];
}

/**
 * @brief Parses a validated JSON number into the nearest double, rounding half to even.
 *
 * Matches Java's `Double.parseDouble`. Small inputs take Clinger's exact fast path; all others
 * start from an estimate and step one ulp at a time, deciding each step with exact big-integer
 * comparisons against the rounding midpoints.
 */
__device__ inline double parse_double(char const* s, int len)
{
  auto const p     = split_number(s, len);
  auto const neg   = p.negative;
  auto const total = p.int_len + p.frac_len;
  auto digit_at    = [&](int i) {
    return i < p.int_len ? p.int_digits[i] : p.frac_digits[i - p.int_len];
  };
  int first = 0;
  while (first < total && digit_at(first) == '0') {
    ++first;
  }
  if (first == total) { return neg ? -0.0 : 0.0; }
  int last = total - 1;
  while (digit_at(last) == '0') {
    --last;
  }
  auto const num_digits = last - first + 1;
  int64_t exponent      = p.exponent - p.frac_len + (total - 1 - last);

  if (num_digits + exponent > 310) { return neg ? -CUDART_INF : CUDART_INF; }
  if (num_digits + exponent < -324) { return neg ? -0.0 : 0.0; }

  uint64_t leading       = 0;
  auto const num_leading = cuda::std::min(num_digits, 19);
  for (int i = 0; i < num_leading; ++i) {
    leading = leading * 10 + (digit_at(first + i) - '0');
  }
  if (num_digits <= 15 && exponent >= -22 && exponent <= 22) {
    auto const w = static_cast<double>(leading);
    auto const r = exponent >= 0 ? w * pow10_exact(static_cast<int>(exponent))
                                 : w / pow10_exact(static_cast<int>(-exponent));
    return neg ? -r : r;
  }

  decimal_digits d{&p, first, num_digits, false, exponent};
  if (num_digits > max_significant_digits) {
    d.count  = max_significant_digits;
    d.sticky = true;
    d.exponent += num_digits - max_significant_digits;
  }

  auto const est_exp = static_cast<int>(exponent + (num_digits - num_leading));
  auto const half    = est_exp / 2;
  double x           = static_cast<double>(leading) * exp10(static_cast<double>(half)) *
             exp10(static_cast<double>(est_exp - half));
  if (!(x < CUDART_INF)) { x = cuda::std::numeric_limits<double>::max(); }

  for (int iter = 0; iter < 64; ++iter) {
    auto const bits = __double_as_longlong(x);
    if (bits == 0x7FF0'0000'0000'0000LL) { break; }
    auto const exp_bits = static_cast<int>((bits >> 52) & 0x7FF);
    auto const frac     = static_cast<uint64_t>(bits) & ((uint64_t{1} << 52) - 1);
    uint64_t m          = exp_bits == 0 ? frac : (frac | (uint64_t{1} << 52));
    int64_t e           = exp_bits == 0 ? -1074 : exp_bits - 1075;
    bool const m_odd    = (m & 1) != 0;
    // Midpoint between x and the next double up: (2m + 1) * 2^(e - 1)
    auto const up = compare_decimal_to_binary(d, 2 * m + 1, e - 1);
    if (up > 0 || (up == 0 && m_odd)) {
      x = __longlong_as_double(bits + 1);
      continue;
    }
    if (m == 0) { break; }
    // Midpoint between x and the next double down; below a power of two the spacing halves
    auto const down = (frac == 0 && exp_bits > 1) ? compare_decimal_to_binary(d, 4 * m - 1, e - 2)
                                                  : compare_decimal_to_binary(d, 2 * m - 1, e - 1);
    if (down < 0 || (down == 0 && m_odd)) {
      x = __longlong_as_double(bits - 1);
      continue;
    }
    break;
  }
  return neg ? -x : x;
}

}  // namespace io::parquet::experimental::detail
}  // namespace CUDF_EXPORT cudf
