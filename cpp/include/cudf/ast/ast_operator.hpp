/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

/**
 * @file
 * @brief Enum defining the supported AST operators.
 */

namespace CUDF_EXPORT cudf {

namespace ast {
/**
 * @addtogroup expressions
 * @{
 */

/**
 * @brief Enum of supported operators.
 */
enum class ast_operator : int32_t {
  // Binary operators
  ADD,        ///< operator +
  SUB,        ///< operator -
  MUL,        ///< operator *
  DIV,        ///< operator / using common type of lhs and rhs
  TRUE_DIV,   ///< operator / after promoting type to floating point
  FLOOR_DIV,  ///< operator / after promoting to the common type of lhs and rhs (integral or
              ///< floating point), and then flooring the result
  MOD,        ///< operator %
  PYMOD,      ///< operator % using Python's sign rules for negatives
  POW,        ///< lhs ^ rhs
  EQUAL,  ///< operator ==: returns NULL when either operand is NULL, otherwise returns whether they
          ///< are equal
  NULL_EQUAL,  ///< null-safe equality (result is never null): returns true if both operands are
               ///< null, false if one is null, otherwise returns whether they are equal
  NOT_EQUAL,   ///< operator !=: returns NULL when either operand is NULL, otherwise returns whether
               ///< they are unequal
  LESS,        ///< operator <
  GREATER,     ///< operator >
  LESS_EQUAL,  ///< operator <=
  GREATER_EQUAL,     ///< operator >=
  BITWISE_AND,       ///< operator &
  BITWISE_OR,        ///< operator |
  BITWISE_XOR,       ///< operator ^
  LOGICAL_AND,       ///< operator &&: returns NULL when either operand is NULL, otherwise returns
                     ///< true only if both operands are true
  NULL_LOGICAL_AND,  ///< three-valued (Kleene) &&: if any operand is false, returns false; if both
                     ///< operands are true, returns true; otherwise returns null
  LOGICAL_OR,  ///< operator ||: returns NULL when either operand is NULL, otherwise returns true
               ///< only if either or both operands are true
  NULL_LOGICAL_OR,  ///< three-valued (Kleene) ||: if any operand is true, returns true; if both
                    ///< operands are false, returns false; otherwise returns null
  // Unary operators
  IDENTITY,            ///< Identity function
  IS_NULL,             ///< Check if operand is null
  SIN,                 ///< Trigonometric sine
  COS,                 ///< Trigonometric cosine
  TAN,                 ///< Trigonometric tangent
  ARCSIN,              ///< Trigonometric sine inverse
  ARCCOS,              ///< Trigonometric cosine inverse
  ARCTAN,              ///< Trigonometric tangent inverse
  SINH,                ///< Hyperbolic sine
  COSH,                ///< Hyperbolic cosine
  TANH,                ///< Hyperbolic tangent
  ARCSINH,             ///< Hyperbolic sine inverse
  ARCCOSH,             ///< Hyperbolic cosine inverse
  ARCTANH,             ///< Hyperbolic tangent inverse
  EXP,                 ///< Exponential (base e, Euler number)
  LOG,                 ///< Natural Logarithm (base e)
  SQRT,                ///< Square-root (x^0.5)
  CBRT,                ///< Cube-root (x^(1.0/3))
  CEIL,                ///< Smallest integer value not less than arg
  FLOOR,               ///< largest integer value not greater than arg
  ABS,                 ///< Absolute value
  RINT,                ///< Rounds the floating-point argument arg to an integer value
  BIT_INVERT,          ///< Bitwise Not (~)
  NOT,                 ///< Logical Not (!)
  CAST_TO_INT64,       ///< Cast value to int64_t
  CAST_TO_UINT64,      ///< Cast value to uint64_t
  CAST_TO_FLOAT64,     ///< Cast value to double
  CAST_TO_BOOL8,       ///< Cast value to bool
  CAST_TO_INT8,        ///< Cast value to int8_t
  CAST_TO_INT16,       ///< Cast value to int16_t
  CAST_TO_INT32,       ///< Cast value to int32_t
  CAST_TO_UINT8,       ///< Cast value to uint8_t
  CAST_TO_UINT16,      ///< Cast value to uint16_t
  CAST_TO_UINT32,      ///< Cast value to uint32_t
  CAST_TO_FLOAT32,     ///< Cast value to float
  CAST_TO_DECIMAL32,   ///< Cast a decimal value to decimal32, preserving its scale
  CAST_TO_DECIMAL64,   ///< Cast a decimal value to decimal64, preserving its scale
  CAST_TO_DECIMAL128,  ///< Cast a decimal value to decimal128, preserving its scale
  RESCALE              ///< Rescale a fixed-point value to operation target scale
};

/** @} */  // end of group
}  // namespace ast

}  // namespace CUDF_EXPORT cudf
