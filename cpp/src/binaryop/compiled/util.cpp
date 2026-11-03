/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operation.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <optional>

namespace cudf::binops::compiled {

namespace {

struct common_type_functor {
  template <typename TypeLhs, typename TypeRhs>
  std::optional<data_type> operator()() const
  {
    if constexpr (binary_op_has_common_type_v<TypeLhs, TypeRhs>) {
      using TypeCommon = binary_op_common_type_t<TypeLhs, TypeRhs>;
      return data_type{type_to_id<TypeCommon>()};
    }

    // A compiler bug may cause a compilation error when using empty
    // initializer list to construct an std::optional object containing no
    // `data_type` value. Therefore, we explicitly return `std::nullopt`
    // instead.
    return std::nullopt;
  }
};

struct has_mutable_element_accessor_functor {
  template <typename T>
  bool operator()() const
  {
    return mutable_column_device_view::has_element_accessor<T>();
  }
};

bool has_mutable_element_accessor(data_type t)
{
  return type_dispatcher(t, has_mutable_element_accessor_functor{});
}

template <typename InputType>
struct is_constructible_functor {
  template <typename TargetType>
  bool operator()() const
  {
    return std::is_constructible_v<TargetType, InputType>;
  }
};

template <typename InputType>
bool is_constructible(data_type target_type)
{
  return type_dispatcher(target_type, is_constructible_functor<InputType>{});
}

/**
 * @brief Functor that return true if BinaryOperator supports given input and output types.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <typename BinaryOperator>
struct is_binary_operation_supported {
  // For types where Out type is fixed. (e.g. comparison types)
  template <typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()() const
  {
    if constexpr (column_device_view::has_element_accessor<TypeLhs>() and
                  column_device_view::has_element_accessor<TypeRhs>()) {
      if constexpr (binary_op_has_common_type_v<TypeLhs, TypeRhs>) {
        using common_t = binary_op_common_type_t<TypeLhs, TypeRhs>;
        return std::is_invocable_v<BinaryOperator, common_t, common_t>;
      } else {
        return std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>;
      }
    } else {
      return false;
    }
  }

  template <typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()(data_type out_type) const
  {
    if constexpr (column_device_view::has_element_accessor<TypeLhs>() and
                  column_device_view::has_element_accessor<TypeRhs>()) {
      if (has_mutable_element_accessor(out_type) or is_fixed_point(out_type)) {
        if constexpr (binary_op_has_common_type_v<TypeLhs, TypeRhs>) {
          using common_t = binary_op_common_type_t<TypeLhs, TypeRhs>;
          if constexpr (std::is_invocable_v<BinaryOperator, common_t, common_t>) {
            using ReturnType = std::invoke_result_t<BinaryOperator, common_t, common_t>;
            return is_constructible<ReturnType>(out_type) or
                   (is_fixed_point<ReturnType>() and is_fixed_point(out_type));
          }
        } else if constexpr (std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>) {
          using ReturnType = std::invoke_result_t<BinaryOperator, TypeLhs, TypeRhs>;
          return is_constructible<ReturnType>(out_type);
        }
      }
    }
    return false;
  }
};

struct is_supported_operation_functor {
  template <typename TypeLhs, typename TypeRhs>
  struct nested_support_functor {
    template <typename BinaryOperator>
    [[nodiscard]] inline constexpr bool call(data_type out_type) const
    {
      return is_binary_operation_supported<BinaryOperator>{}.template operator()<TypeLhs, TypeRhs>(
        out_type);
    }
    inline constexpr bool operator()(binary_operator op, data_type out_type) const
    {
      switch (op) {
          // clang-format off
        case binary_operator::ADD:                  return call<ops::Add>(out_type);
        case binary_operator::SUB:                  return call<ops::Sub>(out_type);
        case binary_operator::MUL:                  return call<ops::Mul>(out_type);
        case binary_operator::DIV:                  return call<ops::Div>(out_type);
        case binary_operator::TRUE_DIV:             return call<ops::TrueDiv>(out_type);
        case binary_operator::FLOOR_DIV:            return call<ops::FloorDiv>(out_type);
        case binary_operator::MOD:                  return call<ops::Mod>(out_type);
        case binary_operator::PYMOD:                return call<ops::PyMod>(out_type);
        case binary_operator::POW:                  return call<ops::Pow>(out_type);
        case binary_operator::INT_POW:              return call<ops::IntPow>(out_type);
        case binary_operator::BITWISE_AND:          return call<ops::BitwiseAnd>(out_type);
        case binary_operator::BITWISE_OR:           return call<ops::BitwiseOr>(out_type);
        case binary_operator::BITWISE_XOR:          return call<ops::BitwiseXor>(out_type);
        case binary_operator::SHIFT_LEFT:           return call<ops::ShiftLeft>(out_type);
        case binary_operator::SHIFT_RIGHT:          return call<ops::ShiftRight>(out_type);
        case binary_operator::SHIFT_RIGHT_UNSIGNED: return call<ops::ShiftRightUnsigned>(out_type);
        case binary_operator::LOG_BASE:             return call<ops::LogBase>(out_type);
        case binary_operator::ATAN2:                return call<ops::ATan2>(out_type);
        case binary_operator::PMOD:                 return call<ops::PMod>(out_type);
        case binary_operator::NULL_MAX:             return call<ops::NullMax>(out_type);
        case binary_operator::NULL_MIN:             return call<ops::NullMin>(out_type);
        /*
        case binary_operator::GENERIC_BINARY:       // defined in jit only.
        */
        default:                                    return false;
          // clang-format on
      }
    }
  };

  template <typename BinaryOperator, typename TypeLhs, typename TypeRhs>
  [[nodiscard]] inline constexpr bool bool_op(data_type out) const
  {
    return out.id() == type_id::BOOL8 and
           is_binary_operation_supported<BinaryOperator>{}.template operator()<TypeLhs, TypeRhs>();
  }
  template <typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()(data_type out, binary_operator op) const
  {
    switch (op) {
      // output type should be bool type.
      case binary_operator::LOGICAL_AND: return bool_op<ops::LogicalAnd, TypeLhs, TypeRhs>(out);
      case binary_operator::LOGICAL_OR: return bool_op<ops::LogicalOr, TypeLhs, TypeRhs>(out);
      case binary_operator::EQUAL: return bool_op<ops::Equal, TypeLhs, TypeRhs>(out);
      case binary_operator::NOT_EQUAL: return bool_op<ops::NotEqual, TypeLhs, TypeRhs>(out);
      case binary_operator::LESS: return bool_op<ops::Less, TypeLhs, TypeRhs>(out);
      case binary_operator::GREATER: return bool_op<ops::Greater, TypeLhs, TypeRhs>(out);
      case binary_operator::LESS_EQUAL: return bool_op<ops::LessEqual, TypeLhs, TypeRhs>(out);
      case binary_operator::GREATER_EQUAL: return bool_op<ops::GreaterEqual, TypeLhs, TypeRhs>(out);
      case binary_operator::NULL_EQUALS: return bool_op<ops::NullEquals, TypeLhs, TypeRhs>(out);
      case binary_operator::NULL_NOT_EQUALS:
        return bool_op<ops::NullNotEquals, TypeLhs, TypeRhs>(out);
      case binary_operator::NULL_LOGICAL_AND:
        return bool_op<ops::NullLogicalAnd, TypeLhs, TypeRhs>(out);
      case binary_operator::NULL_LOGICAL_OR:
        return bool_op<ops::NullLogicalOr, TypeLhs, TypeRhs>(out);
      default: return nested_support_functor<TypeLhs, TypeRhs>{}(op, out);
    }
    return false;
  }
};

}  // namespace

std::optional<data_type> get_common_type(data_type out, data_type lhs, data_type rhs)
{
  // Compute the common type of (out, lhs, rhs) if it exists, or the common
  // type of (lhs, rhs) if it exists, else return a null optional.
  // We can avoid a triple type dispatch by using the definition of
  // std::common_type to compute this with double type dispatches.
  // Specifically, std::common_type_t<TypeOut, TypeLhs, TypeRhs> is the same as
  // std::common_type_t<std::common_type_t<TypeOut, TypeLhs>, TypeRhs>.
  auto common_type = double_type_dispatcher(out, lhs, common_type_functor{});
  if (common_type.has_value()) {
    common_type = double_type_dispatcher(common_type.value(), rhs, common_type_functor{});
  }
  // If no common type of (out, lhs, rhs) exists, fall back to the common type
  // of (lhs, rhs).
  if (!common_type.has_value()) {
    common_type = double_type_dispatcher(lhs, rhs, common_type_functor{});
  }
  return common_type;
}

bool is_supported_operation(data_type out, data_type lhs, data_type rhs, binary_operator op)
{
  // dictionary operands would resolve to their indices instead of their keys
  if (is_dictionary(out) or is_dictionary(lhs) or is_dictionary(rhs)) { return false; }
  return double_type_dispatcher(lhs, rhs, is_supported_operation_functor{}, out, op);
}
}  // namespace cudf::binops::compiled
