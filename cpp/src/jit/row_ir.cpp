/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include "runtime/context.hpp"

#include <cudf/column/column_factories.hpp>

#include <cuda/std/inplace_vector>

#include <algorithm>
#include <format>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace cudf::detail::row_ir {

/**
 * @brief Indicates how an operator propagates null values
 */
enum class [[nodiscard]] null_output : uint8_t {
  PROPAGATE       = 0,
  ALWAYS_VALID    = 1,
  ALWAYS_NULLABLE = 2,
};

/**
 * @brief Get the output scale for a given operator and input scales
 * This function returns the output scale for a given operator based on the input scales and the
 * target scale (if applicable). The output scale is used for decimal operators to determine the
 * scale of the result based on the scales of the inputs and the operator being performed.
 */
[[nodiscard]] int32_t get_op_output_scale(opcode op,
                                          std::span<int32_t const> arg_scales,
                                          std::optional<int32_t> target_scale)
{
  // this implementation must be kept in sync with the implementation of
  // `cudf::ast::detail::operator_functor::fixed_point_result_scale`
  // (https://github.com/rapidsai/cudf/blob/a5dccda20a74fe61e3c4491b0e74bdc0321d60d5/cpp/include/cudf/ast/detail/operator_functor.cuh#L161)
  switch (op) {
    // pseudo-opcode with no argument
    case opcode::GET_INPUT: return 0;
    // divisions
    case opcode::FLOOR_DIV:
    case opcode::DIV_OVERFLOW:
    case opcode::DIV: return arg_scales[0] - arg_scales[1];
    // add/sub/mod
    case opcode::ADD:
    case opcode::SUB:
    case opcode::ADD_OVERFLOW:
    case opcode::SUB_OVERFLOW:
    case opcode::MOD:
    case opcode::MOD_OVERFLOW:
    case opcode::PYMOD: return std::min(arg_scales[0], arg_scales[1]);
    case opcode::MUL:
    case opcode::MUL_OVERFLOW: return arg_scales[0] + arg_scales[1];
    case opcode::RESCALE: return target_scale.value();
    default: return arg_scales[0];
  }
}

/**
 * @brief A type mask used to indicate the types of the arguments and output of an operator.
 */
enum class [[nodiscard]] types : uint64_t {
  NONE                   = 0x0,
  BOOL8                  = 0x1,
  INT8                   = 0x2,
  INT16                  = 0x4,
  INT32                  = 0x8,
  INT64                  = 0x10,
  UINT8                  = 0x20,
  UINT16                 = 0x40,
  UINT32                 = 0x80,
  UINT64                 = 0x100,
  FLOAT32                = 0x200,
  FLOAT64                = 0x400,
  DECIMAL32              = 0x800,
  DECIMAL64              = 0x1000,
  DECIMAL128             = 0x2000,
  TIMESTAMP_DAYS         = 0x4000,
  TIMESTAMP_SECONDS      = 0x8000,
  TIMESTAMP_MILLISECONDS = 0x10000,
  TIMESTAMP_MICROSECONDS = 0x20000,
  TIMESTAMP_NANOSECONDS  = 0x40000,
  DURATION_DAYS          = 0x80000,
  DURATION_SECONDS       = 0x100000,
  DURATION_MILLISECONDS  = 0x200000,
  DURATION_MICROSECONDS  = 0x400000,
  DURATION_NANOSECONDS   = 0x800000,
  STRING                 = 0x1000000,
  SIGNED_INTEGERS        = INT8 | INT16 | INT32 | INT64,
  UNSIGNED_INTEGERS      = UINT8 | UINT16 | UINT32 | UINT64,
  INTEGERS               = SIGNED_INTEGERS | UNSIGNED_INTEGERS,
  INTEGRALS              = INTEGERS | BOOL8,
  FLOATS                 = FLOAT32 | FLOAT64,
  DECIMALS               = DECIMAL32 | DECIMAL64 | DECIMAL128,
  ARITHMETIC             = INTEGERS | FLOATS | DECIMALS,
  SIGNED_ARITHMETIC      = SIGNED_INTEGERS | FLOATS | DECIMALS,
  ALL                    = 0x0FFFFFFF,
  INPUT                  = 0x20000000,
};

/**
 * @brief A reference to an argument of an operator. Used to indicate that an argument is the same
 * type as a previous argument.
 */
enum class arg_ref : uint8_t { ARG0 = 0, ARG1 = 1, ARG2 = 2, ARG3 = 3 };

constexpr types operator|(types lhs, types rhs)
{
  return static_cast<types>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

constexpr types operator&(types lhs, types rhs)
{
  return static_cast<types>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

constexpr types operator~(types t) { return static_cast<types>(~static_cast<uint64_t>(t)); }

struct [[nodiscard]] opcode_info {
  std::string_view name                                                = "";
  null_output null_policy                                              = null_output::PROPAGATE;
  bool is_null_dependent                                               = false;
  bool is_fallible                                                     = false;
  cuda::std::inplace_vector<std::variant<types, arg_ref>, 4> arg_types = {};
  std::variant<types, arg_ref> output_type                             = types::NONE;
};

/**
 * @brief Get the opcode information for a given operator
 */
[[nodiscard]] opcode_info get_op_info(opcode op, error_policy error_policy)
{
  using enum null_output;
  using enum types;
  using enum arg_ref;

  static opcode_info const map[] =  // NOLINT(modernize-avoid-c-arrays)
    {
      {"GET_INPUT", PROPAGATE, false, false, {}, INPUT},
      {"SET_OUTPUT", PROPAGATE, false, false, {ALL}, NONE},
      {"IDENTITY", PROPAGATE, false, false, {ALL}, ARG0},
      {"IS_NULL", ALWAYS_VALID, true, false, {ALL}, BOOL8},
      {"COALESCE", PROPAGATE, true, false, {ALL, ARG0}, ARG0},
      {"PREDICATE", ALWAYS_VALID, true, false, {BOOL8}, ARG0},
      {"ADD", PROPAGATE, false, false, {ARITHMETIC, ARG0}, ARG0},
      {"SUB", PROPAGATE, false, false, {ARITHMETIC, ARG0}, ARG0},
      {"MUL", PROPAGATE, false, false, {ARITHMETIC, ARG0}, ARG0},
      {"DIV", PROPAGATE, false, false, {ARITHMETIC, ARG0}, ARG0},
      {"NEG", PROPAGATE, false, false, {ARITHMETIC}, ARG0},
      {"ABS", PROPAGATE, false, false, {ARITHMETIC}, ARG0},
      {"MOD", PROPAGATE, false, false, {ARITHMETIC, ARG0}, ARG0},
      {"PYMOD", PROPAGATE, false, false, {ARITHMETIC, ARG0}, ARG0},
      {"TRUE_DIV", PROPAGATE, false, false, {FLOATS | INTEGERS, ARG0}, FLOAT64},
      {"FLOOR_DIV", PROPAGATE, false, false, {FLOATS | INTEGERS, ARG0}, ARG0},
      {"ADD_OVERFLOW", PROPAGATE, false, true, {ARITHMETIC, ARG0}, ARG0},
      {"SUB_OVERFLOW", PROPAGATE, false, true, {ARITHMETIC, ARG0}, ARG0},
      {"MUL_OVERFLOW", PROPAGATE, false, true, {ARITHMETIC, ARG0}, ARG0},
      {"DIV_OVERFLOW", PROPAGATE, false, true, {ARITHMETIC, ARG0}, ARG0},
      {"NEG_OVERFLOW", PROPAGATE, false, true, {ARITHMETIC}, ARG0},
      {"ABS_OVERFLOW", PROPAGATE, false, true, {ARITHMETIC}, ARG0},
      {"MOD_OVERFLOW", PROPAGATE, false, true, {ARITHMETIC, ARG0}, ARG0},
      {"CHECK_PRECISION", PROPAGATE, false, true, {DECIMALS, INT32}, ARG0},
      {"BITWISE_AND", PROPAGATE, false, false, {INTEGERS, ARG0}, ARG0},
      {"BITWISE_INVERT", PROPAGATE, false, false, {INTEGERS}, ARG0},
      {"BITWISE_OR", PROPAGATE, false, false, {INTEGERS, ARG0}, ARG0},
      {"BITWISE_XOR", PROPAGATE, false, false, {INTEGERS, ARG0}, ARG0},
      {"BITWISE_SHIFT_LEFT", PROPAGATE, false, false, {INTEGERS, ARG0}, ARG0},
      {"BITWISE_SHIFT_RIGHT", PROPAGATE, false, false, {INTEGERS, ARG0}, ARG0},
      {"CAST_TO_BOOL8", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, BOOL8},
      {"CAST_TO_INT8", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, INT8},
      {"CAST_TO_INT16", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, INT16},
      {"CAST_TO_INT32", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, INT32},
      {"CAST_TO_INT64", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, INT64},
      {"CAST_TO_UINT8", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, UINT8},
      {"CAST_TO_UINT16", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, UINT16},
      {"CAST_TO_UINT32", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, UINT32},
      {"CAST_TO_UINT64", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, UINT64},
      {"CAST_TO_FLOAT32", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, FLOAT32},
      {"CAST_TO_FLOAT64", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, FLOAT64},
      {"CAST_TO_DECIMAL32", PROPAGATE, false, false, {DECIMALS}, DECIMAL32},
      {"CAST_TO_DECIMAL64", PROPAGATE, false, false, {DECIMALS}, DECIMAL64},
      {"CAST_TO_DECIMAL128", PROPAGATE, false, false, {DECIMALS}, DECIMAL128},
      {"RESCALE", PROPAGATE, false, false, {DECIMALS, INT32}, ARG0},
      {"EQUAL", PROPAGATE, false, false, {ALL, ARG0}, BOOL8},
      {"NOT_EQUAL", PROPAGATE, false, false, {ALL, ARG0}, BOOL8},
      {"GREATER", PROPAGATE, false, false, {ALL, ARG0}, BOOL8},
      {"GREATER_EQUAL", PROPAGATE, false, false, {ALL, ARG0}, BOOL8},
      {"LESS", PROPAGATE, false, false, {ALL, ARG0}, BOOL8},
      {"LESS_EQUAL", PROPAGATE, false, false, {ALL, ARG0}, BOOL8},
      {"NULL_EQUAL", ALWAYS_VALID, true, false, {ALL, ARG0}, BOOL8},
      {"NULL_LOGICAL_AND", ALWAYS_NULLABLE, true, false, {ARITHMETIC | BOOL8, ARG0}, BOOL8},
      {"NULL_LOGICAL_OR", ALWAYS_NULLABLE, true, false, {ARITHMETIC | BOOL8, ARG0}, BOOL8},
      {"LOGICAL_AND", PROPAGATE, false, false, {ARITHMETIC | BOOL8, ARG0}, BOOL8},
      {"LOGICAL_OR", PROPAGATE, false, false, {ARITHMETIC | BOOL8, ARG0}, BOOL8},
      {"LOGICAL_NOT", PROPAGATE, false, false, {ARITHMETIC | BOOL8}, BOOL8},
      {"IF_ELSE", PROPAGATE, true, false, {ALL, ARG0, INTEGRALS}, ARG0},
      {"CBRT", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"CEIL", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"FLOOR", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"RINT", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"SQRT", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"POW", PROPAGATE, false, false, {FLOATS | INTEGERS, ARG0}, ARG0},
      {"EXP", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"LOG", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"ARCCOS", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"ARCCOSH", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"ARCSIN", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"ARCSINH", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"ARCTAN", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"ARCTANH", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"COS", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"COSH", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"SIN", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"SINH", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"TAN", PROPAGATE, false, false, {FLOATS}, ARG0},
      {"TANH", PROPAGATE, false, false, {FLOATS}, ARG0},
    };

  auto index = static_cast<size_t>(op);
  CUDF_EXPECTS(index < std::size(map),
               std::format("Invalid opcode: {}", static_cast<int>(op)),
               std::runtime_error);

  auto info = map[index];
  if (error_policy == cudf::error_policy::NULLIFY && info.is_fallible) {
    info.is_fallible = false;
    info.null_policy = null_output::ALWAYS_NULLABLE;
  }

  return info;
}

opcode as_opcode(ast::ast_operator op)
{
  switch (op) {
    case ast::ast_operator::ADD: return opcode::ADD;
    case ast::ast_operator::SUB: return opcode::SUB;
    case ast::ast_operator::MUL: return opcode::MUL;
    case ast::ast_operator::DIV: return opcode::DIV;
    case ast::ast_operator::TRUE_DIV: return opcode::TRUE_DIV;
    case ast::ast_operator::FLOOR_DIV: return opcode::FLOOR_DIV;
    case ast::ast_operator::MOD: return opcode::MOD;
    case ast::ast_operator::PYMOD: return opcode::PYMOD;
    case ast::ast_operator::POW: return opcode::POW;
    case ast::ast_operator::EQUAL: return opcode::EQUAL;
    case ast::ast_operator::NULL_EQUAL: return opcode::NULL_EQUAL;
    case ast::ast_operator::NOT_EQUAL: return opcode::NOT_EQUAL;
    case ast::ast_operator::LESS: return opcode::LESS;
    case ast::ast_operator::GREATER: return opcode::GREATER;
    case ast::ast_operator::LESS_EQUAL: return opcode::LESS_EQUAL;
    case ast::ast_operator::GREATER_EQUAL: return opcode::GREATER_EQUAL;
    case ast::ast_operator::BITWISE_AND: return opcode::BITWISE_AND;
    case ast::ast_operator::BITWISE_OR: return opcode::BITWISE_OR;
    case ast::ast_operator::BITWISE_XOR: return opcode::BITWISE_XOR;
    case ast::ast_operator::LOGICAL_AND: return opcode::LOGICAL_AND;
    case ast::ast_operator::NULL_LOGICAL_AND: return opcode::NULL_LOGICAL_AND;
    case ast::ast_operator::LOGICAL_OR: return opcode::LOGICAL_OR;
    case ast::ast_operator::NULL_LOGICAL_OR: return opcode::NULL_LOGICAL_OR;
    case ast::ast_operator::IDENTITY: return opcode::IDENTITY;
    case ast::ast_operator::IS_NULL: return opcode::IS_NULL;
    case ast::ast_operator::SIN: return opcode::SIN;
    case ast::ast_operator::COS: return opcode::COS;
    case ast::ast_operator::TAN: return opcode::TAN;
    case ast::ast_operator::ARCSIN: return opcode::ARCSIN;
    case ast::ast_operator::ARCCOS: return opcode::ARCCOS;
    case ast::ast_operator::ARCTAN: return opcode::ARCTAN;
    case ast::ast_operator::SINH: return opcode::SINH;
    case ast::ast_operator::COSH: return opcode::COSH;
    case ast::ast_operator::TANH: return opcode::TANH;
    case ast::ast_operator::ARCSINH: return opcode::ARCSINH;
    case ast::ast_operator::ARCCOSH: return opcode::ARCCOSH;
    case ast::ast_operator::ARCTANH: return opcode::ARCTANH;
    case ast::ast_operator::EXP: return opcode::EXP;
    case ast::ast_operator::LOG: return opcode::LOG;
    case ast::ast_operator::SQRT: return opcode::SQRT;
    case ast::ast_operator::CBRT: return opcode::CBRT;
    case ast::ast_operator::CEIL: return opcode::CEIL;
    case ast::ast_operator::FLOOR: return opcode::FLOOR;
    case ast::ast_operator::ABS: return opcode::ABS;
    case ast::ast_operator::RINT: return opcode::RINT;
    case ast::ast_operator::BIT_INVERT: return opcode::BITWISE_INVERT;
    case ast::ast_operator::NOT: return opcode::LOGICAL_NOT;
    case ast::ast_operator::CAST_TO_INT64: return opcode::CAST_TO_INT64;
    case ast::ast_operator::CAST_TO_UINT64: return opcode::CAST_TO_UINT64;
    case ast::ast_operator::CAST_TO_FLOAT64: return opcode::CAST_TO_FLOAT64;
    default:
      CUDF_FAIL(std::format("Unrecognized operator type: {}.", static_cast<int>(op)),
                std::runtime_error);
  }
}

row_ir::types as_type(data_type type)
{
  switch (type.id()) {
    case type_id::BOOL8: return types::BOOL8;
    case type_id::INT8: return types::INT8;
    case type_id::INT16: return types::INT16;
    case type_id::INT32: return types::INT32;
    case type_id::INT64: return types::INT64;
    case type_id::UINT8: return types::UINT8;
    case type_id::UINT16: return types::UINT16;
    case type_id::UINT32: return types::UINT32;
    case type_id::UINT64: return types::UINT64;
    case type_id::FLOAT32: return types::FLOAT32;
    case type_id::FLOAT64: return types::FLOAT64;
    case type_id::DECIMAL32: return types::DECIMAL32;
    case type_id::DECIMAL64: return types::DECIMAL64;
    case type_id::DECIMAL128: return types::DECIMAL128;
    case type_id::TIMESTAMP_DAYS: return types::TIMESTAMP_DAYS;
    case type_id::TIMESTAMP_SECONDS: return types::TIMESTAMP_SECONDS;
    case type_id::TIMESTAMP_MILLISECONDS: return types::TIMESTAMP_MILLISECONDS;
    case type_id::TIMESTAMP_MICROSECONDS: return types::TIMESTAMP_MICROSECONDS;
    case type_id::TIMESTAMP_NANOSECONDS: return types::TIMESTAMP_NANOSECONDS;
    case type_id::DURATION_DAYS: return types::DURATION_DAYS;
    case type_id::DURATION_SECONDS: return types::DURATION_SECONDS;
    case type_id::DURATION_MILLISECONDS: return types::DURATION_MILLISECONDS;
    case type_id::DURATION_MICROSECONDS: return types::DURATION_MICROSECONDS;
    case type_id::DURATION_NANOSECONDS: return types::DURATION_NANOSECONDS;
    case type_id::STRING: return types::STRING;
    default:
      CUDF_FAIL(std::format("Unsupported data type for Row IR: {}", type_to_name(type)),
                std::invalid_argument);
  }
}

type_id as_type_id(types type)
{
  switch (type) {
    case types::BOOL8: return type_id::BOOL8;
    case types::INT8: return type_id::INT8;
    case types::INT16: return type_id::INT16;
    case types::INT32: return type_id::INT32;
    case types::INT64: return type_id::INT64;
    case types::UINT8: return type_id::UINT8;
    case types::UINT16: return type_id::UINT16;
    case types::UINT32: return type_id::UINT32;
    case types::UINT64: return type_id::UINT64;
    case types::FLOAT32: return type_id::FLOAT32;
    case types::FLOAT64: return type_id::FLOAT64;
    case types::DECIMAL32: return type_id::DECIMAL32;
    case types::DECIMAL64: return type_id::DECIMAL64;
    case types::DECIMAL128: return type_id::DECIMAL128;
    case types::TIMESTAMP_DAYS: return type_id::TIMESTAMP_DAYS;
    case types::TIMESTAMP_SECONDS: return type_id::TIMESTAMP_SECONDS;
    case types::TIMESTAMP_MILLISECONDS: return type_id::TIMESTAMP_MILLISECONDS;
    case types::TIMESTAMP_MICROSECONDS: return type_id::TIMESTAMP_MICROSECONDS;
    case types::TIMESTAMP_NANOSECONDS: return type_id::TIMESTAMP_NANOSECONDS;
    case types::DURATION_DAYS: return type_id::DURATION_DAYS;
    case types::DURATION_SECONDS: return type_id::DURATION_SECONDS;
    case types::DURATION_MILLISECONDS: return type_id::DURATION_MILLISECONDS;
    case types::DURATION_MICROSECONDS: return type_id::DURATION_MICROSECONDS;
    case types::DURATION_NANOSECONDS: return type_id::DURATION_NANOSECONDS;
    case types::STRING: return type_id::STRING;
    default:
      CUDF_FAIL(std::format("Invalid typing for {}: {}", __FUNCTION__, static_cast<int>(type)),
                std::invalid_argument);
  }
}

data_type get_return_type(opcode op,
                          std::span<data_type const> args,
                          std::optional<int32_t> target_scale)
{
  std::vector<row_ir::types> arg_types;
  std::vector<int32_t> arg_scales;

  for (auto& type : args) {
    arg_types.emplace_back(as_type(type));
    arg_scales.emplace_back(type.scale());
  }

  auto op_info  = get_op_info(op, error_policy::PROPAGATE);
  auto rescaled = get_op_output_scale(op, arg_scales, target_scale);

  for (size_t i = 0; i < args.size(); ++i) {
    auto spec     = op_info.arg_types[i];
    auto arg_type = arg_types[i];

    if (auto* ref = std::get_if<arg_ref>(&spec)) {
      auto src_index = static_cast<size_t>(*ref);
      CUDF_EXPECTS(
        src_index < i,
        std::format("Invalid type match rule for operator `{}` at argument #{}", op_info.name, i),
        std::runtime_error);
      CUDF_EXPECTS(args[i].id() == args[src_index].id(),
                   std::format("Argument #{} of operator `{}` does not match type of argument "
                               "#{}. Got `{}`, expected `{}`",
                               i,
                               op_info.name,
                               src_index,
                               type_to_name(args[i]),
                               type_to_name(args[src_index])));
    } else {
      auto required_type = std::get<types>(spec);
      CUDF_EXPECTS(
        (arg_type & required_type) != types::NONE,
        std::format("Argument #{} of operator `{}` does not match expected types. Got {}",
                    i,
                    op_info.name,
                    type_to_name(args[i])));
    }
  }

  auto type = [&] {
    if (auto* ref = std::get_if<arg_ref>(&op_info.output_type)) {
      auto arg_index = static_cast<size_t>(*ref);
      return args[arg_index].id();
    } else {
      auto required_type = std::get<types>(op_info.output_type);
      CUDF_EXPECTS(
        required_type != types::NONE,
        std::format("Invalid type match rule for operator `{}` return type", op_info.name),
        std::runtime_error);
      return as_type_id(required_type);
    }
  }();
  return is_fixed_point(data_type{type}) ? data_type{type, numeric::scale_type{rescaled}}
                                         : data_type{type};
}

int32_t instance_context::add_output()
{
  auto id     = static_cast<int32_t>(output_vars_.size());
  auto id_str = std::format("out_{}", id);
  output_vars_.emplace_back(std::move(id_str));
  return id;
}

int32_t instance_context::add_input(input in)
{
  auto id     = static_cast<int32_t>(inputs_.size());
  auto id_str = std::format("in_{}", id);

  data_type const type = [&in] {
    if (auto* col = std::get_if<column_input>(&in)) {
      return col->column.type();
    } else {
      auto& scalar = std::get<scalar_input>(in);
      return scalar.scalar_column->type();
    }
  }();
  inputs_.emplace_back(std::move(in));
  input_vars_.emplace_back(std::move(id_str), type);
  return id;
}

std::string instance_context::make_tmp_id()
{
  return std::format("{}{}", tmp_prefix_, num_tmp_vars_++);
}

bool instance_context::has_nulls() const { return has_nulls_; }

void instance_context::set_has_nulls(bool has_nulls) { has_nulls_ = has_nulls; }

std::span<input const> instance_context::get_inputs() const { return inputs_; }

std::span<var_info const> instance_context::get_input_vars() const { return input_vars_; }

std::span<untyped_var_info const> instance_context::get_output_vars() const { return output_vars_; }

node::node(opcode op,
           std::optional<int32_t> target_scale,
           error_policy error_policy,
           std::vector<std::unique_ptr<node>> args)
  : op_{op}, target_scale_{target_scale}, error_policy_{error_policy}, args_{std::move(args)}
{
  auto op_info = get_op_info(op_, error_policy_);
  CUDF_EXPECTS(op_ != opcode::GET_INPUT && op_ != opcode::SET_OUTPUT,
               std::format("Invalid opcode `{}` for operation node.", op_info.name),
               std::runtime_error);

  if (!get_op_info(op_, error_policy::PROPAGATE).is_fallible) {
    CUDF_EXPECTS(error_policy_ != error_policy::NULLIFY,
                 std::format("Invalid error policy `NULLIFY` for operator `{}`. Only operators "
                             "that are fallible can use the NULLIFY error policy.",
                             op_info.name),
                 std::runtime_error);
  }

  if (op != opcode::RESCALE) {
    auto op_info        = get_op_info(op_, error_policy_);
    auto expected_arity = op_info.arg_types.size();
    auto actual_arity   = args_.size();
    CUDF_EXPECTS(actual_arity == expected_arity,
                 std::format("Invalid number of arguments for operator `{}`. Expected {}, Got {}.",
                             op_info.name,
                             expected_arity,
                             actual_arity),
                 std::runtime_error);
  } else {
    CUDF_EXPECTS(args_.size() == 1,
                 std::format("RESCALE operator expects exactly 1 argument. Got {}.", args_.size()));
    CUDF_EXPECTS(
      target_scale_.has_value(),
      std::format("Target scale must be provided for RESCALE operator and must be nullopt "
                  "for other operators."));
  }
}

node::node(input_reference input)
  : reference_{input}, op_{opcode::GET_INPUT}  // NOLINT(modernize-use-default-member-init)
{
}

node::node(output_reference reference, std::unique_ptr<node> arg)
  : reference_{reference}, op_{opcode::SET_OUTPUT}
{
  args_.emplace_back(std::move(arg));
}

node::node(output_reference reference, node arg)
  : node{reference, std::make_unique<node>(std::move(arg))}
{
}

std::string_view node::get_id() const { return id_; }

data_type node::get_type() const { return type_; }

std::optional<int32_t> node::get_target_scale() const { return target_scale_; }

opcode node::get_opcode() const { return op_; }

std::span<std::unique_ptr<node> const> node::get_args() const { return args_; }

bool node::is_null_aware() const
{
  if (op_ == opcode::GET_INPUT) { return false; }

  // to emit nulls for always-nullable operators, we  need to mark them as null-aware
  auto op_info = get_op_info(op_, error_policy_);

  if (op_info.null_policy == null_output::ALWAYS_NULLABLE) { return true; }

  if (op_info.is_null_dependent) { return true; }

  CUDF_EXPECTS(!args_.empty(),
               "Unexpectedly found an operator node with no arguments. All operator nodes should "
               "have at least one argument.",
               std::runtime_error);

  return std::any_of(args_.begin(), args_.end(), [](auto& a) { return a->is_null_aware(); });
}

bool node::is_always_valid() const
{
  if (op_ == opcode::GET_INPUT) { return false; }

  auto null_policy = get_op_info(op_, error_policy_).null_policy;

  switch (null_policy) {
    case null_output::ALWAYS_NULLABLE: {
      return false;
    }
    case null_output::ALWAYS_VALID: {
      return true;
    }
    case null_output::PROPAGATE:
    default: break;
  }

  CUDF_EXPECTS(!args_.empty(),
               "Unexpectedly found an operator node with no arguments. All operator nodes should "
               "have at least one argument.",
               std::runtime_error);

  return std::all_of(args_.begin(), args_.end(), [](auto& a) { return a->is_always_valid(); });
}

std::string to_cuda_type(cudf::data_type type, bool nullable)
{
  auto name = type_to_name(type);
  return nullable ? std::format("cuda::std::optional<{}>", name) : name;
}

void node::instantiate(instance_context& ctx)
{
  for (auto& arg : args_) {
    arg->instantiate(ctx);
  }

  id_ = ctx.make_tmp_id();

  switch (op_) {
    case opcode::GET_INPUT: {
      type_ = ctx.get_input_vars()[std::get<input_reference>(reference_).index].type;
    } break;
    case opcode::SET_OUTPUT: {
      type_ = args_[0]->get_type();
    } break;
    default: {
      std::vector<data_type> arg_types;
      for (auto& arg : args_) {
        arg_types.emplace_back(arg->get_type());
      }

      if (op_ == opcode::RESCALE) {
        scale_reference_ = input_reference{ctx.add_input(cudf::numeric_scalar<int32_t>{
          target_scale_.value_or(0), ctx.get_stream(), ctx.get_mr()})};
        arg_types.emplace_back(cudf::type_id::INT32);
      }

      type_ = get_return_type(op_, arg_types, target_scale_);
    } break;
  }
}

void node::emit_code(instance_context& instance, target_info const& info, code_sink& sink) const
{
  for (auto& arg : args_) {
    arg->emit_code(instance, info, sink);
  }

  switch (info.id) {
    case target::CUDA: {
      auto type = to_cuda_type(type_, instance.has_nulls());

      switch (op_) {
        case opcode::GET_INPUT: {
          sink.emit(std::format(
            R"***({} {} = {};
)***",
            type,
            id_,
            instance.get_input_vars()[std::get<input_reference>(reference_).index].id));
        } break;

        case opcode::SET_OUTPUT: {
          sink.emit(std::format(
            R"***({} {} = {};
*{} = {};
)***",
            type,
            id_,
            args_[0]->get_id(),
            instance.get_output_vars()[std::get<output_reference>(reference_).index].id,
            id_));
        } break;

        default: {
          auto op_info   = get_op_info(op_, error_policy_);
          auto first_arg = std::format("{}", args_[0]->get_id());
          auto args_str  = (args_.size() == 1)
                             ? std::string{first_arg}
                             : std::accumulate(args_.begin() + 1,
                                              args_.end(),
                                              std::string{first_arg},
                                              [](auto const& a, auto& node) {
                                                return std::format("{}, {}", a, node->get_id());
                                              });

          if (op_ == opcode::RESCALE) {
            args_str =
              std::format("{}, {}", args_str, instance.get_input_vars()[scale_reference_.index].id);
          }

          auto error_policy = error_policy_ == cudf::error_policy::NULLIFY
                                ? "cudf::error_policy::NULLIFY"
                                : "cudf::error_policy::PROPAGATE";

          if (!op_info.is_fallible) {
            sink.emit(std::format(
              R"***({0} {1} = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::{2}, {3}>({4});
)***",
              type,
              id_,
              op_info.name,
              error_policy,
              args_str));
          } else {
            if (error_policy_ != cudf::error_policy::NULLIFY) {
              sink.emit(std::format(
                R"***(auto expected__{1} = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::{2}, {3}>({4});
if(!expected__{1}.has_value()) {{
 return expected__{1}.error();
}}
{0} {1} = expected__{1}.value();
)***",
                type,
                id_,
                op_info.name,
                error_policy,
                args_str));
            } else {
              sink.emit(std::format(
                R"***({0} {1}{{}};
auto expected__{1} = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::{2}, {3}>({4});
if(expected__{1}.has_value()) {{
  {1} = expected__{1}.value();
}} else {{
  {1} = {{}};
}}
)***",
                type,
                id_,
                op_info.name,
                error_policy,
                args_str));
            }
          }
        }
      }
      break;
    }

    default:
      CUDF_FAIL(std::format("Unsupported target: {}", static_cast<int>(info.id)),
                std::invalid_argument);
  }
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::literal const& expr)
{
  auto id = instance_.add_input(expr.get_scalar());
  return std::make_unique<row_ir::node>(input_reference{id});
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::column_reference const& expr)
{
  // resolve the table for a column input spec, preferring left_table/right_table for join cases,
  // falling back to args.table for the single-table case.
  auto resolve = [&](ast::table_reference ref) {
    CUDF_EXPECTS(ref == ast::table_reference::LEFT || ref == ast::table_reference::RIGHT,
                 "Invalid table reference in column expression",
                 std::invalid_argument);
    return ref == ast::table_reference::LEFT ? left_table_ : right_table_;
  };

  auto table = resolve(expr.get_table_source());
  auto id    = instance_.add_input(
    column_input{.column       = table.column(expr.get_column_index()),
                    .table_source = (expr.get_table_source() == ast::table_reference::LEFT ? 0 : 1),
                    .column_index = static_cast<int32_t>(expr.get_column_index())});
  return std::make_unique<row_ir::node>(input_reference{id});
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> args;
  for (auto& operand : expr.get_operands()) {
    args.emplace_back(operand.get().accept(*this));
  }
  return std::make_unique<row_ir::node>(
    as_opcode(expr.get_operator()), std::nullopt, error_policy::PROPAGATE, std::move(args));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::detail::predicate const& expr)
{
  return std::make_unique<row_ir::node>(row_ir::opcode::PREDICATE,
                                        std::nullopt,
                                        error_policy::PROPAGATE,
                                        expr.get_operand().accept(*this));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::jit::detail::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> args;
  for (auto& arg : expr.get_arguments()) {
    args.emplace_back(arg.get().accept(*this));
  }
  return std::make_unique<row_ir::node>(
    expr.get_opcode(), expr.get_target_scale(), expr.get_error_policy(), std::move(args));
}

bool is_nullable(scalar_input const& in) { return in.scalar_column->view().nullable(); }

bool is_nullable(column_input const& in) { return in.column.nullable(); }

std::tuple<std::string, null_aware, output_nullability> ast_converter::generate_code(
  target target_id, ast::expression const& expr, std::string_view function_name)
{
  // add 1 auto-deduced output variable
  [[maybe_unused]] auto output_id = instance_.add_output();

  output_irs_.emplace_back(std::make_unique<row_ir::node>(output_reference{0}, expr.accept(*this)));

  bool has_nullable_inputs =
    std::any_of(instance_.inputs_.begin(), instance_.inputs_.end(), [&](auto& in) {
      return std::visit([](auto& c) { return is_nullable(c); }, in);
    });

  bool is_null_aware = std::any_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_null_aware(); });

  bool output_is_always_valid = std::all_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_always_valid(); });

  bool may_evaluate_null = output_is_always_valid ? false : (has_nullable_inputs || is_null_aware);

  auto null_policy =
    may_evaluate_null ? output_nullability::PRESERVE : output_nullability::ALL_VALID;

  instance_.set_has_nulls(is_null_aware);

  // instantiate the IR nodes
  for (auto& ir : output_irs_) {
    ir->instantiate(instance_);
  }

  target_info target{target_id};

  CUDF_EXPECTS(
    target.id == target::CUDA, "Unsupported target for code generation", std::invalid_argument);

  auto output_decl = [&](auto i) {
    auto& var = instance_.output_vars_[i];
    auto& ir  = output_irs_[i];
    return std::format("{}* {}", to_cuda_type(ir->get_type(), instance_.has_nulls()), var.id);
  };

  auto input_decl = [&](auto i) {
    auto& var = instance_.input_vars_[i];
    return std::format("{} {}", to_cuda_type(var.type, instance_.has_nulls()), var.id);
  };

  std::vector<std::string> arg_decls;

  for (size_t i = 0; i < instance_.output_vars_.size(); ++i) {
    arg_decls.emplace_back(output_decl(i));
  }

  for (size_t i = 0; i < instance_.input_vars_.size(); ++i) {
    arg_decls.emplace_back(input_decl(i));
  }

  auto args_decl = [&] {
    if (arg_decls.empty()) {
      return std::string{};
    } else if (arg_decls.size() == 1) {
      return arg_decls[0];
    } else {
      return std::accumulate(
        arg_decls.begin() + 1, arg_decls.end(), arg_decls[0], [](auto const& a, auto const& b) {
          return std::format("{}, {}", a, b);
        });
    }
  }();

  code_sink sink;
  sink.emit(std::format("__device__ cudf::errc {}(", function_name));
  sink.emit(args_decl);
  sink.emit(")\n{\n");
  for (auto& ir : output_irs_) {
    ir->emit_code(instance_, target, sink);
  }
  sink.emit("return cudf::errc::SUCCESS;\n}");
  return {sink.get_code(), is_null_aware ? null_aware::YES : null_aware::NO, null_policy};
}

std::variant<column_view, scalar_column_view> get_column_view(scalar_input const& in)
{
  return scalar_column_view{in.scalar_column->view()};
}

std::variant<column_view, scalar_column_view> get_column_view(column_input const& in)
{
  return column_view{in.column};
}

// Due to the AST expression tree structure, we can't generate the IR without the target
// tables
transform_args ast_converter::compute_column(target target_id,
                                             ast::expression const& expr,
                                             table_view const& left_table,
                                             table_view const& right_table,
                                             std::string_view function_name,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  ast_converter converter{stream, mr, left_table, right_table};

  // TODO(lamarrr): consider deduplicating ast expression's input column references. See
  // TransformTest/1.DeeplyNestedArithmeticLogicalExpression for reference

  auto [code, is_null_aware, output_nullability] =
    converter.generate_code(target_id, expr, function_name);
  std::vector<std::variant<column_view, scalar_column_view>> inputs;
  std::vector<std::unique_ptr<column>> scalar_columns;
  std::vector<std::optional<int32_t>> table_sources;
  std::vector<std::optional<int32_t>> column_indices;

  for (auto& input : converter.instance_.inputs_) {
    if (std::holds_alternative<column_input>(input)) {
      auto& col = std::get<column_input>(input);
      table_sources.emplace_back(col.table_source);
      column_indices.emplace_back(col.column_index);
    } else {
      table_sources.emplace_back(std::nullopt);
      column_indices.emplace_back(std::nullopt);
    }

    auto view = std::visit([](auto& in) { return get_column_view(in); }, input);
    inputs.emplace_back(view);

    if (std::holds_alternative<scalar_input>(input)) {
      auto& scalar = std::get<scalar_input>(input);
      scalar_columns.emplace_back(std::move(scalar.scalar_column));
    }
  }

  auto& out               = converter.output_irs_[0];
  auto output_column_type = out->get_type();
  auto output   = transform_output{.type = output_column_type, .nullability = output_nullability};
  auto row_size = std::max({left_table.num_rows(), right_table.num_rows()});
  auto result   = transform_args{.scalar_columns       = std::move(scalar_columns),
                                 .input_table_sources  = std::move(table_sources),
                                 .input_column_indices = std::move(column_indices),
                                 .udf                  = std::move(code),
                                 .source_type          = cudf::udf_source_type::CUDA,
                                 .is_null_aware        = is_null_aware,
                                 .user_data            = std::nullopt,
                                 .inputs               = inputs,
                                 .outputs{output},
                                 .string_offsets{},
                                 .row_size = row_size};
  if (get_context().dump_codegen()) {
    std::cout << "Generated code for transform: \n" << result.udf << std::endl;
  }

  return result;
}

transform_args ast_converter::filter(target target_id,
                                     ast::expression const& expr,
                                     table_view const& left_table,
                                     table_view const& right_table,
                                     std::string_view function_name,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto filter = ast::detail::predicate{expr};
  return compute_column(target_id, filter, left_table, right_table, function_name, stream, mr);
}

}  // namespace cudf::detail::row_ir
