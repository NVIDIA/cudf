/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <functional>
#include <iterator>

namespace cudf {

namespace ast {

namespace detail {

device_data_reference::device_data_reference(device_data_reference_type reference_type,
                                             cudf::data_type data_type,
                                             cudf::size_type data_index,
                                             table_reference table_source)
  : reference_type(reference_type),
    data_type(data_type),
    data_index(data_index),
    table_source(table_source)
{
}

device_data_reference::device_data_reference(device_data_reference_type reference_type,
                                             cudf::data_type data_type,
                                             cudf::size_type data_index)
  : reference_type(reference_type),
    data_type(data_type),
    data_index(data_index),
    table_source(table_reference::LEFT)
{
}

bool device_data_reference::operator==(device_data_reference const& rhs) const
{
  return std::tie(data_index, data_type, reference_type, table_source) ==
         std::tie(rhs.data_index, rhs.data_type, rhs.reference_type, rhs.table_source);
}

expression_parser::expression_parser(
  expression const& expr,
  cudf::table_view const& left,
  std::optional<std::reference_wrapper<cudf::table_view const>> right,
  bool has_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : _left{left},
    _right{right},
    _expression_count{0},
    _intermediate_counter{},
    _has_nulls(has_nulls),
    _has_complex_type{false}
{
  expr.accept(*this);
  _has_complex_type =
    std::any_of(_data_references.begin(), _data_references.end(), [&](auto const& ref) {
      return ast::detail::is_complex_type(ref.data_type.id());
    });

  move_to_device(stream, mr);
}

expression_parser::expression_parser(expression const& expr,
                                     cudf::table_view const& table,
                                     bool has_nulls,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  : expression_parser(expr, table, {}, has_nulls, stream, mr)
{
}

void expression_parser::move_to_device(rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  std::vector<cudf::size_type> sizes;
  std::vector<void const*> data_pointers;
  // use a minimum of 4-byte alignment
  cudf::size_type buffer_alignment = 4;

  extract_size_and_pointer(_data_references, sizes, data_pointers, buffer_alignment);
  extract_size_and_pointer(_literals, sizes, data_pointers, buffer_alignment);
  extract_size_and_pointer(_operators, sizes, data_pointers, buffer_alignment);
  extract_size_and_pointer(_operator_arities, sizes, data_pointers, buffer_alignment);
  extract_size_and_pointer(_operator_source_indices, sizes, data_pointers, buffer_alignment);

  // Create device buffer
  auto buffer_offsets = std::vector<cudf::size_type>(sizes.size());
  thrust::exclusive_scan(sizes.cbegin(),
                         sizes.cend(),
                         buffer_offsets.begin(),
                         cudf::size_type{0},
                         [buffer_alignment](auto a, auto b) {
                           // align each component of the AST program
                           return cudf::util::round_up_safe(a + b, buffer_alignment);
                         });

  auto const buffer_size = buffer_offsets.empty() ? 0 : (buffer_offsets.back() + sizes.back());
  auto host_data_buffer  = std::vector<char>(buffer_size);

  for (unsigned int i = 0; i < data_pointers.size(); ++i) {
    std::memcpy(host_data_buffer.data() + buffer_offsets[i], data_pointers[i], sizes[i]);
  }

  _device_data_buffer = rmm::device_buffer(host_data_buffer.data(), buffer_size, stream, mr);
  stream.synchronize();

  // Create device pointers to components of plan
  auto device_data_buffer_ptr            = static_cast<char const*>(_device_data_buffer.data());
  device_expression_data.data_references = device_span<detail::device_data_reference const>(
    reinterpret_cast<detail::device_data_reference const*>(device_data_buffer_ptr +
                                                           buffer_offsets[0]),
    _data_references.size());
  device_expression_data.literals = device_span<generic_scalar_device_view const>(
    reinterpret_cast<generic_scalar_device_view const*>(device_data_buffer_ptr + buffer_offsets[1]),
    _literals.size());
  device_expression_data.operators = device_span<ast_operator const>(
    reinterpret_cast<ast_operator const*>(device_data_buffer_ptr + buffer_offsets[2]),
    _operators.size());
  device_expression_data.operator_arities = device_span<cudf::size_type const>(
    reinterpret_cast<cudf::size_type const*>(device_data_buffer_ptr + buffer_offsets[3]),
    _operators.size());
  device_expression_data.operator_source_indices = device_span<cudf::size_type const>(
    reinterpret_cast<cudf::size_type const*>(device_data_buffer_ptr + buffer_offsets[4]),
    _operator_source_indices.size());
  device_expression_data.num_intermediates = _intermediate_counter.get_max_used();
  shmem_per_thread                         = static_cast<int>(
    (_has_nulls ? sizeof(IntermediateDataType<true>) : sizeof(IntermediateDataType<false>)) *
    device_expression_data.num_intermediates);
}

cudf::size_type expression_parser::intermediate_counter::take()
{
  auto const first_missing = find_first_missing();
  used_values.insert(used_values.cbegin() + first_missing, first_missing);
  max_used = std::max(max_used, first_missing + 1);
  return first_missing;
}

void expression_parser::intermediate_counter::give(cudf::size_type value)
{
  // TODO: add comment
  auto const lower_bound = std::lower_bound(used_values.cbegin(), used_values.cend(), value);
  if ((lower_bound != used_values.cend()) && (*lower_bound == value))
    used_values.erase(lower_bound);
}

cudf::size_type expression_parser::intermediate_counter::find_first_missing() const
{
  if (used_values.empty() || (used_values.front() != 0)) { return 0; }
  // Search for the first non-contiguous pair of elements.
  auto diff_not_one = [](auto a, auto b) { return a != b - 1; };
  auto it           = std::adjacent_find(used_values.cbegin(), used_values.cend(), diff_not_one);
  return it != used_values.cend()
           ? *it + 1              // A missing value was found and is returned.
           : used_values.size();  // No missing elements. Return the next element in the sequence.
}

cudf::size_type expression_parser::visit(literal const& expr)
{
  if (_expression_count == 0) {
    // Handle the trivial case of a literal as the entire expression.
    return visit(operation(ast_operator::IDENTITY, expr));
  } else {
    _expression_count++;                                           // Increment the expression index
    auto const data_type     = expr.get_data_type();               // Resolve expression type
    auto device_view         = expr.get_value();                   // Construct a scalar device view
    auto const literal_index = cudf::size_type(_literals.size());  // Push literal
    _literals.push_back(device_view);
    auto const source = detail::device_data_reference(detail::device_data_reference_type::LITERAL,
                                                      data_type,
                                                      literal_index);  // Push data reference
    return add_data_reference(source);
  }
}

cudf::size_type expression_parser::visit(column_reference const& expr)
{
  if (_expression_count == 0) {
    // Handle the trivial case of a column reference as the entire expression.
    return visit(operation(ast_operator::IDENTITY, expr));
  } else {
    // Increment the expression index
    _expression_count++;
    // Resolve expression type
    cudf::data_type data_type;
    if (expr.get_table_source() == table_reference::LEFT) {
      data_type = expr.get_data_type(_left);
    } else {
      if (_right.has_value()) {
        data_type = expr.get_data_type(*_right);
      } else {
        CUDF_FAIL(
          "Your expression contains a reference to the RIGHT table even though it will only be "
          "evaluated on a single table (by convention, the LEFT table).");
      }
    }
    // Push data reference
    auto const source = detail::device_data_reference(detail::device_data_reference_type::COLUMN,
                                                      data_type,
                                                      expr.get_column_index(),
                                                      expr.get_table_source());
    return add_data_reference(source);
  }
}

cudf::size_type expression_parser::visit(operation const& expr)
{
  // Increment the expression index
  auto const expression_index = _expression_count++;
  // Visit children (operands) of this expression
  auto const operand_data_ref_indices = visit_operands(expr.get_operands());
  // Resolve operand types
  auto data_ref = [this](auto const& index) { return _data_references[index].data_type; };
  auto begin    = thrust::make_transform_iterator(operand_data_ref_indices.cbegin(), data_ref);
  auto end      = begin + operand_data_ref_indices.size();
  auto const operand_types = std::vector<cudf::data_type>(begin, end);

  // Validate type_id's of operand data references match
  // (decimal types with different scales are supported)
  if (std::adjacent_find(
        operand_types.cbegin(), operand_types.cend(), [](auto const& dt1, auto const& dt2) {
          return dt1.id() != dt2.id();
        }) != operand_types.cend()) {
    CUDF_FAIL("An AST expression was provided non-matching operand types.");
  }

  // For decimal128, only comparison operators are supported. All other binary operators
  // (e.g. ADD/SUB/MUL/DIV/MOD/PYMOD) return a decimal128 result whose 16-byte rep exceeds
  // the IntermediateDataType limit and has no GPU kernel support.
  auto const op = expr.get_operator();
  if (operand_types.size() == 2 && operand_types[0].id() == cudf::type_id::DECIMAL128 &&
      !cudf::ast::detail::is_comparison_operator(op)) {
    CUDF_FAIL("decimal128 is not supported for non-comparison binary AST operations.",
              cudf::data_type_error);
  }

  // Give back intermediate storage locations that are consumed by this operation
  std::for_each(
    operand_data_ref_indices.cbegin(),
    operand_data_ref_indices.cend(),
    [this](auto const& data_reference_index) {
      auto const operand_source = _data_references[data_reference_index];
      if (operand_source.reference_type == detail::device_data_reference_type::INTERMEDIATE) {
        auto const intermediate_index = operand_source.data_index;
        _intermediate_counter.give(intermediate_index);
      }
    });
  // Resolve expression type
  auto const data_type = cudf::ast::detail::ast_operator_return_type(op, operand_types);
  _operators.push_back(op);
  _operator_arities.push_back(cudf::ast::detail::ast_operator_arity(op));
  // Push data reference
  auto const output = [&]() {
    if (expression_index == 0) {
      // This expression is the root. Output should be directed to the output column.
      CUDF_EXPECTS(data_type.id() != cudf::type_id::DECIMAL128,
                   "decimal128 is not supported as an AST expression output.",
                   cudf::data_type_error);
      return detail::device_data_reference(
        detail::device_data_reference_type::COLUMN, data_type, 0, table_reference::OUTPUT);
    } else {
      // This expression is not the root. Output is an intermediate value.
      // Ensure that the output type is fixed width and fits in the intermediate storage.
      if (!cudf::is_fixed_width(data_type)) {
        CUDF_FAIL(
          "The output data type is not a fixed-width type but must be stored in an intermediate.");
      } else if (cudf::size_of(data_type) > (_has_nulls ? sizeof(IntermediateDataType<true>)
                                                        : sizeof(IntermediateDataType<false>))) {
        CUDF_FAIL("The output data type is too large to be stored in an intermediate.");
      }
      return detail::device_data_reference(
        detail::device_data_reference_type::INTERMEDIATE, data_type, _intermediate_counter.take());
    }
  }();
  auto const index = add_data_reference(output);
  // Insert source indices from all operands (sources) and this operator (destination)
  _operator_source_indices.insert(_operator_source_indices.end(),
                                  operand_data_ref_indices.cbegin(),
                                  operand_data_ref_indices.cend());
  _operator_source_indices.push_back(index);
  return index;
}

// TODO: Eliminate column name references from expression_parser because
// 2 code paths diverge in supporting column name references:
// 1. column name references are specific to cuIO
// 2. column name references are not supported in the libcudf table operations such as join,
// transform.
cudf::size_type expression_parser::visit(column_name_reference const& expr)
{
  CUDF_FAIL("Column name references are not supported in the AST expression parser.");
}

cudf::data_type expression_parser::output_type() const
{
  return _data_references.empty() ? cudf::data_type(cudf::type_id::EMPTY)
                                  : _data_references.back().data_type;
}

std::vector<cudf::size_type> expression_parser::visit_operands(
  cudf::host_span<std::reference_wrapper<expression const> const> operands)
{
  auto operand_data_reference_indices = std::vector<cudf::size_type>();
  for (auto const& operand : operands) {
    auto const operand_data_reference_index = operand.get().accept(*this);
    operand_data_reference_indices.push_back(operand_data_reference_index);
  }
  return operand_data_reference_indices;
}

cudf::size_type expression_parser::add_data_reference(detail::device_data_reference data_ref)
{
  // If an equivalent data reference already exists, return its index. Otherwise add this data
  // reference and return the new index.
  auto const it = std::find(_data_references.cbegin(), _data_references.cend(), data_ref);
  if (it != _data_references.cend()) {
    return std::distance(_data_references.cbegin(), it);
  } else {
    _data_references.push_back(data_ref);
    return _data_references.size() - 1;
  }
}

}  // namespace detail

}  // namespace ast

}  // namespace cudf
