/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

/**
 * @file
 * @brief Column APIs for transforming rows
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup transformation_transform
 * @{
 */

/**
 * @brief Typedef for inputs to the transform function. Each input can be either a column or a
 * scalar column.
 */
using transform_input = std::variant<column_view, scalar_column_view>;

/**
 * @brief Specification for the outputs of the transform function. This includes the output type and
 * nullability policy for each output column.
 *
 */
struct transform_output {
  data_type type = data_type{type_id::EMPTY};  ///< The output type of the column to be created

  output_nullability nullability =
    output_nullability::PRESERVE;  ///< Signifies if a null mask should be created for the output
                                   ///< column
};

/**
 * @brief Creates a new column by applying a transform function against every element of the input
 * columns.
 *
 * @deprecated in release 26.10. Use `transform` instead.
 *
 * @param inputs Immutable views of the inputs to transform
 * @param udf The PTX/CUDA string of the transform function to apply
 * @param output_type The output type that is compatible with the output type in the UDF
 * @param source_type The source type of the UDF (CUDA or PTX)
 * @param user_data User-defined device data to pass to the UDF
 * @param is_null_aware Signifies the UDF will receive row inputs as optional values
 * @param row_size The row size of the transform operation
 * @param null_policy Signifies if a null mask should be created for the output column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The column resulting from applying the transform function
 */
[[deprecated("Use transform instead")]] std::unique_ptr<column> transform_extended(
  std::span<transform_input const> inputs,
  std::string const& udf,
  data_type output_type,
  udf_source_type source_type,
  std::optional<void*> user_data    = std::nullopt,
  null_aware is_null_aware          = null_aware::NO,
  std::optional<size_type> row_size = std::nullopt,
  output_nullability null_policy    = output_nullability::PRESERVE,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Describes a transform input independently of a particular column.
 *
 * An input specification contains the type information needed to reflect and retrieve a transform
 * kernel. Dictionary specifications recursively describe their indices and keys through `children`.
 * String specifications retain their offsets child type so `INT32` and `INT64` layouts can be
 * distinguished.
 */
struct transform_input_spec {
  type_id type = type_id::EMPTY;  ///< Logical type of the input

  bool is_scalar = false;  ///< Whether the input is presented to the UDF as a scalar

  std::vector<transform_input_spec> children =
    {};  ///< Specifications of dictionary children or string offsets
};

/**
 * @brief Describes a transform output independently of a particular output column.
 *
 * The string-offset setting identifies the device-view representation required by the kernel. The
 * nullability setting is retained so inputs to `transform_program::run` can be validated against
 * the output policy used to construct the program.
 */
struct transform_output_spec {
  type_id type = type_id::EMPTY;  ///< Logical type of the output

  output_nullability nullability =
    output_nullability::PRESERVE;  ///< Null-mask policy for the output

  bool has_string_offsets = false;  ///< Whether a string output uses preallocated offsets
  std::vector<transform_output_spec> children =
    {};  ///< Specifications of string offsets or nested child columns
};

/**
 * @brief A reusable transform program that retains a JIT-compiled kernel.
 *
 * Construction retrieves the kernel for the UDF and the supplied input and output specifications.
 * Subsequent calls to `run` reuse that kernel.
 * Runtime inputs and outputs must be compatible with the specifications used at construction.
 *
 */
struct transform_program {
 private:
  struct impl;

  std::unique_ptr<impl> impl_;  ///< The implementation of the transform program

 public:
  /**
   * @brief Constructs a reusable program by deriving specifications from transform arguments.
   *
   * The UDF kernel is retrieved during construction and retained for subsequent calls to `run`.
   * The input and output objects are inspected only to derive their specifications and are not
   * retained.
   *
   * @param udf The PTX or CUDA source for the transform UDF
   * @param source_type The source type of `udf`
   * @param is_null_aware Whether the UDF receives row inputs as optional values
   * @param user_data User-defined device data, not owned by the program, retained and passed to the
   * UDF by `run`
   * @param inputs Inputs from which to derive the input specifications
   * @param outputs Outputs from which to derive the output type and nullability specifications
   * @param string_offsets Optional string offsets used to determine each string output
   * representation
   */
  transform_program(std::string const& udf,
                    udf_source_type source_type,
                    null_aware is_null_aware,
                    std::optional<void*> user_data,
                    std::span<transform_input const> inputs,
                    std::span<transform_output const> outputs,
                    std::span<std::unique_ptr<column> const> string_offsets);

  /**
   * @brief Constructs a reusable program from explicit input and output specifications.
   *
   * This overload enables composition without requiring concrete columns when the program is
   * created. The UDF kernel is retrieved during construction and retained for subsequent calls to
   * `run`.
   *
   * @param udf The PTX or CUDA source for the transform UDF
   * @param source_type The source type of `udf`
   * @param is_null_aware Whether the UDF receives row inputs as optional values
   * @param user_data User-defined device data, not owned by the program, retained and passed to the
   * UDF by `run`
   * @param inputs Specifications of the transform inputs
   * @param outputs Specifications of the transform outputs
   */
  transform_program(std::string const& udf,
                    udf_source_type source_type,
                    null_aware is_null_aware,
                    std::optional<void*> user_data,
                    std::span<transform_input_spec const> inputs,
                    std::span<transform_output_spec const> outputs);

  /**
   * @brief Constructs a reusable program for an AST expression.
   *
   * The expression is lowered and its kernel is retrieved during construction. Literal values are
   * retained by the program, while column inputs are rebound to the table passed to `run`.
   *
   * @param table A table whose schema is used to lower the expression and retrieve its kernel
   * @param expressions The AST expressions to lower and construct the program from
   * @param stream CUDA stream used for device memory operations during construction
   * @param mr Device memory resource used for device memory allocations during construction
   */
  transform_program(table_view const& table,
                    std::span<std::reference_wrapper<ast::expression const> const> expressions,
                    cuda::stream_ref stream           = cudf::get_default_stream(),
                    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Move constructor for transform_program
   * @param other The transform_program to move from
   */
  transform_program(transform_program&& other);

  /**
   * @brief Move assignment operator for transform_program
   * @param other The transform_program to move from
   * @return A reference to the current transform_program
   */
  transform_program& operator=(transform_program&& other);

  transform_program(transform_program const&)            = delete;  ///< Deleted copy constructor
  transform_program& operator=(transform_program const&) = delete;  ///< Deleted copy assignment
  ~transform_program();                                             ///< Destructor

  /**
   * @brief Runs the transform program on the given inputs and outputs.
   *
   * @throws std::invalid_argument if the inputs, outputs, or string offsets are not compatible with
   * the specifications used to construct the program
   *
   * @param inputs The inputs to the transform program
   * @param outputs The outputs of the transform program
   * @param string_offsets For string output columns, the offsets can be pre-allocated and passed in
   * to prevent overhead of compacting string views into run-end strings column.
   * @param row_size The row size of the transform operation. If not provided, it will be inferred
   * from the inputs.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return A table containing the columns resulting from applying the transform function to every
   * element of the input according to the output specifications
   */
  std::unique_ptr<table> run(
    std::span<transform_input const> inputs,
    std::span<transform_output const> outputs,
    std::vector<std::unique_ptr<column>>&& string_offsets,
    std::optional<size_type> row_size,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Evaluates the AST expressions used to construct this program on a table.
   *
   * @throws std::invalid_argument if the table is not compatible with the specifications used to
   * construct the program
   *
   * @param table The table used for expression evaluation
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column device memory
   * @return The table resulting from evaluating the expression
   */
  std::unique_ptr<table> run(
    table_view const& table,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
};

/**
 * @brief Creates a new table by applying a transform function against every
 * element of the input columns.
 *
 * Computes:
 * `(outputs[i]...) =  UDF(inputs[i]...)`.
 *
 * @throws std::invalid_argument if any of the input columns have different sizes (except scalars)
 * @throws std::invalid_argument if `output_type` or any of the inputs are not fixed-width or string
 * types
 * @throws std::invalid_argument if the inputs only have a scalar with no column inputs and
 * `row_size` is not provided. This is because the row size cannot be inferred from the inputs in
 * this case.
 * @throws std::invalid_argument if any of the output or input types are not supported.
 * CUDA-supported input types are fixed-width, string, and their dictionary types. PTX-supported
 * input types are integrals, floats, and their dictionary types. CUDA-supported output types are
 * fixed-width, string, and their dictionary types. PTX-supported output types are integrals,
 * floats, and their dictionary types.
 * @throws std::invalid_argument if string offsets are provided for non-string output columns, or
 * if the number of string offsets does not match the number of output columns.
 * @throws cudf::evaluation_error if the UDF produces an error during execution.
 *
 * The size of the resulting column is the `row_size` if provided, otherwise it is inferred from
 * the input and pre-allocated output columns.
 *
 * @param udf The PTX/CUDA string of the transform function to apply
 * @param source_type   The source type of the UDF (CUDA or PTX)
 * @param is_null_aware Signifies the UDF will receive row inputs as optional values
 * @param user_data     User-defined device data to pass to the UDF.
 * @param inputs        Immutable views of the inputs to transform (columns and scalar columns)
 * @param outputs       Specification of the output columns to be created
 * @param string_offsets For string output columns, the offsets can be pre-allocated and passed in
 * to prevent overhead of compacting string views into run-end strings column.
 * @param row_size The row size of the transform operation. If not provided, it is inferred from the
 * input columns.
 * @param stream        CUDA stream used for device memory operations and kernel launches
 * @param mr            Device memory resource used to allocate the returned column's device memory
 * @return              A table containing the columns resulting from applying the transform
 * function to every element of the input according to the output specifications
 *
 */
std::unique_ptr<table> transform(
  std::string const& udf,
  udf_source_type source_type,
  null_aware is_null_aware,
  std::optional<void*> user_data,
  std::span<transform_input const> inputs,
  std::span<transform_output const> outputs,
  std::vector<std::unique_ptr<column>>&& string_offsets,
  std::optional<size_type> row_size,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a new table by applying a transform function against every element of the input
 * columns.
 *
 * @deprecated in release 26.10. Use `transform` instead.
 *
 * @param udf The PTX/CUDA string of the transform function to apply
 * @param source_type The source type of the UDF (CUDA or PTX)
 * @param is_null_aware Signifies the UDF will receive row inputs as optional values
 * @param user_data User-defined device data to pass to the UDF
 * @param inputs Immutable views of the inputs to transform (columns and scalar columns)
 * @param outputs Specification of the output columns to be created
 * @param string_offsets Pre-allocated offsets for string output columns
 * @param row_size The row size of the transform operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A table containing the transformed output columns
 */
[[deprecated("Use transform instead")]] std::unique_ptr<table> multi_transform(
  std::string const& udf,
  udf_source_type source_type,
  null_aware is_null_aware,
  std::optional<void*> user_data,
  std::span<transform_input const> inputs,
  std::span<transform_output const> outputs,
  std::vector<std::unique_ptr<column>>&& string_offsets,
  std::optional<size_type> row_size,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief The type of LTO Binary
 */
enum class lto_binary_type : uint8_t {
  LTO_IR,  //< LTO-IR binary
  FATBIN   //< FATBIN binary
};

/**
 * @brief Creates a new table by applying a transform function against every
 * element of the input columns.
 *
 * Computes:
 * `(outputs[i]...) =  UDF(inputs[i]...)`.
 *
 *
 * @throws std::invalid_argument if any of the input columns have different sizes (except scalars)
 * @throws std::invalid_argument if `output_type` or any of the inputs are not fixed-width or string
 * types
 * @throws std::invalid_argument if the inputs only have a scalar with no column inputs and
 * `row_size` is not provided. This is because the row size cannot be inferred from the inputs in
 * this case
 * @throws std::invalid_argument if string offsets are provided for non-string output columns, or
 * if the number of string offsets does not match the number of output columns
 * @throws cudf::evaluation_error if the UDF produces an error during execution
 *
 * The size of the resulting column is the `row_size` if provided, otherwise it is inferred from
 * the input and pre-allocated output columns.
 *
 * @param udf           The LTO-IR fragment containing the transform function to apply. The UDF must
 * be named `transform` and follow the CUDF UDF ABI
 * @param binary_type   The type of the LTO binary provided in `udf`
 * @param is_null_aware Signifies the UDF will receive row inputs as optional values
 * @param user_data     User-defined device data to pass to the UDF
 * @param inputs        Immutable views of the inputs to transform
 * @param outputs       Specification of the output columns to be created
 * @param string_offsets For string output columns, the offsets can be pre-allocated and passed in
 * to prevent overhead of compacting string views into run-end strings column.
 * @param row_size The row size of the transform operation. If not provided, it is inferred from the
 * input columns
 * @param stream        CUDA stream used for device memory operations and kernel launches
 * @param mr            Device memory resource used to allocate the returned column's device memory
 * @return              A table containing the columns resulting from applying the transform
 * function to every element of the input according to the output specifications
 *
 */
std::unique_ptr<table> transform_lto(
  std::span<uint8_t const> udf,
  lto_binary_type binary_type,
  null_aware is_null_aware,
  std::optional<void*> user_data,
  std::span<transform_input const> inputs,
  std::span<transform_output const> outputs,
  std::vector<std::unique_ptr<column>>&& string_offsets,
  std::optional<size_type> row_size,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a null_mask from `input` by converting `NaN` to null and
 * preserving existing null values and also returns new null_count.
 *
 * @deprecated in release 26.04. Use column_nans_to_nulls instead.
 *
 * @throws cudf::logic_error if `input.type()` is a non-floating type
 *
 * @param input  An immutable view of the input column of floating-point type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr     Device memory resource used to allocate the returned bitmask
 * @return A pair containing a `device_buffer` with the new bitmask and its
 * null count obtained by replacing `NaN` in `input` with null.
 */
[[deprecated]] std::pair<std::unique_ptr<rmm::device_buffer>, size_type> nans_to_nulls(
  column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a null_mask from `input` by converting `NaN` elements to null rows
 * and preserving existing null values
 *
 * @throws cudf::logic_error if `input.type()` is a non-floating type
 *
 * @param input  An immutable view of the input column of floating-point type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr     Device memory resource used to allocate the returned bitmask
 * @return       A new column with the null mask created from the input column
 */
std::unique_ptr<column> column_nans_to_nulls(
  column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Compute a new column by evaluating an expression tree on a table.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @throws cudf::logic_error if passed an expression operating on table_reference::RIGHT.
 * @throws cudf::data_type_error if the expression applies a non-comparison binary operator to
 * decimal128 operands.
 * @throws cudf::evaluation_error if the evaluation of the expression results in an error during
 * execution.
 *
 * @param table The table used for expression evaluation
 * @param expr The root of the expression tree
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource
 * @return Output column
 */
std::unique_ptr<column> compute_column(
  table_view const& table,
  ast::expression const& expr,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Compute a new column by evaluating an expression tree on a table using a JIT-compiled
 * kernel.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @throws cudf::logic_error if passed an expression operating on table_reference::RIGHT.
 * @throws cudf::evaluation_error if the evaluation of the expression results in an error during
 * execution.
 *
 * @param table The table used for expression evaluation
 * @param expr The root of the expression tree
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource
 * @return Output column
 */
std::unique_ptr<column> compute_column_jit(
  table_view const& table,
  ast::expression const& expr,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Compute a new table by evaluating expression trees on a table using a JIT-compiled
 * kernel.
 *
 * This evaluates expressions over a table to produce a new table. Also called an n-ary
 * transform. Expressions are evaluated in the order supplied, and output column `i` contains the
 * result of `expressions[i]`. Common subexpressions shared by multiple outputs are evaluated once
 * by the generated function.
 *
 * @pre `expressions` must not be empty.
 *
 * @throws cudf::logic_error if passed an empty collection of expressions.
 * @throws cudf::logic_error if passed an expression operating on table_reference::RIGHT.
 * @throws cudf::evaluation_error if the evaluation of the expression results in an error during
 * execution.
 *
 * @param table The table used for expression evaluation
 * @param expressions Non-empty collection of expression-tree roots, one per output column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource
 * @return Table containing one output column per expression, in the same order as `expressions`
 */
std::unique_ptr<table> compute_table_jit(
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a bitmask from a column of boolean elements.
 *
 * If element `i` in `input` is `true`, bit `i` in the resulting mask is set (`1`). Else,
 * if element `i` is `false` or null, bit `i` is unset (`0`).
 *
 *
 * @throws cudf::logic_error if `input.type()` is a non-boolean type
 *
 * @param input  Boolean elements to convert to a bitmask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr     Device memory resource used to allocate the returned bitmask
 * @return A pair containing a `device_buffer` with the new bitmask and its
 * null count obtained from input considering `true` represent `valid`/`1` and
 * `false` represent `invalid`/`0`.
 */
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Encode the rows of the given table as integers
 *
 * The encoded values are integers in the range [0, n), where `n`
 * is the number of distinct rows in the input table.
 * The result table is such that `keys[result[i]] == input[i]`,
 * where `keys` is a table containing the distinct rows  in `input` in
 * sorted ascending order. Nulls, if any, are sorted to the end of
 * the `keys` table.
 *
 * Examples:
 * @code{.pseudo}
 * input: [{'a', 'b', 'b', 'a'}]
 * output: [{'a', 'b'}], {0, 1, 1, 0}
 *
 * input: [{1, 3, 1, 2, 9}, {1, 2, 1, 3, 5}]
 * output: [{1, 2, 3, 9}, {1, 3, 2, 5}], {0, 2, 0, 1, 3}
 * @endcode
 *
 * @param input Table containing values to be encoded
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return A pair containing the distinct row of the input table in sorter order,
 * and a column of integer indices representing the encoded rows.
 */
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Encodes `input` by generating a new column for each value in `categories` indicating the
 * presence of that value in `input`.
 *
 * The resulting per-category columns are returned concatenated as a single column viewed by a
 * `table_view`.
 *
 * The `i`th row of the `j`th column in the output table equals 1
 * if `input[i] == categories[j]`, and 0 otherwise.
 *
 * The `i`th row of the `j`th column in the output table equals 1
 * if input[i] == categories[j], and 0 otherwise.
 *
 * Examples:
 * @code{.pseudo}
 * input: [{'a', 'c', null, 'c', 'b'}]
 * categories: ['c', null]
 * output: [{0, 1, 0, 1, 0}, {0, 0, 1, 0, 0}]
 * @endcode
 *
 * @throws cudf::logic_error if input and categories are of different types.
 *
 * @param input Column containing values to be encoded
 * @param categories Column containing categories
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return A pair containing the owner to all encoded data and a table view into the data
 */
std::pair<std::unique_ptr<column>, table_view> one_hot_encode(
  column_view const& input,
  column_view const& categories,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a boolean column from given bitmask.
 *
 * Returns a `bool` for each bit in `[begin_bit, end_bit)`. If bit `i` in least-significant bit
 * numbering is set (1), then element `i` in the output is `true`, otherwise `false`.
 *
 * @throws cudf::logic_error if `bitmask` is null and end_bit-begin_bit > 0
 * @throws cudf::logic_error if begin_bit > end_bit
 *
 * Examples:
 * @code{.pseudo}
 * input: {0b10101010}
 * output: [{false, true, false, true, false, true, false, true}]
 * @endcode
 *
 * @param bitmask A device pointer to the bitmask which needs to be converted
 * @param begin_bit position of the bit from which the conversion should start
 * @param end_bit position of the bit before which the conversion should stop
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return A boolean column representing the given mask from [begin_bit, end_bit)
 */
std::unique_ptr<column> mask_to_bools(
  bitmask_type const* bitmask,
  size_type begin_bit,
  size_type end_bit,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an approximate cumulative size in bits of all columns in the `table_view` for
 * each row.
 *
 * This function counts bits instead of bytes to account for the null mask which only has one
 * bit per row.
 *
 * Each row in the returned column is the sum of the per-row size for each column in
 * the table.
 *
 * In some cases, this is an inexact approximation. Specifically, columns of lists and strings
 * require N+1 offsets to represent N rows. It is up to the caller to calculate the small
 * additional overhead of the terminating offset for any group of rows being considered.
 *
 * This function returns the per-row sizes as the columns are currently formed. This can
 * end up being larger than the number you would get by gathering the rows. Specifically,
 * the push-down of struct column validity masks can nullify rows that contain data for
 * string or list columns. In these cases, the size returned is conservative:
 *
 * row_bit_count(column(x)) >= row_bit_count(gather(column(x)))
 *
 * @param t The table view to perform the computation on
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return A 32-bit integer column containing the per-row bit counts
 */
std::unique_ptr<column> row_bit_count(
  table_view const& t,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an approximate cumulative size in bits of all columns in the `table_view` for
 * each segment of rows.
 *
 * This is similar to counting bit size per row for the input table in `cudf::row_bit_count`,
 * except that row sizes are accumulated by segments.
 *
 * Currently, only fixed-length segments are supported. In case the input table has number of rows
 * not divisible by `segment_length`, its last segment is considered as shorter than the others.
 *
 * @throw std::invalid_argument if the input `segment_length` is non-positive or larger than the
 * number of rows in the input table.
 *
 * @param t The table view to perform the computation on
 * @param segment_length The number of rows in each segment for which the total size is computed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return A 32-bit integer column containing the bit counts for each segment of rows
 */
std::unique_ptr<column> segmented_row_bit_count(
  table_view const& t,
  size_type segment_length,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
