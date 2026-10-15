/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/scalar_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/udf.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace CUDF_EXPORT cudf {

namespace ast {
class expression;
}

/** @brief A column or a broadcast scalar input to a JIT reduction. */
using reduce_input = std::variant<column_view, scalar_column_view>;

/** @brief Input schema for a reusable reduction, including decimal scales and dictionary child
 * types. */
struct reduce_input_spec {
  data_type type{};        ///< Logical input type
  bool is_scalar = false;  ///< Broadcast the input's single element
  std::vector<reduce_input_spec>
    children{};  ///< Dictionary children; string offset types are inferred at runtime

  bool operator==(reduce_input_spec const&) const = default;
};

/** @brief Type and independently specified length of a reduction result column. */
struct reduce_output {
  data_type type{};    ///< Fixed-width result type
  size_type size = 1;  ///< Result length, which can differ from other outputs
};

/** @brief Configuration of a reduction's lifecycle callbacks. */
struct reduce_options {
  null_aware is_null_aware = null_aware::NO;           ///< Pass optional row values to update
  error_policy errors      = error_policy::PROPAGATE;  ///< Throw, or nullify all result columns
  std::optional<void*> user_data = std::nullopt;  ///< Borrowed device data passed to all callbacks
};

/**
 * @brief Reusable JIT reduction with custom fixed-width accumulator fields.
 *
 * The CUDA source defines `cudf_reduce_init`, `cudf_reduce_update`, `cudf_reduce_merge`, and
 * `cudf_reduce_finalize`. For state types `S...`, input types `I...`, and output types `O...`,
 * their parameters are:
 *
 * @code{.cpp}
 * init(S*...);
 * update(cudf::size_type row, S*..., I...);
 * merge(S*..., S...);
 * finalize(cudf::size_type index, cuda::std::span<cudf::size_type const> lengths,
 *          cuda::std::optional<O>*..., S...);
 * @endcode
 *
 * When user data is supplied, `void*` is prepended to every signature. Null-aware update receives
 * `cuda::std::optional<I>` instead of `I`. Otherwise rows with any null input are skipped.
 * Finalize is called for each index below the largest output length. Output pointers are null
 * for indices outside that output's length. Active optional outputs start disengaged; assigning
 * a value makes that output element valid. All output lengths may be zero.
 *
 * Init must populate every field with an identity. Merge must be associative and commutative,
 * and compatible with update on independently accumulated rows. Callbacks must not depend on
 * how often they are invoked or on the partitioning of rows. Reduction order is unspecified.
 * Empty and fully skipped inputs are finalized from the identity. Floating-point results may
 * differ from sequential or other parallel reductions.
 *
 * CUDA callbacks can return void, int, or cudf::errc. LTO callbacks must be exported with
 * `extern "C" __device__` linkage and return int. Zero means success; other return values must
 * represent cudf::errc. LTO signatures use the same concrete types as CUDA source, including
 * cudf decimal/chrono wrappers. State and result types are fixed-width; input types follow
 * transform's fixed-width, string, and dictionary support. Dictionary inputs pass decoded keys.
 * Decimal output values must use
 * the scale declared in the output schema.
 *
 * Under PROPAGATE, errors throw cudf::evaluation_error. Under NULLIFY, all elements of all result
 * columns are invalid. Compilation, linking, argument and CUDA failures always throw. Subsequent
 * runs are independent of earlier failures. Programs retain borrowed user data but not input
 * columns. Kernel compilation/linking occurs during construction; output lengths are runtime
 * arguments and can change between runs.
 */
struct reduce_program {
 private:
  struct impl;
  std::unique_ptr<impl> impl_;

 public:
  /**
   * @brief Construct from CUDA source and explicit input schemas.
   *
   * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
   * @param inputs Input views or schemas in update-parameter order
   * @param state_types Fixed-width accumulator field types in callback-parameter order
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param options Null handling, error handling, and optional borrowed device user data
   */
  reduce_program(std::string const& udf,
                 std::span<reduce_input_spec const> inputs,
                 std::span<data_type const> state_types,
                 std::span<reduce_output const> outputs,
                 reduce_options options = {});

  /**
   * @brief Construct from CUDA source and concrete inputs, retaining only their schemas.
   *
   * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
   * @param inputs Input views or schemas in update-parameter order
   * @param state_types Fixed-width accumulator field types in callback-parameter order
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param options Null handling, error handling, and optional borrowed device user data
   */
  reduce_program(std::string const& udf,
                 std::span<reduce_input const> inputs,
                 std::span<data_type const> state_types,
                 std::span<reduce_output const> outputs,
                 reduce_options options = {});

  /**
   * @brief Construct from LTO IR or an LTO-capable fatbinary and explicit schemas.
   *
   * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
   * @param binary_type Format of the supplied device binary
   * @param inputs Input views or schemas in update-parameter order
   * @param state_types Fixed-width accumulator field types in callback-parameter order
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param options Null handling, error handling, and optional borrowed device user data
   */
  reduce_program(std::span<uint8_t const> udf,
                 lto_binary_type binary_type,
                 std::span<reduce_input_spec const> inputs,
                 std::span<data_type const> state_types,
                 std::span<reduce_output const> outputs,
                 reduce_options options = {});

  /**
   * @brief Construct from LTO IR or an LTO-capable fatbinary and concrete inputs.
   *
   * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
   * @param binary_type Format of the supplied device binary
   * @param inputs Input views or schemas in update-parameter order
   * @param state_types Fixed-width accumulator field types in callback-parameter order
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param options Null handling, error handling, and optional borrowed device user data
   */
  reduce_program(std::span<uint8_t const> udf,
                 lto_binary_type binary_type,
                 std::span<reduce_input const> inputs,
                 std::span<data_type const> state_types,
                 std::span<reduce_output const> outputs,
                 reduce_options options = {});

  /**
   * @brief Construct from CUDA lifecycle source and AST expressions mapped into update's inputs.
   *
   * Row IR generates expression evaluation inside the reduction kernel. Literal data is owned
   * by the program and column references are rebound by run(table). Expression-level NULLIFY
   * produces a null mapped value; propagated expression errors follow options.errors. Null
   * skipping is applied to mapped values, rather than to underlying table columns.
   *
   * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
   * @param state_types Fixed-width accumulator field types in callback-parameter order
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param options Null handling, error handling, and optional borrowed device user data
   * @param table Table providing referenced columns and the row count
   * @param expressions AST expressions in update-input order
   * @param stream CUDA stream for device operations
   * @param mr Device resource for allocations
   */
  reduce_program(std::string const& udf,
                 table_view const& table,
                 std::span<std::reference_wrapper<ast::expression const> const> expressions,
                 std::span<data_type const> state_types,
                 std::span<reduce_output const> outputs,
                 reduce_options options            = {},
                 cuda::stream_ref stream           = cudf::get_default_stream(),
                 rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct from LTO lifecycle functions with AST mapping embedded in the kernel.
   *
   * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
   * @param binary_type Format of the supplied device binary
   * @param state_types Fixed-width accumulator field types in callback-parameter order
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param options Null handling, error handling, and optional borrowed device user data
   * @param table Table providing referenced columns and the row count
   * @param expressions AST expressions in update-input order
   * @param stream CUDA stream for device operations
   * @param mr Device resource for allocations
   */
  reduce_program(std::span<uint8_t const> udf,
                 lto_binary_type binary_type,
                 table_view const& table,
                 std::span<std::reference_wrapper<ast::expression const> const> expressions,
                 std::span<data_type const> state_types,
                 std::span<reduce_output const> outputs,
                 reduce_options options            = {},
                 cuda::stream_ref stream           = cudf::get_default_stream(),
                 rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  reduce_program(reduce_program&&);
  reduce_program& operator=(reduce_program&&);
  reduce_program(reduce_program const&)            = delete;
  reduce_program& operator=(reduce_program const&) = delete;
  ~reduce_program();

  /**
   * @brief Run with new inputs and optional replacement output lengths.
   *
   * An empty outputs span uses the constructor's outputs. Non-scalar inputs must have equal
   * lengths matching row_size when provided. If there are no non-scalar inputs, row_size is
   * required. Types, scalar flags, and dictionary child types must match the construction schemas.
   *
   * @param inputs Input views or schemas in update-parameter order
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param stream CUDA stream for device operations
   * @param mr Device resource for allocations
   * @param row_size Explicit row count, required when no input is a non-scalar column
   * @return Result columns in the specified output order
   */
  std::vector<std::unique_ptr<column>> run(
    std::span<reduce_input const> inputs,
    std::span<reduce_output const> outputs = {},
    std::optional<size_type> row_size      = std::nullopt,
    cuda::stream_ref stream                = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

  /**
   * @brief Run an AST program, rebinding its referenced columns to a compatible table.
   *
   * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
   * @param table Table providing referenced columns and the row count
   * @param stream CUDA stream for device operations
   * @param mr Device resource for allocations
   * @return Result columns in the specified output order
   */
  std::vector<std::unique_ptr<column>> run(
    table_view const& table,
    std::span<reduce_output const> outputs = {},
    cuda::stream_ref stream                = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());
};

/**
 * @brief Reduce rows using CUDA lifecycle UDFs. @see reduce_program for the callback contract.
 *
 * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
 * @param inputs Input views or schemas in update-parameter order
 * @param state_types Fixed-width accumulator field types in callback-parameter order
 * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
 * @param options Null handling, error handling, and optional borrowed device user data
 * @param stream CUDA stream for device operations
 * @param mr Device resource for allocations
 * @param row_size Explicit row count, required when no input is a non-scalar column
 * @return Result columns in the specified output order
 */
std::vector<std::unique_ptr<column>> reduce(
  std::string const& udf,
  std::span<reduce_input const> inputs,
  std::span<data_type const> state_types,
  std::span<reduce_output const> outputs,
  reduce_options options            = {},
  std::optional<size_type> row_size = std::nullopt,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reduce rows using LTO lifecycle UDFs. @see reduce_program for the callback contract.
 *
 * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
 * @param binary_type Format of the supplied device binary
 * @param inputs Input views or schemas in update-parameter order
 * @param state_types Fixed-width accumulator field types in callback-parameter order
 * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
 * @param options Null handling, error handling, and optional borrowed device user data
 * @param stream CUDA stream for device operations
 * @param mr Device resource for allocations
 * @param row_size Explicit row count, required when no input is a non-scalar column
 * @return Result columns in the specified output order
 */
std::vector<std::unique_ptr<column>> reduce_lto(
  std::span<uint8_t const> udf,
  lto_binary_type binary_type,
  std::span<reduce_input const> inputs,
  std::span<data_type const> state_types,
  std::span<reduce_output const> outputs,
  reduce_options options            = {},
  std::optional<size_type> row_size = std::nullopt,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Map AST expressions through Row IR and reduce their values using CUDA lifecycle UDFs.
 *
 * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
 * @param state_types Fixed-width accumulator field types in callback-parameter order
 * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
 * @param options Null handling, error handling, and optional borrowed device user data
 * @param table Table providing referenced columns and the row count
 * @param expressions AST expressions in update-input order
 * @param stream CUDA stream for device operations
 * @param mr Device resource for allocations
 * @return Result columns in the specified output order
 */
std::vector<std::unique_ptr<column>> reduce(
  std::string const& udf,
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  std::span<data_type const> state_types,
  std::span<reduce_output const> outputs,
  reduce_options options            = {},
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Map AST expressions through Row IR and reduce their values using LTO lifecycle UDFs.
 *
 * @param udf CUDA lifecycle source, or a device binary containing the lifecycle functions
 * @param binary_type Format of the supplied device binary
 * @param state_types Fixed-width accumulator field types in callback-parameter order
 * @param outputs Fixed-width output types and independent lengths in finalize-parameter order
 * @param options Null handling, error handling, and optional borrowed device user data
 * @param table Table providing referenced columns and the row count
 * @param expressions AST expressions in update-input order
 * @param stream CUDA stream for device operations
 * @param mr Device resource for allocations
 * @return Result columns in the specified output order
 */
std::vector<std::unique_ptr<column>> reduce_lto(
  std::span<uint8_t const> udf,
  lto_binary_type binary_type,
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  std::span<data_type const> state_types,
  std::span<reduce_output const> outputs,
  reduce_options options            = {},
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace CUDF_EXPORT cudf
