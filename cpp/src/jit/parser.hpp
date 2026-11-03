/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace jit {

struct ptx_param {
  std::string register_type;
  std::string identifier;
};

/**
 * @brief Parse and transform a piece of PTX code that contains the implementation
 * of a `__device__` function into a CUDA `__device__` `__inline__` function.
 *
 * @param `src` The input PTX code.
 * @param `function_name` The User defined function that the output CUDA function
 * will have.
 * @param `param_types` The types of the parameters for the the kernel. This will be used to specify
 * the CUDA function declaration for the wrapped PTX code.
 * @return The output CUDA `__device__` `__inline__` function
 */
class ptx_parser {
 private:
  std::string ptx;

  std::string function_name;

  std::map<unsigned int, std::string> param_types;

 private:
  /**
   * @brief parse the PTX function header
   *
   * @param src The header part of the PTX code
   * @return The parsed CUDA header
   */
  std::string parse_function_header(std::string const& src);

  /**
   * @brief parse the parameter list of the PTX code
   *
   * @param src The input parameter list part of the PTX code
   * @return The parsed CUDA parameter list
   */
  static std::vector<ptx_param> parse_param_list(std::string const& src);

  /**
   * @brief parse and transform an input parameter line of the PTX code into the
   * corresponding CUDA form
   *
   * @param param_decl The input parameter line of the PTX code
   * @return The parsed CUDA input parameter
   */
  static ptx_param parse_param(std::string const& param_decl);

  /**
   * @brief Create a CUDA function header to wrap a parsed PTX.
   *
   *
   * The result always has `__device__ __inline__ void`. The types of the
   * parameters are determined from, in descending order of priority:
   *  1. The types provided in the `param_types` map
   *  2. The register types of the parameters found in the PTX function header. i.e. register types
   * `s64` and `b64` would map to `long int`, and `s32` and `b32` would map to `int`
   *
   * @param function_name the function name to use in the CUDA code
   * @param ptx_params the parsed info of each param found in the PTX
   * @param param_types the CUDA source code types to use in place of the ptx parameter register
   * type
   */
  static std::string to_cuda_function_header(
    std::string const& function_name,
    std::vector<ptx_param> const& ptx_params,
    std::map<unsigned int, std::string> const& param_types);

  /**
   * @brief parse function body of the PTX code into statements by `;`s.
   *
   * @param src The function body of the PTX code
   * @return The parsed statements
   */
  static std::vector<std::string> parse_function_body(std::string const& src);

  /**
   * @brief Remove leading white characters and call `parse_instruction`.
   *
   * @param src The statement to be parsed.
   * @return The resulting CUDA statement.
   */
  static std::string parse_statement(std::string const& src);

  /**
   * @brief Convert the input PTX instruction into an inline PTX
   * statement without changing (exceptions exist).
   *
   * Non-alphanumeric characters in register identifiers, except underscores, are replaced with
   underscore. Example:
   *
   *  fma.rn.f32 	%f4, %f3, %f1, %f2
   *
   *    ---> asm volatile ("  fma.rn.f32 _f4, _f3, _f1, _f2;");
   *
   * If a register from the input parameters list is used in an instruction
   * its type is inferred from the instruction and saved in the `input_arg_list`
   * to be used in when parsing the function header.
   *
   * See the document at https://github.com/hummingtree/cudf/wiki/PTX-parser
   * for the detailed description about the exceptions.
   *
   * @param src The statement to be parsed.
   * @return The resulting CUDA inline PTX statement.
   */
  static std::string parse_instruction(std::string const& src);

  /**
   * @brief Convert register type (e.g. ".f32") to the corresponding
   * C++ type (e.g. "float")
   *
   * See the implementation for details
   *
   * @param register_type Registered type string
   * @return The resulting code
   */
  static std::string register_type_to_cpp_type(std::string const& register_type);

  /**
   * @brief Convert register type (e.g. ".f32") to the corresponding
   * constraint in inline PTX syntax (e.g. "f")
   *
   * See the implementation for details
   *
   * @param src The input code
   * @return The resulting code
   */
  static std::string register_type_to_contraint(std::string const& src);

  /**
   * @brief Replace any non-alphanumeric characters that are not underscore with
   * underscore. The leading `[` and trailing `]` are exempted, e.g.
   *
   *  "[t$5]" --> "[t_5]"
   *
   * @param src The input code
   * @return The resulting code
   */
  static std::string remove_nonalphanumeric(std::string const& src);

  /**
   * @brief Replace leading `%` in register identifiers with `_`.
   *
   * According to PTX document `%` can only appear at the start of a register
   * identifier. At the same time `%` is not allowed in inline PTX. This function
   * first looks for the register identifier and if it starts with `%` replaces it
   * with `_`.
   *
   * @param src The input code
   * @return The resulting code
   */
  static std::string escape_percent(std::string const& src);

 public:
  ptx_parser() = delete;

  /**
   * @brief Constructor of the `ptx_parser` class
   *
   * @param ptx_ The input PTX code that contains the function whose
   * CUDA is to be generated.
   * @param function_name_ The function name of the output CUDA function
   * @param param_types the types of the parameters of the function. each entry should map the
   * parameter index to the CUDA type of the parameter. The unspecified indices will be have types
   * based on the PTX's parameter register types.
   */
  ptx_parser(std::string ptx_,
             std::string function_name_,
             std::map<unsigned int, std::string> param_types);

  // parse the source!!!
  std::string parse();
};

/**
 * @brief Parse and Transform a piece of PTX code that contains the implementation
 * of a device function into a CUDA device function.
 *
 * @param src The input PTX code.
 * @param function_name The User defined function that the output CUDA function
 * will have.
 * @param param_types the types of the parameters of the function. each entry should map the
 * parameter index to the CUDA type of the parameter. The unspecified indices will be have types
 * based on the PTX's parameter types.
 * @return The output CUDA device function
 */
std::string parse_single_function_ptx(std::string const& src,
                                      std::string const& function_name,
                                      std::map<unsigned int, std::string> param_types);

/**
 * @brief In a piece of CUDA code that contains the implementation
 * of a device function, locate the function and replace its function name
 * with the specified one.
 *
 * @param src The input CUDA code.
 * @param function_name The User defined function that the output CUDA function
 * will have.
 * @return The output CUDA device function
 */
std::string parse_single_function_cuda(std::string const& src, std::string const& function_name);

}  // namespace jit
}  // namespace CUDF_EXPORT cudf
