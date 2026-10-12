/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parser.hpp"

#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <format>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace jit {
namespace {

inline bool is_white(char const c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

std::string remove_comments(std::string const& src)
{
  std::string output;
  auto f = src.cbegin();
  while (f < src.cend()) {
    auto l = std::find(f, src.cend(), '/');
    output.append(f, l);  // push chunk instead of 1 char at a time
    f = std::next(l);     // skip over '/'
    if (l < src.cend()) {
      char const n = f < src.cend() ? *f : '?';
      if (n == '/') {                        // found "//"
        f = std::find(f, src.cend(), '\n');  // skip to end of line
      } else if (n == '*') {                 // found "/*"
        auto term = std::string("*/");       // skip to end of next "*/"
        f         = std::search(std::next(f), src.cend(), term.cbegin(), term.cend()) + term.size();
      } else {
        output.push_back('/');  // lone '/' should be pushed into output
      }
    }
  }
  return output;
}

}  // namespace

constexpr char percent_escape[] = "_";  // NOLINT

std::string ptx_parser::escape_percent(std::string const& src)
{
  // b/c we're transforming into inline ptx we aren't allowed to have register names starting with %
  auto f = std::ranges::find_if_not(src, [](auto c) { return is_white(c) || c == '['; });
  if (f != src.end() && *f == '%') {
    std::string output = src;
    output.replace(std::distance(src.begin(), f), 1, percent_escape);
    return output;
  }
  return src;
}

std::string ptx_parser::remove_nonalphanumeric(std::string const& src)
{
  std::string out = src;
  auto f          = std::ranges::find_if_not(out, [](auto c) { return is_white(c) || c == '['; });
  auto l          = std::find_if(f, out.end(), [](auto c) { return is_white(c) || c == ']'; });
  std::replace_if(f, l, [](auto c) { return !isalnum(c) && c != '_'; }, '_');
  return std::string(f, l);
}

std::string ptx_parser::register_type_to_contraint(std::string const& src)
{
  if (src == ".b8" || src == ".u8" || src == ".s8")
    return "h";
  else if (src == ".u16" || src == ".s16" || src == ".b16" || src == ".f16")
    return "h";
  else if (src == ".b32" || src == ".u32" || src == ".s32" || src == ".f16x2")
    return "r";
  else if (src == ".u64" || src == ".b64" || src == ".s64")
    return "l";
  else if (src == ".f32")
    return "f";
  else if (src == ".f64")
    return "d";
  else
    return "x_reg";
}

std::string ptx_parser::register_type_to_cpp_type(std::string const& register_type)
{
  if (register_type == ".b8" || register_type == ".s8" || register_type == ".u8")
    return "char";
  else if (register_type == ".b16" || register_type == ".s16" || register_type == ".u16")
    return "short int";
  else if (register_type == ".f16")
    return "half";
  else if (register_type == ".b32" || register_type == ".s32" || register_type == ".u32")
    return "int";
  else if (register_type == ".f16x2")
    return "half2";
  else if (register_type == ".b64" || register_type == ".s64" || register_type == ".u64")
    return "long long int";
  else if (register_type == ".f32")
    return "float";
  else if (register_type == ".f64")
    return "double";
  else
    return "x_cpptype";
}

int32_t get_register_size(std::string_view register_type)
{
  if (register_type == ".b8" || register_type == ".s8" || register_type == ".u8") {
    return 8;
  } else if (register_type == ".b16" || register_type == ".s16" || register_type == ".u16" ||
             register_type == ".f16") {
    return 16;
  } else if (register_type == ".b32" || register_type == ".s32" || register_type == ".u32" ||
             register_type == ".f32" || register_type == ".f16x2") {
    return 32;
  } else if (register_type == ".b64" || register_type == ".s64" || register_type == ".u64" ||
             register_type == ".f64") {
    return 64;
  } else {
    CUDF_FAIL("Unknown register type: " + std::string(register_type));
  }
}

std::string ptx_parser::parse_instruction(std::string const& src)
{
  // I am assuming for an instruction statement the starting phrase is an
  // instruction.
  std::size_t const length = src.size();
  std::string output;
  std::string suffix;

  std::string const original_code = "\n   /**   " + src + "  */\n";

  int piece_count = 0;

  std::size_t start                 = 0;
  std::size_t stop                  = 0;
  bool is_instruction               = true;
  bool is_pragma_instruction        = false;
  bool is_param_loading_instruction = false;
  std::string register_type;
  bool blank = true;
  while (stop < length) {
    while (start < length && (is_white(src[start]) || src[start] == ',' || src[start] == '{' ||
                              src[start] == '}')) {  // running to the first non-white character.
      if (src[start] == ',') output += ',';
      if (src[start] == '{') output += '{';
      if (src[start] == '}') output += '}';
      start++;
    }
    stop = start;
    if (stop < length) {
      blank = false;
      output += " ";
    } else {
      break;
    }
    if (src[start] == '[') {
      while (stop < length && src[stop] != ']') {
        stop++;
      }
      stop++;
    } else {
      while (stop < length && !is_white(src[stop]) && src[stop] != ',' && src[stop] != ':') {
        stop++;
      }
      if (src[stop] == ':') {
        // This is a branch
        stop++;
        output += std::string(src, start, stop - start);
        start = stop;
        continue;
      }
    }
    std::string piece = std::string(src, start, stop - start);
    if (is_instruction) {
      if (piece.find("ld.param") != std::string::npos) {
        is_param_loading_instruction = true;
        register_type                = std::string(piece, 8, stop - 8);
        // This is the ld.param sentence
        if (get_register_size(register_type) < 64) {
          // The trick to support `ld` statement whose destination reg. wider than
          // the instruction width, e.g.
          //
          //  "ld.param.s32 %rd0, [...];",
          //
          // where %rd0 is a 64-bit register. Directly converting to "mov" instruction
          // does not work since "register widening" is ONLY allowed for
          // "ld", "st", and "cvt". So we use cvt instead and something like
          // "cvt.s32.s32". This keep the same operation behavior and when compiling to
          // SASS code "usually" (in cases I have seen) this is optimized away, thus
          // gives no performance penalty.
          output += " cvt" + register_type + register_type;
        } else {
          output += " mov" + register_type;
        }
      } else if (piece.find("st.param") != std::string::npos) {
        return "asm volatile (\"" + output +
               "/** *** The way we parse the CUDA PTX assumes the function returns the return "
               "value through the first function parameter. Thus the `st.param.***` instructions "
               "are not processed. *** */" +
               "\");" + original_code;  // Our port does not support return value;
      } else if (piece.find(".pragma") != std::string::npos) {
        is_pragma_instruction = true;
        output += " " + piece;
      } else if (piece[0] == '@') {
        output += " @" + remove_nonalphanumeric(piece.substr(1, piece.size() - 1));
      } else {
        output += " " + piece;
      }
      is_instruction = false;
    } else {
      // Here it should be the registers.
      if (piece_count == 2 && is_param_loading_instruction) {
        // This is the source of the parameter loading instruction
        output += " %0";

        auto constraint   = register_type_to_contraint(register_type);
        auto cpp_typename = register_type_to_cpp_type(register_type);

        // there's no 8-bit register size constraint in PTX, so we widen the argument to 16-bits
        if (get_register_size(register_type) == 8) {
          suffix = std::format(
            ": : \"{}\"( static_cast<short>( {} ) )", constraint, remove_nonalphanumeric(piece));
        } else {
          // normal case
          suffix = std::format(": : \"{}\"( *reinterpret_cast<{} *>(&{}) )",
                               constraint,
                               cpp_typename,
                               remove_nonalphanumeric(piece));
        }
      } else if (is_pragma_instruction) {
        // quote any string
        std::string transformed_piece;
        for (auto const& c : piece) {
          if (c == '"') {
            transformed_piece += "\\\"";
          } else {
            transformed_piece += c;
          }
        }
        output += transformed_piece;
      } else {
        output += escape_percent(std::string(src, start, stop - start));
      }
    }
    start = stop;
    piece_count++;
  }
  if (!blank) output += ";";
  return "asm volatile (\"" + output + "\"" + suffix + ");" + original_code;
}

std::string ptx_parser::parse_statement(std::string const& src)
{
  auto f = std::ranges::find_if_not(src, [](auto c) { return is_white(c); });
  return f == src.cend() ? " \n" : parse_instruction(std::string(f, src.cend()));
}

std::vector<std::string> ptx_parser::parse_function_body(std::string const& src)
{
  auto f = src.cbegin();
  std::vector<std::string> statements;

  while (f < src.cend()) {
    auto l = std::find(f, src.cend(), ';');
    statements.push_back(parse_statement(std::string(f, l)));
    f = ++l;
  }
  return statements;
}

ptx_param ptx_parser::parse_param(std::string const& src)
{
  auto i = 0;
  auto f = src.cbegin();

  ptx_param param;

  // i.e: .param .b32 _ZNparam
  while (f < src.cend() && i <= 3) {
    f      = std::find_if_not(f, src.cend(), [](auto c) { return is_white(c); });
    auto l = std::find_if(f, src.cend(), [](auto c) { return is_white(c); });

    auto token = std::string(f, l);

    if (i == 1) {
      param.register_type = token;
    } else if (i == 2) {
      param.identifier = remove_nonalphanumeric(token);
    }

    i++;
    f = l;
  }

  return param;
}

std::vector<ptx_param> ptx_parser::parse_param_list(std::string const& src)
{
  auto f = src.begin();

  std::vector<ptx_param> params{};

  while (f < src.end()) {
    auto l = std::find(f, src.end(), ',');
    params.push_back(parse_param(std::string(f, l)));

    f = ++l;
  }

  return params;
}

std::string ptx_parser::parse_function_header(std::string const& src)
{
  // Essentially we only need the information inside the two pairs of parentheses.
  auto f = [&] {
    auto i = std::ranges::find_if_not(src, [](auto c) { return is_white(c); });
    if (i != src.cend() && *i == '(')  // This function has a return type
      // First Pass: output param list
      i = std::find_if_not(std::next(i), src.cend(), [](auto c) { return c == ')'; });
    // The function name
    i = std::find_if_not(std::next(i), src.cend(), [](auto c) { return is_white(c) || c == '('; });
    // Second Pass: input param list
    return std::next(std::find(i, src.cend(), '('));
  }();

  auto l = std::find(f, src.cend(), ')');

  auto const ptx_params = parse_param_list(std::string(f, l));

  CUDF_EXPECTS(std::ranges::all_of(
                 param_types, [&](auto const& entry) { return entry.first < ptx_params.size(); }),
               "Argument index exceeds the number of parameters found in the PTX");

  std::vector<std::string> param_decls;

  std::transform(cuda::counting_iterator<std::size_t>{0},
                 cuda::counting_iterator{ptx_params.size()},
                 std::back_inserter(param_decls),
                 [&](std::size_t param_index) {
                   auto const& param = ptx_params[param_index];

                   if (auto const it = param_types.find(param_index); it != param_types.end()) {
                     return it->second + " " + param.identifier;
                   } else {
                     return std::string{register_type_to_cpp_type(param.register_type)} + " " +
                            param.identifier;
                   }
                 });

  std::string const param_list =
    param_decls.empty() ? ""
                        : std::accumulate(std::next(param_decls.begin()),
                                          param_decls.end(),
                                          param_decls.front(),
                                          [](std::string const& p0, std::string const& p1) {
                                            return p0 + ",\n " + p1;
                                          });

  return "\n__device__ __inline__ void " + function_name + "(" + param_list + "){" + "\n";
}

// The interface
std::string ptx_parser::parse()
{
  std::string const no_comments = remove_comments(ptx);

  auto const _func = std::string(".func");  // Go directly to the .func mark
  auto f           = std::ranges::search(no_comments, _func).end();

  CUDF_EXPECTS(f < no_comments.cend(), "No function (.func) found in the input ptx code.\n");

  auto l = std::find(f, no_comments.cend(), '{');

  auto f2 = std::next(l);
  auto l2 = std::find_if(f2, no_comments.cend(), [brace_count = 0](auto c) mutable {
    if (c == '{') ++brace_count;
    if (c == '}') {
      if (brace_count == 0) return true;  // find matching } to first found {
      --brace_count;
    }
    return false;
  });

  // DO NOT CHANGE ORDER - parse_function_body must be called before parse_function_header
  // because the function parameter types are inferred from their corresponding load
  // instructions in the function body
  auto const fn_body_output   = parse_function_body(std::string(f2, l2));
  auto const fn_header_output = parse_function_header(std::string(f, l));

  // Don't use std::accumulate until C++20 when rvalue references are supported
  auto final_output = fn_header_output + "\n asm volatile (\"{\");";
  for (auto const& line : fn_body_output) {
    std::string output{line};
    std::string_view const ret_instruction = "ret;";
    if (auto start = output.find(ret_instruction); start != std::string::npos) {
      output.replace(start, ret_instruction.size(), "bra RETTGT;");
    }
    final_output += "  " + output + "\n";
  }
  return final_output + " asm volatile (\"RETTGT:}\");}";
}

ptx_parser::ptx_parser(std::string ptx_,
                       std::string function_name_,
                       std::map<unsigned int, std::string> param_types_)
  : ptx(std::move(ptx_)),
    function_name(std::move(function_name_)),
    param_types(std::move(param_types_))
{
}

std::string parse_single_function_ptx(std::string const& src,
                                      std::string const& function_name,
                                      std::map<unsigned int, std::string> param_types)
{
  ptx_parser instance(src, function_name, param_types);

  return instance.parse();
}

// The interface
std::string parse_single_function_cuda(std::string const& src, std::string const& function_name)
{
  std::string no_comments = remove_comments(src);

  // For CUDA device function we just need to find the function
  // name and replace it with the specified one.
  std::size_t const length = no_comments.size();
  std::size_t start        = 0;
  std::size_t stop         = start;

  while (stop < length && no_comments[stop] != '(') {
    stop++;
  }
  CUDF_EXPECTS(stop != length && stop != 0,
               "No CUDA device function found in the input CUDA code.\n");

  stop--;

  while (stop > 0 && is_white(no_comments[stop])) {
    stop--;
  }
  CUDF_EXPECTS(stop != 0 || !is_white(no_comments[0]),
               "No CUDA device function name found in the input CUDA code.\n");

  start = stop;
  while (start > 0 && !is_white(no_comments[start])) {
    start--;
  }
  start++;
  stop++;
  CUDF_EXPECTS(start < stop, "No CUDA device function name found in the input CUDA code.\n");

  no_comments.replace(start, stop - start, function_name);

  return no_comments;
}

}  // namespace jit
}  // namespace cudf
