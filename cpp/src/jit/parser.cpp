/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parser.hpp"

#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
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

std::string parse_single_function_ptx(std::string_view src, std::string const& function_name)
{
  // Keep PTX intact for the linker, including directives, parameter layouts, and helper functions.
  // Tokenize only to find the first function and rename references to that exact symbol.
  struct token {
    size_t begin;
    size_t size;
  };
  std::vector<token> tokens;
  auto const is_identifier = [](unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '$' || c == '.' || c == '%';
  };
  for (size_t i = 0; i < src.size();) {
    if (is_white(src[i])) {
      ++i;
    } else if (src.compare(i, 2, "//") == 0) {
      auto end = src.find('\n', i + 2);
      i        = end == std::string::npos ? src.size() : end;
    } else if (src.compare(i, 2, "/*") == 0) {
      auto end = src.find("*/", i + 2);
      CUDF_EXPECTS(end != std::string::npos, "Unterminated PTX comment");
      i = end + 2;
    } else if (src[i] == '"') {
      ++i;
      while (i < src.size() && src[i] != '"') {
        i += src[i] == '\\' && i + 1 < src.size() ? 2 : 1;
      }
      CUDF_EXPECTS(i < src.size(), "Unterminated PTX string");
      ++i;
    } else {
      auto const begin = i++;
      if (is_identifier(src[begin])) {
        while (i < src.size() && is_identifier(src[i])) {
          ++i;
        }
      }
      tokens.push_back({begin, i - begin});
    }
  }
  auto const text = [&](size_t i) { return src.substr(tokens[i].begin, tokens[i].size); };
  size_t i        = 0;
  while (i < tokens.size() && text(i) != ".func") {
    ++i;
  }
  CUDF_EXPECTS(i < tokens.size(), "No function (.func) found in the input PTX code");
  ++i;
  if (i < tokens.size() && text(i) == "(") {
    // Skip the optional return parameter list.
    int depth = 0;
    do {
      if (text(i) == "(") { ++depth; }
      if (text(i) == ")") { --depth; }
      ++i;
    } while (i < tokens.size() && depth > 0);
    CUDF_EXPECTS(depth == 0, "Unterminated PTX return parameter list");
  }
  CUDF_EXPECTS(i + 1 < tokens.size() && is_identifier(src[tokens[i].begin]) && text(i + 1) == "(",
               "No PTX device function name found");
  auto const original_name = text(i);
  CUDF_EXPECTS(!function_name.empty(), "PTX target function name must not be empty");

  std::string result;
  size_t copied = 0;
  for (size_t j = 0; j < tokens.size(); ++j) {
    if (text(j) == original_name) {
      result.append(src, copied, tokens[j].begin - copied);
      result.append(function_name);
      copied = tokens[j].begin + tokens[j].size;
    }
  }
  result.append(src, copied, src.size() - copied);
  return result;
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
