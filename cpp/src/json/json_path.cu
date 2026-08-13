/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/parsing_utils.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/json/json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/scan.h>
#include <thrust/tuple.h>

namespace cudf {
namespace detail {

namespace {

// change to "\n" and 1 to make output more readable
#define DEBUG_NEWLINE
constexpr int DEBUG_NEWLINE_LEN = 0;

/**
 * @brief Result of calling a parse function.
 *
 * The primary use of this is to distinguish between "success" and
 * "success but no data" return cases.  For example, if you are reading the
 * values of an array you might call a parse function in a while loop. You
 * would want to continue doing this until you either encounter an error (parse_result::ERROR)
 * or you get nothing back (parse_result::EMPTY)
 */
enum class parse_result {
  ERROR,          // failure
  SUCCESS,        // success
  MISSING_FIELD,  // success, but the field is missing
  EMPTY,          // success, but no data
};

/**
 * @brief Base parser class inherited by the (device-side) json_state class and
 * (host-side) path_state class.
 *
 * Contains a number of useful utility functions common to parsing json and
 * JSONPath strings.
 */
class parser {
 protected:
  CUDF_HOST_DEVICE inline parser() {}
  CUDF_HOST_DEVICE inline parser(char const* _input, int64_t _input_len)
    : input(_input), input_len(_input_len), pos(_input)
  {
    parse_whitespace();
  }

  CUDF_HOST_DEVICE inline parser(parser const& p)
    : input(p.input), input_len(p.input_len), pos(p.pos)
  {
  }

  CUDF_HOST_DEVICE inline bool eof(char const* p) { return p - input >= input_len; }
  CUDF_HOST_DEVICE inline bool eof() { return eof(pos); }

  CUDF_HOST_DEVICE inline bool parse_whitespace()
  {
    while (!eof()) {
      if (is_whitespace(*pos)) {
        pos++;
      } else {
        return true;
      }
    }
    return false;
  }

  CUDF_HOST_DEVICE inline bool is_hex_digit(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
  }

  CUDF_HOST_DEVICE inline int64_t chars_left() { return input_len - ((pos - input) + 1); }

  /**
   * @brief Parse an escape sequence.
   *
   * Must be a valid sequence as specified by the JSON format
   * https://www.json.org/json-en.html
   *
   * @returns True on success or false on fail.
   */
  CUDF_HOST_DEVICE inline bool parse_escape_seq()
  {
    if (*pos != '\\') { return false; }
    char c = *++pos;

    // simple case
    if (c == '\"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' ||
        c == 't') {
      pos++;
      return true;
    }

    // hex digits: must be of the form uXXXX  where each X is a valid hex digit
    if (c == 'u' && chars_left() >= 4 && is_hex_digit(pos[1]) && is_hex_digit(pos[2]) &&
        is_hex_digit(pos[3]) && is_hex_digit(pos[4])) {
      pos += 5;
      return true;
    }

    // an illegal escape sequence.
    return false;
  }

  /**
   * @brief Parse a quote-enclosed JSON string.
   *
   * @param[out] str The resulting string.
   * @param can_be_empty Parameter indicating whether it is valid for the string
   * to not be present.
   * @param quote Character expected as the surrounding quotes.  A value of 0
   * indicates allowing either single or double quotes (but not a mixture of both).
   * @returns A result code indicating success, failure or other result.
   */
  CUDF_HOST_DEVICE inline parse_result parse_string(string_view& str, bool can_be_empty, char quote)
  {
    str = string_view(nullptr, 0);

    if (parse_whitespace()) {
      // if the user specifies 0 for quote, allow either ' or ". otherwise
      // use the char directly
      if ((quote == 0 && (*pos == '\'' || *pos == '\"')) || (quote == *pos)) {
        quote = *pos;

        char const* start = ++pos;
        while (!eof()) {
          // handle escaped characters
          if (*pos == '\\') {
            if (!parse_escape_seq()) { return parse_result::ERROR; }
          } else if (*pos == quote) {
            str = string_view(start, pos - start);
            pos++;
            return parse_result::SUCCESS;
          } else {
            pos++;
          }
        }
      }
    }

    return can_be_empty ? parse_result::EMPTY : parse_result::ERROR;
  }

 protected:
  char const* input{nullptr};
  int64_t input_len{0};
  char const* pos{nullptr};

  CUDF_HOST_DEVICE inline bool is_whitespace(char c) { return c <= ' '; }
};

/**
 * @brief Output buffer object.  Used during the preprocess/size-computation step
 * and the actual output step.
 *
 * There is an important distinction between two cases:
 *
 * - producing no output at all. that is, the query matched nothing in the input.
 * - producing empty output. the query matched something in the input, but the
 *   value of the result is an empty string.
 *
 * The `has_output` field is the flag which indicates whether or not the output
 * from the query should be considered empty or null.
 *
 */
struct json_output {
  size_t output_max_len;
  char* output;
  cuda::std::optional<size_t> output_len;

  __device__ void add_output(char const* str, size_t len)
  {
    if (output != nullptr) { memcpy(output + output_len.value_or(0), str, len); }
    output_len = output_len.value_or(0) + len;
  }

  __device__ void add_output(string_view const& str) { add_output(str.data(), str.size_bytes()); }
};

enum json_element_type { NONE, OBJECT, ARRAY, VALUE };

/**
 * @brief Parsing class that holds the current state of the json to be parse and provides
 * functions for navigating through it.
 */
class json_state : private parser {
 public:
  __device__ json_state() : parser() {}
  __device__ json_state(char const* _input,
                        int64_t _input_len,
                        cudf::get_json_object_options _options)
    : parser(_input, _input_len),

      options(_options)
  {
  }

  __device__ json_state(json_state const& j)
    : parser(j),
      cur_el_start(j.cur_el_start),
      cur_el_type(j.cur_el_type),
      parent_el_type(j.parent_el_type),
      options(j.options)
  {
  }

  // retrieve the entire current element into the output
  __device__ parse_result extract_element(json_output* output, bool list_element)
  {
    char const* start = cur_el_start;
    char const* end   = start;

    // if we're a value type, do a simple value parse.
    if (cur_el_type == VALUE) {
      pos = cur_el_start;
      if (parse_value() != parse_result::SUCCESS) { return parse_result::ERROR; }
      end = pos;

      // potentially strip quotes from individually returned string values.
      if (options.get_strip_quotes_from_single_strings() && !list_element && is_quote(*start) &&
          *(end - 1) == *start) {
        start++;
        end--;
      }
    }
    // otherwise, march through everything inside
    else {
      int obj_count = 0;
      int arr_count = 0;

      while (!eof(end)) {
        // parse strings explicitly so we handle all interesting corner cases (such as strings
        // containing {, }, [ or ]
        if (is_quote(*end)) {
          string_view str;
          pos = end;
          if (parse_string(str, false, *end) == parse_result::ERROR) { return parse_result::ERROR; }
          end = pos;
        } else {
          char const c = *end++;
          switch (c) {
            case '{': obj_count++; break;
            case '}': obj_count--; break;
            case '[': arr_count++; break;
            case ']': arr_count--; break;
            default: break;
          }
        }
        if (obj_count == 0 && arr_count == 0) { break; }
      }
      if (obj_count > 0 || arr_count > 0) { return parse_result::ERROR; }
      pos = end;
    }

    // parse trailing ,
    if (parse_whitespace()) {
      if (*pos == ',') { pos++; }
    }

    if (output != nullptr) { output->add_output({start, static_cast<size_type>(end - start)}); }
    return parse_result::SUCCESS;
  }

  // skip the next element
  __device__ parse_result skip_element() { return extract_element(nullptr, false); }

  // advance to the next element
  __device__ parse_result next_element() { return next_element_internal(false); }

  // advance inside the current element
  __device__ parse_result child_element(json_element_type expected_type)
  {
    if (expected_type != NONE && cur_el_type != expected_type) { return parse_result::ERROR; }

    // if we succeed, record our parent element type.
    auto const prev_el_type = cur_el_type;
    auto const result       = next_element_internal(true);
    if (result == parse_result::SUCCESS) { parent_el_type = prev_el_type; }
    return result;
  }

  // return the next element that matches the specified name.
  __device__ parse_result next_matching_element(string_view const& name, bool inclusive)
  {
    // if we're not including the current element, skip it
    if (!inclusive) {
      parse_result result = next_element_internal(false);
      if (result != parse_result::SUCCESS) { return result; }
    }
    // loop until we find a match or there's nothing left
    do {
      if (name.size_bytes() == 1 && name.data()[0] == '*') {
        return parse_result::SUCCESS;
      } else if (cur_el_name == name) {
        return parse_result::SUCCESS;
      }
      // next
      parse_result result = next_element_internal(false);
      if (result != parse_result::SUCCESS) {
        return options.get_missing_fields_as_nulls() && result == parse_result::EMPTY
                 ? parse_result::MISSING_FIELD
                 : result;
      }
    } while (true);

    return parse_result::ERROR;
  }

  /**
   * @brief Parse a name field for a JSON element.
   *
   * When parsing JSON objects, it is not always a requirement that the name
   * actually exists.  For example, the outer object bounded by {} here has
   * no name, while the inner element "a" does.
   *
   * ```
   * {
   *    "a" : "b"
   * }
   * ```
   *
   * The user can specify whether or not the name string must be present via
   * the `can_be_empty` flag.
   *
   * When a name is present, it must be followed by a colon `:`
   *
   * @param[out] name The resulting name.
   * @param can_be_empty Parameter indicating whether it is valid for the name
   * to not be present.
   * @returns A result code indicating success, failure or other result.
   */
  CUDF_HOST_DEVICE inline parse_result parse_name(string_view& name, bool can_be_empty)
  {
    char const quote = options.get_allow_single_quotes() ? 0 : '\"';

    if (parse_string(name, can_be_empty, quote) == parse_result::ERROR) {
      return parse_result::ERROR;
    }

    // if we got a real string, the next char must be a :
    if (name.size_bytes() > 0) {
      if (!parse_whitespace()) { return parse_result::ERROR; }
      if (*pos == ':') {
        pos++;
        return parse_result::SUCCESS;
      }
    }
    return parse_result::EMPTY;
  }

 private:
  /**
   * @brief Parse a non-string JSON value.
   *
   * Non-string values include numbers, true, false, or null. This function does not
   * do any validation of the value.
   *
   * @param val (Output) The string containing the parsed value
   * @returns A result code indicating success, failure or other result.
   */
  CUDF_HOST_DEVICE inline parse_result parse_non_string_value(string_view& val)
  {
    if (!parse_whitespace()) { return parse_result::ERROR; }

    // parse to the end of the value
    char const* start = pos;
    char const* end   = start;
    while (!eof(end)) {
      char const c = *end;
      if (c == ',' || c == '}' || c == ']' || is_whitespace(c)) { break; }

      // illegal chars
      if (c == '[' || c == '{' || c == ':' || is_quote(c)) { return parse_result::ERROR; }
      end++;
    }
    pos = end;

    val = string_view(start, end - start);

    return parse_result::SUCCESS;
  }

  // parse a value - either a string or a number/null/bool
  __device__ parse_result parse_value()
  {
    if (!parse_whitespace()) { return parse_result::ERROR; }

    // string or number?
    string_view unused;
    return is_quote(*pos) ? parse_string(unused, false, *pos) : parse_non_string_value(unused);
  }

  __device__ parse_result next_element_internal(bool child)
  {
    // if we're not getting a child element, skip the current element.
    // this will leave pos as the first character -after- the close of
    // the current element
    if (!child && cur_el_start != nullptr) {
      if (skip_element() == parse_result::ERROR) { return parse_result::ERROR; }
      cur_el_start = nullptr;
    }
    // otherwise pos will be at the first character within the current element

    // can only get the child of an object or array.
    // this could theoretically be handled as an error, but the evaluators I've found
    // seem to treat this as "it's nothing"
    if (child && (cur_el_type == VALUE || cur_el_type == NONE)) { return parse_result::EMPTY; }

    // what's next
    if (!parse_whitespace()) { return parse_result::EMPTY; }
    // if we're closing off a parent element, we're done
    char const c = *pos;
    if (c == ']' || c == '}') { return parse_result::EMPTY; }

    // if we're not accessing elements of an array, check for name.
    bool const array_access =
      (cur_el_type == ARRAY && child) || (parent_el_type == ARRAY && !child);
    if (!array_access && parse_name(cur_el_name, true) == parse_result::ERROR) {
      return parse_result::ERROR;
    }

    // element type
    if (!parse_whitespace()) { return parse_result::EMPTY; }
    switch (*pos++) {
      case '[': cur_el_type = ARRAY; break;
      case '{': cur_el_type = OBJECT; break;

      case ',':
      case ':': return parse_result::ERROR;

      case '\'':
        if (!options.get_allow_single_quotes()) { return parse_result::ERROR; }
        cur_el_type = VALUE;
        break;

      // value type
      default: cur_el_type = VALUE; break;
    }

    // the start of the current element is always at the value, not the name
    cur_el_start = pos - 1;
    return parse_result::SUCCESS;
  }

  CUDF_HOST_DEVICE inline bool is_quote(char c)
  {
    return (c == '\"') || (options.get_allow_single_quotes() && (c == '\''));
  }

  char const* cur_el_start{nullptr};  // pointer to the first character of the -value- of the
                                      // current element - not the name
  string_view cur_el_name;            // name of the current element (if applicable)
  json_element_type cur_el_type{json_element_type::NONE};     // type of the current element
  json_element_type parent_el_type{json_element_type::NONE};  // parent element type
  get_json_object_options options;                            // behavior options
};

enum class path_operator_type { ROOT, CHILD, CHILD_WILDCARD, CHILD_INDEX, ERROR, END };

/**
 * @brief A "command" operator used to query a json string.  A full query is
 * an array of these operators applied to the incoming json string,
 */
struct path_operator {
  CUDF_HOST_DEVICE inline path_operator() {}
  CUDF_HOST_DEVICE inline path_operator(path_operator_type _type,
                                        json_element_type _expected_type = NONE)
    : type(_type), expected_type{_expected_type}
  {
  }

  path_operator_type type{path_operator_type::ERROR};  // operator type
  // the expected element type we're applying this operation to.
  // for example:
  //    - you cannot retrieve a subscripted field (eg [5]) from an object.
  //    - you cannot retrieve a field by name (eg  .book) from an array.
  //    - you -can- use .* for both arrays and objects
  // a value of NONE implies any type accepted
  json_element_type expected_type{NONE};  // the expected type of the element we're working with
  string_view name;                       // name to match against (if applicable)
  int index{-1};                          // index for subscript operator
};

/**
 * @brief Enum to specify whether parsing values enclosed within brackets, like `['book']`.
 */
enum class bracket_state : bool {
  INSIDE,  ///< Parsing inside brackets
  OUTSIDE  ///< Parsing outside brackets
};

/**
 * @brief Parsing class that holds the current state of the JSONPath string to be parsed
 * and provides functions for navigating through it. This is only called on the host
 * during the preprocess step which builds a command buffer that the gpu uses.
 */
class path_state : private parser {
 public:
  path_state(char const* _path, size_t _path_len) : parser(_path, _path_len) {}

  // get the next operator in the JSONPath string
  path_operator get_next_operator()
  {
    if (eof()) { return {path_operator_type::END}; }

    switch (*pos++) {
      case '$': return {path_operator_type::ROOT};

      case '.': {
        path_operator op;
        string_view term{".[", 2};
        if (parse_path_name(op.name, term, bracket_state::OUTSIDE)) {
          // this is another potential use case for __SPARK_BEHAVIORS / configurability
          // Spark currently only handles the wildcard operator inside [*], it does
          // not handle .*
          if (op.name.size_bytes() == 1 && op.name.data()[0] == '*') {
            op.type          = path_operator_type::CHILD_WILDCARD;
            op.expected_type = NONE;
          } else {
            op.type          = path_operator_type::CHILD;
            op.expected_type = OBJECT;
          }
          return op;
        }
      } break;

      // 3 ways this can be used
      // indices:   [0]
      // name:      ['book']
      // wildcard:  [*]
      case '[': {
        path_operator op;
        string_view term{"]", 1};
        bool const is_string = *pos == '\'';
        if (parse_path_name(op.name, term, bracket_state::INSIDE)) {
          pos++;
          if (op.name.size_bytes() == 1 && op.name.data()[0] == '*') {
            op.type          = path_operator_type::CHILD_WILDCARD;
            op.expected_type = NONE;
          } else {
            if (is_string) {
              op.type          = path_operator_type::CHILD;
              op.expected_type = OBJECT;
            } else {
              op.type          = path_operator_type::CHILD_INDEX;
              auto const value = cudf::io::parse_numeric<int>(
                op.name.data(), op.name.data() + op.name.size_bytes(), json_opts);
              op.index = value.value_or(-1);
              CUDF_EXPECTS(op.index >= 0, "Invalid numeric index specified in JSONPath");
              op.expected_type = ARRAY;
            }
          }
          return op;
        }
      } break;

      // wildcard operator
      case '*': {
        pos++;
        return path_operator{path_operator_type::CHILD_WILDCARD};
      } break;

      default: CUDF_FAIL("Unrecognized JSONPath operator", std::invalid_argument); break;
    }
    return {path_operator_type::ERROR};
  }

 private:
  cudf::io::parse_options_view json_opts{',', '\n', '\"', '.'};

  // b_state is set to INSIDE while parsing values enclosed within [ ], otherwise OUTSIDE
  bool parse_path_name(string_view& name, string_view const& terminators, bracket_state b_state)
  {
    switch (*pos) {
      case '*':
        name = string_view(pos, 1);
        pos++;
        break;

      case '\'':
        if (b_state == bracket_state::INSIDE) {
          if (parse_string(name, false, '\'') != parse_result::SUCCESS) { return false; }
          break;
        }
        // if not inside the [ ] -> go to default

      default: {
        size_t const chars_left = input_len - (pos - input);
        char const* end         = std::find_first_of(
          pos, pos + chars_left, terminators.data(), terminators.data() + terminators.size_bytes());
        if (end) {
          name = string_view(pos, end - pos);
          pos  = end;
        } else {
          name = string_view(pos, chars_left);
          pos  = input + input_len;
        }
        break;
      }
    }

    // an empty name is not valid
    CUDF_EXPECTS(
      name.size_bytes() > 0, "Invalid empty name in JSONPath query string", std::invalid_argument);

    return true;
  }
};

/**
 * @brief Preprocess the incoming JSONPath string on the host to generate a
 * command buffer for use by the GPU.
 *
 * @param json_path The incoming json path
 * @param stream Cuda stream to perform any gpu actions on
 * @returns A pair containing the command buffer, and maximum stack depth required.
 */
std::pair<cuda::std::optional<rmm::device_uvector<path_operator>>, int> build_command_buffer(
  cudf::string_scalar const& json_path, rmm::cuda_stream_view stream)
{
  std::string h_json_path = json_path.to_string(stream);
  path_state p_state(h_json_path.data(), static_cast<size_type>(h_json_path.size()));

  std::vector<path_operator> h_operators;

  path_operator op;
  int max_stack_depth = 1;
  do {
    op = p_state.get_next_operator();
    CUDF_EXPECTS(op.type != path_operator_type::ERROR,
                 "Encountered invalid JSONPath input string",
                 std::invalid_argument);
    if (op.type == path_operator_type::CHILD_WILDCARD) { max_stack_depth++; }
    // convert pointer to device pointer
    if (op.name.size_bytes() > 0) {
      op.name =
        string_view(json_path.data() + (op.name.data() - h_json_path.data()), op.name.size_bytes());
    }
    if (op.type == path_operator_type::ROOT) {
      CUDF_EXPECTS(h_operators.size() == 0, "Root operator ($) can only exist at the root");
    }
    // if we have not gotten a root operator to start, and we're not empty, quietly push a
    // root operator now.
    if (h_operators.size() == 0 && op.type != path_operator_type::ROOT &&
        op.type != path_operator_type::END) {
      h_operators.push_back(path_operator{path_operator_type::ROOT});
    }
    h_operators.push_back(op);
  } while (op.type != path_operator_type::END);

  auto const is_empty = h_operators.size() == 1 && h_operators[0].type == path_operator_type::END;
  return is_empty ? std::pair(cuda::std::nullopt, 0)
                  : std::pair(cuda::std::make_optional(cudf::detail::make_device_uvector(
                                h_operators, stream, cudf::get_current_device_resource_ref())),
                              max_stack_depth);
}

#define PARSE_TRY(_x)                                                       \
  do {                                                                      \
    last_result = _x;                                                       \
    if (last_result == parse_result::ERROR) { return parse_result::ERROR; } \
  } while (0)

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * @param j_state The incoming json string and associated parser
 * @param commands The command buffer to be applied to the string. Always ends with a
 * path_operator_type::END
 * @param output Buffer user to store the results of the query
 * @returns A result code indicating success/fail/empty.
 */
template <int max_command_stack_depth>
__device__ parse_result parse_json_path(json_state& j_state,
                                        path_operator const* commands,
                                        json_output& output)
{
  // manually maintained context stack in lieu of calling parse_json_path recursively.
  struct context {
    json_state j_state;
    path_operator const* commands;
    bool list_element;
    bool state_flag;
  };
  context stack[max_command_stack_depth];
  int stack_pos     = 0;
  auto push_context = [&stack, &stack_pos](json_state const& _j_state,
                                           path_operator const* _commands,
                                           bool _list_element = false,
                                           bool _state_flag   = false) {
    if (stack_pos == max_command_stack_depth - 1) { return false; }
    stack[stack_pos++] = context{_j_state, _commands, _list_element, _state_flag};
    return true;
  };
  auto pop_context = [&stack, &stack_pos](context& c) {
    if (stack_pos > 0) {
      c = stack[--stack_pos];
      return true;
    }
    return false;
  };
  push_context(j_state, commands, false);

  parse_result last_result = parse_result::SUCCESS;
  context ctx;
  int element_count = 0;
  while (pop_context(ctx)) {
    path_operator op = *ctx.commands;
    switch (op.type) {
      // whatever the first object is
      case path_operator_type::ROOT:
        PARSE_TRY(ctx.j_state.next_element());
        push_context(ctx.j_state, ctx.commands + 1);
        break;

      // .name
      // ['name']
      // [1]
      // will return a single thing
      case path_operator_type::CHILD: {
        PARSE_TRY(ctx.j_state.child_element(op.expected_type));
        if (last_result == parse_result::SUCCESS) {
          PARSE_TRY(ctx.j_state.next_matching_element(op.name, true));
          if (last_result == parse_result::SUCCESS) {
            push_context(ctx.j_state, ctx.commands + 1, ctx.list_element);
          } else if (last_result == parse_result::MISSING_FIELD) {
            if (ctx.list_element && element_count > 0) {
              output.add_output({"," DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
            }
            output.add_output({"null", 4});
            element_count++;
          }
        }
      } break;

      // .*
      // [*]
      // will return an array of things
      case path_operator_type::CHILD_WILDCARD: {
        // if we're on the first element of this wildcard
        if (!ctx.state_flag) {
          // we will only ever be returning 1 array
          if (!ctx.list_element) { output.add_output({"[" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN}); }

          // step into the child element
          PARSE_TRY(ctx.j_state.child_element(op.expected_type));
          if (last_result == parse_result::EMPTY) {
            if (!ctx.list_element) {
              output.add_output({"]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
            }
            last_result = parse_result::SUCCESS;
            break;
          }

          // first element
          PARSE_TRY(ctx.j_state.next_matching_element({"*", 1}, true));
          if (last_result == parse_result::EMPTY) {
            if (!ctx.list_element) {
              output.add_output({"]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
            }
            last_result = parse_result::SUCCESS;
            break;
          }

          // re-push ourselves
          push_context(ctx.j_state, ctx.commands, ctx.list_element, true);
          // push the next command
          push_context(ctx.j_state, ctx.commands + 1, true);
        } else {
          // next element
          PARSE_TRY(ctx.j_state.next_matching_element({"*", 1}, false));
          if (last_result == parse_result::EMPTY) {
            if (!ctx.list_element) {
              output.add_output({"]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
            }
            last_result = parse_result::SUCCESS;
            break;
          }

          // re-push ourselves
          push_context(ctx.j_state, ctx.commands, ctx.list_element, true);
          // push the next command
          push_context(ctx.j_state, ctx.commands + 1, true);
        }
      } break;

      // [0]
      // [1]
      // etc
      // returns a single thing
      case path_operator_type::CHILD_INDEX: {
        PARSE_TRY(ctx.j_state.child_element(op.expected_type));
        if (last_result == parse_result::SUCCESS) {
          string_view const any{"*", 1};
          PARSE_TRY(ctx.j_state.next_matching_element(any, true));
          if (last_result == parse_result::SUCCESS) {
            int idx;
            for (idx = 1; idx <= op.index; idx++) {
              PARSE_TRY(ctx.j_state.next_matching_element(any, false));
              if (last_result == parse_result::EMPTY) { break; }
            }
            // if we didn't end up at the index we requested, this is an invalid index
            if (idx - 1 != op.index) { return parse_result::ERROR; }
            push_context(ctx.j_state, ctx.commands + 1, ctx.list_element);
          }
        }
      } break;

      // some sort of error.
      case path_operator_type::ERROR: return parse_result::ERROR; break;

      // END case
      default: {
        if (ctx.list_element && element_count > 0) {
          output.add_output({"," DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
        }
        PARSE_TRY(ctx.j_state.extract_element(&output, ctx.list_element));
        if (ctx.list_element && last_result != parse_result::EMPTY) { element_count++; }
      } break;
    }
  }

  return parse_result::SUCCESS;
}

// hardcoding this for now. to reach a stack depth of 8 would require
// a JSONPath containing 7 nested wildcards so this is probably reasonable.
constexpr int max_command_stack_depth = 8;

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * This function exists primarily as a shim for debugging purposes.
 *
 * @param input The incoming json string
 * @param input_len Size of the incoming json string
 * @param commands The command buffer to be applied to the string. Always ends with a
 * path_operator_type::END
 * @param out_buf Buffer user to store the results of the query (nullptr in the size computation
 * step)
 * @param out_buf_size Size of the output buffer
 * @param options Options controlling behavior
 * @returns A pair containing the result code the output buffer.
 */
__device__ cuda::std::pair<parse_result, json_output> get_json_object_single(
  char const* input,
  size_t input_len,
  path_operator const* const commands,
  char* out_buf,
  size_t out_buf_size,
  get_json_object_options options)
{
  json_state j_state(input, input_len, options);
  json_output output{out_buf_size, out_buf};

  auto const result = parse_json_path<max_command_stack_depth>(j_state, commands, output);

  return {result, output};
}

/**
 * @brief Kernel for running the JSONPath query.
 *
 * This kernel operates in a 2-pass way.  On the first pass, it computes
 * output sizes.  On the second pass it fills in the provided output buffers
 * (chars and validity)
 *
 * @param col Device view of the incoming string
 * @param commands JSONPath command buffer
 * @param output_offsets Buffer used to store the string offsets for the results of the query
 * @param out_buf Buffer used to store the results of the query
 * @param out_validity Output validity buffer
 * @param out_valid_count Output count of # of valid bits
 * @param options Options controlling behavior
 */
template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void get_json_object_kernel(column_device_view col,
                              path_operator const* const commands,
                              size_type* d_sizes,
                              cudf::detail::input_offsetalator output_offsets,
                              cuda::std::optional<char*> out_buf,
                              cuda::std::optional<bitmask_type*> out_validity,
                              cuda::std::optional<size_type*> out_valid_count,
                              get_json_object_options options)
{
  auto tid          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffff'ffffu, tid < col.size());
  while (tid < col.size()) {
    bool is_valid         = false;
    string_view const str = col.element<string_view>(tid);
    size_type output_size = 0;
    if (str.size_bytes() > 0) {
      char* dst = out_buf.has_value() ? out_buf.value() + output_offsets[tid] : nullptr;
      size_t const dst_size =
        out_buf.has_value() ? output_offsets[tid + 1] - output_offsets[tid] : 0;

      parse_result result;
      json_output out;
      thrust::tie(result, out) =
        get_json_object_single(str.data(), str.size_bytes(), commands, dst, dst_size, options);
      output_size = out.output_len.value_or(0);
      if (out.output_len.has_value() && result == parse_result::SUCCESS) { is_valid = true; }
    }

    // filled in only during the precompute step. during the compute step, the offsets
    // are fed back in so we do -not- want to write them out
    if (!out_buf.has_value()) { d_sizes[tid] = output_size; }

    // validity filled in only during the output step
    if (out_validity.has_value()) {
      uint32_t mask = __ballot_sync(active_threads, is_valid);
      // 0th lane of the warp writes the validity
      if (!(tid % cudf::detail::warp_size)) {
        out_validity.value()[cudf::word_index(tid)] = mask;
        warp_valid_count += __popc(mask);
      }
    }

    tid += stride;
    active_threads = __ballot_sync(active_threads, tid < col.size());
  }

  // sum the valid counts across the whole block
  if (out_valid_count) {
    size_type block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count.value(), block_valid_count); }
  }
}

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              get_json_object_options options,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  // preprocess the json_path into a command buffer
  auto preprocess = build_command_buffer(json_path, stream);
  CUDF_EXPECTS(std::get<1>(preprocess) <= max_command_stack_depth,
               "Encountered JSONPath string that is too complex");

  if (col.is_empty()) return make_empty_column(type_id::STRING);

  // if the query is empty, return a string column containing all nulls
  if (!std::get<0>(preprocess).has_value()) {
    // Create a proper all-null strings column with valid structure (offsets + chars children)
    auto offsets = cudf::make_column_from_scalar(
      cudf::numeric_scalar<int32_t>(0, true, stream, cudf::get_current_device_resource_ref()),
      col.size() + 1,
      stream,
      mr);

    return make_strings_column(
      col.size(),
      std::move(offsets),
      rmm::device_buffer{0, stream, mr},  // empty chars
      col.size(),                         // null_count
      cudf::detail::create_null_mask(col.size(), mask_state::ALL_NULL, stream, mr));
  }

  // compute output sizes
  auto sizes =
    rmm::device_uvector<size_type>(col.size(), stream, cudf::get_current_device_resource_ref());
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(col.offsets());

  constexpr int block_size = 512;
  cudf::detail::grid_1d const grid{col.size(), block_size};
  auto cdv = column_device_view::create(col.parent(), stream);
  // preprocess sizes (returned in the offsets buffer)
  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *cdv,
      std::get<0>(preprocess).value().data(),
      sizes.data(),
      d_offsets,
      cuda::std::nullopt,
      cuda::std::nullopt,
      cuda::std::nullopt,
      options);
  CUDF_CUDA_TRY(cudaGetLastError());

  // convert sizes to offsets
  auto [offsets, output_size] =
    cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // allocate output string column
  rmm::device_uvector<char> chars(output_size, stream, mr);

  // potential optimization : if we know that all outputs are valid, we could skip creating
  // the validity mask altogether
  rmm::device_buffer validity =
    cudf::detail::create_null_mask(col.size(), mask_state::UNINITIALIZED, stream, mr);

  // compute results
  cudf::detail::device_scalar<size_type> d_valid_count{
    0, stream, cudf::get_current_device_resource_ref()};

  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *cdv,
      std::get<0>(preprocess).value().data(),
      sizes.data(),
      d_offsets,
      chars.data(),
      static_cast<bitmask_type*>(validity.data()),
      d_valid_count.data(),
      options);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto result = make_strings_column(col.size(),
                                    std::move(offsets),
                                    chars.release(),
                                    col.size() - d_valid_count.value(stream),
                                    std::move(validity));
  // unmatched array query may result in unsanitized '[' value in the result
  if (cudf::detail::has_nonempty_nulls(result->view(), stream)) {
    result = cudf::detail::purge_nonempty_nulls(result->view(), stream, mr);
  }
  return result;
}

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              get_json_object_options options,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_json_object(col, json_path, options, stream, mr);
}

}  // namespace cudf
