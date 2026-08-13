/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/regex/regcomp.h"

#include <cudf/strings/detail/utf8.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <array>
#include <cctype>
#include <clocale>
#include <cwctype>
#include <numeric>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {
namespace {
// Bitmask of all operators
enum { OPERATOR_MASK = 0200 };
enum OperatorType : int32_t {
  START        = 0200,  // Start, used for marker on stack
  LBRA_NC      = 0203,  // non-capturing group
  CAT          = 0205,  // Concatenation, implicit operator
  STAR         = 0206,  // Closure, *
  STAR_LAZY    = 0207,
  PLUS         = 0210,  // a+ == aa*
  PLUS_LAZY    = 0211,
  QUEST        = 0212,  // a? == a|nothing, i.e. 0 or 1 a's
  QUEST_LAZY   = 0213,
  COUNTED      = 0214,  // counted repeat a{2} a{3,5}
  COUNTED_LAZY = 0215,
  NOP          = 0302,  // No operation, internal use only
};
enum { ITEM_MASK = 0300 };

static reclass cclass_w(CCLASS_W);   // \w
static reclass cclass_s(CCLASS_S);   // \s
static reclass cclass_d(CCLASS_D);   // \d
static reclass cclass_W(NCCLASS_W);  // \W
static reclass cclass_S(NCCLASS_S);  // \S
static reclass cclass_D(NCCLASS_D);  // \D

// Tables for analyzing quantifiers
std::array<int, 5> const valid_preceding_inst_types{{CHAR, CCLASS, NCCLASS, ANY, ANYNL}};
std::array<char, 5> const quantifiers{{'*', '?', '+', '{', '|'}};
// Valid regex characters that can be escaped and used as literals
std::array<char, 33> const escapable_chars{
  {'.', '-', '+',  '*', '\\', '?', '^', '$', '|', '{', '}', '(', ')', '[', ']', '<', '>',
   '"', '~', '\'', '`', '_',  '@', '=', ';', ':', '!', '#', '%', '&', ',', '/', ' '}};

/**
 * @brief Converts UTF-8 string into fixed-width 32-bit character vector.
 *
 * No character conversion occurs.
 * Each UTF-8 character is promoted into a 32-bit value.
 * The last entry in the returned vector will be a 0 value.
 * The fixed-width vector makes it easier to compile and faster to execute.
 *
 * @param pattern Regular expression encoded with UTF-8.
 * @return Fixed-width 32-bit character vector.
 */
std::vector<char32_t> string_to_char32_vector(std::string_view pattern)
{
  auto size             = static_cast<size_type>(pattern.size());
  size_type const count = std::count_if(pattern.cbegin(), pattern.cend(), [](char ch) {
    return is_begin_utf8_char(static_cast<uint8_t>(ch));
  });
  std::vector<char32_t> result(count + 1);
  char32_t* output_ptr  = result.data();
  char const* input_ptr = pattern.data();
  for (size_type idx = 0; idx < size; ++idx) {
    char_utf8 output_character = 0;
    size_type const ch_width   = to_char_utf8(input_ptr, output_character);
    input_ptr += ch_width;
    idx += ch_width - 1;
    *output_ptr++ = output_character;
  }
  result[count] = 0;  // last entry set to 0
  return result;
}

}  // namespace

int32_t reprog::add_inst(int32_t t)
{
  reinst inst{};
  inst.type        = t;
  inst.u2.left_id  = 0;
  inst.u1.right_id = 0;
  return add_inst(inst);
}

int32_t reprog::add_inst(reinst const& inst)
{
  _insts.push_back(inst);
  return static_cast<int32_t>(_insts.size() - 1);
}

int32_t reprog::add_class(reclass const& cls)
{
  _classes.push_back(cls);
  return static_cast<int32_t>(_classes.size() - 1);
}

reinst& reprog::inst_at(int32_t id) { return _insts[id]; }

reclass const& reprog::class_at(int32_t id) const { return _classes[id]; }

void reprog::set_start_inst(int32_t id) { _startinst_id = id; }

int32_t reprog::get_start_inst() const { return _startinst_id; }

int32_t reprog::insts_count() const { return static_cast<int>(_insts.size()); }

int32_t reprog::classes_count() const { return static_cast<int>(_classes.size()); }

void reprog::set_groups_count(int32_t groups) { _num_capturing_groups = groups; }

int32_t reprog::groups_count() const { return _num_capturing_groups; }

reinst const* reprog::insts_data() const { return _insts.data(); }

reclass const* reprog::classes_data() const { return _classes.data(); }

int32_t const* reprog::starts_data() const { return _startinst_ids.data(); }

int32_t reprog::starts_count() const { return static_cast<int>(_startinst_ids.size()); }

static constexpr auto MAX_REGEX_CHAR = std::numeric_limits<char32_t>::max();

/**
 * @brief Converts pattern into regex classes
 */
class regex_parser {
 public:
  /**
   * @brief Single parsed pattern element.
   */
  struct Item {
    int32_t type;
    union {
      char32_t chr;
      int32_t cclass_id;
      struct {
        int16_t n;
        int16_t m;
      } count;
    } d;
    Item(int32_t type, char32_t chr) : type{type}, d{.chr = chr} {}
    Item(int32_t type, int32_t id) : type{type}, d{.cclass_id = id} {}
    Item(int32_t type, int16_t n, int16_t m) : type{type}, d{.count{n, m}} {}
  };

 private:
  reprog& _prog;
  char32_t const* const _pattern_begin;
  char32_t const* _expr_ptr;
  bool _lex_done{false};
  regex_flags const _flags;
  capture_groups const _capture;

  int32_t _id_cclass_w{-1};  // alphanumeric [a-zA-Z0-9_]
  int32_t _id_cclass_W{-1};  // not alphanumeric plus '\n'
  int32_t _id_cclass_s{-1};  // whitespace including '\t', '\n', '\r'
  int32_t _id_cclass_d{-1};  // digits [0-9]
  int32_t _id_cclass_D{-1};  // not digits

  char32_t _chr{};       // last lex'd char
  int32_t _cclass_id{};  // last lex'd class
  int16_t _min_count{};  // data for counted operators
  int16_t _max_count{};

  std::vector<Item> _items;
  bool _has_counted{false};

  /**
   * @brief Parses octal characters at the current expression position
   * to return the represented character
   *
   * Reads up to 3 octal digits. The first digit should be passed
   * in `in_chr`.
   *
   * @param in_chr The first character of the octal pattern
   * @return The resulting character
   */
  char32_t handle_octal(char32_t in_chr)
  {
    auto out_chr = in_chr - '0';
    auto c       = *_expr_ptr;
    auto digits  = 1;
    while ((c >= '0') && (c <= '7') && (digits < 3)) {
      out_chr = (out_chr * 8) | (c - '0');
      c       = *(++_expr_ptr);
      ++digits;
    }
    return out_chr;
  }

  /**
   * @brief Parses 2 hex characters at the current expression position
   * to return the represented character
   *
   * @return The resulting character
   */
  char32_t handle_hex()
  {
    std::string hex(1, static_cast<char>(*_expr_ptr++));
    hex.append(1, static_cast<char>(*_expr_ptr++));
    return static_cast<char32_t>(std::stol(hex, nullptr, 16));  // 16 = hex
  }

  /**
   * @brief Returns the next character in the expression
   *
   * Handles quoted (escaped) special characters and detecting the end of the expression.
   *
   * @return is-backslash-escape and character
   */
  std::pair<bool, char32_t> next_char()
  {
    if (_lex_done) { return {true, 0}; }

    auto c = *_expr_ptr++;
    if (c == '\\') {
      c = *_expr_ptr++;
      return {true, c};
    }

    if (c == 0) { _lex_done = true; }

    return {false, c};
  }

  // for \d and \D
  void add_ascii_digit_class(std::vector<reclass_range>& ranges, bool negated = false)
  {
    if (!negated) {
      ranges.push_back({'0', '9'});
    } else {
      ranges.push_back({0, '0' - 1});
      ranges.push_back({'9' + 1, MAX_REGEX_CHAR});
    }
  }

  // for \s and \S
  void add_ascii_space_class(std::vector<reclass_range>& ranges, bool negated = false)
  {
    if (!negated) {
      ranges.push_back({'\t', ' '});
    } else {
      ranges.push_back({0, '\t' - 1});
      ranges.push_back({' ' + 1, MAX_REGEX_CHAR});
    }
  }

  // for \w and \W
  void add_ascii_word_class(std::vector<reclass_range>& ranges, bool negated = false)
  {
    add_ascii_digit_class(ranges, negated);
    if (!negated) {
      ranges.push_back({'a', 'z'});
      ranges.push_back({'A', 'Z'});
      ranges.push_back({'_', '_'});
    } else {
      ranges.back().last = 'A' - 1;
      ranges.push_back({'Z' + 1, 'a' - 1});  // {'_'-1, '_' + 1}
      ranges.push_back({'z' + 1, MAX_REGEX_CHAR});
    }
  }

  char32_t swap_case(char32_t chr)
  {
    auto const cp = static_cast<wchar_t>(utf8_to_codepoint(chr));
    auto const lc = std::locale("C.UTF-8");  // always available
    return codepoint_to_utf8(std::isupper(cp, lc) ? std::tolower(cp, lc) : std::toupper(cp, lc));
  }

  int32_t build_cclass()
  {
    int32_t type = CCLASS;
    std::vector<char32_t> literals;
    int32_t builtins = 0;
    std::vector<reclass_range> ranges;

    auto [is_quoted, chr] = next_char();
    // check for negation
    if (!is_quoted && chr == '^') {
      type                     = NCCLASS;
      std::tie(is_quoted, chr) = next_char();
    }

    // parse class into a set of spans
    auto count_char = 0;
    while (true) {
      count_char++;
      if (chr == 0) { return 0; }  // malformed '[]'
      if (is_quoted) {
        switch (chr) {
          case 'n': chr = '\n'; break;
          case 'r': chr = '\r'; break;
          case 't': chr = '\t'; break;
          case 'a': chr = 0x07; break;
          case 'b': chr = 0x08; break;
          case 'f': chr = 0x0C; break;
          case '0' ... '7': {
            chr = handle_octal(chr);
            break;
          }
          case 'x': {
            chr = handle_hex();
            break;
          }
          case 'w':
          case 'W':
            if (is_ascii(_flags)) {
              add_ascii_word_class(ranges, chr == 'W');
            } else {
              builtins |= (chr == 'w' ? cclass_w.builtins : cclass_W.builtins);
            }
            std::tie(is_quoted, chr) = next_char();
            continue;
          case 's':
          case 'S':
            if (is_ascii(_flags)) {
              add_ascii_space_class(ranges, chr == 'S');
            } else {
              builtins |= (chr == 's' ? cclass_s.builtins : cclass_S.builtins);
            }
            std::tie(is_quoted, chr) = next_char();
            continue;
          case 'd':
          case 'D':
            if (is_ascii(_flags)) {
              add_ascii_digit_class(ranges, chr == 'D');
            } else {
              builtins |= (chr == 'd' ? cclass_d.builtins : cclass_D.builtins);
            }
            std::tie(is_quoted, chr) = next_char();
            continue;
        }
      }
      if (!is_quoted && chr == ']' && count_char > 1) { break; }  // done

      // A hyphen '-' here signifies a range of characters in a '[]' class definition.
      // The logic here also gracefully handles a dangling '-' appearing unquoted
      // at the beginning '[-x]' or at the end '[x-]' or by itself '[-]'
      // and treats the '-' as a literal value in this cclass in this case.
      if (!is_quoted && chr == '-' && !literals.empty()) {
        auto [q, n_chr] = next_char();
        if (n_chr == 0) { return 0; }  // malformed: '[x-'
        if (!q && n_chr == ']') {      // handles: '[x-]'
          literals.push_back(chr);
          literals.push_back(0);
          break;
        }
        if (0 == literals.back()) {
          literals.back() = n_chr;  // normal case: '[a-z]' update end-range character
        } else {
          literals.push_back(chr);  // adds '-'
          literals.push_back(chr);
          literals.push_back(n_chr);  // adds new character
          literals.push_back(0);
        }
      } else {
        literals.push_back(chr);
        literals.push_back(0);
      }
      std::tie(is_quoted, chr) = next_char();
    }

    // transform pairs of literals to ranges
    auto const counter = cuda::counting_iterator<std::size_t>{0};
    std::transform(counter,
                   counter + (literals.size() / 2),
                   std::back_inserter(ranges),
                   [&literals, this](auto idx) {
                     auto const lhs  = literals[idx * 2];
                     auto const next = literals[idx * 2 + 1];
                     auto const rhs  = next == 0 ? lhs : next;
                     CUDF_EXPECTS(lhs <= rhs,
                                  "invalid character range in class at " +
                                    std::to_string(std::distance(_pattern_begin, _expr_ptr)));
                     return reclass_range{lhs, rhs};
                   });
    if (is_ignorecase(_flags)) {
      // add the swapped case ranges
      std::transform(counter,
                     counter + (literals.size() / 2),
                     std::back_inserter(ranges),
                     [&literals, this](auto idx) {
                       auto const swap1 = swap_case(literals[idx * 2]);
                       auto const swap2 = swap_case(literals[idx * 2 + 1]);
                       return reclass_range{swap1, swap2};
                     });
    }
    // sort the ranges to help with detecting overlapping entries
    std::sort(ranges.begin(), ranges.end(), [](auto l, auto r) {
      return l.first == r.first ? l.last < r.last : l.first < r.first;
    });
    // combine overlapping entries: [a-f][c-g] => [a-g]
    if (ranges.size() > 1) {
      for (auto itr = ranges.begin() + 1; itr < ranges.end(); ++itr) {
        auto const prev = *(itr - 1);
        if (itr->first <= prev.last + 1) {
          // if these 2 ranges intersect, expand the current one
          *itr = reclass_range{prev.first, std::max(prev.last, itr->last)};
        }
      }
    }
    // remove any duplicates
    auto const end = std::unique(
      ranges.rbegin(), ranges.rend(), [](auto l, auto r) { return l.first == r.first; });
    ranges.erase(ranges.begin(), ranges.begin() + std::distance(end, ranges.rend()));

    _cclass_id = _prog.add_class(reclass{builtins, std::move(ranges)});
    return type;
  }

  int32_t lex(int32_t dot_type)
  {
    _chr = 0;

    auto [is_quoted, chr] = next_char();
    if (is_quoted) {
      switch (chr) {
        case 't': chr = '\t'; break;
        case 'n': chr = '\n'; break;
        case 'r': chr = '\r'; break;
        case 'a': chr = 0x07; break;
        case 'f': chr = 0x0C; break;
        case '0' ... '7': {
          chr = handle_octal(chr);
          break;
        }
        case 'x': {
          chr = handle_hex();
          break;
        }
        case 'w': {
          if (is_ascii(_flags)) {
            reclass cls;
            add_ascii_word_class(cls.literals);
            _cclass_id = _prog.add_class(cls);
          } else {
            if (_id_cclass_w < 0) { _id_cclass_w = _prog.add_class(cclass_w); }
            _cclass_id = _id_cclass_w;
          }
          return CCLASS;
        }
        case 'W': {
          if (is_ascii(_flags)) {
            reclass cls;
            add_ascii_word_class(cls.literals);
            _cclass_id = _prog.add_class(cls);
          } else {
            if (_id_cclass_W < 0) {
              reclass cls = cclass_w;
              cls.literals.push_back({'\n', '\n'});
              _id_cclass_W = _prog.add_class(cls);
            }
            _cclass_id = _id_cclass_W;
          }
          return NCCLASS;
        }
        case 's': {
          if (is_ascii(_flags)) {
            reclass cls;
            add_ascii_space_class(cls.literals);
            _cclass_id = _prog.add_class(cls);
          } else {
            if (_id_cclass_s < 0) { _id_cclass_s = _prog.add_class(cclass_s); }
            _cclass_id = _id_cclass_s;
          }
          return CCLASS;
        }
        case 'S': {
          if (is_ascii(_flags)) {
            reclass cls;
            add_ascii_space_class(cls.literals);
            _cclass_id = _prog.add_class(cls);
          } else {
            if (_id_cclass_s < 0) { _id_cclass_s = _prog.add_class(cclass_s); }
            _cclass_id = _id_cclass_s;
            return NCCLASS;
          }
        }
        case 'd': {
          if (is_ascii(_flags)) {
            reclass cls;
            add_ascii_digit_class(cls.literals);
            _cclass_id = _prog.add_class(cls);
          } else {
            if (_id_cclass_d < 0) { _id_cclass_d = _prog.add_class(cclass_d); }
            _cclass_id = _id_cclass_d;
          }
          return CCLASS;
        }
        case 'D': {
          if (is_ascii(_flags)) {
            reclass cls;
            add_ascii_digit_class(cls.literals);
            _cclass_id = _prog.add_class(cls);
          } else {
            if (_id_cclass_D < 0) {
              reclass cls = cclass_d;
              cls.literals.push_back({'\n', '\n'});
              _id_cclass_D = _prog.add_class(cls);
            }
            _cclass_id = _id_cclass_D;
          }
          return NCCLASS;
        }
        case 'b': return BOW;
        case 'B': return NBOW;
        case 'A': {
          _chr = chr;
          return BOL;
        }
        case 'Z': {
          _chr = chr;
          return EOL;
        }
        default: {
          // let valid escapable chars fall through as literal CHAR
          if (chr &&
              (std::find(escapable_chars.begin(), escapable_chars.end(), static_cast<char>(chr)) !=
               escapable_chars.end())) {
            break;
          }
          // anything else is a bad escape so throw an error
          CUDF_FAIL("invalid regex pattern: bad escape character at position " +
                    std::to_string(_expr_ptr - _pattern_begin - 1));
        }
      }  // end-switch
      _chr = chr;
      return CHAR;
    }

    // handle regex characters
    switch (chr) {
      case 0: return END;
      case '(':
        if (*_expr_ptr == '?' && *(_expr_ptr + 1) == ':')  // non-capturing group
        {
          _expr_ptr += 2;
          return LBRA_NC;
        }
        return (_capture == capture_groups::NON_CAPTURE) ? static_cast<int32_t>(LBRA_NC)
                                                         : static_cast<int32_t>(LBRA);
      case ')': return RBRA;
      case '^': {
        if (is_ext_newline(_flags)) {
          _chr = is_multiline(_flags) ? 'S' : 'N';
        } else {
          _chr = is_multiline(_flags) ? chr : '\n';
        }
        return BOL;
      }
      case '$': {
        if (is_ext_newline(_flags)) {
          _chr = is_multiline(_flags) ? 'S' : 'N';
        } else {
          _chr = is_multiline(_flags) ? chr : '\n';
        }
        return EOL;
      }
      case '[': return build_cclass();
      case '.': {
        _chr = is_ext_newline(_flags) ? 'N' : chr;
        return dot_type;
      }
    }

    if (std::find(quantifiers.begin(), quantifiers.end(), static_cast<char>(chr)) ==
        quantifiers.end()) {
      if (is_ignorecase(_flags)) {
        auto const swap_chr = swap_case(chr);
        _cclass_id =
          _prog.add_class(reclass{0, {reclass_range{chr, chr}, reclass_range{swap_chr, swap_chr}}});
        return CCLASS;
      }
      _chr = chr;
      return CHAR;
    }

    // The quantifiers require at least one "real" previous item.
    // We are throwing errors for invalid quantifiers.
    // Another option is to just return CHAR silently here which effectively
    // treats the chr character as a literal instead as a quantifier.
    // This could lead to confusion where sometimes unescaped quantifier characters
    // are treated as regex expressions and sometimes they are not.
    CUDF_EXPECTS(!_items.empty(), "invalid regex pattern: nothing to repeat at position 0");

    // handle alternation instruction
    if (chr == '|') return OR;

    // Check that the previous item can be used with quantifiers.
    // If the previous item is a capture group, we need to check items inside the
    // capture group can be used with quantifiers too.
    // (Note that capture groups can be nested).
    auto previous_type = _items.back().type;
    if (previous_type == RBRA) {  // previous item is a capture group
      // look for matching LBRA
      auto nested_count = 1;
      auto lbra_itr =
        std::find_if(_items.rbegin(), _items.rend(), [nested_count](auto const& item) mutable {
          auto const is_closing = (item.type == RBRA);
          auto const is_opening = (item.type == LBRA || item.type == LBRA_NC);
          nested_count += is_closing - is_opening;
          return is_opening && (nested_count == 0);
        });
      // search for the first valid item within the LBRA-RBRA range
      auto first_valid = std::find_first_of(
        _items.rbegin() + 1,
        lbra_itr,
        valid_preceding_inst_types.begin(),
        valid_preceding_inst_types.end(),
        [](auto const item, auto const valid_type) { return item.type == valid_type; });
      // set previous_type to be checked in next if-statement
      previous_type = (first_valid == lbra_itr) ? (--lbra_itr)->type : first_valid->type;
    }

    if (std::find(valid_preceding_inst_types.begin(),
                  valid_preceding_inst_types.end(),
                  previous_type) == valid_preceding_inst_types.end()) {
      CUDF_FAIL("invalid regex pattern: nothing to repeat at position " +
                std::to_string(_expr_ptr - _pattern_begin - 1));
    }

    // handle quantifiers
    switch (chr) {
      case '*':
        if (*_expr_ptr == '?') {
          _expr_ptr++;
          return STAR_LAZY;
        }
        return STAR;
      case '?':
        if (*_expr_ptr == '?') {
          _expr_ptr++;
          return QUEST_LAZY;
        }
        return QUEST;
      case '+':
        if (*_expr_ptr == '?') {
          _expr_ptr++;
          return PLUS_LAZY;
        }
        return PLUS;
      case '{':  // counted repetition: {n,m}
      {
        if (!std::isdigit(*_expr_ptr)) { break; }

        // transform char32 to char until null, delimiter, non-digit or end is reached;
        // returns the number of chars read/transformed
        auto transform_until = [](char32_t const* input,
                                  char32_t const* end,
                                  char* output,
                                  std::string_view const delimiters) -> int32_t {
          int32_t count = 0;
          while (*input != 0 && input < end) {
            auto const ch = static_cast<char>(*input++);
            // if ch not a digit or ch is a delimiter, we are done
            if (!std::isdigit(ch) || delimiters.find(ch) != delimiters.npos) { break; }
            output[count] = ch;
            ++count;
          }
          output[count] = 0;  // null-terminate (for the atoi call)
          return count;
        };

        constexpr auto max_read               = 4;    // 3 digits plus the delimiter
        constexpr auto max_value              = 999;  // support only 3 digits
        std::array<char, max_read + 1> buffer = {0};  //(max_read + 1);

        // get left-side (n) value => min_count
        auto bytes_read = transform_until(_expr_ptr, _expr_ptr + max_read, buffer.data(), "},");
        if (_expr_ptr[bytes_read] != '}' && _expr_ptr[bytes_read] != ',') {
          break;  // re-interpret as CHAR
        }
        auto count = std::atoi(buffer.data());
        CUDF_EXPECTS(
          count <= max_value,
          "unsupported repeat value at " + std::to_string(_expr_ptr - _pattern_begin - 1));
        _min_count = static_cast<int16_t>(count);

        auto const expr_ptr_save = _expr_ptr;  // save in case ending '}' is not found
        _expr_ptr += bytes_read;

        // get optional right-side (m) value => max_count
        _max_count = _min_count;
        if (*_expr_ptr++ == ',') {
          bytes_read = transform_until(_expr_ptr, _expr_ptr + max_read, buffer.data(), "}");
          if (_expr_ptr[bytes_read] != '}') {
            _expr_ptr = expr_ptr_save;  // abort, rollback and
            break;                      // re-interpret as CHAR
          }

          count = std::atoi(buffer.data());
          CUDF_EXPECTS(
            count <= max_value,
            "unsupported repeat value at " + std::to_string(_expr_ptr - _pattern_begin - 1));

          // {n,m} and {n,} are both valid
          _max_count = buffer[0] == 0 ? -1 : static_cast<int16_t>(count);
          _expr_ptr += bytes_read + 1;
        }

        // {n,m}? pattern is lazy counted quantifier
        if (*_expr_ptr == '?') {
          _expr_ptr++;
          return COUNTED_LAZY;
        }
        // otherwise, fixed counted quantifier
        return COUNTED;
      }
    }
    _chr = chr;
    return CHAR;
  }

  [[nodiscard]] std::vector<regex_parser::Item> expand_counted_items() const
  {
    std::vector<regex_parser::Item> const& in = _items;
    std::vector<regex_parser::Item> out;
    std::stack<int> lbra_stack;
    auto repeat_start_index = -1;

    for (auto const item : in) {
      if (item.type != COUNTED && item.type != COUNTED_LAZY) {
        out.push_back(item);
        if (item.type == LBRA || item.type == LBRA_NC) {
          lbra_stack.push(out.size() - 1);
          repeat_start_index = -1;
        } else if (item.type == RBRA) {
          repeat_start_index = lbra_stack.top();
          lbra_stack.pop();
        } else if ((item.type & ITEM_MASK) != OPERATOR_MASK) {
          repeat_start_index = out.size() - 1;
        }
      } else {
        // item is of type COUNTED or COUNTED_LAZY
        // here we repeat the previous item(s) based on the count range in item

        CUDF_EXPECTS(repeat_start_index >= 0, "regex: invalid counted quantifier location");

        // range of affected item(s) to repeat
        auto const begin = out.begin() + repeat_start_index;
        auto const end   = out.end();

        // count range values
        auto const n = item.d.count.n;  // minimum count
        auto const m = item.d.count.m;  // maximum count
        assert(n >= 0 && "invalid repeat count value n");
        std::vector<regex_parser::Item> repeat_copy(begin, end);
        // zero-repeat edge-case: need to erase the previous items
        if (n == 0) { out.erase(begin, end); }

        // special handling for quantified capture groups
        if ((n > 1) && (*begin).type == LBRA) {
          (*begin).type = LBRA_NC;  // change first one to non-capture
          // add intermediate groups as non-capture
          std::vector<regex_parser::Item> ncg_copy(begin, end);
          for (int j = 1; j < (n - 1); j++) {
            out.insert(out.end(), ncg_copy.begin(), ncg_copy.end());
          }
          // add the last entry as a regular capture-group
          out.insert(out.end(), repeat_copy.begin(), repeat_copy.end());
        } else {
          // minimum repeats (n)
          for (int j = 1; j < n; j++) {
            out.insert(out.end(), repeat_copy.begin(), repeat_copy.end());
          }
        }

        // optional maximum repeats (m)
        if (m >= 0) {
          for (int j = n; j < m; j++) {
            out.emplace_back(LBRA_NC, 0);
            out.insert(out.end(), repeat_copy.begin(), repeat_copy.end());
          }
          for (int j = n; j < m; j++) {
            out.emplace_back(RBRA, 0);
            out.emplace_back(item.type == COUNTED ? QUEST : QUEST_LAZY, 0);
          }
        } else {
          // infinite repeats
          if (n > 0) {  // append '+' after last repetition
            out.emplace_back(item.type == COUNTED ? PLUS : PLUS_LAZY, 0);
          } else {
            // copy it once then append '*'
            out.insert(out.end(), repeat_copy.begin(), repeat_copy.end());
            out.emplace_back(item.type == COUNTED ? STAR : STAR_LAZY, 0);
          }
        }
      }
    }
    return out;
  }

 public:
  regex_parser(char32_t const* pattern,
               regex_flags const flags,
               capture_groups const capture,
               reprog& prog)
    : _prog(prog), _pattern_begin(pattern), _expr_ptr(pattern), _flags(flags), _capture(capture)
  {
    auto const dot_type = is_dotall(_flags) ? ANYNL : ANY;

    int32_t type = 0;
    while ((type = lex(dot_type)) != END) {
      auto const item = [type, chr = _chr, cid = _cclass_id, n = _min_count, m = _max_count] {
        if (type == CCLASS || type == NCCLASS) return Item{type, cid};
        if (type == COUNTED || type == COUNTED_LAZY) return Item{type, n, m};
        return Item{type, chr};
      }();
      _items.push_back(item);
      if (type == COUNTED || type == COUNTED_LAZY) _has_counted = true;
    }
  }

  [[nodiscard]] std::vector<regex_parser::Item> get_items() const
  {
    return _has_counted ? expand_counted_items() : _items;
  }
};

/**
 * @brief The compiler converts class list into instructions.
 */
class regex_compiler {
  struct and_node {
    int id_first;
    int id_last;
  };

  struct re_operator {
    int t;
    int subid;
  };

  reprog& _prog;
  std::stack<and_node> _and_stack;
  std::stack<re_operator> _operator_stack;
  bool _last_was_and{false};
  int _bracket_count{0};
  regex_flags _flags;

  inline void push_and(int first, int last) { _and_stack.push({first, last}); }

  inline and_node pop_and()
  {
    if (_and_stack.empty()) {
      auto const inst_id = _prog.add_inst(NOP);
      push_and(inst_id, inst_id);
    }
    auto const node = _and_stack.top();
    _and_stack.pop();
    return node;
  }

  inline void push_operator(int token, int subid = 0)
  {
    _operator_stack.push(re_operator{token, subid});
  }

  inline re_operator const pop_operator()
  {
    auto const op = _operator_stack.top();
    _operator_stack.pop();
    return op;
  }

  void eval_until(int min_token)
  {
    while (min_token == RBRA || _operator_stack.top().t >= min_token) {
      auto const op = pop_operator();
      switch (op.t) {
        default:
          // unknown operator
          break;
        case LBRA:  // expects matching RBRA
        {
          auto const operand                        = pop_and();
          auto const id_inst2                       = _prog.add_inst(RBRA);
          _prog.inst_at(id_inst2).u1.subid          = op.subid;
          _prog.inst_at(operand.id_last).u2.next_id = id_inst2;
          auto const id_inst1                       = _prog.add_inst(LBRA);
          _prog.inst_at(id_inst1).u1.subid          = op.subid;
          _prog.inst_at(id_inst1).u2.next_id        = operand.id_first;
          push_and(id_inst1, id_inst2);
          return;
        }
        case OR: {
          auto const operand2                        = pop_and();
          auto const operand1                        = pop_and();
          auto const id_inst2                        = _prog.add_inst(NOP);
          _prog.inst_at(operand2.id_last).u2.next_id = id_inst2;
          _prog.inst_at(operand1.id_last).u2.next_id = id_inst2;
          auto const id_inst1                        = _prog.add_inst(OR);
          _prog.inst_at(id_inst1).u1.right_id        = operand1.id_first;
          _prog.inst_at(id_inst1).u2.left_id         = operand2.id_first;
          push_and(id_inst1, id_inst2);
          break;
        }
        case CAT: {
          auto const operand2                        = pop_and();
          auto const operand1                        = pop_and();
          _prog.inst_at(operand1.id_last).u2.next_id = operand2.id_first;
          push_and(operand1.id_first, operand2.id_last);
          break;
        }
        case STAR: {
          auto const operand                        = pop_and();
          auto const id_inst1                       = _prog.add_inst(OR);
          _prog.inst_at(operand.id_last).u2.next_id = id_inst1;
          _prog.inst_at(id_inst1).u1.right_id       = operand.id_first;
          push_and(id_inst1, id_inst1);
          break;
        }
        case STAR_LAZY: {
          auto const operand                        = pop_and();
          auto const id_inst1                       = _prog.add_inst(OR);
          auto const id_inst2                       = _prog.add_inst(NOP);
          _prog.inst_at(operand.id_last).u2.next_id = id_inst1;
          _prog.inst_at(id_inst1).u2.left_id        = operand.id_first;
          _prog.inst_at(id_inst1).u1.right_id       = id_inst2;
          push_and(id_inst1, id_inst2);
          break;
        }
        case PLUS: {
          auto const operand                        = pop_and();
          auto const id_inst1                       = _prog.add_inst(OR);
          _prog.inst_at(operand.id_last).u2.next_id = id_inst1;
          _prog.inst_at(id_inst1).u1.right_id       = operand.id_first;
          push_and(operand.id_first, id_inst1);
          break;
        }
        case PLUS_LAZY: {
          auto const operand                        = pop_and();
          auto const id_inst1                       = _prog.add_inst(OR);
          auto const id_inst2                       = _prog.add_inst(NOP);
          _prog.inst_at(operand.id_last).u2.next_id = id_inst1;
          _prog.inst_at(id_inst1).u2.left_id        = operand.id_first;
          _prog.inst_at(id_inst1).u1.right_id       = id_inst2;
          push_and(operand.id_first, id_inst2);
          break;
        }
        case QUEST: {
          auto const operand                        = pop_and();
          auto const id_inst1                       = _prog.add_inst(OR);
          auto const id_inst2                       = _prog.add_inst(NOP);
          _prog.inst_at(id_inst1).u2.left_id        = id_inst2;
          _prog.inst_at(id_inst1).u1.right_id       = operand.id_first;
          _prog.inst_at(operand.id_last).u2.next_id = id_inst2;
          push_and(id_inst1, id_inst2);
          break;
        }
        case QUEST_LAZY: {
          auto const operand                        = pop_and();
          auto const id_inst1                       = _prog.add_inst(OR);
          auto const id_inst2                       = _prog.add_inst(NOP);
          _prog.inst_at(id_inst1).u2.left_id        = operand.id_first;
          _prog.inst_at(id_inst1).u1.right_id       = id_inst2;
          _prog.inst_at(operand.id_last).u2.next_id = id_inst2;
          push_and(id_inst1, id_inst2);
          break;
        }
      }
    }
  }

  void handle_operator(int token, int subid = 0)
  {
    if (token == RBRA && --_bracket_count < 0) {
      // unmatched right paren
      return;
    }
    if (token == LBRA) {
      _bracket_count++;
      if (_last_was_and) { handle_operator(CAT, subid); }
    } else {
      eval_until(token);
    }
    if (token != RBRA) { push_operator(token, subid); }

    static std::vector<int> const tokens{STAR, STAR_LAZY, QUEST, QUEST_LAZY, PLUS, PLUS_LAZY, RBRA};
    _last_was_and =
      std::any_of(tokens.cbegin(), tokens.cend(), [token](auto t) { return t == token; });
  }

  void handle_operand(int token, int subid = 0, char32_t yy = 0, int class_id = 0)
  {
    if (_last_was_and) { handle_operator(CAT, subid); }  // catenate is implicit

    auto const inst_id = _prog.add_inst(token);
    if (token == CCLASS || token == NCCLASS) {
      _prog.inst_at(inst_id).u1.cls_id = class_id;
    } else if (token == CHAR) {
      _prog.inst_at(inst_id).u1.c = yy;
    } else if (token == BOL || token == EOL || token == ANY) {
      _prog.inst_at(inst_id).u1.c = yy;
    }
    push_and(inst_id, inst_id);
    _last_was_and = true;
  }

 public:
  regex_compiler(char32_t const* pattern,
                 regex_flags const flags,
                 capture_groups const capture,
                 reprog& prog)
    : _prog(prog), _flags(flags)
  {
    // Parse pattern into items
    auto const items = regex_parser(pattern, _flags, capture, _prog).get_items();

    int cur_subid{};
    int push_subid{};

    // Start with a low priority operator
    push_operator(START - 1);

    for (auto const item : items) {
      auto token = item.type;

      if (token == LBRA) {
        ++cur_subid;
        push_subid = cur_subid;
      } else if (token == LBRA_NC) {
        push_subid = 0;
        token      = LBRA;
      }

      if ((token & ITEM_MASK) == OPERATOR_MASK) {
        handle_operator(token, push_subid);
      } else {
        handle_operand(token, push_subid, item.d.chr, item.d.cclass_id);
      }
    }

    // Close with a low priority operator
    eval_until(START);
    // Force END
    handle_operand(END, push_subid);
    eval_until(START);

    CUDF_EXPECTS(_bracket_count == 0, "unmatched left parenthesis");

    _prog.set_start_inst(_and_stack.top().id_first);
    _prog.optimize();
    _prog.check_for_errors();
    _prog.finalize();
    _prog.set_groups_count(cur_subid);
  }
};

// Convert pattern into program
reprog reprog::create_from(std::string_view pattern,
                           regex_flags const flags,
                           capture_groups const capture)
{
  reprog rtn(flags);
  auto pattern32 = string_to_char32_vector(pattern);
  regex_compiler const compiler(pattern32.data(), flags, capture, rtn);
  // for debugging, it can be helpful to call rtn.print() here to dump
  // out the instructions that have been created from the given pattern
  return rtn;
}

reprog::reprog(regex_flags flags) : _flags{flags} {}

void reprog::optimize() { collapse_nops(); }

void reprog::finalize() { build_start_ids(); }

void reprog::collapse_nops()
{
  // treat non-capturing LBRAs/RBRAs as NOP
  std::transform(_insts.begin(), _insts.end(), _insts.begin(), [](auto inst) {
    if ((inst.type == LBRA || inst.type == RBRA) && (inst.u1.subid < 1)) { inst.type = NOP; }
    return inst;
  });

  // functor for finding the next valid op
  auto find_next_op = [insts = _insts](int id) {
    while (insts[id].type == NOP) {
      id = insts[id].u2.next_id;
    }
    return id;
  };

  // create new routes around NOP chains
  std::transform(_insts.begin(), _insts.end(), _insts.begin(), [find_next_op](auto inst) {
    if (inst.type != NOP) {
      inst.u2.next_id = find_next_op(inst.u2.next_id);
      if (inst.type == OR) { inst.u1.right_id = find_next_op(inst.u1.right_id); }
    }
    return inst;
  });

  // find starting op
  _startinst_id = find_next_op(_startinst_id);

  // build a map of op ids
  // these are used to fix up the ids after the NOPs are removed
  std::vector<int> id_map(insts_count());
  std::transform_exclusive_scan(
    _insts.begin(), _insts.end(), id_map.begin(), 0, std::plus<int>{}, [](auto inst) {
      return static_cast<int>(inst.type != NOP);
    });

  // remove the NOP instructions
  auto end = std::remove_if(_insts.begin(), _insts.end(), [](auto i) { return i.type == NOP; });
  _insts.resize(std::distance(_insts.begin(), end));

  // fix up the ids on the remaining instructions using the id_map
  std::transform(_insts.begin(), _insts.end(), _insts.begin(), [id_map](auto inst) {
    inst.u2.next_id = id_map[inst.u2.next_id];
    if (inst.type == OR) { inst.u1.right_id = id_map[inst.u1.right_id]; }
    return inst;
  });

  // fix up the start instruction id too
  _startinst_id = id_map[_startinst_id];
}

// expand leading ORs to multiple startinst_ids
void reprog::build_start_ids()
{
  _startinst_ids.clear();
  std::stack<int> ids;
  ids.push(_startinst_id);
  while (!ids.empty()) {
    int const id = ids.top();
    ids.pop();
    reinst const& inst = _insts[id];
    if (inst.type == OR) {
      if (inst.u2.left_id != id)  // prevents infinite while-loop here
        ids.push(inst.u2.left_id);
      if (inst.u1.right_id != id)  // prevents infinite while-loop here
        ids.push(inst.u1.right_id);
    } else {
      _startinst_ids.push_back(id);
    }
  }
  _startinst_ids.push_back(-1);  // terminator mark
}

/**
 * @brief Check a specific instruction for errors.
 *
 * Currently this is checking for an infinite-loop condition as documented in this issue:
 * https://github.com/rapidsai/cudf/issues/10006
 *
 * Example instructions list created from pattern `(A?)+`
 * ```
 *   0:    CHAR c='A', next=2
 *   1:      OR right=0, left=2, next=2
 *   2:    RBRA id=1, next=4
 *   3:    LBRA id=1, next=1
 *   4:      OR right=3, left=5, next=5
 *   5:     END
 * ```
 *
 * Following the example above, the instruction at `id==1` (OR)
 * is being checked. If the instruction path returns to `id==1`
 * without including the `0==CHAR` or `5==END` as in this example,
 * then this would cause the runtime to go into an infinite-loop.
 *
 * It appears this example pattern is not valid. But Python interprets
 * its behavior similarly to pattern `(A*)`. Handling this in the same
 * way does not look feasible with the current implementation.
 *
 * @throw cudf::logic_error if instruction logic error is found
 *
 * @param id Instruction to check if repeated.
 * @param next_id Next instruction to process.
 */
void reprog::check_for_errors(int32_t id, int32_t next_id)
{
  auto inst = inst_at(next_id);
  while (inst.type == LBRA || inst.type == RBRA) {
    next_id = inst.u2.next_id;
    inst    = inst_at(next_id);
  }
  if (inst.type == OR) {
    CUDF_EXPECTS(next_id != id, "Unsupported regex pattern");
    check_for_errors(id, inst.u2.left_id);
    check_for_errors(id, inst.u1.right_id);
  }
}

/**
 * @brief Check regex instruction set for any errors.
 *
 * Currently, this checks for OR instructions that eventually point back to themselves with only
 * intervening capture group instructions between causing an infinite-loop during runtime
 * evaluation.
 */
void reprog::check_for_errors()
{
  for (auto id = 0; id < insts_count(); ++id) {
    auto const inst = inst_at(id);
    if (inst.type == OR) {
      check_for_errors(id, inst.u2.left_id);
      check_for_errors(id, inst.u1.right_id);
    }
  }
}

std::pair<literal_fast_path, std::string> reprog::check_for_literal_fast_path() const
{
  if (_flags != regex_flags::DEFAULT) { return {literal_fast_path::NONE, {}}; }
  if (_startinst_ids.size() > 2) { return {literal_fast_path::NONE, {}}; }
  auto const count = static_cast<size_type>(_insts.size());
  if (count < 2) { return {literal_fast_path::NONE, {}}; }

  auto inst = _insts[_startinst_id];

  // Optional BOL at the start of the pattern
  bool const has_bol = (inst.type == BOL);
  if (has_bol) {
    auto const id = inst.u2.next_id;
    if (id < 0 || id >= count) { return {literal_fast_path::NONE, {}}; }
    inst = _insts[id];
  }

  // Accumulate sequential CHAR bytes
  std::string literal;
  while (inst.type == CHAR && inst.u1.c != 0) {
    std::array<char, 5> utf8                     = {};
    utf8[from_char_utf8(inst.u1.c, utf8.data())] = 0;
    literal += utf8.data();
    auto const id = inst.u2.next_id;
    if (id < 0 || id >= count) { return {literal_fast_path::NONE, {}}; }
    inst = _insts[id];
  }
  if (literal.empty()) { return {literal_fast_path::NONE, {}}; }

  // If we are at END then we are literal-only or starts-with.
  if (inst.type == END) {
    return {has_bol ? literal_fast_path::STARTS_WITH : literal_fast_path::LITERAL_ONLY,
            std::move(literal)};
  }
  // Final check for ends-with: EOL followed by END
  if (!has_bol && inst.type == EOL && inst.u1.c == 'Z') {
    auto const id = inst.u2.next_id;
    if (id >= 0 && id < count && _insts[id].type == END) {
      return {literal_fast_path::ENDS_WITH, std::move(literal)};
    }
  }
  return {literal_fast_path::NONE, {}};
}

match_flags reprog::compute_match_flags() const
{
  static std::unordered_set<int> const non_consuming_inst_types{
    OR, BOL, EOL, BOW, NBOW, LBRA, RBRA};

  auto check_paths = [this](auto&& self, int id, std::unordered_set<int>& visited) -> bool {
    if (id < 0 || !std::get<1>(visited.insert(id))) { return false; }
    auto const& inst = _insts[id];
    if (inst.type == END) { return false; }
    if (non_consuming_inst_types.find(inst.type) == non_consuming_inst_types.end()) { return true; }
    if (inst.type == OR) {
      return self(self, inst.u2.left_id, visited) && self(self, inst.u1.right_id, visited);
    }
    return self(self, inst.u2.next_id, visited);
  };

  bool found_non_consuming_path = false;
  for (auto start : _startinst_ids) {
    if (start == -1) break;
    std::unordered_set<int> visited;
    if (!check_paths(check_paths, start, visited)) {
      found_non_consuming_path = true;
      break;
    }
  }
  return found_non_consuming_path ? match_flags::EMPTY_MATCH : match_flags::NONE;
}

#ifndef NDEBUG
void reprog::print() const
{
  printf("Flags = 0x%08x\n", static_cast<uint32_t>(_flags));
  printf("Instructions:\n");
  for (std::size_t i = 0; i < _insts.size(); i++) {
    reinst const& inst = _insts[i];
    printf("%3zu: ", i);
    switch (inst.type) {
      default: printf("Unknown instruction: %d, next=%d", inst.type, inst.u2.next_id); break;
      case CHAR:
        if (inst.u1.c <= 32 || inst.u1.c >= 127) {
          printf("   CHAR c='0x%02x', next=%d", static_cast<unsigned>(inst.u1.c), inst.u2.next_id);
        } else {
          printf("   CHAR c='%c', next=%d", inst.u1.c, inst.u2.next_id);
        }
        break;
      case RBRA: printf("   RBRA id=%d, next=%d", inst.u1.subid, inst.u2.next_id); break;
      case LBRA: printf("   LBRA id=%d, next=%d", inst.u1.subid, inst.u2.next_id); break;
      case OR:
        printf(
          "     OR right=%d, left=%d, next=%d", inst.u1.right_id, inst.u2.left_id, inst.u2.next_id);
        break;
      case STAR: printf("   STAR next=%d", inst.u2.next_id); break;
      case PLUS: printf("   PLUS next=%d", inst.u2.next_id); break;
      case QUEST: printf("  QUEST next=%d", inst.u2.next_id); break;
      case ANY: printf("    ANY '%c', next=%d", inst.u1.c, inst.u2.next_id); break;
      case ANYNL: printf("  ANYNL next=%d", inst.u2.next_id); break;
      case NOP: printf("    NOP next=%d", inst.u2.next_id); break;
      case BOL: {
        printf("    BOL c=");
        if (inst.u1.c == '\n') {
          printf("'\\n'");
        } else {
          printf("'%c'", inst.u1.c);
        }
        printf(", next=%d", inst.u2.next_id);
        break;
      }
      case EOL: {
        printf("    EOL c=");
        if (inst.u1.c == '\n') {
          printf("'\\n'");
        } else {
          printf("'%c'", inst.u1.c);
        }
        printf(", next=%d", inst.u2.next_id);
        break;
      }
      case CCLASS: printf(" CCLASS cls=%d , next=%d", inst.u1.cls_id, inst.u2.next_id); break;
      case NCCLASS: printf("NCCLASS cls=%d, next=%d", inst.u1.cls_id, inst.u2.next_id); break;
      case BOW: printf("    BOW next=%d", inst.u2.next_id); break;
      case NBOW: printf("   NBOW next=%d", inst.u2.next_id); break;
      case END: printf("    END"); break;
    }
    printf("\n");
  }

  printf("startinst_id=%d\n", _startinst_id);
  if (_startinst_ids.size() > 0) {
    printf("startinst_ids: [");
    for (std::size_t i = 0; i < _startinst_ids.size(); i++) {
      printf(" %d", _startinst_ids[i]);
    }
    printf("]\n");
  }

  int count = static_cast<int>(_classes.size());
  printf("\nClasses %d\n", count);
  for (int i = 0; i < count; i++) {
    reclass const& cls = _classes[i];
    auto const size    = static_cast<int>(cls.literals.size());
    printf("%2d: ", i);
    for (int j = 0; j < size; ++j) {
      auto const l = cls.literals[j];
      char32_t c1  = l.first;
      char32_t c2  = l.last;
      if (c1 <= 32 || c1 >= 127 || c2 <= 32 || c2 >= 127) {
        printf("0x%02x-0x%02x", static_cast<unsigned>(c1), static_cast<unsigned>(c2));
      } else {
        printf("%c-%c", static_cast<char>(c1), static_cast<char>(c2));
      }
      if ((j + 1) < size) { printf(", "); }
    }
    printf("\n");
    if (cls.builtins) {
      int mask = cls.builtins;
      printf("   builtins(x%02X):", static_cast<unsigned>(mask));
      if (mask & CCLASS_W) printf(" \\w");
      if (mask & CCLASS_S) printf(" \\s");
      if (mask & CCLASS_D) printf(" \\d");
      if (mask & NCCLASS_W) printf(" \\W");
      if (mask & NCCLASS_S) printf(" \\S");
      if (mask & NCCLASS_D) printf(" \\D");
    }
    printf("\n");
  }
  if (_num_capturing_groups) { printf("Number of capturing groups: %d\n", _num_capturing_groups); }

  auto [fp, literal] = check_for_literal_fast_path();
  switch (fp) {
    case literal_fast_path::LITERAL_ONLY: printf("literal-only: %s\n", literal.c_str()); break;
    case literal_fast_path::STARTS_WITH: printf("starts-with: %s\n", literal.c_str()); break;
    case literal_fast_path::ENDS_WITH: printf("ends-with: %s\n", literal.c_str()); break;
    default: break;
  }
}
#endif

}  // namespace detail
}  // namespace strings
}  // namespace cudf
