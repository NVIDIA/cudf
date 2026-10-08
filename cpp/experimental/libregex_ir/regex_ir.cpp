/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_codegen.hpp"
#include "regex_ir_detail.hpp"
#include "regex_ir_unicode.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <format>
#include <iterator>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

// parser and automata IR

namespace regex_ir {
namespace {

struct compile_failure : std::exception {
  compile_failure(source_span source, std::string message)
    : source{source}, message{std::move(message)}
  {
  }

  source_span source;
  std::string message;
};

enum class node_kind : std::uint8_t {
  EMPTY       = 0,
  PREDICATE   = 1,
  CONCATENATE = 2,
  ALTERNATE   = 3,
  REPEAT      = 4,
  GROUP       = 5,
  ASSERTION   = 6,
};

struct node {
  node_kind kind                              = node_kind::EMPTY;
  source_span source                          = source_span{};
  character_predicate predicate               = character_predicate{};
  assertion_kind assertion                    = assertion_kind::BEGIN_INPUT;
  std::vector<std::unique_ptr<node>> children = std::vector<std::unique_ptr<node>>{};
  std::uint32_t minimum                       = 0;
  std::uint32_t maximum                       = 0;
  bool greedy                                 = true;
  std::uint32_t capture_index                 = 0;
  bool capturing                              = false;
};

bool can_consume_character(node const& value)
{
  switch (value.kind) {
    case node_kind::PREDICATE: return true;
    case node_kind::EMPTY:
    case node_kind::ASSERTION: return false;
    case node_kind::GROUP: return can_consume_character(*value.children.front());
    case node_kind::CONCATENATE:
    case node_kind::ALTERNATE:
      return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
        return can_consume_character(*child);
      });
    case node_kind::REPEAT: return can_consume_character(*value.children.front());
  }
  return false;
}

bool is_unconditional_empty(node const& value)
{
  switch (value.kind) {
    case node_kind::EMPTY: return true;
    case node_kind::GROUP: return is_unconditional_empty(*value.children.front());
    case node_kind::CONCATENATE:
    case node_kind::ALTERNATE:
      return std::all_of(value.children.begin(), value.children.end(), [](auto& child) {
        return is_unconditional_empty(*child);
      });
    case node_kind::PREDICATE:
    case node_kind::REPEAT:
    case node_kind::ASSERTION: return false;
  }
  return false;
}

bool can_match_empty(node const& value)
{
  switch (value.kind) {
    case node_kind::EMPTY:
    case node_kind::ASSERTION: return true;
    case node_kind::PREDICATE: return false;
    case node_kind::GROUP: return can_match_empty(*value.children.front());
    case node_kind::CONCATENATE:
      return std::all_of(value.children.begin(), value.children.end(), [](auto& child) {
        return can_match_empty(*child);
      });
    case node_kind::ALTERNATE:
      return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
        return can_match_empty(*child);
      });
    case node_kind::REPEAT: return value.minimum == 0 || can_match_empty(*value.children.front());
  }
  return false;
}

bool contains_capture(node const& value)
{
  if (value.kind == node_kind::GROUP && value.capturing) return true;
  return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
    return contains_capture(*child);
  });
}

void normalize_ranges(character_predicate& predicate)
{
  if (predicate.ranges.empty()) { return; }

  std::sort(predicate.ranges.begin(), predicate.ranges.end(), [](auto& lhs, auto& rhs) {
    return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.last < rhs.last);
  });

  std::vector<codepoint_range> merged;
  for (auto range : predicate.ranges) {
    if (merged.empty() || static_cast<std::uint64_t>(range.first) >
                            static_cast<std::uint64_t>(merged.back().last) + 1U) {
      merged.push_back(range);
    } else if (range.last > merged.back().last) {
      merged.back().last = range.last;
    }
  }

  predicate.ranges = std::move(merged);
}

template <std::size_t Size>
void append_unicode_ranges(character_predicate& predicate, unicode_data_range const (&ranges)[Size])
{
  predicate.ranges.reserve(predicate.ranges.size() + Size);
  for (unicode_data_range range : ranges) {
    predicate.ranges.push_back(
      {static_cast<char32_t>(range.first), static_cast<char32_t>(range.last)});
  }
}

std::vector<codepoint_range> complement_ranges(std::vector<codepoint_range> ranges)
{
  character_predicate normalized;
  normalized.ranges = std::move(ranges);
  normalize_ranges(normalized);

  std::vector<codepoint_range> result;
  char32_t begin = U'\0';
  for (codepoint_range range : normalized.ranges) {
    if (begin < range.first) result.push_back({begin, static_cast<char32_t>(range.first - 1)});
    if (range.last == static_cast<char32_t>(0x10FFFF)) return result;
    begin = static_cast<char32_t>(range.last + 1);
  }

  result.push_back({begin, static_cast<char32_t>(0x10FFFF)});
  return result;
}

void remove_codepoint(std::vector<codepoint_range>& ranges, char32_t value)
{
  std::vector<codepoint_range> result;
  result.reserve(ranges.size() + 1);

  for (codepoint_range range : ranges) {
    if (value < range.first || value > range.last) {
      result.push_back(range);
      continue;
    }
    if (range.first < value) result.push_back({range.first, static_cast<char32_t>(value - 1)});
    if (value < range.last) result.push_back({static_cast<char32_t>(value + 1), range.last});
  }

  ranges = std::move(result);
}

bool append_posix_class(character_predicate& predicate, std::string_view name)
{
  if (name == "alpha") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'A', U'Z'}, {U'a', U'z'}});
  } else if (name == "alnum") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'0', U'9'}, {U'A', U'Z'}, {U'a', U'z'}});
  } else if (name == "digit") {
    predicate.ranges.push_back({U'0', U'9'});
  } else if (name == "xdigit") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'0', U'9'}, {U'A', U'F'}, {U'a', U'f'}});
  } else if (name == "space") {
    predicate.ranges.insert(predicate.ranges.end(), {{U'\t', U'\r'}, {U' ', U' '}});
  } else if (name == "word") {
    predicate.ranges.insert(predicate.ranges.end(),
                            {{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}});
  } else if (name == "punct") {
    predicate.ranges.insert(predicate.ranges.end(),
                            {{U'!', U'/'}, {U':', U'@'}, {U'[', U'`'}, {U'{', U'~'}});
  } else {
    return false;
  }
  return true;
}

char32_t swap_case(char32_t value)
{
  static std::locale locale{"C.UTF-8"};
  auto wide = static_cast<wchar_t>(value);
  return static_cast<char32_t>(std::isupper(wide, locale) ? std::tolower(wide, locale)
                                                          : std::toupper(wide, locale));
}

void add_case_pair(character_predicate& predicate, char32_t first, char32_t last)
{
  predicate.ranges.push_back({first, last});
  auto swapped_first = swap_case(first);
  auto swapped_last  = swap_case(last);
  if (swapped_first <= swapped_last) predicate.ranges.push_back({swapped_first, swapped_last});
}

class parser {
 public:
  parser(std::string_view pattern, compile_options const& options)
    : pattern_(pattern), options_(options)
  {
  }

  std::unique_ptr<node> parse()
  {
    if (pattern_.size() > options_.limits.max_pattern_bytes) {
      fail({0, pattern_.size()}, "pattern exceeds max_pattern_bytes");
    }

    auto expression = parse_alternation();
    if (position_ != pattern_.size()) { fail({position_, 1}, "unexpected token"); }

    return expression;
  }

  std::uint32_t capture_count = 0;

 private:
  [[noreturn]] void fail(source_span span, std::string message)
  {
    throw compile_failure{span, std::move(message)};
  }

  [[nodiscard]] bool at_end() const noexcept { return position_ >= pattern_.size(); }

  [[nodiscard]] char peek() const noexcept { return at_end() ? '\0' : pattern_[position_]; }

  char take()
  {
    if (at_end()) { fail({position_, 0}, "unexpected end of pattern"); }
    return pattern_[position_++];
  }

  bool consume(char value)
  {
    if (peek() != value) { return false; }
    ++position_;
    return true;
  }

  std::unique_ptr<node> make(node_kind kind, std::size_t start)
  {
    auto result    = std::make_unique<node>();
    result->kind   = kind;
    result->source = {start, position_ - start};
    return result;
  }

  std::unique_ptr<node> parse_alternation()
  {
    auto lhs = parse_concatenation();
    while (consume('|')) {
      auto separator = position_ - 1;
      auto rhs       = parse_concatenation();
      auto alternate = make(node_kind::ALTERNATE, lhs->source.offset);
      if (lhs->kind == node_kind::EMPTY && rhs->kind == node_kind::EMPTY) {
        fail({separator, 1}, "empty alternation");
      }
      alternate->children.push_back(std::move(lhs));
      alternate->children.push_back(std::move(rhs));
      alternate->source.length = position_ - alternate->source.offset;
      lhs                      = std::move(alternate);
    }
    return lhs;
  }

  std::unique_ptr<node> parse_concatenation()
  {
    auto start = position_;
    std::vector<std::unique_ptr<node>> children;
    while (!at_end() && peek() != ')' && peek() != '|') {
      children.push_back(parse_quantified());
    }
    if (children.empty()) { return make(node_kind::EMPTY, start); }
    if (children.size() == 1) { return std::move(children.front()); }
    auto result      = make(node_kind::CONCATENATE, start);
    result->children = std::move(children);
    return result;
  }

  std::uint32_t parse_decimal()
  {
    std::uint64_t value{};
    std::size_t count{};
    while (std::isdigit(static_cast<std::uint8_t>(peek())) != 0) {
      value = value * 10U + static_cast<std::uint32_t>(take() - '0');
      ++count;
      if (value > options_.limits.max_repeat) {
        fail({position_ - count, count}, "repeat bound exceeds max_repeat");
      }
    }
    if (count == 0) { fail({position_, 0}, "expected repeat bound"); }
    return static_cast<std::uint32_t>(value);
  }

  std::unique_ptr<node> parse_quantified()
  {
    auto atom = parse_atom();
    if (at_end()) { return atom; }

    auto start = atom->source.offset;
    std::uint32_t minimum{};
    std::uint32_t maximum{};
    bool quantified = true;
    if (consume('*')) {
      minimum = 0;
      maximum = unbounded_repeat;
    } else if (consume('+')) {
      minimum = 1;
      maximum = unbounded_repeat;
    } else if (consume('?')) {
      minimum = 0;
      maximum = 1;
    } else if (peek() == '{' && position_ + 1 < pattern_.size() &&
               std::isdigit(static_cast<std::uint8_t>(pattern_[position_ + 1])) != 0) {
      ++position_;
      minimum = parse_decimal();
      maximum = minimum;
      if (consume(',')) { maximum = peek() == '}' ? unbounded_repeat : parse_decimal(); }
      if (!consume('}')) { fail({position_, 0}, "unterminated quantifier"); }
      if (maximum != unbounded_repeat && maximum < minimum) {
        fail({start, position_ - start}, "repeat maximum is smaller than minimum");
      }
    } else {
      quantified = false;
    }

    if (!quantified) { return atom; }
    auto greedy = !consume('?');
    if (peek() == '*' || peek() == '+' || peek() == '?' || peek() == '{') {
      fail({position_, 1}, "multiple repeat operators");
    }
    if (is_unconditional_empty(*atom) && !contains_capture(*atom)) {
      // repeating an unconditional empty expression is still empty
      atom->source.length = position_ - start;
      return atom;
    }
    if (!can_consume_character(*atom)) {
      fail({start, position_ - start}, "zero-width assertions cannot be repeated");
    }
    auto result = make(node_kind::REPEAT, start);
    result->children.push_back(std::move(atom));
    result->minimum       = minimum;
    result->maximum       = maximum;
    result->greedy        = greedy;
    result->source.length = position_ - start;
    return result;
  }

  char32_t decode_literal(std::size_t& length)
  {
    if (options_.characters == character_mode::BYTES) {
      length = 1;
      return static_cast<std::uint8_t>(pattern_[position_]);
    }
    auto first = static_cast<std::uint8_t>(pattern_[position_]);
    if (first < 0x80U) {
      length = 1;
      return first;
    }
    std::size_t count = first < 0xE0U ? 2 : (first < 0xF0U ? 3 : 4);
    if (position_ + count > pattern_.size() || first < 0xC2U || first > 0xF4U) {
      fail({position_, 1}, "invalid UTF-8 in pattern");
    }
    char32_t value = first & (count == 2 ? 0x1FU : (count == 3 ? 0x0FU : 0x07U));
    for (std::size_t index = 1; index < count; ++index) {
      auto next = static_cast<std::uint8_t>(pattern_[position_ + index]);
      if ((next & 0xC0U) != 0x80U) { fail({position_, count}, "invalid UTF-8 in pattern"); }
      value = static_cast<char32_t>((value << 6U) | (next & 0x3FU));
    }
    if ((count == 3 && value < 0x800U) || (count == 4 && value < 0x10000U) || value > 0x10FFFFU ||
        (value >= 0xD800U && value <= 0xDFFFU)) {
      fail({position_, count}, "invalid UTF-8 scalar value");
    }
    length = count;
    return value;
  }

  char32_t parse_hex(std::size_t digits, std::size_t escape_start)
  {
    char32_t value{};
    for (std::size_t index = 0; index < digits; ++index) {
      if (at_end()) {
        fail({escape_start, position_ - escape_start}, "truncated hexadecimal escape");
      }
      char const digit = take();
      value <<= 4U;
      if (digit >= '0' && digit <= '9') {
        value |= static_cast<char32_t>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        value |= static_cast<char32_t>(digit - 'a' + 10);
      } else if (digit >= 'A' && digit <= 'F') {
        value |= static_cast<char32_t>(digit - 'A' + 10);
      } else {
        fail({position_ - 1, 1}, "invalid hexadecimal digit");
      }
    }
    return value;
  }

  char32_t parse_octal(char first)
  {
    auto value         = static_cast<char32_t>(first - '0');
    std::size_t digits = 1;
    while (digits < 3 && peek() >= '0' && peek() <= '7') {
      value = static_cast<char32_t>((value << 3U) | static_cast<char32_t>(take() - '0'));
      ++digits;
    }
    return value;
  }

  character_predicate predefined(char value)
  {
    character_predicate result;
    auto append_ascii = [&](char kind) {
      if (kind == 'd') result.ranges = {{U'0', U'9'}};
      if (kind == 'w') { result.ranges = {{U'0', U'9'}, {U'A', U'Z'}, {U'_', U'_'}, {U'a', U'z'}}; }
      if (kind == 's') result.ranges = {{U'\t', U' '}};
    };
    auto append_unicode = [&](char kind) {
      if (kind == 'd') append_unicode_ranges(result, unicode_digit_ranges);
      if (kind == 'w') {
        append_unicode_ranges(result, unicode_word_ranges);
        result.ranges.push_back({U'_', U'_'});
      }
      if (kind == 's') append_unicode_ranges(result, unicode_space_ranges);
    };
    auto base = static_cast<char>(std::tolower(static_cast<std::uint8_t>(value)));
    if (options_.ascii_classes) {
      append_ascii(base);
    } else {
      append_unicode(base);
    }
    normalize_ranges(result);
    bool negative = value == 'D' || value == 'W' || value == 'S';
    if (negative) {
      result.ranges = complement_ranges(std::move(result.ranges));
      if (!options_.ascii_classes && (value == 'D' || value == 'W')) {
        remove_codepoint(result.ranges, U'\n');
      }
    }
    switch (value) {
      case 'd':
      case 'D':
        result.recognized = value == 'd' ? predicate_class::DIGIT : predicate_class::NOT_DIGIT;
        break;
      case 'w':
      case 'W':
        result.recognized = value == 'w' ? predicate_class::WORD : predicate_class::NOT_WORD;
        break;
      case 's':
      case 'S':
        result.recognized = value == 's' ? predicate_class::SPACE : predicate_class::NOT_SPACE;
        break;
      default: break;
    }
    return result;
  }

  char32_t escaped_literal(char value, std::size_t start)
  {
    switch (value) {
      case 'a': return U'\a';
      case 'b': return U'\b';
      case 'n': return U'\n';
      case 'r': return U'\r';
      case 't': return U'\t';
      case 'f': return U'\f';
      case 'v': return U'\v';
      case 'x': return parse_hex(2, start);
      case 'u': return parse_hex(4, start);
      default: return static_cast<std::uint8_t>(value);
    }
  }

  std::unique_ptr<node> parse_escape(bool in_class)
  {
    auto start = position_ - 1;
    if (at_end()) { fail({start, 1}, "trailing backslash"); }
    char const value = take();
    if (!in_class &&
        (value == 'b' || value == 'B' || value == 'A' || value == 'Z' || value == 'z')) {
      auto result = make(node_kind::ASSERTION, start);
      if (value == 'b') result->assertion = assertion_kind::WORD_BOUNDARY;
      if (value == 'B') result->assertion = assertion_kind::NOT_WORD_BOUNDARY;
      if (value == 'A') result->assertion = assertion_kind::BEGIN_INPUT;
      if (value == 'Z' || value == 'z') result->assertion = assertion_kind::END_INPUT;
      return result;
    }
    auto result = make(node_kind::PREDICATE, start);
    if (value == 'p' || value == 'P') {
      if (!consume('{')) { fail({start, position_ - start}, "missing property name"); }
      auto property_begin = position_;
      while (!at_end() && peek() != '}')
        static_cast<void>(take());
      if (!consume('}')) { fail({start, position_ - start}, "unterminated Unicode property"); }
      auto property = pattern_.substr(property_begin, position_ - property_begin - 1U);
      if (property != "Sm") { fail({start, position_ - start}, "unsupported Unicode property"); }
      append_unicode_ranges(result->predicate, unicode_math_symbol_ranges);
      if (value == 'P') {
        result->predicate.ranges = complement_ranges(std::move(result->predicate.ranges));
      }
      normalize_ranges(result->predicate);
    } else if (value == 'd' || value == 'D' || value == 'w' || value == 'W' || value == 's' ||
               value == 'S') {
      result->predicate = predefined(value);
    } else {
      if (std::isalpha(static_cast<std::uint8_t>(value)) != 0 && value != 'a' && value != 'b' &&
          value != 'f' && value != 'n' && value != 'r' && value != 't' && value != 'u' &&
          value != 'v' && value != 'x') {
        fail({start, 2}, "unknown alphabetic escape");
      }
      bool three_digit_octal = value >= '0' && value <= '7' && position_ + 1 < pattern_.size() &&
                               pattern_[position_] >= '0' && pattern_[position_] <= '7' &&
                               pattern_[position_ + 1] >= '0' && pattern_[position_ + 1] <= '7';
      if (value >= '1' && value <= '9' && !three_digit_octal) {
        fail({start, 2}, "backreferences are not supported");
      }
      auto literal = three_digit_octal ? parse_octal(value) : escaped_literal(value, start);
      if (options_.case_insensitive) {
        add_case_pair(result->predicate, literal, literal);
      } else {
        result->predicate.ranges.push_back({literal, literal});
      }
      normalize_ranges(result->predicate);
    }
    result->source.length = position_ - start;
    return result;
  }

  std::unique_ptr<node> parse_class()
  {
    auto start                = position_ - 1;
    auto result               = make(node_kind::PREDICATE, start);
    result->predicate.negated = consume('^');
    bool first                = true;
    bool closed               = false;
    while (!at_end()) {
      if (peek() == ']' && !first) {
        ++position_;
        closed = true;
        break;
      }
      first = false;
      if (peek() == '[' && position_ + 1U < pattern_.size() && pattern_[position_ + 1U] == ':') {
        auto class_start = position_;
        position_ += 2U;
        auto name_start = position_;
        while (position_ + 1U < pattern_.size() &&
               !(pattern_[position_] == ':' && pattern_[position_ + 1U] == ']')) {
          ++position_;
        }
        if (position_ + 1U >= pattern_.size()) {
          fail({class_start, position_ - class_start}, "unterminated POSIX character class");
        }
        auto name = pattern_.substr(name_start, position_ - name_start);
        position_ += 2U;
        if (!append_posix_class(result->predicate, name)) {
          fail({class_start, position_ - class_start}, "unsupported POSIX character class");
        }
        continue;
      }
      char32_t lower{};
      if (consume('\\')) {
        auto escaped = parse_escape(true);
        if (escaped->predicate.ranges.size() != 1 ||
            escaped->predicate.ranges.front().first != escaped->predicate.ranges.front().last) {
          result->predicate.ranges.insert(result->predicate.ranges.end(),
                                          escaped->predicate.ranges.begin(),
                                          escaped->predicate.ranges.end());
          continue;
        }
        lower = escaped->predicate.ranges.front().first;
      } else {
        std::size_t length{};
        lower = decode_literal(length);
        position_ += length;
      }

      char32_t upper = lower;
      if (peek() == '-' && position_ + 1 < pattern_.size() && pattern_[position_ + 1] != ']') {
        ++position_;
        if (consume('\\')) {
          auto escaped = parse_escape(true);
          if (!escaped->predicate.is_singleton()) {
            fail({position_, 1}, "range endpoint must be a literal");
          }
          upper = escaped->predicate.singleton();
        } else {
          std::size_t length{};
          upper = decode_literal(length);
          position_ += length;
        }
        if (upper < lower) { fail({start, position_ - start}, "descending character range"); }
      }
      if (options_.case_insensitive) {
        add_case_pair(result->predicate, lower, upper);
      } else {
        result->predicate.ranges.push_back({lower, upper});
      }
    }
    if (!closed) { fail({start, position_ - start}, "unterminated character class"); }
    if (result->predicate.ranges.empty()) {
      fail({start, position_ - start}, "empty character class");
    }
    normalize_ranges(result->predicate);
    result->source.length = position_ - start;
    return result;
  }

  std::unique_ptr<node> parse_atom()
  {
    auto start       = position_;
    char const value = take();
    if (value == '(') {
      bool capturing = true;
      if (consume('?')) {
        if (consume(':')) {
          capturing = false;
        } else {
          fail({start, position_ - start + 1},
               "lookaround and inline group extensions are not supported");
        }
      }
      if (++depth_ > options_.limits.max_nesting) {
        fail({start, 1}, "group nesting exceeds max_nesting");
      }
      std::uint32_t capture{};
      if (capturing) {
        if (capture_count >= options_.limits.max_captures) {
          fail({start, 1}, "capture count exceeds max_captures");
        }
        capture = ++capture_count;
      }
      auto child = parse_alternation();
      if (!consume(')')) { fail({start, position_ - start}, "unterminated group"); }
      --depth_;
      auto group = make(node_kind::GROUP, start);
      group->children.push_back(std::move(child));
      group->capturing     = capturing;
      group->capture_index = capture;
      return group;
    }
    if (value == '[') { return parse_class(); }
    if (value == '\\') { return parse_escape(false); }
    if (value == '.') {
      auto result                        = make(node_kind::PREDICATE, start);
      result->predicate.recognized       = predicate_class::ANY;
      result->predicate.matches_newline  = options_.dot_all;
      result->predicate.extended_newline = options_.extended_newline;
      return result;
    }
    if (value == '^' || value == '$') {
      auto result       = make(node_kind::ASSERTION, start);
      result->assertion = value == '^' ? assertion_kind::BEGIN_LINE : assertion_kind::END_LINE;
      return result;
    }
    if (value == ')' || value == '|' || value == '*' || value == '+' || value == '?' ||
        (value == '{' && std::isdigit(static_cast<std::uint8_t>(peek())) != 0)) {
      fail({start, 1}, "unexpected metacharacter");
    }

    --position_;
    std::size_t length{};
    auto literal = decode_literal(length);
    position_ += length;
    auto result = make(node_kind::PREDICATE, start);
    if (options_.case_insensitive) {
      add_case_pair(result->predicate, literal, literal);
    } else {
      result->predicate.ranges.push_back({literal, literal});
    }
    normalize_ranges(result->predicate);
    result->source.length = length;
    return result;
  }

  std::string_view pattern_;
  compile_options const& options_;
  std::size_t position_ = 0;
  std::size_t depth_    = 0;
};

struct patch_reference {
  state_id state   = invalid_state;
  std::size_t edge = 0;
};
struct fragment {
  state_id start                    = invalid_state;
  std::vector<patch_reference> outs = std::vector<patch_reference>{};
};

class thompson_builder {
 public:
  thompson_builder(std::string_view pattern, compile_options const& options)
  {
    ir.pattern = std::string(pattern);
    ir.options = options;
  }

  automata_ir ir = automata_ir{};
  fragment build(node const& expression)
  {
    switch (expression.kind) {
      case node_kind::EMPTY: return make_empty(expression.source);
      case node_kind::PREDICATE: return make_predicate(expression);
      case node_kind::ASSERTION: return make_assertion(expression);
      case node_kind::GROUP: return make_group(expression);
      case node_kind::CONCATENATE: return make_concatenate(expression);
      case node_kind::ALTERNATE: ir.has_alternation = true; return make_alternate(expression);
      case node_kind::REPEAT:
        ir.has_lazy_quantifier = ir.has_lazy_quantifier || !expression.greedy;
        return make_repeat(expression);
    }
    return {};
  }

  void finish(fragment expression, std::uint32_t captures)
  {
    auto accept = add_state(automata_state_kind::ACCEPT, {ir.pattern.size(), 0});
    patch(expression.outs, accept);
    auto entry = add_state(automata_state_kind::JUMP, {0, 0});
    add_edge(entry, expression.start, 0);
    ir.entry         = entry;
    ir.accept        = accept;
    ir.capture_count = captures;
  }

 private:
  state_id add_state(automata_state_kind kind, source_span source)
  {
    if (ir.states.size() >= ir.options.limits.max_states) {
      throw compile_failure{source, "state count exceeds max_states"};
    }
    auto id = static_cast<state_id>(ir.states.size());
    automata_state state;
    state.id     = id;
    state.kind   = kind;
    state.source = source;
    ir.states.push_back(std::move(state));
    return id;
  }

  std::size_t add_edge(state_id from, state_id to, std::uint32_t priority)
  {
    if (transition_count_ >= ir.options.limits.max_transitions) {
      throw compile_failure{ir.states[from].source, "transition count exceeds limit"};
    }
    ++transition_count_;
    ir.states[from].edges.push_back({to, priority});
    return ir.states[from].edges.size() - 1;
  }

  patch_reference add_open_edge(state_id from, std::uint32_t priority)
  {
    return {from, add_edge(from, invalid_state, priority)};
  }

  void patch(std::vector<patch_reference> const& references, state_id target)
  {
    // open edges let Thompson fragments be joined without rebuilding either fragment
    for (auto reference : references) {
      ir.states[reference.state].edges[reference.edge].target = target;
    }
  }

  fragment make_empty(source_span span)
  {
    auto state = add_state(automata_state_kind::JUMP, span);
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_predicate(node const& expression)
  {
    auto state                 = add_state(automata_state_kind::CONSUME, expression.source);
    ir.states[state].predicate = expression.predicate;
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_assertion(node const& expression)
  {
    auto state                 = add_state(automata_state_kind::ASSERTION, expression.source);
    ir.states[state].assertion = expression.assertion;
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_group(node const& expression)
  {
    if (!expression.capturing) return build(*expression.children.front());

    auto begin                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[begin].capture       = capture_action::BEGIN;
    ir.states[begin].capture_index = expression.capture_index;

    auto inner = build(*expression.children.front());
    add_edge(begin, inner.start, 0);

    auto end                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[end].capture       = capture_action::END;
    ir.states[end].capture_index = expression.capture_index;
    patch(inner.outs, end);
    return {begin, {add_open_edge(end, 0)}};
  }

  fragment concatenate(fragment left, fragment right)
  {
    patch(left.outs, right.start);
    return {left.start, std::move(right.outs)};
  }

  fragment make_concatenate(node const& expression)
  {
    if (expression.children.empty()) return make_empty(expression.source);
    auto result = build(*expression.children.front());
    for (std::size_t index = 1; index < expression.children.size(); ++index) {
      result = concatenate(std::move(result), build(*expression.children[index]));
    }
    return result;
  }

  fragment alternate(fragment left, fragment right, source_span span)
  {
    auto branch = add_state(automata_state_kind::BRANCH, span);
    add_edge(branch, left.start, 0);
    add_edge(branch, right.start, 1);
    left.outs.insert(left.outs.end(),
                     std::make_move_iterator(right.outs.begin()),
                     std::make_move_iterator(right.outs.end()));
    return {branch, std::move(left.outs)};
  }

  fragment make_alternate(node const& expression)
  {
    if (expression.children.empty()) return make_empty(expression.source);
    auto result = build(*expression.children.front());
    for (std::size_t index = 1; index < expression.children.size(); ++index) {
      result = alternate(std::move(result), build(*expression.children[index]), expression.source);
    }
    return result;
  }

  fragment optional(node const& expression, source_span span, bool greedy)
  {
    auto inner  = build(expression);
    auto branch = add_state(automata_state_kind::BRANCH, span);
    // lower priorities are attempted first, so swapping them implements lazy quantifiers
    auto take_priority = greedy ? 0U : 1U;
    auto exit_priority = greedy ? 1U : 0U;
    add_edge(branch, inner.start, take_priority);
    inner.outs.push_back(add_open_edge(branch, exit_priority));
    return {branch, std::move(inner.outs)};
  }

  fragment star(node const& expression, source_span span, bool greedy)
  {
    auto inner           = build(expression);
    auto branch          = add_state(automata_state_kind::BRANCH, span);
    auto repeat_priority = greedy ? 0U : 1U;
    auto exit_priority   = greedy ? 1U : 0U;
    add_edge(branch, inner.start, repeat_priority);
    patch(inner.outs, branch);
    return {branch, {add_open_edge(branch, exit_priority)}};
  }

  fragment make_repeat(node const& expression)
  {
    auto& repeated = *expression.children.front();
    if (expression.maximum == unbounded_repeat && can_match_empty(repeated)) {
      throw compile_failure{expression.source,
                            "unbounded repetition of a nullable expression is not supported"};
    }
    std::optional<fragment> result;
    auto append = [&](fragment next) {
      if (result) {
        *result = concatenate(std::move(*result), std::move(next));
      } else {
        result = std::move(next);
      }
    };

    for (std::uint32_t count = 0; count < expression.minimum; ++count) {
      append(build(repeated));
    }

    if (expression.maximum == unbounded_repeat) {
      append(star(repeated, expression.source, expression.greedy));
    } else {
      for (std::uint32_t count = expression.minimum; count < expression.maximum; ++count) {
        append(optional(repeated, expression.source, expression.greedy));
      }
    }

    return result ? std::move(*result) : make_empty(expression.source);
  }

  std::size_t transition_count_ = 0;
};

}  // namespace

bool character_predicate::matches(char32_t value) const noexcept
{
  if (recognized == predicate_class::ANY) {
    if (matches_newline) return true;
    if (!extended_newline) return value != U'\n';
    return value != U'\n' && value != U'\r' && value != static_cast<char32_t>(0x85) &&
           value != static_cast<char32_t>(0x2028) && value != static_cast<char32_t>(0x2029);
  }
  bool contained = false;
  for (auto range : ranges) {
    if (value >= range.first && value <= range.last) {
      contained = true;
      break;
    }
  }
  return negated ? !contained : contained;
}

bool character_predicate::is_singleton() const noexcept
{
  return !negated && recognized == predicate_class::NONE && ranges.size() == 1 &&
         ranges.front().first == ranges.front().last;
}

char32_t character_predicate::singleton() const noexcept
{
  return is_singleton() ? ranges.front().first : U'\0';
}

void verify(automata_ir const& ir)
{
  auto invalid = [](std::string_view message) { throw std::logic_error(std::string{message}); };

  if (ir.entry >= ir.states.size()) invalid("entry state is invalid");
  if (ir.accept >= ir.states.size()) invalid("accept state is invalid");

  for (std::size_t index = 0; index < ir.states.size(); ++index) {
    auto& state = ir.states[index];
    if (state.id != index) invalid("state ID does not match storage index");

    for (auto edge : state.edges) {
      if (edge.target >= ir.states.size()) invalid("edge target is invalid");
    }

    if (state.kind == automata_state_kind::ACCEPT && !state.edges.empty()) {
      invalid("accept state has outgoing edges");
    }
    if (state.kind == automata_state_kind::CONSUME && state.edges.size() != 1) {
      invalid("consume state must have one edge");
    }
    if ((state.kind == automata_state_kind::JUMP || state.kind == automata_state_kind::ASSERTION ||
         state.kind == automata_state_kind::CAPTURE) &&
        state.edges.size() != 1) {
      invalid("linear epsilon state must have one edge");
    }
    if (state.kind == automata_state_kind::BRANCH && state.edges.size() < 2) {
      invalid("branch state must have at least two edges");
    }
    if (state.kind == automata_state_kind::CAPTURE &&
        (state.capture_index == 0 || state.capture_index > ir.capture_count)) {
      invalid("capture index is out of range");
    }
  }
}

automata_ir compile_automata(std::string_view pattern, compile_options const& options)
{
  parser parse_pattern(pattern, options);
  auto expression = parse_pattern.parse();

  thompson_builder builder(pattern, options);
  auto fragment = builder.build(*expression);
  builder.finish(std::move(fragment), parse_pattern.capture_count);
  verify(builder.ir);
  return std::move(builder.ir);
}

}  // namespace regex_ir

// instruction IR lowering

namespace regex_ir {
namespace {

void parse_replacement(std::string const& replacement,
                       std::uint32_t capture_count,
                       std::vector<replacement_token>& output)
{
  std::string literal;
  auto flush_literal = [&] {
    if (!literal.empty()) {
      output.push_back({replacement_token::kind::LITERAL, std::move(literal), 0});
      literal.clear();
    }
  };

  for (std::size_t position = 0; position < replacement.size();) {
    if (replacement[position] != '$') {
      literal.push_back(replacement[position++]);
      continue;
    }
    auto start = position++;
    if (position < replacement.size() && replacement[position] == '$') {
      literal.push_back('$');
      ++position;
      continue;
    }
    auto braced = position < replacement.size() && replacement[position] == '{';
    if (braced) { ++position; }
    if (position == replacement.size() ||
        std::isdigit(static_cast<std::uint8_t>(replacement[position])) == 0) {
      throw compile_failure{{start, position - start},
                            "dollar must be followed by a capture number or dollar"};
    }
    std::uint64_t capture{};
    while (position < replacement.size() &&
           std::isdigit(static_cast<std::uint8_t>(replacement[position])) != 0) {
      capture = capture * 10U + static_cast<std::uint32_t>(replacement[position++] - '0');
      if (capture > capture_count) {
        throw compile_failure{{start, position - start}, "replacement capture is out of range"};
      }
    }
    if (braced) {
      if (position == replacement.size() || replacement[position] != '}') {
        throw compile_failure{{start, position - start},
                              "braced replacement capture is not terminated"};
      }
      ++position;
    }
    flush_literal();
    output.push_back({replacement_token::kind::CAPTURE, {}, static_cast<std::uint32_t>(capture)});
  }
  flush_literal();
}

}  // namespace

void verify(instruction_ir const& ir)
{
  auto invalid = [](std::string_view message) { throw std::logic_error(std::string{message}); };

  if (ir.entry >= ir.blocks.size()) invalid("entry block is invalid");
  if (ir.accept >= ir.blocks.size()) invalid("accept block is invalid");

  for (std::size_t index = 0; index < ir.blocks.size(); ++index) {
    auto& block = ir.blocks[index];
    if (block.id != index) invalid("block ID does not match storage index");

    for (auto edge : block.successors) {
      if (edge.target >= ir.blocks.size()) invalid("successor target is invalid");
    }

    bool accepting{};
    std::size_t character_tests{};
    std::size_t advances{};
    for (auto& item : block.instructions) {
      if (std::holds_alternative<emit_accept>(item)) accepting = true;
      if (std::holds_alternative<match_character>(item) ||
          std::holds_alternative<match_literal>(item)) {
        ++character_tests;
      }
      if (std::holds_alternative<advance_cursor>(item)) ++advances;
      if (auto* capture = std::get_if<write_capture>(&item);
          capture != nullptr &&
          (capture->capture_index == 0 || capture->capture_index > ir.capture_count)) {
        invalid("capture write index is out of range");
      }
    }
    if (accepting && !block.successors.empty()) { invalid("accept block has successors"); }
    if (character_tests > 1) invalid("block has multiple character tests");
    if (advances > 1) invalid("block advances more than once");
  }
}

instruction_ir lower(automata_ir const& automata, operation const& selected)
{
  verify(automata);

  instruction_ir result;
  result.pattern            = automata.pattern;
  result.options            = automata.options;
  result.selected_operation = selected;
  switch (selected.kind) {
    case operation_kind::MATCHES: result.control = {false, false, result_shape::BOOLEAN}; break;
    case operation_kind::CONTAINS: result.control = {true, false, result_shape::BOOLEAN}; break;
    case operation_kind::FIND: result.control = {true, false, result_shape::MATCH_SPAN}; break;
    case operation_kind::FIND_ALL: result.control = {true, false, result_shape::MATCH_SPAN}; break;
    case operation_kind::COUNT: result.control = {true, false, result_shape::MATCH_COUNT}; break;
    case operation_kind::EXTRACT: result.control = {true, false, result_shape::CAPTURES}; break;
    case operation_kind::REPLACE: result.control = {true, false, result_shape::REPLACEMENT}; break;
    case operation_kind::SPLIT: result.control = {true, false, result_shape::SPLIT_FIELDS}; break;
  }
  result.entry               = automata.entry;
  result.accept              = automata.accept;
  result.capture_count       = automata.capture_count;
  result.has_alternation     = automata.has_alternation;
  result.has_lazy_quantifier = automata.has_lazy_quantifier;
  result.blocks.reserve(automata.states.size());

  for (auto& state : automata.states) {
    instruction_block block;
    block.id     = state.id;
    block.source = state.source;
    block.successors.reserve(state.edges.size());
    for (auto edge : state.edges)
      block.successors.push_back({edge.target, edge.priority});

    switch (state.kind) {
      case automata_state_kind::JUMP:
      case automata_state_kind::BRANCH: break;
      case automata_state_kind::CONSUME:
        block.instructions.emplace_back(can_peek{1});
        block.instructions.emplace_back(read_character{});
        block.instructions.emplace_back(match_character{state.predicate});
        block.instructions.emplace_back(advance_cursor{1});
        break;
      case automata_state_kind::ASSERTION:
        block.instructions.emplace_back(test_assertion{state.assertion});
        break;
      case automata_state_kind::CAPTURE:
        block.instructions.emplace_back(write_capture{state.capture, state.capture_index});
        break;
      case automata_state_kind::ACCEPT: block.instructions.emplace_back(emit_accept{}); break;
    }
    result.blocks.push_back(std::move(block));
  }

  if (selected.kind == operation_kind::REPLACE) {
    parse_replacement(selected.replacement, result.capture_count, result.replacement);
  }

  verify(result);
  return result;
}

instruction_ir compile_instruction_ir(std::string_view pattern,
                                      operation const& selected,
                                      compile_options const& options,
                                      optimization_options const& optimization)
{
  auto automata = compile_automata(pattern, options);
  return optimize(lower(automata, selected), optimization);
}

}  // namespace regex_ir

// instruction IR optimization

namespace regex_ir {
namespace {

std::optional<char32_t> singleton(instruction_block const& block)
{
  for (auto& item : block.instructions) {
    if (auto* match = std::get_if<match_character>(&item);
        match != nullptr && match->predicate.is_singleton()) {
      return match->predicate.singleton();
    }
  }
  return std::nullopt;
}

void strip_captures(instruction_ir& ir)
{
  std::vector<bool> observed(ir.capture_count + 1U, false);
  if (ir.selected_operation.kind == operation_kind::EXTRACT) {
    std::fill(observed.begin(), observed.end(), true);
  } else if (ir.selected_operation.kind == operation_kind::REPLACE) {
    for (auto& token : ir.replacement) {
      if (token.type == replacement_token::kind::CAPTURE) { observed[token.capture_index] = true; }
    }
  }
  for (auto& block : ir.blocks) {
    block.instructions.erase(std::remove_if(block.instructions.begin(),
                                            block.instructions.end(),
                                            [&](instruction const& item) {
                                              auto* capture = std::get_if<write_capture>(&item);
                                              return capture != nullptr &&
                                                     !observed[capture->capture_index];
                                            }),
                             block.instructions.end());
  }
}

block_id resolve_empty(instruction_ir const& ir, block_id start)
{
  // stop at cycles because nullable repetition can produce an all-empty component
  std::unordered_set<block_id> visited;
  auto current = start;
  while (current < ir.blocks.size() && visited.insert(current).second) {
    auto& block = ir.blocks[current];
    if (!block.instructions.empty() || block.successors.size() != 1) break;
    current = block.successors.front().target;
  }
  return current;
}

void fold_empty_jumps(instruction_ir& ir)
{
  ir.entry = resolve_empty(ir, ir.entry);
  for (auto& block : ir.blocks) {
    for (auto& edge : block.successors)
      edge.target = resolve_empty(ir, edge.target);
  }
}

void fuse_literals(instruction_ir& ir, std::size_t limit)
{
  if (limit < 2) return;
  std::vector<std::size_t> incoming(ir.blocks.size());
  for (auto& block : ir.blocks) {
    for (auto edge : block.successors) {
      if (edge.target < incoming.size()) ++incoming[edge.target];
    }
  }

  for (auto& block : ir.blocks) {
    auto first = singleton(block);
    if (!first || block.successors.size() != 1) continue;

    std::u32string value{*first};
    auto next = block.successors.front().target;
    std::unordered_set<block_id> visited{block.id};
    // a single incoming edge makes it safe to consume the candidate into this block
    while (value.size() < limit && next < ir.blocks.size() && incoming[next] == 1 &&
           visited.insert(next).second) {
      auto& candidate = ir.blocks[next];
      auto character  = singleton(candidate);
      if (!character || candidate.successors.size() != 1) break;
      value.push_back(*character);
      next = candidate.successors.front().target;
    }
    if (value.size() < 2) continue;

    block.instructions.clear();
    block.instructions.emplace_back(can_peek{static_cast<std::uint32_t>(value.size())});
    block.instructions.emplace_back(match_literal{std::move(value)});
    block.instructions.emplace_back(advance_cursor{
      static_cast<std::uint32_t>(std::get<match_literal>(block.instructions[1]).value.size())});
    block.successors = {{next, 0}};
  }
}

void remove_unreachable(instruction_ir& ir)
{
  if (ir.entry >= ir.blocks.size()) return;
  std::vector<bool> reachable(ir.blocks.size());
  std::vector<block_id> work{ir.entry};
  reachable[ir.entry] = true;
  while (!work.empty()) {
    auto current = work.back();
    work.pop_back();
    for (auto edge : ir.blocks[current].successors) {
      if (edge.target < reachable.size() && !reachable[edge.target]) {
        reachable[edge.target] = true;
        work.push_back(edge.target);
      }
    }
  }

  std::vector<block_id> remap(ir.blocks.size(), invalid_block);
  std::vector<instruction_block> blocks;
  blocks.reserve(ir.blocks.size());
  for (std::size_t old = 0; old < ir.blocks.size(); ++old) {
    if (!reachable[old]) continue;
    remap[old] = static_cast<block_id>(blocks.size());
    auto block = std::move(ir.blocks[old]);
    block.id   = static_cast<block_id>(blocks.size());
    blocks.push_back(std::move(block));
  }
  // rewrite dense IDs only after every old-to-new mapping has been established
  for (auto& block : blocks) {
    for (auto& edge : block.successors)
      edge.target = remap[edge.target];
  }
  ir.entry  = remap[ir.entry];
  ir.accept = remap[ir.accept];
  ir.blocks = std::move(blocks);
}

}  // namespace

instruction_ir optimize(instruction_ir ir, optimization_options const& options)
{
  verify(ir);

  if (options.strip_unobserved_captures) strip_captures(ir);
  if (options.fold_epsilon_jumps) fold_empty_jumps(ir);
  if (options.fuse_literals) fuse_literals(ir, options.literal_fusion_limit);
  if (options.fold_epsilon_jumps) fold_empty_jumps(ir);
  if (options.remove_unreachable) remove_unreachable(ir);

  verify(ir);
  return ir;
}

}  // namespace regex_ir

namespace regex_ir {

compile_result compile(std::string_view pattern,
                       operation_kind operation_kind_value,
                       std::optional<std::string> replacement,
                       compile_options const& options)
{
  switch (operation_kind_value) {
    case operation_kind::CONTAINS:
    case operation_kind::MATCHES:
    case operation_kind::COUNT:
    case operation_kind::EXTRACT:
    case operation_kind::FIND:
    case operation_kind::FIND_ALL:
    case operation_kind::SPLIT:
      if (replacement.has_value()) {
        throw std::invalid_argument("replacement is only valid for the REPLACE operation");
      }
      break;
    case operation_kind::REPLACE:
      if (!replacement.has_value()) {
        throw std::invalid_argument("replacement is required for the REPLACE operation");
      }
      break;
    default: throw std::invalid_argument("invalid regex operation");
  }

  try {
    auto compiled = compile_instruction_ir(
      pattern, operation{operation_kind_value, replacement.value_or("")}, options, {});
    return generate_cuda_source(compiled, {});
  } catch (compile_failure const& failure) {
    throw std::invalid_argument(std::format(
      "regex compilation failed at byte {}: {}", failure.source.offset, failure.message));
  }
}

}  // namespace regex_ir
