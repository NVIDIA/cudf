/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvvm_templates.hpp"
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
    while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
      value = value * 10U + static_cast<unsigned>(take() - '0');
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
               std::isdigit(static_cast<unsigned char>(pattern_[position_ + 1])) != 0) {
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
      return static_cast<unsigned char>(pattern_[position_]);
    }
    auto first = static_cast<unsigned char>(pattern_[position_]);
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
      auto next = static_cast<unsigned char>(pattern_[position_ + index]);
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
    auto base = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
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
      default: return static_cast<unsigned char>(value);
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
      if (std::isalpha(static_cast<unsigned char>(value)) != 0 && value != 'a' && value != 'b' &&
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
        (value == '{' && std::isdigit(static_cast<unsigned char>(peek())) != 0)) {
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
        std::isdigit(static_cast<unsigned char>(replacement[position])) == 0) {
      throw compile_failure{{start, position - start},
                            "dollar must be followed by a capture number or dollar"};
    }
    std::uint64_t capture{};
    while (position < replacement.size() &&
           std::isdigit(static_cast<unsigned char>(replacement[position])) != 0) {
      capture = capture * 10U + static_cast<unsigned>(replacement[position++] - '0');
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

instruction_ir optimize(instruction_ir ir, optimization_options const& options);

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

// device IR generation

namespace regex_ir {
namespace {

class source_buffer {
 public:
  template <typename... Args>
  void emit(std::format_string<Args...> format, Args&&... args)
  {
    std::format_to(std::back_inserter(value_), format, std::forward<Args>(args)...);
    value_ += '\n';
  }

  void blank() { value_ += '\n'; }
  [[nodiscard]] std::string take() { return std::move(value_); }

 private:
  std::string value_ = "";
};

void require_identifier(std::string_view value, std::string_view field)
{
  auto first_is_valid = [](unsigned char character) {
    return std::isalpha(character) != 0 || character == '_';
  };
  auto rest_is_valid = [&](unsigned char character) {
    return first_is_valid(character) || std::isdigit(character) != 0;
  };
  if (value.empty() || !first_is_valid(static_cast<unsigned char>(value.front())) ||
      !std::all_of(value.begin() + 1, value.end(), [&](char character) {
        return rest_is_valid(static_cast<unsigned char>(character));
      })) {
    throw std::invalid_argument(std::format("{} must be a valid source identifier", field));
  }
  if (value.starts_with("llvm.") || value.starts_with("nvvm.")) {
    throw std::invalid_argument(std::format("{} uses a reserved identifier", field));
  }
}

void require_codegen_ir(instruction_ir const& ir) { verify(ir); }

std::string nvvm_symbol(std::string_view prefix, std::string_view suffix)
{
  return std::format("{}_{}", prefix, suffix);
}

struct deterministic_nfa_node {
  character_predicate predicate           = character_predicate{};
  std::vector<std::size_t> targets        = std::vector<std::size_t>{};
  std::optional<write_capture> capture    = std::nullopt;
  std::optional<assertion_kind> assertion = std::nullopt;
  bool consumes : 1                       = false;
  bool accepts  : 1                       = false;
};

struct deterministic_nfa_graph {
  std::vector<deterministic_nfa_node> nodes = std::vector<deterministic_nfa_node>{};
  std::size_t entry                         = 0;
};

struct deterministic_capture_action {
  std::uint32_t slot = 0;
  bool reset_end : 1 = false;

  bool operator==(deterministic_capture_action const&) const = default;
};

struct deterministic_interval {
  std::uint32_t first    = 0;
  std::uint32_t last     = 0;
  std::uint16_t class_id = 0;
};

struct deterministic_machine {
  std::array<std::uint16_t, 256> byte_classes           = std::array<std::uint16_t, 256>{};
  std::array<std::uint64_t, 4> start_byte_bitmap        = std::array<std::uint64_t, 4>{};
  std::vector<deterministic_interval> unicode_intervals = std::vector<deterministic_interval>{};
  std::vector<std::uint16_t> transitions                = std::vector<std::uint16_t>{};
  std::vector<std::vector<deterministic_capture_action>> transition_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::vector<deterministic_capture_action>> accept_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::uint8_t> boundary_accepts     = std::vector<std::uint8_t>{};
  std::uint16_t initial_state                    = 0;
  std::uint16_t dead_state                       = 0;
  std::uint16_t class_count                      = 0;
  std::uint16_t state_count                      = 0;
  std::uint16_t state_mask                       = 16383;
  std::uint16_t restart_state                    = std::numeric_limits<std::uint16_t>::max();
  std::uint8_t transition_address_space          = 4;
  std::uint8_t boundary_class_count              = 0;
  std::uint8_t assertion_mask                    = 0;
  std::uint8_t start_byte_range_count            = 0;
  std::optional<assertion_kind> accept_assertion = std::nullopt;
  bool scan_input        : 1                     = false;
  bool accept_at_end     : 1                     = false;
  bool capture_one_pass  : 1                     = false;
  bool assertion_aware   : 1                     = false;
  bool start_byte_filter : 1                     = false;
};

struct glushkov_shift {
  std::uint64_t sources = 0;
  std::uint8_t amount   = 0;
};

struct glushkov_machine {
  deterministic_machine alphabet                        = deterministic_machine{};
  std::vector<std::uint64_t> reach_masks                = std::vector<std::uint64_t>{};
  std::vector<glushkov_shift> shifts                    = std::vector<glushkov_shift>{};
  std::array<std::uint64_t, 64> exception_successors    = std::array<std::uint64_t, 64>{};
  std::uint64_t first_set                               = 0;
  std::uint64_t accept_mask                             = 0;
  std::uint64_t exception_mask                          = 0;
  std::optional<std::uint8_t> start_byte                = std::nullopt;
  std::optional<std::uint32_t> fixed_match_bytes        = std::nullopt;
  std::optional<std::uint32_t> repeated_predicate_count = std::nullopt;
  predicate_class repeated_predicate_class              = predicate_class::NONE;
  std::uint8_t position_count                           = 0;
  bool scan_input    : 1                                = false;
  bool accept_at_end : 1                                = false;
};

bool equivalent_predicates(character_predicate const& left, character_predicate const& right)
{
  if (left.recognized != right.recognized || left.negated != right.negated ||
      left.matches_newline != right.matches_newline ||
      left.extended_newline != right.extended_newline ||
      left.ranges.size() != right.ranges.size()) {
    return false;
  }
  return std::equal(
    left.ranges.begin(), left.ranges.end(), right.ranges.begin(), [](auto& lhs, auto& rhs) {
      return lhs.first == rhs.first && lhs.last == rhs.last;
    });
}

std::optional<builtin_character_class> adapted_builtin(predicate_class value)
{
  switch (value) {
    case predicate_class::DIGIT: return builtin_character_class::DIGIT;
    case predicate_class::NOT_DIGIT: return builtin_character_class::NOT_DIGIT;
    case predicate_class::WORD: return builtin_character_class::WORD;
    case predicate_class::NOT_WORD: return builtin_character_class::NOT_WORD;
    case predicate_class::SPACE: return builtin_character_class::SPACE;
    case predicate_class::NOT_SPACE: return builtin_character_class::NOT_SPACE;
    case predicate_class::NONE:
    case predicate_class::ANY: return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::uint32_t> repeated_predicate_count(deterministic_nfa_graph const& graph)
{
  character_predicate const* predicate = nullptr;
  for (auto& node : graph.nodes) {
    if (!node.consumes) continue;
    if (predicate == nullptr) {
      predicate = &node.predicate;
    } else if (!equivalent_predicates(*predicate, node.predicate)) {
      return std::nullopt;
    }
  }
  if (predicate == nullptr) return std::nullopt;

  enum class visit_state : std::uint8_t { UNVISITED, VISITING, COMPLETE };
  std::vector<visit_state> states(graph.nodes.size(), visit_state::UNVISITED);
  std::vector<std::optional<std::uint32_t>> counts(graph.nodes.size());
  auto visit = [&](auto&& self, std::size_t index) -> std::optional<std::uint32_t> {
    if (index >= graph.nodes.size() || states[index] == visit_state::VISITING) return std::nullopt;
    if (states[index] == visit_state::COMPLETE) return counts[index];
    states[index] = visit_state::VISITING;
    auto& node    = graph.nodes[index];
    std::optional<std::uint32_t> suffix =
      node.accepts ? std::optional<std::uint32_t>{0U} : std::nullopt;
    for (auto target : node.targets) {
      auto target_count = self(self, target);
      if (!target_count.has_value() || (suffix.has_value() && *suffix != *target_count)) {
        states[index] = visit_state::COMPLETE;
        return std::nullopt;
      }
      suffix = target_count;
    }
    if (!suffix.has_value() ||
        (node.consumes && *suffix == std::numeric_limits<std::uint32_t>::max())) {
      states[index] = visit_state::COMPLETE;
      return std::nullopt;
    }
    counts[index] = *suffix + static_cast<std::uint32_t>(node.consumes);
    states[index] = visit_state::COMPLETE;
    return counts[index];
  };
  auto count = visit(visit, graph.entry);
  return count.has_value() && *count != 0U ? count : std::nullopt;
}

std::optional<std::uint32_t> predicate_byte_width(character_predicate const& predicate,
                                                  character_mode characters)
{
  if (characters == character_mode::BYTES) return 1U;
  if (predicate.negated || predicate.ranges.empty()) return std::nullopt;

  auto utf8_width = [](char32_t codepoint) {
    auto value = static_cast<std::uint32_t>(codepoint);
    return value <= 0x7fU ? 1U : value <= 0x7ffU ? 2U : value <= 0xffffU ? 3U : 4U;
  };
  auto width = utf8_width(predicate.ranges.front().first);
  for (auto range : predicate.ranges) {
    if (utf8_width(range.first) != width || utf8_width(range.last) != width) return std::nullopt;
  }
  return width;
}

std::optional<std::uint32_t> fixed_match_byte_width(deterministic_nfa_graph const& graph,
                                                    character_mode characters)
{
  enum class visit_state : std::uint8_t { UNVISITED, VISITING, COMPLETE };
  std::vector<visit_state> states(graph.nodes.size(), visit_state::UNVISITED);
  std::vector<std::optional<std::uint32_t>> widths(graph.nodes.size());

  auto visit = [&](auto&& self, std::size_t index) -> std::optional<std::uint32_t> {
    if (index >= graph.nodes.size()) return std::nullopt;
    if (states[index] == visit_state::VISITING) return std::nullopt;
    if (states[index] == visit_state::COMPLETE) return widths[index];
    states[index] = visit_state::VISITING;

    auto& node = graph.nodes[index];
    auto own   = node.consumes ? predicate_byte_width(node.predicate, characters)
                               : std::optional<std::uint32_t>{0U};
    if (!own.has_value()) {
      states[index] = visit_state::COMPLETE;
      return std::nullopt;
    }

    std::optional<std::uint32_t> suffix =
      node.accepts ? std::optional<std::uint32_t>{0U} : std::nullopt;
    for (auto target : node.targets) {
      auto target_width = self(self, target);
      if (!target_width.has_value() || (suffix.has_value() && *suffix != *target_width)) {
        states[index] = visit_state::COMPLETE;
        return std::nullopt;
      }
      suffix = target_width;
    }
    if (!suffix.has_value() || *own > std::numeric_limits<std::uint32_t>::max() - *suffix) {
      states[index] = visit_state::COMPLETE;
      return std::nullopt;
    }
    widths[index] = *own + *suffix;
    states[index] = visit_state::COMPLETE;
    return widths[index];
  };

  auto width = visit(visit, graph.entry);
  return width.has_value() && *width != 0U ? width : std::nullopt;
}

void build_start_byte_filter(deterministic_machine& machine)
{
  if ((machine.initial_state & 0x8000U) != 0 || machine.dead_state > machine.state_mask ||
      machine.class_count == 0U) {
    return;
  }
  auto initial = static_cast<std::size_t>(machine.initial_state & machine.state_mask);
  auto offset  = initial * machine.class_count;
  if (offset + machine.class_count > machine.transitions.size()) return;

  std::size_t candidates        = 0;
  auto previous_ascii_candidate = false;
  for (std::size_t byte = 0; byte < machine.byte_classes.size(); ++byte) {
    auto transition = machine.transitions[offset + machine.byte_classes[byte]];
    auto target     = static_cast<std::uint16_t>(transition & machine.state_mask);
    auto candidate  = target != machine.dead_state;
    if (byte < 128U && candidate && !previous_ascii_candidate) { ++machine.start_byte_range_count; }
    if (byte < 128U) previous_ascii_candidate = candidate;
    if (!candidate) continue;
    machine.start_byte_bitmap[byte / 64U] |= std::uint64_t{1} << (byte % 64U);
    ++candidates;
  }
  // sparse, simple ranges repay the extra candidate-dispatch control flow.
  machine.start_byte_filter =
    candidates != 0U && candidates <= 16U && machine.start_byte_range_count <= 2U;
}

void build_assertion_start_byte_filter(deterministic_machine& machine)
{
  if (!machine.assertion_aware || machine.class_count == 0U || machine.boundary_class_count == 0U ||
      machine.dead_state > machine.state_mask) {
    return;
  }
  std::size_t candidates        = 0;
  auto previous_ascii_candidate = false;
  for (std::size_t byte = 0; byte < machine.byte_classes.size(); ++byte) {
    auto candidate = false;
    for (std::size_t boundary = 0; boundary < machine.boundary_class_count; ++boundary) {
      auto index = (static_cast<std::size_t>(machine.initial_state) * machine.boundary_class_count +
                    boundary) *
                     machine.class_count +
                   machine.byte_classes[byte];
      if (index >= machine.transitions.size()) return;
      auto target = static_cast<std::uint16_t>(machine.transitions[index] & machine.state_mask);
      candidate |= target != machine.dead_state;
    }
    if (byte < 128U && candidate && !previous_ascii_candidate) { ++machine.start_byte_range_count; }
    if (byte < 128U) previous_ascii_candidate = candidate;
    if (!candidate) continue;
    machine.start_byte_bitmap[byte / 64U] |= std::uint64_t{1} << (byte % 64U);
    ++candidates;
  }
  machine.start_byte_filter =
    candidates != 0U && candidates <= 16U && machine.start_byte_range_count <= 2U;
}

void build_restart_acceleration(deterministic_machine& machine)
{
  if ((machine.initial_state & 0x8000U) != 0 || machine.class_count == 0U ||
      machine.dead_state > machine.state_mask) {
    return;
  }
  auto initial    = static_cast<std::uint16_t>(machine.initial_state & machine.state_mask);
  auto transition = [&](std::uint16_t state, std::uint16_t character_class) {
    return machine
      .transitions[static_cast<std::size_t>(state) * machine.class_count + character_class];
  };

  auto prefix_state = std::numeric_limits<std::uint16_t>::max();
  std::vector<bool> prefix_classes(machine.class_count);
  // every skipped suffix must be in the same prefix state at the failure byte.
  for (std::uint16_t character_class = 0; character_class < machine.class_count;
       ++character_class) {
    auto encoded = transition(initial, character_class);
    auto target  = static_cast<std::uint16_t>(encoded & machine.state_mask);
    if (target == machine.dead_state) continue;
    if ((encoded & 0xC000U) != 0U || target == initial) return;
    if (prefix_state == std::numeric_limits<std::uint16_t>::max()) {
      prefix_state = target;
    } else if (target != prefix_state) {
      return;
    }
    prefix_classes[character_class] = true;
  }
  if (prefix_state == std::numeric_limits<std::uint16_t>::max()) return;

  for (std::uint16_t character_class = 0; character_class < machine.class_count;
       ++character_class) {
    if (!prefix_classes[character_class]) continue;
    auto encoded = transition(prefix_state, character_class);
    if ((encoded & 0x4000U) != 0U || (encoded & machine.state_mask) != prefix_state) return;
  }
  for (std::uint16_t state = 0; state < machine.state_count; ++state) {
    if (state == initial || state == prefix_state) continue;
    for (std::uint16_t character_class = 0; character_class < machine.class_count;
         ++character_class) {
      if ((transition(state, character_class) & machine.state_mask) == prefix_state) return;
    }
  }
  machine.restart_state = prefix_state;
}

void set_machine_bit(std::vector<std::uint64_t>& bits, std::size_t index)
{
  bits[index / 64] |= std::uint64_t{1} << (index % 64);
}

bool machine_bit(std::vector<std::uint64_t> const& bits, std::size_t index)
{
  return (bits[index / 64] & (std::uint64_t{1} << (index % 64))) != 0;
}

character_predicate singleton_predicate(char32_t value)
{
  character_predicate result;
  result.ranges.push_back({value, value});
  return result;
}

std::optional<deterministic_nfa_graph> make_deterministic_graph(instruction_ir const& ir)
{
  if (ir.entry >= ir.blocks.size()) return std::nullopt;
  std::vector<std::size_t> block_starts(ir.blocks.size());
  std::vector<std::size_t> block_lengths(ir.blocks.size(), 1);
  std::size_t node_count = 0;
  for (auto& block : ir.blocks) {
    match_character const* match    = nullptr;
    match_literal const* literal    = nullptr;
    can_peek const* peek            = nullptr;
    advance_cursor const* advance   = nullptr;
    test_assertion const* assertion = nullptr;
    bool accepts                    = false;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<can_peek>(&item)) peek = candidate;
      if (auto* candidate = std::get_if<advance_cursor>(&item)) advance = candidate;
      if (auto* candidate = std::get_if<test_assertion>(&item)) {
        if (assertion != nullptr) return std::nullopt;
        assertion = candidate;
      }
      if (std::holds_alternative<emit_accept>(item)) accepts = true;
    }
    if (match != nullptr && literal != nullptr) return std::nullopt;
    if ((match != nullptr || literal != nullptr) && (accepts || assertion != nullptr)) {
      return std::nullopt;
    }
    if (match != nullptr && (peek == nullptr || peek->characters != 1 || advance == nullptr ||
                             advance->characters != 1)) {
      return std::nullopt;
    }
    if (literal != nullptr) {
      if (literal->value.empty() || peek == nullptr || peek->characters != literal->value.size() ||
          advance == nullptr || advance->characters != literal->value.size()) {
        return std::nullopt;
      }
      block_lengths[block.id] = literal->value.size();
    }
    block_starts[block.id] = node_count;
    node_count += block_lengths[block.id];
  }
  if (node_count == 0) return std::nullopt;

  deterministic_nfa_graph graph;
  graph.nodes.resize(node_count);
  graph.entry = block_starts[ir.entry];
  for (auto& block : ir.blocks) {
    auto start                      = block_starts[block.id];
    match_character const* match    = nullptr;
    match_literal const* literal    = nullptr;
    write_capture const* capture    = nullptr;
    test_assertion const* assertion = nullptr;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<write_capture>(&item)) capture = candidate;
      if (auto* candidate = std::get_if<test_assertion>(&item)) assertion = candidate;
      if (std::holds_alternative<emit_accept>(item)) graph.nodes[start].accepts = true;
    }
    if (capture != nullptr) graph.nodes[start].capture = *capture;
    if (assertion != nullptr) graph.nodes[start].assertion = assertion->kind;

    auto append_successors = [&](deterministic_nfa_node& node) {
      auto successors = block.successors;
      std::stable_sort(successors.begin(), successors.end(), [](auto& left, auto& right) {
        return left.priority < right.priority;
      });
      for (auto edge : successors)
        node.targets.push_back(block_starts[edge.target]);
    };

    if (match != nullptr) {
      graph.nodes[start].predicate = match->predicate;
      graph.nodes[start].consumes  = true;
      append_successors(graph.nodes[start]);
    } else if (literal != nullptr) {
      for (std::size_t index = 0; index < literal->value.size(); ++index) {
        auto& node     = graph.nodes[start + index];
        node.predicate = singleton_predicate(literal->value[index]);
        node.consumes  = true;
        if (index + 1 < literal->value.size()) {
          node.targets.push_back(start + index + 1);
        } else {
          append_successors(node);
        }
      }
    } else {
      append_successors(graph.nodes[start]);
    }
  }
  return graph;
}

std::optional<std::vector<std::uint32_t>> make_deterministic_alphabet(
  std::vector<deterministic_nfa_node> const& nodes, deterministic_machine& machine)
{
  constexpr std::uint32_t unicode_limit = 0x110000;
  auto word_count                       = (nodes.size() + 63U) / 64U;
  std::vector<std::uint32_t> boundaries{0, 256, unicode_limit};
  for (auto& node : nodes) {
    if (!node.consumes) continue;
    if (node.predicate.recognized == predicate_class::ANY && !node.predicate.matches_newline) {
      boundaries.insert(boundaries.end(), {10, 11});
      if (node.predicate.extended_newline) {
        boundaries.insert(boundaries.end(), {13, 14, 133, 134, 8232, 8234});
      }
    }
    for (auto range : node.predicate.ranges) {
      auto first = static_cast<std::uint32_t>(range.first);
      auto last  = static_cast<std::uint32_t>(range.last);
      if (first < unicode_limit) boundaries.push_back(first);
      if (last < unicode_limit - 1) boundaries.push_back(last + 1);
    }
  }
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

  std::map<std::vector<std::uint64_t>, std::uint16_t> class_ids;
  std::vector<std::uint32_t> representatives;
  std::vector<deterministic_interval> intervals;
  for (std::size_t index = 0; index + 1 < boundaries.size(); ++index) {
    auto first = boundaries[index];
    auto last  = boundaries[index + 1] - 1;
    if (first > last || first >= unicode_limit) continue;
    std::vector<std::uint64_t> signature(word_count);
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      if (nodes[node_index].consumes &&
          nodes[node_index].predicate.matches(static_cast<char32_t>(first))) {
        set_machine_bit(signature, node_index);
      }
    }
    auto existing          = class_ids.find(signature);
    std::uint16_t class_id = 0;
    if (existing == class_ids.end()) {
      if (class_ids.size() >= 32767) return std::nullopt;
      class_id = static_cast<std::uint16_t>(class_ids.size());
      class_ids.emplace(std::move(signature), class_id);
      representatives.push_back(first);
    } else {
      class_id = existing->second;
    }
    intervals.push_back({first, last, class_id});
  }
  if (class_ids.empty()) return std::nullopt;

  machine.class_count        = static_cast<std::uint16_t>(class_ids.size());
  std::size_t interval_index = 0;
  for (std::size_t value = 0; value < machine.byte_classes.size(); ++value) {
    while (interval_index + 1 < intervals.size() && value > intervals[interval_index].last)
      ++interval_index;
    machine.byte_classes[value] = intervals[interval_index].class_id;
  }
  for (auto interval : intervals) {
    if (interval.last < 256) continue;
    interval.first = std::max(interval.first, 256U);
    if (!machine.unicode_intervals.empty() &&
        machine.unicode_intervals.back().class_id == interval.class_id &&
        machine.unicode_intervals.back().last + 1 == interval.first) {
      machine.unicode_intervals.back().last = interval.last;
    } else {
      machine.unicode_intervals.push_back(interval);
    }
  }
  return representatives;
}

std::optional<glushkov_machine> make_glushkov_machine(instruction_ir const& ir, bool scan_input)
{
  auto graph = make_deterministic_graph(ir);
  if (!graph) return std::nullopt;

  std::vector<std::size_t> positions;
  std::vector<std::size_t> position_ids(graph->nodes.size(),
                                        std::numeric_limits<std::size_t>::max());
  for (std::size_t index = 0; index < graph->nodes.size(); ++index) {
    auto& node = graph->nodes[index];
    if (node.capture.has_value() || node.assertion.has_value()) return std::nullopt;
    if (!node.consumes) continue;
    if (positions.size() == 64U) return std::nullopt;
    position_ids[index] = positions.size();
    positions.push_back(index);
  }
  if (positions.empty()) return std::nullopt;

  struct closure_result {
    std::uint64_t positions = 0;
    bool accepts : 1        = false;
  };
  // epsilon closure stops at consuming nodes because those nodes are the positions.
  auto close = [&](std::vector<std::size_t> seeds) {
    closure_result result;
    std::vector<bool> visited(graph->nodes.size(), false);
    while (!seeds.empty()) {
      auto index = seeds.back();
      seeds.pop_back();
      if (index >= graph->nodes.size() || visited[index]) continue;
      visited[index] = true;

      auto& node = graph->nodes[index];
      if (node.accepts) result.accepts = true;
      if (node.consumes) {
        result.positions |= std::uint64_t{1} << position_ids[index];
        continue;
      }
      seeds.insert(seeds.end(), node.targets.begin(), node.targets.end());
    }
    return result;
  };

  glushkov_machine machine;
  machine.position_count           = static_cast<std::uint8_t>(positions.size());
  machine.scan_input               = scan_input;
  machine.accept_at_end            = ir.control.require_end;
  machine.fixed_match_bytes        = fixed_match_byte_width(*graph, ir.options.characters);
  machine.repeated_predicate_count = repeated_predicate_count(*graph);
  if (machine.repeated_predicate_count.has_value()) {
    auto repeated = std::find_if(
      graph->nodes.begin(), graph->nodes.end(), [](auto& node) { return node.consumes; });
    machine.repeated_predicate_class = repeated->predicate.recognized;
  }

  auto initial = close({graph->entry});
  if (initial.accepts || initial.positions == 0U) return std::nullopt;
  machine.first_set = initial.positions;

  auto follow = std::array<std::uint64_t, 64>{};
  for (std::size_t position = 0; position < positions.size(); ++position) {
    auto& node       = graph->nodes[positions[position]];
    auto successors  = close(node.targets);
    follow[position] = successors.positions;
    if (successors.accepts) machine.accept_mask |= std::uint64_t{1} << position;
  }
  if (machine.accept_mask == 0U) return std::nullopt;

  auto representatives = make_deterministic_alphabet(graph->nodes, machine.alphabet);
  if (!representatives) return std::nullopt;
  machine.reach_masks.reserve(representatives->size());
  for (auto representative : *representatives) {
    std::uint64_t reach = 0;
    for (std::size_t position = 0; position < positions.size(); ++position) {
      if (graph->nodes[positions[position]].predicate.matches(
            static_cast<char32_t>(representative))) {
        reach |= std::uint64_t{1} << position;
      }
    }
    machine.reach_masks.push_back(reach);
  }

  std::map<std::int32_t, std::uint64_t> span_sources;
  for (std::size_t position = 0; position < positions.size(); ++position) {
    for (std::size_t successor = 0; successor < positions.size(); ++successor) {
      if ((follow[position] & (std::uint64_t{1} << successor)) == 0U) continue;
      auto span = static_cast<std::int32_t>(successor) - static_cast<std::int32_t>(position);
      if (span > 0) {
        span_sources[span] |= std::uint64_t{1} << position;
      } else {
        machine.exception_mask |= std::uint64_t{1} << position;
        machine.exception_successors[position] |= std::uint64_t{1} << successor;
      }
    }
  }

  std::vector<std::pair<std::int32_t, std::uint64_t>> spans(span_sources.begin(),
                                                            span_sources.end());
  // common forward spans become shifts; uncommon and backward edges stay explicit.
  std::stable_sort(spans.begin(), spans.end(), [](auto& left, auto& right) {
    return std::popcount(left.second) > std::popcount(right.second);
  });
  auto shift_count = std::min<std::size_t>(spans.size(), 8U);
  machine.shifts.reserve(shift_count);
  for (std::size_t index = 0; index < shift_count; ++index) {
    machine.shifts.push_back({spans[index].second, static_cast<std::uint8_t>(spans[index].first)});
  }
  for (std::size_t index = shift_count; index < spans.size(); ++index) {
    auto sources = spans[index].second;
    while (sources != 0U) {
      auto position = static_cast<std::size_t>(std::countr_zero(sources));
      sources &= sources - 1U;
      auto successor = position + static_cast<std::size_t>(spans[index].first);
      machine.exception_mask |= std::uint64_t{1} << position;
      machine.exception_successors[position] |= std::uint64_t{1} << successor;
    }
  }

  auto first_positions = machine.first_set;
  std::optional<std::uint8_t> start_byte;
  while (first_positions != 0U) {
    auto position = static_cast<std::size_t>(std::countr_zero(first_positions));
    first_positions &= first_positions - 1U;
    auto& predicate = graph->nodes[positions[position]].predicate;
    if (!predicate.is_singleton() || predicate.singleton() > 0x7f) {
      start_byte.reset();
      break;
    }
    auto byte = static_cast<std::uint8_t>(predicate.singleton());
    if (start_byte.has_value() && *start_byte != byte) {
      start_byte.reset();
      break;
    }
    start_byte = byte;
  }
  machine.start_byte = start_byte;
  return machine;
}

bool prefer_glushkov(glushkov_machine const& machine,
                     std::optional<deterministic_machine> const& deterministic)
{
  if (!deterministic.has_value()) return true;

  auto exceptions = std::popcount(machine.exception_mask);
  // Small position machines stay entirely in registers and avoid even the compact DFA's table
  // lookup. Use larger machines only when they remove a large table or form a long linear graph.
  auto small = machine.position_count <= 8U && exceptions <= 2;
  auto large_linear =
    machine.position_count >= 32U && machine.shifts.size() == 1U && machine.exception_mask == 0U;
  return small || (deterministic->transition_address_space == 1U && exceptions <= 5) ||
         large_linear;
}

std::uint8_t assertion_bit(assertion_kind assertion)
{
  return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(assertion));
}

std::optional<deterministic_machine> make_assertion_deterministic_machine(
  instruction_ir const& ir, deterministic_nfa_graph const& graph, bool scan_input)
{
  deterministic_machine machine;
  machine.scan_input      = scan_input;
  machine.accept_at_end   = ir.control.require_end;
  machine.state_mask      = 32767U;
  machine.assertion_aware = true;
  for (auto& node : graph.nodes) {
    if (node.capture.has_value()) return std::nullopt;
    if (node.assertion.has_value()) machine.assertion_mask |= assertion_bit(*node.assertion);
  }
  if (machine.assertion_mask == 0) return std::nullopt;

  std::vector<std::uint8_t> context_masks;
  for (std::uint8_t mask = 0; mask < 64U; ++mask) {
    auto relevant = static_cast<std::uint8_t>(mask & machine.assertion_mask);
    if (std::find(context_masks.begin(), context_masks.end(), relevant) == context_masks.end()) {
      context_masks.push_back(relevant);
    }
  }
  machine.boundary_class_count = static_cast<std::uint8_t>(context_masks.size());

  auto representatives = make_deterministic_alphabet(graph.nodes, machine);
  if (!representatives) return std::nullopt;

  struct assertion_closure {
    std::vector<std::size_t> consumers = std::vector<std::size_t>{};
    bool accepts : 1                   = false;
  };
  auto close = [&](std::vector<std::size_t> seeds, std::uint8_t assertion_mask) {
    assertion_closure result;
    std::vector<bool> visited(graph.nodes.size());
    std::vector<std::size_t> work;
    for (auto seed = seeds.rbegin(); seed != seeds.rend(); ++seed)
      work.push_back(*seed);
    while (!work.empty()) {
      auto index = work.back();
      work.pop_back();
      if (visited[index]) continue;
      visited[index] = true;
      auto& node     = graph.nodes[index];
      if (node.assertion.has_value() && (assertion_mask & assertion_bit(*node.assertion)) == 0) {
        continue;
      }
      if (node.accepts) result.accepts = true;
      if (node.consumes) {
        result.consumers.push_back(index);
        continue;
      }
      for (auto target = node.targets.rbegin(); target != node.targets.rend(); ++target)
        work.push_back(*target);
    }
    std::sort(result.consumers.begin(), result.consumers.end());
    result.consumers.erase(std::unique(result.consumers.begin(), result.consumers.end()),
                           result.consumers.end());
    return result;
  };
  auto canonicalize = [](std::vector<std::size_t>& state) {
    std::sort(state.begin(), state.end());
    state.erase(std::unique(state.begin(), state.end()), state.end());
  };

  constexpr std::size_t max_table_items = 4U * 1024U * 1024U;
  std::map<std::vector<std::size_t>, std::uint16_t> state_ids;
  std::vector<std::vector<std::size_t>> states{{graph.entry}};
  state_ids.emplace(states.front(), 0);
  machine.dead_state = std::numeric_limits<std::uint16_t>::max();
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (states.size() * context_masks.size() * representatives->size() > max_table_items) {
      return std::nullopt;
    }
    for (auto context : context_masks) {
      auto seeds = states[state_index];
      if (scan_input) {
        seeds.push_back(graph.entry);
        canonicalize(seeds);
      }
      auto closure = close(std::move(seeds), context);
      machine.boundary_accepts.push_back(closure.accepts ? 1U : 0U);
      for (auto representative : *representatives) {
        std::vector<std::size_t> next;
        for (auto node_index : closure.consumers) {
          auto& node = graph.nodes[node_index];
          if (!node.predicate.matches(static_cast<char32_t>(representative))) continue;
          next.insert(next.end(), node.targets.begin(), node.targets.end());
        }
        canonicalize(next);
        auto existing        = state_ids.find(next);
        std::uint16_t target = 0;
        if (existing == state_ids.end()) {
          if (states.size() >= machine.state_mask) return std::nullopt;
          target = static_cast<std::uint16_t>(states.size());
          state_ids.emplace(next, target);
          states.push_back(std::move(next));
        } else {
          target = existing->second;
        }
        if (states[target].empty()) machine.dead_state = target;
        machine.transitions.push_back(
          static_cast<std::uint16_t>(target | (closure.accepts ? 0x8000U : 0U)));
      }
    }
  }
  machine.initial_state = 0;
  machine.state_count   = static_cast<std::uint16_t>(states.size());
  if (machine.transitions.size() * sizeof(std::uint16_t) > 32U * 1024U) {
    machine.transition_address_space = 1;
  }
  build_assertion_start_byte_filter(machine);
  return machine;
}

std::optional<deterministic_machine> make_deterministic_machine(instruction_ir const& ir,
                                                                bool scan_input,
                                                                bool preserve_priority)
{
  auto block_assertion = [](instruction_block const& block) -> std::optional<assertion_kind> {
    if (block.instructions.size() != 1U ||
        !std::holds_alternative<test_assertion>(block.instructions.front())) {
      return std::nullopt;
    }
    return std::get<test_assertion>(block.instructions.front()).kind;
  };
  std::optional<block_id> begin_anchor;
  std::optional<block_id> end_anchor;
  std::optional<assertion_kind> end_assertion;
  bool has_assertions = false;
  if (ir.entry < ir.blocks.size()) {
    auto assertion = block_assertion(ir.blocks[ir.entry]);
    if (assertion == assertion_kind::BEGIN_INPUT ||
        (assertion == assertion_kind::BEGIN_LINE && !ir.options.multiline)) {
      begin_anchor = ir.entry;
    }
  }
  if (ir.accept < ir.blocks.size()) {
    std::vector<block_id> predecessors;
    for (auto& block : ir.blocks) {
      for (auto edge : block.successors) {
        if (edge.target == ir.accept) predecessors.push_back(block.id);
      }
    }
    if (predecessors.size() == 1U) {
      auto assertion = block_assertion(ir.blocks[predecessors.front()]);
      if (assertion == assertion_kind::END_INPUT || assertion == assertion_kind::END_LINE) {
        end_anchor    = predecessors.front();
        end_assertion = assertion;
      }
    }
  }
  for (auto& block : ir.blocks) {
    for (auto& item : block.instructions) {
      if (std::holds_alternative<test_assertion>(item)) {
        has_assertions = true;
        if (preserve_priority && block.id != begin_anchor && block.id != end_anchor) {
          return std::nullopt;
        }
      }
    }
  }
  if (has_assertions && !preserve_priority) {
    auto graph = make_deterministic_graph(ir);
    if (!graph) return std::nullopt;
    return make_assertion_deterministic_machine(ir, *graph, scan_input);
  }
  if ((begin_anchor.has_value() || end_anchor.has_value()) &&
      ir.control.result != result_shape::BOOLEAN) {
    return std::nullopt;
  }

  std::vector<std::size_t> block_starts(ir.blocks.size());
  std::vector<std::size_t> block_lengths(ir.blocks.size(), 1);
  std::size_t node_count = 0;
  for (auto& block : ir.blocks) {
    match_character const* match  = nullptr;
    match_literal const* literal  = nullptr;
    can_peek const* peek          = nullptr;
    advance_cursor const* advance = nullptr;
    bool accepts                  = false;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<can_peek>(&item)) peek = candidate;
      if (auto* candidate = std::get_if<advance_cursor>(&item)) advance = candidate;
      if (std::holds_alternative<emit_accept>(item)) accepts = true;
    }
    if (match != nullptr && literal != nullptr) return std::nullopt;
    if ((match != nullptr || literal != nullptr) && accepts) return std::nullopt;
    if (match != nullptr && (peek == nullptr || peek->characters != 1 || advance == nullptr ||
                             advance->characters != 1)) {
      return std::nullopt;
    }
    if (literal != nullptr) {
      if (literal->value.empty() || peek == nullptr || peek->characters != literal->value.size() ||
          advance == nullptr || advance->characters != literal->value.size()) {
        return std::nullopt;
      }
      block_lengths[block.id] = literal->value.size();
    }
    block_starts[block.id] = node_count;
    node_count += block_lengths[block.id];
  }
  if (node_count == 0) return std::nullopt;

  std::vector<deterministic_nfa_node> nodes(node_count);
  for (auto& block : ir.blocks) {
    auto start                   = block_starts[block.id];
    match_character const* match = nullptr;
    match_literal const* literal = nullptr;
    write_capture const* capture = nullptr;
    for (auto& item : block.instructions) {
      if (auto* candidate = std::get_if<match_character>(&item)) match = candidate;
      if (auto* candidate = std::get_if<match_literal>(&item)) literal = candidate;
      if (auto* candidate = std::get_if<write_capture>(&item)) capture = candidate;
      if (std::holds_alternative<emit_accept>(item)) nodes[start].accepts = true;
    }
    if (capture != nullptr) nodes[start].capture = *capture;

    auto append_successors = [&](deterministic_nfa_node& node) {
      auto successors = block.successors;
      std::stable_sort(successors.begin(), successors.end(), [](auto& left, auto& right) {
        return left.priority < right.priority;
      });
      for (auto edge : successors)
        node.targets.push_back(block_starts[edge.target]);
    };

    if (match != nullptr) {
      nodes[start].predicate = match->predicate;
      nodes[start].consumes  = true;
      append_successors(nodes[start]);
    } else if (literal != nullptr) {
      for (std::size_t index = 0; index < literal->value.size(); ++index) {
        auto& node     = nodes[start + index];
        node.predicate = singleton_predicate(literal->value[index]);
        node.consumes  = true;
        if (index + 1 < literal->value.size()) {
          node.targets.push_back(start + index + 1);
        } else {
          append_successors(node);
        }
      }
    } else {
      append_successors(nodes[start]);
    }
  }

  auto bit_count  = nodes.size() + 1;
  auto word_count = (bit_count + 63) / 64;
  auto accept_bit = nodes.size();
  struct closure_state {
    std::vector<std::uint64_t> bits  = std::vector<std::uint64_t>{};
    std::vector<std::size_t> ordered = std::vector<std::size_t>{};
    std::vector<std::vector<deterministic_capture_action>> captures =
      std::vector<std::vector<deterministic_capture_action>>{};
  };
  struct closure_work_item {
    std::size_t node                                  = 0;
    std::vector<deterministic_capture_action> actions = std::vector<deterministic_capture_action>{};
  };
  auto capture_paths_deterministic = true;
  auto empty_bits                  = [&] { return std::vector<std::uint64_t>(word_count); };
  auto closure                     = [&](std::vector<std::size_t> const& seeds) {
    closure_state result{empty_bits(), {}, {}};
    auto visited = empty_bits();
    std::vector<std::optional<std::vector<deterministic_capture_action>>> visited_actions(
      nodes.size());
    std::vector<closure_work_item> work;
    for (auto seed = seeds.rbegin(); seed != seeds.rend(); ++seed)
      work.push_back({*seed, {}});
    while (!work.empty()) {
      auto current = std::move(work.back());
      work.pop_back();
      auto index = current.node;
      if (machine_bit(visited, index)) {
        if (visited_actions[index] != current.actions) capture_paths_deterministic = false;
        continue;
      }
      set_machine_bit(visited, index);
      visited_actions[index] = current.actions;
      auto& node             = nodes[index];
      if (node.capture.has_value()) {
        auto slot = node.capture->capture_index * 2U +
                    (node.capture->action == capture_action::END ? 1U : 0U);
        current.actions.push_back({slot, node.capture->action == capture_action::BEGIN});
      }
      if (node.accepts) {
        set_machine_bit(result.bits, accept_bit);
        result.ordered.push_back(accept_bit);
        result.captures.push_back(current.actions);
        if (preserve_priority) break;
      }
      if (node.consumes) {
        set_machine_bit(result.bits, index);
        result.ordered.push_back(index);
        result.captures.push_back(current.actions);
        continue;
      }
      for (auto target = node.targets.rbegin(); target != node.targets.rend(); ++target) {
        work.push_back({*target, current.actions});
      }
    }
    if (!preserve_priority) {
      std::sort(result.ordered.begin(), result.ordered.end());
      result.ordered.erase(std::unique(result.ordered.begin(), result.ordered.end()),
                           result.ordered.end());
      result.captures.assign(result.ordered.size(), {});
    }
    return result;
  };

  auto start_state = closure({block_starts[ir.entry]});

  constexpr std::uint32_t unicode_limit = 0x110000;
  std::vector<std::uint32_t> boundaries{0, 256, unicode_limit};
  for (auto& node : nodes) {
    if (!node.consumes) continue;
    if (node.predicate.recognized == predicate_class::ANY && !node.predicate.matches_newline) {
      boundaries.insert(boundaries.end(), {10, 11});
      if (node.predicate.extended_newline) {
        boundaries.insert(boundaries.end(), {13, 14, 133, 134, 8232, 8234});
      }
    }
    for (auto range : node.predicate.ranges) {
      auto first = static_cast<std::uint32_t>(range.first);
      auto last  = static_cast<std::uint32_t>(range.last);
      if (first < unicode_limit) boundaries.push_back(first);
      if (last < unicode_limit - 1) boundaries.push_back(last + 1);
    }
  }
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

  std::map<std::vector<std::uint64_t>, std::uint16_t> class_ids;
  std::vector<std::uint32_t> representatives;
  std::vector<deterministic_interval> intervals;
  for (std::size_t index = 0; index + 1 < boundaries.size(); ++index) {
    auto first = boundaries[index];
    auto last  = boundaries[index + 1] - 1;
    if (first > last || first >= unicode_limit) continue;
    auto signature = empty_bits();
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      if (nodes[node_index].consumes &&
          nodes[node_index].predicate.matches(static_cast<char32_t>(first))) {
        set_machine_bit(signature, node_index);
      }
    }
    auto existing          = class_ids.find(signature);
    std::uint16_t class_id = 0;
    if (existing == class_ids.end()) {
      if (class_ids.size() >= 32767) return std::nullopt;
      class_id = static_cast<std::uint16_t>(class_ids.size());
      class_ids.emplace(std::move(signature), class_id);
      representatives.push_back(first);
    } else {
      class_id = existing->second;
    }
    intervals.push_back({first, last, class_id});
  }
  if (class_ids.empty()) return std::nullopt;

  deterministic_machine machine;
  machine.class_count        = static_cast<std::uint16_t>(class_ids.size());
  machine.scan_input         = scan_input && !begin_anchor.has_value();
  machine.accept_at_end      = ir.control.require_end;
  machine.accept_assertion   = end_assertion;
  machine.state_mask         = preserve_priority ? 16383U : 32767U;
  std::size_t interval_index = 0;
  for (std::size_t value = 0; value < machine.byte_classes.size(); ++value) {
    while (interval_index + 1 < intervals.size() && value > intervals[interval_index].last)
      ++interval_index;
    machine.byte_classes[value] = intervals[interval_index].class_id;
  }
  for (auto interval : intervals) {
    if (interval.last < 256) continue;
    interval.first = std::max(interval.first, 256U);
    if (!machine.unicode_intervals.empty() &&
        machine.unicode_intervals.back().class_id == interval.class_id &&
        machine.unicode_intervals.back().last + 1 == interval.first) {
      machine.unicode_intervals.back().last = interval.last;
    } else {
      machine.unicode_intervals.push_back(interval);
    }
  }

  auto max_dfa_states                   = static_cast<std::size_t>(machine.state_mask);
  constexpr std::size_t max_table_items = 4 * 1024 * 1024;
  std::map<std::vector<std::size_t>, std::uint16_t> state_ids;
  std::vector<std::vector<std::uint64_t>> states;
  std::vector<std::vector<std::size_t>> state_orders;
  std::vector<std::vector<std::vector<deterministic_capture_action>>> state_capture_actions;
  state_ids.emplace(start_state.ordered, 0);
  states.push_back(start_state.bits);
  state_orders.push_back(start_state.ordered);
  state_capture_actions.push_back(start_state.captures);
  machine.dead_state   = std::numeric_limits<std::uint16_t>::max();
  auto strict_one_pass = !machine_bit(start_state.bits, accept_bit);
  auto terminal_accept = true;
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (states.size() * representatives.size() > max_table_items) return std::nullopt;
    for (auto representative : representatives) {
      std::vector<std::size_t> seeds;
      std::vector<deterministic_capture_action> transition_captures;
      std::vector<deterministic_capture_action> deferred_accept_captures;
      auto matching_consumers  = 0U;
      auto has_deferred_accept = false;
      for (std::size_t order_index = 0; order_index < state_orders[state_index].size();
           ++order_index) {
        auto node_index = state_orders[state_index][order_index];
        if (node_index == accept_bit) {
          if (preserve_priority) {
            has_deferred_accept      = true;
            deferred_accept_captures = state_capture_actions[state_index][order_index];
            break;
          }
          continue;
        }
        auto& node = nodes[node_index];
        if (!node.predicate.matches(static_cast<char32_t>(representative))) continue;
        if (matching_consumers++ == 0U) {
          transition_captures = state_capture_actions[state_index][order_index];
        }
        seeds.insert(seeds.end(), node.targets.begin(), node.targets.end());
      }
      if (matching_consumers > 1U) strict_one_pass = false;
      auto next              = closure(seeds);
      auto discovered_accept = machine_bit(next.bits, accept_bit);
      auto stop_before       = preserve_priority && has_deferred_accept && matching_consumers == 0U;
      if (preserve_priority && has_deferred_accept && matching_consumers != 0U &&
          !discovered_accept) {
        // preserve a lower-priority accept while descendants of earlier threads continue.
        set_machine_bit(next.bits, accept_bit);
        next.ordered.push_back(accept_bit);
        next.captures.push_back(std::move(deferred_accept_captures));
      }
      // A deferred acceptance wins before this character. Injecting a new scan start here would
      // make a nullable initial state accepting again and incorrectly consume the character.
      if (machine.scan_input && !stop_before) {
        for (std::size_t word = 0; word < next.bits.size(); ++word)
          next.bits[word] |= start_state.bits[word];
        for (std::size_t start_index = 0; start_index < start_state.ordered.size(); ++start_index) {
          auto node_index = start_state.ordered[start_index];
          if (std::find(next.ordered.begin(), next.ordered.end(), node_index) ==
              next.ordered.end()) {
            next.ordered.push_back(node_index);
            next.captures.push_back(start_state.captures[start_index]);
          }
        }
        if (!preserve_priority) {
          std::sort(next.ordered.begin(), next.ordered.end());
          next.ordered.erase(std::unique(next.ordered.begin(), next.ordered.end()),
                             next.ordered.end());
          next.captures.assign(next.ordered.size(), {});
        }
      }
      auto existing        = state_ids.find(next.ordered);
      std::uint16_t target = 0;
      if (existing == state_ids.end()) {
        if (states.size() >= max_dfa_states) return std::nullopt;
        target = static_cast<std::uint16_t>(states.size());
        state_ids.emplace(next.ordered, target);
        states.push_back(std::move(next.bits));
        state_orders.push_back(std::move(next.ordered));
        state_capture_actions.push_back(std::move(next.captures));
      } else {
        target = existing->second;
        if (state_capture_actions[target] != next.captures) { capture_paths_deterministic = false; }
      }
      if (std::none_of(
            states[target].begin(), states[target].end(), [](auto word) { return word != 0; })) {
        machine.dead_state = target;
      }
      auto accepts  = machine_bit(states[target], accept_bit);
      auto consumes = std::any_of(state_orders[target].begin(),
                                  state_orders[target].end(),
                                  [&](auto node_index) { return node_index != accept_bit; });
      if (accepts && consumes) terminal_accept = false;
      std::vector<deterministic_capture_action> accept_captures;
      if (accepts) {
        auto accept =
          std::find(state_orders[target].begin(), state_orders[target].end(), accept_bit);
        auto index      = static_cast<std::size_t>(accept - state_orders[target].begin());
        accept_captures = state_capture_actions[target][index];
      }
      auto update_accept =
        preserve_priority ? discovered_accept : machine_bit(states[target], accept_bit);
      if (update_accept) target |= 0x8000U;
      if (stop_before) target |= 0x4000U;
      machine.transitions.push_back(target);
      machine.transition_capture_actions.push_back(std::move(transition_captures));
      machine.accept_capture_actions.push_back(std::move(accept_captures));
    }
  }
  machine.state_count      = static_cast<std::uint16_t>(states.size());
  machine.initial_state    = machine_bit(start_state.bits, accept_bit) ? 0x8000U : 0U;
  machine.capture_one_pass = strict_one_pass && terminal_accept && capture_paths_deterministic;
  if (machine.transitions.size() * sizeof(std::uint16_t) > 32U * 1024U) {
    machine.transition_address_space = 1;
  }
  build_start_byte_filter(machine);
  build_restart_acceleration(machine);
  return machine;
}

class nvvm_ir_renderer {
 public:
  nvvm_ir_renderer(instruction_ir const& ir, nvvm_ir_codegen_options const& options)
    : ir_(ir), options_(options), public_execute_function_(options.execute_function)
  {
    exact_ascii_literal_metadata_ = exact_ascii_literal();
    auto result                   = ir_.control.result;
    auto adapt_count              = result == result_shape::MATCH_COUNT;
    auto adapt_find               = result == result_shape::MATCH_SPAN &&
                      ir_.selected_operation.kind == operation_kind::FIND &&
                      !ir_.options.find_match_end_observable;
    if (begins_at_input_start() && (adapt_count || adapt_find)) {
      anchored_boolean_result_  = result;
      options_.execute_function = name("anchored_boolean_execute");
      ir_.control.result        = result_shape::BOOLEAN;
    }
  }

  compile_result render()
  {
    require_codegen_ir(ir_);
    require_identifier(options_.symbol_prefix, "symbol_prefix");
    require_identifier(options_.execute_function, "execute_function");
    require_identifier(public_execute_function_, "execute_function");
    whole_match_captures_ = whole_match_captures();
    capture_slots_        = live_capture_slots();
    if (ir_.control.result == result_shape::CAPTURES && ir_.options.extract_capture_group) {
      for (auto& block : ir_.blocks) {
        std::erase_if(block.instructions, [&](instruction const& item) {
          auto* capture = std::get_if<write_capture>(&item);
          return capture && capture->capture_index != *ir_.options.extract_capture_group;
        });
      }
    }
    if (!whole_match_captures_.empty()) {
      for (auto& block : ir_.blocks) {
        std::erase_if(block.instructions, [&](instruction const& item) {
          auto* capture = std::get_if<write_capture>(&item);
          return capture != nullptr &&
                 std::find(whole_match_captures_.begin(),
                           whole_match_captures_.end(),
                           capture->capture_index) != whole_match_captures_.end();
        });
      }
    }
    auto boolean_result = ir_.control.result == result_shape::BOOLEAN;
    string_operations_  = string_operation_plan();
    fixed_ascii_suffix_ = string_operations_.has_value() ? std::nullopt : fixed_ascii_suffix_plan();
    line_tail_literal_  = string_operations_.has_value() || fixed_ascii_suffix_.has_value()
                            ? std::nullopt
                            : line_tail_literal();
    word_run_minimum_   = string_operations_.has_value() || fixed_ascii_suffix_.has_value() ||
                            line_tail_literal_.has_value()
                            ? std::nullopt
                            : word_run_minimum();
    ascii_literal_      = word_run_minimum_.has_value() || line_tail_literal_.has_value()
                            ? std::nullopt
                            : exact_ascii_literal();
    if (string_operations_.has_value() || fixed_ascii_suffix_.has_value()) {
      ascii_literal_.reset();
    }
    utf8_literal_        = string_operations_.has_value() || fixed_ascii_suffix_.has_value() ||
                        line_tail_literal_.has_value() || word_run_minimum_.has_value() ||
                        ascii_literal_.has_value()
                             ? std::nullopt
                             : exact_utf8_literal();
    exact_literal_bytes_ = exact_ascii_literal_metadata_;
    if (!exact_literal_bytes_.has_value() && utf8_literal_.has_value()) {
      exact_literal_bytes_ = encode_utf8_literal(utf8_literal_->codepoints);
    }
    if (boolean_result && !ir_.control.scan_input) { utf8_literal_.reset(); }
    utf8_literal_pivot_ =
      utf8_literal_.has_value() ? utf8_literal_pivot(*utf8_literal_) : std::nullopt;
    prefix_seek_byte_ = required_ascii_prefix();
    if (string_operations_.has_value() || fixed_ascii_suffix_.has_value() ||
        line_tail_literal_.has_value() || word_run_minimum_.has_value() ||
        ascii_literal_.has_value() || utf8_literal_.has_value()) {
      prefix_seek_byte_.reset();
    }
    // Very short early-hit scans favor the compact DFA; longer literals repay wide candidate
    // scans and packed verification.
    if (boolean_result && ir_.control.scan_input && ascii_literal_.has_value() &&
        ascii_literal_->size() > 1U && ascii_literal_->size() < 8U) {
      ascii_literal_.reset();
    }
    if (!line_tail_literal_.has_value() && !word_run_minimum_.has_value() &&
        !ascii_literal_.has_value() && !utf8_literal_.has_value()) {
      auto replacement_uses_captures =
        std::any_of(ir_.replacement.begin(), ir_.replacement.end(), [](auto& token) {
          return token.type == replacement_token::kind::CAPTURE;
        });
      auto span_or_count =
        ir_.control.result == result_shape::MATCH_SPAN ||
        ir_.control.result == result_shape::MATCH_COUNT ||
        (ir_.control.result == result_shape::REPLACEMENT && !replacement_uses_captures) ||
        ir_.control.result == result_shape::SPLIT_FIELDS ||
        (ir_.control.result == result_shape::CAPTURES && !uses_capture_buffer());
      // The streaming position automaton wins when every input position is a plausible restart.
      // A known prefix instead favors the deterministic executor's restart acceleration, and
      // capture substitutions need its one-pass capture propagation rather than a second
      // span-recovery pass.
      auto accelerated_restart = required_ascii_prefix().has_value();
      auto streaming_span_result =
        span_or_count && !uses_capture_buffer() && !ir_.has_lazy_quantifier && !accelerated_restart;
      if ((boolean_result && !begins_at_input_start()) || streaming_span_result) {
        glushkov_      = make_glushkov_machine(ir_, ir_.control.scan_input);
        deterministic_ = make_deterministic_machine(ir_, ir_.control.scan_input, false);
      } else if (boolean_result && begins_at_input_start() &&
                 ir_.blocks[ir_.entry].successors.size() == 1) {
        // the selected successor is already constrained to byte zero by the removed assertion.
        auto anchored_ir               = ir_;
        anchored_ir.entry              = ir_.blocks[ir_.entry].successors.front().target;
        anchored_ir.control.scan_input = false;
        anchored_ir.blocks[ir_.entry].instructions.clear();
        anchored_ir.blocks[ir_.entry].successors.clear();
        deterministic_ = make_deterministic_machine(anchored_ir, false, false);
      } else {
        deterministic_ = make_deterministic_machine(
          ir_, boolean_result && ir_.control.scan_input, !boolean_result);
      }
      // Nullable expressions and large graphs may not have a streaming position representation.
      // Rebuild their fallback with priority preservation rather than retaining the speculative
      // unordered DFA used only to compare against a successful streaming plan.
      if (streaming_span_result && !glushkov_.has_value()) {
        deterministic_ = make_deterministic_machine(ir_, false, true);
      }
      // This assertion machine records longest matches, so keep priority-sensitive syntax on the
      // ordered executor until assertion closures carry branch-priority metadata.
      auto assertion_candidate = !boolean_result && !deterministic_.has_value() &&
                                 !uses_capture_buffer() && !ir_.has_alternation &&
                                 ir_.pattern.find('?') == std::string::npos;
      auto keep_prefix_count =
        ir_.control.result == result_shape::MATCH_COUNT && required_ascii_prefix().has_value();
      if (assertion_candidate && !keep_prefix_count) {
        auto graph = make_deterministic_graph(ir_);
        if (graph.has_value()) {
          deterministic_ = make_assertion_deterministic_machine(ir_, *graph, false);
        }
      }
    }
    if (glushkov_.has_value()) {
      if (!boolean_result || prefer_glushkov(*glushkov_, deterministic_)) {
        deterministic_.reset();
      } else {
        glushkov_.reset();
      }
    }
    if (boolean_result && !glushkov_.has_value() && !deterministic_.has_value() &&
        ir_.control.scan_input) {
      deterministic_ = make_deterministic_machine(ir_, true, true);
    }
    auto tagged_result = ir_.control.result == result_shape::CAPTURES &&
                         deterministic_.has_value() && deterministic_->capture_one_pass;
    if (!boolean_result &&
        (!deterministic_.has_value() || (uses_capture_buffer() && !tagged_result))) {
      deterministic_.reset();
    }
    if (!ascii_literal_.has_value() && !utf8_literal_.has_value() && !glushkov_.has_value() &&
        !deterministic_.has_value()) {
      mandatory_ascii_literal_ = mandatory_ascii_literal();
    }
    auto executor      = executor_kind::ITERATIVE_THOMPSON;
    auto executor_name = std::string_view{"iterative Thompson"};
    if (string_operations_.has_value()) {
      executor      = executor_kind::STRING_OPERATIONS;
      executor_name = "generated string operations";
    } else if (fixed_ascii_suffix_.has_value()) {
      executor      = executor_kind::STRING_OPERATIONS;
      executor_name = "generated fixed-width ASCII suffix";
    } else if (line_tail_literal_.has_value()) {
      executor      = executor_kind::STRING_OPERATIONS;
      executor_name = "generated final-line tail search";
    } else if (word_run_minimum_.has_value()) {
      executor      = executor_kind::WORD_RUN;
      executor_name = ir_.options.ascii_classes ? "ASCII word run" : "Unicode word run";
    } else if (ascii_literal_.has_value()) {
      executor = ascii_literal_->size() == 1U ? executor_kind::SINGLE_BYTE_LITERAL
                                              : executor_kind::PACKED_ASCII_LITERAL;
      executor_name =
        ascii_literal_->size() == 1U ? "single-byte literal scan" : "packed ASCII literal scan";
    } else if (utf8_literal_.has_value()) {
      executor      = utf8_literal_pivot_.has_value() ? executor_kind::PACKED_UTF8_LITERAL
                                                      : executor_kind::UTF8_KMP_LITERAL;
      executor_name = utf8_literal_pivot_.has_value()
                        ? (ir_.control.result == result_shape::MATCH_COUNT
                             ? "guarded-pivot packed UTF-8 literal scan"
                             : "hybrid guarded-pivot/KMP UTF-8 literal scan")
                        : "UTF-8 KMP literal scan";
    } else if (glushkov_.has_value()) {
      executor =
        boolean_result ? executor_kind::GLUSHKOV : executor_kind::STREAMING_PRIORITIZED_GLUSHKOV;
      executor_name =
        boolean_result ? "bit-parallel Glushkov NFA" : "streaming prioritized Glushkov NFA";
    } else if (deterministic_.has_value()) {
      executor      = deterministic_->assertion_aware ? executor_kind::ASSERTION_AWARE_DETERMINISTIC
                      : boolean_result                ? executor_kind::DETERMINISTIC
                      : tagged_result ? executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC
                                      : executor_kind::PRIORITIZED_DETERMINISTIC;
      executor_name = deterministic_->assertion_aware ? "assertion-aware deterministic table"
                      : boolean_result                ? "deterministic table"
                      : tagged_result                 ? "tagged prioritized deterministic table"
                                                      : "prioritized deterministic table";
    }
    output_.emit("{}",
                 render_nvvm_template(
                   regex_ir_nvvm_templates::regex_module,
                   {{"@PATTERN@", escaped_comment(ir_.pattern)}, {"@EXECUTOR@", executor_name}}));
    if (mandatory_ascii_literal_.has_value()) {
      output_.emit("; mandatory ASCII literal filter: {}",
                   escaped_comment(*mandatory_ascii_literal_));
    }
    if (glushkov_.has_value()) {
      output_.emit("; glushkov positions: {}, alphabet classes: {}, shifts: {}, exceptions: {}",
                   glushkov_->position_count,
                   glushkov_->alphabet.class_count,
                   glushkov_->shifts.size(),
                   std::popcount(glushkov_->exception_mask));
    } else if (deterministic_.has_value()) {
      output_.emit("; dfa states: {}, alphabet classes: {}",
                   deterministic_->state_count,
                   deterministic_->class_count);
    }
    output_.blank();

    if (options_.emit_general_functions) {
      auto emit_function_section = [&](std::string_view section) {
        output_.emit("{}",
                     nvvm_template_section(regex_ir_nvvm_templates::regex_functions, section));
      };
      auto all = options_.emit_all_general_functions;
      emit_function_section("core");
      if (all || ir_.options.characters == character_mode::BYTES) {
        emit_function_section("decode_width_bytes");
        emit_function_section("decode_codepoint_bytes");
      }
      if (all || ir_.options.characters == character_mode::UTF8) {
        emit_function_section("decode_width_utf8");
        emit_function_section("decode_codepoint_utf8");
        emit_function_section("decode_utf8_packed");
      }
      if (all || prefix_seek_byte_.has_value() || line_tail_literal_.has_value() ||
          utf8_literal_pivot_.has_value() || ascii_literal_.has_value() ||
          (glushkov_.has_value() && glushkov_->fixed_match_bytes.has_value() &&
           glushkov_->start_byte.has_value())) {
        auto ascii_boolean = ir_.control.result == result_shape::BOOLEAN &&
                             ir_.control.scan_input && ascii_literal_.has_value() &&
                             ascii_literal_->size() >= 8U;
        emit_function_section(ascii_boolean ? "seek_byte_words" : "seek_byte");
      }
      auto unicode_classifier =
        (glushkov_.has_value() && glushkov_->alphabet.unicode_intervals.size() > 8U) ||
        (deterministic_.has_value() && deterministic_->unicode_intervals.size() > 8U);
      if (all || unicode_classifier) { emit_function_section("unicode_class"); }
      if (all || ir_.options.extended_newline) { emit_function_section("extended_newline"); }
      output_.blank();
    }
    if (string_operations_.has_value()) {
      emit_string_operation_execute(*string_operations_);
    } else if (fixed_ascii_suffix_.has_value()) {
      emit_fixed_ascii_suffix_execute(*fixed_ascii_suffix_);
    } else if (line_tail_literal_.has_value()) {
      emit_line_tail_execute(*line_tail_literal_);
    } else if (word_run_minimum_.has_value()) {
      emit_is_word();
      emit_word_run_execute(*word_run_minimum_);
    } else if (ascii_literal_.has_value() && boolean_result) {
      emit_ascii_literal_execute(*ascii_literal_);
    } else if (utf8_literal_.has_value() && boolean_result) {
      emit_utf8_literal_find_from(*utf8_literal_);
      emit_utf8_kmp_execute();
    } else if (glushkov_.has_value() && boolean_result) {
      emit_glushkov_globals(*glushkov_);
      emit_deterministic_classifier(glushkov_->alphabet);
      emit_glushkov_reach(*glushkov_);
      emit_glushkov_follow(*glushkov_);
      emit_glushkov_execute(*glushkov_);
    } else if (deterministic_.has_value() && boolean_result) {
      if (deterministic_->assertion_aware) {
        auto word_assertions =
          static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                    assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
        if ((deterministic_->assertion_mask & word_assertions) != 0) emit_is_word();
      } else if (deterministic_->accept_assertion.has_value()) {
        emit_advance();
        emit_is_word();
        emit_previous_position();
        emit_assertion();
      }
      emit_deterministic_globals(*deterministic_);
      emit_deterministic_classifier(*deterministic_);
      if (deterministic_->assertion_aware) {
        emit_deterministic_boundary_classifier(*deterministic_);
        emit_assertion_deterministic_execute(*deterministic_);
      } else {
        emit_deterministic_execute(*deterministic_);
      }
    } else {
      emit_optimizer_intrinsics();
      emit_advance();
      emit_replacement_globals();
      if (mandatory_ascii_literal_.has_value()) {
        emit_ascii_literal_at(*mandatory_ascii_literal_);
        emit_mandatory_literal_filter(*mandatory_ascii_literal_);
      }
      if (ascii_literal_.has_value()) {
        if (ascii_literal_->size() == 1U) {
          emit_single_byte_find_from(static_cast<std::uint8_t>(ascii_literal_->front()));
        } else {
          emit_ascii_literal_at(*ascii_literal_);
          emit_ascii_literal_find_from(*ascii_literal_);
        }
      } else if (utf8_literal_.has_value()) {
        emit_utf8_literal_find_from(*utf8_literal_);
      } else if (glushkov_.has_value()) {
        if (ir_.control.result == result_shape::MATCH_COUNT &&
            glushkov_->repeated_predicate_count.has_value() && !ir_.options.ascii_classes) {
          repeated_builtin_ = adapted_builtin(glushkov_->repeated_predicate_class);
        }
        if (repeated_builtin_.has_value()) {
          emit_repeated_builtin_find(*glushkov_->repeated_predicate_count);
        } else {
          emit_glushkov_globals(*glushkov_);
          emit_deterministic_classifier(glushkov_->alphabet);
          emit_glushkov_reach(*glushkov_);
          emit_glushkov_follow(*glushkov_);
          emit_glushkov_find_from(*glushkov_);
        }
      } else if (deterministic_.has_value()) {
        if (deterministic_->assertion_aware) {
          auto word_assertions =
            static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                      assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
          if ((deterministic_->assertion_mask & word_assertions) != 0) emit_is_word();
          emit_previous_position();
        }
        emit_deterministic_globals(*deterministic_);
        emit_deterministic_classifier(*deterministic_);
        if (deterministic_->assertion_aware) {
          emit_deterministic_boundary_classifier(*deterministic_);
          emit_assertion_deterministic_find_from(*deterministic_);
        } else if (tagged_result) {
          emit_tagged_deterministic_find_from(*deterministic_);
        } else {
          emit_deterministic_find_from(*deterministic_);
        }
      } else {
        prepare_thompson();
        emit_is_word();
        emit_previous_position();
        emit_assertion();
        emit_predicate_helpers();
        emit_blocks();
      }
      if (boolean_result) {
        emit_execute();
      } else {
        if (!deterministic_.has_value() && !glushkov_.has_value() && !ascii_literal_.has_value() &&
            !utf8_literal_.has_value()) {
          emit_find_from();
        }
        switch (ir_.control.result) {
          case result_shape::MATCH_SPAN:
            if (ir_.selected_operation.kind == operation_kind::FIND_ALL) {
              emit_find_all_execute();
            } else {
              emit_find_execute();
            }
            break;
          case result_shape::MATCH_COUNT: emit_count_execute(); break;
          case result_shape::CAPTURES: emit_capture_execute(); break;
          case result_shape::REPLACEMENT: emit_replace_execute(); break;
          case result_shape::SPLIT_FIELDS: emit_split_execute(); break;
          case result_shape::BOOLEAN: break;
        }
      }
    }
    emit_anchored_boolean_adapter();
    output_.emit(
      "{}", nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "module_metadata"));
    auto executor_states  = deterministic_.has_value()
                              ? static_cast<std::uint32_t>(deterministic_->state_count)
                            : glushkov_.has_value() ? glushkov_->position_count
                                                    : std::uint32_t{0};
    auto alphabet_classes = deterministic_.has_value() ? deterministic_->class_count
                            : glushkov_.has_value()    ? glushkov_->alphabet.class_count
                                                       : std::uint32_t{0};
    auto module           = output_.take();
    auto replace          = [&](std::string_view token, std::string_view value) {
      for (auto pos = module.find(token); pos != std::string::npos;
           pos      = module.find(token, pos + value.size())) {
        module.replace(pos, token.size(), value);
      }
    };
    replace("@workspace@", workspace_bytes_ != 0 ? "i8* %workspace, " : "");
    if (workspace_bytes_ != 0) replace("nounwind readonly", "nounwind");
    if (thompson_) executor_states = static_cast<std::uint32_t>(thompson_->nodes.size());
    return {std::move(module),
            ir_.capture_count,
            executor,
            executor_states,
            alphabet_classes,
            std::move(exact_ascii_literal_metadata_),
            std::move(exact_literal_bytes_),
            repeated_builtin_,
            workspace_bytes_};
  }

 private:
  [[nodiscard]] std::string name(std::string_view suffix) const
  {
    if (suffix == "load_byte") { return "regex_ir_load_byte"; }
    if (suffix == "decode_width") {
      return ir_.options.characters == character_mode::BYTES ? "regex_ir_decode_width_bytes"
                                                             : "regex_ir_decode_width_utf8";
    }
    if (suffix == "decode_codepoint") {
      return ir_.options.characters == character_mode::BYTES ? "regex_ir_decode_codepoint_bytes"
                                                             : "regex_ir_decode_codepoint_utf8";
    }
    if (suffix == "decode_packed") { return "regex_ir_decode_utf8_packed"; }
    if (suffix == "advance") {
      return ir_.options.characters == character_mode::BYTES ? "regex_ir_advance_bytes"
                                                             : "regex_ir_advance_utf8";
    }
    if (suffix == "is_word" && !uses_unicode_word_boundaries()) { return "regex_ir_is_ascii_word"; }
    if (suffix == "extended_newline_flags") { return "regex_ir_extended_newline_flags"; }
    return nvvm_symbol(options_.symbol_prefix, suffix);
  }

  enum class string_operation_kind : std::uint8_t {
    BEGINS_WITH,
    ENDS_WITH,
    ENDS_LINE,
    EQUALS,
    EQUALS_LINE
  };

  struct string_operation {
    string_operation_kind kind;
    std::string literal;
  };

  struct fixed_ascii_suffix {
    std::vector<character_predicate> predicates;
    bool begins   : 1;
    bool line_end : 1;
  };

  [[nodiscard]] std::optional<fixed_ascii_suffix> fixed_ascii_suffix_plan() const
  {
    if (ir_.control.result != result_shape::BOOLEAN || ir_.entry >= ir_.blocks.size() ||
        ir_.accept >= ir_.blocks.size()) {
      return std::nullopt;
    }

    auto begins             = !ir_.control.scan_input;
    auto ends               = false;
    auto line_end           = false;
    auto predicates         = std::vector<character_predicate>{};
    auto eligible_predicate = [](character_predicate const& predicate) {
      return predicate.recognized != predicate_class::ANY && !predicate.negated &&
             !predicate.ranges.empty() &&
             std::all_of(predicate.ranges.begin(), predicate.ranges.end(), [](auto& range) {
               return range.last <= 0x7f;
             });
    };

    auto visited = std::vector<bool>(ir_.blocks.size(), false);
    auto current = ir_.entry;
    while (current < ir_.blocks.size() && !visited[current]) {
      visited[current] = true;
      auto& block      = ir_.blocks[current];
      if (current == ir_.accept) {
        auto accepting = block.instructions.size() == 1U &&
                         std::holds_alternative<emit_accept>(block.instructions.front()) &&
                         block.successors.empty();
        if (!accepting || predicates.empty() || !ends) return std::nullopt;
        return fixed_ascii_suffix{std::move(predicates), begins, line_end};
      }
      if (block.successors.size() != 1U) return std::nullopt;

      auto consumed = std::vector<character_predicate>{};
      std::optional<std::uint32_t> peek_count;
      std::optional<std::uint32_t> advance_count;
      auto reads_character = false;
      for (auto& item : block.instructions) {
        if (auto* assertion = std::get_if<test_assertion>(&item)) {
          if (assertion->kind == assertion_kind::BEGIN_INPUT ||
              (assertion->kind == assertion_kind::BEGIN_LINE && !ir_.options.multiline)) {
            if (!predicates.empty()) return std::nullopt;
            begins = true;
          } else if (assertion->kind == assertion_kind::END_INPUT) {
            if (consumed.size() != 0U) return std::nullopt;
            ends = true;
          } else if (assertion->kind == assertion_kind::END_LINE && !ir_.options.multiline &&
                     !ir_.options.extended_newline) {
            if (consumed.size() != 0U) return std::nullopt;
            ends     = true;
            line_end = true;
          } else {
            return std::nullopt;
          }
        } else if (auto* peek = std::get_if<can_peek>(&item)) {
          if (peek_count.has_value()) return std::nullopt;
          peek_count = peek->characters;
        } else if (std::holds_alternative<read_character>(item)) {
          if (reads_character) return std::nullopt;
          reads_character = true;
        } else if (auto* match = std::get_if<match_character>(&item)) {
          if (!consumed.empty() || !eligible_predicate(match->predicate)) return std::nullopt;
          consumed.push_back(match->predicate);
        } else if (auto* literal = std::get_if<match_literal>(&item)) {
          if (!consumed.empty()) return std::nullopt;
          for (auto codepoint : literal->value) {
            if (codepoint > 0x7f) return std::nullopt;
            character_predicate predicate;
            predicate.ranges.push_back({codepoint, codepoint});
            consumed.push_back(std::move(predicate));
          }
        } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
          if (advance_count.has_value()) return std::nullopt;
          advance_count = advance->characters;
        } else {
          return std::nullopt;
        }
      }

      if (!consumed.empty()) {
        auto count = static_cast<std::uint32_t>(consumed.size());
        if (!peek_count.has_value() || *peek_count != count || !advance_count.has_value() ||
            *advance_count != count || (reads_character && count != 1U) || ends) {
          return std::nullopt;
        }
        predicates.insert(predicates.end(), consumed.begin(), consumed.end());
      } else if (!block.instructions.empty() &&
                 !std::holds_alternative<test_assertion>(block.instructions.front())) {
        return std::nullopt;
      }
      current = block.successors.front().target;
    }
    return std::nullopt;
  }

  void emit_fixed_ascii_suffix_execute(fixed_ascii_suffix const& plan)
  {
    auto comparisons = std::string{};
    auto result      = std::string{};
    for (std::size_t index = 0; index < plan.predicates.size(); ++index) {
      auto& predicate = plan.predicates[index];
      std::format_to(std::back_inserter(comparisons),
                     "{}",
                     regex_ir::nvvm_template(
                       regex_ir_nvvm_templates::regex_operations, "fixed_ascii_load_byte", index));
      auto predicate_result = std::string{};
      for (std::size_t range_index = 0; range_index < predicate.ranges.size(); ++range_index) {
        auto range        = predicate.ranges[range_index];
        auto range_result = std::format("range_{}_{}", index, range_index);
        if (range.first == range.last) {
          std::format_to(std::back_inserter(comparisons),
                         "  %{0} = icmp eq i32 %value_{1}, {2}\n",
                         range_result,
                         index,
                         static_cast<std::uint32_t>(range.first));
        } else {
          std::format_to(std::back_inserter(comparisons),
                         "{}",
                         regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                                 "fixed_ascii_range",
                                                 index,
                                                 range_index,
                                                 static_cast<std::uint32_t>(range.first),
                                                 static_cast<std::uint32_t>(range.last),
                                                 range_result));
        }
        if (predicate_result.empty()) {
          predicate_result = std::format("%{}", range_result);
        } else {
          auto combined = std::format("predicate_{}_{}", index, range_index);
          std::format_to(std::back_inserter(comparisons),
                         "  %{0} = or i1 {1}, %{2}\n",
                         combined,
                         predicate_result,
                         range_result);
          predicate_result = std::format("%{}", combined);
        }
      }
      if (result.empty()) {
        result = predicate_result;
      } else {
        auto combined = std::format("matched_through_{}", index);
        std::format_to(std::back_inserter(comparisons),
                       "  %{0} = and i1 {1}, {2}\n",
                       combined,
                       result,
                       predicate_result);
        result = std::format("%{}", combined);
      }
    }

    auto matcher = name("fixed_ascii_suffix_at");
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "fixed_ascii_class_at",
                                         matcher,
                                         comparisons,
                                         result));
    auto size = plan.predicates.size();
    if (plan.begins && plan.line_end) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "string_equals_line",
                                           options_.execute_function,
                                           size,
                                           size + 1U,
                                           name("load_byte"),
                                           matcher));
    } else if (plan.begins) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "string_equals",
                                           options_.execute_function,
                                           size,
                                           matcher));
    } else if (plan.line_end) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "string_ends_line",
                                           options_.execute_function,
                                           size,
                                           matcher,
                                           name("load_byte")));
    } else {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "string_ends_with",
                                           options_.execute_function,
                                           size,
                                           matcher));
    }
    output_.blank();
  }

  [[nodiscard]] std::optional<string_operation> string_operation_plan() const
  {
    if (ir_.control.result != result_shape::BOOLEAN || ir_.entry >= ir_.blocks.size() ||
        ir_.accept >= ir_.blocks.size()) {
      return std::nullopt;
    }

    auto begins            = !ir_.control.scan_input;
    auto ends              = ir_.control.require_end;
    auto line_end          = false;
    auto explicit_boundary = false;
    std::vector<bool> visited(ir_.blocks.size(), false);
    std::string literal;
    auto current = ir_.entry;
    while (current < ir_.blocks.size() && !visited[current]) {
      visited[current] = true;
      auto& block      = ir_.blocks[current];
      if (current == ir_.accept) {
        auto accepting = block.instructions.size() == 1U &&
                         std::holds_alternative<emit_accept>(block.instructions.front()) &&
                         block.successors.empty();
        if (!accepting || literal.empty() || !explicit_boundary || (!begins && !ends)) {
          return std::nullopt;
        }
        auto kind = begins && ends && line_end ? string_operation_kind::EQUALS_LINE
                    : begins && ends           ? string_operation_kind::EQUALS
                    : begins                   ? string_operation_kind::BEGINS_WITH
                    : line_end                 ? string_operation_kind::ENDS_LINE
                                               : string_operation_kind::ENDS_WITH;
        return string_operation{kind, std::move(literal)};
      }
      if (block.successors.size() != 1U) return std::nullopt;

      std::u32string consumed;
      std::optional<std::uint32_t> peek_count;
      std::optional<std::uint32_t> advance_count;
      auto reads_character = false;
      for (auto& item : block.instructions) {
        if (auto* assertion = std::get_if<test_assertion>(&item)) {
          if (assertion->kind == assertion_kind::BEGIN_INPUT ||
              (assertion->kind == assertion_kind::BEGIN_LINE && !ir_.options.multiline)) {
            if (!literal.empty()) return std::nullopt;
            begins            = true;
            explicit_boundary = true;
          } else if (assertion->kind == assertion_kind::END_INPUT ||
                     (assertion->kind == assertion_kind::END_LINE && !ir_.options.multiline &&
                      !ir_.options.extended_newline)) {
            if (literal.empty()) return std::nullopt;
            ends              = true;
            line_end          = assertion->kind == assertion_kind::END_LINE;
            explicit_boundary = true;
          } else {
            return std::nullopt;
          }
        } else if (auto* peek = std::get_if<can_peek>(&item)) {
          if (peek_count.has_value()) return std::nullopt;
          peek_count = peek->characters;
        } else if (std::holds_alternative<read_character>(item)) {
          if (reads_character) return std::nullopt;
          reads_character = true;
        } else if (auto* character_match = std::get_if<match_character>(&item)) {
          if (!consumed.empty() || !character_match->predicate.is_singleton()) {
            return std::nullopt;
          }
          consumed.push_back(character_match->predicate.singleton());
        } else if (auto* literal_match = std::get_if<match_literal>(&item)) {
          if (!consumed.empty()) return std::nullopt;
          consumed = literal_match->value;
        } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
          if (advance_count.has_value()) return std::nullopt;
          advance_count = advance->characters;
        } else {
          return std::nullopt;
        }
      }

      if (!consumed.empty()) {
        auto count = static_cast<std::uint32_t>(consumed.size());
        if (!peek_count.has_value() || *peek_count != count || !advance_count.has_value() ||
            *advance_count != count || (reads_character && count != 1U)) {
          return std::nullopt;
        }
        for (auto codepoint : consumed) {
          if (codepoint > 0x7f) return std::nullopt;
          literal.push_back(static_cast<char>(codepoint));
        }
      } else if (!block.instructions.empty() &&
                 !std::holds_alternative<test_assertion>(block.instructions.front())) {
        return std::nullopt;
      }
      current = block.successors.front().target;
    }
    return std::nullopt;
  }

  void emit_string_operation_execute(string_operation const& operation)
  {
    emit_ascii_literal_at(operation.literal);
    auto execute   = options_.execute_function;
    auto literal   = name("ascii_literal_at");
    auto size      = operation.literal.size();
    auto template_ = std::string_view{};
    switch (operation.kind) {
      case string_operation_kind::BEGINS_WITH:
        template_ = "string_begins_with";
        output_.emit("{}",
                     regex_ir::nvvm_template(
                       regex_ir_nvvm_templates::regex_operations, template_, execute, literal));
        break;
      case string_operation_kind::ENDS_WITH:
        output_.emit(
          "{}",
          regex_ir::nvvm_template(
            regex_ir_nvvm_templates::regex_operations, "string_ends_with", execute, size, literal));
        break;
      case string_operation_kind::ENDS_LINE:
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                             "string_ends_line",
                                             execute,
                                             size,
                                             literal,
                                             name("load_byte")));
        break;
      case string_operation_kind::EQUALS:
        output_.emit(
          "{}",
          regex_ir::nvvm_template(
            regex_ir_nvvm_templates::regex_operations, "string_equals", execute, size, literal));
        break;
      case string_operation_kind::EQUALS_LINE:
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                             "string_equals_line",
                                             execute,
                                             size,
                                             size + 1U,
                                             name("load_byte"),
                                             literal));
        break;
    }
    output_.blank();
  }

  /**
   * @brief recognizes a case-sensitive ASCII literal followed by `.*$`
   *
   * The direct executor is deliberately restricted to the default, non-multiline LF semantics.
   * In that dialect only the final logical line can match, so scanning every candidate through
   * the general Thompson executor is unnecessary. Regex metacharacters and escapes in the
   * literal are rejected until the structural IR analysis can prove the same shape.
   *
   * @return literal prefix when the operation can use the final-line tail executor
   */
  [[nodiscard]] std::optional<std::string> line_tail_literal() const
  {
    auto supported_result =
      ir_.control.result == result_shape::BOOLEAN ||
      ir_.control.result == result_shape::MATCH_COUNT ||
      ir_.control.result == result_shape::MATCH_SPAN ||
      (ir_.control.result == result_shape::CAPTURES && ir_.capture_count == 0U);
    if (!supported_result || !ir_.control.scan_input || ir_.options.case_insensitive ||
        ir_.options.multiline || ir_.options.dot_all || ir_.options.extended_newline) {
      return std::nullopt;
    }

    constexpr auto suffix = std::string_view{R"(.*$)"};
    auto pattern          = std::string_view{ir_.pattern};
    if (!pattern.ends_with(suffix) || pattern.size() == suffix.size()) return std::nullopt;
    auto literal                  = pattern.substr(0, pattern.size() - suffix.size());
    constexpr auto metacharacters = std::string_view{R"(\.^$*+?()[]{}|)"};
    for (auto character : literal) {
      auto byte = static_cast<std::uint8_t>(character);
      if (byte > 0x7fU || metacharacters.find(character) != std::string_view::npos) {
        return std::nullopt;
      }
    }
    return std::string{literal};
  }

  /**
   * @brief emits a direct finder for `literal.*$` with default LF semantics
   *
   * The matcher first locates the start of the final logical line, then searches only that range
   * for the literal. A successful match ends at the input end or immediately before its final LF.
   *
   * @param literal non-empty case-sensitive ASCII prefix
   */
  void emit_line_tail_execute(std::string_view literal)
  {
    emit_ascii_literal_at(literal);
    output_.emit("{}",
                 regex_ir::nvvm_template(
                   regex_ir_nvvm_templates::regex_literals,
                   "line_tail_execute",
                   name("find_from"),
                   name("load_byte"),
                   static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
                   literal.size(),
                   name("ascii_literal_at")));
    output_.blank();

    if (ir_.control.result == result_shape::BOOLEAN) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "boolean_find_adapter",
                                           options_.execute_function,
                                           name("find_from")));
      output_.blank();
    } else if (ir_.control.result == result_shape::MATCH_COUNT) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "single_match_count_adapter",
                                           options_.execute_function,
                                           name("find_from")));
      output_.blank();
    } else if (ir_.control.result == result_shape::CAPTURES) {
      emit_capture_execute();
    } else if (ir_.selected_operation.kind == operation_kind::FIND_ALL) {
      emit_find_all_execute();
    } else {
      emit_find_execute();
    }
  }

  [[nodiscard]] std::optional<std::uint32_t> word_run_minimum() const
  {
    if (ir_.options.characters == character_mode::BYTES && !ir_.options.ascii_classes) {
      return std::nullopt;
    }
    if (ir_.control.result != result_shape::BOOLEAN &&
        ir_.control.result != result_shape::MATCH_COUNT &&
        ir_.control.result != result_shape::MATCH_SPAN) {
      return std::nullopt;
    }
    constexpr auto prefix = std::string_view{R"(\b\w{)"};
    constexpr auto suffix = std::string_view{R"(,}\b)"};
    auto pattern          = std::string_view{ir_.pattern};
    if (!pattern.starts_with(prefix) || !pattern.ends_with(suffix) ||
        pattern.size() <= prefix.size() + suffix.size()) {
      return std::nullopt;
    }
    auto digits = pattern.substr(prefix.size(), pattern.size() - prefix.size() - suffix.size());
    auto value  = std::uint64_t{0};
    for (auto character : digits) {
      if (character < '0' || character > '9') return std::nullopt;
      value = value * 10U + static_cast<std::uint64_t>(character - '0');
      if (value > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
    }
    return value == 0 ? std::nullopt
                      : std::optional<std::uint32_t>{static_cast<std::uint32_t>(value)};
  }

  void emit_unicode_word_run_matcher(std::uint32_t minimum)
  {
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_functions,
                                         "unicode_word_run",
                                         name("next_word_run"),
                                         name("decode_codepoint"),
                                         name("decode_width"),
                                         name("is_word")));
  }

  void emit_word_run_execute(std::uint32_t minimum)
  {
    if (!ir_.options.ascii_classes) {
      emit_unicode_word_run_matcher(minimum);
    } else {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_functions,
                                           "ascii_word_run",
                                           name("next_word_run"),
                                           name("load_byte")));
    }
    output_.blank();

    switch (ir_.control.result) {
      case result_shape::BOOLEAN:
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                             "word_run_boolean",
                                             options_.execute_function,
                                             name("next_word_run"),
                                             minimum));
        break;
      case result_shape::MATCH_SPAN:
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                             "word_run_span",
                                             options_.execute_function,
                                             name("next_word_run"),
                                             minimum));
        break;
      case result_shape::MATCH_COUNT:
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                             "word_run_count",
                                             options_.execute_function,
                                             name("next_word_run"),
                                             minimum));
        break;
      default: throw std::invalid_argument("unsupported word-run result shape");
    }
    output_.blank();
  }

  struct utf8_literal {
    std::u32string codepoints;
    std::size_t byte_count;
  };

  [[nodiscard]] static std::string encode_utf8_literal(std::u32string_view codepoints)
  {
    auto bytes = std::string{};
    for (auto codepoint : codepoints) {
      if (codepoint <= 0x7fU) {
        bytes.push_back(static_cast<char>(codepoint));
      } else if (codepoint <= 0x7ffU) {
        bytes.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        bytes.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
      } else if (codepoint <= 0xffffU) {
        bytes.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        bytes.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        bytes.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
      } else {
        bytes.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
        bytes.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
        bytes.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        bytes.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
      }
    }
    return bytes;
  }

  [[nodiscard]] static std::optional<std::size_t> utf8_literal_pivot(utf8_literal const& literal)
  {
    auto bytes = encode_utf8_literal(literal.codepoints);
    if (bytes.size() < sizeof(std::uint64_t)) return std::nullopt;
    return literal_anchor(bytes);
  }

  [[nodiscard]] static std::size_t ascii_literal_pivot(std::string_view literal)
  {
    return literal_anchor(literal);
  }

  [[nodiscard]] std::optional<utf8_literal> exact_literal() const
  {
    if (ir_.entry >= ir_.blocks.size() || ir_.accept >= ir_.blocks.size()) { return std::nullopt; }

    std::vector<bool> visited(ir_.blocks.size(), false);
    std::u32string literal;
    auto byte_count = std::size_t{0};
    auto current    = ir_.entry;
    while (current < ir_.blocks.size() && !visited[current]) {
      visited[current] = true;
      auto& block      = ir_.blocks[current];
      if (current == ir_.accept) {
        auto accepting = block.instructions.size() == 1U &&
                         std::holds_alternative<emit_accept>(block.instructions.front()) &&
                         block.successors.empty();
        if (!accepting || literal.empty()) { return std::nullopt; }
        return utf8_literal{std::move(literal), byte_count};
      }
      if (block.successors.size() != 1U) return std::nullopt;

      std::u32string consumed;
      std::optional<std::uint32_t> peek_count;
      std::optional<std::uint32_t> advance_count;
      auto reads_character = false;
      for (auto& item : block.instructions) {
        if (auto* peek = std::get_if<can_peek>(&item)) {
          if (peek_count.has_value()) return std::nullopt;
          peek_count = peek->characters;
        } else if (std::holds_alternative<read_character>(item)) {
          if (reads_character) return std::nullopt;
          reads_character = true;
        } else if (auto* character_match = std::get_if<match_character>(&item)) {
          if (!consumed.empty() || !character_match->predicate.is_singleton()) {
            return std::nullopt;
          }
          consumed.push_back(character_match->predicate.singleton());
        } else if (auto* literal_match = std::get_if<match_literal>(&item)) {
          if (!consumed.empty()) return std::nullopt;
          consumed = literal_match->value;
        } else if (auto* advance = std::get_if<advance_cursor>(&item)) {
          if (advance_count.has_value()) return std::nullopt;
          advance_count = advance->characters;
        } else {
          return std::nullopt;
        }
      }

      if (!block.instructions.empty()) {
        auto count = static_cast<std::uint32_t>(consumed.size());
        if (count == 0U || !peek_count.has_value() || *peek_count != count ||
            !advance_count.has_value() || *advance_count != count ||
            (reads_character && count != 1U)) {
          return std::nullopt;
        }
        for (auto codepoint : consumed) {
          if (codepoint > 0x10ffffU || (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
            return std::nullopt;
          }
          byte_count += codepoint <= 0x7fU     ? 1U
                        : codepoint <= 0x7ffU  ? 2U
                        : codepoint <= 0xffffU ? 3U
                                               : 4U;
          literal.push_back(codepoint);
        }
      }
      current = block.successors.front().target;
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string> exact_ascii_literal() const
  {
    auto literal = exact_literal();
    if (!literal.has_value() || std::any_of(literal->codepoints.begin(),
                                            literal->codepoints.end(),
                                            [](auto codepoint) { return codepoint > 0x7fU; })) {
      return std::nullopt;
    }
    return encode_utf8_literal(literal->codepoints);
  }

  [[nodiscard]] std::optional<utf8_literal> exact_utf8_literal() const
  {
    if (ir_.options.characters == character_mode::BYTES) return std::nullopt;
    auto literal = exact_literal();
    return literal.has_value() && std::any_of(literal->codepoints.begin(),
                                              literal->codepoints.end(),
                                              [](auto value) { return value > 0x7fU; })
             ? std::move(literal)
             : std::nullopt;
  }

  static std::string escaped_comment(std::string_view value)
  {
    std::string result;
    result.reserve(value.size());
    for (auto character : value) {
      auto byte = static_cast<std::uint8_t>(character);
      if (byte < 0x20 || byte == 0x7f) {
        std::format_to(std::back_inserter(result), "\\x{:02X}", byte);
      } else {
        result.push_back(character);
      }
    }
    return result;
  }

  [[nodiscard]] std::optional<std::uint8_t> required_ascii_prefix() const
  {
    if (!ir_.control.scan_input || ir_.entry >= ir_.blocks.size()) { return std::nullopt; }
    for (auto& instruction : ir_.blocks[ir_.entry].instructions) {
      if (auto* literal = std::get_if<match_literal>(&instruction)) {
        if (!literal->value.empty() && literal->value.front() <= 0x7f) {
          return static_cast<std::uint8_t>(literal->value.front());
        }
        return std::nullopt;
      }
      if (auto* character = std::get_if<match_character>(&instruction)) {
        if (character->predicate.is_singleton() && character->predicate.singleton() <= 0x7f) {
          return static_cast<std::uint8_t>(character->predicate.singleton());
        }
        return std::nullopt;
      }
      if (std::holds_alternative<advance_cursor>(instruction) ||
          std::holds_alternative<emit_accept>(instruction)) {
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string> mandatory_ascii_literal() const
  {
    if (ir_.entry >= ir_.blocks.size() || ir_.accept >= ir_.blocks.size()) { return std::nullopt; }

    std::vector<std::pair<std::size_t, std::string>> candidates;
    for (std::size_t block = 0; block < ir_.blocks.size(); ++block) {
      for (auto& instruction : ir_.blocks[block].instructions) {
        auto* matched = std::get_if<match_literal>(&instruction);
        if (matched == nullptr || matched->value.size() < 2U ||
            std::any_of(matched->value.begin(), matched->value.end(), [](auto codepoint) {
              return codepoint > 0x7f;
            })) {
          continue;
        }
        auto literal = std::string{};
        literal.reserve(matched->value.size());
        for (auto codepoint : matched->value) {
          literal.push_back(static_cast<char>(codepoint));
        }
        candidates.emplace_back(block, std::move(literal));
      }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](auto& lhs, auto& rhs) {
      return lhs.second.size() > rhs.second.size();
    });

    for (auto& [candidate, literal] : candidates) {
      std::vector<bool> reachable(ir_.blocks.size(), false);
      std::vector<std::size_t> pending;
      if (ir_.entry != candidate) {
        reachable[ir_.entry] = true;
        pending.push_back(ir_.entry);
      }
      while (!pending.empty()) {
        auto block = pending.back();
        pending.pop_back();
        for (auto& successor : ir_.blocks[block].successors) {
          if (successor.target >= ir_.blocks.size() || successor.target == candidate ||
              reachable[successor.target]) {
            continue;
          }
          reachable[successor.target] = true;
          pending.push_back(successor.target);
        }
      }
      if (!reachable[ir_.accept]) { return literal; }
    }
    return std::nullopt;
  }

  [[nodiscard]] bool begins_at_input_start() const
  {
    if (ir_.entry >= ir_.blocks.size()) return false;
    auto& entry = ir_.blocks[ir_.entry];
    if (entry.instructions.size() != 1U ||
        !std::holds_alternative<test_assertion>(entry.instructions.front())) {
      return false;
    }
    auto assertion = std::get<test_assertion>(entry.instructions.front()).kind;
    return assertion == assertion_kind::BEGIN_INPUT ||
           (assertion == assertion_kind::BEGIN_LINE && !ir_.options.multiline);
  }

  [[nodiscard]] std::vector<std::size_t> live_capture_slots() const
  {
    std::vector<bool> live(static_cast<std::size_t>(ir_.capture_count + 1U) * 2U, false);
    if (ir_.control.result == result_shape::CAPTURES) {
      for (std::uint32_t capture = 1; capture <= ir_.capture_count; ++capture) {
        if (ir_.options.extract_capture_group && capture != *ir_.options.extract_capture_group)
          continue;
        if (std::find(whole_match_captures_.begin(), whole_match_captures_.end(), capture) !=
            whole_match_captures_.end()) {
          continue;
        }
        auto slot       = static_cast<std::size_t>(capture) * 2U;
        live[slot]      = true;
        live[slot + 1U] = true;
      }
    } else if (ir_.control.result == result_shape::REPLACEMENT) {
      for (auto& token : ir_.replacement) {
        if (token.type == replacement_token::kind::CAPTURE && token.capture_index != 0 &&
            !is_whole_match_capture(token.capture_index)) {
          auto slot       = static_cast<std::size_t>(token.capture_index) * 2U;
          live[slot]      = true;
          live[slot + 1U] = true;
        }
      }
    }
    std::vector<std::size_t> result;
    for (std::size_t slot = 0; slot < live.size(); ++slot) {
      if (live[slot]) result.push_back(slot);
    }
    return result;
  }

  /**
   * @brief identifies explicit captures whose span is necessarily the whole match
   *
   * These captures can be copied from the whole-match span in the public capture adapter instead
   * of forcing the matcher to carry capture state through every transition.
   *
   * @return explicit capture indices equivalent to the whole match
   */
  [[nodiscard]] std::vector<std::uint32_t> whole_match_captures() const
  {
    if (ir_.control.result != result_shape::CAPTURES) return {};
    auto result = std::vector<std::uint32_t>{};
    for (std::uint32_t capture = 1; capture <= ir_.capture_count; ++capture) {
      if (is_whole_match_capture(capture)) result.push_back(capture);
    }
    return result;
  }

  [[nodiscard]] bool uses_capture_buffer() const { return !capture_slots_.empty(); }

  [[nodiscard]] bool is_whole_match_capture(std::uint32_t capture_index) const
  {
    auto is_capture = [&](instruction_block const& block, capture_action action) {
      return block.instructions.size() == 1U &&
             std::holds_alternative<write_capture>(block.instructions.front()) &&
             std::get<write_capture>(block.instructions.front()).capture_index == capture_index &&
             std::get<write_capture>(block.instructions.front()).action == action;
    };

    auto begin_writes = std::size_t{0};
    auto end_writes   = std::size_t{0};
    for (auto& candidate : ir_.blocks) {
      for (auto& item : candidate.instructions) {
        auto* capture = std::get_if<write_capture>(&item);
        if (capture == nullptr || capture->capture_index != capture_index) continue;
        if (capture->action == capture_action::BEGIN) {
          ++begin_writes;
        } else {
          ++end_writes;
        }
      }
    }
    if (begin_writes != 1U || end_writes != 1U) return false;

    std::vector<bool> visited(ir_.blocks.size(), false);
    auto block           = ir_.entry;
    auto begins_at_start = false;
    while (block < ir_.blocks.size() && !visited[block]) {
      visited[block]  = true;
      auto& candidate = ir_.blocks[block];
      if (is_capture(candidate, capture_action::BEGIN)) begins_at_start = true;
      if (!candidate.instructions.empty() && !is_capture(candidate, capture_action::BEGIN)) {
        break;
      }
      if (candidate.successors.size() != 1U) break;
      block = candidate.successors.front().target;
    }
    if (!begins_at_start) return false;

    std::vector<std::vector<block_id>> predecessors(ir_.blocks.size());
    for (auto& candidate : ir_.blocks) {
      for (auto edge : candidate.successors)
        predecessors[edge.target].push_back(candidate.id);
    }
    std::fill(visited.begin(), visited.end(), false);
    block               = ir_.accept;
    auto ends_at_accept = false;
    while (block < ir_.blocks.size() && !visited[block]) {
      visited[block]  = true;
      auto& candidate = ir_.blocks[block];
      if (is_capture(candidate, capture_action::END)) ends_at_accept = true;
      auto is_accept = candidate.instructions.size() == 1U &&
                       std::holds_alternative<emit_accept>(candidate.instructions.front());
      if (!candidate.instructions.empty() && !is_capture(candidate, capture_action::END) &&
          !is_accept) {
        break;
      }
      if (predecessors[block].size() != 1U) break;
      block = predecessors[block].front();
    }
    return ends_at_accept;
  }

  static std::int32_t llvm_i16(std::uint16_t value) { return static_cast<std::int16_t>(value); }

  static std::int64_t llvm_i64(std::uint64_t value) { return static_cast<std::int64_t>(value); }

  static std::string format_i16_array(std::vector<std::uint16_t> const& values)
  {
    std::string result;
    result.reserve(values.size() * 8);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i16 {}", llvm_i16(values[index]));
    }
    return result;
  }

  static std::string format_i32_array(std::vector<std::uint32_t> const& values)
  {
    std::string result;
    result.reserve(values.size() * 14U);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i32 {}", values[index]);
    }
    return result;
  }

  static std::string format_i64_array(std::vector<std::uint64_t> const& values)
  {
    std::string result;
    result.reserve(values.size() * 24U);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i64 {}", llvm_i64(values[index]));
    }
    return result;
  }

  template <typename Range>
  static std::string format_i8_array(Range const& values)
  {
    std::string result;
    result.reserve(values.size() * 7U);
    std::size_t index = 0;
    for (auto value : values) {
      if (index++ != 0) result += ", ";
      std::format_to(std::back_inserter(result), "i8 {}", value);
    }
    return result;
  }

  static std::string format_byte_array(std::string_view value)
  {
    std::string result;
    result.reserve(value.size() * 3U);
    for (auto byte : value) {
      std::format_to(std::back_inserter(result), R"(\{:02X})", static_cast<unsigned char>(byte));
    }
    return result;
  }

  /**
   * @brief emits constant alphabet and reach-mask tables for a Glushkov executor
   *
   * @param machine bit-parallel machine to render
   */
  void emit_glushkov_globals(glushkov_machine const& machine)
  {
    std::vector<std::uint16_t> byte_classes(machine.alphabet.byte_classes.begin(),
                                            machine.alphabet.byte_classes.end());
    auto globals = std::format("@{} = internal addrspace(4) constant [256 x i16] [{}], align 2",
                               name("dfa_byte_classes"),
                               format_i16_array(byte_classes));
    if (machine.reach_masks.size() > 8U) {
      std::format_to(std::back_inserter(globals),
                     "\n@{} = internal addrspace(4) constant [{} x i64] [{}], align 8",
                     name("glushkov_reach_masks"),
                     machine.reach_masks.size(),
                     format_i64_array(machine.reach_masks));
    }
    if (machine.alphabet.unicode_intervals.size() > 8U) {
      std::vector<std::uint32_t> unicode_ends;
      std::vector<std::uint16_t> unicode_classes;
      unicode_ends.reserve(machine.alphabet.unicode_intervals.size());
      unicode_classes.reserve(machine.alphabet.unicode_intervals.size());
      for (auto interval : machine.alphabet.unicode_intervals) {
        unicode_ends.push_back(interval.last);
        unicode_classes.push_back(interval.class_id);
      }
      std::format_to(std::back_inserter(globals),
                     "\n@{} = internal addrspace(1) constant [{} x i32] [{}], align 4",
                     name("dfa_unicode_ends"),
                     unicode_ends.size(),
                     format_i32_array(unicode_ends));
      std::format_to(std::back_inserter(globals),
                     "\n@{} = internal addrspace(1) constant [{} x i16] [{}], align 2",
                     name("dfa_unicode_classes"),
                     unicode_classes.size(),
                     format_i16_array(unicode_classes));
    }
    output_.emit("{}", globals);
    output_.blank();
  }

  /**
   * @brief emits the alphabet-class to active-position reach-mask mapper
   *
   * @param machine bit-parallel machine whose reach masks are rendered
   */
  void emit_glushkov_reach(glushkov_machine const& machine)
  {
    auto function = name("glushkov_reach");
    if (machine.reach_masks.size() == 1U) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                           "glushkov_reach_constant",
                                           function,
                                           llvm_i64(machine.reach_masks.front())));
      output_.blank();
      return;
    }

    if (machine.reach_masks.size() <= 8U) {
      auto result = std::format("{}", llvm_i64(machine.reach_masks.front()));
      std::string body;
      for (std::size_t index = 1; index < machine.reach_masks.size(); ++index) {
        std::format_to(std::back_inserter(body),
                       "{}",
                       regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                               "reach_is_n",
                                               index,
                                               llvm_i64(machine.reach_masks[index]),
                                               result));
        result = std::format("%reach_{}", index);
      }
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                           "glushkov_reach_select",
                                           function,
                                           body,
                                           result));
      output_.blank();
      return;
    }

    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                         "glushkov_reach_table",
                                         function,
                                         machine.reach_masks.size(),
                                         name("glushkov_reach_masks")));
    output_.blank();
  }

  /**
   * @brief emits the bit-parallel Glushkov successor computation
   *
   * @param machine bit-parallel machine whose shift and exception transitions are rendered
   */
  void emit_glushkov_follow(glushkov_machine const& machine)
  {
    std::string body;
    auto result                = std::string{"0"};
    std::size_t combined_index = 0;
    auto append                = [&](std::string_view value) {
      if (result == "0") {
        result = value;
        return;
      }
      std::format_to(
        std::back_inserter(body), "  %follow_{} = or i64 {}, {}\n", combined_index, result, value);
      result = std::format("%follow_{}", combined_index++);
    };

    for (std::size_t index = 0; index < machine.shifts.size(); ++index) {
      auto shift = machine.shifts[index];
      std::format_to(std::back_inserter(body),
                     "{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                             "shift_source_n",
                                             index,
                                             llvm_i64(shift.sources),
                                             shift.amount));
      append(std::format("%shift_value_{}", index));
    }

    for (std::size_t position = 0; position < machine.exception_successors.size(); ++position) {
      auto successors = machine.exception_successors[position];
      if (successors == 0U) continue;
      auto bit = std::uint64_t{1} << position;
      std::format_to(std::back_inserter(body),
                     "{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                             "exception_bit_n",
                                             position,
                                             llvm_i64(bit),
                                             llvm_i64(successors)));
      append(std::format("%exception_value_{}", position));
    }

    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                         "glushkov_follow",
                                         name("glushkov_follow"),
                                         body,
                                         result));
    output_.blank();
  }

  /**
   * @brief emits constant character-class and DFA-transition tables
   *
   * @param machine deterministic machine to render
   */
  void emit_deterministic_globals(deterministic_machine const& machine)
  {
    std::vector<std::uint16_t> byte_classes(machine.byte_classes.begin(),
                                            machine.byte_classes.end());
    auto common = regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                          "dfa_globals",
                                          name("dfa_byte_classes"),
                                          format_i16_array(byte_classes),
                                          name("dfa_transitions"),
                                          machine.transition_address_space,
                                          machine.transitions.size(),
                                          format_i16_array(machine.transitions));
    if (machine.assertion_aware) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                           "dfa_boundary_globals",
                                           common,
                                           name("dfa_boundary_accepts"),
                                           machine.boundary_accepts.size(),
                                           format_i8_array(machine.boundary_accepts)));
    } else {
      output_.emit("{}", common);
    }
    if (machine.unicode_intervals.size() > 8U) {
      std::vector<std::uint32_t> unicode_ends;
      std::vector<std::uint16_t> unicode_classes;
      unicode_ends.reserve(machine.unicode_intervals.size());
      unicode_classes.reserve(machine.unicode_intervals.size());
      for (auto interval : machine.unicode_intervals) {
        unicode_ends.push_back(interval.last);
        unicode_classes.push_back(interval.class_id);
      }
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                           "dfa_unicode_globals",
                                           name("dfa_unicode_ends"),
                                           unicode_ends.size(),
                                           format_i32_array(unicode_ends),
                                           name("dfa_unicode_classes"),
                                           format_i16_array(unicode_classes)));
    }
    output_.blank();
  }

  /**
   * @brief emits the Unicode code-point to deterministic alphabet-class mapper
   *
   * @param machine deterministic machine whose classes are rendered
   */
  void emit_deterministic_classifier(deterministic_machine const& machine)
  {
    auto function = name("dfa_classify");
    auto table    = name("dfa_byte_classes");
    std::vector<std::size_t> frequencies(machine.class_count);
    for (auto character_class : machine.byte_classes)
      ++frequencies[character_class];
    auto default_class = static_cast<std::uint16_t>(
      std::max_element(frequencies.begin(), frequencies.end()) - frequencies.begin());
    std::vector<deterministic_interval> byte_intervals;
    for (std::uint32_t byte = 0; byte < machine.byte_classes.size();) {
      auto character_class = machine.byte_classes[byte];
      if (character_class == default_class) {
        ++byte;
        continue;
      }
      auto first = byte;
      while (byte + 1U < machine.byte_classes.size() &&
             machine.byte_classes[byte + 1U] == character_class) {
        ++byte;
      }
      byte_intervals.push_back({first, byte, character_class});
      ++byte;
    }
    // more fragmented mappings favor one cached table load over comparison chains.
    auto inline_byte_classes = !byte_intervals.empty() && byte_intervals.size() <= 2U;
    std::string byte_classifier;
    if (inline_byte_classes) {
      auto result = std::format("{}", default_class);
      for (std::size_t index = 0; index < byte_intervals.size(); ++index) {
        auto interval = byte_intervals[index];
        if (interval.first == interval.last) {
          std::format_to(std::back_inserter(byte_classifier),
                         "  %byte_in_{} = icmp eq i32 %cp, {}\n",
                         index,
                         interval.first);
        } else {
          std::format_to(std::back_inserter(byte_classifier),
                         "{}",
                         regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                                 "byte_offset_n",
                                                 index,
                                                 interval.first,
                                                 index,
                                                 index,
                                                 interval.last - interval.first));
        }
        std::format_to(std::back_inserter(byte_classifier),
                       "  %byte_class_{} = select i1 %byte_in_{}, i32 {}, i32 {}\n",
                       index,
                       index,
                       interval.class_id,
                       result);
        result = std::format("%byte_class_{}", index);
      }
      std::format_to(std::back_inserter(byte_classifier), "  ret i32 {}", result);
    } else {
      byte_classifier =
        regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa, "byte_index", table);
    }
    output_.emit(
      "{}",
      regex_ir::nvvm_template(
        regex_ir_nvvm_templates::regex_dfa, "dfa_classifier_prefix", function, byte_classifier));

    if (machine.unicode_intervals.empty()) {
      output_.emit("  ret i32 0");
    } else if (machine.unicode_intervals.size() == 1) {
      output_.emit("  ret i32 {}", machine.unicode_intervals.front().class_id);
    } else if (machine.unicode_intervals.size() <= 8U) {
      auto result = std::format("{}", machine.unicode_intervals.back().class_id);
      for (std::size_t reverse = machine.unicode_intervals.size() - 1; reverse > 0; --reverse) {
        auto index     = reverse - 1;
        auto& interval = machine.unicode_intervals[index];
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                             "unicode_le_n",
                                             index,
                                             interval.last,
                                             interval.class_id,
                                             result));
        result = std::format("%unicode_class_{}", index);
      }
      output_.emit("  ret i32 {}", result);
    } else {
      output_.emit(
        "  %unicode_ends = getelementptr [{} x i32], [{} x i32] addrspace(1)* @{}, i64 0, i64 0",
        machine.unicode_intervals.size(),
        machine.unicode_intervals.size(),
        name("dfa_unicode_ends"));
      output_.emit(
        "  %unicode_classes = getelementptr [{} x i16], [{} x i16] addrspace(1)* @{}, i64 0, i64 0",
        machine.unicode_intervals.size(),
        machine.unicode_intervals.size(),
        name("dfa_unicode_classes"));
      output_.emit(
        R"NVVM(  %unicode_class = call i32 @regex_ir_unicode_class(i32 %cp, i32 addrspace(1)* %unicode_ends, i16 addrspace(1)* %unicode_classes, i32 {}))NVVM",
        machine.unicode_intervals.size());
      output_.emit("  ret i32 %unicode_class");
    }
    output_.emit(
      "{}",
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_dfa, "dfa_classifier_suffix"));
    output_.blank();
  }

  /**
   * @brief emits the position-context classifier used by assertion-aware DFA transitions
   *
   * @param machine deterministic machine whose assertion truth values are classified
   */
  void emit_deterministic_boundary_classifier(deterministic_machine const& machine)
  {
    auto function  = name("dfa_boundary_classify");
    auto is_word   = name("is_word");
    auto decode    = name("decode_codepoint");
    auto line_bits = static_cast<std::uint8_t>(assertion_bit(assertion_kind::BEGIN_LINE) |
                                               assertion_bit(assertion_kind::END_LINE));
    output_.emit(
      "{}",
      regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa, "dfa_boundary_prefix", function));

    auto mask_value        = std::string{"0"};
    std::size_t mask_index = 0;
    auto append_bit        = [&](std::string_view condition) {
      auto bit = std::uint32_t{1} << mask_index;
      output_.emit("  %boundary_bit_{} = select i1 {}, i32 {}, i32 0", mask_index, condition, bit);
      if (mask_value == "0") {
        mask_value = std::format("%boundary_bit_{}", mask_index);
      } else {
        output_.emit(
          "  %boundary_mask_{} = or i32 {}, %boundary_bit_{}", mask_index, mask_value, mask_index);
        mask_value = std::format("%boundary_mask_{}", mask_index);
      }
      ++mask_index;
    };

    auto begin_input_bit = assertion_bit(assertion_kind::BEGIN_INPUT);
    auto end_input_bit   = assertion_bit(assertion_kind::END_INPUT);
    auto word_bits       = static_cast<std::uint8_t>(assertion_bit(assertion_kind::WORD_BOUNDARY) |
                                               assertion_bit(assertion_kind::NOT_WORD_BOUNDARY));
    auto needs_begin =
      (machine.assertion_mask &
       static_cast<std::uint8_t>(begin_input_bit | assertion_bit(assertion_kind::BEGIN_LINE))) != 0;
    auto needs_end =
      (machine.assertion_mask &
       static_cast<std::uint8_t>(end_input_bit | assertion_bit(assertion_kind::END_LINE))) != 0;
    if (needs_begin) output_.emit("  %at_begin = icmp eq i64 %position, 0");
    if (needs_end) output_.emit("  %at_end = icmp eq i64 %position, %size");
    if ((machine.assertion_mask & begin_input_bit) != 0) { append_bit("%at_begin"); }
    if ((machine.assertion_mask & end_input_bit) != 0) { append_bit("%at_end"); }

    if ((machine.assertion_mask & word_bits) != 0) {
      output_.emit(
        "{}", regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa, "has_previous", is_word));
      if ((machine.assertion_mask & assertion_bit(assertion_kind::WORD_BOUNDARY)) != 0) {
        append_bit("%word_boundary");
      }
      if ((machine.assertion_mask & assertion_bit(assertion_kind::NOT_WORD_BOUNDARY)) != 0) {
        append_bit("%not_word_boundary");
      }
    }

    if ((machine.assertion_mask & line_bits) != 0) {
      if (ir_.options.extended_newline) {
        output_.emit("  %newline_flags = call i8 @{}(i32 %previous_cp, i32 %current_cp)",
                     name("extended_newline_flags"));
        output_.emit("  %current_newline_bit = and i8 %newline_flags, 1");
        output_.emit("  %current_newline = icmp ne i8 %current_newline_bit, 0");
        output_.emit("  %previous_newline_bit = and i8 %newline_flags, 2");
        output_.emit("  %previous_newline = icmp ne i8 %previous_newline_bit, 0");
        output_.emit("  %not_mid_crlf_bit = and i8 %newline_flags, 4");
        output_.emit("  %not_mid_crlf = icmp ne i8 %not_mid_crlf_bit, 0");
      } else {
        output_.emit(
          "{}", regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_dfa, "current_lf"));
      }

      if ((machine.assertion_mask & assertion_bit(assertion_kind::BEGIN_LINE)) != 0) {
        if (ir_.options.multiline) {
          output_.emit("{}",
                       regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_dfa,
                                                       "begin_after_newline"));
        } else {
          output_.emit("  %begin_line = icmp eq i64 %position, 0");
        }
        append_bit("%begin_line");
      }

      if ((machine.assertion_mask & assertion_bit(assertion_kind::END_LINE)) != 0) {
        if (ir_.options.multiline) {
          output_.emit(
            "{}",
            regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_dfa, "end_at_newline"));
        } else {
          output_.emit(
            "{}",
            regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_dfa, "next_position"));
          auto final_sequence = std::string{"%current_is_final"};
          if (ir_.options.extended_newline) {
            output_.emit(
              "{}",
              regex_ir::nvvm_template(
                regex_ir_nvvm_templates::regex_dfa, "current_cr", decode, name("decode_width")));
            final_sequence = "%final_sequence";
          }
          output_.emit(
            "{}",
            regex_ir::nvvm_template(
              regex_ir_nvvm_templates::regex_dfa, "end_final_newline_0", final_sequence));
        }
        append_bit("%end_line");
      }
    }

    output_.emit("{}",
                 regex_ir::nvvm_template(
                   regex_ir_nvvm_templates::regex_dfa, "dfa_boundary_suffix", mask_value));
    output_.blank();
  }

  /**
   * @brief emits the bit-parallel Glushkov contains or matches executor
   *
   * @param machine bit-parallel machine to execute
   */
  void emit_glushkov_execute(glushkov_machine const& machine)
  {
    auto classify  = name("dfa_classify");
    auto decode    = name("decode_codepoint");
    auto follow    = name("glushkov_follow");
    auto load_byte = name("load_byte");
    auto reach     = name("glushkov_reach");
    auto width     = name("decode_width");
    auto ascii_block =
      regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov, "ascii", classify);
    auto ascii_predecessor    = std::string{"%ascii"};
    auto prefix_loop_position = std::string{};
    auto prefix_loop_state    = std::string{};
    if (machine.scan_input && machine.start_byte.has_value()) {
      ascii_predecessor    = "%ascii_classify";
      prefix_loop_position = ", [ %prefix_next_position, %prefix_skip ]";
      prefix_loop_state    = ", [ 0, %prefix_skip ]";
      ascii_block          = regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                            "glushkov_ascii_prefix",
                                            *machine.start_byte,
                                            classify);
    }

    auto inject =
      machine.scan_input
        ? std::format("  %candidates = or i64 %follow, {}\n", llvm_i64(machine.first_set))
        : regex_ir::nvvm_template(
            regex_ir_nvvm_templates::regex_glushkov, "at_start", llvm_i64(machine.first_set));

    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                         "glushkov_execute",
                                         options_.execute_function,
                                         prefix_loop_position,
                                         prefix_loop_state,
                                         load_byte,
                                         ascii_block,
                                         decode,
                                         width,
                                         classify,
                                         ascii_predecessor,
                                         reach,
                                         follow,
                                         inject));

    if (machine.accept_at_end) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                           "glushkov_accept_end",
                                           llvm_i64(machine.accept_mask)));
    } else {
      output_.emit(
        "{}",
        regex_ir::nvvm_template(
          regex_ir_nvvm_templates::regex_glushkov, "accept_bits", llvm_i64(machine.accept_mask)));
    }
    output_.blank();
  }

  /**
   * @brief emits a streaming leftmost-prioritized Glushkov span finder
   *
   * The first pass advances all candidate starts together until it finds the greedy match end.
   * The second pass recovers the earliest start that reaches an accepting position. This avoids
   * restarting the complete automaton at every input position while preserving the span ABI used
   * by count, replacement, and split.
   *
   * @param machine bit-parallel machine to execute
   */
  void emit_glushkov_find_from(glushkov_machine const& machine)
  {
    auto accept_at_end        = machine.accept_at_end ? "%phase1_at_input_end" : "true";
    auto rescan_accept_at_end = machine.accept_at_end ? "%rescan_at_input_end" : "true";
    if (machine.fixed_match_bytes.has_value()) {
      auto start_seek = machine.start_byte.has_value()
                          ? regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                                    "state_empty",
                                                    static_cast<std::uint32_t>(*machine.start_byte))
                          : std::string{R"NVVM(  %current_position = add i64 %position, 0
)NVVM"};
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                           "glushkov_find_fixed",
                                           name("glushkov_priority_kill"),
                                           name("find_from"),
                                           name("load_byte"),
                                           name("decode_codepoint"),
                                           name("decode_width"),
                                           name("dfa_classify"),
                                           name("glushkov_reach"),
                                           name("glushkov_follow"),
                                           llvm_i64(machine.first_set),
                                           llvm_i64(machine.accept_mask),
                                           accept_at_end,
                                           *machine.fixed_match_bytes,
                                           start_seek));
      output_.blank();
      return;
    }
    output_.emit(
      "{}",
      regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                              "glushkov_find",
                              name("glushkov_priority_kill"),
                              name("find_from"),
                              name("load_byte"),
                              name("decode_codepoint"),
                              name("decode_width"),
                              name("dfa_classify"),
                              name("glushkov_reach"),
                              name("glushkov_follow"),
                              llvm_i64(machine.first_set),
                              llvm_i64(machine.accept_mask),
                              accept_at_end,
                              rescan_accept_at_end,
                              machine.accept_at_end ? "%rescan_loop_at_input_end" : "true"));
    output_.blank();
  }

  void emit_repeated_builtin_find(std::uint32_t count)
  {
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_glushkov,
                                         "predicate_repeat_builtin_find",
                                         name("find_from"),
                                         name("load_byte"),
                                         name("decode_packed"),
                                         count));
    output_.blank();
  }

  /**
   * @brief emits the single-pass deterministic contains or matches executor
   *
   * @param machine deterministic machine to execute
   */
  void emit_deterministic_execute(deterministic_machine const& machine)
  {
    if ((machine.initial_state & 0x8000U) != 0 && !machine.accept_at_end &&
        !machine.accept_assertion.has_value()) {
      output_.emit(
        "{}",
        regex_ir::nvvm_template(
          regex_ir_nvvm_templates::regex_dfa, "always_true_execute", options_.execute_function));
      output_.blank();
      return;
    }

    auto load_byte            = name("load_byte");
    auto decode               = name("decode_codepoint");
    auto width                = name("decode_width");
    auto classify             = name("dfa_classify");
    auto transitions          = name("dfa_transitions");
    auto dead_guard           = std::string{};
    auto reject               = std::string{};
    auto prefix_loop_position = std::string{};
    auto prefix_loop_state    = std::string{};
    auto ascii_block =
      regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa, "ascii", classify);
    auto ascii_predecessor = std::string{"%ascii"};
    auto prefix            = machine.scan_input ? prefix_seek_byte_ : std::nullopt;
    if (prefix.has_value()) {
      prefix_loop_position = ", [ %prefix_next_position, %prefix_skip ]";
      prefix_loop_state    = ", [ " + std::to_string(machine.initial_state) + ", %prefix_skip ]";
      ascii_predecessor    = "%ascii_classify";
      ascii_block          = regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                            "dfa_ascii_prefix",
                                            machine.state_mask,
                                            machine.initial_state & machine.state_mask,
                                            static_cast<std::uint32_t>(*prefix),
                                            classify);
    }
    if (!machine.scan_input && machine.dead_state <= machine.state_mask) {
      dead_guard = regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                           "next_state_index",
                                           machine.state_mask,
                                           machine.dead_state);
      reject     = regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_dfa, "reject");
    }
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "dfa_execute",
                                         options_.execute_function,
                                         machine.initial_state,
                                         load_byte,
                                         classify,
                                         decode,
                                         width,
                                         machine.class_count,
                                         machine.state_mask,
                                         machine.transitions.size(),
                                         machine.transition_address_space,
                                         transitions,
                                         dead_guard,
                                         prefix_loop_position,
                                         prefix_loop_state,
                                         ascii_block,
                                         ascii_predecessor));
    if (!machine.accept_at_end && machine.accept_assertion.has_value()) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                           "dfa_assertion_accept",
                                           name("assertion"),
                                           static_cast<std::uint32_t>(*machine.accept_assertion),
                                           reject,
                                           ir_.options.multiline ? "true" : "false",
                                           ir_.options.extended_newline ? "true" : "false"));
    } else if (!machine.accept_at_end) {
      output_.emit(
        "{}", regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa, "dfa_finish", reject));
    } else {
      auto assertion = std::string{};
      auto result    = std::string{"%accepted"};
      if (machine.accept_assertion.has_value()) {
        assertion = regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                            "done_assertion",
                                            name("assertion"),
                                            static_cast<std::uint32_t>(*machine.accept_assertion),
                                            ir_.options.multiline ? "true" : "false",
                                            ir_.options.extended_newline ? "true" : "false");
        result    = "%done_result";
      }
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                           "branch_br_label_continue",
                                           assertion,
                                           result,
                                           reject));
    }
    output_.blank();
  }

  /**
   * @brief emits a single-pass DFA whose epsilon closure is selected by boundary context
   *
   * @param machine assertion-aware deterministic machine to execute
   */
  void emit_assertion_deterministic_execute(deterministic_machine const& machine)
  {
    auto load_byte        = name("load_byte");
    auto decode           = name("decode_codepoint");
    auto width            = name("decode_width");
    auto classify         = name("dfa_classify");
    auto boundary         = name("dfa_boundary_classify");
    auto transitions      = name("dfa_transitions");
    auto boundary_accepts = name("dfa_boundary_accepts");
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "assertion_dfa_execute",
                                         options_.execute_function,
                                         machine.initial_state,
                                         load_byte,
                                         classify,
                                         decode,
                                         width,
                                         boundary));
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "assertion_dfa_transition",
                                         machine.boundary_class_count,
                                         machine.class_count,
                                         machine.transitions.size(),
                                         machine.transition_address_space,
                                         transitions));
    if (machine.accept_at_end) {
      output_.emit("  br label %continue");
    } else {
      output_.emit("  br i1 %accepted, label %yes, label %continue");
    }
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "assertion_dfa_continue",
                                         machine.boundary_class_count,
                                         boundary,
                                         machine.boundary_accepts.size(),
                                         boundary_accepts));
    output_.blank();
  }

  /**
   * @brief emits NVVM intrinsics used by branch hints
   */
  void emit_optimizer_intrinsics()
  {
    auto expect = required_ascii_prefix().has_value();
    if (expect) {
      output_.emit("{}",
                   regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions,
                                                   "llvm_expect_declaration"));
      output_.blank();
    }
  }

  /**
   * @brief emits the helper that advances a byte cursor by a requested number of characters
   */
  void emit_advance()
  {
    auto function = name("advance");
    auto width    = name("decode_width");
    output_.emit("{}",
                 regex_ir::nvvm_template(
                   regex_ir_nvvm_templates::regex_functions, "advance", function, width));
    output_.blank();
  }

  /**
   * @brief reports whether the program evaluates a Unicode word boundary
   *
   * @return true when a boundary assertion needs cuDF's Unicode word table
   */
  [[nodiscard]] bool uses_unicode_word_boundaries() const
  {
    if (ir_.options.ascii_classes || ir_.options.characters == character_mode::BYTES) return false;
    for (instruction_block const& block : ir_.blocks) {
      for (instruction const& item : block.instructions) {
        auto* assertion = std::get_if<test_assertion>(&item);
        if (assertion != nullptr && (assertion->kind == assertion_kind::WORD_BOUNDARY ||
                                     assertion->kind == assertion_kind::NOT_WORD_BOUNDARY)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * @brief emits the configured word-character classifier used by boundary assertions
   */
  void emit_is_word()
  {
    auto function = name("is_word");
    if (uses_unicode_word_boundaries()) {
      constexpr auto ascii_range_count = std::size_t{3};
      auto unicode_firsts              = std::vector<std::uint32_t>{};
      auto unicode_ends                = std::vector<std::uint32_t>{};
      unicode_firsts.reserve(std::size(unicode_word_ranges) - ascii_range_count);
      unicode_ends.reserve(std::size(unicode_word_ranges) - ascii_range_count);
      for (auto index = ascii_range_count; index < std::size(unicode_word_ranges); ++index) {
        unicode_firsts.push_back(unicode_word_ranges[index].first);
        unicode_ends.push_back(unicode_word_ranges[index].last);
      }
      output_.emit("@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                   name("word_range_firsts"),
                   unicode_firsts.size(),
                   format_i32_array(unicode_firsts));
      output_.emit("@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                   name("word_range_ends"),
                   unicode_ends.size(),
                   format_i32_array(unicode_ends));
      output_.blank();
      output_.emit(
        "{}", regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "is_word"));
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_functions,
                                           "is_word_adapter",
                                           function,
                                           unicode_ends.back(),
                                           unicode_ends.size(),
                                           name("word_range_ends"),
                                           name("word_range_firsts")));
      output_.blank();
      return;
    }
    output_.emit(
      "{}",
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "is_ascii_word"));
    output_.blank();
  }

  /**
   * @brief emits the helper that locates the character immediately before a byte position
   */
  void emit_previous_position()
  {
    auto function = name("previous_position");
    if (ir_.options.characters == character_mode::BYTES) {
      output_.emit(
        "{}",
        regex_ir::nvvm_template(
          regex_ir_nvvm_templates::regex_functions, "previous_position_bytes", function));
      output_.blank();
      return;
    }

    auto load_byte = name("load_byte");
    output_.emit(
      "{}",
      regex_ir::nvvm_template(
        regex_ir_nvvm_templates::regex_functions, "simple_assertion", function, load_byte));
    output_.blank();
  }

  /**
   * @brief emits the dispatcher for begin, end, word-boundary, and non-boundary assertions
   */
  void emit_assertion()
  {
    auto function = name("assertion");
    auto previous = name("previous_position");
    auto decode   = name("decode_codepoint");
    auto is_word  = name("is_word");
    auto advance  = name("advance");
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_functions,
                                         "assertion_execute",
                                         decode,
                                         is_word,
                                         previous,
                                         function,
                                         advance));
    output_.blank();
  }

  /**
   * @brief emits a range-test helper for every character-predicate instruction
   */
  void emit_predicate_helpers()
  {
    for (auto& block : ir_.blocks) {
      for (auto& instruction : block.instructions) {
        auto* match = std::get_if<match_character>(&instruction);
        if (match == nullptr) continue;
        auto function = name(std::format("predicate_{}", block.id));
        auto decode   = name("decode_codepoint");
        output_.emit(
          "{}",
          regex_ir::nvvm_template(
            regex_ir_nvvm_templates::regex_functions, "predicate_prefix", function, decode));
        if (match->predicate.recognized == predicate_class::ANY) {
          if (match->predicate.matches_newline) {
            output_.emit("  ret i1 true");
          } else if (match->predicate.extended_newline) {
            output_.emit(
              "{}",
              regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "not_lf"));
          } else {
            output_.emit("{}",
                         regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions,
                                                         "not_lf_2"));
          }
        } else {
          std::vector<std::string> comparisons;
          for (std::size_t index = 0; index < match->predicate.ranges.size(); ++index) {
            auto range = match->predicate.ranges[index];
            if (range.first == range.last) {
              auto comparison = std::format("equal_{}", index);
              output_.emit(
                "  %{} = icmp eq i32 %cp, {}", comparison, static_cast<std::uint32_t>(range.first));
              comparisons.push_back(comparison);
            } else {
              auto ge     = std::format("ge_{}", index);
              auto le     = std::format("le_{}", index);
              auto inside = std::format("inside_{}", index);
              output_.emit("{}",
                           regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_functions,
                                                   "n",
                                                   ge,
                                                   le,
                                                   inside,
                                                   static_cast<std::uint32_t>(range.first),
                                                   static_cast<std::uint32_t>(range.last)));
              comparisons.push_back(inside);
            }
          }
          if (comparisons.empty()) {
            output_.emit("  ret i1 {}", match->predicate.negated ? "true" : "false");
          } else {
            auto combined = comparisons.front();
            for (std::size_t index = 1; index < comparisons.size(); ++index) {
              auto next = std::format("combined_{}", index);
              output_.emit("  %{} = or i1 %{}, %{}", next, combined, comparisons[index]);
              combined = next;
            }
            if (match->predicate.negated) {
              output_.emit("{}",
                           regex_ir::nvvm_template(
                             regex_ir_nvvm_templates::regex_functions, "negated", combined));
            } else {
              output_.emit("  ret i1 %{}", combined);
            }
          }
        }
        output_.emit("{}",
                     regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions,
                                                     "function_suffix"));
        output_.blank();
      }
    }
  }

  /**
   * @brief emits constant byte arrays used by a specialized replacement executor
   */
  void emit_replacement_globals()
  {
    if (ir_.control.result != result_shape::REPLACEMENT) return;
    for (std::size_t index = 0; index < ir_.replacement.size(); ++index) {
      auto& token = ir_.replacement[index];
      if (token.type != replacement_token::kind::LITERAL || token.literal.empty()) continue;
      output_.emit(R"NVVM(@{} = internal addrspace(4) constant [{} x i8] c"{}", align 1)NVVM",
                   name(std::format("replacement_{}", index)),
                   token.literal.size(),
                   format_byte_array(token.literal));
    }
    output_.blank();
  }

  /**
   * @brief Emit a conservative first-byte seeker for non-nullable scanning fallbacks.
   *
   * Assertions are ignored when finding the first consuming states, giving a
   * superset of possible starts. Predicate and UTF-8 lead-byte boundaries
   * partition membership into constant intervals. Continuation bytes are never
   * candidates in Unicode mode; full matching still checks every assertion.
   */
  void emit_thompson_start_seeker(deterministic_nfa_graph const& graph)
  {
    if (!ir_.control.scan_input || begins_at_input_start() || prefix_seek_byte_) return;

    std::vector<std::size_t> first, pending{graph.entry};
    std::vector<bool> visited(graph.nodes.size());
    while (!pending.empty()) {
      auto id = pending.back();
      pending.pop_back();
      if (visited[id]) continue;
      visited[id] = true;
      auto& node  = graph.nodes[id];
      if (node.accepts) return;  // A nullable expression may match without a byte.
      if (node.consumes)
        first.push_back(id);
      else
        pending.insert(pending.end(), node.targets.begin(), node.targets.end());
    }
    if (first.empty() || first.size() > 8U) return;

    std::vector<char32_t> points;
    for (char32_t cp = 0; cp < 256; ++cp)
      points.push_back(cp);
    for (char32_t cp = 0x80; cp < 0x800; cp += 64)
      points.push_back(cp);
    for (char32_t cp = 0x800; cp < 0x10000; cp += 4096)
      points.push_back(cp);
    points.push_back(0x10000);
    for (char32_t cp = 0x40000; cp <= 0x100000; cp += 0x40000)
      points.push_back(cp);
    for (auto cp : {8232U, 8233U, 8234U})
      points.push_back(cp);
    for (auto id : first) {
      for (auto range : graph.nodes[id].predicate.ranges) {
        points.push_back(range.first);
        if (range.last < 0x10ffff) points.push_back(range.last + 1);
      }
    }
    std::array<std::uint64_t, 4> bitmap{};
    for (auto cp : points) {
      if (cp > 255 && ir_.options.characters == character_mode::BYTES) continue;
      if (!std::any_of(first.begin(), first.end(), [&](auto id) {
            return graph.nodes[id].predicate.matches(cp);
          }))
        continue;
      auto byte = ir_.options.characters == character_mode::BYTES || cp < 128
                    ? static_cast<std::uint32_t>(cp)
                  : cp < 0x800   ? 0xc0U | (cp >> 6)
                  : cp < 0x10000 ? 0xe0U | (cp >> 12)
                                 : 0xf0U | (cp >> 18);
      bitmap[byte / 64] |= std::uint64_t{1} << (byte % 64);
    }
    auto candidates = std::size_t{0};
    for (auto word : bitmap)
      candidates += std::popcount(word);
    if (candidates == 0 || candidates > 32U) return;
    candidate_seeker_ = true;
    output_.emit(
      "{}",
      nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "llvm_expect_declaration"));
    output_.emit("{}",
                 nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                               "candidate_functions",
                               name("candidate_byte"),
                               bitmap[0],
                               bitmap[1],
                               bitmap[2],
                               bitmap[3],
                               name("seek_candidates")));
  }

  /**
   * @brief Normalize the fallback into single-code-point states with ordered edges.
   */
  void prepare_thompson()
  {
    thompson_ = make_deterministic_graph(ir_);
    if (!thompson_) { throw std::invalid_argument("unsupported Thompson instruction graph"); }
    auto& graph = *thompson_;
    emit_thompson_start_seeker(graph);
    ir_.blocks.clear();
    ir_.entry = static_cast<block_id>(graph.entry);
    for (std::size_t id = 0; id < graph.nodes.size(); ++id) {
      auto& node = graph.nodes[id];
      instruction_block block;
      block.id = static_cast<block_id>(id);
      if (node.capture) block.instructions.emplace_back(*node.capture);
      if (node.assertion) block.instructions.emplace_back(test_assertion{*node.assertion});
      if (node.consumes) {
        block.instructions.emplace_back(can_peek{1});
        block.instructions.emplace_back(read_character{});
        block.instructions.emplace_back(match_character{node.predicate});
        block.instructions.emplace_back(advance_cursor{1});
      }
      if (node.accepts) {
        block.instructions.emplace_back(emit_accept{});
        ir_.accept = block.id;
      }
      for (std::size_t edge = 0; edge < node.targets.size(); ++edge) {
        block.successors.push_back(
          {static_cast<block_id>(node.targets[edge]), static_cast<std::uint32_t>(edge)});
      }
      ir_.blocks.push_back(std::move(block));
    }
  }

  /**
   * @brief Emit ordered Thompson execution with pattern-bounded worklists.
   *
   * Each state is expanded at most once per input position. The first arrival
   * wins, including its capture history. Acceptance removes lower-priority
   * threads but allows higher-priority consuming threads to continue.
   */
  void emit_blocks()
  {
    auto& graph = *thompson_;
    std::string scan_cases, scan_bodies, single_cases, single_bodies;
    std::string low_inputs, high_inputs;
    std::vector<character_predicate const*> eligible_predicates;
    for (std::size_t start = 0; start < graph.nodes.size(); ++start) {
      auto& branch = graph.nodes[start];
      if (branch.consumes || branch.capture || branch.assertion || branch.accepts ||
          branch.targets.size() != 2U)
        continue;
      auto consume = branch.targets[0];
      auto& loop   = graph.nodes[consume];
      if (!loop.consumes || loop.capture || loop.assertion || loop.accepts ||
          loop.targets.size() != 1U || loop.targets[0] != start)
        continue;
      // Every nonempty suffix must start outside the repeated class; nullable
      // suffixes may accept at the final greedy boundary. Assertions and cycles
      // conservatively disable skipping.
      auto disjoint = [&](character_predicate const& other) {
        std::vector<char32_t> points{0, 10, 11, 13, 14, 133, 134, 8232, 8233, 8234};
        for (auto* predicate : std::array<character_predicate const*, 2>{&loop.predicate, &other}) {
          for (auto range : predicate->ranges) {
            points.push_back(range.first);
            if (range.last < 0x10ffff) points.push_back(range.last + 1);
          }
        }
        return std::none_of(points.begin(), points.end(), [&](auto cp) {
          return loop.predicate.matches(cp) && other.matches(cp);
        });
      };
      std::vector<std::uint8_t> visited(graph.nodes.size());
      std::size_t proof_steps = 0;
      auto safe               = [&](auto&& self, std::size_t state) -> bool {
        if (++proof_steps > 256U) return false;
        if (visited[state] == 1) return false;
        if (visited[state] == 2) return true;
        auto& node = graph.nodes[state];
        if (node.assertion) return false;
        if (node.consumes) return disjoint(node.predicate);
        if (node.accepts) return true;
        visited[state] = 1;
        for (auto target : node.targets)
          if (!self(self, target)) return false;
        visited[state] = 2;
        return true;
      };
      if (!safe(safe, branch.targets[1])) continue;
      eligible_predicates.push_back(&loop.predicate);
      scan_cases += std::format("    i64 {}, label %multi_run_{}\n", start, start);
      single_cases += std::format("    i64 {}, label %run_{}\n", start, start);
      single_bodies += nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                     "scan_case",
                                     start,
                                     name(std::format("predicate_{}", consume)));
      std::uint64_t low = 0, high = 0;
      for (char32_t cp = 0; cp < 128; ++cp)
        if (loop.predicate.matches(cp)) {
          if (cp < 64)
            low |= std::uint64_t{1} << cp;
          else
            high |= std::uint64_t{1} << (cp - 64);
        }
      scan_bodies +=
        nvvm_template(regex_ir_nvvm_templates::regex_thompson, "multi_scan_case", start, low, high);
      if (!low_inputs.empty()) {
        low_inputs += ", ";
        high_inputs += ", ";
      }
      low_inputs += std::format("[ %low_{0}, %multi_run_{0} ]", start);
      high_inputs += std::format("[ %high_{0}, %multi_run_{0} ]", start);
    }
    // Multiple loops can skip together only if their ASCII predicates overlap.
    // Otherwise emit the original single-frontier helper and call signature,
    // leaving its generated code unchanged rather than adding a runtime check.
    auto multi_run = false;
    for (std::size_t i = 0; i < eligible_predicates.size(); ++i)
      for (std::size_t j = i + 1; j < eligible_predicates.size(); ++j)
        for (char32_t cp = 0; cp < 128; ++cp)
          multi_run |= eligible_predicates[i]->matches(cp) && eligible_predicates[j]->matches(cp);
    output_.emit("{}",
                 multi_run ? nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                           "run_scanner",
                                           name("scan_run"),
                                           scan_cases,
                                           scan_bodies,
                                           capture_slots_.size() + 1U,
                                           single_cases,
                                           single_bodies,
                                           low_inputs,
                                           high_inputs)
                           : nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                           "single_run_scanner",
                                           name("scan_run"),
                                           single_cases,
                                           single_bodies));
    auto states = graph.nodes.size();
    auto record = capture_slots_.size() + 1U;
    std::vector<bool> frontier_targets(states);
    auto closure_capacity = std::size_t{1};
    for (auto& node : graph.nodes) {
      if (node.consumes) {
        for (auto target : node.targets)
          frontier_targets[target] = true;
      } else if (node.targets.size() > 1U) {
        closure_capacity += node.targets.size() - 1U;
      }
    }
    // Only consuming successors enter a frontier. The closure stack contains
    // lower-priority epsilon branches; the highest-priority edge is expanded
    // directly. Both bounds depend solely on the graph, never input length.
    auto other =
      std::max<std::size_t>(1, std::count(frontier_targets.begin(), frontier_targets.end(), true)) *
      record;
    auto stack        = other * 2U;
    auto active       = stack + closure_capacity * record;
    auto seen         = active + record;
    auto bitset_words = (states + 63U) / 64U;
    auto queued       = seen + bitset_words;
    auto bytes        = (queued + bitset_words) * sizeof(std::int64_t);
    workspace_bytes_  = bytes > 32768U ? bytes : 0U;
    auto copy         = name("copy_thread");
    output_.emit("{}",
                 nvvm_template(regex_ir_nvvm_templates::regex_thompson, "copy", copy, record));
    std::string cases, body, initial, accepted;
    for (std::size_t index = 0; index < capture_slots_.size(); ++index) {
      initial += nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                               "initial_capture",
                               index,
                               capture_slots_[index],
                               index + 1U);
      accepted += nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                "accepted_capture",
                                index,
                                capture_slots_[index],
                                index + 1U);
    }
    for (std::size_t id = 0; id < states; ++id) {
      auto& node = graph.nodes[id];
      std::format_to(std::back_inserter(cases), "    i64 {}, label %state_{}\n", id, id);
      std::format_to(std::back_inserter(body), "state_{}:\n", id);
      if (node.capture) {
        auto slot = node.capture->capture_index * 2U +
                    (node.capture->action == capture_action::END ? 1U : 0U);
        auto write = [&](std::size_t target, std::string_view value) {
          auto found = std::find(capture_slots_.begin(), capture_slots_.end(), target);
          if (found == capture_slots_.end()) return;
          body += nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                "capture",
                                id,
                                target,
                                std::distance(capture_slots_.begin(), found) + 1,
                                value);
        };
        write(slot, "%pos");
        if (node.capture->action == capture_action::BEGIN) write(slot + 1U, "-1");
      }
      if (node.assertion) {
        body += nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                              "assertion",
                              id,
                              name("assertion"),
                              static_cast<std::uint32_t>(*node.assertion),
                              ir_.options.multiline ? "true" : "false",
                              ir_.options.extended_newline ? "true" : "false");
      }
      if (node.accepts) {
        body += ir_.control.require_end
                  ? nvvm_template(regex_ir_nvvm_templates::regex_thompson, "accept_end", id)
                  : std::string{"  br label %accept\n"};
        continue;
      }
      if (node.consumes) {
        body += nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                              "consume",
                              id,
                              name(std::format("predicate_{}", id)));
      }
      for (std::size_t index = 0; index < node.targets.size(); ++index) {
        // Closure uses a LIFO worklist; enqueue consuming successors in forward
        // priority order, epsilon successors in reverse order.
        auto target = node.targets[node.consumes ? index : node.targets.size() - index - 1U];
        if (!node.consumes && index + 1U == node.targets.size()) {
          // Tail-expand the highest-priority epsilon edge without copying its
          // capture record through the closure stack. The visited check still
          // terminates nullable cycles.
          body += nvvm_template(regex_ir_nvvm_templates::regex_thompson, "direct", id, target);
          continue;
        }
        body += nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                              node.consumes ? "enqueue" : "push",
                              std::format("{}_{}", id, index),
                              record,
                              copy,
                              target);
      }
      if (node.consumes || node.targets.empty()) body += "  br label %pop\n";
    }
    auto storage =
      workspace_bytes_ == 0
        ? nvvm_template(regex_ir_nvvm_templates::regex_thompson, "local_storage", bytes)
        : std::string{
            nvvm_template_section(regex_ir_nvvm_templates::regex_thompson, "external_storage")};
    output_.emit("{}",
                 nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                               "run",
                               name("run_block"),
                               states <= 256U && workspace_bytes_ == 0 ? "alwaysinline" : "",
                               storage,
                               other,
                               stack,
                               active,
                               seen,
                               queued,
                               initial,
                               bitset_words,
                               record,
                               copy,
                               cases,
                               body,
                               accepted,
                               name("advance"),
                               name("scan_run"),
                               name("decode_codepoint"),
                               multi_run ? "i64* %current" : "i64 %front_state"));
  }

  /**
   * @brief Render the start filter with its actual predicate, not a dummy literal.
   */
  std::string start_candidate_test(std::string_view byte) const
  {
    return candidate_seeker_
             ? std::format("call i1 @{}(i32 {})", name("candidate_byte"), byte)
             : std::format(
                 "icmp eq i32 {}, {}", byte, static_cast<std::uint32_t>(prefix_seek_byte_.value()));
  }

  /**
   * @brief Name the selected literal or first-byte-bitmap seeking function.
   */
  std::string start_seeker_name() const
  {
    return candidate_seeker_ ? name("seek_candidates") : "regex_ir_seek_byte";
  }

  /**
   * @brief Supply the literal argument only to the literal seeking function.
   */
  std::string start_seeker_arguments() const
  {
    return candidate_seeker_
             ? std::string{}
             : std::format(", i32 {}", static_cast<std::uint32_t>(prefix_seek_byte_.value()));
  }

  /**
   * @brief emits the externally callable anchored or scanning regex execution function
   */
  void emit_execute()
  {
    auto run_block = name("run_block");
    auto advance   = name("advance");
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                         "boolean_execute_prefix",
                                         options_.execute_function));
    auto mandatory_filter =
      mandatory_ascii_literal_.has_value() && ir_.control.scan_input && !begins_at_input_start();
    if (mandatory_filter) {
      output_.emit("  %mandatory_present = call i1 @{}(i8* %data, i64 %size, i64 0)",
                   name("mandatory_literal_present"));
      output_.emit("  br i1 %mandatory_present, label %mandatory_filter_done, label %no");
      output_.emit("mandatory_filter_done:");
    }
    auto initial_predecessor = mandatory_filter ? "%mandatory_filter_done" : "%entry";
    if (!ir_.control.scan_input || begins_at_input_start()) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                           "store_store_i64_0_i64_position_align_8",
                                           run_block,
                                           ir_.entry));
      output_.blank();
      return;
    }

    auto prefix = prefix_seek_byte_;
    if (prefix.has_value() || candidate_seeker_) {
      auto hint =
        std::string{R"NVVM(  %candidate_likely = call i1 @llvm.expect.i1(i1 %candidate, i1 false)
)NVVM"};
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                           "thompson_execute_prefix_search",
                                           name("load_byte"),
                                           start_candidate_test("%start_byte"),
                                           hint,
                                           "%candidate_likely",
                                           run_block,
                                           ir_.entry,
                                           initial_predecessor,
                                           start_seeker_name(),
                                           start_seeker_arguments()));
      output_.blank();
      return;
    }

    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                         "thompson_execute_suffix",
                                         run_block,
                                         ir_.entry,
                                         advance,
                                         initial_predecessor));
    output_.blank();
  }

  /**
   * @brief emits capture stores selected by a deterministic transition index
   *
   * @param prefix unique label and temporary prefix
   * @param programs capture-update program for each deterministic transition
   * @param position SSA value recorded by every capture update
   * @param continuation block reached after applying a program
   */
  void emit_capture_action_dispatch(
    std::string_view prefix,
    std::vector<std::vector<deterministic_capture_action>> const& programs,
    std::string_view position,
    std::string_view continuation)
  {
    std::vector<std::size_t> active;
    for (std::size_t index = 0; index < programs.size(); ++index) {
      if (!programs[index].empty()) active.push_back(index);
    }
    if (active.empty()) {
      output_.emit("  br label %{}", continuation);
      return;
    }

    std::string source;
    std::format_to(
      std::back_inserter(source), "  switch i32 %transition_index, label %{}_done [\n", prefix);
    for (auto index : active) {
      std::format_to(
        std::back_inserter(source), "    i32 {}, label %{}_{}\n", index, prefix, index);
    }
    std::format_to(std::back_inserter(source), "  ]\n");
    for (auto index : active) {
      std::format_to(std::back_inserter(source), "{}_{}:\n", prefix, index);
      for (std::size_t action_index = 0; action_index < programs[index].size(); ++action_index) {
        auto action = programs[index][action_index];
        std::format_to(std::back_inserter(source),
                       "{}",
                       regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                               "n_n_n_ptr",
                                               prefix,
                                               index,
                                               action_index,
                                               action.slot,
                                               position));
        if (action.reset_end) {
          std::format_to(std::back_inserter(source),
                         "{}",
                         regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                                 "n_n_n_end_ptr",
                                                 prefix,
                                                 index,
                                                 action_index,
                                                 action.slot + 1U));
        }
      }
      std::format_to(std::back_inserter(source), "  br label %{}_done\n", prefix);
    }
    std::format_to(
      std::back_inserter(source),
      "{}",
      regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa, "n_done", prefix, continuation));
    output_.emit("{}", source);
  }

  std::string render_start_byte_filter(deterministic_machine const& machine,
                                       std::string_view candidate_target,
                                       bool direct_advance)
  {
    if (!machine.start_byte_filter) return {};

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    for (std::uint32_t byte = 0; byte < 128U;) {
      auto is_candidate =
        (machine.start_byte_bitmap[byte / 64U] & (std::uint64_t{1} << (byte % 64U))) != 0U;
      if (!is_candidate) {
        ++byte;
        continue;
      }
      auto first = byte;
      while (byte + 1U < 128U && (machine.start_byte_bitmap[(byte + 1U) / 64U] &
                                  (std::uint64_t{1} << ((byte + 1U) % 64U))) != 0U) {
        ++byte;
      }
      ranges.emplace_back(first, byte);
      ++byte;
    }

    std::string checks;
    std::string predicate = "false";
    for (std::size_t index = 0; index < ranges.size(); ++index) {
      auto [first, last] = ranges[index];
      auto range_value   = std::format("%start_filter_range_{}", index);
      if (first == last) {
        std::format_to(std::back_inserter(checks),
                       "  {} = icmp eq i32 %start_filter_byte, {}\n",
                       range_value,
                       first);
      } else {
        std::format_to(std::back_inserter(checks),
                       "{}",
                       regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                               "start_filter_offset_n",
                                               index,
                                               first,
                                               range_value,
                                               index,
                                               last - first));
      }
      if (index == 0U) {
        predicate = range_value;
      } else {
        auto any_value = std::format("%start_filter_any_{}", index);
        std::format_to(
          std::back_inserter(checks), "  {} = or i1 {}, {}\n", any_value, predicate, range_value);
        predicate = std::move(any_value);
      }
    }

    auto miss_target  = direct_advance ? "start_filter_advance" : "advance_start";
    auto direct_block = !direct_advance ? std::string{}
                        : prefix_seek_byte_.has_value()
                          ? regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                                    "start_filter_advance",
                                                    static_cast<std::uint32_t>(*prefix_seek_byte_))
                          : std::string{regex_ir::nvvm_template_section(
                              regex_ir_nvvm_templates::regex_dfa, "start_filter_advance_2")};
    return regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                   "start_byte_filter",
                                   candidate_target,
                                   name("load_byte"),
                                   checks,
                                   predicate,
                                   miss_target,
                                   direct_block);
  }

  /**
   * @brief emits a direct tagged-DFA matcher for capture-safe deterministic automata
   *
   * @param machine deterministic machine and transition capture programs to execute
   */
  void emit_tagged_deterministic_find_from(deterministic_machine const& machine)
  {
    auto search_target = machine.start_byte_filter ? "start_filter" : "initialize";
    auto start_filter  = render_start_byte_filter(machine, "initialize", false);
    auto restart =
      machine.restart_state <= machine.state_mask
        ? regex_ir::nvvm_template(
            regex_ir_nvvm_templates::regex_dfa, "restart_state_match", machine.restart_state)
        : std::string{};
    auto advance_phi =
      machine.restart_state <= machine.state_mask && machine.start_byte_filter
        ? std::
            string{R"NVVM(  %restart_advance_base = phi i64 [ %restart_base, %candidate_fail ], [ %start, %start_filter_ascii_byte ]
)NVVM"}
        : std::string{};
    auto advance_base = machine.restart_state <= machine.state_mask
                          ? (machine.start_byte_filter ? "%restart_advance_base" : "%restart_base")
                          : "%start";
    auto advance_candidate =
      prefix_seek_byte_.has_value()
        ? regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                  "prefix_seek_start",
                                  advance_base,
                                  static_cast<std::uint32_t>(*prefix_seek_byte_))
        : std::format(
            R"NVVM(  %next_start = call i64 @{}(i8* %data, i64 %size, i64 {}, i64 1))NVVM",
            name("advance"),
            advance_base);
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "tagged_dfa_find_prefix",
                                         name("find_from"),
                                         search_target,
                                         start_filter));
    for (auto slot : capture_slots_) {
      output_.emit(
        "{}", regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa, "capture_n_ptr", slot));
    }
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "tagged_dfa_find_loop",
                                         machine.initial_state,
                                         name("load_byte"),
                                         name("dfa_classify"),
                                         name("decode_codepoint"),
                                         name("decode_width"),
                                         machine.class_count,
                                         machine.transitions.size(),
                                         machine.transition_address_space,
                                         name("dfa_transitions"),
                                         machine.dead_state));
    emit_capture_action_dispatch(
      "transition_capture", machine.transition_capture_actions, "%position", "consume");
    output_.emit("{}",
                 regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_dfa, "consume"));
    emit_capture_action_dispatch(
      "accept_capture", machine.accept_capture_actions, "%next_position", "yes");
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "tagged_dfa_accept",
                                         restart,
                                         advance_phi,
                                         advance_candidate));
    output_.blank();
  }

  /**
   * @brief emits a packed fixed-width comparison at one byte position
   *
   * @param literal non-empty ASCII literal to compare
   */
  void emit_ascii_literal_at(std::string_view literal)
  {
    std::string comparisons;
    std::string matched;
    std::size_t offset = 0;
    std::size_t index  = 0;
    while (offset < literal.size()) {
      auto remaining      = literal.size() - offset;
      auto width          = remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
      std::uint64_t value = 0;
      for (std::size_t byte = 0; byte < width; ++byte) {
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal[offset + byte]))
                 << (byte * 8U);
      }
      auto bits = width * 8U;
      std::format_to(
        std::back_inserter(comparisons),
        "  %literal_byte_ptr_{} = getelementptr i8, i8* %data, i64 %literal_offset_{}\n",
        index,
        index);
      if (width == 1U) {
        std::format_to(std::back_inserter(comparisons),
                       "  %literal_chunk_{} = load i8, i8* %literal_byte_ptr_{}, align 1\n",
                       index,
                       index);
      } else {
        std::format_to(std::back_inserter(comparisons),
                       "{}",
                       regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                               "literal_chunk_ptr_n",
                                               index,
                                               index,
                                               bits,
                                               index,
                                               bits,
                                               bits,
                                               index));
      }
      std::format_to(std::back_inserter(comparisons),
                     "  %literal_equal_{} = icmp eq i{} %literal_chunk_{}, {}\n",
                     index,
                     bits,
                     index,
                     value);
      if (matched.empty()) {
        matched = std::format("%literal_equal_{}", index);
      } else {
        std::format_to(std::back_inserter(comparisons),
                       "  %literal_equal_through_{} = and i1 {}, %literal_equal_{}\n",
                       index,
                       matched,
                       index);
        matched = std::format("%literal_equal_through_{}", index);
      }
      offset += width;
      ++index;
    }

    std::string offsets;
    offset = 0;
    for (std::size_t chunk = 0; chunk < index; ++chunk) {
      std::format_to(std::back_inserter(offsets),
                     "  %literal_offset_{} = add i64 %position, {}\n",
                     chunk,
                     offset);
      auto remaining = literal.size() - offset;
      offset += remaining >= 8U ? 8U : remaining >= 4U ? 4U : remaining >= 2U ? 2U : 1U;
    }

    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "tagged_dfa_execute_adapter",
                                         name("ascii_literal_at"),
                                         literal.size(),
                                         offsets,
                                         comparisons,
                                         matched));
    output_.blank();
  }

  /**
   * @brief Emits a row-level search for a mandatory ASCII literal.
   *
   * @param literal Mandatory literal already emitted through `emit_ascii_literal_at`
   */
  void emit_mandatory_literal_filter(std::string_view literal)
  {
    output_.emit("{}",
                 regex_ir::nvvm_template(
                   regex_ir_nvvm_templates::regex_literals,
                   "mandatory_literal_filter",
                   name("mandatory_literal_present"),
                   literal.size(),
                   name("load_byte"),
                   static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
                   name("ascii_literal_at")));
    output_.blank();
  }

  /**
   * @brief emits a direct boolean executor for an exact ASCII literal
   *
   * @param literal non-empty ASCII literal to search or match
   */
  void emit_ascii_literal_execute(std::string_view literal)
  {
    emit_ascii_literal_at(literal);
    if (!ir_.control.scan_input) {
      auto required_size =
        ir_.control.require_end
          ? std::format("  %required_size = icmp eq i64 %size, {}\n", literal.size())
          : std::string{"  %required_size = icmp uge i64 %size, 0\n"};
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_literals,
                                           "exact_ascii_execute",
                                           options_.execute_function,
                                           required_size,
                                           name("ascii_literal_at")));
      output_.blank();
      return;
    }

    if (literal.size() >= 8U) {
      emit_ascii_literal_find_from(literal);
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "boolean_find_adapter",
                                           options_.execute_function,
                                           name("find_from")));
      output_.blank();
      return;
    }

    auto verify = literal.size() == 1U
                    ? std::string{"  br i1 %candidate, label %yes, label %continue\n"}
                    : regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_literals,
                                              "branch_br_i1_candidate_label_verify_label_conti",
                                              name("ascii_literal_at"));
    output_.emit("{}",
                 regex_ir::nvvm_template(
                   regex_ir_nvvm_templates::regex_literals,
                   "ascii_pivot_seek",
                   options_.execute_function,
                   literal.size(),
                   name("load_byte"),
                   static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal.front())),
                   verify));
    output_.blank();
  }

  /**
   * @brief emits a selective-pivot seeker with packed verification for an exact UTF-8 literal
   *
   * @param literal Exact UTF-8 literal plan
   * @param pivot Offset of a selective ASCII byte within the encoded literal
   */
  void emit_utf8_pivot_find_from(utf8_literal const& literal,
                                 std::size_t pivot,
                                 std::string_view function)
  {
    auto encoded      = encode_utf8_literal(literal.codepoints);
    auto guard_offset = std::min(pivot, encoded.size() - sizeof(std::uint64_t));
    auto guard_value  = std::uint64_t{0};
    for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
      guard_value |=
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(encoded[guard_offset + byte]))
        << (byte * 8U);
    }
    emit_ascii_literal_at(encoded);
    output_.emit(
      "{}",
      regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_literals,
                              "utf8_pivot_seek",
                              function,
                              pivot,
                              static_cast<std::uint32_t>(static_cast<std::uint8_t>(encoded[pivot])),
                              encoded.size(),
                              name("ascii_literal_at"),
                              guard_offset,
                              llvm_i64(guard_value)));
    output_.blank();
  }

  void emit_utf8_literal_find_from(utf8_literal const& literal)
  {
    if (!utf8_literal_pivot_.has_value()) {
      emit_utf8_kmp_find_from(literal, name("find_from"));
      return;
    }
    if (ir_.control.result == result_shape::BOOLEAN ||
        ir_.control.result == result_shape::MATCH_COUNT) {
      emit_utf8_pivot_find_from(literal, *utf8_literal_pivot_, name("find_from"));
      return;
    }

    auto pivot = name("utf8_pivot_find_from");
    auto kmp   = name("utf8_kmp_find_from");
    emit_utf8_pivot_find_from(literal, *utf8_literal_pivot_, pivot);
    emit_utf8_kmp_find_from(literal, kmp);
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_literals,
                                         "utf8_literal_find_adapter",
                                         name("find_from"),
                                         pivot,
                                         kmp));
    output_.blank();
  }

  /**
   * @brief Emits a byte-domain KMP seeker for an exact UTF-8 literal.
   *
   * Every non-ASCII literal begins with a UTF-8 lead byte, so a byte-domain match cannot begin at a
   * continuation byte. Comparing the complete encoded sequence therefore preserves code-point
   * matching while eliminating decoder work. Input positions and returned spans use byte offsets.
   *
   * @param literal Exact UTF-8 literal plan
   */
  void emit_utf8_kmp_find_from(utf8_literal const& literal, std::string_view function)
  {
    auto encoded = encode_utf8_literal(literal.codepoints);
    auto bytes   = std::vector<std::uint8_t>(encoded.begin(), encoded.end());
    auto failure = std::vector<std::uint32_t>(bytes.size());
    for (std::size_t index = 1, prefix = 0; index < bytes.size();) {
      if (bytes[index] == bytes[prefix]) {
        failure[index++] = static_cast<std::uint32_t>(++prefix);
      } else if (prefix != 0U) {
        prefix = failure[prefix - 1U];
      } else {
        failure[index++] = 0U;
      }
    }
    output_.emit("@{} = internal addrspace(4) constant [{} x i8] [{}], align 1",
                 name("utf8_kmp_literal"),
                 bytes.size(),
                 format_i8_array(bytes));
    output_.emit("@{} = internal addrspace(4) constant [{} x i32] [{}], align 4",
                 name("utf8_kmp_failure"),
                 failure.size(),
                 format_i32_array(failure));
    output_.blank();
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_literals,
                                         "utf8_kmp_seek",
                                         function,
                                         name("load_byte"),
                                         bytes.size(),
                                         name("utf8_kmp_literal"),
                                         name("utf8_kmp_failure")));
    output_.blank();
  }

  /**
   * @brief Emits the boolean executor ABI around the UTF-8 KMP seeker.
   */
  void emit_utf8_kmp_execute()
  {
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "boolean_find_adapter",
                                         options_.execute_function,
                                         name("find_from")));
    output_.blank();
  }

  /**
   * @brief emits a packed ASCII-literal finder for span-producing operations
   *
   * @param literal non-empty multi-byte ASCII literal to search
   */
  void emit_ascii_literal_find_from(std::string_view literal)
  {
    auto pivot = ascii_literal_pivot(literal);
    output_.emit("{}",
                 regex_ir::nvvm_template(
                   regex_ir_nvvm_templates::regex_literals,
                   "ascii_literal_find",
                   name("find_from"),
                   literal.size(),
                   name("load_byte"),
                   pivot,
                   static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal[pivot])),
                   name("ascii_literal_at"),
                   literal.size() == 2U ? 129U : 64U,
                   static_cast<std::uint32_t>(static_cast<std::uint8_t>(literal[pivot - 1U]))));
    output_.blank();
  }

  /**
   * @brief emits a direct byte-search primitive for an exact ASCII character
   *
   * @param byte ASCII byte that forms the complete regex
   */
  void emit_single_byte_find_from(std::uint8_t byte)
  {
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_literals,
                                         "single_byte_find",
                                         name("find_from"),
                                         name("load_byte"),
                                         static_cast<std::uint32_t>(byte)));
    output_.blank();
  }

  void emit_assertion_deterministic_find_from(deterministic_machine const& machine)
  {
    auto search_target = machine.start_byte_filter ? "start_filter" : "attempt";
    auto start_filter  = render_start_byte_filter(machine, "attempt", false);
    auto advance_candidate =
      prefix_seek_byte_.has_value()
        ? regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                  "prefix_seek_start_2",
                                  static_cast<std::uint32_t>(*prefix_seek_byte_))
        : std::format(
            R"NVVM(  %next_start = call i64 @{}(i8* %data, i64 %size, i64 %start, i64 1))NVVM",
            name("advance"));
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "assertion_dfa_find",
                                         name("assertion_candidate"),
                                         name("previous_position"),
                                         name("decode_codepoint"),
                                         machine.initial_state,
                                         name("load_byte"),
                                         name("dfa_classify"),
                                         name("decode_width"),
                                         name("dfa_boundary_classify"),
                                         machine.boundary_class_count,
                                         machine.class_count,
                                         machine.transitions.size(),
                                         machine.transition_address_space,
                                         name("dfa_transitions"),
                                         machine.dead_state,
                                         machine.boundary_accepts.size(),
                                         name("dfa_boundary_accepts")));
    output_.blank();

    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "assertion_candidate",
                                         name("find_from"),
                                         name("assertion_candidate"),
                                         advance_candidate,
                                         search_target,
                                         start_filter));
    output_.blank();
  }

  /**
   * @brief emits a non-recursive leftmost-match primitive for an ordered deterministic automaton
   *
   * @param machine deterministic machine whose transitions are known to preserve priority
   */
  void emit_deterministic_find_from(deterministic_machine const& machine)
  {
    auto initial_accept = (machine.initial_state & 0x8000U) != 0 ? "%start" : "-1";
    // the direct edge wins for count but disrupts larger materializing control-flow graphs.
    auto direct_filter_advance = ir_.control.result == result_shape::MATCH_COUNT;
    auto search_target         = machine.start_byte_filter ? "start_filter" : "candidate";
    auto start_filter      = render_start_byte_filter(machine, "candidate", direct_filter_advance);
    auto filter_search_phi = machine.start_byte_filter && direct_filter_advance
                               ? ", [ %start_filter_next, %start_filter_advance ]"
                               : "";
    auto restart =
      machine.restart_state <= machine.state_mask
        ? regex_ir::nvvm_template(
            regex_ir_nvvm_templates::regex_dfa, "restart_state_match_2", machine.restart_state)
        : std::string{};
    auto advance_phi =
      machine.restart_state <= machine.state_mask && machine.start_byte_filter &&
          !direct_filter_advance
        ? std::
            string{R"NVVM(  %restart_advance_base = phi i64 [ %restart_base, %candidate_fail ], [ %start, %start_filter_ascii_byte ]
)NVVM"}
        : std::string{};
    auto advance_base =
      machine.restart_state <= machine.state_mask
        ? (machine.start_byte_filter && !direct_filter_advance ? "%restart_advance_base"
                                                               : "%restart_base")
        : "%start";
    auto advance_candidate =
      prefix_seek_byte_.has_value()
        ? regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                  "prefix_seek_start_3",
                                  advance_base,
                                  static_cast<std::uint32_t>(*prefix_seek_byte_))
        : std::format(
            R"NVVM(  %next_start = call i64 @{}(i8* %data, i64 %size, i64 {}, i64 1))NVVM",
            name("advance"),
            advance_base);
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_dfa,
                                         "deterministic_find",
                                         name("find_from"),
                                         search_target,
                                         start_filter,
                                         filter_search_phi,
                                         machine.initial_state,
                                         initial_accept,
                                         name("load_byte"),
                                         name("dfa_classify"),
                                         name("decode_codepoint"),
                                         name("decode_width"),
                                         machine.class_count,
                                         machine.transitions.size(),
                                         machine.transition_address_space,
                                         name("dfa_transitions"),
                                         machine.dead_state,
                                         restart,
                                         advance_phi,
                                         advance_candidate));
    output_.blank();
  }

  /**
   * @brief emits the shared leftmost-match primitive used by materializing operations
   */
  void emit_find_from()
  {
    auto function         = name("find_from");
    auto run_block        = name("run_block");
    auto advance          = name("advance");
    auto mandatory_filter = mandatory_ascii_literal_.has_value()
                              ? regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                                        "mandatory_at_begin",
                                                        name("mandatory_literal_present"))
                              : std::string{"  br label %search\n"};
    auto initial_predecessor =
      mandatory_ascii_literal_.has_value() ? "%mandatory_filter_done" : "%entry";
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                         "find_from_prefix",
                                         function,
                                         mandatory_filter,
                                         initial_predecessor));

    auto prefix = required_ascii_prefix();
    if (prefix.has_value() || candidate_seeker_) {
      auto hint = std::string{
        R"NVVM(  %prefix_likely = call i1 @llvm.expect.i1(i1 %prefix_candidate, i1 false)
)NVVM"};
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                           "branch_br_i1_in_range_label_prefix_end_label_no",
                                           name("load_byte"),
                                           start_candidate_test("%prefix_byte"),
                                           hint,
                                           "%prefix_likely"));
    } else {
      output_.emit("  br i1 %in_range, label %initialize, label %no");
    }

    output_.emit("initialize:");
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit("{}",
                   regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_thompson,
                                                   "find_capture_ptr_0"));
    }
    for (auto slot : capture_slots_) {
      output_.emit("{}",
                   regex_ir::nvvm_template(
                     regex_ir_nvvm_templates::regex_thompson, "find_capture_ptr_n", slot));
    }
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit("  store i64 %start, i64* %find_capture_ptr_0, align 8");
    }
    auto captures = uses_capture_buffer() ? "%captures" : "null";
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                         "store_store_i64_start_i64_position_align_8",
                                         run_block,
                                         ir_.entry,
                                         captures));
    if (prefix.has_value() || candidate_seeker_) {
      output_.emit("  br i1 %at_end, label %no, label %continue");
    } else {
      output_.emit(
        "{}", regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_thompson, "at_end"));
    }
    auto next_candidate =
      prefix_seek_byte_.has_value() || candidate_seeker_
        ? regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson,
                                  "prefix_seek_start",
                                  start_seeker_name(),
                                  start_seeker_arguments())
        : std::format(
            R"NVVM(  %next_start = call i64 @{}(i8* %data, i64 %size, i64 %start, i64 1))NVVM",
            advance);
    output_.emit(
      "{}",
      regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson, "continue", next_candidate));
    if (ir_.control.result == result_shape::CAPTURES) {
      output_.emit("  store i64 %accepted_end, i64* %find_capture_ptr_1, align 8");
    }
    output_.emit(
      "{}", regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_thompson, "return_ret_i1_true"));
    output_.blank();
  }

  /**
   * @brief adapts an input-anchored boolean executor to a cardinality or match-start ABI
   */
  void emit_anchored_boolean_adapter()
  {
    if (!anchored_boolean_result_.has_value()) return;
    if (*anchored_boolean_result_ == result_shape::MATCH_COUNT) {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "boolean_count_adapter",
                                           public_execute_function_,
                                           options_.execute_function));
    } else {
      output_.emit("{}",
                   regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                           "operation_execute_adapter",
                                           public_execute_function_,
                                           options_.execute_function));
    }
    output_.blank();
  }

  /**
   * @brief emits the first-match span ABI for a find operation
   */
  void emit_find_execute()
  {
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "find_execute",
                                         options_.execute_function,
                                         name("find_from")));
    output_.blank();
  }

  /**
   * @brief emits the search-offset whole-match ABI used by findall wrappers
   */
  void emit_find_all_execute()
  {
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "find_all_execute",
                                         options_.execute_function,
                                         name("find_from")));
    output_.blank();
  }

  /**
   * @brief emits the first-match capture ABI used by extract and enumeration wrappers
   */
  void emit_capture_execute()
  {
    auto aliases = std::string{};
    if (!whole_match_captures_.empty()) {
      aliases = regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_operations,
                                                "branch_br_i1_matched_label_alias_whole_match_la");
      for (auto capture : whole_match_captures_) {
        auto slot = static_cast<std::size_t>(capture) * 2U;
        std::format_to(std::back_inserter(aliases),
                       "{}",
                       regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                               "whole_capture_begin_ptr_n",
                                               capture,
                                               slot,
                                               slot + 1U));
      }
      aliases += regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_operations,
                                                 "branch_br_label_done");
    }
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "capture_execute",
                                         options_.execute_function,
                                         name("find_from"),
                                         aliases));
    output_.blank();
  }

  /**
   * @brief emits a complete non-overlapping match-count executor
   */
  void emit_count_execute()
  {
    output_.emit("{}",
                 regex_ir::render_nvvm_template_section(
                   regex_ir_nvvm_templates::regex_operations,
                   repeated_builtin_.has_value() ? "count_builtin_execute" : "count_execute",
                   {{"@EXECUTE@", std::format("@{}", options_.execute_function)},
                    {"@FIND_FROM@", std::format("@{}", name("find_from"))},
                    {"@ADVANCE@", std::format("@{}", name("advance"))}}));
    output_.blank();
  }

  /**
   * @brief emits the range-copy primitive used by the replacement executor
   */
  void emit_append_range()
  {
    output_.emit(
      "{}",
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "append_range"));
    output_.blank();
  }

  /**
   * @brief emits a sizing-and-materialization executor specialized for a replacement template
   */
  void emit_replace_execute()
  {
    emit_append_range();
    auto capture_argument    = std::string{"null"};
    auto capture_declaration = std::string{};
    if (uses_capture_buffer()) {
      auto slots = static_cast<std::size_t>(ir_.capture_count + 1U) * 2U;
      capture_declaration =
        regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations, "capture_array", slots);
      capture_argument = "%captures";
    }
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "replace_execute",
                                         options_.execute_function,
                                         capture_declaration,
                                         name("find_from"),
                                         capture_argument,
                                         "regex_ir_append_range"));
    std::string cursor = "%unmatched_cursor";
    for (std::size_t index = 0; index < ir_.replacement.size(); ++index) {
      auto& token = ir_.replacement[index];
      if (token.type == replacement_token::kind::LITERAL) {
        if (token.literal.empty()) continue;
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                             "replacement_constant_n",
                                             index,
                                             token.literal.size(),
                                             name(std::format("replacement_{}", index)),
                                             "regex_ir_append_range",
                                             cursor));
      } else if (token.capture_index == 0 || is_whole_match_capture(token.capture_index)) {
        output_.emit(
          "{}",
          std::format(
            R"NVVM(  %replacement_cursor_{0} = call i64 @{1}(i8* %data, i64 %match_begin_value, i64 %match_end_value, i8* %output, i64 {2}))NVVM",
            index,
            "regex_ir_append_range",
            cursor));
      } else {
        auto slot = static_cast<std::size_t>(token.capture_index) * 2U;
        output_.emit("{}",
                     regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                             "replacement_capture_begin_ptr_n",
                                             index,
                                             slot,
                                             slot + 1U,
                                             "regex_ir_append_range",
                                             cursor));
      }
      cursor = std::format("%replacement_cursor_{}", index);
    }
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "replacement_capture_step",
                                         cursor,
                                         name("advance"),
                                         "regex_ir_append_range"));
    output_.blank();
  }

  /**
   * @brief emits the optional span-store primitive used by split sizing and materialization
   */
  void emit_write_span()
  {
    output_.emit(
      "{}",
      regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "write_span"));
    output_.blank();
  }

  /**
   * @brief emits a sizing-and-span-materialization executor specialized for split
   */
  void emit_split_execute()
  {
    emit_write_span();
    output_.emit("{}",
                 regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                         "split_execute",
                                         options_.execute_function,
                                         name("find_from"),
                                         "regex_ir_write_span",
                                         name("advance")));
    output_.blank();
  }

  instruction_ir ir_;
  nvvm_ir_codegen_options options_;
  std::string public_execute_function_;
  std::optional<result_shape> anchored_boolean_result_     = std::nullopt;
  std::optional<std::string> exact_ascii_literal_metadata_ = std::nullopt;
  std::optional<std::string> exact_literal_bytes_          = std::nullopt;
  std::optional<builtin_character_class> repeated_builtin_ = std::nullopt;
  std::optional<deterministic_machine> deterministic_      = std::nullopt;
  std::optional<glushkov_machine> glushkov_                = std::nullopt;
  std::optional<string_operation> string_operations_       = std::nullopt;
  std::optional<fixed_ascii_suffix> fixed_ascii_suffix_    = std::nullopt;
  std::optional<std::string> line_tail_literal_            = std::nullopt;
  std::optional<std::uint32_t> word_run_minimum_           = std::nullopt;
  std::optional<std::string> ascii_literal_                = std::nullopt;
  std::optional<utf8_literal> utf8_literal_                = std::nullopt;
  std::optional<std::size_t> utf8_literal_pivot_           = std::nullopt;
  std::optional<std::string> mandatory_ascii_literal_      = std::nullopt;
  std::optional<std::uint8_t> prefix_seek_byte_            = std::nullopt;
  bool candidate_seeker_                                   = false;
  std::vector<std::uint32_t> whole_match_captures_         = std::vector<std::uint32_t>{};
  std::vector<std::size_t> capture_slots_                  = std::vector<std::size_t>{};
  std::optional<deterministic_nfa_graph> thompson_;
  std::size_t workspace_bytes_ = 0;
  source_buffer output_        = source_buffer{};
};

std::string_view module_body(std::string_view module)
{
  auto begin = std::string_view::npos;
  for (auto marker :
       {std::string_view{"\n@"}, std::string_view{"\ndefine "}, std::string_view{"\ndeclare "}}) {
    auto candidate = module.find(marker);
    if (candidate != std::string_view::npos) begin = std::min(begin, candidate + 1U);
  }
  if (begin == std::string_view::npos) {
    throw std::invalid_argument("generated alternative module has no declarations");
  }
  auto end = module.find("\n!nvvmir.version", begin);
  return module.substr(begin, end == std::string_view::npos ? end : end - begin);
}

void append_module_body(std::string& output, std::string_view body)
{
  while (!body.empty()) {
    auto newline = body.find(static_cast<char>(10));
    auto length  = newline == std::string_view::npos ? body.size() : newline + 1U;
    auto line    = body.substr(0, length);
    if (!line.starts_with("declare ") || output.find(line) == std::string::npos) {
      output.append(line);
    }
    body.remove_prefix(length);
  }
}

std::string_view module_without_metadata(std::string_view module)
{
  auto end = module.find("\n!nvvmir.version");
  return module.substr(0, end);
}

std::optional<compile_result> render_large_boolean_alternation(
  instruction_ir const& ir, nvvm_ir_codegen_options const& options)
{
  if (ir.control.result != result_shape::BOOLEAN || ir.blocks.size() < 80U ||
      ir.entry >= ir.blocks.size()) {
    return std::nullopt;
  }
  auto& entry = ir.blocks[ir.entry];
  if (!entry.instructions.empty() || entry.successors.size() < 2U) return std::nullopt;

  std::vector<instruction_ir> alternatives;
  alternatives.reserve(entry.successors.size());
  for (auto edge : entry.successors) {
    auto branch  = ir;
    branch.entry = edge.target;
    alternatives.push_back(optimize(std::move(branch), {}));
  }

  std::string result;
  std::vector<std::string> functions;
  std::vector<bool> uses_workspace;
  std::size_t workspace_bytes = 0;
  functions.reserve(alternatives.size());
  for (std::size_t index = 0; index < alternatives.size(); ++index) {
    auto branch_options                       = options;
    branch_options.emit_general_functions     = options.emit_general_functions && index == 0;
    branch_options.emit_all_general_functions = branch_options.emit_general_functions;
    branch_options.symbol_prefix += std::format("_alternative_{}", index);
    branch_options.execute_function += std::format("_alternative_{}", index);
    functions.push_back(branch_options.execute_function);
    auto nested    = render_large_boolean_alternation(alternatives[index], branch_options);
    auto generated = nested.has_value()
                       ? std::move(*nested)
                       : nvvm_ir_renderer(alternatives[index], branch_options).render();
    uses_workspace.push_back(generated.workspace_bytes != 0);
    workspace_bytes = std::max(workspace_bytes, generated.workspace_bytes);
    if (index == 0) {
      result = module_without_metadata(generated.nvvm_ir);
    } else {
      result += '\n';
      append_module_body(result, module_body(generated.nvvm_ir));
    }
  }

  std::string branches;
  for (std::size_t index = 0; index < functions.size(); ++index) {
    auto label = index == 0 ? std::string{"entry"} : std::format("alternative_{}", index);
    branches += regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                        "n",
                                        label,
                                        index,
                                        functions[index],
                                        uses_workspace[index] ? "i8* %workspace, " : "");
    if (index + 1U < functions.size()) {
      branches +=
        std::format("  br i1 %matched_{}, label %yes, label %alternative_{}\n", index, index + 1U);
    } else {
      branches += std::format("  ret i1 %matched_{}\n", index);
    }
  }
  result += regex_ir::nvvm_template(regex_ir_nvvm_templates::regex_operations,
                                    "boolean_alternation_execute",
                                    options.execute_function,
                                    branches,
                                    workspace_bytes ? "i8* %workspace, " : "");
  result +=
    regex_ir::nvvm_template_section(regex_ir_nvvm_templates::regex_functions, "module_metadata");
  return compile_result{std::move(result),
                        ir.capture_count,
                        executor_kind::BOOLEAN_ALTERNATION,
                        0,
                        0,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        workspace_bytes};
}

}  // namespace

compile_result generate_nvvm_ir(instruction_ir const& ir, nvvm_ir_codegen_options const& options)
{
  if (auto alternatives = render_large_boolean_alternation(ir, options)) {
    return std::move(*alternatives);
  }
  return nvvm_ir_renderer(ir, options).render();
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
    return generate_nvvm_ir(compiled, {});
  } catch (compile_failure const& failure) {
    throw std::invalid_argument(std::format(
      "regex compilation failed at byte {}: {}", failure.source.offset, failure.message));
  }
}

}  // namespace regex_ir
