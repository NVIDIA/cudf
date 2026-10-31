/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file parser.cpp
 * @brief Regex parsing, character predicates, and Thompson automata construction.
 *
 * Builds the private AST, assigns capture groups, and constructs ordered automata
 * and lookaround subprograms while enforcing compilation limits.
 */

#include <libregex_ir/detail/case_mapping.hpp>
#include <libregex_ir/detail/detail.hpp>
#include <libregex_ir/detail/unicode.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace regex_ir {
namespace {

enum class node_kind : std::uint8_t {
  EMPTY       = 0,
  PREDICATE   = 1,
  CONCATENATE = 2,
  ALTERNATE   = 3,
  REPEAT      = 4,
  GROUP       = 5,
  ASSERTION   = 6,
  LOOKAROUND  = 7,
};

struct flag_context {
  bool case_insensitive = false;
  bool multiline        = false;
  bool dot_all          = false;
  bool ascii_classes    = true;
  bool verbose          = false;
  bool inline_case      = false;
  bool ascii_case       = false;
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
  bool reverse                                = false;
  bool negative                               = false;
};

bool can_consume_character(node const& value)
{
  switch (value.kind) {
    case node_kind::PREDICATE: return true;
    case node_kind::EMPTY:
    case node_kind::ASSERTION:
    case node_kind::LOOKAROUND: return false;
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
    case node_kind::ASSERTION:
    case node_kind::LOOKAROUND: return false;
  }
  return false;
}

bool can_match_empty(node const& value)
{
  switch (value.kind) {
    case node_kind::EMPTY:
    case node_kind::ASSERTION:
    case node_kind::LOOKAROUND: return true;
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

bool contains_lookaround(node const& value)
{
  if (value.kind == node_kind::LOOKAROUND) return true;
  return std::any_of(value.children.begin(), value.children.end(), [](auto& child) {
    return contains_lookaround(*child);
  });
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

void add_case_pair(character_predicate& predicate, char32_t first, char32_t last)
{
  predicate.ranges.push_back({first, last});
  auto mapping = std::lower_bound(
    std::begin(unicode_case_ranges),
    std::end(unicode_case_ranges),
    static_cast<std::uint32_t>(first),
    [](unicode_case_range const& range, std::uint32_t value) { return range.last < value; });
  for (; mapping != std::end(unicode_case_ranges) && mapping->first <= last; ++mapping) {
    auto begin = std::max(static_cast<std::uint32_t>(first), mapping->first);
    auto end   = std::min(static_cast<std::uint32_t>(last), mapping->last);
    if (mapping->delta == 0) {
      // Complete swapped pairs are already in the original range. Only cut pairs extend it.
      if ((begin - mapping->first) % 2) predicate.ranges.push_back({begin - 1, begin - 1});
      if ((end - mapping->first) % 2 == 0) predicate.ranges.push_back({end + 1, end + 1});
    } else {
      predicate.ranges.push_back({static_cast<char32_t>(begin + mapping->delta),
                                  static_cast<char32_t>(end + mapping->delta)});
    }
  }
}

void add_inline_case_pair(character_predicate& predicate, char32_t first, char32_t last, bool ascii)
{
  character_predicate folded;
  folded.ranges.push_back({first, last});
  if (ascii) {
    for (auto range : {codepoint_range{U'A', U'Z'}, codepoint_range{U'a', U'z'}}) {
      auto begin = std::max(first, range.first);
      auto end   = std::min(last, range.last);
      if (begin <= end) {
        auto const delta = range.first == U'A' ? 32 : -32;
        folded.ranges.push_back(
          {static_cast<char32_t>(begin + delta), static_cast<char32_t>(end + delta)});
      }
    }
  } else {
    // Include inverse mappings as well: S/s/long-s and sigma/final-sigma share a class.
    for (;;) {
      auto const previous = folded.ranges;
      for (auto range : previous)
        add_case_pair(folded, range.first, range.last);
      for (auto mapping : unicode_case_ranges) {
        if (mapping.delta == 0) continue;  // Adjacent pairs are already symmetric.
        auto target_first = static_cast<char32_t>(mapping.first + mapping.delta);
        auto target_last  = static_cast<char32_t>(mapping.last + mapping.delta);
        for (auto range : previous) {
          auto begin = std::max(range.first, target_first);
          auto end   = std::min(range.last, target_last);
          if (begin <= end)
            folded.ranges.push_back({static_cast<char32_t>(begin - mapping.delta),
                                     static_cast<char32_t>(end - mapping.delta)});
        }
      }
      normalize_ranges(folded);
      if (std::equal(previous.begin(),
                     previous.end(),
                     folded.ranges.begin(),
                     folded.ranges.end(),
                     [](auto left, auto right) {
                       return left.first == right.first && left.last == right.last;
                     }))
        break;
    }
  }
  predicate.ranges.insert(predicate.ranges.end(), folded.ranges.begin(), folded.ranges.end());
}

class parser {
 public:
  parser(std::string_view pattern, compile_options const& options)
    : pattern_(pattern),
      options_(options),
      flags_{options.case_insensitive, options.multiline, options.dot_all, options.ascii_classes}
  {
  }

  std::unique_ptr<node> parse()
  {
    if (pattern_.size() > options_.limits.max_pattern_bytes) {
      fail({0, pattern_.size()}, "pattern exceeds max_pattern_bytes");
    }

    while (starts_flags()) {
      auto const start       = position_;
      auto const saved_flags = flags_;
      position_ += 2;
      if (parse_flags(start)) {
        position_ = start;
        flags_    = saved_flags;
        break;
      }
      skip_ignored();
    }
    root_flags_     = flags_;
    auto expression = parse_alternation();
    if (position_ != pattern_.size()) { fail({position_, 1}, "unexpected token"); }

    return expression;
  }

  std::uint32_t capture_count = 0;
  bool has_lookaround         = false;
  void apply_root_flags(compile_options& options) const
  {
    options.case_insensitive = root_flags_.case_insensitive;
    options.multiline        = root_flags_.multiline;
    options.dot_all          = root_flags_.dot_all;
    options.ascii_classes    = root_flags_.ascii_classes;
  }

 private:
  static bool is_flag(char value)
  {
    return value == 'i' || value == 'm' || value == 's' || value == 'a' || value == 'u' ||
           value == 'x';
  }

  bool starts_flags() const
  {
    return pattern_.substr(position_).starts_with("(?") && position_ + 2 < pattern_.size() &&
           (is_flag(pattern_[position_ + 2]) || pattern_[position_ + 2] == '-');
  }

  void skip_ignored()
  {
    if (!flags_.verbose) return;
    while (!at_end()) {
      char const value = peek();
      if (value == '#') {
        while (!at_end() && take() != '\n') {}
      } else if (value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' ||
                 value == '\v') {
        ++position_;
      } else {
        break;
      }
    }
  }

  // Return true for a scoped group and false for a prefix directive.
  bool parse_flags(std::size_t start)
  {
    constexpr std::string_view flags = "imsaux";
    auto bit                         = [&](char value) { return 1U << flags.find(value); };
    unsigned enabled                 = 0;
    unsigned disabled                = 0;
    bool removing                    = false;
    while (!at_end() && peek() != ':' && peek() != ')') {
      auto const value = take();
      if (value == '-' && !removing) {
        removing = true;
        continue;
      }
      if (!is_flag(value)) fail({start, position_ - start}, "unsupported inline flag");
      if (removing && (value == 'a' || value == 'u'))
        fail({start, position_ - start}, "ASCII and Unicode flags cannot be disabled");
      (removing ? disabled : enabled) |= bit(value);
    }
    if ((enabled | disabled) == 0 || (enabled & disabled) != 0 ||
        ((enabled & bit('a')) && (enabled & bit('u'))) || (removing && disabled == 0))
      fail({start, position_ - start}, "invalid inline flag combination");
    auto const scoped = consume(':');
    if (!scoped && (!consume(')') || removing))
      fail({start, position_ - start}, "invalid inline flag directive");
    if ((enabled & bit('u')) && options_.characters == character_mode::BYTES)
      fail({start, position_ - start}, "Unicode inline mode requires UTF-8 input");
    auto set = [&](char flag, bool& value) {
      if (enabled & bit(flag)) value = true;
      if (disabled & bit(flag)) value = false;
    };
    set('i', flags_.case_insensitive);
    set('m', flags_.multiline);
    set('s', flags_.dot_all);
    set('x', flags_.verbose);
    if ((enabled | disabled) & (bit('i') | bit('a') | bit('u'))) flags_.inline_case = true;
    if (enabled & bit('a')) flags_.ascii_case = true;
    if (enabled & bit('u')) flags_.ascii_case = false;
    if (enabled & bit('a')) flags_.ascii_classes = true;
    if (enabled & bit('u')) flags_.ascii_classes = false;
    return scoped;
  }

  void add_case(character_predicate& predicate, char32_t first, char32_t last)
  {
    if (flags_.inline_case)
      add_inline_case_pair(
        predicate, first, last, flags_.ascii_case || options_.characters == character_mode::BYTES);
    else
      add_case_pair(predicate, first, last);
  }

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
    skip_ignored();
    while (!at_end() && peek() != ')' && peek() != '|') {
      children.push_back(parse_quantified());
      skip_ignored();
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
    skip_ignored();
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
    if (!can_consume_character(*atom) && !contains_lookaround(*atom)) {
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
      if (kind == 's') result.ranges = {{U'\t', U'\r'}, {U' ', U' '}};
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
    if (flags_.ascii_classes) {
      append_ascii(base);
    } else {
      append_unicode(base);
    }
    normalize_ranges(result);
    bool negative = value == 'D' || value == 'W' || value == 'S';
    if (negative) {
      result.ranges = complement_ranges(std::move(result.ranges));
      if (!flags_.ascii_classes && (value == 'D' || value == 'W')) {
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
    // Adapter predicates use the program-wide class mode. A scoped override
    // must retain its explicit ranges instead of selecting that fast path.
    if (flags_.ascii_classes != root_flags_.ascii_classes)
      result.recognized = predicate_class::NONE;
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

  assertion_kind contextual_assertion(assertion_kind kind) const
  {
    if (flags_.multiline != root_flags_.multiline) {
      if (kind == assertion_kind::BEGIN_LINE)
        kind = flags_.multiline ? assertion_kind::SCOPED_BEGIN_LINE : assertion_kind::BEGIN_INPUT;
      if (kind == assertion_kind::END_LINE) kind = assertion_kind::SCOPED_END_LINE;
    }
    if (flags_.ascii_classes != root_flags_.ascii_classes) {
      if (kind == assertion_kind::WORD_BOUNDARY) kind = assertion_kind::SCOPED_WORD_BOUNDARY;
      if (kind == assertion_kind::NOT_WORD_BOUNDARY)
        kind = assertion_kind::SCOPED_NOT_WORD_BOUNDARY;
    }
    return kind;
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
      result->assertion = contextual_assertion(result->assertion);
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
      if (flags_.case_insensitive && !in_class) {
        add_case(result->predicate, literal, literal);
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
      if (flags_.case_insensitive) {
        add_case(result->predicate, lower, upper);
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
    for (;;) {
      auto start       = position_;
      char const value = take();
      if (value == '(') {
        auto const saved_flags = flags_;
        bool capturing         = true;
        bool lookaround        = false;
        bool reverse           = false;
        bool negative          = false;
        if (consume('?')) {
          capturing = false;
          if (is_flag(peek()) || peek() == '-') {
            if (!parse_flags(start)) {
              skip_ignored();
              if (peek() == '*' || peek() == '+' || peek() == '?' ||
                  (peek() == '{' && position_ + 1 < pattern_.size() &&
                   std::isdigit(static_cast<std::uint8_t>(pattern_[position_ + 1])) != 0))
                fail({position_, 1}, "inline flag directives cannot be repeated");
              if (at_end() || peek() == ')' || peek() == '|') return make(node_kind::EMPTY, start);
              continue;
            }
          } else if (!consume(':')) {
            reverse = consume('<');
            if (consume('=')) {
              lookaround = true;
            } else if (consume('!')) {
              lookaround = true;
              negative   = true;
            } else {
              fail({start, position_ - start + 1}, "unsupported inline group extension");
            }
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
        flags_ = saved_flags;
        --depth_;
        has_lookaround  = has_lookaround || lookaround;
        auto group      = make(lookaround ? node_kind::LOOKAROUND : node_kind::GROUP, start);
        group->reverse  = reverse;
        group->negative = negative;
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
        result->predicate.matches_newline  = flags_.dot_all;
        result->predicate.extended_newline = options_.extended_newline;
        return result;
      }
      if (value == '^' || value == '$') {
        auto result       = make(node_kind::ASSERTION, start);
        result->assertion = contextual_assertion(value == '^' ? assertion_kind::BEGIN_LINE
                                                              : assertion_kind::END_LINE);
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
      if (flags_.case_insensitive) {
        add_case(result->predicate, literal, literal);
      } else {
        result->predicate.ranges.push_back({literal, literal});
      }
      normalize_ranges(result->predicate);
      result->source.length = length;
      return result;
    }
  }

  std::string_view pattern_;
  compile_options const& options_;
  flag_context flags_;
  flag_context root_flags_;
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

struct construction_budget {
  std::size_t states      = 0;
  std::size_t transitions = 0;
  bool lookaround         = false;
};

class thompson_builder {
 public:
  thompson_builder(std::string_view pattern,
                   compile_options const& options,
                   construction_budget& budget)
    : budget_(budget)
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
      case node_kind::LOOKAROUND: return make_lookaround(expression);
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
    ir.entry          = entry;
    ir.accept         = accept;
    ir.capture_count  = captures;
    auto set_captures = [&](auto&& self, automata_ir& program) -> void {
      program.capture_count = captures;
      for (auto& nested : program.lookarounds)
        self(self, nested);
    };
    set_captures(set_captures, ir);
  }

 private:
  state_id add_state(automata_state_kind kind, source_span source)
  {
    if (budget_.states >= ir.options.limits.max_states) {
      throw compile_failure{source, "state count exceeds max_states"};
    }
    auto id = static_cast<state_id>(ir.states.size());
    automata_state state;
    state.id     = id;
    state.kind   = kind;
    state.source = source;
    ir.states.push_back(std::move(state));
    ++budget_.states;
    return id;
  }

  std::size_t add_edge(state_id from, state_id to, std::uint32_t priority)
  {
    if (budget_.transitions >= ir.options.limits.max_transitions) {
      throw compile_failure{ir.states[from].source, "transition count exceeds limit"};
    }
    ++budget_.transitions;
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

  fragment make_lookaround(node const& expression)
  {
    if (!lookaround_programs_) lookaround_programs_.emplace();
    auto& programs = *lookaround_programs_;
    auto found     = programs.find(&expression);
    std::uint32_t index;
    if (found != programs.end()) {
      index = found->second;
    } else {
      thompson_builder nested(ir.pattern, ir.options, budget_);
      nested.ir.reverse = expression.reverse;
      auto inner        = nested.build(*expression.children.front());
      nested.finish(std::move(inner), 0);
      index = static_cast<std::uint32_t>(ir.lookarounds.size());
      ir.lookarounds.push_back(std::move(nested.ir));
      programs.emplace(&expression, index);
    }
    auto state                     = add_state(automata_state_kind::ASSERTION, expression.source);
    ir.states[state].assertion     = assertion_kind::LOOKAROUND;
    ir.states[state].operand_index = index;
    ir.states[state].negative      = expression.negative;
    return {state, {add_open_edge(state, 0)}};
  }

  fragment make_group(node const& expression)
  {
    if (!expression.capturing) return build(*expression.children.front());

    auto begin                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[begin].capture       = ir.reverse ? capture_action::END : capture_action::BEGIN;
    ir.states[begin].operand_index = expression.capture_index;

    auto inner = build(*expression.children.front());
    add_edge(begin, inner.start, 0);

    auto end                     = add_state(automata_state_kind::CAPTURE, expression.source);
    ir.states[end].capture       = ir.reverse ? capture_action::BEGIN : capture_action::END;
    ir.states[end].operand_index = expression.capture_index;
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
    auto child = [&](std::size_t index) -> node const& {
      return *expression.children[ir.reverse ? expression.children.size() - index - 1 : index];
    };
    auto result = build(child(0));
    for (std::size_t index = 1; index < expression.children.size(); ++index) {
      result = concatenate(std::move(result), build(child(index)));
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
    auto first           = ir.states.size();
    auto inner           = build(expression);
    auto last            = ir.states.size();
    auto branch          = add_state(automata_state_kind::BRANCH, span);
    auto repeat_priority = greedy ? 0U : 1U;
    auto exit_priority   = greedy ? 1U : 0U;
    add_edge(branch, inner.start, repeat_priority);
    std::vector<patch_reference> exits{add_open_edge(branch, exit_priority)};
    if (budget_.lookaround && can_match_empty(expression)) {
      // Track whether this iteration consumed a character. Empty iterations exit
      // immediately, after exporting their captures. Consuming iterations repeat.
      // Distinct states keep epsilon-closure deduplication valid for both paths.
      std::vector<state_id> consumed(last - first);
      for (auto id = first; id < last; ++id) {
        auto original = ir.states[id];
        auto copy     = add_state(original.kind, original.source);
        original.id   = copy;
        original.edges.clear();
        ir.states[copy]      = std::move(original);
        consumed[id - first] = copy;
      }
      for (auto id = first; id < last; ++id) {
        auto edges = ir.states[id].edges;
        for (auto edge : edges)
          add_edge(consumed[id - first],
                   edge.target == invalid_state ? invalid_state : consumed[edge.target - first],
                   edge.priority);
        if (ir.states[id].kind == automata_state_kind::CONSUME)
          for (auto& edge : ir.states[id].edges)
            if (edge.target != invalid_state) edge.target = consumed[edge.target - first];
      }
      for (auto out : inner.outs) {
        ir.states[consumed[out.state - first]].edges[out.edge].target = branch;
        if (ir.states[out.state].kind == automata_state_kind::CONSUME)
          ir.states[out.state].edges[out.edge].target = branch;
        else
          exits.push_back(out);
      }
    } else {
      patch(inner.outs, branch);
    }
    return {branch, std::move(exits)};
  }

  fragment make_repeat(node const& expression)
  {
    auto& repeated = *expression.children.front();
    if (expression.maximum == unbounded_repeat && can_match_empty(repeated) &&
        !budget_.lookaround) {
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

  construction_budget& budget_;
  std::optional<std::map<node const*, std::uint32_t>> lookaround_programs_;
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

  for (auto& nested : ir.lookarounds)
    verify(nested);
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
    if (state.kind == automata_state_kind::ASSERTION &&
        state.assertion == assertion_kind::LOOKAROUND &&
        (state.operand_index >= ir.lookarounds.size() || state.edges.size() != 1))
      invalid("lookaround state is invalid");
    if (state.kind == automata_state_kind::CAPTURE &&
        (state.operand_index == 0 || state.operand_index > ir.capture_count)) {
      invalid("capture index is out of range");
    }
  }
}

automata_ir compile_automata(std::string_view pattern, compile_options const& options)
{
  parser parse_pattern(pattern, options);
  auto expression = parse_pattern.parse();

  construction_budget budget;
  budget.lookaround = parse_pattern.has_lookaround;
  thompson_builder builder(pattern, options, budget);
  parse_pattern.apply_root_flags(builder.ir.options);
  auto fragment = builder.build(*expression);
  builder.finish(std::move(fragment), parse_pattern.capture_count);
  verify(builder.ir);
  return std::move(builder.ir);
}

}  // namespace regex_ir
