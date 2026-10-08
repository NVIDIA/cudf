/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_codegen.hpp"

#include "cuda_abi.hpp"
#include "execution_plan.hpp"
#include "regex_ir_unicode.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace regex_ir::detail {
namespace {

class source_buffer {
 public:
  template <typename... Arguments>
  void emit(std::format_string<Arguments...> format, Arguments&&... arguments)
  {
    std::format_to(std::back_inserter(value_), format, std::forward<Arguments>(arguments)...);
    value_ += '\n';
  }

  void blank() { value_ += '\n'; }
  [[nodiscard]] std::string take() { return std::move(value_); }

 private:
  std::string value_ = "";
};

void require_identifier(std::string_view value, std::string_view field)
{
  auto first_is_valid = [](std::uint8_t character) {
    return std::isalpha(character) != 0 || character == '_';
  };
  auto rest_is_valid = [&](std::uint8_t character) {
    return first_is_valid(character) || std::isdigit(character) != 0;
  };
  if (value.empty() || !first_is_valid(static_cast<std::uint8_t>(value.front())) ||
      !std::all_of(value.begin() + 1, value.end(), [&](char character) {
        return rest_is_valid(static_cast<std::uint8_t>(character));
      })) {
    throw std::invalid_argument(std::format("{} must be a valid source identifier", field));
  }
}

// Emit symbolic enum constants; their numeric ordinals do not form the fragment ABI.
std::string executor_name(executor_kind kind)
{
  switch (kind) {
    case executor_kind::ITERATIVE_THOMPSON:
      return "regex_ir::device::executor_kind::ITERATIVE_THOMPSON";
    case executor_kind::STRING_OPERATIONS:
      return "regex_ir::device::executor_kind::STRING_OPERATIONS";
    case executor_kind::WORD_RUN: return "regex_ir::device::executor_kind::WORD_RUN";
    case executor_kind::SINGLE_BYTE_LITERAL:
      return "regex_ir::device::executor_kind::SINGLE_BYTE_LITERAL";
    case executor_kind::PACKED_ASCII_LITERAL:
      return "regex_ir::device::executor_kind::PACKED_ASCII_LITERAL";
    case executor_kind::PACKED_UTF8_LITERAL:
      return "regex_ir::device::executor_kind::PACKED_UTF8_LITERAL";
    case executor_kind::UTF8_KMP_LITERAL:
      return "regex_ir::device::executor_kind::UTF8_KMP_LITERAL";
    case executor_kind::GLUSHKOV: return "regex_ir::device::executor_kind::GLUSHKOV";
    case executor_kind::STREAMING_PRIORITIZED_GLUSHKOV:
      return "regex_ir::device::executor_kind::STREAMING_PRIORITIZED_GLUSHKOV";
    case executor_kind::DETERMINISTIC: return "regex_ir::device::executor_kind::DETERMINISTIC";
    case executor_kind::ASSERTION_AWARE_DETERMINISTIC:
      return "regex_ir::device::executor_kind::ASSERTION_AWARE_DETERMINISTIC";
    case executor_kind::PRIORITIZED_DETERMINISTIC:
      return "regex_ir::device::executor_kind::PRIORITIZED_DETERMINISTIC";
    case executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC:
      return "regex_ir::device::executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC";
    case executor_kind::BOOLEAN_ALTERNATION:
      return "regex_ir::device::executor_kind::BOOLEAN_ALTERNATION";
  }
  throw std::invalid_argument("invalid executor kind");
}

std::string assertion_name(assertion_kind kind)
{
  switch (kind) {
    case assertion_kind::BEGIN_INPUT: return "regex_ir::device::assertion_kind::BEGIN_INPUT";
    case assertion_kind::END_INPUT: return "regex_ir::device::assertion_kind::END_INPUT";
    case assertion_kind::WORD_BOUNDARY: return "regex_ir::device::assertion_kind::WORD_BOUNDARY";
    case assertion_kind::NOT_WORD_BOUNDARY:
      return "regex_ir::device::assertion_kind::NOT_WORD_BOUNDARY";
    case assertion_kind::BEGIN_LINE: return "regex_ir::device::assertion_kind::BEGIN_LINE";
    case assertion_kind::END_LINE: return "regex_ir::device::assertion_kind::END_LINE";
  }
  throw std::invalid_argument("invalid assertion kind");
}

class cuda_source_emitter {
 public:
  cuda_source_emitter(execution_plan const& plan, cuda_codegen_options options)
    : plan_(plan), options_(std::move(options))
  {
  }

  compile_result render()
  {
    require_identifier(options_.symbol_prefix, "symbol_prefix");
    require_identifier(options_.execute_function, "execute_function");
    output_.emit(R"CUDA(namespace {} {{
using namespace regex_ir::device;)CUDA",
                 options_.symbol_prefix);
    globals();
    output_.emit(R"CUDA(
struct pattern : policy {{)CUDA");
    constant("executor", executor_name(plan_.executor_));
    constant("bytes", plan_.ir_.options.characters == character_mode::BYTES);
    constant("ascii_word_classes", plan_.ir_.options.ascii_classes);
    constant("multiline", plan_.ir_.options.multiline);
    constant("extended_newline", plan_.ir_.options.extended_newline);
    constant("scan_input", plan_.ir_.control.scan_input && !plan_.begins_at_input_start());
    constant("require_end", plan_.ir_.control.require_end);
    bool positive_unicode_predicate = false;
    bool unicode_predicate          = false;
    for (auto& block : plan_.ir_.blocks) {
      for (auto& instruction : block.instructions) {
        auto* character_match = std::get_if<match_character>(&instruction);
        if (character_match != nullptr) {
          auto unicode_range = std::any_of(character_match->predicate.ranges.begin(),
                                           character_match->predicate.ranges.end(),
                                           [](codepoint_range range) { return range.last > 127; });
          unicode_predicate |= character_match->predicate.negated || unicode_range;
          positive_unicode_predicate |=
            !character_match->predicate.negated && unicode_range &&
            character_match->predicate.recognized != predicate_class::ANY;
        }
      }
    }
    constant("fused_unicode_decode", unicode_predicate);
    constant("aligned_prefix_seek",
             plan_.ir_.control.result != result_shape::MATCH_COUNT || !positive_unicode_predicate);
    constant("capture_slots", (plan_.ir_.capture_count + 1) * 2);
    constant("live_captures", plan_.capture_slots_.size());
    constant("external_workspace", plan_.workspace_bytes_ != 0);
    constant("prefix",
             plan_.prefix_seek_byte_ ? static_cast<std::int32_t>(*plan_.prefix_seek_byte_) : -1);
    constant("repeated_builtin", plan_.repeated_builtin_.has_value());
    emit_word();
    emit_captures();
    if (literal_) emit_literal(*literal_);
    if (plan_.string_operations_ || plan_.fixed_ascii_suffix_ || plan_.line_tail_literal_)
      emit_string();
    if (plan_.word_run_minimum_) constant("minimum", *plan_.word_run_minimum_);
    if (plan_.glushkov_) emit_glushkov(*plan_.glushkov_);
    if (plan_.deterministic_) emit_dfa(*plan_.deterministic_);
    if (plan_.thompson_) emit_thompson(*plan_.thompson_);
    emit_replacement();
    output_.emit(R"CUDA(}};
}} // namespace {})CUDA",
                 options_.symbol_prefix);
    emit_entry();
    auto states = plan_.thompson_        ? static_cast<std::uint32_t>(plan_.thompson_->nodes.size())
                  : plan_.deterministic_ ? plan_.deterministic_->state_count
                  : plan_.glushkov_      ? plan_.glushkov_->position_count
                                         : 0;
    std::uint32_t const classes = plan_.deterministic_ ? plan_.deterministic_->class_count
                                  : plan_.glushkov_    ? plan_.glushkov_->alphabet.class_count
                                                       : 0;
    return {output_.take(),
            plan_.ir_.capture_count,
            plan_.executor_,
            states,
            classes,
            plan_.exact_ascii_literal_metadata_,
            plan_.exact_literal_bytes_,
            plan_.repeated_builtin_,
            plan_.workspace_bytes_};
  }

 private:
  template <typename Value>
  void constant(std::string_view name, Value value)
  {
    output_.emit("static constexpr auto {} = {};", name, value);
  }

  template <typename Range>
  void array(std::string_view name, std::string_view type, Range const& values, bool small = true)
  {
    output_.emit(R"CUDA(
__device__ {} {} const {}[] = {{)CUDA",
                 small ? "__constant__" : "",
                 type,
                 name);
    if (std::empty(values)) output_.emit("0,");
    for (auto value : values)
      output_.emit("{}{},", static_cast<std::uint64_t>(value), type == "u64" ? "ULL" : "");
    output_.emit("}};");
  }

  std::string predicate(character_predicate const& predicate_value,
                        std::string_view code_point = "code_point")
  {
    if (predicate_value.recognized == predicate_class::ANY) {
      return predicate_value.matches_newline
               ? "true"
               : std::format("!newline<{}>({})", predicate_value.extended_newline, code_point);
    }
    std::string result = "false";
    for (auto& range : predicate_value.ranges) {
      auto first = static_cast<std::uint32_t>(range.first);
      auto last  = static_cast<std::uint32_t>(range.last);
      result += first == last
                  ? std::format(" || {} == {}U", code_point, first)
                  : std::format(" || ({} - {}U <= {}U)", code_point, first, last - first);
    }
    return std::format("{}({})", predicate_value.negated ? "!" : "", result);
  }

  void globals()
  {
    literal_ = plan_.ascii_literal_;
    if (plan_.utf8_literal_)
      literal_ = execution_plan::encode_utf8_literal(plan_.utf8_literal_->codepoints);
    if (plan_.string_operations_) literal_ = plan_.string_operations_->literal;
    if (plan_.line_tail_literal_) literal_ = *plan_.line_tail_literal_;
    if (literal_) {
      std::vector<std::uint8_t> bytes(literal_->begin(), literal_->end());
      array("literal_bytes", "u8", bytes, bytes.size() <= 32768);
      if (plan_.utf8_literal_) {
        std::vector<std::uint32_t> failure(bytes.size());
        std::size_t length = 0;
        for (std::size_t index = 1; index < bytes.size(); ++index) {
          while (length && bytes[index] != bytes[length])
            length = failure[length - 1];
          if (bytes[index] == bytes[length]) ++length;
          failure[index] = length;
        }
        array("literal_failure", "u32", failure, failure.size() * 4 <= 32768);
      }
    }
    auto* machine = plan_.glushkov_        ? &plan_.glushkov_->alphabet
                    : plan_.deterministic_ ? &*plan_.deterministic_
                                           : nullptr;
    if (machine) {
      array("byte_classes", "u16", machine->byte_classes);
      std::vector<std::uint32_t> ends;
      std::vector<std::uint32_t> classes;
      for (auto interval : machine->unicode_intervals) {
        ends.push_back(interval.last);
        classes.push_back(interval.class_id);
      }
      if (ends.size() > 8) {
        array("unicode_ends", "u32", ends, false);
        array("unicode_classes", "u32", classes, false);
      }
    }
    if (plan_.deterministic_) {
      auto& machine = *plan_.deterministic_;
      array("transitions", "u16", machine.transitions, machine.transition_address_space == 4);
      if (machine.assertion_aware) array("boundary_accepts", "u8", machine.boundary_accepts);
    }
    if (plan_.glushkov_ && plan_.glushkov_->reach_masks.size() > 8)
      array("reach_masks", "u64", plan_.glushkov_->reach_masks);
    auto unicode_word = plan_.uses_unicode_word_boundaries() ||
                        (plan_.word_run_minimum_ && !plan_.ir_.options.ascii_classes);
    if (unicode_word) {
      std::vector<std::uint32_t> firsts;
      std::vector<std::uint32_t> ends;
      for (auto& range : unicode_word_ranges) {
        firsts.push_back(range.first);
        ends.push_back(range.last);
      }
      array("word_firsts", "u32", firsts);
      array("word_ends", "u32", ends);
    }
    for (std::size_t index = 0; index < plan_.ir_.replacement.size(); ++index) {
      auto& token = plan_.ir_.replacement[index];
      if (token.type == replacement_token::kind::LITERAL && !token.literal.empty()) {
        std::vector<std::uint8_t> bytes(token.literal.begin(), token.literal.end());
        array(std::format("replacement_{}", index), "u8", bytes, bytes.size() <= 32768);
      }
    }
  }

  void emit_word()
  {
    if (!(plan_.uses_unicode_word_boundaries() ||
          (plan_.word_run_minimum_ && !plan_.ir_.options.ascii_classes)))
      return;
    output_.emit(R"CUDA(
__device__ __forceinline__ static bool is_word(u32 code_point) {{
if (code_point < 128) return ascii_word(code_point);
u32 low = 0; u32 high = {};
while (low < high) {{
  auto mid = low + (high - low) / 2;
  if (word_ends[mid] < code_point) low = mid + 1; else {{ high = mid; }}
}}
return low < {} && code_point >= word_firsts[low];
}})CUDA",
                 std::size(unicode_word_ranges),
                 std::size(unicode_word_ranges));
  }

  void emit_captures()
  {
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static u32 capture_slot(u32 index) {{ switch (index) {{)CUDA");
    for (std::size_t index = 0; index < plan_.capture_slots_.size(); ++index)
      output_.emit("case {}: return {};", index, plan_.capture_slots_[index]);
    output_.emit(R"CUDA(default: return 0; }} }}

__device__ __forceinline__ static void whole_captures(i64* spans, i64 begin, i64 end) {{)CUDA");
    for (auto capture : plan_.whole_match_captures_)
      output_.emit("spans[{}] = begin; spans[{}] = end;", capture * 2, capture * 2 + 1);
    output_.emit("}}");
  }

  void emit_literal(std::string_view literal)
  {
    constant("literal_size", literal.size());
    auto short_boolean = !plan_.utf8_literal_ &&
                         plan_.ir_.control.result == result_shape::BOOLEAN && literal.size() < 8;
    auto pivot = short_boolean || literal.size() == 1 ? 0
                 : plan_.utf8_literal_ && !plan_.utf8_literal_pivot_
                   ? -1
                   : static_cast<std::int32_t>(literal_anchor(literal));
    constant("pivot", pivot);
    constant("vector_seek", !short_boolean);
    constant("seek_threshold", plan_.utf8_literal_ ? 0 : literal.size() == 2 ? 129 : 64);
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static bool literal_guard(input input_value, i64 pos) {{)CUDA");
    if (plan_.utf8_literal_pivot_) {
      auto offset = std::min(*plan_.utf8_literal_pivot_, literal.size() - sizeof(std::uint64_t));
      std::uint64_t value = 0;
      for (std::uint32_t index = 0; index < 8; ++index)
        value |= std::uint64_t(static_cast<std::uint8_t>(literal[offset + index])) << (8 * index);
      output_.emit(
        "return load_unaligned<u64>(input_value.data + pos + {}) == {}ULL; }}", offset, value);
    } else if (pivot > 0) {
      output_.emit("return input_value.byte(pos + {}) == {}; }}",
                   pivot - 1,
                   static_cast<std::uint8_t>(literal[pivot - 1]));
    } else {
      output_.emit("return true; }}");
    }
    constant("kmp", plan_.utf8_literal_.has_value());
    constant("hybrid",
             plan_.utf8_literal_ && plan_.ir_.control.result != result_shape::BOOLEAN &&
               plan_.ir_.control.result != result_shape::MATCH_COUNT);
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static u32 literal_byte(u32 index) {{ return literal_bytes[index]; }})CUDA");
    if (plan_.utf8_literal_)
      output_.emit(
        R"CUDA(
__device__ __forceinline__ static u32 failure(u32 index) {{ return literal_failure[index]; }})CUDA");
    output_.emit(R"CUDA(
__device__ __forceinline__ static bool literal_at(input input_value, i64 pos) {{
if (pos < 0 || pos > input_value.size || input_value.size - pos < {}) return false;)CUDA",
                 literal.size());
    std::string comparisons = "true";
    for (std::size_t offset = 0; offset < literal.size();) {
      auto remaining      = literal.size() - offset;
      auto chunk          = remaining >= 8 ? 8 : remaining >= 4 ? 4U : remaining >= 2 ? 2U : 1U;
      std::uint64_t value = 0;
      for (std::uint32_t index = 0; index < chunk; ++index)
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(literal[offset + index]))
                 << (8 * index);
      // All chunks are row-bounded. String operations compare them eagerly so independent
      // loads can issue together; scanning literals retain their selective comparisons.
      comparisons +=
        std::format(" {} (load_unaligned<u{}>(input_value.data + pos + {}) == {}ULL)",
                    plan_.string_operations_.has_value() && literal.size() <= 16 ? "&" : "&&",
                    chunk * 8,
                    offset,
                    value);
      offset += chunk;
    }
    output_.emit(R"CUDA(return {};
}}

__device__ __forceinline__ static bool literal_candidate_at(input input_value, i64 pos) {{)CUDA",
                 comparisons);
    if (!plan_.utf8_literal_ && literal.size() == 2 && pivot == 1)
      output_.emit("return true; }}");
    else {
      output_.emit("return literal_at(input_value, pos); }}");
    }
  }

  void emit_string()
  {
    if (plan_.line_tail_literal_) {
      output_.emit(R"CUDA(
__device__ __forceinline__ static bool string_find(input input_value,
                                                   i64 search,
                                                   i64& begin,
                                                   i64& end)
{{
  return line_tail_find<pattern>(input_value, search, begin, end);
}}

__device__ __forceinline__ static bool string_boolean(input input_value)
{{
  i64 begin, end;
  return string_find(input_value, 0, begin, end);
}})CUDA");
      return;
    }
    std::string_view kind;
    if (plan_.string_operations_) switch (plan_.string_operations_->kind) {
        case execution_plan::string_operation_kind::BEGINS_WITH:
          kind = "string_operation_kind::BEGINS_WITH";
          break;
        case execution_plan::string_operation_kind::ENDS_WITH:
          kind = "string_operation_kind::ENDS_WITH";
          break;
        case execution_plan::string_operation_kind::ENDS_LINE:
          kind = "string_operation_kind::ENDS_LINE";
          break;
        case execution_plan::string_operation_kind::EQUALS:
          kind = "string_operation_kind::EQUALS";
          break;
        case execution_plan::string_operation_kind::EQUALS_LINE:
          kind = "string_operation_kind::EQUALS_LINE";
          break;
      }
    else {
      auto& suffix = *plan_.fixed_ascii_suffix_;
      kind         = suffix.begins     ? suffix.line_end ? "string_operation_kind::EQUALS_LINE"
                                                         : "string_operation_kind::EQUALS"
                     : suffix.line_end ? "string_operation_kind::ENDS_LINE"
                                       : "string_operation_kind::ENDS_WITH";
      constant("literal_size", suffix.predicates.size());
      output_.emit(R"CUDA(
__device__ __forceinline__ static bool literal_at(input input_value, i64 pos) {{
if (pos < 0 || pos > input_value.size || input_value.size - pos < {}) return false;)CUDA",
                   suffix.predicates.size());
      // Folded first classes use eager verification; other classes retain
      // the selective byte guard before the branchless remainder.
      bool case_folded_first = false;
      for (char32_t uppercase = U'A'; uppercase <= U'Z'; ++uppercase)
        case_folded_first |= suffix.predicates.front().matches(uppercase) &&
                             suffix.predicates.front().matches(uppercase + 32);
      if (!case_folded_first)
        output_.emit("if (!{}) return false;",
                     predicate(suffix.predicates.front(), "input_value.byte(pos)"));
      output_.emit("bool matched = true;");
      for (std::size_t index = case_folded_first ? 0 : 1; index < suffix.predicates.size(); ++index)
        output_.emit(
          "matched &= {};",
          predicate(suffix.predicates[index], std::format("input_value.byte(pos + {})", index)));
      output_.emit("return matched; }}");
    }
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static bool string_boolean(input input_value)
{{
  return regex_ir::device::string_boolean<pattern, {}>(input_value);
}})CUDA",
      kind);
  }

  void emit_classifier(deterministic_machine const& machine)
  {
    std::vector<std::size_t> frequency(machine.class_count);
    for (auto cls : machine.byte_classes)
      ++frequency[cls];
    auto default_class = std::max_element(frequency.begin(), frequency.end()) - frequency.begin();
    std::vector<deterministic_interval> intervals;
    for (std::uint32_t byte = 0; byte < 256;) {
      auto cls = machine.byte_classes[byte];
      if (cls == default_class) {
        ++byte;
        continue;
      }
      auto first = byte;
      while (byte + 1 < 256 && machine.byte_classes[byte + 1] == cls)
        ++byte;
      intervals.push_back({first, byte++, cls});
    }
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static u32 classify(u32 code_point) {{
if (code_point < 256) {{)CUDA");
    if (!intervals.empty() && intervals.size() <= 2) {
      output_.emit("u32 result = {};", default_class);
      for (auto interval : intervals)
        output_.emit("result = code_point - {}U <= {}U ? {} : result;",
                     interval.first,
                     interval.last - interval.first,
                     interval.class_id);
      output_.emit("return result;");
    } else {
      output_.emit("return byte_classes[code_point];");
    }
    output_.emit("}}");
    if (machine.unicode_intervals.empty())
      output_.emit("return 0;");
    else if (machine.unicode_intervals.size() <= 8) {
      output_.emit("u32 result = {};", machine.unicode_intervals.back().class_id);
      for (std::size_t reverse = machine.unicode_intervals.size() - 1; reverse > 0; --reverse) {
        auto interval = machine.unicode_intervals[reverse - 1];
        output_.emit("result = code_point <= {}U ? {} : result;", interval.last, interval.class_id);
      }
      output_.emit("return result;");
    } else {
      output_.emit(R"CUDA(u32 low = 0; u32 high = {};
while (low < high) {{
  auto mid = low + (high - low) / 2;
  if (unicode_ends[mid] < code_point) low = mid + 1; else {{ high = mid; }}
}}
return low < {} ? unicode_classes[low] : 0;)CUDA",
                   machine.unicode_intervals.size(),
                   machine.unicode_intervals.size());
    }
    output_.emit("}}");
  }

  void emit_dfa(deterministic_machine const& machine)
  {
    constexpr auto transition_accept_mask = 0x8000U;
    auto accepts_any_first                = machine.class_count != 0 && !machine.assertion_aware &&
                             !plan_.ir_.control.require_end &&
                             !machine.accept_assertion.has_value();
    if (accepts_any_first) {
      auto initial = machine.initial_state & machine.state_mask;
      for (std::uint32_t character_class = 0; character_class < machine.class_count;
           ++character_class)
        accepts_any_first &= (machine.transitions[initial * machine.class_count + character_class] &
                              transition_accept_mask) != 0;
    }
    constant("accepts_any_first", accepts_any_first);
    emit_classifier(machine);
    constant("class_count", machine.class_count);
    constant("initial_state", machine.initial_state);
    constant("state_mask", machine.state_mask);
    constant("dead_state", machine.dead_state);
    constant("restart_state", machine.restart_state);
    constant("assertion_aware", machine.assertion_aware);
    constant("assertion_mask", machine.assertion_mask);
    constant("boundary_classes",
             machine.boundary_class_count == 0 ? 1 : machine.boundary_class_count);
    constant("tagged", plan_.executor_ == executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC);
    std::int32_t filter_prefix = -1;
    if (machine.start_byte_filter && machine.start_byte_bitmap[2] == 0 &&
        machine.start_byte_bitmap[3] == 0 &&
        std::popcount(machine.start_byte_bitmap[0]) + std::popcount(machine.start_byte_bitmap[1]) ==
          1)
      filter_prefix = machine.start_byte_bitmap[0] != 0
                        ? std::countr_zero(machine.start_byte_bitmap[0])
                        : 64 + std::countr_zero(machine.start_byte_bitmap[1]);
    // A sparse ASCII bitmap alone does not exclude Unicode alternatives.
    if (filter_prefix >= 0 && plan_.ir_.options.characters != character_mode::BYTES) {
      auto boundary_count = machine.boundary_class_count == 0 ? 1U : machine.boundary_class_count;
      for (auto interval : machine.unicode_intervals) {
        if (interval.last < 256) continue;
        for (std::uint32_t context = 0; context < boundary_count; ++context) {
          auto index = ((machine.initial_state & machine.state_mask) * boundary_count + context) *
                         machine.class_count +
                       interval.class_id;
          auto transition = machine.transitions[index];
          if ((transition & machine.state_mask) != machine.dead_state ||
              (transition & transition_accept_mask) != 0)
            filter_prefix = -1;
        }
      }
    }
    // Initial and dead states must be equivalent in every assertion context.
    // Other characters must return to that equivalence class without accepting.
    auto boolean_seek_byte =
      plan_.prefix_seek_byte_ ? static_cast<std::int32_t>(*plan_.prefix_seek_byte_) : filter_prefix;
    bool boolean_prefix_skip =
      machine.assertion_aware && boolean_seek_byte >= 0 && boolean_seek_byte < 128;
    if (boolean_prefix_skip) {
      auto boundary_count = machine.boundary_class_count == 0 ? 1U : machine.boundary_class_count;
      auto initial        = machine.initial_state & machine.state_mask;
      boolean_prefix_skip &= machine.dead_state < machine.boundary_accepts.size() / boundary_count;
      for (std::uint32_t context = 0; context < boundary_count && boolean_prefix_skip; ++context) {
        boolean_prefix_skip &=
          machine.boundary_accepts[initial * boundary_count + context] == 0 &&
          machine.boundary_accepts[machine.dead_state * boundary_count + context] == 0;
        for (std::uint32_t character_class = 0; character_class < machine.class_count;
             ++character_class) {
          boolean_prefix_skip &=
            machine.transitions[(initial * boundary_count + context) * machine.class_count +
                                character_class] ==
            machine
              .transitions[(machine.dead_state * boundary_count + context) * machine.class_count +
                           character_class];
        }
        for (std::uint32_t code_point = 0; code_point < 256; ++code_point) {
          if (code_point == static_cast<std::uint32_t>(boolean_seek_byte)) continue;
          auto transition =
            machine.transitions[(initial * boundary_count + context) * machine.class_count +
                                machine.byte_classes[code_point]];
          boolean_prefix_skip &= ((transition & machine.state_mask) == initial ||
                                  (transition & machine.state_mask) == machine.dead_state) &&
                                 (transition & transition_accept_mask) == 0;
        }
        for (auto interval : machine.unicode_intervals) {
          if (interval.last < 256) continue;
          auto transition =
            machine.transitions[(initial * boundary_count + context) * machine.class_count +
                                interval.class_id];
          boolean_prefix_skip &= ((transition & machine.state_mask) == initial ||
                                  (transition & machine.state_mask) == machine.dead_state) &&
                                 (transition & transition_accept_mask) == 0;
        }
      }
    }
    constant("boolean_prefix_skip", boolean_prefix_skip);
    constant("boolean_seek_byte", boolean_seek_byte);
    constant("start_filter_prefix", filter_prefix);
    constant("accept_assertion",
             machine.accept_assertion ? assertion_name(*machine.accept_assertion)
                                      : "regex_ir::device::assertion_kind::NONE");
    output_.emit(R"CUDA(
__device__ __forceinline__ static u32 transition_index(u32 index) {{ return transitions[index]; }}

__device__ __forceinline__ static u32 transition(u32 state, u32 cls, u32 context)
{{
  return transitions[(state * boundary_classes + context) * class_count + cls];
}})CUDA");
    if (machine.assertion_aware)
      output_.emit(
        R"CUDA(
__device__ __forceinline__ static bool boundary_accept(u32 state, u32 context)
{{
  return boundary_accepts[state * boundary_classes + context] != 0;
}})CUDA");
    output_.emit(R"CUDA(
__device__ __forceinline__ static bool start_byte(u32 code_point) {{)CUDA");
    if (machine.start_byte_filter) {
      output_.emit("if (code_point >= 128) return true;");
      for (std::uint32_t word = 0; word < 2; ++word)
        output_.emit(
          "if (code_point / 64 == {}) return ({}ULL & (1ULL << (code_point % 64))) != 0;",
          word,
          machine.start_byte_bitmap[word]);
      output_.emit("return true;");
    } else {
      output_.emit("return true;");
    }
    output_.emit("}}");
    auto captures = [&](std::string_view name, auto& actions) {
      output_.emit(
        R"CUDA(
__device__ __forceinline__ static void {}(u32 index, i64 pos, i64* spans) {{ switch (index) {{)CUDA",
        name);
      std::map<std::vector<std::pair<std::uint32_t, bool>>, std::vector<std::size_t>> groups;
      for (std::size_t index = 0; index < actions.size(); ++index) {
        std::vector<std::pair<std::uint32_t, bool>> key;
        for (auto action : actions[index])
          if (std::find(plan_.capture_slots_.begin(), plan_.capture_slots_.end(), action.slot) !=
              plan_.capture_slots_.end())
            key.emplace_back(action.slot, static_cast<bool>(action.reset_end));
        if (!key.empty()) groups[key].push_back(index);
      }
      for (auto& [actions, indices] : groups) {
        for (auto index : indices)
          output_.emit("case {}:", index);
        for (auto [slot, reset] : actions) {
          output_.emit("spans[{}] = pos;", slot);
          if (reset) output_.emit("spans[{}] = -1;", slot + 1);
        }
        output_.emit("break;");
      }
      output_.emit("default: break; }} }}");
    };
    if (plan_.executor_ == executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC) {
      captures("capture_transition", machine.transition_capture_actions);
      captures("capture_accept", machine.accept_capture_actions);
    }
  }

  void emit_glushkov(glushkov_machine const& machine)
  {
    auto accepts_any_first =
      !plan_.ir_.control.require_end &&
      std::all_of(machine.reach_masks.begin(), machine.reach_masks.end(), [&](auto reach_mask) {
        return (reach_mask & machine.first_set & machine.accept_mask) != 0;
      });
    constant("accepts_any_first", accepts_any_first);
    emit_classifier(machine.alphabet);
    output_.emit(
      R"CUDA(static constexpr u64 first_set = {}ULL;
static constexpr u64 accept_mask = {}ULL;)CUDA",
      machine.first_set,
      machine.accept_mask);
    constant(
      "fixed_match_bytes",
      machine.fixed_match_bytes ? static_cast<std::int64_t>(*machine.fixed_match_bytes) : -1);
    constant("glushkov_prefix",
             machine.start_byte ? static_cast<std::int32_t>(*machine.start_byte) : -1);
    if (plan_.repeated_builtin_) constant("repeat_count", *machine.repeated_predicate_count);
    output_.emit(R"CUDA(
__device__ __forceinline__ static u64 reach(u32 cls) {{)CUDA");
    if (machine.reach_masks.size() > 8)
      output_.emit("return reach_masks[cls];");
    else {
      output_.emit("u64 result = {}ULL;", machine.reach_masks.back());
      for (std::size_t index = 0; index + 1 < machine.reach_masks.size(); ++index)
        output_.emit("result = cls == {} ? {}ULL : result;", index, machine.reach_masks[index]);
      output_.emit("return result;");
    }
    output_.emit(
      R"CUDA(}}

__device__ __forceinline__ static u64 follow(u64 state) {{
u64 result = 0;)CUDA");
    for (auto shift : machine.shifts)
      output_.emit("result |= (state & {}ULL) << {};", shift.sources, shift.amount);
    for (std::uint32_t index = 0; index < 64; ++index)
      if ((machine.exception_mask >> index) & 1)
        output_.emit("result |= (state & {}ULL) != 0 ? {}ULL : 0ULL;",
                     std::uint64_t{1} << index,
                     machine.exception_successors[index]);
    output_.emit("return result; }}");
  }

  void emit_thompson(deterministic_nfa_graph const& graph);

  void emit_replacement()
  {
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static i64 replacement(input input_value,
                                                  i64 const* spans,
                                                  char* output,
                                                  i64 cursor)
{{)CUDA");
    for (std::size_t index = 0; index < plan_.ir_.replacement.size(); ++index) {
      auto& token = plan_.ir_.replacement[index];
      if (token.type == replacement_token::kind::LITERAL) {
        if (!token.literal.empty())
          output_.emit(
            R"CUDA(cursor = append(reinterpret_cast<char const*>(replacement_{}), 0, {}, output, cursor);)CUDA",
            index,
            token.literal.size());
      } else if (token.capture_index == 0 || plan_.is_whole_match_capture(token.capture_index)) {
        output_.emit("cursor = append(input_value.data, spans[0], spans[1], output, cursor);");
      } else {
        auto slot = token.capture_index * 2;
        output_.emit(
          R"CUDA(if (spans[{}] >= 0 && spans[{}] >= spans[{}])
  cursor = append(input_value.data, spans[{}], spans[{}], output, cursor);)CUDA",
          slot,
          slot + 1,
          slot,
          slot,
          slot + 1);
      }
    }
    output_.emit("return cursor; }}");
  }

  void emit_entry()
  {
    auto shape = plan_.anchored_boolean_result_.value_or(plan_.ir_.control.result);
    auto abi   = shape == result_shape::BOOLEAN ? matcher_abi::BOOLEAN
                 : shape == result_shape::MATCH_COUNT
                   ? plan_.repeated_builtin_ ? matcher_abi::BUILTIN_COUNT : matcher_abi::COUNT
                 : shape == result_shape::MATCH_SPAN
                   ? plan_.ir_.selected_operation.kind == operation_kind::FIND_ALL
                       ? matcher_abi::CAPTURES
                       : matcher_abi::FIND
                 : shape == result_shape::CAPTURES    ? matcher_abi::CAPTURES
                 : shape == result_shape::REPLACEMENT ? matcher_abi::REPLACE
                                                      : matcher_abi::SPLIT;
    output_.emit(
      R"CUDA({} {{
using Pattern = {}::pattern;
using namespace regex_ir::device;
input input_value{{data, size}};
auto* scratch = {};)CUDA",
      matcher_signature(abi, options_.execute_function, plan_.workspace_bytes_ != 0),
      options_.symbol_prefix,
      plan_.workspace_bytes_ ? "reinterpret_cast<i64*>(workspace)" : "static_cast<i64*>(nullptr)");
    if (plan_.anchored_boolean_result_) {
      output_.emit("bool matched = boolean<Pattern>(input_value, scratch);");
      if (shape == result_shape::MATCH_SPAN)
        output_.emit("if (matched) spans[0] = 0; return matched;");
      else {
        output_.emit("return static_cast<i64>(matched);");
      }
    } else if (shape == result_shape::BOOLEAN) {
      output_.emit("return boolean<Pattern>(input_value, scratch);");
    } else if (shape == result_shape::MATCH_COUNT) {
      output_.emit("return count<Pattern>(input_value, scratch{});",
                   plan_.repeated_builtin_ ? ", flags" : "");
    } else if (shape == result_shape::CAPTURES) {
      output_.emit("return captures<Pattern>(input_value, search, spans, scratch);");
    } else if (shape == result_shape::MATCH_SPAN) {
      output_.emit("return find<Pattern>(input_value, {}, spans[0], spans[1], nullptr, scratch);",
                   abi == matcher_abi::CAPTURES ? "search" : "0");
    } else if (shape == result_shape::REPLACEMENT) {
      output_.emit("return replace<Pattern>(input_value, output, scratch);");
    } else {
      output_.emit("return split<Pattern>(input_value, spans, scratch);");
    }
    output_.emit("}}");
    if (shape == result_shape::SPLIT_FIELDS) {
      output_.blank();
      output_.emit(R"CUDA({}
{{
  using namespace regex_ir::device;
  return split<{}::pattern>(
    {{data, size}}, spans, {}, limit, capacity, overflow);
}})CUDA",
                   matcher_signature(matcher_abi::LIMITED_SPLIT,
                                     "regex_ir_split_execute_limited",
                                     plan_.workspace_bytes_ != 0),
                   options_.symbol_prefix,
                   plan_.workspace_bytes_ ? "reinterpret_cast<i64*>(workspace)" : "nullptr");
    }
  }

  execution_plan const& plan_;
  cuda_codegen_options options_;
  source_buffer output_;
  std::optional<std::string> literal_;
};

void cuda_source_emitter::emit_thompson(deterministic_nfa_graph const& graph)
{
  constant("entry", graph.entry);
  constant("frontier_words", plan_.frontier_words_);
  constant("closure_records", plan_.closure_records_);
  constant("storage_words", plan_.storage_words_);
  constant("bitset_words", (graph.nodes.size() + 63) / 64);
  // A slot is live before a state if some accepting continuation reads it before
  // the state writes it. BEGIN also clears its matching END slot.
  auto capture_masks = std::vector<std::uint64_t>(graph.nodes.size());
  if (plan_.capture_slots_.size() <= 64) {
    auto all_captures = plan_.capture_slots_.size() == 64
                          ? ~std::uint64_t{0}
                          : (std::uint64_t{1} << plan_.capture_slots_.size()) - 1;
    bool changed;
    do {
      changed = false;
      for (std::size_t reverse = graph.nodes.size(); reverse > 0; --reverse) {
        auto state = reverse - 1;
        auto& node = graph.nodes[state];
        auto live  = node.accepts ? all_captures : std::uint64_t{0};
        for (auto target : node.targets)
          live |= capture_masks[target];
        if (node.capture) {
          auto slot =
            node.capture->capture_index * 2 + (node.capture->action == capture_action::END);
          for (std::size_t capture_index = 0; capture_index < plan_.capture_slots_.size();
               ++capture_index) {
            auto capture_slot = plan_.capture_slots_[capture_index];
            if (capture_slot == slot ||
                (node.capture->action == capture_action::BEGIN && capture_slot == slot + 1))
              live &= ~(std::uint64_t{1} << capture_index);
          }
        }
        changed |= capture_masks[state] != live;
        capture_masks[state] = live;
      }
    } while (changed);
  }
  output_.emit(R"CUDA(
__device__ static constexpr u64 capture_mask(u32 state) {{ switch (state) {{)CUDA");
  std::map<std::uint64_t, std::vector<std::size_t>> capture_groups;
  for (std::size_t state = 0; state < capture_masks.size(); ++state)
    capture_groups[capture_masks[state]].push_back(state);
  for (auto& [mask, states] : capture_groups) {
    for (auto state : states)
      output_.emit("case {}:", state);
    output_.emit("return {}ULL;", mask);
  }
  output_.emit(R"CUDA(default: return 0; }} }}

__device__ __forceinline__ static bool candidate_byte(u32 code_point) {{)CUDA");
  if (plan_.candidate_seeker_) {
    for (std::uint32_t index = 0; index < 4; ++index)
      output_.emit("if (code_point / 64 == {}) return ({}ULL & (1ULL << (code_point % 64))) != 0;",
                   index,
                   plan_.candidate_bitmap_[index]);
    output_.emit("return false;");
  } else if (plan_.prefix_seek_byte_) {
    output_.emit("return code_point == {};", *plan_.prefix_seek_byte_);
  } else {
    output_.emit("return true;");
  }
  output_.emit(R"CUDA(}}
template <typename Work>
__device__ __forceinline__ static i32 dispatch(
  u32 state, input input_value, i64 pos, u32 code_point, Work& work)
{{
  switch (state) {{)CUDA");
  for (std::size_t id = 0; id < graph.nodes.size(); ++id) {
    auto& node = graph.nodes[id];
    output_.emit("case {}:", id);
    if (node.capture) {
      auto slot  = node.capture->capture_index * 2 + (node.capture->action == capture_action::END);
      auto write = [&](std::size_t slot, std::string_view value) {
        auto found = std::find(plan_.capture_slots_.begin(), plan_.capture_slots_.end(), slot);
        if (found != plan_.capture_slots_.end())
          output_.emit("work.active[{}] = {};", found - plan_.capture_slots_.begin() + 1, value);
      };
      write(slot, "pos");
      if (node.capture->action == capture_action::BEGIN) write(slot + 1, "-1");
    }
    if (node.assertion)
      output_.emit(
        R"CUDA(if (!assertion<pattern>(input_value, pos, {})) return static_cast<i32>(dispatch_status::EXHAUSTED);)CUDA",
        assertion_name(*node.assertion));
    if (node.accepts) {
      output_.emit(
        R"CUDA(return {} ? static_cast<i32>(dispatch_status::ACCEPTED)
                       : static_cast<i32>(dispatch_status::EXHAUSTED);)CUDA",
        plan_.ir_.control.require_end ? "pos == input_value.size" : "true");
      continue;
    }
    if (node.consumes)
      output_.emit(
        "if (pos >= input_value.size || !{}) return static_cast<i32>(dispatch_status::EXHAUSTED);",
        predicate(node.predicate));
    for (std::size_t index = 0; index < node.targets.size(); ++index) {
      auto target = node.targets[node.consumes ? index : node.targets.size() - index - 1];
      if (!node.consumes && index + 1 == node.targets.size())
        output_.emit("return {};", target);
      else {
        output_.emit("work.{}({});", node.consumes ? "enqueue" : "push", target);
      }
    }
    if (node.consumes || node.targets.empty())
      output_.emit("return static_cast<i32>(dispatch_status::EXHAUSTED);");
  }
  output_.emit("default: return static_cast<i32>(dispatch_status::EXHAUSTED); }} }}");

  struct run_loop {
    std::size_t state;
    std::size_t consume;
    std::uint64_t low;
    std::uint64_t high;
  };

  std::vector<run_loop> loops;
  for (std::size_t start = 0; start < graph.nodes.size(); ++start) {
    auto& branch = graph.nodes[start];
    if (branch.consumes || branch.capture || branch.assertion || branch.accepts ||
        branch.targets.size() != 2)
      continue;
    auto consume = branch.targets[0];
    auto& loop   = graph.nodes[consume];
    if (!loop.consumes || loop.capture || loop.assertion || loop.accepts ||
        loop.targets.size() != 1 || loop.targets[0] != start)
      continue;
    auto disjoint = [&](character_predicate const& other) {
      std::vector<char32_t> points{0, 10, 11, 13, 14, 133, 134, 8232, 8233, 8234};
      for (auto* predicate_value :
           std::array<character_predicate const*, 2>{&loop.predicate, &other})
        for (auto range : predicate_value->ranges) {
          points.push_back(range.first);
          if (range.last < 0x10ffff) points.push_back(range.last + 1);
        }
      return std::none_of(points.begin(), points.end(), [&](auto code_point) {
        return loop.predicate.matches(code_point) && other.matches(code_point);
      });
    };
    std::vector<std::uint8_t> visited(graph.nodes.size());
    std::size_t proof_steps = 0;
    auto safe               = [&](auto&& self, std::size_t state) -> bool {
      if (++proof_steps > 256 || visited[state] == 1) return false;
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
    std::uint64_t low  = 0;
    std::uint64_t high = 0;
    for (char32_t code_point = 0; code_point < 128; ++code_point)
      if (loop.predicate.matches(code_point)) {
        if (code_point < 64)
          low |= std::uint64_t{1} << code_point;
        else {
          high |= std::uint64_t{1} << (code_point - 64);
        }
      }
    loops.push_back({start, consume, low, high});
  }
  // A linear capture path with proven non-backtracking greedy runs does not
  // need a frontier or capture records. Any other graph keeps the general executor.
  auto simple_path    = std::vector<std::size_t>{};
  auto visited_path   = std::vector<bool>(graph.nodes.size());
  auto simple_state   = static_cast<std::size_t>(graph.entry);
  bool simple_capture = plan_.ir_.control.result == result_shape::CAPTURES;
  while (simple_capture) {
    if (visited_path[simple_state]) {
      simple_capture = false;
      break;
    }
    visited_path[simple_state] = true;
    simple_path.push_back(simple_state);
    auto& node = graph.nodes[simple_state];
    if (node.accepts) break;
    auto loop = std::find_if(
      loops.begin(), loops.end(), [&](auto& candidate) { return candidate.state == simple_state; });
    if (loop != loops.end()) {
      simple_state = node.targets[1];
    } else if (node.targets.size() == 1) {
      simple_state = node.targets[0];
    } else {
      simple_capture = false;
    }
  }
  constant("simple_capture", simple_capture);
  if (simple_capture) {
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static bool simple_capture_find(
  input input_value, i64 search, i64& begin, i64& end, i64* captures)
{{
  for (auto start = search; start <= input_value.size;) {{
    auto attempt = [&]() {{
      auto pos = start;)CUDA");
    for (auto state : simple_path) {
      auto& node = graph.nodes[state];
      if (node.capture) {
        auto slot = node.capture->capture_index * 2 + (node.capture->action == capture_action::END);
        if (std::find(plan_.capture_slots_.begin(), plan_.capture_slots_.end(), slot) !=
            plan_.capture_slots_.end())
          output_.emit("if (captures != nullptr) captures[{}] = pos;", slot);
      }
      if (node.assertion)
        output_.emit("if (!assertion<pattern>(input_value, pos, {})) return false;",
                     assertion_name(*node.assertion));
      auto loop = std::find_if(
        loops.begin(), loops.end(), [&](auto& candidate) { return candidate.state == state; });
      if (loop != loops.end()) {
        output_.emit(
          R"CUDA(pos = scan_ascii_run<pattern, {}>(input_value, pos);
while (pos < input_value.size) {{
  auto character  = input_value.read<bytes, fused_unicode_decode>(pos);
  auto code_point = character.code_point;
  if (!{}) break;
  pos += character.width;
}})CUDA",
          state,
          predicate(graph.nodes[loop->consume].predicate));
      } else if (node.consumes) {
        output_.emit(
          R"CUDA({{
  if (pos >= input_value.size) return false;
  auto character  = input_value.read<bytes, fused_unicode_decode>(pos);
  auto code_point = character.code_point;
  if (!{}) return false;
  pos += character.width;
}})CUDA",
          predicate(node.predicate));
      }
      if (node.accepts)
        output_.emit(
          R"CUDA(if (require_end && pos != input_value.size) return false; begin = start; end = pos; return true;)CUDA");
    }
    output_.emit(
      R"CUDA(}};
if (start < input_value.size && !candidate_byte(input_value.byte(start))) {{
  if constexpr (prefix >= 0)
    start = input_value.seek<aligned_prefix_seek, !aligned_prefix_seek>(start + 1, prefix);
  else {{
    ++start;
  }}
  continue;
}}
if (attempt()) return true;
if (!scan_input || start == input_value.size) return false;
if constexpr (prefix >= 0)
  start = input_value.seek<aligned_prefix_seek, !aligned_prefix_seek>(start + 1, prefix);
else {{
  start = input_value.advance<bytes>(start);
}}
}}
return false;
}})CUDA");
  }
  bool multi_run = false;
  for (std::size_t index = 0; index < loops.size(); ++index)
    for (std::size_t other_index = index + 1; other_index < loops.size(); ++other_index)
      multi_run |= (loops[index].low & loops[other_index].low) != 0 ||
                   (loops[index].high & loops[other_index].high) != 0;
  constant("multi_run", multi_run);
  output_.emit(
    R"CUDA(
__device__ static constexpr ascii_run_filter run_filter(u32 state) {{ switch (state) {{)CUDA");
  for (auto loop : loops) {
    auto excluded_low   = ~loop.low;
    auto excluded_high  = ~loop.high;
    auto excluded_count = std::popcount(excluded_low) + std::popcount(excluded_high);
    if (excluded_count <= 1) {
      auto excluded = excluded_count == 0 ? 0
                      : excluded_low != 0 ? std::countr_zero(excluded_low)
                                          : 64 + std::countr_zero(excluded_high);
      output_.emit("case {}: return {{ascii_run_kind::{}, {}}};",
                   loop.state,
                   excluded_count == 0 ? "ALL_ASCII" : "EXCEPT_BYTE",
                   excluded);
    }
  }
  output_.emit(R"CUDA(default: return {{ascii_run_kind::GENERAL, 0}}; }} }}

__device__ __forceinline__ static bool run_predicate(u32 state, u32 code_point) {{ switch (state) {{)CUDA");
  for (auto loop : loops)
    output_.emit("case {}: return {};", loop.state, predicate(graph.nodes[loop.consume].predicate));
  output_.emit(R"CUDA(default: return false;
  }}
  }}

  __device__ __forceinline__ static bool run_masks(u32 state, u64& low, u64& high)
  {{
    switch (state) {{)CUDA");
  for (auto loop : loops)
    output_.emit(
      "case {}: low = {}ULL; high = {}ULL; return true;", loop.state, loop.low, loop.high);
  output_.emit(R"CUDA(default: return false;
  }}
  }}

  template <typename StateWord>
  __device__ __forceinline__ static i64 scan_run(input input_value,
                                                 i64 pos,
                                                 StateWord const* current,
                                                 i64 count)
  {{
    if (count == 1) {{
      switch (current[0]) {{)CUDA");
  for (auto loop : loops)
    output_.emit(
      "case {}: return scan_ascii_run<pattern, {}>(input_value, pos);", loop.state, loop.state);
  output_.emit(
    "default: return pos; }} }} return scan_runs<pattern>(input_value, pos, current, count); }}");
  if (plan_.mandatory_ascii_literal_) {
    auto& literal = *plan_.mandatory_ascii_literal_;
    constant("mandatory_size", literal.size());
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static bool mandatory_at(input input_value, i64 pos)
{{
  return input_value.byte(pos) == {}U)CUDA",
      static_cast<std::uint8_t>(literal.front()));
    for (std::size_t offset = 0; offset < literal.size();) {
      auto remaining            = literal.size() - offset;
      std::uint32_t const width = remaining >= 4 ? 4U : remaining >= 2 ? 2U : 1U;
      std::uint64_t value       = 0;
      for (std::uint32_t index = 0; index < width; ++index)
        value |= std::uint64_t(static_cast<std::uint8_t>(literal[offset + index])) << (8 * index);
      output_.emit(
        " && load_unaligned<u{}>(input_value.data + pos + {}) == {}ULL", width * 8, offset, value);
      offset += width;
    }
    output_.emit(R"CUDA(;
}}

__device__ __forceinline__ static bool mandatory_present(input input_value)
{{
  return mandatory_literal<pattern>(input_value);
}})CUDA");
  } else {
    output_.emit(
      R"CUDA(
__device__ __forceinline__ static bool mandatory_present(input) {{ return true; }})CUDA");
  }
}

compile_result render_cuda(instruction_ir const& ir, cuda_codegen_options const& options)
{
  if (ir.control.result == result_shape::BOOLEAN && ir.blocks.size() >= 80 &&
      ir.entry < ir.blocks.size() && ir.blocks[ir.entry].instructions.empty() &&
      ir.blocks[ir.entry].successors.size() >= 2) {
    std::string source;
    std::vector<std::pair<std::string, bool>> functions;
    std::size_t workspace = 0;
    for (auto edge : ir.blocks[ir.entry].successors) {
      auto branch         = ir;
      branch.entry        = edge.target;
      auto branch_options = options;
      auto suffix         = std::format("_alternative_{}", functions.size());
      branch_options.symbol_prefix += suffix;
      branch_options.execute_function += suffix;
      auto result = render_cuda(optimize(std::move(branch), {}), branch_options);
      source += result.cuda_source;
      functions.emplace_back(branch_options.execute_function, result.workspace_bytes != 0);
      workspace = std::max(workspace, result.workspace_bytes);
    }
    source +=
      matcher_signature(matcher_abi::BOOLEAN, options.execute_function, workspace != 0) + R"CUDA( {
)CUDA";
    for (auto& [function, external] : functions)
      source += std::format(
        R"CUDA(if ({}({}data, size)) return true;
)CUDA",
        function,
        external ? "workspace, " : "");
    source += R"CUDA(return false;
}
)CUDA";
    return {std::move(source),
            ir.capture_count,
            executor_kind::BOOLEAN_ALTERNATION,
            0,
            0,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            workspace};
  }
  return cuda_source_emitter(execution_plan(ir), options).render();
}

}  // namespace
}  // namespace regex_ir::detail

namespace regex_ir {

compile_result generate_cuda_source(instruction_ir const& ir, cuda_codegen_options const& options)
{
  return detail::render_cuda(ir, options);
}

}  // namespace regex_ir
