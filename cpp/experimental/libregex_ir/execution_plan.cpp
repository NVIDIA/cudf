/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "execution_plan.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace regex_ir::detail {
namespace {

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

  enum class visit_state : std::uint8_t { UNVISITED = 0, VISITING = 1, COMPLETE = 2 };
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
  enum class visit_state : std::uint8_t { UNVISITED = 0, VISITING = 1, COMPLETE = 2 };
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

}  // namespace

execution_plan::execution_plan(instruction_ir ir) : ir_(std::move(ir))
{
  exact_ascii_literal_metadata_ = exact_ascii_literal();
  auto result                   = ir_.control.result;
  auto adapt_count              = result == result_shape::MATCH_COUNT;
  auto adapt_find               = result == result_shape::MATCH_SPAN &&
                    ir_.selected_operation.kind == operation_kind::FIND &&
                    !ir_.options.find_match_end_observable;
  if (begins_at_input_start() && (adapt_count || adapt_find)) {
    anchored_boolean_result_ = result;
    ir_.control.result       = result_shape::BOOLEAN;
  }
  select();
}

std::optional<execution_plan::fixed_ascii_suffix> execution_plan::fixed_ascii_suffix_plan() const
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

std::optional<execution_plan::string_operation> execution_plan::string_operation_plan() const
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

std::optional<std::string> execution_plan::line_tail_literal() const
{
  auto supported_result = ir_.control.result == result_shape::BOOLEAN ||
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

std::optional<std::uint32_t> execution_plan::word_run_minimum() const
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

std::string execution_plan::encode_utf8_literal(std::u32string_view codepoints)
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

std::optional<std::size_t> execution_plan::utf8_literal_pivot(utf8_literal const& literal)
{
  auto bytes = encode_utf8_literal(literal.codepoints);
  if (bytes.size() < sizeof(std::uint64_t)) return std::nullopt;
  return literal_anchor(bytes);
}

std::optional<execution_plan::utf8_literal> execution_plan::exact_literal() const
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

std::optional<std::string> execution_plan::exact_ascii_literal() const
{
  auto literal = exact_literal();
  if (!literal.has_value() || std::any_of(literal->codepoints.begin(),
                                          literal->codepoints.end(),
                                          [](auto codepoint) { return codepoint > 0x7fU; })) {
    return std::nullopt;
  }
  return encode_utf8_literal(literal->codepoints);
}

std::optional<execution_plan::utf8_literal> execution_plan::exact_utf8_literal() const
{
  if (ir_.options.characters == character_mode::BYTES) return std::nullopt;
  auto literal = exact_literal();
  return literal.has_value() && std::any_of(literal->codepoints.begin(),
                                            literal->codepoints.end(),
                                            [](auto value) { return value > 0x7fU; })
           ? std::move(literal)
           : std::nullopt;
}

std::optional<std::uint8_t> execution_plan::required_ascii_prefix() const
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

std::optional<std::string> execution_plan::mandatory_ascii_literal() const
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

bool execution_plan::begins_at_input_start() const
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

std::vector<std::size_t> execution_plan::live_capture_slots() const
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

std::vector<std::uint32_t> execution_plan::whole_match_captures() const
{
  if (ir_.control.result != result_shape::CAPTURES) return {};
  auto result = std::vector<std::uint32_t>{};
  for (std::uint32_t capture = 1; capture <= ir_.capture_count; ++capture) {
    if (is_whole_match_capture(capture)) result.push_back(capture);
  }
  return result;
}

bool execution_plan::uses_capture_buffer() const { return !capture_slots_.empty(); }

bool execution_plan::is_whole_match_capture(std::uint32_t capture_index) const
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
    if (!candidate.instructions.empty() && !is_capture(candidate, capture_action::BEGIN)) { break; }
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

bool execution_plan::uses_unicode_word_boundaries() const
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

void execution_plan::prepare_start_seeker(deterministic_nfa_graph const& graph)
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
    else {
      pending.insert(pending.end(), node.targets.begin(), node.targets.end());
    }
  }
  if (first.empty() || first.size() > 8U) return;

  std::vector<char32_t> points;
  for (char32_t code_point = 0; code_point < 256; ++code_point)
    points.push_back(code_point);
  for (char32_t code_point = 0x80; code_point < 0x800; code_point += 64)
    points.push_back(code_point);
  for (char32_t code_point = 0x800; code_point < 0x10000; code_point += 4096)
    points.push_back(code_point);
  points.push_back(0x10000);
  for (char32_t code_point = 0x40000; code_point <= 0x100000; code_point += 0x40000)
    points.push_back(code_point);
  for (auto code_point : {8232U, 8233U, 8234U})
    points.push_back(code_point);
  for (auto id : first) {
    for (auto range : graph.nodes[id].predicate.ranges) {
      points.push_back(range.first);
      if (range.last < 0x10ffff) points.push_back(range.last + 1);
    }
  }
  std::array<std::uint64_t, 4> bitmap{};
  for (auto code_point : points) {
    if (code_point > 255 && ir_.options.characters == character_mode::BYTES) continue;
    if (!std::any_of(first.begin(), first.end(), [&](auto id) {
          return graph.nodes[id].predicate.matches(code_point);
        }))
      continue;
    auto byte = ir_.options.characters == character_mode::BYTES || code_point < 128
                  ? static_cast<std::uint32_t>(code_point)
                : code_point < 0x800   ? 0xc0U | (code_point >> 6)
                : code_point < 0x10000 ? 0xe0U | (code_point >> 12)
                                       : 0xf0U | (code_point >> 18);
    bitmap[byte / 64] |= std::uint64_t{1} << (byte % 64);
  }
  auto candidates = std::size_t{0};
  for (auto word : bitmap)
    candidates += std::popcount(word);
  if (candidates == 0 || candidates > 32U) return;
  candidate_seeker_ = true;
  candidate_bitmap_ = bitmap;
}

void execution_plan::prepare_thompson()
{
  thompson_ = make_deterministic_graph(ir_);
  if (!thompson_) { throw std::invalid_argument("unsupported Thompson instruction graph"); }
  auto& graph = *thompson_;
  prepare_start_seeker(graph);
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
  auto states = graph.nodes.size();
  auto record = capture_slots_.size() + 1;
  std::vector<bool> frontier_targets(states);
  closure_records_ = 1;
  for (auto& node : graph.nodes) {
    if (node.consumes) {
      for (auto target : node.targets)
        frontier_targets[target] = true;
    } else if (node.targets.size() > 1) {
      closure_records_ += node.targets.size() - 1;
    }
  }
  frontier_words_ =
    std::max<std::size_t>(1, std::count(frontier_targets.begin(), frontier_targets.end(), true)) *
    record;
  storage_words_ =
    2 * frontier_words_ + closure_records_ * record + record + 2 * ((states + 63) / 64);
  auto bytes       = storage_words_ * sizeof(std::int64_t);
  workspace_bytes_ = bytes > 32768 ? bytes : 0;
}

void execution_plan::select()
{
  verify(ir_);
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
  if (string_operations_.has_value() || fixed_ascii_suffix_.has_value()) { ascii_literal_.reset(); }
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
      deterministic_ =
        make_deterministic_machine(ir_, boolean_result && ir_.control.scan_input, !boolean_result);
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
  auto tagged_result = ir_.control.result == result_shape::CAPTURES && deterministic_.has_value() &&
                       deterministic_->capture_one_pass;
  if (!boolean_result &&
      (!deterministic_.has_value() || (uses_capture_buffer() && !tagged_result))) {
    deterministic_.reset();
  }
  if (!ascii_literal_.has_value() && !utf8_literal_.has_value() && !glushkov_.has_value() &&
      !deterministic_.has_value()) {
    mandatory_ascii_literal_ = mandatory_ascii_literal();
  }
  executor_ = executor_kind::ITERATIVE_THOMPSON;
  if (string_operations_.has_value()) {
    executor_ = executor_kind::STRING_OPERATIONS;
  } else if (fixed_ascii_suffix_.has_value()) {
    executor_ = executor_kind::STRING_OPERATIONS;
  } else if (line_tail_literal_.has_value()) {
    executor_ = executor_kind::STRING_OPERATIONS;
  } else if (word_run_minimum_.has_value()) {
    executor_ = executor_kind::WORD_RUN;
  } else if (ascii_literal_.has_value()) {
    executor_ = ascii_literal_->size() == 1U ? executor_kind::SINGLE_BYTE_LITERAL
                                             : executor_kind::PACKED_ASCII_LITERAL;
  } else if (utf8_literal_.has_value()) {
    executor_ = utf8_literal_pivot_.has_value() ? executor_kind::PACKED_UTF8_LITERAL
                                                : executor_kind::UTF8_KMP_LITERAL;
  } else if (glushkov_.has_value()) {
    executor_ =
      boolean_result ? executor_kind::GLUSHKOV : executor_kind::STREAMING_PRIORITIZED_GLUSHKOV;
  } else if (deterministic_.has_value()) {
    executor_ = deterministic_->assertion_aware ? executor_kind::ASSERTION_AWARE_DETERMINISTIC
                : boolean_result                ? executor_kind::DETERMINISTIC
                : tagged_result                 ? executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC
                                                : executor_kind::PRIORITIZED_DETERMINISTIC;
  }
  if (executor_ == executor_kind::ITERATIVE_THOMPSON) prepare_thompson();
  if (ir_.control.result == result_shape::MATCH_COUNT && glushkov_ &&
      glushkov_->repeated_predicate_count && !ir_.options.ascii_classes) {
    repeated_builtin_ = adapted_builtin(glushkov_->repeated_predicate_class);
  }
}

}  // namespace regex_ir::detail
