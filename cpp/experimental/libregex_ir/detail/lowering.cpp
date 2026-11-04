/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file lowering.cpp
 * @brief Lowering from Thompson automata to operation-specific instruction graphs.
 *
 * Parses replacement references and validates instruction graphs, including
 * independently anchored lookaround programs and their capture observation.
 */

#include <libregex_ir/detail/detail.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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

  for (auto& nested : ir.lookarounds)
    verify(nested);
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
      if (auto* look = std::get_if<test_assertion>(&item);
          look && look->kind == assertion_kind::LOOKAROUND &&
          look->program_index >= ir.lookarounds.size())
        invalid("lookaround program index is invalid");
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
  result.reverse = automata.reverse;
  for (std::size_t index = 0; index < automata.lookarounds.size(); ++index) {
    // Captures inside a negative assertion cannot escape it. Its complete
    // subtree therefore needs only a Boolean result, even for EXTRACT/REPLACE.
    auto negative = std::any_of(automata.states.begin(), automata.states.end(), [&](auto& state) {
      return state.kind == automata_state_kind::ASSERTION &&
             state.assertion == assertion_kind::LOOKAROUND && state.operand_index == index &&
             state.negative;
    });
    auto program  = negative ? lower(automata.lookarounds[index], {operation_kind::MATCHES, {}})
                             : lower(automata.lookarounds[index], selected);
    program.assertion_program   = true;
    program.control.scan_input  = false;
    program.control.require_end = false;
    result.lookarounds.push_back(std::move(program));
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
        block.instructions.emplace_back(
          test_assertion{state.assertion, state.operand_index, state.negative});
        break;
      case automata_state_kind::CAPTURE:
        block.instructions.emplace_back(write_capture{state.capture, state.operand_index});
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

}  // namespace regex_ir
