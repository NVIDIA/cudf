/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file optimizer.cpp
 * @brief Optimization passes over regex instruction graphs.
 *
 * Removes unobserved capture writes, folds empty jumps, fuses literal sequences,
 * and removes unreachable blocks, including within lookaround subprograms.
 */

#include <libregex_ir/detail/detail.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

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

  for (auto& nested : ir.lookarounds)
    nested = optimize(std::move(nested), options);
  if (options.strip_unobserved_captures) strip_captures(ir);
  if (options.fold_epsilon_jumps) fold_empty_jumps(ir);
  if (options.fuse_literals) fuse_literals(ir, options.literal_fusion_limit);
  if (options.remove_unreachable) remove_unreachable(ir);

  verify(ir);
  return ir;
}

}  // namespace regex_ir
