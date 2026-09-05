/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <charconv>
#include <cstddef>
#include <map>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

namespace {

// Dot-notation field names accept any byte except the structural characters '.' and '['.
[[nodiscard]] constexpr bool is_name_char(char c) { return c != '.' && c != '['; }

// Reads a maximal run of name characters from the front of `tail`.
[[nodiscard]] std::string read_unquoted_name(std::string_view tail)
{
  std::size_t n = 0;
  while (n < tail.size() && is_name_char(tail[n])) {
    ++n;
  }
  return std::string{tail.substr(0, n)};
}

// Reads a bracket step "[<non-negative integer>]" from the front of `tail`.
// The returned token keeps its brackets (e.g. "[42]").
[[nodiscard]] std::string read_bracket_step(std::string_view tail)
{
  CUDF_EXPECTS(!tail.empty() && tail.front() == '[',
               "expected '[' to open variant path index",
               std::invalid_argument);

  // Consume the maximal run of decimal digits.
  std::size_t n = 1;
  while (n < tail.size() && tail[n] >= '0' && tail[n] <= '9') {
    ++n;
  }
  CUDF_EXPECTS(
    n != 1, "expected non-negative integer after '[' in variant path", std::invalid_argument);

  // Reject indices that cannot be a valid array position (don't fit in cudf::size_type)
  cudf::size_type index = 0;
  auto const result     = std::from_chars(tail.data() + 1, tail.data() + n, index);
  CUDF_EXPECTS(
    result.ec == std::errc{}, "variant path index is out of range", std::invalid_argument);

  CUDF_EXPECTS(n < tail.size() && tail[n] == ']',
               "expected ']' to close variant path index",
               std::invalid_argument);
  return std::string{tail.substr(0, n + 1)};  // include the closing ']'
}

// A prefix-tree node built while merging paths. Children are keyed by step token, which is also how
// the device walk tells the steps apart, so an index step and a same-spelled name cannot collide.
// The map keeps the child order deterministic across runs.
struct trie_builder_node {
  std::map<std::string, std::size_t> children;
  bool ends_a_path = false;
};

// A prefix is worth caching only if more than one path continues past it, or a path ends there.
// Any other prefix is folded into its single descendant's step range.
[[nodiscard]] bool needs_slot(trie_builder_node const& node)
{
  return node.ends_a_path || node.children.size() > 1;
}

}  // namespace

std::vector<std::string> parse_variant_path(std::string_view path)
{
  std::vector<std::string> steps;
  auto const len  = path.size();
  std::size_t pos = 0;

  // Optional leading '$'
  if (pos < len && path[pos] == '$') { ++pos; }

  bool first = true;
  while (pos < len) {
    char const c = path[pos];
    if (c == '[') {
      steps.emplace_back(read_bracket_step(path.substr(pos)));
    } else {
      if (c == '.') {
        ++pos;
        CUDF_EXPECTS(pos < len && is_name_char(path[pos]),
                     "trailing '.' with no field name",
                     std::invalid_argument);
      } else {
        // Neither a '.'/'[' step nor a valid leading name (e.g. a stray ']' or a name after a step)
        CUDF_EXPECTS(
          first && is_name_char(c), "unexpected character in variant path", std::invalid_argument);
      }
      steps.emplace_back(read_unquoted_name(path.substr(pos)));
    }
    pos += steps.back().size();
    first = false;
  }

  CUDF_EXPECTS(!steps.empty(), "variant path is empty", std::invalid_argument);

  return steps;
}

variant_path_trie build_variant_path_trie(host_span<std::string_view const> paths)
{
  // Node 0 is the root: the value blob itself, before any step is applied.
  std::vector<trie_builder_node> nodes(1);
  std::vector<std::size_t> path_end_node(paths.size());

  for (std::size_t p = 0; p < paths.size(); ++p) {
    std::size_t node = 0;
    for (auto const& step : parse_variant_path(paths[p])) {
      auto const child = nodes[node].children.find(step);
      if (child != nodes[node].children.end()) {
        node = child->second;
      } else {
        auto const new_node = nodes.size();
        nodes.emplace_back();
        nodes[node].children.emplace(step, new_node);
        node = new_node;
      }
    }
    nodes[node].ends_a_path = true;
    path_end_node[p]        = node;
  }

  variant_path_trie trie;
  trie.slot_steps.push_back(0);
  std::vector<size_type> slot_of_node(nodes.size(), -1);

  // Depth-first descent from the root, emitting slots in pre-order. `pending` carries the steps of
  // the collapsed single-child chain walked since the last slot, and becomes that slot's step
  // range.
  struct descent_state {
    std::size_t node;
    size_type depth;
    std::vector<std::string> pending;
  };
  std::vector<descent_state> stack;
  stack.push_back({0, 0, {}});

  while (!stack.empty()) {
    auto state = std::move(stack.back());
    stack.pop_back();

    // The root is the value blob itself and never becomes a slot.
    auto child_depth = state.depth;
    std::vector<std::string> child_pending;
    if (state.node != 0 && needs_slot(nodes[state.node])) {
      slot_of_node[state.node] = static_cast<size_type>(trie.slot_depth.size());
      trie.steps.insert(trie.steps.end(), state.pending.begin(), state.pending.end());
      trie.slot_steps.push_back(static_cast<size_type>(trie.steps.size()));
      trie.slot_depth.push_back(state.depth);
      child_depth = state.depth + 1;
    } else {
      child_pending = std::move(state.pending);
    }

    // Reverse order, so that popping visits the first child first and keeps each subtree
    // contiguous.
    for (auto const& [step, child] : std::ranges::reverse_view(nodes[state.node].children)) {
      auto pending = child_pending;
      pending.push_back(step);
      stack.push_back({child, child_depth, std::move(pending)});
    }
  }

  // Invert the path-to-slot mapping into CSR form, counting then filling.
  auto const num_slots = trie.slot_depth.size();
  trie.output_offsets.assign(num_slots + 1, 0);
  for (auto const node : path_end_node) {
    ++trie.output_offsets[slot_of_node[node] + 1];
  }
  for (std::size_t slot = 0; slot < num_slots; ++slot) {
    trie.output_offsets[slot + 1] += trie.output_offsets[slot];
  }
  trie.output_paths.resize(paths.size());
  auto fill_position   = trie.output_offsets;
  size_type path_index = 0;
  for (auto const node : path_end_node) {
    trie.output_paths[fill_position[slot_of_node[node]]++] = path_index++;
  }

  return trie;
}

}  // namespace cudf::io::parquet::experimental::detail
