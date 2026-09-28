/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Serialized trie implementation for C++/CUDA
 * @file trie.cu
 */

#include "trie.cuh"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda_runtime.h>

#include <deque>
#include <string>
#include <vector>

namespace cudf {
namespace detail {

rmm::device_uvector<serial_trie_node> create_serialized_trie(std::vector<std::string> const& keys,
                                                             cuda::stream_ref stream)
{
  if (keys.empty()) { return rmm::device_uvector<serial_trie_node>{0, stream}; }

  static constexpr int alphabet_size = std::numeric_limits<char>::max() + 1;
  struct TreeTrieNode {
    using TrieNodePtr = std::unique_ptr<TreeTrieNode>;
    std::array<TrieNodePtr, alphabet_size> children;
    bool is_end_of_word = false;
  };

  // Construct a tree-structured trie
  // The trie takes a lot of memory, but the lookup is fast:
  // allows direct addressing of children nodes
  TreeTrieNode tree_trie;
  for (auto const& key : keys) {
    auto* current_node = &tree_trie;

    for (char const character : key) {
      if (current_node->children[character] == nullptr)
        current_node->children[character] = std::make_unique<TreeTrieNode>();

      current_node = current_node->children[character].get();
    }

    current_node->is_end_of_word = true;
  }

  struct IndexedTrieNode {
    TreeTrieNode const* const pnode;
    int16_t const idx;
    IndexedTrieNode(TreeTrieNode const* const node, int16_t index) : pnode(node), idx(index) {}
  };

  // Serialize the tree trie
  std::deque<IndexedTrieNode> to_visit;
  std::vector<serial_trie_node> nodes;

  // If the Tree trie matches empty strings, the root node is marked as 'end of word'.
  // The first node in the serialized trie is also used to match empty strings, so we're
  // initializing it using the `is_end_of_word` value from the root node.
  nodes.push_back(serial_trie_node(trie_terminating_character, tree_trie.is_end_of_word));

  // Add root node to queue. this node is not included to the serialized trie
  to_visit.emplace_back(&tree_trie, -1);
  while (!to_visit.empty()) {
    auto const node_and_idx = to_visit.front();
    auto const node         = node_and_idx.pnode;
    auto const idx          = node_and_idx.idx;
    to_visit.pop_front();

    bool has_children = false;
    for (size_t i = 0; i < node->children.size(); ++i) {
      if (node->children[i] != nullptr) {
        // Update the children offset of the parent node, unless at the root
        if (idx >= 0 && nodes[idx].children_offset < 0) {
          nodes[idx].children_offset = static_cast<uint16_t>(nodes.size() - idx);
        }
        // Add node to the trie
        nodes.emplace_back(static_cast<char>(i), node->children[i]->is_end_of_word);
        // Add to the queue, with the index within the new trie
        to_visit.emplace_back(node->children[i].get(), static_cast<uint16_t>(nodes.size()) - 1);

        has_children = true;
      }
    }
    // Only add the terminating character if any nodes were added
    if (has_children) { nodes.emplace_back(trie_terminating_character); }
  }
  return cudf::detail::make_device_uvector(nodes, stream, cudf::get_current_device_resource_ref());
}

}  // namespace detail
}  // namespace cudf
