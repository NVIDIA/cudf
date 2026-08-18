/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Serialized trie implementation for C++/CUDA
 * @file trie.cuh
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <optional>
#include <string>
#include <vector>

namespace cudf {
namespace detail {
static constexpr char trie_terminating_character = '\n';

/**
 * @brief Node in the serialized trie.
 *
 * A serialized trie is an array of nodes. Each node represents a matching character, except for the
 * last child node, which denotes the end of the children list. Children of a node are stored
 * contiguously. The `children_offset` member is the offset between the node and its first child.
 * Matching is successful if all characters are matched and the final node is the last character of
 * a word (i.e. `is_leaf` is true).
 *
 */
struct serial_trie_node {
  int16_t children_offset{-1};
  char character{trie_terminating_character};
  bool is_leaf{false};
  explicit serial_trie_node(char c, bool leaf = false) noexcept : character(c), is_leaf(leaf) {}
};

using trie          = rmm::device_uvector<serial_trie_node>;
using optional_trie = std::optional<trie>;
using trie_view     = device_span<serial_trie_node const>;

inline trie_view make_trie_view(optional_trie const& t)
{
  if (!t) return {};
  return trie_view{t->data(), t->size()};
}

/**
 * @brief Creates a serialized trie for cache-friendly string search.
 *
 * The resulting trie is a compact array - children array size is equal to the
 * actual number of children nodes, not the size of the alphabet.
 *
 * @param keys Array of strings to insert into the trie
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return A host vector of nodes representing the serialized trie
 */
CUDF_EXPORT trie create_serialized_trie(std::vector<std::string> const& keys,
                                        cuda::stream_ref stream);

}  // namespace detail
}  // namespace cudf
