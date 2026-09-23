/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/json.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf::io::json {

/// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;

/// Type used to represent the stack alphabet (i.e.: empty-stack, struct, list)
using StackSymbolT = char;

/// Type used to index into the symbols within the JSON input
using SymbolOffsetT = uint32_t;

/// Type large enough to support indexing up to max nesting level (must be signed)
using StackLevelT = int8_t;

/// Type used to represent a symbol group id of the input alphabet in the pushdown automaton
using PdaInputSymbolGroupIdT = char;

/// Type used to represent a symbol group id of the stack alphabet in the pushdown automaton
using PdaStackSymbolGroupIdT = char;

/// Type used to represent a (input-symbol, stack-symbol)-tuple in stack-symbol-major order
using PdaSymbolGroupIdT = char;

/// Type being emitted by the pushdown automaton transducer
using PdaTokenT = char;

/// Type used to represent the class of a node (or a node "category") within the tree representation
using NodeT = char;

/// Type used to index into the nodes within the tree of structs, lists, field names, and value
/// nodes
using NodeIndexT = size_type;

/// Type large enough to represent tree depth from [0, max-tree-depth); may be an unsigned type
using TreeDepthT = StackLevelT;

constexpr NodeIndexT parent_node_sentinel = -1;

/**
 * @brief Class of a node (or a node "category") within the tree representation
 */
enum node_t : NodeT {
  /// A node representing a struct
  NC_STRUCT,
  /// A node representing a list
  NC_LIST,
  /// A node representing a field name
  NC_FN,
  /// A node representing a string value
  NC_STR,
  /// A node representing a numeric or literal value (e.g., true, false, null)
  NC_VAL,
  /// A node representing a parser error
  NC_ERR,
  /// Total number of node classes
  NUM_NODE_CLASSES
};

/**
 * @brief Tokens emitted while parsing a JSON input
 */
enum token_t : PdaTokenT {
  /// Beginning-of-struct token (on encounter of semantic '{')
  StructBegin,
  /// End-of-struct token (on encounter of semantic '}')
  StructEnd,
  /// Beginning-of-list token (on encounter of semantic '[')
  ListBegin,
  /// End-of-list token (on encounter of semantic ']')
  ListEnd,
  // Beginning-of-struct-member token
  StructMemberBegin,
  // End-of-struct-member token
  StructMemberEnd,
  /// Beginning-of-field-name token (on encounter of first quote)
  FieldNameBegin,
  /// End-of-field-name token (on encounter of a field name's second quote)
  FieldNameEnd,
  /// Beginning-of-string-value token (on encounter of the string's first quote)
  StringBegin,
  /// End-of-string token (on encounter of a string's second quote)
  StringEnd,
  /// Beginning-of-value token (first character of literal or numeric)
  ValueBegin,
  /// Post-value token (first character after a literal or numeric string)
  ValueEnd,
  /// Beginning-of-error token (on first encounter of a parsing error)
  ErrorBegin,
  /// Delimiting a JSON line for error recovery
  LineEnd,
  /// Total number of tokens
  NUM_TOKENS
};

namespace CUDF_EXPORT detail {

/**
 * @brief Parses the given JSON string and emits a sequence of tokens that demarcate relevant
 * sections from the input.
 *
 * @param json_in The JSON input
 * @param options Parsing options specifying the parsing behaviour
 * @param stream The CUDA stream to which kernels are dispatched
 * @param mr Optional, resource with which to allocate
 * @return Pair of device vectors, where the first vector represents the token types and the second
 * vector represents the index within the input corresponding to each token
 */
std::pair<rmm::device_uvector<PdaTokenT>, rmm::device_uvector<SymbolOffsetT>> get_token_stream(
  device_span<SymbolT const> json_in,
  cudf::io::json_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace CUDF_EXPORT detail

}  // namespace cudf::io::json
