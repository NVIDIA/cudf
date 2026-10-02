/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

/**
 * @brief Parse a JSONPath-like VARIANT path string into an ordered sequence of steps.
 *
 * Grammar — object descent and array indexing:
 *   path  := "$"? first_step (("." name) | index)*
 *   first := name | "." name | index
 *   name  := [^.\[]+
 *   index := "[" [0-9]+ "]"
 *
 * A step is either an object-key name or an array index. Names accept any byte except '.' (step
 * separator) and '[' (start of an index step). Index steps hold a non-negative integer and are
 * returned with their brackets kept (e.g. "[42]"), which is how downstream consumers tell an index
 * step apart from an object key.
 *
 * @throws std::invalid_argument on an empty path or malformed syntax (e.g. a non-integer, negative,
 *         or out-of-range array index, an unterminated '[', or a trailing '.')
 */
[[nodiscard]] std::vector<std::string> parse_variant_path(std::string_view path);

/**
 * @brief A set of VARIANT paths merged into a prefix tree, flattened for a device walk.
 *
 * Slots are the units of work of the device walk: slot `k` starts from the value located by its
 * parent slot (or from the whole value blob when `slot_depth[k]` is 0) and applies the steps
 * `steps[slot_steps[k] : slot_steps[k + 1]]`.
 *
 * Only prefixes worth caching get a slot: a prefix shared by more than one path (a branch point) or
 * one that ends a path. A prefix with a single continuation and no path ending on it is folded into
 * its descendant's step range, so `$.user.addr.zip` is one slot holding three steps rather than
 * three slots.
 *
 * Slots are in depth-first pre-order, so a slot's subtree is contiguous and immediately follows it.
 * A walk can therefore keep one located value per depth: when it reaches slot `k`, the entry at
 * `slot_depth[k] - 1` still holds its parent's value. That bounds the walk's scratch by the depth
 * of the trie rather than by its slot count.
 *
 * `output_offsets` and `output_paths` are the reverse of the path-to-slot mapping in CSR form: the
 * value of slot `k` is the output of the input paths `output_paths[output_offsets[k] :
 * output_offsets[k + 1]]`, which is empty for a slot that only exists as a shared prefix. Duplicate
 * paths land on one slot with several outputs.
 */
struct variant_path_trie {
  std::vector<std::string> steps;         ///< Step tokens of every slot, concatenated in slot order
  std::vector<size_type> slot_steps;      ///< Size `num_slots + 1`; each slot's range in `steps`
  std::vector<size_type> slot_depth;      ///< Depth of each slot; 0 starts from the value blob
  std::vector<size_type> output_offsets;  ///< Size `num_slots + 1`; CSR offsets into `output_paths`
  std::vector<size_type> output_paths;    ///< Input paths that each slot's value is the output of
};

/**
 * @brief Merge VARIANT paths into a `variant_path_trie` so shared prefixes resolve once per row.
 *
 * Each path is parsed with `parse_variant_path`, so every path is validated before any of them is
 * used.
 *
 * @param paths JSONPath-like path strings, in output order
 * @return The flattened trie
 *
 * @throws std::invalid_argument if any path is empty or malformed
 */
[[nodiscard]] variant_path_trie build_variant_path_trie(host_span<std::string_view const> paths);

}  // namespace cudf::io::parquet::experimental::detail
