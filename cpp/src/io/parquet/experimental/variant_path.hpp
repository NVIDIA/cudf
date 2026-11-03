/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

}  // namespace cudf::io::parquet::experimental::detail
