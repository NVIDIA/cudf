/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernel_ir.hpp"

#include <cuda_abi.hpp>
#include <regex_ir_detail.hpp>

#include <algorithm>
#include <cctype>
#include <format>
#include <stdexcept>

namespace cudf::experimental::detail::regex_jit {
namespace {
using regex_ir::matcher_abi;

std::string preamble(bool offset64, std::optional<matcher_abi> abi = std::nullopt)
{
  std::string result =
    std::format(R"CUDA(using namespace cudf::experimental::detail::regex_jit::device;

    using input_offset = {};
)CUDA",
                offset64 ? "i64" : "i32");
  if (abi) {
    auto symbol =
      *abi == matcher_abi::LIMITED_SPLIT ? "regex_ir_split_execute_limited" : "regex_ir_execute";
    result += std::format(R"CUDA(#if CUDF_REGEX_WORKSPACE_BYTES
{};
#else
{};
#endif
)CUDA",
                          regex_ir::matcher_signature(*abi, symbol, true),
                          regex_ir::matcher_signature(*abi, symbol, false));
    auto spec = regex_ir::layout(*abi);
    result += std::format(
      R"CUDA(
struct matcher {{
  __device__ static {} call(char* workspace, char const* data, i64 size{})
  {{
    return {}(RX_MATCHER_ARGUMENT data, size{});
  }}
}};
)CUDA",
      spec.result,
      spec.parameters,
      symbol,
      spec.arguments);
  }
  return result;
}

std::string entry(std::string source,
                  std::string_view name,
                  std::string_view parameters,
                  std::string_view body)
{
  if (name.empty() || !(std::isalpha(static_cast<std::uint8_t>(name[0])) || name[0] == '_') ||
      !std::all_of(name.begin() + 1, name.end(), [](std::uint8_t character) {
        return std::isalnum(character) || character == '_';
      }))
    throw std::invalid_argument("kernel_name must be a valid source identifier");
  source += std::format(
    R"CUDA(
extern "C" __global__ void {}(RX_KERNEL_PARAMETERS char const* chars,
                                           input_offset const* offsets,
                                           u32 const* validity,
                                           i32 row_offset,
                                           i32 rows{})
{{
  RX_WORKER
  column<input_offset> strings{{chars, offsets, validity, row_offset, rows}};
  {}
}}
)CUDA",
    name,
    parameters,
    body);
  return source;
}

std::string bytes(std::string_view value)
{
  std::string result;
  for (std::uint8_t character : value)
    result += std::format("{},", std::uint32_t(character));
  return result;
}

std::string_view output_type(bool offset64) { return offset64 ? "i64" : "i32"; }
}  // namespace

std::string make_module(std::string wrapper, std::size_t workspace_bytes)
{
  return std::format(R"CUDA(#define CUDF_REGEX_WORKSPACE_BYTES {}
#if CUDF_REGEX_WORKSPACE_BYTES
#define RX_KERNEL_PARAMETERS char* scratch, i32 tile_begin, i32 tile_end,
#define RX_MATCHER_ARGUMENT workspace,
#define RX_WORKER \
  worker worker_state{{scratch, tile_begin, tile_end, CUDF_REGEX_WORKSPACE_BYTES}};
#else
#define RX_KERNEL_PARAMETERS
#define RX_MATCHER_ARGUMENT
#define RX_WORKER worker worker_state{{}};
#endif
{}
)CUDA",
                     workspace_bytes,
                     wrapper);
}

std::string make_fixed_kernel(bool offset64,
                              regex_ir::operation_kind op,
                              std::optional<regex_ir::builtin_character_class> builtin,
                              std::string_view name)
{
  bool const boolean =
    op == regex_ir::operation_kind::CONTAINS || op == regex_ir::operation_kind::MATCHES;
  auto abi    = boolean ? matcher_abi::BOOLEAN
                : op == regex_ir::operation_kind::COUNT
                  ? builtin ? matcher_abi::BUILTIN_COUNT : matcher_abi::COUNT
                  : matcher_abi::FIND;
  auto source = preamble(offset64, abi);
  if (builtin) {
    std::string_view predicate;
    bool negated         = false;
    bool exclude_newline = false;
    switch (*builtin) {
      case regex_ir::builtin_character_class::NOT_DIGIT: negated = true; [[fallthrough]];
      case regex_ir::builtin_character_class::DIGIT:
        predicate       = "(bits & digit_mask) != 0";
        exclude_newline = true;
        break;
      case regex_ir::builtin_character_class::NOT_WORD: negated = true; [[fallthrough]];
      case regex_ir::builtin_character_class::WORD:
        predicate       = "(bits & word_mask) != 0 || code_point == '_'";
        exclude_newline = true;
        break;
      case regex_ir::builtin_character_class::NOT_SPACE: negated = true; [[fallthrough]];
      case regex_ir::builtin_character_class::SPACE: predicate = "(bits & space_mask) != 0"; break;
    }
    std::string ascii_check;
    if (*builtin == regex_ir::builtin_character_class::WORD ||
        *builtin == regex_ir::builtin_character_class::NOT_WORD) {
      ascii_check = std::format(
        R"CUDA(if (code_point < 128) {{
  bool positive = ((code_point | 32U) - 'a') < 26U || (code_point - '0') < 10U || code_point == '_';
  return {};
}})CUDA",
        negated ? "!positive && code_point != '\\n'" : "positive");
    } else if (*builtin == regex_ir::builtin_character_class::DIGIT ||
               *builtin == regex_ir::builtin_character_class::NOT_DIGIT) {
      ascii_check = std::format(
        R"CUDA(if (code_point < 128) {{ bool positive = (code_point - '0') < 10U; return {}; }})CUDA",
        negated ? "!positive && code_point != '\\n'" : "positive");
    }
    source += std::format(
      R"CUDA(extern "C" __device__ bool regex_ir_repeated_builtin(u32 code_point, char const* flags) {{
      {}
      constexpr u32 digit_mask = 4;
      constexpr u32 word_mask = 15;
      constexpr u32 space_mask = 16;
      u32 bits = code_point < 0x10000 ? static_cast<u8>(flags[code_point]) : 0;
      bool positive = {};
      return {};
    }}
)CUDA",
      ascii_check,
      predicate,
      negated ? exclude_newline ? "!positive && code_point != '\\n'" : "!positive" : "positive");
  }
  return entry(source,
               name,
               builtin ? ", void* output, char const* flags" : ", void* output",
               std::format("fixed<matcher, {}, {}>(worker_state, strings, output{});",
                           boolean                                 ? "fixed_output_kind::BOOLEAN"
                           : op == regex_ir::operation_kind::COUNT ? "fixed_output_kind::COUNT"
                                                                   : "fixed_output_kind::FIND",
                           builtin.has_value(),
                           builtin ? ", flags" : ""));
}

std::string make_warp_literal_contains_kernel(bool offset64,
                                              std::string_view literal,
                                              std::string_view name)
{
  if (literal.empty()) throw std::invalid_argument("literal must not be empty");
  auto source = preamble(offset64);
  auto pivot  = regex_ir::literal_anchor(literal);
  source += std::format(
    R"CUDA(struct literal {{
  static constexpr i64 size   = {};
  static constexpr i32 pivot  = {};
  static constexpr u32 anchor = {};
  __device__ static bool matches(input input_value, i64 pos)
  {{
    if (input_value.byte(pos) != {}U) {{ return false; }}
    return true)CUDA",
    literal.size(),
    pivot,
    std::uint32_t(static_cast<std::uint8_t>(literal[pivot])),
    std::uint32_t(static_cast<std::uint8_t>(literal.front())));
  for (std::size_t offset = 0; offset < literal.size();) {
    auto remaining      = literal.size() - offset;
    std::uint32_t width = remaining >= 8 ? 8 : remaining >= 4 ? 4 : remaining >= 2 ? 2 : 1;
    std::uint64_t value = 0;
    for (std::uint32_t piece_index = 0; piece_index < width; ++piece_index)
      value |= std::uint64_t(static_cast<std::uint8_t>(literal[offset + piece_index]))
               << (8 * piece_index);
    source += std::format(
      " && load_unaligned<u{}>(input_value.data + pos + {}) == {}ULL", width * 8, offset, value);
    offset += width;
  }
  source += R"CUDA(; } };
)CUDA";
  return entry(source, name, ", char* output", "warp_literal<literal>(strings, output);");
}

std::string make_capture_kernel(bool offset64,
                                std::int32_t slots,
                                std::int32_t first,
                                std::int32_t groups,
                                bool major,
                                std::string_view name)
{
  return entry(preamble(offset64, matcher_abi::CAPTURES),
               name,
               ", string_pair* output",
               std::format("capture<matcher, {}, {}, {}, {}>(worker_state, strings, output);",
                           slots,
                           first,
                           groups,
                           major));
}

std::string make_enumeration_size_kernel(bool offset64,
                                         std::int32_t slots,
                                         std::int32_t multiplier,
                                         bool require,
                                         bool cache,
                                         std::string_view name)
{
  return entry(
    preamble(offset64, matcher_abi::CAPTURES),
    name,
    cache ? ", i32* output, char* output_validity, i64* cache, i32 capacity, char* overflow"
          : ", i32* output, char* output_validity",
    std::format(
      R"CUDA(enumeration_size<matcher, {}, {}, {}, {}>(worker_state, strings, output, output_validity{});)CUDA",
      slots,
      multiplier,
      require,
      cache,
      cache ? ", cache, capacity, overflow" : ""));
}

std::string make_enumeration_emit_kernel(bool offset64,
                                         std::int32_t slots,
                                         std::int32_t groups,
                                         bool findall,
                                         bool overflow,
                                         std::string_view name)
{
  return entry(
    preamble(offset64, matcher_abi::CAPTURES),
    name,
    overflow ? ", string_pair* output, i32 const* output_offsets, char const* overflow"
             : ", string_pair* output, i32 const* output_offsets",
    std::format(
      "enumeration_emit<matcher, {}, {}, {}, {}>(worker_state, strings, output, output_offsets{});",
      slots,
      groups,
      findall,
      overflow,
      overflow ? ", overflow" : ""));
}

std::string make_limited_replace_kernel(bool offset64,
                                        bool emit,
                                        bool output64,
                                        bool cache,
                                        std::span<replacement_piece const> pieces,
                                        std::int32_t slots,
                                        std::int32_t limit,
                                        std::string_view name)
{
  auto source = preamble(offset64, matcher_abi::CAPTURES);
  for (std::size_t piece_index = 0; piece_index < pieces.size(); ++piece_index)
    if (!pieces[piece_index].literal.empty())
      source += std::format(R"CUDA(__device__ __constant__ u8 replacement_{}[] = {{{}}};
)CUDA",
                            piece_index,
                            bytes(pieces[piece_index].literal));
  source +=
    R"CUDA(struct replacement {
  __device__ static i64 apply(input input_value, i64 const* spans, char* output, i64 cursor)
  {
)CUDA";
  for (std::size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
    auto& piece = pieces[piece_index];
    if (piece.capture) {
      auto slot = *piece.capture * 2;
      source += std::format(
        R"CUDA(if (spans[{}] >= 0 && spans[{}] >= spans[{}])
  cursor = append(input_value.data, spans[{}], spans[{}], output, cursor);
)CUDA",
        slot,
        slot + 1,
        slot,
        slot,
        slot + 1);
    } else if (!piece.literal.empty()) {
      source += std::format(
        R"CUDA(cursor = append(reinterpret_cast<char const*>(replacement_{}), 0, {}, output, cursor);
)CUDA",
        piece_index,
        piece.literal.size());
    }
  }
  source += R"CUDA(return cursor; } };
)CUDA";
  std::string parameters =
    emit ? std::format(", void* output, {} const* output_offsets", output_type(output64))
         : ", void* output";
  if (cache) parameters += ", i64* cache, i32 capacity, char* overflow, i32* counts";
  if (!emit) parameters += ", i32* size_overflow";
  return entry(source,
               name,
               parameters,
               std::format(
                 R"CUDA(limited_replace<matcher,
                replacement,
                {},
                {},
                {},
                {},
                {}>(worker_state,
                                 strings,
                                 output,
                                 {},
                                 {},
                                 {},
                                 {},
                                 {},
                                 {});)CUDA",
                 slots,
                 limit,
                 emit,
                 cache,
                 output_type(output64),
                 emit ? "output_offsets" : "nullptr",
                 cache ? "cache" : "nullptr",
                 cache ? "capacity" : "0",
                 cache ? "overflow" : "nullptr",
                 cache ? "counts" : "nullptr",
                 emit ? "nullptr" : "size_overflow"));
}

std::string encode_replacement(std::span<replacement_piece const> replacement)
{
  std::string result;
  for (auto& piece : replacement) {
    if (piece.capture) {
      result += std::format("${{{}}}", *piece.capture);
      continue;
    }
    for (char character : piece.literal) {
      result += character;
      if (character == '$') result += '$';
    }
  }
  return result;
}

std::string make_replace_kernel(bool offset64, bool emit, bool output64, std::string_view name)
{
  return entry(preamble(offset64, matcher_abi::REPLACE),
               name,
               emit ? std::format(", void* output, {} const* output_offsets", output_type(output64))
                    : ", void* output",
               std::format("replace<matcher, {}, {}>(worker_state, strings, output{});",
                           emit,
                           output_type(output64),
                           emit ? ", output_offsets" : ""));
}

std::string make_split_size_kernel(bool offset64,
                                   std::int32_t limit,
                                   bool cache,
                                   std::string_view name)
{
  return entry(preamble(offset64, matcher_abi::LIMITED_SPLIT),
               name,
               cache ? ", i32* output, i64* cache, i32 capacity, char* overflow" : ", i32* output",
               std::format("split_size<matcher, {}, {}>(worker_state, strings, output{});",
                           limit,
                           cache,
                           cache ? ", cache, capacity, overflow" : ""));
}

std::string make_split_emit_kernel(
  bool offset64, bool reverse, std::int32_t limit, bool overflow, std::string_view name)
{
  return entry(
    preamble(offset64, matcher_abi::LIMITED_SPLIT),
    name,
    std::format(", string_pair* output, i32 const* effective, i32 const* full, i64* spans{}",
                overflow ? ", char const* overflow" : ""),
    std::format(
      "split_emit<matcher, {}, {}, {}>(worker_state, strings, output, effective, full, spans{});",
      limit,
      reverse,
      overflow,
      overflow ? ", overflow" : ""));
}

std::string make_span_cache_sample_kernel(
  bool offset64, bool split, std::int32_t slots, std::int32_t limit, std::string_view name)
{
  return entry(
    preamble(offset64, split ? matcher_abi::LIMITED_SPLIT : matcher_abi::CAPTURES),
    name,
    ", i32 samples, i32 capacity, u64* stats",
    std::format("sample<matcher, {}, {}, {}>(worker_state, strings, samples, capacity, stats);",
                split,
                std::max(slots, 2),
                limit));
}
}  // namespace cudf::experimental::detail::regex_jit
