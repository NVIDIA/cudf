/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace regex_ir::device {
// Types.
using i8  = signed char;
using i16 = signed short;
using i32 = signed int;
using i64 = signed long long;
using u8  = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

static_assert(sizeof(i8) == 1);
static_assert(sizeof(i16) == 2);
static_assert(sizeof(i32) == 4);
static_assert(sizeof(i64) == 8);
static_assert(sizeof(u8) == 1);
static_assert(sizeof(u16) == 2);
static_assert(sizeof(u32) == 4);
static_assert(sizeof(u64) == 8);

// Enums.
enum class executor_kind : u8 {
  ITERATIVE_THOMPSON               = 0,
  STRING_OPERATIONS                = 1,
  WORD_RUN                         = 2,
  SINGLE_BYTE_LITERAL              = 3,
  PACKED_ASCII_LITERAL             = 4,
  PACKED_UTF8_LITERAL              = 5,
  UTF8_KMP_LITERAL                 = 6,
  GLUSHKOV                         = 7,
  STREAMING_PRIORITIZED_GLUSHKOV   = 8,
  DETERMINISTIC                    = 9,
  ASSERTION_AWARE_DETERMINISTIC    = 10,
  PRIORITIZED_DETERMINISTIC        = 11,
  TAGGED_PRIORITIZED_DETERMINISTIC = 12,
  BOOLEAN_ALTERNATION              = 13
};

enum class assertion_kind : u8 {
  BEGIN_INPUT       = 0,
  END_INPUT         = 1,
  WORD_BOUNDARY     = 2,
  NOT_WORD_BOUNDARY = 3,
  BEGIN_LINE        = 4,
  END_LINE          = 5,
  NONE              = 6
};

enum class string_operation_kind : u8 {
  BEGINS_WITH = 0,
  ENDS_WITH   = 1,
  ENDS_LINE   = 2,
  EQUALS      = 3,
  EQUALS_LINE = 4
};

enum class dispatch_status : i32 { EXHAUSTED = -1, ACCEPTED = -2 };

enum class ascii_run_kind : u8 { GENERAL = 0, ALL_ASCII = 1, EXCEPT_BYTE = 2 };

inline constexpr u32 transition_accept_mask      = 0x8000U;
inline constexpr u32 transition_stop_before_mask = 0x4000U;

enum class fixed_output_kind : u8 { BOOLEAN = 0, COUNT = 1, FIND = 2 };

enum class unicode_decoder_kind : u8 { SCALAR = 0, BUFFERED = 1 };

__device__ constexpr bool is_literal_executor(executor_kind kind)
{
  return kind == executor_kind::SINGLE_BYTE_LITERAL ||
         kind == executor_kind::PACKED_ASCII_LITERAL ||
         kind == executor_kind::PACKED_UTF8_LITERAL || kind == executor_kind::UTF8_KMP_LITERAL;
}

__device__ constexpr bool is_deterministic_executor(executor_kind kind)
{
  return kind == executor_kind::DETERMINISTIC ||
         kind == executor_kind::ASSERTION_AWARE_DETERMINISTIC ||
         kind == executor_kind::PRIORITIZED_DETERMINISTIC ||
         kind == executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC;
}

// Input.
__device__ __forceinline__ u32 trailing_zeroes(u64 value)
{
  return value == 0 ? 64 : __ffsll(static_cast<i64>(value)) - 1;
}

template <typename Value>
__device__ __forceinline__ Value load_unaligned(char const* address)
{
  Value value;
  auto* bytes = reinterpret_cast<char*>(&value);
#pragma unroll
  for (u32 byte_index = 0; byte_index < sizeof(Value); ++byte_index)
    bytes[byte_index] = address[byte_index];
  return value;
}

struct input {
  char const* data;
  i64 size;

  __device__ __forceinline__ u32 byte(i64 position) const
  {
    return static_cast<u8>(data[position]);
  }

  template <bool Bytes>
  __device__ __forceinline__ i64 width(i64 position) const
  {
    if (position >= size) return 0;
    if constexpr (Bytes) return 1;
    auto first = byte(position);
    if (first < 128) return 1;
    auto count = first >= 194 && first <= 223   ? 2
                 : first >= 224 && first <= 239 ? 3
                 : first >= 240 && first <= 244 ? 4
                                                : 1;
    if (size - position < count) return 1;
    for (i32 byte_index = 1; byte_index < count; ++byte_index)
      if ((byte(position + byte_index) & 192U) != 128U) return 1;
    return count;
  }

  struct decoded {
    u32 code_point;
    i64 width;
  };

  template <bool Bytes, bool FuseContinuationLoads = true>
  __device__ __forceinline__ decoded read(i64 position) const
  {
    if (position >= size) return {0, 0};
    auto code_point = byte(position);
    if constexpr (Bytes) return {code_point, 1};
    if (code_point < 128) return {code_point, 1};
    if constexpr (!FuseContinuationLoads) {
      auto count = width<false>(position);
      if (count == 1) return {code_point, 1};
      code_point &= count == 2 ? 31U : count == 3 ? 15U : 7U;
      for (i64 byte_index = 1; byte_index < count; ++byte_index)
        code_point = (code_point << 6U) | (byte(position + byte_index) & 63U);
      return {code_point, count};
    } else {
      auto first_byte = code_point;
      auto count      = first_byte >= 194 && first_byte <= 223   ? 2
                        : first_byte >= 224 && first_byte <= 239 ? 3
                        : first_byte >= 240 && first_byte <= 244 ? 4
                                                                 : 1;
      if (count == 1 || size - position < count) return {first_byte, 1};
      code_point &= count == 2 ? 31U : count == 3 ? 15U : 7U;
      // Validate and assemble from the same loads. Invalid sequences still consume one byte.
      for (i32 byte_index = 1; byte_index < count; ++byte_index) {
        auto continuation = byte(position + byte_index);
        if ((continuation & 192U) != 128U) return {first_byte, 1};
        code_point = (code_point << 6U) | (continuation & 63U);
      }
      return {code_point, count};
    }
  }

  template <bool Bytes, bool FuseContinuationLoads = true>
  __device__ __forceinline__ u32 decode(i64 position) const
  {
    return read<Bytes, FuseContinuationLoads>(position).code_point;
  }

  template <bool Bytes>
  __device__ __forceinline__ i64 advance(i64 position) const
  {
    return position + width<Bytes>(position);
  }

  template <bool Bytes>
  __device__ __forceinline__ i64 previous(i64 position) const
  {
    if (position == 0) return 0;
    --position;
    if constexpr (!Bytes) {
      while (position > 0 && (byte(position) & 192U) == 128U)
        --position;
    }
    return position;
  }

  // Eight bytes must be available within this string. Match the original NVVM
  // unaligned load without reading before or beyond the row.
  template <bool AlignedWords, bool AlignShortRows = false>
  __device__ __forceinline__ u64 search_word(i64 position) const
  {
    constexpr i64 short_row_bytes = 256;
    if constexpr (AlignedWords || AlignShortRows) {
      if (AlignedWords || size < short_row_bytes) {
        auto address  = reinterpret_cast<u64>(data + position);
        auto base     = reinterpret_cast<u64>(data);
        auto aligned  = address & ~u64{3};
        auto offset   = static_cast<u32>(address & 3U);
        auto required = offset == 0 ? 8U : 12U;
        if (aligned >= base && aligned + required <= base + size) {
          auto* aligned_words = reinterpret_cast<u32 const*>(data + (aligned - base));
          auto first_word     = aligned_words[0];
          auto second_word    = aligned_words[1];
          if (offset == 0)
            return static_cast<u64>(first_word) | (static_cast<u64>(second_word) << 32U);
          auto third_word = aligned_words[2];
          auto shift      = offset * 8U;
          return static_cast<u64>(__funnelshift_r(first_word, second_word, shift)) |
                 (static_cast<u64>(__funnelshift_r(second_word, third_word, shift)) << 32U);
        }
      }
    }
    return load_unaligned<u64>(data + position);
  }

  template <bool AlignedWords = false, bool AlignShortRows = false>
  __device__ __forceinline__ i64 seek(i64 position, u32 needle) const
  {
    auto repeated = static_cast<u64>(needle) * 0x0101010101010101ULL;
    while (size - position >= 8) {
      auto differences = search_word<AlignedWords, AlignShortRows>(position) ^ repeated;
      auto candidates =
        (differences - 0x0101010101010101ULL) & ~differences & 0x8080808080808080ULL;
      while (candidates) {
        auto candidate = position + trailing_zeroes(candidates) / 8;
        // Subtraction can propagate a borrow into a neighboring byte.
        if (byte(candidate) == needle) return candidate;
        candidates &= candidates - 1;
      }
      position += 8;
    }
    while (position < size && byte(position) != needle)
      ++position;
    return position;
  }
};

__device__ __forceinline__ bool ascii_word(u32 code_point)
{
  return (code_point >= 'a' && code_point <= 'z') || (code_point >= 'A' && code_point <= 'Z') ||
         (code_point >= '0' && code_point <= '9') || code_point == '_';
}

template <bool Extended>
__device__ __forceinline__ bool newline(u32 code_point)
{
  return code_point == 10 || (Extended && (code_point == 13 || code_point == 133 ||
                                           code_point == 8232 || code_point == 8233));
}

template <typename Pattern>
__device__ __forceinline__ bool assertion(
  input input_value, i64 position, assertion_kind kind, u32 before, u32 current, i64 current_width)
{
  if (kind == assertion_kind::BEGIN_INPUT) return position == 0;
  if (kind == assertion_kind::END_INPUT) return position == input_value.size;
  auto mid_crlf = Pattern::extended_newline && before == 13 && current == 10;
  if (kind == assertion_kind::WORD_BOUNDARY || kind == assertion_kind::NOT_WORD_BOUNDARY) {
    auto boundary = (position > 0 && Pattern::is_word(before)) !=
                    (position < input_value.size && Pattern::is_word(current));
    return kind == assertion_kind::WORD_BOUNDARY ? boundary : !boundary;
  }
  if (kind == assertion_kind::BEGIN_LINE)
    return position == 0 ||
           (Pattern::multiline && newline<Pattern::extended_newline>(before) && !mid_crlf);
  if (position == input_value.size) return true;
  if (!newline<Pattern::extended_newline>(current) || mid_crlf) return false;
  if (Pattern::multiline) return true;
  auto next = position + current_width;
  return next == input_value.size ||
         (Pattern::extended_newline && current == 13 &&
          input_value.template decode<Pattern::bytes, Pattern::fused_unicode_decode>(next) == 10 &&
          input_value.template advance<Pattern::bytes>(next) == input_value.size);
}

template <typename Pattern>
__device__ __forceinline__ bool assertion(input input_value, i64 position, assertion_kind kind)
{
  auto previous = input_value.template previous<Pattern::bytes>(position);
  auto before =
    position > 0
      ? input_value.template decode<Pattern::bytes, Pattern::fused_unicode_decode>(previous)
      : 0;
  auto current = input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(position);
  return assertion<Pattern>(input_value, position, kind, before, current.code_point, current.width);
}

__device__ __forceinline__ i64
append(char const* source, i64 begin, i64 end, char* output, i64 cursor)
{
  auto count = end - begin;
  if (output != nullptr && count > 0) {
    i64 chunk_offset = 0;
    for (; count - chunk_offset >= 8; chunk_offset += 8) {
      auto value  = load_unaligned<u64>(source + begin + chunk_offset);
      auto* bytes = reinterpret_cast<char const*>(&value);
#pragma unroll
      for (u32 byte_index = 0; byte_index < sizeof(value); ++byte_index)
        output[cursor + chunk_offset + byte_index] = bytes[byte_index];
    }
    for (; chunk_offset < count; ++chunk_offset)
      output[cursor + chunk_offset] = source[begin + chunk_offset];
  }
  return cursor + count;
}

}  // namespace regex_ir::device

extern "C" __device__ bool regex_ir_repeated_builtin(regex_ir::device::u32, char const*);

namespace regex_ir::device {

// Sequential decoding keeps a row-bounded aligned window in registers. A character
// that crosses the window, or an unaligned row edge, uses the scalar decoder.
struct buffered_unicode_input {
  input input_value;
  u64 buffered_bytes  = 0;
  i32 remaining_bytes = 0;

  __device__ __forceinline__ input::decoded read(i64 position)
  {
    if (remaining_bytes == 0) {
      constexpr u64 alignment_mask = 7;
      auto address                 = reinterpret_cast<u64>(input_value.data + position);
      auto aligned_address         = address & ~alignment_mask;
      auto leading_bytes           = static_cast<i32>(address & alignment_mask);
      auto available_bytes         = 8 - leading_bytes;
      if (aligned_address >= reinterpret_cast<u64>(input_value.data) &&
          input_value.size - position >= available_bytes) {
        buffered_bytes = *reinterpret_cast<u64 const*>(aligned_address);
        buffered_bytes >>= 8U * leading_bytes;
        remaining_bytes = available_bytes;
      }
    }
    if (remaining_bytes == 0) { return input_value.read<false, true>(position); }
    auto first_byte = static_cast<u32>(buffered_bytes & 255U);
    auto width      = first_byte < 128                         ? 1
                      : first_byte >= 194 && first_byte <= 223 ? 2
                      : first_byte >= 224 && first_byte <= 239 ? 3
                      : first_byte >= 240 && first_byte <= 244 ? 4
                                                               : 1;
    if (width > remaining_bytes) {
      remaining_bytes = 0;
      return input_value.read<false, true>(position);
    }
    auto code_point = first_byte;
    if (width > 1) {
      code_point &= width == 2 ? 31U : width == 3 ? 15U : 7U;
#pragma unroll 3
      for (i32 byte_index = 1; byte_index < width; ++byte_index) {
        auto continuation = static_cast<u32>((buffered_bytes >> (8U * byte_index)) & 255U);
        if ((continuation & 192U) != 128U) {
          code_point = first_byte;
          width      = 1;
          break;
        }
        code_point = (code_point << 6U) | (continuation & 63U);
      }
    }
    buffered_bytes >>= 8U * width;
    remaining_bytes -= width;
    return {code_point, width};
  }
};

// Policy.
// Defaults let a generated policy specify only the capabilities it needs.
struct policy {
  static constexpr bool bytes                      = false;
  static constexpr bool fused_unicode_decode       = true;
  static constexpr bool ascii_word_classes         = false;
  static constexpr bool multiline                  = false;
  static constexpr bool extended_newline           = false;
  static constexpr bool scan_input                 = false;
  static constexpr bool require_end                = false;
  static constexpr bool assertion_aware            = false;
  static constexpr u32 capture_slots               = 2;
  static constexpr u32 live_captures               = 0;
  static constexpr i32 prefix                      = -1;
  static constexpr assertion_kind accept_assertion = assertion_kind::NONE;
  static constexpr i64 fixed_match_bytes           = -1;
  static constexpr u32 initial_state               = 0;
  static constexpr u32 state_mask                  = 16383;
  static constexpr u32 dead_state                  = 65535;
  static constexpr u32 restart_state               = 65535;
  static constexpr u32 assertion_mask              = 0;
  static constexpr u32 boundary_classes            = 1;
  static constexpr bool start_filter               = false;
  static constexpr bool simple_capture             = false;
  static constexpr bool accepts_any_first          = false;
  static constexpr bool boolean_prefix_skip        = false;
  static constexpr bool aligned_prefix_seek        = true;

  __device__ __forceinline__ static bool is_word(u32 code_point) { return ascii_word(code_point); }

  __device__ __forceinline__ static bool start_byte(u32) { return true; }

  __device__ __forceinline__ static bool candidate_byte(u32) { return true; }
};

struct ascii_run_filter {
  ascii_run_kind kind;
  u32 excluded_byte;
};

template <typename Pattern>
__device__ __forceinline__ bool literal_find(input, i64, i64&, i64&);

template <typename Pattern>
__device__ __forceinline__ bool dfa_boolean(input);

template <typename Pattern>
__device__ __forceinline__ bool dfa_find(input, i64, i64&, i64&, i64*);

template <typename Pattern>
__device__ __forceinline__ bool glushkov_boolean(input);

template <typename Pattern>
__device__ __forceinline__ bool glushkov_find(input, i64, i64&, i64&);

template <typename Pattern>
__device__ __forceinline__ bool thompson_run(input, i64, i64&, i64*, i64*);

// Literals.
template <typename Pattern>
__device__ __forceinline__ bool literal_find(input input_value, i64 start, i64& begin, i64& end)
{
  constexpr i64 hybrid_literal_row_bytes = 256;
  if constexpr (Pattern::kmp && Pattern::hybrid && Pattern::scan_input && !Pattern::require_end &&
                Pattern::literal_size >= 8 && Pattern::literal_size <= 16) {
    if (input_value.size >= hybrid_literal_row_bytes) {
      // UTF-8 leading bytes can be common throughout a row. Preserve the generated
      // selective pivot and its verification guard instead of always seeking byte zero.
      constexpr i32 candidate_offset = Pattern::pivot < 0 ? 0 : Pattern::pivot;
      auto position                  = start;
      auto needle                    = Pattern::literal_byte(candidate_offset);
      while (input_value.size - position >= Pattern::literal_size) {
        auto found = input_value.template seek<true>(position + candidate_offset, needle);
        if (found == input_value.size) { return false; }
        position = found - candidate_offset;
        if (input_value.size - position < Pattern::literal_size) { return false; }
        if (Pattern::literal_guard(input_value, position) &&
            Pattern::literal_at(input_value, position)) {
          begin = position;
          end   = position + Pattern::literal_size;
          return true;
        }
        ++position;
      }
      return false;
    }
  }
  if constexpr (Pattern::kmp) {
    if (Pattern::pivot < 0 || (Pattern::hybrid && input_value.size >= 256)) {
      u32 matched = 0;
      for (auto pos = start; pos < input_value.size; ++pos) {
        auto value = input_value.byte(pos);
        while (matched && value != Pattern::literal_byte(matched))
          matched = Pattern::failure(matched - 1);
        if (value == Pattern::literal_byte(matched)) ++matched;
        if (matched == Pattern::literal_size) {
          end   = pos + 1;
          begin = end - Pattern::literal_size;
          return true;
        }
      }
      return false;
    }
  }
  constexpr auto pivot = Pattern::pivot < 0 ? 0 : Pattern::pivot;
  auto pos             = start;
  bool const vector_seek =
    Pattern::vector_seek && input_value.size - start >= Pattern::seek_threshold;
  while (pos <= input_value.size && input_value.size - pos >= Pattern::literal_size) {
    if (vector_seek) {
      auto found =
        input_value.template seek<!Pattern::kmp>(pos + pivot, Pattern::literal_byte(pivot));
      if (found == input_value.size) return false;
      pos = found - pivot;
      if (input_value.size - pos < Pattern::literal_size) return false;
    }
    bool matched;
    if constexpr (!Pattern::kmp && Pattern::literal_size == 2)
      matched = (vector_seek || input_value.byte(pos + pivot) == Pattern::literal_byte(pivot)) &&
                Pattern::literal_guard(input_value, pos) &&
                Pattern::literal_candidate_at(input_value, pos);
    else {
      matched = input_value.byte(pos + pivot) == Pattern::literal_byte(pivot) &&
                Pattern::literal_guard(input_value, pos) && Pattern::literal_at(input_value, pos);
    }
    if (matched) {
      begin = pos;
      end   = pos + Pattern::literal_size;
      return true;
    }
    ++pos;
  }
  return false;
}

// Strings.
template <typename Pattern, string_operation_kind Kind>
__device__ __forceinline__ bool string_boolean(input input_value)
{
  constexpr bool begins = Kind == string_operation_kind::BEGINS_WITH ||
                          Kind == string_operation_kind::EQUALS ||
                          Kind == string_operation_kind::EQUALS_LINE;
  constexpr bool ends = Kind != string_operation_kind::BEGINS_WITH;
  constexpr bool line =
    Kind == string_operation_kind::ENDS_LINE || Kind == string_operation_kind::EQUALS_LINE;
  auto matches = [&](i64 end) {
    if (end < Pattern::literal_size) return false;
    if constexpr (begins && ends)
      if (end != Pattern::literal_size) return false;
    return Pattern::literal_at(input_value, begins ? 0 : end - Pattern::literal_size);
  };
  if (matches(input_value.size)) return true;
  if constexpr (line) {
    return input_value.size > Pattern::literal_size &&
           input_value.byte(input_value.size - 1) == 10 && matches(input_value.size - 1);
  }
  return false;
}

template <typename Pattern>
__device__ __forceinline__ bool line_tail_find(input input_value, i64 search, i64& begin, i64& end)
{
  end = input_value.size;
  if (end > 0 && input_value.byte(end - 1) == 10) --end;
  auto start = end;
  while (start > 0 && input_value.byte(start - 1) != 10)
    --start;
  if (search > start) start = search;
  i64 literal_end;
  return literal_find<Pattern>({input_value.data, end}, start, begin, literal_end);
}

// Dfa.
template <typename Pattern>
__device__ __forceinline__ u32
boundary(input input_value, i64 position, u32 before, u32 current, i64 width)
{
  u32 context = 0;
  u32 bit     = 1;
  // Context classes pack only the assertion bits present in this machine.
  for (u32 kind = 0; kind < 6; ++kind) {
    if ((Pattern::assertion_mask & (1U << kind)) == 0) continue;
    if (assertion<Pattern>(
          input_value, position, static_cast<assertion_kind>(kind), before, current, width))
      context |= bit;
    bit <<= 1;
  }
  return context;
}

template <typename Pattern>
__device__ __forceinline__ bool dfa_boolean(input input_value)
{
  auto state = Pattern::initial_state;
  if constexpr (!Pattern::assertion_aware && !Pattern::require_end &&
                Pattern::accept_assertion == assertion_kind::NONE) {
    if (state & transition_accept_mask) return true;
  }
  i64 pos                 = 0;
  u32 previous_code_point = 0;
  for (; pos < input_value.size;) {
    auto character  = input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(pos);
    auto code_point = character.code_point;
    if constexpr (!Pattern::assertion_aware && Pattern::prefix >= 0) {
      if ((state & Pattern::state_mask) == (Pattern::initial_state & Pattern::state_mask) &&
          code_point < 128 && code_point != Pattern::prefix) {
        pos =
          input_value.template seek<Pattern::aligned_prefix_seek, !Pattern::aligned_prefix_seek>(
            pos + 1, Pattern::prefix);
        continue;
      }
    }
    if constexpr (Pattern::assertion_aware && Pattern::boolean_prefix_skip) {
      if ((state == (Pattern::initial_state & Pattern::state_mask) ||
           state == Pattern::dead_state) &&
          code_point != Pattern::boolean_seek_byte) {
        pos =
          input_value.template seek<Pattern::aligned_prefix_seek, !Pattern::aligned_prefix_seek>(
            pos + 1, Pattern::boolean_seek_byte);
        previous_code_point =
          input_value.template decode<Pattern::bytes, Pattern::fused_unicode_decode>(
            input_value.template previous<Pattern::bytes>(pos));
        continue;
      }
    }
    auto context =
      Pattern::assertion_aware
        ? boundary<Pattern>(input_value, pos, previous_code_point, code_point, character.width)
        : 0;
    auto encoded =
      Pattern::transition(state & Pattern::state_mask, Pattern::classify(code_point), context);
    previous_code_point = code_point;
    pos += character.width;
    state = Pattern::assertion_aware ? encoded & Pattern::state_mask : encoded;
    if constexpr (!Pattern::assertion_aware && !Pattern::scan_input) {
      if ((state & Pattern::state_mask) == Pattern::dead_state) return false;
    }
    if constexpr (!Pattern::require_end) {
      if ((encoded & transition_accept_mask) &&
          (Pattern::accept_assertion == assertion_kind::NONE ||
           assertion<Pattern>(input_value, pos, Pattern::accept_assertion)))
        return true;
    }
  }
  if constexpr (Pattern::assertion_aware) {
    return Pattern::boundary_accept(state,
                                    boundary<Pattern>(input_value, pos, previous_code_point, 0, 0));
  } else {
    return (state & transition_accept_mask) &&
           (Pattern::require_end || Pattern::accept_assertion != assertion_kind::NONE) &&
           (Pattern::accept_assertion == assertion_kind::NONE ||
            assertion<Pattern>(input_value, pos, Pattern::accept_assertion));
  }
}

template <typename Pattern>
__device__ __forceinline__ bool dfa_find(
  input input_value, i64 search, i64& begin, i64& end, i64* captures)
{
  for (auto start = search; start <= input_value.size;) {
    if (start < input_value.size && !Pattern::start_byte(input_value.byte(start))) {
      if constexpr (Pattern::start_filter_prefix >= 0)
        start = input_value.seek(start + 1, Pattern::start_filter_prefix);
      else if constexpr (Pattern::prefix >= 0) {
        start =
          input_value.template seek<Pattern::aligned_prefix_seek, !Pattern::aligned_prefix_seek>(
            start + 1, Pattern::prefix);
      } else {
        ++start;
      }
      continue;
    }
    auto state              = Pattern::initial_state;
    auto pos                = start;
    u32 previous_code_point = 0;
    if constexpr (Pattern::assertion_aware)
      if (start > 0)
        previous_code_point =
          input_value.template decode<Pattern::bytes, Pattern::fused_unicode_decode>(
            input_value.template previous<Pattern::bytes>(start));
    auto accepted = (state & transition_accept_mask) ? start : i64{-1};
    if constexpr (Pattern::tagged) {
      for (u32 slot = 0; slot < Pattern::live_captures; ++slot)
        captures[Pattern::capture_slot(slot)] = -1;
    }
    for (; pos < input_value.size;) {
      auto character =
        input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(pos);
      auto code_point = character.code_point;
      auto context =
        Pattern::assertion_aware
          ? boundary<Pattern>(input_value, pos, previous_code_point, code_point, character.width)
          : 0;
      auto index = ((state & Pattern::state_mask) * Pattern::boundary_classes + context) *
                     Pattern::class_count +
                   Pattern::classify(code_point);
      auto encoded = Pattern::transition_index(index);
      if constexpr (Pattern::assertion_aware) {
        if (encoded & transition_accept_mask) accepted = pos;
      } else if (encoded & transition_stop_before_mask) {
        break;
      }
      auto next = encoded & Pattern::state_mask;
      if (next == Pattern::dead_state) {
        // A restart state is inspected before advancing the candidate start.
        if constexpr (Pattern::assertion_aware) state = next;
        break;
      }
      if constexpr (Pattern::tagged) Pattern::capture_transition(index, pos, captures);
      previous_code_point = code_point;
      pos += character.width;
      state = next;
      if constexpr (!Pattern::assertion_aware) {
        if (encoded & transition_accept_mask) {
          accepted = pos;
          if constexpr (Pattern::tagged) {
            Pattern::capture_accept(index, pos, captures);
            begin = start;
            end   = pos;
            return true;
          }
        }
      }
    }
    if constexpr (Pattern::assertion_aware) {
      if (pos == input_value.size &&
          Pattern::boundary_accept(state,
                                   boundary<Pattern>(input_value, pos, previous_code_point, 0, 0)))
        accepted = pos;
    }
    if (accepted != -1) {
      begin = start;
      end   = accepted;
      return true;
    }
    if (start == input_value.size) return false;
    if constexpr (Pattern::prefix >= 0) {
      auto base =
        state == Pattern::restart_state && pos > start && pos < input_value.size ? pos : start;
      start =
        input_value.template seek<Pattern::aligned_prefix_seek, !Pattern::aligned_prefix_seek>(
          base + 1, Pattern::prefix);
    } else {
      start = input_value.template advance<Pattern::bytes>(start);
    }
  }
  return false;
}

// Glushkov.
template <typename Pattern>
__device__ __forceinline__ u64 priority_trim(u64 state)
{
  if constexpr (Pattern::accept_mask != 0 &&
                (Pattern::accept_mask & (Pattern::accept_mask - 1)) == 0) {
    // A singleton accept bit makes the priority prefix constant. Unsigned wrap
    // gives an all-ones mask for bit 63, preserving the highest-bit case.
    constexpr u64 priority_mask = Pattern::accept_mask + (Pattern::accept_mask - 1);
    return (state & Pattern::accept_mask) != 0 ? state & priority_mask : state;
  }
  auto accepts = state & Pattern::accept_mask;
  if (accepts == 0) return state;
  auto first = trailing_zeroes(accepts);
  return first == 63 ? state : state & ((u64{1} << (first + 1)) - 1);
}

template <typename Pattern>
__device__ __forceinline__ bool glushkov_boolean(input input_value)
{
  u64 state = 0;
  for (i64 pos = 0; pos < input_value.size;) {
    auto character  = input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(pos);
    auto code_point = character.code_point;
    auto seed       = Pattern::scan_input || pos == 0 ? Pattern::first_set : 0;
    state = (Pattern::follow(state) | seed) & Pattern::reach(Pattern::classify(code_point));
    pos += character.width;
    if constexpr (!Pattern::require_end) {
      if (state & Pattern::accept_mask) return true;
    } else if (state == 0) {
      return false;
    }
  }
  return Pattern::require_end && (state & Pattern::accept_mask) != 0;
}

template <typename Pattern>
__device__ __forceinline__ bool glushkov_find(input input_value, i64 search, i64& begin, i64& end)
{
  if (search >= input_value.size) return false;
  u64 state     = 0;
  i64 saved_end = -1;
  i64 window    = search;
  for (auto pos = search; pos < input_value.size;) {
    if constexpr (Pattern::fixed_match_bytes >= 0 && Pattern::glushkov_prefix >= 0) {
      if (state == 0 && saved_end == -1) pos = input_value.seek(pos, Pattern::glushkov_prefix);
      if (pos == input_value.size) return false;
    }
    auto character  = input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(pos);
    auto code_point = character.code_point;
    auto raw        = (Pattern::follow(state) | (saved_end == -1 ? Pattern::first_set : 0)) &
               Pattern::reach(Pattern::classify(code_point));
    if (state == 0 && saved_end == -1) window = pos;
    pos += character.width;
    if ((raw & Pattern::accept_mask) && (!Pattern::require_end || pos == input_value.size))
      saved_end = pos;
    state = priority_trim<Pattern>(raw);
    if (state == 0 && saved_end != -1) break;
  }
  if (saved_end == -1) return false;
  if constexpr (Pattern::fixed_match_bytes >= 0) {
    begin = saved_end - Pattern::fixed_match_bytes;
    end   = saved_end;
    return true;
  }
  // Recover the earliest start using the same prioritized automaton, without
  // reinjecting starts. Only the last live window from the streaming pass matters.
  for (auto start = window; start < saved_end;
       start      = input_value.template advance<Pattern::bytes>(start)) {
    state        = 0;
    i64 accepted = -1;
    for (auto pos = start; pos < input_value.size;) {
      auto character =
        input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(pos);
      auto code_point = character.code_point;
      auto raw        = (Pattern::follow(state) | (pos == start ? Pattern::first_set : 0)) &
                 Pattern::reach(Pattern::classify(code_point));
      pos += character.width;
      if ((raw & Pattern::accept_mask) && (!Pattern::require_end || pos == input_value.size))
        accepted = pos;
      state = priority_trim<Pattern>(raw);
      if (state == 0) break;
    }
    if (accepted != -1) {
      begin = start;
      end   = accepted;
      return true;
    }
  }
  return false;
}

// Runs.
template <typename Pattern, u32 State>
__device__ __forceinline__ i64 scan_ascii_run(input input_value, i64 pos)
{
  constexpr auto filter = Pattern::run_filter(State);
  if constexpr (filter.kind != ascii_run_kind::GENERAL) {
    // Peel at most seven bytes, then keep every wide load aligned and row-bounded.
    while (pos < input_value.size && (reinterpret_cast<u64>(input_value.data + pos) & 7U)) {
      auto code_point = input_value.byte(pos);
      if (code_point >= 128 || !Pattern::run_predicate(State, code_point)) return pos;
      ++pos;
    }
  }
  while (input_value.size - pos >= 8) {
    u64 characters;
    if constexpr (filter.kind == ascii_run_kind::GENERAL)
      characters = input_value.template search_word<true>(pos);
    else {
      characters = *reinterpret_cast<u64 const*>(input_value.data + pos);
    }
    bool matched;
    if constexpr (filter.kind != ascii_run_kind::GENERAL) {
      matched = (characters & 0x8080808080808080ULL) == 0;
      if constexpr (filter.kind == ascii_run_kind::EXCEPT_BYTE) {
        auto differences =
          characters ^ (static_cast<u64>(filter.excluded_byte) * 0x0101010101010101ULL);
        auto candidates =
          (differences - 0x0101010101010101ULL) & ~differences & 0x8080808080808080ULL;
        matched &= candidates == 0;
      }
    } else {
      matched = true;
#pragma unroll
      for (u32 byte_index = 0; byte_index < 8; ++byte_index) {
        auto code_point = static_cast<u32>((characters >> (byte_index * 8)) & 255U);
        matched &= code_point < 128 && Pattern::run_predicate(State, code_point);
      }
    }
    if (!matched) break;
    pos += 8;
  }
  while (pos < input_value.size) {
    auto code_point = input_value.byte(pos);
    if (code_point >= 128 || !Pattern::run_predicate(State, code_point)) break;
    ++pos;
  }
  return pos;
}

template <typename Pattern, typename StateWord>
__device__ __forceinline__ i64
scan_runs(input input_value, i64 pos, StateWord const* current, i64 count)
{
  if (count == 1) {
    auto state = static_cast<u32>(current[0]);
    while (pos < input_value.size) {
      auto code_point = input_value.byte(pos);
      if (code_point >= 128 || !Pattern::run_predicate(state, code_point)) break;
      ++pos;
    }
    return pos;
  }
  if constexpr (Pattern::multi_run) {
    if (count > 8) return pos;
    u64 low  = ~u64{0};
    u64 high = ~u64{0};
    for (i64 capture_index = 0; capture_index < count; ++capture_index) {
      u64 next_low;
      u64 next_high;
      if (!Pattern::run_masks(
            static_cast<u32>(current[capture_index * (Pattern::live_captures + 1)]),
            next_low,
            next_high))
        return pos;
      low &= next_low;
      high &= next_high;
    }
    while (pos < input_value.size) {
      auto code_point = input_value.byte(pos);
      if (code_point >= 128 || (((code_point < 64 ? low : high) >> (code_point & 63U)) & 1U) == 0)
        break;
      ++pos;
    }
  }
  return pos;
}

// Thompson.
template <typename Pattern>
__device__ __forceinline__ bool mandatory_literal(input input_value)
{
  for (i64 pos = 0; pos <= input_value.size && input_value.size - pos >= Pattern::mandatory_size;
       ++pos)
    if (Pattern::mandatory_at(input_value, pos)) return true;
  return false;
}

// Dispatch callbacks are specialized by the generated Thompson switch. Records
// retain state ID plus only observable captures, with the original bounded layout.
template <bool HasCaptures>
struct record_word_type {
  using type = i64;
};

template <>
struct record_word_type<false> {
  using type = u32;
};

template <typename Pattern>
struct closure {
  using record_word = typename record_word_type<(Pattern::live_captures != 0)>::type;

  record_word* active;
  record_word* stack;
  record_word* next;
  u64* queued;
  i64 depth                   = 0;
  i64 produced                = 0;
  static constexpr u32 record = Pattern::live_captures + 1;

  __device__ __forceinline__ void copy(record_word* target, record_word const* source, u32 state)
  {
    target[0] = state;
    if constexpr (Pattern::live_captures > 0) {
      if constexpr (Pattern::live_captures <= 64) {
        auto live = Pattern::capture_mask(state);
#pragma unroll
        for (u32 capture_index = 0; capture_index < Pattern::live_captures; ++capture_index)
          if (live & (u64{1} << capture_index))
            target[capture_index + 1] = source[capture_index + 1];
      } else {
#pragma unroll
        for (u32 capture_index = 0; capture_index < Pattern::live_captures; ++capture_index)
          target[capture_index + 1] = source[capture_index + 1];
      }
    }
  }

  __device__ __forceinline__ void copy(record_word* target, record_word const* source)
  {
    copy(target, source, static_cast<u32>(source[0]));
  }

  __device__ __forceinline__ void push(u32 state)
  {
    auto* target = stack + depth++ * record;
    copy(target, active, state);
  }

  __device__ __forceinline__ void enqueue(u32 state)
  {
    auto mask  = u64{1} << (state & 63U);
    auto& word = queued[state / 64];
    if (word & mask) return;
    word |= mask;
    auto* target = next + produced++ * record;
    copy(target, active, state);
  }
};

template <typename Pattern>
__device__ __forceinline__ bool thompson_run(
  input input_value, i64 start, i64& end, i64* captures, i64* storage)
{
  constexpr auto record = Pattern::live_captures + 1;
  using record_word     = typename closure<Pattern>::record_word;
  auto* storage_bytes   = reinterpret_cast<u8*>(storage);
  auto* current         = reinterpret_cast<record_word*>(storage_bytes);
  auto* next =
    reinterpret_cast<record_word*>(storage_bytes + Pattern::frontier_words * sizeof(i64));
  auto* stack =
    reinterpret_cast<record_word*>(storage_bytes + u64{2} * Pattern::frontier_words * sizeof(i64));
  auto* active_bytes =
    storage_bytes +
    (u64{2} * Pattern::frontier_words + Pattern::closure_records * record) * sizeof(i64);
  auto* active = reinterpret_cast<record_word*>(active_bytes);
  auto* seen   = reinterpret_cast<u64*>(active_bytes + record * sizeof(i64));
  auto* queued = seen + Pattern::bitset_words;
  current[0]   = Pattern::entry;
  for (u32 capture_index = 0; capture_index < Pattern::live_captures; ++capture_index)
    current[capture_index + 1] =
      captures == nullptr ? -1 : captures[Pattern::capture_slot(capture_index)];
  i64 count  = 1;
  i64 pos    = start;
  bool found = false;
  for (;;) {
    pos             = Pattern::scan_run(input_value, pos, current, count);
    auto character  = input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(pos);
    auto code_point = character.code_point;
    for (u32 capture_index = 0; capture_index < Pattern::bitset_words; ++capture_index)
      seen[capture_index] = queued[capture_index] = 0;
    closure<Pattern> work{active, stack, next, queued};
    bool accepted = false;
    for (i64 seed = 0; seed < count && !accepted; ++seed) {
      work.copy(active, current + seed * record);
      work.depth  = 0;
      bool expand = true;
      while (expand) {
        auto state = static_cast<u32>(active[0]);
        auto mask  = u64{1} << (state & 63U);
        auto& word = seen[state / 64];
        i32 direct = static_cast<i32>(dispatch_status::EXHAUSTED);
        if ((word & mask) == 0) {
          word |= mask;
          direct = Pattern::dispatch(state, input_value, pos, code_point, work);
          if (direct == static_cast<i32>(dispatch_status::ACCEPTED)) {
            accepted = true;
            break;
          }
        }
        if (direct >= 0) {
          active[0] = direct;
          continue;
        }
        if (work.depth == 0) break;
        work.copy(active, stack + --work.depth * record);
      }
    }
    if (accepted) {
      found = true;
      end   = pos;
      if (captures != nullptr)
        for (u32 capture_index = 0; capture_index < Pattern::live_captures; ++capture_index)
          captures[Pattern::capture_slot(capture_index)] = active[capture_index + 1];
    }
    if (work.produced == 0) return found;
    count      = work.produced;
    auto* swap = current;
    current    = next;
    next       = swap;
    pos += character.width;
  }
}

// Dispatch.
template <typename Pattern>
__device__ __forceinline__ bool find(input input_value,
                                     i64 search,
                                     i64& begin,
                                     i64& end,
                                     i64* captures,
                                     i64* external,
                                     char const* flags = nullptr)
{
  if constexpr (is_literal_executor(Pattern::executor)) {
    return literal_find<Pattern>(input_value, search, begin, end);
  } else if constexpr (Pattern::executor == executor_kind::STREAMING_PRIORITIZED_GLUSHKOV) {
    if constexpr (Pattern::repeated_builtin) {
      i64 run_begin = search;
      u32 count     = 0;
      for (auto pos = search; pos < input_value.size;) {
        auto character =
          input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(pos);
        auto code_point = character.code_point;
        pos += character.width;
        if (regex_ir_repeated_builtin(code_point, flags)) {
          if (++count == Pattern::repeat_count) {
            begin = run_begin;
            end   = pos;
            return true;
          }
        } else {
          count     = 0;
          run_begin = pos;
        }
      }
      return false;
    } else {
      return glushkov_find<Pattern>(input_value, search, begin, end);
    }
  } else if constexpr (is_deterministic_executor(Pattern::executor)) {
    return dfa_find<Pattern>(input_value, search, begin, end, captures);
  } else if constexpr (Pattern::executor == executor_kind::WORD_RUN) {
    if constexpr (Pattern::ascii_word_classes) {
      for (auto pos = search; pos < input_value.size;) {
        if (!Pattern::is_word(input_value.byte(pos))) {
          ++pos;
          continue;
        }
        auto start = pos;
        do {
          ++pos;
        } while (pos < input_value.size && Pattern::is_word(input_value.byte(pos)));
        if (pos - start >= Pattern::minimum) {
          begin = start;
          end   = pos;
          return true;
        }
      }
      return false;
    }
    auto position = search;
    while (position < input_value.size) {
      auto character =
        input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(position);
      auto start = position;
      u32 count  = 0;
      while (Pattern::is_word(character.code_point)) {
        ++count;
        position += character.width;
        if (position == input_value.size) break;
        character =
          input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(position);
      }
      if (count >= Pattern::minimum) {
        begin = start;
        end   = position;
        return true;
      }
      // The decoded delimiter also supplies its width; never decode it again.
      position += character.width;
    }
    return false;
  } else if constexpr (Pattern::executor == executor_kind::STRING_OPERATIONS) {
    return Pattern::string_find(input_value, search, begin, end);
  } else {
    if (!Pattern::mandatory_present(input_value)) return false;
    if constexpr (Pattern::simple_capture) {
      if (input_value.size <= 256)
        return Pattern::simple_capture_find(input_value, search, begin, end, captures);
    }
    alignas(8) u8 local[(Pattern::external_workspace ? 1 : Pattern::storage_words) * sizeof(i64)];
    auto* storage = Pattern::external_workspace ? external : reinterpret_cast<i64*>(local);
    for (auto start = search; start <= input_value.size;) {
      if (start < input_value.size && !Pattern::candidate_byte(input_value.byte(start))) {
        if constexpr (Pattern::prefix >= 0)
          start =
            input_value.template seek<Pattern::aligned_prefix_seek, !Pattern::aligned_prefix_seek>(
              start + 1, Pattern::prefix);
        else {
          ++start;
        }
        continue;
      }
      if (thompson_run<Pattern>(input_value, start, end, captures, storage)) {
        begin = start;
        return true;
      }
      if (!Pattern::scan_input || start == input_value.size) return false;
      if constexpr (Pattern::prefix >= 0)
        start =
          input_value.template seek<Pattern::aligned_prefix_seek, !Pattern::aligned_prefix_seek>(
            start + 1, Pattern::prefix);
      else {
        start = input_value.template advance<Pattern::bytes>(start);
      }
    }
    return false;
  }
}

template <typename Pattern>
__device__ __forceinline__ bool boolean(input input_value, i64* workspace)
{
  // The generated transition proof includes every possible classifier result.
  // Empty input still follows the executor's original nullable-match policy.
  if constexpr (Pattern::accepts_any_first) {
    if (input_value.size > 0) return true;
  }

  if constexpr (Pattern::executor == executor_kind::GLUSHKOV)
    return glushkov_boolean<Pattern>(input_value);
  else if constexpr (is_deterministic_executor(Pattern::executor)) {
    return dfa_boolean<Pattern>(input_value);
  } else if constexpr (Pattern::executor == executor_kind::STRING_OPERATIONS) {
    return Pattern::string_boolean(input_value);
  } else {
    i64 begin;
    i64 end;
    if constexpr (is_literal_executor(Pattern::executor) && !Pattern::scan_input)
      return Pattern::literal_at(input_value, 0);
    return find<Pattern>(input_value, 0, begin, end, nullptr, workspace);
  }
}

// Operations.
template <typename Pattern>
__device__ __forceinline__ bool captures(input input_value, i64 search, i64* output, i64* workspace)
{
  for (u32 capture_index = 0; capture_index < Pattern::capture_slots; ++capture_index)
    output[capture_index] = -1;
  i64 begin;
  i64 end;
  auto matched = find<Pattern>(input_value, search, begin, end, output, workspace);
  if (matched) {
    output[0] = begin;
    output[1] = end;
    Pattern::whole_captures(output, begin, end);
  }
  return matched;
}

template <typename Pattern, unicode_decoder_kind DecoderKind>
__device__ __forceinline__ i64 count_repeated_predicate(
  input input_value, i64 position, u32 run_length, i64 total, char const* flags)
{
  buffered_unicode_input reader{input_value};
  for (; position < input_value.size;) {
    auto character =
      DecoderKind == unicode_decoder_kind::BUFFERED
        ? reader.read(position)
        : input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(position);
    position += character.width;
    if (regex_ir_repeated_builtin(character.code_point, flags)) {
      if (++run_length == Pattern::repeat_count) {
        ++total;
        run_length = 0;
      }
    } else {
      run_length = 0;
    }
  }
  return total;
}

template <typename Pattern>
__device__ __forceinline__ i64 count(input input_value, i64* workspace, char const* flags = nullptr)
{
  if constexpr (Pattern::repeated_builtin) {
    if constexpr (Pattern::bytes || !Pattern::fused_unicode_decode) {
      return count_repeated_predicate<Pattern, unicode_decoder_kind::SCALAR>(
        input_value, 0, 0, 0, flags);
    }
    constexpr i64 buffered_row_bytes = 256;
    if (input_value.size < buffered_row_bytes) {
      return count_repeated_predicate<Pattern, unicode_decoder_kind::SCALAR>(
        input_value, 0, 0, 0, flags);
    }
    constexpr u32 scalar_prefix_characters = 16;
    i64 position                           = 0;
    u32 run_length                         = 0;
    i64 total                              = 0;
    // Count the prefix normally, carrying partial runs into the remaining scan.
    // Completed prefix matches retain the scalar path for dense inputs.
    for (u32 character_index = 0;
         character_index < scalar_prefix_characters && position < input_value.size;
         ++character_index) {
      auto character =
        input_value.template read<Pattern::bytes, Pattern::fused_unicode_decode>(position);
      position += character.width;
      if (regex_ir_repeated_builtin(character.code_point, flags)) {
        if (++run_length == Pattern::repeat_count) {
          ++total;
          run_length = 0;
        }
      } else {
        run_length = 0;
      }
    }
    if (total != 0) {
      return count_repeated_predicate<Pattern, unicode_decoder_kind::SCALAR>(
        input_value, position, run_length, total, flags);
    }
    return count_repeated_predicate<Pattern, unicode_decoder_kind::BUFFERED>(
      input_value, position, run_length, total, flags);
  }
  if constexpr (Pattern::executor == executor_kind::SINGLE_BYTE_LITERAL && Pattern::scan_input &&
                !Pattern::require_end) {
    auto needle                  = Pattern::literal_byte(0);
    auto repeated                = static_cast<u64>(needle) * 0x0101010101010101ULL;
    constexpr u64 byte_low_bits  = 0x7f7f7f7f7f7f7f7fULL;
    constexpr u64 byte_high_bits = 0x8080808080808080ULL;
    constexpr i64 packed_bytes   = sizeof(u64);
    constexpr u64 alignment_mask = packed_bytes - 1;
    i64 total                    = 0;
    i64 position                 = 0;
    while (position < input_value.size &&
           (reinterpret_cast<u64>(input_value.data + position) & alignment_mask)) {
      total += input_value.byte(position) == needle;
      ++position;
    }
    while (input_value.size - position >= packed_bytes) {
      auto characters  = *reinterpret_cast<u64 const*>(input_value.data + position);
      auto differences = characters ^ repeated;
      // Exact zero-byte detection avoids borrow propagation into neighboring bytes.
      auto matching_bytes =
        ~(((differences & byte_low_bits) + byte_low_bits) | differences) & byte_high_bits;
      total += __popcll(matching_bytes);
      position += packed_bytes;
    }
    while (position < input_value.size) {
      total += input_value.byte(position) == needle;
      ++position;
    }
    return total;
  }
  i64 total  = 0;
  i64 search = 0;
  i64 begin;
  i64 end;
  while (find<Pattern>(input_value, search, begin, end, nullptr, workspace, flags)) {
    ++total;
    if (end != begin)
      search = end;
    else {
      if (end == input_value.size) break;
      search = input_value.template advance<Pattern::bytes>(end);
    }
  }
  return total;
}

template <typename Pattern>
__device__ __forceinline__ i64 replace(input input_value, char* output, i64* workspace)
{
  i64 capture[Pattern::capture_slots];
  i64 search = 0;
  i64 copied = 0;
  i64 cursor = 0;
  while (captures<Pattern>(input_value, search, capture, workspace)) {
    auto begin = capture[0];
    auto end   = capture[1];
    cursor     = append(input_value.data, copied, begin, output, cursor);
    cursor     = Pattern::replacement(input_value, capture, output, cursor);
    copied     = end;
    if (begin != end)
      search = end;
    else {
      if (end == input_value.size) break;
      search = input_value.template advance<Pattern::bytes>(end);
    }
  }
  return append(input_value.data, copied, input_value.size, output, cursor);
}

template <typename Pattern>
__device__ __forceinline__ i64 split(input input_value,
                                     i64* spans,
                                     i64* workspace,
                                     i64 limit      = -1,
                                     i64 capacity   = -1,
                                     char* overflow = nullptr)
{
  i64 search = 0;
  i64 copied = 0;
  i64 count  = 0;
  i64 begin;
  i64 end;
  auto write = [&](i64 end) {
    if (spans == nullptr) return;
    if (capacity >= 0 && count >= capacity) {
      if (overflow != nullptr) *overflow = 1;
      return;
    }
    spans[2 * count]     = copied;
    spans[2 * count + 1] = end;
  };
  while ((limit <= 0 || count < limit) &&
         find<Pattern>(input_value, search, begin, end, nullptr, workspace)) {
    write(begin);
    ++count;
    copied = end;
    if (begin != end)
      search = end;
    else {
      if (end == input_value.size) break;
      search = input_value.template advance<Pattern::bytes>(end);
    }
  }
  write(input_value.size);
  return count + 1;
}
}  // namespace regex_ir::device
