/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "pair.cuh"

#include <executor.cuh>

namespace cudf::experimental::detail::regex_jit::device {
using namespace regex_ir::device;

struct worker {
  i32 index = threadIdx.x + blockDim.x * blockIdx.x;
  i32 end;
  char* workspace;

  __device__ worker(char* scratch       = nullptr,
                    i32 begin           = 0,
                    i32 limit           = 2147483647,
                    i64 workspace_bytes = 0)
    : end(limit), workspace(scratch == nullptr ? nullptr : scratch + i64(index) * workspace_bytes)
  {
    index += begin;
  }

  __device__ bool active(i32 rows) const { return index < rows && index < end; }
};

template <typename Offset>
struct column {
  char const* chars;
  Offset const* offsets;
  u32 const* mask;
  i32 row_offset;
  i32 rows;

  __device__ bool valid(i32 row) const
  {
    auto physical = row + row_offset;
    return mask == nullptr || (mask[physical / 32] & (1u << (physical % 32)));
  }

  __device__ input load(i32 row) const
  {
    auto physical = row + row_offset;
    return {chars + offsets[physical], i64(offsets[physical + 1]) - offsets[physical]};
  }

  __device__ string_pair pair(input input_value, i64 begin, i64 end) const
  {
    if (begin < 0 || end < begin) return {nullptr, 0};
    return {end == begin ? reinterpret_cast<char const*>(offsets) : input_value.data + begin,
            i32(end - begin)};
  }
};

__device__ inline i64 advance(input input_value, i64 pos)
{
  auto strings = input_value.byte(pos);
  auto width   = (strings & 0xe0) == 0xc0   ? 2
                 : (strings & 0xf0) == 0xe0 ? 3
                 : (strings & 0xf8) == 0xf0 ? 4
                                            : 1;
  return pos + width < input_value.size ? pos + width : input_value.size;
}

__device__ inline i32 character_index(input input_value, i64 byte_offset)
{
  i32 count          = 0;
  i64 position       = 0;
  auto aligned_input = (reinterpret_cast<u64>(input_value.data) & 7U) == 0;
  while (byte_offset - position >= 8) {
    auto characters = aligned_input ? *reinterpret_cast<u64 const*>(input_value.data + position)
                                    : load_unaligned<u64>(input_value.data + position);
    if ((characters & 0x8080808080808080ULL) != 0) break;
    position += 8;
    count += 8;
  }
  // Keep the existing character-width behavior after the first non-ASCII block.
  for (; position < byte_offset; position = advance(input_value, position))
    ++count;
  return count;
}

__device__ inline bool next(input input_value, i64 begin, i64 end, i64& search)
{
  if (begin != end) {
    search = end;
    return true;
  }
  if (end == input_value.size) return false;
  search = advance(input_value, end);
  return true;
}

template <typename Matcher, fixed_output_kind Kind, bool Builtin, typename Column>
__device__ void fixed(worker worker_state,
                      Column strings,
                      void* output,
                      char const* flags = nullptr)
{
  if (!worker_state.active(strings.rows)) return;
  auto row = worker_state.index;
  if constexpr (Kind == fixed_output_kind::BOOLEAN) {
    static_cast<char*>(output)[row] =
      strings.valid(row) &&
      Matcher::call(worker_state.workspace, strings.load(row).data, strings.load(row).size);
  } else if constexpr (Kind == fixed_output_kind::COUNT) {
    i64 value = 0;
    if (strings.valid(row)) {
      auto input_value = strings.load(row);
      if constexpr (Builtin)
        value = Matcher::call(worker_state.workspace, input_value.data, input_value.size, flags);
      else {
        value = Matcher::call(worker_state.workspace, input_value.data, input_value.size);
      }
    }
    static_cast<i32*>(output)[row] = i32(value);
  } else {
    i32 value = -1;
    if (strings.valid(row)) {
      auto input_value = strings.load(row);
      i64 spans[2];
      if (Matcher::call(worker_state.workspace, input_value.data, input_value.size, spans)) {
        value = character_index(input_value, spans[0]);
      }
    }
    static_cast<i32*>(output)[row] = value;
  }
}

template <typename Matcher, i32 Slots, i32 First, i32 Groups, bool ColumnMajor, typename Column>
__device__ void capture(worker worker_state, Column strings, string_pair* output)
{
  if (!worker_state.active(strings.rows)) return;
  auto row = worker_state.index;
  i64 spans[Slots];
  input input_value{};
  bool matched = false;
  if (strings.valid(row)) {
    input_value = strings.load(row);
    matched = Matcher::call(worker_state.workspace, input_value.data, input_value.size, 0, spans);
  }
  for (i32 group = 0; group < Groups; ++group) {
    auto slot  = 2 * (First + group + 1);
    auto index = ColumnMajor ? i64(group) * strings.rows + row : row;
    output[index] =
      matched ? strings.pair(input_value, spans[slot], spans[slot + 1]) : string_pair{nullptr, 0};
  }
}

template <i32 Slots>
__device__ void cache_match(
  i64* cache, i32 row, i32 count, i32 capacity, char* overflow, i64 const* spans)
{
  if (count >= capacity) {
    overflow[row] = 1;
    return;
  }
  auto record = cache + (i64(row) * capacity + count) * Slots;
  for (i32 slot = 0; slot < Slots; ++slot)
    record[slot] = spans[slot];
}

template <typename Matcher, i32 Slots, i32 Multiplier, bool Require, bool Cache, typename Column>
__device__ void enumeration_size(worker worker_state,
                                 Column strings,
                                 i32* sizes,
                                 char* validity,
                                 i64* cache     = nullptr,
                                 i32 capacity   = 0,
                                 char* overflow = nullptr)
{
  if (!worker_state.active(strings.rows)) return;
  auto row   = worker_state.index;
  i32 count  = 0;
  bool valid = strings.valid(row);
  if (valid) {
    auto input_value = strings.load(row);
    i64 spans[Slots];
    i64 search = 0;
    while (
      Matcher::call(worker_state.workspace, input_value.data, input_value.size, search, spans)) {
      if constexpr (Cache) cache_match<Slots>(cache, row, count, capacity, overflow, spans);
      ++count;
      if (!next(input_value, spans[0], spans[1], search)) break;
    }
  }
  sizes[row]    = count * Multiplier;
  validity[row] = valid && (!Require || count != 0);
}

template <typename Matcher, i32 Slots, i32 Groups, bool FindAll, bool OverflowOnly, typename Column>
__device__ void enumeration_emit(worker worker_state,
                                 Column strings,
                                 string_pair* output,
                                 i32 const* offsets,
                                 char const* overflow = nullptr)
{
  if (!worker_state.active(strings.rows) || !strings.valid(worker_state.index)) return;
  auto row = worker_state.index;
  if constexpr (OverflowOnly)
    if (!overflow[row]) return;
  auto input_value = strings.load(row);
  i64 spans[Slots];
  i64 search = 0;
  i64 cursor = offsets[row];
  while (Matcher::call(worker_state.workspace, input_value.data, input_value.size, search, spans)) {
    if constexpr (FindAll) {
      constexpr i32 slot = Groups == 0 ? 0 : 2;
      output[cursor++]   = strings.pair(input_value, spans[slot], spans[slot + 1]);
    } else {
      for (i32 group = 1; group <= Groups; ++group)
        output[cursor++] = strings.pair(input_value, spans[group * 2], spans[group * 2 + 1]);
    }
    if (!next(input_value, spans[0], spans[1], search)) break;
  }
}

template <typename Matcher, bool Emit, typename OutOffset, typename Column>
__device__ void replace(worker worker_state,
                        Column strings,
                        void* output,
                        OutOffset const* offsets = nullptr)
{
  if (!worker_state.active(strings.rows)) return;
  auto row = worker_state.index;
  if constexpr (Emit) {
    if (strings.valid(row)) {
      auto input_value = strings.load(row);
      Matcher::call(worker_state.workspace,
                    input_value.data,
                    input_value.size,
                    static_cast<char*>(output) + offsets[row]);
    }
  } else {
    i64 size = 0;
    if (strings.valid(row)) {
      auto input_value = strings.load(row);
      size = Matcher::call(worker_state.workspace, input_value.data, input_value.size, nullptr);
    }
    static_cast<i32*>(output)[row] = i32(size);
  }
}

template <typename Matcher,
          typename Replacement,
          i32 Slots,
          i32 Limit,
          bool Emit,
          bool Cache,
          typename OutOffset,
          typename Column>
__device__ void limited_replace(worker worker_state,
                                Column strings,
                                void* output,
                                OutOffset const* offsets,
                                i64* cache,
                                i32 capacity,
                                char* overflow,
                                i32* counts,
                                i32* size_overflow)
{
  if (!worker_state.active(strings.rows)) return;
  auto row   = worker_state.index;
  i64 cursor = 0;
  i64 copied = 0;
  i64 search = 0;
  i32 count  = 0;
  if (strings.valid(row)) {
    auto input_value  = strings.load(row);
    char* destination = Emit ? static_cast<char*>(output) + offsets[row] : nullptr;
    bool cached       = false;
    if constexpr (Emit && Cache) cached = !overflow[row];
    i64 spans[Slots];
    while (count < Limit) {
      if (cached) {
        if (count >= counts[row]) break;
        auto record = cache + (i64(row) * capacity + count) * Slots;
        for (i32 slot = 0; slot < Slots; ++slot)
          spans[slot] = record[slot];
      } else if (!Matcher::call(
                   worker_state.workspace, input_value.data, input_value.size, search, spans)) {
        break;
      }
      if constexpr (Cache && !Emit)
        cache_match<Slots>(cache, row, count, capacity, overflow, spans);
      cursor = append(input_value.data, copied, spans[0], destination, cursor);
      cursor = Replacement::apply(input_value, spans, destination, cursor);
      copied = spans[1];
      ++count;
      if (!next(input_value, spans[0], spans[1], search)) break;
    }
    cursor = append(input_value.data, copied, input_value.size, destination, cursor);
  }
  if constexpr (!Emit) {
    static_cast<i32*>(output)[row] = i32(cursor);
    if (cursor > 2147483647) atomicMax(size_overflow, 1);
    if constexpr (Cache) counts[row] = count;
  }
}

template <typename Matcher, i32 Limit, bool Cache, typename Column>
__device__ void split_size(worker worker_state,
                           Column strings,
                           i32* output,
                           i64* cache     = nullptr,
                           i32 capacity   = 0,
                           char* overflow = nullptr)
{
  if (!worker_state.active(strings.rows)) return;
  auto row  = worker_state.index;
  i64 count = 0;
  if (strings.valid(row)) {
    auto input_value = strings.load(row);
    auto spans       = Cache ? cache + i64(row) * (capacity + 1) * 2 : nullptr;
    count            = Matcher::call(worker_state.workspace,
                          input_value.data,
                          input_value.size,
                          spans,
                          Limit,
                          Cache ? capacity + 1 : -1,
                          Cache ? overflow + row : nullptr);
  }
  output[row] = i32(count);
}

template <typename Matcher, i32 Limit, bool Reverse, bool OverflowOnly, typename Column>
__device__ void split_emit(worker worker_state,
                           Column strings,
                           string_pair* output,
                           i32 const* effective,
                           i32 const* full,
                           i64* buffer,
                           char const* overflow = nullptr)
{
  if (!worker_state.active(strings.rows) || !strings.valid(worker_state.index)) return;
  auto row = worker_state.index;
  if constexpr (OverflowOnly)
    if (!overflow[row]) return;
  auto input_value = strings.load(row);
  auto spans       = buffer + i64(full[row]) * 2;
  Matcher::call(worker_state.workspace,
                input_value.data,
                input_value.size,
                spans,
                Reverse ? -1 : Limit,
                -1,
                nullptr);
  i32 field_count = effective[row + 1] - effective[row];
  i32 total       = full[row + 1] - full[row];
  for (i32 index = 0; index < field_count; ++index) {
    i32 source = Reverse && field_count < total && index != 0 ? total - field_count + index : index;
    auto begin = spans[source * 2];
    auto end   = spans[source * 2 + 1];
    if constexpr (Reverse) {
      if (field_count < total && index == 0) end = spans[(total - field_count) * 2 + 1];
    } else {
      if (field_count < total && index == field_count - 1) end = input_value.size;
    }
    output[effective[row] + index] = strings.pair(input_value, begin, end);
  }
}

template <typename Matcher, bool Split, i32 Slots, i32 Limit, typename Column>
__device__ void sample(worker worker_state, Column strings, i32 samples, i32 capacity, u64* stats)
{
  if (!worker_state.active(samples)) return;
  auto row = i32((u64(worker_state.index) * strings.rows) / samples);
  if (!strings.valid(row)) return;
  auto input_value = strings.load(row);
  i64 count        = 0;
  if constexpr (Split)
    count =
      Matcher::call(
        worker_state.workspace, input_value.data, input_value.size, nullptr, Limit, -1, nullptr) -
      1;
  else {
    i64 spans[Slots];
    i64 search = 0;
    while (
      (Limit <= 0 || count < Limit) &&
      Matcher::call(worker_state.workspace, input_value.data, input_value.size, search, spans)) {
      ++count;
      if (count > capacity || !next(input_value, spans[0], spans[1], search)) break;
    }
  }
  atomicAdd(stats, 1ULL);
  atomicAdd(stats + 1, static_cast<u64>(input_value.size));
  atomicAdd(stats + 2, static_cast<u64>(count > capacity ? capacity + 1 : count));
  atomicAdd(stats + 3, static_cast<u64>(count > capacity));
}

template <typename Pattern, typename Column>
__device__ void warp_literal(Column strings, char* output)
{
  i32 lane = threadIdx.x % 32;
  i32 row  = (threadIdx.x + blockDim.x * blockIdx.x) / 32;
  if (row >= strings.rows) return;
  bool found = false;
  if (strings.valid(row)) {
    auto input_value = strings.load(row);
    i64 base         = 0;
    i64 last         = input_value.size - Pattern::size;
    bool first_round = true;
    while (base <= last) {
      auto lane_position = base + (first_round ? lane : lane * 4);
      auto candidates    = first_round ? 1 : 4;
      bool hit           = false;
#pragma unroll 1
      for (i32 candidate_index = 0; candidate_index < candidates; ++candidate_index) {
        auto pos = lane_position + candidate_index;
        if (pos > last) break;
        if (input_value.byte(pos + Pattern::pivot) == Pattern::anchor &&
            Pattern::matches(input_value, pos)) {
          hit = true;
          break;
        }
      }
      if (__any_sync(0xffffffffu, hit)) {
        found = true;
        break;
      }
      base += first_round ? 32 : 128;
      first_round = false;
    }
  }
  if (lane == 0) output[row] = found;
}
}  // namespace cudf::experimental::detail::regex_jit::device
