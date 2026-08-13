/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"

#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/bit>
#include <cuda/std/array>
#include <cuda/std/optional>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

/**
 * @brief Device-side Glushkov NFA program data structure.
 *
 * The device program is a compact representation of the Glushkov NFA:
 *  - State is a 64-bit bitmask (one bit per position, no working memory needed).
 *  - Transition = shift-and + exception lookup (Hyperscan technique).
 *  - Character matching reuses the existing reclass_device::is_match path.
 */

namespace cudf::strings::detail {

struct gkprog;

/**
 * @brief Block-shared cache of read-only Glushkov program data.
 *
 * Loaded by thread-0 via gkprog_device::load/store calls.
 * Subsequent calls to gkprog_device::find read from shared memory (~20-cycle latency)
 * instead of global/L2 (~100 cycles).
 *
 * Total size: ~1608 bytes — negligible impact on occupancy.
 */
struct glushkov_shmem_cache {
  cuda::std::array<glushkov_state_t, GLUSHKOV_ASCII_TABLE_SIZE>
    reach_ascii;  ///< precomputed ASCII reach masks
  cuda::std::array<glushkov_state_t, GLUSHKOV_MAX_SHIFTS>
    shift_masks;  ///< [shift_count] shift-and source masks
  cuda::std::array<glushkov_state_t, GLUSHKOV_MAX_STATES>
    exception_successors;  ///< [num_states] exception successor masks
  cuda::std::array<uint8_t, GLUSHKOV_MAX_SHIFTS> shift_amounts;  ///< [shift_count] shift amounts
};

/**
 * @brief GPU-resident Glushkov NFA program.
 *
 * All array pointers address device-global memory packed into a single flat
 * buffer (see gkprog_device::create).
 */
struct gkprog_device {
 public:
  gkprog_device()                                = default;
  ~gkprog_device()                               = default;
  gkprog_device(gkprog_device const&)            = default;
  gkprog_device(gkprog_device&&)                 = default;
  gkprog_device& operator=(gkprog_device const&) = default;
  gkprog_device& operator=(gkprog_device&&)      = default;

  static std::unique_ptr<gkprog_device, std::function<void(gkprog_device*)>> create(
    gkprog const& prog, rmm::cuda_stream_view stream);

  /**
   * @brief Called automatically by the unique_ptr returned from create().
   */
  void destroy();

  [[nodiscard]] __device__ inline bool is_empty() const { return false; }

  [[nodiscard]] CUDF_HOST_DEVICE bool is_empty_match_possible() const { return false; }

  /**
   * @brief Stores this instance into the given buffer (shared_memory pointer)
   */
  __device__ inline void store(void* buffer) const;

  /**
   * @brief Load an instance from a device buffer (e.g. shared memory).
   */
  [[nodiscard]] __device__ static inline gkprog_device load(gkprog_device const prog, void* buffer);

  /**
   * @brief Returns the size of shared memory required to hold this instance.
   *
   * This can be called on the CPU for specifying the shared-memory size in the
   * kernel launch parameters.
   */
  [[nodiscard]] int32_t compute_shared_memory_size() const;

  /**
   * @brief Returns amount of working memory needed to hold state for a stride of threads.
   */
  [[nodiscard]] std::pair<std::size_t, int32_t> compute_strided_working_memory(
    int32_t rows, int32_t min_rows = 0, std::size_t requested_max_size = 0) const;

  /**
   * @brief Sets the working memory pointer for this instance.
   */
  void set_working_memory(void* buffer, int32_t thread_count, int32_t max_insts = 0);

  /**
   * @brief Returns number of threads set by set_working_memory
   */
  [[nodiscard]] __device__ inline int32_t thread_count() const { return _thread_count; }

  /**
   * @brief Does a find evaluation using the compiled expression on the given string.
   *
   * @tparam P Desired positional values. Default includes valid begin and end match positions.
   * @param thread_idx The index used for mapping the state memory for this string in global memory.
   * @param d_str The string to search.
   * @param begin Position to begin the search within `d_str`.
   * @param end Character position index to end the search within `d_str`.
   *            Specify -1 to match any virtual positions past the end of the string.
   * @return If match found, returns character positions of the matches.
   */
  template <positional P = positional::BEGIN_END>
  [[nodiscard]] __device__ inline match_result find(int32_t const thread_idx,
                                                    string_view const d_str,
                                                    string_view::const_iterator begin,
                                                    cudf::size_type end = -1) const;

  uint32_t num_states{};                      ///< Number of Glushkov positions (≤ 64)
  uint32_t shift_count{};                     ///< Number of shift slots (≤ GLUSHKOV_MAX_SHIFTS_DEV)
  glushkov_state_t first_set{};               ///< Initial active positions (before first character)
  glushkov_state_t accept_mask{};             ///< Positions whose match completes the pattern
  glushkov_state_t exception_mask{};          ///< Positions with non-shift follow transitions
  cuda::std::optional<char32_t> startchar{};  ///< First literal

  uint8_t const* _codepoint_flags{};       ///< Character-type lookup table (shared with Thompson)
  reclass_device const* _classes{};        ///< Character class data
  reinst const* _positions{};              ///< [num_states] per-position descriptors
  glushkov_state_t const* _shift_masks{};  ///< [shift_count] shift-and source masks
  uint8_t const* _shift_amounts{};         ///< [shift_count] shift amounts
  glushkov_state_t const* _reach_ascii{};  ///< precomputed reach bitmasks for ASCII chars
  glushkov_state_t const* _exception_successors{};  ///< [num_states] exception successor masks

  std::size_t _prog_size{};  ///< Total buffer size (for potential shmem loading)
  int32_t _thread_count{};   ///< Used for strided kernel loops

  [[nodiscard]] __device__ glushkov_state_t priority_kill(glushkov_state_t const state,
                                                          glushkov_state_t const accept_mask) const;
  [[nodiscard]] __device__ bool position_matches(reinst const& pos,
                                                 char32_t const c,
                                                 reclass_device const* classes,
                                                 uint8_t const* codepoint_flags) const;
  [[nodiscard]] __device__ glushkov_state_t compute_follow_impl(glushkov_state_t const state) const;
  [[nodiscard]] __device__ glushkov_state_t compute_reach_impl(char32_t const c) const;
};

/**
 * Matching algorithm
 * ------------------
 * One thread processes one input string.  The NFA state is a single glushkov_state_t
 * (uint64_t) bitmask held in a register – no global working memory required.
 *
 * Unanchored search: two-phase O(n) per match.
 *   Phase 1 (forward scan): inject first_set at every character; track the
 *     earliest injection position (inject_start_pos) since state was last 0.
 *     When state & accept_mask fires, record the match end; continue extending
 *     greedily until state dies.
 *   Phase 2 (BEGIN_END only): rescan from inject_start_pos, trying each valid
 *     start position with a single-seed NFA until one reaches accept.  The
 *     earliest start that accepts gives the leftmost match.
 *   END_ONLY (contains_re, matches_re): phase 1 result is returned directly;
 *     the start position is not needed so phase 2 is skipped.
 *
 * Anchored (end >= 0): inner-loop from start_pos, already O(n) per string.
 *
 *
 * The shift-and follow computation (Hyperscan-style):
 *   follow(state) ≈ OR_k( (state & shift_masks[k]) << shift_amounts[k] )
 *                   | OR over active exception states (exception_successors lookup)
 *
 */

/**
 * @brief Emulate Thompson's first-alternative-wins priority by killing lower-priority states.
 *
 * When multiple alternatives can accept at the same position, the lowest-indexed accepting
 * position corresponds to the highest-priority alternative.  All bits above the lowest accepting
 * bit are cleared so that lower-priority paths are killed while self-loops and continuations of the
 * winning path survive.
 *
 * @param state       Current NFA state bitmask.
 * @param accept_mask Bitmask of accepting positions.
 * @return            State with lower-priority alternatives killed.
 */
__device__ __forceinline__ glushkov_state_t
gkprog_device::priority_kill(glushkov_state_t const state, glushkov_state_t const accept_mask) const
{
  auto const lsb_a = static_cast<uint32_t>(cuda::std::countr_zero(state & accept_mask));
  return (lsb_a < 63) ? (state & ((glushkov_state_t(1) << (lsb_a + 1)) - 1)) : state;
}

/**
 * @brief Returns true if character @p c is matched by Glushkov position @p pos.
 *
 * Mirrors the character-dispatch logic in the Thompson NFA regexec loop.
 */
__device__ __forceinline__ bool gkprog_device::position_matches(
  reinst const& pos,
  char32_t const c,
  reclass_device const* classes,
  uint8_t const* codepoint_flags) const
{
  switch (pos.type) {
    case CHAR: return c == pos.u1.c;
    case ANY: {
      // pos.ch == 'N' → EXT_NEWLINE: reject all is_newline() characters
      // otherwise     → default: reject only '\n' (matches Thompson NFA)
      return (pos.u1.c == 'N') ? !is_newline(c) : (c != '\n');
    }
    case ANYNL: return true;
    case CCLASS: return classes[pos.u1.cls_id].is_match(c, codepoint_flags);
    case NCCLASS: return !classes[pos.u1.cls_id].is_match(c, codepoint_flags);
    default: return false;
  }
}

/**
 * @brief Compute the Glushkov follow set for the current state bitmask.
 *
 * Two-phase:
 *   1. Shift masks  – fast, covers most forward transitions.
 *   2. Exception table – backward and large-span transitions.
 */
__device__ __forceinline__ glushkov_state_t
gkprog_device::compute_follow_impl(glushkov_state_t const state) const
{
  glushkov_state_t follow = 0;

  // Phase 1: shift-and transitions
  for (uint32_t k = 0; k < shift_count; ++k) {
    follow |= (state & _shift_masks[k]) << _shift_amounts[k];
  }

  // Phase 2: exception transitions (backward / large-span)
  glushkov_state_t exc = state & exception_mask;
  while (exc) {
    auto const p = cuda::std::countr_zero(exc);
    exc &= exc - 1;  // clear lowest set bit
    follow |= _exception_successors[p];
  }

  return follow;
}

/**
 * @brief Compute the reach bitmask for character @p c.
 *
 * reach(c) = { p | position p matches character c }
 *
 * Fast path (c < GLUSHKOV_ASCII_TABLE_SIZE): single lookup into the precomputed
 *   reach_ascii table — O(1).
 * Slow path (non-ASCII): O(num_states) loop over position descriptors.
 */
__device__ __forceinline__ glushkov_state_t
gkprog_device::compute_reach_impl(char32_t const c) const
{
  if (c < static_cast<uint32_t>(GLUSHKOV_ASCII_TABLE_SIZE)) { return _reach_ascii[c]; }

  // Non-ASCII fallback: must use prog._positions (global memory)
  glushkov_state_t reach = 0;
  for (uint32_t p = 0; p < num_states; ++p) {
    if (position_matches(_positions[p], c, _classes, _codepoint_flags)) {
      reach |= glushkov_state_t(1) << p;
    }
  }
  return reach;
}

/**
 * @brief Glushkov NFA find: locate the leftmost (greedy) match in @p d_str.
 *
 * Semantics match reprog_device::find (Thompson NFA) as closely as possible:
 *   - Leftmost start position.
 *   - Greedy (longest match from that start).
 *
 * For unanchored patterns, uses a two-phase O(n) algorithm:
 * Phase 1 injects first_set at every character and extends greedily; when
 * state & accept_mask fires, the match end is recorded.  Phase 2 (BEGIN_END
 * only) rescans from inject_start_pos (the earliest injection position since
 * state was last 0) to find the leftmost valid start for that match end.
 *
 *
 * @tparam P     Positional mode (BEGIN_END or END_ONLY)
 *
 * @param d_str  Input string.
 * @param begin  Iterator to the first character to consider.
 * @param end    Max start-position to try (−1 = try all positions).
 *               `end=1` means only try starting at 0
 *               (used by matches_re / beginning_only)
 * @return       Match pair or nullopt.
 */
template <positional P>
__device__ __forceinline__ match_result gkprog_device::find(int32_t const,
                                                            string_view const d_str,
                                                            string_view::const_iterator begin,
                                                            cudf::size_type end) const
{
  auto const size_bytes = static_cast<int32_t>(d_str.size_bytes());  // O(1)

  int32_t const start_pos = begin.position();

  // --- Anchored (end >= 0): inner-loop approach, O(n) ---
  // Outer loop runs at most (end - start_pos) iterations (typically 1 for matches_re).
  if (end >= 0) {
    auto outer_itr = begin;
    for (int32_t start = start_pos;; ++start, ++outer_itr) {
      if (start >= end) { break; }
      if (outer_itr.byte_offset() >= size_bytes) { break; }

      if (startchar.has_value()) {
        while (outer_itr.byte_offset() < size_bytes && *outer_itr != startchar.value()) {
          ++outer_itr;
          ++start;
        }
        if (outer_itr.byte_offset() >= size_bytes) { break; }
        if (start >= end) { break; }
      } else {
        char32_t const first_c = *outer_itr;
        if ((compute_reach_impl(first_c) & first_set) == 0) { continue; }
      }

      match_result cur_match{};

      glushkov_state_t state = first_set;
      auto inner_itr         = outer_itr;
      for (int32_t pos = start; inner_itr.byte_offset() < size_bytes; ++pos, ++inner_itr) {
        char32_t const c = *inner_itr;
        if (pos == start) {
          state = state & compute_reach_impl(c);
        } else {
          state = compute_follow_impl(state) & compute_reach_impl(c);
        }
        if (state == 0) { break; }
        if (state & accept_mask) {
          if constexpr (P == positional::BEGIN_END) {
            cur_match = match_pair{start, pos + 1};
          } else {
            cur_match = match_pair{-1, pos + 1};
          }
          state = priority_kill(state, accept_mask);
        }
      }
      if (cur_match) { return cur_match; }
    }
    return {};
  }

  // --- Unanchored: two-phase O(n) search ---
  //
  // Phase 1 (forward scan): inject first_set at every step; detect match END
  // via state & accept_mask (any active seed reaching accept); greedy extension
  // until state dies.
  //
  // inject_start_pos records the earliest position where seeds were injected
  // since state was last 0.  Phase 2 rescans from there to pinpoint the start.
  //
  // Phase 2 (BEGIN_END only): rescan from inject_start_pos, trying each valid
  // start position with a single-seed NFA (mirrors the anchored inner loop).
  // The first start whose NFA reaches accept gives the leftmost match.
  //
  // Cost: O(n) for phase 1; O(gap + match_length) per match for phase 2.
  // Each character is processed at most twice overall.

  glushkov_state_t state = 0;
  match_result cur_match{};

  // Earliest position where seeds were injected since state was last 0.
  int32_t inject_start_pos = start_pos;
  auto inject_start_itr    = begin;

  auto itr = begin;
  for (int32_t pos = start_pos; itr.byte_offset() < size_bytes; ++pos, ++itr) {
    // Fast skip when no active states and no match found yet.
    if (state == 0 && !cur_match) {
      if (startchar.has_value()) {
        // Tight byte scan for the literal start character.
        while (itr.byte_offset() < size_bytes && *itr != startchar.value()) {
          ++itr;
          ++pos;
        }
        if (itr.byte_offset() >= size_bytes) { break; }
      }
      // Record injection start for phase 2 rescan.
      inject_start_pos = pos;
      inject_start_itr = itr;
    }

    char32_t const c = *itr;
    auto const reach = compute_reach_impl(c);

    glushkov_state_t state_next = (state != 0) ? compute_follow_impl(state) : glushkov_state_t{0};

    // Inject fresh starts unless in greedy-extension mode.
    if (!cur_match) { state_next |= first_set; }

    state = state_next & reach;

    if (state == 0) {
      if (cur_match) { break; }  // greedy extension died
      continue;
    }

    // Detect match: any active state reaching accept.
    if (state & accept_mask) {
      if constexpr (P == positional::BEGIN_END) {
        cur_match = match_pair{inject_start_pos, pos + 1};  // start is provisional
      } else {
        cur_match = match_pair{-1, pos + 1};
      }
      // Greedy: continue extending.  Fresh start injection stops via
      // the !cur_match gate above.
      state = priority_kill(state, accept_mask);
    }
  }

  if (!cur_match) { return {}; }

  // For END_ONLY (contains_re, matches_re), start position is unused.
  if constexpr (P != positional::BEGIN_END) { return cur_match; }

  // --- Phase 2: find the earliest match start (BEGIN_END only) ---
  //
  // Rescan from inject_start_pos.  For each position where first_set
  // matches, run a single-seed NFA forward.  The first start whose NFA
  // reaches accept gives the leftmost match with greedy extension.

  auto rescan_itr = inject_start_itr;
  for (int32_t s = inject_start_pos; s < cur_match->second; ++s, ++rescan_itr) {
    char32_t const c      = *rescan_itr;
    auto const reach      = compute_reach_impl(c);
    glushkov_state_t seed = first_set & reach;
    if (seed == 0) { continue; }

    // Check immediate accept (single-position pattern, e.g. `\d`).
    int32_t greedy_end = -1;
    if (seed & accept_mask) {
      greedy_end = s + 1;
      seed       = priority_kill(seed, accept_mask);
    }

    // Advance the single-seed NFA forward.
    auto scan_itr = rescan_itr;
    ++scan_itr;
    for (int32_t p = s + 1; scan_itr.byte_offset() < size_bytes; ++p, ++scan_itr) {
      seed = compute_follow_impl(seed) & compute_reach_impl(*scan_itr);
      if (seed == 0) { break; }
      if (seed & accept_mask) {
        greedy_end = p + 1;
        seed       = priority_kill(seed, accept_mask);
      }
    }

    if (greedy_end >= 0) { return match_pair{s, greedy_end}; }
  }

  // Fallback: should not reach here if phase 1 detected a match correctly.
  return cur_match;
}

__device__ inline void gkprog_device::store(void* buffer) const
{
  auto cache = static_cast<glushkov_shmem_cache*>(buffer);
  for (uint32_t i = 0; i < static_cast<uint32_t>(GLUSHKOV_ASCII_TABLE_SIZE); ++i) {
    cache->reach_ascii[i] = _reach_ascii[i];
  }
  for (uint32_t i = 0; i < shift_count; ++i) {
    cache->shift_masks[i]   = _shift_masks[i];
    cache->shift_amounts[i] = _shift_amounts[i];
  }
  for (uint32_t i = 0; i < num_states; ++i) {
    cache->exception_successors[i] = _exception_successors[i];
  }
}

__device__ gkprog_device gkprog_device::load(gkprog_device const prog, void* buffer)
{
  auto result                  = gkprog_device(prog);
  auto cache                   = static_cast<glushkov_shmem_cache const*>(buffer);
  result._reach_ascii          = cache->reach_ascii.data();
  result._shift_amounts        = cache->shift_amounts.data();
  result._shift_masks          = cache->shift_masks.data();
  result._exception_successors = cache->exception_successors.data();
  return result;
}

}  // namespace cudf::strings::detail
