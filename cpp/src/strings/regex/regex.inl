/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/types.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief This holds the state information when evaluating a string
 * against a regex pattern.
 *
 * There are 2 instances of this per string managed in the reljunk class.
 * As each regex instruction is evaluated for a string, the result is
 * reflected here. The regexec function updates and manages this state data.
 */
struct alignas(8) relist {
  /**
   * @brief Compute the memory size for the state data.
   */
  CUDF_HOST_DEVICE constexpr inline static std::size_t data_size_for(int32_t insts)
  {
    return ((sizeof(ranges[0]) + sizeof(inst_ids[0])) * insts) +
           cudf::util::div_rounding_up_unsafe(insts, 8);
  }

  /**
   * @brief Compute the aligned memory allocation size.
   */
  CUDF_HOST_DEVICE constexpr inline static std::size_t alloc_size(int32_t insts,
                                                                  int32_t num_threads)
  {
    return cudf::util::round_up_unsafe<size_t>(data_size_for(insts) * num_threads, sizeof(restate));
  }

  struct alignas(16) restate {
    int2 range;
    int32_t inst_id;
    int32_t reserved;
  };

  __device__ __forceinline__
  relist(int16_t insts, int32_t num_threads, u_char* gp_ptr, int32_t index)
    : masksize(cudf::util::div_rounding_up_unsafe(insts, 8)), stride(num_threads)
  {
    auto const rdata_size = sizeof(ranges[0]);
    auto const idata_size = sizeof(inst_ids[0]);
    ranges                = reinterpret_cast<decltype(ranges)>(gp_ptr + (index * rdata_size));
    inst_ids =
      reinterpret_cast<int16_t*>(gp_ptr + (rdata_size * stride * insts) + (index * idata_size));
    mask = gp_ptr + ((rdata_size + idata_size) * stride * insts) + (index * masksize);
  }

  __device__ __forceinline__ void reset()
  {
    memset(mask, 0, masksize);
    size = 0;
  }

  template <positional P = positional::BEGIN_END>
  __device__ __forceinline__ bool activate(int32_t id, int32_t begin, int32_t end)
  {
    if (readMask(id)) { return false; }
    writeMask(id);
    inst_ids[size * stride] = static_cast<int16_t>(id);
    if constexpr (P == positional::BEGIN_END) { ranges[size * stride] = int2{begin, end}; }
    ++size;
    return true;
  }

  template <positional P = positional::BEGIN_END>
  [[nodiscard]] __device__ __forceinline__ restate get_state(int16_t idx) const
  {
    if constexpr (P == positional::BEGIN_END) {
      return restate{ranges[idx * stride], inst_ids[idx * stride]};
    }
    return restate{{-1, -1}, inst_ids[idx * stride]};
  }

  [[nodiscard]] __device__ __forceinline__ int16_t get_size() const { return size; }

 private:
  int16_t size{};
  int16_t const masksize;
  int32_t const stride;
  int2* __restrict__ ranges;       // pair per instruction
  int16_t* __restrict__ inst_ids;  // one per instruction
  u_char* __restrict__ mask;       // bit per instruction

  __device__ __forceinline__ void writeMask(int32_t pos) const
  {
    u_char const uc = 1 << (pos & 7);
    mask[pos >> 3] |= uc;
  }

  [[nodiscard]] __device__ __forceinline__ bool readMask(int32_t pos) const
  {
    u_char const uc = mask[pos >> 3];
    return static_cast<bool>((uc >> (pos & 7)) & 1);
  }
};

template <positional P = positional::BEGIN_END>
struct reljunk {
  relist* __restrict__ list1;
  relist* __restrict__ list2;
  int32_t starttype{};
  char32_t startchar{};

  __device__ inline reljunk(relist* list1, relist* list2, reinst const inst)
    : list1(list1), list2(list2)
  {
    if (inst.type == CHAR || inst.type == BOL) {
      starttype = inst.type;
      startchar = inst.u1.c;
    }
  }
  __device__ inline void swaplist()
  {
    auto tmp = list1;
    list1    = list2;
    list2    = tmp;
  }
};

__device__ __forceinline__ reinst reprog_device::get_inst(int32_t id) const { return _insts[id]; }

__device__ __forceinline__ reclass_device reprog_device::get_class(int32_t id) const
{
  return _classes[id];
}

__device__ __forceinline__ bool reprog_device::is_empty() const
{
  return insts_counts() == 0 || get_inst(0).type == END;
}

__device__ __forceinline__ void reprog_device::store(void* buffer) const
{
  if (_prog_size > MAX_SHARED_MEM) { return; }

  auto ptr = static_cast<u_char*>(buffer);

  // create instance inside the given buffer
  auto result = new (ptr) reprog_device(*this);

  // add the insts array
  ptr += sizeof(reprog_device);
  auto insts     = reinterpret_cast<reinst*>(ptr);
  result->_insts = insts;
  for (int idx = 0; idx < _insts_count; ++idx)
    *insts++ = _insts[idx];

  // add the startinst_ids array
  ptr += cudf::util::round_up_unsafe(_insts_count * sizeof(_insts[0]), sizeof(_startinst_ids[0]));
  auto ids               = reinterpret_cast<int32_t*>(ptr);
  result->_startinst_ids = ids;
  for (int idx = 0; idx < _starts_count; ++idx)
    *ids++ = _startinst_ids[idx];

  // add the classes array
  ptr += cudf::util::round_up_unsafe(_starts_count * sizeof(int32_t), sizeof(_classes[0]));
  auto classes     = reinterpret_cast<reclass_device*>(ptr);
  result->_classes = classes;
  // fill in each class
  auto d_ptr = reinterpret_cast<reclass_range*>(classes + _classes_count);
  for (int idx = 0; idx < _classes_count; ++idx) {
    classes[idx]          = _classes[idx];
    classes[idx].literals = d_ptr;
    for (int jdx = 0; jdx < _classes[idx].count; ++jdx)
      *d_ptr++ = _classes[idx].literals[jdx];
  }
}

__device__ __forceinline__ reprog_device reprog_device::load(reprog_device const prog, void* buffer)
{
  return (prog._prog_size > MAX_SHARED_MEM) ? reprog_device(prog)
                                            : reinterpret_cast<reprog_device*>(buffer)[0];
}

__device__ __forceinline__ static string_view::const_iterator find_char(
  cudf::char_utf8 chr, string_view const d_str, string_view::const_iterator itr)
{
  while (itr.byte_offset() < d_str.size_bytes() && *itr != chr) {
    ++itr;
  }
  return itr;
}

/**
 * @brief Evaluate a specific string against regex pattern compiled to this instance.
 *
 * This is the main function for executing the regex against an individual string.
 *
 * @param dstr String used for matching.
 * @param jnk State data object for this string.
 * @param[in,out] begin Character position to start evaluation. On return, it is the position of the
 * match.
 * @param[in,out] end Character position to stop evaluation. On return, it is the end of the matched
 * substring.
 * @param group_id Index of the group to match in a multi-group regex pattern.
 * @return >0 if match found
 */
template <positional P>
__device__ __forceinline__ match_result reprog_device::regexec(string_view const dstr,
                                                               reljunk<P>& jnk,
                                                               string_view::const_iterator itr,
                                                               cudf::size_type end,
                                                               cudf::size_type const group_id) const
{
  int32_t match       = 0;
  auto begin          = itr.position();
  auto pos            = begin;
  auto eos            = end;
  auto checkstart     = jnk.starttype != 0;
  auto last_character = false;

  jnk.list1->reset();
  do {
    // fast check for first CHAR or BOL
    if (checkstart) {
      auto startchar = static_cast<char_utf8>(jnk.startchar);
      switch (jnk.starttype) {
        case BOL: {
          if (pos == 0) { break; }
          if (startchar != '^' && startchar != 'S') { return cuda::std::nullopt; }
          if (startchar != '\n') { break; }
          --itr;
          startchar = static_cast<char_utf8>('\n');
          [[fallthrough]];
        }
        case CHAR: {
          auto const find_itr = find_char(startchar, dstr, itr);
          if (find_itr.byte_offset() >= dstr.size_bytes()) { return cuda::std::nullopt; }
          itr = find_itr + (jnk.starttype == BOL);
          pos = itr.position();
          break;
        }
      }
    }

    if (((eos < 0) || (pos < eos)) && match == 0) {
      auto ids = _startinst_ids;
      while (*ids >= 0) {
        jnk.list1->template activate<P>(*ids++, (group_id == 0 ? pos : -1), -1);
      }
    }

    last_character = itr.byte_offset() >= dstr.size_bytes();

    char_utf8 const c = last_character ? 0 : *itr;

    // expand the non-character types like: LBRA, RBRA, BOL, EOL, BOW, NBOW, and OR
    bool expanded = false;
    do {
      jnk.list2->reset();
      expanded = false;

      for (int16_t i = 0; i < jnk.list1->get_size(); i++) {
        auto state          = jnk.list1->template get_state<P>(i);
        auto range          = state.range;
        auto const inst     = get_inst(state.inst_id);
        int32_t id_activate = -1;

        switch (inst.type) {
          case CHAR:
          case ANY:
          case ANYNL:
          case CCLASS:
          case NCCLASS:
          case END: id_activate = state.inst_id; break;
          case LBRA:
            if (inst.u1.subid == group_id) { range.x = pos; }
            id_activate = inst.u2.next_id;
            expanded    = true;
            break;
          case RBRA:
            if (inst.u1.subid == group_id) { range.y = pos; }
            id_activate = inst.u2.next_id;
            expanded    = true;
            break;
          case BOL: {
            auto titr         = itr;
            auto const prev_c = pos > 0 ? *(--titr) : 0;
            // For EXT_NEWLINE, \r\n is a single terminator: ^ matches AFTER the \n and
            // never between the \r and \n.
            if ((pos == 0) || ((inst.u1.c == '^') && (prev_c == '\n')) ||
                ((inst.u1.c == 'S') && is_newline(prev_c) && !((prev_c == '\r') && (c == '\n')))) {
              id_activate = inst.u2.next_id;
              expanded    = true;
            }
            break;
          }
          case EOL: {
            // EOL matches at end-of-string, or before a line terminator (MULTILINE: any
            // terminator; otherwise: only the final terminator). For EXT_NEWLINE the
            // two-character CRLF (\r\n) is treated as a SINGLE terminator: '$' matches before
            // the \r and never between \r and \n.
            if (last_character) {
              id_activate = inst.u2.next_id;
              expanded    = true;
              break;
            }
            bool const ext = (inst.u1.c == 'S' || inst.u1.c == 'N');
            bool const nl  = ext ? is_newline(c) : (c == '\n');
            if (nl && (inst.u1.c != 'Z')) {
              // For EXT_NEWLINE, suppress a match wedged between the CR and LF of a CRLF.
              auto titr     = itr;
              bool mid_crlf = ext && (c == '\n') && (pos > 0) && (*(--titr) == '\r');
              if (!mid_crlf) {
                // MULTILINE matches before any terminator; otherwise only the final one.
                bool matched = (inst.u1.c == '$' || inst.u1.c == 'S');
                if (!matched) {
                  matched = (itr.byte_offset() + bytes_in_char_utf8(c) == dstr.size_bytes());
                  if (!matched && ext && (c == '\r')) {
                    // a CR beginning a trailing CRLF also ends the string
                    auto nitr = itr;
                    ++nitr;
                    matched =
                      (nitr.byte_offset() < dstr.size_bytes()) && (*nitr == '\n') &&
                      (nitr.byte_offset() + bytes_in_char_utf8(static_cast<char_utf8>('\n')) ==
                       dstr.size_bytes());
                  }
                }
                if (matched) {
                  id_activate = inst.u2.next_id;
                  expanded    = true;
                }
              }
            }
            break;
          }
          case BOW:
          case NBOW: {
            auto titr               = itr;
            auto const prev_c       = pos > 0 ? *(--titr) : 0;
            auto const word_class   = reclass_device{CCLASS_W};
            bool const curr_is_word = word_class.is_match(c, _codepoint_flags);
            bool const prev_is_word = word_class.is_match(prev_c, _codepoint_flags);
            if ((curr_is_word == prev_is_word) != (inst.type == BOW)) {
              id_activate = inst.u2.next_id;
              expanded    = true;
            }
            break;
          }
          case OR:
            jnk.list2->template activate<P>(inst.u1.right_id, range.x, range.y);
            id_activate = inst.u2.left_id;
            expanded    = true;
            break;
        }
        if (id_activate >= 0) { jnk.list2->template activate<P>(id_activate, range.x, range.y); }
      }
      jnk.swaplist();

    } while (expanded);

    // execute instructions
    bool continue_execute = true;
    jnk.list2->reset();
    for (int16_t i = 0; continue_execute && i < jnk.list1->get_size(); i++) {
      auto const state    = jnk.list1->template get_state<P>(i);
      auto const range    = state.range;
      auto const inst     = get_inst(state.inst_id);
      int32_t id_activate = -1;

      switch (inst.type) {
        case CHAR:
          if (inst.u1.c == c) id_activate = inst.u2.next_id;
          break;
        case ANY: {
          if ((c == '\n') || ((inst.u1.c == 'N') && is_newline(c))) { break; }
          [[fallthrough]];
        }
        case ANYNL: id_activate = inst.u2.next_id; break;
        case NCCLASS:
        case CCLASS: {
          auto const cls = get_class(inst.u1.cls_id);
          if (cls.is_match(static_cast<char32_t>(c), _codepoint_flags) == (inst.type == CCLASS)) {
            id_activate = inst.u2.next_id;
          }
          break;
        }
        case END:
          match = 1;
          begin = range.x;
          end   = group_id == 0 ? pos : range.y;
          // done with execute
          continue_execute = false;
          break;
      }
      if (continue_execute && (id_activate >= 0)) {
        jnk.list2->template activate<P>(id_activate, range.x, range.y);
      }
    }

    ++pos;
    ++itr;
    jnk.swaplist();
    checkstart = jnk.list1->get_size() == 0;
  } while (!last_character && (!checkstart || !match));

  return match ? match_result({begin, end}) : cuda::std::nullopt;
}

template <positional P>
__device__ __forceinline__ match_result reprog_device::find(int32_t const thread_idx,
                                                            string_view const dstr,
                                                            string_view::const_iterator begin,
                                                            cudf::size_type end) const
{
  return call_regexec<P>(thread_idx, dstr, begin, end);
}

__device__ __forceinline__ match_result reprog_device::extract(int32_t const thread_idx,
                                                               string_view const dstr,
                                                               string_view::const_iterator begin,
                                                               cudf::size_type end,
                                                               cudf::size_type const group_id) const
{
  end               = begin.position() + 1;
  auto const result = call_regexec(thread_idx, dstr, begin, end, group_id + 1);
  // a capture group that did not participate in the overall match
  // (e.g. an unmatched optional group) has an invalid range
  return (result && (result->first >= 0)) ? result : cuda::std::nullopt;
}

template <positional P>
__device__ __forceinline__ match_result
reprog_device::call_regexec(int32_t const thread_idx,
                            string_view const dstr,
                            string_view::const_iterator begin,
                            cudf::size_type end,
                            cudf::size_type const group_id) const
{
  auto gp_ptr = reinterpret_cast<u_char*>(_buffer);
  relist list1(static_cast<int16_t>(_max_insts), _thread_count, gp_ptr, thread_idx);

  gp_ptr += relist::alloc_size(_max_insts, _thread_count);
  relist list2(static_cast<int16_t>(_max_insts), _thread_count, gp_ptr, thread_idx);

  reljunk<P> jnk(&list1, &list2, get_inst(_startinst_id));
  return regexec<P>(dstr, jnk, begin, end, group_id);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
