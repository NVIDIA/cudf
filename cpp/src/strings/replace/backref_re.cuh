/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "strings/regex/regex.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace cudf {
namespace strings {
namespace detail {

using backref_type = cuda::std::pair<size_type, size_type>;

/**
 * @brief This functor handles replacing strings by applying the compiled regex pattern
 * and inserting the at the backref position indicated in the replacement template.
 *
 * The logic includes computing the size of each string and also writing the output.
 */
template <typename Iterator>
struct backrefs_fn {
  column_device_view const d_strings;
  string_view const d_repl;  // string replacement template
  Iterator backrefs_begin;
  Iterator backrefs_end;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type const idx, reprog_device const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const in_ptr = d_str.data();
    auto const nchars = d_str.length();      // number of characters in input string
    auto nbytes       = d_str.size_bytes();  // number of bytes for the output string
    auto out_ptr      = d_chars ? (d_chars + d_offsets[idx]) : nullptr;
    auto itr          = d_str.begin();
    auto last_pos     = itr;

    // copy input to output replacing strings as we go
    while (itr.position() <= nchars)  // inits the begin/end vars
    {
      auto const match = prog.find(prog_idx, d_str, itr);
      if (!match) { break; }

      auto const [start_pos, end_pos] = match_positions_to_bytes(*match, d_str, itr);
      nbytes += d_repl.size_bytes() - (end_pos - start_pos);  // compute the output size

      // copy the string data before the matched section
      if (out_ptr) {
        out_ptr = copy_and_increment(
          out_ptr, in_ptr + last_pos.byte_offset(), start_pos - last_pos.byte_offset());
      }
      size_type lpos_template = 0;              // last end pos of replace template
      auto const repl_ptr     = d_repl.data();  // replace template pattern

      itr += (match->first - itr.position());
      thrust::for_each(
        thrust::seq, backrefs_begin, backrefs_end, [&] __device__(backref_type backref) {
          if (out_ptr) {
            auto const copy_length = backref.second - lpos_template;
            out_ptr = copy_and_increment(out_ptr, repl_ptr + lpos_template, copy_length);
            lpos_template += copy_length;
          }
          // extract the specific group's string for this backref's index
          auto extracted = prog.extract(prog_idx, d_str, itr, match->second, backref.first - 1);
          if (!extracted || (extracted->second < extracted->first)) {
            return;  // no value for this backref number; that is ok
          }
          auto const d_str_ex = string_from_match(*extracted, d_str, itr);
          nbytes += d_str_ex.size_bytes();
          if (out_ptr) { out_ptr = copy_string(out_ptr, d_str_ex); }
        });

      // copy remainder of template
      if (out_ptr && (lpos_template < d_repl.size_bytes())) {
        out_ptr = copy_and_increment(
          out_ptr, repl_ptr + lpos_template, d_repl.size_bytes() - lpos_template);
      }

      // setup to match the next section
      last_pos += (match->second - last_pos.position());
      itr = last_pos + (match->first == match->second);
    }

    // finally, copy remainder of input string
    if (out_ptr) {
      thrust::copy_n(thrust::seq,
                     in_ptr + last_pos.byte_offset(),
                     d_str.size_bytes() - last_pos.byte_offset(),
                     out_ptr);
    } else {
      d_sizes[idx] = nbytes;
    }
  }
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
