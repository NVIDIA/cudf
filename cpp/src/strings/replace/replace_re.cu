/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/regex/regex_program_impl.hpp"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/stream>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Size of the match positions buffer as a multiple of the input chars size
 *
 * This allows one match per 4 bytes of input before a row must be re-processed.
 */
constexpr int64_t match_slots_factor = 2;

/**
 * @brief Allots match slots to each row in proportion to its size in bytes
 *
 * Row `i` owns slots `[begin(i), begin(i+1))` where `begin(i)` is its byte offset in the
 * input scaled by `match_slots_factor / sizeof(match_pair)`. No per-row storage is needed
 * to locate them.
 */
struct match_slots {
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Returns the number of slots for the given number of input bytes
   */
  static constexpr CUDF_HOST_DEVICE int64_t slots_for_bytes(int64_t bytes)
  {
    return (bytes * match_slots_factor) / static_cast<int64_t>(sizeof(match_pair));
  }

  __device__ int64_t begin(size_type row) const
  {
    return slots_for_bytes(d_offsets[row] - d_offsets[0]);
  }
  __device__ size_type capacity(size_type row) const
  {
    return static_cast<size_type>(begin(row + 1) - begin(row));
  }
};

/**
 * @brief Records the output size of each string and the byte positions of its regex matches
 *
 * Up to `slots.capacity(idx)` match positions are stored at `d_matches + slots.begin(idx)`.
 * The match count is also recorded so rows with more matches than slots can be identified.
 */
struct find_matches_fn {
  column_device_view const d_strings;
  string_view const d_repl;
  size_type const maxrepl;
  match_pair* d_matches;
  match_slots slots;
  size_type* d_counts;
  size_type* d_sizes;

  template <typename ProgDevice>
  __device__ void operator()(size_type const idx, ProgDevice const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) {
      d_counts[idx] = 0;
      d_sizes[idx]  = 0;
      return;
    }

    auto const d_str    = d_strings.element<string_view>(idx);
    auto const nchars   = d_str.length();
    auto nbytes         = d_str.size_bytes();
    auto mxn            = maxrepl < 0 ? nchars + 1 : maxrepl;
    auto const capacity = slots.capacity(idx);
    auto d_output       = d_matches + slots.begin(idx);
    auto itr            = d_str.begin();
    auto last_pos       = itr;

    size_type count = 0;
    while (mxn-- > 0 && itr.position() <= nchars && !prog.is_empty()) {
      auto const match = prog.find(prog_idx, d_str, itr);
      if (!match) { break; }

      auto const positions = match_positions_to_bytes(*match, d_str, last_pos);
      nbytes += d_repl.size_bytes() - (positions.second - positions.first);
      if (count < capacity) { d_output[count] = positions; }
      ++count;

      last_pos += (match->second - last_pos.position());
      itr = last_pos + (match->first == match->second);
    }

    d_counts[idx] = count;
    d_sizes[idx]  = nbytes;
  }
};

/**
 * @brief Writes the output for the rows whose matches did not fit in their slots
 *
 * The regex is run again on each of the rows in `d_rows` and the output is written directly.
 */
struct replace_overflow_fn {
  column_device_view const d_strings;
  string_view const d_repl;
  size_type const maxrepl;
  size_type const* d_rows;
  cudf::detail::input_offsetalator d_offsets;
  char* d_chars;

  template <typename ProgDevice>
  __device__ void operator()(size_type const idx, ProgDevice const prog, int32_t const prog_idx)
  {
    auto const row    = d_rows[idx];
    auto const d_str  = d_strings.element<string_view>(row);
    auto const nchars = d_str.length();
    auto mxn      = maxrepl < 0 ? nchars + 1 : maxrepl;  // max possible replaces for this string
    auto in_ptr   = d_str.data();                        // input pointer (i)
    auto out_ptr  = d_chars + d_offsets[row];            // output pointer (o)
    auto itr      = d_str.begin();
    auto last_pos = itr;

    // copy input to output replacing strings as we go
    while (mxn-- > 0 && itr.position() <= nchars && !prog.is_empty()) {
      auto const match = prog.find(prog_idx, d_str, itr);
      if (!match) { break; }  // no more matches

      auto const [start_pos, end_pos] = match_positions_to_bytes(*match, d_str, last_pos);
      // replace:
      //  i:bbbbsssseeee
      //    ^
      out_ptr = copy_and_increment(out_ptr,                              // o:bbbb
                                   in_ptr + last_pos.byte_offset(),      //       ^
                                   start_pos - last_pos.byte_offset());  //
      out_ptr = copy_string(out_ptr, d_repl);                            // o:bbbbrrrrrr
                                                                         //  out_ptr ---^
      last_pos += (match->second - last_pos.position());                 // i:bbbbsssseeee
                                                                         //  in_ptr --^
      itr = last_pos + (match->first == match->second);
    }

    copy_and_increment(out_ptr,                                       // copy the remainder
                       in_ptr + last_pos.byte_offset(),               // o:bbbbrrrrrreeee
                       d_str.size_bytes() - last_pos.byte_offset());  //             ^   ^
  }
};

/**
 * @brief Writes the output for rows whose matches fit in their slots
 *
 * The output is built from the recorded match positions without re-running the regex.
 * One warp processes each row with the lanes copying each piece together.
 */
CUDF_KERNEL void write_output_kernel(column_device_view const d_strings,
                                     string_view const d_repl,
                                     match_pair const* d_matches,
                                     match_slots const slots,
                                     size_type const* d_counts,
                                     cudf::detail::input_offsetalator const d_offsets,
                                     char* d_chars)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const idx     = static_cast<size_type>(tid / cudf::detail::warp_size);
  auto const lane_id = static_cast<size_type>(tid % cudf::detail::warp_size);
  if (idx >= d_strings.size() || d_strings.is_null(idx)) { return; }

  auto const count = d_counts[idx];
  if (count > slots.capacity(idx)) { return; }  // handled by replace_overflow_fn

  auto const d_str = d_strings.element<string_view>(idx);
  auto out_ptr     = d_chars + d_offsets[idx];
  auto in_ptr      = d_str.data();
  auto warp_copy   = [lane_id](char* dst, char const* src, size_type bytes) {
    for (auto i = lane_id; i < bytes; i += cudf::detail::warp_size) {
      dst[i] = src[i];
    }
    return dst + bytes;
  };

  auto const d_row_matches = d_matches + slots.begin(idx);
  size_type last           = 0;
  for (size_type i = 0; i < count; ++i) {
    auto const [begin, end] = d_row_matches[i];
    out_ptr                 = warp_copy(out_ptr, in_ptr + last, begin - last);
    out_ptr                 = warp_copy(out_ptr, d_repl.data(), d_repl.size_bytes());
    last                    = end;
  }
  warp_copy(out_ptr, in_ptr + last, d_str.size_bytes() - last);
}

/**
 * @brief Regex replace using a single regex pass that computes the output sizes and records
 * the match positions, followed by a regex-free pass that writes the output
 *
 * Only rows with more matches than their slots can hold are processed by a second regex pass.
 */
template <typename ProgDevice>
std::unique_ptr<column> replace_re_fn(strings_column_view const& input,
                                      ProgDevice& d_prog,
                                      string_view d_repl,
                                      size_type maxrepl,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const strings_count = input.size();
  auto const temp_mr       = cudf::get_current_device_resource_ref();
  auto const d_column      = column_device_view::create(input.parent(), stream);
  auto const d_strings     = *d_column;

  auto const slots = match_slots{
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset())};
  auto const [first_offset, last_offset] = get_first_and_last_offset(input, stream);
  auto const total_slots                 = match_slots::slots_for_bytes(last_offset - first_offset);

  // compute the output sizes and record the match positions that fit
  auto counts  = cuda::device_buffer<size_type>(stream, temp_mr, strings_count, cuda::no_init);
  auto sizes   = cuda::device_buffer<size_type>(stream, temp_mr, strings_count, cuda::no_init);
  auto matches = cuda::device_buffer<match_pair>(stream, temp_mr, total_slots, cuda::no_init);
  launch_for_each_kernel(
    find_matches_fn{d_strings, d_repl, maxrepl, matches.data(), slots, counts.data(), sizes.data()},
    d_prog,
    strings_count,
    stream);

  auto [offsets, bytes] = make_offsets_child_column(
    cudf::device_span<size_type const>(sizes.data(), strings_count), stream, mr);
  auto chars = rmm::device_uvector<char>(bytes, stream, mr);
  if (bytes > 0) {
    auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

    // write the rows whose matches fit in their slots
    cudf::detail::grid_1d grid{static_cast<int64_t>(strings_count) * cudf::detail::warp_size, 256};
    write_output_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      d_strings, d_repl, matches.data(), slots, counts.data(), d_offsets, chars.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    // run the regex again to write the rows whose matches did not fit
    auto overflow_rows =
      cuda::device_buffer<size_type>(stream, temp_mr, strings_count, cuda::no_init);
    auto const d_counts = counts.data();
    auto const rows_end = cudf::detail::copy_if(
      cuda::counting_iterator<size_type>(0),
      cuda::counting_iterator<size_type>(strings_count),
      overflow_rows.begin(),
      [d_counts, slots] __device__(size_type idx) -> bool {
        return d_counts[idx] > slots.capacity(idx);
      },
      stream);
    auto const overflow_count =
      static_cast<size_type>(cuda::std::distance(overflow_rows.begin(), rows_end));
    if (overflow_count > 0) {
      launch_for_each_kernel(
        replace_overflow_fn{
          d_strings, d_repl, maxrepl, overflow_rows.data(), d_offsets, chars.data()},
        d_prog,
        overflow_count,
        stream);
    }
  }

  return make_strings_column(strings_count,
                             std::move(offsets),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

//
std::unique_ptr<column> replace_re(strings_column_view const& input,
                                   regex_program const& prog,
                                   string_scalar const& replacement,
                                   std::optional<size_type> max_replace_count,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }

  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");

  auto const maxrepl = max_replace_count.value_or(-1);

  auto [fp, literal] = prog.get_literal_fast_path();
  if (fp == literal_fast_path::LITERAL_ONLY) {
    auto const target =
      cudf::string_scalar(literal, true, stream, cudf::get_current_device_resource_ref());
    return replace(input, target, replacement, maxrepl, stream, mr);
  }

  auto const d_repl = string_view(replacement.data(), replacement.size());

  if (regex_device_builder::glushkov_fast_path_supported(prog)) {
    auto d_prog = regex_device_builder::create_gkprog_device(prog, stream);
    return replace_re_fn(input, *d_prog, d_repl, maxrepl, stream, mr);
  }
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);
  return replace_re_fn(input, *d_prog, d_repl, maxrepl, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> replace_re(strings_column_view const& strings,
                                   regex_program const& prog,
                                   string_scalar const& replacement,
                                   std::optional<size_type> max_replace_count,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_re(strings, prog, replacement, max_replace_count, stream, mr);
}

}  // namespace strings
}  // namespace cudf
