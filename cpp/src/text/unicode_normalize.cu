/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/unicode_normalize.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/span>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cstdint>

namespace nvtext {
namespace detail {
namespace {

// Composition exclusion: ~70 codepoints explicitly excluded from NFC/NFKC
// composition (Unicode 15, DerivedNormalizationProps.txt).
// Must be sorted ascending for thrust::binary_search.
// 0x2ADC (Supplemental Mathematical Operators) is placed after 0x0FB9 (Tibetan),
// not adjacent to the Hebrew block where it was previously listed out of order.
// clang-format off
__device__ __constant__ cuda::std::array COMPOSITION_EXCLUSIONS{
  0x0958u, 0x0959u, 0x095Au, 0x095Bu, 0x095Cu, 0x095Du, 0x095Eu, 0x095Fu, // Devanagari
  0x09DCu, 0x09DDu, 0x09DFu, // Bengali
  0x0A33u, 0x0A36u, // Gurmukhi
  0x0A59u, 0x0A5Au, 0x0A5Bu, 0x0A5Cu, 0x0A5Eu, // Gujarati
  0x0B5Cu, 0x0B5Du, // Oriya
  0x0F43u, 0x0F4Du, 0x0F52u, 0x0F57u, 0x0F5Cu, 0x0F69u, 0x0F76u, 0x0F78u, // Tibetan
  0x0F80u, 0x0F93u, 0x0F9Du, 0x0FA2u, 0x0FA7u, 0x0FACu, 0x0FB9u,
  0x2ADCu, // Supplemental Mathematical Operators
  0xFB1Du, 0xFB1Fu, 0xFB2Au, 0xFB2Bu, 0xFB2Cu, 0xFB2Du, 0xFB2Eu, // Hebrew Presentation Forms
  0xFB2Fu, 0xFB30u, 0xFB31u, 0xFB32u, 0xFB33u, 0xFB34u, 0xFB35u,
  0xFB36u, 0xFB38u, 0xFB39u, 0xFB3Au, 0xFB3Bu, 0xFB3Cu, 0xFB3Eu,
  0xFB40u, 0xFB41u, 0xFB43u, 0xFB44u, 0xFB46u, 0xFB47u, 0xFB48u,
  0xFB49u, 0xFB4Au, 0xFB4Bu, 0xFB4Cu, 0xFB4Du, 0xFB4Eu,
  0x1D15Eu, 0x1D15Fu, 0x1D160u, 0x1D161u, 0x1D162u, 0x1D163u, 0x1D164u, // Musical Symbols
  0x1D1BBu, 0x1D1BCu, 0x1D1BDu, 0x1D1BEu, 0x1D1BFu, 0x1D1C0u,
};
// clang-format on

/**
 * Scatter Canonical Combining Class (CCC) values from the CCC column into a codepoint-indexed
 * table. Codepoints are pre-converted from the hex column via hex_to_integers. One invocation per
 * UnicodeData.txt row.
 */
struct scatter_ccc_fn {
  cudf::column_device_view ccc_col;              // INT32: CCC values
  cuda::std::span<uint32_t const> d_codepoints;  // pre-converted codepoints
  cuda::std::span<uint8_t> ccc_table;            // output: CCC indexed by codepoint

  __device__ void operator()(cudf::size_type idx) const
  {
    uint32_t const cp = d_codepoints[idx];
    if (cp <= MAX_CODEPOINT) {
      ccc_table[cp] = static_cast<uint8_t>(ccc_col.element<int32_t>(idx));
    }
  }
};

/**
 * Set one bit in the NFC/NFKC quick-check bitset for each codepoint that is
 * not stable under NFC or NFKC normalization:
 *
 *  - Compatibility decompositions (mapping starts with '<'): unstable under NFKC.
 *  - Singleton canonical decompositions (exactly one canonical token, e.g.
 *    U+212B ANGSTROM SIGN → U+00C5): unstable under NFC because the codepoint
 *    decomposes to a different codepoint that does not recompose back to itself.
 *
 * One invocation per UnicodeData.txt row.
 */
struct scatter_compat_flag_fn {
  cudf::column_device_view decomp_map;
  cuda::std::span<uint32_t const> d_codepoints;
  cuda::std::span<int32_t const> d_counts;           // token count per row (apply_compat-aware)
  cuda::std::span<cudf::bitmask_type> compat_flags;  // bitset: 1 bit per codepoint

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const sv        = decomp_map.element<cudf::string_view>(idx);
    bool const is_compat = sv.size_bytes() > 0 && sv.data()[0] == '<';
    // Flag if compat decomp OR singleton canonical decomp (NFC-unstable)
    if (!is_compat && d_counts[idx] != 1) { return; }
    uint32_t const cp = d_codepoints[idx];
    if (cp <= MAX_CODEPOINT) {
      cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(cp));
    }
  }
};

/**
 * Invoke `fn` for each space-separated hex token in a decomp mapping string.
 * Returns immediately for empty strings or, when `apply_compat==false`, for
 * compatibility mappings (strings that begin with '<').  When `apply_compat==true`
 * the leading "<tag> " prefix is consumed before the iteration starts.
 */
template <typename Fn>
__device__ void for_each_decomp_token(cudf::string_view d_str, bool apply_compat, Fn fn)
{
  auto const size = d_str.size_bytes();
  if (size == 0) { return; }
  char const* const ptr = d_str.data();
  bool const is_compat  = (ptr[0] == '<');
  cudf::size_type pos   = 0;
  if (is_compat) {
    if (!apply_compat) { return; }
    while (pos < size && ptr[pos] != '>') {
      ++pos;
    }
    pos += 2;  // skip '>' and the following space
  }
  while (pos < size) {
    while (pos < size && ptr[pos] == ' ') {
      ++pos;
    }
    cudf::size_type const tok_start = pos;
    while (pos < size && ptr[pos] != ' ') {
      ++pos;
    }
    if (pos > tok_start) { fn(ptr + tok_start, pos - tok_start); }
  }
}

/**
 * Count space-separated hex tokens in a decomp mapping string. One invocation per row.
 */
struct count_decomp_tokens_fn {
  cudf::column_device_view decomp_map;
  bool apply_compat;

  __device__ cudf::size_type operator()(cudf::size_type idx) const
  {
    auto count = cudf::size_type{0};
    auto fn    = [&count](char const*, cudf::size_type) { ++count; };
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx), apply_compat, fn);
    return count;
  }
};

/**
 * Write decomposition codepoints into the flat decomp_table.
 * One invocation per row; uses pre-computed per-codepoint offsets for placement.
 */
struct write_decomp_tokens_fn {
  cudf::column_device_view decomp_map;
  bool apply_compat;
  cuda::std::span<uint32_t const> d_codepoints;       // parsed codepoint per row
  cuda::std::span<uint32_t const> decomp_cp_offsets;  // write-start per codepoint
  cuda::std::span<uint32_t> decomp_table;             // flat output decomp codepoints

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const cp = d_codepoints[idx];
    if (cp > MAX_CODEPOINT) { return; }
    auto write_pos = decomp_cp_offsets[cp];
    auto fn        = [this, &write_pos](char const* ptr, cudf::size_type size) {
      decomp_table[write_pos++] = hex_to_cp(ptr, size);
    };
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx), apply_compat, fn);
  }
};

/**
 * Build composition table entries from canonical two-token decompositions.
 * Writes a (key, value) pair per qualifying row; zero for non-qualifying rows.
 */
struct build_comp_table_fn {
  cudf::column_device_view decomp_map;
  cuda::std::span<uint32_t const> d_codepoints;      // parsed codepoint per row
  cuda::std::span<int32_t const> d_counts;           // token count per row
  cuda::std::span<uint8_t const> ccc_table;          // CCC indexed by codepoint
  cuda::std::span<cudf::bitmask_type> compat_flags;  // NFC/NFKC quick-check bitset
  cuda::std::span<uint64_t> d_comp_keys;             // output: composition key
  cuda::std::span<uint32_t> d_comp_values;           // output: composed codepoint

  __device__ void operator()(cudf::size_type idx) const
  {
    d_comp_keys[idx]   = 0;
    d_comp_values[idx] = 0;
    if (d_counts[idx] != 2) { return; }
    // apply_compat=false: skip compatibility mappings (<tag> prefix)
    uint32_t tokens[2] = {0, 0};
    int32_t tok        = 0;
    auto fn            = [&tokens, &tok](char const* ptr, cudf::size_type size) {
      if (tok < 2) { tokens[tok++] = hex_to_cp(ptr, size); }
    };
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx), false, fn);

    if (tok < 2) { return; }
    auto const composed = d_codepoints[idx];
    if (composed > MAX_CODEPOINT) { return; }
    auto const starter   = tokens[0];
    auto const combining = tokens[1];
    if (thrust::binary_search(
          thrust::seq, COMPOSITION_EXCLUSIONS.begin(), COMPOSITION_EXCLUSIONS.end(), composed)) {
      // Script/explicit exclusion: NFC_QC=No, flag it so quick check catches it
      cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(composed));
      return;
    }

    if (starter > MAX_CODEPOINT || combining > MAX_CODEPOINT) { return; }
    if (ccc_table[starter] != 0) {
      // Non-starter decomposition: NFC_QC=No, flag it so quick check catches it
      cudf::set_bit(compat_flags.data(), static_cast<cudf::size_type>(composed));
      return;
    }
    d_comp_keys[idx]   = (static_cast<uint64_t>(starter) << 32) | combining;
    d_comp_values[idx] = composed;
  }
};

struct is_zero_comp_key {
  __device__ bool operator()(uint64_t k) const { return k == uint64_t{0}; }
};

}  // namespace
}  // namespace detail

struct unicode_normalizer::unicode_normalizer_impl {
  rmm::device_uvector<uint32_t> decomp_offsets;  // size DECOMP_OFFSETS_SIZE
  rmm::device_uvector<uint32_t> decomp_table;    // flat replacement codepoints
  rmm::device_uvector<uint8_t> ccc_table;        // size CODEPOINT_TABLE_SIZE
  rmm::device_uvector<cudf::bitmask_type> compat_decomp_flags;
  rmm::device_uvector<uint64_t> comp_keys;    // sorted (starter<<32|combining)
  rmm::device_uvector<uint32_t> comp_values;  // parallel composed codepoints
  unicode_normalization_form form;

  unicode_normalizer_impl(rmm::device_uvector<uint32_t>&& decomp_offsets,
                          rmm::device_uvector<uint32_t>&& decomp_table,
                          rmm::device_uvector<uint8_t>&& ccc_table,
                          rmm::device_uvector<cudf::bitmask_type>&& compat_decomp_flags,
                          rmm::device_uvector<uint64_t>&& comp_keys,
                          rmm::device_uvector<uint32_t>&& comp_values,
                          unicode_normalization_form form)
    : decomp_offsets(std::move(decomp_offsets)),
      decomp_table(std::move(decomp_table)),
      ccc_table(std::move(ccc_table)),
      compat_decomp_flags(std::move(compat_decomp_flags)),
      comp_keys(std::move(comp_keys)),
      comp_values(std::move(comp_values)),
      form(form)
  {
  }
};

unicode_normalizer::unicode_normalizer(cudf::table_view const& unicode_data,
                                       unicode_normalization_form form,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(unicode_data.num_columns() == 3,
               "unicode_data table must have exactly 3 columns",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(0).type().id() == cudf::type_id::STRING,
               "unicode_data column[0] must be STRING",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(1).type().id() == cudf::type_id::INT32,
               "unicode_data column[1] must be INT32",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(2).type().id() == cudf::type_id::STRING,
               "unicode_data column[2] must be STRING",
               std::invalid_argument);
  CUDF_EXPECTS(!cudf::has_nulls(unicode_data),
               "unicode_data table must not contain nulls",
               std::invalid_argument);

  cudf::size_type const num_rows = unicode_data.num_rows();
  CUDF_EXPECTS(num_rows > 0, "unicode_data table must not be empty", std::invalid_argument);

  auto temp_mr = cudf::get_current_device_resource_ref();
  auto codepoints_col =
    cudf::strings::detail::hex_to_integers(cudf::strings_column_view(unicode_data.column(0)),
                                           cudf::data_type{cudf::type_id::UINT32},
                                           stream,
                                           temp_mr);
  auto d_codepoints = cuda::std::span<uint32_t const>(codepoints_col->view().data<uint32_t>(),
                                                      static_cast<std::size_t>(num_rows));

  auto const d_ccc_col    = cudf::column_device_view::create(unicode_data.column(1), stream);
  auto const d_decomp_map = cudf::column_device_view::create(unicode_data.column(2), stream);
  bool const apply_compat =
    (form == unicode_normalization_form::NFKD || form == unicode_normalization_form::NFKC);
  auto const policy   = rmm::exec_policy_nosync(stream, temp_mr);
  auto const row_iter = cuda::make_counting_iterator(cudf::size_type{0});

  // Build Canonical Combining Class (CCC) table
  auto ccc_table = rmm::device_uvector<uint8_t>(detail::CODEPOINT_TABLE_SIZE, stream, mr);
  thrust::uninitialized_fill(policy, ccc_table.begin(), ccc_table.end(), uint8_t{0});
  thrust::for_each_n(
    policy, row_iter, num_rows, detail::scatter_ccc_fn{*d_ccc_col, d_codepoints, ccc_table});

  // Count decomposition tokens per row
  auto d_counts        = rmm::device_uvector<cudf::size_type>(num_rows, stream, temp_mr);
  auto count_tokens_fn = detail::count_decomp_tokens_fn{*d_decomp_map, apply_compat};
  thrust::transform(policy, row_iter, row_iter + num_rows, d_counts.begin(), count_tokens_fn);

  // Build NFC/NFKC quick-check bitset: flag compat decompositions (NFKC-unstable)
  // and singleton canonical decompositions like U+212B → U+00C5 (NFC-unstable).
  auto compat_decomp_flags = rmm::device_uvector<cudf::bitmask_type>(
    cudf::num_bitmask_words(detail::CODEPOINT_TABLE_SIZE), stream, mr);
  thrust::uninitialized_fill(
    policy, compat_decomp_flags.begin(), compat_decomp_flags.end(), uint32_t{0});
  auto scatter_compat_fn =
    detail::scatter_compat_flag_fn{*d_decomp_map, d_codepoints, d_counts, compat_decomp_flags};
  thrust::for_each_n(policy, row_iter, num_rows, scatter_compat_fn);

  // Build codepoint-indexed decomp offsets
  auto decomp_offsets = rmm::device_uvector<uint32_t>(detail::DECOMP_OFFSETS_SIZE, stream, mr);
  thrust::uninitialized_fill(policy, decomp_offsets.begin(), decomp_offsets.end(), uint32_t{0});
  // Scatter per-row token counts to the codepoint-indexed positions, then
  // exclusive-scan to get start offsets. The extra sentinel slot at
  // MAX_CODEPOINT+1 accumulates the total via the scan.
  thrust::scatter(
    policy, d_counts.begin(), d_counts.end(), d_codepoints.begin(), decomp_offsets.begin());
  auto const total_decomp_size = cudf::detail::sizes_to_offsets(
    decomp_offsets.begin(), decomp_offsets.end(), decomp_offsets.begin(), 0, stream);

  // Fill decomp_table
  auto decomp_table    = rmm::device_uvector<uint32_t>(total_decomp_size, stream, mr);
  auto write_tokens_fn = detail::write_decomp_tokens_fn{
    *d_decomp_map, apply_compat, d_codepoints, decomp_offsets, decomp_table};
  thrust::for_each_n(policy, row_iter, num_rows, write_tokens_fn);

  if (form != unicode_normalization_form::NFC && form != unicode_normalization_form::NFKC) {
    _impl = std::make_unique<unicode_normalizer_impl>(std::move(decomp_offsets),
                                                      std::move(decomp_table),
                                                      std::move(ccc_table),
                                                      std::move(compat_decomp_flags),
                                                      rmm::device_uvector<uint64_t>(0, stream, mr),
                                                      rmm::device_uvector<uint32_t>(0, stream, mr),
                                                      form);
    return;
  }

  // Build composition table (NFC/NFKC only)
  auto d_comp_keys    = rmm::device_uvector<uint64_t>(num_rows, stream, mr);
  auto d_comp_values  = rmm::device_uvector<uint32_t>(num_rows, stream, mr);
  auto build_table_fn = detail::build_comp_table_fn{*d_decomp_map,
                                                    d_codepoints,
                                                    d_counts,
                                                    ccc_table,
                                                    compat_decomp_flags,
                                                    d_comp_keys,
                                                    d_comp_values};
  thrust::for_each_n(policy, row_iter, num_rows, build_table_fn);

  auto zero_fn = detail::is_zero_comp_key{};
  thrust::remove_if(
    policy, d_comp_values.begin(), d_comp_values.end(), d_comp_keys.begin(), zero_fn);
  auto const end_itr = thrust::remove(policy, d_comp_keys.begin(), d_comp_keys.end(), uint64_t{0});
  auto const comp_size = cuda::std::distance(d_comp_keys.begin(), end_itr);
  d_comp_keys.resize(comp_size, stream);
  d_comp_values.resize(comp_size, stream);

  thrust::sort_by_key(policy, d_comp_keys.begin(), d_comp_keys.end(), d_comp_values.begin());

  _impl = std::make_unique<unicode_normalizer_impl>(std::move(decomp_offsets),
                                                    std::move(decomp_table),
                                                    std::move(ccc_table),
                                                    std::move(compat_decomp_flags),
                                                    std::move(d_comp_keys),
                                                    std::move(d_comp_values),
                                                    form);
}

unicode_normalizer::~unicode_normalizer() {}

std::unique_ptr<unicode_normalizer> create_unicode_normalizer(cudf::table_view const& unicode_data,
                                                              unicode_normalization_form form,
                                                              cuda::stream_ref stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<unicode_normalizer>(unicode_data, form, stream, mr);
}

namespace detail {
namespace {
/**
 * Transitively decompose a single Unicode codepoint and invoke `fn` with the result.
 * Runs the full NFD/NFKD ping-pong expansion loop.  The `fn` is called as
 * `fn(buf, count)` where `buf[0..count)` holds the expanded codepoints.
 * Returns immediately for intermediate UTF-8 bytes.
 */
template <typename Fn>
__device__ void for_each_decomposed_cp(int64_t idx,
                                       cuda::std::span<char const> chars,
                                       cuda::std::span<uint32_t const> decomp_offsets,
                                       cuda::std::span<uint32_t const> decomp_table,
                                       Fn fn)
{
  if (!cudf::strings::detail::is_begin_utf8_char(chars[idx])) { return; }
  cudf::char_utf8 ch = static_cast<unsigned char>(chars[idx]);  // cast preserves high order bit
  if (ch > 0x7F) { cudf::strings::detail::to_char_utf8(chars.data() + idx, ch); }
  uint32_t buf_a[MAX_DECOMP_EXPAND];
  uint32_t buf_b[MAX_DECOMP_EXPAND];
  int32_t count_a = 1;
  buf_a[0]        = cudf::strings::detail::utf8_to_codepoint(ch);
  for (int32_t depth = 0; depth < MAX_DECOMP_DEPTH; ++depth) {
    int32_t count_b = 0;
    bool expanded   = false;
    for (int32_t i = 0; i < count_a; ++i) {
      auto const cp = buf_a[i];
      if (cp >= HANGUL_SBASE && cp <= HANGUL_SEND) {
        if (count_b + 3 <= MAX_DECOMP_EXPAND) {
          count_b += hangul_decompose(cp, buf_b + count_b);
          expanded = true;
        }
      } else if (cp > MAX_CODEPOINT) {
        if (count_b < MAX_DECOMP_EXPAND) { buf_b[count_b++] = cp; }  // out-of-range: pass through
      } else {
        auto const start = decomp_offsets[cp];
        auto const end   = decomp_offsets[cp + 1];
        if (start == end) {
          buf_b[count_b++] = cp;
        } else {
          auto copy_size =
            cuda::std::min(end - start, static_cast<uint32_t>(MAX_DECOMP_EXPAND - count_b));
          cuda::std::memcpy(
            buf_b + count_b, decomp_table.data() + start, copy_size * sizeof(uint32_t));
          count_b += copy_size;
          expanded = true;
        }
      }
    }
    cuda::std::memcpy(buf_a, buf_b, count_b * sizeof(uint32_t));
    count_a = count_b;
    if (!expanded) { break; }
  }
  fn(buf_a, count_a);
}

/**
 * Count output codepoints for the input byte at @p idx (size pass).
 * Non-lead bytes return 0.
 */
struct decompose_size_fn {
  cuda::std::span<char const> d_input_chars;
  cuda::std::span<uint32_t const> decomp_offsets;
  cuda::std::span<uint32_t const> decomp_table;

  __device__ int32_t operator()(int64_t idx) const
  {
    auto count = int32_t{0};
    auto fn    = [&count](uint32_t const*, int32_t n) { count = n; };
    for_each_decomposed_cp(idx, d_input_chars, decomp_offsets, decomp_table, fn);
    return count;
  }
};

/**
 * Write decomposed codepoints and Canonical Combining Class (CCC) values for the input byte at @p
 * idx (fill pass). Non-lead bytes are skipped. Writes to pre-scanned positions in d_out_cps /
 * d_out_ccc.
 */
struct decompose_fill_fn {
  cuda::std::span<char const> d_input_chars;
  cuda::std::span<uint32_t const> decomp_offsets;
  cuda::std::span<uint32_t const> decomp_table;
  cuda::std::span<uint8_t const> ccc_table;
  cuda::std::span<int64_t const> d_out_positions;  // exclusive-scan of expanded sizes
  cuda::std::span<uint32_t> d_out_cps;
  cuda::std::span<uint8_t> d_out_ccc;

  __device__ void operator()(int64_t idx) const
  {
    auto fn = [this, idx](uint32_t const* cps, int32_t count) {
      auto const out_pos = d_out_positions[idx];
      for (int32_t i = 0; i < count; ++i) {
        auto const cp          = cps[i];
        d_out_cps[out_pos + i] = cp;
        d_out_ccc[out_pos + i] = (cp <= MAX_CODEPOINT) ? ccc_table[cp] : 0u;
      }
    };
    for_each_decomposed_cp(idx, d_input_chars, decomp_offsets, decomp_table, fn);
  }
};

/**
 * Stable-sort combining mark runs within a string's codepoint slice.
 * One invocation per string; insertion-sort each maximal run of CCC>0 marks.
 */
struct reorder_fn {
  cuda::std::span<uint32_t> d_cps;
  cuda::std::span<uint8_t> d_ccc;
  cuda::std::span<int64_t const> d_str_cp_offsets;

  __device__ void operator()(cudf::size_type str_idx) const
  {
    auto const cp_start = d_str_cp_offsets[str_idx];
    auto const cp_end   = d_str_cp_offsets[str_idx + 1];
    auto run_start      = cp_start;
    for (int64_t i = cp_start; i <= cp_end; ++i) {
      bool const is_combining = (i < cp_end) && (d_ccc[i] > 0);
      if (is_combining) { continue; }
      auto const run_len = i - run_start;
      if (run_len > 1) {
        // upper_bound locates the insertion point in O(log n)
        for (int64_t j = run_start + 1; j < i; ++j) {
          auto const ccc_j = d_ccc[j];
          auto const ins =
            thrust::upper_bound(thrust::seq, d_ccc.begin() + run_start, d_ccc.begin() + j, ccc_j) -
            d_ccc.begin();
          if (ins < j) {
            auto const cps_begin = d_cps.begin() + j;
            cuda::std::rotate(d_cps.begin() + ins, cps_begin, cps_begin + 1);
            auto const ccc_begin = d_ccc.begin() + j;
            cuda::std::rotate(d_ccc.begin() + ins, ccc_begin, ccc_begin + 1);
          }
        }
      }
      run_start = i + 1;
    }
  }
};

/**
 * Canonical composition pass (NFC/NFKC only).
 * One invocation per string.  The composition table is small (~600 entries,
 * ~7 KB) and accessed by all strings, so it stays L2-hot throughout execution.
 * Consumed codepoints are zeroed and compacted during the UTF-8 encoding pass.
 */
struct compose_fn {
  cuda::std::span<uint32_t> d_cps;
  cuda::std::span<uint8_t> d_ccc;
  cuda::std::span<int64_t const> d_str_cp_offsets;
  cuda::std::span<uint64_t const> comp_keys;
  cuda::std::span<uint32_t const> comp_values;

  __device__ void operator()(cudf::size_type str_idx) const
  {
    auto const cp_start  = d_str_cp_offsets[str_idx];
    auto const cp_end    = d_str_cp_offsets[str_idx + 1];
    int64_t last_starter = -1;
    uint8_t last_class   = 0;

    for (int64_t i = cp_start; i < cp_end; ++i) {
      if (d_cps[i] == 0) { continue; }  // already consumed
      uint8_t const ccc = d_ccc[i];
      if (last_starter < 0) {
        last_starter = ccc == 0 ? i : last_starter;
        last_class   = ccc;
        continue;
      }
      if (ccc == 0) {
        // Consume hangul if composed value is found.  Otherwise, treat as a new starter.
        if (last_class == 0) {
          auto const composed_hangul = hangul_compose(d_cps[last_starter], d_cps[i]);
          if (composed_hangul != 0) {
            d_cps[last_starter] = composed_hangul;
            d_cps[i]            = 0;
            d_ccc[i]            = 0;
            continue;
          }
        }
        last_starter = i;
      } else {
        // Combining mark: compose with last_starter
        if (last_class < ccc) {
          auto const key = (static_cast<uint64_t>(d_cps[last_starter]) << 32) | d_cps[i];
          auto const it = thrust::lower_bound(thrust::seq, comp_keys.begin(), comp_keys.end(), key);
          if (it != comp_keys.end() && *it == key) {
            d_cps[last_starter] = comp_values[cuda::std::distance(comp_keys.begin(), it)];
            d_cps[i]            = 0;
            d_ccc[i]            = 0;
            continue;
          }
        }
      }
      last_class = ccc;
    }
  }
};

/**
 * NFC/NFKC quick-check predicate.
 *
 * Returns true for the first byte of any UTF-8 sequence whose codepoint
 * requires the full normalization pipeline:
 *   - Non-zero CCC (combining mark): may need reorder or table-based composition.
 *   - Hangul V jamo (U+1161–U+1175) or T jamo (U+11A8–U+11C2): NFC_QC=Maybe;
 *     can compose algorithmically with a preceding L or LV syllable.
 *   - Compat-decomp or singleton-canonical flag: unstable under NFC/NFKC.
 *
 * If no such byte exists the column is already in NFC/NFKC form and the
 * early-return copy path fires.
 */
struct nfc_quick_check_fn {
  cuda::std::span<char const> chars;
  cuda::std::span<uint8_t const> ccc_table;
  cuda::std::span<cudf::bitmask_type const> compat_flags;

  __device__ bool operator()(int64_t idx) const
  {
    if (!cudf::strings::detail::is_begin_utf8_char(chars[idx])) { return false; }
    auto ch = static_cast<cudf::char_utf8>(chars[idx]);
    if (ch > 0x7F) { cudf::strings::detail::to_char_utf8(chars.data() + idx, ch); }
    auto const cp = cudf::strings::detail::utf8_to_codepoint(ch);
    if (cp > MAX_CODEPOINT) { return false; }
    if (ccc_table[cp] > 0) { return true; }
    if ((cp >= HANGUL_VBASE && cp <= HANGUL_VEND) || (cp >= HANGUL_TSTART && cp <= HANGUL_TEND)) {
      return true;
    }
    return !compat_flags.empty() &&
           cudf::bit_is_set(compat_flags.data(), static_cast<cudf::size_type>(cp));
  }
};

/**
 * Output codepoints to UTF-8 bytes.
 */
struct output_fn {
  uint32_t* d_cps;
  int64_t* d_scp;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets{};

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const cp_start   = d_scp[idx];
    auto const cp_end     = d_scp[idx + 1];
    cudf::size_type bytes = 0;
    auto d_output         = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (int64_t i = cp_start; i < cp_end; ++i) {
      auto const cp = d_cps[i];
      if (cp == 0) { continue; }
      auto const utf8 = cudf::strings::detail::codepoint_to_utf8(cp);
      bytes += cudf::strings::detail::bytes_in_char_utf8(utf8);
      if (d_output != nullptr) {
        cudf::strings::detail::from_char_utf8(utf8, d_output);
        d_output += cudf::strings::detail::bytes_in_char_utf8(utf8);
      }
    }
    if (d_sizes) { d_sizes[idx] = bytes; }
  }
};

}  // namespace

std::unique_ptr<cudf::column> normalize_unicode(cudf::strings_column_view const& input,
                                                unicode_normalizer const& normalizer,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto const [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const chars_size = last_offset - first_offset;
  if (chars_size == 0) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }

  auto const& p          = *normalizer._impl;
  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const policy      = rmm::exec_policy_nosync(stream, temp_mr);
  auto const byte_iter   = cuda::make_counting_iterator(int64_t{0});
  auto const d_raw_chars = input.chars_begin(stream) + first_offset;
  auto const chars_span  = cuda::std::span<char const>(d_raw_chars, chars_size);

  // NFC/NFKC quick check: scan for any codepoint that is NFC_QC=No or NFC_QC=Maybe
  // (non-zero Canonical Combining Class (CCC), Hangul V/T jamo, compat decomp, singleton canonical,
  // script exclusion, or non-starter decomposition).  If none found the column is already
  // normalized.
  if (p.form == unicode_normalization_form::NFC || p.form == unicode_normalization_form::NFKC) {
    auto nfc_qc_fn = detail::nfc_quick_check_fn{chars_span, p.ccc_table, p.compat_decomp_flags};
    if (!cudf::detail::any_of(byte_iter, byte_iter + chars_size, nfc_qc_fn, stream)) {
      return std::make_unique<cudf::column>(input.parent(), stream, mr);
    }
  }

  // Decomposition: first count output codepoints per input byte
  auto expanded_sizes = rmm::device_uvector<int32_t>(chars_size + 1, stream, temp_mr);
  thrust::uninitialized_fill(policy, expanded_sizes.begin(), expanded_sizes.end(), int32_t{0});
  auto size_fn = detail::decompose_size_fn{chars_span, p.decomp_offsets, p.decomp_table};
  thrust::transform(policy, byte_iter, byte_iter + chars_size, expanded_sizes.begin(), size_fn);

  // Exclusive scan to get per-byte output positions
  auto out_positions   = rmm::device_uvector<int64_t>(chars_size + 1, stream, temp_mr);
  auto const total_cps = cudf::detail::sizes_to_offsets(
    expanded_sizes.begin(), expanded_sizes.end(), out_positions.begin(), int64_t{0}, stream);

  // Fill codepoints and CCCs at pre-scanned positions
  auto cps = rmm::device_uvector<uint32_t>(total_cps, stream, temp_mr);
  auto ccc = rmm::device_uvector<uint8_t>(total_cps, stream, temp_mr);
  thrust::fill(policy, cps.begin(), cps.end(), uint32_t{0});
  auto decomp_fill_fn = detail::decompose_fill_fn{
    chars_span, p.decomp_offsets, p.decomp_table, p.ccc_table, out_positions, cps, ccc};
  thrust::for_each_n(policy, byte_iter, chars_size, decomp_fill_fn);

  // Build per-string codepoint offset boundaries
  auto str_cp_offsets = rmm::device_uvector<int64_t>(input.size() + 1, stream, temp_mr);
  {
    auto const input_char_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    auto const d_out_pos   = out_positions.data();
    auto const d_exp_sizes = expanded_sizes.data();
    int64_t const first    = first_offset;
    thrust::transform(policy,
                      input_char_offsets,
                      input_char_offsets + input.size() + 1,
                      str_cp_offsets.begin(),
                      cuda::proclaim_return_type<int64_t>(
                        [d_out_pos, d_exp_sizes, first] __device__(int64_t offset) {
                          auto const local = offset - first;
                          if (local <= 0) { return 0L; }
                          return d_out_pos[local - 1] + d_exp_sizes[local - 1];
                        }));
  }
  expanded_sizes.release();
  out_positions.release();

  auto const row_iter = cuda::make_counting_iterator(cudf::size_type{0});
  auto const d_cps    = cps.data();
  auto const d_scp    = str_cp_offsets.data();

  // Canonical Reorder using codepoints (cps) and Canonical Combining Class (ccc) values
  thrust::for_each_n(policy, row_iter, input.size(), detail::reorder_fn{cps, ccc, str_cp_offsets});
  // Canonical Composition (NFC/NFKC only)
  // Run composition for NFC/NFKC regardless of whether the table has entries:
  // Hangul algorithmic composition is inside compose_fn and requires no table.
  if (p.form == unicode_normalization_form::NFC || p.form == unicode_normalization_form::NFKC) {
    auto fn = detail::compose_fn{cps, ccc, str_cp_offsets, p.comp_keys, p.comp_values};
    thrust::for_each_n(policy, row_iter, input.size(), fn);
  }
  ccc.release();

  auto output_fn = detail::output_fn{d_cps, d_scp};
  auto [offsets_column, chars] =
    cudf::strings::detail::make_strings_children(output_fn, input.size(), stream, mr);
  return cudf::make_strings_column(input.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> normalize_unicode(cudf::strings_column_view const& input,
                                                unicode_normalizer const& normalizer,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_unicode(input, normalizer, stream, mr);
}

}  // namespace nvtext
