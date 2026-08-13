/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/tokenize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
using hash_value_type    = string_hasher_type::result_type;

/**
 * @brief Hasher function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (string).
 */
struct vocab_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  // used by insert
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    return hasher(d_strings.element<cudf::string_view>(index));
  }
  // used by find
  __device__ hash_value_type operator()(cudf::string_view const& s) const { return hasher(s); }
};

/**
 * @brief Equal function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (string).
 */
struct vocab_equal {
  cudf::column_device_view const d_strings;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are expected to be unique
  }
  // used by find
  __device__ bool operator()(cudf::string_view const& lhs, cudf::size_type rhs) const noexcept
  {
    return d_strings.element<cudf::string_view>(rhs) == lhs;
  }
};

using probe_scheme        = cuco::linear_probing<1, vocab_hasher>;
using cuco_storage        = cuco::storage<1>;
using vocabulary_map_type = cuco::static_map<cudf::size_type,
                                             cudf::size_type,
                                             cuco::extent<std::size_t>,
                                             cuda::thread_scope_device,
                                             vocab_equal,
                                             probe_scheme,
                                             rmm::mr::polymorphic_allocator<char>,
                                             cuco_storage>;
}  // namespace
}  // namespace detail

// since column_device_view::create returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type in a maintainable way
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             rmm::cuda_stream_view,
                                             rmm::device_async_resource_ref>;

struct tokenize_vocabulary::tokenize_vocabulary_impl {
  std::unique_ptr<cudf::column> const vocabulary;
  col_device_view const d_vocabulary;
  std::unique_ptr<detail::vocabulary_map_type> vocabulary_map;

  auto get_map_ref() const { return vocabulary_map->ref(cuco::op::find); }

  tokenize_vocabulary_impl(std::unique_ptr<cudf::column>&& vocab,
                           col_device_view&& d_vocab,
                           std::unique_ptr<detail::vocabulary_map_type>&& map)
    : vocabulary(std::move(vocab)), d_vocabulary(std::move(d_vocab)), vocabulary_map(std::move(map))
  {
  }
};

struct key_pair {
  __device__ auto operator()(cudf::size_type idx) const noexcept
  {
    return cuco::make_pair(idx, idx);
  }
};

tokenize_vocabulary::tokenize_vocabulary(cudf::strings_column_view const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not input.is_empty(), "vocabulary must not be empty");
  CUDF_EXPECTS(not input.has_nulls(), "vocabulary must not have nulls");

  // need to hold a copy of the input
  auto vocabulary   = std::make_unique<cudf::column>(input.parent(), stream, mr);
  auto d_vocabulary = cudf::column_device_view::create(vocabulary->view(), stream);

  auto vocab_map = std::make_unique<detail::vocabulary_map_type>(
    static_cast<size_t>(vocabulary->size() * 2),
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    detail::vocab_equal{*d_vocabulary},
    detail::probe_scheme{detail::vocab_hasher{*d_vocabulary}},
    cuco::thread_scope_device,
    detail::cuco_storage{},
    rmm::mr::polymorphic_allocator<char>{mr},
    stream.value());

  // the row index is the token id (value for each key in the map)
  auto iter = cudf::detail::make_counting_transform_iterator(0, key_pair{});
  vocab_map->insert_async(iter, iter + vocabulary->size(), stream.value());

  _impl = new tokenize_vocabulary_impl(
    std::move(vocabulary), std::move(d_vocabulary), std::move(vocab_map));
}
tokenize_vocabulary::~tokenize_vocabulary() { delete _impl; }

std::unique_ptr<tokenize_vocabulary> load_vocabulary(cudf::strings_column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<tokenize_vocabulary>(input, stream, mr);
}

namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the warp-parallel function is used to compute the output sizes.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the vocab_tokenize benchmark results.
 */
constexpr cudf::size_type AVG_CHAR_BYTES_THRESHOLD = 128;

constexpr int block_size = 256;

__device__ bool is_delimiter(cudf::string_view const& d_delimiters, cudf::char_utf8 chr)
{
  return d_delimiters.empty() ? (chr <= ' ') :  // whitespace check
           thrust::any_of(thrust::seq,
                          d_delimiters.begin(),
                          d_delimiters.end(),
                          [chr] __device__(cudf::char_utf8 c) { return c == chr; });
}

CUDF_KERNEL void mark_delimiters_fn(char const* d_chars,
                                    int64_t chars_size,
                                    cudf::string_view const d_delimiter,
                                    int8_t* d_results)

{
  auto idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= chars_size) { return; }
  auto const ptr = d_chars + idx;
  if (cudf::strings::detail::is_utf8_continuation_char(*ptr)) { return; }
  cudf::char_utf8 chr = 0;
  auto ch_size        = cudf::strings::detail::to_char_utf8(ptr, chr);
  auto const output   = is_delimiter(d_delimiter, chr);
  while (ch_size > 0) {
    d_results[idx++] = output;
    --ch_size;
  }
}

CUDF_KERNEL void token_counts_fn(cudf::column_device_view const d_strings,
                                 cudf::string_view const d_delimiter,
                                 cudf::size_type* d_counts,
                                 int8_t* d_results)
{
  // string per warp
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / cudf::detail::warp_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) {
    d_counts[str_idx] = 0;
    return;
  }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    d_counts[str_idx] = 0;
    return;
  }

  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto const offsets     = d_strings.child(cudf::strings_column_view::offsets_column_index);
  auto const offsets_itr = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
  auto const offset = offsets_itr[str_idx + d_strings.offset()] - offsets_itr[d_strings.offset()];
  auto const chars_begin = d_strings.data<char>() + offsets_itr[d_strings.offset()];

  auto const begin        = d_str.data();
  auto const end          = begin + d_str.size_bytes();
  auto const d_output     = d_results + offset;
  auto const d_output_end = d_output + d_str.size_bytes();

  cudf::size_type count = 0;
  if (lane_idx == 0) {
    cudf::char_utf8 chr = 0;
    auto ch_size        = cudf::strings::detail::to_char_utf8(begin, chr);
    auto output         = 1;
    if (begin > chars_begin) {
      auto ptr = begin - 1;
      while (ptr > chars_begin && cudf::strings::detail::is_utf8_continuation_char(*ptr)) {
        --ptr;
      }
      cudf::strings::detail::to_char_utf8(ptr, chr);
      output = !is_delimiter(d_delimiter, chr);
    }
    auto ptr = d_output;
    while (ch_size > 0) {
      *ptr++ = output;
      --ch_size;
    }
    count = ((begin + ch_size) == end);
  }
  warp.sync();

  for (auto itr = d_output + lane_idx + 1; itr < d_output_end; itr += cudf::detail::warp_size) {
    // add one if at the edge of a token or if at the string's end
    if (*itr) {
      count += !(*(itr - 1));
    } else {
      count += (itr + 1 == d_output_end);
    }
  }
  warp.sync();

  // add up the counts from the other threads to compute the total token count for this string
  auto const total_count = cg::reduce(warp, count, cg::plus<cudf::size_type>{});
  if (lane_idx == 0) { d_counts[str_idx] = total_count; }
}

/**
 * @brief Tokenizes each string and uses the map to assign token id values
 *
 * @tparam MapRefType Type of the static_map reference for calling find()
 */
template <typename MapRefType>
struct vocabulary_tokenizer_fn {
  cudf::column_device_view const d_strings;
  cudf::string_view const d_delimiter;
  MapRefType d_map;
  cudf::size_type const default_id;
  cudf::detail::input_offsetalator d_offsets;
  cudf::size_type* d_results;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return; }

    auto const d_str = d_strings.element<cudf::string_view>(idx);
    characters_tokenizer tokenizer(d_str, d_delimiter);
    auto d_tokens = d_results + d_offsets[idx];

    cudf::size_type token_idx = 0;
    while (tokenizer.next_token()) {
      auto const pos   = tokenizer.token_byte_positions();
      auto const token = cudf::string_view{d_str.data() + pos.first, (pos.second - pos.first)};
      // lookup token in map
      auto const itr = d_map.find(token);
      auto const id  = (itr != d_map.end()) ? itr->second : default_id;
      // set value into the output
      d_tokens[token_idx++] = id;
    }
  }
};

template <typename MapRefType>
struct transform_tokenizer_fn {
  cudf::string_view const d_delimiter;
  MapRefType d_map;
  cudf::size_type const default_id;

  __device__ cudf::size_type operator()(cudf::string_view d_str) const
  {
    auto const begin = d_str.data();
    auto const end   = begin + d_str.size_bytes();

    auto itr = begin;
    while (itr < end) {
      cudf::char_utf8 chr = 0;
      auto const ch_size  = cudf::strings::detail::to_char_utf8(itr, chr);
      if (!is_delimiter(d_delimiter, chr)) break;
      itr += ch_size;
    }

    auto const size  = static_cast<cudf::size_type>(cuda::std::distance(itr, end));
    auto const token = cudf::string_view{itr, size};
    // lookup token in map
    auto const fitr = d_map.find(token);
    return (fitr != d_map.end()) ? fitr->second : default_id;
  }
};

}  // namespace

std::unique_ptr<cudf::column> tokenize_with_vocabulary(cudf::strings_column_view const& input,
                                                       tokenize_vocabulary const& vocabulary,
                                                       cudf::string_scalar const& delimiter,
                                                       cudf::size_type default_id,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.size() == input.null_count()) { return cudf::make_empty_column(output_type); }

  // count the tokens per string and build the offsets from the counts
  auto const d_strings   = cudf::column_device_view::create(input.parent(), stream);
  auto const d_delimiter = delimiter.value(stream);
  auto map_ref           = vocabulary._impl->get_map_ref();

  if ((input.chars_size(stream) / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    auto const zero_itr = cuda::counting_iterator<cudf::size_type>{0};
    auto d_sizes        = rmm::device_uvector<cudf::size_type>(input.size(), stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      zero_itr,
                      zero_itr + input.size(),
                      d_sizes.begin(),
                      strings_tokenizer{*d_strings, d_delimiter});
    auto [token_offsets, total_count] =
      cudf::detail::make_offsets_child_column(d_sizes.begin(), d_sizes.end(), stream, mr);

    // build the output column to hold all the token ids
    auto tokens = cudf::make_numeric_column(
      output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d_tokens  = tokens->mutable_view().data<cudf::size_type>();
    auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(token_offsets->view());
    vocabulary_tokenizer_fn<decltype(map_ref)> tokenizer{
      *d_strings, d_delimiter, map_ref, default_id, d_offsets, d_tokens};
    thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       zero_itr,
                       input.size(),
                       tokenizer);
    return cudf::make_lists_column(input.size(),
                                   std::move(token_offsets),
                                   std::move(tokens),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  // longer strings perform better with warp-parallel approach

  auto const first_offset  = (input.offset() == 0) ? 0
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? input.chars_size(stream)
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  rmm::device_uvector<cudf::size_type> d_token_counts(input.size(), stream);
  rmm::device_uvector<int8_t> d_marks(chars_size, stream);

  // mark position of all delimiters
  auto grid_chars = cudf::detail::grid_1d{chars_size, block_size};
  mark_delimiters_fn<<<grid_chars.num_blocks,
                       grid_chars.num_threads_per_block,
                       0,
                       stream.value()>>>(d_input_chars, chars_size, d_delimiter, d_marks.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // launch warp per string to compute token counts
  constexpr cudf::thread_index_type warp_size = cudf::detail::warp_size;
  cudf::detail::grid_1d grid{input.size() * warp_size, block_size};
  token_counts_fn<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, d_delimiter, d_token_counts.data(), d_marks.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  auto [token_offsets, total_count] = cudf::detail::make_offsets_child_column(
    d_token_counts.begin(), d_token_counts.end(), stream, mr);

  auto d_tmp_offsets = rmm::device_uvector<int64_t>(total_count + 1, stream);
  d_tmp_offsets.set_element(total_count, chars_size, stream);
  cudf::detail::copy_if_async(
    cuda::counting_iterator<int64_t>{0},
    cuda::counting_iterator<int64_t>{chars_size},
    d_tmp_offsets.begin(),
    [d_marks = d_marks.data()] __device__(auto idx) -> bool {
      if (idx == 0) return true;
      return d_marks[idx] && !d_marks[idx - 1];
    },
    stream);

  auto tmp_offsets =
    std::make_unique<cudf::column>(std::move(d_tmp_offsets), rmm::device_buffer{}, 0);
  auto const tmp_input = cudf::column_view(
    input.parent().type(), total_count, d_input_chars, nullptr, 0, 0, {tmp_offsets->view()});

  auto const d_tmp_strings = cudf::column_device_view::create(tmp_input, stream);

  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_tokens = tokens->mutable_view().data<cudf::size_type>();

  transform_tokenizer_fn<decltype(map_ref)> tokenizer{d_delimiter, map_ref, default_id};
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    d_tmp_strings->begin<cudf::string_view>(),
                    d_tmp_strings->end<cudf::string_view>(),
                    d_tokens,
                    tokenizer);

  return cudf::make_lists_column(input.size(),
                                 std::move(token_offsets),
                                 std::move(tokens),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> tokenize_with_vocabulary(cudf::strings_column_view const& input,
                                                       tokenize_vocabulary const& vocabulary,
                                                       cudf::string_scalar const& delimiter,
                                                       cudf::size_type default_id,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize_with_vocabulary(input, vocabulary, delimiter, default_id, stream, mr);
}

}  // namespace nvtext
