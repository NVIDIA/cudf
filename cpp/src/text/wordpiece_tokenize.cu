/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/wordpiece_tokenize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cub/cub.cuh>
#include <cuco/static_map.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/remove.h>

namespace nvtext {
namespace detail {
namespace {

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
using hash_value_type    = string_hasher_type::result_type;

/**
 * @brief Hasher used for vocabulary map
 */
struct vocab_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    return hasher(d_strings.element<cudf::string_view>(index));
  }
  __device__ hash_value_type operator()(cudf::string_view const& s) const { return hasher(s); }
};
/**
 * @brief Equality operator for vocabulary map
 */
struct vocab_equal {
  cudf::column_device_view const d_strings;
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are expected to be unique
  }
  __device__ bool operator()(cudf::string_view const& lhs, cudf::size_type rhs) const noexcept
  {
    return d_strings.element<cudf::string_view>(rhs) == lhs;
  }
};

using cuco_storage        = cuco::storage<1>;
using probe_scheme        = cuco::linear_probing<1, vocab_hasher>;
using vocabulary_map_type = cuco::static_map<cudf::size_type,
                                             cudf::size_type,
                                             cuco::extent<std::size_t>,
                                             cuda::thread_scope_thread,
                                             vocab_equal,
                                             probe_scheme,
                                             rmm::mr::polymorphic_allocator<char>,
                                             cuco_storage>;

/**
 * @brief Hasher used for the subword vocabulary map
 */
struct sub_vocab_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    auto const d_str = d_strings.element<cudf::string_view>(index);
    // skip over the '##' prefix
    return hasher(cudf::string_view(d_str.data() + 2, d_str.size_bytes() - 2));
  }
  __device__ hash_value_type operator()(cudf::string_view const& s) const { return hasher(s); }
};
/**
 * @brief Equality operator used for the subword vocabulary map
 *
 * The subwords start with '##' prefix in the original vocabulary map
 */
struct sub_vocab_equal {
  cudf::column_device_view const d_strings;
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are expected to be unique
  }
  __device__ bool operator()(cudf::string_view const& lhs, cudf::size_type rhs) const noexcept
  {
    auto const d_str = d_strings.element<cudf::string_view>(rhs);
    // skip over the '##' prefix
    return lhs == cudf::string_view(d_str.data() + 2, d_str.size_bytes() - 2);
  }
};

// This 2nd subword map helps avoid requiring temporary strings in device code
using sub_probe_scheme        = cuco::linear_probing<1, sub_vocab_hasher>;
using sub_vocabulary_map_type = cuco::static_map<cudf::size_type,
                                                 cudf::size_type,
                                                 cuco::extent<std::size_t>,
                                                 cuda::thread_scope_thread,
                                                 sub_vocab_equal,
                                                 sub_probe_scheme,
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

/**
 * @brief Internal class manages all the data held by the vocabulary object
 */
struct wordpiece_vocabulary::wordpiece_vocabulary_impl {
  std::unique_ptr<cudf::column> const vocabulary;  // copy of the original vocabulary input
  col_device_view const d_vocabulary;
  std::unique_ptr<detail::vocabulary_map_type> vocabulary_map;
  std::unique_ptr<detail::sub_vocabulary_map_type> vocabulary_sub_map;
  cudf::size_type unk_id{};  // resolved [UNK] id from vocabulary

  auto get_map_ref() const { return vocabulary_map->ref(cuco::op::find); }
  auto get_sub_map_ref() const { return vocabulary_sub_map->ref(cuco::op::find); }

  wordpiece_vocabulary_impl(std::unique_ptr<cudf::column>&& vocab,
                            col_device_view&& d_vocab,
                            std::unique_ptr<detail::vocabulary_map_type>&& map,
                            std::unique_ptr<detail::sub_vocabulary_map_type>&& sub_map,
                            cudf::size_type unk_id)
    : vocabulary(std::move(vocab)),
      d_vocabulary(std::move(d_vocab)),
      vocabulary_map(std::move(map)),
      vocabulary_sub_map(std::move(sub_map)),
      unk_id{unk_id}
  {
  }
};

namespace {
/**
 * @brief Identifies the column indices as the values in the vocabulary map
 */
struct key_pair {
  __device__ auto operator()(cudf::size_type idx) const noexcept
  {
    return cuco::make_pair(idx, idx);
  }
};

/**
 * @brief For filtering the subword ('##' prefixed) entries in the vocabulary
 */
struct copy_pieces_fn {
  cudf::column_device_view d_strings;
  __device__ bool operator()(cudf::size_type idx)
  {
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.size_bytes() < 2) { return false; }
    return (d_str.data()[0] == '#') and (d_str.data()[1] == '#');
  }
};

/**
 * @brief Resolves the [UNK] entry from the vocabulary
 *
 * This saves inlining the lookup code in several places in device code.
 */
template <typename MapRefType>
struct resolve_unk_id {
  MapRefType d_map;
  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    // look for both since the normalizer may change the case to match the vocab table
    auto const unk = idx == 0 ? cudf::string_view("[UNK]", 5) : cudf::string_view("[unk]", 5);
    auto const fnd = d_map.find(unk);
    return fnd != d_map.end() ? fnd->second : -1;
  }
};

}  // namespace

wordpiece_vocabulary::wordpiece_vocabulary(cudf::strings_column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not input.is_empty(), "vocabulary must not be empty", std::invalid_argument);
  CUDF_EXPECTS(not input.has_nulls(), "vocabulary must not have nulls", std::invalid_argument);

  // hold a copy of the input (not expected to be very large)
  auto vocabulary   = std::make_unique<cudf::column>(input.parent(), stream, mr);
  auto d_vocabulary = cudf::column_device_view::create(vocabulary->view(), stream);

  // build the vocabulary map: each row is a single term and is the key for the map
  auto vocab_map = std::make_unique<detail::vocabulary_map_type>(
    static_cast<size_t>(vocabulary->size() * 2),
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    detail::vocab_equal{*d_vocabulary},
    detail::probe_scheme{detail::vocab_hasher{*d_vocabulary}},
    cuco::thread_scope_thread,
    detail::cuco_storage{},
    rmm::mr::polymorphic_allocator<char>{mr},
    stream.value());
  // the row index is the token id (data value for each key in the map)
  auto iter = cudf::detail::make_counting_transform_iterator(0, key_pair{});
  vocab_map->insert_async(iter, iter + vocabulary->size(), stream.value());
  auto const zero_itr = cuda::counting_iterator<cudf::size_type>{0};

  // get the indices of all the ## prefixed entries
  auto sub_map_indices = rmm::device_uvector<cudf::size_type>(vocabulary->size(), stream);
  auto const end       = cudf::detail::copy_if(
    zero_itr,
    cuda::counting_iterator{static_cast<cudf::size_type>(sub_map_indices.size())},
    sub_map_indices.begin(),
    copy_pieces_fn{*d_vocabulary},
    stream);
  sub_map_indices.resize(cuda::std::distance(sub_map_indices.begin(), end), stream);

  // build a 2nd map with just the ## prefixed items
  auto vocab_sub_map = std::make_unique<detail::sub_vocabulary_map_type>(
    sub_map_indices.size() * 2,
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    detail::sub_vocab_equal{*d_vocabulary},
    detail::sub_probe_scheme{detail::sub_vocab_hasher{*d_vocabulary}},
    cuco::thread_scope_thread,
    detail::cuco_storage{},
    rmm::mr::polymorphic_allocator<char>{mr},
    stream.value());
  // insert them without the '##' prefix since that is how they will be looked up
  auto iter_sub = thrust::make_transform_iterator(sub_map_indices.begin(), key_pair{});
  vocab_sub_map->insert_async(iter_sub, iter_sub + sub_map_indices.size(), stream.value());

  // prefetch the [unk] vocab entry
  auto unk_ids = rmm::device_uvector<cudf::size_type>(2, stream);
  auto d_map   = vocab_map->ref(cuco::op::find);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    zero_itr,
                    zero_itr + unk_ids.size(),
                    unk_ids.begin(),
                    resolve_unk_id<decltype(d_map)>{d_map});
  auto const id0    = unk_ids.front_element(stream);
  auto const id1    = unk_ids.back_element(stream);
  auto const unk_id = id0 >= 0 ? id0 : id1;

  _impl = std::make_unique<wordpiece_vocabulary_impl>(std::move(vocabulary),
                                                      std::move(d_vocabulary),
                                                      std::move(vocab_map),
                                                      std::move(vocab_sub_map),
                                                      unk_id);
}

wordpiece_vocabulary::~wordpiece_vocabulary() {}

std::unique_ptr<wordpiece_vocabulary> load_wordpiece_vocabulary(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<wordpiece_vocabulary>(input, stream, mr);
}

namespace detail {
namespace {

constexpr auto block_size    = 128;
constexpr auto no_token      = cuda::std::numeric_limits<cudf::size_type>::max();
constexpr auto max_word_size = 200;  // words longer than this are not tokenized

/**
 * @brief Returns a new string_view truncating the last character of the input string
 *
 * This is more efficient than using substr() which is more generic.
 */
__device__ cudf::string_view remove_last_char(cudf::string_view d_str)
{
  if (d_str.size_bytes() < 2) { return cudf::string_view(); }
  auto const begin = d_str.data();
  auto end         = begin + d_str.size_bytes() - 1;
  while ((end > begin) && cudf::strings::detail::is_utf8_continuation_char(*end)) {
    --end;
  }
  auto const size = static_cast<cudf::size_type>(cuda::std::distance(begin, end));
  return cudf::string_view(begin, size);
}

/**
 * @brief The wordpiece tokenizer
 *
 * The given word is looked up in the d_map and if found the corresponding
 * token (integer) is returned.
 *
 * If not found, the function will iteratively remove the last character
 * from the word and check the substring exists in the d_map until the
 * the substring(s) is found. If still not found, the unk_id is returned.
 * If found, the characters removed are iteratively checked against
 * the d_sub_map until all have been located. If any of these are not found,
 * the unk_id is returned.
 *
 * Example: word="GPU" and d_map contains { ... {"G",10}, {"##U",7}, {"##P",3}, ... }
 * which means the d_sub_map contains { ... {"U",7}, {"P",3}, ... }
 * Since "GPU" is not found in d_map, the 'U' is removed and rechecked.
 * And "GP" is also not found so another character is removed leaving "G".
 * The "G" is found in d_map so now the removed characters are processed
 * starting with "PU" which is not found in d_sub_map. Removing the 'U'
 * again results in "P" which is found in d_sub_map and iterating again
 * locates 'U' in d_sub_map as well. The end result is that "GPU" produces
 * 3 tokens [10,3,7].
 *
 * @param word Word to tokenize
 * @param d_map Vocabulary to check for word and sub-words
 * @param d_sub_map Partial vocabulary of '##' entries
 * @param unk_id The unknown token id returned when no token is found
 * @param d_tokens Output token ids are returned here
 * @return The number of resolved tokens
 */
template <typename MapRefType, typename SubMapRefType>
__device__ cudf::size_type wp_tokenize_fn(cudf::string_view word,
                                          MapRefType const& d_map,
                                          SubMapRefType const& d_sub_map,
                                          cudf::size_type unk_id,
                                          cudf::size_type* d_tokens)
{
  // lookup word in map
  auto token_idx = 0;
  auto itr       = d_map.find(word);
  if (itr != d_map.end()) {
    d_tokens[token_idx++] = itr->second;
    return token_idx;
  }

  // reduce word by one character and try again
  auto piece = remove_last_char(word);
  while (!piece.empty()) {
    itr = d_map.find(piece);
    if (itr == d_map.end()) {
      piece = remove_last_char(piece);
      continue;
    }
    d_tokens[token_idx++] = itr->second;
    break;
  }
  if (piece.empty()) {
    // did not find anything; this is not common
    d_tokens[token_idx++] = unk_id;
    return token_idx;
  }

  word =
    cudf::string_view(word.data() + piece.size_bytes(), word.size_bytes() - piece.size_bytes());
  piece = word;
  while (!piece.empty()) {
    auto itr = d_sub_map.find(piece);
    if (itr == d_sub_map.end()) {
      piece = remove_last_char(piece);
      continue;
    }
    d_tokens[token_idx++] = itr->second;

    word =
      cudf::string_view(word.data() + piece.size_bytes(), word.size_bytes() - piece.size_bytes());
    piece = word;
  }
  if (!word.empty()) {
    // very uncommon
    d_tokens[0] = unk_id;
    // need to reset any previous ids too
    for (auto i = 1; i < token_idx; ++i) {
      d_tokens[i] = no_token;
    }
    token_idx = 1;
  }

  return token_idx;
}

/**
 * @brief Kernel for tokenizing all words
 *
 * Launched as a thread per edge value in d_edges.
 *
 * Each value in d_edge is the beginning of a word.
 * The kernel searches for a matching space character (or the next d_edge value)
 * to find the end of the word.
 * The result is then tokenized using the wp_tokenize_fn utility.
 *
 * @param d_edges The offset to the beginning of each word
 * @param d_chars Pointer to the characters of the input column
 * @param d_map Lookup table for the wp_tokenize_fn utility
 * @param d_sub_map 2nd lookup table for the wp_tokenize_fn utility
 * @param unk_id Unknown token id when a token cannot be resolved
 * @param d_tokens Output tokens are written here
 */
template <typename MapRefType, typename SubMapRefType>
CUDF_KERNEL void tokenize_all_kernel(cudf::device_span<int64_t const> d_edges,
                                     char const* d_chars,
                                     MapRefType const d_map,
                                     SubMapRefType const d_sub_map,
                                     cudf::size_type unk_id,
                                     cudf::size_type* d_tokens)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= (d_edges.size() - 1)) { return; }
  auto const begin    = d_chars + d_edges[idx];
  auto const end      = d_chars + d_edges[idx + 1];
  auto const word_end = thrust::find(thrust::seq, begin, end, ' ');
  auto const size     = static_cast<cudf::size_type>(cuda::std::distance(begin, word_end));
  if (size == 0) { return; }
  auto d_output = d_tokens + d_edges[idx];
  if (size >= max_word_size) {
    *d_output = unk_id;
    return;
  }
  auto const word = cudf::string_view{begin, size};
  wp_tokenize_fn(word, d_map, d_sub_map, unk_id, d_output);
}

/**
 * @brief Count the number of tokens per output row
 *
 * Uses segmented-reduce to compute the number of tokens per row.
 *
 * @param d_tokens The tokens to count
 * @param offsets The offsets for the segmented-reduce
 * @param offset Maybe non-zero if the input column has been sliced
 * @param size The number of output rows (same as the number of input rows)
 * @param stream Stream used for device allocations and kernel launches
 * @return The number of tokens per row
 */
template <typename OffsetType>
rmm::device_uvector<cudf::size_type> count_tokens(cudf::size_type const* d_tokens,
                                                  OffsetType offsets,
                                                  int64_t offset,
                                                  cudf::size_type size,
                                                  rmm::cuda_stream_view stream)
{
  auto d_counts = rmm::device_uvector<cudf::size_type>(size, stream);

  // transform iterator used for counting the number of !no_tokens
  auto const d_in = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<cudf::size_type>([d_tokens] __device__(auto idx) {
      return static_cast<cudf::size_type>(d_tokens[idx] != no_token);
    }));

  auto const d_offsets = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<int64_t>([offsets, offset] __device__(auto idx) {
      return offsets[idx] - offset;
    }));

  auto temp  = std::size_t{0};
  auto d_out = d_counts.data();
  cub::DeviceSegmentedReduce::Sum(
    nullptr, temp, d_in, d_out, size, d_offsets, d_offsets + 1, stream.value());
  auto d_temp = rmm::device_buffer{temp, stream};
  cub::DeviceSegmentedReduce::Sum(
    d_temp.data(), temp, d_in, d_out, size, d_offsets, d_offsets + 1, stream.value());

  return d_counts;
}

/**
 * @brief Compute all tokens for the input column
 *
 * @param input Input strings column
 * @param first_offset Offset to first row in chars for `input`
 * @param chars_size Size of the character data for `input`
 * @param vocabulary Vocabulary data needed by the tokenizer
 * @param stream Stream used for device allocations and kernel launches
 * @return The tokens (and non-tokens) for the input
 */
rmm::device_uvector<cudf::size_type> compute_all_tokens(
  cudf::strings_column_view const& input,
  int64_t first_offset,
  int64_t chars_size,
  wordpiece_vocabulary::wordpiece_vocabulary_impl const& vocabulary,
  rmm::cuda_stream_view stream)
{
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  // find beginnings of words
  auto d_edges = rmm::device_uvector<int64_t>(chars_size / 2L, stream);
  // beginning of a word is a non-space preceded by a space
  auto edges_end = cudf::detail::copy_if(
    cuda::counting_iterator<int64_t>{0},
    cuda::counting_iterator<int64_t>{chars_size},
    d_edges.begin(),
    [d_input_chars] __device__(auto idx) -> bool {
      if (idx == 0) { return d_input_chars[idx] == ' '; }
      return (d_input_chars[idx] != ' ' && d_input_chars[idx - 1] == ' ');
    },
    stream);

  auto const edges_count =
    input.size() + 1 + static_cast<int64_t>(cuda::std::distance(d_edges.begin(), edges_end));
  // thrust::merge has an int32 max limit currently
  CUDF_EXPECTS(edges_count < std::numeric_limits<int32_t>::max(), "words exceed internal limit");

  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const d_offsets = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<int64_t>([input_offsets, first_offset] __device__(auto idx) {
      return input_offsets[idx] - first_offset;
    }));

  // merge in the input offsets to identify words starting each row
  auto d_all_edges = [&] {
    auto d_all_edges = rmm::device_uvector<int64_t>(edges_count, stream);
    thrust::merge(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  d_offsets,
                  d_offsets + input.size() + 1,
                  d_edges.begin(),
                  edges_end,
                  d_all_edges.begin());
    d_edges.release();  // done with this
    return d_all_edges;
  }();

  auto const map_ref     = vocabulary.get_map_ref();
  auto const sub_map_ref = vocabulary.get_sub_map_ref();
  auto const unk_id      = vocabulary.unk_id;

  auto d_tokens = rmm::device_uvector<cudf::size_type>(chars_size, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_tokens.begin(),
    d_tokens.end(),
    no_token);

  cudf::detail::grid_1d grid{static_cast<cudf::size_type>(d_all_edges.size()), 512};
  tokenize_all_kernel<decltype(map_ref), decltype(sub_map_ref)>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      d_all_edges, d_input_chars, map_ref, sub_map_ref, unk_id, d_tokens.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  return d_tokens;
}

constexpr cudf::size_type no_word = cuda::std::numeric_limits<cudf::size_type>::max();
constexpr int64_t no_word64       = cuda::std::numeric_limits<int64_t>::max();

/**
 * @brief Find word boundaries kernel
 *
 * Launched as a warp per string in the input column.
 *
 * Finds the edges of words within each string and stores them into 'starts' and 'sizes'.
 * This kernel is used when a maximum number of words are to be processed per row.
 *
 * @param d_strings Input strings column
 * @param d_chars The beginning of the character data for d_strings
 *                already adjusted for any sliced offset
 * @param offsets The offsets for the output arrays: starts and sizes
 * @param starts The output offsets within d_chars identifying the beginning of words
 * @param sizes The output size of the words corresponding to starts
 */
template <int32_t tile_size = cudf::detail::warp_size>
CUDF_KERNEL void find_words_kernel(cudf::column_device_view const d_strings,
                                   char const* d_chars,
                                   int64_t const* offsets,
                                   int64_t* starts,
                                   cudf::size_type* sizes)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) { return; }
  auto const str_offset = static_cast<int64_t>(cuda::std::distance(d_chars, d_str.data()));

  auto const d_start_words = starts + offsets[str_idx];
  auto const d_word_sizes  = sizes + offsets[str_idx];
  auto const max_words     = static_cast<cudf::size_type>(offsets[str_idx + 1] - offsets[str_idx]);

  constexpr auto bytes_per_thread = 6;  // average 5 chars per word plus space
  constexpr auto words_size       = block_size * bytes_per_thread;
  __shared__ cudf::size_type s_start_words[words_size];
  __shared__ cudf::size_type s_end_words[words_size];
  // compiler is not able to find this for some reason so defining it here as well
  constexpr auto no_word = cuda::std::numeric_limits<cudf::size_type>::max();

  namespace cg     = cooperative_groups;
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<tile_size>(block);

  auto const lane_idx   = tile.thread_rank();
  auto const warp_idx   = tile.meta_group_rank();
  auto const warp_words = words_size / tile.meta_group_size();

  cudf::size_type word_count = 0;
  cudf::size_type byte_count = 0;

  auto first_word  = no_word;  // only used by lane_idx==0
  auto const begin = d_str.data();
  auto const end   = begin + d_str.size_bytes();

  auto start_words = s_start_words + (warp_idx * warp_words);
  auto end_words   = s_end_words + (warp_idx * warp_words);

  // continue until all bytes have been consumed or the max word count has been reached
  auto itr = begin + lane_idx;
  while (word_count < max_words && byte_count < d_str.size_bytes()) {
    // initialize all intermediate results
    start_words[lane_idx] = lane_idx > 0 ? no_word : first_word;
    end_words[lane_idx]   = no_word;
    for (auto j = lane_idx + tile_size; j < warp_words; j += tile_size) {
      start_words[j] = no_word;
      end_words[j]   = no_word;
    }
    tile.sync();

    cudf::size_type last_idx = 0;
    // each thread processes bytes_per_thread of the d_str
    for (auto k = lane_idx; k < warp_words && itr < end; k += tile_size) {
      // look for word starts (non-space preceded by a space)
      if ((*itr != ' ') && ((itr == begin) || (*(itr - 1) == ' '))) {
        last_idx              = (k / 2) + 1;
        start_words[last_idx] = static_cast<cudf::size_type>(cuda::std::distance(begin, itr));
      }
      // look for word ends (space preceded by non-space)
      if (((itr + 1) == end) || ((itr != begin) && (*itr == ' ') && (*(itr - 1) != ' '))) {
        auto const adjust = static_cast<cudf::size_type>(*itr != ' ');  // edge case
        last_idx          = (k / 2) + adjust;
        end_words[last_idx] =
          static_cast<cudf::size_type>(cuda::std::distance(begin, itr)) + adjust;
      }
      itr += tile_size;
    }
    tile.sync();
    // keep track of how much of start_words/end_words we used
    last_idx = cg::reduce(tile, last_idx, cg::greater<cudf::size_type>{}) + 1;

    cudf::size_type output_count = 0;
    if (lane_idx == 0) {
      // compress out the no-words
      auto const count       = static_cast<cudf::size_type>(cuda::std::distance(
        start_words, thrust::remove(thrust::seq, start_words, start_words + last_idx, no_word)));
      auto const words_found = static_cast<cudf::size_type>(cuda::std::distance(
        end_words, thrust::remove(thrust::seq, end_words, end_words + last_idx, no_word)));
      // this partially resolved word wraps around for the next iteration
      first_word   = (count > words_found) ? start_words[words_found] : no_word;
      output_count = cuda::std::min(words_found, max_words - word_count);
    }
    tile.sync();

    // copy results to the output
    auto out_starts = d_start_words + word_count;
    auto out_sizes  = d_word_sizes + word_count;
    output_count    = tile.shfl(output_count, 0);  // copy output_count to all threads
    for (auto k = lane_idx; k < output_count; k += tile_size) {
      auto const start = start_words[k];
      out_starts[k]    = start + str_offset;
      out_sizes[k]     = end_words[k] - start;
    }

    word_count += output_count;
    byte_count += tile_size * bytes_per_thread;
    tile.sync();
  }

  // fill in the remainder of the output
  auto out_starts = d_start_words + word_count;
  auto out_sizes  = d_word_sizes + word_count;
  for (auto k = lane_idx; k < (max_words - word_count); k += tile_size) {
    out_starts[k] = no_word64;
    out_sizes[k]  = no_word;
  }
}

/**
 * @brief Limiting tokenizing kernel
 *
 * Launched as a thread per d_starts (and d_sizes) values.
 *
 * This kernel is provided word boundaries as d_starts and d_sizes.
 * The start of the word at index idx is d_start[idx].
 * The size of that word is d_size[idx].
 *
 * The wp_tokenize_fn is used to output the tokens for each word
 * appropriately into d_tokens.
 *
 * @param d_starts The start of each word in d_chars
 * @param d_sizes The corresponding size of the word pointed to by d_starts
 * @param d_chars Points to the beginning of the characters of the input column
 * @param d_map Lookup table for the wp_tokenize_fn utility
 * @param d_sub_map 2nd lookup table for the wp_tokenize_fn utility
 * @param unk_id Unknown token id when a token cannot be resolved
 * @param d_tokens Output tokens are written here
 */
template <typename MapRefType, typename SubMapRefType>
CUDF_KERNEL void tokenize_kernel(cudf::device_span<int64_t const> d_starts,
                                 cudf::device_span<cudf::size_type const> d_sizes,
                                 char const* d_chars,
                                 MapRefType const d_map,
                                 SubMapRefType const d_sub_map,
                                 cudf::size_type unk_id,
                                 cudf::size_type* d_tokens)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= d_starts.size()) { return; }
  auto const size = d_sizes[idx];
  if (size <= 0 || size == no_word) { return; }
  auto const start = d_starts[idx];
  auto const begin = d_chars + start;
  auto d_output    = d_tokens + start;
  if (size >= max_word_size) {
    *d_output = unk_id;
    return;
  }
  auto const word = cudf::string_view{begin, size};
  wp_tokenize_fn(word, d_map, d_sub_map, unk_id, d_output);
}

/**
 * @brief Compute tokens limited to `max_words_per_row`
 *
 * @param input Input strings column
 * @param first_offset Offset to first row in chars for `input`
 * @param chars_size Size of the character data for `input`
 * @param max_words_per_row Maximum number of words to tokenize in each row
 * @param vocabulary Vocabulary data needed by the tokenizer
 * @param stream Stream used for device allocations and kernel launches
 * @return The tokens (and non-tokens) for the input
 */
rmm::device_uvector<cudf::size_type> compute_some_tokens(
  cudf::strings_column_view const& input,
  int64_t first_offset,
  int64_t chars_size,
  cudf::size_type max_words_per_row,
  wordpiece_vocabulary::wordpiece_vocabulary_impl const& vocabulary,
  rmm::cuda_stream_view stream)
{
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  auto const d_strings  = cudf::column_device_view::create(input.parent(), stream);
  auto max_word_offsets = rmm::device_uvector<int64_t>(input.size() + 1, stream);

  // compute max word counts for each row
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{input.size()},
                    max_word_offsets.begin(),
                    cuda::proclaim_return_type<cudf::size_type>(
                      [d_strings = *d_strings, max_words_per_row] __device__(auto idx) {
                        if (idx >= d_strings.size()) { return 0; }
                        if (d_strings.is_null(idx)) { return 0; }
                        auto const d_str = d_strings.element<cudf::string_view>(idx);
                        return cuda::std::min(max_words_per_row, d_str.size_bytes() / 2);
                      }));

  auto const max_size = cudf::detail::sizes_to_offsets(
    max_word_offsets.begin(), max_word_offsets.end(), max_word_offsets.begin(), 0, stream);

  auto start_words = rmm::device_uvector<int64_t>(max_size, stream);
  auto word_sizes  = rmm::device_uvector<cudf::size_type>(max_size, stream);

  // find start/end for each row up to max_words_per_row words;
  // store word positions in start_words and sizes in word_sizes
  constexpr cudf::thread_index_type warp_size = cudf::detail::warp_size;
  cudf::detail::grid_1d grid_find{input.size() * warp_size, block_size};
  find_words_kernel<warp_size>
    <<<grid_find.num_blocks, grid_find.num_threads_per_block, 0, stream.value()>>>(
      *d_strings, d_input_chars, max_word_offsets.data(), start_words.data(), word_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // remove the non-words
  auto const end =
    thrust::remove(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   start_words.begin(),
                   start_words.end(),
                   no_word64);
  auto const check =
    thrust::remove(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   word_sizes.begin(),
                   word_sizes.end(),
                   no_word);

  auto const total_words = static_cast<int64_t>(cuda::std::distance(start_words.begin(), end));
  // this should only trigger if there is a bug in the code above
  CUDF_EXPECTS(total_words == static_cast<int64_t>(cuda::std::distance(word_sizes.begin(), check)),
               "error resolving word locations from input column");
  start_words.resize(total_words, stream);  // always
  word_sizes.resize(total_words, stream);   // smaller

  auto const map_ref     = vocabulary.get_map_ref();
  auto const sub_map_ref = vocabulary.get_sub_map_ref();
  auto const unk_id      = vocabulary.unk_id;

  auto d_tokens = rmm::device_uvector<cudf::size_type>(chars_size, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_tokens.begin(),
    d_tokens.end(),
    no_token);

  cudf::detail::grid_1d grid{total_words, 512};
  tokenize_kernel<decltype(map_ref), decltype(sub_map_ref)>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      start_words, word_sizes, d_input_chars, map_ref, sub_map_ref, unk_id, d_tokens.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  return d_tokens;
}

}  // namespace

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 cudf::size_type max_words_per_row,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    max_words_per_row >= 0, "Invalid value for max_words_per_row argument", std::invalid_argument);

  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.size() == input.null_count()) {
    return input.has_nulls() ? cudf::lists::detail::make_all_nulls_lists_column(
                                 input.size(), output_type, stream, mr)
                             : cudf::lists::detail::make_empty_lists_column(output_type);
  }

  auto [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  auto const chars_size = last_offset - first_offset;

  auto d_tokens =
    max_words_per_row == 0
      ? compute_all_tokens(input, first_offset, chars_size, *(vocabulary._impl), stream)
      : compute_some_tokens(
          input, first_offset, chars_size, max_words_per_row, *(vocabulary._impl), stream);

  // compute token counts by doing a segmented reduce over valid d_tokens
  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const d_token_counts =
    count_tokens(d_tokens.data(), input_offsets, first_offset, input.size(), stream);

  auto [token_offsets, total_count] = cudf::detail::make_offsets_child_column(
    d_token_counts.begin(), d_token_counts.end(), stream, mr);

  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto output = tokens->mutable_view().begin<cudf::size_type>();
  thrust::remove_copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      d_tokens.begin(),
                      d_tokens.end(),
                      output,
                      no_token);

  return cudf::make_lists_column(input.size(),
                                 std::move(token_offsets),
                                 std::move(tokens),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr));
}
}  // namespace detail

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 cudf::size_type max_words_per_row,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::wordpiece_tokenize(input, vocabulary, max_words_per_row, stream, mr);
}

}  // namespace nvtext
