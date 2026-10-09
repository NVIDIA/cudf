/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/wordpiece_tokenize.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuco/static_map.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
#include <thrust/scan.h>

namespace nvtext {
namespace detail {
namespace {

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
using hash_value_type    = string_hasher_type::result_type;

/**
 * @brief Key type for the vocabulary maps
 *
 * The hash of the vocabulary entry is stored in the upper bits and
 * the row index of the entry is stored in the lower bits.
 * Storing the hash allows most non-matching entries to be rejected
 * without comparing the strings.
 */
using map_key_type = int64_t;

/// Number of bits the hash is shifted to place it in the upper bits of the key
constexpr int key_hash_shift = cuda::std::numeric_limits<hash_value_type>::digits;
/// Mask for the row index in the lower bits of the key
constexpr uint64_t key_row_mask = (uint64_t{1} << key_hash_shift) - 1;

__device__ map_key_type make_map_key(hash_value_type hash, cudf::size_type row)
{
  return static_cast<map_key_type>((static_cast<uint64_t>(hash) << key_hash_shift) |
                                   static_cast<uint32_t>(row));
}
__device__ hash_value_type key_hash(map_key_type key)
{
  return static_cast<hash_value_type>(static_cast<uint64_t>(key) >> key_hash_shift);
}
__device__ cudf::size_type key_row(map_key_type key)
{
  return static_cast<cudf::size_type>(static_cast<uint64_t>(key) & key_row_mask);
}

/**
 * @brief String used to search the vocabulary maps along with its hash
 */
struct vocab_probe {
  cudf::string_view str;
  hash_value_type hash;
};

__device__ vocab_probe make_probe(cudf::string_view str)
{
  return {str, string_hasher_type{}(str)};
}

/**
 * @brief Hasher used for the vocabulary maps
 *
 * The hash values are computed when the keys and probes are created.
 */
struct vocab_hasher {
  __device__ hash_value_type operator()(map_key_type key) const { return key_hash(key); }
  __device__ hash_value_type operator()(vocab_probe const& probe) const { return probe.hash; }
};

/**
 * @brief Equality operator for the vocabulary maps
 *
 * The vocabulary entries are compared after skipping `prefix_size` bytes.
 * This allows the subword map to skip the '##' prefix of its entries.
 */
struct vocab_equal {
  cudf::column_device_view const d_strings;
  cudf::size_type prefix_size;
  __device__ bool operator()(map_key_type lhs, map_key_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are expected to be unique
  }
  __device__ bool operator()(vocab_probe const& lhs, map_key_type rhs) const noexcept
  {
    if (key_hash(rhs) != lhs.hash) { return false; }
    auto const d_str = d_strings.element<cudf::string_view>(key_row(rhs));
    return lhs.str ==
           cudf::string_view(d_str.data() + prefix_size, d_str.size_bytes() - prefix_size);
  }
};

/**
 * @brief Capacity of the vocabulary maps relative to the number of entries
 *
 * A low load factor reduces the number of slots checked for each lookup.
 * Most lookups by the tokenizer are for prefixes not in the vocabulary
 * and these must check slots until an empty one is found.
 */
constexpr std::size_t map_capacity_factor = 4;

using cuco_storage        = cuco::storage<1>;
using probe_scheme        = cuco::linear_probing<1, vocab_hasher>;
using vocabulary_map_type = cuco::static_map<map_key_type,
                                             cudf::size_type,
                                             cuco::extent<std::size_t>,
                                             cuda::thread_scope_thread,
                                             vocab_equal,
                                             probe_scheme,
                                             rmm::mr::polymorphic_allocator<char>,
                                             cuco_storage>;
// This 2nd subword map holds the '##' entries without the prefix
// which helps avoid requiring temporary strings in device code
using sub_vocabulary_map_type = vocabulary_map_type;
}  // namespace
}  // namespace detail

// since column_device_view::create returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type in a maintainable way
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             cuda::stream_ref,
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
 * @brief Creates the key and value for each vocabulary map entry
 *
 * The key holds the hash and the row index of the entry.
 * The value is the row index which is also the token id.
 */
struct key_pair {
  cudf::column_device_view const d_strings;
  cudf::size_type prefix_size;  // number of bytes to skip when hashing the entry
  __device__ cuco::pair<detail::map_key_type, cudf::size_type> operator()(
    cudf::size_type idx) const noexcept
  {
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    auto const hash  = detail::string_hasher_type{}(
      cudf::string_view(d_str.data() + prefix_size, d_str.size_bytes() - prefix_size));
    return cuco::make_pair(detail::make_map_key(hash, idx), idx);
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
    auto const fnd = d_map.find(detail::make_probe(unk));
    return fnd != d_map.end() ? fnd->second : -1;
  }
};

}  // namespace

wordpiece_vocabulary::wordpiece_vocabulary(cudf::strings_column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not input.is_empty(), "vocabulary must not be empty", std::invalid_argument);
  CUDF_EXPECTS(not input.has_nulls(), "vocabulary must not have nulls", std::invalid_argument);

  // hold a copy of the input (not expected to be very large)
  auto vocabulary   = std::make_unique<cudf::column>(input.parent(), stream, mr);
  auto d_vocabulary = cudf::column_device_view::create(vocabulary->view(), stream);

  // build the vocabulary map: each row is a single term and is the key for the map
  auto vocab_map = std::make_unique<detail::vocabulary_map_type>(
    static_cast<std::size_t>(vocabulary->size()) * detail::map_capacity_factor,
    cuco::empty_key{detail::map_key_type{-1}},
    cuco::empty_value{-1},
    detail::vocab_equal{*d_vocabulary, 0},
    detail::probe_scheme{detail::vocab_hasher{}},
    cuco::thread_scope_thread,
    detail::cuco_storage{},
    rmm::mr::polymorphic_allocator<char>{mr},
    stream.get());
  // the row index is the token id (data value for each key in the map)
  auto iter = cudf::detail::make_counting_transform_iterator(0, key_pair{*d_vocabulary, 0});
  vocab_map->insert_async(iter, iter + vocabulary->size(), stream.get());
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
    sub_map_indices.size() * detail::map_capacity_factor,
    cuco::empty_key{detail::map_key_type{-1}},
    cuco::empty_value{-1},
    detail::vocab_equal{*d_vocabulary, 2},
    detail::probe_scheme{detail::vocab_hasher{}},
    cuco::thread_scope_thread,
    detail::cuco_storage{},
    rmm::mr::polymorphic_allocator<char>{mr},
    stream.get());
  // insert them without the '##' prefix since that is how they will be looked up
  auto iter_sub = cuda::transform_iterator(sub_map_indices.begin(), key_pair{*d_vocabulary, 2});
  vocab_sub_map->insert_async(iter_sub, iter_sub + sub_map_indices.size(), stream.get());

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
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<wordpiece_vocabulary>(input, stream, mr);
}

namespace detail {
namespace {

constexpr auto block_size    = 128;
constexpr auto max_word_size = 200;  // words longer than this are not tokenized

/**
 * @brief Finds the longest prefix of `str` that is in `d_map`
 *
 * The tile checks `tile.size()` prefix sizes at a time starting with `max_size`
 * and stops at the first set of sizes where a match is found.
 * Only prefixes ending on a character boundary are checked.
 *
 * @param tile Threads cooperating on this search
 * @param str String to search
 * @param max_size Largest prefix size to check
 * @param d_map Map to search
 * @param token Token of the matched prefix returned to all threads in the tile
 * @return Size of the longest matching prefix or 0 if there is no match
 */
template <typename Tile, typename MapRefType>
__device__ cudf::size_type find_longest_prefix(Tile const& tile,
                                               cudf::string_view str,
                                               cudf::size_type max_size,
                                               MapRefType const& d_map,
                                               cudf::size_type& token)
{
  auto const lane      = static_cast<cudf::size_type>(tile.thread_rank());
  auto const tile_size = static_cast<cudf::size_type>(tile.size());
  for (auto base = max_size; base > 0; base -= tile_size) {
    auto const size  = base - lane;
    auto found_token = cudf::size_type{0};
    auto found       = false;
    if ((size > 0) && ((size == str.size_bytes()) ||
                       !cudf::strings::detail::is_utf8_continuation_char(str.data()[size]))) {
      auto const itr = d_map.find(make_probe(cudf::string_view(str.data(), size)));
      if (itr != d_map.end()) {
        found       = true;
        found_token = itr->second;
      }
    }
    auto const mask = tile.ballot(found);
    if (mask != 0) {
      auto const src = __ffs(mask) - 1;  // lowest lane has the longest prefix
      token          = tile.shfl(found_token, src);
      return base - src;
    }
  }
  return 0;
}

/**
 * @brief The wordpiece tokenizer
 *
 * The longest prefix of the word found in d_map is the first token.
 * If no prefix is found, the unk_id is the only token.
 * The longest prefix of the remaining characters found in d_sub_map is the next token
 * and this repeats until all characters have been resolved. If any of the remaining
 * characters cannot be resolved, the unk_id is the only token.
 *
 * Example: word="GPU" and d_map contains { ... {"G",10}, {"##U",7}, {"##P",3}, ... }
 * which means the d_sub_map contains { ... {"U",7}, {"P",3}, ... }
 * The longest prefix of "GPU" found in d_map is "G".
 * The longest prefix of the remaining "PU" found in d_sub_map is "P"
 * and the remaining "U" is also found in d_sub_map.
 * The end result is that "GPU" produces 3 tokens [10,3,7].
 *
 * All threads in the tile must call this function with the same word.
 * Words with `max_word_size` or more bytes resolve to the `unk_id`.
 *
 * @param tile Threads cooperating to tokenize the word
 * @param word Word to tokenize
 * @param d_map Vocabulary to check for word and sub-words
 * @param d_sub_map Partial vocabulary of '##' entries
 * @param unk_id The unknown token id returned when no token is found
 * @param output Called by all threads with each resolved token id in order;
 *               `output.reset(unk_id)` replaces all previous tokens with just `unk_id`
 */
template <typename Tile, typename MapRefType, typename SubMapRefType, typename OutputFn>
__device__ void wp_tokenize_fn(Tile const& tile,
                               cudf::string_view word,
                               MapRefType const& d_map,
                               SubMapRefType const& d_sub_map,
                               cudf::size_type unk_id,
                               OutputFn& output)
{
  if (word.empty()) { return; }
  if (word.size_bytes() >= max_word_size) {
    output(unk_id);
    return;
  }

  cudf::size_type token = 0;
  auto size             = find_longest_prefix(tile, word, word.size_bytes(), d_map, token);
  if (size == 0) {
    output(unk_id);
    return;
  }
  output(token);

  auto rest = cudf::string_view(word.data() + size, word.size_bytes() - size);
  while (!rest.empty()) {
    size = find_longest_prefix(tile, rest, rest.size_bytes(), d_sub_map, token);
    if (size == 0) {
      output.reset(unk_id);
      return;
    }
    output(token);
    rest = cudf::string_view(rest.data() + size, rest.size_bytes() - size);
  }
}

/**
 * @brief Output functor for wp_tokenize_fn holding the tokens across the tile
 *
 * Token `k` is held by the thread with `lane==k` and only the first
 * tile-size tokens are held. All tokens are counted.
 */
struct tile_tokens_fn {
  cudf::size_type lane;
  cudf::size_type token = 0;
  cudf::size_type count = 0;
  __device__ void operator()(cudf::size_type value)
  {
    if (count == lane) { token = value; }
    ++count;
  }
  __device__ void reset(cudf::size_type value)
  {
    if (lane == 0) { token = value; }
    count = 1;
  }
};

/**
 * @brief Output functor for wp_tokenize_fn writing the tokens to device memory
 */
struct device_tokens_fn {
  cudf::size_type* d_tokens;
  cudf::size_type lane;
  cudf::size_type count = 0;
  __device__ void operator()(cudf::size_type value)
  {
    if (lane == 0) { d_tokens[count] = value; }
    ++count;
  }
  __device__ void reset(cudf::size_type value)
  {
    if (lane == 0) { d_tokens[0] = value; }
    count = 1;
  }
};

/// Number of bits in each word of the row-start bitmask
constexpr int64_t row_starts_word_bits = cudf::detail::size_in_bits<cudf::bitmask_type>();

/**
 * @brief Returns true if `pos` is the first byte of a row
 *
 * The cudf bit utilities are not used since `pos` may exceed the range of `size_type`.
 */
__device__ bool is_row_start(cudf::bitmask_type const* d_row_starts, int64_t pos)
{
  return (d_row_starts[pos / row_starts_word_bits] >> (pos % row_starts_word_bits)) & 1u;
}

/**
 * @brief Returns each word found in the input
 *
 * A word ends at a space character or at the start of the next row.
 * Only the first `max_word_size` bytes are needed to tokenize a word.
 */
struct all_words_fn {
  char const* d_chars;
  int64_t chars_size;
  int64_t const* d_starts;
  cudf::bitmask_type const* d_row_starts;
  __device__ cudf::string_view operator()(cudf::size_type idx) const
  {
    auto const start = d_starts[idx];
    auto last        = cuda::std::min(chars_size, start + max_word_size);
    // the word cannot extend past the start of the next row
    for (auto pos = start + 1; pos < last;
         pos      = cudf::util::round_up_unsafe(pos + 1, row_starts_word_bits)) {
      auto const bits = d_row_starts[pos / row_starts_word_bits] >> (pos % row_starts_word_bits);
      if (bits != 0) {
        last = cuda::std::min(last, pos + __ffs(bits) - 1);
        break;
      }
    }
    auto end = start + 1;
    while ((end < last) && (d_chars[end] != ' ')) {
      ++end;
    }
    return cudf::string_view(d_chars + start, static_cast<cudf::size_type>(end - start));
  }

  /**
   * @brief Returns the word at `idx` whose size is already known
   */
  __device__ cudf::string_view word(cudf::size_type idx, cudf::size_type size) const
  {
    return cudf::string_view(d_chars + d_starts[idx], size);
  }
};

constexpr int32_t words_block_size = 256;

/// Mask for all the threads in a warp used with the warp-level intrinsics
constexpr uint32_t all_lanes = 0xFFFF'FFFFu;

/// Number of threads cooperating to tokenize each deferred word
constexpr int32_t deferred_tile_size = 8;

/**
 * @brief Tokenizes words that are found in the vocabulary
 *
 * Launched as a thread per word.
 *
 * Most words are found directly in the vocabulary and produce a single token.
 * Words that are not found are deferred to tokenize_deferred_kernel,
 * their count is set to 0, and their size (in bytes) is stored in `d_values`.
 *
 * @param words Returns the word for each index
 * @param num_words Number of words to tokenize
 * @param d_map Vocabulary of all words and sub-words
 * @param unk_id Unknown token id
 * @param d_counts Number of tokens for each word; 0 for deferred words
 * @param d_values Token for each word or the size of each deferred word
 */
template <typename WordsFn, typename MapRefType>
CUDF_KERNEL void tokenize_words_kernel(WordsFn words,
                                       cudf::size_type num_words,
                                       MapRefType const d_map,
                                       cudf::size_type unk_id,
                                       uint8_t* d_counts,
                                       cudf::size_type* d_values)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= num_words) { return; }

  auto const word = words(static_cast<cudf::size_type>(idx));
  uint8_t count   = 0;
  auto value      = cudf::size_type{0};
  if (word.size_bytes() >= max_word_size) {
    count = 1;
    value = unk_id;
  } else if (!word.empty()) {
    auto const itr = d_map.find(make_probe(word));
    if (itr != d_map.end()) {
      count = 1;
      value = itr->second;
    } else {
      value = word.size_bytes();  // deferred words keep their size for tokenize_deferred_kernel
    }
  }
  d_counts[idx] = count;
  d_values[idx] = value;
}

/**
 * @brief Tokenizes the words not found by tokenize_words_kernel
 *
 * Launched as a tile of `tile_size` threads per deferred word.
 *
 * The number of tokens for each word is stored in `d_counts`.
 * For words with a single token, the token is stored in `d_values`.
 * Otherwise, the tokens are stored in `d_overflow` and `d_values`
 * holds the position of the word's tokens in `d_overflow`.
 *
 * The `d_overflow_size` is the total number of overflow tokens reserved.
 * If this is larger than `d_overflow.size()` then some words did not fit and
 * their count remains 0 so they can be tokenized again with a larger `d_overflow`.
 *
 * @param words Returns the word for each index
 * @param d_deferred Indices of the words to tokenize
 * @param d_map Vocabulary of all words and sub-words
 * @param d_sub_map Partial vocabulary of '##' entries
 * @param unk_id Unknown token id
 * @param d_counts Number of tokens for each word
 * @param d_values Size of each deferred word; set to the token or overflow position for each word
 * @param d_overflow Tokens for words with more than one token
 * @param d_overflow_size Number of tokens needed for `d_overflow`
 */
template <int32_t tile_size, typename WordsFn, typename MapRefType, typename SubMapRefType>
CUDF_KERNEL void tokenize_deferred_kernel(WordsFn words,
                                          cudf::device_span<cudf::size_type const> d_deferred,
                                          MapRefType const d_map,
                                          SubMapRefType const d_sub_map,
                                          cudf::size_type unk_id,
                                          uint8_t* d_counts,
                                          cudf::size_type* d_values,
                                          cudf::device_span<cudf::size_type> d_overflow,
                                          int64_t* d_overflow_size)
{
  namespace cg      = cooperative_groups;
  auto const block  = cg::this_thread_block();
  auto const tile   = cg::tiled_partition<tile_size>(block);
  auto const lane   = static_cast<cudf::size_type>(tile.thread_rank());
  auto const tid    = cudf::detail::grid_1d::global_thread_id() / tile_size;
  auto const active = tid < static_cast<cudf::thread_index_type>(d_deferred.size());

  auto const word_idx = active ? d_deferred[tid] : 0;
  auto const word     = active ? words.word(word_idx, d_values[word_idx]) : cudf::string_view{};

  tile_tokens_fn tokens{lane};
  wp_tokenize_fn(tile, word, d_map, d_sub_map, unk_id, tokens);
  auto const count = tokens.count;

  // Reserve space in d_overflow for the words with more than one token.
  // This only synchronizes within the warp since the tokenize time varies by word.
  constexpr auto warp_size = cudf::detail::warp_size;
  auto const warp_lane     = threadIdx.x % warp_size;
  auto const needed        = static_cast<int64_t>((lane == 0 && count > 1) ? count : 0);
  auto offset              = needed;  // inclusive scan within the warp
  for (uint32_t delta = 1; delta < warp_size; delta *= 2) {
    auto const value = __shfl_up_sync(all_lanes, offset, delta);
    if (warp_lane >= delta) { offset += value; }
  }
  auto const warp_needed = __shfl_sync(all_lanes, offset, warp_size - 1);
  int64_t warp_overflow  = 0;
  if ((warp_lane == warp_size - 1) && (warp_needed > 0)) {
    warp_overflow =
      cuda::atomic_ref<int64_t, cuda::thread_scope_device>{*d_overflow_size}.fetch_add(
        warp_needed, cuda::memory_order_relaxed);
  }
  warp_overflow = __shfl_sync(all_lanes, warp_overflow, warp_size - 1);
  offset        = tile.shfl(warp_overflow + offset - needed, 0);

  if (!active) { return; }
  // words whose tokens do not fit in d_overflow are left as deferred to be tokenized again
  auto const stored = (count <= 1) || (offset + count <= static_cast<int64_t>(d_overflow.size()));
  if ((count > 1) && stored) {
    auto const d_output = d_overflow.data() + offset;
    if (count <= tile_size) {
      if (lane < count) { d_output[lane] = tokens.token; }
    } else {
      // too many tokens to hold in the tile so tokenize again writing directly to the output
      device_tokens_fn output{d_output, lane};
      wp_tokenize_fn(tile, word, d_map, d_sub_map, unk_id, output);
    }
  }
  if ((lane == 0) && stored) {
    d_counts[word_idx] = static_cast<uint8_t>(count);
    d_values[word_idx] = count == 1 ? tokens.token : static_cast<cudf::size_type>(offset);
  }
}

/**
 * @brief Computes the number of tokens for each block of words
 *
 * Launched as a thread per word using the same blocks as write_tokens_kernel.
 *
 * @param num_words Number of words
 * @param d_counts Number of tokens for each word
 * @param d_block_counts Number of tokens for each block of words
 */
CUDF_KERNEL void block_counts_kernel(cudf::size_type num_words,
                                     uint8_t const* d_counts,
                                     int64_t* d_block_counts)
{
  auto const idx   = cudf::detail::grid_1d::global_thread_id();
  auto const count = idx < num_words ? static_cast<int64_t>(d_counts[idx]) : int64_t{0};

  using block_reduce = cub::BlockReduce<int64_t, words_block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;
  auto const block_count = block_reduce(reduce_storage).Sum(count);
  if (threadIdx.x == 0) { d_block_counts[blockIdx.x] = block_count; }
}

/**
 * @brief Writes the tokens for each word to the output
 *
 * Launched as a thread per word using the same blocks as block_counts_kernel.
 *
 * @param num_words Number of words
 * @param d_counts Number of tokens for each word
 * @param d_values Token or overflow position for each word
 * @param d_overflow Tokens for words with more than one token
 * @param d_block_offsets Output offset for each block of words
 * @param d_output Output tokens
 */
CUDF_KERNEL void write_tokens_kernel(cudf::size_type num_words,
                                     uint8_t const* d_counts,
                                     cudf::size_type const* d_values,
                                     cudf::size_type const* d_overflow,
                                     int64_t const* d_block_offsets,
                                     cudf::size_type* d_output)
{
  auto const idx   = cudf::detail::grid_1d::global_thread_id();
  auto const count = idx < num_words ? static_cast<int32_t>(d_counts[idx]) : 0;

  using block_scan = cub::BlockScan<int32_t, words_block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;
  int32_t offset = 0;
  block_scan(scan_storage).ExclusiveSum(count, offset);

  if (idx >= num_words) { return; }
  auto d_tokens    = d_output + d_block_offsets[blockIdx.x] + offset;
  auto const value = d_values[idx];
  if (count == 1) {
    *d_tokens = value;
    return;
  }
  for (int32_t i = 0; i < count; ++i) {
    d_tokens[i] = d_overflow[value + i];
  }
}

/**
 * @brief The tokenizer results for each word
 */
struct tokenized_words {
  rmm::device_uvector<uint8_t> counts;             ///< Number of tokens for each word
  rmm::device_uvector<cudf::size_type> values;     ///< Token or overflow position for each word
  rmm::device_uvector<cudf::size_type> overflow;   ///< Tokens for words with multiple tokens
  rmm::device_uvector<cudf::size_type> row_words;  ///< Word index boundaries for each row
};

/**
 * @brief Tokenizes all the given words
 *
 * @param words Returns the word for each index
 * @param d_starts Position of each word in the input characters; must be sorted
 * @param num_words Number of words
 * @param input Input strings column
 * @param first_offset Offset to first row in chars for `input`
 * @param vocabulary Vocabulary data needed by the tokenizer
 * @param stream Stream used for device allocations and kernel launches
 * @return The tokenizer results
 */
template <typename WordsFn>
tokenized_words tokenize_words(WordsFn words,
                               int64_t const* d_starts,
                               cudf::size_type num_words,
                               cudf::strings_column_view const& input,
                               int64_t first_offset,
                               wordpiece_vocabulary::wordpiece_vocabulary_impl const& vocabulary,
                               cuda::stream_ref stream)
{
  auto const map_ref     = vocabulary.get_map_ref();
  auto const sub_map_ref = vocabulary.get_sub_map_ref();
  auto const unk_id      = vocabulary.unk_id;

  auto result = tokenized_words{rmm::device_uvector<uint8_t>(num_words, stream),
                                rmm::device_uvector<cudf::size_type>(num_words, stream),
                                rmm::device_uvector<cudf::size_type>(0, stream),
                                rmm::device_uvector<cudf::size_type>(input.size() + 1, stream)};

  if (num_words > 0) {
    // tokenize the words found directly in the vocabulary
    cudf::detail::grid_1d grid{num_words, words_block_size};
    tokenize_words_kernel<WordsFn, decltype(map_ref)>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        words, num_words, map_ref, unk_id, result.counts.data(), result.values.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    // collect the indices of the words that were not found
    auto const d_counts    = result.counts.data();
    auto const is_deferred = [d_counts] __device__(cudf::size_type idx) -> bool {
      return d_counts[idx] == 0;
    };
    auto const begin = cuda::counting_iterator<cudf::size_type>{0};
    auto const end   = cuda::counting_iterator<cudf::size_type>{num_words};
    auto const num_deferred =
      static_cast<cudf::size_type>(cudf::detail::count_if(begin, end, is_deferred, stream));
    auto d_deferred = rmm::device_uvector<cudf::size_type>(num_deferred, stream);
    cudf::detail::copy_if(begin, end, d_deferred.begin(), is_deferred, stream);

    // the initial overflow size is a guess and is increased if needed
    auto overflow_capacity = static_cast<int64_t>(num_deferred) * 3;
    result.overflow        = rmm::device_uvector<cudf::size_type>(overflow_capacity, stream);
    auto d_overflow_size   = cudf::detail::device_scalar<int64_t>(0, stream);
    while (!d_deferred.is_empty()) {
      cudf::detail::grid_1d grid_deferred{
        static_cast<cudf::thread_index_type>(d_deferred.size()) * deferred_tile_size,
        words_block_size};
      tokenize_deferred_kernel<deferred_tile_size,
                               WordsFn,
                               decltype(map_ref),
                               decltype(sub_map_ref)>
        <<<grid_deferred.num_blocks, grid_deferred.num_threads_per_block, 0, stream.get()>>>(
          words,
          d_deferred,
          map_ref,
          sub_map_ref,
          unk_id,
          result.counts.data(),
          result.values.data(),
          cudf::device_span<cudf::size_type>(result.overflow),
          d_overflow_size.data());
      CUDF_CUDA_TRY(cudaGetLastError());
      auto const overflow_size = d_overflow_size.value(stream);
      if (overflow_size <= overflow_capacity) { break; }

      // Some words did not fit so they are tokenized again.
      // Their reserved space ends at overflow_size so they fit if the
      // overflow is increased by overflow_size and reserved from the old capacity.
      auto const remaining_begin = d_deferred.begin();
      auto const remaining_end   = d_deferred.end();
      auto const num_remaining   = static_cast<cudf::size_type>(
        cudf::detail::count_if(remaining_begin, remaining_end, is_deferred, stream));
      auto d_remaining = rmm::device_uvector<cudf::size_type>(num_remaining, stream);
      cudf::detail::copy_if(
        remaining_begin, remaining_end, d_remaining.begin(), is_deferred, stream);
      d_deferred = std::move(d_remaining);

      CUDF_EXPECTS(overflow_capacity + overflow_size < std::numeric_limits<cudf::size_type>::max(),
                   "number of tokens exceeds the column size limit",
                   std::overflow_error);
      d_overflow_size.set_value_async(overflow_capacity, stream);
      overflow_capacity += overflow_size;
      result.overflow.resize(overflow_capacity, stream);
    }
  }

  // identify the range of words for each row
  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const d_offsets = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<int64_t>([input_offsets, first_offset] __device__(auto idx) {
      return input_offsets[idx] - first_offset;
    }));
  thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      d_starts,
                      d_starts + num_words,
                      d_offsets,
                      d_offsets + input.size() + 1,
                      result.row_words.begin());

  return result;
}

/**
 * @brief Creates the output lists column from the tokenizer results
 *
 * @param tokens The tokenizer results
 * @param input Input strings column
 * @param stream Stream used for device allocations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column of tokens for each row
 */
std::unique_ptr<cudf::column> make_tokens_column(tokenized_words&& tokens,
                                                 cudf::strings_column_view const& input,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const num_words = static_cast<cudf::size_type>(tokens.counts.size());

  // compute the token counts for each row by doing a segmented reduce over the word counts
  auto d_token_counts = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  {
    // the uint8 counts are accumulated using the output type
    auto const mr_prop     = cuda::std::execution::prop{cuda::mr::get_memory_resource,
                                                    cudf::get_current_device_resource_ref()};
    auto const env         = cuda::std::execution::env{stream, mr_prop};
    auto const d_row_words = tokens.row_words.data();
    CUDF_CUDA_TRY(cub::DeviceSegmentedReduce::Sum(tokens.counts.data(),
                                                  d_token_counts.data(),
                                                  input.size(),
                                                  d_row_words,
                                                  d_row_words + 1,
                                                  env));
  }

  auto [token_offsets, total_count] = cudf::detail::make_offsets_child_column(
    d_token_counts.begin(), d_token_counts.end(), stream, mr);

  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  auto output =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);

  if (num_words > 0) {
    // compute the output offset for each block of words
    cudf::detail::grid_1d grid{num_words, words_block_size};
    auto d_block_offsets = rmm::device_uvector<int64_t>(grid.num_blocks, stream);
    block_counts_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      num_words, tokens.counts.data(), d_block_offsets.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           d_block_offsets.begin(),
                           d_block_offsets.end(),
                           d_block_offsets.begin());
    write_tokens_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      num_words,
      tokens.counts.data(),
      tokens.values.data(),
      tokens.overflow.data(),
      d_block_offsets.data(),
      output->mutable_view().data<cudf::size_type>());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return cudf::make_lists_column(input.size(),
                                 std::move(token_offsets),
                                 std::move(output),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

/**
 * @brief Identifies the first byte of each row with a bit
 *
 * @param input Input strings column
 * @param first_offset Offset to first row in chars for `input`
 * @param chars_size Size of the character data for `input`
 * @param stream Stream used for device allocations and kernel launches
 * @return Bits set for the first byte of each row
 */
rmm::device_uvector<cudf::bitmask_type> make_row_starts(cudf::strings_column_view const& input,
                                                        int64_t first_offset,
                                                        int64_t chars_size,
                                                        cuda::stream_ref stream)
{
  auto d_row_starts = rmm::device_uvector<cudf::bitmask_type>(
    cudf::util::div_rounding_up_safe(chars_size, row_starts_word_bits), stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    d_row_starts.data(), 0, d_row_starts.size() * sizeof(cudf::bitmask_type), stream.get()));
  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    input.size(),
    [input_offsets, first_offset, chars_size, d_row_starts = d_row_starts.data()] __device__(
      cudf::size_type idx) {
      auto const pos = input_offsets[idx] - first_offset;
      if (pos >= chars_size) { return; }
      cuda::atomic_ref<cudf::bitmask_type, cuda::thread_scope_device>{
        d_row_starts[pos / row_starts_word_bits]}
        .fetch_or(cudf::bitmask_type{1} << (pos % row_starts_word_bits),
                  cuda::memory_order_relaxed);
    });
  return d_row_starts;
}

/**
 * @brief Compute all tokens for the input column
 *
 * @param input Input strings column
 * @param first_offset Offset to first row in chars for `input`
 * @param chars_size Size of the character data for `input`
 * @param vocabulary Vocabulary data needed by the tokenizer
 * @param stream Stream used for device allocations and kernel launches
 * @return The tokenizer results
 */
tokenized_words compute_all_tokens(
  cudf::strings_column_view const& input,
  int64_t first_offset,
  int64_t chars_size,
  wordpiece_vocabulary::wordpiece_vocabulary_impl const& vocabulary,
  cuda::stream_ref stream)
{
  auto const d_input_chars = input.chars_begin(stream) + first_offset;
  auto const d_row_starts  = make_row_starts(input, first_offset, chars_size, stream);

  // beginning of a word is a non-space preceded by a space or at the beginning of a row
  auto const is_word_start = [d_input_chars,
                              d_row_starts = d_row_starts.data()] __device__(int64_t idx) -> bool {
    return (d_input_chars[idx] != ' ') &&
           (is_row_start(d_row_starts, idx) || (d_input_chars[idx - 1] == ' '));
  };
  auto const begin     = cuda::counting_iterator<int64_t>{0};
  auto const end       = cuda::counting_iterator<int64_t>{chars_size};
  auto const num_words = cudf::detail::count_if(begin, end, is_word_start, stream);
  CUDF_EXPECTS(num_words < static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "words exceed internal limit",
               std::overflow_error);

  auto d_starts = rmm::device_uvector<int64_t>(num_words, stream);
  cudf::detail::copy_if(begin, end, d_starts.begin(), is_word_start, stream);

  return tokenize_words(
    all_words_fn{d_input_chars, chars_size, d_starts.data(), d_row_starts.data()},
    d_starts.data(),
    static_cast<cudf::size_type>(num_words),
    input,
    first_offset,
    vocabulary,
    stream);
}

/**
 * @brief Locates the first `max_words` words in each row
 *
 * Launched as a warp per row.
 *
 * A word begins with a non-space character that is either preceded by a space
 * or is the first character of the row.
 *
 * If `d_starts==nullptr` the number of words found (up to `max_words`) for each
 * row is stored in `d_counts`. Otherwise, the position of each word is stored
 * in `d_starts` beginning at `d_offsets[row]`.
 *
 * @param d_strings Input strings column
 * @param d_chars Beginning of the character data for d_strings adjusted for any sliced offset
 * @param max_words Maximum number of words to locate in each row
 * @param d_counts Number of words found in each row
 * @param d_offsets Output offset for the words of each row
 * @param d_starts Position of each word within `d_chars`
 */
CUDF_KERNEL void find_row_words_kernel(cudf::column_device_view const d_strings,
                                       char const* d_chars,
                                       cudf::size_type max_words,
                                       int64_t* d_counts,
                                       int64_t const* d_offsets,
                                       int64_t* d_starts)
{
  auto const idx  = cudf::detail::grid_1d::global_thread_id();
  auto const row  = idx / cudf::detail::warp_size;
  auto const lane = static_cast<uint32_t>(idx % cudf::detail::warp_size);
  if (row >= d_strings.size()) { return; }

  cudf::size_type count = 0;
  if (d_strings.is_valid(row)) {
    auto const d_str  = d_strings.element<cudf::string_view>(row);
    auto const begin  = d_str.data();
    auto const size   = d_str.size_bytes();
    auto const offset = static_cast<int64_t>(cuda::std::distance(d_chars, begin));
    for (cudf::size_type pos = 0; (pos < size) && (count < max_words);
         pos += cudf::detail::warp_size) {
      auto const p        = pos + static_cast<cudf::size_type>(lane);
      auto const is_start = (p < size) && (begin[p] != ' ') && ((p == 0) || (begin[p - 1] == ' '));
      auto const mask     = __ballot_sync(all_lanes, is_start);
      auto const word_idx = count + __popc(mask & ((1u << lane) - 1u));
      if (is_start && (word_idx < max_words) && (d_starts != nullptr)) {
        d_starts[d_offsets[row] + word_idx] = offset + p;
      }
      count += __popc(mask);
    }
  }
  if ((lane == 0) && (d_starts == nullptr)) { d_counts[row] = cuda::std::min(count, max_words); }
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
 * @return The tokenizer results
 */
tokenized_words compute_some_tokens(
  cudf::strings_column_view const& input,
  int64_t first_offset,
  int64_t chars_size,
  cudf::size_type max_words_per_row,
  wordpiece_vocabulary::wordpiece_vocabulary_impl const& vocabulary,
  cuda::stream_ref stream)
{
  auto const d_input_chars = input.chars_begin(stream) + first_offset;
  auto const d_row_starts  = make_row_starts(input, first_offset, chars_size, stream);
  auto const d_strings     = cudf::column_device_view::create(input.parent(), stream);

  // count the words in each row (up to max_words_per_row) and convert to offsets
  auto d_offsets = rmm::device_uvector<int64_t>(input.size() + 1, stream);
  cudf::detail::grid_1d grid{
    static_cast<cudf::thread_index_type>(input.size()) * cudf::detail::warp_size, block_size};
  find_row_words_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    *d_strings, d_input_chars, max_words_per_row, d_offsets.data(), nullptr, nullptr);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const num_words = cudf::detail::sizes_to_offsets(d_offsets.begin(),
                                                        d_offsets.end(),
                                                        d_offsets.begin(),
                                                        0,
                                                        stream,
                                                        cudf::get_current_device_resource_ref());
  CUDF_EXPECTS(num_words < static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "words exceed internal limit",
               std::overflow_error);

  // store the position of each word
  auto d_starts = rmm::device_uvector<int64_t>(num_words, stream);
  find_row_words_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    *d_strings, d_input_chars, max_words_per_row, nullptr, d_offsets.data(), d_starts.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  return tokenize_words(
    all_words_fn{d_input_chars, chars_size, d_starts.data(), d_row_starts.data()},
    d_starts.data(),
    static_cast<cudf::size_type>(num_words),
    input,
    first_offset,
    vocabulary,
    stream);
}

}  // namespace

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 cudf::size_type max_words_per_row,
                                                 cuda::stream_ref stream,
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

  auto tokens =
    max_words_per_row == 0
      ? compute_all_tokens(input, first_offset, chars_size, *(vocabulary._impl), stream)
      : compute_some_tokens(
          input, first_offset, chars_size, max_words_per_row, *(vocabulary._impl), stream);

  return make_tokens_column(std::move(tokens), input, stream, mr);
}
}  // namespace detail

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 cudf::size_type max_words_per_row,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::wordpiece_tokenize(input, vocabulary, max_words_per_row, stream, mr);
}

}  // namespace nvtext
