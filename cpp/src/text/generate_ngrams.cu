/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/generate_ngrams.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/copy.h>

#include <stdexcept>

namespace nvtext {
namespace detail {
namespace {
// long strings threshold found with benchmarking
constexpr cudf::size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Generate ngrams from strings column.
 *
 * Adjacent strings are concatenated with the provided separator.
 * The number of adjacent strings join depends on the specified ngrams value.
 * For example: for bigrams (ngrams=2), pairs of strings are concatenated.
 */
struct ngram_generator_fn {
  cudf::column_device_view const d_strings;
  cudf::size_type ngrams;
  cudf::string_view const d_separator;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Build ngram for each string.
   *
   * This is called for each thread and processed for each string.
   * Each string will produce the number of ngrams specified.
   *
   * @param idx Index of the kernel thread.
   * @return Number of bytes required for the string for this thread.
   */
  __device__ void operator()(cudf::size_type idx)
  {
    char* out_ptr         = d_chars ? d_chars + d_offsets[idx] : nullptr;
    cudf::size_type bytes = 0;
    for (cudf::size_type n = 0; n < ngrams; ++n) {
      auto const d_str = d_strings.element<cudf::string_view>(n + idx);
      bytes += d_str.size_bytes();
      if (out_ptr) out_ptr = cudf::strings::detail::copy_string(out_ptr, d_str);
      if ((n + 1) >= ngrams) continue;
      bytes += d_separator.size_bytes();
      if (out_ptr) out_ptr = cudf::strings::detail::copy_string(out_ptr, d_separator);
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

}  // namespace

std::unique_ptr<cudf::column> generate_ngrams(cudf::strings_column_view const& strings,
                                              cudf::size_type ngrams,
                                              cudf::string_scalar const& separator,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    separator.is_valid(stream), "Parameter separator must be valid", std::invalid_argument);
  cudf::string_view const d_separator(separator.data(), separator.size());
  CUDF_EXPECTS(ngrams > 1,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);

  auto strings_count = strings.size();
  if (strings_count == 0)  // if no strings, return an empty column
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // first create a new offsets vector removing nulls and empty strings from the input column
  std::unique_ptr<cudf::column> non_empty_offsets_column = [&] {
    cudf::column_view offsets_view(
      strings.offsets().type(), strings_count + 1, strings.offsets().head(), nullptr, 0);
    auto table_offsets = cudf::detail::copy_if(
                           cudf::table_view({offsets_view}),
                           [d_strings, strings_count] __device__(cudf::size_type idx) -> bool {
                             if (idx == strings_count) return true;
                             if (d_strings.is_null(idx)) return false;
                             return !d_strings.element<cudf::string_view>(idx).empty();
                           },
                           stream,
                           cudf::get_current_device_resource_ref())
                           ->release();
    strings_count = table_offsets.front()->size() - 1;
    auto result   = std::move(table_offsets.front());
    return result;
  }();  // this allows freeing the temporary table_offsets

  CUDF_EXPECTS(strings_count >= ngrams, "Insufficient number of strings to generate ngrams");
  // create a temporary column view from the non-empty offsets and chars column views
  cudf::column_view strings_view(cudf::data_type{cudf::type_id::STRING},
                                 strings_count,
                                 strings.chars_begin(stream),
                                 nullptr,
                                 0,
                                 0,
                                 {non_empty_offsets_column->view()});
  strings_column = cudf::column_device_view::create(strings_view, stream);
  d_strings      = *strings_column;

  // compute the number of strings of ngrams
  auto const ngrams_count = strings_count - ngrams + 1;

  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    ngram_generator_fn{d_strings, ngrams, d_separator}, ngrams_count, stream, mr);

  // make the output strings column from the offsets and chars column
  return cudf::make_strings_column(
    ngrams_count, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

}  // namespace detail

std::unique_ptr<cudf::column> generate_ngrams(cudf::strings_column_view const& strings,
                                              cudf::size_type ngrams,
                                              cudf::string_scalar const& separator,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::generate_ngrams(strings, ngrams, separator, stream, mr);
}

namespace detail {
namespace {

constexpr cudf::thread_index_type block_size       = 256;
constexpr cudf::thread_index_type bytes_per_thread = 4;

/**
 * @brief Counts the number of ngrams in each row of the given strings column
 *
 * Each warp/thread processes a single string.
 * Formula is `count = max(0,str.length() - ngrams + 1)`
 * If a string has less than ngrams characters, its count is 0.
 */
CUDF_KERNEL void count_char_ngrams_kernel(cudf::column_device_view const d_strings,
                                          cudf::size_type ngrams,
                                          cudf::size_type tile_size,
                                          cudf::size_type* d_counts)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();

  auto const str_idx = idx / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) {
    d_counts[str_idx] = 0;
    return;
  }

  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (tile_size == 1) {
    d_counts[str_idx] = cuda::std::max(0, (d_str.length() + 1 - ngrams));
    return;
  }

  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());

  auto const end = d_str.data() + d_str.size_bytes();

  auto const lane_idx   = warp.thread_rank();
  cudf::size_type count = 0;
  for (auto itr = d_str.data() + (lane_idx * bytes_per_thread); itr < end;
       itr += tile_size * bytes_per_thread) {
    for (auto s = itr; (s < (itr + bytes_per_thread)) && (s < end); ++s) {
      count += static_cast<cudf::size_type>(cudf::strings::detail::is_begin_utf8_char(*s));
    }
  }
  auto const char_count = cg::reduce(warp, count, cg::plus<int>());
  if (lane_idx == 0) { d_counts[str_idx] = cuda::std::max(0, char_count - ngrams + 1); }
}

/**
 * @brief Generate character ngrams for each string
 *
 * Each string produces many strings depending on the ngram width and the string size.
 * This functor can be used with `make_strings_children` to build the offsets and
 * the chars child columns.
 */
struct character_ngram_generator_fn {
  cudf::column_device_view const d_strings;
  cudf::size_type ngrams;
  cudf::size_type const* d_ngram_offsets{};
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return;
    auto itr                = d_str.begin();
    auto const ngram_offset = d_ngram_offsets[idx];
    auto const ngram_count  = d_ngram_offsets[idx + 1] - ngram_offset;
    auto d_output_sizes     = d_sizes + ngram_offset;
    auto out_ptr            = d_chars ? d_chars + d_offsets[ngram_offset] : nullptr;
    for (cudf::size_type n = 0; n < ngram_count; ++n, ++itr) {
      auto const begin = itr.byte_offset();
      auto const end   = (itr + ngrams).byte_offset();
      if (d_chars) {
        out_ptr =
          cudf::strings::detail::copy_and_increment(out_ptr, d_str.data() + begin, (end - begin));
      } else {
        *d_output_sizes++ = end - begin;
      }
    }
  }
};
}  // namespace

std::unique_ptr<cudf::column> generate_character_ngrams(cudf::strings_column_view const& input,
                                                        cudf::size_type ngrams,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(ngrams >= 2,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);

  if (input.is_empty()) {  // if no strings, return an empty column
    return cudf::lists::detail::make_empty_lists_column(cudf::data_type{cudf::type_id::STRING});
  }
  if (input.size() == input.null_count()) {
    return cudf::lists::detail::make_all_nulls_lists_column(
      input.size(), cudf::data_type{cudf::type_id::STRING}, stream, mr);
  }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto [offsets, total_ngrams] = [&] {
    auto counts               = rmm::device_uvector<cudf::size_type>(input.size(), stream);
    auto const avg_char_bytes = (input.chars_size(stream) / (input.size() - input.null_count()));
    auto const tile_size      = (avg_char_bytes < AVG_CHAR_BYTES_THRESHOLD)
                                  ? 1                         // thread per row
                                  : cudf::detail::warp_size;  // warp per row
    auto const grid           = cudf::detail::grid_1d(
      static_cast<cudf::thread_index_type>(input.size()) * tile_size, block_size);
    count_char_ngrams_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_strings, ngrams, tile_size, counts.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    return cudf::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr);
  }();
  auto d_offsets = offsets->view().data<cudf::size_type>();

  CUDF_EXPECTS(total_ngrams > 0,
               "Insufficient number of characters in each string to generate ngrams");

  character_ngram_generator_fn generator{*d_strings, ngrams, d_offsets};
  auto [offsets_column, chars] =
    cudf::strings::detail::make_strings_children(generator, input.size(), total_ngrams, stream, mr);

  auto output = cudf::make_strings_column(
    total_ngrams, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});

  return make_lists_column(
    input.size(), std::move(offsets), std::move(output), 0, rmm::device_buffer{});
}

namespace {

/**
 * @brief Computes the hash of each character ngram
 *
 * Each warp processes a single string. Substrings are resolved for every character
 * of the string and hashed.
 */
CUDF_KERNEL void character_ngram_hash_kernel(cudf::column_device_view const d_strings,
                                             cudf::size_type ngrams,
                                             uint32_t seed,
                                             cudf::size_type const* d_ngram_offsets,
                                             cudf::hash_value_type* d_results)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / cudf::detail::warp_size;
  if (str_idx >= d_strings.size() or d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) { return; }

  __shared__ cudf::hash_value_type hvs[block_size];  // temp store for hash values

  auto const ngram_offset = d_ngram_offsets[str_idx];
  auto const hasher       = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{seed};

  auto const end        = d_str.data() + d_str.size_bytes();
  auto const warp_count = (d_str.size_bytes() / cudf::detail::warp_size) + 1;

  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto d_hashes = d_results + ngram_offset;
  auto itr      = d_str.data() + lane_idx;
  for (auto i = 0; i < warp_count; ++i) {
    cudf::hash_value_type hash = 0;
    if (itr < end && cudf::strings::detail::is_begin_utf8_char(*itr)) {
      // resolve ngram substring
      auto const sub_str =
        cudf::string_view(itr, static_cast<cudf::size_type>(cuda::std::distance(itr, end)));
      auto const [bytes, left] =
        cudf::strings::detail::bytes_to_character_position(sub_str, ngrams);
      if (left == 0) { hash = hasher(cudf::string_view(itr, bytes)); }
    }
    hvs[threadIdx.x] = hash;  // store hash into shared memory
    warp.sync();
    if (lane_idx == 0) {
      // copy valid hash values into d_hashes
      auto const hashes = &hvs[threadIdx.x];
      d_hashes          = thrust::copy_if(
        thrust::seq, hashes, hashes + cudf::detail::warp_size, d_hashes, [](auto h) {
          return h != 0;
        });
    }
    warp.sync();
    itr += cudf::detail::warp_size;
  }
}
}  // namespace

std::unique_ptr<cudf::column> hash_character_ngrams(cudf::strings_column_view const& input,
                                                    cudf::size_type ngrams,
                                                    uint32_t seed,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(ngrams >= 2,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);

  auto output_type = cudf::data_type{cudf::type_to_id<cudf::hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto const grid      = cudf::detail::grid_1d(
    static_cast<cudf::thread_index_type>(input.size()) * cudf::detail::warp_size, block_size);

  // build offsets column by computing the number of ngrams per string
  auto [offsets, total_ngrams] = [&] {
    auto counts = rmm::device_uvector<cudf::size_type>(input.size(), stream);
    count_char_ngrams_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_strings, ngrams, cudf::detail::warp_size, counts.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    return cudf::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr);
  }();
  auto d_offsets = offsets->view().data<cudf::size_type>();

  CUDF_EXPECTS(total_ngrams > 0,
               "Insufficient number of characters in each string to generate ngrams");

  // compute ngrams and build hashes
  auto hashes =
    cudf::make_numeric_column(output_type, total_ngrams, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_hashes = hashes->mutable_view().data<cudf::hash_value_type>();

  character_ngram_hash_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, ngrams, seed, d_offsets, d_hashes);
  CUDF_CUDA_TRY(cudaGetLastError());

  return make_lists_column(
    input.size(), std::move(offsets), std::move(hashes), 0, rmm::device_buffer{});
}

}  // namespace detail

std::unique_ptr<cudf::column> generate_character_ngrams(cudf::strings_column_view const& strings,
                                                        cudf::size_type ngrams,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::generate_character_ngrams(strings, ngrams, stream, mr);
}

std::unique_ptr<cudf::column> hash_character_ngrams(cudf::strings_column_view const& strings,
                                                    cudf::size_type ngrams,
                                                    uint32_t seed,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::hash_character_ngrams(strings, ngrams, seed, stream, mr);
}

}  // namespace nvtext
