/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#include <nanoarrow/nanoarrow.hpp>
#include <nvbench/nvbench.cuh>

#include <cstdlib>
#include <format>
#include <string>

namespace {

// Runtime switch to use ArrowStringView instead of cudf's Arrow string format.
//
// Set to anything to use ArrowStringView, and unset to use cudf.
// Example command line to generate ArrowStringView numbers for sv_hash benchmark:
//   CUDF_BM_ARROWSTRINGVIEW=1 benchmarks/STRINGS_EXPERIMENTAL_NVBENCH -d 0 -b sv_hash
//
// This will generate nvbench benchmark outputs that can be compared directly
// using the `nvbench compare.py` script.
auto const BM_ARROWSTRINGVIEW = "CUDF_BM_ARROWSTRINGVIEW";

/**
 * Creates ArrowBinaryView objects from a strings column.
 */
struct strings_to_binary_view {
  cudf::column_device_view d_strings;
  cudf::detail::input_offsetalator d_offsets;
  ArrowBinaryView* d_items;  // output

  __device__ void operator()(cudf::size_type idx) const
  {
    auto& item = d_items[idx];
    if (d_strings.is_null(idx)) {
      item.inlined.size = 0;  // not used in this benchmark
      return;
    }

    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    item.inlined.size = d_str.size_bytes();
    // copy the string data to the inlined buffer if it fits
    if (d_str.size_bytes() <= NANOARROW_BINARY_VIEW_INLINE_SIZE) {
      thrust::copy(thrust::seq, d_str.data(), d_str.data() + d_str.size_bytes(), item.inlined.data);
      thrust::uninitialized_fill(thrust::seq,
                                 item.inlined.data + item.inlined.size,
                                 item.inlined.data + NANOARROW_BINARY_VIEW_INLINE_SIZE,
                                 0);
    } else {
      // otherwise, copy the prefix and set the offset to the data buffer
      thrust::copy(thrust::seq,
                   d_str.data(),
                   d_str.data() + NANOARROW_BINARY_VIEW_PREFIX_SIZE,
                   item.ref.prefix);
      auto const offset     = d_offsets[idx];
      item.ref.buffer_index = 0;  // only one buffer in this benchmark
      item.ref.offset       = static_cast<int32_t>(offset);
    }
  }
};

/**
 * Returns a string_view from an ArrowBinaryView.
 * This helps in the comparison by both implementations using `cudf::string_view`
 * as the base type so the actual operations are the same and only the
 * format (how the data is organized) is different.
 */
__device__ cudf::string_view get_string_view(ArrowBinaryView const& item, char const* d_chars)
{
  auto const data = item.inlined.size <= NANOARROW_BINARY_VIEW_INLINE_SIZE
                      ? reinterpret_cast<char const*>(item.inlined.data)
                      : d_chars + item.ref.offset;
  return cudf::string_view(data, item.inlined.size);
}

/**
 * Hashes a string from an ArrowBinaryView.
 */
struct hash_arrow_sv {
  ArrowBinaryView* d_items;
  char const* d_chars;
  __device__ cudf::hash_value_type operator()(cudf::size_type idx) const
  {
    auto& item        = d_items[idx];
    auto const d_str  = get_string_view(item, d_chars);
    auto const hasher = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{0};
    return hasher(d_str);
  }
};

/**
 * Checks if a string from an ArrowBinaryView starts with a target string.
 */
struct starts_arrow_sv {
  ArrowBinaryView* d_items;
  char const* d_chars;
  cudf::size_type tgt_size;
  __device__ bool operator()(cudf::size_type idx) const
  {
    // note that this requires tgt_size <= 26
    auto const d_tgt = cudf::string_view("abcdefghijklmnopqrstuvwxyz", tgt_size);
    auto& item       = d_items[idx];
    auto const size  = item.inlined.size;
    auto const data  = (size <= NANOARROW_BINARY_VIEW_INLINE_SIZE) || (tgt_size <= 4)
                         ? reinterpret_cast<char const*>(item.inlined.data)
                         : d_chars + item.ref.offset;
    auto const d_str = cudf::string_view(data, size);
    return d_str.size_bytes() >= d_tgt.size_bytes() &&
           d_tgt.compare(d_str.data(), d_tgt.size_bytes()) == 0;
  }
};

/**
 * Compares two strings from ArrowBinaryView objects.
 */
struct compare_arrow_sv {
  ArrowBinaryView* d_items;
  char const* d_chars;
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs)
  {
    auto& item_lhs = d_items[lhs];
    auto& item_rhs = d_items[rhs];

    // shortcut to check preview bytes
    auto pv_lhs = reinterpret_cast<uint32_t const*>(item_lhs.inlined.data)[0];
    auto pv_rhs = reinterpret_cast<uint32_t const*>(item_rhs.inlined.data)[0];
    if (pv_lhs != pv_rhs) {
      return cudf::hashing::detail::swap_endian(pv_lhs) <
             cudf::hashing::detail::swap_endian(pv_rhs);
    }

    // prefix matches so check how many bytes are left to compare
    constexpr auto prefix_size = static_cast<cudf::size_type>(sizeof(uint32_t));
    auto const size_lhs        = item_lhs.inlined.size;
    auto const size_rhs        = item_rhs.inlined.size;
    // if no bytes left to compare, we are done (strings are equal)
    if (size_lhs <= prefix_size && size_rhs <= prefix_size) { return false; }

    // compare the remaining bytes
    auto const d_str_lhs = cudf::string_view(
      get_string_view(item_lhs, d_chars).data() + prefix_size, size_lhs - prefix_size);
    auto const d_str_rhs = cudf::string_view(
      get_string_view(item_rhs, d_chars).data() + prefix_size, size_rhs - prefix_size);

    return d_str_lhs < d_str_rhs;
  }
};

/**
 * Hashes a string from a cudf column
 */
struct hash_sv {
  cudf::column_device_view d_strings;
  __device__ cudf::hash_value_type operator()(cudf::size_type idx) const
  {
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const hasher = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{0};
    return hasher(d_str);
  }
};

/**
 * Checks if a string from a cudf column starts with a target string
 */
struct starts_sv {
  cudf::column_device_view d_strings;
  cudf::size_type tgt_size;
  __device__ bool operator()(cudf::size_type idx) const
  {
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    auto const d_tgt = cudf::string_view("abcdefghijklmnopqrstuvwxyz", tgt_size);
    return d_str.size_bytes() >= d_tgt.size_bytes() &&
           d_tgt.compare(d_str.data(), d_tgt.size_bytes()) == 0;
  }
};

/**
 * Compares two strings from a cudf column
 */
struct compare_sv {
  cudf::column_device_view d_strings;
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs)
  {
    auto const d_str_lhs = d_strings.element<cudf::string_view>(lhs);
    auto const d_str_rhs = d_strings.element<cudf::string_view>(rhs);
    return d_str_lhs < d_str_rhs;
  }
};

/**
 * Creates an ArrowBinaryView vector and data buffer from a strings column.
 */
std::pair<rmm::device_uvector<ArrowBinaryView>, rmm::device_buffer> create_sv_array(
  cudf::strings_column_view const& input, cuda::stream_ref stream)
{
  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // count the (longer) strings that will need to be stored in the data buffer
  auto const num_longer_strings = thrust::count_if(
    rmm::exec_policy_nosync(stream),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{input.size()},
    [d_offsets] __device__(auto idx) {
      return d_offsets[idx + 1] - d_offsets[idx] > NANOARROW_BINARY_VIEW_INLINE_SIZE;
    });

  // gather all the long-ish strings into a single strings column
  auto [unused_col, longer_strings] = [&] {
    if (num_longer_strings == input.size()) {
      // we can use the input column as is for the remainder of this function
      return std::pair{cudf::make_empty_column(cudf::type_id::STRING), input};
    }
    auto indices = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<cudf::strings::detail::string_index_pair>(
        [d_strings = *d_strings] __device__(auto idx) {
          if (d_strings.is_null(idx)) {
            return cudf::strings::detail::string_index_pair{nullptr, 0};
          }
          auto const d_str = d_strings.element<cudf::string_view>(idx);
          return (d_str.size_bytes() > NANOARROW_BINARY_VIEW_INLINE_SIZE)
                   ? cudf::strings::detail::string_index_pair{d_str.data(), d_str.size_bytes()}
                   : cudf::strings::detail::string_index_pair{"", 0};
        }));
    auto longer_strings = cudf::strings::detail::make_strings_column(
      indices, indices + input.size(), stream, cudf::get_current_device_resource_ref());
    stream.sync();
    auto const sv = cudf::strings_column_view(longer_strings->view());
    return std::pair{std::move(longer_strings), sv};
  }();
  auto [first, last] = cudf::strings::detail::get_first_and_last_offset(longer_strings, stream);
  auto const longer_chars_size = last - first;

  // Make sure only one buffer is needed.
  // Using a single data buffer makes the two formats more similar focusing on the layout.
  constexpr int64_t max_size = std::numeric_limits<cudf::size_type>::max() / 2;
  auto const num_buffers     = cudf::util::div_rounding_up_safe(longer_chars_size, max_size);
  CUDF_EXPECTS(num_buffers <= 1, "num_buffers must be <= 1");

  // now build BinaryView objects from the strings in device memory
  // (for-each works better than transform due to the prefix/data of the ArrowBinaryView)
  auto d_items = rmm::device_uvector<ArrowBinaryView>(input.size(), stream);
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     cuda::counting_iterator<cudf::size_type>{0},
                     input.size(),
                     strings_to_binary_view{*d_strings, d_offsets, d_items.data()});

  rmm::device_buffer data_buffer(longer_chars_size, stream);
  auto const chars_data = longer_strings.chars_begin(stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    data_buffer.data(), chars_data, longer_chars_size, cudaMemcpyDefault, stream.get()));

  return std::pair{std::move(d_items), std::move(data_buffer)};
}

template <typename MapIterator>
std::pair<rmm::device_uvector<ArrowBinaryView>, rmm::device_buffer> gather_sv_array(
  rmm::device_uvector<ArrowBinaryView> const& d_items,
  rmm::device_buffer const& data,
  MapIterator begin,
  cudf::size_type map_size,
  cuda::stream_ref stream)
{
  auto output   = rmm::device_uvector<ArrowBinaryView>(map_size, stream);
  auto d_output = output.data();
  thrust::gather(
    rmm::exec_policy_nosync(stream), begin, begin + map_size, d_items.data(), d_output);
  // Although the above should be enough, in reality it is impractical to share a data buffer
  // between two columns in libcudf. Sharing data in column_views is expected but a libcudf
  // gather (all other libcudf APIs) would return a new column with newly owned (non-shared)
  // data. At best, the data-buffer could be simply copied but it seems impractical in general
  // to not compact the buffer and remove any unused data characters from it.
  // Also, the compaction could help coalesce accessing adjacent row data in down stream calls.
  //
  // The rest of the code compacts the data buffer appropriately.

  // record sizes of the long buffers (only single data buffer is supported in this benchmark)
  auto offsets   = rmm::device_uvector<int32_t>(map_size + 1, stream);
  auto d_offsets = offsets.data();
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     cuda::counting_iterator<cudf::size_type>{0},
                     map_size,
                     [d_offsets, d_output] __device__(cudf::size_type idx) {
                       auto& item      = d_output[idx];
                       auto const size = static_cast<int32_t>(item.inlined.size);
                       d_offsets[idx]  = size > NANOARROW_BINARY_VIEW_INLINE_SIZE ? size : 0;
                     });
  // convert the sizes to offsets (offsets are only for compacting the data)
  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream), offsets.begin(), offsets.end(), offsets.begin());
  auto total_size  = offsets.element(map_size, stream);
  auto output_data = rmm::device_buffer(total_size, stream);
  if (total_size > 0) {
    auto d_output_data = static_cast<char*>(output_data.data());
    auto d_input_data  = static_cast<char const*>(data.data());

    // rebuild the data buffer with only the data from the gather
    // and reset each binary-view offset appropriately
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream),
      cuda::counting_iterator<cudf::size_type>{0},
      map_size,
      [d_offsets, d_input_data, d_output, d_output_data] __device__(cudf::size_type idx) {
        auto& item      = d_output[idx];
        auto const size = item.inlined.size;
        if (size <= NANOARROW_BINARY_VIEW_INLINE_SIZE) { return; }
        auto const offset = d_offsets[idx];
        auto const d_out  = d_output_data + offset;
        auto const d_in   = d_input_data + item.ref.offset;
        memcpy(d_out, d_in, size);
        item.ref.offset = offset;
      });
  }

  return std::pair{std::move(output), std::move(output_data)};
}

}  // namespace

static void BM_sv_hash(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const min_width = state.get_int64("fw") ? max_width : 1;  // fw = fixed width

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .no_validity();
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto col_view     = column->view();
  auto stream       = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_writes(num_rows * sizeof(cudf::hash_value_type));
  auto output = rmm::device_uvector<cudf::hash_value_type>(num_rows, stream);
  auto begin  = cuda::counting_iterator<cudf::size_type>{0};
  auto end    = cuda::counting_iterator<cudf::size_type>{num_rows};

  if (std::getenv(BM_ARROWSTRINGVIEW)) {
    auto [d_items, data_buffer] = create_sv_array(col_view, stream);
    auto const d_chars          = reinterpret_cast<char const*>(data_buffer.data());
    state.add_global_memory_reads(num_rows * sizeof(ArrowBinaryView) + data_buffer.size());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      thrust::transform(rmm::exec_policy_nosync(stream),
                        begin,
                        end,
                        output.begin(),
                        hash_arrow_sv{d_items.data(), d_chars});
    });
  } else {
    auto d_strings = cudf::column_device_view::create(col_view, stream);
    auto col_size  = column->alloc_size();
    state.add_global_memory_reads(col_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      thrust::transform(
        rmm::exec_policy_nosync(stream), begin, end, output.begin(), hash_sv{*d_strings});
    });
  }
}

static void BM_sv_starts(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const min_width = state.get_int64("fw") ? max_width : 1;
  auto const tgt_size  = static_cast<cudf::size_type>(state.get_int64("tgt_size"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .no_validity();
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto col_view     = column->view();
  auto stream       = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_writes(num_rows * sizeof(bool));
  auto output = rmm::device_uvector<bool>(num_rows, stream);
  auto begin  = cuda::counting_iterator<cudf::size_type>{0};
  auto end    = cuda::counting_iterator<cudf::size_type>{num_rows};

  if (std::getenv(BM_ARROWSTRINGVIEW)) {
    auto [d_items, data_buffer] = create_sv_array(col_view, stream);
    auto const d_chars          = reinterpret_cast<char const*>(data_buffer.data());
    state.add_global_memory_reads(num_rows * sizeof(ArrowBinaryView) + data_buffer.size());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      thrust::transform(rmm::exec_policy_nosync(stream),
                        begin,
                        end,
                        output.begin(),
                        starts_arrow_sv{d_items.data(), d_chars, tgt_size});
    });
  } else {
    auto d_strings = cudf::column_device_view::create(col_view, stream);
    auto col_size  = column->alloc_size();
    state.add_global_memory_reads(col_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      thrust::transform(rmm::exec_policy_nosync(stream),
                        begin,
                        end,
                        output.begin(),
                        starts_sv{*d_strings, tgt_size});
    });
  }
}

static void BM_sv_sort(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const card      = static_cast<cudf::size_type>(state.get_int64("card"));

  auto h_data = std::vector<std::string>(card);
  std::transform(cuda::counting_iterator<cudf::size_type>{0},
                 cuda::counting_iterator<cudf::size_type>{card},
                 h_data.begin(),
                 [max_width](auto idx) {
                   auto const fmt = std::format("{{:0{}d}}", max_width);
                   return std::vformat(fmt, std::make_format_args(idx));
                 });
  auto d_data = cudf::test::strings_column_wrapper(h_data.begin(), h_data.end()).release();

  data_profile gather_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, d_data->size() - 1);
  auto gather_map = create_random_column(cudf::type_id::INT32, row_count{num_rows}, gather_profile);

  auto table  = cudf::gather(cudf::table_view({d_data->view()}), gather_map->view());
  auto column = std::move(table->release().front());

  auto col_view = column->view();
  auto stream   = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_writes(num_rows * sizeof(cudf::size_type));

  // indices are the keys that are sorted (not inplace)
  auto keys      = rmm::device_uvector<cudf::size_type>(num_rows, stream);
  auto in_keys   = cuda::counting_iterator<cudf::size_type>{0};
  auto out_keys  = keys.begin();
  auto tmp_bytes = std::size_t{0};

  if (std::getenv(BM_ARROWSTRINGVIEW)) {
    auto [d_items, data_buffer] = create_sv_array(col_view, stream);
    auto const d_chars          = reinterpret_cast<char const*>(data_buffer.data());
    auto comparator             = compare_arrow_sv{d_items.data(), d_chars};
    cub::DeviceMergeSort::SortKeysCopy(
      nullptr, tmp_bytes, in_keys, out_keys, num_rows, comparator, stream.get());
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    state.add_global_memory_reads(num_rows * sizeof(ArrowBinaryView) + data_buffer.size());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cub::DeviceMergeSort::SortKeysCopy(
        tmp_stg.data(), tmp_bytes, in_keys, out_keys, num_rows, comparator, stream.get());
    });
  } else {
    auto d_strings = cudf::column_device_view::create(col_view, stream);
    auto col_size  = column->alloc_size();
    state.add_global_memory_reads(col_size);
    auto comparator = compare_sv{*d_strings};
    cub::DeviceMergeSort::SortKeysCopy(
      nullptr, tmp_bytes, in_keys, out_keys, num_rows, comparator, stream.get());
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    state.add_global_memory_reads(col_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cub::DeviceMergeSort::SortKeysCopy(
        tmp_stg.data(), tmp_bytes, in_keys, out_keys, num_rows, comparator, stream.get());
    });
  }
}

static void BM_sv_gather(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const width    = static_cast<cudf::size_type>(state.get_int64("width"));
  auto const map_rows = static_cast<cudf::size_type>(state.get_int64("map_rows"));

  data_profile profile = data_profile_builder().no_validity().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, width, width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto col_view     = column->view();

  data_profile map_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, num_rows - 1);
  auto map      = create_random_column(cudf::type_id::INT32, row_count{map_rows}, map_profile);
  auto map_view = map->view();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  if (std::getenv(BM_ARROWSTRINGVIEW)) {
    auto [d_items, data_buffer] = create_sv_array(col_view, stream);

    auto const begin  = map_view.begin<cudf::size_type>();
    auto const result = gather_sv_array(d_items, data_buffer, begin, map_rows, stream);
    auto gather_size  = result.first.size() * sizeof(ArrowBinaryView) + result.second.size();
    state.add_global_memory_reads(gather_size);
    state.add_global_memory_writes(gather_size);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      gather_sv_array(d_items, data_buffer, begin, map_rows, stream);
    });
  } else {
    auto const result = cudf::gather(
      cudf::table_view({col_view}), map_view, cudf::out_of_bounds_policy::DONT_CHECK, stream);

    state.add_global_memory_reads(result->alloc_size());
    state.add_global_memory_writes(result->alloc_size());

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::gather(
        cudf::table_view({col_view}), map_view, cudf::out_of_bounds_policy::DONT_CHECK, stream);
    });
  }
}

NVBENCH_BENCH(BM_sv_hash)
  .set_name("sv_hash")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000})
  .add_int64_axis("max_width", {5, 10, 15, 20, 30, 60})
  .add_int64_axis("fw", {1, 0});

NVBENCH_BENCH(BM_sv_starts)
  .set_name("sv_starts")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000})
  .add_int64_axis("max_width", {10, 20, 30, 60})
  .add_int64_axis("tgt_size", {4, 8, 16})
  .add_int64_axis("fw", {1, 0});

NVBENCH_BENCH(BM_sv_sort)
  .set_name("sv_sort")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000})
  .add_int64_axis("max_width", {10, 20, 30, 60})
  .add_int64_axis("card", {100, 1000});

NVBENCH_BENCH(BM_sv_gather)
  .set_name("sv_gather")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000})
  .add_int64_axis("width", {6, 12, 24, 48, 64})
  .add_int64_axis("map_rows", {10'000, 100'000});
