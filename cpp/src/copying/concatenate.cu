/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/concatenate.hpp>
#include <cudf/detail/concatenate_masks.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/structs/detail/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cudf {
namespace detail {
namespace {

// From benchmark data, the fused kernel optimization appears to perform better
// when there are more than a trivial number of columns, or when the null mask
// can also be computed at the same time
constexpr bool use_fused_kernel_heuristic(bool const has_nulls, size_t const num_columns)
{
  return has_nulls || num_columns > 4;
}

auto create_device_views(host_span<column_view const> views, rmm::cuda_stream_view stream)
{
  // Create device views for each input view
  using CDViewPtr         = decltype(column_device_view::create(std::declval<column_view>(),
                                                        std::declval<rmm::cuda_stream_view>()));
  auto device_view_owners = std::vector<CDViewPtr>(views.size());
  std::transform(views.begin(), views.end(), device_view_owners.begin(), [stream](auto const& col) {
    return column_device_view::create(col, stream);
  });

  // Assemble contiguous array of device views
  auto device_views =
    cudf::detail::make_empty_host_vector<column_device_view>(views.size(), stream);
  std::transform(device_view_owners.cbegin(),
                 device_view_owners.cend(),
                 std::back_inserter(device_views),
                 [](auto const& col) { return *col; });

  auto d_views =
    make_device_uvector_async(device_views, stream, cudf::get_current_device_resource_ref());

  // Compute the partition offsets
  auto offsets = cudf::detail::make_pinned_vector_async<size_t>(views.size() + 1, stream);
  thrust::transform_inclusive_scan(
    thrust::host,
    device_views.cbegin(),
    device_views.cend(),
    std::next(offsets.begin()),
    [](auto const& col) { return col.size(); },
    cuda::std::plus{});
  auto d_offsets =
    make_device_uvector_async(offsets, stream, cudf::get_current_device_resource_ref());
  auto const output_size = offsets.back();

  return std::make_tuple(
    std::move(device_view_owners), std::move(d_views), std::move(d_offsets), output_size);
}

/**
 * @brief Concatenates the null mask bits of all the column device views in the
 * `views` array to the destination bitmask.
 *
 * @tparam block_size Block size for using with single_lane_block_sum_reduce
 *
 * @param views Array of column_device_view
 * @param output_offsets Prefix sum of sizes of elements of `views`
 * @param number_of_views Size of `views` array
 * @param dest_mask The output buffer to copy null masks into
 * @param number_of_mask_bits The total number of null masks bits that are being copied
 * @param out_valid_count To hold the total number of valid bits set
 */
template <size_type block_size>
CUDF_KERNEL void concatenate_masks_kernel(column_device_view const* views,
                                          size_t const* output_offsets,
                                          size_type number_of_views,
                                          bitmask_type* dest_mask,
                                          size_type number_of_mask_bits,
                                          size_type* out_valid_count)
{
  auto tidx         = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  auto active_mask  = __ballot_sync(0xFFFF'FFFFu, tidx < number_of_mask_bits);

  size_type warp_valid_count = 0;

  while (tidx < number_of_mask_bits) {
    auto const mask_index = static_cast<cudf::size_type>(tidx);
    size_type const source_view_index =
      thrust::upper_bound(
        thrust::seq, output_offsets, output_offsets + number_of_views, mask_index) -
      output_offsets - 1;
    bool bit_is_set = true;
    if (source_view_index < number_of_views) {
      size_type const column_element_index = mask_index - output_offsets[source_view_index];
      bit_is_set = views[source_view_index].is_valid(column_element_index);
    }
    bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

    if (threadIdx.x % detail::warp_size == 0) {
      dest_mask[word_index(mask_index)] = new_word;
      warp_valid_count += __popc(new_word);
    }

    tidx += stride;
    active_mask = __ballot_sync(active_mask, tidx < number_of_mask_bits);
  }

  using detail::single_lane_block_sum_reduce;
  auto const block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
  if (threadIdx.x == 0) { atomicAdd(out_valid_count, block_valid_count); }
}
}  // namespace

size_type concatenate_masks(device_span<column_device_view const> d_views,
                            device_span<size_t const> d_offsets,
                            bitmask_type* dest_mask,
                            size_type output_size,
                            rmm::cuda_stream_view stream)
{
  cudf::detail::device_scalar<size_type> d_valid_count(
    0, stream, cudf::get_current_device_resource_ref());
  constexpr size_type block_size{256};
  cudf::detail::grid_1d config(output_size, block_size);
  concatenate_masks_kernel<block_size>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      d_views.data(),
      d_offsets.data(),
      static_cast<size_type>(d_views.size()),
      dest_mask,
      output_size,
      d_valid_count.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  return output_size - d_valid_count.value(stream);
}

size_type concatenate_masks(host_span<column_view const> views,
                            bitmask_type* dest_mask,
                            rmm::cuda_stream_view stream)
{
  // Preprocess and upload inputs to device memory
  auto const device_views = create_device_views(views, stream);
  auto const& d_views     = std::get<1>(device_views);
  auto const& d_offsets   = std::get<2>(device_views);
  auto const output_size  = std::get<3>(device_views);

  return concatenate_masks(d_views, d_offsets, dest_mask, output_size, stream);
}

namespace {
template <typename T, size_type block_size, bool Nullable>
CUDF_KERNEL void fused_concatenate_kernel(column_device_view const* input_views,
                                          size_t const* input_offsets,
                                          size_type num_input_views,
                                          mutable_column_device_view output_view,
                                          size_type* out_valid_count)
{
  auto const output_size = output_view.size();
  auto* output_data      = output_view.data<T>();

  auto output_index          = cudf::detail::grid_1d::global_thread_id();
  auto const stride          = cudf::detail::grid_1d::grid_stride();
  size_type warp_valid_count = 0;

  unsigned active_mask;
  if (Nullable) { active_mask = __ballot_sync(0xFFFF'FFFFu, output_index < output_size); }
  while (output_index < output_size) {
    // Lookup input index by searching for output index in offsets
    auto const offset_it            = cuda::std::prev(thrust::upper_bound(
      thrust::seq, input_offsets, input_offsets + num_input_views, output_index));
    size_type const partition_index = offset_it - input_offsets;

    // Copy input data to output
    auto const offset_index   = output_index - *offset_it;
    auto const& input_view    = input_views[partition_index];
    auto const* input_data    = input_view.data<T>();
    output_data[output_index] = input_data[offset_index];

    if (Nullable) {
      bool const bit_is_set       = input_view.is_valid(offset_index);
      bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

      // First thread writes bitmask word
      if (threadIdx.x % detail::warp_size == 0) {
        output_view.null_mask()[word_index(output_index)] = new_word;
      }

      warp_valid_count += __popc(new_word);
    }

    output_index += stride;
    if (Nullable) { active_mask = __ballot_sync(active_mask, output_index < output_size); }
  }

  if (Nullable) {
    using detail::single_lane_block_sum_reduce;
    auto block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count, block_valid_count); }
  }
}

template <typename T>
std::unique_ptr<column> fused_concatenate(host_span<column_view const> views,
                                          bool const has_nulls,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  using mask_policy = cudf::mask_allocation_policy;

  // Preprocess and upload inputs to device memory
  auto const device_views = create_device_views(views, stream);
  auto const& d_views     = std::get<1>(device_views);
  auto const& d_offsets   = std::get<2>(device_views);
  auto const output_size  = std::get<3>(device_views);

  CUDF_EXPECTS(output_size <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated rows exceeds the column size limit",
               std::overflow_error);

  // Allocate output
  auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
  auto out_col      = detail::allocate_like(views.front(), output_size, policy, stream, mr);
  auto out_view     = out_col->mutable_view();
  auto d_out_view   = mutable_column_device_view::create(out_view, stream);

  cudf::detail::device_scalar<size_type> d_valid_count(
    0, stream, cudf::get_current_device_resource_ref());

  // Launch kernel
  constexpr size_type block_size{256};
  cudf::detail::grid_1d config(output_size, block_size);
  auto const kernel = has_nulls ? fused_concatenate_kernel<T, block_size, true>
                                : fused_concatenate_kernel<T, block_size, false>;
  kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    d_views.data(),
    d_offsets.data(),
    static_cast<size_type>(d_views.size()),
    *d_out_view,
    d_valid_count.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  if (has_nulls) {
    out_col->set_null_count(output_size - d_valid_count.value(stream));
  } else {
    out_col->set_null_count(0);  // prevent null count from being materialized
  }

  return out_col;
}

template <typename T>
std::unique_ptr<column> for_each_concatenate(host_span<column_view const> views,
                                             bool const has_nulls,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  size_type const total_element_count =
    std::accumulate(views.begin(), views.end(), 0, [](auto accumulator, auto const& v) {
      return accumulator + v.size();
    });

  using mask_policy = cudf::mask_allocation_policy;
  auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
  auto col = cudf::detail::allocate_like(views.front(), total_element_count, policy, stream, mr);

  auto m_view = col->mutable_view();

  auto const num_views = views.size();
  std::vector<void*> dsts(num_views);
  std::vector<void const*> srcs(num_views);
  std::vector<std::size_t> sizes(num_views);
  size_type count = 0;
  for (std::size_t i = 0; i < num_views; ++i) {
    dsts[i]  = m_view.begin<T>() + count;
    srcs[i]  = views[i].begin<T>();
    sizes[i] = views[i].size() * sizeof(T);
    count += views[i].size();
  }
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_batch_async(dsts.data(), srcs.data(), sizes.data(), num_views, stream));

  // If concatenated column is nullable, proceed to calculate it
  if (has_nulls) {
    col->set_null_count(
      cudf::detail::concatenate_masks(views, (col->mutable_view()).null_mask(), stream));
  } else {
    col->set_null_count(0);  // prevent null count from being materialized
  }

  return col;
}

struct concatenate_dispatch {
  host_span<column_view const> views;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  // fixed width
  template <typename T>
  std::unique_ptr<column> operator()()
  {
    bool const has_nulls =
      std::any_of(views.begin(), views.end(), [](auto const& col) { return col.has_nulls(); });

    // Use a heuristic to guess when the fused kernel will be faster
    if (use_fused_kernel_heuristic(has_nulls, views.size())) {
      return fused_concatenate<T>(views, has_nulls, stream, mr);
    } else {
      return for_each_concatenate<T>(views, has_nulls, stream, mr);
    }
  }
};

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::dictionary32>()
{
  return cudf::dictionary::detail::concatenate(views, stream, mr);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::string_view>()
{
  return cudf::strings::detail::concatenate(views, stream, mr);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::list_view>()
{
  return cudf::lists::detail::concatenate(views, stream, mr);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::struct_view>()
{
  return cudf::structs::detail::concatenate(views, stream, mr);
}

void bounds_and_type_check(host_span<column_view const> cols, rmm::cuda_stream_view stream);

/**
 * @brief Functor for traversing child columns and recursively verifying concatenation
 * bounds and types.
 */
class traverse_children {
 public:
  // nothing to do for simple types.
  template <typename T>
  void operator()(host_span<column_view const>, rmm::cuda_stream_view)
  {
  }

 private:
  // verify length of concatenated offsets.
  void check_offsets_size(host_span<column_view const> cols)
  {
    // offsets.  we can't just add up the total sizes of all offset child columns because each one
    // has an extra value, regardless of the # of parent rows.  So we have to add up the total # of
    // rows in the base column and add 1 at the end
    size_t const total_offset_count =
      std::accumulate(cols.begin(),
                      cols.end(),
                      std::size_t{},
                      [](size_t a, auto const& b) -> size_t { return a + b.size(); }) +
      1;
    CUDF_EXPECTS(total_offset_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
                 "Total number of concatenated offsets exceeds the column size limit",
                 std::overflow_error);
  }
};

template <>
void traverse_children::operator()<cudf::string_view>(host_span<column_view const> cols,
                                                      rmm::cuda_stream_view stream)
{
  // verify offsets
  check_offsets_size(cols);

  // chars -- checked in call to cudf::strings::detail::concatenate
}

template <>
void traverse_children::operator()<cudf::struct_view>(host_span<column_view const> cols,
                                                      rmm::cuda_stream_view stream)
{
  // march each child
  auto child_iter         = cuda::counting_iterator<cudf::size_type>{0};
  auto const num_children = cols.front().num_children();
  std::vector<column_view> nth_children;
  nth_children.reserve(cols.size());
  std::for_each(child_iter, child_iter + num_children, [&](auto child_index) {
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(nth_children),
                   [child_index, stream](column_view const& col) {
                     structs_column_view scv(col);
                     return scv.get_sliced_child(child_index, stream);
                   });

    bounds_and_type_check(nth_children, stream);
    nth_children.clear();
  });
}

template <>
void traverse_children::operator()<cudf::list_view>(host_span<column_view const> cols,
                                                    rmm::cuda_stream_view stream)
{
  // verify offsets
  check_offsets_size(cols);

  // recurse into the child columns
  std::vector<column_view> nth_children;
  nth_children.reserve(cols.size());
  std::transform(
    cols.begin(), cols.end(), std::back_inserter(nth_children), [stream](column_view const& col) {
      lists_column_view lcv(col);
      return lcv.get_sliced_child(stream);
    });
  bounds_and_type_check(nth_children, stream);
}

/**
 * @brief Verifies that the sum of the sizes of all the columns to be concatenated
 * will not exceed the max value of size_type, and verifies all column types match
 *
 * @param cols Span of columns to check
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @throws cudf::logic_error if the total length of the concatenated columns would
 * exceed the max value of size_type
 *
 * @throws cudf::logic_error if all of the input column types don't match
 */
void bounds_and_type_check(host_span<column_view const> cols, rmm::cuda_stream_view stream)
{
  // total size of all concatenated rows
  size_t const total_row_count =
    std::accumulate(cols.begin(), cols.end(), std::size_t{}, [](size_t a, auto const& b) {
      return a + static_cast<size_t>(b.size());
    });
  CUDF_EXPECTS(total_row_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated rows exceeds the column size limit",
               std::overflow_error);

  if (std::any_of(cols.begin(), cols.end(), [](column_view const& c) {
        return c.type().id() == cudf::type_id::EMPTY;
      })) {
    CUDF_EXPECTS(
      std::all_of(cols.begin(),
                  cols.end(),
                  [](column_view const& c) { return c.type().id() == cudf::type_id::EMPTY; }),
      "Mismatch in columns to concatenate.",
      cudf::data_type_error);
    return;
  }
  CUDF_EXPECTS(cudf::all_have_same_types(cols.begin(), cols.end()),
               "Type mismatch in columns to concatenate.",
               cudf::data_type_error);

  // traverse children
  cudf::type_dispatcher(cols.front().type(), traverse_children{}, cols, stream);
}

}  // anonymous namespace

// Concatenates the elements from a vector of column_views
std::unique_ptr<column> concatenate(std::span<column_view const> columns_to_concat,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not columns_to_concat.empty(), "Unexpected empty list of columns to concatenate.");

  auto const views_as_host_span =
    host_span<column_view const>{columns_to_concat.data(), columns_to_concat.size()};

  // verify all types match and that we won't overflow size_type in output size
  bounds_and_type_check(views_as_host_span, stream);

  if (std::all_of(columns_to_concat.begin(), columns_to_concat.end(), [](column_view const& c) {
        return c.is_empty();
      })) {
    return empty_like(columns_to_concat.front());
  }

  // For empty columns, we can just create an EMPTY column of the appropriate length.
  if (columns_to_concat.front().type().id() == cudf::type_id::EMPTY) {
    auto length = std::accumulate(
      columns_to_concat.begin(), columns_to_concat.end(), 0, [](auto a, auto const& b) {
        return a + b.size();
      });
    return std::make_unique<column>(
      data_type(type_id::EMPTY), length, rmm::device_buffer{}, rmm::device_buffer{}, length);
  }
  return type_dispatcher<dispatch_storage_type>(
    columns_to_concat.front().type(), concatenate_dispatch{views_as_host_span, stream, mr});
}

std::unique_ptr<table> concatenate(std::span<table_view const> tables_to_concat,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  if (tables_to_concat.empty()) { return std::make_unique<table>(); }

  table_view const first_table = tables_to_concat.front();
  CUDF_EXPECTS(std::all_of(tables_to_concat.begin(),
                           tables_to_concat.end(),
                           [&first_table](auto const& t) {
                             return t.num_columns() == first_table.num_columns();
                           }),
               "Mismatch in table columns to concatenate.");

  // Zero-column tables carry only a row count; concatenation sums their rows.
  if (first_table.num_columns() == 0) {
    auto const total_rows = std::accumulate(
      tables_to_concat.begin(),
      tables_to_concat.end(),
      std::size_t{0},
      [](std::size_t acc, auto const& t) { return acc + static_cast<std::size_t>(t.num_rows()); });
    CUDF_EXPECTS(total_rows <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
                 "Total number of rows exceeds the column size limit",
                 std::overflow_error);
    return std::make_unique<table>(std::vector<std::unique_ptr<column>>{},
                                   static_cast<size_type>(total_rows));
  }

  std::vector<std::unique_ptr<column>> concat_columns;
  for (size_type i = 0; i < first_table.num_columns(); ++i) {
    std::vector<column_view> cols;
    std::transform(tables_to_concat.begin(),
                   tables_to_concat.end(),
                   std::back_inserter(cols),
                   [i](auto const& t) { return t.column(i); });

    // verify all types match and that we won't overflow size_type in output size
    bounds_and_type_check(cols, stream);
    concat_columns.emplace_back(detail::concatenate(cols, stream, mr));
  }
  return std::make_unique<table>(std::move(concat_columns));
}

rmm::device_buffer concatenate_masks(std::span<column_view const> views,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  bool const has_nulls =
    std::any_of(views.begin(), views.end(), [](column_view const col) { return col.has_nulls(); });
  if (has_nulls) {
    size_type const total_element_count =
      std::accumulate(views.begin(), views.end(), 0, [](auto accumulator, auto const& v) {
        return accumulator + v.size();
      });

    rmm::device_buffer null_mask =
      cudf::detail::create_null_mask(total_element_count, mask_state::UNINITIALIZED, stream, mr);

    detail::concatenate_masks(host_span<column_view const>{views.data(), views.size()},
                              static_cast<bitmask_type*>(null_mask.data()),
                              stream);

    return null_mask;
  }
  // no nulls, so return an empty device buffer
  return rmm::device_buffer{0, stream, mr};
}

}  // namespace detail

rmm::device_buffer concatenate_masks(std::span<column_view const> views,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_masks(views, stream, mr);
}

// Concatenates the elements from a vector of column_views
std::unique_ptr<column> concatenate(std::span<column_view const> columns_to_concat,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(columns_to_concat, stream, mr);
}

std::unique_ptr<table> concatenate(std::span<table_view const> tables_to_concat,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(tables_to_concat, stream, mr);
}

}  // namespace cudf
