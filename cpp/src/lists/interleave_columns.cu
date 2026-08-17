/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
/**
 * @brief Generate list offsets and list validities for the output lists column from the table_view
 * of the input lists columns.
 */
std::pair<std::unique_ptr<column>, rmm::device_uvector<int8_t>>
generate_list_offsets_and_validities(table_view const& input,
                                     bool has_null_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const num_cols         = input.num_columns();
  auto const num_rows         = input.num_rows();
  auto const num_output_lists = num_rows * num_cols;
  auto const table_dv_ptr     = table_device_view::create(input, stream);

  // The output offsets column.
  auto list_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_output_lists + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_offsets = list_offsets->mutable_view().template begin<int32_t>();

  // The array of int8_t to store validities for list elements.
  auto validities = rmm::device_uvector<int8_t>(has_null_mask ? num_output_lists : 0, stream);

  // Compute list sizes and validities.
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{num_output_lists},
    d_offsets,
    cuda::proclaim_return_type<size_type>([num_cols,
                                           table_dv     = *table_dv_ptr,
                                           d_validities = validities.begin(),
                                           has_null_mask] __device__(size_type const idx) {
      auto const col_id     = idx % num_cols;
      auto const list_id    = idx / num_cols;
      auto const& lists_col = table_dv.column(col_id);
      if (has_null_mask) { d_validities[idx] = static_cast<int8_t>(lists_col.is_valid(list_id)); }
      auto const list_offsets =
        lists_col.child(lists_column_view::offsets_column_index).template data<int32_t>() +
        lists_col.offset();
      return list_offsets[list_id + 1] - list_offsets[list_id];
    }));

  // Compute offsets from sizes.
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         d_offsets,
                         d_offsets + num_output_lists + 1,
                         d_offsets);

  return {std::move(list_offsets), std::move(validities)};
}

/**
 * @brief Concatenate all input columns into one column and gather its rows to generate an output
 * column that is the result of interleaving the input columns.
 */
std::unique_ptr<column> concatenate_and_gather_lists(host_span<column_view const> columns_to_concat,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  // Concatenate all columns into a single (temporary) column.
  auto const concatenated_col =
    cudf::detail::concatenate(columns_to_concat, stream, cudf::get_current_device_resource_ref());

  // The number of input columns is known to be non-zero thus it's safe to call `front()` here.
  auto const num_cols       = columns_to_concat.size();
  auto const num_input_rows = columns_to_concat.front().size();

  // Generate the gather map that interleaves the input columns.
  auto const iter_gather = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<std::size_t>([num_cols, num_input_rows] __device__(auto const idx) {
      auto const source_col_idx = idx % num_cols;
      auto const source_row_idx = idx / num_cols;
      return source_col_idx * num_input_rows + source_row_idx;
    }));

  // The gather API should be able to handle any data type for the input columns.
  auto result = cudf::detail::gather(table_view{{concatenated_col->view()}},
                                     iter_gather,
                                     iter_gather + concatenated_col->size(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

// Error case when no other overload or specialization is available
template <typename T, typename Enable = void>
struct interleave_list_entries_impl {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Called `interleave_list_entries_fn()` on non-supported types.");
  }
};

/**
 * @brief Interleave array of string_index_pair objects for a list of strings
 *
 * Each thread processes the strings for the corresponding list row
 */
struct compute_string_sizes_and_interleave_lists_fn {
  table_device_view const table_dv;

  // Store list offsets of the output lists column.
  int32_t const* const dst_list_offsets;

  using string_index_pair = cudf::strings::detail::string_index_pair;
  string_index_pair* indices;  // output

  // thread per list row per column
  __device__ void operator()(size_type const idx)
  {
    auto const num_cols = table_dv.num_columns();
    auto const col_id   = idx % num_cols;
    auto const list_id  = idx / num_cols;

    auto const& lists_col = table_dv.column(col_id);
    if (lists_col.is_null(list_id)) { return; }

    auto const list_offsets =
      lists_col.child(lists_column_view::offsets_column_index).template data<int32_t>() +
      lists_col.offset();
    auto const& str_col = lists_col.child(lists_column_view::child_column_index);

    // The range of indices of the strings within the source list.
    auto const start_str_idx = list_offsets[list_id];
    auto const end_str_idx   = list_offsets[list_id + 1];

    // In case of empty list (i.e. it doesn't contain any string element), we just ignore it because
    // there will not be anything to store for that list in the child column.
    if (start_str_idx == end_str_idx) { return; }

    // read_idx and write_idx are indices of string elements.
    size_type write_idx = dst_list_offsets[idx];

    for (auto read_idx = start_str_idx; read_idx < end_str_idx; ++read_idx, ++write_idx) {
      if (str_col.is_null(read_idx)) {
        indices[write_idx] = string_index_pair{nullptr, 0};
        continue;
      }
      auto const d_str   = str_col.element<string_view>(read_idx);
      indices[write_idx] = d_str.empty() ? string_index_pair{"", 0}
                                         : string_index_pair{d_str.data(), d_str.size_bytes()};
    }
  }
};

template <typename T>
struct interleave_list_entries_impl<T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>> {
  std::unique_ptr<column> operator()(table_view const& input,
                                     column_view const& output_list_offsets,
                                     size_type num_output_lists,
                                     size_type num_output_entries,
                                     bool,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const noexcept
  {
    auto const table_dv_ptr   = table_device_view::create(input, stream);
    auto const d_list_offsets = output_list_offsets.template begin<int32_t>();

    rmm::device_uvector<cudf::strings::detail::string_index_pair> indices(num_output_entries,
                                                                          stream);
    auto comp_fn =
      compute_string_sizes_and_interleave_lists_fn{*table_dv_ptr, d_list_offsets, indices.data()};
    thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<size_type>{0},
                       num_output_lists,
                       comp_fn);
    return cudf::strings::detail::make_strings_column(indices.begin(), indices.end(), stream, mr);
  }
};

template <typename T>
struct interleave_list_entries_impl<T, std::enable_if_t<cudf::is_fixed_width<T>()>> {
  std::unique_ptr<column> operator()(table_view const& input,
                                     column_view const& output_list_offsets,
                                     size_type num_output_lists,
                                     size_type num_output_entries,
                                     bool data_has_null_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const noexcept
  {
    auto const num_cols     = input.num_columns();
    auto const table_dv_ptr = table_device_view::create(input, stream);

    // The output child column.
    auto output        = cudf::detail::allocate_like(lists_column_view(*input.begin()).child(),
                                              num_output_entries,
                                              mask_allocation_policy::NEVER,
                                              stream,
                                              mr);
    auto output_dv_ptr = mutable_column_device_view::create(*output, stream);

    // The array of int8_t to store entry validities.
    auto validities =
      rmm::device_uvector<int8_t>(data_has_null_mask ? num_output_entries : 0, stream);

    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>{0},
      num_output_lists,
      [num_cols,
       table_dv     = *table_dv_ptr,
       d_validities = validities.begin(),
       d_offsets    = output_list_offsets.template begin<int32_t>(),
       d_output     = output_dv_ptr->template begin<T>(),
       data_has_null_mask] __device__(size_type const idx) {
        auto const col_id     = idx % num_cols;
        auto const list_id    = idx / num_cols;
        auto const& lists_col = table_dv.column(col_id);
        auto const list_offsets =
          lists_col.child(lists_column_view::offsets_column_index).template data<int32_t>() +
          lists_col.offset();
        auto const& data_col = lists_col.child(lists_column_view::child_column_index);

        // The range of indices of the entries within the source list.
        auto const start_idx = list_offsets[list_id];
        auto const end_idx   = list_offsets[list_id + 1];

        auto const write_start = d_offsets[idx];

        // Fill the validities array if necessary.
        if (data_has_null_mask) {
          for (auto read_idx = start_idx, write_idx = write_start; read_idx < end_idx;
               ++read_idx, ++write_idx) {
            d_validities[write_idx] = static_cast<int8_t>(data_col.is_valid(read_idx));
          }
        }

        // Do a copy for the entire list entries.
        auto const input_ptr =
          reinterpret_cast<char const*>(data_col.template data<T>() + start_idx);
        auto const output_ptr = reinterpret_cast<char*>(&d_output[write_start]);
        thrust::copy(
          thrust::seq, input_ptr, input_ptr + sizeof(T) * (end_idx - start_idx), output_ptr);
      });

    if (data_has_null_mask) {
      auto [null_mask, null_count] = cudf::detail::valid_if(
        validities.begin(), validities.end(), cuda::std::identity{}, stream, mr);
      if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
    }

    return output;
  }
};

/**
 * @brief Struct used in type_dispatcher to interleave list entries of the input lists columns and
 * output the results into a destination column.
 */
struct interleave_list_entries_fn {
  template <class T>
  std::unique_ptr<column> operator()(table_view const& input,
                                     column_view const& output_list_offsets,
                                     size_type num_output_lists,
                                     size_type num_output_entries,
                                     bool data_has_null_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    return interleave_list_entries_impl<T>{}(input,
                                             output_list_offsets,
                                             num_output_lists,
                                             num_output_entries,
                                             data_has_null_mask,
                                             stream,
                                             mr);
  }
};

}  // anonymous namespace

/**
 * @copydoc cudf::lists::detail::interleave_columns
 *
 */
std::unique_ptr<column> interleave_columns(table_view const& input,
                                           bool has_null_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const entry_type = lists_column_view(*input.begin()).child().type();
  for (auto const& col : input) {
    CUDF_EXPECTS(col.type().id() == type_id::LIST,
                 "All columns of the input table must be of lists column type.");

    auto const child_col = lists_column_view(col).child();
    CUDF_EXPECTS(entry_type == child_col.type(),
                 "The types of entries in the input columns must be the same.");
  }

  if (input.num_rows() == 0) { return cudf::empty_like(input.column(0)); }
  if (input.num_columns() == 1) { return std::make_unique<column>(*(input.begin()), stream, mr); }

  // For nested types, we rely on the `concatenate_and_gather` method, which costs more memory due
  // to concatenation of the input columns into a temporary column. For non-nested types, we can
  // directly interleave the input columns into the output column for better efficiency.
  if (cudf::is_nested(entry_type)) {
    auto const input_columns = std::vector<column_view>(input.begin(), input.end());
    return concatenate_and_gather_lists(host_span<column_view const>{input_columns}, stream, mr);
  }

  // Generate offsets of the output lists column.
  auto [list_offsets, list_validities] =
    generate_list_offsets_and_validities(input, has_null_mask, stream, mr);
  auto const offsets_view = list_offsets->view();

  // Copy entries from the input lists columns to the output lists column - this needed to be
  // specialized for different types.
  auto const num_output_lists = input.num_rows() * input.num_columns();
  auto const num_output_entries =
    cudf::detail::get_value<int32_t>(offsets_view, num_output_lists, stream);
  auto const data_has_null_mask =
    std::any_of(std::cbegin(input), std::cend(input), [](auto const& col) {
      return col.child(lists_column_view::child_column_index).nullable();
    });
  auto list_entries = type_dispatcher<dispatch_storage_type>(entry_type,
                                                             interleave_list_entries_fn{},
                                                             input,
                                                             offsets_view,
                                                             num_output_lists,
                                                             num_output_entries,
                                                             data_has_null_mask,
                                                             stream,
                                                             mr);

  if (not has_null_mask) {
    return make_lists_column(
      num_output_lists, std::move(list_offsets), std::move(list_entries), 0, rmm::device_buffer{});
  }

  auto [null_mask, null_count] = cudf::detail::valid_if(
    list_validities.begin(), list_validities.end(), cuda::std::identity{}, stream, mr);
  return make_lists_column(num_output_lists,
                           std::move(list_offsets),
                           std::move(list_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{});
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
