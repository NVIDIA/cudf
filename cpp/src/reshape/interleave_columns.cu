/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/reshape.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/detail/interleave_columns.hpp>
#include <cudf/reshape.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/for_each.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {
// Error case when no other overload or specialization is available
template <typename T, typename Enable = void>
struct interleave_columns_impl {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type in `interleave_columns`.");
  }
};

struct interleave_columns_functor {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(table_view const& input,
                                           bool create_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    return interleave_columns_impl<T>{}(input, create_mask, stream, mr);
  }
};

template <typename T>
struct interleave_columns_impl<T, std::enable_if_t<std::is_same_v<T, cudf::list_view>>> {
  std::unique_ptr<column> operator()(table_view const& lists_columns,
                                     bool create_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return lists::detail::interleave_columns(lists_columns, create_mask, stream, mr);
  }
};

template <typename T>
struct interleave_columns_impl<T, std::enable_if_t<std::is_same_v<T, cudf::struct_view>>> {
  std::unique_ptr<cudf::column> operator()(table_view const& structs_columns,
                                           bool create_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    // We can safely call `column(0)` as the number of columns is known to be non zero.
    auto const num_children = structs_columns.column(0).num_children();
    CUDF_EXPECTS(
      std::all_of(structs_columns.begin(),
                  structs_columns.end(),
                  [num_children](auto const& col) { return col.num_children() == num_children; }),
      "Number of children of the input structs columns must be the same");

    auto const num_columns = structs_columns.num_columns();
    auto const num_rows    = structs_columns.num_rows();
    auto const output_size = num_columns * num_rows;

    // Interleave the children of the structs columns.
    std::vector<std::unique_ptr<cudf::column>> output_struct_members;
    for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
      // Collect children columns from the input structs columns at index `child_idx`.
      auto const child_iter = cuda::transform_iterator(
        structs_columns.begin(), [&stream = stream, child_idx](auto const& col) {
          return structs_column_view(col).get_sliced_child(child_idx, stream);
        });
      auto children = std::vector<column_view>(child_iter, child_iter + num_columns);

      auto const child_type = children.front().type();
      CUDF_EXPECTS(
        std::all_of(children.cbegin(),
                    children.cend(),
                    [child_type](auto const& col) { return child_type == col.type(); }),
        "Children of the input structs columns at the same child index must have the same type");

      auto const children_nullable = std::any_of(
        children.cbegin(), children.cend(), [](auto const& col) { return col.nullable(); });
      output_struct_members.emplace_back(
        type_dispatcher<dispatch_storage_type>(child_type,
                                               interleave_columns_functor{},
                                               table_view{std::move(children)},
                                               children_nullable,
                                               stream,
                                               mr));
    }

    auto const create_mask_fn = [&] {
      auto const input_dv_ptr = table_device_view::create(structs_columns, stream);
      auto const validity_fn  = [input_dv = *input_dv_ptr, num_columns] __device__(auto const idx) {
        return input_dv.column(idx % num_columns).is_valid(idx / num_columns);
      };
      return cudf::detail::valid_if(cuda::counting_iterator<size_type>{0},
                                    cuda::counting_iterator<size_type>{output_size},
                                    validity_fn,
                                    stream,
                                    mr);
    };

    // Only create null mask if at least one input structs column is nullable.
    auto [null_mask, null_count] =
      create_mask ? create_mask_fn() : std::pair{rmm::device_buffer{0, stream, mr}, size_type{0}};
    return make_structs_column(
      output_size, std::move(output_struct_members), null_count, std::move(null_mask), stream, mr);
  }
};

struct interleave_strings_fn {
  using string_index_pair = cudf::strings::detail::string_index_pair;
  table_device_view d_table;

  __device__ string_index_pair operator()(size_type idx)
  {
    auto const num_columns    = d_table.num_columns();
    auto const source_col_idx = idx % num_columns;
    auto const source_row_idx = idx / num_columns;
    auto const col            = d_table.column(source_col_idx);
    if (col.is_null(source_row_idx)) { return string_index_pair{nullptr, 0}; }
    auto const d_str = col.element<string_view>(source_row_idx);
    // ensures an empty string is not identified as a null row
    return !d_str.empty() ? string_index_pair{d_str.data(), d_str.size_bytes()}
                          : string_index_pair{"", 0};
  }
};

template <typename T>
struct interleave_columns_impl<T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>> {
  std::unique_ptr<cudf::column> operator()(table_view const& strings_columns,
                                           bool,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto num_columns = strings_columns.num_columns();
    if (num_columns == 1) {  // Single strings column returns a copy
      return std::make_unique<column>(*(strings_columns.begin()), stream, mr);
    }

    auto strings_count = strings_columns.num_rows();
    if (strings_count == 0) {  // All columns have 0 rows
      return make_empty_column(type_id::STRING);
    }

    // Create device views from the strings columns.
    auto d_table     = table_device_view::create(strings_columns, stream);
    auto num_strings = num_columns * strings_count;

    rmm::device_uvector<cudf::strings::detail::string_index_pair> indices(num_strings, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{num_strings},
                      indices.begin(),
                      interleave_strings_fn{*d_table});

    return cudf::strings::detail::make_strings_column(indices.begin(), indices.end(), stream, mr);
  }
};

template <typename T>
struct interleave_columns_impl<T, std::enable_if_t<cudf::is_fixed_width<T>()>> {
  std::unique_ptr<cudf::column> operator()(table_view const& input,
                                           bool create_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto arch_column = input.column(0);
    auto output_size = input.num_columns() * input.num_rows();
    auto output =
      detail::allocate_like(arch_column, output_size, mask_allocation_policy::NEVER, stream, mr);
    auto device_input  = table_device_view::create(input, stream);
    auto device_output = mutable_column_device_view::create(*output, stream);
    auto index_begin   = cuda::counting_iterator<size_type>{0};
    auto index_end     = cuda::counting_iterator<size_type>{output_size};

    auto func_value = cuda::proclaim_return_type<T>(
      [input = *device_input, divisor = input.num_columns()] __device__(size_type idx) {
        return input.column(idx % divisor).element<T>(idx / divisor);
      });

    if (not create_mask) {
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        index_begin,
                        index_end,
                        device_output->begin<T>(),
                        func_value);

      return output;
    }

    auto func_validity = [input   = *device_input,
                          divisor = input.num_columns()] __device__(size_type idx) {
      return input.column(idx % divisor).is_valid(idx / divisor);
    };

    thrust::transform_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         index_begin,
                         index_end,
                         device_output->begin<T>(),
                         func_value,
                         func_validity);

    auto [mask, null_count] = valid_if(index_begin, index_end, func_validity, stream, mr);

    output->set_null_mask(std::move(mask), null_count);

    return output;
  }
};

}  // anonymous namespace

std::unique_ptr<column> interleave_columns(table_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.num_columns() > 0, "input must have at least one column to determine dtype.");

  auto const dtype = input.column(0).type();
  CUDF_EXPECTS(std::all_of(std::cbegin(input),
                           std::cend(input),
                           [dtype](auto const& col) { return dtype == col.type(); }),
               "Input columns must have the same type");

  auto const output_needs_mask = std::any_of(
    std::cbegin(input), std::cend(input), [](auto const& col) { return col.nullable(); });

  return type_dispatcher<dispatch_storage_type>(
    dtype, detail::interleave_columns_functor{}, input, output_needs_mask, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> interleave_columns(table_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::interleave_columns(input, stream, mr);
}

}  // namespace cudf
