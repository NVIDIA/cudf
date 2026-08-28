/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/reshape.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/for_each.h>

#include <type_traits>

namespace cudf {
namespace detail {
namespace {

// Data type of the output data column after conversion.
constexpr data_type output_type{type_id::UINT8};

template <typename T, typename Enable = void>
struct byte_list_conversion_fn {
  template <typename... Args>
  static std::unique_ptr<column> invoke(Args&&...)
  {
    CUDF_FAIL("Unsupported non-numeric and non-string column");
  }
};

struct byte_list_conversion_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     flip_endianness configuration,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
  {
    return byte_list_conversion_fn<T>::invoke(input, configuration, stream, mr);
  }
};

template <typename T>
struct byte_list_conversion_fn<T, std::enable_if_t<cudf::is_numeric<T>()>> {
  static std::unique_ptr<column> invoke(column_view const& input,
                                        flip_endianness configuration,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
  {
    if (input.size() == 0) { return cudf::lists::detail::make_empty_lists_column(output_type); }
    if (input.size() == input.null_count()) {
      return cudf::lists::detail::make_all_nulls_lists_column(
        input.size(), output_type, stream, mr);
    }

    auto const num_bytes = static_cast<size_type>(input.size() * sizeof(T));
    auto byte_column =
      make_numeric_column(output_type, num_bytes, mask_state::UNALLOCATED, stream, mr);

    auto const d_inp = reinterpret_cast<char const*>(input.data<T>());
    auto const d_out = byte_column->mutable_view().data<char>();

    if (configuration == flip_endianness::YES) {
      thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<cudf::size_type>{0},
                       cuda::counting_iterator{num_bytes},
                       [d_inp, d_out] __device__(auto index) {
                         constexpr auto mask = static_cast<size_type>(sizeof(T) - 1);
                         d_out[index]        = d_inp[index + mask - ((index & mask) << 1)];
                       });
    } else {
      thrust::copy_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     d_inp,
                     num_bytes,
                     d_out);
    }

    auto const it = cuda::make_constant_iterator(sizeof(T));
    auto offsets_column =
      std::get<0>(cudf::detail::make_offsets_child_column(it, it + input.size(), stream, mr));

    auto result = make_lists_column(input.size(),
                                    std::move(offsets_column),
                                    std::move(byte_column),
                                    input.null_count(),
                                    detail::copy_bitmask(input, stream, mr));

    // If any nulls are present, the corresponding lists must be purged so that
    // the result is sanitized.
    if (auto const result_cv = result->view();
        cudf::detail::has_nonempty_nulls(result_cv, stream)) {
      return cudf::detail::purge_nonempty_nulls(result_cv, stream, mr);
    }

    return result;
  }
};

template <typename T>
struct byte_list_conversion_fn<T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>> {
  static std::unique_ptr<column> invoke(column_view const& input,
                                        flip_endianness,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
  {
    if (input.size() == 0) { return cudf::lists::detail::make_empty_lists_column(output_type); }
    if (input.size() == input.null_count()) {
      return cudf::lists::detail::make_all_nulls_lists_column(
        input.size(), output_type, stream, mr);
    }

    auto const num_chars = strings_column_view(input).chars_size(stream);
    CUDF_EXPECTS(num_chars < static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                 "Cannot convert strings column to lists column due to size_type limit",
                 std::overflow_error);

    auto col_content = std::make_unique<column>(input, stream, mr)->release();

    auto uint8_col = std::make_unique<column>(
      output_type, num_chars, std::move(*(col_content.data)), rmm::device_buffer{}, 0);

    auto offsets_col = std::move(col_content.children[strings_column_view::offsets_column_index]);
    if (offsets_col->type().id() != type_id::INT32) {
      offsets_col = cudf::detail::cast(offsets_col->view(), data_type{type_id::INT32}, stream, mr);
    }

    auto result = make_lists_column(input.size(),
                                    std::move(offsets_col),
                                    std::move(uint8_col),
                                    input.null_count(),
                                    detail::copy_bitmask(input, stream, mr));

    // If any nulls are present, the corresponding lists must be purged so that
    // the result is sanitized.
    if (auto const result_cv = result->view();
        cudf::detail::has_nonempty_nulls(result_cv, stream)) {
      return cudf::detail::purge_nonempty_nulls(result_cv, stream, mr);
    }

    return result;
  }
};

}  // namespace

std::unique_ptr<column> byte_cast(column_view const& input,
                                  flip_endianness endian_configuration,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  return type_dispatcher(
    input.type(), byte_list_conversion_dispatcher{}, input, endian_configuration, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> byte_cast(column_view const& input,
                                  flip_endianness endian_configuration,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_cast(input, endian_configuration, stream, mr);
}

}  // namespace cudf
