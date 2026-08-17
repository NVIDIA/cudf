/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace cudf {
namespace {
inline bool __device__ out_of_bounds(size_type size, size_type idx)
{
  return idx < 0 || idx >= size;
}

std::pair<rmm::device_buffer, size_type> create_null_mask(column_device_view const& input,
                                                          size_type offset,
                                                          scalar const& fill_value,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  auto const size = input.size();
  auto func_validity =
    [size, offset, fill = fill_value.validity_data(), input] __device__(size_type idx) {
      auto src_idx = idx - offset;
      return out_of_bounds(size, src_idx) ? *fill : input.is_valid(src_idx);
    };
  return detail::valid_if(cuda::counting_iterator<size_type>{0},
                          cuda::counting_iterator<size_type>{size},
                          func_validity,
                          stream,
                          mr);
}

struct shift_functor {
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(not cudf::is_fixed_width<T>() and not std::is_same_v<cudf::string_view, T>)
  {
    CUDF_FAIL("shift only supports fixed-width or string types.", cudf::data_type_error);
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(column_view const& input,
                                     size_type offset,
                                     scalar const& fill_value,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(std::is_same_v<cudf::string_view, T>)
  {
    auto output = cudf::strings::detail::shift(
      cudf::strings_column_view(input), offset, fill_value, stream, mr);

    if (input.nullable() || not fill_value.is_valid(stream)) {
      auto const d_input           = column_device_view::create(input, stream);
      auto [null_mask, null_count] = create_null_mask(*d_input, offset, fill_value, stream, mr);
      output->set_null_mask(std::move(null_mask), null_count);
    }

    return output;
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     size_type offset,
                                     scalar const& fill_value,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_width<T>())
  {
    using ScalarType = cudf::scalar_type_t<T>;
    auto& scalar     = static_cast<ScalarType const&>(fill_value);

    auto device_input = column_device_view::create(input, stream);
    auto output =
      detail::allocate_like(input, input.size(), mask_allocation_policy::NEVER, stream, mr);
    auto device_output = mutable_column_device_view::create(*output, stream);

    auto const scalar_is_valid = scalar.is_valid(stream);

    if (input.nullable() || not scalar_is_valid) {
      auto [null_mask, null_count] =
        create_null_mask(*device_input, offset, fill_value, stream, mr);
      output->set_null_mask(std::move(null_mask), null_count);
    }

    auto const size  = input.size();
    auto index_begin = cuda::counting_iterator<size_type>{0};
    auto index_end   = cuda::counting_iterator<size_type>{size};
    auto data        = device_output->data<T>();

    // avoid assigning elements we know to be invalid.
    if (not scalar_is_valid) {
      if (std::abs(offset) > size) { return output; }
      if (offset > 0) {
        index_begin = cuda::counting_iterator<size_type>{offset};
        data        = data + offset;
      } else if (offset < 0) {
        index_end = cuda::counting_iterator<size_type>{size + offset};
      }
    }

    auto func_value =
      [size, offset, fill = scalar.data(), input = *device_input] __device__(size_type idx) {
        auto src_idx = idx - offset;
        return out_of_bounds(size, src_idx) ? *fill : input.element<T>(src_idx);
      };

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      index_begin,
                      index_end,
                      data,
                      func_value);

    return output;
  }
};

}  // anonymous namespace

namespace detail {

std::unique_ptr<column> shift(column_view const& input,
                              size_type offset,
                              scalar const& fill_value,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::have_same_types(input, fill_value),
               "shift requires each fill value type to match the corresponding column type.",
               cudf::data_type_error);

  if (input.is_empty()) { return empty_like(input); }

  return type_dispatcher<dispatch_storage_type>(
    input.type(), shift_functor{}, input, offset, fill_value, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> shift(column_view const& input,
                              size_type offset,
                              scalar const& fill_value,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::shift(input, offset, fill_value, stream, mr);
}

}  // namespace cudf
