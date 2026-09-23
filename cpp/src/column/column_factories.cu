/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <thrust/uninitialized_fill.h>

namespace cudf {

namespace {

struct column_from_scalar_dispatch {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(scalar const& value,
                                           size_type size,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
  {
    if (size == 0) return make_empty_column(value.type());
    if (!value.is_valid(stream))
      return make_fixed_width_column(value.type(), size, mask_state::ALL_NULL, stream, mr);
    auto output_column =
      make_fixed_width_column(value.type(), size, mask_state::UNALLOCATED, stream, mr);
    auto view = output_column->mutable_view();
    detail::fill_in_place(view, 0, size, value, stream);
    return output_column;
  }
};

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::string_view>(
  scalar const& value,
  size_type size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  if (size == 0) return make_empty_column(value.type());

  if (!value.is_valid(stream)) {
    return make_strings_column(
      size,
      make_column_from_scalar(
        numeric_scalar<int32_t>(0, true, stream, cudf::get_current_device_resource_ref()),
        size + 1,
        stream,
        mr),
      rmm::device_buffer{},
      size,
      cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr));
  }

  auto& ss         = static_cast<scalar_type_t<cudf::string_view> const&>(value);
  auto const d_str = ss.value(stream);  // no actual data is copied

  // fill the column with the scalar
  rmm::device_uvector<cudf::strings::detail::string_index_pair> indices(size, stream);
  auto const row_value =
    d_str.empty() ? cudf::strings::detail::string_index_pair{"", 0}
                  : cudf::strings::detail::string_index_pair{d_str.data(), d_str.size_bytes()};
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    indices.begin(),
    indices.end(),
    row_value);
  return cudf::make_strings_column(indices, stream, mr);
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::dictionary32>(
  scalar const&, size_type, cuda::stream_ref, rmm::device_async_resource_ref) const
{
  CUDF_FAIL("dictionary not supported when creating from scalar");
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::list_view>(
  scalar const& value,
  size_type size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  auto lv = static_cast<list_scalar const*>(&value);
  return lists::detail::make_lists_column_from_scalar(*lv, size, stream, mr);
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::struct_view>(
  scalar const& value,
  size_type size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(size != 0, "0-length struct column is unsupported.");
  auto& ss  = static_cast<scalar_type_t<cudf::struct_view> const&>(value);
  auto iter = cuda::make_constant_iterator(0);

  auto children =
    detail::gather(ss.view(), iter, iter + size, out_of_bounds_policy::NULLIFY, stream, mr);
  auto const is_valid = ss.is_valid(stream);
  return make_structs_column(size,
                             std::move(children->release()),
                             is_valid ? 0 : size,
                             is_valid
                               ? rmm::device_buffer{}
                               : detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr),
                             stream,
                             mr);
}

}  // anonymous namespace

std::unique_ptr<column> make_column_from_scalar(scalar const& s,
                                                size_type size,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  return type_dispatcher(s.type(), column_from_scalar_dispatch{}, s, size, stream, mr);
}

}  // namespace cudf
