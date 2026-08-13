/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {
namespace {
struct size_of_helper {
  cudf::data_type type;
  template <typename T>
  constexpr int operator()() const
    requires(not is_fixed_width<T>())
  {
    CUDF_FAIL("Invalid, non fixed-width element type.");
    return 0;
  }

  template <typename T>
  constexpr int operator()() const noexcept
    requires(is_fixed_width<T>() && not is_fixed_point<T>())
  {
    return sizeof(T);
  }

  template <typename T>
  constexpr int operator()() const noexcept
    requires(is_fixed_point<T>())
  {
    // Only want the sizeof fixed_point::Rep as fixed_point::scale is stored in data_type
    return sizeof(typename T::rep);
  }
};
}  // namespace

std::size_t size_of(data_type element_type)
{
  CUDF_EXPECTS(is_fixed_width(element_type), "Invalid element type.");
  return cudf::type_dispatcher(element_type, size_of_helper{element_type});
}

// Empty column of specified type
std::unique_ptr<column> make_empty_column(data_type type)
{
  CUDF_EXPECTS(type.id() == type_id::EMPTY || !cudf::is_nested(type),
               "make_empty_column is invalid to call on nested types",
               cudf::data_type_error);
  return std::make_unique<column>(type, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
}

// Empty column of specified type id
std::unique_ptr<column> make_empty_column(type_id id) { return make_empty_column(data_type{id}); }

// Allocate storage for a specified number of numeric elements
std::unique_ptr<column> make_numeric_column(data_type type,
                                            size_type size,
                                            mask_state state,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(type.id() != type_id::EMPTY && is_numeric(type),
               "Invalid, non-numeric type.",
               cudf::data_type_error);
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");

  return std::make_unique<column>(
    type,
    size,
    rmm::device_buffer{size * cudf::size_of(type), stream, mr},
    detail::create_null_mask(size, state, stream, mr),
    state == mask_state::UNINITIALIZED ? 0 : state_null_count(state, size),
    std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of numeric elements
std::unique_ptr<column> make_fixed_point_column(data_type type,
                                                size_type size,
                                                mask_state state,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_fixed_point(type), "Invalid, non-fixed_point type.", cudf::data_type_error);
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");

  return std::make_unique<column>(
    type,
    size,
    rmm::device_buffer{size * cudf::size_of(type), stream, mr},
    detail::create_null_mask(size, state, stream, mr),
    state == mask_state::UNINITIALIZED ? 0 : state_null_count(state, size),
    std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of timestamp elements
std::unique_ptr<column> make_timestamp_column(data_type type,
                                              size_type size,
                                              mask_state state,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.", cudf::data_type_error);
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");

  return std::make_unique<column>(
    type,
    size,
    rmm::device_buffer{size * cudf::size_of(type), stream, mr},
    detail::create_null_mask(size, state, stream, mr),
    state == mask_state::UNINITIALIZED ? 0 : state_null_count(state, size),
    std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of duration elements
std::unique_ptr<column> make_duration_column(data_type type,
                                             size_type size,
                                             mask_state state,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_duration(type), "Invalid, non-duration type.", cudf::data_type_error);
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");

  return std::make_unique<column>(
    type,
    size,
    rmm::device_buffer{size * cudf::size_of(type), stream, mr},
    detail::create_null_mask(size, state, stream, mr),
    state == mask_state::UNINITIALIZED ? 0 : state_null_count(state, size),
    std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of fixed width elements
std::unique_ptr<column> make_fixed_width_column(data_type type,
                                                size_type size,
                                                mask_state state,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(type.id() != type_id::EMPTY && is_fixed_width(type),
               "Invalid, non-fixed-width type.",
               cudf::data_type_error);

  // clang-format off
  if      (is_timestamp  (type)) return make_timestamp_column  (type, size, state, stream, mr);
  else if (is_duration   (type)) return make_duration_column   (type, size, state, stream, mr);
  else if (is_fixed_point(type)) return make_fixed_point_column(type, size, state, stream, mr);
  else                           return make_numeric_column    (type, size, state, stream, mr);
  // clang-format on
}

std::unique_ptr<column> make_dictionary_from_scalar(scalar const& s,
                                                    size_type size,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (size == 0) return make_empty_column(type_id::DICTIONARY32);
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");
  CUDF_EXPECTS(s.is_valid(stream), "cannot create a dictionary with a null key");
  return make_dictionary_column(
    make_column_from_scalar(s, 1, stream, mr),
    make_column_from_scalar(
      numeric_scalar<int32_t>(0, true, stream, cudf::get_current_device_resource_ref()),
      size,
      stream,
      mr),
    rmm::device_buffer{0, stream, mr},
    0);
}

}  // namespace cudf
