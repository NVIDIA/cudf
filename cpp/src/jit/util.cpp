/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace jit {
struct get_data_ptr_functor {
  /**
   * @brief Gets the data pointer from a column_view
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  void const* operator()(column_view const& view)
  {
    return static_cast<void const*>(view.template data<T>());
  }

  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>())>
  void const* operator()(column_view const& view)
  {
    CUDF_FAIL("Invalid data type for JIT operation");
  }

  /**
   * @brief Gets the data pointer from a scalar
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  void const* operator()(scalar const& s)
  {
    using ScalarType = scalar_type_t<T>;
    auto s1          = static_cast<ScalarType const*>(&s);
    return static_cast<void const*>(s1->data());
  }

  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>())>
  void const* operator()(scalar const& s)
  {
    CUDF_FAIL("Invalid data type for JIT operation");
  }
};

void const* get_data_ptr(column_view const& view)
{
  return type_dispatcher<dispatch_storage_type>(view.type(), get_data_ptr_functor{}, view);
}

void const* get_data_ptr(scalar const& s)
{
  return type_dispatcher<dispatch_storage_type>(s.type(), get_data_ptr_functor{}, s);
}

data_type physical_type_of(data_type type)
{
  using enum type_id;
  switch (type.id()) {
    case EMPTY: return data_type{EMPTY};
    case BOOL8:
    case INT8:
    case UINT8: return data_type{UINT8};
    case INT16:
    case UINT16: return data_type{UINT16};
    case INT32:
    case UINT32:
    case FLOAT32: return data_type{UINT32};
    case INT64:
    case UINT64:
    case FLOAT64: return data_type{UINT64};
    case TIMESTAMP_DAYS: return data_type{type_to_id<timestamp_D::rep>()};
    case TIMESTAMP_SECONDS: return data_type{type_to_id<timestamp_s::rep>()};
    case TIMESTAMP_MILLISECONDS: return data_type{type_to_id<timestamp_ms::rep>()};
    case TIMESTAMP_MICROSECONDS: return data_type{type_to_id<timestamp_us::rep>()};
    case TIMESTAMP_NANOSECONDS: return data_type{type_to_id<timestamp_ns::rep>()};
    case DURATION_DAYS: return data_type{type_to_id<duration_D::rep>()};
    case DURATION_SECONDS: return data_type{type_to_id<duration_s::rep>()};
    case DURATION_MILLISECONDS: return data_type{type_to_id<duration_ms::rep>()};
    case DURATION_MICROSECONDS: return data_type{type_to_id<duration_us::rep>()};
    case DURATION_NANOSECONDS: return data_type{type_to_id<duration_ns::rep>()};
    case DICTIONARY32:
    case STRING:
    case LIST:
    case DECIMAL32:
    case DECIMAL64:
    case DECIMAL128:
    case STRUCT:
    default: return type;
  }
}

}  // namespace jit
}  // namespace cudf
