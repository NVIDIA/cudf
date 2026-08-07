/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utilities.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <nanoarrow/nanoarrow.h>

#include <stdexcept>

namespace cudf {
namespace detail {
data_type arrow_to_cudf_type(ArrowSchemaView const* arrow_view)
{
  switch (arrow_view->type) {
    case NANOARROW_TYPE_NA: return data_type(type_id::EMPTY);
    case NANOARROW_TYPE_BOOL: return data_type(type_id::BOOL8);
    case NANOARROW_TYPE_INT8: return data_type(type_id::INT8);
    case NANOARROW_TYPE_INT16: return data_type(type_id::INT16);
    case NANOARROW_TYPE_INT32: return data_type(type_id::INT32);
    case NANOARROW_TYPE_INT64: return data_type(type_id::INT64);
    case NANOARROW_TYPE_UINT8: return data_type(type_id::UINT8);
    case NANOARROW_TYPE_UINT16: return data_type(type_id::UINT16);
    case NANOARROW_TYPE_UINT32: return data_type(type_id::UINT32);
    case NANOARROW_TYPE_UINT64: return data_type(type_id::UINT64);
    case NANOARROW_TYPE_FLOAT: return data_type(type_id::FLOAT32);
    case NANOARROW_TYPE_DOUBLE: return data_type(type_id::FLOAT64);
    case NANOARROW_TYPE_DATE32: return data_type(type_id::TIMESTAMP_DAYS);
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_STRING_VIEW:
    case NANOARROW_TYPE_LARGE_STRING: return data_type(type_id::STRING);
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_FIXED_SIZE_LIST: return data_type(type_id::LIST);
    case NANOARROW_TYPE_DICTIONARY: return data_type(type_id::DICTIONARY32);
    case NANOARROW_TYPE_STRUCT: return data_type(type_id::STRUCT);
    case NANOARROW_TYPE_TIMESTAMP: {
      switch (arrow_view->time_unit) {
        case NANOARROW_TIME_UNIT_SECOND: return data_type(type_id::TIMESTAMP_SECONDS);
        case NANOARROW_TIME_UNIT_MILLI: return data_type(type_id::TIMESTAMP_MILLISECONDS);
        case NANOARROW_TIME_UNIT_MICRO: return data_type(type_id::TIMESTAMP_MICROSECONDS);
        case NANOARROW_TIME_UNIT_NANO: return data_type(type_id::TIMESTAMP_NANOSECONDS);
        default: CUDF_FAIL("Unsupported timestamp unit in arrow", cudf::data_type_error);
      }
    }
    case NANOARROW_TYPE_DURATION: {
      switch (arrow_view->time_unit) {
        case NANOARROW_TIME_UNIT_SECOND: return data_type(type_id::DURATION_SECONDS);
        case NANOARROW_TIME_UNIT_MILLI: return data_type(type_id::DURATION_MILLISECONDS);
        case NANOARROW_TIME_UNIT_MICRO: return data_type(type_id::DURATION_MICROSECONDS);
        case NANOARROW_TIME_UNIT_NANO: return data_type(type_id::DURATION_NANOSECONDS);
        default: CUDF_FAIL("Unsupported duration unit in arrow", cudf::data_type_error);
      }
    }
    case NANOARROW_TYPE_DECIMAL32: return data_type{type_id::DECIMAL32, -arrow_view->decimal_scale};
    case NANOARROW_TYPE_DECIMAL64: return data_type{type_id::DECIMAL64, -arrow_view->decimal_scale};
    case NANOARROW_TYPE_DECIMAL128:
      return data_type{type_id::DECIMAL128, -arrow_view->decimal_scale};
    default: CUDF_FAIL("Unsupported type_id conversion to cudf", cudf::data_type_error);
  }
}

bool is_fixed_size_list(ArrowSchemaView const* arrow_view)
{
  return arrow_view->type == NANOARROW_TYPE_FIXED_SIZE_LIST;
}

size_type fixed_size_list_width(ArrowSchemaView const* arrow_view)
{
  CUDF_EXPECTS(
    is_fixed_size_list(arrow_view), "Expected a fixed-size-list schema", cudf::data_type_error);
  CUDF_EXPECTS(arrow_view->fixed_size >= 0,
               "fixed-size-list width must be non-negative",
               std::invalid_argument);
  return static_cast<size_type>(arrow_view->fixed_size);
}

ArrowType id_to_arrow_type(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::BOOL8: return NANOARROW_TYPE_BOOL;
    case cudf::type_id::INT8: return NANOARROW_TYPE_INT8;
    case cudf::type_id::INT16: return NANOARROW_TYPE_INT16;
    case cudf::type_id::INT32: return NANOARROW_TYPE_INT32;
    case cudf::type_id::INT64: return NANOARROW_TYPE_INT64;
    case cudf::type_id::UINT8: return NANOARROW_TYPE_UINT8;
    case cudf::type_id::UINT16: return NANOARROW_TYPE_UINT16;
    case cudf::type_id::UINT32: return NANOARROW_TYPE_UINT32;
    case cudf::type_id::UINT64: return NANOARROW_TYPE_UINT64;
    case cudf::type_id::FLOAT32: return NANOARROW_TYPE_FLOAT;
    case cudf::type_id::FLOAT64: return NANOARROW_TYPE_DOUBLE;
    case cudf::type_id::TIMESTAMP_DAYS: return NANOARROW_TYPE_DATE32;
    case cudf::type_id::DECIMAL32: return NANOARROW_TYPE_DECIMAL32;
    case cudf::type_id::DECIMAL64: return NANOARROW_TYPE_DECIMAL64;
    case cudf::type_id::DECIMAL128: return NANOARROW_TYPE_DECIMAL128;
    default: CUDF_FAIL("Unsupported type_id conversion to arrow type", cudf::data_type_error);
  }
}

ArrowType id_to_arrow_storage_type(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::TIMESTAMP_DAYS: return NANOARROW_TYPE_INT32;
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return NANOARROW_TYPE_INT64;
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS: return NANOARROW_TYPE_INT64;
    default: return id_to_arrow_type(id);
  }
}

int initialize_array(ArrowArray* arr, ArrowType storage_type, cudf::column_view column)
{
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(arr, storage_type));
  arr->length     = column.size();
  arr->null_count = column.null_count();
  return NANOARROW_OK;
}

}  // namespace detail
}  // namespace cudf
