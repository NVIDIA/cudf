/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/protobuf/kernels.cuh"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>

namespace cudf::io::protobuf::detail {

std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (num_rows == 0) { return cudf::make_empty_column(dtype); }

  switch (dtype.id()) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8:
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
      return cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    case cudf::type_id::STRING: {
      rmm::device_uvector<cudf::strings::detail::string_index_pair> pairs(num_rows, stream, mr);
      thrust::fill(rmm::exec_policy_nosync(stream, mr),
                   pairs.data(),
                   pairs.end(),
                   cudf::strings::detail::string_index_pair{nullptr, 0});
      return cudf::strings::detail::make_strings_column(pairs.data(), pairs.end(), stream, mr);
    }
    case cudf::type_id::LIST:
      return cudf::lists::detail::make_all_nulls_lists_column(
        num_rows, cudf::data_type{cudf::type_id::UINT8}, stream, mr);
    case cudf::type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> empty_children;
      auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
      return cudf::make_structs_column(
        num_rows, std::move(empty_children), num_rows, std::move(null_mask), stream, mr);
    }
    default: CUDF_FAIL("Unsupported type for null column creation");
  }
}

std::unique_ptr<cudf::column> make_empty_column_safe(cudf::data_type dtype,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  switch (dtype.id()) {
    case cudf::type_id::LIST: {
      return cudf::lists::detail::make_empty_lists_column(cudf::data_type{cudf::type_id::UINT8});
    }
    case cudf::type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> empty_children;
      return cudf::make_structs_column(
        0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
    }
    default: return cudf::make_empty_column(dtype);
  }
}

std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto offsets =
    cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(num_rows + 1, stream, mr);
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);
  auto null_mask   = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), num_rows, std::move(null_mask));
}

std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     rmm::cuda_stream_view,
                                                     rmm::device_async_resource_ref)
{
  auto offsets_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  return cudf::make_lists_column(
    0, std::move(offsets_col), std::move(element_col), 0, rmm::device_buffer{});
}

}  // namespace cudf::io::protobuf::detail
