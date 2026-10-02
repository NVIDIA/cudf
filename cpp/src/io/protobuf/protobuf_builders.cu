/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/protobuf/protobuf_host_helpers.hpp"

#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/fill.h>

#include <memory>
#include <utility>
#include <vector>

namespace cudf::io::protobuf::detail {

std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               cuda::stream_ref stream,
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
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  switch (dtype.id()) {
    case cudf::type_id::LIST: {
      auto offsets_col =
        std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                       1,
                                       rmm::device_buffer(sizeof(int32_t), stream, mr),
                                       cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                       0);
      CUDF_CUDA_TRY(cudaMemsetAsync(
        offsets_col->mutable_view().data<int32_t>(), 0, sizeof(int32_t), stream.get()));
      auto child_col =
        std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                       0,
                                       rmm::device_buffer{},
                                       cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                       0);
      return cudf::make_lists_column(0,
                                     std::move(offsets_col),
                                     std::move(child_col),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    }
    case cudf::type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> empty_children;
      return cudf::make_structs_column(0,
                                       std::move(empty_children),
                                       0,
                                       cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                       stream,
                                       mr);
    }
    default: return cudf::make_empty_column(dtype);
  }
}

std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
  thrust::fill(rmm::exec_policy_nosync(stream, mr), offsets.begin(), offsets.end(), 0);
  auto offsets_col = make_offsets_column(num_rows, std::move(offsets));
  auto null_mask   = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), num_rows, std::move(null_mask));
}

std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto offsets_col =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   1,
                                   rmm::device_buffer(sizeof(int32_t), stream, mr),
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                   0);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(offsets_col->mutable_view().data<int32_t>(), 0, sizeof(int32_t), stream.get()));
  return cudf::make_lists_column(0,
                                 std::move(offsets_col),
                                 std::move(element_col),
                                 0,
                                 cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace cudf::io::protobuf::detail
