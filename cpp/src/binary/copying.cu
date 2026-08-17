/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/binary/binary_column_factories.hpp>
#include <cudf/binary/detail/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform.h>

namespace cudf::binary::detail {
namespace {

int64_t get_offset(column_view const& offsets, size_type index, rmm::cuda_stream_view stream)
{
  if (offsets.type().id() == type_id::INT32) {
    return cudf::detail::get_value<int32_t>(offsets, index, stream);
  }
  CUDF_EXPECTS(offsets.type().id() == type_id::INT64,
               "Binary offsets must have type INT32 or INT64");
  return cudf::detail::get_value<int64_t>(offsets, index, stream);
}

}  // namespace

std::unique_ptr<column> copy_slice(binary_column_view const& input,
                                   size_type start,
                                   size_type end,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(start >= 0 and start <= end and end <= input.size(),
               "Invalid BINARY slice range");
  if (start == end) { return make_empty_binary_column(); }

  auto const row_count      = end - start;
  auto const offsets_offset = input.offset() + start;
  auto offsets_column       = std::make_unique<column>(
    cudf::detail::slice(
      input.offsets(), {offsets_offset, offsets_offset + row_count + 1}, stream)
      .front(),
    stream,
    mr);

  auto const first_offset = get_offset(offsets_column->view(), 0, stream);
  if (first_offset != 0) {
    auto output_offsets =
      cudf::detail::offsetalator_factory::make_output_iterator(offsets_column->mutable_view());
    auto input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), offsets_offset);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input_offsets,
                      input_offsets + offsets_column->size(),
                      output_offsets,
                      cuda::proclaim_return_type<int64_t>(
                        [first_offset] __device__(auto offset) { return offset - first_offset; }));
  }

  auto const payload_size =
    static_cast<std::size_t>(get_offset(offsets_column->view(), row_count, stream));
  auto payload = rmm::device_buffer{input.bytes_begin() + first_offset, payload_size, stream, mr};

  auto null_mask = cudf::detail::copy_bitmask(
    input.null_mask(), offsets_offset, offsets_offset + row_count, stream, mr);
  auto const null_count = cudf::detail::null_count(
    static_cast<bitmask_type const*>(null_mask.data()), 0, row_count, stream);

  return make_binary_column(row_count,
                            std::move(offsets_column),
                            std::move(payload),
                            null_count,
                            std::move(null_mask));
}

}  // namespace cudf::binary::detail
