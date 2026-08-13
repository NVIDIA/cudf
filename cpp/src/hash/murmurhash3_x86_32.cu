/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "murmurhash3_x86_32.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_for.cuh>

#include <cstdint>
#include <memory>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

template <typename Nullate>
std::unique_ptr<column> murmurhash3_x86_32_impl(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& input,
  size_type num_rows,
  uint32_t seed,
  Nullate nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(
    data_type(type_to_id<hash_value_type>()), num_rows, mask_state::UNALLOCATED, stream, mr);

  if (num_rows == 0) { return output; }

  auto const row_hasher = cudf::detail::row::hash::row_hasher(input);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  auto const output_begin = output_view.begin<hash_value_type>();
  auto const hasher       = row_hasher.device_hasher<MurmurHash3_x86_32>(nulls, seed);
  // thrust::tabulate is slow here, see NVIDIA/cccl#9070
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    num_rows,
    [output_begin, hasher] __device__(size_type i) mutable { output_begin[i] = hasher(i); },
    stream.value()));

  return output;
}

}  // namespace

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input, stream);
  return murmurhash3_x86_32_impl(
    preprocessed_input, input.num_rows(), seed, nullate::DYNAMIC{has_nulls(input)}, stream, mr);
}

std::unique_ptr<column> murmurhash3_x86_32(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& input,
  size_type num_rows,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return murmurhash3_x86_32_impl(input, num_rows, seed, nullate::YES{}, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
