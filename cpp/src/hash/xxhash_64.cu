/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/stream>

namespace cudf {
namespace hashing {
namespace detail {

using hash_value_type = uint64_t;

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  cuda::stream_ref stream,
                                  cudf::memory_resources mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  if (input.num_rows() == 0) { return output; }

  bool const nullable = has_nulls(input);
  auto const row_hasher =
    cudf::detail::row::hash::row_hasher(input, stream, cudf::get_current_device_resource_ref());
  auto output_view = output->mutable_view();

  // Compute the hash value for each row
  auto const output_begin = output_view.begin<hash_value_type>();
  auto const hasher       = row_hasher.device_hasher<XXHash_64>(nullable, seed);
  // thrust::tabulate is slow here, see NVIDIA/cccl#9070
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    input.num_rows(),
    [output_begin, hasher] __device__(size_type i) mutable { output_begin[i] = hasher(i); },
    stream.get()));

  return output;
}

std::unique_ptr<column> xxhash_64_bytes(column_view const& input,
                                        uint64_t seed,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  // Each row is a contiguous range [offsets[i], offsets[i + 1]) of bytes in `bytes`.
  // `offsets` already accounts for the parent offset; `bytes` for the child offset.
  cudf::detail::input_offsetalator offsets;
  uint8_t const* bytes = nullptr;
  if (input.type().id() == type_id::STRING) {
    if (input.size() > 0) {
      auto const scv = strings_column_view{input};
      offsets =
        cudf::detail::offsetalator_factory::make_input_iterator(scv.offsets(), scv.offset());
      bytes = reinterpret_cast<uint8_t const*>(scv.chars_begin(stream));
    }
  } else if (input.type().id() == type_id::LIST) {
    auto const lcv   = lists_column_view{input};
    auto const child = lcv.child();
    CUDF_EXPECTS(child.type().id() == type_id::INT8 || child.type().id() == type_id::UINT8,
                 "xxhash_64_bytes supports only LIST<INT8> and LIST<UINT8> list columns.",
                 cudf::data_type_error);
    if (input.size() > 0) {
      // Only the elements of the selected rows matter: `child` is not sliced with `input`.
      // Slice it only if it has nulls; slicing reads offsets back to the host.
      CUDF_EXPECTS(!child.has_nulls() || !lcv.get_sliced_child(stream).has_nulls(),
                   "xxhash_64_bytes does not support null elements within a list.",
                   std::invalid_argument);
      offsets =
        cudf::detail::offsetalator_factory::make_input_iterator(lcv.offsets(), lcv.offset());
      bytes = child.head<uint8_t>() + child.offset();
    }
  } else {
    CUDF_FAIL("xxhash_64_bytes supports only STRING, LIST<INT8> and LIST<UINT8> columns.",
              cudf::data_type_error);
  }

  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.size(),
                                    cudf::detail::copy_bitmask(input, stream, mr),
                                    input.null_count(),
                                    stream,
                                    mr);

  if (input.size() == 0) { return output; }

  auto const d_input      = column_device_view::create(input, stream);
  auto const output_begin = output->mutable_view().begin<hash_value_type>();
  auto const hasher       = XXHash_64<uint8_t>{seed};
  // thrust::tabulate is slow here, see NVIDIA/cccl#9070
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    input.size(),
    [d_input = *d_input, offsets, bytes, output_begin, hasher] __device__(size_type i) -> void {
      if (d_input.is_null(i)) { return; }
      auto const begin = offsets[i];
      auto const size  = offsets[i + 1] - begin;
      output_begin[i] =
        hasher.compute_bytes(reinterpret_cast<cuda::std::byte const*>(bytes + begin), size);
    },
    stream.get()));

  return output;
}

}  // namespace detail

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  cuda::stream_ref stream,
                                  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::xxhash_64(input, seed, stream, mr);
}

std::unique_ptr<column> xxhash_64_bytes(column_view const& input,
                                        uint64_t seed,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::xxhash_64_bytes(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
