/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/spark_hashing.cuh>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/spark_murmurhash3.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

#include <functional>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

void check_spark_murmurhash3_compatibility(table_view const& input)
{
  using column_checker_fn_t = std::function<void(column_view const&)>;

  column_checker_fn_t check_column = [&](column_view const& c) {
    if (c.type().id() == type_id::LIST) {
      auto const& list_col = lists_column_view(c);
      CUDF_EXPECTS(list_col.child().type().id() != type_id::STRUCT,
                   "Cannot compute hash of a table with a LIST of STRUCT columns.");
      check_column(list_col.child());
    } else if (c.type().id() == type_id::STRUCT) {
      for (auto child = c.child_begin(); child != c.child_end(); ++child) {
        check_column(*child);
      }
    }
  };

  for (column_view const& c : input) {
    check_column(c);
  }
}

}  // namespace

std::unique_ptr<column> spark_murmurhash3_x86_32(table_view const& input,
                                                 uint32_t seed,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  // The hasher works in unsigned; Spark reports a signed `Int`, so the output column is INT32.
  using output_type = int32_t;

  auto output = make_numeric_column(
    data_type(type_to_id<output_type>()), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  // Return early if there's nothing to hash
  if (input.num_rows() == 0) { return output; }

  // Lists of structs are not supported
  check_spark_murmurhash3_compatibility(input);

  auto const nullable = nullate::DYNAMIC{has_nested_nulls(input)};
  auto const row_hasher =
    cudf::detail::row::hash::row_hasher(input, stream, cudf::get_current_device_resource_ref());
  auto const output_begin = output->mutable_view().begin<output_type>();

  // Compute the hash value for each row
  auto const hasher =
    row_hasher
      .device_hasher<Spark_MurmurHash3_x86_32, cudf::detail::row::hash::spark_device_row_hasher>(
        nullable, seed);
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    input.num_rows(),
    [output_begin, hasher] __device__(size_type i) mutable {
      output_begin[i] = static_cast<output_type>(hasher(i));
    },
    stream.get()));

  return output;
}

}  // namespace detail

std::unique_ptr<column> spark_murmurhash3_x86_32(table_view const& input,
                                                 uint32_t seed,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::spark_murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
