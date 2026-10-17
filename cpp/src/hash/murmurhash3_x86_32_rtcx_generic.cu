/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "murmurhash3_x86_32_rtcx.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32_rtcx_tags.hpp>

#include <rtcx/algorithm_planner.hpp>

namespace cudf::hashing::detail {

std::unique_ptr<column> murmurhash3_x86_32_rtcx_generic(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& input,
  size_type num_rows,
  uint32_t seed,
  bool nullable,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(
    data_type(type_to_id<hash_value_type>()), num_rows, mask_state::UNALLOCATED, stream, mr);
  if (num_rows == 0) { return output; }
  // The row-hasher ABI requires CUDA headers even though the kernel lives only in the LTO fragment.
  auto const hasher = cudf::detail::row::hash::row_hasher(input).device_hasher<MurmurHash3_x86_32>(
    nullate::DYNAMIC{nullable}, seed);
  static rtcx::launcher_jit_cache cache;
  rtcx::algorithm_planner planner{"cudf_murmurhash3_x86_32_rtcx_generic_entry", cache};
  planner.add_static_fragment<rtcx_murmur::fragment_tag_entry_generic>();
  auto const launcher = planner.get_launcher();
  launcher->dispatch<void(hash_value_type*, size_type, decltype(hasher))>(
    stream.get(),
    dim3((num_rows + 255) / 256),
    dim3(256),
    0,
    output->mutable_view().begin<hash_value_type>(),
    num_rows,
    hasher);
  return output;
}

}  // namespace cudf::hashing::detail
