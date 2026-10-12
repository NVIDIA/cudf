/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "insert.cuh"

#include <thrust/transform.h>

namespace cudf::groupby {

void compute_batch_hashes(
  std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
  cudf::nullate::DYNAMIC has_null,
  bitmask_type const* batch_bitmask,
  hash_value_type* batch_hash_cache,
  size_type batch_size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const batch_hasher_obj = cudf::detail::row::hash::row_hasher{preprocessed_batch};
  auto const d_batch_hash     = batch_hasher_obj.device_hasher(has_null);
  thrust::transform(rmm::exec_policy_nosync(stream, mr),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    batch_hash_cache,
                    conditional_hash_fn<decltype(d_batch_hash)>{d_batch_hash, batch_bitmask});
}

template streaming_groupby::impl::batch_insert_result
streaming_groupby::impl::probe_and_insert_impl<false>(table_view const& batch_keys,
                                                      cuda::stream_ref stream);

}  // namespace cudf::groupby
