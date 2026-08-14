/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

namespace cudf::detail {

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_inner_join(cudf::join_partition_context const& context,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr) const
{
  return this->partitioned_join_retrieve(join_kind::INNER_JOIN, context, stream, mr);
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::partitioned_inner_join(cudf::join_partition_context const& context,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
