/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "retrieve_impl.cuh"

namespace cudf::detail {

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::left_join(cudf::table_view const& left,
                             std::optional<std::size_t> output_size,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr) const
{
  return this->template join_retrieve<join_kind::LEFT_JOIN>(left, output_size, stream, mr);
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::left_join(cudf::table_view const& left,
                                       std::optional<std::size_t> output_size,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
