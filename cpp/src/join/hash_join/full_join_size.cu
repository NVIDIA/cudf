/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

namespace cudf::detail {

template <typename Hasher>
std::size_t hash_join<Hasher>::full_join_size(cudf::table_view const& left,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr) const
{
  return this->template join_size<join_kind::FULL_JOIN>(left, stream, mr);
}

template std::size_t hash_join<hash_join_hasher>::full_join_size(cudf::table_view const& left,
                                                                 cuda::stream_ref stream,
                                                                 cudf::memory_resources mr) const;

}  // namespace cudf::detail
