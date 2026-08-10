/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace cudf::detail {

/**
 * @brief Implementation of `cudf::streaming_hash_join`.
 *
 * Owns a persistent hash table and supports probing rows stored across multiple right-side
 * partitions.
 */
class streaming_hash_join {
 public:
  streaming_hash_join(std::span<data_type const> right_schema,
                      std::span<size_type const> right_key_indices,
                      size_type total_right_rows,
                      size_type max_num_batches,
                      nullable_join has_nulls,
                      null_equality compare_nulls,
                      double load_factor,
                      rmm::cuda_stream_view stream,
                      cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  ~streaming_hash_join();
  streaming_hash_join(streaming_hash_join const&)            = delete;
  streaming_hash_join& operator=(streaming_hash_join const&) = delete;
  streaming_hash_join(streaming_hash_join&&) noexcept;
  streaming_hash_join& operator=(streaming_hash_join&&) noexcept;

  void insert(cudf::table_view const& right_partition, rmm::cuda_stream_view stream);

  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                                    std::unique_ptr<rmm::device_uvector<size_type>>>>
  inner_join(cudf::table_view const& left,
             std::optional<std::size_t> output_size,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr) const;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace cudf::detail
