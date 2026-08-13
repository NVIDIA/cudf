/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <memory>
#include <optional>

// Forward declaration
namespace cudf::detail::row::equality {
class preprocessed_table;
}

namespace cudf {
namespace detail {
/**
 * @brief Hash join that builds a hash table with the right table on construction and probes
 * results in subsequent `*_join` member functions.
 *
 * User-defined hash function can be passed via the template parameter `Hasher`
 *
 * @tparam Hasher Unary callable type
 */
template <typename Hasher>
class hash_join {
 public:
  /// Opaque CUDA implementation holding the hash table and related device functors.
  struct impl;

  hash_join() = delete;
  ~hash_join();
  hash_join(hash_join const&)            = delete;
  hash_join(hash_join&&)                 = delete;
  hash_join& operator=(hash_join const&) = delete;
  hash_join& operator=(hash_join&&)      = delete;

  /**
   * @brief Constructor that internally builds the hash table from the given `right` table.
   *
   * @throw cudf::logic_error if the number of columns in `right` table is 0.
   *
   * @param right The right table, from which the hash table is built.
   * @param has_nulls Flag to indicate if the there exists any nulls in the `right` table or
   *        any `left` table that will be used later for join.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the internal hash table.
   */
  hash_join(cudf::table_view const& right,
            bool has_nulls,
            cudf::null_equality compare_nulls,
            rmm::cuda_stream_view stream,
            cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  /**
   * @copydoc hash_join(cudf::table_view const&, bool, null_equality, rmm::cuda_stream_view,
   * cuda::mr::any_resource<cuda::mr::device_accessible>)
   *
   * @param load_factor The hash table occupancy ratio in (0,1]. A value of 0.5 means 50% occupancy.
   */
  hash_join(cudf::table_view const& right,
            bool has_nulls,
            cudf::null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream,
            cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  /**
   * @copydoc cudf::hash_join::inner_join
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& left,
             std::optional<std::size_t> output_size,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::left_join
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& left,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::full_join
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& left,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::inner_join_size
   */
  [[nodiscard]] std::size_t inner_join_size(cudf::table_view const& left,
                                            rmm::cuda_stream_view stream) const;

  /**
   * @copydoc cudf::hash_join::left_join_size
   */
  [[nodiscard]] std::size_t left_join_size(cudf::table_view const& left,
                                           rmm::cuda_stream_view stream) const;

  /**
   * @copydoc cudf::hash_join::full_join_size
   */
  [[nodiscard]] std::size_t full_join_size(cudf::table_view const& left,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::inner_join_match_context
   */
  [[nodiscard]] cudf::join_match_context inner_join_match_context(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::left_join_match_context
   */
  [[nodiscard]] cudf::join_match_context left_join_match_context(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::full_join_match_context
   */
  [[nodiscard]] cudf::join_match_context full_join_match_context(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::partitioned_inner_join
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(cudf::join_partition_context const& context,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::partitioned_left_join
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_left_join(cudf::join_partition_context const& context,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr) const;

  /**
   * @copydoc cudf::hash_join::partitioned_full_join
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_full_join(cudf::join_partition_context const& context,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr) const;

 private:
  bool const _is_empty;   ///< true if `_hash_table` is empty
  bool const _has_nulls;  ///< true if nulls are present in either right table or any left table
  cudf::null_equality const _nulls_equal;  ///< whether to consider nulls as equal
  cudf::table_view _right;                 ///< input table to build the hash map
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table>
    _preprocessed_right;        ///< input table preprocssed for row operators
  std::unique_ptr<impl> _impl;  ///< CUDA hash table implementation

  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> make_match_counts(
    join_kind join,
    cudf::table_view const& left,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_join_retrieve(join_kind join,
                            cudf::join_partition_context const& context,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const;

  template <join_kind Join>
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  join_retrieve(cudf::table_view const& probe,
                std::optional<std::size_t> output_size,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr) const;

  template <join_kind Join>
  [[nodiscard]] std::size_t join_size(cudf::table_view const& left,
                                      rmm::cuda_stream_view stream) const;

  template <join_kind Join>
  [[nodiscard]] std::size_t join_size(cudf::table_view const& left,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr) const;
};

}  // namespace detail
}  // namespace cudf
