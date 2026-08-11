/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/types.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace cudf::detail::row::equality {
struct preprocessed_table;
}

namespace cudf {
namespace detail {

using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

/**
 * @brief Implementation of filtered join using set hash tables.
 *
 * Implements hash-based semi and anti joins using set semantics, where duplicate
 * keys are not stored in the hash table.
 */
class filtered_join {
 public:
  /**
   * @brief Adapter for insertion operations in the hash table
   *
   * Returns result of self comparator passed
   */
  template <typename T>
  struct insertion_adapter {
    insertion_adapter(T const& _c) : _comparator{_c} {}
    __device__ constexpr bool operator()(
      cuco::pair<hash_value_type, lhs_index_type> const& lhs,
      cuco::pair<hash_value_type, lhs_index_type> const& rhs) const noexcept
    {
      if (lhs.first != rhs.first) { return false; }
      auto const lhs_index = static_cast<size_type>(lhs.second);
      auto const rhs_index = static_cast<size_type>(rhs.second);
      return _comparator(lhs_index, rhs_index);
    }

   private:
    T _comparator;
  };

  /**
   * @brief Adapter for extracting hash values from key-value pairs
   */
  struct hash_extract_fn {
    template <typename T>
    __device__ constexpr hash_value_type operator()(
      cuco::pair<hash_value_type, T> const& key) const noexcept
    {
      return key.first;
    }
  };

  /**
   * @brief Adapter for generating key-value pairs from indices
   *
   * @tparam T Index type
   * @tparam Hasher Hash function type
   */
  template <typename T, typename Hasher>
  struct key_pair_fn {
    CUDF_HOST_DEVICE constexpr key_pair_fn(Hasher const& hasher) : _hasher{hasher} {}

    __device__ __forceinline__ auto operator()(size_type i) const noexcept
    {
      return cuco::pair{_hasher(i), T{i}};
    }

   private:
    Hasher _hasher;
  };

  /**
   * @brief Adapter for comparing key-value pairs
   *
   * Compares hash values first for performance, then uses the provided equality comparator
   *
   * @tparam Equal Equality comparator type
   */
  template <typename Equal>
  struct comparator_adapter {
    comparator_adapter(Equal const& d_equal) : _d_equal{d_equal} {}

    __device__ constexpr auto operator()(
      cuco::pair<hash_value_type, rhs_index_type> const& rhs,
      cuco::pair<hash_value_type, lhs_index_type> const& lhs) const noexcept
    {
      if (lhs.first != rhs.first) { return false; }
      return _d_equal(lhs.second, rhs.second);
    }

   private:
    Equal _d_equal;
  };

  /**
   * @brief Constructor for filtered_join
   *
   * Initializes the hash table with the right table and prepares it for join operations.
   *
   * @param right The right table used to build the hash table
   * @param compare_nulls How null values should be compared
   * @param load_factor Target load factor for the hash table
   * @param stream CUDA stream on which to perform operations
   * @param mr Device memory resource used to allocate the internal hash table
   */
  filtered_join(cudf::table_view const& right,
                cudf::null_equality compare_nulls,
                double load_factor,
                rmm::cuda_stream_view stream,
                cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  /**
   * @brief Returns indices of left table rows that have matching keys in the right table
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& left, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Returns indices of left table rows that do not have matching keys in the right table
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& left, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

 private:
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_anti_join(
    cudf::table_view const& left,
    join_kind kind,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  // Queries the hash table for every left row and writes the matches to contains_map.
  template <int32_t CGSize, typename Iterator, typename Ref>
  void query_right_table(cudf::table_view const& left,
                         Iterator left_iter,
                         Ref query_ref,
                         cudf::device_span<bool> contains_map,
                         rmm::cuda_stream_view stream);

  void query_right_table_primitive(
    cudf::table_view const& left,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
    cudf::device_span<bool> contains_map,
    rmm::cuda_stream_view stream);

  void query_right_table_flat(
    cudf::table_view const& left,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
    cudf::device_span<bool> contains_map,
    rmm::cuda_stream_view stream);

  void query_right_table_nested(
    cudf::table_view const& left,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
    cudf::device_span<bool> contains_map,
    rmm::cuda_stream_view stream);

  enum class row_operator_mode : uint8_t { PRIMITIVE, FLAT, NESTED };

  // Key type used in the hash table
  using key = cuco::pair<hash_value_type, lhs_index_type>;

  // Storage type for the hash table buckets
  using storage_type =
    cuco::bucket_storage<key,
                         1,  /// fixing bucket size to be 1 i.e each thread handles one slot
                         cuco::extent<std::size_t>,
                         rmm::mr::polymorphic_allocator<char>>;

  using single_probing_scheme = cuco::linear_probing<1, hash_extract_fn>;
  // Nested rows use four cooperating threads per probe.
  using nested_probing_scheme = cuco::linear_probing<4, hash_extract_fn>;

  row_operator_mode const _right_mode;
  storage_type _bucket_storage;  ///< Storage for hash table buckets

  // Empty sentinel key used to mark empty slots in the hash table
  static constexpr auto empty_sentinel_key = cuco::empty_key{
    cuco::pair{std::numeric_limits<hash_value_type>::max(), lhs_index_type{cudf::JoinNoMatch}}};
  cudf::table_view _right;                 ///< input table used to build the hash map
  cudf::null_equality const _nulls_equal;  ///< whether to consider nulls as equal
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table>
    _preprocessed_right;  ///< input table preprocessed for row operators

  // Build and probe row operators must use matching nullate modes. Since probe nullability is
  // unknown at build time, primitive paths use DYNAMIC true and other paths use YES.
  void insert_right_table_primitive(rmm::cuda_stream_view stream);
  // Populates the hash table from the right-row iterator.
  template <int32_t CGSize, typename Iterator, typename Ref>
  void insert_right_table(Iterator right_iter, Ref const& insert_ref, rmm::cuda_stream_view stream);

  void insert_right_table_flat(rmm::cuda_stream_view stream);
  void insert_right_table_nested(rmm::cuda_stream_view stream);

  /**
   * @brief Calculates the required storage size for the hash table
   *
   * Computes the appropriate size for the bucket storage based on the input
   * table size and desired load factor.
   *
   * @param num_rows Number of rows to store
   * @param load_factor Target load factor for the hash table
   * @param mode Row-operator mode used by the hash table
   * @return Calculated bucket storage size
   */
  static std::size_t compute_bucket_storage_size(cudf::size_type num_rows,
                                                 double load_factor,
                                                 row_operator_mode mode);

  static row_operator_mode select_row_operator_mode(cudf::table_view const& table);
};

}  // namespace detail
}  // namespace cudf
