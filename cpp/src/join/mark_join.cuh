/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/types.cuh>
#include <cuda/std/limits>

#include <memory>

namespace cudf::detail {

using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

static constexpr hash_value_type mark_bit_mask = hash_value_type{1}
                                                 << (sizeof(hash_value_type) * 8 - 1);

CUDF_HOST_DEVICE constexpr hash_value_type set_mark(hash_value_type value) noexcept
{
  return value | mark_bit_mask;
}

CUDF_HOST_DEVICE constexpr hash_value_type unset_mark(hash_value_type value) noexcept
{
  return value & ~mark_bit_mask;
}

CUDF_HOST_DEVICE constexpr bool is_marked(hash_value_type value) noexcept
{
  return (value & mark_bit_mask) != 0;
}

struct masked_hash_fn {
  template <typename T>
  CUDF_HOST_DEVICE constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, T> const& key) const noexcept
  {
    return unset_mark(key.first);
  }
};

struct secondary_hash_fn {
  uint32_t _seed{0};

  CUDF_HOST_DEVICE secondary_hash_fn() = default;
  CUDF_HOST_DEVICE secondary_hash_fn(uint32_t seed) : _seed{seed} {}

  template <typename T>
  CUDF_HOST_DEVICE auto operator()(cuco::pair<hash_value_type, T> const& key) const noexcept
  {
    return cuco::xxhash_32<hash_value_type>{_seed}(unset_mark(key.first));
  }
};

template <typename T, typename Hasher>
struct masked_key_fn {
  CUDF_HOST_DEVICE constexpr masked_key_fn(Hasher const& hasher) : _hasher{hasher} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{unset_mark(_hasher(i)), T{i}};
  }

 private:
  Hasher _hasher;
};

template <typename Hasher>
struct masked_hash_value_fn {
  CUDF_HOST_DEVICE constexpr masked_hash_value_fn(Hasher const& hasher) : _hasher{hasher} {}

  __device__ __forceinline__ hash_value_type operator()(size_type i) const noexcept
  {
    return unset_mark(_hasher(i));
  }

 private:
  Hasher _hasher;
};

template <typename IndexType>
struct hash_pair_fn {
  CUDF_HOST_DEVICE constexpr hash_pair_fn(hash_value_type const* hashes) : _hashes{hashes} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{_hashes[i], IndexType{i}};
  }

 private:
  hash_value_type const* _hashes;
};

template <typename Equal>
struct masked_comparator_fn {
  masked_comparator_fn(Equal const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& lhs,
    cuco::pair<hash_value_type, lhs_index_type> const& rhs) const noexcept
  {
    if (unset_mark(lhs.first) != unset_mark(rhs.first)) { return false; }
    return _d_equal(lhs.second, rhs.second);
  }

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, rhs_index_type> const& right,
    cuco::pair<hash_value_type, lhs_index_type> const& left) const noexcept
  {
    if (unset_mark(right.first) != unset_mark(left.first)) { return false; }
    return _d_equal(left.second, right.second);
  }

 private:
  Equal _d_equal;
};

template <typename T>
struct insertion_adapter {
  insertion_adapter(T const&) {}
  __device__ constexpr bool operator()(
    cuco::pair<hash_value_type, lhs_index_type> const&,
    cuco::pair<hash_value_type, lhs_index_type> const&) const noexcept
  {
    return false;
  }
};

static constexpr uint32_t mark_join_cg_size{1};
static constexpr uint32_t mark_join_bucket_size{1};

using masked_probing_scheme =
  cuco::double_hashing<mark_join_cg_size, masked_hash_fn, secondary_hash_fn>;

using mark_key_type = cuco::pair<hash_value_type, lhs_index_type>;

using mark_storage_type = cuco::bucket_storage<mark_key_type,
                                               mark_join_bucket_size,
                                               cuco::extent<std::size_t>,
                                               rmm::mr::polymorphic_allocator<char>>;

using storage_ref_type =
  cuco::bucket_storage_ref<mark_key_type, mark_join_bucket_size, cuco::extent<std::size_t>>;
using right_key_type = cuco::pair<hash_value_type, rhs_index_type>;

using bloom_filter_policy_type    = cuco::default_filter_policy<hash_value_type>;
using bloom_filter_allocator_type = rmm::mr::polymorphic_allocator<cuda::std::byte>;
using bloom_filter_type           = cuco::bloom_filter<hash_value_type,
                                                       cuco::extent<std::size_t>,
                                                       cuda::thread_scope_device,
                                                       bloom_filter_policy_type,
                                                       bloom_filter_allocator_type>;

static constexpr auto masked_empty_sentinel =
  cuco::empty_key{cuco::pair{unset_mark(cuda::std::numeric_limits<hash_value_type>::max()),
                             lhs_index_type{cudf::JoinNoMatch}}};

class mark_join {
 public:
  mark_join(cudf::table_view const& left,
            cudf::null_equality compare_nulls,
            double load_factor,
            cudf::join_prefilter prefilter,
            rmm::cuda_stream_view stream,
            cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& right, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& right, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

 private:
  using primitive_row_hasher =
    cudf::detail::row::primitive::row_hasher<cudf::hashing::detail::default_hash>;
  using primitive_row_comparator = cudf::detail::row::primitive::row_equality_comparator;
  using row_hasher =
    cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash, nullate::YES>;
  bool _has_nested_columns;
  cudf::table_view _left;
  cudf::null_equality _nulls_equal;
  cudf::join_prefilter _prefilter;
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> _preprocessed_left;
  mark_storage_type _bucket_storage;
  std::unique_ptr<bloom_filter_type> _bloom_filter;
  cudf::size_type _num_left_inserted{0};

  [[nodiscard]] cudf::size_type num_left_inserted() const { return _num_left_inserted; }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_anti_join(
    cudf::table_view const& right,
    join_kind kind,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  template <typename Comparator>
  cudf::size_type mark_probe_without_prefilter(storage_ref_type storage_ref,
                                               Comparator comparator,
                                               right_key_type const* right_rows,
                                               cudf::size_type num_right_rows,
                                               bitmask_type const* right_row_bitmask,
                                               rmm::cuda_stream_view stream);

  template <typename Comparator>
  cudf::size_type mark_probe_with_prefilter(storage_ref_type storage_ref,
                                            Comparator comparator,
                                            right_key_type const* right_rows,
                                            cudf::size_type num_right_rows,
                                            bitmask_type const* right_row_bitmask,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

  template <typename Comparator>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_probe_and_retrieve(
    cudf::table_view const& right,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_right,
    join_kind kind,
    Comparator comparator,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  void clear_marks(rmm::cuda_stream_view stream);
};

}  // namespace cudf::detail
