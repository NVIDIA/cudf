/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/join_common_utils.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/hash_functions.cuh>
#include <cuco/pair.cuh>
#include <cuco/static_multiset.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream_ref>

#include <algorithm>
#include <atomic>
#include <bit>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::detail {
namespace {

using slot_type = cuco::pair<hash_value_type, size_type>;

template <typename Equality, typename Factory>
auto make_device_comparators(
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  Factory factory,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using allocator_type  = cudf::detail::rmm_host_allocator<Equality>;
  auto host_comparators = std::vector<Equality, allocator_type>{
    allocator_type{cudf::get_pinned_memory_resource(), stream}};
  host_comparators.reserve(preprocessed_right.size());
  std::transform(preprocessed_right.begin(),
                 preprocessed_right.end(),
                 std::back_inserter(host_comparators),
                 factory);
  auto d_comparators = cudf::detail::make_device_uvector_async(
    cudf::host_span<Equality const>{
      host_comparators.data(), host_comparators.size(), /*is_device_accessible=*/true},
    stream,
    temp_mr);
  stream.sync();  // wait for host_comparators to finish copying before it goes out of scope
  return d_comparators;
}

template <bool has_nested>
auto make_device_row_comparators(
  std::shared_ptr<row::equality::preprocessed_table> const& preprocessed_left,
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using equality_type =
    row::equality::device_row_comparator<has_nested,
                                         nullate::DYNAMIC,
                                         row::equality::nan_equal_physical_equality_comparator>;

  return make_device_comparators<equality_type>(
    preprocessed_right,
    [&](auto const& right) {
      auto const comparator = row::equality::two_table_comparator{preprocessed_left, right};
      return comparator.equal_to<has_nested>(has_nulls, compare_nulls).comparator;
    },
    stream,
    temp_mr);
}

auto make_device_primitive_row_comparators(
  std::shared_ptr<row::equality::preprocessed_table> const& preprocessed_left,
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using equality_type = row::primitive::row_equality_comparator;

  return make_device_comparators<equality_type>(
    preprocessed_right,
    [&](auto const& right) {
      return equality_type{has_nulls, preprocessed_left, right, compare_nulls};
    },
    stream,
    temp_mr);
}

size_type checked_batch_count(size_type batches)
{
  CUDF_EXPECTS(
    batches > 0, "streaming_hash_join requires max_num_batches > 0.", std::invalid_argument);
  return batches;
}

/**
 * @brief Describes how a slot's first 32-bit word is divided between hash and batch ID.
 *
 * The high `batch_bits` store the batch ID and the remaining low bits store the truncated row hash.
 * CUCO's probing hash functions use only the low hash bits, ensuring that equal rows from different
 * batches share the same probe sequence.
 */
struct batch_hash_layout {
  static constexpr int32_t num_hash_bits = std::numeric_limits<hash_value_type>::digits;
  static_assert(num_hash_bits == 32, "streaming_hash_join requires a 32-bit row hash");

  explicit batch_hash_layout(size_type max_num_batches)
    : batch_bits{std::bit_width(
        static_cast<hash_value_type>(checked_batch_count(max_num_batches) - 1))},
      batch_shift{num_hash_bits - batch_bits},
      hash_mask{std::numeric_limits<hash_value_type>::max() >> batch_bits}
  {
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr hash_value_type masked_hash(
    hash_value_type hash) const noexcept
  {
    return hash & hash_mask;
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr hash_value_type pack(hash_value_type hash,
                                                                size_type batch_id) const noexcept
  {
    if (batch_bits == 0) { return hash; }
    return masked_hash(hash) | (static_cast<hash_value_type>(batch_id) << batch_shift);
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type batch_id(
    hash_value_type packed_hash) const noexcept
  {
    if (batch_bits == 0) { return 0; }
    return static_cast<size_type>(packed_hash >> batch_shift);
  }

  int32_t batch_bits;
  int32_t batch_shift;
  hash_value_type hash_mask;
};

struct always_not_equal {
  __device__ constexpr bool operator()(slot_type const&, slot_type const&) const noexcept
  {
    return false;
  }
};

struct masked_hasher1 {
  hash_value_type hash_mask{std::numeric_limits<hash_value_type>::max()};

  __device__ constexpr hash_value_type operator()(slot_type const& key) const noexcept
  {
    return key.first & hash_mask;
  }
};

struct masked_hasher2 {
  masked_hasher2(hash_value_type mask = std::numeric_limits<hash_value_type>::max(),
                 hash_value_type seed = cudf::DEFAULT_HASH_SEED)
    : hash_mask{mask}, hash{seed}
  {
  }

  __device__ constexpr hash_value_type operator()(slot_type const& key) const noexcept
  {
    return hash(key.first & hash_mask);
  }

  hash_value_type hash_mask;
  cuco::xxhash_32<hash_value_type> hash;
};

using probing_scheme  = cuco::double_hashing<DEFAULT_JOIN_CG_SIZE, masked_hasher1, masked_hasher2>;
using hash_table_type = cuco::static_multiset<slot_type,
                                              cuco::extent<std::size_t>,
                                              cuda::thread_scope_device,
                                              always_not_equal,
                                              probing_scheme,
                                              rmm::mr::polymorphic_allocator<char>,
                                              cuco::storage<2>>;

template <typename Hasher>
struct build_pair_fn {
  Hasher hash;
  batch_hash_layout layout;
  size_type batch_id;

  __device__ slot_type operator()(size_type row_index) const noexcept
  {
    return slot_type{layout.pack(hash(row_index), batch_id), row_index};
  }
};

template <typename Hasher>
struct probe_pair_fn {
  Hasher hash;
  batch_hash_layout layout;

  __device__ slot_type operator()(size_type row_index) const noexcept
  {
    return slot_type{layout.masked_hash(hash(row_index)), row_index};
  }
};

template <typename RowEqual>
struct n_table_pair_equal {
  RowEqual const* comparators;
  batch_hash_layout layout;

  __device__ bool operator()(slot_type const& probe, slot_type const& build) const noexcept
  {
    auto const batch_id = layout.batch_id(build.first);
    return layout.masked_hash(probe.first) == layout.masked_hash(build.first) &&
           comparators[batch_id](probe.second, build.second);
  }
};

struct extract_index_fn {
  __device__ constexpr size_type operator()(slot_type const& value) const noexcept
  {
    return value.second;
  }
};

struct decode_slot_fn {
  batch_hash_layout layout;

  __device__ cuda::std::tuple<size_type, size_type> operator()(slot_type const& slot) const noexcept
  {
    return {layout.batch_id(slot.first), slot.second};
  }
};

}  // namespace

class streaming_hash_join_impl {
 public:
  // Empty copy of the right-side schema, owned so validation never depends on caller lifetimes.
  std::unique_ptr<table> right_schema;
  std::vector<size_type> right_key_indices;
  size_type total_right_rows;
  size_type max_num_batches;
  bool has_nulls;
  null_equality compare_nulls;
  std::atomic<size_type> inserted_rows{0};
  std::atomic<size_type> inserted_batches{0};
  bool has_nested_keys{false};
  batch_hash_layout layout;

  // Declared before the hash table so the resource outlives allocations that refer to it.
  cudf::memory_resources mr;
  hash_table_type hash_table;

  std::vector<std::shared_ptr<row::equality::preprocessed_table>> preprocessed_right;
  // Views of the inserted partitions, indexed by batch ID so callers can resolve the batch IDs
  // reported by `inner_join()` without tracking concurrent insert ordering themselves.
  std::vector<table_view> right_partitions;

  streaming_hash_join_impl(table_view const& schema,
                           std::span<size_type const> key_indices,
                           size_type total_rows,
                           size_type maximum_batches,
                           nullable_join nullable,
                           null_equality nulls_equal,
                           double load_factor,
                           cuda::stream_ref stream,
                           cudf::memory_resources resource)
    // Deep copy a zero-row slice rather than `empty_like`, which drops children such as a
    // dictionary's keys column and would make `have_same_types` reject every partition.
    : right_schema{std::make_unique<table>(
        cudf::slice(schema, {0, 0}).front(), stream, resource.get_output_mr())},
      right_key_indices{key_indices.begin(), key_indices.end()},
      total_right_rows{total_rows},
      max_num_batches{checked_batch_count(maximum_batches)},
      has_nulls{nullable == nullable_join::YES},
      compare_nulls{nulls_equal},
      layout{max_num_batches},
      mr{resource},
      hash_table{
        cuco::extent{[&] {
          CUDF_EXPECTS(total_rows >= 0,
                       "streaming_hash_join requires total_right_rows >= 0.",
                       std::invalid_argument);
          return static_cast<std::size_t>(total_rows);
        }()},
        checked_load_factor(load_factor),
        cuco::empty_key{slot_type{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
        always_not_equal{},
        probing_scheme{masked_hasher1{layout.hash_mask}, masked_hasher2{layout.hash_mask}},
        {},
        {},
        rmm::mr::polymorphic_allocator<char>{mr.get_output_mr()},
        stream.get()},
      preprocessed_right(static_cast<std::size_t>(max_num_batches)),
      right_partitions(static_cast<std::size_t>(max_num_batches))
  {
    CUDF_EXPECTS(right_schema->num_columns() > 0,
                 "streaming_hash_join requires at least one right-side column.",
                 std::invalid_argument);
    CUDF_EXPECTS(!right_key_indices.empty(),
                 "streaming_hash_join requires at least one right-side key column.",
                 std::invalid_argument);
    auto const schema_size = right_schema->num_columns();
    for (auto const index : right_key_indices) {
      CUDF_EXPECTS(index >= 0 && index < schema_size,
                   "streaming_hash_join key index is out of range for the provided schema.",
                   std::invalid_argument);
    }
    // The schema exemplar carries nesting, so this is settled once here rather than on first
    // insert, which is what previously forced the insert path to lock.
    has_nested_keys =
      cudf::detail::has_nested_columns(right_schema->view().select(right_key_indices));
  }

  void insert(table_view const& right_partition, cuda::stream_ref stream)
  {
    CUDF_EXPECTS(right_partition.num_columns() == right_schema->num_columns(),
                 "streaming_hash_join: inserted partition column count does not match schema.",
                 std::invalid_argument);
    CUDF_EXPECTS(cudf::have_same_types(right_schema->view(), right_partition),
                 "streaming_hash_join: inserted partition schema does not match the schema passed "
                 "to the constructor.",
                 cudf::data_type_error);
    auto const keys = right_partition.select(right_key_indices);

    auto preprocessed = row::equality::preprocessed_table::create(keys, stream, mr.get_output_mr());
    auto const batch_rows = keys.num_rows();
    auto row_bitmask      = [&]() -> std::optional<rmm::device_buffer> {
      if (batch_rows > 0 && compare_nulls == null_equality::UNEQUAL && nullable(keys)) {
        return cudf::detail::bitmask_and(keys, stream, mr.get_temporary_mr()).first;
      }
      return std::nullopt;
    }();

    auto old_inserted_rows = inserted_rows.load(std::memory_order_relaxed);
    do {
      CUDF_EXPECTS(batch_rows <= total_right_rows - old_inserted_rows,
                   "streaming_hash_join: cumulative inserted rows would exceed total_right_rows.",
                   std::invalid_argument);
    } while (!inserted_rows.compare_exchange_weak(
      old_inserted_rows, old_inserted_rows + batch_rows, std::memory_order_relaxed));

    auto batch_id = inserted_batches.load(std::memory_order_relaxed);
    do {
      if (batch_id >= max_num_batches) {
        inserted_rows.fetch_sub(batch_rows, std::memory_order_relaxed);
        CUDF_FAIL("streaming_hash_join: number of inserted batches would exceed max_num_batches.",
                  std::invalid_argument);
      }
    } while (
      !inserted_batches.compare_exchange_weak(batch_id, batch_id + 1, std::memory_order_relaxed));

    preprocessed_right[batch_id] = std::move(preprocessed);
    right_partitions[batch_id]   = right_partition;

    if (batch_rows > 0) {
      auto const nulls = nullate::DYNAMIC{has_nulls};
      auto insert_rows = [&](auto const& row_hasher) {
        auto const input_begin = cudf::detail::make_counting_transform_iterator(
          0, build_pair_fn{row_hasher, layout, batch_id});

        if (compare_nulls == null_equality::EQUAL || !nullable(keys)) {
          hash_table.insert_async(input_begin, input_begin + batch_rows, stream.get());
        } else {
          hash_table.insert_if_async(
            input_begin,
            input_begin + batch_rows,
            cuda::counting_iterator<size_type>{0},
            row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask->data())},
            stream.get());
        }
      };

      if (cudf::detail::is_primitive_row_op_compatible(keys)) {
        insert_rows(row::primitive::row_hasher{nulls, preprocessed_right[batch_id]});
      } else {
        insert_rows(row::hash::row_hasher{preprocessed_right[batch_id]}.device_hasher(nulls));
      }
    }
  }

  template <bool has_nested, bool use_primitive>
  auto probe(table_view const& left,
             std::optional<std::size_t> output_size,
             cuda::stream_ref stream,
             cudf::memory_resources resources) const
  {
    static_assert(!(has_nested && use_primitive),
                  "primitive row comparators are never used for nested keys");
    auto preprocessed_left =
      row::equality::preprocessed_table::create(left, stream, resources.get_temporary_mr());
    auto const num_batches = inserted_batches.load(std::memory_order_relaxed);
    auto const right_comparators =
      std::span{preprocessed_right}.first(static_cast<std::size_t>(num_batches));
    auto comparators = [&] {
      if constexpr (use_primitive) {
        return make_device_primitive_row_comparators(preprocessed_left,
                                                     right_comparators,
                                                     nullate::DYNAMIC{has_nulls},
                                                     compare_nulls,
                                                     stream,
                                                     resources.get_temporary_mr());
      } else {
        return make_device_row_comparators<has_nested>(preprocessed_left,
                                                       right_comparators,
                                                       nullate::DYNAMIC{has_nulls},
                                                       compare_nulls,
                                                       stream,
                                                       resources.get_temporary_mr());
      }
    }();
    auto const equality   = n_table_pair_equal{comparators.data(), layout};
    auto const row_hasher = [&] {
      if constexpr (use_primitive) {
        return row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_left};
      } else {
        return row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls});
      }
    }();
    auto const input_begin =
      cudf::detail::make_counting_transform_iterator(0, probe_pair_fn{row_hasher, layout});

    auto const join_size = output_size ? *output_size
                                       : hash_table.count(input_begin,
                                                          input_begin + left.num_rows(),
                                                          equality,
                                                          hash_table.hash_function(),
                                                          stream.get());

    auto const output_mr = resources.get_output_mr();
    auto left_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);
    auto batch_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);
    auto row_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);

    if (join_size > 0) {
      cudf::prefetch::detail::prefetch(*left_indices, stream);
      cudf::prefetch::detail::prefetch(*batch_indices, stream);
      cudf::prefetch::detail::prefetch(*row_indices, stream);
      auto const probe_output =
        cuda::transform_output_iterator{left_indices->begin(), extract_index_fn{}};
      auto const build_output = cuda::transform_output_iterator{
        cuda::zip_iterator(batch_indices->begin(), row_indices->begin()), decode_slot_fn{layout}};
      hash_table.retrieve(input_begin,
                          input_begin + left.num_rows(),
                          equality,
                          hash_table.hash_function(),
                          probe_output,
                          build_output,
                          stream.get());
    }

    return std::pair{std::move(left_indices),
                     std::pair{std::move(batch_indices), std::move(row_indices)}};
  }

  [[nodiscard]] table_view get_partition(size_type batch_id) const
  {
    CUDF_EXPECTS(batch_id >= 0 and batch_id < inserted_batches.load(std::memory_order_relaxed),
                 "streaming_hash_join: no partition has been inserted with this batch ID.",
                 std::out_of_range);
    return right_partitions[batch_id];
  }

  auto inner_join(table_view const& left,
                  std::optional<std::size_t> output_size,
                  cuda::stream_ref stream,
                  cudf::memory_resources resources) const
  {
    CUDF_EXPECTS(inserted_batches.load(std::memory_order_relaxed) > 0,
                 "streaming_hash_join: inner_join called before any insert().",
                 std::logic_error);
    validate_hash_join_probe(right_schema->view().select(right_key_indices), left, has_nulls);

    if (left.num_rows() == 0 || inserted_rows.load(std::memory_order_relaxed) == 0) {
      return std::pair{
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr()),
        std::pair{
          std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr()),
          std::make_unique<rmm::device_uvector<size_type>>(0, stream, resources.get_output_mr())}};
    }
    if (cudf::detail::is_primitive_row_op_compatible(left)) {
      return probe<false, true>(left, output_size, stream, resources);
    }
    return has_nested_keys ? probe<true, false>(left, output_size, stream, resources)
                           : probe<false, false>(left, output_size, stream, resources);
  }
};

}  // namespace cudf::detail

namespace cudf {

streaming_hash_join::streaming_hash_join(table_view const& right_schema,
                                         std::span<size_type const> right_key_indices,
                                         size_type total_right_rows,
                                         size_type max_num_batches,
                                         nullable_join has_nulls,
                                         null_equality compare_nulls,
                                         double load_factor,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
  : _impl{std::make_unique<cudf::detail::streaming_hash_join_impl>(right_schema,
                                                                   right_key_indices,
                                                                   total_right_rows,
                                                                   max_num_batches,
                                                                   has_nulls,
                                                                   compare_nulls,
                                                                   load_factor,
                                                                   stream,
                                                                   mr)}
{
}

streaming_hash_join::~streaming_hash_join()                                         = default;
streaming_hash_join::streaming_hash_join(streaming_hash_join&&) noexcept            = default;
streaming_hash_join& streaming_hash_join::operator=(streaming_hash_join&&) noexcept = default;

void streaming_hash_join::insert(table_view const& right_partition, cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  _impl->insert(right_partition, stream);
}

table_view streaming_hash_join::get_partition(size_type batch_id) const
{
  return _impl->get_partition(batch_id);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                    std::unique_ptr<rmm::device_uvector<size_type>>>>
streaming_hash_join::inner_join(table_view const& left,
                                std::optional<std::size_t> output_size,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->inner_join(left, output_size, stream, mr);
}

}  // namespace cudf
