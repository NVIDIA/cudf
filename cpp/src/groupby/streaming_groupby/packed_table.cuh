/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/functional>
#include <cuda/stream>

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::packed {

using key_t = uint64_t;

constexpr key_t key_sentinel = std::numeric_limits<key_t>::max();

/// The set is kept at most this full, so that a probe ends within a few slots.
constexpr double max_load = 0.6;

/// Slots are numbered in `size_type` when the groups are written out.
constexpr int64_t max_slots = static_cast<int64_t>(std::numeric_limits<size_type>::max()) - 1;

/// Rows go into the table as many at a time as could all be new keys and still fit, and at
/// least this many (or the rest of the batch), so that a nearly full table is grown rather than
/// fed a few rows per kernel.
constexpr size_type min_chunk_rows = size_type{1} << 18;

constexpr uint32_t max_keys   = 8;
constexpr uint32_t max_values = 16;

using set_t = cuco::static_set<key_t,
                               cuco::extent<int64_t>,
                               cuda::thread_scope_device,
                               cuda::std::equal_to<key_t>,
                               cuco::linear_probing<1, cuco::xxhash_64<key_t>>,
                               rmm::mr::polymorphic_allocator<char>,
                               cuco::storage<1>>;

using insert_ref_t = decltype(std::declval<set_t const&>().ref(cuco::op::insert_and_find));

/// How a row's value is read: the bits of the accumulator it folds into.
enum class source : int8_t {
  none,          ///< No value is read; the row contributes 1 (COUNT).
  int_signed,    ///< A signed integer of `size` bytes, widened to int64.
  int_unsigned,  ///< An unsigned integer of `size` bytes, widened to uint64.
  float32,       ///< A float, widened to a double.
  float64,       ///< A double.
};

/// How a value folds into its accumulator: what it is added to or compared with, and how.
enum class fold : int8_t {
  sum_int,
  sum_float,
  min_signed,
  max_signed,
  min_unsigned,
  max_unsigned,
  min_float,
  max_float,
};

/// How a group's accumulator is written into its output column.
enum class store : int8_t {
  bits,      ///< All eight bytes as they are.
  truncate,  ///< The low `size` bytes of the integer.
  narrow,    ///< A double narrowed to a float.
};

struct key_layout {
  char const* data              = nullptr;
  bitmask_type const* null_mask = nullptr;
  size_type mask_offset         = 0;
  uint32_t size                 = 0;
  uint32_t shift                = 0;
  uint32_t valid_shift          = 0;
};

struct key_layouts {
  key_layout columns[max_keys];
  uint32_t count = 0;
  bool may_equal_sentinel = false;
  bool nulls_as_keys = false;
};

struct value_layout {
  char const* data              = nullptr;
  bitmask_type const* null_mask = nullptr;
  size_type mask_offset         = 0;
  uint32_t size                 = 0;
  source src                    = source::none;
  fold f                        = fold::sum_int;
  bool square                   = false;
  bool skip_nulls               = true;
};

struct value_layouts {
  value_layout columns[max_values];
  uint32_t count = 0;
};

struct value_folds {
  fold folds[max_values];
  uint32_t count = 0;
};

struct identities {
  uint64_t values[max_values];
  uint32_t count = 0;
};

struct accumulators {
  uint64_t* records = nullptr;
  uint32_t words    = 0;

  __host__ __device__ __forceinline__ uint64_t* of(int64_t slot) const
  {
    return records + slot * words;
  }
};

struct table_ref {
  insert_ref_t set;
  key_t const* slots = nullptr;
  int64_t capacity   = 0;
  accumulators acc;
  uint32_t num_values = 0;
};

struct counters {
  size_type* num_groups    = nullptr;
  size_type* sentinel_seen = nullptr;
};

struct chunk {
  key_layouts keys;
  value_layouts values;
  bitmask_type const* row_bitmask = nullptr;
  size_type row_bitmask_offset    = 0;
  size_type rows                  = 0;
  bool track_seen = false;
};

struct output_layout {
  char* data           = nullptr;
  uint8_t* valid       = nullptr;
  uint32_t size        = 0;
  uint32_t shift       = 0;
  uint32_t valid_shift = 0;
  store st             = store::bits;
};

struct output_layouts {
  output_layout keys[max_keys];
  uint32_t num_keys = 0;
  output_layout values[max_values];
  uint32_t num_values = 0;
};

struct group_list {
  size_type const* slots = nullptr;
  size_type num_regular  = 0;
  bool with_sentinel     = false;
};

struct value_spec {
  data_type type;
  aggregation::Kind kind;
  bool intermediate;
};

class packed_state {
 public:
  [[nodiscard]] static bool supported(std::vector<data_type> const& key_types,
                                      std::vector<value_spec> const& values,
                                      null_policy null_handling);

  packed_state(std::vector<data_type> key_types,
               std::vector<value_spec> values,
               null_policy null_handling,
               size_type initial_distinct_keys,
               cuda::mr::any_resource<cuda::mr::device_accessible> mr,
               cuda::stream_ref stream);

  ~packed_state();
  packed_state(packed_state const&)            = delete;
  packed_state& operator=(packed_state const&) = delete;

  void aggregate(table_view const& keys, table_view const& values, cuda::stream_ref stream);

  void merge(packed_state const& other, cuda::stream_ref stream);

  [[nodiscard]] std::pair<std::unique_ptr<table>, std::unique_ptr<table>> write_groups(
    cuda::stream_ref stream, rmm::device_async_resource_ref mr) const;

  [[nodiscard]] size_type num_groups() const noexcept { return _groups + (_sentinel_seen ? 1 : 0); }

 private:
  struct group_table;

  [[nodiscard]] uint32_t record_words() const noexcept
  {
    return static_cast<uint32_t>(_values.size()) + 1;
  }
  [[nodiscard]] int64_t room() const noexcept;
  void ensure_room(size_type more, cuda::stream_ref stream);
  void read_counters(cuda::stream_ref stream);
  void enable_seen_tracking(cuda::stream_ref stream);
  [[nodiscard]] table_ref ref() const;
  [[nodiscard]] counters counter_ptrs() noexcept;
  [[nodiscard]] chunk chunk_of(table_view const& keys,
                               table_view const& values,
                               bitmask_type const* row_bitmask,
                               size_type offset,
                               size_type rows) const;

  std::vector<data_type> _key_types;
  std::vector<uint32_t> _key_shifts;
  uint32_t _key_bits = 0;
  std::vector<value_spec> _values;
  std::vector<source> _sources;
  std::vector<fold> _folds;
  identities _identities;
  null_policy _null_handling;
  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;

  std::unique_ptr<group_table> _table;
  rmm::device_uvector<size_type> _counters;
  size_type _groups   = 0;
  bool _sentinel_seen = false;

  bool _track_seen = false;
  std::vector<bool> _ever_null;
  std::vector<bool> _key_nullable;
};

}  // namespace cudf::groupby::packed
