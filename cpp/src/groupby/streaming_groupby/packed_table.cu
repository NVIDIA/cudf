/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/utils.hpp"
#include "packed_table.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <thrust/copy.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <string>

namespace cudf::groupby::packed {

namespace {

constexpr unsigned threads_per_block = 256;
constexpr int64_t max_blocks         = 32768;

unsigned blocks_for(int64_t work)
{
  return static_cast<unsigned>(
    std::min<int64_t>((work + threads_per_block - 1) / threads_per_block, max_blocks));
}


__device__ __forceinline__ uint64_t load_bits(char const* data, int64_t row, uint32_t size)
{
  switch (size) {
    case 1: return reinterpret_cast<uint8_t const*>(data)[row];
    case 2: return reinterpret_cast<uint16_t const*>(data)[row];
    case 4: return reinterpret_cast<uint32_t const*>(data)[row];
    default: return reinterpret_cast<uint64_t const*>(data)[row];
  }
}

__device__ __forceinline__ int64_t load_signed(char const* data, int64_t row, uint32_t size)
{
  switch (size) {
    case 1: return reinterpret_cast<int8_t const*>(data)[row];
    case 2: return reinterpret_cast<int16_t const*>(data)[row];
    case 4: return reinterpret_cast<int32_t const*>(data)[row];
    default: return reinterpret_cast<int64_t const*>(data)[row];
  }
}

__device__ __forceinline__ void store_bits(char* data, int64_t row, uint32_t size, uint64_t bits)
{
  switch (size) {
    case 1: reinterpret_cast<uint8_t*>(data)[row] = static_cast<uint8_t>(bits); break;
    case 2: reinterpret_cast<uint16_t*>(data)[row] = static_cast<uint16_t>(bits); break;
    case 4: reinterpret_cast<uint32_t*>(data)[row] = static_cast<uint32_t>(bits); break;
    default: reinterpret_cast<uint64_t*>(data)[row] = bits; break;
  }
}

__device__ __forceinline__ bool is_valid(bitmask_type const* mask, size_type offset, int64_t row)
{
  return mask == nullptr || cudf::bit_is_set(mask, offset + static_cast<size_type>(row));
}

__device__ __forceinline__ void atomic_min_double(uint64_t* accumulator, double value)
{
  if (value >= 0) {
    atomicMin(reinterpret_cast<long long*>(accumulator), __double_as_longlong(value));
  } else {
    atomicMax(reinterpret_cast<unsigned long long*>(accumulator),
              static_cast<unsigned long long>(__double_as_longlong(value)));
  }
}

__device__ __forceinline__ void atomic_max_double(uint64_t* accumulator, double value)
{
  if (value >= 0) {
    atomicMax(reinterpret_cast<long long*>(accumulator), __double_as_longlong(value));
  } else {
    atomicMin(reinterpret_cast<unsigned long long*>(accumulator),
              static_cast<unsigned long long>(__double_as_longlong(value)));
  }
}

__device__ __forceinline__ uint64_t load_value_bits(value_layout const& value, int64_t row)
{
  switch (value.src) {
    case source::none: return 1;
    case source::int_signed: {
      auto const v = static_cast<uint64_t>(load_signed(value.data, row, value.size));
      return value.square ? v * v : v;
    }
    case source::int_unsigned: {
      auto const v = load_bits(value.data, row, value.size);
      return value.square ? v * v : v;
    }
    case source::float32:
    case source::float64: {
      auto const v = value.src == source::float32
                       ? static_cast<double>(reinterpret_cast<float const*>(value.data)[row])
                       : reinterpret_cast<double const*>(value.data)[row];
      return static_cast<uint64_t>(__double_as_longlong(value.square ? v * v : v));
    }
  }
  return 0;
}

__device__ __forceinline__ void fold_bits(fold f, uint64_t bits, uint64_t* accumulator)
{
  switch (f) {
    case fold::sum_int:
      atomicAdd(reinterpret_cast<unsigned long long*>(accumulator),
                static_cast<unsigned long long>(bits));
      break;
    case fold::sum_float:
      atomicAdd(reinterpret_cast<double*>(accumulator),
                __longlong_as_double(static_cast<long long>(bits)));
      break;
    case fold::min_signed:
      atomicMin(reinterpret_cast<long long*>(accumulator), static_cast<long long>(bits));
      break;
    case fold::max_signed:
      atomicMax(reinterpret_cast<long long*>(accumulator), static_cast<long long>(bits));
      break;
    case fold::min_unsigned:
      atomicMin(reinterpret_cast<unsigned long long*>(accumulator),
                static_cast<unsigned long long>(bits));
      break;
    case fold::max_unsigned:
      atomicMax(reinterpret_cast<unsigned long long*>(accumulator),
                static_cast<unsigned long long>(bits));
      break;
    case fold::min_float:
      atomic_min_double(accumulator, __longlong_as_double(static_cast<long long>(bits)));
      break;
    case fold::max_float:
      atomic_max_double(accumulator, __longlong_as_double(static_cast<long long>(bits)));
      break;
  }
}

__device__ __forceinline__ uint64_t fold_row(value_layouts const& values,
                                             int64_t row,
                                             uint64_t* record)
{
  uint64_t seen = 0;
  for (uint32_t i = 0; i < values.count; ++i) {
    auto const& value = values.columns[i];
    if (value.skip_nulls && !is_valid(value.null_mask, value.mask_offset, row)) { continue; }
    fold_bits(value.f, load_value_bits(value, row), record + i);
    seen |= uint64_t{1} << i;
  }
  return seen;
}

__device__ __forceinline__ key_t pack_key(key_layouts const& keys, int64_t row)
{
  key_t key = 0;
  for (uint32_t i = 0; i < keys.count; ++i) {
    auto const& column = keys.columns[i];
    auto const valid   = is_valid(column.null_mask, column.mask_offset, row);
    if (valid) { key |= load_bits(column.data, row, column.size) << column.shift; }
    if (keys.nulls_as_keys) { key |= static_cast<key_t>(valid) << column.valid_shift; }
  }
  return key;
}

__device__ __forceinline__ int64_t slot_of(table_ref& table,
                                           key_layouts const& keys,
                                           key_t key,
                                           counters count)
{
  if (keys.may_equal_sentinel && key == key_sentinel) {
    *count.sentinel_seen = 1;
    return table.capacity;
  }
  auto const [it, inserted] = table.set.insert_and_find(key);
  if (inserted) { atomicAdd(count.num_groups, 1); }
  return static_cast<int64_t>(it - table.slots);
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

CUDF_KERNEL void init_records(accumulators acc, int64_t capacity, identities ids)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto slot = cudf::detail::grid_1d::global_thread_id(); slot <= capacity; slot += stride) {
    auto* record = acc.of(slot);
    for (uint32_t i = 0; i < ids.count; ++i) {
      record[i] = ids.values[i];
    }
    record[ids.count] = 0;
  }
}

CUDF_KERNEL void aggregate_rows(table_ref table, chunk rows, counters count)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto row = cudf::detail::grid_1d::global_thread_id(); row < rows.rows; row += stride) {
    if (!is_valid(rows.row_bitmask, rows.row_bitmask_offset, row)) { continue; }

    auto const key  = pack_key(rows.keys, row);
    auto* record    = table.acc.of(slot_of(table, rows.keys, key, count));
    auto const seen = fold_row(rows.values, row, record);
    if (rows.track_seen) {
      atomicOr(reinterpret_cast<unsigned long long*>(record + table.num_values),
               static_cast<unsigned long long>(seen));
    }
  }
}

CUDF_KERNEL void move_groups(key_t const* old_slots,
                             int64_t old_capacity,
                             accumulators from,
                             table_ref to)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto old_slot = cudf::detail::grid_1d::global_thread_id(); old_slot < old_capacity;
       old_slot += stride) {
    auto const key = old_slots[old_slot];
    if (key == key_sentinel) { continue; }

    auto const* old_record = from.of(old_slot);
    auto* record = to.acc.of(static_cast<int64_t>(to.set.insert_and_find(key).first - to.slots));
    for (uint32_t i = 0; i < from.words; ++i) {
      record[i] = old_record[i];
    }
  }
}

CUDF_KERNEL void mark_seen(key_t const* slots,
                           int64_t capacity,
                           accumulators acc,
                           uint32_t num_values,
                           size_type const* sentinel_seen)
{
  auto const full_mask = num_values == 64 ? ~uint64_t{0} : (uint64_t{1} << num_values) - 1;
  auto const stride    = cudf::detail::grid_1d::grid_stride();
  for (auto slot = cudf::detail::grid_1d::global_thread_id(); slot <= capacity; slot += stride) {
    auto const occupied = slot == capacity ? *sentinel_seen != 0 : slots[slot] != key_sentinel;
    if (occupied) { acc.of(slot)[num_values] = full_mask; }
  }
}

struct other_table {
  key_t const* slots = nullptr;
  int64_t capacity   = 0;
  accumulators acc;
  size_type const* sentinel_seen = nullptr;
  bool track_seen = false;
};

CUDF_KERNEL void merge_groups(
  table_ref table, other_table other, value_folds folds, bool track_seen, counters count)
{
  auto const full_mask =
    table.num_values == 64 ? ~uint64_t{0} : (uint64_t{1} << table.num_values) - 1;
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto slot = cudf::detail::grid_1d::global_thread_id(); slot <= other.capacity;
       slot += stride) {
    int64_t target_slot;
    if (slot == other.capacity) {
      if (*other.sentinel_seen == 0) { continue; }
      *count.sentinel_seen = 1;
      target_slot          = table.capacity;
    } else {
      auto const key = other.slots[slot];
      if (key == key_sentinel) { continue; }
      auto const [it, inserted] = table.set.insert_and_find(key);
      if (inserted) { atomicAdd(count.num_groups, 1); }
      target_slot = static_cast<int64_t>(it - table.slots);
    }

    auto const* source = other.acc.of(slot);
    auto* record       = table.acc.of(target_slot);
    auto const seen    = other.track_seen ? source[table.num_values] : full_mask;
    for (uint32_t i = 0; i < folds.count; ++i) {
      if ((seen >> i) & 1) { fold_bits(folds.folds[i], source[i], record + i); }
    }
    if (track_seen) {
      atomicOr(reinterpret_cast<unsigned long long*>(record + table.num_values),
               static_cast<unsigned long long>(seen));
    }
  }
}

CUDF_KERNEL void write_groups_kernel(group_list groups, table_ref table, output_layouts out)
{
  auto const num_groups = static_cast<int64_t>(groups.num_regular) + (groups.with_sentinel ? 1 : 0);
  auto const stride     = cudf::detail::grid_1d::grid_stride();
  for (auto group = cudf::detail::grid_1d::global_thread_id(); group < num_groups;
       group += stride) {
    auto const sentinel = group == groups.num_regular;
    auto const slot     = sentinel ? table.capacity : static_cast<int64_t>(groups.slots[group]);
    auto const* record  = table.acc.of(slot);
    auto const key      = sentinel ? key_sentinel : table.slots[slot];
    auto const seen     = record[table.num_values];

    for (uint32_t i = 0; i < out.num_keys; ++i) {
      auto const& column = out.keys[i];
      store_bits(column.data, group, column.size, key >> column.shift);
      if (column.valid) { column.valid[group] = (key >> column.valid_shift) & 1; }
    }

    for (uint32_t i = 0; i < out.num_values; ++i) {
      auto const& value = out.values[i];
      auto const bits   = record[i];
      switch (value.st) {
        case store::bits: reinterpret_cast<uint64_t*>(value.data)[group] = bits; break;
        case store::truncate: store_bits(value.data, group, value.size, bits); break;
        case store::narrow:
          reinterpret_cast<float*>(value.data)[group] =
            static_cast<float>(__longlong_as_double(static_cast<long long>(bits)));
          break;
      }
      if (value.valid) { value.valid[group] = (seen >> i) & 1; }
    }
  }
}

struct is_occupied {
  __device__ bool operator()(key_t key) const { return key != key_sentinel; }
};

struct is_nonzero {
  __device__ bool operator()(uint8_t v) const { return v != 0; }
};

// ---------------------------------------------------------------------------------------------
// Host-side classification of cudf types
// ---------------------------------------------------------------------------------------------

struct element_class {
  uint32_t size = 0;
  source src    = source::none;
  bool ok       = false;
};

struct element_class_fn {
  template <typename T>
  element_class operator()() const
  {
    if constexpr (cudf::is_chrono<T>()) {
      using rep = typename T::rep;
      return {sizeof(rep), std::is_signed_v<rep> ? source::int_signed : source::int_unsigned, true};
    } else if constexpr (std::is_same_v<T, float>) {
      return {4, source::float32, true};
    } else if constexpr (std::is_same_v<T, double>) {
      return {8, source::float64, true};
    } else if constexpr (cudf::is_integral<T>()) {
      return {sizeof(T), std::is_signed_v<T> ? source::int_signed : source::int_unsigned, true};
    } else {
      return {};
    }
  }
};

element_class classify(data_type type) { return cudf::type_dispatcher(type, element_class_fn{}); }

bool is_count(aggregation::Kind kind)
{
  return kind == aggregation::COUNT_VALID || kind == aggregation::COUNT_ALL;
}

/// Whether one aggregate can be kept in a record, and how its value is read.
element_class classify_value(value_spec const& value)
{
  switch (value.kind) {
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL: return {0, source::none, true};
    case aggregation::SUM: {
      if (cudf::is_timestamp(value.type)) { return {}; }
      return classify(value.type);
    }
    case aggregation::SUM_OF_SQUARES: {
      if (cudf::is_chrono(value.type)) { return {}; }
      return classify(value.type);
    }
    case aggregation::MIN:
    case aggregation::MAX: return classify(value.type);
    default: return {};
  }
}

fold fold_of(aggregation::Kind kind, source src)
{
  auto const is_float = src == source::float32 || src == source::float64;
  switch (kind) {
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL: return fold::sum_int;
    case aggregation::SUM:
    case aggregation::SUM_OF_SQUARES: return is_float ? fold::sum_float : fold::sum_int;
    case aggregation::MIN:
      return is_float ? fold::min_float
                      : (src == source::int_signed ? fold::min_signed : fold::min_unsigned);
    case aggregation::MAX:
      return is_float ? fold::max_float
                      : (src == source::int_signed ? fold::max_signed : fold::max_unsigned);
    default: CUDF_FAIL("Internal error: aggregation kind has no fold.");
  }
}

uint64_t identity_of(fold f)
{
  switch (f) {
    case fold::sum_int:
    case fold::sum_float:
    case fold::max_unsigned: return 0;
    case fold::min_signed: return static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    case fold::max_signed: return static_cast<uint64_t>(std::numeric_limits<int64_t>::min());
    case fold::min_unsigned: return std::numeric_limits<uint64_t>::max();
    case fold::min_float: return std::bit_cast<uint64_t>(std::numeric_limits<double>::infinity());
    case fold::max_float: return std::bit_cast<uint64_t>(-std::numeric_limits<double>::infinity());
  }
  return 0;
}

store store_of(data_type target)
{
  auto const size = cudf::size_of(target);
  if (cudf::is_floating_point(target) && size == 4) { return store::narrow; }
  return size == 8 ? store::bits : store::truncate;
}

}  // namespace

struct packed_state::group_table {
  std::unique_ptr<set_t> set;
  key_t const* slots = nullptr;
  int64_t capacity   = 0;
  rmm::device_uvector<uint64_t> records;
  accumulators acc;

  group_table(int64_t requested,
              identities const& ids,
              cuda::mr::any_resource<cuda::mr::device_accessible>& mr,
              cuda::stream_ref stream)
    : set{std::make_unique<set_t>(cuco::extent<int64_t>{requested},
                                  cuco::empty_key<key_t>{key_sentinel},
                                  cuda::std::equal_to<key_t>{},
                                  cuco::linear_probing<1, cuco::xxhash_64<key_t>>{},
                                  cuco::cuda_thread_scope<cuda::thread_scope_device>{},
                                  cuco::storage<1>{},
                                  rmm::mr::polymorphic_allocator<char>{mr},
                                  stream)},
      capacity{static_cast<int64_t>(set->capacity())},
      records{0, stream, rmm::device_async_resource_ref{mr}}
  {
    CUDF_EXPECTS(capacity <= max_slots,
                 "Distinct key count would exceed the largest supported capacity (" +
                   std::to_string(max_slots) + " slots).");
    slots = set->data();

    auto const words = ids.count + 1;
    records          = rmm::device_uvector<uint64_t>(
      static_cast<size_t>(capacity + 1) * words, stream, rmm::device_async_resource_ref{mr});
    acc = accumulators{records.data(), words};

    init_records<<<blocks_for(capacity + 1), threads_per_block, 0, stream.get()>>>(
      acc, capacity, ids);
    CUDF_CHECK_CUDA(stream.get());
  }

  [[nodiscard]] table_ref ref(uint32_t num_values) const
  {
    return table_ref{set->ref(cuco::op::insert_and_find), slots, capacity, acc, num_values};
  }

  [[nodiscard]] int64_t room(int64_t groups) const
  {
    auto const fits = static_cast<int64_t>(max_load * static_cast<double>(capacity));
    return fits > groups ? fits - groups : 0;
  }

  void take_groups_of(group_table const& from, uint32_t num_values, cuda::stream_ref stream)
  {
    CUDF_CUDA_TRY(cudaMemcpyAsync(acc.of(capacity),
                                  from.acc.of(from.capacity),
                                  acc.words * sizeof(uint64_t),
                                  cudaMemcpyDeviceToDevice,
                                  stream.get()));
    move_groups<<<blocks_for(from.capacity), threads_per_block, 0, stream.get()>>>(
      from.slots, from.capacity, from.acc, ref(num_values));
    CUDF_CHECK_CUDA(stream.get());
  }
};

// ---------------------------------------------------------------------------------------------
// packed_state
// ---------------------------------------------------------------------------------------------

bool packed_state::supported(std::vector<data_type> const& key_types,
                             std::vector<value_spec> const& values,
                             null_policy null_handling)
{
  if (key_types.empty() || key_types.size() > max_keys) { return false; }
  if (values.size() > max_values) { return false; }

  uint32_t bits = 0;
  for (auto const& type : key_types) {
    auto const cls = classify(type);
    if (!cls.ok || cls.src == source::float32 || cls.src == source::float64) { return false; }
    bits += 8 * cls.size;
  }
  if (null_handling == null_policy::INCLUDE) { bits += static_cast<uint32_t>(key_types.size()); }
  if (bits > 64) { return false; }

  for (auto const& value : values) {
    if (!classify_value(value).ok) { return false; }
    if (!is_count(value.kind)) {
      auto const target = cudf::detail::target_type(value.type, value.kind);
      if (!cudf::is_fixed_width(target)) { return false; }
      auto const size = cudf::size_of(target);
      if (size != 1 && size != 2 && size != 4 && size != 8) { return false; }
    }
  }
  return true;
}

packed_state::packed_state(std::vector<data_type> key_types,
                           std::vector<value_spec> values,
                           null_policy null_handling,
                           size_type initial_distinct_keys,
                           cuda::mr::any_resource<cuda::mr::device_accessible> mr,
                           cuda::stream_ref stream)
  : _key_types{std::move(key_types)},
    _values{std::move(values)},
    _null_handling{null_handling},
    _mr{std::move(mr)},
    _counters{2, stream, rmm::device_async_resource_ref{_mr}},
    _ever_null(_values.size(), false),
    _key_nullable(_key_types.size(), false)
{
  CUDF_EXPECTS(supported(_key_types, _values, _null_handling),
               "Internal error: packed groupby table used for unsupported keys or aggregates.");

  for (auto const& type : _key_types) {
    _key_shifts.push_back(_key_bits);
    _key_bits += 8 * classify(type).size;
  }

  _identities.count = static_cast<uint32_t>(_values.size());
  for (size_t i = 0; i < _values.size(); ++i) {
    auto const cls = classify_value(_values[i]);
    _sources.push_back(cls.src);
    _folds.push_back(fold_of(_values[i].kind, cls.src));
    _identities.values[i] = identity_of(_folds.back());
  }

  CUDF_CUDA_TRY(
    cudaMemsetAsync(_counters.data(), 0, _counters.size() * sizeof(size_type), stream.get()));

  auto const requested =
    static_cast<int64_t>(static_cast<double>(initial_distinct_keys) / max_load) + 1;
  _table = std::make_unique<group_table>(requested, _identities, _mr, stream);
}

packed_state::~packed_state() = default;

table_ref packed_state::ref() const { return _table->ref(static_cast<uint32_t>(_values.size())); }

counters packed_state::counter_ptrs() noexcept
{
  return counters{_counters.data(), _counters.data() + 1};
}

int64_t packed_state::room() const noexcept { return _table->room(_groups); }

void packed_state::read_counters(cuda::stream_ref stream)
{
  auto const host = cudf::detail::make_host_vector(device_span<size_type const>{_counters}, stream);
  _groups         = host[0];
  _sentinel_seen  = host[1] != 0;
}

void packed_state::ensure_room(size_type more, cuda::stream_ref stream)
{
  if (room() >= more) { return; }

  auto const needed    = static_cast<int64_t>(_groups) + more;
  auto const requested = std::max<int64_t>(
    static_cast<int64_t>(static_cast<double>(needed) / max_load) + 1, 2 * _table->capacity);
  CUDF_EXPECTS(requested <= max_slots,
               "Distinct key count (" + std::to_string(needed) +
                 ") would exceed the largest supported capacity.");

  auto larger = std::make_unique<group_table>(requested, _identities, _mr, stream);
  larger->take_groups_of(*_table, static_cast<uint32_t>(_values.size()), stream);

  // The old table is released on the host below, and it was not necessarily allocated on
  // `stream`, so the kernel that reads it has to be done first.
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  _table = std::move(larger);
}

void packed_state::enable_seen_tracking(cuda::stream_ref stream)
{
  if (_track_seen) { return; }
  mark_seen<<<blocks_for(_table->capacity + 1), threads_per_block, 0, stream.get()>>>(
    _table->slots,
    _table->capacity,
    _table->acc,
    static_cast<uint32_t>(_values.size()),
    _counters.data() + 1);
  CUDF_CHECK_CUDA(stream.get());
  _track_seen = true;
}

chunk packed_state::chunk_of(table_view const& keys,
                             table_view const& values,
                             bitmask_type const* row_bitmask,
                             size_type offset,
                             size_type rows) const
{
  chunk result;
  result.rows               = rows;
  result.row_bitmask        = row_bitmask;
  result.row_bitmask_offset = offset;
  result.track_seen         = _track_seen;

  result.keys.count              = static_cast<uint32_t>(_key_types.size());
  result.keys.nulls_as_keys      = _null_handling == null_policy::INCLUDE;
  auto const total_bits          = _key_bits + (result.keys.nulls_as_keys ? result.keys.count : 0);
  result.keys.may_equal_sentinel = total_bits == 64;
  for (uint32_t i = 0; i < result.keys.count; ++i) {
    auto const& column     = keys.column(i);
    auto const size        = static_cast<uint32_t>(cudf::size_of(column.type()));
    auto const row0        = static_cast<int64_t>(column.offset()) + offset;
    result.keys.columns[i] = key_layout{column.head<char>() + row0 * size,
                                        column.null_mask(),
                                        static_cast<size_type>(row0),
                                        size,
                                        _key_shifts[i],
                                        _key_bits + i};
    if (result.keys.nulls_as_keys) {
      result.keys.columns[i].null_mask = column.nullable() ? column.null_mask() : nullptr;
    } else {
      // Rows with null keys are skipped through `row_bitmask`; a null key's bytes are not read.
      result.keys.columns[i].null_mask = nullptr;
    }
  }

  result.values.count = static_cast<uint32_t>(_values.size());
  for (uint32_t i = 0; i < result.values.count; ++i) {
    auto const& column = values.column(i);
    auto const src     = _sources[i];
    auto const size =
      src == source::none ? 0u : static_cast<uint32_t>(cudf::size_of(column.type()));
    auto const row0 = static_cast<int64_t>(column.offset()) + offset;
    result.values.columns[i] =
      value_layout{src == source::none ? nullptr : column.head<char>() + row0 * size,
                   column.nullable() ? column.null_mask() : nullptr,
                   static_cast<size_type>(row0),
                   size,
                   src,
                   _folds[i],
                   _values[i].kind == aggregation::SUM_OF_SQUARES,
                   _values[i].kind != aggregation::COUNT_ALL};
  }
  return result;
}

void packed_state::aggregate(table_view const& keys,
                             table_view const& values,
                             cuda::stream_ref stream)
{
  auto const rows = keys.num_rows();
  if (rows == 0) { return; }
  CUDF_EXPECTS(static_cast<size_t>(keys.num_columns()) == _key_types.size() &&
                 static_cast<size_t>(values.num_columns()) == _values.size(),
               "Internal error: packed groupby table fed a batch of another shape.");

  auto needs_tracking = false;
  for (size_t i = 0; i < _values.size(); ++i) {
    if (values.column(i).has_nulls()) {
      _ever_null[i] = true;
      if (_values[i].kind != aggregation::COUNT_ALL) { needs_tracking = true; }
    }
  }
  if (needs_tracking) { enable_seen_tracking(stream); }
  for (size_t i = 0; i < _key_types.size(); ++i) {
    if (keys.column(i).nullable()) { _key_nullable[i] = true; }
  }

  auto const skip_null_keys = _null_handling == null_policy::EXCLUDE && cudf::has_nulls(keys);
  auto [bitmask_buffer, row_bitmask] =
    skip_null_keys
      ? detail::compute_row_bitmask(keys, stream)
      : std::pair<rmm::device_buffer, bitmask_type const*>{rmm::device_buffer{0, stream}, nullptr};

  for (size_type offset = 0; offset < rows;) {
    ensure_room(std::min(rows - offset, min_chunk_rows), stream);
    auto const n = static_cast<size_type>(std::min<int64_t>(rows - offset, room()));

    aggregate_rows<<<blocks_for(n), threads_per_block, 0, stream.get()>>>(
      ref(), chunk_of(keys, values, row_bitmask, offset, n), counter_ptrs());
    CUDF_CHECK_CUDA(stream.get());
    read_counters(stream);

    offset += n;
  }
}

void packed_state::merge(packed_state const& other, cuda::stream_ref stream)
{
  CUDF_EXPECTS(other._key_types == _key_types && other._values.size() == _values.size() &&
                 std::equal(_values.begin(),
                            _values.end(),
                            other._values.begin(),
                            [](value_spec const& a, value_spec const& b) {
                              return a.type == b.type && a.kind == b.kind;
                            }) &&
                 other._null_handling == _null_handling,
               "Cannot merge streaming_groupby objects with different key or value types.",
               std::invalid_argument);

  if (other._groups == 0 && !other._sentinel_seen) { return; }

  for (size_t i = 0; i < _values.size(); ++i) {
    if (other._ever_null[i]) { _ever_null[i] = true; }
  }
  for (size_t i = 0; i < _key_types.size(); ++i) {
    if (other._key_nullable[i]) { _key_nullable[i] = true; }
  }
  if (other._track_seen) { enable_seen_tracking(stream); }

  ensure_room(other._groups, stream);

  value_folds folds;
  folds.count = static_cast<uint32_t>(_folds.size());
  std::copy(_folds.begin(), _folds.end(), folds.folds);

  auto const source = other_table{other._table->slots,
                                  other._table->capacity,
                                  other._table->acc,
                                  other._counters.data() + 1,
                                  other._track_seen};
  merge_groups<<<blocks_for(other._table->capacity + 1), threads_per_block, 0, stream.get()>>>(
    ref(), source, folds, _track_seen, counter_ptrs());
  CUDF_CHECK_CUDA(stream.get());
  read_counters(stream);
}

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> packed_state::write_groups(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const num_regular = _groups;
  auto const num_groups  = this->num_groups();

  // The occupied slots, in slot order, so that the keys and the values come out in one order.
  rmm::device_uvector<size_type> group_slots(num_regular, stream, temp_mr);
  if (num_regular > 0) {
    auto const end =
      thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(static_cast<size_type>(_table->capacity)),
                      _table->slots,
                      group_slots.begin(),
                      is_occupied{});
    CUDF_EXPECTS(end - group_slots.begin() == num_regular,
                 "Internal error: occupied slots do not match the group count.");
  }

  output_layouts out;
  std::vector<std::unique_ptr<column>> key_columns;
  std::vector<rmm::device_uvector<uint8_t>> key_valid;
  out.num_keys = static_cast<uint32_t>(_key_types.size());
  for (uint32_t i = 0; i < out.num_keys; ++i) {
    key_columns.push_back(
      make_fixed_width_column(_key_types[i], num_groups, mask_state::UNALLOCATED, stream, mr));
    // With `null_policy::INCLUDE` a key's validity bit is the group's; with EXCLUDE every group
    // is valid, but the output column is nullable when the input was, as gathering it would be.
    auto const from_bits = _null_handling == null_policy::INCLUDE && _key_nullable[i];
    key_valid.emplace_back(from_bits ? num_groups : 0, stream, temp_mr);
    out.keys[i] = output_layout{key_columns.back()->mutable_view().head<char>(),
                                from_bits ? key_valid.back().data() : nullptr,
                                static_cast<uint32_t>(cudf::size_of(_key_types[i])),
                                _key_shifts[i],
                                _key_bits + i,
                                store::truncate};
  }

  std::vector<std::unique_ptr<column>> value_columns;
  std::vector<rmm::device_uvector<uint8_t>> value_valid;
  std::vector<bool> value_nullable;
  out.num_values = static_cast<uint32_t>(_values.size());
  for (uint32_t i = 0; i < out.num_values; ++i) {
    auto const target = cudf::detail::target_type(_values[i].type, _values[i].kind);
    value_columns.push_back(
      make_fixed_width_column(target, num_groups, mask_state::UNALLOCATED, stream, mr));
    // Nullable on the same terms as the generic path's results table: a final (not
    // intermediate) aggregate other than a count, over a column that has had nulls.
    auto const nullable = !_values[i].intermediate && !is_count(_values[i].kind) && _ever_null[i];
    value_nullable.push_back(nullable);
    value_valid.emplace_back(nullable ? num_groups : 0, stream, temp_mr);
    out.values[i] = output_layout{value_columns.back()->mutable_view().head<char>(),
                                  nullable ? value_valid.back().data() : nullptr,
                                  static_cast<uint32_t>(cudf::size_of(target)),
                                  0,
                                  0,
                                  store_of(target)};
  }

  if (num_groups > 0) {
    auto const groups = group_list{group_slots.data(), num_regular, _sentinel_seen};
    write_groups_kernel<<<blocks_for(num_groups), threads_per_block, 0, stream.get()>>>(
      groups, ref(), out);
    CUDF_CHECK_CUDA(stream.get());

    for (uint32_t i = 0; i < out.num_keys; ++i) {
      if (out.keys[i].valid != nullptr) {
        auto [mask, null_count] = cudf::detail::valid_if(
          key_valid[i].begin(), key_valid[i].end(), is_nonzero{}, stream, mr);
        key_columns[i]->set_null_mask(std::move(mask), null_count);
      } else if (_key_nullable[i]) {
        key_columns[i]->set_null_mask(
          cudf::create_null_mask(num_groups, mask_state::ALL_VALID, stream, mr), 0);
      }
    }
    for (uint32_t i = 0; i < out.num_values; ++i) {
      if (!value_nullable[i]) { continue; }
      auto [mask, null_count] = cudf::detail::valid_if(
        value_valid[i].begin(), value_valid[i].end(), is_nonzero{}, stream, mr);
      value_columns[i]->set_null_mask(std::move(mask), null_count);
    }
  }

  return {std::make_unique<table>(std::move(key_columns)),
          std::make_unique<table>(std::move(value_columns))};
}

}  // namespace cudf::groupby::packed
