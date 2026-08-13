/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

// Copies a benchmark column's offsets (INT32 or INT64) to a pinned host int64 vector: `count + 1`
// entries. The pinned vector makes later H2D copies sourced from it genuinely asynchronous. The
// offsetalator's device-only iterators are unavailable in this host-only TU, so the width
// dispatch stays explicit.
static cudf::detail::host_vector<int64_t> offsets_to_host_int64(cudf::column_view const& offsets_cv,
                                                                cudf::size_type count,
                                                                rmm::cuda_stream_view stream)
{
  auto offsets =
    cudf::detail::make_pinned_vector_async<int64_t>(static_cast<std::size_t>(count) + 1, stream);
  if (offsets_cv.type().id() == cudf::type_id::INT64) {
    auto const host_offsets = cudf::detail::make_pinned_vector<int64_t>(
      cudf::device_span<int64_t const>{offsets_cv.data<int64_t>(),
                                       static_cast<std::size_t>(count) + 1},
      stream);
    std::copy(host_offsets.begin(), host_offsets.end(), offsets.begin());
  } else {
    auto const host_offsets = cudf::detail::make_pinned_vector<cudf::size_type>(
      cudf::device_span<cudf::size_type const>{offsets_cv.data<cudf::size_type>(),
                                               static_cast<std::size_t>(count) + 1},
      stream);
    std::copy(host_offsets.begin(), host_offsets.end(), offsets.begin());
  }
  return offsets;
}

// The list generator forces its final offset to the child size, so the last row absorbs every
// leftover element -- an artifact often far above `max_list_size`. The numeric gates average over
// the whole column, so routing cannot flip on one row, but the trim (untimed) keeps the timed
// input inside the declared [0, max_list_size] regime, matching the strings benchmark. Returns
// nullptr when the last row already fits.
static std::unique_ptr<cudf::column> trim_forced_last_row(cudf::column_view const& list_col,
                                                          cudf::size_type max_list_size,
                                                          rmm::cuda_stream_view stream)
{
  auto const mr       = cudf::get_current_device_resource_ref();
  auto const lists    = cudf::lists_column_view{list_col};
  auto const num_rows = lists.size();

  auto const offsets_cv = lists.offsets();
  auto const offsets    = offsets_to_host_int64(offsets_cv, num_rows, stream);
  if (num_rows == 0 or offsets[num_rows] - offsets[num_rows - 1] <= max_list_size) {
    return nullptr;
  }

  auto new_offsets          = offsets;
  new_offsets[num_rows]     = new_offsets[num_rows - 1] + max_list_size;
  auto const new_child_size = static_cast<cudf::size_type>(new_offsets[num_rows]);

  // Rebuild offsets at the input's own width to keep the generator's offset type. Both staging
  // buffers are pinned; `new_offsets` must outlive the trailing synchronize since the INT64
  // branch's H2D copy reads it asynchronously, while the INT32 branch's buffer does not outlive
  // this scope and so synchronizes immediately.
  std::unique_ptr<cudf::column> offsets_col;
  if (offsets_cv.type().id() == cudf::type_id::INT64) {
    offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT64},
      num_rows + 1,
      rmm::device_buffer{new_offsets.data(), new_offsets.size() * sizeof(int64_t), stream, mr},
      rmm::device_buffer{},
      0);
  } else {
    auto new_offsets_i32 =
      cudf::detail::make_pinned_vector_async<cudf::size_type>(new_offsets.size(), stream);
    std::copy(new_offsets.begin(), new_offsets.end(), new_offsets_i32.begin());
    offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      num_rows + 1,
      rmm::device_buffer{
        new_offsets_i32.data(), new_offsets_i32.size() * sizeof(cudf::size_type), stream, mr},
      rmm::device_buffer{},
      0);
    stream.synchronize();
  }
  auto new_leaf =
    std::make_unique<cudf::column>(cudf::slice(lists.child(), {0, new_child_size})[0], stream, mr);
  auto result = cudf::make_lists_column(num_rows,
                                        std::move(offsets_col),
                                        std::move(new_leaf),
                                        list_col.null_count(),
                                        cudf::copy_bitmask(list_col, stream, mr));
  stream.synchronize();
  return result;
}

// Benchmarks cudf::lists::sort_lists on LIST<numeric>, the operation Spark array_sort lowers to.
// Axis values straddle the fast-path routing crossovers; per-axis rationale at the registrations.
template <typename Type>
void bench_sort_list_of_numbers(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_list_size  = static_cast<cudf::size_type>(state.get_int64("max_list_size"));
  auto const null_frequency = state.get_float64("null_frequency");
  auto const column_order =
    state.get_string("order") == "DESC" ? cudf::order::DESCENDING : cudf::order::ASCENDING;
  auto const null_precedence =
    state.get_string("null_order") == "BEFORE" ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

  // Skip states whose worst-case leaf count overflows int32 offsets.
  if (static_cast<double>(num_rows) * max_list_size >
      static_cast<double>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
    return;
  }

  // Nulls apply at both the list-row and leaf levels; cardinality 0 leaves distinct counts
  // uncapped. null_frequency == 0 must pass std::nullopt: a literal null_probability(0.0) still
  // samples per-element validity, and thrust's uniform_real<float> endpoint rounding injects stray
  // nulls past ~2^24 leaf elements, routing the "no-null" cell through the has-nulls path.
  auto const null_prob =
    null_frequency > 0 ? std::optional<double>{null_frequency} : std::optional<double>{};
  data_profile profile =
    data_profile_builder()
      .list_type(cudf::type_to_id<Type>())
      .list_depth(1)
      .distribution(cudf::type_id::LIST, distribution_id::UNIFORM, 0, max_list_size)
      .null_probability(null_prob)
      .cardinality(0);
  // The bound type selects the set_distribution_params overload and must match the leaf type -- a
  // mismatched call is a silent no-op.
  if constexpr (cudf::is_floating_point<Type>()) {
    profile.set_distribution_params(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0., 1'000'000.);
  } else if constexpr (cudf::is_fixed_point<Type>()) {
    // dec_regime picks the leaf magnitude regime: u64 (< 2^32), key96 (within int64), twophase
    // (past int64). __int128 bounds make >2^63 expressible.
    auto const regime = state.get_string("dec_regime");
    auto const hi     = regime == "twophase" ? (static_cast<__int128_t>(1) << 64)
                        : regime == "key96"  ? (static_cast<__int128_t>(1) << 40)
                                             : static_cast<__int128_t>(1'000'000);
    profile.set_distribution_params(cudf::type_to_id<Type>(),
                                    distribution_id::UNIFORM,
                                    static_cast<__int128_t>(0),
                                    hi,
                                    numeric::scale_type{0});
  } else {
    profile.set_distribution_params(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 1'000'000);
  }

  auto const table = create_random_table({cudf::type_id::LIST}, row_count{num_rows}, profile);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const trimmed  = trim_forced_last_row(table->view().column(0), max_list_size, stream);
  auto const base_col = trimmed ? trimmed->view() : table->view().column(0);
  auto const input    = cudf::lists_column_view{base_col};
  stream.synchronize();  // Ensure untimed setup completes before timing.

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::lists::sort_lists(
      input, column_order, null_precedence, stream, cudf::get_current_device_resource_ref());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

// Chrono is omitted: its packed key is byte-identical to the int32/int64 rep it extracts to, so
// those cells already measure it. decimal128 exercises the range-gated radix keys (8-byte
// min-biased / prefix_key96 / two-phase hi64/lo64) and needs its own type-string mapping.
NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal128, "decimal128", "decimal128");

NVBENCH_BENCH_TYPES(
  bench_sort_list_of_numbers,
  NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, std::int64_t, float, double>))
  .set_name("sort_list_of_numbers")
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  // 4: tiny lists (tiered network/warp tiers); 32: the mid band where no-null routing splits by
  // type between CUB DeviceSegmentedSort and the tiered kernel; 256: past the packed-radix cutoff.
  .add_int64_axis("max_list_size", {4, 16, 32, 64, 128, 256})
  // The null-bearing case routes every supported type to the tiered kernel.
  .add_float64_axis("null_frequency", {0, 0.1})
  // The fast path folds (order, null_order) into its keys, so all four combinations exercise it.
  .add_string_axis("order", {"ASC", "DESC"})
  .add_string_axis("null_order", {"AFTER", "BEFORE"});

// decimal128 registers separately to carry the dec_regime axis; other axes mirror the bench above.
NVBENCH_BENCH_TYPES(bench_sort_list_of_numbers,
                    NVBENCH_TYPE_AXES(nvbench::type_list<numeric::decimal128>))
  .set_name("sort_list_of_decimal128")
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("max_list_size", {4, 16, 32, 64, 128, 256})
  // Leaf magnitude classes -- span < 2^32, within int64, past int64 -- bounds set in the body.
  .add_string_axis("dec_regime", {"u64", "key96", "twophase"})
  .add_float64_axis("null_frequency", {0, 0.1})
  .add_string_axis("order", {"ASC", "DESC"})
  .add_string_axis("null_order", {"AFTER", "BEFORE"});
