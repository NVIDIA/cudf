/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

// Mixed-type benches: the routed batch_gather entry vs one stock `cudf::gather` call on the
// SAME mixed table (stock's internal per-column launches are the comparison point). The
// `column_mix` axis selects the schema of the non-fixed-width part:
//   0 = strings only                     1 = strings + lists<int64>
//   2 = strings + structs{int64,double}  3 = dictionary<int64>(int64 keys) + strings
// Column roles are positional and deterministic: every index with `i % 5 < 2` is a
// non-fixed-width slot (40% of the columns exactly when num_cols is a multiple of 5 — 40/60
// for num_cols=100, 50/50 for num_cols=8), the remaining indices cycle int64/double/int8.
// Within the non-fixed slots, even ranks are strings and odd ranks are the mix's nested type
// (mix 0 keeps every non-fixed slot strings).

namespace {

constexpr cudf::size_type strings_only_mix    = 0;
constexpr cudf::size_type strings_lists_mix   = 1;
constexpr cudf::size_type strings_structs_mix = 2;
constexpr cudf::size_type strings_dict_mix    = 3;

// All column owners returned here MUST be held by the caller through the nvbench exec lambda:
// the table_view/column_view vectors are non-owning, and the exec lambda runs inside the bench
// function's scope where the owners (this returned vector plus the map owner) are still alive.
std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>>
build_mixed_table(cudf::size_type num_rows, cudf::size_type num_cols, cudf::size_type column_mix)
{
  auto const stream    = cudf::get_default_stream();
  auto const mr        = cudf::get_current_device_resource_ref();
  auto const valid_all = cudf::test::iterators::no_nulls();

  // One hoisted iterator per lambda (a single closure type): two separately-written
  // make_counting_transform_iterator expressions are two DISTINCT closure types and break
  // (InputIterator, InputIterator) deduction. End bounds derive by addition.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(num_cols);
  cudf::size_type nonfixed_rank = 0;
  cudf::size_type fixed_rank    = 0;
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    if (i % 5 < 2) {
      auto const is_nested = (column_mix != strings_only_mix) and (nonfixed_rank % 2 == 1);
      ++nonfixed_rank;
      if (not is_nested) {
        // Strings: generated device-side (scales to any num_rows; identical for both benches).
        columns.push_back(create_random_column(cudf::type_id::STRING, row_count{num_rows}));
        continue;
      }
      switch (column_mix) {
        case strings_lists_mix: {
          // Every row is a 2-element list<int64>; nullable parent mask (ALL_VALID), nullable
          // zero-null child, offsets dense via sequence.
          auto const elems = cudf::detail::make_counting_transform_iterator(
            0, [](cudf::size_type e) { return static_cast<int64_t>(e); });
          auto child = cudf::test::fixed_width_column_wrapper<int64_t>(
            elems, elems + 2 * num_rows, valid_all, stream, mr);
          auto offsets = cudf::sequence(num_rows + 1,
                                        cudf::numeric_scalar<cudf::size_type>(0),
                                        cudf::numeric_scalar<cudf::size_type>(2));
          columns.push_back(cudf::make_lists_column(
            num_rows,
            std::move(offsets),
            child.release(),
            0,
            cudf::create_null_mask(num_rows, cudf::mask_state::ALL_VALID, stream, mr)));
          break;
        }
        case strings_structs_mix: {
          // Struct{int64,double} with a 10%-null parent and nullable zero-null children.
          auto const int_elems = cudf::detail::make_counting_transform_iterator(
            0, [](cudf::size_type e) { return static_cast<int64_t>(e); });
          auto int_child = cudf::test::fixed_width_column_wrapper<int64_t>(
            int_elems, int_elems + num_rows, valid_all, stream, mr);
          auto const dbl_elems = cudf::detail::make_counting_transform_iterator(
            0, [](cudf::size_type e) { return static_cast<double>(e); });
          auto dbl_child = cudf::test::fixed_width_column_wrapper<double>(
            dbl_elems, dbl_elems + num_rows, valid_all, stream, mr);
          auto const parent_validity = cudf::detail::make_counting_transform_iterator(
            0, [](cudf::size_type r) { return r % 10 != 0; });
          columns.push_back(
            cudf::test::structs_column_wrapper({int_child, dbl_child}, parent_validity, stream, mr)
              .release());
          break;
        }
        case strings_dict_mix: {
          // dictionary<int64>(int64 keys): low-cardinality int64 payload encoded in place.
          auto const payload_elems = cudf::detail::make_counting_transform_iterator(
            0, [](cudf::size_type e) { return static_cast<int64_t>(e % 7); });
          auto payload = cudf::test::fixed_width_column_wrapper<int64_t>(
            payload_elems, payload_elems + num_rows, valid_all, stream, mr);
          columns.push_back(
            cudf::dictionary::encode(payload, cudf::data_type{cudf::type_id::INT32}, stream, mr));
          break;
        }
        default: CUDF_FAIL("Unhandled column_mix");
      }
      continue;
    }
    // Fixed-width slots cycle int64/double/int8 (all non-null, like the single-table benches).
    switch (fixed_rank % 3) {
      case 0: {
        auto const elems = cudf::detail::make_counting_transform_iterator(
          0, [](cudf::size_type e) { return static_cast<int64_t>(e); });
        columns.push_back(cudf::test::fixed_width_column_wrapper<int64_t>(
                            elems, elems + num_rows, valid_all, stream, mr)
                            .release());
        break;
      }
      case 1: {
        auto const elems = cudf::detail::make_counting_transform_iterator(
          0, [](cudf::size_type e) { return static_cast<double>(e); });
        columns.push_back(cudf::test::fixed_width_column_wrapper<double>(
                            elems, elems + num_rows, valid_all, stream, mr)
                            .release());
        break;
      }
      default: {
        auto const elems = cudf::detail::make_counting_transform_iterator(
          0, [](cudf::size_type e) { return static_cast<int8_t>(e); });
        columns.push_back(cudf::test::fixed_width_column_wrapper<int8_t>(
                            elems, elems + num_rows, valid_all, stream, mr)
                            .release());
        break;
      }
    }
    ++fixed_rank;
  }

  std::vector<cudf::column_view> views;
  views.reserve(columns.size());
  for (auto const& col : columns) {
    views.push_back(col->view());
  }
  return {std::move(views), std::move(columns)};
}

// Map datagen identical to the single/multi-table benches: reversed-sequence coalesced or
// uniform random, non-null, size_type.
// Recursive byte accounting: fixed-width columns via `size_of`; strings/lists/structs via
// their children (chars + offsets, list children, struct fields). `size_of` throws on
// non-fixed-width types, so a type-blind sum crashes mixed tables.
// Sliced views (offset != 0) would overcount here: child sizes cover the full parent column,
// not the offset window. Unreachable today — the mixed benches build unsliced columns — but do
// not reuse this helper on sliced inputs without accounting for the offset.
static long column_bytes(cudf::column_view const& cv)
{
  long bytes =
    cudf::is_fixed_width(cv.type()) ? static_cast<long>(cudf::size_of(cv.type())) * cv.size() : 0;
  for (cudf::size_type i = 0; i < cv.num_children(); ++i) {
    bytes += column_bytes(cv.child(i));
  }
  return bytes;
}

std::unique_ptr<cudf::column> make_gather_map(cudf::size_type num_rows, bool coalesce)
{
  if (coalesce) {
    return cudf::sequence(num_rows,
                          cudf::numeric_scalar<cudf::size_type>(num_rows - 1),
                          cudf::numeric_scalar<cudf::size_type>(-1));
  }
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, num_rows - 1);
  return create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
}

}  // namespace

static void bench_batch_gather_mixed(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols   = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const coalesce   = static_cast<bool>(state.get_int64("coalesce"));
  auto const column_mix = static_cast<cudf::size_type>(state.get_int64("column_mix"));

  if (static_cast<long>(num_rows) * static_cast<long>(num_cols) >
      std::numeric_limits<int32_t>::max()) {  // reasonable limit
    state.skip("input benchmark size too large");
    return;
  }

  // Owners (`owners` + `gather_map`) live to the END of this function; the exec lambda runs
  // inside that scope, so the non-owning views cannot dangle.
  auto [column_views, owners] = build_mixed_table(num_rows, num_cols, column_mix);
  auto gather_map             = make_gather_map(num_rows, coalesce);
  cudf::table_view const source_table(column_views);

  auto const source_bytes =
    std::accumulate(column_views.begin(),
                    column_views.end(),
                    static_cast<long>(0),
                    [](long acc, cudf::column_view const& cv) { return acc + column_bytes(cv); });

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int8_t>(source_bytes);
  state.add_global_memory_writes<int8_t>(source_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::detail::batch_gather(source_table,
                               gather_map->view(),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               cudf::negative_index_policy::NOT_ALLOWED,
                               stream,
                               cudf::memory_resources{cudf::get_current_device_resource_ref()});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_gather_mixed(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols   = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const coalesce   = static_cast<bool>(state.get_int64("coalesce"));
  auto const column_mix = static_cast<cudf::size_type>(state.get_int64("column_mix"));

  if (static_cast<long>(num_rows) * static_cast<long>(num_cols) >
      std::numeric_limits<int32_t>::max()) {  // reasonable limit
    state.skip("input benchmark size too large");
    return;
  }

  // Owners outlive the exec lambda (same discipline as bench_batch_gather_mixed).
  auto [column_views, owners] = build_mixed_table(num_rows, num_cols, column_mix);
  auto gather_map             = make_gather_map(num_rows, coalesce);
  cudf::table_view const source_table(column_views);

  auto const source_bytes =
    std::accumulate(column_views.begin(),
                    column_views.end(),
                    static_cast<long>(0),
                    [](long acc, cudf::column_view const& cv) { return acc + column_bytes(cv); });

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int8_t>(source_bytes);
  state.add_global_memory_writes<int8_t>(source_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::gather(source_table, gather_map->view()); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_batch_gather_mixed)
  .set_name("batch_gather_mixed")
  .add_int64_axis("num_rows", {65536, 262144, 1048576})
  .add_int64_axis("num_cols", {8, 100})
  .add_int64_axis("coalesce", {0, 1})
  .add_int64_axis("column_mix", {0, 1, 2, 3});

NVBENCH_BENCH(bench_gather_mixed)
  .set_name("gather_mixed")
  .add_int64_axis("num_rows", {65536, 262144, 1048576})
  .add_int64_axis("num_cols", {8, 100})
  .add_int64_axis("coalesce", {0, 1})
  .add_int64_axis("column_mix", {0, 1, 2, 3});
