/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

// Copies a benchmark column's offsets (INT32 or INT64) to host int64 -- `count + 1` entries --
// through the input offsetalator, cudf's canonical width-erased view of an offsets column. The
// pinned return vector also makes later H2D copies sourced from it genuinely asynchronous.
static cudf::detail::host_vector<int64_t> offsets_to_host_int64(cudf::column_view const& offsets_cv,
                                                                cudf::size_type count,
                                                                rmm::cuda_stream_view stream)
{
  auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets_cv);
  auto widened         = rmm::device_uvector<int64_t>(static_cast<std::size_t>(count) + 1, stream);
  thrust::copy(rmm::exec_policy_nosync(stream), d_offsets, d_offsets + count + 1, widened.begin());
  return cudf::detail::make_pinned_vector(
    cudf::device_span<int64_t const>{widened.data(), widened.size()}, stream);
}

// Overwrites the first `min(plen, byte_length)` bytes of every non-null leaf string with 'A',
// returning a new LIST<STRING> column: strings stay distinct in their tails but share a
// `plen`-byte prefix -- the regime where the packed-prefix key is uninformative and the tie-break
// does the real work. Untimed setup, so a host round-trip is fine.
// Assumes an unsliced leaf, which holds for the freshly generated column.
static std::unique_ptr<cudf::column> apply_shared_prefix(cudf::column_view const& list_col,
                                                         cudf::size_type plen,
                                                         rmm::cuda_stream_view stream)
{
  auto const mr = cudf::get_current_device_resource_ref();

  auto const lists    = cudf::lists_column_view{list_col};
  auto const scv      = cudf::strings_column_view{lists.child()};
  auto const num_leaf = scv.size();

  auto const chars_bytes = static_cast<std::size_t>(scv.chars_size(stream));
  auto host_chars        = cudf::detail::make_pinned_vector<char>(
    cudf::device_span<char const>{scv.chars_begin(stream), chars_bytes}, stream);

  auto const offsets_cv = scv.offsets();
  auto const offsets    = offsets_to_host_int64(offsets_cv, num_leaf, stream);

  std::vector<cudf::bitmask_type> host_mask;
  if (scv.null_mask() != nullptr) {
    auto const host_mask_words = cudf::detail::make_pinned_vector<cudf::bitmask_type>(
      cudf::device_span<cudf::bitmask_type const>{
        scv.null_mask(), static_cast<std::size_t>(cudf::num_bitmask_words(num_leaf))},
      stream);
    host_mask.assign(host_mask_words.begin(), host_mask_words.end());
  }
  auto const* mask_ptr = host_mask.empty() ? nullptr : host_mask.data();

  for (cudf::size_type i = 0; i < num_leaf; ++i) {
    if (!cudf::bit_value_or(mask_ptr, i, true)) { continue; }  // Skip null elements.
    auto const len = offsets[i + 1] - offsets[i];
    auto const k   = std::min<int64_t>(plen, len);
    std::fill_n(host_chars.begin() + offsets[i], k, 'A');
  }

  auto new_chars = rmm::device_buffer{host_chars.data(), chars_bytes, stream, mr};
  auto new_leaf  = cudf::make_strings_column(num_leaf,
                                            std::make_unique<cudf::column>(offsets_cv, stream, mr),
                                            std::move(new_chars),
                                            scv.null_count(),
                                            cudf::copy_bitmask(scv.parent(), stream, mr));

  auto result = cudf::make_lists_column(lists.size(),
                                        std::make_unique<cudf::column>(lists.offsets(), stream, mr),
                                        std::move(new_leaf),
                                        list_col.null_count(),
                                        cudf::copy_bitmask(list_col, stream, mr));

  // The async H2D chars copy reads `host_chars`; synchronize before it goes out of scope.
  stream.synchronize();
  return result;
}

// The list generator forces its final offset to the child size, so the last row absorbs every
// leftover element -- an artifact often far above `max_list_size` that would disqualify the whole
// column from the graduated-warp path and silently demote it to the prefix path. Trimming that row
// (untimed) restores the declared [0, max_list_size] regime. Returns nullptr when the last row
// already fits. The generator purges nonempty nulls, so the trim cannot create a nonempty null.
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

  auto new_offsets      = offsets;
  new_offsets[num_rows] = new_offsets[num_rows - 1] + max_list_size;
  // The trim only shrinks the final offset below the existing child size, so the size_type
  // narrowings here and inside the output offsetalator cannot overflow; the guard makes any
  // future violation fail loudly instead of wrapping.
  CUDF_EXPECTS(new_offsets[num_rows] <= std::numeric_limits<cudf::size_type>::max(),
               "trimmed final offset exceeds size_type range");
  auto const new_child_size = static_cast<cudf::size_type>(new_offsets[num_rows]);

  // Rebuild offsets at the input's own width: H2D the widened offsets (pinned, so the copy is
  // async), then store through the output offsetalator, which narrows to the source type on
  // device. The host and device staging must outlive the trailing synchronize.
  auto offsets_col = cudf::make_numeric_column(
    offsets_cv.type(), num_rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const d_new_offsets = cudf::detail::make_device_uvector_async(new_offsets, stream, mr);
  auto const d_out_offsets =
    cudf::detail::offsetalator_factory::make_output_iterator(offsets_col->mutable_view());
  thrust::copy(
    rmm::exec_policy_nosync(stream), d_new_offsets.begin(), d_new_offsets.end(), d_out_offsets);
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

// Benchmarks cudf::lists::sort_lists on LIST<STRING>, the operation Spark array_sort lowers to and
// which a plain table sort does not exercise; per-axis rationale at the registration below.
static void bench_sort_list_of_strings(nvbench::state& state)
{
  auto const num_rows          = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_list_size     = static_cast<cudf::size_type>(state.get_int64("max_list_size"));
  auto const row_width         = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const null_frequency    = state.get_float64("null_frequency");
  auto const shared_prefix_len = static_cast<cudf::size_type>(state.get_int64("shared_prefix_len"));
  auto const column_order =
    state.get_string("order") == "DESC" ? cudf::order::DESCENDING : cudf::order::ASCENDING;
  auto const null_precedence =
    state.get_string("null_order") == "BEFORE" ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

  // Skip when estimated leaf chars exceed the int32 char cap.
  auto const estimated_chars =
    static_cast<double>(num_rows) * (max_list_size / 2.0) * (row_width / 2.0);
  if (estimated_chars > static_cast<double>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
    return;
  }

  // Leaf strings are all-distinct (cardinality 0) to match the near-unique elements of real
  // array_sort workloads -- the hardest tie-break case, stressed further by shared_prefix_len --
  // so leaf cardinality is not a separate axis. Nulls apply at both the list-row and leaf levels.
  // null_frequency == 0 must pass std::nullopt: a literal null_probability(0.0) still samples
  // per-element validity, and thrust's uniform_real<float> endpoint rounding injects stray nulls
  // past ~2^24 leaf elements, routing the "no-null" cell through the has-nulls path.
  auto const null_prob =
    null_frequency > 0 ? std::optional<double>{null_frequency} : std::optional<double>{};
  data_profile const profile =
    data_profile_builder()
      .list_type(cudf::type_id::STRING)
      .list_depth(1)
      .distribution(cudf::type_id::LIST, distribution_id::UNIFORM, 0, max_list_size)
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width)
      .null_probability(null_prob)
      .cardinality(0);

  auto const table = create_random_table({cudf::type_id::LIST}, row_count{num_rows}, profile);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const trimmed  = trim_forced_last_row(table->view().column(0), max_list_size, stream);
  auto const base_col = trimmed ? trimmed->view() : table->view().column(0);

  auto const prefixed =
    shared_prefix_len > 0 ? apply_shared_prefix(base_col, shared_prefix_len, stream) : nullptr;
  auto const input = cudf::lists_column_view{prefixed ? prefixed->view() : base_col};
  stream.synchronize();  // Ensure untimed setup completes before timing.

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::lists::sort_lists(
      input, column_order, null_precedence, stream, cudf::get_current_device_resource_ref());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_sort_list_of_strings)
  .set_name("sort_list_of_strings")
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  // Tiny lists stress per-segment overhead; large ones amortize it.
  .add_int64_axis("max_list_size", {4, 16, 32, 64, 128, 256})
  // Short strings mostly fit the packed key; wide strings push work into the byte-window tie-break.
  .add_int64_axis("row_width", {16, 64})
  // 0: control; 8: exactly the packed-key width (first tie-break window); 32: several windows.
  // 96: exceeds both row widths, so bytes never differ and compares resolve by length alone.
  // Shared prefixes are the realistic regime for keyed array_sort data.
  .add_int64_axis("shared_prefix_len", {0, 8, 32, 96})
  .add_float64_axis("null_frequency", {0, 0.1})
  // The fast path folds (order, null_order) into its keys, so all four combinations exercise it.
  .add_string_axis("order", {"ASC", "DESC"})
  .add_string_axis("null_order", {"AFTER", "BEFORE"});
