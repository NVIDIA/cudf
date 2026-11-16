/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/filling.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr cudf::size_type benchmark_rows = 100'000;

using key_scalar = cudf::numeric_scalar<cudf::size_type>;

std::unique_ptr<cudf::column> make_right_batch(cudf::size_type offset,
                                               cudf::size_type rows,
                                               bool duplicate_heavy)
{
  if (!duplicate_heavy) { return cudf::sequence(rows, key_scalar{offset}); }
  // Keep 90% of build rows at key zero. Probing [0, 100k) produces a representative
  // duplicate-heavy result without a quadratic output.
  auto const keys = cudf::sequence(rows, key_scalar{offset - (benchmark_rows * 9 / 10 - 1)});
  return cudf::clamp(keys->view(), key_scalar{0}, key_scalar{0, false});
}

std::unique_ptr<cudf::column> make_probe() { return cudf::sequence(benchmark_rows, key_scalar{0}); }

}  // namespace

void nvbench_streaming_hash_join(nvbench::state& state)
{
  auto const scenario        = state.get_string("scenario");
  auto const duplicate_heavy = scenario == "duplicate_heavy";
  auto const interleaved     = scenario == "interleaved";
  auto const repeated_probe  = scenario == "repeated_probe";
  auto const batch_count     = scenario == "single_batch" || repeated_probe ? 1 : 16;
  auto const batch_rows      = benchmark_rows / batch_count;

  std::vector<std::unique_ptr<cudf::column>> right_columns;
  right_columns.reserve(batch_count);
  std::vector<cudf::table_view> right_batches;
  right_batches.reserve(batch_count);
  for (auto batch = cudf::size_type{0}; batch < batch_count; ++batch) {
    right_columns.push_back(make_right_batch(batch * batch_rows, batch_rows, duplicate_heavy));
    right_batches.emplace_back(cudf::table_view{{right_columns.back()->view()}});
  }
  auto probe                 = make_probe();
  auto const probe_keys      = cudf::table_view{{probe->view()}};
  auto const right_key_index = std::array<cudf::size_type, 1>{0};
  auto const input_bytes = static_cast<std::size_t>(benchmark_rows) * sizeof(cudf::size_type) * 2;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_element_count(input_bytes, "input_bytes");
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (repeated_probe) {
    cudf::streaming_hash_join joiner{right_batches.front(),
                                     right_key_index,
                                     benchmark_rows,
                                     /*max_num_batches=*/1,
                                     cudf::nullable_join::NO,
                                     cudf::null_equality::UNEQUAL,
                                     /*load_factor=*/0.5};
    joiner.insert(right_batches.front());
    // Seal outside the timed operation, then measure stable repeated probes.
    static_cast<void>(joiner.inner_join(probe_keys));
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch&) { auto result = joiner.inner_join(probe_keys); });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
    return;
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::streaming_hash_join joiner{right_batches.front(),
                                     right_key_index,
                                     benchmark_rows,
                                     batch_count,
                                     cudf::nullable_join::NO,
                                     cudf::null_equality::UNEQUAL,
                                     /*load_factor=*/0.5};
    joiner.insert(right_batches.front());
    if (interleaved) {
      static_cast<void>(joiner.inner_join(probe_keys));
      for (auto batch = 1; batch < batch_count; ++batch) {
        joiner.insert(right_batches[batch]);
        static_cast<void>(joiner.inner_join(probe_keys));
      }
    } else {
      for (auto batch = 1; batch < batch_count; ++batch) {
        joiner.insert(right_batches[batch]);
      }
      auto result = joiner.inner_join(probe_keys);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(nvbench_streaming_hash_join)
  .set_name("streaming_hash_join")
  .add_string_axis(
    "scenario",
    {"single_batch", "sixteen_batches", "duplicate_heavy", "repeated_probe", "interleaved"});
