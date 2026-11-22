/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "lto_benchmark.hpp"

#include <benchmarks/common/generate_input.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf_reduce_lto_benchmark_fragments.hpp>

#include <memory>
#include <numeric>
#include <utility>

namespace {
using columns      = std::vector<std::unique_ptr<cudf::column>>;
using host_columns = std::vector<std::vector<std::optional<int64_t>>>;

struct shape {
  char const* name;
  cudf::size_type segments;
  cudf::size_type length;
  bool skewed;
};

std::vector<cudf::size_type> make_offsets(shape const& dimensions)
{
  auto const rows = dimensions.segments * dimensions.length;
  std::vector<cudf::size_type> lengths(dimensions.segments, dimensions.length);
  if (dimensions.skewed) {
    std::fill(lengths.begin(), lengths.end(), 0);
    lengths.front() = static_cast<int64_t>(rows) * 9 / 10;
    std::vector<cudf::size_type> active;
    for (cudf::size_type index = 1; index < dimensions.segments; ++index) {
      if (index % 8 != 0) { active.push_back(index); }
    }
    auto const remaining = rows - lengths.front();
    for (size_t index = 0; index < active.size(); ++index) {
      lengths[active[index]] = remaining / active.size() + (index < remaining % active.size());
    }
  }
  std::vector<cudf::size_type> offsets(dimensions.segments + 1);
  std::partial_sum(lengths.begin(), lengths.end(), offsets.begin() + 1);
  return offsets;
}

cudf::reduce_udf make_udf(bool fused)
{
  auto const range = cudf_reduce_lto_benchmark_fragments::file_ranges
    [cudf_reduce_lto_benchmark_fragments::segmented_sum_max];
  static cudf::lto_fragment const fragments[]{
    {cudf::lto_binary_type::FATBIN,
     cudf_reduce_lto_benchmark_fragments::files.subspan(range[0], range[1])}};
  if (fused) {
    return {fragments,
            "segmented_sum_max_init",
            "segmented_sum_max_update",
            "segmented_sum_max_merge",
            "segmented_sum_max_finalize"};
  }
  return {fragments,
          "segmented_sum_init",
          "segmented_sum_update",
          "segmented_sum_merge",
          "segmented_sum_finalize"};
}
}  // namespace

namespace reduction_benchmark {
void run_segmented(options const& settings, std::ostream& output)
{
  auto const shapes = settings.smoke ? std::vector<shape>{{"small", 32, 16, false},
                                                          {"large", 4, 1024, false},
                                                          {"uneven", 16, 1024, true}}
                                     : std::vector<shape>{{"small", 4096, 16, false},
                                                          {"large", 32, 1 << 20, false},
                                                          {"uneven", 256, 1 << 16, true}};
  auto const stream = cudf::get_default_stream();
  auto const type   = cudf::data_type{cudf::type_id::INT64};
  for (auto const& dimensions : shapes) {
    auto host_offsets   = make_offsets(dimensions);
    auto device_offsets = cudf::detail::make_device_buffer(
      host_offsets, stream, cudf::get_current_device_resource_ref());
    cudf::device_span<cudf::size_type const> offsets{device_offsets.data(), device_offsets.size()};
    cudf::size_type longest = 0;
    for (size_t index = 1; index < host_offsets.size(); ++index) {
      longest = std::max(longest, host_offsets[index] - host_offsets[index - 1]);
    }
    auto const rows = host_offsets.back();
    auto const nulls =
      settings.smoke ? std::vector<int32_t>{0, 10, 100} : std::vector<int32_t>{0, 10};
    for (auto null_percent : nulls) {
      auto profile = data_profile_builder();
      profile.distribution(cudf::type_id::INT64, distribution_id::UNIFORM, 0, 100);
      if (null_percent == 0) {
        profile.no_validity();
      } else {
        profile.null_probability(null_percent / 100.0);
      }
      auto input      = create_random_column(cudf::type_id::INT64, row_count{rows}, profile, 42);
      auto const view = input->view();
      cudf::reduce_input const inputs[]{view};
      for (bool fused : {false, true}) {
        auto sum      = cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
        auto maximum  = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
        auto baseline = [&] {
          columns result;
          result.push_back(cudf::segmented_reduce(
            view, offsets, *sum, type, cudf::null_policy::EXCLUDE, std::nullopt, stream));
          if (fused) {
            result.push_back(cudf::segmented_reduce(
              view, offsets, *maximum, type, cudf::null_policy::EXCLUDE, std::nullopt, stream));
          }
          return result;
        };
        host_columns expected;
        for (auto const& column : baseline()) {
          expected.push_back(host_values<int64_t>(column->view(), stream));
        }
        auto validate_baseline = [&](columns const& result) {
          if (result.size() != expected.size()) {
            throw std::runtime_error{"Output count mismatch"};
          }
          for (size_t index = 0; index < result.size(); ++index) {
            expect_equal(host_values<int64_t>(result[index]->view(), stream), expected[index]);
          }
        };
        workload work{fused ? "segmented_sum_max" : "segmented_sum",
                      dimensions.name,
                      rows,
                      dimensions.segments,
                      null_percent,
                      "none",
                      rows * (sizeof(int64_t) + (null_percent == 0 ? 0.0 : 1.0 / CHAR_BIT)) +
                        host_offsets.size() * sizeof(cudf::size_type)};
        measure(output, work, "cudf", "packed_columns", stream, baseline, validate_baseline);

        std::vector<cudf::data_type> types(fused ? 2 : 1, type);
        auto const udf = make_udf(fused);
        auto validate  = [&](columns const& result) {
          if (result.size() != dimensions.segments * types.size()) {
            throw std::runtime_error{"Segmented output count mismatch"};
          }
          for (size_t field = 0; field < types.size(); ++field) {
            std::vector<std::optional<int64_t>> actual;
            for (cudf::size_type segment = 0; segment < dimensions.segments; ++segment) {
              auto values =
                host_values<int64_t>(result[segment * types.size() + field]->view(), stream);
              if (values.size() != 1) { throw std::runtime_error{"Expected a single-row output"}; }
              actual.push_back(values.front());
            }
            expect_equal(actual, expected[field]);
          }
        };
        for (bool known : {false, true}) {
          auto const hint = known ? std::optional{longest} : std::nullopt;
          work.hint       = known ? "known" : "unknown";
          auto execute    = [&] {
            return cudf::segmented_reduce_lto(
              udf, cudf::null_aware::YES, inputs, offsets, types, types, hint, stream);
          };
          measure(output, work, "lto", "single_row_columns", stream, execute, validate);
        }
      }
    }
  }
}
}  // namespace reduction_benchmark
