/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief contains(column, scalar) over columns whose row lengths follow skewed distributions.
 *
 * `dist` axis values:
 *   fixed_L              every row L bytes
 *   uniform_A_B          lengths uniform in [A, B]
 *   bimodal_P_S_L        P percent of rows are L bytes, the rest S bytes
 *   lognormal_M_S        lengths ~ lognormal with median M and sigma S/100, clamped to [0, 1 MiB]
 *   pareto_XM_A          lengths ~ pareto(xm=XM, alpha=A/100), clamped to [0, 1 MiB]
 *   fewhuge_S_H_K        all rows S bytes except K rows of H bytes
 *
 * Row content is random text over a 64-symbol alphabet. The target is inserted at a random
 * position in `hit_rate` percent of the rows long enough to hold it, and `null_pct` percent of
 * the rows are null.
 */

namespace {

constexpr char const* target_string = "0987 5W43";

/**
 * @brief Generate one length per row for the distribution named by `dist`
 */
std::vector<int32_t> make_lengths(std::string const& dist, int64_t num_rows, std::mt19937_64& rng)
{
  std::vector<std::string> parts;
  {
    std::stringstream ss(dist);
    std::string token;
    while (std::getline(ss, token, '_')) {
      parts.push_back(token);
    }
  }
  auto const kind  = parts.at(0);
  auto const param = [&](std::size_t i) { return std::stoll(parts.at(i)); };

  auto constexpr max_length = int64_t{1} << 20;
  auto const clamp_length   = [&](double v) {
    return static_cast<int32_t>(std::clamp<double>(std::llround(v), 0, max_length));
  };

  std::vector<int32_t> lengths(num_rows);
  if (kind == "fixed") {
    std::fill(lengths.begin(), lengths.end(), static_cast<int32_t>(param(1)));
  } else if (kind == "uniform") {
    std::uniform_int_distribution<int32_t> d(static_cast<int32_t>(param(1)),
                                             static_cast<int32_t>(param(2)));
    std::generate(lengths.begin(), lengths.end(), [&] { return d(rng); });
  } else if (kind == "bimodal") {
    auto const pct   = param(1);
    auto const small = static_cast<int32_t>(param(2));
    auto const large = static_cast<int32_t>(param(3));
    std::uniform_int_distribution<int> d(0, 99);
    std::generate(lengths.begin(), lengths.end(), [&] { return d(rng) < pct ? large : small; });
  } else if (kind == "lognormal") {
    auto const median = static_cast<double>(param(1));
    auto const sigma  = static_cast<double>(param(2)) / 100.0;
    std::lognormal_distribution<double> d(std::log(median), sigma);
    std::generate(lengths.begin(), lengths.end(), [&] { return clamp_length(d(rng)); });
  } else if (kind == "pareto") {
    auto const xm    = static_cast<double>(param(1));
    auto const alpha = static_cast<double>(param(2)) / 100.0;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::generate(lengths.begin(), lengths.end(), [&] {
      return clamp_length(xm / std::pow(1.0 - u(rng), 1.0 / alpha));
    });
  } else if (kind == "fewhuge") {
    std::fill(lengths.begin(), lengths.end(), static_cast<int32_t>(param(1)));
    std::uniform_int_distribution<int64_t> d(0, num_rows - 1);
    for (int64_t i = 0; i < param(3); ++i) {
      lengths[d(rng)] = static_cast<int32_t>(param(2));
    }
  } else {
    CUDF_FAIL("unknown dist: " + dist);
  }
  return lengths;
}

/**
 * @brief Create the offsets column from host offsets, using int64 when the total exceeds int32
 */
std::unique_ptr<cudf::column> make_offsets_column(std::vector<int64_t> const& offsets)
{
  if (offsets.back() <= std::numeric_limits<int32_t>::max()) {
    return cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end())
      .release();
  }
  return cudf::test::fixed_width_column_wrapper<int64_t>(offsets.begin(), offsets.end()).release();
}

struct skewed_column {
  std::unique_ptr<cudf::column> column;
  int64_t chars_size;
  int64_t max_length;
};

/**
 * @brief Build a strings column whose row lengths follow `dist`
 */
skewed_column make_skewed_column(
  std::string const& dist, int64_t num_rows, int hit_rate, int null_pct, cuda::stream_ref stream)
{
  std::mt19937_64 rng(0x5eed1234 + num_rows);
  auto lengths = make_lengths(dist, num_rows, rng);

  std::vector<bool> valid(num_rows, true);
  if (null_pct > 0) {
    std::uniform_int_distribution<int> d(0, 99);
    for (int64_t i = 0; i < num_rows; ++i) {
      if (d(rng) < null_pct) {
        valid[i]   = false;
        lengths[i] = 0;
      }
    }
  }

  std::vector<int64_t> offsets(num_rows + 1, 0);
  std::inclusive_scan(
    lengths.begin(), lengths.end(), offsets.begin() + 1, std::plus<int64_t>{}, int64_t{0});
  auto const chars_size = offsets.back();

  // Random text over a 64-symbol alphabet that includes digits and space so the first bytes of
  // the target match often but the whole target matches only where it is inserted.
  static constexpr char alphabet[] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -";
  std::vector<char> chars(chars_size);
  {
    uint64_t x = 0x9E3779B97F4A7C15ull;
    int64_t i  = 0;
    for (; i + 8 <= chars_size; i += 8) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      auto v = x;
      for (int k = 0; k < 8; ++k) {
        chars[i + k] = alphabet[v & 63];
        v >>= 6;
      }
    }
    std::fill(chars.begin() + i, chars.end(), 'a');
  }

  // insert the target into hit_rate percent of the rows long enough to hold it
  auto const target_size = static_cast<int32_t>(std::char_traits<char>::length(target_string));
  std::uniform_int_distribution<int> hit_dist(0, 99);
  for (int64_t i = 0; i < num_rows; ++i) {
    if (!valid[i] || lengths[i] < target_size || hit_dist(rng) >= hit_rate) { continue; }
    auto const pos = std::uniform_int_distribution<int32_t>(0, lengths[i] - target_size)(rng);
    std::copy_n(target_string, target_size, chars.begin() + offsets[i] + pos);
  }

  auto [null_mask, null_count] =
    null_pct > 0 ? cudf::test::detail::make_null_mask(valid.begin(), valid.end(), stream)
                 : std::pair{cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0};
  auto column = cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                          make_offsets_column(offsets),
                                          rmm::device_buffer(chars.data(), chars.size(), stream),
                                          null_count,
                                          std::move(null_mask));
  stream.sync();
  return {std::move(column), chars_size, *std::max_element(lengths.begin(), lengths.end())};
}

}  // namespace

static void bench_find_skewed(nvbench::state& state)
{
  auto const num_rows = state.get_int64("num_rows");
  auto const hit_rate = static_cast<int>(state.get_int64("hit_rate"));
  auto const null_pct = static_cast<int>(state.get_int64("null_pct"));
  auto const dist     = state.get_string("dist");

  auto const stream = cudf::get_default_stream();
  auto const data   = make_skewed_column(dist, num_rows, hit_rate, null_pct, stream);
  auto const input  = cudf::strings_column_view(data.column->view());
  auto const target = cudf::string_scalar(target_string);

  state.add_element_count(num_rows, "rows");
  state.add_summary("avg_bytes").set_int64("value", data.chars_size / num_rows);
  state.add_summary("max_bytes").set_int64("value", data.max_length);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<nvbench::int8_t>(data.column->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(input.size());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::strings::contains(input, target); });
}

NVBENCH_BENCH(bench_find_skewed)
  .set_name("find_skewed")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("hit_rate", {20})
  .add_int64_axis("null_pct", {0})
  .add_string_axis("dist",
                   {"fixed_16",
                    "fixed_64",
                    "fixed_128",
                    "fixed_1024",
                    "uniform_1_128",
                    "uniform_1_512",
                    "bimodal_10_16_1024",
                    "bimodal_50_16_256",
                    "lognormal_40_100",
                    "lognormal_80_150",
                    "pareto_8_110",
                    "fewhuge_32_1048576_8"});
