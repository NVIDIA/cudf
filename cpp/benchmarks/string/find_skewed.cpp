/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file find_skewed.cpp
 * @brief contains(column, scalar) benchmark over *heterogeneous* row-length distributions.
 *
 * The standard `find_string` benchmark (find.cpp) builds rows from ten sample strings that are
 * all ~32 bytes (or an exact multiple), so every row in a column has essentially the same width
 * and the thread-per-row vs warp-per-row dispatch on the *average* width is nearly optimal.
 * Real string columns are rarely that uniform. This benchmark generates columns whose row
 * lengths follow several skewed distributions so the scheduling strategy inside
 * cudf::strings::contains can be compared under intra-column width variance.
 *
 * `dist` axis values (parsed at runtime):
 *   fixed_L              every row L bytes
 *   uniform_A_B          lengths uniform in [A, B]
 *   bimodal_P_S_L        P percent of rows are L bytes, the rest S bytes
 *   lognormal_M_S        lengths ~ lognormal with median M and sigma S/100, clamped to [0, 1<<20]
 *   pareto_XM_A          lengths ~ pareto(xm=XM, alpha=A/100), clamped to [0, 1<<20]
 *   fewhuge_S_H_K        all rows S bytes except K rows of H bytes
 *
 * Row *content* is random lowercase text; `hit_rate` percent of rows have the 9-byte target
 * "0987 5W43" inserted at a random position (rows shorter than the target are never hits).
 * `null_pct` percent of rows are null.
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr char const* TARGET = "0987 5W43";

/// Split `s` on `sep`.
std::vector<std::string> split(std::string const& s, char sep)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, sep))
    out.push_back(tok);
  return out;
}

/// Generate per-row byte lengths for the named distribution.
std::vector<int32_t> make_lengths(std::string const& dist, int64_t num_rows, std::mt19937_64& rng)
{
  auto const parts = split(dist, '_');
  auto const kind  = parts.at(0);
  // i-th numeric parameter of the distribution spec
  auto p = [&](size_t i) { return std::stoll(parts.at(i)); };
  std::vector<int32_t> lens(num_rows);
  auto constexpr max_len = int64_t{1} << 20;
  // round and clamp a sampled length to [0, max_len]
  auto clampf = [&](double v) {
    return static_cast<int32_t>(std::clamp<double>(std::llround(v), 0, max_len));
  };

  if (kind == "fixed") {
    std::fill(lens.begin(), lens.end(), static_cast<int32_t>(p(1)));
  } else if (kind == "uniform") {
    std::uniform_int_distribution<int32_t> d(static_cast<int32_t>(p(1)),
                                             static_cast<int32_t>(p(2)));
    for (auto& l : lens)
      l = d(rng);
  } else if (kind == "bimodal") {
    auto const pct = p(1);
    auto const s   = static_cast<int32_t>(p(2));
    auto const l   = static_cast<int32_t>(p(3));
    std::uniform_int_distribution<int> d(0, 99);
    for (auto& x : lens)
      x = d(rng) < pct ? l : s;
  } else if (kind == "lognormal") {
    auto const median = static_cast<double>(p(1));
    auto const sigma  = static_cast<double>(p(2)) / 100.0;
    std::lognormal_distribution<double> d(std::log(median), sigma);
    for (auto& l : lens)
      l = clampf(d(rng));
  } else if (kind == "pareto") {
    auto const xm    = static_cast<double>(p(1));
    auto const alpha = static_cast<double>(p(2)) / 100.0;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (auto& l : lens)
      l = clampf(xm / std::pow(1.0 - u(rng), 1.0 / alpha));
  } else if (kind == "fewhuge") {
    auto const s = static_cast<int32_t>(p(1));
    auto const h = static_cast<int32_t>(p(2));
    auto const k = p(3);
    std::fill(lens.begin(), lens.end(), s);
    std::uniform_int_distribution<int64_t> d(0, num_rows - 1);
    for (int64_t i = 0; i < k; ++i)
      lens[d(rng)] = h;
  } else {
    CUDF_FAIL("unknown dist: " + dist);
  }
  return lens;
}

struct skewed_column {
  std::unique_ptr<cudf::column> col;
  int64_t chars_bytes;
  int64_t max_len;
  std::vector<uint8_t> expected;  // 1 where the target was inserted, 2 for null rows
};

/**
 * @brief Build a strings column whose row lengths follow `dist`, with the target inserted at a
 * random position in `hit_rate` percent of the eligible rows and `null_pct` percent null rows.
 *
 * Uses int32 offsets when the characters fit, otherwise int64 (a cudf large-strings column).
 */
skewed_column make_skewed_column(std::string const& dist,
                                 int64_t num_rows,
                                 int hit_rate,
                                 int null_pct,
                                 rmm::cuda_stream_view stream)
{
  std::mt19937_64 rng(0x5eed1234 + num_rows);
  auto lens = make_lengths(dist, num_rows, rng);

  std::vector<bool> valid(num_rows, true);
  if (null_pct > 0) {
    std::uniform_int_distribution<int> d(0, 99);
    for (int64_t i = 0; i < num_rows; ++i) {
      if (d(rng) < null_pct) {
        valid[i] = false;
        lens[i]  = 0;
      }
    }
  }

  std::vector<int64_t> offsets(num_rows + 1, 0);
  for (int64_t i = 0; i < num_rows; ++i)
    offsets[i + 1] = offsets[i] + lens[i];
  auto const total = offsets[num_rows];

  // Random text over a 64-symbol alphabet that includes digits and space, so partial matches of
  // the target's leading bytes do occur (about 1 in 64 positions match the first byte) and the
  // per-position compare cost is realistic. A full 9-byte accidental match has probability 64^-9
  // per position and is ignored; results are validated against the inserted hits below.
  static constexpr char alphabet[65] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -";
  std::vector<char> chars(total);
  {
    uint64_t x = 0x9E3779B97F4A7C15ull;
    int64_t i  = 0;
    for (; i + 8 <= total; i += 8) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      uint64_t v = x;
      for (int k = 0; k < 8; ++k) {
        chars[i + k] = alphabet[v & 63];
        v >>= 6;
      }
    }
    for (; i < total; ++i)
      chars[i] = 'a';
  }

  // Insert the target into hit_rate percent of eligible rows at a random position.
  auto const tlen = static_cast<int32_t>(std::string(TARGET).size());
  std::uniform_int_distribution<int> hit_d(0, 99);
  int64_t max_len = 0;
  std::vector<uint8_t> expected(num_rows, 0);
  for (int64_t i = 0; i < num_rows; ++i) {
    max_len = std::max<int64_t>(max_len, lens[i]);
    if (!valid[i]) {
      expected[i] = 2;  // null: value unspecified, only the copied null mask matters
      continue;
    }
    if (lens[i] < tlen) continue;
    if (hit_d(rng) < hit_rate) {
      std::uniform_int_distribution<int32_t> pos_d(0, lens[i] - tlen);
      auto const pos = pos_d(rng);
      std::copy_n(TARGET, tlen, chars.begin() + offsets[i] + pos);
      expected[i] = 1;
    }
  }

  // int32 offsets when they fit, otherwise int64 (cudf large-strings column)
  auto offsets_col = [&]() -> std::unique_ptr<cudf::column> {
    if (total < std::numeric_limits<int32_t>::max()) {
      std::vector<int32_t> offsets32(offsets.begin(), offsets.end());
      rmm::device_uvector<int32_t> d_offsets(offsets32.size(), stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_offsets.data(),
                                    offsets32.data(),
                                    offsets32.size() * sizeof(int32_t),
                                    cudaMemcpyDefault,
                                    stream.value()));
      return std::make_unique<cudf::column>(std::move(d_offsets), rmm::device_buffer{}, 0);
    }
    rmm::device_uvector<int64_t> d_offsets(offsets.size(), stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_offsets.data(),
                                  offsets.data(),
                                  offsets.size() * sizeof(int64_t),
                                  cudaMemcpyDefault,
                                  stream.value()));
    return std::make_unique<cudf::column>(std::move(d_offsets), rmm::device_buffer{}, 0);
  }();
  rmm::device_buffer d_chars(chars.data(), chars.size(), stream);

  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;
  if (null_pct > 0) {
    auto const mask_words =
      cudf::bitmask_allocation_size_bytes(num_rows) / sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> h_mask(mask_words, 0);
    for (int64_t i = 0; i < num_rows; ++i) {
      if (valid[i]) {
        h_mask[i / 32] |= (cudf::bitmask_type{1} << (i % 32));
      } else {
        ++null_count;
      }
    }
    null_mask =
      rmm::device_buffer(h_mask.data(), h_mask.size() * sizeof(cudf::bitmask_type), stream);
  }

  auto col = cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                       std::move(offsets_col),
                                       std::move(d_chars),
                                       null_count,
                                       std::move(null_mask));
  stream.synchronize();
  return {std::move(col), total, max_len, std::move(expected)};
}

/// Run contains() once and check every row against the inserted hits; throws on any mismatch.
void validate(cudf::strings_column_view const& input,
              cudf::string_scalar const& target,
              std::vector<uint8_t> const& expected,
              rmm::cuda_stream_view stream)
{
  auto result = cudf::strings::contains(input, target);
  std::vector<uint8_t> got(expected.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    got.data(), result->view().data<bool>(), got.size(), cudaMemcpyDefault, stream.value()));
  stream.synchronize();
  int64_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (expected[i] == 2) continue;
    if ((got[i] != 0) != (expected[i] != 0)) ++bad;
  }
  CUDF_EXPECTS(bad == 0, "contains() returned " + std::to_string(bad) + " wrong rows");
}

}  // namespace

/**
 * @brief contains(column, scalar) over the row-length distribution named by the `dist` axis.
 */
static void bench_find_skewed(nvbench::state& state)
{
  auto const num_rows = state.get_int64("num_rows");
  auto const hit_rate = static_cast<int>(state.get_int64("hit_rate"));
  auto const null_pct = static_cast<int>(state.get_int64("null_pct"));
  auto const dist     = state.get_string("dist");

  auto const stream = cudf::get_default_stream();
  auto data         = make_skewed_column(dist, num_rows, hit_rate, null_pct, stream);
  auto const input  = cudf::strings_column_view(data.col->view());
  auto target       = cudf::string_scalar(TARGET);

  validate(input, target, data.expected, stream);

  state.add_element_count(num_rows, "rows");
  state.add_summary("avg_bytes").set_int64("value", data.chars_bytes / num_rows);
  state.add_summary("max_bytes").set_int64("value", data.max_len);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<nvbench::int8_t>(data.col->alloc_size());
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
