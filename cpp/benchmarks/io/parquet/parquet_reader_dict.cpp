/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/column/column.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// Benchmark for the parquet-dictionary -> cudf DICTIONARY32 transcode fast path enabled by
// `parquet_reader_options::output_dict_columns`. A flat fully dictionary-encoded string column is
// read under three modes, selected by the `mode` axis, so the transcode path can be judged against
// both a lower and an upper reference:
//
//   - "materialize_string": reader default; the column materializes as STRING. The cheapest
//   possible
//                      read (no dictionary built); serves as the lower-bound reference.
//   - "materialize_string_and_encode_dict": materialize as STRING, then `cudf::dictionary::encode`
//                      to DICTIONARY32. The pre-existing way to obtain a dictionary column and the
//                      fair apples-to-apples baseline the transcode fast path aims to beat.
//   - "direct_dict_transcode": `output_dict_columns=true`; the reader keeps the dictionary
//                      representation and produces DICTIONARY32 directly, skipping string
//                      materialization.
//
// Both "materialize_string_and_encode_dict" and "direct_dict_transcode" produce DICTIONARY32
// output, so their times and peak memory are directly comparable; "materialize_string" shows the
// floor cost of just decoding. A relative comparison table (materialize_string_and_encode_dict =
// 100%%) is printed at program exit (see comparison_collector).
//
// The sweep varies cardinality, rows per row group, and rows per data page at a fixed table size
// A single column (num_cols == 1) is used, so a row group can hold as many distinct values as
// possible. The writer picks the dictionary index bit width per row group from the distinct values
// it contains, capped at MAX_DICT_BITS (24). Cardinality therefore ranges up to 2^24, the point at
// which 24-bit indices are required. At high distinct-per-row-group counts the writer may abandon
// dictionary encoding (indices exceed 24 bits, or plain encoding is smaller); when that leaves the
// column ineligible for transcode, that state is skipped rather than measured (the benchmark
// omits skipped states from its own table).

namespace {

constexpr cudf::size_type num_cols = 1;

enum class bench_mode {
  materialize_string,
  materialize_string_and_encode_dict,
  direct_dict_transcode
};

[[nodiscard]] bench_mode parse_mode(std::string const& mode)
{
  if (mode == "materialize_string") { return bench_mode::materialize_string; }
  if (mode == "materialize_string_and_encode_dict") {
    return bench_mode::materialize_string_and_encode_dict;
  }
  if (mode == "direct_dict_transcode") { return bench_mode::direct_dict_transcode; }
  CUDF_FAIL("Unknown benchmark mode: " + mode);
}

// Upper-bound estimate of the dictionary index bit width the writer will use for a row group. The
// width is derived per row group from its distinct value count, which is at most
// min(cardinality, rows in the row group). This over-estimates when the (last) row group is shorter
// than `row_group_size_rows` or when hash collisions reduce distinct counts.
[[nodiscard]] int approx_dict_bits(std::int64_t cardinality, std::int64_t row_group_size_rows)
{
  auto const distinct = std::min(cardinality, row_group_size_rows);
  if (distinct <= 1) { return 1; }
  return static_cast<int>(std::bit_width(static_cast<std::uint64_t>(distinct - 1)));
}

// nvbench invokes the benchmark once per axis combination, prints its own results table, and omits
// skipped states from it; there is also no cross-state hook, so a single invocation cannot group
// the three modes of a configuration together. This collector accumulates each run's CPU/GPU mean
// time (and any direct_dict_transcode skip reason), keyed by every setting except `mode`, and
// prints one row per configuration from its destructor -- i.e. at program exit, after nvbench's own
// output
// -- with all three modes in fixed order (materialize_string, materialize_string_and_encode_dict,
// direct_dict_transcode) so each configuration is
// grouped and ordered regardless of nvbench's state ordering or its omission of skipped states.
struct run_settings {
  std::int64_t cardinality;
  std::int64_t data_size;
  std::int64_t row_group_size_rows;
  std::int64_t max_page_size_rows;
  std::int64_t avg_string_length;

  bool operator<(run_settings const& o) const
  {
    return std::tie(
             cardinality, data_size, row_group_size_rows, max_page_size_rows, avg_string_length) <
           std::tie(o.cardinality,
                    o.data_size,
                    o.row_group_size_rows,
                    o.max_page_size_rows,
                    o.avg_string_length);
  }
};

struct mode_timing {
  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
  bool present  = false;
};

class comparison_collector {
 public:
  void record(run_settings const& key, bench_mode mode, double cpu_ms, double gpu_ms)
  {
    auto& r    = _rows[key];
    auto& slot = (mode == bench_mode::materialize_string_and_encode_dict)
                   ? r.materialize_string_and_encode_dict
                   : ((mode == bench_mode::direct_dict_transcode) ? r.direct_dict_transcode
                                                                  : r.materialize_string);
    slot       = mode_timing{cpu_ms, gpu_ms, true};
  }

  // Record that direct_dict_transcode was skipped for a configuration, with a short reason shown in
  // the table. `direct_dict_transcode` is the only mode this benchmark ever skips.
  void record_skip(run_settings const& key, std::string reason)
  {
    _rows[key].direct_dict_transcode_note = std::move(reason);
  }

  ~comparison_collector() { print(); }

 private:
  struct row {
    mode_timing materialize_string;
    mode_timing materialize_string_and_encode_dict;
    mode_timing direct_dict_transcode;
    // reason shown when `direct_dict_transcode` was skipped as ineligible
    std::string direct_dict_transcode_note;
  };

  void print() const
  {
    if (_rows.empty()) { return; }

    std::printf(
      "\n# Per-configuration mode comparison "
      "(order: materialize_string, materialize_string_and_encode_dict, "
      "direct_dict_transcode)\n\n");
    std::printf(
      "| cardinality | ~dict_bits | data_size (MiB) | row_group_size_rows | max_page_size_rows | "
      "materialize_string CPU (ms) | materialize_string_and_encode_dict CPU (ms) | "
      "direct_dict_transcode CPU (ms) | materialize_string GPU (ms) | "
      "materialize_string_and_encode_dict GPU (ms) | direct_dict_transcode GPU (ms) | "
      "direct_dict_transcode CPU speedup %% | direct_dict_transcode GPU speedup %% |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");

    auto const num = [](double v) {
      std::array<char, 32> buf{};
      std::snprintf(buf.data(), buf.size(), "%.3f", v);
      return std::string{buf.data()};
    };
    // Timing cell: the value if the mode ran, else the skip reason (direct_dict_transcode only) or
    // "-".
    auto const cell =
      [&](mode_timing const& t, double mode_timing::* field, std::string const& note) {
        if (t.present) { return num(t.*field); }
        return note.empty() ? std::string{"-"} : note;
      };

    // Speedup of direct_dict_transcode over the materialize_string_and_encode_dict baseline, as a
    // signed percentage of baseline time saved:
    //   100 * (materialize_string_and_encode_dict - direct_dict_transcode) /
    //         materialize_string_and_encode_dict.
    // Positive = direct_dict_transcode faster, negative = slower. "-" when either mode is missing.
    auto const speedup =
      [](mode_timing const& base, mode_timing const& cand, double mode_timing::* field) {
        if (not(base.present and cand.present)) { return std::string{"-"}; }
        std::array<char, 16> buf{};
        std::snprintf(
          buf.data(), buf.size(), "%+.1f%%", 100.0 * (base.*field - cand.*field) / (base.*field));
        return std::string{buf.data()};
      };

    for (auto const& [key, r] : _rows) {
      std::printf(
        "| %lld | %d | %lld | %lld | %lld | %s | %s | %s | %s | %s | %s | %s | %s |\n",
        static_cast<long long>(key.cardinality),
        approx_dict_bits(key.cardinality, key.row_group_size_rows),
        static_cast<long long>(key.data_size >> 20),
        static_cast<long long>(key.row_group_size_rows),
        static_cast<long long>(key.max_page_size_rows),
        cell(r.materialize_string, &mode_timing::cpu_ms, std::string{}).c_str(),
        cell(r.materialize_string_and_encode_dict, &mode_timing::cpu_ms, std::string{}).c_str(),
        cell(r.direct_dict_transcode, &mode_timing::cpu_ms, r.direct_dict_transcode_note).c_str(),
        cell(r.materialize_string, &mode_timing::gpu_ms, std::string{}).c_str(),
        cell(r.materialize_string_and_encode_dict, &mode_timing::gpu_ms, std::string{}).c_str(),
        cell(r.direct_dict_transcode, &mode_timing::gpu_ms, r.direct_dict_transcode_note).c_str(),
        speedup(r.materialize_string_and_encode_dict, r.direct_dict_transcode, &mode_timing::cpu_ms)
          .c_str(),
        speedup(r.materialize_string_and_encode_dict, r.direct_dict_transcode, &mode_timing::gpu_ms)
          .c_str());
    }
    std::printf("\n");
  }

  std::map<run_settings, row> _rows;
};

comparison_collector g_comparison_collector;

// The transcode fast path requires every data page of an eligible column to be dictionary-encoded.
// Forcing `dictionary_policy::ALWAYS` maximizes the chance of full dictionary encoding; the writer
// can still fall back to plain when indices exceed MAX_DICT_BITS or plain is smaller, in which case
// the direct_dict_transcode state is skipped by the caller.
void write_dict_encoded_parquet(cudf::table_view const& view,
                                cuio_source_sink_pair& source_sink,
                                std::int64_t row_group_size_rows,
                                std::int64_t max_page_size_rows)
{
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  if (row_group_size_rows > 0) {
    write_opts.set_row_group_size_rows(static_cast<cudf::size_type>(row_group_size_rows));
  }
  if (max_page_size_rows > 0) {
    write_opts.set_max_page_size_rows(static_cast<cudf::size_type>(max_page_size_rows));
  }
  cudf::io::write_parquet(write_opts);
}

}  // namespace

void BM_parquet_read_dict_transcode(nvbench::state& state)
{
  auto const cardinality       = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const data_size         = static_cast<size_t>(state.get_int64("data_size"));
  auto const rg_size_rows      = state.get_int64("row_group_size_rows");
  auto const page_size_rows    = state.get_int64("max_page_size_rows");
  auto const mode              = parse_mode(state.get_string("mode"));
  auto const avg_string_length = static_cast<cudf::size_type>(state.get_int64("avg_string_length"));
  auto const source_type       = retrieve_io_type_enum(state.get_string("io_type"));

  // corresponds to 3 sigma (full width 6 sigma: 99.7% of range)
  auto const half_width = avg_string_length >> 3;
  auto const length_min = avg_string_length - half_width;
  auto const length_max = avg_string_length + half_width;

  data_profile const profile =
    data_profile_builder()
      .cardinality(cardinality)
      .avg_run_length(1)
      .distribution(data_type::STRING, distribution_id::NORMAL, length_min, length_max);

  auto const d_type = get_type_or_group(static_cast<int32_t>(data_type::STRING));
  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  write_dict_encoded_parquet(view, source_sink, rg_size_rows, page_size_rows);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info())
      .output_dict_columns(mode == bench_mode::direct_dict_transcode);

  // Perform the full work for the selected mode: read, and for `materialize_string_and_encode_dict`
  // additionally encode each STRING column to DICTIONARY32. Returns the resulting table so it can
  // be reused for both the outside-the-timed-region verification and the timed measurement.
  auto const run_mode = [&]() -> std::unique_ptr<cudf::table> {
    auto result = cudf::io::read_parquet(read_opts);
    if (mode == bench_mode::materialize_string_and_encode_dict) {
      std::vector<std::unique_ptr<cudf::column>> encoded;
      encoded.reserve(result.tbl->num_columns());
      for (auto const& col : result.tbl->view()) {
        encoded.push_back(cudf::dictionary::encode(col));
      }
      return std::make_unique<cudf::table>(std::move(encoded));
    }
    return std::move(result.tbl);
  };

  // Verification (outside the timed region, run for every mode so warm-up is symmetric). For
  // `direct_dict_transcode`, the writer may have fallen back to plain encoding at high cardinality
  // / large row groups, leaving the column ineligible for the fast path; in that case skip the
  // state rather than silently measuring the plain path or aborting the whole sweep.
  {
    auto const probe = run_mode();
    // Bind the table_view to a local: `probe->view()` returns a temporary, so calling it separately
    // for begin() and end() would yield iterators into two different temporaries (mismatched-
    // iterator UB). Iterate a single view instead.
    auto const probe_view = probe->view();
    CUDF_EXPECTS(probe_view.num_columns() == num_cols, "Unexpected number of columns");
    auto const all_of_type = [&](cudf::type_id id) {
      return std::all_of(probe_view.begin(), probe_view.end(), [id](auto const& col) {
        return col.type().id() == id;
      });
    };
    auto const actual_type_id = static_cast<int>(probe_view.column(0).type().id());
    if (mode == bench_mode::materialize_string) {
      if (not all_of_type(cudf::type_id::STRING)) {
        state.skip(
          "materialize_string produced unexpected type_id=" + std::to_string(actual_type_id) +
          " (expected STRING=" + std::to_string(static_cast<int>(cudf::type_id::STRING)) + ")");
        return;
      }
    } else if (mode == bench_mode::materialize_string_and_encode_dict) {
      if (not all_of_type(cudf::type_id::DICTIONARY32)) {
        state.skip("materialize_string_and_encode_dict produced unexpected type_id=" +
                   std::to_string(actual_type_id) + " (expected DICTIONARY32=" +
                   std::to_string(static_cast<int>(cudf::type_id::DICTIONARY32)) + ")");
        return;
      }
    } else if (not all_of_type(cudf::type_id::DICTIONARY32)) {
      // Record the skip so the end-of-program per-configuration table can show why
      // direct_dict_transcode has no
      // timing for this configuration (nvbench omits skipped states from its own table).
      g_comparison_collector.record_skip(run_settings{cardinality,
                                                      static_cast<std::int64_t>(data_size),
                                                      rg_size_rows,
                                                      page_size_rows,
                                                      avg_string_length},
                                         "skipped: plain fallback");
      state.skip(
        "direct_dict_transcode did not produce DICTIONARY32: at this cardinality / row-group size "
        "the writer "
        "fell back to plain encoding, making the column ineligible for the fast path");
      return;
    }
  }

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(read_opts.get_source().filepaths());

               timer.start();
               auto const result = run_mode();
               timer.stop();

               CUDF_EXPECTS(result->num_columns() == num_cols, "Unexpected number of columns");
             });

  auto const time     = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const cpu_time = state.get_summary("nv/cold/time/cpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_element_count(static_cast<double>(view.num_rows()) / time, "rows_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");

  // Record this run for the end-of-program
  // direct_dict_transcode-vs-materialize_string_and_encode_dict comparison table. Times are
  // reported by nvbench in seconds; store as milliseconds.
  g_comparison_collector.record(run_settings{cardinality,
                                             static_cast<std::int64_t>(data_size),
                                             rg_size_rows,
                                             page_size_rows,
                                             avg_string_length},
                                mode,
                                cpu_time * 1e3,
                                time * 1e3);
}

NVBENCH_BENCH(BM_parquet_read_dict_transcode)
  .set_name("parquet_read_dict_transcode")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_string_axis(
    "mode", {"materialize_string", "materialize_string_and_encode_dict", "direct_dict_transcode"})
  // Cardinality: low, mid, and 2^24 -- the point at which per-row-group dictionary indices need the
  // maximum 24 bits the writer supports (MAX_DICT_BITS); beyond that the writer abandons dictionary
  // encoding. Achieved bits = ceil(log2(min(cardinality, rows per row group))).
  .add_int64_axis("cardinality", {1 << 10, 1 << 20, 1 << 24})
  // Fixed table size, kept modest so the sweep stays light for local/CI runs (peak memory and
  // per-state runtime scale with this).
  .add_int64_axis("data_size", {std::int64_t{512} << 20})
  // Rows per row group: small (many row groups -> stresses per-row-group key concatenation),
  // default (1M), and very large (>= 2^24 so a single row group can reach the 24-bit dict
  // boundary).
  .add_int64_axis("row_group_size_rows", {100'000, 1'000'000, 20'000'000})
  // Rows per data page: small and large (page size is a second-order factor for this benchmark).
  .add_int64_axis("max_page_size_rows", {20'000, 1'000'000})
  .add_int64_axis("avg_string_length", {16});
