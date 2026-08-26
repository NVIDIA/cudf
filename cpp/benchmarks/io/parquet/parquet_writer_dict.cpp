/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/parquet/compact_protocol_reader.hpp"

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr auto frequent_pages_ratio =
  0.8;  ///< 80% of the pages will only contain elements from the frequent set

/**
 * @brief Build a string column such that certain pages only contain elements from the frequent set
 and others only contain elements from the rare set
 *
 * Each generated value is of the form `"k_<v>"`, where `v` is a `cudf::size_type` index in `[0,
 cardinality)`, so cardinality maps 1:1 to distinct strings.
 *
 * @param num_rows Total number of rows
 * @param page_size_rows Number of rows per page
 * @param cardinality Total number of distinct values
 * @param frequent_set_ratio Fraction of `cardinality` assigned to the frequent set
 * @return Constructed column values
 */
std::vector<std::string> build_string_column(cudf::size_type num_rows,
                                             cudf::size_type page_size_rows,
                                             cudf::size_type cardinality,
                                             double frequent_set_ratio)
{
  static constexpr auto dict_rng_seed = 0xC0DEFACE;

  CUDF_EXPECTS(frequent_set_ratio > 0.0 and frequent_set_ratio < 1.0,
               "frequent_set_ratio must be between 0.0 and 1.0");
  CUDF_EXPECTS(num_rows % page_size_rows == 0, "num_rows must be a multiple of page_size_rows");
  static_assert(frequent_pages_ratio > 0.0 and frequent_pages_ratio < 1.0,
                "frequent_pages_ratio must be between 0.0 and 1.0");

  auto const total_pages = num_rows / page_size_rows;
  auto const frequent_set_threshold =
    static_cast<cudf::size_type>(total_pages * frequent_pages_ratio) * page_size_rows;

  auto const frequent_set_size =
    static_cast<cudf::size_type>(static_cast<double>(cardinality) * frequent_set_ratio);

  std::mt19937 rng{dict_rng_seed};
  std::uniform_int_distribution<cudf::size_type> freq_dist(0, frequent_set_size - 1);
  std::uniform_int_distribution<cudf::size_type> rare_dist(frequent_set_size, cardinality - 1);

  cudf::size_type row_idx = 0;
  std::vector<std::string> values(num_rows);
  std::generate_n(values.begin(), num_rows, [&]() {
    auto const v = row_idx++ < frequent_set_threshold ? freq_dist(rng) : rare_dist(rng);
    return "k_" + std::to_string(v);
  });

  return values;
}

/**
 * @brief Build a table with a single STRING column
 *
 * @tparam reverse_order Whether to reverse the order of the values
 * @param num_rows Number of rows
 * @param page_size_rows Number of rows per page
 * @param cardinality Total number of distinct values
 * @param frequent_set_ratio Fraction of `cardinality` assigned to the frequent set
 * @return std::unique_ptr<cudf::table>
 */
template <bool reverse_order = false>
[[nodiscard]] std::unique_ptr<cudf::table> build_table(cudf::size_type num_rows,
                                                       cudf::size_type page_size_rows,
                                                       cudf::size_type cardinality,
                                                       double frequent_set_ratio)
{
  constexpr cudf::size_type num_cols = 1;

  auto values = build_string_column(num_rows, page_size_rows, cardinality, frequent_set_ratio);
  if constexpr (reverse_order) { std::reverse(values.begin(), values.end()); }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(num_cols);
  cols.emplace_back(cudf::test::strings_column_wrapper(values.begin(), values.end()).release());
  return std::make_unique<cudf::table>(std::move(cols));
}

/**
 * @brief Compute per-page RLE bit widths for dictionary-encoded pages from the parquet page index
 *
 * Assumption: All parquet pages are dictionary-encoded, no nulls, no rep/def levels
 *
 * @param source Datasource
 * @param footer File metadata
 * @return Vector of number of bits per page for dictionary-encoded pages
 */
[[nodiscard]] std::vector<int> compute_page_dict_bits(cudf::io::datasource& source,
                                                      cudf::io::parquet::FileMetaData const& footer)
{
  using namespace cudf::io::parquet;

  std::vector<int> bits;

  for (auto const& rg : footer.row_groups) {
    for (auto const& chunk : rg.columns) {
      if (not chunk.offset_index.has_value()) { continue; }
      for (auto const& page_loc : chunk.offset_index->page_locations) {
        if (page_loc.offset <= 0 or page_loc.compressed_page_size <= 0) { continue; }
        auto const buffer = source.host_read(page_loc.offset, page_loc.compressed_page_size);
        detail::CompactProtocolReader cp(buffer->data(), buffer->size());
        PageHeader page_header;
        cp.read(&page_header);
        // Check if the page is dictionary-encoded.
        auto const is_dict_encoded =
          (page_header.type == PageType::DATA_PAGE and
           (page_header.data_page_header.encoding == Encoding::PLAIN_DICTIONARY or
            page_header.data_page_header.encoding == Encoding::RLE_DICTIONARY)) or
          (page_header.type == PageType::DATA_PAGE_V2 and
           (page_header.data_page_header_v2.encoding == Encoding::PLAIN_DICTIONARY or
            page_header.data_page_header_v2.encoding == Encoding::RLE_DICTIONARY));
        if (not is_dict_encoded) { continue; }
        // `cp` is positioned at the first byte of the page payload after the
        // header thrift; that byte is the RLE bit width for dict-indexed
        // pages (valid only with no rep/def levels).
        bits.push_back(cp.getb());
      }
    }
  }
  return bits;
}

}  // namespace

void BM_parq_write_dict_encoding(nvbench::state& state)
{
  auto const num_rows           = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const reverse_order      = static_cast<bool>(state.get_int64("reverse_order"));
  auto const cardinality        = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const frequent_set_ratio = static_cast<double>(state.get_float64("freq_set_ratio"));
  auto const page_size_rows     = static_cast<cudf::size_type>(state.get_int64("page_size_rows"));

  CUDF_EXPECTS(page_size_rows <= num_rows and num_rows % page_size_rows == 0,
               "num_rows must be a multiple of page_size_rows");

  cuio_source_sink_pair source_sink(io_type::FILEPATH);

  auto const table = [&]() {
    if (reverse_order) {
      return build_table<true>(num_rows, page_size_rows, cardinality, frequent_set_ratio);
    } else {
      return build_table<false>(num_rows, page_size_rows, cardinality, frequent_set_ratio);
    }
  }();

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::timer | nvbench::exec_tag::sync, [&](nvbench::launch&, auto& timer) {
      timer.start();
      auto const write_opts =
        cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), table->view())
          .compression(cudf::io::compression_type::NONE)
          .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
          .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
          .row_group_size_rows(num_rows)
          .max_page_size_rows(page_size_rows)
          .max_page_size_bytes(std::size_t{64} << 20)
          .build();
      cudf::io::write_parquet(write_opts);
      timer.stop();
    });

  state.add_element_count(static_cast<double>(table->num_rows()), "rows");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");

  // Use hybrid scan reader to get footer with page index.
  {
    auto const datasource =
      std::move(cudf::io::make_datasources(source_sink.make_source_info()).front());
    auto const datasource_ref = std::ref(*datasource);
    auto const footer_buf     = cudf::io::parquet::fetch_footer_to_host(datasource_ref);
    cudf::io::parquet::experimental::hybrid_scan_reader reader(*footer_buf,
                                                               cudf::io::parquet_reader_options{});
    auto const page_index_bytes = reader.page_index_byte_range();
    CUDF_EXPECTS(not page_index_bytes.is_empty(), "Page index is required");
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(datasource_ref, page_index_bytes);
    reader.setup_page_index(*page_index_buffer);

    auto const metadata       = reader.parquet_metadata();
    auto const page_dict_bits = compute_page_dict_bits(datasource_ref, metadata);

    CUDF_EXPECTS(not page_dict_bits.empty(), "No dictionary-encoded pages found");

    auto const [min_it, max_it] = std::minmax_element(page_dict_bits.begin(), page_dict_bits.end());
    auto const sum =
      std::accumulate(page_dict_bits.begin(), page_dict_bits.end(), std::uint64_t{0});
    auto const mean =
      std::round(static_cast<double>(sum) / static_cast<double>(page_dict_bits.size()));
    state.add_element_count(static_cast<double>(*min_it), "dict_rle_bits_min");
    state.add_element_count(static_cast<double>(*max_it), "dict_rle_bits_max");
    state.add_element_count(mean, "dict_rle_bits_mean");
  }
}

NVBENCH_BENCH(BM_parq_write_dict_encoding)
  .set_name("parquet_write_dict_encoding")
  .set_min_samples(4)
  .add_int64_axis("reverse_order", {false, true})
  .add_int64_axis("num_rows", {1'000'000})
  .add_int64_axis("page_size_rows", {10'000, 100'000})
  .add_int64_axis("cardinality", {64'000, 100'000})
  .add_float64_axis("freq_set_ratio", {0.001, 0.01});
