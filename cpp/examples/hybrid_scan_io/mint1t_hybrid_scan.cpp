/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"
#include "io_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using cudf::io::parquet::experimental::hybrid_scan_multifile;
using cudf::io::parquet::experimental::use_data_page_mask;

auto constexpr original_dataset_root  = "/MINT1T_benchmarking_subset";
auto constexpr rewritten_dataset_root = "/MINT1T_rewritten";
auto constexpr default_refs =
  "/MINT1T_benchmarking_subset/parquet/interleaved_image_url_refs/image_url_refs.parquet";
auto constexpr default_image_dir =
  "/MINT1T_rewritten/parquet/mint_1t_html_images_stable_row_ids/data";
auto constexpr default_expected = "/MINT1T_rewritten/output/image_url_payloads.parquet";
auto constexpr writable_root    = "/MINT1T_rewritten";

std::vector<std::string> const payload_columns{
  "image", "image_format", "mime_type", "image_size_bytes", "md5", "sha256", "width", "height"};

struct arguments {
  std::filesystem::path dataset_root{rewritten_dataset_root};
  std::filesystem::path refs{default_refs};
  std::filesystem::path image_dir{default_image_dir};
  std::optional<cudf::size_type> limit;
  std::size_t pass_read_limit{};
  bool pass_read_limit_set{false};
  bool use_page_mask{true};
  bool use_sparse_page_io{true};
  bool drop_cache{false};
  bool validate{false};
  std::filesystem::path expected_output{default_expected};
  std::optional<std::filesystem::path> output;
};

struct reference {
  std::string image_parquet;
  uint32_t row_offset;

  bool operator<(reference const& other) const
  {
    return std::tie(image_parquet, row_offset) < std::tie(other.image_parquet, other.row_offset);
  }

  bool operator==(reference const& other) const
  {
    return image_parquet == other.image_parquet and row_offset == other.row_offset;
  }
};

struct timing_data {
  std::vector<std::pair<std::string, double>> stages;

  void add_seconds(std::string name, double seconds)
  {
    std::cout << "TIMING " << std::left << std::setw(38) << name << std::right << std::fixed
              << std::setprecision(6) << seconds << " s\n"
              << std::flush;
    stages.emplace_back(std::move(name), seconds);
  }

  void add(std::string name, clock_type::time_point start)
  {
    auto const seconds = std::chrono::duration<double>(clock_type::now() - start).count();
    auto const existing = std::find_if(
      stages.begin(), stages.end(), [&](auto const& item) { return item.first == name; });
    if (existing == stages.end()) {
      add_seconds(std::move(name), seconds);
    } else {
      existing->second += seconds;
    }
  }

  [[nodiscard]] double get(std::string_view name) const
  {
    auto const it = std::find_if(
      stages.begin(), stages.end(), [&](auto const& item) { return item.first == name; });
    return it == stages.end() ? 0.0 : it->second;
  }

  [[nodiscard]] double sum(std::span<std::string_view const> names) const
  {
    return std::accumulate(
      names.begin(), names.end(), 0.0, [&](double total, auto name) { return total + get(name); });
  }

  void print() const
  {
    std::cout << "\nStage timings:\n";
    for (auto const& [name, seconds] : stages) {
      std::cout << "  " << std::left << std::setw(38) << name << std::right << std::fixed
                << std::setprecision(6) << seconds << " s\n";
    }
  }
};

struct cache_drop_result {
  bool attempted{};
  std::size_t files{};
  std::size_t errors{};
};

void print_usage()
{
  std::cout
    << "Usage: mint1t_hybrid_scan [options]\n\n"
    << "  --dataset-root PATH            Dataset root; sets the default image-shard directory\n"
    << "  --refs PATH                    Reference Parquet (default: original manifest)\n"
    << "  --image-parquet-dir PATH       Directory containing image Parquet shards\n"
    << "  --limit N                      Use the first N manifest rows\n"
    << "  --pass-read-limit-mib N        Per-pass limit; default is 105% of GPU memory, 0 unbounded\n"
    << "  --use-data-page-mask YES|NO    Toggle payload data-page pruning (default YES)\n"
    << "  --use-sparse-page-io YES|NO    Toggle sparse physical payload I/O (default YES)\n"
    << "  --best-effort-drop-cache       Apply POSIX_FADV_DONTNEED before timed source setup\n"
    << "  --validate                     Compare with the projected Python output\n"
    << "  --expected-output PATH         Validation Parquet used by --validate\n"
    << "  --output PATH                  Write the payload-only C++ result under /MINT1T_rewritten\n"
    << "  -h, --help                     Show this message\n";
}

arguments parse_args(int argc, char const** argv)
{
  arguments args;
  auto image_dir_set       = false;
  auto dataset_root_set    = false;
  auto expected_output_set = false;
  auto require_value = [&](int& index, std::string_view option) -> std::string {
    if (++index >= argc) {
      throw std::invalid_argument("Missing value for " + std::string{option});
    }
    return argv[index];
  };

  for (int index = 1; index < argc; ++index) {
    auto const option = std::string_view{argv[index]};
    if (option == "-h" or option == "--help") {
      print_usage();
      std::exit(0);
    } else if (option == "--dataset-root") {
      args.dataset_root = require_value(index, option);
      dataset_root_set  = true;
    } else if (option == "--refs") {
      args.refs = require_value(index, option);
    } else if (option == "--image-parquet-dir") {
      args.image_dir = require_value(index, option);
      image_dir_set   = true;
    } else if (option == "--limit") {
      auto const value = std::stoll(require_value(index, option));
      CUDF_EXPECTS(value >= 0 and value <= std::numeric_limits<cudf::size_type>::max(),
                   "Invalid --limit");
      args.limit = static_cast<cudf::size_type>(value);
    } else if (option == "--pass-read-limit-mib") {
      auto const value = std::stoull(require_value(index, option));
      CUDF_EXPECTS(value <= std::numeric_limits<std::size_t>::max() / (1024 * 1024),
                   "Invalid --pass-read-limit-mib");
      args.pass_read_limit     = value * 1024 * 1024;
      args.pass_read_limit_set = true;
    } else if (option == "--use-data-page-mask") {
      args.use_page_mask = get_boolean(require_value(index, option));
    } else if (option == "--use-sparse-page-io") {
      args.use_sparse_page_io = get_boolean(require_value(index, option));
    } else if (option == "--best-effort-drop-cache") {
      args.drop_cache = true;
    } else if (option == "--validate") {
      args.validate = true;
    } else if (option == "--expected-output") {
      args.expected_output = require_value(index, option);
      args.validate        = true;
      expected_output_set  = true;
    } else if (option == "--output") {
      args.output = require_value(index, option);
    } else {
      throw std::invalid_argument("Unknown option: " + std::string{option});
    }
  }
  if (not image_dir_set) {
    args.image_dir =
      args.dataset_root / "parquet/mint_1t_html_images_stable_row_ids/data";
  }
  if (dataset_root_set and not expected_output_set) {
    auto const output_dir =
      args.dataset_root == std::filesystem::path{original_dataset_root}
        ? std::filesystem::path{writable_root} / "original"
        : std::filesystem::path{writable_root} / "output";
    args.expected_output = output_dir / "image_url_payloads.parquet";
  }
  return args;
}

void require_writable_output_path(std::filesystem::path const& path)
{
  auto const output = std::filesystem::absolute(path).lexically_normal();
  auto const root   = std::filesystem::path{writable_root}.lexically_normal();
  auto const rel    = output.lexically_relative(root);
  CUDF_EXPECTS(not rel.empty() and *rel.begin() != "..",
               "Output paths must be under " + root.string());
}

std::vector<reference> refs_to_host(cudf::table_view refs, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(refs.num_columns() == 2, "Unexpected reference table column count");
  CUDF_EXPECTS(refs.column(0).null_count() == 0 and refs.column(1).null_count() == 0,
               "Null reference fields are not supported");

  auto const strings = cudf::strings_column_view{refs.column(0)};
  auto host_offsets  = std::vector<cudf::size_type>(strings.size() + 1);
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_offsets.data(),
                                strings.offsets().data<cudf::size_type>() + strings.offset(),
                                host_offsets.size() * sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  auto const first_char = host_offsets.front();
  auto const chars_size = host_offsets.back() - first_char;
  auto host_chars       = std::vector<char>(chars_size);
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_chars.data(),
                                strings.chars_begin(stream) + first_char,
                                host_chars.size(),
                                cudaMemcpyDeviceToHost,
                                stream.value()));

  auto host_row_offsets = std::vector<uint32_t>(refs.num_rows());
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_row_offsets.data(),
                                refs.column(1).data<uint32_t>() + refs.column(1).offset(),
                                host_row_offsets.size() * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  auto result = std::vector<reference>{};
  result.reserve(refs.num_rows());
  for (cudf::size_type row = 0; row < refs.num_rows(); ++row) {
    auto const begin = host_offsets[row] - first_char;
    auto const end   = host_offsets[row + 1] - first_char;
    result.push_back({std::string{host_chars.data() + begin, static_cast<std::size_t>(end - begin)},
                      host_row_offsets[row]});
  }
  return result;
}

cache_drop_result drop_file_cache(std::vector<std::filesystem::path> const& paths)
{
  cache_drop_result result{.attempted = true};
  for (auto const& path : paths) {
    auto const fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      ++result.errors;
      continue;
    }
    if (::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0) {
      ++result.files;
    } else {
      ++result.errors;
    }
    ::close(fd);
  }
  return result;
}

std::unique_ptr<cudf::column> make_device_column(std::span<uint8_t const> host_data,
                                                 cudf::data_type type,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto result =
    cudf::make_numeric_column(type, host_data.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(result->mutable_view().data<uint8_t>(),
                                host_data.data(),
                                host_data.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  stream.synchronize();
  return result;
}

std::unique_ptr<cudf::column> make_gather_map(std::span<cudf::size_type const> host_data,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto result = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          host_data.size(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(result->mutable_view().data<cudf::size_type>(),
                                host_data.data(),
                                host_data.size_bytes(),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  stream.synchronize();
  return result;
}

uint64_t payload_bytes(cudf::table_view table, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(table.num_columns() == static_cast<cudf::size_type>(payload_columns.size()),
               "Unexpected payload table schema");
  auto const& sizes = table.column(3);
  auto host_sizes   = std::vector<int64_t>(table.num_rows());
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_sizes.data(),
                                sizes.data<int64_t>() + sizes.offset(),
                                host_sizes.size() * sizeof(int64_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  return std::accumulate(host_sizes.begin(), host_sizes.end(), uint64_t{0});
}

void write_output(std::filesystem::path const& path, cudf::table_view table)
{
  require_writable_output_path(path);
  std::filesystem::create_directories(path.parent_path());
  auto metadata = cudf::io::table_input_metadata{table};
  for (std::size_t index = 0; index < payload_columns.size(); ++index) {
    metadata.column_metadata[index].set_name(payload_columns[index]);
  }
  metadata.column_metadata.front().set_output_as_binary(true);
  auto options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{path.string()}, table)
      .metadata(std::move(metadata))
      .compression(cudf::io::compression_type::ZSTD)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .build();
  cudf::io::write_parquet(options);
}

std::unique_ptr<cudf::table> read_expected(std::filesystem::path const& path)
{
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path.string()})
                   .column_names(payload_columns)
                   .build();
  return std::move(cudf::io::read_parquet(options).tbl);
}

void print_json(timing_data const& timings,
                cache_drop_result const& cache,
                bool use_page_mask,
                bool use_sparse_page_io,
                std::size_t files,
                std::size_t row_groups,
                cudf::size_type rows,
                uint64_t selected_payload_bytes,
                uint64_t requested_bytes,
                std::size_t output_allocated_bytes,
                double benchmark_elapsed,
                double end_to_end_elapsed)
{
  auto const payload_mib = selected_payload_bytes / double{1024 * 1024};
  auto const fetch_time  = timings.get("payload byte-range fetch");
  auto const decode_time = timings.get("payload materialization/decode");

  std::cout << "\n{"
            << R"("method":"hybrid_scan_multifile_payload_mask",)"
            << "\"use_data_page_mask\":" << std::boolalpha << use_page_mask << ","
            << "\"use_sparse_page_io\":" << use_sparse_page_io << ","
            << "\"elapsed_s\":" << benchmark_elapsed << ","
            << "\"end_to_end_elapsed_s\":" << end_to_end_elapsed << ","
            << "\"rows\":" << rows << ","
            << "\"payload_bytes\":" << selected_payload_bytes << ","
            << "\"payload_mib\":" << payload_mib << ","
            << "\"rows_per_s\":" << (benchmark_elapsed > 0 ? rows / benchmark_elapsed : 0) << ","
            << "\"payload_mib_per_s\":"
            << (benchmark_elapsed > 0 ? payload_mib / benchmark_elapsed : 0) << ","
            << "\"referenced_files\":" << files << ","
            << "\"row_groups_read\":" << row_groups << ","
            << "\"compressed_payload_bytes_requested\":" << requested_bytes << ","
            << "\"output_allocated_bytes\":" << output_allocated_bytes << ","
            << "\"payload_fetch_mib_per_s\":"
            << (fetch_time > 0 ? requested_bytes / double{1024 * 1024} / fetch_time : 0) << ","
            << "\"materialization_mib_per_s\":"
            << (decode_time > 0 ? output_allocated_bytes / double{1024 * 1024} / decode_time : 0)
            << ","
            << R"("best_effort_cache_drop":{"attempted":)" << cache.attempted
            << ",\"files\":" << cache.files << ",\"errors\":" << cache.errors << "},"
            << "\"stage_seconds\":{";
  for (std::size_t index = 0; index < timings.stages.size(); ++index) {
    if (index != 0) { std::cout << ","; }
    std::cout << "\"" << timings.stages[index].first << "\":" << timings.stages[index].second;
  }
  std::cout << "}}\n";
}

}  // namespace

int main(int argc, char const** argv)
{
  auto const program_start = clock_type::now();
  auto timings             = timing_data{};
  auto args                = parse_args(argc, argv);
  if (not args.pass_read_limit_set) {
    auto free_bytes  = std::size_t{};
    auto total_bytes = std::size_t{};
    CUDF_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
    args.pass_read_limit = total_bytes + total_bytes / 20;
  }
  auto const stream        = cudf::get_default_stream();
  auto resource            = create_memory_resource(false);
  auto stats_mr            = rmm::mr::statistics_resource_adaptor{resource};
  rmm::mr::set_current_device_resource(stats_mr);

  auto start = clock_type::now();
  auto refs_options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{args.refs.string()})
      .column_names({"image_parquet", "_row_offset"})
      .build();
  if (args.limit.has_value()) { refs_options.set_num_rows(args.limit.value()); }
  auto refs_table = std::move(cudf::io::read_parquet(refs_options, stream).tbl);
  stream.synchronize();
  timings.add("reference parquet read", start);

  start     = clock_type::now();
  auto refs = refs_to_host(refs_table->view(), stream);
  std::stable_sort(refs.begin(), refs.end());
  CUDF_EXPECTS(not refs.empty(), "No references to read");

  auto unique_refs = std::vector<reference>{};
  auto gather_map  = std::vector<cudf::size_type>{};
  unique_refs.reserve(refs.size());
  gather_map.reserve(refs.size());
  for (auto const& ref : refs) {
    if (unique_refs.empty() or not(unique_refs.back() == ref)) { unique_refs.push_back(ref); }
    gather_map.push_back(static_cast<cudf::size_type>(unique_refs.size() - 1));
  }

  auto source_names   = std::vector<std::string>{};
  auto source_offsets = std::vector<std::vector<uint32_t>>{};
  for (auto const& ref : unique_refs) {
    if (source_names.empty() or source_names.back() != ref.image_parquet) {
      source_names.push_back(ref.image_parquet);
      source_offsets.emplace_back();
    }
    source_offsets.back().push_back(ref.row_offset);
  }
  auto source_paths = std::vector<std::filesystem::path>{};
  auto source_files = std::vector<std::string>{};
  source_paths.reserve(source_names.size());
  source_files.reserve(source_names.size());
  for (auto const& name : source_names) {
    auto const path = args.image_dir / name;
    CUDF_EXPECTS(std::filesystem::is_regular_file(path), "Missing source file: " + path.string());
    source_paths.push_back(path);
    source_files.push_back(path.string());
  }
  timings.add("reference host extraction/sort/group", start);

  auto cache_result = cache_drop_result{};
  start             = clock_type::now();
  if (args.drop_cache) {
    auto cache_paths = source_paths;
    cache_paths.insert(cache_paths.begin(), args.refs);
    cache_result = drop_file_cache(cache_paths);
  }
  timings.add("best-effort cache drop", start);

  auto const benchmark_start = clock_type::now();

  start       = clock_type::now();
  auto inputs = multifile_inputs{cudf::io::source_info{source_files}};
  timings.add("datasource construction", start);

  start = clock_type::now();
  inputs.fetch_footers();
  timings.add("footer fetch", start);

  auto options = cudf::io::parquet_reader_options::builder().column_names(payload_columns).build();
  start        = clock_type::now();
  auto reader  = hybrid_scan_multifile{inputs.footer_byte_spans, options};
  timings.add("hybrid reader construction", start);

  start                = clock_type::now();
  auto const metadatas = reader.parquet_metadatas();
  auto row_groups      = std::vector<std::vector<cudf::size_type>>(source_names.size());
  auto host_row_mask   = std::vector<uint8_t>{};
  std::size_t selected_row_groups{};
  for (std::size_t source = 0; source < source_names.size(); ++source) {
    auto const& metadata = metadatas[source];
    auto const& offsets  = source_offsets[source];
    auto offset_index    = std::size_t{0};
    auto file_row_start  = uint64_t{0};
    for (std::size_t rg = 0; rg < metadata.row_groups.size() and offset_index < offsets.size();
         ++rg) {
      auto const rows = metadata.row_groups[rg].num_rows;
      CUDF_EXPECTS(rows >= 0, "Negative row count in Parquet metadata");
      auto const rows_unsigned = static_cast<uint64_t>(rows);
      auto const file_row_stop = file_row_start + rows_unsigned;
      if (offsets[offset_index] < file_row_start) {
        throw std::logic_error("References are not ordered");
      }
      if (offsets[offset_index] < file_row_stop) {
        row_groups[source].push_back(static_cast<cudf::size_type>(rg));
        ++selected_row_groups;
        auto const mask_start = host_row_mask.size();
        host_row_mask.resize(mask_start + static_cast<std::size_t>(rows_unsigned), uint8_t{0});
        while (offset_index < offsets.size() and offsets[offset_index] < file_row_stop) {
          auto const local_offset =
            static_cast<std::size_t>(offsets[offset_index] - file_row_start);
          host_row_mask[mask_start + local_offset] = uint8_t{1};
          ++offset_index;
        }
      }
      file_row_start = file_row_stop;
    }
    CUDF_EXPECTS(offset_index == offsets.size(),
                 "Out-of-range row offset in " + source_names[source]);
  }
  CUDF_EXPECTS(
    host_row_mask.size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "Selected row groups exceed cudf column size limit");
  timings.add("row-group planning/host mask", start);

  start             = clock_type::now();
  auto page_ranges  = reader.page_index_byte_ranges();
  auto missing_page = std::find_if(
    page_ranges.begin(), page_ranges.end(), [](auto const& range) { return range.is_empty(); });
  CUDF_EXPECTS(missing_page == page_ranges.end(), "A referenced source has no Parquet page index");
  timings.add("page-index range planning", start);

  start = clock_type::now();
  auto page_buffers =
    cudf::io::parquet::fetch_page_indexes_to_host(inputs.datasource_refs, page_ranges);
  auto page_spans = std::vector<cudf::host_span<uint8_t const>>{};
  page_spans.reserve(page_buffers.size());
  std::transform(page_buffers.begin(),
                 page_buffers.end(),
                 std::back_inserter(page_spans),
                 [](auto const& buffer) { return cudf::host_span<uint8_t const>{*buffer}; });
  timings.add("page-index fetch", start);

  start = clock_type::now();
  reader.setup_page_indexes(page_spans);
  timings.add("page-index setup", start);

  start                        = clock_type::now();
  auto const indexed_metadatas = reader.parquet_metadatas();
  for (std::size_t source = 0; source < row_groups.size(); ++source) {
    for (auto const rg : row_groups[source]) {
      auto const& chunks = indexed_metadatas[source].row_groups[rg].columns;
      for (auto const& name : payload_columns) {
        auto const chunk = std::find_if(chunks.begin(), chunks.end(), [&](auto const& candidate) {
          auto const& path = candidate.meta_data.path_in_schema;
          return not path.empty() and path.back() == name;
        });
        CUDF_EXPECTS(chunk != chunks.end(),
                     "Missing payload column " + name + " in " + source_names[source]);
        CUDF_EXPECTS(chunk->offset_index.has_value(),
                     "Missing offset index for " + source_names[source] + " row group " +
                       std::to_string(rg) + " column " + name);
      }
    }
  }
  timings.add("selected index validation", start);

  start = clock_type::now();
  auto row_mask =
    make_device_column(host_row_mask, cudf::data_type{cudf::type_id::BOOL8}, stream, stats_mr);
  timings.add("row-mask host-to-device", start);

  start             = clock_type::now();
  auto const passes = reader.construct_row_group_passes(row_groups, args.pass_read_limit);
  timings.add("row-group pass construction", start);

  auto requested_bytes       = uint64_t{0};
  auto selected_payload_bytes = uint64_t{0};
  auto output_allocated_bytes = std::size_t{0};
  auto global_mask_start      = cudf::size_type{0};
  auto pass_tables            = std::vector<std::unique_ptr<cudf::table>>{};
  auto const retain_output =
    args.validate or args.output.has_value() or refs.size() != unique_refs.size();

  auto range_planning_seconds = double{0};
  auto range_fetch_seconds    = double{0};
  auto chunking_setup_seconds = double{0};
  auto materialize_seconds    = double{0};
  auto stat_seconds           = double{0};

  std::cout << "PROGRESS constructed " << passes.size() << " multifile passes (limit "
            << args.pass_read_limit / double{1024 * 1024} << " MiB)\n"
            << std::flush;
  for (std::size_t pass_index = 0; pass_index < passes.size(); ++pass_index) {
    auto const& pass      = passes[pass_index];
    auto const pass_start = clock_type::now();
    auto const pass_row_groups =
      std::accumulate(pass.begin(),
                      pass.end(),
                      std::size_t{0},
                      [](auto count, auto const& source_row_groups) {
                        return count + source_row_groups.size();
                      });
    auto const pass_rows = reader.total_rows_in_row_groups(pass);
    auto const pass_mask = cudf::column_view(
      row_mask->type(), pass_rows, row_mask->view().data<bool>(), nullptr, 0, global_mask_start);
    std::cout << "PROGRESS pass " << pass_index + 1 << "/" << passes.size() << " started ("
              << pass_row_groups << " row groups, " << pass_rows << " input rows)\n"
              << std::flush;

    auto payload_data = multisource_device_data{};
    if (args.use_page_mask and args.use_sparse_page_io) {
      start = clock_type::now();
      auto const payload_ranges = reader.payload_column_chunks_byte_ranges(
        pass, pass_mask, use_data_page_mask::YES, options, stream);
      for (auto const& source_ranges : payload_ranges) {
        requested_bytes =
          std::accumulate(source_ranges.begin(),
                          source_ranges.end(),
                          requested_bytes,
                          [](uint64_t sum, auto const& range) { return sum + range.size(); });
      }
      range_planning_seconds += std::chrono::duration<double>(clock_type::now() - start).count();

      start        = clock_type::now();
      payload_data = fetch_multisource_device_data(inputs, payload_ranges, stream, stats_mr);
      range_fetch_seconds += std::chrono::duration<double>(clock_type::now() - start).count();

      start = clock_type::now();
      reader.setup_chunking_for_payload_columns(0,
                                                0,
                                                pass,
                                                pass_mask,
                                                use_data_page_mask::YES,
                                                payload_data.per_source_spans,
                                                options,
                                                stream,
                                                stats_mr);
    } else {
      start               = clock_type::now();
      auto payload_ranges = reader.payload_column_chunks_byte_ranges(pass, options);
      requested_bytes +=
        std::accumulate(payload_ranges.first.begin(),
                        payload_ranges.first.end(),
                        uint64_t{0},
                        [](uint64_t sum, auto const& range) { return sum + range.size(); });
      range_planning_seconds += std::chrono::duration<double>(clock_type::now() - start).count();

      start        = clock_type::now();
      payload_data = fetch_multisource_device_data(inputs, payload_ranges, stream, stats_mr);
      range_fetch_seconds += std::chrono::duration<double>(clock_type::now() - start).count();

      start = clock_type::now();
      reader.setup_chunking_for_payload_columns(0,
                                                0,
                                                pass,
                                                pass_mask,
                                                args.use_page_mask ? use_data_page_mask::YES
                                                                   : use_data_page_mask::NO,
                                                payload_data.flat_spans,
                                                options,
                                                stream,
                                                stats_mr);
    }
    stream.synchronize();
    chunking_setup_seconds += std::chrono::duration<double>(clock_type::now() - start).count();

    auto current_tables = std::vector<std::unique_ptr<cudf::table>>{};
    start               = clock_type::now();
    while (reader.has_next_table_chunk()) {
      current_tables.push_back(reader.materialize_payload_columns_chunk(pass_mask).tbl);
    }
    stream.synchronize();
    materialize_seconds += std::chrono::duration<double>(clock_type::now() - start).count();

    if (retain_output) {
      std::move(current_tables.begin(), current_tables.end(), std::back_inserter(pass_tables));
    } else {
      start = clock_type::now();
      for (auto const& table : current_tables) {
        selected_payload_bytes += payload_bytes(table->view(), stream);
        output_allocated_bytes += table->alloc_size();
      }
      stream.synchronize();
      stat_seconds += std::chrono::duration<double>(clock_type::now() - start).count();
    }
    global_mask_start += pass_rows;
    auto const pass_seconds =
      std::chrono::duration<double>(clock_type::now() - pass_start).count();
    std::cout << "PROGRESS pass " << pass_index + 1 << "/" << passes.size() << " completed in "
              << std::fixed << std::setprecision(3) << pass_seconds << " s\n"
              << std::flush;
  }
  CUDF_EXPECTS(std::cmp_equal(global_mask_start, host_row_mask.size()),
               "Row-group passes do not span the global row mask");

  timings.add_seconds("payload byte-range planning", range_planning_seconds);
  timings.add_seconds("payload byte-range fetch", range_fetch_seconds);
  timings.add_seconds("payload chunking setup", chunking_setup_seconds);
  timings.add_seconds("payload materialization/decode", materialize_seconds);

  start = clock_type::now();
  auto output = [&]() -> std::unique_ptr<cudf::table> {
    if (not retain_output) { return nullptr; }
    CUDF_EXPECTS(not pass_tables.empty(), "Payload materialization produced no table chunks");
    if (pass_tables.size() == 1) { return std::move(pass_tables.front()); }
    auto views = std::vector<cudf::table_view>{};
    views.reserve(pass_tables.size());
    std::transform(
      pass_tables.begin(), pass_tables.end(), std::back_inserter(views), [](auto const& table) {
        return table->view();
      });
    return cudf::concatenate(views, stream, stats_mr);
  }();
  stream.synchronize();
  timings.add("payload pass concatenation", start);

  start = clock_type::now();
  if (refs.size() != unique_refs.size()) {
    CUDF_EXPECTS(output != nullptr, "Duplicate references require a retained output table");
    auto device_gather_map = make_gather_map(gather_map, stream, stats_mr);
    output                 = cudf::gather(output->view(),
                          device_gather_map->view(),
                          cudf::out_of_bounds_policy::DONT_CHECK,
                          cudf::negative_index_policy::NOT_ALLOWED,
                          stream,
                          stats_mr);
    stream.synchronize();
  }
  timings.add("duplicate-order gather", start);

  if (retain_output) {
    start                    = clock_type::now();
    selected_payload_bytes   = payload_bytes(output->view(), stream);
    output_allocated_bytes   = output->alloc_size();
    stream.synchronize();
    stat_seconds = std::chrono::duration<double>(clock_type::now() - start).count();
  }
  timings.add_seconds("payload byte/stat extraction", stat_seconds);

  start = clock_type::now();
  std::unique_ptr<cudf::table> expected;
  if (args.validate) {
    CUDF_EXPECTS(std::filesystem::is_regular_file(args.expected_output),
                 "Expected output does not exist: " + args.expected_output.string());
    expected = read_expected(args.expected_output);
    stream.synchronize();
  }
  timings.add("expected-output read", start);

  start = clock_type::now();
  if (args.validate) {
    auto const equal =
      cudf::tables_equal(output->view(), expected->view(), cudf::null_equality::EQUAL, stream);
    stream.synchronize();
    CUDF_EXPECTS(equal, "Hybrid Scan output differs from Python output");
  }
  timings.add("table comparison", start);

  auto const benchmark_elapsed =
    std::chrono::duration<double>(clock_type::now() - benchmark_start).count();

  start = clock_type::now();
  if (args.output.has_value()) { write_output(args.output.value(), output->view()); }
  stream.synchronize();
  timings.add("output parquet write", start);

  start = clock_type::now();
  expected.reset();
  output.reset();
  pass_tables.clear();
  row_mask.reset();
  stream.synchronize();
  timings.add("result cleanup", start);

  auto const end_to_end_elapsed =
    std::chrono::duration<double>(clock_type::now() - program_start).count();
  timings.print();
  std::cout << "\nSummary:\n"
            << "  setup + payload benchmark: " << benchmark_elapsed << " s\n"
            << "  end-to-end:               " << end_to_end_elapsed << " s\n"
            << "  referenced files:         " << source_names.size() << "\n"
            << "  selected row groups:      " << selected_row_groups << "\n"
            << "  selected rows:            " << refs.size() << "\n"
            << "  selected image bytes:     " << selected_payload_bytes << "\n"
            << "  requested compressed bytes: " << requested_bytes << "\n"
            << "  use data page mask:        " << std::boolalpha << args.use_page_mask << "\n"
            << "  use sparse page I/O:       "
            << (args.use_page_mask and args.use_sparse_page_io) << "\n";
  print_json(timings,
             cache_result,
             args.use_page_mask,
             args.use_page_mask and args.use_sparse_page_io,
             source_names.size(),
             selected_row_groups,
             static_cast<cudf::size_type>(refs.size()),
             selected_payload_bytes,
             requested_bytes,
             output_allocated_bytes,
             benchmark_elapsed,
             end_to_end_elapsed);
  std::cout << "Peak device memory: " << stats_mr.get_bytes_counter().peak / double{1024 * 1024}
            << " MiB\n";
  return 0;
}
