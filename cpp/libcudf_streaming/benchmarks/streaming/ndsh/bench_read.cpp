/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils.hpp"

#include <cudf/context.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <cudf_streaming/parquet.hpp>

#include <rmm/detail/format.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <coro/when_all.hpp>
#include <getopt.h>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

namespace {

rapidsmpf::streaming::Actor read_parquet(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                         std::shared_ptr<rapidsmpf::Communicator> comm,
                                         std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                         std::size_t num_producers,
                                         cudf::size_type num_rows_per_chunk,
                                         std::optional<std::vector<std::string>> columns,
                                         std::string const& input_directory,
                                         std::string const& input_file)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, input_file));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files)).build();
  if (columns.has_value()) { options.set_column_names(*columns); }
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
}

rapidsmpf::streaming::Actor consume_channel_parallel(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::size_t num_consumers)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in};
  std::atomic<std::size_t> estimated_total_bytes{0};
  auto task = [&]() -> rapidsmpf::streaming::Actor {
    co_await ctx->executor()->schedule();
    while (true) {
      auto msg = co_await ch_in->receive();
      if (msg.empty()) { break; }
      if (msg.holds<cudf_streaming::table_chunk>()) {
        auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
        ctx->logger()->print("Consumed chunk with ",
                             chunk.table_view().num_rows(),
                             " rows and ",
                             chunk.table_view().num_columns(),
                             " columns");
        estimated_total_bytes.fetch_add(chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE));
      }
    }
  };
  std::vector<rapidsmpf::streaming::Actor> tasks;
  for (std::size_t i = 0; i < num_consumers; i++) {
    tasks.push_back(task());
  }
  rapidsmpf::streaming::coro_results(co_await coro::when_all(std::move(tasks)));
  ctx->logger()->print("Table was around ",
                       rmm::detail::format_bytes(estimated_total_bytes.load()));
}

///< @brief Configuration options for the benchmark
struct ProgramOptions {
  int num_streaming_threads{1};  ///< Number of streaming threads to use
  int num_iterations{2};         ///< Number of iterations of query to run
  int num_streams{16};           ///< Number of streams in stream pool
  rapidsmpf::ndsh::CommType comm_type{
    rapidsmpf::ndsh::ProgramOptions{}.comm_type};   ///< Type of communicator to create
  cudf::size_type num_rows_per_chunk{100'000'000};  ///< Number of rows to produce per chunk read
  std::size_t num_producers{1};  ///< Number of simultaneous read_parquet chunk producers.
  std::size_t num_consumers{1};  ///< Number of simultaneous chunk consumers.
  std::string input_directory;   ///< Directory containing input files.
  std::string input_file;        ///< Basename of input file to read.
  std::optional<std::vector<std::string>> columns{std::nullopt};  ///< Columns to read.
};

ProgramOptions parse_arguments(int argc, char** argv)
{
  ProgramOptions options;

  auto const comm_names = rapidsmpf::ndsh::comm_type_names();

  auto print_usage = [&argv, &comm_names, &options]() {
    std::cerr << "Usage: " << argv[0] << " [options]\n"
              << "Options:\n"
              << "  --num-streaming-threads <n>  Number of streaming threads (default: "
              << options.num_streaming_threads << ")\n"
              << "  --num-iterations <n>         Number of iterations (default: "
              << options.num_iterations << ")\n"
              << "  --num-streams <n>            Number of streams in stream pool "
                 "(default: "
              << options.num_streams << ")\n"
              << "  --num-rows-per-chunk <n>     Number of rows per chunk (default: "
              << options.num_rows_per_chunk << ")\n"
              << "  --num-producers <n>          Number of concurrent read_parquet "
                 "producers (default: "
              << options.num_producers << ")\n"
              << "  --num-consumers <n>          Number of concurrent consumers (default: "
              << options.num_consumers << ")\n"
              << "  --comm-type <type>           Communicator type: "
              << rapidsmpf::ndsh::available_comm_types()
              << " "
                 "(default: "
              << comm_names[static_cast<std::size_t>(options.comm_type)] << ")\n"
              << "  --input-directory <path>     Input directory path (required)\n"
              << "  --input-file <file>          Input file basename relative to input "
                 "directory (required)\n"
              << "  --columns <a,b,c>            Comma-separated column names to read "
                 "(optional, default all columns)\n"
              << "  --help                       Show this help message\n";
  };

  // NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)
  static struct option long_options[] = {{"num-streaming-threads", required_argument, nullptr, 1},
                                         {"num-rows-per-chunk", required_argument, nullptr, 2},
                                         {"num-producers", required_argument, nullptr, 3},
                                         {"num-consumers", required_argument, nullptr, 4},
                                         {"input-directory", required_argument, nullptr, 5},
                                         {"input-file", required_argument, nullptr, 6},
                                         {"help", no_argument, nullptr, 7},
                                         {"num-iterations", required_argument, nullptr, 8},
                                         {"num-streams", required_argument, nullptr, 9},
                                         {"comm-type", required_argument, nullptr, 10},
                                         {"columns", required_argument, nullptr, 11},
                                         {nullptr, 0, nullptr, 0}};
  // NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)

  int opt;
  int option_index = 0;

  bool saw_input_directory = false;
  bool saw_input_file      = false;

  auto parse_i64 = [](char const* s, char const* opt_name) -> long long {
    if (s == nullptr || *s == '\0') {
      std::cerr << "Error: " << opt_name << " requires a value\n";
      std::exit(1);
    }
    errno        = 0;
    char* end    = nullptr;
    auto const v = std::strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
      std::cerr << "Error: invalid integer for " << opt_name << ": '" << s << "'\n";
      std::exit(1);
    }
    return v;
  };

  auto parse_u64 = [](char const* s, char const* opt_name) -> unsigned long long {
    if (s == nullptr || *s == '\0') {
      std::cerr << "Error: " << opt_name << " requires a value\n";
      std::exit(1);
    }
    errno        = 0;
    char* end    = nullptr;
    auto const v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
      std::cerr << "Error: invalid non-negative integer for " << opt_name << ": '" << s << "'\n";
      std::exit(1);
    }
    return v;
  };

  auto require_positive_i32 = [&](char const* s, char const* opt_name) -> int {
    auto const v = parse_i64(s, opt_name);
    if (v <= 0 || v > std::numeric_limits<int>::max()) {
      std::cerr << "Error: " << opt_name << " must be in [1, " << std::numeric_limits<int>::max()
                << "], got '" << s << "'\n";
      std::exit(1);
    }
    return static_cast<int>(v);
  };

  auto require_positive_size_t = [&](char const* s, char const* opt_name) -> std::size_t {
    auto const v = parse_u64(s, opt_name);
    if (v == 0 || v > std::numeric_limits<std::size_t>::max()) {
      std::cerr << "Error: " << opt_name << " must be in [1, "
                << std::numeric_limits<std::size_t>::max() << "], got '" << s << "'\n";
      std::exit(1);
    }
    return static_cast<std::size_t>(v);
  };

  auto parse_columns = [](char const* s) -> std::optional<std::vector<std::string>> {
    if (s == nullptr) { return std::nullopt; }
    std::string str{s};
    if (str.empty()) { return std::nullopt; }
    std::vector<std::string> cols;
    std::size_t start = 0;
    while (start <= str.size()) {
      auto const comma = str.find(',', start);
      auto const end   = (comma == std::string::npos) ? str.size() : comma;
      auto const token = str.substr(start, end - start);
      if (token.empty()) {
        std::cerr << "Error: --columns contains an empty column name\n";
        std::exit(1);
      }
      cols.push_back(token);
      if (comma == std::string::npos) { break; }
      start = comma + 1;
    }
    return cols;
  };

  while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
    switch (opt) {
      case 1:  // --num-streaming-threads
        options.num_streaming_threads = require_positive_i32(optarg, "--num-streaming-threads");
        break;
      case 2:  // --num-rows-per-chunk
        options.num_rows_per_chunk = require_positive_i32(optarg, "--num-rows-per-chunk");
        break;
      case 3:  // --num-producers
        options.num_producers = require_positive_size_t(optarg, "--num-producers");
        break;
      case 4:  // --num-consumers
        options.num_consumers = require_positive_size_t(optarg, "--num-consumers");
        break;
      case 5:  // --input-directory
        if (optarg == nullptr || *optarg == '\0') {
          std::cerr << "Error: --input-directory requires a non-empty value\n";
          std::exit(1);
        }
        options.input_directory = optarg;
        saw_input_directory     = true;
        break;
      case 6:  // --input-file
        if (optarg == nullptr || *optarg == '\0') {
          std::cerr << "Error: --input-file requires a non-empty value\n";
          std::exit(1);
        }
        options.input_file = optarg;
        saw_input_file     = true;
        break;
      case 7:  // --help
        print_usage();
        std::exit(0);
      case 8:  // --num-iterations
        options.num_iterations = require_positive_i32(optarg, "--num-iterations");
        break;
      case 9:  // --num-streams
        options.num_streams = require_positive_i32(optarg, "--num-streams");
        break;
      case 10: {  // --comm-type
        if (optarg == nullptr || *optarg == '\0') {
          std::cerr << "Error: --comm-type requires a value\n";
          std::exit(1);
        }
        std::string_view const s{optarg};
        auto parsed = rapidsmpf::ndsh::parse_comm_type(s);
        if (!parsed.has_value()) {
          std::cerr << "Error: invalid --comm-type '" << s << "' (expected: single, mpi, ucxx)\n";
          std::exit(1);
        }
        if (!rapidsmpf::ndsh::is_comm_type_available(*parsed)) {
          std::cerr << "Error: communicator '" << s
                    << "' is not available in this build (available: "
                    << rapidsmpf::ndsh::available_comm_types() << ")\n";
          std::exit(1);
        }
        options.comm_type = *parsed;
        break;
      }
      case 11:  // --columns
        options.columns = parse_columns(optarg);
        break;
      case '?':
        if (optopt == 0 && optind > 1) {
          std::cerr << "Error: Unknown option '" << argv[optind - 1] << "'\n\n";
        }
        print_usage();
        std::exit(1);
      default: print_usage(); std::exit(1);
    }
  }

  // Check if required options were provided
  if (!saw_input_directory || !saw_input_file) {
    if (!saw_input_directory) { std::cerr << "Error: --input-directory is required\n"; }
    if (!saw_input_file) { std::cerr << "Error: --input-file is required\n"; }
    std::cerr << std::endl;
    print_usage();
    std::exit(1);
  }

  return options;
}

}  // namespace

/**
 * @brief Run a simple benchmark reading a table from parquet files.
 */
int main(int argc, char** argv)
{
  rapidsmpf::ndsh::FinalizeMPI finalize{};
  CUDF_CUDA_TRY(cudaFree(nullptr));
  // work around https://github.com/rapidsai/cudf/issues/20849
  cudf::initialize();
  auto mr        = rmm::mr::cuda_async_memory_resource{};
  auto arguments = parse_arguments(argc, argv);
  rapidsmpf::ndsh::ProgramOptions ctx_arguments{
    .num_streaming_threads = arguments.num_streaming_threads,
    .num_iterations        = arguments.num_iterations,
    .num_streams           = arguments.num_streams,
    .comm_type             = arguments.comm_type,
    .num_rows_per_chunk    = arguments.num_rows_per_chunk,
    .output_file           = "",
    .input_directory       = arguments.input_directory};

  auto [ctx, comm] = rapidsmpf::ndsh::create_context(ctx_arguments, std::move(mr));
  std::vector<double> timings;
  for (int i = 0; i < arguments.num_iterations; i++) {
    std::vector<rapidsmpf::streaming::Actor> actors;
    auto start = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing read_parquet pipeline");

      // Input data channels
      auto ch_out = ctx->create_channel();
      actors.push_back(read_parquet(ctx,
                                    comm,
                                    ch_out,
                                    arguments.num_producers,
                                    arguments.num_rows_per_chunk,
                                    arguments.columns,
                                    arguments.input_directory,
                                    arguments.input_file));
      actors.push_back(consume_channel_parallel(ctx, ch_out, arguments.num_consumers));
    }
    auto end                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> pipeline = end - start;
    start                                  = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("read_parquet iteration");
      rapidsmpf::streaming::run_actor_network(std::move(actors));
    }
    end                                   = std::chrono::steady_clock::now();
    std::chrono::duration<double> compute = end - start;
    timings.push_back(pipeline.count());
    timings.push_back(compute.count());
    auto statistics = ctx->statistics();
    comm->logger()->print(
      statistics->report({.mr = ctx->br()->device_mr(), .pinned_mr = ctx->br()->try_pinned_mr()}));
    statistics->clear();
  }

  if (comm->rank() == 0) {
    for (int i = 0; i < arguments.num_iterations; i++) {
      comm->logger()->print("Iteration ",
                            i,
                            " pipeline construction time [s]: ",
                            timings[rapidsmpf::safe_cast<std::size_t>(2 * i)]);
      comm->logger()->print("Iteration ",
                            i,
                            " compute time [s]: ",
                            timings[rapidsmpf::safe_cast<std::size_t>(2 * i + 1)]);
    }
  }
  return 0;
}
