/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <getopt.h>
#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>

#ifdef CUDF_STREAMING_HAVE_MPI
#include <mpi.h>
#include <rapidsmpf/communicator/mpi.hpp>
#endif

#ifdef CUDF_STREAMING_HAVE_UCXX
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#endif

#if defined(CUDF_STREAMING_HAVE_MPI) && defined(CUDF_STREAMING_HAVE_UCXX)
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#endif

namespace rapidsmpf::ndsh {
namespace detail {
std::vector<std::string> list_parquet_files(std::string const root_path)
{
  auto root_entry = std::filesystem::directory_entry(std::filesystem::path(root_path));
  RAPIDSMPF_EXPECTS(
    root_entry.exists() && (root_entry.is_regular_file() || root_entry.is_directory()),
    "Invalid file path",
    std::runtime_error);
  if (root_entry.is_regular_file()) {
    RAPIDSMPF_EXPECTS(root_path.ends_with(".parquet"), "Invalid filename", std::runtime_error);
    return {root_path};
  }
  std::vector<std::string> result;
  for (auto const& entry : std::filesystem::directory_iterator(root_path)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      if (filename.ends_with(".parquet")) { result.push_back(entry.path()); }
    }
  }
  return result;
}

std::string get_table_path(std::string const& input_directory, std::string const& table_name)
{
  auto dir       = input_directory.empty() ? "." : input_directory;
  auto file_path = dir + "/" + table_name + ".parquet";

  if (std::filesystem::exists(file_path)) { return file_path; }

  return dir + "/" + table_name + "/";
}

std::map<std::string, cudf::data_type> get_column_types(std::string const& input_directory,
                                                        std::string const& table_name)
{
  auto files = list_parquet_files(get_table_path(input_directory, table_name));
  RAPIDSMPF_EXPECTS(!files.empty(), "No parquet files found for table " + table_name);

  auto metadata    = cudf::io::read_parquet_metadata(cudf::io::source_info(files[0]));
  auto const& root = metadata.schema().root();

  std::map<std::string, cudf::data_type> result;
  for (std::size_t i = 0; i < root.num_children(); ++i) {
    auto const& column = root.child(safe_cast<int>(i));
    result.emplace(column.name(), column.cudf_type());
  }
  return result;
}

}  // namespace detail

bool is_comm_type_available(CommType comm_type)
{
  switch (comm_type) {
    case CommType::SINGLE: return true;
    case CommType::MPI:
#ifdef CUDF_STREAMING_HAVE_MPI
      return true;
#endif
      break;
    case CommType::UCXX:
#ifdef CUDF_STREAMING_HAVE_UCXX
      return true;
#endif
      break;
    case CommType::MAX: break;
  }
  return false;
}

std::string available_comm_types()
{
  auto const names = comm_type_names();
  std::stringstream out;
  bool first = true;
  for (std::size_t i = 0; i < static_cast<std::size_t>(CommType::MAX); ++i) {
    auto const comm_type = static_cast<CommType>(i);
    if (!is_comm_type_available(comm_type)) { continue; }
    if (!first) { out << ", "; }
    out << names[i];
    first = false;
  }
  return out.str();
}

std::optional<CommType> parse_comm_type(std::string_view name)
{
  auto const names = comm_type_names();
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (name == names[i]) { return static_cast<CommType>(i); }
  }
  return std::nullopt;
}

streaming::Actor sink_channel(std::shared_ptr<streaming::Context> ctx,
                              std::shared_ptr<streaming::Channel> ch)
{
  co_await ctx->executor()->schedule();
  co_await ch->shutdown();
}

streaming::Actor consume_channel(std::shared_ptr<streaming::Context> ctx,
                                 std::shared_ptr<streaming::Channel> ch_in)
{
  streaming::ShutdownAtExit c{ch_in};
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
    }
  }
}

std::pair<std::shared_ptr<streaming::Context>, std::shared_ptr<Communicator>> create_context(
  ProgramOptions& arguments, cuda::mr::any_resource<cuda::mr::device_accessible> mr)
{
  rmm::mr::set_current_device_resource(mr);
  std::unordered_map<MemoryType, std::int64_t> memory_limits{};
  if (arguments.spill_device_limit.has_value()) {
    auto limit_size =
      rmm::align_down((rmm::available_device_memory().second *
                       static_cast<std::size_t>(arguments.spill_device_limit.value() * 100) / 100),
                      rmm::CUDA_ALLOCATION_ALIGNMENT);

    memory_limits[MemoryType::DEVICE] = static_cast<std::int64_t>(limit_size);
  }
  auto statistics = Statistics::create();

  RAPIDSMPF_EXPECTS(
    arguments.no_pinned_host_memory || is_pinned_memory_resources_supported(),
    "Pinned host memory is not supported on this system. "
    "CUDA " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
    " is one of the requirements, but additional platform or driver constraints may "
    "apply. If needed, use `--no-pinned-host-memory` to disable pinned host memory, "
    "noting that this may significantly degrade spilling performance.",
    std::invalid_argument);

  auto pinned_pool_properties =
    arguments.no_pinned_host_memory ? PinnedMemoryDisabled : PinnedPoolProperties{};
  auto br                              = BufferResource::create(std::move(mr),
                                   std::move(pinned_pool_properties),
                                   std::move(memory_limits),
                                   arguments.periodic_spill,
                                   std::make_shared<rmm::cuda_stream_pool>(
                                     arguments.num_streams, rmm::cuda_stream::flags::non_blocking),
                                   statistics);
  auto environment                     = config::get_environment_variables();
  environment["NUM_STREAMING_THREADS"] = std::to_string(arguments.num_streaming_threads);
  auto options                         = config::Options(environment);
  auto log                             = Logger::from_options(options);
  auto progress_thread                 = std::make_shared<rapidsmpf::ProgressThread>(statistics);
  std::shared_ptr<Communicator> comm;
  switch (arguments.comm_type) {
    case CommType::MPI:
#ifdef CUDF_STREAMING_HAVE_MPI
      RAPIDSMPF_EXPECTS(!bootstrap::is_running_with_rrun(), "Can't use MPI communicator with rrun");
      mpi::init(nullptr, nullptr);

      comm = std::make_shared<MPI>(MPI_COMM_WORLD, progress_thread, log);
#else
      RAPIDSMPF_FAIL("MPI communicator is not available in this build", std::invalid_argument);
#endif
      break;
    case CommType::SINGLE: comm = std::make_shared<Single>(progress_thread, log); break;
    case CommType::UCXX:
#ifdef CUDF_STREAMING_HAVE_UCXX
      if (bootstrap::is_running_with_rrun()) {
        comm =
          bootstrap::create_ucxx_comm(progress_thread, bootstrap::BackendType::AUTO, options, log);
      } else {
#ifdef CUDF_STREAMING_HAVE_MPI
        mpi::init(nullptr, nullptr);
        comm = ucxx::init_using_mpi(MPI_COMM_WORLD, options, progress_thread, log);
#else
        RAPIDSMPF_FAIL("UCXX without MPI support requires bootstrap mode", std::invalid_argument);
#endif
      }
#else
      RAPIDSMPF_FAIL("UCXX communicator is not available in this build", std::invalid_argument);
#endif
      break;
    default: RAPIDSMPF_FAIL("Unknown communicator type");
  }
  auto ctx = std::make_shared<streaming::Context>(options, log, br);
  if (comm->rank() == 0) {
    log->print("Execution context on ",
               comm->nranks(),
               " ranks has ",
               ctx->executor()->num_streaming_threads(),
               " threads");
  }
  return {ctx, comm};
}

ProgramOptions parse_arguments(int argc, char** argv)
{
  ProgramOptions options;

  auto const comm_names = comm_type_names();

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
              << "  --spill-device-limit <n>     Fractional spill device limit as "
                 "fraction "
                 "of total device memory (default: "
              << (options.spill_device_limit.has_value()
                    ? std::to_string(options.spill_device_limit.value())
                    : "None")
              << ")\n"
              << "  --no-pinned-host-memory      Disable pinned host memory (default: "
              << (options.no_pinned_host_memory ? "true" : "false") << ")\n"
              << "  --periodic-spill <n>         Duration in milliseconds between periodic "
                 "spilling checks (default: "
              << (options.periodic_spill.has_value()
                    ? std::to_string(options.periodic_spill.value().count())
                    : "None")
              << ")\n"
              << "  --comm-type <type>           Communicator type: " << available_comm_types()
              << " "
                 "(default: "
              << comm_names[static_cast<std::size_t>(options.comm_type)] << ")\n"
              << "  --use-shuffle-join           Use shuffle join (default: "
              << (options.use_shuffle_join ? "true" : "false") << ")\n"
              << "  --output-file <path>         Output file path (required)\n"
              << "  --input-directory <path>     Input directory path (required)\n"
              << "  --help                       Show this help message\n";
  };

  // NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)
  static struct option long_options[] = {{"num-streaming-threads", required_argument, nullptr, 1},
                                         {"num-rows-per-chunk", required_argument, nullptr, 2},
                                         {"use-shuffle-join", no_argument, nullptr, 3},
                                         {"output-file", required_argument, nullptr, 4},
                                         {"input-directory", required_argument, nullptr, 5},
                                         {"help", no_argument, nullptr, 6},
                                         {"spill-device-limit", required_argument, nullptr, 7},
                                         {"num-iterations", required_argument, nullptr, 8},
                                         {"num-streams", required_argument, nullptr, 9},
                                         {"comm-type", required_argument, nullptr, 10},
                                         {"periodic-spill", required_argument, nullptr, 11},
                                         {"no-pinned-host-memory", no_argument, nullptr, 12},
                                         {nullptr, 0, nullptr, 0}};
  // NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)

  int opt;
  int option_index = 0;

  bool saw_output_file     = false;
  bool saw_input_directory = false;

  while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
    switch (opt) {
      case 1: {
        char* endptr;
        long val = std::strtol(optarg, &endptr, 10);
        if (*endptr != '\0' || val <= 0) {
          std::cerr << "Error: Invalid value for --num-streaming-threads: " << optarg << "\n\n";
          print_usage();
          std::exit(1);
        }
        options.num_streaming_threads = static_cast<int>(val);
        break;
      }
      case 2: {
        char* endptr;
        long val = std::strtol(optarg, &endptr, 10);
        if (*endptr != '\0' || val <= 0) {
          std::cerr << "Error: Invalid value for --num-rows-per-chunk: " << optarg << "\n\n";
          print_usage();
          std::exit(1);
        }
        options.num_rows_per_chunk = static_cast<int>(val);
        break;
      }
      case 3: options.use_shuffle_join = true; break;
      case 4:
        options.output_file = optarg;
        saw_output_file     = true;
        break;
      case 5:
        options.input_directory = optarg;
        saw_input_directory     = true;
        break;
      case 6: print_usage(); std::exit(0);
      case 7: {
        char* endptr;
        double val = std::strtod(optarg, &endptr);
        if (*endptr != '\0' || val < 0.0 || val > 1.0) {
          std::cerr << "Error: Invalid value for --spill-device-limit: " << optarg
                    << " (must be between 0.0 and 1.0)\n\n";
          print_usage();
          std::exit(1);
        }
        options.spill_device_limit = val;
        break;
      }
      case 8: {
        char* endptr;
        long val = std::strtol(optarg, &endptr, 10);
        if (*endptr != '\0' || val <= 0) {
          std::cerr << "Error: Invalid value for --num-iterations: " << optarg << "\n\n";
          print_usage();
          std::exit(1);
        }
        options.num_iterations = static_cast<int>(val);
        break;
      }
      case 9: {
        char* endptr;
        long val = std::strtol(optarg, &endptr, 10);
        if (*endptr != '\0' || val <= 0) {
          std::cerr << "Error: Invalid value for --num-streams: " << optarg << "\n\n";
          print_usage();
          std::exit(1);
        }
        options.num_streams = static_cast<int>(val);
        break;
      }
      case 10: {
        std::string_view comm_type = optarg;
        auto parsed                = parse_comm_type(comm_type);
        if (!parsed.has_value()) {
          std::cerr << "Error: Invalid value for --comm-type: " << optarg << " (must be one of "
                    << comm_names[0];
          for (std::size_t i = 1; i < comm_names.size(); ++i) {
            std::cerr << ", " << comm_names[i];
          }
          std::cerr << ")\n\n";
          print_usage();
          std::exit(1);
        }
        if (!is_comm_type_available(*parsed)) {
          std::cerr << "Error: communicator '" << comm_type
                    << "' is not available in this build (available: " << available_comm_types()
                    << ")\n\n";
          print_usage();
          std::exit(1);
        }
        options.comm_type = *parsed;
        break;
      }
      case 11: {
        char* endptr;
        long val = std::strtol(optarg, &endptr, 10);
        if (*endptr != '\0' || val <= 0) {
          std::cerr << "Error: Invalid value for --periodic-spill: " << optarg << "\n\n";
          print_usage();
          std::exit(1);
        }
        options.periodic_spill = std::chrono::milliseconds(val);
        break;
      }
      case 12: options.no_pinned_host_memory = true; break;
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
  if (!saw_output_file || !saw_input_directory) {
    if (!saw_output_file) { std::cerr << "Error: --output-file is required\n"; }
    if (!saw_input_directory) { std::cerr << "Error: --input-directory is required\n"; }
    std::cerr << std::endl;
    print_usage();
    std::exit(1);
  }

  return options;
}

FinalizeMPI::~FinalizeMPI() noexcept
{
#ifdef CUDF_STREAMING_HAVE_MPI
  if (rapidsmpf::mpi::is_initialized()) {
    int flag;
    RAPIDSMPF_MPI(MPI_Finalized(&flag));
    if (!flag) { RAPIDSMPF_MPI(MPI_Finalize()); }
  }
#endif
}
}  // namespace rapidsmpf::ndsh
