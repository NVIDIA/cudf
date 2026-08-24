/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils/comm.hpp"
#include "../utils/misc.hpp"
#include "../utils/rmm_utils.hpp"
#include "data_generator.hpp"

#include <cudf_streaming/partition.hpp>
#include <cudf_streaming/partition_utils.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>
#include <unistd.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

class ArgumentParser {
 public:
  ArgumentParser(int argc, char* const* argv, bool use_mpi = true)
  {
    int rank           = 0;
    int nranks         = 1;
    auto abort_or_exit = [&](int code) {
#ifdef CUDF_STREAMING_HAVE_MPI
      if (use_mpi) { RAPIDSMPF_MPI(MPI_Abort(MPI_COMM_WORLD, code)); }
#endif
      std::exit(code);
    };

    if (use_mpi) {
#ifdef CUDF_STREAMING_HAVE_MPI
      RAPIDSMPF_EXPECTS(rapidsmpf::mpi::is_initialized() == true, "MPI is not initialized");

      RAPIDSMPF_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
      RAPIDSMPF_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
#else
      RAPIDSMPF_FAIL("MPI support is not available in this build", std::runtime_error);
#endif
    } else {
      // When not using MPI, expect to be using bootstrap mode (rrun)
#ifdef CUDF_STREAMING_HAVE_UCXX
      nranks = rapidsmpf::bootstrap::get_nranks();
#else
      RAPIDSMPF_FAIL("UCXX bootstrap support is not available in this build", std::runtime_error);
#endif
    }
    try {
      int option;
      while ((option = getopt(argc, argv, "C:r:w:c:n:p:o:m:l:Lxh")) != -1) {
        switch (option) {
          case 'h': {
            std::stringstream ss;
            ss << "Usage: " << argv[0] << " [options]\n"
               << "Options:\n"
               << "  -C <comm>  Communicator {"
               << cudf_streaming::benchmarks::available_communicators()
               << "} (default: " << comm_type << ")\n"
               << "  -r <num>   Number of runs (default: 1)\n"
               << "  -w <num>   Number of warmup runs (default: 0)\n"
               << "  -c <num>   Number of columns in the input tables "
                  "(default: 1)\n"
               << "  -n <num>   Number of rows per rank (default: 1M)\n"
               << "  -p <num>   Number of partitions (input tables) per "
                  "rank (default: 1)\n"
               << "  -o <num>   Number of output partitions per rank "
                  "(default: 1)\n"
               << "  -m <mr>    RMM memory resource {cuda, pool, async, "
                  "managed} (default: pool)\n"
               << "  -l <num>   Device memory limit in MiB (default:-1, "
                  "unlimited)\n"
               << "  -L         Disable Pinned host memory (default: "
                  " unlimited)\n"
               << "  -x         Enable memory profiler (default: disabled)\n"
               << "  -h         Display this help message\n";
            if (rank == 0) { std::cerr << ss.str(); }
            abort_or_exit(0);
          } break;
          case 'C':
            comm_type = std::string{optarg};
            if (!cudf_streaming::benchmarks::is_communicator_available(comm_type)) {
              if (rank == 0) {
                std::cerr << "-C (Communicator) must be one of {"
                          << cudf_streaming::benchmarks::available_communicators() << "}"
                          << std::endl;
              }
              abort_or_exit(-1);
            }
            break;
          case 'r': parse_integer(num_runs, optarg, 1); break;
          case 'w': parse_integer(num_warmups, optarg); break;
          case 'c': parse_integer(num_columns, optarg); break;
          case 'n': parse_integer(num_local_rows, optarg); break;
          case 'p': parse_integer(num_local_partitions, optarg); break;
          case 'o': parse_integer(num_output_partitions, optarg); break;
          case 'm':
            rmm_mr = std::string{optarg};
            if (!(rmm_mr == "cuda" || rmm_mr == "pool" || rmm_mr == "async" ||
                  rmm_mr == "managed")) {
              if (rank == 0) {
                std::cerr << "-m (RMM memory resource) must be one of "
                             "{cuda, pool, async, managed}"
                          << std::endl;
              }
              abort_or_exit(-1);
            }
            break;
          case 'l': parse_integer(device_mem_limit_mb, optarg); break;
          case 'L': pinned_mem_disable = true; break;
          case 'x': enable_memory_profiler = true; break;
          case '?': abort_or_exit(-1); break;
          default: RAPIDSMPF_FAIL("unknown option", std::invalid_argument);
        }
      }
      if (optind < argc) { RAPIDSMPF_FAIL("unknown option", std::invalid_argument); }
    } catch (std::exception const& e) {
      if (rank == 0) { std::cerr << "Error parsing arguments: " << e.what() << std::endl; }
      abort_or_exit(-1);
    }

    local_nbytes = num_columns * num_local_rows * num_local_partitions * sizeof(std::int32_t);
    total_nbytes = local_nbytes * static_cast<std::uint64_t>(nranks);
    if (rmm_mr == "cuda") {
      if (rank == 0) {
        std::cout << "WARNING: using the default cuda memory resource "
                     "(-m cuda) might leak memory! A limitation in UCX "
                     "means that device memory send through IPC can "
                     "never be freed."
                  << std::endl;
      }
    }
  }

  void pprint(rapidsmpf::Communicator& comm) const
  {
    if (comm.rank() > 0) { return; }
    std::stringstream ss;
    ss << "Arguments:\n";
    ss << "  -c " << comm_type << " (communicator)\n";
    ss << "  -r " << num_runs << " (number of runs)\n";
    ss << "  -w " << num_warmups << " (number of warmup runs)\n";
    ss << "  -c " << num_columns << " (number of columns)\n";
    ss << "  -n " << num_local_rows << " (number of rows per rank)\n";
    ss << "  -p " << num_local_partitions << " (number of input partitions per rank)\n";
    ss << "  -o " << num_output_partitions << " (number of output partitions per rank)\n";
    ss << "  -m " << rmm_mr << " (RMM memory resource)\n";
    if (device_mem_limit_mb >= 0) {
      ss << "  -l " << device_mem_limit_mb << " (device memory limit in MiB)\n";
    }
    if (pinned_mem_disable) { ss << "  -L (disable pinned host memory)\n"; }
    if (enable_memory_profiler) { ss << "  -x (enable memory profiling)\n"; }
    ss << "Local size: " << rapidsmpf::format_nbytes(local_nbytes) << "\n";
    ss << "Total size: " << rapidsmpf::format_nbytes(total_nbytes) << "\n";
    comm.logger()->print(ss.str());
  }

  std::uint64_t num_runs{1};
  std::uint64_t num_warmups{0};
  std::uint32_t num_columns{1};
  std::uint64_t num_local_rows{1 << 20};
  rapidsmpf::shuffler::PartID num_local_partitions{1};
  rapidsmpf::shuffler::PartID num_output_partitions{1};
  std::string rmm_mr{"pool"};
  std::string comm_type{cudf_streaming::benchmarks::default_communicator()};
  std::uint64_t local_nbytes;
  std::uint64_t total_nbytes;
  bool enable_memory_profiler{false};
  std::int64_t device_mem_limit_mb{-1};
  bool pinned_mem_disable{false};
};

rapidsmpf::streaming::Actor consumer(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                     std::shared_ptr<rapidsmpf::streaming::Channel> ch_in)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in};
  co_await ctx->executor()->schedule();
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
  }
}

rapidsmpf::Duration run(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                        std::shared_ptr<rapidsmpf::Communicator> comm,
                        ArgumentParser const& args,
                        rmm::cuda_stream_view stream)
{
  constexpr std::int32_t min_val        = 0;
  constexpr std::int32_t max_val        = 10;
  constexpr cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;
  constexpr std::uint32_t seed          = cudf::DEFAULT_HASH_SEED;
  rapidsmpf::shuffler::PartID const total_num_partitions =
    args.num_output_partitions * static_cast<rapidsmpf::shuffler::PartID>(comm->nranks());
  constexpr rapidsmpf::OpID op_id = 0;

  // Create streaming pipeline.
  std::vector<rapidsmpf::streaming::Actor> actors;
  {
    auto ch1                  = ctx->create_channel();
    auto const num_columns    = rapidsmpf::safe_cast<cudf::size_type>(args.num_columns);
    auto const num_local_rows = rapidsmpf::safe_cast<cudf::size_type>(args.num_local_rows);
    actors.push_back(rapidsmpf::streaming::actor::random_table_generator(
      ctx, stream, ch1, args.num_local_partitions, num_columns, num_local_rows, min_val, max_val));
    auto ch2 = ctx->create_channel();
    actors.push_back(cudf_streaming::actor::partition_and_pack(
      ctx, ch1, ch2, {0}, static_cast<int>(total_num_partitions), hash_function, seed));
    auto ch3 = ctx->create_channel();
    actors.push_back(
      rapidsmpf::streaming::actor::shuffler(ctx, comm, ch2, ch3, op_id, total_num_partitions));
    auto ch4 = ctx->create_channel();
    actors.push_back(cudf_streaming::actor::unpack_and_concat(ctx, ch3, ch4));
    actors.push_back(consumer(ctx, ch4));
  }
  auto const t0_elapsed = rapidsmpf::Clock::now();
  rapidsmpf::streaming::run_actor_network(std::move(actors));
  return rapidsmpf::Clock::now() - t0_elapsed;
}

int main(int argc, char** argv)
{
  bool use_bootstrap = rapidsmpf::bootstrap::is_running_with_rrun();

  // Explicitly initialize MPI with thread support, as this is needed for both mpi
  // and ucxx communicators when not using bootstrap mode.
  int provided = 0;
  if (!use_bootstrap) {
#ifdef CUDF_STREAMING_HAVE_MPI
    RAPIDSMPF_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    RAPIDSMPF_EXPECTS(provided == MPI_THREAD_MULTIPLE,
                      "didn't get the requested thread level support: MPI_THREAD_MULTIPLE");
#else
    std::cerr << "Error: this build has no MPI support. Use UCXX bootstrap mode or build with MPI."
              << std::endl;
    return 1;
#endif
  } else {
#ifndef CUDF_STREAMING_HAVE_UCXX
    std::cerr << "Error: this build has no UCXX support. Bootstrap mode is unavailable."
              << std::endl;
    return 1;
#endif
  }
  ArgumentParser args{argc, argv, !use_bootstrap};

  // Initialize configuration options from environment variables.
  rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};
  auto log             = rapidsmpf::Logger::from_options(options);
  auto progress_thread = std::make_shared<rapidsmpf::ProgressThread>();

  std::shared_ptr<rapidsmpf::Communicator> comm;
  if (args.comm_type == "mpi") {
#ifdef CUDF_STREAMING_HAVE_MPI
    if (use_bootstrap) {
      std::cerr << "Error: MPI communicator requires MPI initialization. Don't use with "
                   "rrun or unset RRUN_RANK."
                << std::endl;
      return 1;
    }
    rapidsmpf::mpi::init(&argc, &argv);
    comm = std::make_shared<rapidsmpf::MPI>(MPI_COMM_WORLD, progress_thread, log);
#else
    std::cerr << "Error: MPI communicator is not available in this build." << std::endl;
    return 1;
#endif
  } else if (args.comm_type == "ucxx") {
#ifdef CUDF_STREAMING_HAVE_UCXX
    if (use_bootstrap) {
      // Launched with rrun - use bootstrap backend
      comm = rapidsmpf::bootstrap::create_ucxx_comm(
        progress_thread, rapidsmpf::bootstrap::BackendType::AUTO, options, log);
    } else {
#ifdef CUDF_STREAMING_HAVE_MPI
      // Launched with mpirun - use MPI bootstrap
      comm = rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD, options, progress_thread, log);
#else
      std::cerr << "Error: UCXX without MPI support requires bootstrap mode." << std::endl;
      return 1;
#endif
    }
#else
    std::cerr << "Error: UCXX communicator is not available in this build." << std::endl;
    return 1;
#endif
  } else {
    std::cerr << "Error: Unknown communicator type: " << args.comm_type << std::endl;
    return 1;
  }

  args.pprint(*comm);

  RAPIDSMPF_EXPECTS(comm->nranks() == 1, "only single-rank runs are supported");

  auto rmm_mr = create_rmm_resource(args.rmm_mr);
  std::unordered_map<rapidsmpf::MemoryType, std::int64_t> memory_limits{};
  if (args.device_mem_limit_mb >= 0) {
    memory_limits[rapidsmpf::MemoryType::DEVICE] = args.device_mem_limit_mb << 20;
  }

  auto stats = rapidsmpf::Statistics::create();

  RAPIDSMPF_EXPECTS(args.pinned_mem_disable || rapidsmpf::is_pinned_memory_resources_supported(),
                    "pinned host memory is not supported on this system; pass `-L` to disable it.",
                    std::runtime_error);
  auto pinned_pool_properties =
    args.pinned_mem_disable ? rapidsmpf::PinnedMemoryDisabled : rapidsmpf::PinnedPoolProperties{};
  auto br = rapidsmpf::BufferResource::create(
    rmm_mr,
    std::move(pinned_pool_properties),
    std::move(memory_limits),
    std::nullopt,
    std::make_shared<rmm::cuda_stream_pool>(16, rmm::cuda_stream::flags::non_blocking),
    stats);
  // `BufferResource` wraps the device resource in an internal tracking
  // `RmmResourceAdaptor` (exposed via `device_mr_adaptor()`). Install it as
  // the current device resource so libcudf temp allocations are also tracked.
  auto& stat_enabled_mr = br->device_mr_adaptor();
  rmm::mr::set_current_device_resource(stat_enabled_mr);

  rmm::cuda_stream_view stream = cudf::get_default_stream();

  // Print benchmark/hardware info.
  {
    std::stringstream ss;
    auto const cur_dev = rmm::get_current_cuda_device().value();
    std::string pci_bus_id(16, '\0');  // Preallocate space for the PCI bus ID
    RAPIDSMPF_CUDA_TRY(cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), cur_dev));
    cudaDeviceProp properties;
    RAPIDSMPF_CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
    ss << "Hardware setup: \n";
    ss << "  GPU (" << properties.name << "): \n";
    ss << "    Device number: " << cur_dev << "\n";
    ss << "    PCI Bus ID: " << pci_bus_id.substr(0, pci_bus_id.find('\0')) << "\n";
    ss << "    Total Memory: " << rapidsmpf::format_nbytes(properties.totalGlobalMem, 0) << "\n";
    ss << "  Comm: " << *comm << "\n";
    log->print(ss.str());
  }

  auto ctx = std::make_shared<rapidsmpf::streaming::Context>(options, comm->logger(), br);

  std::vector<double> elapsed_vec;
  std::uint64_t const total_num_runs = args.num_warmups + args.num_runs;
  for (std::uint64_t i = 0; i < total_num_runs; ++i) {
    // Clear statistics before the last run so only the final run is reported.
    if (i == total_num_runs - 1) { ctx->statistics()->clear(); }
    double const elapsed = run(ctx, comm, args, stream).count();
    std::stringstream ss;
    ss << "elapsed: " << rapidsmpf::format_duration(elapsed)
       << " | local throughput: " << rapidsmpf::format_nbytes(args.local_nbytes / elapsed)
       << "/s | global throughput: " << rapidsmpf::format_nbytes(args.total_nbytes / elapsed)
       << "/s";
    if (i < args.num_warmups) { ss << " (warmup run)"; }
    log->print(ss.str());
    if (i >= args.num_warmups) { elapsed_vec.push_back(elapsed); }
  }

  if (!use_bootstrap) {
#ifdef CUDF_STREAMING_HAVE_MPI
    RAPIDSMPF_MPI(MPI_Barrier(MPI_COMM_WORLD));
#else
    RAPIDSMPF_FAIL("MPI barrier requested, but MPI support is not available in this build",
                   std::runtime_error);
#endif
  } else {
#ifdef CUDF_STREAMING_HAVE_UCXX
    auto ucxx = std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm);
    RAPIDSMPF_EXPECTS(
      ucxx != nullptr, "Expected UCXX communicator when using bootstrap mode", std::runtime_error);
    ucxx->barrier();
#else
    RAPIDSMPF_FAIL("UCXX bootstrap barrier requested, but UCXX support is not available",
                   std::runtime_error);
#endif
  }

  {
    auto const elapsed_mean = harmonic_mean(elapsed_vec);
    std::stringstream ss;
    ss << "means: " << rapidsmpf::format_duration(elapsed_mean)
       << " | local throughput: " << rapidsmpf::format_nbytes(args.local_nbytes / elapsed_mean)
       << "/s | global throughput: " << rapidsmpf::format_nbytes(args.total_nbytes / elapsed_mean)
       << "/s"
       << " | in_parts: " << args.num_local_partitions
       << " | out_parts: " << args.num_output_partitions << " | nranks: " << comm->nranks();
    if (args.enable_memory_profiler) {
      auto record = stat_enabled_mr.get_main_record();
      ss << " | device memory peak: " << rapidsmpf::format_nbytes(record.peak())
         << " | device memory total: "
         << rapidsmpf::format_nbytes(record.total() / static_cast<std::int64_t>(total_num_runs))
         << " (avg)";
    }
    log->print(ss.str());
  }

  auto statistics = ctx->statistics();
  if (args.enable_memory_profiler) {
    log->print(statistics->report({
      .mr        = stat_enabled_mr,
      .pinned_mr = br->try_pinned_mr(),
      .header    = "Statistics (of the last run):",
    }));
  } else {
    log->print(statistics->report({.header = "Statistics (of the last run):"}));
  }

#ifdef CUDF_STREAMING_HAVE_MPI
  if (!use_bootstrap) { RAPIDSMPF_MPI(MPI_Finalize()); }
#endif
  return 0;
}
