/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_streaming/partition_utils.hpp>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>
#include <unistd.h>

#include <functional>
#include <string>
#include <vector>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>
#endif

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

#include "utils/comm.hpp"
#include "utils/misc.hpp"
#include "utils/random_data.hpp"
#include "utils/rmm_utils.hpp"

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
      while ((option = getopt(argc, argv, "C:r:w:c:n:p:o:m:l:LigsbxhM:")) != -1) {
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
               << "  -g         Use pre-partitioned (hash) input tables "
                  "(default: unset, hash partition during insertion)\n"
               << "  -s         Discard output chunks to simulate streaming "
                  "(default: disabled)\n"
               << "  -b         Disallow memory overbooking when generating "
                  "input data (default: allow memory overbooking)\n"
               << "  -x         Enable memory profiler (default: disabled)\n"
#ifdef RAPIDSMPF_HAVE_CUPTI
               << "  -M <path>  Enable CUPTI memory monitoring and save CSV "
                  "files with given path prefix. For example, /tmp/test will "
                  "write files to /tmp/test_<rank>.csv (default: disabled)\n"
#endif
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
          case 'r': parse_integer(num_runs, optarg); break;
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
          case 'g': hash_partition_with_datagen = true; break;
          case 's': enable_output_discard = true; break;
          case 'b': input_data_allow_overbooking = rapidsmpf::AllowOverbooking::NO; break;
          case 'x': enable_memory_profiler = true; break;
#ifdef RAPIDSMPF_HAVE_CUPTI
          case 'M':
            cupti_csv_prefix        = std::string{optarg};
            enable_cupti_monitoring = true;
            break;
#endif
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
    if (enable_output_discard) { ss << "  -s (enable output discard to simulate streaming)\n"; }
    if (input_data_allow_overbooking == rapidsmpf::AllowOverbooking::NO) {
      ss << "  -b (disallow memory overbooking when generating input data)\n";
    }
    if (enable_memory_profiler) { ss << "  -x (enable memory profiling)\n"; }
    if (hash_partition_with_datagen) { ss << "  -g (use pre-partitioned input tables)\n"; }
    if (enable_cupti_monitoring) {
      ss << "  -M " << cupti_csv_prefix << " (CUPTI memory monitoring enabled)\n";
    }
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
  bool enable_output_discard{false};
  rapidsmpf::AllowOverbooking input_data_allow_overbooking{rapidsmpf::AllowOverbooking::YES};
  bool enable_memory_profiler{false};
  bool hash_partition_with_datagen{false};
  std::int64_t device_mem_limit_mb{-1};
  bool pinned_mem_disable{false};
  bool enable_cupti_monitoring{false};
  std::string cupti_csv_prefix;
};

void barrier(std::shared_ptr<rapidsmpf::Communicator>& comm)
{
  bool use_bootstrap = rapidsmpf::bootstrap::is_running_with_rrun();
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
}

rapidsmpf::Duration do_run(rapidsmpf::shuffler::PartID const total_num_partitions,
                           std::shared_ptr<rapidsmpf::Communicator>& comm,
                           ArgumentParser const& args,
                           rmm::cuda_stream_view stream,
                           rapidsmpf::BufferResource* br,
                           std::shared_ptr<rapidsmpf::Statistics> statistics,
                           auto&& shuffle_insert_fn)
{
  std::vector<std::unique_ptr<cudf::table>> output_partitions;
  output_partitions.reserve(total_num_partitions);

  barrier(comm);

  auto const t0_elapsed = rapidsmpf::Clock::now();
  {
    RAPIDSMPF_NVTX_SCOPED_RANGE("Shuffling", total_num_partitions);
    if (args.enable_memory_profiler) {
      RAPIDSMPF_MEMORY_PROFILE(statistics, br->device_mr(), "shuffling");
    }
    rapidsmpf::shuffler::Shuffler shuffler(comm,
                                           0,  // op_id
                                           total_num_partitions,
                                           br,
                                           rapidsmpf::shuffler::Shuffler::round_robin);

    // insert partitions into the shuffler
    shuffle_insert_fn(shuffler);

    shuffler.wait();
    for (auto finished_partition : shuffler.local_partitions()) {
      auto packed_chunks    = shuffler.extract(finished_partition);
      auto output_partition = cudf_streaming::unpack_and_concat(
        rapidsmpf::unspill_partitions(
          std::move(packed_chunks), br, rapidsmpf::AllowOverbooking::YES),
        stream,
        br);
      if (!args.enable_output_discard) {
        output_partitions.emplace_back(std::move(output_partition));
      }
    }
    stream.synchronize();
  }

  auto const elapsed = rapidsmpf::Clock::now() - t0_elapsed;

  // Check the shuffle result (this test only works for non-empty partitions
  // thus we only check large shuffles).
  if (args.num_local_rows >= 1000000) {
    for (auto const& output_partition : output_partitions) {
      auto [parts, owner] =
        cudf_streaming::partition_and_split(output_partition->view(),
                                            {0},
                                            static_cast<std::int32_t>(total_num_partitions),
                                            cudf::hash_id::HASH_MURMUR3,
                                            cudf::DEFAULT_HASH_SEED,
                                            stream,
                                            br);
      RAPIDSMPF_EXPECTS(
        std::count_if(
          parts.begin(), parts.end(), [](auto const& table) { return table.num_rows() > 0; }) == 1,
        "all rows in an output partition should hash to the same");
    }
  }

  barrier(comm);

  return elapsed;
}

// generate input partitions by applying a transform function to each table
template <typename TransformFn,
          typename InputPartitionsT =
            std::remove_reference_t<std::invoke_result_t<TransformFn, cudf::table&&>>>
std::vector<InputPartitionsT> generate_input_partitions(ArgumentParser const& args,
                                                        rmm::cuda_stream_view stream,
                                                        rapidsmpf::BufferResource* br,
                                                        TransformFn&& transform_fn)
{
  auto const num_columns     = rapidsmpf::safe_cast<cudf::size_type>(args.num_columns);
  auto const num_local_rows  = rapidsmpf::safe_cast<cudf::size_type>(args.num_local_rows);
  std::int32_t const min_val = 0;
  std::int32_t const max_val = num_local_rows;

  std::vector<InputPartitionsT> input_partitions;
  input_partitions.reserve(args.num_local_partitions);
  for (rapidsmpf::shuffler::PartID i = 0; i < args.num_local_partitions; ++i) {
    std::size_t size_lb = random_table_size_lower_bound(num_columns, num_local_rows);

    // reserve at least size_lb and spill if necessary.
    auto res = br->reserve_device_memory_and_spill(size_lb, args.input_data_allow_overbooking);
    cudf::table table =
      random_table(num_columns, num_local_rows, min_val, max_val, stream, br->device_mr());
    input_partitions.emplace_back(transform_fn(std::move(table)));
  }
  stream.synchronize();
  return input_partitions;
}

/**
 * Helper function to iterate over input partitions and insert them into the shuffler.
 *
 * @param shuffler Shuffler to insert the partitions into.
 * @param input_partitions This is either a vector<cudf::table> or
 * vector<unordered_map<PartID, PackedData>>. Former will be forwarded to to
 * partition_and_pack to generate a unordered_map<PartID, PackedData> for each table.
 * @param make_chunk_fn Function to make a chunk from a partition.
 */
void do_insert(rapidsmpf::shuffler::Shuffler& shuffler,
               auto&& input_partitions,
               auto&& make_chunk_fn)
{
  // Convert a partition into chunks and insert into the shuffler.
  for (auto&& partition : input_partitions) {
    shuffler.insert(std::move(make_chunk_fn(partition)));
  }

  // Tell the shuffler that we have no more data.
  shuffler.insert_finished();
}

/**
 * @brief Runs shuffle by partitioning the input tables and inserting them into the
 * shuffler.
 *
 * This function generates random input tables and partitions them into the number of
 * input partitions specified by the user. It then inserts the partitions into the
 * shuffler and runs the shuffle. Each input partition will be partitioned into
 * `num_output_partitions * nranks`, resulting in, `num_local_partitions *
 * num_output_partitions * nranks` chunks being inserted into the shuffler. Each chunk
 * size will be `~num_local_rows/(num_output_partitions * nranks)` rows.
 *
 * @param comm Communicator for the shuffler
 * @param args Command line arguments
 * @param stream CUDA stream for the shuffler
 * @param br Buffer resource for the shuffler
 * @param statistics Statistics for the shuffler
 * @return Duration of the run
 */
rapidsmpf::Duration run_hash_partition_inline(std::shared_ptr<rapidsmpf::Communicator>& comm,
                                              ArgumentParser const& args,
                                              rmm::cuda_stream_view stream,
                                              rapidsmpf::BufferResource* br,
                                              std::shared_ptr<rapidsmpf::Statistics> statistics)
{
  rapidsmpf::shuffler::PartID const total_num_partitions =
    args.num_output_partitions * static_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

  std::vector<cudf::table> input_partitions =
    generate_input_partitions(args, stream, br, std::identity{});

  auto make_chunk_fn = [&](cudf::table const& partition) {
    return cudf_streaming::partition_and_pack(partition,
                                              {0},
                                              static_cast<std::int32_t>(total_num_partitions),
                                              cudf::hash_id::HASH_MURMUR3,
                                              cudf::DEFAULT_HASH_SEED,
                                              stream,
                                              br);
  };

  return do_run(total_num_partitions, comm, args, stream, br, statistics, [&](auto& shuffler) {
    do_insert(shuffler, std::move(input_partitions), std::move(make_chunk_fn));
  });
}

/**
 * @brief Runs shuffle by using pre-partitioned input tables.
 *
 * This is similar to the hash partitioning, but the input tables are already
 * partitioned before being inserted into the shuffler.
 *
 * @param comm Communicator for the shuffler
 * @param args Command line arguments
 * @param stream CUDA stream for the shuffler
 * @param br Buffer resource for the shuffler
 * @param statistics Statistics for the shuffler
 * @return Duration of the run
 */
rapidsmpf::Duration run_hash_partition_with_datagen(
  std::shared_ptr<rapidsmpf::Communicator>& comm,
  ArgumentParser const& args,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  std::shared_ptr<rapidsmpf::Statistics> statistics)
{
  rapidsmpf::shuffler::PartID const total_num_partitions =
    args.num_output_partitions * static_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

  std::vector<std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>>
    input_partitions = generate_input_partitions(args, stream, br, [&](cudf::table&& table) {
      return cudf_streaming::partition_and_pack(table,
                                                {0},
                                                static_cast<std::int32_t>(total_num_partitions),
                                                cudf::hash_id::HASH_MURMUR3,
                                                cudf::DEFAULT_HASH_SEED,
                                                stream,
                                                br);
    });

  return do_run(total_num_partitions, comm, args, stream, br, statistics, [&](auto& shuffler) {
    do_insert(shuffler, std::move(input_partitions), std::identity{});
  });
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

  auto rmm_mr = create_rmm_resource(args.rmm_mr);

  std::unordered_map<rapidsmpf::MemoryType, std::int64_t> memory_limits{};
  if (args.device_mem_limit_mb >= 0) {
    memory_limits[rapidsmpf::MemoryType::DEVICE] = args.device_mem_limit_mb << 20;
  }

  auto stats = rapidsmpf::Statistics::create();

  // We're only going to measure the last run, so disable initially.
  stats->disable();
  RAPIDSMPF_EXPECTS(args.pinned_mem_disable || rapidsmpf::is_pinned_memory_resources_supported(),
                    "pinned host memory is not supported on this system; pass `-L` to disable it.",
                    std::runtime_error);
  auto pinned_pool_properties =
    args.pinned_mem_disable ? rapidsmpf::PinnedMemoryDisabled : rapidsmpf::PinnedPoolProperties{};
  auto br = rapidsmpf::BufferResource::create(
    rmm_mr,
    std::move(pinned_pool_properties),
    std::move(memory_limits),
    std::chrono::milliseconds{1},
    std::make_shared<rmm::cuda_stream_pool>(16, rmm::cuda_stream::flags::non_blocking),
    stats);
  // `BufferResource` wraps the device resource in an internal tracking
  // `RmmResourceAdaptor` (exposed via `device_mr_adaptor()`). Install the
  // tracking adaptor as the current device resource so libcudf temporary
  // allocations are also tracked.
  auto& stat_enabled_mr = br->device_mr_adaptor();
  rmm::mr::set_current_device_resource(stat_enabled_mr);

  auto log = rapidsmpf::Logger::from_options(options);

  std::shared_ptr<rapidsmpf::Communicator> comm;
  auto progress_thread = std::make_shared<rapidsmpf::ProgressThread>(stats);
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

#ifdef RAPIDSMPF_HAVE_CUPTI
  // Create CUPTI monitor if enabled
  std::unique_ptr<rapidsmpf::CuptiMonitor> cupti_monitor;
  if (args.enable_cupti_monitoring) {
    cupti_monitor = std::make_unique<rapidsmpf::CuptiMonitor>();
    cupti_monitor->start_monitoring();
    log->print("CUPTI memory monitoring enabled");
  }
#endif

  std::vector<double> elapsed_vec;
  std::uint64_t const total_num_runs = args.num_warmups + args.num_runs;
  for (std::uint64_t i = 0; i < total_num_runs; ++i) {
    // Enable statistics before the last run so only last-run data is reported.
    if (i == total_num_runs - 1) { stats->enable(); }
    double elapsed;
    if (args.hash_partition_with_datagen) {
      elapsed = run_hash_partition_with_datagen(comm, args, stream, br.get(), stats).count();
    } else {
      elapsed = run_hash_partition_inline(comm, args, stream, br.get(), stats).count();
    }
    std::stringstream ss;
    ss << "elapsed: " << rapidsmpf::format_duration(elapsed)
       << " | local throughput: " << rapidsmpf::format_nbytes(args.local_nbytes / elapsed)
       << "/s | global throughput: " << rapidsmpf::format_nbytes(args.total_nbytes / elapsed)
       << "/s";
    if (i < args.num_warmups) { ss << " (warmup run)"; }
    log->print(ss.str());
    if (i >= args.num_warmups) { elapsed_vec.push_back(elapsed); }
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

  if (args.enable_memory_profiler) {
    log->print(stats->report({.mr = stat_enabled_mr, .header = "Statistics (of the last run):"}));
  } else {
    log->print(stats->report({.header = "Statistics (of the last run):"}));
  }

#ifdef RAPIDSMPF_HAVE_CUPTI
  // Save CUPTI monitoring results to CSV file
  if (args.enable_cupti_monitoring && cupti_monitor) {
    cupti_monitor->stop_monitoring();

    std::string csv_filename = args.cupti_csv_prefix + std::to_string(comm->rank()) + ".csv";
    try {
      cupti_monitor->write_csv(csv_filename);
      log->print("CUPTI memory data written to " + csv_filename + " (" +
                 std::to_string(cupti_monitor->get_sample_count()) + " samples, " +
                 std::to_string(cupti_monitor->get_total_callback_count()) + " callbacks)");

      // Print callback summary for rank 0
      if (comm->rank() == 0) {
        log->print("CUPTI Callback Summary:\n" + cupti_monitor->get_callback_summary());
      }
    } catch (std::exception const& e) {
      log->print("Failed to write CUPTI CSV file: " + std::string(e.what()));
    }
  }
#endif

#ifdef CUDF_STREAMING_HAVE_MPI
  if (!use_bootstrap) { RAPIDSMPF_MPI(MPI_Finalize()); }
#endif
  return 0;
}
