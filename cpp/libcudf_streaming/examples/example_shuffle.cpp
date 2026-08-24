/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../benchmarks/utils/random_data.hpp"

#include <cudf_streaming/partition_utils.hpp>

#include <mpi.h>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>
#include <unistd.h>

#include <vector>

// An example of how to use the shuffler.
int main(int argc, char** argv)
{
  // In this example we use the MPI backed. For convenience, rapidsmpf provides an
  // optional MPI-init function that initialize MPI with thread support.
  rapidsmpf::mpi::init(&argc, &argv);

  // Initialize configuration options from environment variables.
  rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

  // Create a statistics instance for the shuffler that tracks useful information.
  auto stats = rapidsmpf::Statistics::create();

  // The communicator has a progress thread where the shuffler event loop executes. A
  // single progress thread may be used by multiple shufflers simultaneously.
  auto progress_thread = std::make_shared<rapidsmpf::ProgressThread>(stats);

  auto log = rapidsmpf::Logger::from_options(options);

  // Now we have to create a Communicator, which we will use throughout the
  // example. Multiple concurrent shuffles are possible on the same communicator by
  // providing differentiating "OpID" arguments.
  std::shared_ptr<rapidsmpf::Communicator> comm =
    std::make_shared<rapidsmpf::MPI>(MPI_COMM_WORLD, progress_thread, log);

  // We will use the same stream, memory, and buffer resource throughout the example.
  rmm::cuda_stream_view stream      = cudf::get_default_stream();
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
  auto br                           = rapidsmpf::BufferResource::create(mr);

  // As input data, we use a helper function from the benchmark suite. It creates a
  // random cudf table with 2 columns and 100 rows. In this example, each MPI rank
  // creates its own local input and we only have one input per rank but each rank
  // could take any number of inputs.
  cudf::table local_input = random_table(2, 100, 0, 10, stream, mr);

  // The total number of inputs equals the number of ranks, in this case.
  auto const total_num_partitions = static_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

  // We create a new shuffler instance, which represents a single shuffle. It takes
  // a Communicator, the total number of partitions, and a "owner function", which
  // map partitions to their destination ranks. All ranks must use the same owner
  // function, in this example we use the included round-robin owner function.
  rapidsmpf::shuffler::Shuffler shuffler(
    comm,
    0,  // op_id
    total_num_partitions,
    br.get(),
    rapidsmpf::shuffler::Shuffler::round_robin  // partition owner
  );

  // It is our own responsibility to partition and pack (serialize) the input for
  // the shuffle. The shuffler only handles raw host and device buffers. However, it
  // does provide a convenience function that hash partitions a cudf table and packs
  // each partition. The result is a mapping of `PartID`, globally unique partition
  // identifiers, to their packed partitions.
  std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> packed_inputs =
    cudf_streaming::partition_and_pack(local_input,
                                       {0},  // columns_to_hash
                                       static_cast<int>(total_num_partitions),
                                       cudf::hash_id::HASH_MURMUR3,
                                       cudf::DEFAULT_HASH_SEED,
                                       stream,
                                       br.get());

  // Now, we can insert the packed partitions into the shuffler. This operation is
  // non-blocking and we can continue inserting new input partitions. E.g., a pipeline
  // could read, hash-partition, pack, and insert, one parquet-file at a time while the
  // distributed shuffle is being processed underneath.
  shuffler.insert(std::move(packed_inputs));

  // When we are finished inserting data, we tell the shuffler. This sends one control
  // message per target rank, informing each that this rank has finished inserting data.
  shuffler.insert_finished();

  // Vector to hold the local results of the shuffle operation.
  std::vector<std::unique_ptr<cudf::table>> local_outputs;

  // Wait for all partitions to finish.
  shuffler.wait();

  // Process the shuffle results for each partition.
  for (auto finished_partition : shuffler.local_partitions()) {
    // Extract the finished partition's data from the Shuffler.
    auto packed_chunks = shuffler.extract(finished_partition);

    // Unpack (deserialize) and concatenate the chunks into a single table using a
    // convenience function.
    local_outputs.push_back(cudf_streaming::unpack_and_concat(
      rapidsmpf::unspill_partitions(
        std::move(packed_chunks), br.get(), rapidsmpf::AllowOverbooking::YES),
      stream,
      br.get()));
  }
  // At this point, `local_outputs` contains the local result of the shuffle.
  // Let's log the result.
  log->print("Finished shuffle with ", local_outputs.size(), " local output partitions");

  // Log the statistics report.
  log->print(stats->report());

  // Shutdown the Shuffler explicitly or let it go out of scope for cleanup.
  shuffler.shutdown();

  // Finalize the execution, `RAPIDSMPF_MPI` is a convenience macro that
  // checks for MPI errors.
  RAPIDSMPF_MPI(MPI_Finalize());
}
