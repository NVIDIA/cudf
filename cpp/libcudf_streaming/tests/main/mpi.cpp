/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../environment.hpp"

#include <cudf_test/cudf_gtest.hpp>

#include <mpi.h>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/progress_thread.hpp>

#include <memory>

namespace {
class MPIEnvironment : public Environment {
 public:
  using Environment::Environment;

  [[nodiscard]] TestEnvironmentType type() const override { return TestEnvironmentType::MPI; }

  void SetUp() override
  {
    rapidsmpf::mpi::init(&argc_, &argv_);

    RAPIDSMPF_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_));

    options_ = rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());

    comm_ = std::make_shared<rapidsmpf::MPI>(mpi_comm_,
                                             std::make_shared<rapidsmpf::ProgressThread>(),
                                             rapidsmpf::Logger::from_options(options_));
  }

  void TearDown() override
  {
    comm_ = nullptr;  // Clean up the communicator.

    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm_));
    RAPIDSMPF_MPI(MPI_Finalize());
  }

  void barrier() override { RAPIDSMPF_MPI(MPI_Barrier(mpi_comm_)); }

  std::shared_ptr<rapidsmpf::Communicator> split_comm() override
  {
    // Create the new split communicator
    int rank;
    RAPIDSMPF_MPI(MPI_Comm_rank(mpi_comm_, &rank));
    MPI_Comm split_comm = MPI_COMM_NULL;
    RAPIDSMPF_MPI(MPI_Comm_split(mpi_comm_, rank, 0, &split_comm));
    auto new_logger = rapidsmpf::Logger::from_options(options_);
    new_logger->set_name(std::to_string(rank));
    return std::shared_ptr<rapidsmpf::MPI>(
      new rapidsmpf::MPI(split_comm, comm_->progress_thread(), std::move(new_logger)),
      // Don't leak the split handle.
      [comm = split_comm](rapidsmpf::MPI* x) mutable {
        delete x;
        MPI_Comm_free(&comm);
      });
  }

 private:
  MPI_Comm mpi_comm_{MPI_COMM_NULL};
};
}  // namespace

Environment* GlobalEnvironment = nullptr;

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  GlobalEnvironment = new MPIEnvironment(argc, argv);
  ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
  return RUN_ALL_TESTS();
}
