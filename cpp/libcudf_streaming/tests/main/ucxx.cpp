/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../environment.hpp"

#include <cudf_test/cudf_gtest.hpp>

#include <cuda_runtime_api.h>

#include <mpi.h>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <ucxx/listener.h>

#include <memory>

namespace {
class UCXXEnvironment : public Environment {
 public:
  using Environment::Environment;

  [[nodiscard]] TestEnvironmentType type() const override { return TestEnvironmentType::UCXX; }

  void SetUp() override
  {
    // Ensure CUDA context is created before UCX is initialized.
    RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));

    // Explicitly initialize MPI. We can not use rapidsmpf::mpi::init as it checks some
    // rapidsmpf::MPI communicator specific conditions
    int provided;
    RAPIDSMPF_MPI(MPI_Init_thread(&argc_, &argv_, MPI_THREAD_MULTIPLE, &provided));
    RAPIDSMPF_EXPECTS(provided == MPI_THREAD_MULTIPLE,
                      "didn't get the requested thread level support: MPI_THREAD_MULTIPLE");

    options_ = rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());
    comm_    = rapidsmpf::ucxx::init_using_mpi(MPI_COMM_WORLD,
                                            options_,
                                            std::make_shared<rapidsmpf::ProgressThread>(),
                                            rapidsmpf::Logger::from_options(options_));
  }

  void TearDown() override
  {
    // Ensure UCXX cleanup before MPI. If this is not done failures related to
    // accessing the CUDA context may be thrown during shutdown.
    comm_       = nullptr;  // Clean up the communicator.
    split_comm_ = nullptr;
    RAPIDSMPF_MPI(MPI_Finalize());
  }

  void barrier() override { std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm_)->barrier(); }

  std::shared_ptr<rapidsmpf::Communicator> split_comm() override
  {
    if (split_comm_ == nullptr) {
      split_comm_ = std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm_)->split();
    }
    return split_comm_;
  }

 private:
  std::shared_ptr<rapidsmpf::Communicator> split_comm_{nullptr};
};
}  // namespace

Environment* GlobalEnvironment = nullptr;

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  GlobalEnvironment = new UCXXEnvironment(argc, argv);
  ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
  return RUN_ALL_TESTS();
}
