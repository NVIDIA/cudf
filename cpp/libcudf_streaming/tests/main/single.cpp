/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../environment.hpp"

#include <cudf_test/cudf_gtest.hpp>

#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/progress_thread.hpp>

#include <memory>

namespace {
class SingleEnvironment : public Environment {
 public:
  using Environment::Environment;

  [[nodiscard]] TestEnvironmentType type() const override { return TestEnvironmentType::SINGLE; }

  void SetUp() override
  {
    options_ = rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());
    comm_    = std::make_shared<rapidsmpf::Single>(std::make_shared<rapidsmpf::ProgressThread>(),
                                                rapidsmpf::Logger::from_options(options_));
  }

  void TearDown() override { comm_ = nullptr; }

  void barrier() override {}

  std::shared_ptr<rapidsmpf::Communicator> split_comm() override { return comm_; }
};
}  // namespace

Environment* GlobalEnvironment = nullptr;

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  GlobalEnvironment = new SingleEnvironment(argc, argv);
  ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
  return RUN_ALL_TESTS();
}
