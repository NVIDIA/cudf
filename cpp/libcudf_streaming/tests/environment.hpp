/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/cudf_gtest.hpp>

#include <rapidsmpf/communicator/communicator.hpp>

#include <memory>

enum class TestEnvironmentType : int {
  MPI,
  UCXX,
  SINGLE,
};

/**
 * Base test environment that exposes a communicator to the test suite. Concrete,
 * communicator-specific environments (single/MPI/UCXX) derive from this class and are defined in
 * the corresponding file in main dir, where the communicator-specific state (e.g. the MPI
 * communicator) and headers live. Exactly one of them is linked into each test executable.
 */
class Environment : public ::testing::Environment {
 public:
  Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}
  ~Environment() override = default;

  void SetUp() override = 0;

  void TearDown() override = 0;

  virtual void barrier() = 0;

  [[nodiscard]] virtual TestEnvironmentType type() const = 0;

  rapidsmpf::config::Options& options() { return options_; }

  virtual std::shared_ptr<rapidsmpf::Communicator> split_comm() = 0;

  std::shared_ptr<rapidsmpf::Communicator> comm_;

 protected:
  int argc_;
  char** argv_;
  rapidsmpf::config::Options options_;
};

extern Environment* GlobalEnvironment;
