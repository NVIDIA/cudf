/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/ndsh/fixture_cache.hpp>

#include <cudf_test/cudf_gtest.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace {

// Keep this CPU-only and use nonthrowing cleanup, including during constructor unwinding.
class temporary_directory {
 public:
  temporary_directory()
  {
    auto pattern = (std::filesystem::temp_directory_path() / "ndsh-fixture-cache-XXXXXX").string();
    auto const created = ::mkdtemp(pattern.data());
    if (created == nullptr) { throw std::runtime_error("mkdtemp failed"); }
    path = created;
  }

  ~temporary_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }

  temporary_directory(temporary_directory const&)            = delete;
  temporary_directory& operator=(temporary_directory const&) = delete;

  std::filesystem::path path;
};

// Each test uses a distinct type so its function-static cache is independent.
template <int Id>
struct fake_files {
  static inline int attempts               = 0;
  static inline int constructions          = 0;
  static inline int destructions           = 0;
  static inline bool fail_next             = false;
  static inline bool old_directory_present = false;
  static inline std::filesystem::path previous_directory;

  static void prepare_test()
  {
    fail_next = false;
    // Replace any cached fixture from a previous gtest_repeat iteration before counting operations.
    ndsh::local_fixture<fake_files>(0.0);
    ndsh::local_fixture<fake_files>(1.0);
    attempts = constructions = destructions = 0;
    old_directory_present                   = false;
  }

  static temporary_directory make_directory()
  {
    ++attempts;
    old_directory_present = std::filesystem::exists(previous_directory);
    return temporary_directory{};
  }

  temporary_directory directory{make_directory()};

  explicit fake_files(double)
  {
    previous_directory = directory.path;
    std::ofstream marker_file(marker());
    marker_file << "fixture";
    marker_file.close();
    if (!marker_file) { throw std::runtime_error("Unable to write fixture marker"); }
    if (fail_next) {
      fail_next = false;
      throw std::runtime_error("Fixture construction failed");
    }
    ++constructions;
  }

  ~fake_files() { ++destructions; }

  std::filesystem::path marker() const { return directory.path / "marker"; }
};

TEST(FixtureCacheTest, SameScaleReusesFiles)
{
  using files = fake_files<0>;
  files::prepare_test();
  auto const& first  = ndsh::local_fixture<files>(1.0);
  auto const marker  = first.marker();
  auto const& second = ndsh::local_fixture<files>(1.0);

  EXPECT_EQ(&first, &second);
  EXPECT_EQ(second.marker(), marker);
  EXPECT_TRUE(std::filesystem::exists(marker));
  EXPECT_EQ(files::attempts, 0);
  EXPECT_EQ(files::constructions, 0);
  EXPECT_EQ(files::destructions, 0);
}

TEST(FixtureCacheTest, DeletesOldFilesBeforeReplacementConstruction)
{
  using files = fake_files<1>;
  files::prepare_test();
  auto const& first        = ndsh::local_fixture<files>(1.0);
  auto const old_directory = first.directory.path;
  auto const old_marker    = first.marker();
  ASSERT_TRUE(std::filesystem::exists(old_marker));

  auto const& replacement = ndsh::local_fixture<files>(2.0);

  EXPECT_FALSE(files::old_directory_present);
  EXPECT_FALSE(std::filesystem::exists(old_directory));
  EXPECT_FALSE(std::filesystem::exists(old_marker));
  EXPECT_TRUE(std::filesystem::exists(replacement.marker()));
  EXPECT_EQ(files::attempts, 1);
  EXPECT_EQ(files::constructions, 1);
  EXPECT_EQ(files::destructions, 1);
}

TEST(FixtureCacheTest, RevisitedScaleRegeneratesFiles)
{
  using files = fake_files<2>;
  files::prepare_test();
  auto const first_directory  = ndsh::local_fixture<files>(1.0).directory.path;
  auto const second_directory = ndsh::local_fixture<files>(2.0).directory.path;
  auto const& revisited       = ndsh::local_fixture<files>(1.0);

  EXPECT_FALSE(files::old_directory_present);
  EXPECT_FALSE(std::filesystem::exists(first_directory));
  EXPECT_FALSE(std::filesystem::exists(second_directory));
  EXPECT_TRUE(std::filesystem::exists(revisited.marker()));
  EXPECT_EQ(files::attempts, 2);
  EXPECT_EQ(files::constructions, 2);
  EXPECT_EQ(files::destructions, 2);
}

TEST(FixtureCacheTest, FailedConstructionCleansPartialFilesAndAllowsRetry)
{
  using files = fake_files<3>;
  files::prepare_test();
  auto const old_directory = ndsh::local_fixture<files>(1.0).directory.path;
  files::fail_next         = true;

  EXPECT_THROW(ndsh::local_fixture<files>(2.0), std::runtime_error);
  auto const partial_directory = files::previous_directory;
  EXPECT_FALSE(files::old_directory_present);
  EXPECT_FALSE(std::filesystem::exists(old_directory));
  EXPECT_FALSE(std::filesystem::exists(partial_directory));
  EXPECT_EQ(files::attempts, 1);
  EXPECT_EQ(files::constructions, 0);
  EXPECT_EQ(files::destructions, 1);

  auto const& retried = ndsh::local_fixture<files>(2.0);

  EXPECT_FALSE(files::old_directory_present);
  EXPECT_TRUE(std::filesystem::exists(retried.marker()));
  EXPECT_FALSE(std::filesystem::exists(partial_directory));
  EXPECT_EQ(files::attempts, 2);
  EXPECT_EQ(files::constructions, 1);
  EXPECT_EQ(files::destructions, 1);
  EXPECT_EQ(&ndsh::local_fixture<files>(2.0), &retried);
  EXPECT_EQ(files::attempts, 2);
}

}  // namespace
