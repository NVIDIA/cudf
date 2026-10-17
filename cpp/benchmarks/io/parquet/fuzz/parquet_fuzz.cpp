/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Fuzz test of the Parquet writer and readers.
 *
 * Each test case is a seed. It writes a random table to Parquet files with random writer settings,
 * reads the files back with each Parquet reader and random read settings, and checks that every
 * read returns its rows of the table. The values that the cases draw from are in `fuzz_recipe`.
 *
 * A failed case does not stop the other cases, unless it leaves a CUDA error behind, which stops
 * the remaining cases of the process (see `device_failed` for how to test them).
 */

#include "checked_read.hpp"
#include "data_generation.hpp"
#include "tests/io/parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/table/table.hpp>
#include <cudf/version_config.hpp>

#include <cuda_runtime_api.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <ostream>
#include <random>
#include <string>
#include <vector>

namespace {

/**
 * @brief Values that the reads of a case draw their settings from
 */
struct read_recipe {
  int whole_reads           = 1;
  int chunked_reads         = 3;
  int hybrid_scan_reads     = 2;
  int bounded_reads         = 1;  ///< Whole reads of random row ranges
  int bounded_chunked_reads = 2;  ///< Chunked reads of random row ranges
  /// Read limits of the chunked and hybrid scan reads, where 0 means no limit
  std::vector<std::size_t> chunk_read_limits{0, 1, 1'000, 64'000, 1'000'000, 16'000'000};
  std::vector<std::size_t> pass_read_limits{0, 0, 64'000, 1'000'000, 16'000'000};
  /// Read limits of the bounded chunked reads
  std::vector<std::size_t> bounded_chunk_read_limits{1, 1'000, 64'000, 1'000'000};
  std::vector<std::size_t> bounded_pass_read_limits{0, 64'000, 1'000'000};
};

/**
 * @brief Draws the reads of files of `num_rows` rows from `recipe`
 */
std::vector<read_settings> pick_reads(read_recipe const& recipe,
                                      cudf::size_type num_rows,
                                      std::mt19937& rng)
{
  std::vector<read_settings> reads(recipe.whole_reads, read_settings{parquet_reader_kind::WHOLE});
  for (int i = 0; i < recipe.chunked_reads + recipe.hybrid_scan_reads; ++i) {
    auto const reader =
      i < recipe.chunked_reads ? parquet_reader_kind::CHUNKED : parquet_reader_kind::HYBRID_SCAN;
    auto const chunk_read_limit = pick(rng, recipe.chunk_read_limits);
    auto const pass_read_limit  = pick(rng, recipe.pass_read_limits);
    reads.push_back({reader, chunk_read_limit, pass_read_limit});
  }
  // Row ranges have at least one row
  for (int i = 0; num_rows > 0 and i < recipe.bounded_reads + recipe.bounded_chunked_reads; ++i) {
    read_settings read{i < recipe.bounded_reads ? parquet_reader_kind::WHOLE
                                                : parquet_reader_kind::CHUNKED};
    read.skip_rows = std::uniform_int_distribution<cudf::size_type>(0, num_rows - 1)(rng);
    read.num_rows =
      std::uniform_int_distribution<cudf::size_type>(1, num_rows - read.skip_rows)(rng);
    if (read.reader == parquet_reader_kind::CHUNKED) {
      read.chunk_read_limit = pick(rng, recipe.bounded_chunk_read_limits);
      read.pass_read_limit  = pick(rng, recipe.bounded_pass_read_limits);
    }
    reads.push_back(read);
  }
  return reads;
}

/**
 * @brief Values that the cases of a fuzz run draw their tables, files and reads from
 */
struct fuzz_recipe {
  table_recipe table;
  writer_recipe writer;
  read_recipe reads;
};

/**
 * @brief A case of a fuzz run, which draws its settings from `recipe` with engines seeded by `seed`
 */
struct fuzz_case {
  fuzz_recipe recipe;
  int seed;
};

/**
 * @brief Prints a case as its seed, in gtest messages and test names
 */
void PrintTo(fuzz_case const& c, std::ostream* os) { *os << c.seed; }

/**
 * @brief Cases of `recipe` with the seeds in [`begin`, `end`), which must not be negative since
 * cases are named by their seeds
 */
auto seeds(fuzz_recipe const& recipe, int begin, int end)
{
  std::vector<fuzz_case> cases;
  for (int seed = begin; seed < end; ++seed) {
    cases.push_back({recipe, seed});
  }
  return ::testing::ValuesIn(cases);
}

/**
 * @brief Stages of a case, which draw from their own engines so that a change to the draws of one
 * stage leaves the other stages as they were
 */
enum class case_stage { TABLE, WRITER, READS };

std::mt19937 stage_rng(int seed, case_stage stage)
{
  std::seed_seq sequence{seed, static_cast<int>(stage)};
  return std::mt19937{sequence};
}

/**
 * @brief Whether the device has a CUDA error
 *
 * A bug can leave a sticky CUDA error, such as an illegal memory access in a kernel. The process
 * cannot recover from it, since every later CUDA call fails, so no other case can be tested in the
 * process. The case that hit the error fails, the remaining cases of the process are skipped
 * rather than failed, and the run fails at the end (also when the error only shows when the next
 * case starts). To test the remaining cases, rerun them in a new process without the case that
 * caused the error (the one that failed, or else the one before the first skipped case), for
 * example with `--gtest_filter=-Default/ParquetFuzzTest.ReadBack/<seed>`.
 */
bool device_failed() { return cudaDeviceSynchronize() != cudaSuccess; }

/**
 * @brief Checks of a whole fuzz run, after all its cases
 */
struct run_checks : public ::testing::Environment {
  int cases_read = 0;  ///< Cases that got as far as the reads

  /**
   * @brief Fails a run that left a CUDA error (see `device_failed`), and a run of several cases in
   * which none got as far as the reads, so that a run that skips every case cannot pass
   */
  void TearDown() override
  {
    EXPECT_FALSE(device_failed()) << "A case left a CUDA error that makes the device unusable";
    if (::testing::UnitTest::GetInstance()->test_to_run_count() > 1) {
      EXPECT_GT(cases_read, 0) << "No case got as far as reading its files back";
    }
  }
};

auto* const checks = static_cast<run_checks*>(::testing::AddGlobalTestEnvironment(new run_checks));

}  // namespace

/**
 * @brief Fixture of a fuzz case
 */
struct ParquetFuzzTest : public cudf::test::BaseFixtureWithParam<fuzz_case> {
  /**
   * @brief Generates the table of the case, writes it to Parquet files and draws the reads, or
   * skips the case after a CUDA error in an earlier case (see `device_failed`)
   */
  void SetUp() override
  {
    if (device_failed()) {
      GTEST_SKIP() << "An earlier case left a CUDA error that makes the device unusable";
    }
    generate_table();
    if (table == nullptr) { return; }
    write_files();
    if (IsSkipped() or HasFailure()) { return; }
    auto rng = stage_rng(GetParam().seed, case_stage::READS);
    reads    = pick_reads(GetParam().recipe.reads, table->num_rows(), rng);
    ++checks->cases_read;
  }

  /**
   * @brief Removes the files of the case, after keeping a copy if the case failed
   */
  void TearDown() override
  {
    if (HasFailure()) { keep_failure(); }
    for (auto const& path : paths) {
      std::filesystem::remove(path);
    }
  }

  /**
   * @brief Copies the files of a failed case to `parquet_fuzz_failures/<seed>/`, with a `case.txt`
   * that holds the case's settings, its reads and the command that reruns it
   */
  void keep_failure() const
  {
    auto const dir =
      std::filesystem::path{"parquet_fuzz_failures"} / std::to_string(GetParam().seed);
    std::filesystem::create_directories(dir);
    for (auto const& path : paths) {
      if (std::filesystem::exists(path)) {
        std::filesystem::copy_file(path,
                                   dir / std::filesystem::path{path}.filename(),
                                   std::filesystem::copy_options::overwrite_existing);
      }
    }
    auto const* test = ::testing::UnitTest::GetInstance()->current_test_info();
    auto const name  = std::string{test->test_suite_name()} + "." + test->name();
    std::ofstream info{dir / "case.txt"};
    info << "rerun: PARQUET_FUZZ --gtest_filter=" << name << "\n"
         << "built with cuDF " << CUDF_VERSION_MAJOR << "." << CUDF_VERSION_MINOR << "."
         << CUDF_VERSION_PATCH << "\n"
         << "settings: " << description << "\n";
    for (auto const& read : reads) {
      info << "read: " << to_string(read) << "\n";
    }
    std::cout << "PARQUET-FUZZ-FAILURE kept in " << std::filesystem::absolute(dir) << std::endl;
  }

  /**
   * @brief Generates the random table of the case, or skips the case if the generator fails
   */
  void generate_table()
  {
    auto const& [recipe, seed] = GetParam();
    auto rng                   = stage_rng(seed, case_stage::TABLE);
    auto const settings        = pick_table_settings(recipe.table, rng);
    description                = "seed=" + std::to_string(seed) + " " + settings.description;
    SCOPED_TRACE(description);
    try {
      table =
        create_random_table(settings.types, row_count{settings.num_rows}, settings.profile, seed);
    } catch (std::exception const& e) {
      ASSERT_FALSE(device_failed())
        << "Generating the table left the device unusable: " << e.what();
      // Some settings ask for more data than a column can hold, which is not a Parquet issue
      GTEST_SKIP() << "The generator failed: " << e.what();
    }
  }

  /**
   * @brief Writes the table to random Parquet files, or skips the case if the table is beyond the
   * writer's limits or does not fit in device memory
   */
  void write_files()
  {
    auto const& [recipe, seed] = GetParam();
    auto rng                   = stage_rng(seed, case_stage::WRITER);
    auto const settings        = pick_writer_settings(recipe.writer, table->view(), rng);
    description += " " + settings.description;
    SCOPED_TRACE(description);
    for (int f = 0; f < settings.num_files; ++f) {
      paths.push_back(temp_env->get_temp_filepath("parquet_fuzz_" + std::to_string(seed) + "_" +
                                                  std::to_string(f) + ".parquet"));
    }
    try {
      write_parquet_files(table->view(), settings, paths);
    } catch (std::exception const& e) {
      ASSERT_FALSE(device_failed()) << "Writing left the device unusable: " << e.what();
      if (is_writer_limit(e)) {
        GTEST_SKIP() << "The table is beyond a writer limit: " << e.what();
      }
      // How much device memory a table needs depends on the GPU, so running out is not a failure
      if (dynamic_cast<std::bad_alloc const*>(&e) != nullptr) {
        GTEST_SKIP() << "The table does not fit in device memory: " << e.what();
      }
      FAIL() << "Writing failed: " << e.what();
    }
  }

  std::string description;  ///< Settings of the case, for failure messages
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> paths;
  std::vector<read_settings> reads;
};

TEST_P(ParquetFuzzTest, ReadBack)
{
  SCOPED_TRACE(description);
  for (auto const& read : reads) {
    SCOPED_TRACE(to_string(read));
    EXPECT_NO_THROW(checked_read(read, paths, table->view()));
  }
}

// To fuzz other values, instantiate the test with another recipe, for example
//   fuzz_recipe large_tables()
//   {
//     fuzz_recipe recipe;
//     recipe.table.num_rows = {1'000'000, 10'000'000};
//     return recipe;
//   }
//   INSTANTIATE_TEST_SUITE_P(LargeTables,
//                            ParquetFuzzTest,
//                            seeds(large_tables(), 0, 100),
//                            ::testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(Default,
                         ParquetFuzzTest,
                         seeds(fuzz_recipe{}, 0, 1000),
                         ::testing::PrintToStringParamName());

CUDF_TEST_PROGRAM_MAIN()
