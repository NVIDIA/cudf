/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/table/table.hpp>
#include <cudf/timezone.hpp>
#include <cudf/utilities/error.hpp>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr std::string_view canonical_zone_name = "America/Los_Angeles";

// Candidate locations for the system TZif database, probed in order.
constexpr std::array<std::string_view, 3> candidate_tz_dirs{
  "/usr/share/zoneinfo",
  "/usr/lib/zoneinfo",
  "/etc/zoneinfo",
};

std::optional<std::filesystem::path> find_system_tz_dir()
{
  static std::optional<std::filesystem::path> const cached = [] {
    namespace fs      = std::filesystem;
    auto const usable = [](fs::path const& dir) {
      std::error_code ec;
      return fs::is_regular_file(dir / canonical_zone_name, ec);
    };
    if (auto const* env = std::getenv("TZDIR")) {
      if (fs::path const d{env}; usable(d)) { return std::optional{d}; }
    }
    for (auto const sv : candidate_tz_dirs) {
      if (fs::path const d{sv}; usable(d)) { return std::optional{d}; }
    }
    return std::optional<fs::path>{};
  }();
  return cached;
}

}  // namespace

class TimezoneTransitionTableTest : public cudf::test::BaseFixture {};

TEST_F(TimezoneTransitionTableTest, UtcShortCircuitsWithoutReadingFile)
{
  auto const table = cudf::make_timezone_transition_table(std::nullopt, "UTC");
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 0);
}

TEST_F(TimezoneTransitionTableTest, EmptyZoneNameShortCircuitsWithoutReadingFile)
{
  auto const table = cudf::make_timezone_transition_table(std::nullopt, "");
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 0);
}

TEST_F(TimezoneTransitionTableTest, CanonicalZoneProducesTwoColumnTable)
{
  auto const tz_dir = find_system_tz_dir();
  if (!tz_dir) { GTEST_SKIP() << "No system zoneinfo directory with " << canonical_zone_name; }

  auto const table = cudf::make_timezone_transition_table(tz_dir->string(), canonical_zone_name);
  ASSERT_EQ(table->num_columns(), 2);
  // Sanity: the future cycle dominates the row count, so we expect hundreds of rows.
  EXPECT_GT(table->num_rows(), 100);
}

TEST_F(TimezoneTransitionTableTest, UnknownZoneThrows)
{
  auto const tz_dir = find_system_tz_dir();
  if (!tz_dir) { GTEST_SKIP() << "No system zoneinfo directory with " << canonical_zone_name; }

  EXPECT_THROW(cudf::make_timezone_transition_table(tz_dir->string(), "Not_A/Real_Zone_bXYZ"),
               cudf::logic_error);
}

class TimezoneAliasResolutionTest : public cudf::test::BaseFixture {
 protected:
  void SetUp() override
  {
    // make the directory name process-unique
    auto const tmpl =
      (std::filesystem::temp_directory_path() / (std::string{"cudf_tz_alias_test_"} + ".XXXXXX"))
        .string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    ASSERT_NE(::mkdtemp(buf.data()), nullptr) << "mkdtemp failed: " << std::strerror(errno);
    tz_dir_ = buf.data();
  }

  void TearDown() override
  {
    std::error_code ec;
    std::filesystem::remove_all(tz_dir_, ec);
  }

  [[nodiscard]] bool install_zone(std::string_view zone_name) const
  {
    auto const src_dir = find_system_tz_dir();
    if (!src_dir) { return false; }

    std::error_code ec;
    auto const dst = tz_dir_ / zone_name;
    std::filesystem::create_directories(dst.parent_path(), ec);
    if (ec) {
      ADD_FAILURE() << "create_directories(" << dst.parent_path() << ") failed: " << ec.message();
      return false;
    }
    std::filesystem::copy_file(
      *src_dir / canonical_zone_name, dst, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      ADD_FAILURE() << "copy_file(" << (*src_dir / canonical_zone_name) << " -> " << dst
                    << ") failed: " << ec.message();
      return false;
    }
    return true;
  }

  void write_tzdata_zi(std::string_view contents) const
  {
    std::ofstream{tz_dir_ / "tzdata.zi"} << contents;
  }

  [[nodiscard]] std::string dir() const { return tz_dir_.string(); }

 private:
  std::filesystem::path tz_dir_;
};

TEST_F(TimezoneAliasResolutionTest, DirectLookupUnaffectedByNewFallback)
{
  if (!install_zone(canonical_zone_name)) {
    GTEST_SKIP() << "No system zoneinfo directory with " << canonical_zone_name;
  }

  auto const table = cudf::make_timezone_transition_table(dir(), canonical_zone_name);
  EXPECT_GT(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(TimezoneAliasResolutionTest, ResolvesShortFormLinkFromTzdataZi)
{
  if (!install_zone(canonical_zone_name)) {
    GTEST_SKIP() << "No system zoneinfo directory with " << canonical_zone_name;
  }
  write_tzdata_zi(
    "# synthetic tzdata.zi for libcudf tests\n"
    "L America/Los_Angeles US/Pacific\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), canonical_zone_name);
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

TEST_F(TimezoneAliasResolutionTest, ResolvesLongFormLinkFromTzdataZi)
{
  if (!install_zone(canonical_zone_name)) {
    GTEST_SKIP() << "No system zoneinfo directory with " << canonical_zone_name;
  }
  write_tzdata_zi("Link America/Los_Angeles US/Pacific\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), canonical_zone_name);
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

TEST_F(TimezoneAliasResolutionTest, ResolvesChainedLinks)
{
  if (!install_zone(canonical_zone_name)) {
    GTEST_SKIP() << "No system zoneinfo directory with " << canonical_zone_name;
  }
  // Neither "intermediate" nor "US/Pacific" exists on disk. The resolver must traverse
  //   US/Pacific -> intermediate -> America/Los_Angeles to find a real file.
  write_tzdata_zi(
    "L America/Los_Angeles intermediate\n"
    "L intermediate US/Pacific\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), canonical_zone_name);
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

TEST_F(TimezoneAliasResolutionTest, ThrowsWhenLinkTargetIsAlsoMissing)
{
  // No zone files installed in tz_dir_.
  write_tzdata_zi("L Also/Missing US/Pacific\n");

  EXPECT_THROW(cudf::make_timezone_transition_table(dir(), "US/Pacific"), cudf::logic_error);
}

TEST_F(TimezoneAliasResolutionTest, ThrowsWhenNoTzdataZiPresent)
{
  EXPECT_THROW(cudf::make_timezone_transition_table(dir(), "US/Pacific"), cudf::logic_error);
}

TEST_F(TimezoneAliasResolutionTest, IgnoresCommentsAndNonLinkDirectives)
{
  if (!install_zone(canonical_zone_name)) {
    GTEST_SKIP() << "No system zoneinfo directory with " << canonical_zone_name;
  }
  write_tzdata_zi(
    "# a leading comment\n"
    "\n"
    "R SomeRule 1970 o - Jan 1 0 0 S\n"      // `Rule` entry, must be ignored
    "Z Fake/Zone 0 - LMT\n"                  // `Zone` entry, must be ignored
    "   L America/Los_Angeles US/Pacific\n"  // link with leading whitespace
    "# trailing comment\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), canonical_zone_name);
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

CUDF_TEST_PROGRAM_MAIN()
