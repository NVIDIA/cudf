/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <map>
#include <memory>
#include <vector>

namespace cudf::test {
class LargeStringsData {
 public:
  using DataPointer = std::unique_ptr<cudf::table>;

  virtual ~LargeStringsData() {}

  void add_table(std::string_view name, std::unique_ptr<cudf::table>&& data)
  {
    _data[std::string(name)] = std::move(data);
  }

  [[nodiscard]] cudf::table_view get_table(std::string_view name) const
  {
    std::string key{name};
    return _data.find(key) != _data.end() ? _data.at(key)->view() : cudf::table_view{};
  }

  void add_column(std::string_view name, std::unique_ptr<cudf::column>&& data)
  {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.emplace_back(std::move(data));
    _data[std::string(name)] = std::make_unique<cudf::table>(std::move(cols));
  }

  [[nodiscard]] cudf::column_view get_column(std::string_view name) const
  {
    std::string key{name};
    return _data.find(key) != _data.end() ? _data.at(key)->view().column(0) : cudf::column_view{};
  }

  [[nodiscard]] bool has_key(std::string_view name) const
  {
    return _data.find(std::string(name)) != _data.end();
  }

 protected:
  std::map<std::string, DataPointer> _data;
};

cudf::column_view StringsLargeTest::wide_column()
{
  std::string name{"wide1"};
  if (!g_ls_data->has_key(name)) {
    auto input =
      cudf::test::strings_column_wrapper({"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "The result does not include the value in the sum in"});
    auto counts = cudf::test::fixed_width_column_wrapper<int>({8, 8, 8, 8, 8});
    auto result = cudf::strings::repeat_strings(cudf::strings_column_view(input), counts);
    g_ls_data->add_column(name, std::move(result));
  }
  return g_ls_data->get_column(name);
}

cudf::column_view StringsLargeTest::long_column()
{
  std::string name("long1");
  if (!g_ls_data->has_key(name)) {
    auto itr = cuda::constant_iterator<std::string_view>(
      "abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY");                // 50 bytes
    auto input = cudf::test::strings_column_wrapper(itr, itr + 5'000'000);  // 250MB
    g_ls_data->add_column(name, input.release());
  }
  return g_ls_data->get_column(name);
}

cudf::column_view StringsLargeTest::very_long_column()
{
  std::string name("long2");
  if (!g_ls_data->has_key(name)) {
    auto itr   = cuda::constant_iterator<std::string_view>("12345");
    auto input = cudf::test::strings_column_wrapper(itr, itr + 30'000'000);
    g_ls_data->add_column(name, input.release());
  }
  return g_ls_data->get_column(name);
}

std::unique_ptr<LargeStringsData> StringsLargeTest::get_ls_data()
{
  CUDF_EXPECTS(g_ls_data == nullptr, "invalid call to get_ls_data");
  auto lsd_data = std::make_unique<LargeStringsData>();
  g_ls_data     = lsd_data.get();
  return lsd_data;
}

LargeStringsData* StringsLargeTest::g_ls_data = nullptr;
}  // namespace cudf::test

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  cudf::test::config config;
  config.rmm_mode = "cuda";
  init_cudf_test(argc, argv, config);
  // create object to automatically be destroyed at the end of main()
  auto lsd = cudf::test::StringsLargeTest::get_ls_data();

  if (std::getenv("GTEST_CUDF_MEMORY_PEAK")) {
    auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
    cudf::set_current_device_resource(mr);
    auto rc = RUN_ALL_TESTS();
    std::cout << "Peak memory usage " << mr.get_bytes_counter().peak << " bytes" << std::endl;
    rmm::mr::reset_current_device_resource();
    return rc;
  }
  rmm::mr::reset_current_device_resource();
  return RUN_ALL_TESTS();
}
