/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/asof_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/buffer>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

using cudf::experimental::asof_join;
using cudf::experimental::asof_join_exact_matches;
using cudf::experimental::asof_join_strategy;
using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

void expect_indices(cuda::device_buffer<cudf::size_type> const& actual,
                    std::vector<cudf::size_type> const& expected,
                    cuda::stream_ref stream)
{
  auto const host_actual = cudf::detail::make_std_vector(actual, stream);
  EXPECT_EQ(host_actual, expected);
}

}  // anonymous namespace

class AsofJoinTest : public cudf::test::BaseFixtureWithHarness {
 public:
  cudf::memory_resources setup_resources() { return {harness().setup_mr(), harness().setup_mr()}; }
};

template <typename T>
class AsofJoinOrderedKeyTest : public AsofJoinTest {};

using OrderedKeyTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::ChronoTypes>;
TYPED_TEST_SUITE(AsofJoinOrderedKeyTest, OrderedKeyTypes);

TYPED_TEST(AsofJoinOrderedKeyTest, GroupedBackward)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> right_by({0, 0, 0, 0, 0, 1, 1, 1}, stream, setup_mr);
  fixed_width_column_wrapper<TypeParam, int32_t> right_on(
    {0, 1, 3, 3, 7, 0, 2, 6}, cudf::test::iterators::nulls_at({0, 5}), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_by({0, 0, 0, 0, 0, 1, 1, 1, 2, 0}, stream, setup_mr);
  fixed_width_column_wrapper<TypeParam, int32_t> left_on(
    {0, 1, 2, 3, 8, 1, 5, 6, 9, 3}, cudf::test::iterators::nulls_at({9}), stream, setup_mr);

  asof_join join(cudf::table_view{{right_by}}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(
    result,
    {cudf::JoinNoMatch, 1, 1, 3, 4, cudf::JoinNoMatch, 6, 7, cudf::JoinNoMatch, cudf::JoinNoMatch},
    stream);
}

TYPED_TEST(AsofJoinOrderedKeyTest, UngroupedBackward)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<TypeParam, int32_t> right_on(
    {0, 1, 3, 3, 7}, cudf::test::iterators::nulls_at({0}), stream, setup_mr);
  fixed_width_column_wrapper<TypeParam, int32_t> left_on(
    {0, 1, 2, 3, 8, 3}, cudf::test::iterators::nulls_at({5}), stream, setup_mr);

  asof_join join(cudf::table_view{}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(result, {cudf::JoinNoMatch, 1, 1, 3, 4, cudf::JoinNoMatch}, stream);
}

TEST_F(AsofJoinTest, MultipleGroupingKeys)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  strings_column_wrapper right_by_1({"a", "a", "a", "b", "b"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_by_2({1, 1, 2, 1, 1}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on({1, 4, 2, 3, 5}, stream, setup_mr);
  strings_column_wrapper left_by_1({"b", "a", "a"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_by_2({1, 2, 1}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on({4, 1, 4}, stream, setup_mr);

  asof_join join(cudf::table_view{{right_by_1, right_by_2}}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{{left_by_1, left_by_2}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(result, {3, cudf::JoinNoMatch, 1}, stream);
}

template <typename T>
class AsofJoinGroupingKeyTest : public AsofJoinTest {
 public:
  void run_test()
    requires(cudf::is_floating_point<T>())
  {
    auto const stream   = this->stream();
    auto const setup_mr = this->setup_resources();

    auto const nan = std::numeric_limits<T>::quiet_NaN();
    auto const inf = std::numeric_limits<T>::infinity();
    auto right_by =
      fixed_width_column_wrapper<T>({T{0}, T{0}, T{0}, T{0}, T{0}, T{1}, T{1}, inf, nan, nan},
                                    cudf::test::iterators::nulls_at({0}),
                                    stream,
                                    setup_mr)
        .release();
    auto left_by = fixed_width_column_wrapper<T>(
                     {T{1}, T{0}, T{2}, T{0}, T{1}, T{0}, T{1}, T{0}, T{0}, inf, nan, nan, T{0}},
                     cudf::test::iterators::nulls_at({12}),
                     stream,
                     setup_mr)
                     .release();
    std::vector<int32_t> right_ordered    = {1, 1, 3, 3, 7, 2, 6, 2, 1, 5};
    std::vector<int32_t> left_ordered     = {5, 0, 9, 3, 1, 1, 6, 8, 2, 2, 5, 0, 9};
    std::vector<cudf::size_type> expected = {5,
                                             cudf::JoinNoMatch,
                                             cudf::JoinNoMatch,
                                             3,
                                             cudf::JoinNoMatch,
                                             1,
                                             6,
                                             4,
                                             1,
                                             7,
                                             9,
                                             cudf::JoinNoMatch,
                                             cudf::JoinNoMatch};

    check(right_by->view(), left_by->view(), right_ordered, left_ordered, expected);
  }

  void run_test()
    requires(cudf::is_fixed_point<T>())
  {
    auto const stream   = this->stream();
    auto const setup_mr = this->setup_resources();

    auto right_by = cudf::test::fixed_point_column_wrapper<typename T::rep>(
                      {0, 0, 0, 0, 1, 1}, numeric::scale_type{-2}, stream, setup_mr)
                      .release();
    auto left_by = cudf::test::fixed_point_column_wrapper<typename T::rep>(
                     {1, 0, 2, 0, 1, 0, 1, 0, 0}, numeric::scale_type{-2}, stream, setup_mr)
                     .release();

    check(right_by->view(), left_by->view());
  }

  void run_test()
    requires(cudf::is_fixed_width<T>() && !cudf::is_floating_point<T>() &&
             !cudf::is_fixed_point<T>())
  {
    auto const stream   = this->stream();
    auto const setup_mr = this->setup_resources();

    auto right_by =
      fixed_width_column_wrapper<T, int32_t>({0, 0, 0, 0, 1, 1}, stream, setup_mr).release();
    auto left_by =
      fixed_width_column_wrapper<T, int32_t>({1, 0, 2, 0, 1, 0, 1, 0, 0}, stream, setup_mr)
        .release();

    check(right_by->view(), left_by->view());
  }

  void run_test()
    requires(!cudf::is_fixed_width<T>())
  {
    auto const stream   = this->stream();
    auto const setup_mr = this->setup_resources();

    auto right_by = strings_column_wrapper({"", "", "a", "a", "a", "a", "b", "b"},
                                           cudf::test::iterators::nulls_at({0, 1}),
                                           stream,
                                           setup_mr)
                      .release();
    auto left_by = strings_column_wrapper({"b", "a", "c", "a", "b", "a", "b", "a", "a", "", ""},
                                          cudf::test::iterators::nulls_at({9, 10}),
                                          stream,
                                          setup_mr)
                     .release();
    std::vector<int32_t> right_ordered    = {1, 2, 1, 3, 3, 7, 2, 6};
    std::vector<int32_t> left_ordered     = {5, 0, 9, 3, 1, 1, 6, 8, 2, 2, 5};
    std::vector<cudf::size_type> expected = {6,
                                             cudf::JoinNoMatch,
                                             cudf::JoinNoMatch,
                                             4,
                                             cudf::JoinNoMatch,
                                             2,
                                             7,
                                             5,
                                             2,
                                             cudf::JoinNoMatch,
                                             cudf::JoinNoMatch};
    check(right_by->view(), left_by->view(), right_ordered, left_ordered, expected);
  }

 private:
  void check(cudf::column_view right_by,
             cudf::column_view left_by,
             std::vector<int32_t> const& right_ordered    = {1, 3, 3, 7, 2, 6},
             std::vector<int32_t> const& left_ordered     = {5, 0, 9, 3, 1, 1, 6, 8, 2},
             std::vector<cudf::size_type> const& expected = {
               4, cudf::JoinNoMatch, cudf::JoinNoMatch, 2, cudf::JoinNoMatch, 0, 5, 3, 0})
  {
    auto const stream   = this->stream();
    auto const setup_mr = this->setup_resources();
    auto const mr       = this->resources();

    fixed_width_column_wrapper<int32_t> right_on(
      right_ordered.begin(), right_ordered.end(), stream, setup_mr);
    fixed_width_column_wrapper<int32_t> left_on(
      left_ordered.begin(), left_ordered.end(), stream, setup_mr);
    asof_join join(cudf::table_view{{right_by}}, right_on, stream, mr);
    auto result = join.join(cudf::table_view{{left_by}},
                            left_on,
                            asof_join_strategy::BACKWARD,
                            asof_join_exact_matches::YES,
                            stream,
                            mr);

    expect_indices(result, expected, stream);
  }
};

using GroupingKeyTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                            cudf::test::FloatingPointTypes,
                                            cudf::test::FixedPointTypes,
                                            cudf::test::ChronoTypes,
                                            cudf::test::StringTypes>;
TYPED_TEST_SUITE(AsofJoinGroupingKeyTest, GroupingKeyTypes);

TYPED_TEST(AsofJoinGroupingKeyTest, GroupedBackward) { this->run_test(); }

TEST_F(AsofJoinTest, ConcurrentMultiStreamProbes)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();

  constexpr std::size_t num_probes = 4;
  fixed_width_column_wrapper<int32_t> right_by({0, 0, 1, 1, 2, 2}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on({1, 4, 2, 6, 3, 8}, stream, setup_mr);

  fixed_width_column_wrapper<int32_t> left_by_0({0, 1, 2}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on_0({4, 5, 9}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_by_1({2, 0, 3}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on_1({2, 2, 7}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_by_2({1, 2, 0}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on_2({6, 8, 0}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_by_3({0, 2, 1}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on_3({3, 4, 1}, stream, setup_mr);

  auto const left_by =
    std::array<cudf::column_view, num_probes>{left_by_0, left_by_1, left_by_2, left_by_3};
  auto const left_on =
    std::array<cudf::column_view, num_probes>{left_on_0, left_on_1, left_on_2, left_on_3};
  auto const expected = std::array<std::vector<cudf::size_type>, num_probes>{
    std::vector<cudf::size_type>{1, 2, 5},
    std::vector<cudf::size_type>{cudf::JoinNoMatch, 0, cudf::JoinNoMatch},
    std::vector<cudf::size_type>{3, 5, cudf::JoinNoMatch},
    std::vector<cudf::size_type>{0, 4, cudf::JoinNoMatch}};

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  auto const stream_device = cuda::device_ref{device};
  std::vector<std::unique_ptr<cuda::stream>> streams;
  streams.reserve(num_probes);
  for (std::size_t i = 0; i < num_probes; ++i) {
    streams.push_back(std::make_unique<cuda::stream>(stream_device));
  }

  // Construct on a stream of its own so the probes below run on different streams than the
  // group index was built on, then synchronize it as the `join()` docs require.
  stream.sync();
  cuda::stream const build_stream{stream_device};
  auto const mr = this->resources();
  asof_join join(cudf::table_view{{right_by}}, right_on, build_stream, mr);
  build_stream.sync();

  // Bound only the host start gate, not join() or CUDA/future cleanup.
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  std::mutex mutex;
  std::condition_variable gate;
  std::size_t ready{0};
  bool start{false};
  bool cancelled{false};
  std::exception_ptr launch_error;
  std::vector<std::future<cuda::device_buffer<cudf::size_type>>> results;
  results.reserve(num_probes);
  try {
    for (std::size_t i = 0; i < num_probes; ++i) {
      results.push_back(std::async(std::launch::async, [&, i, mr] {
        {
          std::unique_lock lock{mutex};
          ++ready;
          gate.notify_all();
          if (!gate.wait_until(lock, deadline, [&] { return start || cancelled; })) {
            cancelled = true;
            gate.notify_all();
          }
          if (cancelled) {
            throw std::runtime_error("Concurrent probe start gate timed out or was cancelled");
          }
        }
        CUDF_CUDA_TRY(cudaSetDevice(device));
        return join.join(cudf::table_view{{left_by[i]}},
                         left_on[i],
                         asof_join_strategy::BACKWARD,
                         asof_join_exact_matches::YES,
                         *streams[i],
                         mr);
      }));
    }
  } catch (...) {
    launch_error = std::current_exception();
    std::lock_guard lock{mutex};
    cancelled = true;
  }
  {
    std::unique_lock lock{mutex};
    if (!gate.wait_until(lock, deadline, [&] { return ready == num_probes || cancelled; })) {
      cancelled = true;
    }
    start = true;
  }
  // Release all workers, including on partial launch failure, before waiting for cleanup.
  gate.notify_all();
  for (auto& result : results) {
    result.wait();
  }
  for (auto const& probe_stream : streams) {
    probe_stream->sync();
  }
  if (launch_error) { std::rethrow_exception(launch_error); }
  for (std::size_t i = 0; i < num_probes; ++i) {
    expect_indices(results[i].get(), expected[i], stream);
  }
}

TEST_F(AsofJoinTest, EmptyGroupedInputs)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> empty(std::initializer_list<int32_t>{}, stream, setup_mr);
  auto empty_strings  = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto const empty_by = cudf::table_view{{empty, empty_strings->view()}};
  fixed_width_column_wrapper<int32_t> by({0, 1}, stream, setup_mr);
  strings_column_wrapper strings_by({"a", "b"}, stream, setup_mr);
  auto const nonempty_by = cudf::table_view{{by, strings_by}};
  fixed_width_column_wrapper<int32_t> on({1, 2}, stream, setup_mr);

  asof_join empty_right_join(empty_by, empty, stream, mr);
  auto no_matches = empty_right_join.join(
    nonempty_by, on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES, stream, mr);
  expect_indices(no_matches, {cudf::JoinNoMatch, cudf::JoinNoMatch}, stream);

  asof_join join(nonempty_by, on, stream, mr);
  auto empty_result = join.join(
    empty_by, empty, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES, stream, mr);
  expect_indices(empty_result, {}, stream);
}

TEST_F(AsofJoinTest, GroupedMemoryResources)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();

  strings_column_wrapper right_by(
    {"", "a", "a", "b"}, cudf::test::iterators::nulls_at({0}), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on(
    {1, 0, 3, 2}, cudf::test::iterators::nulls_at({1}), stream, setup_mr);
  strings_column_wrapper left_by(
    {"a", "b", "", "c", "a"}, cudf::test::iterators::nulls_at({2}), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on(
    {3, 4, 9, 9, 3}, cudf::test::iterators::nulls_at({4}), stream, setup_mr);
  // Keep retained index allocations separate from the probe's exact output-byte accounting.
  auto build_resources  = cudf::test::memory_resource_test_harness{mr()};
  auto& probe_resources = harness();
  {
    auto join = [&] {
      auto guard  = build_resources.fail_on_current_device_resource_use();
      auto result = std::make_unique<asof_join>(
        cudf::table_view{{right_by}}, right_on, stream, build_resources.resources());
      build_resources.synchronize(stream);
      return result;
    }();
    auto const retained = build_resources.expect_output_allocations_live(stream).value;
    EXPECT_EQ(build_resources.output_mr().get_bytes_counter().total, retained);
    build_resources.expect_temporary_allocation_activity(stream);
    build_resources.expect_temporary_allocations_released(stream);

    cudf::test::expect_api_uses_memory_resources(
      probe_resources,
      [&](cudf::memory_resources mr) {
        return join->join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);
      },
      [](auto const& result) { return result.size() * sizeof(cudf::size_type); },
      [&](auto const& result) {
        expect_indices(
          result, {2, 3, cudf::JoinNoMatch, cudf::JoinNoMatch, cudf::JoinNoMatch}, stream);
      },
      {cudf::test::output_allocation_expectation::EXACT,
       cudf::test::temporary_allocation_expectation::SOME},
      stream);
    EXPECT_EQ(build_resources.output_mr().get_bytes_counter().value, retained);
  }
  build_resources.expect_no_live_allocations(stream);
}

TEST_F(AsofJoinTest, EmptyAndUngroupedMemoryResources)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();

  fixed_width_column_wrapper<int32_t> empty(std::initializer_list<int32_t>{}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> on({1, 2}, stream, setup_mr);
  strings_column_wrapper by({"a", "b"}, stream, setup_mr);
  auto empty_strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  for (bool grouped : {false, true}) {
    for (bool empty_right : {false, true}) {
      for (bool empty_left : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "grouped=" << grouped << " empty_right=" << empty_right
                                          << " empty_left=" << empty_left);
        auto const right_by  = !grouped      ? cudf::table_view{}
                               : empty_right ? cudf::table_view{{empty_strings->view()}}
                                             : cudf::table_view{{by}};
        auto const left_by   = !grouped     ? cudf::table_view{}
                               : empty_left ? cudf::table_view{{empty_strings->view()}}
                                            : cudf::table_view{{by}};
        auto build_resources = cudf::test::memory_resource_test_harness{mr()};
        auto probe_resources = cudf::test::memory_resource_test_harness{mr()};
        {
          auto join = [&] {
            auto guard  = build_resources.fail_on_current_device_resource_use();
            auto result = std::make_unique<asof_join>(
              right_by,
              empty_right ? cudf::column_view{empty} : cudf::column_view{on},
              stream,
              build_resources.resources());
            build_resources.synchronize(stream);
            return result;
          }();
          auto const retained = build_resources.expect_output_allocations_live(stream);
          if (!grouped || empty_right) {
            EXPECT_EQ(retained.value, (empty_right ? 1 : 2) * sizeof(cudf::size_type));
            EXPECT_EQ(build_resources.temporary_mr().get_bytes_counter().total, 0);
          }
          build_resources.expect_temporary_allocations_released(stream);
          cudf::test::expect_api_uses_memory_resources(
            probe_resources,
            [&](cudf::memory_resources mr) {
              return join->join(left_by,
                                empty_left ? cudf::column_view{empty} : cudf::column_view{on},
                                asof_join_strategy::BACKWARD,
                                asof_join_exact_matches::YES,
                                stream,
                                mr);
            },
            [](auto const& result) { return result.size() * sizeof(cudf::size_type); },
            [&](auto const& result) {
              expect_indices(result,
                             empty_left ? std::vector<cudf::size_type>{}
                             : empty_right
                               ? std::vector<cudf::size_type>{cudf::JoinNoMatch, cudf::JoinNoMatch}
                               : std::vector<cudf::size_type>{0, 1},
                             stream);
            },
            {cudf::test::output_allocation_expectation::EXACT,
             grouped && !empty_left && !empty_right
               ? cudf::test::temporary_allocation_expectation::SOME
               : cudf::test::temporary_allocation_expectation::NONE},
            stream);
        }
        build_resources.expect_no_live_allocations(stream);
      }
    }
  }
}

TEST_F(AsofJoinTest, MismatchedOrderedKeyTypes)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> right_on({1}, stream, setup_mr);
  fixed_width_column_wrapper<int64_t> left_on({1}, stream, setup_mr);
  asof_join join(cudf::table_view{}, right_on, stream, mr);

  EXPECT_THROW((void)join.join(cudf::table_view{},
                               left_on,
                               asof_join_strategy::BACKWARD,
                               asof_join_exact_matches::YES,
                               stream,
                               mr),
               cudf::data_type_error);
}

TEST_F(AsofJoinTest, MismatchedGroupingKeyTypes)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> right_by({1}, stream, setup_mr);
  fixed_width_column_wrapper<int64_t> left_by({1}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> on({1}, stream, setup_mr);
  asof_join join(cudf::table_view{{right_by}}, on, stream, mr);

  EXPECT_THROW((void)join.join(cudf::table_view{{left_by}},
                               on,
                               asof_join_strategy::BACKWARD,
                               asof_join_exact_matches::YES,
                               stream,
                               mr),
               cudf::data_type_error);
}

TEST_F(AsofJoinTest, MismatchedGroupingKeyCounts)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> key({1}, stream, setup_mr);
  asof_join join(cudf::table_view{{key}}, key, stream, mr);

  for (auto const& left_by : {cudf::table_view{}, cudf::table_view{{key, key}}}) {
    EXPECT_THROW(
      (void)join.join(
        left_by, key, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES, stream, mr),
      std::invalid_argument);
  }
}

TEST_F(AsofJoinTest, MismatchedKeyRowCounts)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> by({1, 2}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> on({1}, stream, setup_mr);

  EXPECT_THROW((asof_join{cudf::table_view{{by}}, on, stream, mr}), std::invalid_argument);
  asof_join join(cudf::table_view{{by}}, by, stream, mr);
  EXPECT_THROW((void)join.join(cudf::table_view{{by}},
                               on,
                               asof_join_strategy::BACKWARD,
                               asof_join_exact_matches::YES,
                               stream,
                               mr),
               std::invalid_argument);
}

TEST_F(AsofJoinTest, UnsupportedKeyTypes)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> on({1}, stream, setup_mr);
  strings_column_wrapper strings({"a"}, stream, setup_mr);
  cudf::test::lists_column_wrapper<int32_t> lists({{1}}, stream, setup_mr);

  EXPECT_THROW((asof_join{cudf::table_view{}, strings, stream, mr}), cudf::data_type_error);
  EXPECT_THROW((asof_join{cudf::table_view{{lists}}, on, stream, mr}), cudf::data_type_error);
}

TEST_F(AsofJoinTest, LeftKeyBetweenGroups)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  strings_column_wrapper right_by({"a", "c"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on({1, 1}, stream, setup_mr);
  strings_column_wrapper left_by({"b", "0", "d"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on({1, 1, 1}, stream, setup_mr);

  asof_join join(cudf::table_view{{right_by}}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(result, {cudf::JoinNoMatch, cudf::JoinNoMatch, cudf::JoinNoMatch}, stream);
}

TEST_F(AsofJoinTest, NullOrderedKeyAtGroupFront)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  strings_column_wrapper right_by({"a", "a", "b"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on(
    {0, 5, 3}, cudf::test::iterators::nulls_at({0}), stream, setup_mr);
  strings_column_wrapper left_by({"a", "a"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on({4, 5}, stream, setup_mr);

  asof_join join(cudf::table_view{{right_by}}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(result, {cudf::JoinNoMatch, 1}, stream);
}

TEST_F(AsofJoinTest, AllNullOrderedKeyGroups)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  strings_column_wrapper right_by({"a", "a", "b"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on(
    {1, 2, 3}, cudf::test::iterators::all_nulls(), stream, setup_mr);
  strings_column_wrapper left_by({"a", "a", "b", "b"}, stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on({0, 9, 0, 9}, stream, setup_mr);

  asof_join join(cudf::table_view{{right_by}}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(
    result, {cudf::JoinNoMatch, cudf::JoinNoMatch, cudf::JoinNoMatch, cudf::JoinNoMatch}, stream);
}

TEST_F(AsofJoinTest, SlicedColumns)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  fixed_width_column_wrapper<int32_t> right_on_base(
    {9, 0, 1, 3, 3, 7, 9}, cudf::test::iterators::nulls_at({1}), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on_base(
    {9, 0, 1, 2, 3, 3, 9}, cudf::test::iterators::nulls_at({5}), stream, setup_mr);
  auto const right_on = cudf::slice(right_on_base, {1, 6}, stream)[0];
  auto const left_on  = cudf::slice(left_on_base, {1, 6}, stream)[0];

  asof_join join(cudf::table_view{}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(result, {cudf::JoinNoMatch, 1, 1, 3, cudf::JoinNoMatch}, stream);
}

TEST_F(AsofJoinTest, SlicedGroupedColumns)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  strings_column_wrapper right_by_base(
    {"z", "", "a", "a", "b", "b", "z"}, cudf::test::iterators::nulls_at({1}), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on_base(
    {9, 1, 0, 3, 2, 4, 9}, cudf::test::iterators::nulls_at({2}), stream, setup_mr);
  auto const right_by = cudf::slice(right_by_base, {1, 6}, stream)[0];
  auto const right_on = cudf::slice(right_on_base, {1, 6}, stream)[0];
  strings_column_wrapper left_by_base(
    {"z", "b", "a", "a", "", "b", "z"}, cudf::test::iterators::nulls_at({4}), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on_base(
    {9, 3, 2, 3, 9, 4, 9}, cudf::test::iterators::nulls_at({5}), stream, setup_mr);
  auto const left_by = cudf::slice(left_by_base, {1, 6}, stream)[0];
  auto const left_on = cudf::slice(left_on_base, {1, 6}, stream)[0];

  asof_join join(cudf::table_view{{right_by}}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  expect_indices(result, {3, cudf::JoinNoMatch, 2, cudf::JoinNoMatch, cudf::JoinNoMatch}, stream);
}

TEST_F(AsofJoinTest, LargeMultiBlock)
{
  auto const stream   = this->stream();
  auto const setup_mr = this->setup_resources();
  auto const mr       = this->resources();

  // 2000 contiguous groups of 1000 rows each, ordered keys ascending within each group.
  // Exercises comparator lifetimes and dispatch across many thread blocks.
  auto constexpr num_rows   = 2'000'000;
  auto constexpr group_size = 1000;
  std::vector<int32_t> right_by_host(num_rows);
  std::vector<int32_t> right_on_host(num_rows);
  for (int32_t i = 0; i < num_rows; ++i) {
    right_by_host[i] = i / group_size;
    right_on_host[i] = i % group_size;
  }
  fixed_width_column_wrapper<int32_t> right_by(
    right_by_host.begin(), right_by_host.end(), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> right_on(
    right_on_host.begin(), right_on_host.end(), stream, setup_mr);

  auto constexpr num_groups = num_rows / group_size;
  std::vector<int32_t> left_by_host(num_groups);
  std::vector<int32_t> left_on_host(num_groups, group_size / 2);
  for (int32_t g = 0; g < num_groups; ++g) {
    left_by_host[g] = g;
  }
  fixed_width_column_wrapper<int32_t> left_by(
    left_by_host.begin(), left_by_host.end(), stream, setup_mr);
  fixed_width_column_wrapper<int32_t> left_on(
    left_on_host.begin(), left_on_host.end(), stream, setup_mr);

  asof_join join(cudf::table_view{{right_by}}, right_on, stream, mr);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES,
                          stream,
                          mr);

  std::vector<cudf::size_type> expected(num_groups);
  for (int32_t g = 0; g < num_groups; ++g) {
    expected[g] = g * group_size + group_size / 2;
  }
  expect_indices(result, expected, stream);
}
