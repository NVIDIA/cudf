/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/direct_join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

using key_wrapper = cudf::test::fixed_width_column_wrapper<std::uint32_t>;

struct DirectJoinTest : public cudf::test::BaseFixture {
  // Runs the join and checks the returned pairs against a host-side reference
  void compare_to_reference(std::vector<std::uint32_t> const& left_keys,
                            std::vector<std::uint32_t> const& right_keys,
                            std::size_t capacity)
  {
    auto const left  = key_wrapper(left_keys.begin(), left_keys.end());
    auto const right = key_wrapper(right_keys.begin(), right_keys.end());

    auto const [left_indices, right_indices] = cudf::direct_inner_join(left, right, capacity);

    auto const stream          = cudf::get_default_stream();
    auto const h_left_indices  = cudf::detail::make_std_vector(*left_indices, stream);
    auto const h_right_indices = cudf::detail::make_std_vector(*right_indices, stream);

    auto right_row_of = std::unordered_map<std::uint32_t, cudf::size_type>{};
    for (std::size_t i = 0; i < right_keys.size(); ++i) {
      right_row_of[right_keys[i]] = static_cast<cudf::size_type>(i);
    }

    auto expected_left  = std::vector<cudf::size_type>{};
    auto expected_right = std::vector<cudf::size_type>{};
    for (std::size_t i = 0; i < left_keys.size(); ++i) {
      if (auto const it = right_row_of.find(left_keys[i]); it != right_row_of.end()) {
        expected_left.push_back(static_cast<cudf::size_type>(i));
        expected_right.push_back(it->second);
      }
    }

    // The probe is a single stable pass over the left keys, so output order is deterministic
    EXPECT_EQ(h_left_indices, expected_left);
    EXPECT_EQ(h_right_indices, expected_right);
  }
};

TEST_F(DirectJoinTest, DenseKeys)
{
  auto right_keys = std::vector<std::uint32_t>(1000);
  std::iota(right_keys.begin(), right_keys.end(), 0);

  auto left_keys = std::vector<std::uint32_t>{};
  for (std::uint32_t i = 0; i < 3000; ++i) {
    left_keys.push_back(i % 1500);  // keys in [1000, 1500) are unmatched
  }

  compare_to_reference(left_keys, right_keys, 1500);
}

TEST_F(DirectJoinTest, SparseKeys)
{
  auto const right_keys = std::vector<std::uint32_t>{7, 0, 42, 999, 512, 3};
  auto const left_keys  = std::vector<std::uint32_t>{42, 42, 1, 999, 0, 998, 3, 7, 100};

  compare_to_reference(left_keys, right_keys, 1000);
}

TEST_F(DirectJoinTest, EmptyInputs)
{
  auto const empty    = key_wrapper{};
  auto const nonempty = key_wrapper{0, 1, 2};

  {
    auto const [left_indices, right_indices] = cudf::direct_inner_join(empty, nonempty, 3);
    EXPECT_EQ(left_indices->size(), 0);
    EXPECT_EQ(right_indices->size(), 0);
  }
  {
    auto const [left_indices, right_indices] = cudf::direct_inner_join(nonempty, empty, 3);
    EXPECT_EQ(left_indices->size(), 0);
    EXPECT_EQ(right_indices->size(), 0);
  }
}

TEST_F(DirectJoinTest, InvalidKeyType)
{
  auto const keys = cudf::test::fixed_width_column_wrapper<std::int32_t>{0, 1, 2};

  EXPECT_THROW(std::ignore = cudf::direct_inner_join(keys, keys, 3), cudf::data_type_error);
}

TEST_F(DirectJoinTest, NullKeys)
{
  auto const valid      = key_wrapper{0, 1, 2};
  auto const with_nulls = key_wrapper{{0, 1, 2}, cudf::test::iterators::null_at(1)};

  EXPECT_THROW(std::ignore = cudf::direct_inner_join(with_nulls, valid, 3), std::invalid_argument);
  EXPECT_THROW(std::ignore = cudf::direct_inner_join(valid, with_nulls, 3), std::invalid_argument);
}

TEST_F(DirectJoinTest, InsufficientCapacity)
{
  auto const keys = key_wrapper{0, 1, 2};

  EXPECT_THROW(std::ignore = cudf::direct_inner_join(keys, keys, 2), std::invalid_argument);
}
