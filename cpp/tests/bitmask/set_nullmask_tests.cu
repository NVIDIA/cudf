/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iostream>

namespace {
struct valid_bit_functor {
  cudf::bitmask_type const* _null_mask;
  __device__ bool operator()(cudf::size_type element_index) const noexcept
  {
    return cudf::bit_is_set(_null_mask, element_index);
  }
};
}  // namespace

struct SetBitmaskTest : public cudf::test::BaseFixture {
  void expect_bitmask_equal(cudf::bitmask_type const* bitmask,  // Device Ptr
                            cudf::size_type start_bit,
                            thrust::host_vector<bool> const& expect,
                            cuda::stream_ref stream = cudf::get_default_stream())
  {
    rmm::device_uvector<bool> result(expect.size(), stream);
    auto counting_iter = cuda::counting_iterator<cudf::size_type>{0};
    thrust::transform(rmm::exec_policy_nosync(stream),
                      counting_iter + start_bit,
                      counting_iter + start_bit + expect.size(),
                      result.begin(),
                      valid_bit_functor{bitmask});

    auto host_result = cudf::detail::make_host_vector(result, stream);
    EXPECT_THAT(host_result, testing::ElementsAreArray(expect));
  }

  void test_set_null_range(cudf::size_type size,
                           cudf::size_type begin,
                           cudf::size_type end,
                           bool valid)
  {
    thrust::host_vector<bool> expected(end - begin, valid);
    // TEST
    rmm::device_buffer mask = create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    // valid ? cudf::mask_state::ALL_NULL : cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), begin, end, valid);
    expect_bitmask_equal(static_cast<cudf::bitmask_type*>(mask.data()), begin, expected);
  }

  void test_null_partition(cudf::size_type size, cudf::size_type middle, bool valid)
  {
    thrust::host_vector<bool> expected(size);
    std::generate(expected.begin(), expected.end(), [n = 0, middle, valid]() mutable {
      auto i = n++;
      return (!valid) ^ (i < middle);
    });
    // TEST
    rmm::device_buffer mask = create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, middle, valid);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), middle, size, !valid);
    expect_bitmask_equal(static_cast<cudf::bitmask_type*>(mask.data()), 0, expected);
  }

  void test_null_partition_bulk_unsafe(cudf::size_type size, cudf::size_type middle, bool valid)
  {
    thrust::host_vector<bool> expected1(size);
    thrust::host_vector<bool> expected2(size);
    std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                  cuda::counting_iterator<cudf::size_type>{size},
                  [&](auto i) {
                    expected1[i] = (!valid) ^ (i < middle);
                    expected2[i] = (valid) ^ (i < middle);
                  });
    // TEST
    rmm::device_buffer mask1 = create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    rmm::device_buffer mask2 = create_null_mask(size, cudf::mask_state::UNINITIALIZED);

    std::vector<cudf::bitmask_type*> masks{static_cast<cudf::bitmask_type*>(mask1.data()),
                                           static_cast<cudf::bitmask_type*>(mask2.data())};

    // Set first halves of bitmasks
    std::vector<cudf::size_type> begins{0, 0};
    std::vector<cudf::size_type> ends{middle, middle};
    auto valids = cudf::detail::make_host_vector<bool>(begins.size(), cudf::get_default_stream());
    valids[0]   = valid;
    valids[1]   = !valid;
    cudf::set_null_masks_unsafe(masks, begins, ends, valids);

    // Set second halves of bitmasks
    begins    = {middle, middle};
    ends      = {size, size};
    valids[0] = !valid;
    valids[1] = valid;
    cudf::set_null_masks_unsafe(
      masks, begins, ends, cudf::host_span<bool const>{valids.data(), valids.size()});

    // Verify bitmasks
    expect_bitmask_equal(static_cast<cudf::bitmask_type*>(mask1.data()), 0, expected1);
    expect_bitmask_equal(static_cast<cudf::bitmask_type*>(mask2.data()), 0, expected2);
  }

  void test_null_partition_bulk_safe(cudf::size_type size, cudf::size_type middle, bool valid)
  {
    thrust::host_vector<bool> expected(size);
    std::generate(expected.begin(), expected.end(), [n = 0, middle, valid]() mutable {
      auto i = n++;
      return (!valid) ^ (i < middle);
    });

    rmm::device_buffer mask = create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    std::vector<cudf::size_type> begins{0, middle};
    std::vector<cudf::size_type> ends{middle, size};

    std::vector<cudf::bitmask_type*> masks{static_cast<cudf::bitmask_type*>(mask.data()),
                                           static_cast<cudf::bitmask_type*>(mask.data())};
    auto valids = cudf::detail::make_host_vector<bool>(masks.size(), cudf::get_default_stream());
    valids[0]   = valid;
    valids[1]   = !valid;

    cudf::set_null_masks_safe(masks, begins, ends, valids);
    // Verify bitmasks
    expect_bitmask_equal(static_cast<cudf::bitmask_type*>(mask.data()), 0, expected);
  }
};

// tests for set_null_mask
TEST_F(SetBitmaskTest, fill_range)
{
  cudf::size_type size = 121;
  for (auto begin = 0; begin < size; begin += 5)
    for (auto end = begin + 1; end <= size; end += 7) {
      this->test_set_null_range(size, begin, end, true);
      this->test_set_null_range(size, begin, end, false);
    }
}

TEST_F(SetBitmaskTest, null_mask_partition)
{
  cudf::size_type size = 67;
  for (auto middle = 1; middle < size; middle++) {
    this->test_null_partition(size, middle, true);
    this->test_null_partition(size, middle, false);
  }
}

TEST_F(SetBitmaskTest, null_mask_partition_bulk_unsafe)
{
  auto const sizes = std::vector<cudf::size_type>{67, 121};
  for (auto size : sizes) {
    for (auto middle = 1; middle < size; middle++) {
      this->test_null_partition_bulk_safe(size, middle, true);
      this->test_null_partition_bulk_safe(size, middle, false);
    }
  }
}

TEST_F(SetBitmaskTest, null_mask_partition_bulk_safe)
{
  auto const sizes = std::vector<cudf::size_type>{67, 121};
  for (auto size : sizes) {
    for (auto middle = 1; middle < size; middle++) {
      this->test_null_partition_bulk_safe(size, middle, true);
      this->test_null_partition_bulk_safe(size, middle, false);
    }
  }
}

TEST_F(SetBitmaskTest, null_mask_bulk_empty)
{
  EXPECT_NO_THROW(cudf::set_null_masks_unsafe({}, {}, {}, {}));
  EXPECT_NO_THROW(cudf::set_null_masks_safe({}, {}, {}, {}));
}

TEST_F(SetBitmaskTest, error_range)
{
  cudf::size_type size = 121;
  using size_pair      = std::pair<cudf::size_type, cudf::size_type>;
  std::vector<size_pair> begin_end_fail{
    {-1, size},  // begin>=0
    {-2, -1},    // begin>=0
    {9, 8},      // begin<=end
  };
  for (auto begin_end : begin_end_fail) {
    auto begin = begin_end.first, end = begin_end.second;
    EXPECT_ANY_THROW(this->test_set_null_range(size, begin, end, true));
    EXPECT_ANY_THROW(this->test_set_null_range(size, begin, end, false));
  }
  std::vector<size_pair> begin_end_pass{
    {0, size},         // begin>=0
    {0, 1},            // begin>=0
    {8, 8},            // begin==end
    {8, 9},            // begin<=end
    {size - 1, size},  // begin<=end
  };
  for (auto begin_end : begin_end_pass) {
    auto begin = begin_end.first, end = begin_end.second;
    EXPECT_NO_THROW(this->test_set_null_range(size, begin, end, true));
    EXPECT_NO_THROW(this->test_set_null_range(size, begin, end, false));
  }
}
