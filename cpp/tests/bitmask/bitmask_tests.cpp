/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/iterator>

#include <stdexcept>

struct BitmaskUtilitiesTest : public cudf::test::BaseFixture {};

TEST_F(BitmaskUtilitiesTest, StateNullCount)
{
  EXPECT_EQ(0, cudf::state_null_count(cudf::mask_state::UNALLOCATED, 42));
  EXPECT_EQ(42, cudf::state_null_count(cudf::mask_state::ALL_NULL, 42));
  EXPECT_EQ(0, cudf::state_null_count(cudf::mask_state::ALL_VALID, 42));
  EXPECT_THROW(cudf::state_null_count(cudf::mask_state::UNINITIALIZED, 42), std::invalid_argument);
}

TEST_F(BitmaskUtilitiesTest, BitmaskAllocationSize)
{
  EXPECT_EQ(0u, cudf::bitmask_allocation_size_bytes(0));
  EXPECT_EQ(64u, cudf::bitmask_allocation_size_bytes(1));
  EXPECT_EQ(64u, cudf::bitmask_allocation_size_bytes(512));
  EXPECT_EQ(128u, cudf::bitmask_allocation_size_bytes(513));
  EXPECT_EQ(128u, cudf::bitmask_allocation_size_bytes(1023));
  EXPECT_EQ(128u, cudf::bitmask_allocation_size_bytes(1024));
  EXPECT_EQ(192u, cudf::bitmask_allocation_size_bytes(1025));
}

TEST_F(BitmaskUtilitiesTest, NumBitmaskWords)
{
  EXPECT_EQ(0, cudf::num_bitmask_words(0));
  EXPECT_EQ(1, cudf::num_bitmask_words(1));
  EXPECT_EQ(1, cudf::num_bitmask_words(31));
  EXPECT_EQ(1, cudf::num_bitmask_words(32));
  EXPECT_EQ(2, cudf::num_bitmask_words(33));
  EXPECT_EQ(2, cudf::num_bitmask_words(63));
  EXPECT_EQ(2, cudf::num_bitmask_words(64));
  EXPECT_EQ(3, cudf::num_bitmask_words(65));
}

struct CountBitmaskTest : public cudf::test::BaseFixture {};

TEST_F(CountBitmaskTest, NullMask)
{
  EXPECT_EQ(0, cudf::null_count(nullptr, 0, 32));

  std::vector<cudf::size_type> indices = {0, 32, 7, 25};

  auto counts = cudf::segmented_null_count(nullptr, indices);
  EXPECT_THAT(counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));
  counts = cudf::segmented_valid_count(nullptr, indices);
  EXPECT_THAT(counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{32, 18}));
}

// Utility to construct a mask vector. If fill_valid is false (default), it is initialized to all
// null. Otherwise it is initialized to all valid.
rmm::device_uvector<cudf::bitmask_type> make_mask(cudf::size_type size, bool fill_valid = false)
{
  if (!fill_valid) {
    return cudf::detail::make_zeroed_device_uvector<cudf::bitmask_type>(
      size, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  } else {
    auto ret = rmm::device_uvector<cudf::bitmask_type>(size, cudf::get_default_stream());
    CUDF_CUDA_TRY(cudaMemsetAsync(ret.data(),
                                  ~cudf::bitmask_type{0},
                                  size * sizeof(cudf::bitmask_type),
                                  cudf::get_default_stream().get()));
    return ret;
  }
}

TEST_F(CountBitmaskTest, NegativeStart)
{
  auto mask = make_mask(1);
  EXPECT_THROW(cudf::null_count(mask.data(), -1, 32), std::invalid_argument);
  EXPECT_THROW(cudf::index_of_first_set_bit(mask.data(), -1, 32), std::invalid_argument);

  auto indices = std::vector<cudf::size_type>{0, 16, -1, 32};
  EXPECT_THROW(cudf::segmented_null_count(mask.data(), indices), std::out_of_range);
  EXPECT_THROW(cudf::segmented_valid_count(mask.data(), indices), std::out_of_range);
}

TEST_F(CountBitmaskTest, StartLargerThanStop)
{
  auto mask = make_mask(1);
  EXPECT_THROW(cudf::null_count(mask.data(), 32, 31), std::invalid_argument);
  EXPECT_THROW(cudf::index_of_first_set_bit(mask.data(), 32, 31), std::invalid_argument);

  auto indices = std::vector<cudf::size_type>{0, 16, 31, 30};
  EXPECT_THROW(cudf::segmented_null_count(mask.data(), indices), std::invalid_argument);
  EXPECT_THROW(cudf::segmented_valid_count(mask.data(), indices), std::invalid_argument);
}

TEST_F(CountBitmaskTest, EmptyRange)
{
  auto mask = make_mask(1);
  EXPECT_EQ(0, cudf::null_count(mask.data(), 17, 17));
  EXPECT_THROW(cudf::index_of_first_set_bit(mask.data(), 17, 17), std::invalid_argument);

  auto const indices     = std::vector<cudf::size_type>{0, 0, 17, 17};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));
  auto const valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));
}

TEST_F(CountBitmaskTest, SingleWordAllZero)
{
  auto mask = make_mask(1, false);
  EXPECT_EQ(32, cudf::null_count(mask.data(), 0, 32));

  auto const indices     = std::vector<cudf::size_type>{0, 32, 0, 32};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{32, 32}));
  auto const valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));
}

TEST_F(CountBitmaskTest, SingleBitAllZero)
{
  auto mask = make_mask(1, false);
  EXPECT_EQ(1, cudf::null_count(mask.data(), 17, 18));

  auto const indices     = std::vector<cudf::size_type>{17, 18, 7, 8};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{1, 1}));
  auto const valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));
}

TEST_F(CountBitmaskTest, SingleBitAllSet)
{
  auto mask = make_mask(1, true);
  EXPECT_EQ(0, cudf::null_count(mask.data(), 13, 14));

  auto const indices     = std::vector<cudf::size_type>{13, 14, 0, 1};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));
  auto const valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{1, 1}));
}

TEST_F(CountBitmaskTest, SingleWordAllBitsSet)
{
  auto mask = make_mask(1, true);
  EXPECT_EQ(0, cudf::null_count(mask.data(), 0, 32));

  auto const indices     = std::vector<cudf::size_type>{0, 32, 0, 32};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));
  auto const valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{32, 32}));
}

TEST_F(CountBitmaskTest, SingleWordPreSlack)
{
  auto mask = make_mask(1, false);
  EXPECT_EQ(25, cudf::null_count(mask.data(), 7, 32));

  auto const indices     = std::vector<cudf::size_type>{7, 32, 8, 32};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{25, 24}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(1, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{25, 24}));
}

TEST_F(CountBitmaskTest, SingleWordPostSlack)
{
  auto mask = make_mask(1, false);
  EXPECT_EQ(17, cudf::null_count(mask.data(), 0, 17));

  auto const indices     = std::vector<cudf::size_type>{0, 17, 0, 18};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{17, 18}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(1, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{17, 18}));
}

TEST_F(CountBitmaskTest, SingleWordSubset)
{
  auto mask = make_mask(1, false);
  EXPECT_EQ(30, cudf::null_count(mask.data(), 1, 31));

  auto const indices     = std::vector<cudf::size_type>{1, 31, 7, 17};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{30, 10}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(1, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{30, 10}));
}

TEST_F(CountBitmaskTest, SingleWordSubset2)
{
  auto mask = make_mask(1, false);
  EXPECT_EQ(28, cudf::null_count(mask.data(), 2, 30));

  auto const indices     = std::vector<cudf::size_type>{4, 16, 2, 30};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{12, 28}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(1, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{12, 28}));
}

TEST_F(CountBitmaskTest, MultipleWordsAllBits)
{
  auto mask = make_mask(10, false);
  EXPECT_EQ(320, cudf::null_count(mask.data(), 0, 320));

  auto const indices     = std::vector<cudf::size_type>{0, 320, 0, 320};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{320, 320}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(10, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{320, 320}));
}

TEST_F(CountBitmaskTest, MultipleWordsSubsetWordBoundary)
{
  auto mask = make_mask(10, false);
  EXPECT_EQ(256, cudf::null_count(mask.data(), 32, 288));

  auto const indices     = std::vector<cudf::size_type>{32, 192, 32, 288};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{160, 256}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(10, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{160, 256}));
}

TEST_F(CountBitmaskTest, MultipleWordsSplitWordBoundary)
{
  auto mask = make_mask(10, false);
  EXPECT_EQ(2, cudf::null_count(mask.data(), 31, 33));

  auto const indices     = std::vector<cudf::size_type>{31, 33, 60, 67};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{2, 7}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(10, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{2, 7}));
}

TEST_F(CountBitmaskTest, MultipleWordsSubset)
{
  auto mask = make_mask(10, false);
  EXPECT_EQ(226, cudf::null_count(mask.data(), 67, 293));

  auto const indices     = std::vector<cudf::size_type>{67, 293, 37, 319};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{226, 282}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0}));

  mask         = make_mask(10, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{226, 282}));
}

TEST_F(CountBitmaskTest, MultipleWordsSingleBit)
{
  auto mask = make_mask(10, false);
  EXPECT_EQ(1, cudf::null_count(mask.data(), 67, 68));

  auto const indices     = std::vector<cudf::size_type>{67, 68, 31, 32, 192, 193};
  auto const null_counts = cudf::segmented_null_count(mask.data(), indices);
  EXPECT_THAT(null_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{1, 1, 1}));
  auto valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{0, 0, 0}));

  mask         = make_mask(10, true);
  valid_counts = cudf::segmented_valid_count(mask.data(), indices);
  EXPECT_THAT(valid_counts, ::testing::ElementsAreArray(std::vector<cudf::size_type>{1, 1, 1}));
}

TEST_F(CountBitmaskTest, BatchNullCount)
{
  using cudf::test::iterators::nulls_at;
  using col_type  = cudf::test::fixed_width_column_wrapper<int32_t>;
  auto const col0 = col_type{{0, 1, 2, 3, 4, 5}, nulls_at({1, 3, 5})};
  auto const col1 = col_type{{0, 1, 2, 3, 4, 5}, nulls_at({0, 2, 4})};
  auto const col2 = col_type{{0, 1, 2, 3, 4, 5}, nulls_at({2, 3})};
  auto const col3 = col_type{{0, 1, 2, 3, 4, 5}, nulls_at({4})};
  auto const col4 = col_type{{0, 1, 2, 3, 4, 5}, nulls_at({0, 1, 2, 3, 4, 5})};
  auto const col5 = col_type{0, 1, 2, 3, 4, 5};

  auto const bitmasks =
    std::vector<cudf::bitmask_type const*>{static_cast<cudf::column_view>(col0).null_mask(),
                                           static_cast<cudf::column_view>(col1).null_mask(),
                                           static_cast<cudf::column_view>(col2).null_mask(),
                                           static_cast<cudf::column_view>(col3).null_mask(),
                                           static_cast<cudf::column_view>(col4).null_mask(),
                                           static_cast<cudf::column_view>(col5).null_mask()};

  auto const null_counts =
    cudf::batch_null_count(bitmasks, 0, static_cast<cudf::column_view>(col0).size());
  EXPECT_THAT(null_counts,
              ::testing::ElementsAreArray(std::vector<cudf::size_type>{3, 3, 2, 1, 6, 0}));
}

struct iofub_test_parameter {
  cudf::size_type size;
  cudf::size_type set_index;
  cudf::size_type start_index;
  cudf::size_type result;
};

TEST_F(CountBitmaskTest, IndexOfFirstUnsetBit)
{
  auto parameters = std::vector<iofub_test_parameter>({
    // clang-format off
    // rows  set  start result
    {  28,    0,    0,    0},  // less than sizeof bitmask_type
    {  28,   10,   10,    0},
    {  32,    0,    0,    0},  // equal to sizeof bitmask_type
    {  32,   10,   10,    0},
    {  32,   10,   11,   21},  // set bit is not in range
    {  64,   33,    0,   33},  // exactly 2x sizeof bitmask_type
    { 260,  258,    0,  258},  // greater than 256 bits
    { 260,  258,   32,  226},
    { 320,  260,   60,  200},
    { 320,  260,  256,    4},
    {9000,    2,   0,     2},  // more than one CUDA block
    {9000,  260,  256,    4},
    {9000, 8193,    0, 8193},
    {9000, 8193, 8192,    1},
    {9000,    8,   80, 8920},  // set bit is not in range
    // clang-format on
  });
  for (auto parm : parameters) {
    auto data            = std::vector<bool>(parm.size, false);
    data[parm.set_index] = true;
    std::fill(data.begin() + parm.set_index + 1, data.end(), (parm.set_index >= parm.start_index));
    auto input =
      cudf::test::fixed_width_column_wrapper<int, bool>(data.begin(), data.end(), data.begin())
        .release();
    auto result = cudf::index_of_first_set_bit(
      input->view().null_mask(), parm.start_index, input->view().size());
    EXPECT_EQ(result, parm.result);
  }
}

struct CopyBitmaskTest : public cudf::test::BaseFixture, cudf::test::UniformRandomGenerator<int> {
  CopyBitmaskTest() : cudf::test::UniformRandomGenerator<int>{0, 1} {}
};

void cleanEndWord(rmm::device_buffer& mask, int begin_bit, int end_bit)
{
  auto ptr = static_cast<cudf::bitmask_type*>(mask.data());

  auto number_of_mask_words = cudf::num_bitmask_words(static_cast<size_t>(end_bit - begin_bit));
  auto number_of_bits       = end_bit - begin_bit;
  if (number_of_bits % 32 != 0) {
    cudf::bitmask_type end_mask = 0;
    CUDF_CUDA_TRY(
      cudaMemcpy(&end_mask, ptr + number_of_mask_words - 1, sizeof(end_mask), cudaMemcpyDefault));
    end_mask = end_mask & ((1 << (number_of_bits % 32)) - 1);
    CUDF_CUDA_TRY(
      cudaMemcpy(ptr + number_of_mask_words - 1, &end_mask, sizeof(end_mask), cudaMemcpyDefault));
  }
}

TEST_F(CopyBitmaskTest, NegativeStart)
{
  auto mask = make_mask(1);
  EXPECT_THROW(cudf::copy_bitmask(mask.data(), -1, 32), cudf::logic_error);
}

TEST_F(CopyBitmaskTest, StartLargerThanStop)
{
  auto mask = make_mask(1);
  EXPECT_THROW(cudf::copy_bitmask(mask.data(), 32, 31), cudf::logic_error);
}

TEST_F(CopyBitmaskTest, EmptyRange)
{
  auto mask = make_mask(1);
  auto buff = cudf::copy_bitmask(mask.data(), 17, 17);
  EXPECT_EQ(0, static_cast<int>(buff.size()));
}

TEST_F(CopyBitmaskTest, NullPtr)
{
  auto buff = cudf::copy_bitmask(nullptr, 17, 17);
  EXPECT_EQ(0, static_cast<int>(buff.size()));
}

TEST_F(CopyBitmaskTest, TestZeroOffset)
{
  std::vector<int> validity_bit(1000);
  for (auto& m : validity_bit) {
    m = this->generate();
  }
  auto input_mask =
    std::get<0>(cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end()));

  int begin_bit         = 0;
  int end_bit           = 800;
  auto gold_splice_mask = std::get<0>(cudf::test::detail::make_null_mask(
    validity_bit.begin() + begin_bit, validity_bit.begin() + end_bit));

  auto splice_mask = cudf::copy_bitmask(
    static_cast<cudf::bitmask_type const*>(input_mask.data()), begin_bit, end_bit);

  cleanEndWord(splice_mask, begin_bit, end_bit);
  auto number_of_bits = end_bit - begin_bit;
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    gold_splice_mask.data(), splice_mask.data(), cudf::num_bitmask_words(number_of_bits));
}

TEST_F(CopyBitmaskTest, TestNonZeroOffset)
{
  std::vector<int> validity_bit(1000);
  for (auto& m : validity_bit) {
    m = this->generate();
  }
  auto input_mask =
    std::get<0>(cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end()));

  int begin_bit         = 321;
  int end_bit           = 998;
  auto gold_splice_mask = std::get<0>(cudf::test::detail::make_null_mask(
    validity_bit.begin() + begin_bit, validity_bit.begin() + end_bit));

  auto splice_mask = cudf::copy_bitmask(
    static_cast<cudf::bitmask_type const*>(input_mask.data()), begin_bit, end_bit);

  cleanEndWord(splice_mask, begin_bit, end_bit);
  auto number_of_bits = end_bit - begin_bit;
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    gold_splice_mask.data(), splice_mask.data(), cudf::num_bitmask_words(number_of_bits));
}

TEST_F(CopyBitmaskTest, TestCopyColumnViewVectorContiguous)
{
  cudf::data_type t{cudf::type_id::INT32};
  cudf::size_type num_elements = 1001;
  std::vector<int> validity_bit(num_elements);
  for (auto& m : validity_bit) {
    m = this->generate();
  }
  auto [gold_mask, null_count] =
    cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end());

  rmm::device_buffer copy_mask{gold_mask, cudf::get_default_stream()};
  cudf::column original{t,
                        num_elements,
                        rmm::device_buffer{num_elements * sizeof(int), cudf::get_default_stream()},
                        std::move(copy_mask),
                        null_count};
  std::vector<cudf::size_type> indices{0,
                                       104,
                                       104,
                                       128,
                                       128,
                                       152,
                                       152,
                                       311,
                                       311,
                                       491,
                                       491,
                                       583,
                                       583,
                                       734,
                                       734,
                                       760,
                                       760,
                                       num_elements};
  std::vector<cudf::column_view> views    = cudf::slice(original, indices);
  rmm::device_buffer concatenated_bitmask = cudf::concatenate_masks(views);
  cleanEndWord(concatenated_bitmask, 0, num_elements);
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    concatenated_bitmask.data(), gold_mask.data(), cudf::num_bitmask_words(num_elements));
}

TEST_F(CopyBitmaskTest, TestCopyColumnViewVectorDiscontiguous)
{
  cudf::data_type t{cudf::type_id::INT32};
  cudf::size_type num_elements = 1001;
  std::vector<int> validity_bit(num_elements);
  for (auto& m : validity_bit) {
    m = this->generate();
  }
  auto gold_mask =
    std::get<0>(cudf::test::detail::make_null_mask(validity_bit.begin(), validity_bit.end()));
  std::vector<cudf::size_type> split{0, 104, 128, 152, 311, 491, 583, 734, 760, num_elements};

  std::vector<cudf::column> cols;
  std::vector<cudf::column_view> views;
  for (unsigned i = 0; i < split.size() - 1; i++) {
    auto [null_mask, null_count] = cudf::test::detail::make_null_mask(
      validity_bit.begin() + split[i], validity_bit.begin() + split[i + 1]);
    cols.emplace_back(
      t,
      split[i + 1] - split[i],
      rmm::device_buffer{sizeof(int) * (split[i + 1] - split[i]), cudf::get_default_stream()},
      std::move(null_mask),
      null_count);
    views.push_back(cols.back());
  }
  rmm::device_buffer concatenated_bitmask = cudf::concatenate_masks(views);
  cleanEndWord(concatenated_bitmask, 0, num_elements);
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    concatenated_bitmask.data(), gold_mask.data(), cudf::num_bitmask_words(num_elements));
}

struct MergeBitmaskTest : public cudf::test::BaseFixture {};

TEST_F(MergeBitmaskTest, TestBitmaskAnd)
{
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col3({0, 2, 1, 0, 255});

  auto const input1 = cudf::table_view({bools_col3});
  auto const input2 = cudf::table_view({bools_col1, bools_col2});
  auto const input3 = cudf::table_view({bools_col1, bools_col2, bools_col3});

  auto [result1_mask, result1_null_count] = cudf::bitmask_and(input1);
  auto [result2_mask, result2_null_count] = cudf::bitmask_and(input2);
  auto [result3_mask, result3_null_count] = cudf::bitmask_and(input3);

  constexpr cudf::size_type gold_null_count = 3;

  EXPECT_EQ(result1_null_count, 0);
  EXPECT_EQ(result2_null_count, gold_null_count);
  EXPECT_EQ(result3_null_count, gold_null_count);

  auto odd_indices = cudf::test::iterators::nulls_at_multiples_of(2);
  auto odd =
    std::get<0>(cudf::test::detail::make_null_mask(odd_indices, odd_indices + input2.num_rows()));

  EXPECT_EQ(nullptr, result1_mask.data());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    result2_mask.data(), odd.data(), cudf::num_bitmask_words(input2.num_rows()));
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    result3_mask.data(), odd.data(), cudf::num_bitmask_words(input2.num_rows()));
}

TEST_F(MergeBitmaskTest, TestSegmentedBitmaskAndSingleSegment)
{
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col3({0, 2, 1, 0, 255}, {1, 1, 1, 1, 1});

  auto const num_rows = 5;
  {
    std::vector<cudf::column_view> const colviews{bools_col3};
    std::vector<cudf::size_type> const segment_offsets{0, 1};
    auto const [result_masks, result_null_count] =
      cudf::segmented_bitmask_and(colviews, segment_offsets);
    EXPECT_EQ(result_null_count.size(), 1);
    EXPECT_EQ(result_masks.size(), 1);
    EXPECT_EQ(result_null_count[0], 0);
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(result_masks[0]->data(),
                                   static_cast<cudf::column_view>(bools_col3).null_mask(),
                                   cudf::num_bitmask_words(num_rows));
  }

  {
    std::vector<cudf::column_view> const colviews{bools_col1, bools_col2};
    std::vector<cudf::size_type> const segment_offsets{0, 2};
    auto const [result_masks, result_null_count] =
      cudf::segmented_bitmask_and(colviews, segment_offsets);
    EXPECT_EQ(result_null_count.size(), 1);
    EXPECT_EQ(result_masks.size(), 1);
    EXPECT_EQ(result_null_count[0], 3);
    auto odd_indices = cudf::test::iterators::nulls_at_multiples_of(2);
    auto const odd =
      std::get<0>(cudf::test::detail::make_null_mask(odd_indices, odd_indices + num_rows));
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(
      result_masks[0]->data(), odd.data(), cudf::num_bitmask_words(num_rows));
  }

  {
    std::vector<cudf::column_view> const colviews{bools_col1, bools_col2, bools_col3};
    std::vector<cudf::size_type> const segment_offsets{0, 3};
    auto const [result_masks, result_null_count] =
      cudf::segmented_bitmask_and(colviews, segment_offsets);
    EXPECT_EQ(result_null_count.size(), 1);
    EXPECT_EQ(result_masks.size(), 1);
    EXPECT_EQ(result_null_count[0], 3);
    auto odd_indices = cudf::test::iterators::nulls_at_multiples_of(2);
    auto const odd =
      std::get<0>(cudf::test::detail::make_null_mask(odd_indices, odd_indices + num_rows));
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(
      result_masks[0]->data(), odd.data(), cudf::num_bitmask_words(num_rows));
  }
}

TEST_F(MergeBitmaskTest, TestSegmentedBitmaskAndMultipleSegments)
{
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col3({0, 2, 1, 0, 255}, {1, 1, 1, 1, 1});

  auto const num_rows = 5;
  {
    std::vector<cudf::column_view> const colviews{bools_col1, bools_col2};
    std::vector<cudf::size_type> const segment_offsets{0, 1, 2};
    auto const [result_masks, result_null_count] =
      cudf::segmented_bitmask_and(colviews, segment_offsets);
    EXPECT_EQ(result_null_count.size(), 2);
    EXPECT_EQ(result_masks.size(), 2);
    EXPECT_EQ(result_null_count[0], 2);
    EXPECT_EQ(result_null_count[1], 2);
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(result_masks[0]->data(),
                                   static_cast<cudf::column_view>(bools_col1).null_mask(),
                                   cudf::num_bitmask_words(num_rows));
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(result_masks[1]->data(),
                                   static_cast<cudf::column_view>(bools_col2).null_mask(),
                                   cudf::num_bitmask_words(num_rows));
  }

  {
    std::vector<cudf::column_view> const colviews{bools_col1, bools_col2, bools_col3};
    std::vector<cudf::size_type> const segment_offsets{0, 2, 3};
    auto const [result_masks, result_null_count] =
      cudf::segmented_bitmask_and(colviews, segment_offsets);
    EXPECT_EQ(result_null_count.size(), 2);
    EXPECT_EQ(result_masks.size(), 2);
    EXPECT_EQ(result_null_count[0], 3);
    EXPECT_EQ(result_null_count[1], 0);
    auto odd_indices = cudf::test::iterators::nulls_at_multiples_of(2);
    auto const odd =
      std::get<0>(cudf::test::detail::make_null_mask(odd_indices, odd_indices + num_rows));
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(
      result_masks[0]->data(), odd.data(), cudf::num_bitmask_words(num_rows));
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(result_masks[1]->data(),
                                   static_cast<cudf::column_view>(bools_col3).null_mask(),
                                   cudf::num_bitmask_words(num_rows));
  }
}

TEST_F(MergeBitmaskTest, TestSegmentedBitmaskAndEmptySegments)
{
  // Columns span several bitmask words with a partial last word
  auto const num_rows = 300;
  cudf::test::fixed_width_column_wrapper<int32_t> const col1(
    cuda::make_counting_iterator(0),
    cuda::make_counting_iterator(num_rows),
    cudf::test::iterators::nulls_at_multiples_of(3));
  cudf::test::fixed_width_column_wrapper<int32_t> const col2(
    cuda::make_counting_iterator(0),
    cuda::make_counting_iterator(num_rows),
    cudf::test::iterators::nulls_at_multiples_of(5));

  // An empty segment should yield the identity of bitwise AND: every row valid.
  auto const expect_all_valid = [&](void const* mask) {
    EXPECT_EQ(cudf::null_count(static_cast<cudf::bitmask_type const*>(mask), 0, num_rows), 0);
  };

  // Empty leading, interior and trailing segments
  {
    std::vector<cudf::column_view> const colviews{col1, col2};
    std::vector<cudf::size_type> const segment_offsets{0, 0, 1, 1, 2, 2};
    auto const [result_masks, result_null_count] =
      cudf::segmented_bitmask_and(colviews, segment_offsets);
    ASSERT_EQ(result_masks.size(), 5);
    EXPECT_EQ(result_null_count, std::vector<cudf::size_type>({0, 100, 0, 60, 0}));
    expect_all_valid(result_masks[0]->data());
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(result_masks[1]->data(),
                                   static_cast<cudf::column_view>(col1).null_mask(),
                                   cudf::num_bitmask_words(num_rows));
    expect_all_valid(result_masks[2]->data());
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(result_masks[3]->data(),
                                   static_cast<cudf::column_view>(col2).null_mask(),
                                   cudf::num_bitmask_words(num_rows));
    expect_all_valid(result_masks[4]->data());
  }

  // Call the raw-mask overload
  {
    std::vector<cudf::bitmask_type const*> const masks{
      static_cast<cudf::column_view>(col1).null_mask(),
      static_cast<cudf::column_view>(col2).null_mask()};
    std::vector<cudf::size_type> const segment_offsets{0, 0, 2};
    auto const [result_masks, result_null_count] =
      cudf::segmented_bitmask_and(masks, segment_offsets, num_rows);
    ASSERT_EQ(result_masks.size(), 2);
    auto const [expected_mask, expected_null_count] =
      cudf::bitmask_and(cudf::table_view({col1, col2}));
    EXPECT_EQ(result_null_count, std::vector<cudf::size_type>({0, expected_null_count}));
    expect_all_valid(result_masks[0]->data());
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(
      result_masks[1]->data(), expected_mask.data(), cudf::num_bitmask_words(num_rows));
  }
}

TEST_F(MergeBitmaskTest, TestSegmentedBitmaskAndMultipleBlocksPerSegment)
{
  // Wide enough that a segment spans more than one block of the reduction kernel
  auto const num_rows = 100'003;
  cudf::test::fixed_width_column_wrapper<int32_t> const col1(
    cuda::make_counting_iterator(0),
    cuda::make_counting_iterator(num_rows),
    cudf::test::iterators::nulls_at_multiples_of(3));
  cudf::test::fixed_width_column_wrapper<int32_t> const col2(
    cuda::make_counting_iterator(0),
    cuda::make_counting_iterator(num_rows),
    cudf::test::iterators::nulls_at_multiples_of(5));
  cudf::test::fixed_width_column_wrapper<int32_t> const col3(
    cuda::make_counting_iterator(0),
    cuda::make_counting_iterator(num_rows),
    cudf::test::iterators::nulls_at_multiples_of(7));

  auto const multiples_of = [&](int n) { return (num_rows - 1) / n + 1; };
  std::vector<cudf::size_type> const expected_null_counts{
    multiples_of(3) + multiples_of(5) - multiples_of(15), multiples_of(7)};

  std::vector<cudf::column_view> const colviews{col1, col2, col3};
  std::vector<cudf::size_type> const segment_offsets{0, 2, 3};
  auto const [result_masks, result_null_count] =
    cudf::segmented_bitmask_and(colviews, segment_offsets);

  ASSERT_EQ(result_masks.size(), 2);
  EXPECT_EQ(result_null_count, expected_null_counts);

  auto const [expected_mask, expected_null_count] =
    cudf::bitmask_and(cudf::table_view({col1, col2}));
  EXPECT_EQ(expected_null_count, expected_null_counts[0]);
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    result_masks[0]->data(), expected_mask.data(), cudf::num_bitmask_words(num_rows));
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(result_masks[1]->data(),
                                 static_cast<cudf::column_view>(col3).null_mask(),
                                 cudf::num_bitmask_words(num_rows));
}

TEST_F(MergeBitmaskTest, TestSegmentedBitmaskAndNoColumns)
{
  std::vector<cudf::column_view> const colviews{};
  std::vector<cudf::size_type> const segment_offsets{0};
  auto const [result_masks, result_null_count] =
    cudf::segmented_bitmask_and(colviews, segment_offsets);
  EXPECT_TRUE(result_masks.empty());
  EXPECT_TRUE(result_null_count.empty());
}

TEST_F(MergeBitmaskTest, TestBitmaskOr)
{
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {0, 0, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col3({0, 2, 1, 0, 255});

  auto const input1 = cudf::table_view({bools_col3});
  auto const input2 = cudf::table_view({bools_col1, bools_col2});
  auto const input3 = cudf::table_view({bools_col1, bools_col2, bools_col3});

  auto [result1_mask, result1_null_count] = cudf::bitmask_or(input1);
  auto [result2_mask, result2_null_count] = cudf::bitmask_or(input2);
  auto [result3_mask, result3_null_count] = cudf::bitmask_or(input3);

  EXPECT_EQ(result1_null_count, 0);
  EXPECT_EQ(result2_null_count, 1);
  EXPECT_EQ(result3_null_count, 0);

  auto all_but_index3 = cudf::test::iterators::null_at(3);
  auto null3          = std::get<0>(
    cudf::test::detail::make_null_mask(all_but_index3, all_but_index3 + input2.num_rows()));

  EXPECT_EQ(nullptr, result1_mask.data());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(
    result2_mask.data(), null3.data(), cudf::num_bitmask_words(input2.num_rows()));
  EXPECT_EQ(nullptr, result3_mask.data());
}

CUDF_TEST_PROGRAM_MAIN()
