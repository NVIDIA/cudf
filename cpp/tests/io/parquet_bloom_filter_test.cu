/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/reduction/bloom_filter.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/bloom_filter.cuh>

#include <cstdint>

using StringType = cudf::string_view;

class ParquetBloomFilterTest : public cudf::test::BaseFixture {};

TEST_F(ParquetBloomFilterTest, TestStrings)
{
  using key_type    = StringType;
  using policy_type = cudf::arrow_filter_policy<cudf::hashing::detail::XXHash_64<key_type>>;
  using word_type   = policy_type::word_type;

  std::size_t constexpr num_filter_blocks = 4;
  auto stream                             = cudf::get_default_stream();

  // strings keys to insert
  auto keys = cudf::test::strings_column_wrapper(
    {"seventh",    "fifteenth",   "second",      "tenth",      "fifth",       "first",
     "seventh",    "tenth",       "ninth",       "ninth",      "seventeenth", "eighteenth",
     "thirteenth", "fifth",       "fourth",      "twelfth",    "second",      "second",
     "fourth",     "seventh",     "seventh",     "tenth",      "thirteenth",  "seventeenth",
     "fifth",      "seventeenth", "eighth",      "fourth",     "second",      "eighteenth",
     "fifteenth",  "second",      "seventeenth", "thirteenth", "eighteenth",  "fifth",
     "seventh",    "tenth",       "fourteenth",  "first",      "fifth",       "fifth",
     "tenth",      "thirteenth",  "fourteenth",  "third",      "third",       "sixth",
     "first",      "third"});

  auto d_keys = cudf::column_device_view::create(keys);

  // Spawn a bloom filter
  cuco::bloom_filter<key_type,
                     cuco::extent<size_t>,
                     cuda::thread_scope_device,
                     policy_type,
                     rmm::mr::polymorphic_allocator<char>>
    filter{num_filter_blocks,
           cuco::thread_scope_device,
           {{cudf::DEFAULT_HASH_SEED}},
           rmm::mr::polymorphic_allocator<char>{},
           stream};

  // Add strings to the bloom filter
  filter.add(d_keys->begin<key_type>(), d_keys->end<key_type>(), stream);

  // Number of words in the filter
  cudf::size_type const num_words = filter.block_extent() * filter.words_per_block;

  // Filter bitset as a column
  auto const bitset = cudf::column_view{
    cudf::data_type{cudf::type_id::UINT32}, num_words, filter.data(), nullptr, 0, 0, {}};

  // Expected filter bitset words computed using Arrow's implementation here:
  // https://godbolt.org/z/oKfqcPWbY
  auto expected = cudf::test::fixed_width_column_wrapper<word_type>(
    {4194306U,    4194305U,    2359296U,  1073774592U, 524544U,    1024U,      268443648U,
     8519680U,    2147500040U, 8421380U,  269500416U,  4202624U,   8396802U,   100665344U,
     2147747840U, 5243136U,    131146U,   655364U,     285345792U, 134222340U, 545390596U,
     2281717768U, 51201U,      41943553U, 1619656708U, 67441680U,  8462730U,   361220U,
     2216738864U, 587333888U,  4219272U,  873463873U});

  // Check the bitset for equality
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bitset, expected);
}
